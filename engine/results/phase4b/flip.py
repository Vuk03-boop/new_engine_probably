"""4B Q4 and M3 (a): LDR-FLIP (flip-evaluator 1.7, A-008).

usage: python flip.py <NE_4B_DIR> <out dir> [<eight-bounce ref dir> <one-bounce ref dir>]

Q4 (docs/changes/2026-09-26-phase4b-many-lights.md): for every still arm (street, low x blue hour,
night) the raw and the filtered display images at ages 1, 16, 64, and for the night motion frames 8,
16, 32, each against its comparison image, as written by `gpu/tests/night.rs`. Criterion: filtered
FLIP <= raw FLIP at the same age / frame.

M3 (a), data, when the two reference directories are given (`ref_light` PFMs, 960x540): FLIP between
the one-bounce and the eight-bounce night references (blue hour, night; both cameras), both shown
at the eight-bounce image's metric exposure (light::exposure::metric_exposure), the ACES fit, sRGB.

Writes `flip.jsonl` (one line per arm, then a summary) and a contact sheet `stills_sheet.png`.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import flip_evaluator as flip

TIMES = ["blue_hour", "night"]
CAMERAS = ["street", "low"]
AGES = [1, 16, 64]
MOTION = [8, 16, 32]


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def score(ref, test):
    err, mean, _ = flip.evaluate(ref, test, "LDR", applyMagma=False)
    return float(mean)


def load_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data[::-1].astype(np.float64)  # PFM rows run bottom to top


def metric_exposure(img):
    y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mean = y.mean()
    if not (np.isfinite(mean) and mean > 0):
        return None
    lit = y[y >= mean / 1024.0]
    return 0.18 / np.exp(np.log(lit).mean())


def display(img, exposure):
    x = np.clip(img, 0.0, None) * exposure
    a = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    s = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)
    return (np.floor(s * 255.0 + 0.5) / 255.0).astype(np.float32)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines, failed, rows = [], [], []
    for time in TIMES:
        for cam in CAMERAS:
            key = f"still_{cam}_{time}"
            ref = load(src / f"{key}_ref.ppm")
            rec = {"arm": key}
            row = [ref]
            for age in AGES:
                raw, filt = load(src / f"{key}_raw_{age}.ppm"), load(src / f"{key}_filt_{age}.ppm")
                r, f = score(ref, raw), score(ref, filt)
                ok = f <= r
                rec[f"age{age}"] = {"raw": round(r, 5), "filtered": round(f, 5), "q4": "pass" if ok else "FAIL"}
                if not ok:
                    failed.append(f"{key} age {age}")
                if age == 1:
                    row += [raw, filt]
                if age == 64:
                    row.append(filt)
            rows.append(row)
            lines.append(rec)
            print(json.dumps(rec), flush=True)
    for k in MOTION:
        key = f"motion_night_{k}"
        ref = load(src / f"{key}_ref.ppm")
        r, f = score(ref, load(src / f"{key}_raw.ppm")), score(ref, load(src / f"{key}_filt.ppm"))
        ok = f <= r
        rec = {"arm": key, "raw": round(r, 5), "filtered": round(f, 5), "q4": "pass" if ok else "FAIL"}
        if not ok:
            failed.append(key)
        lines.append(rec)
        print(json.dumps(rec), flush=True)
    if len(sys.argv) >= 5:
        eight, one = Path(sys.argv[3]), Path(sys.argv[4])
        for time in TIMES:
            for cam in CAMERAS:
                a, b = load_pfm(eight / f"{cam}_{time}.pfm"), load_pfm(one / f"{cam}_{time}.pfm")
                e = metric_exposure(a)
                rec = {"m3a": f"{cam}_{time}", "exposure": e, "flip_one_vs_eight_bounces": round(score(display(a, e), display(b, e)), 5),
                       "image_mean_luminance_ratio": float((0.2126 * b[..., 0] + 0.7152 * b[..., 1] + 0.0722 * b[..., 2]).mean() / (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).mean())}
                lines.append(rec)
                print(json.dumps(rec), flush=True)
    summary = {"q4_failed": failed, "q4": "pass" if not failed else "FAIL", "flip": "flip-evaluator 1.7", "ppd": 67}
    lines.append(summary)
    print(json.dumps(summary), flush=True)
    (out / "flip.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    # Contact sheet: rows (street, low) x (blue hour, night); columns: comparison image, raw age 1,
    # filtered age 1, filtered age 64; each at a quarter of 1080p.
    tw, th = 480, 270
    sheet = Image.new("RGB", (tw * 4, th * len(rows)))
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            im = Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (j * tw, i * th))
    sheet.save(out / "stills_sheet.png")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
