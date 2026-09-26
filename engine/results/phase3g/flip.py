"""3G Q4: LDR-FLIP (flip-evaluator 1.7, A-008) of the display images written by `gpu/tests/gate.rs`.

usage: python flip.py <NE_GATE_DIR> <out dir> [image prefix]

The optional prefix selects a candidate filter's images (`NE_FILTER`, e.g. `conservative-4_`; the
G4 filter record, docs/changes/2026-09-26-4b-filter-energy.md); without it, the default filter's.

For every still arm (camera x time) it compares the raw and the filtered image at ages 1, 16, 64 with
the reference; for every motion frame the raw and the filtered one. Writes `flip.jsonl` (one line per
arm with the mean FLIP of each image and the Q4 verdict) and a contact sheet and error maps as PNG.
Criterion (docs/changes/2026-09-25-phase3g-gate.md): filtered FLIP <= raw FLIP at the same age/frame.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import flip_evaluator as flip

TIMES = ["dawn", "morning", "midday", "dusk", "twilight"]
CAMERAS = ["street", "low"]
AGES = [1, 16, 64]
MOTION = [("morning", k) for k in (8, 16, 32)] + [("dusk", k) for k in (8, 16, 32)]


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def score(ref, test):
    err, mean, _ = flip.evaluate(ref, test, "LDR", applyMagma=False)
    return float(mean), err


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    pre = sys.argv[3] if len(sys.argv) > 3 else ""
    out.mkdir(parents=True, exist_ok=True)
    lines, failed, sheet_rows, maps = [], [], [], []
    for time in TIMES:
        for cam in CAMERAS:
            key = f"{pre}still_{cam}_{time}"
            ref = load(src / f"{key}_ref.ppm")
            rec = {"arm": key}
            row = [ref]
            for age in AGES:
                raw, filt = load(src / f"{key}_raw_{age}.ppm"), load(src / f"{key}_filt_{age}.ppm")
                (r, re), (f, fe) = score(ref, raw), score(ref, filt)
                ok = f <= r
                rec[f"age{age}"] = {"raw": round(r, 5), "filtered": round(f, 5), "q4": "pass" if ok else "FAIL"}
                if not ok:
                    failed.append(f"{key} age {age}")
                if age in (1, 64):
                    row += [raw, filt] if age == 1 else [filt]
                if cam == "street" and time in ("morning", "dusk") and age == 1:
                    maps.append((f"{key}_age1", re, fe))
            if time in ("morning", "dusk", "twilight"):
                sheet_rows.append(row)
            lines.append(rec)
            print(json.dumps(rec), flush=True)
    for time, k in MOTION:
        key = f"{pre}motion_{time}_{k}"
        ref = load(src / f"{key}_ref.ppm")
        (r, _), (f, _) = score(ref, load(src / f"{key}_raw.ppm")), score(ref, load(src / f"{key}_filt.ppm"))
        ok = f <= r
        rec = {"arm": key, "raw": round(r, 5), "filtered": round(f, 5), "q4": "pass" if ok else "FAIL"}
        if not ok:
            failed.append(key)
        lines.append(rec)
        print(json.dumps(rec), flush=True)
    summary = {"q4_failed": failed, "q4": "pass" if not failed else "FAIL", "flip": flip.__name__ + " 1.7", "ppd": 67}
    lines.append(summary)
    print(json.dumps(summary), flush=True)
    (out / "flip.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    # Contact sheet: rows (morning, dusk, twilight) x (street, low); columns reference, raw age 1,
    # filtered age 1, filtered age 64; each image at a quarter of 1080p.
    tw, th = 480, 270
    sheet = Image.new("RGB", (tw * 4, th * len(sheet_rows)))
    for i, row in enumerate(sheet_rows):
        for j, img in enumerate(row):
            im = Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (j * tw, i * th))
    sheet.save(out / "stills_sheet.png")
    # FLIP error maps (grey, 1 = maximal difference) of raw and filtered at age 1: half size.
    for name, re, fe in maps:
        pair = np.concatenate([re, fe], axis=1)
        pair = np.clip(pair.squeeze(), 0.0, 1.0)
        Image.fromarray((pair * 255 + 0.5).astype(np.uint8)).resize((pair.shape[1] // 2, pair.shape[0] // 2), Image.LANCZOS).save(out / f"flip_{name}_raw_vs_filtered.png")


if __name__ == "__main__":
    main()
