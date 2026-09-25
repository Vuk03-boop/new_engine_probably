"""4B G4 Q4: LDR-FLIP (flip-evaluator 1.7, A-008) of the night display images written by
`gpu/tests/lights.rs` (`night_against_the_reference`).

usage: python flip.py <NE_GATE_DIR> <out dir>

For every arm (camera x time) it compares the raw and the filtered image at ages 1, 16, 64 with the
reference (all with emission added and the reference's metric exposure). Writes `flip.jsonl` (one line
per arm with the mean FLIP of each image and the Q4 verdict), a contact sheet and error maps as PNG.
Criterion (docs/changes/2026-09-25-phase4b-many-lights.md, G4): filtered FLIP <= raw FLIP at the same
age, judged at night; blue hour is data.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import flip_evaluator as flip

TIMES = [("night", True), ("blue_hour", False)]
CAMERAS = ["street", "low"]
AGES = [1, 16, 64]


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def score(ref, test):
    err, mean, _ = flip.evaluate(ref, test, "LDR", applyMagma=False)
    return float(mean), err


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines, failed, sheet_rows, maps = [], [], [], []
    for time, judged in TIMES:
        for cam in CAMERAS:
            key = f"still4b_{cam}_{time}"
            ref = load(src / f"{key}_ref.ppm")
            rec = {"arm": key, "judged": judged}
            row = [ref]
            for age in AGES:
                raw, filt = load(src / f"{key}_raw_{age}.ppm"), load(src / f"{key}_filt_{age}.ppm")
                (r, re), (f, fe) = score(ref, raw), score(ref, filt)
                ok = f <= r
                rec[f"age{age}"] = {"raw": round(r, 5), "filtered": round(f, 5), "q4": ("pass" if ok else "FAIL") if judged else "data"}
                if judged and not ok:
                    failed.append(f"{key} age {age}")
                if age == 1:
                    row += [raw, filt]
                    maps.append((f"{cam}_{time}_age1", re, fe))
                if age == 64:
                    row.append(filt)
            sheet_rows.append(row)
            lines.append(rec)
            print(json.dumps(rec), flush=True)
    summary = {"q4_failed": failed, "q4": "pass" if not failed else "FAIL", "flip": flip.__name__ + " 1.7", "ppd": 67}
    lines.append(summary)
    print(json.dumps(summary), flush=True)
    (out / "flip.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    # Contact sheet: rows (night, blue hour) x (street, low); columns reference, raw age 1, filtered
    # age 1, filtered age 64; each image at a quarter of 1080p.
    tw, th = 480, 270
    sheet = Image.new("RGB", (tw * 4, th * len(sheet_rows)))
    for i, row in enumerate(sheet_rows):
        for j, img in enumerate(row):
            im = Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
            sheet.paste(im, (j * tw, i * th))
    sheet.save(out / "stills_sheet.png")
    # FLIP error maps (grey, 1 = maximal difference) of raw and filtered at age 1: half size.
    for name, re, fe in maps:
        pair = np.clip(np.concatenate([re, fe], axis=1).squeeze(), 0.0, 1.0)
        Image.fromarray((pair * 255 + 0.5).astype(np.uint8)).resize((pair.shape[1] // 2, pair.shape[0] // 2), Image.LANCZOS).save(out / f"flip_{name}_raw_vs_filtered.png")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
