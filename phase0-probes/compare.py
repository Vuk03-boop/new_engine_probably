"""Compare probe PPM outputs pixel-by-pixel and write PNG copies. usage: python compare.py a.ppm b.ppm [...]"""
import sys
import numpy as np
from PIL import Image

imgs = [(p, np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)) for p in sys.argv[1:]]
for p, a in imgs:
    Image.fromarray(a.astype(np.uint8)).save(p[:-4] + ".png")
ref_path, ref = imgs[0]
for p, a in imgs[1:]:
    if a.shape != ref.shape:
        print(f"{p}: SHAPE MISMATCH {a.shape} vs {ref.shape}")
        continue
    d = np.abs(a - ref)
    print(f"{p} vs {ref_path}: max_abs={d.max()} differing_px={(d.max(axis=2) > 0).sum()} of {d.shape[0]*d.shape[1]} mean_abs={d.mean():.5f}")
