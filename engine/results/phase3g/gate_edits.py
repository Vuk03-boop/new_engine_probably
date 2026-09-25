"""3G E1 / E3: the edit budget in the viewer's frame loop (docs/changes/2026-09-25-phase3g-gate.md).

usage: python gate_edits.py <out dir> [--validation]   (from engine/, after building the viewer)

E1 (default): release, validation off, 1920x1080, the light view with every default, the scripted fly
path, `--edit-script --edit-size N` for N = 1, 8, 32, 2,000 frames (50 removals, 50 restorations),
2 repetitions interleaved. Pass: in every run p95 edit-to-visible <= 50 ms (1, 8) and <= 100 ms (32),
every edit shown, 0 deferred.
E3 (`--validation`): one 400-frame run per size with validation on: exit 0, 0 errors, 0 warnings.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

VIEWER = Path("target/release/viewer.exe")
SIZES = [1, 8, 32]
BUDGET = {1: 50.0, 8: 50.0, 32: 100.0}


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else None


def main():
    out = Path(sys.argv[1])
    validation = "--validation" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    tag = "e3" if validation else "e1"
    log = out / f"edits_{tag}.jsonl"
    reps = 1 if validation else 2
    frames = "400" if validation else "2000"
    env = dict(os.environ)
    if validation:
        env.pop("NE_NO_VALIDATION", None)
    else:
        env["NE_NO_VALIDATION"] = "1"
    results = []
    for rep in range(1, reps + 1):
        for n in SIZES if rep % 2 else SIZES[::-1]:
            before = log.read_text().count("\n") if log.exists() else 0
            cmd = [str(VIEWER), "--size", "1920x1080", "--frames", frames, "--edit-script", "--edit-size", str(n), "--log", str(log)]
            with open(out / f"edits_{tag}.log", "a") as f:
                f.write(f"== rep {rep} size {n}: {' '.join(cmd)}\n")
                f.flush()
                code = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
                f.write(f"== exit {code}\n")
            lines = log.read_text().splitlines()
            j = json.loads(lines[-1]) if len(lines) > before else {}
            e = j.get("edits", {})
            vis = e.get("visible_all_ms", [])
            rec = {
                "rep": rep,
                "size": n,
                "exit": code,
                "validation": j.get("validation"),
                "errors": j.get("validation_errors"),
                "warnings": j.get("validation_warnings"),
                "applied": e.get("applied"),
                "shown": e.get("shown"),
                "rejected": e.get("rejected"),
                "deferred": e.get("deferred"),
                "not_shown": e.get("not_shown_at_exit"),
                "voxels": e.get("voxels"),
                "regions": e.get("regions"),
                "visible_p50": sorted(vis)[len(vis) // 2] if vis else None,
                "visible_p95": p95(vis),
                "visible_max": max(vis) if vis else None,
                "accel_ms": e.get("accel_ms"),
                "frame_p99": j.get("frame_ms", {}).get("p99"),
            }
            if validation:
                rec["pass"] = code == 0 and rec["errors"] == 0 and rec["warnings"] == 0
            else:
                rec["pass"] = code == 0 and rec["visible_p95"] is not None and rec["visible_p95"] <= BUDGET[n] and rec["deferred"] == 0 and rec["not_shown"] == 0 and rec["applied"] == rec["shown"]
            results.append(rec)
            with open(out / f"edits_{tag}_analysis.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    summary = {"summary": tag, "pass": all(r["pass"] for r in results)}
    with open(out / f"edits_{tag}_analysis.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
