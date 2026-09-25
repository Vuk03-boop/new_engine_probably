"""3G P: the M3 performance series in the M1 method (docs/changes/2026-09-25-phase3g-gate.md).

usage: python gate_perf.py <out dir>   (from engine/, after `cargo build --release -j 2 -p viewer`)

Release viewer, validation off, 1920x1080, the light view with every default, MAILBOX, greedy /
chunk, the looping walk, 10,000 frames per run. Arms are interleaved in each repetition (the order
rotates by one arm per repetition). P1: every undisturbed run of a pass arm has p99 frame interval
<= 16.67 ms, exit 0, 0 overlaps, 0 falls; runs with focus-lost or occluded events are excluded; each
pass arm needs 3 undisturbed runs (at most 6 repetitions). nvidia-smi samples every 250 ms per run.
Added after the first series (keys reached the viewer without a focus change, 3G record): a run is also
disturbed if the viewer received any key, mouse-button or wheel input, or if its view or lighting
settings at exit differ from the arm's.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

VIEWER = Path("target/release/viewer.exe")
COMMON = ["--size", "1920x1080", "--frames", "10000", "--walk"]
PASS_ARMS = [("dawn", ["--hour", "6.25"]), ("midday", ["--hour", "12"]), ("dusk", ["--hour", "17.75"])]
DATA_ARMS = [("run_day", ["--hour", "6.25", "--run-day"]), ("fifo_midday", ["--hour", "12", "--present", "fifo"])]
MIN_REPS, MAX_REPS, NEED = 3, 6, 3
LIMIT_MS = 16.67


def run(out, rep, name, extra):
    log = out / "gate.jsonl"
    before = log.read_text().count("\n") if log.exists() else 0
    smi_path = out / f"nvidia_smi_{rep}_{name}.csv"
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.gr,power.draw,temperature.gpu,clocks_throttle_reasons.active", "--format=csv", "-lms", "250", "-f", str(smi_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    env = dict(os.environ, NE_NO_VALIDATION="1")
    cmd = [str(VIEWER), *COMMON, *extra, "--log", str(log), "--trace", str(out / f"trace_{rep}_{name}.csv")]
    with open(out / "gate.log", "a") as f:
        f.write(f"== rep {rep} {name}: {' '.join(cmd)}\n")
        f.flush()
        code = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"== exit {code}\n")
    smi.terminate()
    smi.wait()
    lines = log.read_text().splitlines() if log.exists() else []
    j = json.loads(lines[-1]) if len(lines) > before else {}
    smi_rows = [l.split(", ") for l in smi_path.read_text().splitlines()[1:]] if smi_path.exists() else []
    walk = j.get("walk", {})
    fm = j.get("frame_ms", {})
    shade = j.get("gpu_ms", {}).get("shade", {})
    rec = {
        "rep": rep,
        "arm": name,
        "exit": code,
        "p50": fm.get("p50"),
        "p99": fm.get("p99"),
        "max": fm.get("max"),
        "fps": j.get("fps"),
        "over_16_7": j.get("phases_ms", {}).get("over_16_7"),
        "over_33_3": j.get("phases_ms", {}).get("over_33_3"),
        "shade_p50": shade.get("p50"),
        "shade_p99": shade.get("p99"),
        "shade_max": shade.get("max"),
        "gbuffer_p50": j.get("gpu_ms", {}).get("gbuffer", {}).get("p50"),
        "present": j.get("present"),
        "hour_at_exit": j.get("hour"),
        "sky_correction": j.get("sky_correction"),
        "bounce": j.get("bounce"),
        "denoise": j.get("denoise"),
        "validation": j.get("validation"),
        "focus_lost": j.get("focus_lost"),
        "occluded": j.get("occluded"),
        "input_events": j.get("input_events"),
        "view": j.get("view"),
        "accumulate": j.get("accumulate"),
        "overlaps": walk.get("overlap_frames"),
        "falls": walk.get("respawns"),
        "loops": walk.get("script_loops"),
        "pstates": sorted({r[1] for r in smi_rows if len(r) > 5}),
        "throttle": sorted({r[5] for r in smi_rows if len(r) > 5}),
        "gr_clock_min": min((int(r[2].split()[0]) for r in smi_rows if len(r) > 5), default=None),
        "temp_max": max((int(r[4]) for r in smi_rows if len(r) > 5), default=None),
    }
    hour_ok = "--run-day" in extra or abs((rec["hour_at_exit"] or -1) - float(extra[extra.index("--hour") + 1])) < 1e-3
    settings_ok = rec["view"] == "light" and rec["bounce"] and rec["denoise"] and rec["accumulate"] and rec["sky_correction"] and hour_ok
    rec["disturbed"] = bool(rec["focus_lost"]) or bool(rec["occluded"]) or rec["input_events"] != 0 or not settings_ok
    rec["pass"] = code == 0 and rec["p99"] is not None and rec["p99"] <= LIMIT_MS and rec["overlaps"] == 0 and rec["falls"] == 0
    return rec


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms = PASS_ARMS + DATA_ARMS
    undisturbed = {n: 0 for n, _ in PASS_ARMS}
    for rep in range(1, MAX_REPS + 1):
        k = (rep - 1) % len(arms)
        for name, extra in arms[k:] + arms[:k]:
            rec = run(out, rep, name, extra)
            if name in undisturbed and not rec["disturbed"]:
                undisturbed[name] += 1
            with open(out / "analysis.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        if rep >= MIN_REPS and all(v >= NEED for v in undisturbed.values()):
            break
    recs = [json.loads(l) for l in (out / "analysis.jsonl").read_text().splitlines()]
    verdict = {}
    for name, _ in PASS_ARMS:
        runs = [r for r in recs if r["arm"] == name and not r["disturbed"]]
        verdict[name] = {"undisturbed": len(runs), "p99": [r["p99"] for r in runs], "all_pass": all(r["pass"] for r in runs), "enough": len(runs) >= NEED}
    p1 = all(v["all_pass"] and v["enough"] for v in verdict.values())
    summary = {"summary": verdict, "p1": "pass" if p1 else "FAIL"}
    with open(out / "analysis.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
