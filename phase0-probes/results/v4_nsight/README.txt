Nsight Graphics 2026.3.1 GPU Trace attempt, 2026-09-22 (rt_probe small scene, PROBE_NO_VALIDATION=1).
small.out.log: ngfx launched and attached to rt_probe, then the target reported
"GPU Performance Counters unavailable. Please enable access to GPU performance counters."
No trace was produced. Counter access is restricted to administrators by the NVIDIA driver by default.
Earlier attempt passing --platform "Windows (x86_64)" failed before launch: Qt parsed --platform as its own
QPA plugin option (err: Could not find the Qt platform plugin "windows (x86_64)"). Omit --platform.

2026-09-23, after the user rebooted (counter access enabled, RmProfilingAdminOnly=0, Nsight GUI closed):
- small/: GPU Trace succeeded (--start-after-ms 0 --max-duration-ms 5000). 9.5 MB .ngfx-gputrace plus exported BASE/*.xls.
- bistro_tex/: succeeded on the 5th attempt with PROBE_HOLD_MS=12000, --start-after-ms 3000 --max-duration-ms 9999. 108 MB report.
  Failures before that:
  (a) --max-duration-ms 20000 is rejected; the maximum is below 10000.
  (b)-(c) The start time was too late and/or the process exited mid-trace. A diagnostic run on the small scene, exiting mid-trace, also failed, so exiting during a trace loses the report.
  (d) With the hold but a 10000 ms start: "Data fetching never started". The timer counts from attach (~3 s uptime), so the trace began after the last GPU submit.
- Nsight locks GPU clocks to base: trace median 0.669 ms (small) and 2.12 ms (Bistro tex), against 0.59 / 1.87 ms unprofiled. Do not compare profiled and unprofiled timings.
- Exported xls metrics are whole-window averages ("Async Compute Triage" metric set). Bistro window: 88 dispatches, 1.93e9 SM instructions,
  L1 hit 64.0%, L2 hit 69.7% (small: 95.7% / 93.7%). Per-dispatch data needs the GUI (open the .ngfx-gputrace in ngfx-ui).
- REPRO_INFO "VRAM Requested 41 MiB" in the Bistro trace is presumably a snapshot at attach time, before uploads; the probe's own measurement is 1331.5 MiB. Not investigated.
