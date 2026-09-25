# Change: Phase 3G — the M3 gate

Status: **in progress.** Criteria declared 2026-09-25, before any gate run.
Date and baseline: 2026-09-25, after the sky correction (S-020); no Git.
Authorization: S-017 (slices 3A–3G); started by the user ("3g is also greenlit the meachine is free right now"). FLIP installed with the user's permission (A-008).

## Objective

Show that the M3 target holds as one system: *"the street lit by sun and sky (dawn to dusk), with real shadows and one bounce, 1080p 60 fps, checked against a reference"* ([Phase 3 proposal](2026-09-24-phase3-proposal.md) §3G). Every lighting default is on: sun, sky with the S-020 correction, one bounce, the temporal pass and the filter.

## Criteria (declared before running)

### P — performance (the M1 method, [M1 record](2026-09-23-m1-walk-shade-gate.md) §3b)

- Release, validation off (`NE_NO_VALIDATION`), 1920×1080, the light view with every default, MAILBOX (S-014), greedy / chunk, the looping walk (`--walk`), 10,000 frames per run.
- **Arms**, interleaved in each repetition:
  - pass arms: dawn (`--hour 6.25`), midday (`--hour 12`), dusk (`--hour 17.75`);
  - data arms: the running day from dawn (`--hour 6.25 --run-day`: the sun moves, histories are capped by sun motion), and FIFO at midday.
- **P1:** in every undisturbed run of a pass arm, the p99 frame interval is ≤ 16.67 ms, with exit 0, 0 overlaps and 0 falls. Runs with focus-lost or occluded events are excluded; each pass arm needs 3 undisturbed runs (at most 6 repetitions).
- **Data:** p50, fps lows, per-pass GPU time (G-buffer, the shade slot: shade, temporal and filter), GPU clocks and throttling (nvidia-smi).
- If P1 fails, the record reports the per-pass cost and the user chooses (proposal decision 2). Lowering resolution or sample count is an appearance change and needs the user's approval.

### Q — quality against the reference

- **Reference:** `gpu::reference` with `max_bounces` 1 (the real-time path's transport: sun, sky, one bounce) at 1920×1080. 4,096 samples per pixel; 16,384 at twilight.
- **Real-time:** the shown image, with the viewer's settings (bounce on, temporal defaults with `max_age` 64, filter defaults, sky corrected), from a fresh history. The frame index is the age.
- **Cameras:**
  - stills: the 3A reference cameras (street, low) × the five reference times (6.25, 8, 12, 17.75, 18.25 h);
  - motion: the 3E D4 path (street camera, 0.25 voxel sideways and 0.2° of yaw per frame) at 8 h and 17.75 h, frames 8, 16 and 32.
- **Metrics:**
  - relative MSE and mean-luminance bias as in 3E (`error`), over surface pixels;
  - LDR-FLIP 1.7 over the whole image. The image uses the viewer's display mapping (exposure from sun and sky, the ACES fit, sRGB) and FLIP's default 67 pixels per degree.
- **Q1 energy (stills):** the shown mean luminance is within ±2% of the reference at ages 1, 4, 16 and 64, in all 10 arms (the proposal's budget).
- **Q2 error (stills):** the filtered relative MSE against the reference is ≤ the raw one at gain × age, with gains 8 / 4 / 4 / 1 at ages 1 / 4 / 16 / 64, in all 10 arms. This is 3E D1 moved onto the reference, with the bounce, the low camera and 1080p.
- **Q3 motion energy:** at frames 8, 16 and 32 of both paths, the filtered mean luminance is within 1% of the raw one in the same frame and within 2% of the reference. The motion relative MSE is data: D4 stays failed as accepted (S-018) and is not restated.
- **Q4 FLIP:** the filter never looks worse than no filter: filtered FLIP ≤ raw FLIP at the same age (ages 1, 16, 64, all 10 still arms) and in the same frame (motion frames 8, 16, 32). The FLIP values themselves are data; this is the first FLIP measurement, so there is no basis for an absolute limit.

### E — edits

- **E1 edit budget** (S-016):
  - in the viewer: release, validation off, 1080p, the light view with every default, the scripted fly path;
  - the new `--edit-size N` (a box of N³ voxels) with N = 1, 8, 32; 2,000 frames, so 50 removals and 50 restorations;
  - 2 interleaved runs per size.
  - Pass: in every run, p95 edit-to-visible ≤ 50 ms (1 and 8³) and ≤ 100 ms (32³), every edit shown, and 0 deferred.
- **E2 relight with the bounce:** 3E R1 and R2 re-run with the bounce on: the same edit and thresholds, and the controls without boxes must still fail.
- **E3:** the viewer with box edits of every size and validation on: exit 0, 0 errors and 0 warnings.

### T — the reset-frame tail (S-019), data

- GPU time of the temporal pass and the filter at 1080p on frames 1–8 after a full reset, against the steady state (ages ≥ 9).
- Intervals over 16.67 ms in the P runs.
- No pass limit beyond P1. The held levers are applied only if P1 fails because of the tail: RGBA16F buffers, a cheaper young-pixel variance, tiling.

### U — the user's walk

The user walks the street and judges it. This closes M3.

**Work units (3):** P 1, Q and E 1, U 1.

**Budget:** the references take about 45 min of GPU once (cached in the scratch directory by the test when `NE_GATE_DIR` is set); P about 25 min; E about 5 min. Each criterion runs once. Failures are diagnosed and reported, never rebaselined.

## Results

### Q — quality (validation on, 0 errors / 0 warnings)

Logs in `engine/results/`: `test_gpu_3g_stills.log` (exit 101: Q2 fails in one arm), `test_gpu_3g_motion.log`, `phase3g/flip.jsonl`, `phase3g/flip.log`. Images: `phase3g/stills_sheet.png` (rows: morning, dusk, twilight × street, low; columns: reference, raw age 1, filtered age 1, filtered age 64); FLIP error maps `phase3g/flip_*`.

- **References:** 1920×1080, one bounce, 4,096 samples per pixel (about 90–100 s each), and 16,384 at twilight (300–330 s). Their own noise in the metric is 0.5–3.9 × 10⁻³ relative MSE; it adds equally to every arm's error.

| # | Result |
|---|---|
| Q1 | **pass**, all 10 arms × 4 ages. Worst bias −1.47% (low camera, twilight, age 1). Day arms are within 0.43%; twilight is −0.8% to −1.5%, of which −0.74% to −0.87% is already in the raw accumulation, i.e. the known twilight sky bias (S-020 record). |
| Q2 | **fail in 1 of 10 arms** (38 of 40 checks pass): low camera at midday, ages 16 and 64. Filtered 0.0098 against raw-at-64 0.0077 at age 16; filtered 0.0090 against raw 0.0077 at age 64. All other arms pass with a margin: at age 64 the filter is 4.6–9.8× below raw (street at midday: 0.0016 against 0.0162). |
| Q3 | **pass**, both paths, frames 8 / 16 / 32. The filtered bias is within 0.4% of raw and 0.21% of the reference. Data: at 1080p the filtered motion error is 4.6–13× below raw (morning frame 32: 0.0032 against 0.0163). |
| Q4 | **pass**, 36 of 36 comparisons. For example, street at morning, FLIP raw / filtered: age 1 0.084 / 0.047, age 16 0.031 / 0.021, age 64 0.021 / 0.016. The failing Q2 arm (low, midday) also passes: age 64 raw 0.0227, filtered 0.0196. Twilight shows the highest FLIP (age 64 filtered 0.048–0.050), in part from the reference's own noise. |

**Q2 diagnosis** (logs `test_gpu_3g_diag_q2.log`, `test_gpu_3g_sweep_prefilter_*.log`, test `diagnose_q2`):

1. **Where:**
   - 0.0084 of the 0.0090 at age 64 lies on lighting-edge pixels (the reference's 3×3 luminance max / min > 1.5; 4.6% of the surface);
   - it is a thin band along the far shadow edge across the road, seen at a grazing angle (`phase3g/q2_low_midday_where.png`: reference, filtered age 64, and red where filtered is further from the reference than raw, blue where closer);
   - elsewhere the filter reduces the error (rest: 0.0006 filtered against 0.0068 raw).
2. **Hypothesis 1, refuted by reading the code:** "the edge-stopping variance does not shrink with age". The shader already divides it by the age.
3. **Hypothesis 2, confirmed by varying one setting at a time:**
   - only `prefilter_levels 0` removes the excess (age 64: 0.0009);
   - `variance_blur` off, `sigma_l` 2 or 1 and `levels` 1 leave it at 0.0076–0.0094;
   - the control `levels 0` reproduces raw exactly.
   - So the level-1 guide prefilter (3E: 3×3 same-surface means) straddles hard lighting edges and blurs converged images there.
4. **Fix attempt:** `DenoiseSettings::prefilter_age`. Pixels at or above that age compare their own values; the default is `u32::MAX`, the unchanged 3E filter, and the control reproduces the earlier numbers exactly. The sweep over 2 / 4 / 8 / 16 / 32 shows the tradeoff the prefilter was made for:
   - it removes the Q2 excess (low midday at age 64 0.0009, at age 16 0.0018);
   - but it brings back the energy loss: street twilight bias at age 16 −2.14% (over Q1's 2%), at age 64 −1.12%; street dawn at age 16 −0.95%.
   - No age threshold satisfies both. After two hypotheses and one fix attempt the method stops here: the default is unchanged and the choice goes to the user (below).
5. **For the user's look:** the viewer's P key switches between the 3E filter and `prefilter_age` = `--prefilter-age N` (default 8), and the title shows which; `--camera low` starts flying at the low camera. The default is still the 3E filter. Smoke run: `--camera low --hour 12 --prefilter-age 8 --frames 120` with validation on, exit 0.
6. **User decision (S-021, 2026-09-25):** after comparing with P the user saw no difference and kept the 3E filter; Q2 stays failed as measured and is a known limitation.

### T — the reset-frame tail (validation off, `test_gpu_3g_reset_cost.log`)

- Street camera at 8 h, 1080p, 6 fresh histories after a warm-up.
- **Filter:** 3.83 ms at ages 1–7 against 3.05 ms from age 8. The difference (+0.8 ms) is the 7×7 neighbour variance, which is used until `moments_age`.
- **Shade and temporal:** 1.66 ms and 1.31 ms at every age (temporal 0.92 ms on the first frame).
- **Frame slot:** 6.8 ms after a reset against 6.0 ms steady.
- The ~20 ms frames seen in 3E / 3F viewer runs are therefore **not** explained by the young-pixel variance. P shows whether they occur on the walk.

### P — performance, series 1 (stopped: disturbed)

Evidence: `engine/results/phase3g/perf_series1_disturbed/` (`analysis.jsonl`, `gate.jsonl`, `gate.log`, traces, nvidia-smi).

- **Undisturbed pass-arm runs, all passing:** dawn p99 10.75 and 12.12 ms; dusk 11.39 and 11.03 ms.
- **Midday:** both runs were excluded, for focus loss (10.58 ms; 16.79 ms).
- **Frame p50** is 6.8–7.2 ms (about 140 fps); the shade slot p50 is 6.1–6.7 ms.
- **The series was stopped in repetition 3, because input reached the viewer windows:**
  - The run-day run ended with the bounce off, which only the B key does.
  - Dusk in repetition 3 showed the shade slot at 0 ms: a number key had switched to a debug view.
  - The FIFO run had a 74 s gap, the window being minimized.
  - None of these left a focus-lost event, so the M1 disturbance rule could not see them. The two runs above that recorded a focus loss were excluded by that rule anyway.
- **Rule added after the declaration** (the pass limit is unchanged):
  - the viewer's JSON counts key, mouse-button and wheel input (`input_events`);
  - a run is also disturbed if it received any input, or if its view or lighting settings at exit differ from its arm's (`gate_perf.py`).
  - The series is rerun in full under it.
- **Scripted-walk falls in disturbed runs** (run day 5, FIFO 8,232): while minimized, `frame()` returns after `step_camera`, so the script repeats the same step and the walker walks off the world. This is a harness robustness bug in the viewer's scripted mode (debt; undisturbed runs are unaffected).
- **The slow-frame bursts** (7–28 consecutive frames at 20–40 ms, waiting in `begin`):
  - In the dawn trace they fall at the GPU's power and thermal transitions in the first ~10 s: power-capped at about 95 W (reason 0x4), a 0x400 event, then thermal slowdown (0x20) with clocks falling to 1,530 MHz at up to 83 °C.
  - They do not follow the walk loop's position.
  - Together with T, this places the S-019 tail in the laptop GPU's power management rather than in the filter's young pixels. This is an observation from one trace, not a controlled test.

### P — performance, series 2 (under the tightened rule)

Evidence: `engine/results/phase3g/perf/` (`analysis.jsonl`, `gate.jsonl`, `gate.log`, traces, nvidia-smi) and `perf_driver.log` (exit 0). 6 repetitions, 30 runs. The viewer build includes the Q2 comparison switch at its default (`"prefilter_age": 4294967295`, the 3E filter) in every run.

- **P1: pass.** Every undisturbed pass-arm run has p99 ≤ 16.67 ms with exit 0, 0 overlaps and 0 falls:

| Arm | Undisturbed / disturbed | p99 (ms) | p50 (ms) | Max (ms) | 1% low (fps) | Shade slot p50 (ms) |
|---|---|---|---|---|---|---|
| dawn | 6 / 0 | 10.18–11.18 | 6.59–7.37 | 32.3–39.8 | 52.7–68.8 | 5.98–6.74 |
| midday | 3 / 3 | 10.00–10.37 | 7.15–7.43 | 31.7–44.7 | 64.9–72.9 | 6.56–6.82 |
| dusk | 4 / 2 | 9.95–10.20 | 6.93–7.20 | 39.2–39.7 | 59.7–64.4 | 6.34–6.59 |
| run day (data) | 4 / 2 | 10.35–11.10 | 7.11–7.48 | 39.8–41.1 | 68.3–80.5 | 6.52–6.87 |
| FIFO midday (data) | 5 / 1 | 11.48–15.92 | 7.99–8.16 | 43.8–48.7 | 45.2–61.5 | 6.88–7.07 |

- **Disturbed runs** (all by focus loss; no run received input): repetition 1 (midday, dusk, run day, FIFO), 2 (midday, dusk, run day) and 5 (midday). The dusk run of repetition 1 exited 1 with a 522 ms stall and 72 scripted-walk falls: the minimized-window harness bug recorded under series 1 (debt).
- **Margin:** the worst undisturbed pass-arm p99 is 11.18 ms, 5.5 ms under the limit. 24–48 frames of 10,000 per run exceed 16.7 ms (the bursts described under series 1; max ≤ 45 ms).
- **FIFO (data):** every undisturbed run is under 16.67 ms, with less margin (up to 15.92 ms). M1 saw FIFO fail on sky-heavy views; not re-diagnosed here.
- **GPU state:** P0 throughout; clocks fall to 1,530 MHz under thermal slowdown (0x20) at up to 84 °C in every arm, so the pass holds with the laptop at its thermal limit.

### E — edits

- **E2: pass** (`engine/results/test_gpu_3g_denoise.log`, validation 0 / 0), with the bounce on:
  - R1: 920 px relit, and 0 of 57,600 decisions differ from the host rule. The control without boxes differs on 920 (caught).
  - R2: 171 changed px. Four frames after the edit, relight 0.0300 = reset 0.0300; without boxes 0.2996 (caught).
- **The same run re-checks 3E** after the `prefilter_age` setting was added (default = the 3E filter), with identical numbers:
  - D1–D3 pass;
  - S1 0.0130;
  - D4 fails as accepted (0.0238 / 0.0210 / 0.0208, S-018);
  - C1 fails as accepted (3.49 ms median with validation on, S-019).
- **E1: pass** (`engine/results/phase3g/edits/`, `edits_driver.log` exit 0; release, validation off, 1080p, the light view with every default, 2 repetitions per size, interleaved). Every edit applied was shown, 0 deferred:

| Size | Voxels per run | Edits shown | Regions per edit (p50 / max) | Edit-to-visible p50 (ms) | p95 (ms) | Max (ms) | Budget (p95) |
|---|---|---|---|---|---|---|---|
| 1 | 67 | 67, 67 | 2 / 3 | 21.5, 23.7 | 27.7, 29.6 | 34.9, 51.5 | 50 ms |
| 8³ | 17,736 | 66, 66 | 2 / 8 | 23.8, 23.3 | 29.6, 29.0 | 51.2, 53.2 | 50 ms |
| 32³ | 533,510 | 66, 66 | 8 / 12 | 23.7, 24.1 | 31.1, 31.1 | 32.6, 56.8 | 100 ms |

  - 33–34 of 100 requests per run were rejected: nothing under the screen centre on the scripted fly path (sky), as in 2E. They are not edits and are not timed.
  - Acceleration-structure update p50 13–16 ms per edit; frame p99 17.5–20.2 ms in these runs (edits every 20 frames; not a P measurement).
- **E3: pass** (`engine/results/phase3g/edits_validation/`, `edits_validation_driver.log` exit 0): one 400-frame run per size with validation on, exit 0, 0 errors, 0 warnings; 14 edits shown per run, p95 29.3 / 29.7 / 30.4 ms.

### Regression checks after the 3G changes (2026-09-25)

- **3G changed:** `gpu::denoise` (the `prefilter_age` setting, default = the 3E filter), `denoise.slang`, the new `gpu/tests/gate.rs`, `gpu/tests/denoise.rs` (E2 with the bounce), and the viewer (`--edit-size`, `input_events`, the Q2 comparison `P` / `--prefilter-age` / `--camera`).
- **Pure suite:** 137 pass, 0 fail (`engine/results/test_pure_3g.log`).
- **Clippy** (workspace, all targets): exit 0; only the 6 old `world` lints, as after 3F (8 warning lines: the library's 2 repeat in its test build; `engine/results/clippy_3g.log`). *(Corrected 2026-09-25: first written as 8 lints.)* The 2 lints in the new `gate.rs` were fixed (a type alias and an iterator loop; no behaviour change).
- **`-p gpu`** (`engine/results/test_gpu_3g_regress_failfast.log`, exit 101): lib 12, bounce 4 pass; denoise 4 pass, 2 fail as accepted (C1 cost, S-019; D4 motion, S-018). Cargo stops at the first failing test file, so the files after `denoise` did not run in it. A rerun with `--no-fail-fast` was started and **stopped at the user's request** ("do we really need all o that").
  - **NOT RUN after 3G:** `device`, `edit`, `raster`, `ray`, `reference`, `shade`, `sky`, `temporal` (and `gate` as part of the suite). None of them uses the filter, the only engine library code 3G changed; `gate` and `denoise` ran on their own after the change (Q, E2 above). This is the reasoning for not running them, not evidence that they pass.
- **Viewer:** 30 performance runs and 9 edit runs (3 with validation on, 0 / 0) on the final build; smoke run of the Q2 switch with validation on, exit 0.

## Status (2026-09-25)

- **P: pass** (series 2). **Q:** Q1, Q3, Q4 pass; Q2 fails in 1 of 10 arms, accepted as a known limitation by the user (S-021). **E:** E1, E2, E3 pass. **T:** data, recorded.
- **U: met.** The user accepted the result ("I am fine with it", then "Sure i accept it"), S-022.
- **3G closed 2026-09-25; M3 accepted as met.**
