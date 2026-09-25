# Change: Phase 3E — native reconstruction and lighting change

Status: **closed 2026-09-24.** All quality criteria pass except D4 (motion; gap accepted, S-018). C1 (cost) fails on mains, 3.47 ms against 3.0 ms after optimization; the user accepted about 3.5 ms for now, and cost is judged on the whole frame at the 3G gate (S-019). Neither failed criterion is restated. Criteria declared 2026-09-24, after measuring the unfiltered input and before running the filter; none has been changed since.
Date and baseline: 2026-09-24, after 3D (no Git).
Authorization: S-017 (slices 3A–3G); resumed by the user after the pause ("continue now where you left off").

## Objective

Show a clean image from the 1-sample-per-pixel lighting, without hiding light, blurring across edges or lagging behind edits and the moving sun ([Phase 3 proposal](2026-09-24-phase3-proposal.md) §3E, decision 3). Close the 3D known limitation (lighting that changes under an accepted history).

## What is built

- **`gpu::denoise`** + `shaders/denoise.slang`: SVGF-style (Schied et al. 2017):
  - demodulate by the material albedo;
  - variance from the history moments (temporal pass, w channel) once the history is `moments_age` samples old (default 8; SVGF uses 4), before that from the 7×7 same-surface neighbours;
  - `levels` à-trous levels (default 2; steps 1, 2, …, 5×5 B-spline). A tap counts only on the **same surface** (face, material, integer plane: the exact ADR-0006 guides, so no depth or normal tolerance is involved), weighted by luminance edge-stopping `exp(-|Δl| / (σ_l · sqrt(variance)))` (σ_l 4) with the variance pre-blurred 3×3;
  - **guide prefilter** (`prefilter_levels`, default 1): the first level's edge-stopping compares 3×3 same-surface means of luminance (σ scaled to their variance, var/9) instead of the taps' own values;
  - remodulate into the shade radiance buffer.
  - The history stays unfiltered: the running mean is exact (3D) and the filter only shapes what is shown.
  - Planted fault `ignore_guides` mixes all surface pixels.
- **Temporal pass** ([ADR-0006 Amendment 1](../adr/ADR-0006-guides-and-history.md)):
  - **relight boxes:** an edit lists its voxel boxes for the frame after it is published. A pixel whose history would be accepted restarts, with the new reason **relit**, when the ray from its surface point towards the sun crosses the box (grown by 1 voxel), or when it lies within the box's sky radius (4 × the largest side + 1 voxel).
  - **sun-motion age cap:** the history spans at most `sun_tolerance_deg` (default 0.5°) of sun motion, so the cap is 0.5° / (the sun's motion this frame).
  - The existing global reset (sun moves > 1° in one frame) stays.
  - **history resampling:** Catmull-Rom over the 4×4 previous pixels when every tap with a weight is valid, clamped to the 2×2 range; otherwise bilinear as in 3D (`TemporalSettings::bilinear` restores 3D). The moment channel is resampled as each tap's variance, and the moment is rebuilt from the resampled mean.
- **Memory:** filter buffers 44 B/px (two 16 B ping-pong buffers, a 4 B luminance guide, an 8 B compact surface key), `GpuTemporal`: 91 MB at 1080p.

## Measured input (before the filter)

Log: `engine/results/test_gpu_3e_raw.log`. Street camera, 320×180, 43,288 surface pixels, of which 13,197 are **edge pixels** (within 2 px of a different surface or the sky). Errors are measured against the same estimator converged over 2048 frames. That target's own noise: relative MSE 1.2–1.7 × 10⁻³ between two independent targets, so about 0.7 × 10⁻³ each.

| Hour | Age 1 | 4 | 16 | 64 | Edge pixels at 1 / 64 |
|---|---|---|---|---|---|
| 8 | 1.25 | 0.311 | 0.076 | 0.019 | 2.04 / 0.031 |
| 12 | 1.35 | 0.331 | 0.080 | 0.020 | 2.30 / 0.033 |
| 17.75 | 1.44 | 0.378 | 0.096 | 0.025 | 2.06 / 0.037 |
| 18.25 | 1.78 | 0.444 | 0.111 | 0.029 | 2.41 / 0.040 |

The raw relative MSE falls as about 1.3–1.8 / age. Its mean-luminance bias is noise (within ±0.9% at age 1).

## Criteria (declared before the filter runs)

**Metrics.**

- **Relative MSE:** per channel, (x − t)² / (t² + ε), with ε = (0.1 × the channel's mean)². Averaged over surface pixels, or over edge pixels.
- **Bias:** the relative difference of the mean luminance.
- **Equivalent-sample gain:** a filtered frame at age a with gain g has a relative MSE no larger than the raw accumulation at age g·a.

| # | Criterion |
|---|---|
| D1 | Stills (street camera, hours 8, 12, 17.75, 18.25): gain ≥ 8 at age 1 (filtered ≤ raw at age 8), ≥ 4 at ages 4 and 16, and ≥ 1 at age 64 (the filter never makes a converged image worse). |
| D2 | Energy: at every age and hour, the filtered mean luminance is within 1% of the raw one in the same frame, and within 2% of the target (proposal budget). |
| D3 | Edges: at age 64 on edge pixels, filtered ≤ raw. The planted `ignore_guides` must fail D3. |
| D4 | Motion (8 h: 0.25 voxel sideways and 0.2° of yaw per frame, 32 frames): at frames 8, 16 and 32, filtered relative MSE ≤ ¼ of the raw one in the same frame, and D2's 1% energy rule holds. |
| R1 | Relight decisions: after an edit, the pixels the GPU marks relit equal a host re-derivation of the ADR rule on every pixel whose history is otherwise accepted, with ≤ 0.05% of pixels differing and ≥ 100 relit. The same check against an arm run without boxes must fail. |
| R2 | Relight lag: take the pixels whose converged value changed by > 25% because of the edit, outside the rebuilt regions. 4 frames after the edit, their filtered relative MSE is ≤ 3 × that of an arm fully reset at the edit. The arm without boxes must fail this. |
| S1 | Sun motion (from 8 h at 0.0625°/frame, the viewer's day speed at 60 fps, 128 frames): at the last frame, filtered relative MSE with the cap < without it (tolerance ∞), and ≤ 2 × the still filtered error at age 8. |
| C1 | Cost: the filter at 1080p ≤ 3.0 ms median (the frame so far is about 2.9 ms of the 16.7 ms budget). |
| — | Full `-p gpu` release, the pure suite, clippy, viewer scripted runs: exit 0, validation clean. |

**Ablations, reported with no pass limit:** unfiltered, temporal only (the 3D accumulation), spatial only (max_age 1 + filter), and both. Swept sliders: levels 0–6, σ_l 1–16 and max_age 16–128. The defaults are taken from the measured plateau.

**NOT RUN by decision:** FLIP (A-008). Its download waits for the user's permission, so the metric here is relative MSE and bias only.

## Results

Logs in `engine/results/`: first run `test_gpu_3e_denoise.log`; final `test_gpu_3e_denoise_run2.log`; sweeps `test_gpu_3e_diag1.log` (first), `test_gpu_3e_sweep2.log`, `test_gpu_3e_sweep3.log`; motion diagnostics `test_gpu_3e_diag2.log`–`diag7`; `test_gpu_3e_temporal_run2.log`, `test_pure_3e.log`, `clippy_3e.log`, `viewer_3e.jsonl`. Images: `denoise_3e_sheet.png` (8 h and 17.75 h; raw at 1, filtered at 1, filtered at 64, target); `motion_3e_explained.png` (D4: the motion path's frame 32 held still and moving, error heat maps, and the hard-sun shadow edge smeared by the moving history; test `motion_images`, log `test_gpu_3e_motion_images.log`).

**First run (SVGF defaults, 5 levels):** D1, D3, R1 and S1 passed. D2 failed (filtered 4–10% darker at ages 1–4). D4 failed. R2 measured nothing: the box around the median occluder did not let the sun in, because thick walls still shadowed through the hole. The **test setup** was corrected to pick the occluder whose box frees the most shadow samples. The criterion is unchanged.

**Diagnosis and changes (each measured on its own):**

| Finding | Evidence | Change |
|---|---|---|
| D2: weights that depend on the samples they average lose energy on heavy-tailed 1-spp input | no edge-stopping: bias +1%; levels 0 reproduce raw exactly | level-0 guide prefilter: bias at 8 h age 1 −4.0% → +0.3%, 18.25 h −9.8% → −1.2% (errors unchanged). Prefiltering later levels made it **worse** (−1.7%, −2.1%; refuted, kept at 1) |
| D2 at age 4: a 4-sample moment variance grows with the pixel's own bright sample | loss appears where moments take over | `moments_age` 8: 18.25 h age 4 −1.1% → +0.3% |
| Levels: 5 was SVGF's value, not ours | sweep: 2–3 levels beat 5 at ages 16–64 and under motion; 3 levels fail D2 at dusk age 1 (−1.2% vs raw) | `levels` 2 |
| D4: moving history, not view | same end view held still: filtered 0.0084; moving: 0.0272 | Catmull-Rom resampling (0.0272 → 0.0254); variance resampling (→ 0.0242) |
| D4 remainder | deterministic hard-sun input: still error 0, moving blurs a shadow edge 2–3 px after 32 frames; `max_age` 4 under motion is no better (0.0276) | none found within the method |

**Final (defaults: levels 2, σ_l 4, prefilter 1, moments_age 8, Catmull-Rom):**

| Hour | Age 1 filtered (limit: raw at 8) | Age 4 (raw at 16) | Age 16 (raw at 64) | Age 64 (raw at 64) | Edges at 64 filtered / raw / `ignore_guides` |
|---|---|---|---|---|---|
| 8 | 0.101 (0.155) | 0.037 (0.076) | 0.0099 (0.019) | 0.0047 (0.019) | 0.012 / 0.031 / 0.84 |
| 12 | 0.136 (0.164) | 0.056 (0.080) | 0.0098 (0.020) | 0.0044 (0.020) | 0.011 / 0.033 / 0.77 |
| 17.75 | 0.089 (0.201) | 0.038 (0.096) | 0.018 (0.025) | 0.0088 (0.025) | 0.022 / 0.037 / 1.25 |
| 18.25 | 0.096 (0.229) | 0.036 (0.111) | 0.015 (0.029) | 0.0084 (0.029) | 0.020 / 0.040 / 1.61 |

| # | Result |
|---|---|
| D1 | **pass** at every hour and age (12 h and 17.75 h were not in the sweep: held out) |
| D2 | **pass**: filtered − raw bias within −0.44% … +0.11%; filtered bias ≤ 0.63% |
| D3 | **pass**; the planted `ignore_guides` fails it at every hour |
| D4 | **fail, accepted as a known limitation (S-018)**: frame 8 0.0238 vs raw 0.0940 (3.95×, needs 4×), frame 16 0.0210 vs 0.0523 (2.5×), frame 32 0.0208 vs 0.0341 (1.6×); energy within 1% |
| R1 | **pass**: 920 px relit, 0 of 57,600 decisions differ from the host; the control without boxes differs on 920 (caught) |
| R2 | **pass**: 172 changed px; 4 frames after the edit relight 0.0319 = reset 0.0319; without boxes 0.312 (caught) |
| S1 | **pass**: capped 0.0130 < uncapped 0.0695; ≤ 2 × still at age 8 (0.0165) |
| C1 | **fail** (mains, validation off): 3.47 ms median, p90 3.60 (limit 3.0). See *Cost* below |
| — | temporal (3D) 7/7; pure suite 133 pass, 2 ignored; clippy clean apart from the 6 old `world` lints; viewer: all views with a resize, 53 scripted edits at 1080p, a walk through sunrise with the day running: all exit 0, validation 0 / 0, no overlaps. Edit-to-visible p99 43 ms (budget 50; 16.4 ms in 3D on mains): battery, to be re-measured |

**D4, open for the user.** A moving history is resampled every frame. That blurs fine lighting detail and makes the remaining noise spatially correlated (blotches), which a per-pixel spatial filter removes much less well than independent noise. The filtered moving error (about 0.021) is what a still camera reaches after 4–6 frames, whatever `max_age` is. The criterion compares against the raw moving image, which that blur already makes smoother, so ¼ of it is out of reach at frames 16–32 with this method. Options: (a) keep D4 and treat it as the target of a later, bigger method (for example temporal-gradient history control, A-SVGF), leaving 3E open; (b) restate D4 against the still-camera filtered error at the same frame (motion ≤ k × still) and accept 3E; (c) accept the measured gap as a known limitation.

**User judgment (2026-09-24, S-018).** In the viewer, the motion gap is barely visible while moving, even when looked for. The filtered and unfiltered images are also hard to tell apart while moving (consistent with the 1.6× measured at frame 32). The filter matters most where the history is young: stills (4× at 64 frames), newly revealed areas, the frames after an edit, and the running day (history capped at 8 frames). With H off, the viewer shows the raw 1-sample image: it skips the filter when not accumulating.

## Cost (C1, on mains, validation off)

Logs: `test_gpu_3e_cost.log` (first), `test_gpu_3e_cost_breakdown.log`, `test_gpu_3e_cost_fused.log`, `test_gpu_3e_cost_keys.log`, `viewer_3e_cost.jsonl`.

- **First measurement:** 4.47 ms. Breakdown with stages added one at a time (warm GPU): init + remodulate 1.28, level 1 +1.04, level-1 prefilter +0.39, level 2 +1.06 = 3.77 ms. Every pass reads and writes full-frame 16 B/px buffers, so the filter is **memory-bound**. The first (cold) run read 4.47 ms against 3.77 ms warm.
- **Exact optimizations (image unchanged, every quality number identical, `test_gpu_3e_denoise_run3.log`):**
  - the level-1 guide is computed inside init, and the last level remodulates, saving 2 full-frame passes: 3.68 ms (the cold/warm gap closed);
  - init copies an 8 B same-surface key, so the levels read 8 B guides instead of 16 B: **3.47 ms**.
- **Not adopted:** without the 3×3 variance pre-blur 3.05 ms (changes the image, still over the limit).
- **In the viewer's frame loop** (1080p, fly path, 2 interleaved pairs): the filter adds about 4.1 ms (shade slot median 6.5 / 7.0 ms against 2.4 / 2.9 without it). The slot p99 rises from 3.5–3.8 ms to 11.6–13.3 ms. Likely cause: frames right after a reset, where every pixel is young and takes the 7×7 neighbour variance (not yet measured separately). Frame p50 is 7.0 / 7.6 ms, and p99 18.1 / 14.3 ms.
- **Levers left:**
  - RGBA16F filter buffers, which the Phase 3 proposal's memory budget already planned; they need pre-exposure for range;
  - a cheaper variance for young pixels (the reset-frame tail);
  - tiling the levels in shared memory.

**C1 decision (user, 2026-09-24, S-019): option (b).** Accept about 3.5 ms now and judge cost on the whole frame at the 3G gate (1080p 60, p99); the levers above are held for 3G. Evidence the user had: their own 1080p session with the filter on (`fps` added to the viewer for it) averaged 223 fps, 10% low 130, 1% low 70, frame p99 8.3 ms. That session also had a run of 8 frames at about 20 ms (the CPU waiting on the GPU; shade slot max 20.6 ms), most likely the reset-frame tail named above; it is a named 3G item, not yet measured separately.
