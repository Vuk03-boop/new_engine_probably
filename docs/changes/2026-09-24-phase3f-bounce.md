# Change: Phase 3F — one bounce of diffuse light

Status: **closed 2026-09-24.** B1 and B2 pass; B3, B4 and C are measured (C after the user allowed benchmarks again: "feel free to bench now"). Criteria were declared 2026-09-24, before the bounce was run against the reference; none has been changed.
Date and baseline: 2026-09-24, after 3E closed (S-019) (no Git).
Authorization: S-017 (slices 3A–3G); 3F starts after 3E closes (S-019, user: "Greenlight b)").

## Objective

Light that reaches a surface after one diffuse bounce ([Phase 3 proposal](2026-09-24-phase3-proposal.md) §3F). Without it, shadowed street areas are darker than the reference. That missing light must not be hidden: energy from bounces ≥ 2 stays missing and is reported per reference camera.

## What is built

- **`gpu::shade`, `bounce`** (a new `ShadeSettings` field, on by default; flag `BOUNCE` = 16). The 3C sky ray becomes the reference's continuation ray. When it hits a surface instead of escaping, that point is lit exactly as the reference lights its second vertex with `max_bounces` 1:
  - the sun by next-event estimation, with its own shadow ray;
  - the sky through one more cosine-sampled visibility ray (sky-view table);
  - both times the two albedos.
  - The pixel's PCG32 stream continues in the reference's order.
- **Hit material:** the hit's face, material and snapped point come from the ray query and the region table (`Accel::table`), bound as `regions` (binding 7), as in `reference.slang`. The layout is checked from reflection like every other binding.
- **Rays per pixel:** up to 4 (3C: 2). They are one shadow ray, one closest-hit continuation, and a shadow ray plus a sky-visibility ray at the hit.
- **Unchanged:**
  - the result is added to the shade radiance, so the temporal pass and the filter reconstruct it;
  - no new passes and no new memory.
- The 3B–3E test files pin their shading to `DIRECT` (bounce off), the lighting their criteria were measured with, so their recorded numbers stay comparable.

## Criteria (declared before running)

| ID | Criterion |
|---|---|
| B1 | **Exact, per pixel.** Frame f of the shade pass with the bounce equals the reference's one-bounce sample f (`max_bounces` 1) per pixel, relative 1e-5. This is checked in three arms: uniform sky of 1 with the sun off; the point sun with the sky off; the disk sun with the sky off. It uses three cameras at 480×270, with the sun arms at 8, 12 and 17.75 h. Each arm allows at most 0.1% of pixels mismatched (the 3B/3C edge set: the surface point rebuilt from depth against the ray hit) and 0 bad ids, and at least 1000 pixels must be changed by the bounce (the path is exercised). **Controls:** the bounce off must fail at least half of the arms with at least 10× the correct arms' mismatches, and a different frame must differ. |
| B2 | **Convergence, real lighting.** The shade pass with the bounce is averaged over 256 frames (160×90, street and low cameras). It is compared with the reference with one bounce (Monte Carlo sky, disk sun, 4096 spp). At morning and midday: the image mean per channel within 5% (the 3C table-sky budget), and the **indirect component** (bounce − no bounce) within 5% per channel. Both sides' streams make the indirect difference exact per sample: with and without the bounce, a path differs only at the continuation's hit. Dawn, dusk and twilight are reported as data. The table sky's known bias there is 3C criterion 4, the user's open decision. |
| B3 | **Data: missing energy.** Reference with 8 bounces against reference with 1, per camera and time of day, for the whole image and for the pixels not directly sunlit. This is the energy 3F does not add. |
| B4 | **Data: input noise.** The shade pass's 1-sample relative MSE with and without the bounce, and the energy the bounce adds. This is the filter's input for 3G. |
| C | **Data: cost.** The shade pass at 1080p with and without the bounce, on mains power with validation off. It is judged on the whole frame at the 3G gate (S-019). |
| — | Regression: the 3B–3E tests (pinned to `DIRECT`) pass; the pure suite; clippy; viewer scripted runs validation-clean. |

## Correction during the run

The regression run's `accumulated_images` (3D, data only) used the default shading, so it overwrote `engine/results/temporal_3d_*.ppm` with bounce-lit images. It is now pinned to `DIRECT` and was re-run, and the files again show the 3D lighting (`test_gpu_3f_temporal_images.log`). The 3D record's evidence, `temporal_3d_sheet.png`, was not touched.

## Results

Logs: `engine/results/test_gpu_3f_b1.log`, `test_gpu_3f_b2.log`, `test_gpu_3f_b3.log`. All were validation-clean (0 / 0).

| ID | Result |
|---|---|
| B1 | **pass.** 21 arms (3 cameras × uniform sky, and point and disk sun at 8, 12 and 17.75 h): 39 mismatched pixels in total, at most 7 in one arm (0.005%); 0 bad ids. 1485–17,722 pixels per arm are changed by the bounce. **Controls:** the bounce off fails 21 of 21 arms (130,992 px, 3,359× the correct arms); frame 30 against reference frame 29 mismatches 9,977 px. |
| B2 | **pass** (morning, midday). Surface pixels, per channel R G B. |
| B3 | Data: see below. |
| B4 | Data: see below. |
| C | **Measured** (data, judged at 3G): the bounce adds 0.29–0.77 ms to the shade pass at 1080p (greedy / chunk: street 1.07 → 1.84 ms, low 0.76 → 1.17, overhead 0.59 → 0.90). In the viewer's frame the difference is within run-to-run noise. See *Cost* below |
| — | **Regression** (`test_gpu_3f_regress.log`, cost tests skipped): lib 12, bounce 3, shade 3, sky 3 (+1 ignored, the known 3C failure), temporal 6, denoise 3 pass. **D4 fails as before** (S-018), with identical numbers (frame 8 0.0238 against raw 0.0940), so the `DIRECT` pin keeps the 3E evidence. Clippy exit 0, only the old `world` lints (`clippy_3f.log`). Viewer, validation on: 600 frames with the edit script and the bounce, 300 frames with `--no-bounce`: exit 0, validation 0 / 0, 20 of 20 edits shown (`viewer_3f.jsonl`). The pure suite is NOT RUN (no pure crate changed). The other `gpu` test files (raster, ray, edit, device, reference) do not use `gpu::shade` and were not re-run. |

**B2, the shade pass with the bounce against the one-bounce reference** (256 frames against 4096 spp; twilight 16,384):

| Time | Camera | Image (R G B) | Indirect (R G B) | Bounce share of the reference |
|---|---|---|---|---|
| morning | street | −0.8% −2.0% −3.1% | +0.2% −0.8% −2.2% | 9.2% |
| morning | low | −0.3% −1.0% −1.9% | −0.9% −1.5% −2.5% | 4.6% |
| midday | street | +0.3% −0.6% −1.0% | +0.5% +0.0% −0.4% | 11.5% |
| midday | low | +0.0% −0.2% −0.4% | −0.1% −0.4% −0.7% | 4.2% |
| dawn (data) | street / low | −3.3 −4.4 −4.5% / −2.3 −3.6 −4.0% | −2.0 −3.5 −4.0% / −2.4 −3.8 −4.4% | 7.7% / 5.9% |
| dusk (data) | street / low | −2.7 −4.0 −4.5% / −1.7 −3.1 −3.9% | −1.8 −3.3 −4.0% / −1.8 −3.5 −4.4% | 7.4% / 4.9% |
| twilight (data) | street / low | −0.5 +5.3 +10.6% / −0.3 +5.2 +10.3% | −0.1 +4.7 +9.5% / −0.6 +4.4 +9.1% | 6.8% / 5.2% |

The indirect component carries the same sign and size of error as the whole image at every time. **Update 2026-09-25:** with the sky correction (S-020), B2 re-run as data gives the image within 1.1% and the indirect within 1.7% at every time ([sky correction record](2026-09-24-phase3c-sky-correction.md), K6). So the error at dawn, dusk and twilight is the table sky's known bias (3C criterion 4: dawn/dusk blue −5.7%, twilight blue +15.5%), now seen through one more bounce, not a bounce error.

**B3, missing energy of bounces ≥ 2** (reference with 0, 1 and 8 bounces; 2048 spp, twilight 8192; 160×90):

| Time | Camera | Direct only reaches | One bounce reaches | Missing | Not directly sunlit: one bounce reaches |
|---|---|---|---|---|---|
| dawn | street / low | 90.9% / 93.0% | 98.5% / 98.9% | 1.5% / 1.1% | 98.3% / 98.4% |
| morning | street / low | 89.7% / 94.8% | 98.7% / 99.4% | 1.3% / 0.6% | 98.4% / 98.4% |
| midday | street / low | 87.3% / 95.3% | 98.6% / 99.5% | 1.4% / 0.5% | 98.5% / 98.5% |
| dusk | street / low | 91.2% / 94.2% | 98.5% / 99.0% | 1.5% / 1.0% | 98.2% / 98.3% |
| twilight | street / low | 91.8% / 93.7% | 98.4% / 98.9% | 1.6% / 1.1% | (sun below the horizon: all pixels) |

Before 3F the real-time image missed 5–15% of the light, most of it in the shadows (up to 15.3% there). With one bounce it misses 0.5–1.8%.

**Images:** `engine/results/bounce_3f_sheet.png`: street camera, 8 h and 17.75 h, 256 frames, without and with the bounce (`--ignored bounce_images`). The shadowed street and the shaded facades are lighter; the change is subtle, consistent with the 5–15% the direct image was missing.

**Cost (C)** (validation off, laptop reported plugged in by the user earlier in the day, not re-checked). Logs: `test_gpu_3f_cost.log`, `viewer_3f_cost.jsonl`.

- **Shade pass** (`shade_cost_at_1080p`: both arms in one command buffer, alternating order, a barrier between them; 40 reps; medians, p90 within 0.01 ms):

| Camera | Greedy / chunk: 2 rays → with the bounce | Greedy / 2³ chunks | Greedy / brick | Unmerged / chunk |
|---|---|---|---|---|
| street | 1.065 → 1.838 | 0.916 → 1.599 | 1.020 → 1.942 | 1.008 → 2.082 |
| low | 0.758 → 1.169 | 0.664 → 1.037 | 0.760 → 1.282 | 0.750 → 1.307 |
| overhead | 0.591 → 0.898 | 0.578 → 0.869 | 0.653 → 1.046 | 0.615 → 1.069 |

  The 2-ray arm agrees with the 3C record (0.68–1.07 ms). The default setting stays the cheapest with the bounce, apart from 2³ chunks (as for the other passes).
- **Viewer frame** (1080p, scripted fly path at 8 h, filter on, 1500 frames, 2 interleaved pairs, MAILBOX): with / without the bounce, frame p50 6.10 / 6.01 and 6.23 / 6.18 ms, p99 9.99 / 14.45 and 11.62 / 10.87 ms; shade slot (shade, temporal pass and filter) p50 5.53 / 5.32 and 5.46 / 5.42 ms. The mean difference (about 0.1 ms) is within the pairs' spread. Both arms have 1% lows of 46–61 fps on this path (the reset-frame tail named in S-019, not the bounce), a 3G item.

**B4, input noise** (street camera, 1-sample relative MSE against the 256-frame mean): direct / with the bounce: dawn 0.601 / 0.500, morning 0.570 / 0.460, midday 0.740 / 0.609, dusk 0.833 / 0.653, twilight 1.214 / 0.996. The bounce lowers the relative noise by about 17%. Its light lands where the direct sky ray used to be blocked (0), so fewer samples are black; the mean rises 1–4%.
