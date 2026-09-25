# Change: 3C follow-up — the sky correction baked from the reference

Status: **closed 2026-09-25.** K1–K5 pass; K6 and K7 measured. Criteria were declared 2026-09-24, before the bake was run, and none has been changed (the pilots changed only the grid and the sample counts).
Date and baseline: 2026-09-24, after 3F closed (no Git).
Authorization: S-020 (user: "i greenlight A go on"), option A of [3C criterion 4](2026-09-24-phase3c-sky.md).

## Objective

Remove the table sky's known bias (the isotropic multiple-scattering approximation, 3C record) so that sky light on surfaces meets the budget declared in 3C (criterion 4: image mean per channel within 5% dawn to dusk, 15% twilight). The budget is not changed.

## Design

- **What is stored:** the reference's sky radiance, not a ratio. The Monte Carlo sky (`sky_sample`, the reference's estimator and settings) is averaged per direction on a grid of:
  - 32 azimuths × 32 view zeniths, at the sky-view table's own unit coordinates (azimuth toward the sun's side, zenith concentrated at the horizon);
  - 33 sun elevations: −6° to 10° every 1°, then every 5° to 90° (first planned from −10°, changed after the pilots below).
  - The reference's mean and standard error per texel are stored, with the sample count per slice.
- **The ratio** (reference ÷ table, per channel) is computed when the file is loaded, from the current table code (`light::sky`, f64). A change to the table code therefore needs no new bake; a change to the atmosphere or the estimator settings does. Ratios are clamped to [0.5, 2]; where the table value is below 1e-9, the ratio is 1.
- **Applied** when the sky-view table is built (host `SkyLuts::view`, GPU `sky.slang`): each texel is multiplied by the ratio, interpolated bilinearly in the view coordinates and linearly in sun elevation. Below −6° the −6° slice is used. The shade pass, the background and the bounce all read the corrected table. The per-frame cost is zero; the sky-view pass gains one lookup per texel.
- **Uncorrected:** `SkyLuts::new` has a ratio of exactly 1 everywhere, so 3B–3F tests keep their lighting bit for bit.
- **Data file** `engine/light/data/sky_reference_v1.bin` (811 KB), little-endian:
  - the magic, version 1, and the grid;
  - seed, estimator settings, and samples per slice;
  - a fingerprint of the `Atmosphere` and `SkyOptions` fields;
  - the elevations, then the mean and standard error per texel;
  - a checksum.
  - A mismatch refuses the file. The viewer then runs uncorrected, warns, and reports `"sky_correction": false`.
- **Bake tool** `gpu/src/bin/sky_bake.rs`, `gpu::sky_bake`, `shaders/sky_bake.slang`:
  - directions are made on the host (`light::sky_ref::texel`);
  - slices with the same sample count are baked together, each job as a power-of-two number of replicas so that one slice's 1,024 texels still fill the GPU;
  - each thread adds 1,024 samples per dispatch in f32, and the host sums in f64;
  - streams are keyed by the global texel index, with seed `0x5C0`, independent of the tests' reference images.
  - **Samples per texel** (`sky_ref::bake_samples`): 2^20 from 5° up, 2^21 from −2° to 4°, 2^22 at −3°, 2^24 / 2^25 / 2^26 at −4° / −5° / −6°.
- **Viewer:** corrected by default; `--no-sky-correction` for comparison.

## Pilots (before the bake; K1 not changed)

- **Pilot 1** (1,024 samples per texel, the first grid; scratch file, not kept):
  - the estimator is noisy per sample: relative standard error p95 10–17% per texel above the horizon;
  - at −2° to −6° it was 25–100%;
  - from −7° to −10°, up to 2,727 of 3,072 texel values were exactly 0 (almost no path reaches sunlit air). That range cannot be converged at a sane cost, so the grid stops at −6° (the end of civil twilight) and gets a sample count per slice.
- **Pilot 2** (the per-slice schedule ÷ 512): the projected full-scale p95 missed K1 at −3° to −6° (0.9–1.7%) and was borderline at 5° (0.49%). The counts were raised there, to the schedule above; the projection became 0.30% (sun ≥ 0°) and 0.70% (below 0°).
- **Throughput:** 106–151 M samples per second. The full bake is about 160 G samples.

## Criteria (declared before running)

| ID | Criterion |
|---|---|
| K1 | **Bake noise.** Relative standard error of the stored mean, per texel and channel (mean > 0): p95 ≤ 0.5% for sun elevations ≥ 0°, ≤ 1% below 0°. Checked from the file by a pure test. |
| K2 | **Format.** Pure tests: a round trip is exact; a wrong magic, version, size, checksum, atmosphere or estimator setting is refused. The ratio is exactly 1 when uncorrected. |
| K3 | **GPU equals host with the correction.** 3C criterion 1 re-run with the correction: GPU sky-view table against the host's at the five reference times, ≤ 0.5% per texel; the sky pass's layout check (new binding). |
| K4 | **The goal: 3C criterion 4 passes at its declared budget with the correction.** Sky light on surfaces, 256 frames against the reference's direct-sky component (4096 spp, twilight 16,384); image mean per channel within 5% (dawn to dusk), 15% (twilight), street and low cameras. **Control:** the uncorrected run still fails, with the recorded numbers. |
| K5 | 3C criterion 3 (table sky against the Monte Carlo sky on sky-only cameras) passes with the correction; the raw error means are reported. |
| K6 | Data: 3F criterion B2 re-run with the correction (dawn, dusk and twilight were outside 5% only because of this bias). |
| K7 | Data: the bake's time; the sky-view pass cost with and without the correction; device memory added. |
| — | Regression: the pure suite (`light` changed); clippy; `-p gpu` shade, sky, bounce, temporal and denoise tests; a viewer run with validation, corrected and `--no-sky-correction`. |

## Results

Logs: `engine/results/sky_bake_k.jsonl` and `sky_bake_k.log` (the bake), `test_light_k.log`, `test_gpu_k_sky.log`, `test_gpu_k_cost.log`, `test_gpu_k_bounce.log`, `test_gpu_k_regress.log` and `test_gpu_k_regress2.log`, `test_release_k.log`, `clippy_k.log`, `viewer_k.jsonl`. All GPU runs were validation-clean (0 / 0).

| ID | Result |
|---|---|
| K1 | **pass.** Relative standard error p95 0.311% (sun ≥ 0°; p50 0.117%, max 1.02%); 0.734% below 0° (p50 0.332%, max 1.22%). |
| K2 | **pass.** The round trip is exact; every mismatch is refused. The grid's directions land on their texels in the lookup (the poles and a vertical sun excepted, where azimuth has no meaning); the correction is exactly 1 uncorrected. |
| K3 | **pass.** GPU against host with the correction, 5 times: worst 9.8 × 10⁻⁵, 0 of 311,040 values over 0.5%. The sky pass's layout check includes the new binding; the shade reflection is still refused. |
| K4 | **pass.** Sky light on surfaces with the correction, all within 0.2% from dawn to dusk and 1.2% at twilight (the budget is 5% / 15%). See the table below. **Control:** the uncorrected run fails with the 3C numbers to the fourth digit. |
| K5 | **pass.** Beyond 3 standard errors, the luminance p95 is 0 from dawn to dusk (uncorrected: 1.8%, 0.9%, 0, 1.8%) and 1.9% at twilight (8.1%). Raw means: 2.1%, 1.2%, 1.1%, 2.1%, 4.1% (uncorrected: 4.2%, 2.1%, 1.3%, 4.2%, 7.8%). What remains is the reference's own noise at 4096 spp (at 65,536 at twilight). |
| K6 | Data: 3F's B2 with the correction: the image is within 1.1% of the one-bounce reference at every time, and the indirect component within 1.7% (before: dawn/dusk −2 to −4.5%, twilight blue +10.6%). Morning and midday still pass. |
| K7 | Data. **Bake:** 1,085 s for 163 G samples (115–170 M/s), once. **Sky-view pass:** 0.066 ms at 1080p, the same with the corrected data and with ones (validation off, `test_gpu_k_cost.log`). Both arms run the new shader, so the lookup's cost against the old shader is not isolated; the pass runs only when the hour changes. **Memory:** +0.54 MB of device memory (the correction, `GpuTemporal`; sky tables 1.15 MB in total). **Loading:** 0.08 s on the host to form the ratio. |
| — | **Regression:** pure suite 137 pass, 2 ignored. Clippy: only the 6 old `world` lints. `-p gpu`: lib 12, bounce 4 (+1 ignored), shade 3, temporal 6, sky 7 pass (+ the ignored control); denoise 3 pass and **D4 fails as accepted** (S-018), with numbers identical to the 3F run. Viewer, validation on: 600 frames with edits at 18 h (20 of 20 edits shown), 900 frames walking from 5.5 h with the day running (the sun from −6° through the slices), 300 frames with `--no-sky-correction`. All exit 0, 0 / 0; the JSON reports `"sky_correction"`. |

**K4, sky light on surfaces** (image mean against the reference's direct-sky component; R G B):

| Time | Camera | Uncorrected (3C) | Corrected |
|---|---|---|---|
| dawn | street / low | −3.0 −4.3 −5.7% / −3.6 −4.7 −5.4% | +0.03 −0.06 −0.11% / +0.12 −0.00 −0.08% |
| morning | street / low | −2.0 −3.1 −4.3% / −1.6 −2.8 −4.0% | −0.04 −0.04 −0.04% / +0.08 +0.06 +0.02% |
| midday | street / low | −0.1 −1.2 −1.6% / +0.1 −0.9 −1.5% | −0.09 −0.08 −0.09% / +0.06 +0.03 −0.02% |
| dusk | street / low | −4.5 −5.2 −5.8% / −4.2 −4.9 −5.4% | −0.09 −0.13 −0.17% / +0.05 −0.04 −0.13% |
| twilight | street / low | +1.2 +10.1 +15.5% / +1.4 +10.1 +15.2% | −0.91 −1.04 −0.54% / −0.95 −1.20 −0.84% |

**The correction** (`sky_bake_k.jsonl`, per slice: mean over its texels):
- **From 45° up:** within 1%.
- **Low sun:** the reference is 2–4% brighter than the table from 0° to 10° (blue up to +17% near the horizon).
- **Below the horizon:** the reference is darker. At −2° blue is 0.93× on average; at −4° to −6° it is 0.6–0.8×, where the table's isotropic multiple scattering overestimates the shadowed atmosphere.
- **Clamped:** 715 values, all at −4° to −6° (lowest raw ratio 0.36).
- Between slices it changes smoothly.

## Observations

- The display exposure still comes from the uncorrected ground-irradiance table (display only, not lighting).
- Below −6° the −6° slice is used; there the sky is dark and its error is not measured.
- 3C criterion 4 now passes with the correction; the uncorrected test stays ignored as the control.

## Not run

- A debug build; another GPU. The viewer frame-time comparison with and without the correction (validation-off, interleaved) was not run: the change touches only the sky-view pass, which runs when the hour changes.
- The user's visual judgement of the corrected sky (`--no-sky-correction` shows the old one).
