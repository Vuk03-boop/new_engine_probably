# Change: Phase 3A — light conventions and the reference renderer

Status: **passed on the RTX 3050 (2026-09-24).**
Date and baseline: 2026-09-24, after 2E (no Git). Toolchain per ADR-0001 (slangc 2026.13.1, Vulkan SDK 1.4.357.0).
Authorization: S-017 (Phase 3 slices 3A–3G; sky model and lighting technique delegated).

## Objective

One trustworthy definition of light for every M3 term to be measured against ([Phase 3 proposal](2026-09-24-phase3-proposal.md) §3A):
- the conventions, in [ADR-0005](../adr/ADR-0005-light-transport-conventions.md): units, sun path, physical atmosphere, Lambertian surfaces, ray offsets, light-path structure, RNG;
- a reference renderer on the GPU, checked against an independent CPU implementation of the same estimator.

## What was built

- **`engine/light`** (new pure crate, default workspace member, no dependencies beyond `world`):
  - `sample`: PCG32 RXS-M-XS (the shared RNG), cosine / cone / sphere / Henyey–Greenstein samplers, phase functions.
  - `sun`: E_SUN = 1, the 0.2664° disk, `SunPath` (latitude 45°, equinox) and the five reference times.
  - `atmosphere`: Rayleigh + Mie + ozone on a spherical planet; the discretised medium; transmittance; the Monte Carlo sky (`sky_sample`, all orders, spectral MIS, sun next-event estimation, ground bounce, Russian roulette); a single-scattering quadrature as its oracle.
  - `reference`: the CPU path tracer over `world::reference::trace` (f64), the albedo table check (refuses values outside [0, 1]), `Accum` (sums and sums of squares).
- **`gpu::reference`** + `gpu/shaders/reference.slang`: the same estimator on the GPU (ray query on the TLAS, f32), reflection-checked layout (128-byte push constants), a transmittance probe for constant agreement, planted faults (`wrong_pdf`, `no_cosine`). Accumulation is 32 B/px under `GpuTemporal`; albedos under `GpuMaterial`.
- **`ref_light`** (`gpu/src/bin`): renders the reference set (2 cameras × 5 times) to `.pfm` (linear HDR data) and `.ppm` (display only: log-average exposure, ACES fit, sRGB).

## Criteria and results

Logs: `engine/results/test_light_3a.log`, `engine/results/test_gpu_3a_reference.log` (the first failing run: `test_gpu_3a_reference_first.log`; the diagnostic: `test_gpu_3a_tail_diagnostic.log`).

| # | Criterion | Result |
|---|---|---|
| CPU | Zenith transmittance against the closed form | 64 steps 1.6 × 10⁻⁴, 4096 steps 3.8 × 10⁻⁸ relative |
| CPU | One-event Monte Carlo sky against the single-scattering quadrature (4 directions × 3 channels) | all within 4.5 standard errors + 0.5% |
| CPU | Default steps (64 view / 32 sun) converged within 1% | quadrature at all 5 times: worst 0.68% (twilight, near-horizon red); same-stream Monte Carlo at dawn and midday: worst 0.65% |
| CPU | White furnace, point-sun plane closed form, samplers against their PDFs, albedo refusal | pass (furnace exactly 1; plane exact to 10⁻¹²) |
| 1 | GPU layout checked from reflection (the ray-primary reflection is refused); GPU transmittance equals the CPU's at the same steps | 41,218 channel values, worst 4.9 × 10⁻⁶ relative; 0 ground/sky disagreements |
| 2 | White furnace on the street (albedo 1, uniform sky, 64 bounces, 320×180, 16 frames); PDF fault caught | 57,600 of 57,600 pixels exactly 1; fault: mean 1.147 |
| 3 | Point sun, per pixel, GPU against CPU at 8 h, 12 h, 17.75 h, two cameras; missing-cosine fault caught | lit pixels agree to ≤ 1.7 × 10⁻⁷; **0** shadow disagreements of 57,600 in every arm; fault fails every arm |
| 4 | GPU (4096 spp) against CPU (256 spp), 96×54, sun + sky + 2 bounces, dawn / midday / dusk / twilight, two cameras: image mean |z| < 4 per channel, pixel |z| > 4 rate within the metric's own null rate; PDF fault caught | image means agree within 0.52% (|z| ≤ 1.62); pixel rates 0.13–0.25% (day), 1.6–1.9% (twilight, null 1.74–1.76%); fault caught in all 8 arms (|z| up to 33) |

Validation: 0 errors, 0 warnings in every GPU test. Bad surface ids: 0.

Broader gates: the pure suite (`cargo test --release -j 2`) exits 0 with 130 tests (115 before 3A + 15 in `light`; 1 diagnostic ignored), `results/test_release_3a.log`. Clippy exits 0 with only the 6 pre-existing `world` lints after fixing the 4 that 3A introduced (`results/clippy_3a.log`). The full `-p gpu` suite was not rerun for 3A (no existing module changed; `build.rs` gained one shader); it runs at the end of 3B, which changes `debug_view`.

## Failures on the way (diagnosed, not rebaselined)

1. **Point-sun non-vacuity guard.** The first run required ≥ 10% lit pixels so the comparison could not pass empty. The noon street view has 2,588 lit pixels (4.5%), with every value agreeing to 10⁻⁷. The guard became "≥ 1,000 lit pixels"; the accuracy criteria are unchanged.
2. **Pixel |z| rate at twilight** (1.6–1.9% against a 1% limit, while the image means agreed). Hypothesis: heavy-tailed twilight pixels underestimate their standard error at 256 samples. Discriminating diagnostic (`diagnostic_tail_rate_of_the_pixel_metric`): an independent GPU image at the same 256 samples shows the same rate (1.76%); at 1,024 samples both CPU and null drop to 0.07–0.09%. So the metric, not the renderer, produced it. The criterion now measures the null rate in the same scene and requires the CPU rate ≤ max(1%, 1.5 × null + 0.2%). The image-mean criterion, which catches the planted fault, is unchanged.
3. **Step-convergence check at twilight** (first version: 2.4% in blue). Diagnostic (`diagnostic_step_dependence_at_twilight`): with the same streams every step count is within ±0.7% of 1024/512, with no trend; independent streams differ by 8–14% at 40,000 samples. It was path-divergence noise. The check now uses the deterministic quadrature at every time, and same-stream Monte Carlo only at dawn and midday. A NaN (0/0 for a fully blocked twilight path) that `f64::max` silently ignored was also fixed.

## Observations

- **The twilight sky is very noisy for this estimator:** most scattering events happen low, where the set sun is blocked; the light comes from rare events high up. References after sunset need many more samples (or a better distance-sampling strategy, e.g. sampling proportional to sunlit scattering). It matters for 3C, where the real-time sky tables are compared with this reference.
- GPU reference throughput: about 25 million paths per second at 96×54 (sun + sky + 2 bounces).
- Direct sun at the street (E_SUN × transmittance): midday [0.917, 0.818, 0.682]; the colour shifts to orange as the sun lowers (see the image table).

## Reference images

Rendered by `ref_light results\phase3a_ref --size 960 540 --spp 2048` (8 bounces, validation on, 0 errors, 0 bad samples). Data: `engine/results/phase3a_ref/*.pfm`, per-image JSON `engine/results/phase3a_ref.jsonl`, overview `engine/results/phase3a_ref/contact_sheet.png` (top: street camera, bottom: low camera; left to right dawn, morning, midday, dusk, twilight; display exposure per image).

| Image | Sun elevation | Direct sun at the street (R / G / B) | Mean radiance (R / G / B) | Mean pixel relative std. error | Seconds |
|---|---|---|---|---|---|
| street_dawn | 2.65° | 0.389 / 0.120 / 0.013 | 3.31E-001 / 1.05E-001 / 1.58E-002 | 0.045 | 16.8 |
| street_morning | 20.70° | 0.842 / 0.672 / 0.467 | 4.93E-001 / 4.00E-001 / 2.89E-001 | 0.036 | 16.8 |
| street_midday | 45.00° | 0.917 / 0.818 / 0.682 | 1.16E-002 / 1.57E-002 / 2.41E-002 | 0.038 | 17.9 |
| street_dusk | 2.65° | 0.389 / 0.120 / 0.013 | 5.02E-003 / 4.20E-003 / 4.15E-003 | 0.045 | 18.5 |
| street_twilight | -2.65° | 0.000 / 0.000 / 0.000 | 3.67E-004 / 1.74E-004 / 2.62E-004 | 0.137 | 15.5 |
| low_dawn | 2.65° | 0.389 / 0.120 / 0.013 | 1.31E-001 / 4.49E-002 / 1.11E-002 | 0.041 | 14.3 |
| low_morning | 20.70° | 0.842 / 0.672 / 0.467 | 3.55E-001 / 2.92E-001 / 2.19E-001 | 0.028 | 14.4 |
| low_midday | 45.00° | 0.917 / 0.818 / 0.682 | 2.60E-002 / 3.20E-002 / 4.39E-002 | 0.029 | 14.4 |
| low_dusk | 2.65° | 0.389 / 0.120 / 0.013 | 6.80E-003 / 5.97E-003 / 6.24E-003 | 0.041 | 14.4 |
| low_twilight | -2.65° | 0.000 / 0.000 / 0.000 | 4.98E-004 / 2.60E-004 / 4.15E-004 | 0.131 | 12.4 |

- The street camera faces east: at dawn and in the morning the sun disk (radiance about E × T / Ω ≈ 5,700 at dawn, in red) is in frame and dominates the image mean. At dusk it is behind the camera.
- Twilight images carry about 3× the relative noise of the day images at the same sample count (see Observations).

## Not run

- Another GPU; scenes other than the street.
- A controlled cold-start shader-cache measurement of the reference pipeline.
- References at 1080p (the 3E/3G quality comparisons will render them at the resolution they need).

## Closeout

- Docs: ADR-0005 (new, accepted), this record, `engine/README.md` (crate `light`, `gpu::reference`, `ref_light`, commands), DECISIONS (S-017, A-008), NOW.
- Next: 3B, the real-time sun term (a G-buffer shading pass with one shadow ray), its exact point-sun control and its convergence to the reference's sun component.
