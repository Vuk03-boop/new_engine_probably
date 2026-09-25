# Change: Phase 4A — emitters, units and the reference

Status: **in progress. Part 1 passed** (conventions, emitter table, CPU and GPU reference, `street_night`): C1–C5 in the cloud session, G1–G5 and the M3 GPU regressions on the RTX 3050 (`run-local.cmd`, 2026-09-25 19:12). **Part 2 started**: its criteria are frozen below (C6–C7, G6–G9, M1–M2).
Date and baseline: 2026-09-25, branch `claude/tender-keller-9u9d6t` from `main` after the cloud rules commit. Written in a cloud session (no GPU; [CLOUD.md](../CLOUD.md)).
Authorization: S-024 (4A and 4B authorized); the user's "green light for the session" (2026-09-25); part 2: "For now i accept it part 2 time" (the proposed night lights accepted for now).

## Objective

- Give emission a unit and a sampling rule, and add it to the reference, so every later M4 term has something to be measured against ([Phase 4 proposal](2026-09-25-phase4-proposal.md) §4A).
- Conventions: [ADR-0005 Amendment 3](../adr/ADR-0005-light-transport-conventions.md) (units, emitter model, sampling) and [ADR-0003 Amendment 3](../adr/ADR-0003-device-budget-and-visibility-buffers.md) (the emitter table in the publication set).

## Scope

- **Part 1 (this record, now):**
  - `light::emitters`: the emitter table and its alias sampler;
  - `light::reference`: emission and emitter next-event estimation, with planted faults;
  - `gpu::reference` and `shaders/reference.slang`: the same on the GPU;
  - `gpu::emitters`: the table from region meshes, and its device upload;
  - `world::scene::street_night` with a dressing parameter;
  - tests for all of the above.
- **Part 2:** the table inside `GpuScene` (build, edits, swap, edit latency); the viewer's `--scene`; the lights rule; the metric exposure; `ref_light` night references; the magnitudes (bounces ≥ 2 share at dusk, blue hour and night; one-sample noise). Details and criteria below.
- **Excluded:** real-time emitter light (4B), reservoirs (4C+), any change to `street_block` or to M3 defaults.

## Criteria (frozen before the first run)

**Pure CPU (cloud and laptop):**

| # | Criterion |
|---|---|
| C1 | Alias table: 10⁶ selections from a 50-emitter table whose powers span 10⁴ : 1, χ² against the realized probabilities < 85.35 (df 49, p = 0.001). Every emitter with power > 0 has probability > 0; realized probability within 10⁻⁵ relative of Φ / ΣΦ for emitters with Φ / ΣΦ ≥ 10⁻³. |
| C2 | Emissive rectangle over a plane (direct emitters only, 0 bounces) against Lambert's polygon formula: every pixel |z| < 5, image mean within 3 standard errors and 0.5%. The faults "area PDF without solid angle" and "no emitter cosine" each move the image mean by more than 10 standard errors. |
| C3 | Emissive furnace (closed box, albedo 0.5, `max_bounces` 16, emission + direct + indirect): image mean equals L_e Σ₀¹⁷ 0.5ᵏ within 3 standard errors and 0.5%. The fault "emission counted twice" moves it by more than 10 standard errors. |
| C4 | Table identity: brick-grouped and region-grouped quads give the same table (same order, geometry and probabilities); removing one lamp voxel changes the table; a table built for another snapshot is refused. |
| C5 | Emitters off changes nothing: the pure suite passes unchanged, and the `diagnostic_street_fingerprint` of the default street render is bit-identical before and after on the same machine. |

**GPU (laptop, `run-local.cmd`):**

| # | Criterion |
|---|---|
| G1 | The reference module's reflection matches the host layout, including the emitter bindings; 0 validation errors; 0 bad surface ids. |
| G2 | Emission at the primary hit only (deterministic): GPU equals CPU per pixel within 10⁻⁶ relative on `street_night` (Full). |
| G3 | Emissive furnace on the GPU as C3 (image mean within 3 SE and 0.5%); "emission counted twice" is caught (> 10 SE). |
| G4 | GPU (4096 spp) against CPU (256 spp), 96×54, `street_night` (Full) at 21 h, emission + direct + indirect, `max_bounces` 1, two cameras: image mean |z| < 4 per channel; pixel |z| > 4 rate ≤ max(1%, 1.5 × null + 0.2%) (the 3A metric); "area PDF without solid angle" caught in every arm. |
| G5 | The emitter table build for `street_night` (Dense) takes ≤ 5 ms on the host (10% of the 50 ms edit budget), median of 20. |

## Part 2: plan and criteria (frozen 2026-09-25, before its first run)

**What part 2 builds:**

- **The table in the publication set** (ADR-0003 Amendment 3):
  - `GpuScene::build_lit` builds the table with the meshes. `GpuScene::update` rebuilds it from every region's emissive quads (`gpu::emitters::EmitterSet`, kept on the host per region) and uploads it before the acceleration update. It is swapped with the meshes and the TLAS, and its replaced buffers are retired like mesh buffers.
  - A refused grant leaves meshes, TLAS and table on the previous snapshot.
  - `GpuScene::emitters` refuses a table whose snapshot is not the scene's.
  - Planted faults: `stale_emitters` (the update keeps the old table) and `refuse_emitters` (the table's upload is refused).
  - `UpdateStats` gains the table's host time.
  - A scene built with `GpuScene::build` has no table, so every M3 path is unchanged.
- **Viewer:** `--scene street|night` (default `street`, the M3 scene) and `--dressing lamps|windows|full|dense` (default `full`). The night scene is built with its table, and every edit rebuilds the table inside the update. The JSON line gains the scene, the emitter count and the table's time per edit.
- **L moves to 4B.** The real-time path has no emitter light until 4B, so in 4A the key would switch nothing on screen. 4B adds L, the automatic switch and the night exposure together.
- **The lights rule** (Phase 4 decision 3): the lights are on while the sun is below the horizon (elevation < 0°), `light::emitters::lights_on`. So at M4's dusk (17.75 h, +2.65°) they are off, and at blue hour (18.5 h) and night (21 h) they are on.
- **Metric exposure** (`light::exposure::metric_exposure`): 0.18 divided by the log-average luminance of the reference image, over the pixels whose luminance is at least 2⁻¹⁰ of the image mean.
  - Leaving darker pixels out stops the black night sky and noise-level pixels from setting the exposure.
  - By day no pixel is that dark, so it is the plain log-average.
  - `ref_light` writes its display images with it and prints it.
- **Night references** (`ref_light --scene night`): 2 cameras × dusk, blue hour and night; 960×540, 16,384 spp, 8 bounces; the lights by the rule (emission and emitter light when on). Cached in `engine/results/phase4a_ref/`.
- **Magnitudes** (M1–M2) on the CPU reference in the cloud session. It is the same estimator as the GPU's (G2–G4), and the magnitudes are ratios, so they do not need 960×540.

**Pure CPU (cloud and laptop):**

| # | Criterion |
|---|---|
| C6 | The table follows edits (host, `gpu --lib`). On `street_night` (Full), chunk regions, through the real 1C pipeline, after each of these edits: remove a lamp-head voxel; remove a bulb voxel; add an emissive voxel in open air; remove a voxel in a region without emitters; restore everything. After each: the incremental table equals a from-scratch build of the snapshot's regions in order, geometry, radiance, probabilities, thresholds and aliases, with the same region keys and quad indices. Emitters of the edited regions carry the new snapshot; all others keep theirs. After the edit without emitters, no emitter id changes. The restored table equals the original except for the edited regions' snapshots. Negative control: leaving the lamp's region out of the update makes the comparison fail. |
| C7 | Metric exposure and the lights rule (`light` unit tests). A uniform image of luminance c gives 0.18 / c. Scaling an image by k divides its exposure by k (both within 10⁻¹² relative). An image with no pixel below 2⁻¹⁰ of its mean gives the plain log-average. Adding black pixels leaves the exposure unchanged (in a test image with no pixel between the old and new thresholds). An all-black image gives none. The lights are off at every 3A time with the sun up and at M4's dusk, and on at twilight, blue hour and night. |

**GPU and viewer (laptop, `run-local.cmd`):**

| # | Criterion |
|---|---|
| G6 | The table in `GpuScene` on the device (`gpu --test emitters`): the C6 sequence through `GpuScene::update` with ray tracing. After each update, `emitters()` returns the table of the scene's snapshot, equal to C6's from-scratch build, and the device buffers read back byte-equal to it (the 80 B rows and the per-material emission). Replaced table buffers are retired, and after collection the `GpuMaterial` ledger holds exactly the table's buffers beyond what it held before the scene. `stale_emitters` is refused (`StaleTable`) after the next edit. `refuse_emitters` leaves meshes, TLAS and table on the previous snapshot and counts a deferral, and the retry succeeds. 0 validation errors, no leaks. |
| G7 | Edit latency with the table (viewer, release, validation off, 1920×1080, the scripted fly path, as 3G's E1). `--scene night` (Full) with `--edit-script --edit-size N`, N = 1, 8, 32, 2,000 frames each; and Dense at N = 1. Pass: p95 edit-to-visible ≤ 50 ms (N = 1, 8) and ≤ 100 ms (N = 32), every edit shown, 0 deferred. The table's median host time per update is reported. With validation on (400 frames, N = 1, 8, 32, Full): exit 0, 0 errors, 0 warnings. |
| G8 | Night references (`ref_light --scene night`, 16,384 spp): all 6 images with 0 bad samples, 0 validation errors and no leaked buffers. Recorded per image: the mean, the mean relative standard error, the metric exposure and the render time. |
| G9 | Regressions: pure suite; `gpu --lib`; `gpu --test emitters` (G1–G5 as before); `gpu --test edit` (M2's edits through the changed update); `gpu --test temporal` (edits with relight). The viewer's M3 edit script on `street` (N = 1, 2,000 frames, validation off): p95 ≤ 50 ms, every edit shown, and the JSON reports no table. |

**Magnitudes (measurements with a fixed method, not pass/fail):**

| # | Measurement |
|---|---|
| M1 | Share of the image in bounces ≥ 2: 1 − L̄₁ / L̄₈, on the luminance of the image mean. Emission at the primary hit is left out (bounces do not change it). Paired samples: the same random streams at `max_bounces` 1 and 8, so each difference is exactly the light after 2 or more bounces. `street_night` (Full), 2 cameras × dusk, blue hour and night, the lights by the rule, 128×72 pixels. Samples are added until the standard error is ≤ 0.5 percentage points. Per pixel as well: the median and 90th percentile of the share over pixels with light. |
| M2 | One-sample noise: σ/μ of a single sample per pixel (median and 90th percentile over pixels with μ > 0). (a) The one-bounce estimator (sun, sky, and emitters at the primary and bounce vertices: the reference at `max_bounces` 1, what 4B computes per frame), same arms as M1. Dusk (lights off) is the noise the M3 filter already handles, for scale. (b) Emitter light alone at the primary hit (one emitter sample, `max_bounces` 0), at night, for each dressing (Lamps, Windows, Full, Dense). |

## Night lights (proposed; accepted by the user for now, 2026-09-25)

Luminance is the value in cd/m² (converted with 1 unit = 128,000 cd/m²); the colour is normalized to Y = 1. They are starting points from the proposal's ranges, chosen so a lamp lights the road below it to tens of lux; they are judged by you from 4B, when the street can be seen at night.

| Light | Colour (linear RGB, before normalizing) | Luminance cd/m² | Why |
|---|---|---|---|
| Street lamp heads (3) | warm white 3000 K (1.0, 0.80, 0.60) | 7,000 | about 8,000 lm from the head's ~0.37 m² of faces |
| Shop signs (3) | as today (1.0, 0.75, 0.35) | 400 | a backlit sign |
| Neon outlines | shop 1 pink (1.0, 0.10, 0.45), shop 2 cyan (0.10, 0.75, 1.0), shop 3 green (0.20, 1.0, 0.30) | 300 | a 1.5 cm tube averaged over a 6.25 cm voxel |
| Lit upper windows | warm interior (1.0, 0.72, 0.42) | 60 | a lit room seen through glass |
| String-light bulbs | warm 2200 K (1.0, 0.55, 0.20) | 800 | about 40 lm per voxel-sized bulb |

Dressing levels (`world::scene::Dressing`, for the light-count sweep; measured counts are in the results):

| Level | Content |
|---|---|
| `Lamps` | lamps and signs only |
| `Windows` | + lit upper windows |
| `Full` (default) | + neon outlines + 4 string lights across the street, a bulb every 0.5 m |
| `Dense` | + string lights every 1 m along the street, a bulb every 0.25 m (measurement only) |

## Commands and evidence actually produced

Cloud session, Linux x86_64, 4 cores, no GPU. The `gpu` crate was built with the pinned slangc 2026.13.1 (the Linux release, downloaded to the session's scratch space) and the distribution's `spirv-val` (SPIRV-Tools v2025.1, not the SDK's): a supplemental compile check, not the ADR-0001 toolchain on the laptop.

| Check | Result | Scope |
|---|---|---|
| Pure suite before the change | exit 0, 137 pass | cloud |
| Pure suite after (`cargo test --release -j 2`) | exit 0, **150 pass**, 3 ignored (2 existing + the fingerprint diagnostic); `engine/results/test_pure_4a_cloud.log` | cloud |
| C1 alias (`emitters::tests::alias_selection_matches_the_realized_probabilities`) | pass; the negative control (index draw without the coin) fails the same χ² | cloud |
| C2 rectangle (`tests/emitters.rs`) | pass. By solid angle: mean ratio 0.99977 ± 0.00108, worst pixel \|z\| 2.37; by area: 0.99952 ± 0.00156, 2.42. Faults: no solid angle 188.7× / 185.0×, no emitter cosine 1.140 ± 0.001 (> 100 SE) | cloud |
| C3 furnace | pass: 0.99844 ± 0.00149 at 4096 spp; double emission 1.4958 ± 0.0058 | cloud |
| C4 identity | pass (brick vs chunk-region grouping on the CPU; brick / chunk / 2³-chunk regions in `gpu --lib`) | cloud |
| C5 unchanged | pass: fingerprints bit-identical before and after; `emitters_off_ignores_the_table` passes | cloud |
| Spherical rectangle | solid angle equals Van Oosterom–Strackee within 10⁻⁹ at 3 points (one 0.01 voxel below an edge); sample mean direction matches the exact one within 5 × 10⁻³ | cloud |
| `street_night` smoke (all terms, 21 h) | finite, non-negative; lights on 1.56 × 10⁻⁴, off exactly 0 | cloud |
| `gpu --lib` | 14 pass (`engine/results/test_gpu_lib_4a_cloud.log`), including the reflection check with a negative control (a moved field is refused) | cloud, no device |
| Clippy (`--workspace --all-targets`, with `gpu`) | exit 0; only the 6 pre-existing `world` lints (`engine/results/clippy_4a_cloud.log`) | cloud |
| G5 table build (host only) | Dense, 7,003 emitters: median 3.16 ms, max 4.82 ms | cloud CPU: **not** evidence for the laptop |

**RTX 3050 laptop**, `run-local.cmd` at commit `07272fc` (cargo 1.98.1, Vulkan SDK 1.4.357.0), logs in `engine/results/local-run/2026-09-25_1912-4a/` (`summary.txt`: 8 of 8 steps exit 0). An earlier start (`2026-09-25_1902-4a/`) was stopped with Ctrl+C during the build (`STATUS_CONTROL_C_EXIT`); it is not a result.

| Check | Result |
|---|---|
| Pure suite | exit 0, 150 pass, 3 ignored |
| `gpu --lib` | 14 pass (the reflection check included) |
| G1 layout and validation | pass: `Reference::new` accepts the module with the emitter bindings; 0 validation errors and 0 warnings in all four `emitters` tests; 0 bad surface ids |
| G2 emission at the primary hit | pass: 3,948 (street view) and 1,257 (low) emissive pixels, 0 over 10⁻⁶, worst difference exactly 0 |
| G3 furnace on the GPU | pass: 1.999681 ± 0.000714 against 1.999992 (ratio 0.99984, 0.44 SE); double emission 2.999674 (caught) |
| G4 night street, GPU against CPU | pass. Street view: z [0.31, −1.88, −2.10], \|z\| > 4 in 0.649% (limit 1.123%); low: z [0.94, −0.33, −0.60], 0.791% (limit 1.498%). Fault "no solid angle": z ≈ 1,400–1,600 in both (caught) |
| G5 table build (Dense, 7,003 emitters) | pass: median 2.877 ms, max 3.432 ms (≤ 5 ms) |
| M3 regressions (emitters off) | `reference` 4 pass (1 ignored): all 39 result lines equal the 3A log's (`results/test_gpu_3a_reference.log`, whose lines were saved cut at a fixed width): the same GPU image means, point sun to ≤ 1.7 × 10⁻⁷ with 0 shadow disagreements, the furnace exactly 1. `shade` 4, `sky` 7 (1 ignored as recorded), `temporal` 7, `bounce` 4 (1 ignored) pass; 0 validation errors |
| **NOT RUN** on purpose | `gate` (35+ min; the frame path did not change) and `denoise` (its accepted failures C1/D4; it does not bind the reference) |

## Part 2: evidence

Cloud session (as part 1: Linux, 4 cores, no GPU; the pinned slangc's Linux release and a non-SDK `spirv-val` for the compile check).

| Check | Result | Scope |
|---|---|---|
| C6 the table follows edits (`gpu --lib`, `emitters::tests::edits_keep_the_table_equal_to_a_fresh_build`) | pass. Remove a lamp-head voxel: 3 regions rebuilt, 1,143 → 1,147 emitters. Remove a bulb voxel: 3 regions, 1,142. Add a neon voxel in open air: 2 regions, 1,148. Remove a voxel in a region without emitters: 1 region, no emitter id changed. Restore everything: 8 regions, the original 1,143 with the original geometry and probabilities. After every edit, the incremental table equals the from-scratch one; the negative control (the lamp's region left out) is caught | cloud |
| C7 metric exposure and lights rule (`light`: `exposure::tests::exposure_follows_the_lit_pixels`, `emitters::tests::lights_follow_the_sun`) | pass | cloud |
| The viewer's night start (`walk`: `night_street_start_is_free`) | pass for all four dressings (the reference feet are free) | cloud |
| Builds: `gpu` (every test binary, G6 included), `viewer`, `ref_light` | exit 0, no warnings | cloud, substitute SDK |
| Pure suite (`cargo test --release -j 2`) | exit 0, **153 pass** (150 + the 3 new), 4 ignored (+ `diagnostic_magnitudes`); `engine/results/test_pure_4a2_cloud.log` | cloud |
| `gpu --lib` | exit 0, 15 pass (14 + C6); `engine/results/test_gpu_lib_4a2_cloud.log` | cloud, no device |
| Clippy (`--workspace --all-targets`) | exit 0; only the 6 pre-existing `world` lints (5 new ones in my test code were fixed first); `engine/results/clippy_4a2_cloud.log` | cloud |
| M1–M2 magnitudes (`light`: `diagnostic_magnitudes`) | below; `engine/results/magnitudes_4a_cloud.log` (22 min on 4 threads) | cloud CPU |

**Magnitudes** (CPU reference, `street_night` Full, 128×72, the method frozen above):

| Arm (sun, lights) | M1: bounces ≥ 2, image | M1 per pixel: median / p90 | M2a one-bounce σ/μ: median / p90 | Emission seen directly, share of the full image mean |
|---|---|---|---|---|
| street, dusk (+2.65°, off) | 0.48% ± 0.00 (1,024 spp) | 0.54% / 3.4% | 1.87 / 2.60 | 0 |
| street, blue hour (−5.3°, on) | 6.63% ± 0.34 (4,096 spp) | 0.68% / 6.2% | 11.65 / 27.62 † | 78.8% |
| street, night (−30°, on) | 10.11% ± 0.38 (8,192 spp) | 1.62% / 10.5% (7,035 lit pixels; the rest is black sky) | 4.74 / 21.40 | 85.7% |
| low, dusk | 0.22% ± 0.00 (1,024 spp) | 0.07% / 1.5% | 1.70 / 2.32 | 0 |
| low, blue hour | 3.25% ± 0.41 (1,024 spp) | 0.04% / 2.0% | 13.16 / 22.97 † | 81.8% |
| low, night | 6.80% ± 0.48 (4,096 spp) | 0.64% / 6.0% (4,810 lit pixels) | 3.31 / 12.96 | 90.5% |

† The reference estimates the sky radiance of every ray by Monte Carlo (`Atmosphere::sky_sample`); the real-time path reads the sky table. So M2a at blue hour, and a little at dusk, includes noise the real-time path does not have: those values are upper bounds. At night the sky is black and M2a is clean.

M2b, emitter light alone at the primary hit, one sample, night (σ/μ median / p90):

| Dressing (emitter quads) | street | low |
|---|---|---|
| Lamps (165) | 2.37 / 4.08 | 2.38 / 3.74 |
| Windows (325) | 2.48 / 4.11 | 2.46 / 3.82 |
| Full (1,143) | 2.88 / 7.02 | 2.78 / 5.55 |
| Dense (7,003) | 2.76 / 8.62 | 2.82 / 8.45 |

**Laptop (`run-local.cmd`, 45–60 min):** G6–G9 are **NOT RUN** until the user runs it and pushes `engine/results`. Steps: the pure suite; `gpu --lib`; `gpu --test emitters` (G1–G6); `gpu --test edit`; `gpu --test temporal`; the viewer's edit runs (night Full at N = 1, 8, 32 and Dense at N = 1 with validation off; the street at N = 1; night Full at N = 1, 8, 32 with validation on, 400 frames), all appending to `viewer_edits.jsonl`; `ref_light --scene night --spp 16384` into `results/phase4a_ref/`. Not rerun on purpose: `gate`, `denoise`, and `reference`, `shade`, `sky`, `bounce` (their code did not change since part 1's run).

## Failures and changes on the way (diagnosed, not rebaselined)

1. **Area sampling had infinite variance (C3, first runs).** At 256 spp the furnace mean was 0.9932 ± 0.0092: inside 1 SE, but the standard error could not resolve the 0.5% bound. At 8192 and 32768 spp the standard error went *up* (0.21% → 0.27%). Cause: near the edge of an adjacent emitting wall, area sampling's 1/d² term has a second moment ∝ 1/δ² (δ the distance to that wall), which diverges when averaged over receivers. `street_night` has this everywhere (façade points next to neon). **Change:** emitters are sampled uniformly in solid angle (spherical rectangles, Ureña et al. 2013), with area sampling below 10⁻² sr and as a control setting. ADR-0005 Amendment 3 was revised before any GPU run. By solid angle the standard error now halves per 4× samples (0.30% / 0.15% / 0.075% at 1k / 4k / 16k spp); by area at the same seeds 0.56% / 0.26% / 0.15%. The criterion is unchanged; the test uses 4096 spp.
2. **The solid-angle oracle (unit test, first version)** was a 400² quadrature, which is too coarse 0.01 voxel below an edge (it was the side in error: 3.079 against the exact 3.128). Replaced by exact formulas.
3. **Switch threshold:** first 10⁻³ sr, raised to 10⁻² sr before any GPU run, because the f32 solid angle Σgᵢ − 2π carries about 10⁻⁶ sr of cancellation error (10⁻³ relative at the old threshold).

4. **G4's colour pattern (passed, then diagnosed).** On the street view the GPU image mean was +0.56% / −3.01% / −4.55% (R / G / B) against the CPU's 256 samples, z ≈ −2 in G and B, with the same signs, smaller, on the low camera: within the criterion, but a colour-dependent sign could have been a per-emitter bias. Discriminating diagnostic (`diagnostic_g4_cpu_convergence`, CPU only, run in the cloud session): the CPU at 4096 independent samples against the CPU at G4's 256 moves by −0.06% / −2.89% / −4.58% on the street view (low: +0.69% / +0.09% / −0.36%). So the 256-sample CPU image carried a few bright G/B samples, and the GPU agrees with the 4096-sample CPU within +0.6% / −0.1% / +0.03% (street) and 0.6% (low). No bias; heavy tails at 256 samples, as at twilight in 3A.

The spherical-rectangle code is written from the published algorithm as I recall it; it is validated by the exact-formula tests above, not by a copy of the paper.

## Observations

**Part 2 magnitudes** (observations; what they suggest is under Next):

- Bounces ≥ 2 carry 7–10% of the reflected light at night (street 10.1%, low 6.8%) and 3–7% at blue hour, against 0.2–0.5% at dusk (as by day in 3F). Per pixel the share is concentrated: median 0.6–1.6%, 90th percentile 6–10.5% at night.
- Directly seen emission is 86–90% of the night image's mean luminance (79–82% at blue hour). A criterion on the mean of the whole displayed image would therefore hide a 7–10% loss in reflected light.
- One sample of the one-bounce estimator is noisier at night than at dusk: median σ/μ 3.3–4.7 against 1.7–1.9, and at the 90th percentile 13–21 against 2.3–2.6.
- Emitter light alone: the median σ/μ barely changes with the light count (2.4 → 2.8 from 165 to 7,003 emitter quads), but the 90th percentile doubles (3.7–4.1 → 8.5–8.6). The pixels lit by many comparable small lights (string lights, neon) are where one sample chosen by power is noisiest.

**Part 1:**

- Night luminance of the Full street at 21 h (street camera, CPU smoke render): mean G about 1.6 × 10⁻⁴ units, about 20 cd/m². With the lights off the image is exactly 0 (the sky is fully dark at −30°).
- Lamps and signs are 165 quads by themselves (greedy quads split at brick boundaries), not 6; the proposal's counts were for lights.

## Next

1. **The laptop run:** the user runs `run-local.cmd` (45–60 min) and pushes `engine/results`. G6–G9 are then analysed from the logs (G7's p95 from `viewer_edits.jsonl`). If they pass, 4A is done (its 6 units count).
2. **Then 4B** (authorized): one emitter sample per pixel at the primary and bounce hits, then the temporal pass and the filter; L, the automatic switch and the night exposure (moved here from 4A); relight rules for emitters; the equal-time curve over samples per pixel and light counts.
3. **What the magnitudes suggest** (recommendations, not decisions):
   - 4B's energy criterion should be on reflected light (emission excluded) or per region, not on the mean of the whole image, which emission dominates at night.
   - Per-frame noise at night is 2–3× dusk's in the median and 5–8× in the 90th percentile, and its tail grows with the light count. That is the case reservoir reuse (4C) targets; 4B's filtered error decides whether it pays.
   - One bounce misses 7–10% of the reflected light at night (0.2–0.5% at dusk). That bears on 4D's entry condition ("the indirect term matters at night as a share of bounces ≥ 2"), which set no threshold: the call is yours once 4B shows whether the loss is visible after filtering.
