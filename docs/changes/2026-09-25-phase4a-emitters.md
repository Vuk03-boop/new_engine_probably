# Change: Phase 4A — emitters, units and the reference

Status: **in progress.** Part 1 (conventions, emitter table, CPU and GPU reference, `street_night`) is built. Its GPU checks are **NOT RUN**: they wait for `run-local.cmd` on the RTX 3050. Part 2 (viewer `--scene` and L, night exposure, magnitudes) has not started.
Date and baseline: 2026-09-25, branch `claude/tender-keller-9u9d6t` from `main` after the cloud rules commit. Written in a cloud session (no GPU; [CLOUD.md](../CLOUD.md)).
Authorization: S-024 (4A and 4B authorized); the user's "green light for the session" (2026-09-25).

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
- **Part 2 (next):** the viewer's `--scene` and L; the metric exposure at night; `ref_light` night references; the magnitudes (bounces ≥ 2 share at dusk, blue hour and night; one-sample noise); the table inside `GpuScene::update` (edit latency).
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

## Proposed night lights (for the user to adjust)

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
| G1–G4, and G5 on the laptop | **NOT RUN** | `run-local.cmd` |

## Failures and changes on the way (diagnosed, not rebaselined)

1. **Area sampling had infinite variance (C3, first runs).** At 256 spp the furnace mean was 0.9932 ± 0.0092: inside 1 SE, but the standard error could not resolve the 0.5% bound. At 8192 and 32768 spp the standard error went *up* (0.21% → 0.27%). Cause: near the edge of an adjacent emitting wall, area sampling's 1/d² term has a second moment ∝ 1/δ² (δ the distance to that wall), which diverges when averaged over receivers. `street_night` has this everywhere (façade points next to neon). **Change:** emitters are sampled uniformly in solid angle (spherical rectangles, Ureña et al. 2013), with area sampling below 10⁻² sr and as a control setting. ADR-0005 Amendment 3 was revised before any GPU run. By solid angle the standard error now halves per 4× samples (0.30% / 0.15% / 0.075% at 1k / 4k / 16k spp); by area at the same seeds 0.56% / 0.26% / 0.15%. The criterion is unchanged; the test uses 4096 spp.
2. **The solid-angle oracle (unit test, first version)** was a 400² quadrature, which is too coarse 0.01 voxel below an edge (it was the side in error: 3.079 against the exact 3.128). Replaced by exact formulas.
3. **Switch threshold:** first 10⁻³ sr, raised to 10⁻² sr before any GPU run, because the f32 solid angle Σgᵢ − 2π carries about 10⁻⁶ sr of cancellation error (10⁻³ relative at the old threshold).

The spherical-rectangle code is written from the published algorithm as I recall it; it is validated by the exact-formula tests above, not by a copy of the paper.

## Observations

- Night luminance of the Full street at 21 h (street camera, CPU smoke render): mean G about 1.6 × 10⁻⁴ units, about 20 cd/m². With the lights off the image is exactly 0 (the sky is fully dark at −30°).
- Lamps and signs are 165 quads by themselves (greedy quads split at brick boundaries), not 6; the proposal's counts were for lights.

## Next

1. Run `run-local.cmd` on the RTX 3050 and read its logs against G1–G5.
2. Part 2: the viewer's `--scene` and L; the night exposure; `ref_light` night references; the magnitudes; the table inside `GpuScene::update`.
