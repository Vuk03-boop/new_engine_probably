# engine — new engine source tree

Rust workspace, toolchain pinned to 1.98.1 by `rust-toolchain.toml` ([ADR-0001](../docs/adr/ADR-0001-host-and-shader-language.md)).
- External crates (A-006): `ash =0.38.0` in `gpu` and `viewer`; `winit =0.30.13` and `ash-window =0.13.0` in `viewer` only (resolved offline to the Phase 0 lock versions).
- `gpu` and `viewer` are not default members. Plain `cargo test` runs the pure CPU crates (including `walk`); the GPU tests run with `-p gpu` and need the Vulkan SDK and an RT GPU; `viewer` also needs a display.

| Crate | Contents |
|---|---|
| `memory` | Phase 1D memory categories, reported and budgeted (`Ledger`) accounting, container bounds, and an opt-in counting allocator for verification. Record: [1D](../docs/changes/2026-09-23-phase1d-memory-accounting.md) |
| `world` | Phase 1A authoritative sparse world, Phase 1E CPU reference ray query, and Phase 1B transactions and save/reload (`edit`, `persist`). Records: [1A + 1E](../docs/changes/2026-09-23-phase1-world-core.md), [1B](../docs/changes/2026-09-23-phase1b-persistence.md) |
| `gpu` | Phase 2B headless Vulkan: capability-based device, validation, ledger-backed block allocator (buffers and images), timeline retirement (also of 1C reader tokens), staging ring, ADR-0004 region meshes, and `slangc` reflection and capability checks. Phase 2C-1: raster G-buffer and the per-pixel equivalence check. Phase 2C-2: optional window surface, swapchain and frames in flight, debug views, GPU timing. Phase 2D: one BLAS per region, a TLAS, and ray-query primary visibility checked against raster. Records: [2B](../docs/changes/2026-09-23-phase2b-gpu-bring-up.md), [2C-1](../docs/changes/2026-09-23-phase2c1-raster-equivalence.md), [2C-2](../docs/changes/2026-09-23-phase2c2-window-frame-loop.md), [2D](../docs/changes/2026-09-24-phase2d-ray-query.md). 2E: `scene::GpuScene`, edits reaching both representations (changed regions re-uploaded and their BLAS rebuilt, TLAS rebuilt, swap, retirement), and a ray-query source for the debug views, [2E](../docs/changes/2026-09-24-phase2e-edits.md). 4A ([ADR-0003 Amendment 3](../docs/adr/ADR-0003-device-budget-and-visibility-buffers.md)): `emitters`, the emitter table on the device (`RefEmitters`, 80 B per emitter, `GpuMaterial`); `GpuScene::build_lit` publishes it with the meshes and TLAS, every update rebuilds it (`EmitterSet`) and swaps it with them, and `GpuScene::emitters` refuses a table of another snapshot; `GpuScene::build` has none, [4A](../docs/changes/2026-09-25-phase4a-emitters.md) |
| `viewer` | 3B: the light view (key 0, default; real-time lighting of `gpu::shade`), the M1 lit view on key 9, time of day (`[` `]`, `T`, `--hour`, `--run-day`); 3C: the sky and its light; 3D: accumulation (H), history views (8, `--view 10`–`12`), `--max-age`; 3E: the filter (N, `--no-denoise`), edits relight what they change; 3F: one bounce (B, `--no-bounce`); the sky correction (S-020, `--no-sky-correction`); fps average and 10% / 1% lows in the title (last 1000 frames) and on exit (whole run, also `"fps"` in the JSON). Phase 2C-2: the window. Street block through a 1C snapshot, 7 debug views, 2 frames in flight, per-pass GPU timing, scripted runs (`--frames N`). M1: walking camera (default; F toggles free-fly), lit view (default), frame-phase diagnostics (`--trace`), optional display pacing (`--present-wait N`). M2 (2E): voxel edits (left click removes, E or middle click places), R shows the ray-query result, `--edit-script`, `--source`, MAILBOX by default. 3G: `--edit-size N` (the edit script removes and restores N³ boxes; the JSON lists every edit-to-visible time); the Q2 comparison: P switches the filter between the 3E filter and `--prefilter-age N` (default 8), `--camera low` starts at the 3A low camera. 4A: `--scene street|night` (default `street`, the M3 street block) and `--dressing lamps|windows|full|dense` (default `full`); the night scene keeps its emitter table current through edits, and the JSON reports `scene`, `emitters` and the table's time per edit (`edits.emitters_ms`); its lights are not rendered in real time until 4B. Records: [2C-2](../docs/changes/2026-09-23-phase2c2-window-frame-loop.md), [M1 steps](../docs/changes/2026-09-23-m1-walk-shade-gate.md), [2E](../docs/changes/2026-09-24-phase2e-edits.md), [4A](../docs/changes/2026-09-25-phase4a-emitters.md) |
| `walk` | M1 walking collision: an axis-aligned box against the authoritative world, exact per-axis voxel sweeps (no tunnelling), fixed 240 Hz substeps, step-up and ground snap within `step_height`, gravity and jumps. Pure CPU. Record: [M1 steps](../docs/changes/2026-09-23-m1-walk-shade-gate.md) |
| `light` | 3C: `sky`, the real-time sky tables (Hillaire 2020 + Bruneton's ground irradiance), the oracle of the GPU sky; `sky_ref` (S-020): the baked reference sky that corrects the sky-view table, its file `data/sky_reference_v1.bin` and format. Phase 3A light transport ([ADR-0005](../docs/adr/ADR-0005-light-transport-conventions.md)): units (E_SUN = 1), the sun path through the day, the physical atmosphere (Rayleigh + Mie + ozone, Monte Carlo sky), the shared PCG32 RNG and samplers, and the CPU reference path tracer over `world::reference::trace`. Pure CPU. Record: [3A](../docs/changes/2026-09-24-phase3a-reference.md). 4A ([ADR-0005 Amendment 3](../docs/adr/ADR-0005-light-transport-conventions.md)): `emitters`, the emitter table (one emitter per emissive mesh quad, sorted by geometry, alias selection by power with exact realized probabilities, solid-angle rectangle sampling) and the reference's emitter terms (`Settings::emission`, `emitters_direct`, `emitters_indirect`, off by default; `render_lit`); `emitters::lights_on` (the lights are on while the sun is below the horizon); `exposure::metric_exposure` (the metric exposure from a reference image, for night comparisons). Record: [4A](../docs/changes/2026-09-25-phase4a-emitters.md) |
| `derived` | Phase 1C versioned jobs, bounded queue, atomic group publication and reader-safe retirement for data derived from `world`; Phase 2A per-brick surface meshes as the product. Records: [1C](../docs/changes/2026-09-23-phase1c-jobs-retirement.md), [2A](../docs/changes/2026-09-23-phase2a-surface-extraction.md). 2E: snapshots share unchanged chunk shards, [2E](../docs/changes/2026-09-24-phase2e-edits.md) |

## `world` in one screen

- **Layout:** a chunk hierarchy of sparse bricks. Chunks are keyed by `ChunkCoord` in a `BTreeMap`; each has 64 optional brick slots. Only non-empty bricks and chunks exist.
- **Dimensions** (`world/src/dims.rs`, starting tuning values):
  - 8³-voxel bricks, 4³-brick chunks (32³ voxels, 2 m)
  - voxel 1/16 m
  - right-handed, Y up
- **Brick payload:** a 512-bit occupancy mask and a separate `u16` `MaterialId` array.
  - No material means "empty": emptiness lives in the mask.
  - Unoccupied slots hold a canonical value, so content comparison is exact.
- **Materials:** `MaterialRegistry` is append-only.
  - IDs are registry indices and stable once assigned; names are unique.
  - Numerical parameters (`MaterialParams`) are stored separately from the ID.
  - Writing an unregistered ID is rejected.
- **Identity and versions:**
  - `BrickKey` (chunk + slot) is the logical brick identity.
  - Every real content change stamps the brick with the next world-wide `ContentVersion`. No-op writes stamp nothing.
  - Freeing a brick advances the world version, so a recreated brick never reuses an earlier version.
- **Reference query** (`world/src/reference.rs`):
  - `trace` is a voxel DDA; `trace_brute` is the brute-force oracle.
  - Both use the half-open-voxel convention in the module docs, and must agree exactly.
- **Test scenes:** `scene::street_block()` is a deterministic procedural street with three shops and street lamps (unchanged since Phase 2). `scene::street_night(Dressing)` (4A) is the same geometry and material ids with the lamps and signs at night luminance, plus lit windows, neon and string lights by level (`Lamps`, `Windows`, `Full`, `Dense`; 165 / 325 / 1,143 / 7,003 emitter quads). Emitted radiance is in ADR-0005 units; `material::CANDELA_PER_UNIT` converts from cd/m².
- **Transactions** (`world/src/edit.rs`): `Transaction` is an ordered list of `Set`/`Fill` ops. `World::apply` is atomic: materials are validated first, so a rejected transaction changes nothing. It returns `Applied` (versions before/after, changed voxels for `derived::Pipeline::notify_edits`).
- **Persistence** (`world/src/persist/`, [ADR-0002](../docs/adr/ADR-0002-save-format.md)):
  - A store directory holds `world.snap` and `world.journal`.
  - `Store::create`, `commit` (apply, append, `sync_data`), `snapshot` (atomic rewrite plus journal restart), `open` (load and repair a torn tail); `store::load` is read-only.
  - Reload is exact, versions included. Replay checks every record's sequence and versions.
  - Materials are adopted from the file, or mapped by name with a loud failure (`MaterialPolicy`).
  - Format pin: `world/tests/fixtures/v1_small.*`. Regenerate only for an intentional format change: `set NE_WRITE_FIXTURES=1` then `cargo test --release -j 2 -p world --test persist format_matches`.

## `memory` in one screen

- `Usage {live, reserved}` per `Category`, collected in a `Report`. `live` is the exact payload; `reserved` is an upper bound including capacity and container nodes.
- `World::memory()` and `Pipeline::memory()` report what they hold. The pipeline counts every snapshot alive at once. Dispatched jobs are held by the caller: `Job::heap()`.
- `device_budget(driver_heap_budget)` (ADR-0003): min(3.5 GB, driver budget) − 10%, which is 3.15 × 10⁹ bytes on this laptop.
- `Ledger` + `Budget`: pools ask before allocating.
  - Refusal is deterministic and changes nothing but a counter.
  - Tracks high-water and a resettable transient peak; `outstanding()` finds leaks.
  - Not used by any pool yet; Phase 2 GPU and staging pools are its first users.
- `CountingAlloc`: install it as `#[global_allocator]` in a test or tool to measure real heap. The `memory_account` tests use it to check that reported live ≤ measured ≤ reported reserved.

## `derived` in one screen

- **`slots::SlotPool`:** generational handles. A free bumps the slot's generation, a slot whose generation would wrap is retired permanently, and free slots are reused lowest-first.
- **`mesh`** (2A, [ADR-0004](../docs/adr/ADR-0004-surface-mesh-layout.md)): the pipeline's product, a `BrickMesh` of 8-byte `Quad`s.
  - Quads are brick-local, half-open and wound counter-clockwise from outside, each with an exact `MaterialId`.
  - `Merge::{None, Greedy}` (`Config::merge`) is the mesh-merge slider. Both modes cover exactly the same exposed faces.
  - `BrickMesh::triangulate(Split::Watertight)` gives brick-local triangles with every edge shared (ADR-0004 Amendment 1); `Split::CornersOnly` is for controls.
  - `exposed_faces` is the coverage oracle. `trace_mesh` is the CPU ray–quad reference; it agrees exactly with `world::reference::trace` for rays in general position.
- **`surface`:** the dependency record every per-brick surface product shares, and exposed-face counts per material (`SurfaceSummary`), kept as a cross-check of mesh area.
  - `InputSnapshot` holds immutable copies of the brick and its six face neighbours.
  - `Dependencies` is the exact record used to decide staleness: versions first, then content (target brick and six face layers).
- **`pipeline::Pipeline`:** the contract is in its module docs.
  - Edit batches become publication groups, merged when they share a brick.
  - Results are rejected if cancelled or stale.
  - Complete groups publish atomically into a new snapshot.
  - Readers (`acquire`/`release`) keep their snapshot's resources alive until they release.
  - Snapshots are sharded by chunk and share unchanged shards (`Arc`), so a publication copies the chunk index and the changed shards only (2E); `memory()` counts a shared shard once.
  - The queue and in-flight count are bounded, with an ordered overflow set.
- **`derived::Faults`:** planted faults used only by negative-control tests; all are off by default.

## `gpu` in one screen

- `Gpu::new()` requires Vulkan 1.3 plus acceleration structures and ray query, `geometryShader` (fragment `PrimitiveID`) and `dynamicRendering`. `NE_GPU_ALLOW_NO_RT=1` is a diagnostic that accepts a non-RT device; its results are supplemental. `NE_NO_VALIDATION` turns validation off.
- `Allocator`: 64 MiB blocks, each one ledger grant (ADR-0003 budget from `Gpu::device_budget`), sub-allocated first-fit. A refusal changes nothing.
- `Timeline` + `Retirement<T>`: free only after the GPU passes a value. `FrameReaders` does the same for 1C `ReaderToken`s.
- `Uploader` is a staging ring with back-pressure; `staging::download` does batched readback.
- `layout::build_regions` builds ADR-0004 region images (`RegionSize` is the region slider): vertices, indices, the per-triangle quad index `tri_quad`, and quads. `GpuMeshes::upload` is all-or-nothing.
- **Shaders:**
  - `build.rs` compiles `gpu/shaders/*.slang` with the pinned `slangc` 2026.13.1, runs `spirv-val` and allow-lists SPIR-V capabilities.
  - `decode::Decoder` checks the reflection JSON against the host layout before creating its pipeline.
- **Raster (2C-1):**
  - `raster::Camera` holds the conventions: right-handed Y-up, camera-relative, reverse-Z infinite, negative viewport height, CCW front faces, pixel centres.
  - `raster::Targets` are the ADR-0003 G-buffer, under `GpuTemporal`.
  - `raster::Raster` checks bindings, push constants and varyings from reflection first.
  - `equivalence::{reference, compare}` is the ADR-0003 Amendment 1 check against the CPU DDA. Its tolerances are declared in the module docs.
- **Window and frame (2C-2):**
  - `Gpu::with_surface` makes a presenting device; the caller supplies the surface (the viewer uses `ash-window`).
  - `present::{Swapchain, Frames}`: the caller picks the present mode (the viewer defaults to MAILBOX since 2E); per frame a scene submission before the acquire and a present submission after it; `Frames::begin` waits for the slot's previous frame (2 in flight in the viewer).
  - `debug_view::{DebugView, Tables, View}`: seven views of the G-buffer, read with `Load`; sampled images declared with format Unknown.
  - `View::Lit` (M1) in the same pass: base colour × (sun × N·L + hemisphere ambient) + emissive, exposure, ACES fit; no shadows. Tunables in `debug_view::Lighting`; material rows (`MaterialRow`: base, emissive) under `GpuMaterial`; 128 B push constants.
  - `timing::GpuTimer`: begin/end timestamps per pass per slot; `percentile` (nearest rank).
  - Swapchain images are driver-owned and not in the ledger (`Swapchain::estimated_bytes`).
- **Ray query (2D):**
  - `accel::Accel::build`: one BLAS per region, from the region's own mesh buffer (the vertices and indices raster draws); one TLAS, with the instance custom index = region index; a region table of quad and `tri_quad` device addresses and lengths.
    - `GpuAccel` holds BLAS, TLAS, instances and table; build scratch is under `GpuAccelScratch`, freed when the build completes, and batched to at most 32 MiB.
    - All-or-nothing on a refused grant. Instances carry no facing flag: Vulkan's default facing matches the ADR-0004 winding.
  - `ray::RayPrimary`: one ray-query ray per pixel centre, writing the G-buffer's values (depth, normal bits, material, surface id) to `RayTargets` (24 B/px, `GpuTemporal`). Every table read is bounds-checked; a bad id writes `BAD_MATERIAL`.
  - `equivalence::agreement` compares two frames directly (raster against ray); only non-edge disagreements fail.
- **Edits (2E):**
  - `scene::GpuScene` is the device copy of one published 1C snapshot: region meshes plus (on RT devices) the `Accel`.
  - `affected_regions` + `region_meshes` rebuild the changed regions from a snapshot; `GpuScene::update` uploads them into new buffers (`GpuMeshes::upload_regions`), rebuilds their BLAS and the top level (`Accel::update`), then swaps. It is all-or-nothing: a refused grant leaves the previous snapshot, counts a deferral and returns the error, and the caller retries.
  - Replaced buffers, BLAS, the old top level and descriptor pools go to `GpuScene::retire` and are freed by `collect` after the last submission that may read them. The acceleration update waits for its build (and so for earlier frames); the mesh-only path does not wait.
  - `debug_view::Source::{Raster, Ray}` selects what the views read; `Tables::replace_regions` follows the scene. Faults for controls: `scene::SceneFaults`.
- **Reference (3A):** `reference::Reference` is the ADR-0005 path tracer on the GPU (`shaders/reference.slang`): one sample per pixel per dispatch into `RefAccum` (sum and sum of squares, 32 B/px, `GpuTemporal`), albedos in `RefMaterials` (`GpuMaterial`). It draws the same PCG32 streams as `light::reference` and is checked against it statistically (`tests/reference.rs`). `probe_transmittance` checks the atmosphere constants; `RefFaults` are the planted controls. `ref_light` renders the reference set.
- **Real-time lighting (3B):** `shade::Shade` (`shaders/shade.slang`) reads the G-buffer (after `debug_view::targets_to_read`) and the TLAS, and writes linear HDR radiance to `ShadeTargets` (16 B/px, `GpuTemporal`): the direct sun with one shadow ray, on the reference's PCG32 streams, so frame f equals the reference's sun-only sample f per pixel (`tests/shade.rs`). `debug_view::View::Light` tone-maps it (`bind_radiance`, `Lighting::light_exposure`). Shared shader code: `shaders/light_common.slang`.
- **Sky (3C):** `sky::SkyTables` holds the `light::sky` tables on the device (transmittance, multiple scattering and ground irradiance built once on the host in f64 and uploaded; the sky-view table rebuilt on the GPU by `sky::SkyView` when the sun moves; `shaders/sky.slang`, `sky_common.slang`). `shade` adds the sky term (one cosine visibility ray per pixel into the table sky) and the sky behind the scene. Tests: `tests/sky.rs` (one known failure is ignored with its reason: the 3C record).
- **Sky correction (S-020):** the sky-view pass multiplies each texel by the ratio reference ÷ table (`SkyTables::corr`, from `SkyLuts::apply_reference`; exactly 1 for `SkyLuts::new`, so the 3B–3F tests keep their lighting). The reference is baked once by `sky_bake` (`sky_bake::bake`, `shaders/sky_bake.slang`) into `light/data/sky_reference_v1.bin`; rerun it if the atmosphere or the sky estimator changes (the file is refused otherwise). Tests: the `corrected_*` tests in `tests/sky.rs` and `tests/bounce.rs`; record: [sky correction](../docs/changes/2026-09-24-phase3c-sky-correction.md).
- **Temporal (3D, ADR-0006):** `temporal::Temporal` reprojects every surface pixel into the previous frame, validates four bilinear taps against the stored guides (face, integer plane, material, region key and snapshot) and accumulates (age capped at `max_age`); `History` owns the ping-pong guide / colour / state buffers and motion (72 B/px, `GpuTemporal`) and decides global resets. The resolved colour and state go back into the shade radiance buffer; views 9–11 show age, reason and motion. Tests: `tests/temporal.rs`. 3E (ADR-0006 Amendment 1): edits pass `temporal::Relight` boxes, and pixels whose sun ray crosses one or that lie within its sky radius restart as *relit*; the age cap shrinks while the sun moves (`TemporalSettings::sun_tolerance_deg`); the history is resampled with Catmull-Rom when all 16 taps are valid (`bilinear` restores 3D).
- **One bounce (3F):** `ShadeSettings::bounce` (on by default) follows the sky ray to its hit and lights that point by the sun (a shadow ray) and the sky (a visibility ray), exactly as the reference's second vertex with `max_bounces` 1; the hit's material comes from the region table (`Accel::table`, binding `regions`). Up to 4 rays per pixel. Tests: `tests/bounce.rs` (per pixel against the one-bounce reference, convergence, missing energy of higher bounces; criteria in the [3F record](../docs/changes/2026-09-24-phase3f-bounce.md)). The 3B–3E test files pin `bounce: false` (`DIRECT`), the lighting their criteria were measured with.
- **M3 gate (3G):** `tests/gate.rs` runs the viewer's frame (sun, the corrected sky, one bounce, the temporal pass and the filter) at 1080p against the one-bounce reference (stills on both 3A cameras at the five reference times, the 3E motion path) and times the frames after a history reset. With `NE_GATE_DIR` set it caches the references there and writes display images for `results/phase3g/flip.py` (FLIP, A-008). `results/phase3g/gate_perf.py` runs the performance series and `gate_edits.py` the edit budget in the viewer. Criteria and results: [3G record](../docs/changes/2026-09-25-phase3g-gate.md).
- **Reconstruction (3E):** `denoise::Denoise` (`shaders/denoise.slang`, SVGF-style) demodulates the resolved radiance by albedo, estimates variance from the history moments, runs à-trous levels (default 2) that mix only pixels on the same surface (exact guides) with luminance edge-stopping (the first level on 3×3 means), and remodulates into the shade radiance buffer; `DenoiseTargets` 44 B/px, `GpuTemporal`. Diagnostics: `--ignored sweep_sliders`, `denoise_cost_breakdown`, `motion_images`, `diagnose_motion_edges` (`NE_DIAG_ARM`, `NE_DIAG_HARD_ONLY`, `NE_DIAG_NO_HARD`). Tests: `tests/denoise.rs` (against the estimator converged on a still camera; criteria in the [3E record](../docs/changes/2026-09-24-phase3e-denoise.md)).

## Verified commands (2026-09-23, this laptop)

Run from `engine/`:

```bat
cargo test --release -j 2
cargo test -j 2
cargo build --release -j 2
target\release\ref_render.exe results\street_ref.ppm
target\release\persist_cost.exe %TEMP%\ne_persist_cost
target\release\perf_phase1.exe --rounds 5
cargo clippy --release -j 2 --workspace --all-targets
cargo test --release -j 2 -p gpu -- --test-threads=1
cargo test --release -j 2 -p gpu --test raster -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test ray -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test edit -- --test-threads=1 --nocapture
set NE_NO_VALIDATION=1 && cargo test --release -j 2 -p gpu --test edit -- --test-threads=1 --nocapture edits_reach
python ..\phase0-probes\compare.py results\street_ref.ppm
cargo build --release -j 2 -p viewer
target\release\viewer.exe
target\release\viewer.exe --frames 600 --cycle-views --resize-at 300 1600x900 --log results\viewer_2c2.jsonl
target\release\viewer.exe --frames 1200 --size 1920x1080 --present mailbox --merge none
target\release\viewer.exe --frames 1500 --walk --size 1920x1080 --present mailbox --log results\viewer_m1.jsonl
set NE_NO_VALIDATION=1 && target\release\viewer.exe --frames 3000 --walk --size 1920x1080 --trace results\trace.csv
set NE_NO_VALIDATION=1 && target\release\viewer.exe --frames 10000 --walk --size 1920x1080 --present fifo --present-wait 1 --log results\gate.jsonl
target\release\viewer.exe --frames 2400 --edit-script --source ray --size 1920x1080 --log results\viewer_2e.jsonl
cargo test --release -j 2 -p light
cargo test --release -j 2 -p gpu --test reference -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test shade -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test sky -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test temporal -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test denoise -- --test-threads=1 --nocapture
cargo test --release -j 2 -p gpu --test bounce -- --test-threads=1 --nocapture
target\release\viewer.exe --frames 1500 --walk --size 1920x1080 --hour 5.5 --run-day --log results\viewer_3c_day.jsonl
set NE_NO_VALIDATION=1 && cargo test --release -j 2 -p gpu --test shade -- --test-threads=1 --nocapture shade_cost
target\release\viewer.exe --frames 1200 --edit-script --size 1920x1080 --hour 17.5 --log results\viewer_3b.jsonl
cargo build --release -j 2 -p gpu --bin ref_light
cargo build --release -j 2 -p gpu --bin sky_bake
target\release\sky_bake.exe > results\sky_bake_k.jsonl
target\release\ref_light.exe results\phase3a_ref --size 960 540 --spp 2048
set NE_GATE_DIR=%TEMP%\ne_gate && cargo test --release -j 2 -p gpu --test gate -- --test-threads=1 --nocapture
python results\phase3g\flip.py %TEMP%\ne_gate results\phase3g
set NE_NO_VALIDATION=1 && cargo test --release -j 2 -p gpu --test gate -- --test-threads=1 --nocapture reset_frame_cost
python results\phase3g\gate_perf.py results\phase3g\perf
python results\phase3g\gate_edits.py results\phase3g\edits
python results\phase3g\gate_edits.py results\phase3g\edits --validation
```

Notes:
- `-p gpu` needs `VULKAN_SDK` (SDK 1.4.357.0) and an RT GPU. Verified on the RTX 3050 on 2026-09-23: 17 of 17 at 2B (`results/test_gpu_2b_rtx.log`); 24 of 24 after 2C-1 and ADR-0004 Amendment 1 (`results/test_gpu_2c1_watertight_rtx.log`). Without an RT GPU it fails with `NoDevice`, by design. `set NE_GPU_ALLOW_NO_RT=1` allows a supplemental run on a non-RT device; it passed on the Intel iGPU (`results/test_gpu_2b_igpu_supplemental.log`).
- `--test raster` (2C-1) passes on the RTX 3050 since ADR-0004 Amendment 1 (watertight greedy): 0 cracks in all 32 settings. Full `-p gpu` run: 11 + 9 + 4 tests, `results/test_gpu_2c1_watertight_rtx.log`. The pre-fix run is `results/test_gpu_2c1_rtx.log`. It writes `results/raster_2c1_*.bmp` for viewing.
- `--test ray` (2D, 2026-09-24): ray query agrees with the CPU reference and with raster per pixel in all 32 settings × cameras, the planted faults are caught, and the ledger holds exactly the acceleration buffers (`results/test_gpu_2d_ray_rtx.log`). Full `-p gpu` after 2D: 12 + 9 + 5 + 4 tests (`results/test_gpu_2d_rtx.log`, debug `results/test_gpu_2d_rtx_debug.log`). Record: [2D](../docs/changes/2026-09-24-phase2d-ray-query.md).
- `--test edit` (2E, 2026-09-24): edits reach raster and ray query in all 8 settings, checked per pixel against the CPU reference of the edited world after the 1-voxel, 8³ and 32³ edits, and byte for byte against a from-scratch build; refused updates keep the previous snapshot; planted faults are caught. With `NE_NO_VALIDATION=1` it is the timing run of the granularity sweep (`SWEEP` JSON lines); validation checks are then NOT RUN, as the log says. See the [2E record](../docs/changes/2026-09-24-phase2e-edits.md).
- The debug test run takes several minutes, because of the randomized world and pipeline tests.
- `ref_render` usage: `ref_render <out.ppm> [--size W H] [--threads N]` (default 640×360, 2 threads). It prints one JSON line. The image does not depend on the thread count.
- `compare.py` writes a PNG copy next to the PPM, and needs numpy and Pillow.
- `perf_phase1 [--rounds N]` is the CPU baseline: generation, edits, derived build by worker count, mesh size and build time per merge mode (2A), publication scaling, edit-to-publish latency and reference trace. It prints JSON lines with raw samples and verifies every snapshot. Outputs from 2026-09-23: `results/perf_phase1.jsonl` (Phase 1, face-count product) and `results/perf_phase2a.jsonl` (meshes). Machine speed varied between those runs, so compare arms only within one run.
- `viewer` (2C-2): controls are in `viewer/src/main.rs`. `--frames N` flies a fixed path, prints one JSON summary and exits non-zero on any validation error or warning or a leak. Verified 2026-09-23: FIFO 720p with resize, MAILBOX 1080p greedy and unmerged, a debug build; all clean (`results/viewer_2c2.jsonl`). Full `-p gpu` after 2C-2: 12 + 9 + 5 tests (`results/test_gpu_2c2_rtx.log`).
- Granularity default (ADR-0003 Amendment 2): greedy / chunk; `--region 2x2x2_chunks` is the alternative for larger, less dense scenes.
- `viewer` (M1): walking starts at the reference view's feet; `--walk` scripts the walker in a closed 1308-frame loop through the street (restarting at the spawn point each loop) and fails the run on any overlap with a solid voxel or any fall off the world. `--present-wait N` (off by default; needs `VK_KHR_present_wait`) waits before each frame until at most N presents are not yet displayed. The summary adds the frame-phase breakdown (`begin`, `acquire`, `present`, `inside`), every interval over 16.67 ms, the refresh rate and focus/occlusion counts. Verified 2026-09-23 (`results/viewer_m1*.jsonl`); 60 fps gate results in the M1 record. Pure suite after M1: 114 tests (`results/test_release_m1.log`); full `-p gpu`: 12 + 9 + 5 (`results/test_gpu_m1_rtx.log`). Rerun after present wait and the loop, 2026-09-24: the same counts in release and debug (`results/test_*_m1_close*.log`); the 60 fps gate on the looping walk passes (`results/m1_gate_loop/`).
- `--test reference` and `light` (3A, 2026-09-24): results in the [3A record](../docs/changes/2026-09-24-phase3a-reference.md). `ref_light <out dir> [--size W H] [--spp N] [--bounces N] [--times a,b] [--scene street|night [--dressing ...]]` writes `<camera>_<time>.pfm` (linear HDR) and `.ppm` (display only) and one JSON line per image; 960×540 at 2048 spp takes about 17 s per image on the RTX 3050. Since 4A part 2 the display images use `light::exposure::metric_exposure` (the 3A images were written with the earlier log-average, which differs only in how black pixels count), and the tool exits non-zero on any bad sample. `--scene night` renders `street_night` at the M4 times (dusk, blue hour, night) with its emitter table and the lights by `lights_on`.
- `clippy` currently reports 6 pre-existing style lints and nothing else; see `results/clippy_m1_close.log`.
- `--test emitters` (4A, `gpu::emitters`, `shaders/emitters_common.slang`): the GPU reference's emitter terms against the CPU; criteria G1–G5 in the [4A record](../docs/changes/2026-09-25-phase4a-emitters.md); passed on the RTX 3050 on 2026-09-25 (`results/local-run/2026-09-25_1912-4a/`). `--ignored diagnostic_g4_cpu_convergence` is CPU only (no device) and checks the CPU side of G4 at 4096 samples. `Reference::bind_lit` binds an emitter table; `bind` binds none and refuses emitter settings. `gpu --lib` includes a device-free check of the reference module's reflection against the host layout.
- 4A part 2 (the [4A record](../docs/changes/2026-09-25-phase4a-emitters.md); G6–G9 passed on the RTX 3050 on 2026-09-25, `results/local-run/2026-09-25_2029-4a2/` and `results/phase4a_ref/`; each night reference takes about 90–145 s at 16,384 spp): `--test emitters` gains G6 (the table in `GpuScene` through edits); `viewer.exe --scene night --size 1920x1080 --frames 2000 --edit-script --edit-size N` with `NE_NO_VALIDATION=1` measures the edit latency with the table; `ref_light.exe results\phase4a_ref --scene night --spp 16384` renders the night references. `gpu --lib` (C6, device-free) and `light` (C7, the lights rule and the metric exposure) run anywhere. `cargo test --release -j 2 -p light --test emitters diagnostic_magnitudes -- --ignored --nocapture` measures M1–M2 on the CPU (about 20–40 min on 4 threads).
- Cloud sessions (no GPU) prepare `run-local.cmd` in the repository root for the checks they cannot run ([CLOUD.md](../docs/CLOUD.md)); its logs go to `results/local-run/<date>-<task>/`.
- `persist_cost <scratch dir> [--trials N] [--commits N]` saves and loads the street block in a scratch directory (removed afterwards), and prints one JSON line of sizes and timings. Output from 2026-09-23: `results/persist_cost.json`.

## Not here yet

- A byte budget on CPU-side pools. Derived pools are bounded by world size and by the 1C queue limits.
- Off-thread jobs and a no-wait acceleration update (the edit frame costs about 4 ms of host time); BLAS refit.
- BLAS compaction (not measured).
- Light from two or more bounces: the real-time image has the direct sun (3B), the sky (3C) and one bounce of both (3F) at one sample per pixel, accumulated over frames (3D) and filtered (3E). The missing share is reported in the [3F record](../docs/changes/2026-09-24-phase3f-bounce.md).
- A clean 60 fps result under every condition: on this laptop FIFO passes the street walk (p99 about 14.6 ms) but failed on sky-heavy views, cause undiagnosed (M1 record §3b).
- `ref_render` (world) shading is a debug view; the lighting reference is `ref_light`.
