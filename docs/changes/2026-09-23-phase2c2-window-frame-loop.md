# Change: Phase 2C-2 — window, frame loop, debug views, free-fly camera, GPU timing

Status: **implemented; release and debug checks pass on the RTX 3050.** This completes Phase 2C. It is not the 60 fps gate.
Date and baseline: 2026-09-23, after [2C-1](2026-09-23-phase2c1-raster-equivalence.md) and the [watertight greedy fix](2026-09-23-phase2c1-watertight-greedy.md). No Git repository.
Authorization (S-011), user: "If there is a need to run debug versions of test run them if not then greenlight for 2c-2". Debug runs were needed (the code had changed since the last ones), so they were run first; see "Debug runs of S-010".
Plan: the [2C proposal](2026-09-23-phase2c-proposal.md), slice 2C-2.

## What was built

**Dependencies (A-006):**
- `winit =0.30.13` and `ash-window =0.13.0`, in the new `viewer` crate only; `gpu` stays windowless.
- Resolved with `--offline`. Every locked version is identical to the Phase 0 probe lock (`phase0-probes/rust/Cargo.lock`).
- The lock lists winit's cross-platform graph, but 22 crates compile for this target: ash, ash_window, bitflags, cfg_aliases, cursor_icon, dpi, libloading, pin_project_lite, raw_window_handle, smol_str, tracing, tracing_core, unicode_segmentation, windows_link, windows_sys, windows_targets, windows_x86_64_msvc, winit, and the engine crates.

**`gpu`:**
- `context`: `Gpu::with_surface(extensions, make_surface)`.
  - The caller's instance extensions are enabled, and the surface is created from the new instance and owned by the `Gpu`.
  - A device qualifies only with `VK_KHR_swapchain` and a queue family that presents to the surface.
  - `DeviceInfo::timestamp_period` added.
- `present`: `Swapchain` and `Frames`.
  - FIFO by default. MAILBOX or IMMEDIATE only when asked for and supported.
  - `B8G8R8A8_SRGB` when offered; per-image "rendered" semaphores; recreate on resize, out-of-date or suboptimal.
  - **Two submissions per frame:** the scene (G-buffer) is submitted *before* the swapchain acquire, with no wait. The view pass is submitted after it, waiting on the acquire at colour output. (The first version used one submission, so the acquire wait also held the G-buffer's colour writes; it was split before any measurement was recorded.)
  - 2 frames in flight: `Frames::begin` waits for the slot's last timeline value.
- `debug_view` plus `shaders/debug_view.slang`: a fullscreen pass reading the targets with `Load`. Seven views: material colour (`MaterialParams` base colour, face-axis shaded), normal, region hash, brick hash (from reconstructed position), surface-id hash, linear depth, snapshot version.
  - `Tables`: material colours under `GpuMaterial`, and the region table (key and 1C snapshot id) under `GpuMesh`.
  - Reflection is checked before any Vulkan object is created.
- `timing`: `GpuTimer`, a begin and end timestamp per pass per frame slot, read only after the slot's frame completes. `percentile` is nearest-rank.
- `raster::Targets` gained `SAMPLED` usage. `reflect::varyings` now reads a bare (non-struct) fragment result.

**`viewer` (new crate, not a default member):**
- Builds the street block through a 1C `derived::Pipeline` snapshot and uploads it.
- Each frame acquires a reader token, and `FrameReaders` holds it until that frame's timeline value completes.
- An assertion checks that at most 2 tokens are held (one per frame in flight). Shutdown asserts none are left, and that the allocator leaked nothing.
- Free-fly camera: right mouse button to look; W A S D, Space and Ctrl to move; Shift for ×4; the wheel sets speed. Keys 1–7 choose the view.
- The title shows fps, frame p50/p99 and GPU pass times each second.
- `--frames N`: a scripted camera path indexed by frame number, then one JSON summary line (`--log` appends it). The run exits non-zero on any validation error **or warning**.
- `--resize-at FRAME WxH` exercises recreation under the same checks.

**Companion change:** `derived::SnapshotId::raw()`, a read-only accessor for the snapshot-version view and the logs. It is a small public-interface addition.

## Problems found and fixed on the way

- **SPIR-V capabilities:** the `build.rs` allow-list stopped two unplanned capabilities.
  - `SV_VertexID` → DrawParameters (4427). Replaced with `SV_VulkanVertexID`, which is the same value with first vertex 0.
  - Two-component texel types → StorageImageExtendedFormats (49).
- **Sampled-image formats:** Slang gave the sampled images explicit Format operands (`R32ui`, `Rgba32ui`). They differ from the views (`R16_UINT`, `R32G32_UINT`), which the spec makes undefined.
  - Validation reported it only as a *warning*, while every pixel still matched.
  - Fixed with `[[vk::image_format("unknown")]]`, which also removes capability 49.
  - The debug-view test and the viewer now fail on warnings as well; that first run is the evidence the check can fail.

## Results (RTX 3050, validation on unless stated)

**Debug views, headless:** `debug_views_show_the_stored_gbuffer_values_and_the_timer_measures` in `gpu/tests/raster.rs`, street_view camera, greedy meshes, chunk regions, 640×360.
- Every pixel of every view is re-derived on the CPU from the read-back G-buffer. The tolerances were declared before measuring: 1 LSB (2 for depth, because of the GPU's `log2`), and up to 0.1% of drawn pixels for the brick view.
- Result: **0 of 230,400 pixels differ in each of the 7 views** (173,137 drawn).
- Planted control: the raster modules' reflection handed to the debug view is refused, with 20 layout errors.
- The timer returns nothing before a slot is written, and positive times after.

**Viewer, scripted:** every run exits 0, with 0 validation errors and 0 warnings. Summaries: `engine/results/viewer_2c2.jsonl`; logs `viewer_2c2_*.log`.

| Run | Extent, present | Recreates | CPU frame ms p50 / p99 / max | GPU G-buffer ms p50 / p99 | GPU view ms p50 |
|---|---|---|---|---|---|
| greedy, all 7 views, resize at frame 300 | 1280×720 → 1600×900, FIFO | 2 | 6.93 / 14.8 / 27.8 | 0.177 / 0.234 | 0.042 |
| greedy, rep 1 / 2 | 1920×1080, MAILBOX | 1 | 3.05 / 6.34 / 18.1 and 2.82 / 6.29 / 13.0 | 0.213 / 0.361 and 0.214 / 0.312 | 0.076 |
| none (unmerged), rep 1 / 2 | 1920×1080, MAILBOX | 1 | 2.91 / 6.09 / 20.0 and 2.81 / 6.51 / 17.6 | 0.387 / 0.479 and 0.389 / 0.482 | 0.079 |
| greedy, validation off, 2 reps | 1920×1080, MAILBOX | 1 | 1.72 / 7.14 and 1.53 / 7.71 | 0.215 / 0.363 and 0.216 / 0.313 | — |
| none, validation off, 2 reps | 1920×1080, MAILBOX | 1 | 1.82 / 11.0 and 1.83 / 10.5 | 0.391 / 0.559 and 0.392 / 0.558 | — |
| debug build, greedy, all views, resize | 1280×720 → 1600×900, FIFO | 2 | 6.91 / 18.3 / 28.0 | 0.177 / 0.235 | 0.043 |

Observations (not claims about the 60 fps gate):
- FIFO paces at the panel's 144 Hz (6.9 ms).
- At 1080p the G-buffer takes 0.21 ms (greedy, 276 k triangles) or 0.39 ms (unmerged, 1.04 M) of GPU time. Frame time is set by the CPU, not the GPU.
- With validation off, p50 falls to about 1.6–1.8 ms, but p99 rises to 7–11 ms, and the max is about 19–20 ms in every arm. The cause of those tails is **not diagnosed**: candidates include OS scheduling, the compositor and MAILBOX pacing. It belongs to the 60 fps gate step, not to 2C-2.
- Ledger at 720p: GpuMesh 6.82 MB, GpuMaterial 272 B, GpuTemporal 17.7 MB (the targets), Staging 16.8 MB (the ring). Swapchain images are driver-owned and outside the ledger: about 11 MB at 720p and 24.9 MB at 1080p (3 images × 4 B/px, an estimate).
- The window's startup mesh build is 0.11 s: pipeline, extraction and region build for the street block.

**Other gates:**
- `cargo test --release -j 2 -p gpu -- --test-threads=1 --nocapture`: exit 0, 12 unit + 9 device + 5 raster tests.
  - The only validation error is the planted early-free control.
  - The 32 equivalence lines are byte-identical to the S-010 run, so `SAMPLED` usage changed nothing.
  - Log: `engine/results/test_gpu_2c2_rtx.log`.
- `cargo test --release -j 2 --no-fail-fast`: exit 0, 101 tests (`results/test_release.log`).
- `cargo clippy --release -j 2 --workspace --all-targets`: exit 0. The same 6 pre-existing lints, all in `world`; none in `gpu` or `viewer` (`results/clippy.log`).

## Debug runs of S-010 (run first, as asked)

- `cargo test -j 2 --no-fail-fast` (pure crates, debug): exit 0, 101 tests. Log: `results/test_debug.log`.
- `cargo test -j 2 -p gpu -- --test-threads=1 --nocapture` (debug): all 24 tests of the S-010 code passed: 11 unit, 9 device and 4 raster, with 0 cracks in all 32 settings. The debug reference takes 42–87 s per camera.
  - The command still **exited 1**, at the doc-test step. It compiled the `gpu` sources I had started editing for 2C-2, against shaders not yet built.
  - The test binaries had been built from the S-010 code before those edits. The log is kept as-is: `results/test_gpu_2c1_watertight_rtx_debug.log`.

## Risks

- Frame-time tails without validation (p99 up to 11 ms, max about 20 ms at 1080p) are undiagnosed. A 20 ms frame misses 60 fps.
- Swapchain memory is outside the ADR-0003 ledger. At 1080p that is about 25 MB of a 3.15 GB budget, but it is not budgeted.
- The viewer's interactive input (mouse look, keys, wheel) was not exercised by any scripted run; only the scripted path and resize were.
- Resize is tested only through `request_inner_size`; minimize and restore were not tested.
- The CPU frame loop draws one call per region (291 at chunk regions). CPU cost per region has not been measured against the region-size slider.
- The depth-tolerance risk from 2C-1 (0.95 of the tolerance at grazing incidence) is unchanged.

## NOT RUN

- the 60 fps p99 gate (a later M1 step)
- interactive input, minimize and restore
- region sizes other than chunk in the viewer
- a cold-process timing series
- 1080p equivalence
- BLAS and ray query (2D)

## Closeout

- **Docs:** this record; the [2C proposal](2026-09-23-phase2c-proposal.md) status; [DECISIONS](../DECISIONS.md) (S-011, A-006); `engine/README.md`; `docs/NOW.md`.
- **Debug GPU suite for the 2C-2 code:** `cargo test -j 2 -p gpu -- --test-threads=1 --nocapture` exits 0: 12 + 9 + 5 tests and the doc-test step. The only validation error is the planted early-free control; the 7 views are pixel-exact, and the equivalence lines are identical to the release run. Log: `results/test_gpu_2c2_rtx_debug.log`.
- **Revert:**
  - delete `viewer/`, `gpu/src/{present,debug_view,timing}.rs` and `gpu/shaders/debug_view.slang`
  - remove the workspace member and the `build.rs` debug-view entries
  - remove `Gpu::with_surface` and `timestamp_period`, `SAMPLED` on the targets, the bare-result branch in `reflect::varyings`, and `SnapshotId::raw`
  - rebuild so cargo drops the viewer's lock entries (the pre-change lockfile was only copied to the session scratchpad)
