# Change: M1 steps — walking camera with voxel collision, simple shading, the 60 fps gate

Status: **M1 accepted as met by the user on 2026-09-24** ("I accept M1 as met"; they run on mains power, so battery behaviour does not concern them). **Walking and shading implemented, all checks pass. The 60 fps gate passes on the representative (looping street) walk: every undisturbed run in all three arms (§3b, 2026-09-24).** The earlier series (§3) ran a flawed script that mostly showed sky; FIFO failed there in an unexplained regime, which remains a risk for sky-heavy views.
Date and baseline: 2026-09-23, after [2C-2](2026-09-23-phase2c2-window-frame-loop.md). No Git repository.
Authorization (S-012), user: "worked fine looked meh but that is to be expected at this point I will follow your reccomendations green light given", then "Go ahead". The recommended order: walking camera with collision, simple shading, then the 60 fps gate (diagnosing the tails first).

## 1. Walking camera: new pure crate `walk`

A default workspace member with no external dependencies (depends on `world`), so its tests run in the pure suite.

- **Collision shape:** the walker is an axis-aligned box: 0.5 m wide, 1.8 m tall, eyes at 1.7 m (voxel units internally). Every occupied voxel is solid; materials carry no collision flag yet.
- **Fixed substeps:** motion runs at 240 Hz. `Walker::update` carries the remainder of each frame, and `render_eye` interpolates into it, so the path does not depend on the frame rate.
- **Exact axis sweeps**, x then z then y:
  - every voxel layer the leading face crosses is tested over the whole cross-section, and the box stops flush;
  - nothing tunnels at any speed or substep;
  - cells are taken with an `EPS` = 1e-7 tolerance, so a box resting on a face never counts the voxel behind it (no snagging).
- **Step-up:** a grounded, horizontally blocked walker tries each whole voxel level up to `step_height`, lowest first (up, across, down). It keeps the first try that gets further.
  - The first version rose the full step at once and failed the headroom test: a low ceiling blocked it where a lower rise fits. It was fixed before any viewer run.
- **Ground snap** down to `step_height`, so walking off the curb is not a short fall.
- **Visual smoothing:** step-ups and snaps move the camera by an offset that decays (14/s), not the box.
- **Defaults, all in `walk::Params`:**

  | Parameter | Default |
  |---|---|
  | step height | 0.25 m (4 voxels; the curb is 3) |
  | walk speed | 1.4 m/s |
  | run speed (Shift) | 4 m/s |
  | jump apex | 0.45 m |
  | gravity | 9.81 m/s² |
  | max fall speed | 50 m/s |
  | longest simulated frame | 0.1 s (a longer hitch is not simulated) |

- **Viewer:**
  - Walking is the start mode, at the reference view's feet on the road.
  - F toggles fly/walk. Switching to walk lifts the walker out of any solid voxels.
  - Falling 256 voxels below the world respawns at the start.
  - `--frames N --walk` runs a fixed input script at 1/60 s of simulated time per frame.
    - **Since 2026-09-24:** a closed loop of 1308 frames (21.8 s): up the curb to the window sill, run along the shop fronts, back over the curb with jumps, run back along the road, back to the start.
    - Each loop restarts at the spawn point. Every loop ends 0.005 m from it with identical stats (1 step-up, 3 jumps).
    - The original script's last leg walked diagonally off the world, so after 16 s it fell and respawned repeatedly, mostly showing sky. §3b explains the consequence.
  - The run fails if the box ever overlaps a solid voxel, and (since 2026-09-24) if the scripted walk ever falls off the world.
    - The old script fell 53 times in every undisturbed 10,000-frame run, so this check would have caught it.

**Tests** (`walk/tests/collision.rs`, 13, release): all pass (`engine/results/test_release_m1.log`).

- **Rest:** lands flush at y = 0 and does not drift over 10 s.
- **Step-height slider**, obstacles 1–8 voxels. Each walker climbs exactly the obstacles no taller than its step height, and otherwise stops flush:

  | step | climbs obstacles of height |
  |---|---|
  | 0 | none |
  | 2 | 1–2 |
  | 3 | 1–3 |
  | 4 | 1–4 |
  | 6 | 1–6 |

- **Headroom:** a 28-voxel gap blocks the step; a 29-voxel gap passes.
- **Snaps:** 3-voxel curb: grounded every substep, 1 snap. 8-voxel drop: a fall, no snap.
- **Sliding:** along a seamed wall, the slide distance is exact (1e-6). The inside corner holds. The floor distance is exact.
- **Tunnelling:** none through a 1-voxel wall at 5000 voxels/s with 1/30 s substeps, or onto a 1-voxel slab from 20,000 voxels up.
- **Ceiling:** head flush; the free jump apex is 0.45 m.
- **Frame rate:** 30 and 144 fps paths agree within one substep.
- **Street block:** from the reference feet, up the curb (1 step) to z = 170, flush against the sill. With a 2-voxel step, the curb blocks at z = 124.
- **Random walks:** 8 seeded walkers for 20 s each through the street block, no overlap after any substep:
  - 25,705 substeps, 6 steps up, 3 snaps, 68 jumps, 6 respawns;
  - 1.8 layers and 111 voxels tested per substep;
  - 2.26 µs per substep.
- **Negative control:** the overlap check detects half a voxel into the road and a box inside the sill, and accepts flush.

## 2. Simple shading: `View::Lit` in the fullscreen view pass

- **Formula:** base colour × (sun × max(N·L, 0) + hemisphere ambient) + emissive, then × exposure and the Narkowicz ACES fit. The output is linear; the sRGB swapchain encodes it.
  - The sun direction is `scene::View::sun_dir`. The sky background is a horizon-to-zenith gradient.
  - No shadows or occlusion: sunlit-facing faces under awnings are lit.
- **Interfaces:**
  - `debug_view::View` gains `Lit`, so `View::ALL` has 8 entries.
  - New `Lighting { sun_dir, sun_intensity, ambient, exposure }`. `DebugView::record` and `params` take it.
  - The push constants grow to 128 B, the guaranteed minimum limit; a compile-time assert checks the size.
  - The material table rows are now `MaterialRow { base, emissive }` (32 B), checked by reflection. `Tables::upload` takes `&[MaterialParams]`.
  - These are public-interface changes inside `gpu`, made for this step.
- **Viewer:**
  - The lit view is the default; key 0 selects it, and keys 1–7 are the debug views as before.
  - `-` and `=` change exposure by quarter stops.
- **Defaults**, sun 3.0, ambient 0.6, exposure 0.5, were chosen by eye from the swept renders:
  - `results/lit_m1_sweep_ambient_{0.4,0.6,0.9}.bmp`: rows exposure 0.4 / 0.6 / 0.8, columns sun 2 / 3 / 4, all rendered on the CPU, which is exact to the GPU;
  - a first attempt at exposure 1.0 looked washed out;
  - ambient changes this view little, because most visible faces are sunlit.
- **Test:** the debug-view test re-derives the lit view per pixel on the CPU, including the sky gradient.
  - Result: **0 of 230,400 px differ**, within 1 LSB, like the other 7 views.
  - Planted control: a 2% brighter sun on the CPU gives 45,220 differing pixels.
  - The image is `results/lit_m1_street_view.bmp`, for inspection only.

## 3. The 60 fps gate

**Gate, declared before measuring:**
- Configuration: release build, validation off (`NE_NO_VALIDATION`), 1920×1080, lit view, `--walk` script, greedy meshes, chunk regions.
- Runs: 3 FIFO and 3 MAILBOX, interleaved, 3000 frames each.
- **Pass condition:** p99 CPU frame interval ≤ 16.67 ms in every run.
- Also reported: the maximum, and frame counts over 16.67 and 33.3 ms.
- One scripted fly run per mode is supplemental.

**Diagnostics added to the viewer:**
- The frame's CPU time is split into:
  - `begin` (waiting for the slot's previous frame);
  - `acquire` (scene submit plus `vkAcquireNextImageKHR`);
  - `present` (view submit plus `vkQueuePresentKHR`);
  - `inside` (the whole frame), with the interval minus `inside` giving time outside (the event loop).
- Every interval over 16.67 ms is listed with that breakdown.
- Also recorded: the monitor refresh rate, focus-lost and occluded events, and an optional per-frame CSV (`--trace`). The summary gains `mean`.
- The machine was on mains power; the display is 144 Hz on the Intel iGPU, with rendering on the RTX 3050 (Optimus).

**Results** (`results/viewer_m1_gate.jsonl`, `viewer_m1_gate_repeat.jsonl`, and the diagnostic series):

| Series | FIFO p99 (ms) | MAILBOX p99 (ms) | Verdict |
|---|---|---|---|
| First declared series | 14.93, **20.43**, **19.90** | 7.24, 7.00, 6.90 | **fails** (2 FIFO runs) |
| 3 FIFO back to back (with trace) | 14.83, 14.01, 12.20 | — | pass |
| MAILBOX then FIFO, 3 pairs | 15.09, 14.15, 14.77 | 7.00, 7.01, 7.14 | pass |
| Repeat of the declared series | 9.39, 15.70, 14.26 | 7.00, 6.81, 6.97 | **pass** |
| Supplemental fly, first and repeat series | 14.01, 11.95 | 6.74, 6.75 | pass |

- **Walker cost:** update p99 about 0.065 ms per frame.
- **GPU passes:** G-buffer p50 about 0.07 ms, lit view pass about 0.05 ms, at 1080p.
- **No walker overlaps** in any run.

**What the diagnostics show** (observations):
- **MAILBOX:**
  - Its tail (p99 about 7 ms) is `acquire` waiting for the display to release an image, about one 144 Hz refresh.
  - Its only interval over 16.67 ms in every run is the start-up frame (frames 17–20, 18–20 ms, in `acquire`, around the initial swapchain recreate).
  - This explains the 2C-2 "max about 20 ms in every arm": it was a start-up frame, not a steady-state tail.
- **FIFO:**
  - Paced runs have median 6.95 ms and mean 7.0 ms (144 Hz), with the wait in `begin`.
  - Intervals over 16.67 ms are frames that missed more than two refreshes.
  - The largest listed spikes (51–84 ms, one per run in 3 of 14 FIFO runs) are inside `vkQueuePresentKHR` itself, not in engine CPU work or GPU passes.
  - One failing first-series run reached 142.8 ms. The slow list keeps only the first 40 intervals, so that interval's breakdown was not captured.
- **The two failing FIFO runs** in the first series were in a different regime:
  - median 1.4–2.2 ms, meaning FIFO was not pacing, with periodic 19–20 ms waits in `begin`;
  - the view pass GPU time was 0.119 ms instead of 0.051 ms, which suggests a different GPU clock or load state at that time.
  - The regime did not recur in the 9 FIFO runs after it.
  - Focus and occlusion were not yet recorded in that series; in all later runs they were 0.
  - The cause is **not diagnosed**: external GPU load or power state is suspected, not shown.

**Verdict:**
- The gate as declared **failed once and passed on repeat**.
- MAILBOX has a wide margin.
- FIFO passes with a thin margin (worst p99 15.7 ms) and rare present-call stalls.
- Not claimed: a clean 60 fps pass under every condition. Over all 25 validation-off 1080p runs (walk and fly, both modes), p99 ≤ 16.67 ms in 23.
- **Superseded caveat (2026-09-24):** these runs used the original script, which walks off the world after 16 s of simulated time (frame 960). Most of every walk run after that point rendered sky, so §3 does not measure the street. §3b is the representative gate.

## 3b. Closing the gate (2026-09-24, user: "close the gate run the tests")

**Soak, original script** (`results/m1_soak/`; 6 FIFO and 6 MAILBOX runs interleaved, 10,000 frames each; nvidia-smi every 250 ms):
- A disturbance rule was declared after run 1, because the user was using the machine: runs with focus-lost or occluded events are excluded.
- **Undisturbed FIFO runs (3): all fail**, p99 17.7–19.9 ms, with 139–163 intervals over 16.67 ms per run, almost all waiting in `begin`.
- **Undisturbed MAILBOX runs (4): all pass**, p99 6.5–7.8 ms.
- GPU: P0 throughout, with no thermal or power throttling reason. The GPU is not the limit.

**New viewer option `--present-wait N`** (off by default):
- Before each frame, it waits until at most N presents are not yet on the display (`VK_KHR_present_id` and `VK_KHR_present_wait`).
- The extensions are enabled only when the presenting device supports them; the RTX 3050 driver does. They appear in the summary as `present_wait_supported` and `present_wait_timeouts`, and in the trace as a `pwait_ms` column.
- Purpose: to test whether FIFO's slow intervals are the engine running ahead and then blocking in bursts, or the displayed cadence itself.
- Sweep (`results/m1_present_wait/`, original script):
  - Rep 1 was undisturbed: FIFO 8.79, FIFO with `--present-wait 0` 20.59, FIFO with `--present-wait 1` 18.87, MAILBOX 6.67 ms.
  - Rep 2 was disturbed. Rep 3 was cut short when the user stopped the work.
  - Waiting on the display did not help. This suggests the displayed cadence itself sometimes skips refreshes on those views. Not diagnosed further.

**Script flaw found** (from the user's screenshot of a sky-only viewer window). The script was fixed as described in §1, then checked:
- validation on, 1349 and 2800 frames: 0 errors, 0 overlaps, 0 falls;
- 1308/2616/3924 frames end at the same point;
- screenshots at 3, 15 and 26 s show the street.

**Gate, declared before running** (`results/m1_gate_loop/`):
- Release, validation off, 1920×1080, lit view, looping walk.
- 3 arms interleaved: FIFO, MAILBOX, FIFO with `--present-wait 1`. 10,000 frames each (7 loops).
- Pass: every undisturbed run has p99 ≤ 16.67 ms, 0 overlaps and 0 falls. Runs with focus or occlusion events are excluded, and each arm needs 3 undisturbed runs.
- Rep 1 had one focus loss in two arms, which left FIFO with 2 undisturbed runs, so rep 4 was added as the rule requires.

| Arm | Undisturbed p99 (ms) | Worst max (ms) | Verdict |
|---|---|---|---|
| FIFO | 14.61, 14.51, 14.71 | 32.6 | **pass** 3 of 3 |
| MAILBOX | 7.06, 7.08, 7.20, 7.16 | 19.2 | **pass** 4 of 4 |
| FIFO, present wait 1 | 14.09, 14.22, 14.08 | 24.8 | **pass** 3 of 3 |
| Excluded (focus loss) | FIFO 14.76 (would pass); FIFO present wait 20.16 (would fail) | 54.6 | — |

- **All 12 runs:** exit 0, 0 overlaps, 0 falls, 7 loops each.
- **GPU:** P0 in every run with no throttling. G-buffer 0.26–0.71 ms and view pass 0.11–0.12 ms at p50, with the street in view.
- **FIFO timing:**
  - p50 6.9 ms, which is 144 Hz;
  - 3–5 intervals over 16.67 ms per undisturbed run, none over 33.3 ms;
  - the waits are in `begin` (or `pwait`), and no undisturbed run showed the 51–84 ms `vkQueuePresentKHR` stall.

**Observations:**
- With the street in view, the view pass takes 0.11–0.12 ms in every run, passing or failing. §3's "0.119 ms in the failing runs" clue therefore does not indicate a different GPU state.
- FIFO fails on the sky-heavy workload and passes on the street workload. The cause is **not diagnosed**.
- A person can look at the sky, so this remains a FIFO risk. MAILBOX passes in both.

**Verdict:** the M1 60 fps p99 gate at 1080p **passes** on the representative walk in all three arms. `--present-wait 1` adds nothing measurable over plain FIFO here, so it stays off. No default was changed.

## Checks run

- **Pure crates, release:** `cargo test --release -j 2 --no-fail-fast` exits 0 with 114 tests (101 + 13 `walk`). Log: `results/test_release_m1.log`.
- **Clippy:** `cargo clippy --release -j 2 --workspace --all-targets` exits 0, with the same 6 pre-existing `world` lints and none in `walk`, `gpu` or `viewer` (`results/clippy_m1.log`).
- **GPU, release:** `cargo test --release -j 2 -p gpu -- --test-threads=1 --nocapture` exits 0 with 12 + 9 + 5 tests.
  - The 32 equivalence lines are identical to 2C-2.
  - 8 views pixel-exact; lit control detected.
  - The only validation error is the planted early-free control.
  - Log: `results/test_gpu_m1_rtx.log`.
- **Viewer, validation on** (`results/viewer_m1_validation.log`, summaries in `results/viewer_m1.jsonl`), all exit 0 with 0 validation errors, 0 warnings and 0 overlaps:
  - walk script 1080p MAILBOX, greedy and unmerged;
  - fly with all 8 views and a resize to 1600×900 (2 recreates);
  - also the first walk run, before shading (`viewer_m1_walk_script.log`), and a fly/resize regression run (`viewer_m1_fly_resize.log`).
- **Debug suites:** see Closeout.
- **After the present-wait and loop changes (2026-09-24)**, all exit 0:
  - **Pure crates:** release and debug, 114 tests each (`results/test_release_m1_close.log`, `test_debug_m1_close.log`).
  - **Clippy:** the same 6 pre-existing `world` lints only (`results/clippy_m1_close.log`).
  - **GPU, release and debug:** 12 + 9 + 5 tests each (`results/test_gpu_m1_close_rtx.log`, `test_gpu_m1_close_rtx_debug.log`).
    - The 34 equivalence result lines are identical to the earlier M1 run, and debug matches release.
    - 8 views are pixel-exact; the lit control gives 45,220 px.
    - The only validation error is the planted early-free control.
  - **Viewer, validation on** (`results/viewer_m1_close_validation.jsonl`), all with 0 errors and 0 warnings:
    - walk FIFO with `--present-wait 1` at 1080p;
    - walk MAILBOX unmerged at 1080p;
    - fly with all views and a resize (2 recreates).
    - 0 overlaps and 0 falls in all of them.

## Risks

- **FIFO on this laptop:**
  - On the street walk it passes: p99 14.5–14.7 ms, 2 ms of margin.
  - On sky-heavy views it failed every undisturbed soak run (p99 17.7–19.9 ms), cause undiagnosed.
  - Earlier 51–84 ms stalls inside `vkQueuePresentKHR` (3 of 14 original FIFO runs; Optimus composed presentation suspected) did not appear in the undisturbed loop runs.
  - Heavier shading (shadows, 2D) will eat into that margin through the CPU wait, not only the GPU.
- **Use of the machine during a run breaks the measurement** (focus loss): the gate needs an idle machine.
- The walking script covers the street block's curb, sill, shop fronts and edges, not every obstacle. Random walks cover more but found only 6 step-ups in 20 s × 8.
- Collision reads the authoritative world directly. When edits arrive (2E), the walker sees them immediately, possibly before the GPU meshes do; this is acceptable for now, but a declared choice.
- There are no shadows, so the lit view has no occlusion cue. This is by design for M1; it is the next visible gap.
- Public-interface changes in `gpu`: `View::Lit`, `Lighting`, `MaterialRow`, and the `Tables::upload`, `record` and `params` signatures. `walk` is a new public crate API.
- Added 2026-09-24: `Gpu::present_wait()` and `Swapchain::wait_presented()`. The device now enables `VK_KHR_present_id` and `VK_KHR_present_wait` when they are supported. These are standard Vulkan extensions through the existing `ash` dependency, with no new crate.
- The summary JSON gains `present_wait*`, `start_unix_ms` and `walk.script_loops`. The trace CSV gains `at_ms`, `pwait_ms` and GPU columns. These are diagnostic outputs and are not versioned.

## NOT RUN

- Interactive input (keys, mouse, F toggle) by a person. The scripted runs cover the same code paths except input handling.
- Minimize and restore.
- Exclusive fullscreen, and presentation through the dGPU directly (MUX or dGPU-only mode).
- A cold-process timing series.
- Battery power.

## Closeout

- **Docs:** this record; [DECISIONS](../DECISIONS.md) S-012; `engine/README.md`; `docs/NOW.md`.
- **Debug suites** (run 2026-09-23/24):
  - `cargo test -j 2 --no-fail-fast` exits 0 with 114 tests (`results/test_debug_m1.log`).
  - `cargo test -j 2 -p gpu -- --test-threads=1 --nocapture` exits 0 with 12 + 9 + 5 tests and the doc-test step. The lit view is exact and its control is detected. The 32 equivalence lines are identical to the release run (`results/test_gpu_m1_rtx_debug.log`).
- **2026-09-24 docs:** this record (§1 script, §3b, checks, risks), `engine/README.md`, DECISIONS S-012, `docs/NOW.md`.
- **Revert:**
  - present wait: remove the `present_wait` detection and enablement in `gpu/src/context.rs`, and `waiter`, `last_present`, `wait_presented` and the `PresentIdKHR` chain in `gpu/src/present.rs`;
  - delete `engine/walk/` and remove it from the workspace members and `viewer/Cargo.toml`;
  - revert the viewer's walk, lit, phase and trace code;
  - remove `View::Lit`, `Lighting`, `MaterialRow` and the 128 B push constants in `gpu/src/debug_view.rs` and `shaders/debug_view.slang`;
  - restore `Tables::upload` to base colours;
  - remove the lit branch and control from `gpu/tests/raster.rs`.
