# Change: Phase 2A — surface extraction (CPU)

Status: passed (within the limits below)
Date and baseline: 2026-09-23, after the Phase 1 perf/prep record. No Git repository.
Authorization (S-006), user: "You may add them and for 2A sure".
- "them" is the engine dependencies `ash =0.38.0`, `winit =0.30.13`, `ash-window =0.13.0`. They are approved (A-006) and get added with the first crate that uses them, the 2B `gpu` crate. 2A needs no dependency, so none was added.
- The user's question "LoD 0 wont doing it bit exact prevent improvements?" led to a correction of ADR-0003 (Amendment 1).

## What was done

1. **`derived/src/mesh.rs`:**
   - `Quad`: 8 bytes, brick-local and half-open, with an exact `MaterialId`.
   - `BrickMesh`.
   - `Merge::{None, Greedy}`: the mesh-merge "slider" from ADR-0003.
   - `extract` / `extract_world`.
   - `exposed_faces`: the coverage oracle, a direct voxel scan.
   - `trace_mesh`: the CPU ray–quad reference for the later ray-query path.
   - Greedy merging runs per face direction and plane over an 8×8 mask, row by row. It never crosses a material, a plane or a brick.
2. **The pipeline now carries the real product.**
   - `Job::run` extracts a `BrickMesh` from the job's input copies, instead of the 1C face-count stand-in.
   - `Config::merge` selects the mode.
   - The dependency record is unchanged: the brick plus its six face layers is exactly what a mesh reads.
   - `SurfaceSummary` stays as a cross-check: a mesh's area per material must equal the direct face count.
3. **[ADR-0004](../adr/ADR-0004-surface-mesh-layout.md):**
   - the CPU quad record, the winding and triangulation, and the order
   - the GPU layout for 2B: half-float region-local vertices, u32 indices, an 8-byte quad table, and one instance per region
   - the (instance, primitive) → quad mapping
   - the T-junction risk, and how it will be measured
4. **ADR-0003 Amendment 1:** "bit-identical image across settings" is replaced by a per-pixel contract. Resolved voxel face, material and normal must be exact, depth within tolerance, and crack pixels zero.
5. **Tests and controls:**
   - `mesh` unit tests: winding, merging, neighbour-hidden faces, and the documented diagonal-edge difference.
   - `derived/tests/mesh.rs`: coverage, ray agreement, negative controls, and pipeline equality.
   - Existing 1C tests now run on meshes.
6. **Harness:** `perf_phase1` gained a `mesh` section (per merge mode: size, projected GPU bytes, full build), and its verification compares meshes.

## Results

**Correctness** (`derived/tests/mesh.rs`, release; debug below):
- **Coverage is exact in both modes:**
  - on the street block and 4 random worlds spanning chunk boundaries
  - every exposed unit face is covered once, with its voxel's material
  - nothing else is covered
  - area per material equals the direct count for every brick
- **Ray agreement:** 0 disagreements with the voxel DDA (bit-equal t, same voxel, material and face):
  - street: 3,000 rays (1,311 hits) per mode
  - random worlds: 3 × 2,000 rays (474–634 hits) per mode
  - 200 rays lying in the chunk plane x = 0 (62 hits)
- **Negative controls, all reported:**
  - coverage checker: a dropped quad, a duplicated quad, a shifted plane, and a cross-material merge
  - ray checker: shifted faces
  - The existing 1C planted faults, including a missed neighbour-dirty, are still caught with meshes as the product.
- **Documented difference:** a ray through the exact shared edge of two diagonal voxels. The DDA hits; no quad is crossed. It is measure-zero, and pinned by a unit test.

**Size and cost** (`engine/results/perf_phase2a.jsonl`, 5 rounds, AC power, High performance):

| Street block (5,103 bricks) | `none` | `greedy` |
|---|---|---|
| Quads (triangles) | 520,540 (1,041,080) | 10,503 (21,006) |
| Most quads in one brick | 240 | 11 |
| CPU quad bytes | 4.16 MB | 84 KB |
| Projected GPU geometry (ADR-0004, 64 B/quad, BLAS not included) | 33.3 MB | 0.67 MB |
| Full inline build, same run, interleaved (median) | 100.6 ms | 92.8 ms |

- **Same run, default greedy:**
  - full build 101 ms inline, 63 ms with 4 workers
  - edit-to-publish on the CPU: 0.18 ms (1 voxel), 0.32 ms (8³), 5.7 ms (32³)
- **Not comparable with the Phase 1 baseline.** The unchanged reference DDA ran at 0.289 M rays/s in this run, against 0.174 in the Phase 1 run. The machine's speed differed between the runs by more than any code change could explain, so only arms within one run are compared. No claim is made that 2A is faster or slower than 1C.
- Both merge modes are far below the 3.15 × 10⁹ B device budget. The granularity choice will be decided by BLAS size, trace time and edit latency in 2E, not by geometry bytes.

## Commands and evidence

| Check | Result | Evidence |
|---|---|---|
| `cargo test --release -j 2 --no-fail-fast` | exit 0, 99 tests (89 before, plus 4 `mesh` unit and 6 `mesh` integration tests) | `engine/results/test_release.log` (overwrites the Phase 1 log from the same command) |
| `cargo test -j 2 --no-fail-fast` (debug) | exit 0, 99 tests pass | `engine/results/test_debug.log` |
| `cargo clippy --release -j 2 --workspace --all-targets` | exit 0; the same 6 pre-existing style lints, 0 new (one lint in `mesh.rs` fixed) | `engine/results/clippy.log` |
| `perf_phase1 --rounds 5` | exit 0; every snapshot verified against direct extraction | `engine/results/perf_phase2a.jsonl` |

**Debug run:** `cargo test -j 2 --no-fail-fast` exits 0; 99 of 99 tests pass (the debug ray tests use fewer rays: 300 per set).

## Limits and risks

- **T-junctions:** greedy quads have them, and they can crack in raster and ray tests. This is unmeasured until 2C/2D; it is the main open risk of the greedy default.
- **Scope of the ray agreement:** it is checked on the CPU, with f64, against quads. The GPU (float, hardware triangles) is 2D's job.
- `notify_edits` still does per-voxel dirty work: 3.3 ms of the 5.7 ms for a 32³ edit.
- Publication still clones the snapshot table (2E).
- The Phase 1 test logs were overwritten by the reruns of the same commands. The Phase 1 counts stay in the Phase 1 records.

## Closeout

- **Docs:**
  - ADR-0004 (new)
  - ADR-0003 (Amendment 1)
  - Phase 2 proposal (2A done, decision 1 accepted)
  - `docs/DECISIONS.md` (S-006, A-006, A-007)
  - `engine/README.md`
  - `docs/NOW.md`
- **Revert:**
  - delete `derived/src/mesh.rs` and `derived/tests/mesh.rs`
  - restore `Job::run` → `summarize` and `SlotPool<SurfaceSummary>`
  - remove `Config::merge` and `Job::merge`
  - drop the `mesh` perf section
