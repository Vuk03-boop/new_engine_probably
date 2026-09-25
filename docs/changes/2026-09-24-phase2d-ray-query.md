# Change: Phase 2D — ray-query representation and raster-versus-ray equivalence

Status: **passed on the RTX 3050 (2026-09-24).** Every declared criterion is met. The criteria were declared before the first GPU run; two bugs found by the first focused runs were fixed before the recorded suite ran (see Iterations).
Date and baseline: 2026-09-24, after M1 ([M1 steps](2026-09-23-m1-walk-shade-gate.md)). No Git repository.
Authorization (S-013), user, 2026-09-24: "I accept M1 as met so continue". Read as: continue with the next slice of the approved Phase 2 plan ([Phase 2 proposal](2026-09-23-phase2-proposal.md) §4, 2D). 2E is not started by this record.

## Objective

Build the ray-query representation of the committed LoD 0 meshes, and show per pixel that ray query sees exactly what raster sees (ADR-0003 Amendment 1), with its memory in the ledger.

## Scope and contracts

- **Paths:**
  - `engine/gpu`: new `src/accel.rs`, `src/ray.rs`, `shaders/ray_primary.slang`, `tests/ray.rs`;
  - `equivalence::agreement` (a direct frame-against-frame check);
  - `build.rs`: the new shader, and two allowed SPIR-V capabilities.
- **No new dependency, device feature, format or default.**
  - The shader needs `RayQueryKHR`, enabled by the required `rayQuery` feature, and `PhysicalStorageBufferAddresses`, enabled by `bufferDeviceAddress`. Both were already on.
  - It needs no `Int64`: `slangc` emits pointer loads without it (checked in the SPIR-V disassembly).
- **Representation:**
  - One BLAS per region, built from the region's own device buffer. These are the same vertices (`R16G16B16A16_SFLOAT`, stride 8) and indices raster draws. Geometry is opaque, and builds prefer fast trace.
  - One TLAS, with one instance per region in key order:
    - the transform translates by the region origin;
    - the instance custom index is the region index;
    - instances carry no facing flag: Vulkan's default facing matches ADR-0004's winding (corrected after iteration 1; the plan said front-counter-clockwise).
- **Surface id:** (InstanceCustomIndex, PrimitiveIndex), the ADR-0003 pair.
  - Material and face come from `quads[tri_quad[primitive]]`, read through a per-region table of device addresses and table lengths (24 B per region; the plan said 16 B, see iteration 2).
  - This design keeps per-region buffers addressable for 2E. It avoids a concatenated copy of the tables.
- **Ray:**
  - one ray per pixel centre, with the `raster::Camera` basis, so the forward component is 1 and t is the view distance;
  - TMin = near, since raster clips there too;
  - back faces culled, as raster culls them;
  - depth written as near / t.
- **Memory:**
  - BLAS, TLAS, instances and table go under `GpuAccel`; build scratch goes under `GpuAccelScratch` and is freed when the build completes;
  - trace outputs (24 B/px) go under `GpuTemporal`;
  - building is all-or-nothing on a refused grant;
  - BLAS builds are batched with at most 32 MiB of scratch per batch.
- **Excluded:** edits and rebuilds (2E), compaction, the granularity decision (the 2E sweep), viewer integration, shadows and lighting.

## Declared pass criteria (before any GPU run)

1. **Ray against the CPU reference:** for all 8 settings (merge none/greedy × region brick/2³ bricks/chunk/2³ chunks) × the 4 2C-1 cameras at 640×360, `equivalence::compare` passes. That means 0 crack, material, normal, face, depth, extra and bad-id pixels, and exact checks cover at least a quarter of the reference hits. These are the same criteria and tolerances as raster in 2C-1.
2. **Ray against raster, directly:** `equivalence::agreement` over the same 32 cases has 0 non-edge disagreements. That covers hit/miss, material, normal bits, resolved (voxel, face), and depth within twice the declared tolerance, since each path is allowed that tolerance from the reference. Near-edge disagreements are reported, not failed.
3. **Planted faults are caught** (greedy, chunk, street view):
   - instance translation off by one voxel: face + depth > 0;
   - the busiest region left out of the TLAS: crack > 0;
   - instances without the front-counter-clockwise flag: failures > half the reference hits;
     - *Correction after iteration 1:* the unflagged build is the correct one, so the planted inverted facing is the flagged build. The criterion itself (inverted facing: failures > half the hits) is unchanged.
   - instance custom index shifted by one: material + face + bad_id > 0;
   - the raster reflection passed to the trace pipeline: refused as a layout error before any Vulkan object is made.
4. **Ledger:**
   - after a build, `GpuAccel` live bytes equal the sum of the build's buffer ranges, and `GpuAccelScratch` live is 0;
   - with a budget too small for the BLAS, the build is refused, nothing stays allocated, and the refusal is counted.
5. **Validation:** 0 errors in every test. Warnings are reported.
6. **Cost, data only (no gate):**
   - At 1920×1080, street view camera: trace-pass and raster-G-buffer GPU times, interleaved, 30 repetitions, median and p90.
   - Per setting: BLAS and TLAS build GPU ms, host ms, and BLAS/TLAS/instance/table/scratch bytes.
   - These are inputs to the 2E sweep, which chooses the default.

## Iterations before the recorded run (focused test: `--test ray planted`)

1. **All hits were back faces.**
   - The first build set `TRIANGLE_FRONT_COUNTERCLOCKWISE` on every instance.
   - The hit count was right (173,137, the same as the reference and raster). But every checked pixel had the wrong face, normal and depth, and 172,048 were cracks (a farther surface drawn).
   - The cause: Vulkan's default ray-triangle facing already treats ADR-0004's winding (counter-clockwise from outside, right-handed, Y up) as front-facing. The flag inverted it.
   - Fix: no facing flag. The flagged build is now the planted `flipped_facing` control, and it reproduces the first run's counts exactly.
2. **A wrong id lost the device.**
   - With the facing fixed, the baseline passed and three controls were caught. `custom_index_shifted` ended in `ERROR_DEVICE_LOST`: a shifted region index made the shader read past the end of another region's `tri_quad` table through a raw device address.
   - Fix: `RegionRef` gained the quad and triangle counts (16 → 24 B), and the shader bounds-checks the region, primitive and quad.
     - An out-of-range id writes `BAD_MATERIAL` (0xFFFE) and keeps the surface id, so the host reports it as `bad_id`.
     - The reflection check covers the new fields.
3. The third focused run passed. The recorded suite below ran after it, with no further change.

The focused logs are scratch (`ray_planted_iter1..3.log` in the session scratchpad), not preserved evidence; the counts above are copied from them.

## Commands and evidence actually produced

All runs: RTX 3050 Laptop GPU, validation on, release unless stated.

| Command/config or check | Exit/result | Evidence path | Scope/limitation |
|---|---|---|---|
| `cargo test --release -j 2 -p gpu --test ray -- --test-threads=1 --nocapture` | exit 0; 4 of 4; 0 validation errors, 0 warnings | `engine/results/test_gpu_2d_ray_rtx.log` | The declared run: criteria 1–6 |
| `slangc` + `spirv-val` on `ray_primary.slang`; capabilities from the disassembly | pass; `RayQueryKHR`, `PhysicalStorageBufferAddresses`, `Shader`; no `Int64` | `build.rs` checks it on every build | |
| `cargo test --release -j 2 -p gpu -- --test-threads=1 --nocapture` | exit 0; 12 unit + 9 device + 5 raster + 4 ray; 0 validation errors | `engine/results/test_gpu_2d_rtx.log` | Regression check after the shared-code changes. The 39 raster equivalence lines are identical to the M1 close run; 8 views pixel-exact; lit control 45,220 px, as before |
| `cargo test -j 2 -p gpu -- --test-threads=1 --nocapture` (debug) | exit 0; same 30 tests; 0 validation errors | `engine/results/test_gpu_2d_rtx_debug.log` | Debug build of the same suite |
| `cargo clippy --release -j 2 --workspace --all-targets` | exit 0; only the 6 pre-existing `world` lints | `engine/results/clippy_2d.log` | Also compiles `viewer` against the changed `gpu` |
| Pure suite (`cargo test --release -j 2`) | NOT RUN | | No pure crate changed since the M1 close run (`results/test_release_m1_close.log`) |
| Viewer runs | NOT RUN | | The viewer does not call 2D code; it only recompiled |

## Result and full cost

- **Criterion 1 (ray against the reference): met in all 32 cases.**
  - 0 crack, material, normal, face, depth, extra and bad-id pixels.
  - The ray hit count equals the reference's in every case (street view: 173,137).
  - Checked pixels are 73–87% of reference hits; the declared floor is 25%.
  - The largest depth error is 0.0063 of the tolerance.
- **Criterion 2 (ray against raster): met in all 32 cases.**
  - 0 non-edge hit/miss, differ and depth disagreements.
  - Near-edge disagreements are reported, not failed. They are at most 205 of 230,400 px, plus 3 near-edge hit/miss pixels on the overhead camera.
  - Raw surface ids are identical on 99.6–99.9% of the pixels both paths hit. Where the ids differ, the two paths picked different triangles; away from edges those still resolve to the same voxel face.
  - The largest depth difference is 1.63 × the doubled tolerance, on a near-edge pixel of the grazing camera. Non-edge pixels are within it.
- **Criterion 3 (planted faults): all caught.**

  | Fault | Caught by |
  |---|---|
  | instance translation off by one voxel | 4,612 face and 4,896 depth pixels |
  | the busiest instance dropped (8,866 px) | 8,866 cracks |
  | flipped facing | 172,048 cracks; 142,318 face |
  | custom index shifted by one | 50,911 bad id; 33,163 material; 78,357 face |
  | raster reflection given to the trace pipeline | refused with 7 layout errors, before any Vulkan object |

- **Criterion 4 (ledger): met.**
  - After every build, `GpuAccel` live equals the build's buffer ranges exactly, and `GpuAccelScratch` live is 0. `GpuAccel` returns to 0 after the free.
  - With a budget of meshes + ring + 4 MiB, the build was refused on a `GpuAccel` grant. Nothing stayed reserved, the live-buffer count did not change, and the refusal was counted.
- **Criterion 5:** 0 validation errors and 0 warnings in all four tests.
- **Criterion 6 (cost, data only; the 2E sweep chooses).** Street block: 1,041,080 unmerged or 275,890 greedy triangles.

  | Merge | Region | BLAS | BLAS MiB | TLAS MiB | Scratch MiB | GPU BLAS ms | GPU TLAS ms | Host ms |
  |---|---|---|---|---|---|---|---|---|
  | none | brick | 5,103 | 69.7 | 1.47 | 32.0 | 26.2 | 0.20 | 166 |
  | none | 2³ bricks | 1,176 | 71.4 | 0.34 | 32.0 | 10.4 | 0.14 | 34 |
  | none | chunk | 291 | 70.7 | 0.09 | 31.8 | 8.9 | 0.10 | 7 |
  | none | 2³ chunks | 67 | 69.7 | 0.02 | 31.8 | 8.5 | 0.07 | 4 |
  | greedy | brick | 5,103 | 26.1 | 1.47 | 23.0 | 25.5 | 0.19 | 189 |
  | greedy | 2³ bricks | 1,176 | 18.5 | 0.34 | 14.3 | 6.1 | 0.14 | 17 |
  | greedy | chunk | 291 | 19.0 | 0.09 | 15.2 | 3.0 | 0.10 | 6 |
  | greedy | 2³ chunks | 67 | 18.7 | 0.02 | 15.3 | 2.8 | 0.07 | 3 |

  - These are full-scene builds; 2E measures per-edit rebuilds.
  - Host ms covers size queries, allocation, uploads and recording. Scratch is freed after each build.
  - Mesh bytes, for comparison: 35.7 MiB unmerged, 6.5 MiB greedy.

  GPU pass times at 1920×1080, 30 interleaved repetitions (median; p90 within 0.01 ms). Cameras are listed as street view / chunk corner / overhead / grazing.

  | Setting | Raster G-buffer ms | Ray trace ms |
  |---|---|---|
  | greedy, chunk | 0.330 / 0.321 / 0.295 / 0.255 | 1.091 / 1.063 / 0.797 / 0.974 |
  | greedy, brick | 0.346 / 0.358 / 0.295 / 0.271 | 1.003 / 1.014 / 0.788 / 0.939 |
  | none, chunk | 0.546 / 0.526 / 0.517 / 0.504 | 1.027 / 0.997 / 0.794 / 0.912 |

- **Observations** (not decisions):
  - Primary rays cost about 0.8–1.1 ms at 1080p, about 3× the greedy raster G-buffer. That fits A-001: raster for primary visibility, ray query for secondary rays such as shadows and lighting.
  - Trace time barely depends on region size. BLAS build time and memory do: greedy chunk needs 3.0 ms and 19 MiB; greedy brick needs 25.5 ms and 26 MiB.
  - Greedy merging cuts BLAS memory to about 27% of unmerged.

## Closeout

- **Changed:**
  - new: `engine/gpu/src/accel.rs`, `engine/gpu/src/ray.rs`, `engine/gpu/shaders/ray_primary.slang`, `engine/gpu/tests/ray.rs`;
  - edited: `engine/gpu/src/equivalence.rs` (`agreement`), `engine/gpu/src/lib.rs`, `engine/gpu/build.rs` (shader entry; capabilities 4472 and 5347), `engine/README.md`.
- **Interfaces:**
  - `accel::RegionRef` (24 B: two addresses and two counts) and `ray::{Hit, Params}` are host/shader layouts. They are checked against `slangc` reflection before the pipeline is made.
  - The ray outputs are test and diagnostic buffers, not a new G-buffer format.
  - No ADR: ADR-0003 already fixes the shared id pair, and this record documents the table.
- **Fallback and revert:** only the new tests call 2D. Removing the four new files, the `lib.rs` lines, the `build.rs` entry and capabilities, and `equivalence::agreement` restores the M1 state.
- **Not run:** ray query in the viewer; builds after edits (2E); BLAS compaction and `ALLOW_UPDATE` refits; non-street scenes; another GPU or driver.
- **Next:** 2E, edits end to end. That covers the BLAS rebuild of touched regions, the snapshot swap after the old frames retire, edit-to-visible latency, publication sharing, and the granularity sweep that picks the default. It needs authorization.
