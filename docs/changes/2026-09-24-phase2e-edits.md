# Change: Phase 2E — edits end to end, granularity sweep, and the M2 viewer edit step

Status: **done on the RTX 3050 (2026-09-24).** Decided afterwards (S-015): the user accepted greedy / chunk as the default, kept 2³ chunks as the alternative for larger, less dense scenes ([ADR-0003 Amendment 2](../adr/ADR-0003-device-budget-and-visibility-buffers.md)), and accepted M2 as met. The edit budget stays provisional.
- Criteria 1–6, 8 and 9 are met.
- Criterion 7, the default rule, did not give a stable answer. The granularity default is unchanged (greedy / chunk) and waits for the user, together with the provisional edit budget.
- MAILBOX is the viewer's default.
Date and baseline: 2026-09-24, after 2D ([record](2026-09-24-phase2d-ray-query.md)). No Git repository.
Authorization (S-014), user, 2026-09-24: "Greenlight for 2E and M2 default display mode MAILBOX is fine". Read as:
- 2E as planned in the [Phase 2 proposal](2026-09-23-phase2-proposal.md) §5;
- M2 confirmed as proposed in NOW.md, including the viewer edit step (place and remove voxels, see the edit in raster and in a ray-query view);
- the viewer's default present mode changes from FIFO to MAILBOX.

## Objective

An edit committed to the world reaches both GPU representations as one coherent snapshot, with measured per-stage cost and edit-to-visible latency. The granularity sweep then picks the default mesh merge and acceleration region by the rule declared below. In the viewer, you can place and remove voxels and see the edit in raster and in ray query.

## Scope and contracts

- **Paths:**
  - `engine/derived`: `pipeline.rs` (sharing), `tests/scenarios.rs`, `tests/memory_account.rs`;
  - `engine/gpu`: `accel.rs` (incremental update), new `scene.rs`, `mesh.rs`, `debug_view.rs` and `shaders/debug_view.slang` (the ray-query source), new `tests/edit.rs`;
  - `engine/viewer`: edits, the ray-query view, the MAILBOX default;
  - docs.
- **No new dependency, device feature or format.** Defaults changed: the present mode (authorized), and the granularity if the sweep rule picks a different one.
- **Edit path:** world transaction (1B) → `notify_edits` → jobs → `try_publish` (1C) → the published brick keys → the acceleration regions containing them → those regions' meshes rebuilt from the new snapshot → upload → BLAS of those regions rebuilt, TLAS and region table rebuilt → swap.
  - **Coherence:** the device shows one published snapshot at a time. The swap happens only after every new resource was granted and built. A refused grant leaves the previous snapshot visible and correct, counts a deferral, and keeps the regions dirty for a retry.
  - **Retirement:** replaced mesh buffers, BLAS, the old TLAS/instances/table and old descriptor pools are retired at the timeline value of the last submission that may read them, and freed only after it completes.
  - The acceleration update waits on the host for its build, as 2D's build does. So in this slice the swap also waits for earlier frames, and the edit frame can hitch; the hitch is measured. Without that wait, in-place descriptor rebinding would need per-version descriptor sets (recorded, not built).
- **Publication sharing:** snapshot entries are sharded by chunk and shared between snapshots (`Arc`); a publication copies the chunk index and only the shards it changes.
- **Excluded:** asynchronous (off-thread) jobs, BLAS refit or compaction, shadows and lighting, LoD, streaming.

## Declared pass criteria

Criterion 1 was written after the first sharing run (its A/B numbers are below, with the before-run log kept). Criteria 2–9 were declared before any 2E GPU run.

1. **Sharing:**
   - a 1-voxel publication on the street copies only the shards of the chunks it touches, and a reader of the old snapshot keeps the old entry;
   - the 1D memory test's bracket still holds, with shared shards counted once;
   - the 1D edit's publish transient drops by at least the size of one entry table (1C: 4,762,632 B).
2. **Edits reach both paths, every setting:** for all 8 settings (merge none/greedy × region brick/2³ bricks/chunk/2³ chunks), after the 1-voxel, 8³ and 32³ edits of `perf_phase1`, at an edit close-up camera and the street view (640×360):
   - raster and ray each pass `equivalence::compare` against the CPU reference of the edited world (0 failures);
   - raster against ray: 0 non-edge disagreements;
   - the incrementally updated device meshes equal a from-scratch build of the same snapshot, byte for byte;
   - after collection, `GpuMesh` and `GpuAccel` live bytes equal the scene's buffers, and `GpuAccelScratch` is 0.
3. **Refused grant:** with the budget too small for an update, the update is refused, the scene still shows the previous snapshot and passes the reference check of the previous world, nothing new stays allocated, and the deferral is counted. After space is freed, the retry shows the new snapshot and passes.
4. **Planted faults are caught:**
   - one changed region left out of the update: the reference check fails;
   - replaced buffers freed at the swap instead of after retirement, with a frame in flight (mesh-only path, which does not wait): validation reports an error.
5. **Validation:** 0 errors and 0 warnings in every correct-path test.
6. **Measurements (per setting):** device memory by category; BLAS and TLAS build GPU ms; 1080p trace and raster G-buffer GPU ms (4 cameras, 30 interleaved repetitions, median); per edit kind: stage times and edit-to-visible latency (commit start to the completion of the first frame rendered with the new snapshot, raster and ray), over 10 repetitions, median and p95.
7. **Default rule (from the proposal, made exact here):**
   - eligible: settings whose p95 edit-to-visible latency fits the edit budget for all three edits;
   - among them, those whose trace time (mean over the 4 cameras of the median) is within 5% of the best eligible trace time;
   - the default is the smallest region among those; between merge modes, the one with less `GpuMesh` + `GpuAccel` memory.
   - **Edit budget (provisional, needs the user's yes):** p95 ≤ 50 ms for the 1-voxel and 8³ edits (3 frames at 60 Hz), ≤ 100 ms for the 32³ edit. No budget was set before; P-001 says "mostly static". The record also shows which default other budgets would pick.
8. **Direct-voxel fork check:** triggered if the chosen default misses the budget and extraction plus BLAS/TLAS rebuild are more than half of its latency. Otherwise it is recorded as not triggered, with the measured shares.
9. **Viewer (M2):**
   - left click removes the voxel under the cursor, E (or middle click) places one against the face under the cursor; R switches the views between the raster G-buffer and the ray-query result;
   - a scripted run (`--frames N --edit-script`) places and removes voxels in view, with 0 validation errors and warnings, and logs every edit's latency;
   - the default present mode is MAILBOX.

## Iterations

1. **Refusal test harness.** The first run of `refused_update_keeps_the_previous_snapshot` panicked: the exact test budget also refused the test's own readback buffer (`Staging`, 4 MiB) when checking the image. This was a test design error, not an engine error. The checks now read back through a separate, unlimited allocator, so the test budget governs only the scene (`results/edit_focused_iter1.log`, then `_iter2.log`).
2. **Debug-view comparison coverage.** The first version compared the two sources only on pixels whose depth bits were identical: 20,161 and 58,878 pixels. Raster and ray depth differ in the last bits almost everywhere. Views that do not read depth are now compared on every pixel both paths hit with identical normal, material and surface id (211,795 and 172,843 pixels). The views that read depth (brick hash, linear depth) keep the exact-depth rule.
3. **Timing with validation off.** In the first sweep (validation on, `results/edit_sweep_iter1.log`), the validation layer dominated the host stages. For example, the brick settings spent about 30 ms rebuilding 5,103 descriptor sets and about 80 ms recording and waiting for one frame, so its latencies were 100–150 ms. That run also measured greedy trace at 1.74–1.88 ms against 0.92–0.98 ms with validation off. The cause is not diagnosed: 2D's validation-on run measured 0.8–1.06 ms for greedy / chunk.
   - Latency and trace time come from two validation-off runs of the same test (`NE_NO_VALIDATION=1`; the test says "TIMING RUN" and reports its validation checks as NOT RUN).
   - Correctness and validation come from the validation-on runs. Every correctness check also runs, and passed, in the timing runs.
4. **Viewer review fix (after the recorded viewer runs).** Edits were attached to the next submitted frame even when the device update had been refused, so a deferred edit would have been reported as visible early. They now count as shown only once no published keys are waiting for the device. The recorded runs had 0 deferrals, so their numbers are unaffected; a re-check run with the fixed binary passed (`results/viewer_2e_fixcheck.log`: 25 of 25 edits shown, validation clean).

## Evidence

| Check | Result | Evidence | Scope |
|---|---|---|---|
| 1D memory test before sharing (1C table clone) | exit 0; publish transient 4,762,632 B | `results/memory_account_2e_before.log` | release, CPU |
| memory + scenario tests after sharing | exit 0; transient 2,481,896 B; 1-voxel edit copies 1 of 292 shards | `results/derived_2e_sharing.log` | release, CPU |
| focused 2E tests, iteration 1 | exit 101: refusal harness error (iteration 1); debug views and planted faults passed | `results/edit_focused_iter1.log` | RTX 3050, validation on |
| focused 2E tests, iteration 2 | exit 0: debug views, refusal (both scenarios) | `results/edit_focused_iter2.log` | RTX 3050, validation on |
| sweep, validation on | exit 0; 288 check lines, 0 failures; validation 0 errors, 0 warnings | `results/edit_sweep_iter1.log` | correctness record; its timings are not used |
| sweep timing run 1 and repeat | exit 0 both; all correctness checks pass; validation NOT RUN | `results/edit_sweep_timing_iter1.log`, `results/edit_sweep_timing_repeat.log`; `results/perf_2e_sweep.jsonl` (the `SWEEP` lines of all three runs, tagged by run) | timings |
| pure suite, release | exit 0, 115 tests (114 + the sharing scenario) | `results/test_release_2e.log` | CPU |
| `-p gpu`, release | exit 0: 12 + 9 + 4 (edit) + 5 (raster) + 4 (ray); validation 0 errors, 0 warnings in every test that checks it | `results/test_gpu_2e_rtx.log` | RTX 3050 |
| raster and ray test output against 2D's run | identical: 54 raster and 74 ray equivalence lines | `results/test_gpu_2d_rtx.log` vs `results/test_gpu_2e_rtx.log` | regression |
| `-p gpu`, debug | exit 0: 12 + 9 + 4 + 5 + 4; validation 0 errors, 0 warnings; the sweep's 288 checks 0 failures (the edit tests take 25 min in debug) | `results/test_gpu_2e_rtx_debug.log` | RTX 3050 |
| clippy, workspace | exit 0; only the 6 pre-existing `world` lints | `results/clippy_2e.log` | |
| viewer, scripted edits (raster; ray; ray without validation; walk; ray with resize and view cycling) | exit 0 each; every edit shown; validation 0 errors, 0 warnings (where on); 0 walker overlaps, 0 respawns | `results/viewer_2e.jsonl`, `results/viewer_2e_*.log`, `results/viewer_2e_ray_trace.csv` | RTX 3050, MAILBOX |
| NOT RUN | edits on another GPU or a non-RT device (the mesh-only path runs only in the planted-fault test); scenes other than the street; interactive editing by a person; minimize/restore; the M1 60 fps gate series with the new viewer (the scripted runs below are not that gate) | | |

## Results

### 1. Sharing — met

- 1-voxel edit: the new snapshot has 292 shards, 291 of them shared with the previous one; a reader of the old snapshot keeps the old entry.
- The 1D edit's publish transient: 4,762,632 B → 2,481,896 B (−2.28 MB). One snapshot table is 1.28 MB live (4.6 MB reserved bound), so the drop is larger than one table.
- Cost: one snapshot now measures 3.03 MB of heap instead of 2.81 MB (292 small shard maps), and the reported reserved bound is looser (5.04 MB instead of 3.98 MB). The bracket (live ≤ measured ≤ reserved) holds.
- The 1D assertion "two snapshots > 1.5 × one" encoded the clone; it is now "two > one" and "two < 1.5 × one" (shared shards counted once). This changes the assertion because the approved behaviour changed; it is not a rebaseline of a mismatch.

### 2. Edits reach both paths, every setting — met

- 8 settings × 6 edit states (each edit kind placed, then cleared) × 2 cameras: raster and ray pass against the CPU reference of the edited world, and raster against ray has 0 non-edge disagreements (at most 162 near-edge pixels). That is 288 check lines, all 0 failures, in the validation-on run; the timing runs repeat them, also 0.
- In all 8 settings, after the initial build and after every checked state, the device meshes equal a from-scratch build byte for byte, the BLAS set equals the region set, `GpuMesh` and `GpuAccel` live equal the scene's buffers, and scratch is 0.

### 3. Refused grant — met

- Scenario A (total budget with a ballast) and scenario B (`GpuAccel` limit): the update is refused, the scene keeps the previous snapshot, and it passes the reference check of the previous world. Live bytes and live buffers are unchanged, the deferral is counted, and the ledger counts the refusal.
- In both scenarios the refusal came at the acceleration step (the new mesh buffers fit in free block space), so the rollback of already-uploaded mesh buffers is what ran. A refusal at the mesh step is not exercised by this test; that step is `GpuMeshes::upload_regions`, the 2B all-or-nothing code.
- Scenario A: after the ballast is freed, the retry shows the new snapshot, passes the new world's reference, and is byte-exact.

### 4. Planted faults — met

- One changed region (the one containing the edit) left out: 268,619 failures at the edit close-up (raster and ray together).
- Replaced buffers freed at the swap while a gated frame is pending: 2 validation errors (`VUID-vkDestroyBuffer-buffer-00922`). The same sequence with retirement: 0.

### 5. Validation — met

0 errors and 0 warnings in every correct-path test and viewer run with validation on.

### 6. Measurements

1080p, 4 cameras, 30 interleaved repetitions (median, mean over cameras). Edit latency: 10 repetitions per edit, from the world commit to the completion of one 1080p frame (raster + ray) with the new snapshot; inline jobs; validation off. Two runs:

| Merge / region | Trace ms (run 1 / 2) | Raster ms | Mesh MiB | Accel MiB | Full BLAS build ms | p95 edit-to-visible ms: voxel | 8³ | 32³ |
|---|---|---|---|---|---|---|---|---|
| greedy / brick | 0.935 / 0.918 | 0.335 | 7.5 | 28.3 | 25.1 | 12.0 / 40.3 | 8.6 / 8.9 | 20.1 / 15.9 |
| greedy / 2³ bricks | 0.945 / 0.995 | 0.302 | 6.6 | 19.0 | 6.0 | 7.8 / 12.6 | 6.8 / 11.7 | 16.9 / 16.2 |
| greedy / chunk | 0.980 / 0.981 | 0.316 | 6.5 | 19.2 | 3.0 | 9.5 / 6.4 | 6.7 / 6.4 | 16.6 / 17.3 |
| greedy / 2³ chunks | 0.919 / 0.940 | 0.292 | 6.5 | 18.7 | 2.8 | 5.8 / 5.8 | 5.8 / 6.4 | 17.4 / 16.6 |
| none / brick | 0.919 / 0.919 | 0.506 | 35.8 | 71.8 | 25.8 | 9.8 / 10.1 | 14.0 / 9.7 | 22.2 / 33.4 |
| none / 2³ bricks | 0.936 / 0.936 | 0.553 | 35.8 | 71.9 | 10.3 | 8.2 / 8.5 | 7.3 / 7.0 | 18.9 / 18.5 |
| none / chunk | 0.931 / 0.906 | 0.527 | 35.8 | 70.8 | 8.9 | 9.3 / 8.0 | 6.9 / 6.9 | 15.5 / 18.5 |
| none / 2³ chunks | 0.906 / 0.924 | 0.539 | 35.7 | 69.8 | 8.4 | 7.5 / 7.5 | 10.5 / 7.5 | 28.7 / 23.1 |

Stage medians, greedy / chunk, run 1 (ms):

| Edit | Commit + dirty | Jobs (extraction) | Publish | Region meshes | Upload | Accel update (host, incl. wait) | of which BLAS / TLAS GPU | Rebind | Frame | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| voxel (1 region, 12 BLAS triangles) | 0.003 | 0.016 | 0.011 | 0.007 | 0.031 | 3.05 | 0.07 / 0.09 | 0.04 | 1.75 | 4.9 |
| 8³ (2 regions) | 0.06 | 0.12 | 0.015 | 0.025 | 0.040 | 2.80 | 0.11 / 0.09 | 0.04 | 1.59 | 4.7 |
| 32³ (10 regions, 98 KiB) | 3.8 | 1.48 | 0.11 | 0.19 | 0.091 | 3.58 | 0.22 / 0.10 | 0.05 | 1.78 | 11.3 |

- The acceleration update's host time (about 3 ms) is mostly fixed per-update overhead (a timer and command pool made per build, size queries, the instance and table upload, submit and wait), not BLAS work.
- In the brick settings every edit rebuilds 5,103 descriptor sets (0.8 ms) and every frame records 5,103 draws.

### 7. Default rule — not usable as declared; default unchanged

- **What the rule gives:**
  - All 8 settings fit the provisional budget in both runs, with a large margin (worst p95: 40.3 ms against 50 for the 1-voxel edit, 33.4 ms against 100 for the 32³ edit).
  - Trace times are a plateau: 0.906–0.995 ms, and the same setting moves by up to 0.05 ms (5%) between runs.
  - So "within 5% of the best trace" keeps six or seven settings, and "the smallest region" then picks **greedy / brick** (run 1, and run 2 at the 50/100 budget) or **none / brick** (run 2 at a 33/67 budget, because of one 40 ms outlier).
- **Why I did not apply it:**
  - The rule assumed smaller regions make edits cheaper. They do not here.
  - Brick regions cost 25 ms for a full BLAS build instead of 3 ms, 28 MiB of acceleration memory instead of 19, and 5,103 draws per frame.
  - Their edit latency is not better (p95 12.0 / 40.3 ms against 9.5 / 6.4 for greedy / chunk).
  - The pick also changed with run-to-run noise.
  - Changing a default needs the user's yes, so the viewer keeps **greedy / chunk**.
- **Recommendation:** keep greedy / chunk.
  - Greedy uses about a quarter of the unmerged memory (26 MiB against 106 MiB), and its raster is 40% cheaper.
  - Chunk is on the edit and trace plateau.
  - 2³ chunks measured marginally better here (trace −5%, p95 about 6 ms), but a 1-voxel edit rebuilds 8× the geometry of a chunk, and denser scenes will pay for that.
  - If you prefer, the rule can be restated: among settings on the trace plateau (within run-to-run noise) that fit the budget, the least memory, then the cheaper full build.

### 8. Direct-voxel fork check — not triggered

- The default fits the provisional budget with a margin of more than 3×: greedy / chunk p95 is 9.5 ms for 1 voxel and 17.3 ms for 32³.
- Extraction plus acceleration update are 39–63% of the median latency. Most of that is the fixed host overhead above: extraction is at most 1.5 ms (32³), and BLAS + TLAS GPU time is at most 0.32 ms.
- Reopen if a denser scene or a higher edit rate misses the budget.

### 9. Viewer (M2) — met

- Left click removes, E / middle click places (rejected when nothing is hit or when it would overlap the walker), R switches raster / ray query. The title shows the snapshot, the edit count and the last edit's edit-to-visible time.
- Scripted runs at 1920×1080, MAILBOX (the new default):

| Run | Validation | Edits applied / shown / rejected / deferred | Edit-to-visible p50 / max ms | Frame p50 / p99 / max ms |
|---|---|---|---|---|
| raster, `--edit-script` | on, 0 / 0 | 81 / 81 / 39 / 0 | 11.3 / 14.2 | 2.83 / 9.12 / 13.7 |
| ray query (`--source ray`) | on, 0 / 0 | 81 / 81 / 39 / 0 | 11.8 / 27.5 | 4.46 / 12.4 / 27.1 |
| ray query | off | 81 / 81 / 39 / 0 | 7.4 / 16.6 | 2.89 / 7.43 / 31.7 |
| walk + edits (`--walk`) | on, 0 / 0 | 79 / 79 / 52 / 0; 0 overlaps, 0 respawns | 11.5 / 22.4 | 4.39 / 10.8 / 25.6 |
| ray, view cycling, resize at frame 400 | on, 0 / 0 | 25 / 25 / 15 / 0 | 10.3 / 13.3 | 2.50 / 8.29 / 15.8 |

- Rejected edits are mostly aims at the sky; the walk run also rejects places that would overlap the walker.
- Frame trace (ray, validation off, `results/viewer_2e_ray_trace.csv`): edit frames spend 5.5 ms (p50) and at most 8.5 ms inside the frame, against 1.7 ms for other frames. No interval over 16.7 ms was an edit frame; the two slow intervals (18.5 and 18.9 ms) were ordinary frames.

## Closeout

- **Contracts added:** `GpuScene` and `SceneFaults`, `Accel::update` and `AccelGarbage`, `GpuMeshes::upload_regions`, `debug_view::Source` and bindings 6–7 (checked by reflection), `Tables::replace_regions`, `Snapshot::shards_shared_with`, and the viewer flags `--edit-script` and `--source`.
- **ADR:** none yet. The granularity default, once decided, belongs in ADR-0003 as an amendment together with the edit budget.
- **Revert:** the viewer takes `--present fifo`; the other 2E parts are additive (edits are a new path; the start-up build is `GpuScene::build`, the same upload and build as before).
- **Lessons:**
  - Measure host latency with validation off; the layer's overhead scales with draws and descriptor writes.
  - A default rule needs a noise estimate before a threshold is set on it.
- **Next (user decisions):**
  - the edit budget (provisional 50 / 100 ms);
  - the granularity default (recommended: keep greedy / chunk);
  - whether M2 is met;
  - what comes after Phase 2 (a Phase 3 proposal needs asking).
