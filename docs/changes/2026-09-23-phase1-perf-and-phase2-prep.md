# Change: Phase 1 performance baseline, a publication fix, and Phase 2 prep

Status: passed (within the limits below)
Date and baseline: 2026-09-23, after 1D. No Git repository.
Authorization: user (S-005):
- "gpu memory budget sure max is 3.5 so make sure it fits there"
- "mesh and rays we can test it cant we make a slider for it"
- "screne buffers i will leave that up to you"
- "greenlight testing performance and get us ready for the next phase"

Not included: engine dependencies (`ash`, `winit`, `ash-window`) and authorization of any Phase 2 slice.

## What was done

1. **Performance baseline** (`derived/src/bin/perf_phase1.rs`). Covers the street block: generation, world edits, full derived build at 0/1/2/4/8 workers, publication scaling, CPU edit-to-publish latency for 1-voxel/8³/32³ edits, reference-trace throughput, and memory.
   - Arms are interleaved per round, with raw samples kept.
   - Every published snapshot is verified against a recomputation, outside the timed sections.
   - Environment: AC power (battery status 2, 100%), Windows "High performance" scheme, i5-12500H (16 threads), release build.
2. **A defect found and fixed:** 1C publication rescanned all staged work on every call.
3. **`clippy`** run, which was previously NOT RUN.
4. **Phase 2 prep:**
   - [ADR-0003](../adr/ADR-0003-device-budget-and-visibility-buffers.md): the device budget (implemented as `memory::device_budget`), G-buffer formats, and granularity as a swept parameter.
   - The [Phase 2 proposal](2026-09-23-phase2-proposal.md) updated, including a correction.

## The publication defect (diagnosed before changing)

- **Observation:** a full build spent about 76 ms of about 214 ms in `try_publish`. It is one batch and publishes once, and more workers barely helped.
- **Hypothesis:** every call re-validates every staged result (7 version lookups each) and re-checks every group's members. The work per call is O(staged), and there is one call per dispatch round, so the total is quadratic in batch size.
- **Discriminating test** (`publish_scaling` section): with the in-flight limit varied, publish time scaled with the call count. 320 calls took 254 ms and 6 calls took 15.5 ms, at 0.8–2.6 ms per call. Confirmed.
- **Fix** (`derived/src/pipeline.rs`):
  - Each staged result records the world version it was last validated at. Revalidation skips results already checked at the current version: equal versions mean equal content, which the 1C fast path already relies on.
  - Only groups that gained a staged member since the last call are checked for readiness. Dirtying a key always un-stages it, so readiness can only arise in `complete`.
  - The module contract is updated.
- **Coverage gap closed:** publish-time staleness (`rejected_stale_at_publish`) was 0 in every stress seed and was not tested anywhere. The new scenario `staged_result_is_revalidated_at_publication_after_an_unnotified_change` exercises it.
- **Planted faults** in the fix (`engine/results/publish_fix_negative_controls.log`):
  - P1 (never revalidate): caught by the new scenario.
  - P2 (completion does not touch its group): caught by several scenarios.
  - P3 (revalidated entries keep their old stamp) is a cost-only control. It stays correct, as designed.
  - The 4 existing 1C stress faults are still detected on 4 of 4 seeds.
- **A/B:** before and after binaries built from the same harness, 3 interleaved runs × 5 rounds (`engine/results/perf_publish_ab.jsonl`). Medians of run medians:

| Measure | Before | After |
|---|---|---|
| Full build publish time, any worker count | 72–80 ms | 19–26 ms (0.28–0.35×) |
| Full build, inline | 207–216 ms | 149–164 ms (0.72×) |
| Full build, 4 workers | 153–185 ms | 67–122 ms (0.56×) |
| Publish, in-flight 16 (320 calls) | 251–289 ms | 66–68 ms |
| Edit-to-publish, 1 voxel / 8³ / 32³ | unchanged (ratio 0.99–1.00) | — |

## Baseline after the fix (`engine/results/perf_phase1.jsonl`, 5 rounds, medians)

| Measure | Result |
|---|---|
| Street block generation | 135 ms (5,103 bricks, 1.52 M voxels) |
| World 32³ fill / clear (32,768 voxels, one transaction) | 0.79 / 0.47 ms |
| Full derived build, workers 0 / 1 / 2 / 4 / 8 | 122 / 206 / 115 / 83 / 100 ms. 4 workers is the best, at 61.5 k jobs/s. 1 worker is slower than inline: every job pays a channel round trip while the main thread waits. Scaling is limited by main-thread work: input capture (7 brick copies per job), the completion check, and publication (about 25 ms) |
| Publication per call, in-flight 16 / 64 / 256 / 1024 | 0.22 / 0.30 / 0.57 / 1.53 ms |
| CPU edit-to-publish, 1 voxel | 0.30 ms. Publication 0.26 ms of it: the snapshot-table clone (O(entries)) that 2E replaces |
| CPU edit-to-publish, 8³ box (511 voxels, 8 jobs) | 0.64 ms |
| CPU edit-to-publish, 32³ box (32,256 voxels, 160 jobs) | 10.8 ms: notify 5.7, jobs 3.2, apply 1.1, publish 0.6. Notify does per-voxel dirty-key work; coalescing per brick is a known improvement, not made |
| Reference DDA, single thread | 0.174 M rays/s (200 k rays, 162 k hits). A correctness oracle, not a renderer |
| Memory at the end (1D report) | world 5.90 MB live / 6.19 MB reserved; pipeline 1.53 / 3.96 MB |

The stages in edit-to-publish are sequential on one thread, so their sum is the latency. **Not included:** anything on the GPU (upload, BLAS rebuild, frame latency). That is 2E.

## clippy (`engine/results/clippy.log`)

- `cargo clippy --release --workspace --all-targets` exits 0.
- It found 8 style lints and no correctness lints.
- The two in code written this session are fixed: items after a test module in `derived/src/surface.rs`, and a complex type in `persist/snapshot.rs`, now a type alias.
- **Left as found**, as pre-existing style only: 5 × manual `is_multiple_of` (`world/src/dims.rs` and 4 in tests) and 1 × index loop (`world/src/reference.rs`).

## Commands and evidence

| Check | Result | Evidence |
|---|---|---|
| `cargo test --release -j 2 --no-fail-fast` | exit 0, 89 tests: memory 9; world 34 + 1 + 14 + 2; derived 8 + 1 + 1 + 14 + 5 | `engine/results/test_release.log` |
| `cargo test -j 2 --no-fail-fast` (debug) | exit 0, 89 tests pass | `engine/results/test_debug.log` |
| `perf_phase1 --rounds 5` | exit 0; every verify passed | `engine/results/perf_phase1.jsonl` |
| Publication A/B | as above | `engine/results/perf_publish_ab.jsonl` |
| Fix controls | P1, P2 caught; P3 cost-only, as designed | `engine/results/publish_fix_negative_controls.log` |
| `memory::device_budget` | laptop 3.15 × 10⁹ B (the cap governs: the driver's 3,367.7 MiB is about 3.53 × 10⁹ B); smaller driver budget governs; larger GPU capped; ledger refuses one byte over | memory unit test |

## Limits

- Single machine, warm process. "Cold" is only the first round inside a run; there is no cold-process or cold-cache control.
- Worker-count medians vary by ±20–40% between runs (for example, 4 workers ran at 67–122 ms), so a single median is not a stable ranking. Differences under about 1.3× are not claimed.
- No thermal or clock logging.
- Timings are CPU only.

## Closeout

- Docs:
  - ADR-0003
  - the Phase 2 proposal (decision status, the sweep, the correction)
  - [the 1C record](2026-09-23-phase1c-jobs-retirement.md) (addendum)
  - [engine/README.md](../../engine/README.md)
  - `docs/DECISIONS.md` (S-005, A-005)
  - `docs/NOW.md`
- Revert:
  - the fix: restore `try_publish`'s full rescans, and remove `Staged::validated_at` and `Pipeline::touched`
  - the harness: delete `perf_phase1.rs`
  - the budget: remove `device_budget` and its constants
