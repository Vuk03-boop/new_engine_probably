# Change: Phase 1C versioned jobs, publication and retirement

Status: passed (within the limits below)
Date and baseline: 2026-09-23. Builds on [Phase 1A + 1E](2026-09-23-phase1-world-core.md). No Git repository.
Authorization: user, "1C greenlight".

## Objective and smallest useful test

- **Objective:** the lifetime substrate for data derived from the world ([BUILD-ROADMAP](../../BUILD-ROADMAP.md) Phase 1; [PROPOSITION](../../PROPOSITION.md) §2 "Transactional derived data").
  - Jobs carry the exact inputs they read.
  - Outdated results are rejected.
  - Publication is coherent.
  - Resources are freed only after their last reader.
  - Reused handles cannot expose old data.
  - Overload falls back deterministically.
- **Gate** (roadmap): boundary edits and reused handles cannot expose stale results; overload has a deterministic fallback; tested with artificial delays and out-of-order jobs.
- **Stop condition:** every invariant holds in scripted, seeded-random and threaded runs, and each planted fault is detected.

## Scope and contracts

- **Paths:**
  - `engine/derived/` (new crate: `slots`, `surface`, `pipeline`)
  - `engine/world/src/coords.rs` (`BrickKey::origin`, `BrickKey::offset`, with a test)
  - `engine/Cargo.toml` (workspace member)
- **Derived product:** a per-brick exposed-face summary (`surface::SurfaceSummary`).
  - It stands in for surface extraction, with the same dependency shape: the brick plus its six face-neighbour layers.
  - It is not a mesh; meshing is Phase 2.
- **The contract** is in the `engine/derived/src/pipeline.rs` module docs. In short:
  1. **Dirty tracking:** each `notify_edits` call is one batch. It dirties the edited bricks plus face neighbours of boundary voxels, and forms one publication group. A batch that touches an unpublished group merges into it.
  2. **Jobs:** each job captures immutable copies of its seven input bricks. It is pure and runs on any thread.
  3. **Rejection:** a result is rejected when *cancelled* (its target was re-dirtied after dispatch) or *stale* (an input it actually depends on changed). Versions are the fast path; on a version mismatch the recorded content is compared (target brick, neighbour face layers), which is exact. The two guards are independent.
  4. **Publication:** accepted results are staged. `try_publish` re-validates them, then publishes every complete group in one new snapshot. Unfinished groups keep their previous values, so a batch is never half-visible.
  5. **Retirement:** replaced resources are freed once no reader holds a snapshot at or before the last one that contained them. `SlotPool` generations bump on free, and a slot whose generation would wrap is retired permanently.
  6. **Overload:** the queue and the in-flight count are bounded. Excess keys go to an ordered overflow set; nothing is dropped, and the order is deterministic.
- **Design corrections found by testing** (the first design was wrong on both):
  - Publishing only when *nothing* was pending anywhere starved under steady edits. The fix is per-group atomic publication, which is what PROPOSITION §2 actually specifies ("the affected publication set").
  - Brick-granular version checks rejected valid results after unrelated edits inside a neighbour brick (837 rejections at publish in one run). The fix is exact dependency comparison on version mismatch (`surface::Dependencies`).

## Commands and evidence actually produced

| Check | Result | Evidence |
|---|---|---|
| `cargo test --release -j 2` (workspace) | exit 0. 57 tests: world 29 unit + 2 integration; derived 8 unit + 13 scenario + 5 stress | `engine/results/test_release.log` |
| `cargo test -j 2` (debug: debug asserts, overflow checks) | exit 0; all 57 tests pass (stress 17.8 s, world agreement 248 s) | `engine/results/test_debug.log` |
| Scripted scenarios | pass. Cancellation with the newer result finishing first; neighbour-interior edit accepted by content; unnotified face edit rejected as stale; duplicate completion rejected; boundary edit re-derives the neighbour and removes an emptied brick; half-finished group not published while readers keep a whole old snapshot; merged batches publish together; 50 frames of sustained edits defer publication (old snapshot valid) then recover; retirement waits for readers; a reused slot rejects the old handle; overload with queue 4 and in-flight 2 over 60 bricks is lossless and repeatable | same log |
| Seeded stress (4 seeds × 6000 steps, queue 8, in-flight 6) | every invariant held on every step (details below) | same log |
| Planted faults (`derived::Faults`) | each detected on **4 of 4** seeds: missing neighbour dirtying, freeing without waiting for readers, cancel and stale check both disabled, publishing partial groups | same log |
| Independent guards | cancellation alone off, or the stale check alone off: all invariants still hold. With cancellation off, the stale check did its work (rejected_stale > 0, rejected_cancelled = 0) | same log |
| Threaded run (3 workers, random 0–1.5 ms delays) | 278 publications checked; no leaks | same log |
| Determinism | the same seed gives identical `Stats` | same log |

**Seeded stress details:**
- Invariants checked on every step: all live-snapshot handles resolve; every reader sees exactly its snapshot's published values.
- At each publication: non-awaiting keys equal a from-scratch recomputation, awaiting keys are unchanged, and no edit batch is half-published.
- At the end: drains fully and matches the oracle, with no leaked slots, retiring entries or snapshots.
- Per seed: 66–107 publications checked, 235–253 edit batches checked, 40k–355k reader reads.
- The run exercises every path: cancellations 15–31, stale rejections 9–13, accepted by content 23–32, overflow, coalescing, deferred publications and group merges.

## Result and full cost

- **Limits:**
  - The "GPU readers" are simulated tokens. Real fence and queue integration is Phase 2.
  - Each publication clones the entry map, O(entries).
  - The derived product is a stand-in.
  - Groups merge transitively, so under sustained editing of one region that region does not publish: the prior snapshot stays valid, and `publish_deferred` and `max_group_len` report it. A direct-voxel overlay, if immediacy is needed, is a separate decision (PROPOSITION §2).
  - The stress workload was set so processing keeps up on average. Sustained overload is tested separately rather than inside the random run.
  - No performance measurement.
- **Decision:** keep.

## Addendum 2026-09-23: publication cost fix

- **Measured:** `try_publish` rescanned all staged work on every call, so the total cost was quadratic in batch size (a full street build spent about 76 ms of 214 ms publishing).
- **Fixed:** per-entry `validated_at`, plus checking only groups that gained a staged member.
- **Coverage gap:** publish-time staleness had no test. It was 0 in every stress seed above. The new scenario `staged_result_is_revalidated_at_publication_after_an_unnotified_change` covers it.
- All results above still hold.
- Record: [Phase 1 perf and Phase 2 prep](2026-09-23-phase1-perf-and-phase2-prep.md).

## Closeout

- Docs: [engine/README.md](../../engine/README.md), this record, `docs/DECISIONS.md` (S-002), `docs/NOW.md`.
- No ADR: this is an internal reference contract. The publication rule and GPU-side resource retirement become an interface decision when Phase 2 adds real GPU resources.
- Revert: delete `engine/derived/`, remove it from the workspace members, and remove the two `BrickKey` helpers.
