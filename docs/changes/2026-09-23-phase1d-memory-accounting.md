# Change: Phase 1D memory accounting

Status: passed (within the limits below)
Date and baseline: 2026-09-23. Builds on 1A, 1B, 1C and 1E. No Git repository.
Authorization: user, "is 1d worth it now ... if it is and wont cause issues later on do it now then do p2" (S-004).

## Why now

- [VALIDATION-AND-BUDGETS](../../VALIDATION-AND-BUDGETS.md) §2 is headed "Memory accounting that must exist from day one". It asks for live, reserved, high-water and transient peak recorded separately, every simultaneous snapshot counted, and a deterministic overflow response for every pool.
- PROPOSITION §1 says all duplicated representations are budgeted together, and §3 asks for budgeted pools.
- Phase 2 adds the representations that actually use the 3.5 GB VRAM budget (meshes, BLAS/TLAS, build scratch, staging). They need one shared ledger to report into from their first allocation. Retrofitting that later means touching every allocation site.
- **Kept deliberately small:** accounting only. No allocator, no paging, no default budget values. The roadmap says to start with simple allocation behind logical handles, and P-001's 3.5 GB limit is applied by Phase 2 code, not chosen here.

## Scope and contracts

- **New crate `engine/memory`** (workspace, no external dependencies):
  - `Category`: world, derived, and reserved device and staging categories, with unique names.
  - `Usage {live, reserved}`: `live` is the exact payload; `reserved` is an upper bound including capacity and node overhead.
  - `Report`: per category, host and device totals kept apart, JSON output.
  - `Tracker`: high-water over successive reports.
  - `Ledger` + `Budget` + `Grant`: budgeted grants.
    - Refusal is deterministic, and changes only the refusal counter.
    - Grants are not `Clone`; a foreign grant is rejected.
    - `live` stays within the reserved size.
    - Tracks high-water, a resettable transient peak, and outstanding grants for leak checks.
  - `containers`: `Vec`, `VecDeque` and `BTreeMap`/`BTreeSet` accounting. The B-tree bound is based on the std node layout, and is tested against measured heap.
  - `CountingAlloc`: an opt-in global allocator for tests and tools. No library installs it.
- **`world`:**
  - `World::memory()`: bricks exact (header box + material box), chunk index (B-tree bound), registry.
  - `MaterialRegistry::memory()`.
  - The `WorldStats` doc now points to `memory()`; `payload_bytes` is unchanged.
- **`derived`:**
  - `Pipeline::memory()`: slots + summaries, every live snapshot's table, and all bookkeeping, including staged dependencies.
  - `SlotPool::memory`.
  - `Job::heap`: jobs are held by the caller and reported separately.
  - `heap()` on `SurfaceSummary`, `Dependencies` and `InputSnapshot`.
- **Not wired yet:** no pool allocates through `Ledger` yet. The first user is Phase 2's GPU/staging pools. Derived pools are bounded by world size and by the 1C queue limits, not by a byte budget.

## Commands and evidence actually produced

| Check | Result | Evidence |
|---|---|---|
| `cargo test --release -j 2 --no-fail-fast` | exit 0, 87 tests: memory 8; world 34 unit + 1 memory + 14 persist + 2 agreement; derived 8 unit + 1 memory + 1 persisted + 13 scenario + 5 stress | `engine/results/test_release.log` |
| `cargo test -j 2 --no-fail-fast` (debug) | exit 0, all 87 tests pass; both memory brackets also hold with the debug build's allocations | `engine/results/test_debug.log` |
| World bracket (`CountingAlloc`) | reported live ≤ measured heap ≤ reported reserved, for the street block (5.83 / 5.92 / 6.12 MB) and 4 seeded random worlds with 3,000 fill/clear transactions each (measured / reserved 0.947–0.952). Dropping the world returns the heap exactly to its starting value | `engine/results/memory_account.log` |
| Pipeline bracket | holds with jobs in flight (1.09 MB measured), one snapshot (2.75 MB), two snapshots while a reader holds the old one (5.21 MB), and after release (2.87 MB). The old snapshot's table is counted (snapshot live 1.27 → 2.60 MB) and freed after release. Dropping the pipeline returns the heap exactly to its start | same |
| Controls inside the tests | removing the bricks category, or the snapshot category, makes the reserved bound fall below measured, so the bracket can discriminate | same |
| Ledger unit tests | grants, live, release, high-water, transient-peak window, refusal at category and total limits (state unchanged, counter +1, exactly-at-limit allowed), foreign grant, live above reserved | memory unit tests |
| Planted faults | 4 of 4 detected: only the current snapshot counted, brick material arrays omitted, refusal still reserving, B-tree bound assuming full nodes. Sources restored byte for byte, with fresh mtimes | `engine/results/memory_negative_controls.log` |
| Observation | a publication with 5,103 entries peaks 4.75 MB above the pre-publish heap. That is more than one full snapshot table (about 2.6 MB reserved), because of 1C's O(entries) snapshot clone plus the new summaries | `memory_account.log` |

**Two test mistakes, fixed:**
1. My first "everything freed" checks counted the test's own `Report` and `ids` vector.
2. Printing inside the measured window allocates (the harness captures output), which made the exact-return check fail by 2,040 bytes in the normal, captured run.

Results are now kept on the stack and printed after the last measurement.

## Result and full cost

- **Limits:**
  - The B-tree bound depends on std's node layout (B = 6). A std change fails the bracket tests rather than silently mis-reporting.
  - Omissions smaller than the bound's slack are not detected: about 5% of the world total, about 30% of the derived total. The derived tables are dominated by B-tree nodes, whose fill varies.
  - `reserved` for the chunk and snapshot indexes is an upper bound, 1.4–2.8× their live payload.
  - Reported (sampled) usage cannot see transients inside one call; `CountingAlloc` can, but only in tests and tools.
  - Device categories are declared but unused.
- **Consequence for Phase 2:**
  - The snapshot-clone transient (above) is real memory, and becomes device memory once snapshots hold GPU resources.
  - Phase 2 publication should share unchanged entries, or copy only changed ones, and measure it.
- **Decision:** keep.

## Closeout

- Docs: [engine/README.md](../../engine/README.md), `docs/DECISIONS.md` (S-004), `docs/NOW.md`, and the [Phase 2 proposal](2026-09-23-phase2-proposal.md), which takes up the snapshot-transient consequence.
- No ADR: category names and the ledger are internal until Phase 2 fixes device budgets.
- Revert:
  - Delete `engine/memory/` and both `tests/memory_account.rs`.
  - Remove the `memory` workspace member and dependencies.
  - Remove the `memory()`/`heap()` methods.
