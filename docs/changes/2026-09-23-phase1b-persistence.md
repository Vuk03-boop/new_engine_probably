# Change: Phase 1B edit transactions, journal, snapshots and exact save/reload

Status: passed (within the limits below)
Date and baseline: 2026-09-23. Builds on [1A + 1E](2026-09-23-phase1-world-core.md) and [1C](2026-09-23-phase1c-jobs-retirement.md). No Git repository.
Authorization: user, "1b greenlit" (S-003). The five proposal decisions were taken as recommended; see [ADR-0002](../adr/ADR-0002-save-format.md).
Plan: [1B proposal](2026-09-23-phase1b-proposal.md).

## Objective and smallest useful test

- **Objective:** the world becomes a durable artefact. It is saved and reloaded bit for bit, versions included, and reproduced from its edit history. Material identity survives a registry change.
- **Gate** ([BUILD-ROADMAP](../../BUILD-ROADMAP.md) Phase 1): exact save/reload and material identity; headless world/edit replay.
- **Stop condition:** round trip, replay, torn tail, corruption, fixture and mapping tests pass in release and debug, and planted faults are detected.

## Scope

- **New:**
  - `engine/world/src/edit.rs`: `Transaction`, `Op`, `Applied`, `World::apply`, with atomic material pre-validation.
  - `engine/world/src/persist/`:
    - `codec`: little-endian encoding and CRC-32
    - `snapshot`, `journal`: the file layouts
    - `mod`: `load_bytes`, replay, the material policy
    - `store`: `Store`, `load`
  - `engine/world/src/bin/persist_cost.rs`.
  - Tests: `engine/world/tests/persist.rs` with fixtures in `engine/world/tests/fixtures/v1_small.{snap,journal}`, and `engine/derived/tests/persisted.rs`.
- **Changed (crate-private restore hooks only):**
  - `Brick::from_parts`
  - `World::from_parts`
  - `BrickIndex::from_raw`
  - the `world/src/lib.rs` module list

  No existing public behavior changed.
- **Excluded:**
  - compression, undo
  - streaming or partial loads
  - multi-file worlds
  - saving derived data
  - 1D allocation accounting

## Commands and evidence actually produced

| Check | Result | Evidence |
|---|---|---|
| `cargo test --release -j 2` (workspace) | exit 0, 77 tests: world 34 unit + 14 persist + 2 agreement; derived 8 unit + 1 persisted + 13 scenario + 5 stress | `engine/results/test_release.log` |
| `cargo test -j 2` (debug: debug asserts, overflow checks) | exit 0, all 77 tests pass (world agreement 229 s, persist 2.8 s) | `engine/results/test_debug.log` |
| Round trip | street block and 10 other worlds (empty, one material, 8 seeded random worlds reaching negative chunks) reload `==`: content, versions, registry. Encoding is deterministic | persist tests |
| Replay | 6 seeds: snapshot at seq 5, then 40 random transactions; the replayed world `==` the live one | same |
| Store | 30 commits with a snapshot at 13; reload `==`; reopen + 5 commits `==`. Unjournaled edit refused; a rejected transaction writes nothing | same |
| Torn tail | every truncation inside the final record loads the world before that transaction, with `Tail::Dropped` at the right offset. Boundary cut and zero fill also covered. `Store::open` truncates and continues | same |
| Corruption | every single-byte flip (1,350 bytes) and every truncation of a snapshot fails to load. Every flip in a journal header or non-final record is a hard error; a flip in the final record's payload or CRC is dropped as the tail, and the exact counts are asserted | same |
| Continuity | lineage mismatch, a journal starting after the snapshot, a skipped record and a forged `version_after` are each a hard error | same |
| Format | unknown major version refused; unknown section skipped and reported. The fixture is reproduced byte for byte and decodes. The fixture was also decoded independently in Python (zlib CRCs match, and bricks, versions and ops are as expected) | same; one-off check, not a test |
| Materials | reordered running registry with an extra material and different parameters: same material name at every voxel, IDs really remapped, parameter difference reported. Two missing names both listed. After a remapped open, commits are refused until a snapshot is written | same |
| Crash window | new snapshot with old journal: records skipped. Snapshot ahead of the whole journal: `open` restarts the journal and commits continue | same |
| 1C integration | street block + 2 commits, reopened with `MapByName`: `mark_all` → published snapshot equals the oracle. A further commit → `notify_edits(applied.changed)` → equals the oracle, and a reload equals the live world | `derived/tests/persisted.rs` |
| Planted faults | 6 of 6 detected, each by the test aimed at it: section CRC ignored, mid-journal CRC failure treated as tail, `version_after` unchecked, silent minor bump, torn tail not truncated, apply without pre-validation. Sources restored and compared byte for byte | `engine/results/persist_negative_controls.log` |
| Costs, recorded not judged (`persist_cost`, 5 trials, this laptop) | snapshot 3,465,655 B (5,825,856 B in-memory estimate). Encode 11–17 ms, decode 16–20 ms, create store (write + fsync) 18–22 ms, load from files 20–22 ms. Commit of a 2×2×2 fill: p50 0.87 ms, p90 1.73 ms, max 7.6 ms (fsync). Journal record 68 B. Load with 50 journaled commits 17–24 ms | `engine/results/persist_cost.json` |

**Harness incident** (recorded because it briefly produced a false failure):
- The first fault harness restored files with `shutil.copy2`, which also restores the old mtime. Cargo therefore kept the last faulted build, and the next full run failed on the fault itself (F6).
- The sources were verified clean. The harness now restores with a fresh mtime, and both the harness and the suite were re-run.
- The individual fault runs were valid either way, since each fault edit had a new mtime and the crate recompiles from current sources.

## Result and full cost

- **Limits:**
  - Directory entries are not fsynced, so a rename can be lost on power failure (ADR-0002).
  - Records are verified by CRC-32, which detects all single-byte and burst errors up to 32 bits. It is not a cryptographic check.
  - Loads read whole files into memory.
  - A transaction must fit in one record (256 MiB payload cap).
  - `persist_cost` timings are single-machine and warm-cache; no paired comparison exists, because there is nothing to compare against yet.
- **Decision:** keep.

## Closeout

- Docs: ADR-0002, [engine/README.md](../../engine/README.md), `docs/DECISIONS.md` (S-003, A-004), `docs/NOW.md`. The proposal record is marked implemented.
- Revert: delete `world/src/edit.rs`, `world/src/persist/`, `world/src/bin/persist_cost.rs`, `world/tests/persist.rs`, `world/tests/fixtures/` and `derived/tests/persisted.rs`. Then remove the three `from_parts`/`from_raw` hooks and the `lib.rs` lines.
