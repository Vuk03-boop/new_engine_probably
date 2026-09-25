# ADR 0002 — World save format, edit journal and versioning policy

Status: accepted
Date: 2026-09-23
Decision authority: user, "1b greenlit". This authorized the [1B proposal](../changes/2026-09-23-phase1b-proposal.md) without changing its five open decisions, so the proposal's recommendations were adopted, consistent with the user's standing delegation of secondary choices. Any of them can be reopened by the user.
Related change/evidence: [Phase 1B persistence](../changes/2026-09-23-phase1b-persistence.md)

## Context and actual constraint

- The world existed only in memory; the street scene was regenerated on every run.
- The roadmap gate is exact save/reload, stable material identity, and a headless edit replay.
- PROPOSITION §2 asks for persistent edits in a journal or transactional store.
- A file format is a lasting interface: once worlds are saved, changing it needs migration.

## Options and tradeoffs

- **Snapshot only** (rewrite the whole world per save). Simplest, but a 3.5 MB write per edit and no history. Rejected.
- **Journal of resulting bricks.** Replay is trivial, but each record carries 1 KB+ per touched brick. Rejected for now.
- **Journal of operations + periodic snapshots (chosen).** Records are small (68 bytes per small fill), and replay reuses `World::apply`. Exactness depends on replay determinism, which is checked on every record (below).
- **An external serialization crate or database.** It would need new-dependency approval, and it would not remove the need for our own exactness and torn-write rules. Not now.
- **Compression.** Deferred until size is measured to matter. The street snapshot is 3.47 MB.

## Decision

1. **Content versions are persisted.** A reload is bit-identical, versions included: `load(save(w)) == w`.
2. **One journal record per transaction.** An `edit::Transaction` (ordered `Set`/`Fill` ops) is atomic. It is the same unit `derived` uses as an edit batch (`Applied::changed` → `notify_edits`).
3. **Materials map by name.** IDs in files are file-local.
   - `MaterialPolicy::Adopt` takes the file's registry.
   - `MaterialPolicy::MapByName(running)` maps every file material onto the running registry by name. Any missing name fails, and the error lists all of them. Parameter differences are reported, and the running parameters are used.
4. **No compression, no undo** in format 1. The record layout leaves room for a later "prior values" block.
5. **Location:** `world::persist`, a module of the `world` crate, not a separate crate. Restoring exact versions needs crate-private constructors (`Brick::from_parts`, `World::from_parts`) that must not become public mutation paths.

**Format 1** (little-endian; full layouts in `world/src/persist/snapshot.rs` and `journal.rs` module docs):
- `world.snap`: magic `NEWORLD\0`, major/minor u16, section count, header CRC-32. Then the sections `META`, `MATS` and `BRKS`, each with tag, length and a CRC-32 over tag+length+payload. Bricks store only occupied voxels' material IDs.
- `world.journal`: magic `NEJOURN\0`, major/minor, lineage, base sequence, header CRC. Then the records, each laid out as: length, !length, payload (seq, world version before/after, ops), CRC-32 of payload.

## Contracts and consequences

- **Replay is verified, not trusted.**
  - Every record must continue the sequence, start at the recorded world version, and end at the recorded one.
  - Anything else is `Divergence`, `SequenceGap` or `LineageMismatch`, and is a hard error.
- **Torn tail rule:**
  - An incomplete final record, a final record with a bad payload CRC, or trailing zeros is dropped and reported (`Tail::Dropped`).
  - Anything earlier is a hard error. So is a length word that disagrees with its complement, because lengths locate later records.
  - `Store::open` truncates a dropped tail before appending.
- **Durability:**
  - `commit` applies, appends and `sync_data`s before returning. A failed append poisons the store.
  - `snapshot` writes temp+fsync+rename for the snapshot, then for a fresh journal. A crash between the two renames is recoverable: load skips records the snapshot contains, and `open` restarts a journal the snapshot has overtaken.
  - **Not covered:** directory entries are not fsynced (no portable directory fsync on Windows), so a rename can be lost on power failure.
- **Versioning:**
  - An unknown major version is refused.
  - Unknown snapshot sections are skipped and reported.
  - Minor bumps must stay readable by older readers.
  - Any encoding change fails the fixture test (`world/tests/fixtures/v1_small.*`) until the fixture is regenerated deliberately.
- **Guards:**
  - Edits made around the store are refused (`UnjournaledEdits`).
  - After a remapping load, commits are refused until a snapshot is rewritten in the running registry (`SnapshotRequired`).
- **Not saved:** derived data (rebuilt after load), streaming or partial loads (Phase 6), multi-file worlds.

## Validation and revisiting

- **Acceptance evidence:** [Phase 1B record](../changes/2026-09-23-phase1b-persistence.md). Covers round trip, replay, torn tails at every offset, every single-byte flip, format fixture, material mapping, crash-window recovery, 1C integration, and six planted faults detected.
- **Reopen if:**
  - measured snapshot size or load time matters (compression, palettes)
  - undo is wanted
  - streaming needs per-region files
  - a new edit operation is needed (a minor or major bump, with a fixture)

## Implementation status

Implemented with gates passed, as listed in the change record.

## Supersession or errata

None.
