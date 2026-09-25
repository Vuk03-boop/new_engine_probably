# Change (planned): Phase 1B edit journal, snapshots and exact save/reload

Status: **implemented**. Authorized by "1b greenlit" with the recommendations below adopted; see [ADR-0002](../adr/ADR-0002-save-format.md) and the [1B record](2026-09-23-phase1b-persistence.md). The text below is the original plan, kept unchanged.
Originally: planned, not authorized. The user asked for an expanded proposal ("1b expand on").
Date: 2026-09-23. Builds on 1A ([world core](2026-09-23-phase1-world-core.md)) and 1C ([jobs/retirement](2026-09-23-phase1c-jobs-retirement.md)).

## What 1B is for

- **Roadmap gate** ([BUILD-ROADMAP](../../BUILD-ROADMAP.md) Phase 1): exact save/reload and material identity, and a headless world/edit replay.
- **PROPOSITION §2:** store persistent edits in a journal or transactional store. Generation produces world data; it is not a substitute for edited geometry.
- Today the world exists only in memory, and the street scene is regenerated on every run. 1B makes the world a durable artefact: saved, reloaded bit-for-bit, and reproducible from its edit history.

## Proposed design

1. **Transactions.**
   - An edit transaction is an ordered list of operations (`set voxel`, `fill box`), applied atomically to the world.
   - It is the same unit that `derived` uses as a publication batch: one transaction is one `notify_edits`.
   - Every transaction gets a sequence number and records the world version before and after.
2. **Journal** (append-only file).
   - One record per committed transaction: length, sequence number, operations, then a checksum.
   - It stores the *operations*, not the resulting bricks, so it stays small for typical edits.
   - An optional "prior values" block would make undo possible later. It is not in 1B.
3. **Snapshot** (full save).
   - Contents:
     - the material registry (names and parameters, in ID order)
     - the world version counter
     - every stored brick: key, occupancy mask, material array, content version
     - the journal sequence number it includes
   - Writing is atomic: write a temporary file, flush, then rename over the old one. Windows `std::fs::rename` replaces the existing file.
4. **Load = snapshot + journal tail.**
   - Replay the journal records newer than the snapshot.
   - A torn or corrupt final record is detected by its length and checksum, dropped, and reported. It is never half-applied.
   - Corruption earlier in the file is a hard error.
5. **Format.**
   - Own little-endian binary: magic, format version, sections with lengths and checksums. No external crates; the checksum would be a small in-house CRC32.
   - An unknown major version is refused; an unknown section is skipped, for forward compatibility.
   - Uncompressed at first. The street block is about 5.6 MB raw. Per-brick palettes or RLE come later, and only when measured size matters.
6. **Material identity across saves.** IDs are file-local. On load, the file's registry is either adopted as-is, or mapped by name onto the running engine's registry. A missing name is a loud error listing every missing material, never a silent substitution.

## Decisions needed from you (my recommendation first)

1. **Persist content versions?**
   - Recommended: yes. A reload is then bit-identical, including versions, and replaying a journal tail reproduces the same stamps.
   - The alternative is to restamp on load: simpler, but "reload equals the saved world" then holds only for content.
2. **Transaction granularity.**
   - Recommended: one transaction per edit batch (the `notify_edits` unit), so persistence and publication agree.
   - The alternative is one record per voxel write: simpler, but larger and slower to replay.
3. **Material mismatch policy.**
   - Recommended: map by name and fail loudly on any missing name.
   - The alternative is adopting the file's registry wholesale.
4. **Compression.** Recommended: none in 1B; measure first.
5. **Undo.** Recommended: out of 1B scope; the journal layout leaves room for it.

The file format is a lasting interface, so 1B would begin with **ADR-0002 (save format and versioning policy)**, recording choices 1–3.

## Proposed tests (smallest discriminating set first)

- **Round trip:** generate, save, load, and require `World ==` the original (content, versions and registry). This runs on the street block and on seeded random worlds, including negative coordinates and chunk edges.
- **Replay:** snapshot at a random point, then N random transactions, then load snapshot + journal. The result must equal the live world. The live and replayed runs use the same seed.
- **Torn tail:** truncate the journal at every byte offset inside its last record. Load must succeed, with the last transaction absent and reported.
- **Corruption (negative control):** flipping any single byte in a snapshot section or an earlier journal record must be detected. This proves the checksums can fail.
- **Format pin:** a small saved file checked in as a fixture, so an accidental format change fails a test.
- **Material mapping:** a file saved with registry order A, loaded into registry order B, gives identical material *names* per voxel. Loading with one name missing fails and names it.
- **Integration with 1C:** after load, `Pipeline::mark_all` publishes a snapshot that matches the oracle.
- **Costs to record, not to judge yet:** save and load time and file size for the street block.

## Scope and exclusions

- **Proposed paths:** a new `engine/persist/` crate, or a module in `world`; the ADR would settle that. Plus tests and `engine/README.md`.
- **Excluded:**
  - compression
  - undo
  - streaming or partial loads (Phase 6)
  - multi-file worlds
  - saving derived data (it is rebuilt after load)
  - allocation accounting (1D)
