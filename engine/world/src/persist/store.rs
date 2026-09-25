//! A world's files on disk: `world.snap` and `world.journal` in one directory.
//!
//! - [`Store::commit`] applies a transaction to the world, then appends its record and flushes it
//!   to the device (`sync_data`) before returning. The world in memory is ahead of the files only
//!   while `commit` runs; if the append fails, the store refuses further writes ([`PersistError::Poisoned`]).
//! - [`Store::snapshot`] writes a full snapshot to a temporary file, flushes it and renames it over
//!   the old one, then restarts the journal the same way. A crash between the two renames leaves a
//!   new snapshot with an old journal whose records it already contains; load skips them.
//! - [`Store::open`] loads, then truncates a dropped journal tail so that new records follow the
//!   last valid one. [`load`] is the read-only variant.
//!
//! Limit: directory entries are not flushed separately (Windows has no portable directory fsync),
//! so a rename may be lost on power failure even though the renamed file's data was flushed.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::journal::{self, Record};
use super::snapshot::{self, SnapshotMeta};
use super::{load_bytes, LoadReport, MaterialPolicy, PersistError};
use crate::brick::ContentVersion;
use crate::edit::{Applied, Transaction};
use crate::world::World;

pub const SNAPSHOT_FILE: &str = "world.snap";
pub const JOURNAL_FILE: &str = "world.journal";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Ready,
    SnapshotRequired,
    Poisoned,
}

pub struct Store {
    dir: PathBuf,
    lineage: u64,
    journal: File,
    /// The last committed sequence number.
    seq: u64,
    /// The world version after the last commit or load; the world must still be at it.
    expected: ContentVersion,
    state: State,
}

/// A lineage value that differs between stores created at different times or by different
/// processes. Not cryptographic; it only has to make mixing up two worlds' files unlikely.
pub fn fresh_lineage() -> u64 {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let mut x = nanos ^ (std::process::id() as u64).rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    // splitmix64 finaliser
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), PersistError> {
    let tmp = dir.join(format!("{name}.tmp"));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, dir.join(name))?;
    Ok(())
}

fn open_append(dir: &Path) -> Result<File, PersistError> {
    Ok(OpenOptions::new().append(true).open(dir.join(JOURNAL_FILE))?)
}

/// Reads and loads a store's files without modifying them.
pub fn load(dir: &Path, policy: &MaterialPolicy) -> Result<(World, LoadReport), PersistError> {
    let snap = fs::read(dir.join(SNAPSHOT_FILE))?;
    let jour = fs::read(dir.join(JOURNAL_FILE))?;
    let loaded = load_bytes(&snap, &jour, policy)?;
    Ok((loaded.world, loaded.report))
}

impl Store {
    /// Creates a new store for `world` in `dir` (created if needed). Refuses to overwrite files.
    pub fn create(dir: &Path, world: &World, lineage: u64) -> Result<Store, PersistError> {
        fs::create_dir_all(dir)?;
        for name in [SNAPSHOT_FILE, JOURNAL_FILE] {
            if dir.join(name).exists() {
                return Err(PersistError::AlreadyExists(dir.join(name)));
            }
        }
        write_atomic(dir, SNAPSHOT_FILE, &snapshot::encode(world, SnapshotMeta { lineage, journal_seq: 0 }))?;
        write_atomic(dir, JOURNAL_FILE, &journal::encode_header(lineage, 0))?;
        Ok(Store { dir: dir.to_owned(), lineage, journal: open_append(dir)?, seq: 0, expected: world.version(), state: State::Ready })
    }

    /// Loads the store, drops a torn journal tail from the file, and returns the world ready for
    /// more commits. With a remapping material policy, commits wait for [`Store::snapshot`].
    pub fn open(dir: &Path, policy: &MaterialPolicy) -> Result<(World, Store, LoadReport), PersistError> {
        let snap = fs::read(dir.join(SNAPSHOT_FILE))?;
        let jour = fs::read(dir.join(JOURNAL_FILE))?;
        let loaded = load_bytes(&snap, &jour, policy)?;
        let r = &loaded.report;
        if r.journal_end_seq < r.last_seq {
            // The snapshot is newer than the whole journal: restart the journal after it.
            write_atomic(dir, JOURNAL_FILE, &journal::encode_header(r.lineage, r.last_seq))?;
        } else if r.journal_valid_len < jour.len() as u64 {
            let f = OpenOptions::new().write(true).open(dir.join(JOURNAL_FILE))?;
            f.set_len(r.journal_valid_len)?;
            f.sync_all()?;
        }
        let state = if loaded.world.materials() == &loaded.file_registry { State::Ready } else { State::SnapshotRequired };
        let store = Store { dir: dir.to_owned(), lineage: r.lineage, journal: open_append(dir)?, seq: r.last_seq, expected: loaded.world.version(), state };
        Ok((loaded.world, store, loaded.report))
    }

    pub fn lineage(&self) -> u64 {
        self.lineage
    }

    /// The last committed sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    fn check(&self, world: &World) -> Result<(), PersistError> {
        if self.state == State::Poisoned {
            return Err(PersistError::Poisoned);
        }
        if world.version() != self.expected {
            return Err(PersistError::UnjournaledEdits { expected: self.expected, found: world.version() });
        }
        Ok(())
    }

    /// Applies `tx` to `world` and makes it durable. Returns the new sequence number and what changed.
    pub fn commit(&mut self, world: &mut World, tx: &Transaction) -> Result<(u64, Applied), PersistError> {
        self.check(world)?;
        if self.state == State::SnapshotRequired {
            return Err(PersistError::SnapshotRequired);
        }
        let applied = world.apply(tx).map_err(PersistError::World)?;
        let rec = Record { seq: self.seq + 1, version_before: applied.version_before, version_after: applied.version_after, tx: tx.clone() };
        let bytes = journal::encode_record(&rec);
        if let Err(e) = self.journal.write_all(&bytes).and_then(|_| self.journal.sync_data()) {
            self.state = State::Poisoned;
            return Err(PersistError::Io(e));
        }
        self.seq = rec.seq;
        self.expected = applied.version_after;
        Ok((rec.seq, applied))
    }

    /// Writes a full snapshot of `world` and restarts the journal after it.
    pub fn snapshot(&mut self, world: &World) -> Result<(), PersistError> {
        self.check(world)?;
        let result = (|| {
            write_atomic(&self.dir, SNAPSHOT_FILE, &snapshot::encode(world, SnapshotMeta { lineage: self.lineage, journal_seq: self.seq }))?;
            write_atomic(&self.dir, JOURNAL_FILE, &journal::encode_header(self.lineage, self.seq))?;
            open_append(&self.dir)
        })();
        match result {
            Ok(f) => {
                self.journal = f;
                self.state = State::Ready;
                Ok(())
            }
            Err(e) => {
                self.state = State::Poisoned;
                Err(e)
            }
        }
    }
}
