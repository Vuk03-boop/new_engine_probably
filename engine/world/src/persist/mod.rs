//! Phase 1B persistence: an append-only edit journal, atomic snapshots and exact reload
//! ([ADR-0002](../../../../docs/adr/ADR-0002-save-format.md)).
//!
//! - A **snapshot** holds the whole world: registry, world version, every brick with its content
//!   version, and the last journal sequence number it contains.
//! - The **journal** holds one record per committed [`Transaction`], as operations.
//! - **Load** = snapshot + replay of the newer journal records. Replaying the same operations from
//!   the same state reproduces the same content versions; each record also carries the world
//!   version before and after, and any disagreement is a hard error ([`PersistError::Divergence`]).
//! - Reload is exact: `load(save(w)) == w`, versions included.
//!
//! Material identity: IDs in files are file-local. [`MaterialPolicy::Adopt`] takes the file's
//! registry; [`MaterialPolicy::MapByName`] maps every file material onto the running registry by
//! name and fails listing every missing name. Nothing is ever substituted silently.
//!
//! [`encode_snapshot`], [`scan_journal`] and [`load_bytes`] are pure; [`store::Store`] does the file I/O.

mod codec;
pub mod journal;
pub mod snapshot;
pub mod store;

use std::fmt;

pub use codec::crc32;
pub use journal::{encode_header as encode_journal_header, encode_record, scan as scan_journal, Record, Scanned, Tail, TailReason};
pub use snapshot::{encode as encode_snapshot, SnapshotMeta};
pub use store::Store;

use crate::brick::ContentVersion;
use crate::edit::{Op, Transaction};
use crate::material::{MaterialId, MaterialRegistry};
use crate::world::{World, WorldError};

#[derive(Debug)]
pub enum PersistError {
    Io(std::io::Error),
    BadMagic(&'static str),
    UnsupportedVersion { file: &'static str, major: u16 },
    Checksum(String),
    Truncated(&'static str),
    Malformed(String),
    MissingSection(String),
    /// Every file material name that the running registry lacks.
    MissingMaterials(Vec<String>),
    /// The journal does not continue this snapshot.
    LineageMismatch { snapshot: u64, journal: u64 },
    /// Records are missing between the snapshot and the journal, or inside the journal.
    SequenceGap { expected: u64, found: u64 },
    /// Replaying a record did not reproduce the recorded world versions.
    Divergence { seq: u64, detail: String },
    /// A journal operation names a material outside the file registry.
    UnknownMaterial { seq: u64, raw: u16 },
    World(WorldError),
    /// The world was edited without going through [`Store::commit`].
    UnjournaledEdits { expected: ContentVersion, found: ContentVersion },
    /// The loaded world uses a different registry than the files; write a snapshot first.
    SnapshotRequired,
    /// An earlier write failed; the files may not match memory. Reopen the store.
    Poisoned,
    AlreadyExists(std::path::PathBuf),
}

impl fmt::Display for PersistError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PersistError {}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        PersistError::Io(e)
    }
}

#[derive(Clone, Debug)]
pub enum MaterialPolicy {
    /// Use the registry stored in the file.
    Adopt,
    /// Map file materials by name onto this registry, which the loaded world then uses.
    MapByName(MaterialRegistry),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaterialReport {
    /// The loaded world's registry differs from the file's (order, names or parameters).
    pub remapped: bool,
    /// Names whose parameters differ between file and running registry; the running ones are used.
    pub param_differences: Vec<String>,
}

/// Returns the loaded world's registry, the file-ID → loaded-ID map, and the report.
pub(crate) fn map_materials(file: &MaterialRegistry, policy: &MaterialPolicy) -> Result<(MaterialRegistry, Vec<MaterialId>, MaterialReport), PersistError> {
    match policy {
        MaterialPolicy::Adopt => Ok((file.clone(), file.iter().map(|(id, _)| id).collect(), MaterialReport::default())),
        MaterialPolicy::MapByName(running) => {
            let mut map = Vec::with_capacity(file.len());
            let mut missing = Vec::new();
            let mut param_differences = Vec::new();
            for (_, def) in file.iter() {
                match running.id_of(&def.name) {
                    Some(id) => {
                        if running.get(id).expect("registered").params != def.params {
                            param_differences.push(def.name.clone());
                        }
                        map.push(id);
                    }
                    None => missing.push(def.name.clone()),
                }
            }
            if !missing.is_empty() {
                return Err(PersistError::MissingMaterials(missing));
            }
            let report = MaterialReport { remapped: running != file, param_differences };
            Ok((running.clone(), map, report))
        }
    }
}

fn remap(tx: &Transaction, map: &[MaterialId], seq: u64) -> Result<Transaction, PersistError> {
    let m = |id: Option<MaterialId>| match id {
        None => Ok(None),
        Some(id) => map.get(id.raw() as usize).copied().map(Some).ok_or(PersistError::UnknownMaterial { seq, raw: id.raw() }),
    };
    let ops = tx
        .ops
        .iter()
        .map(|op| {
            Ok(match *op {
                Op::Set { at, material } => Op::Set { at, material: m(material)? },
                Op::Fill { min, max, material } => Op::Fill { min, max, material: m(material)? },
            })
        })
        .collect::<Result<_, PersistError>>()?;
    Ok(Transaction { ops })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadReport {
    pub lineage: u64,
    pub snapshot_seq: u64,
    /// Journal records already contained in the snapshot.
    pub skipped: u64,
    pub replayed: u64,
    /// The sequence number of the last transaction in the loaded world.
    pub last_seq: u64,
    /// The sequence number of the last valid journal record (its base if it has none). Below
    /// `last_seq` when a snapshot was written but the journal was not yet restarted.
    pub journal_end_seq: u64,
    pub tail: Tail,
    /// Journal length up to the last valid record; appends continue from here.
    pub journal_valid_len: u64,
    pub materials: MaterialReport,
    pub skipped_sections: Vec<String>,
}

/// A loaded world and what the store needs to continue it.
pub struct Loaded {
    pub world: World,
    pub report: LoadReport,
    /// The registry stored in the snapshot file.
    pub file_registry: MaterialRegistry,
}

/// Snapshot + journal tail → world. Pure: reads bytes, writes nothing.
pub fn load_bytes(snapshot_bytes: &[u8], journal_bytes: &[u8], policy: &MaterialPolicy) -> Result<Loaded, PersistError> {
    let snap = snapshot::decode(snapshot_bytes, policy)?;
    let journal = journal::scan(journal_bytes)?;
    if journal.lineage != snap.meta.lineage {
        return Err(PersistError::LineageMismatch { snapshot: snap.meta.lineage, journal: journal.lineage });
    }
    let s = snap.meta.journal_seq;
    if journal.base_seq > s {
        return Err(PersistError::SequenceGap { expected: s + 1, found: journal.base_seq + 1 });
    }
    let mut world = snap.world;
    let (mut skipped, mut replayed) = (0, 0);
    for rec in &journal.records {
        if rec.seq <= s {
            skipped += 1;
            continue;
        }
        if world.version() != rec.version_before {
            return Err(PersistError::Divergence {
                seq: rec.seq,
                detail: format!("world version {} before replay, record says {}", world.version().raw(), rec.version_before.raw()),
            });
        }
        let tx = remap(&rec.tx, &snap.map, rec.seq)?;
        let applied = world.apply(&tx).map_err(PersistError::World)?;
        if applied.version_after != rec.version_after {
            return Err(PersistError::Divergence {
                seq: rec.seq,
                detail: format!("replay ended at version {}, record says {}", applied.version_after.raw(), rec.version_after.raw()),
            });
        }
        replayed += 1;
    }
    let last_seq = s + replayed;
    let report = LoadReport {
        lineage: snap.meta.lineage,
        snapshot_seq: s,
        skipped,
        replayed,
        last_seq,
        journal_end_seq: journal.base_seq + journal.records.len() as u64,
        tail: journal.tail,
        journal_valid_len: journal.valid_len,
        materials: snap.materials,
        skipped_sections: snap.skipped_sections.into_iter().map(snapshot::tag_str).collect(),
    };
    Ok(Loaded { world, report, file_registry: snap.file_registry })
}
