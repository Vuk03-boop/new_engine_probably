//! Snapshot file: the complete world at one journal sequence number (format in ADR-0002).
//!
//! ```text
//! header   magic "NEWORLD\0" | major u16 | minor u16 | section count u32 | crc32(previous 16 bytes) u32
//! section  tag [u8;4] | payload length u64 | crc32(tag, length, payload) u32 | payload
//! META     lineage u64 | world version u64 | journal seq u64 | brick count u64
//! MATS     count u32, then per material in ID order:
//!          name length u32 | UTF-8 name | base colour 3 x f32 bits | emissive 3 x f32 bits
//! BRKS     count u64, then per brick in `World::bricks()` order:
//!          chunk x, y, z i32 | slot u8 | content version u64 | occupancy 8 x u64 |
//!          one u16 material ID per occupied voxel, in slot order
//! ```
//!
//! All integers are little-endian. Sections end exactly at the end of the file. Unknown sections
//! are skipped; an unknown major version is refused.

use std::collections::BTreeMap;

use super::codec::{crc32, crc32_update, Reader, Writer};
use super::{MaterialPolicy, MaterialReport, PersistError};
use crate::brick::{Brick, ContentVersion};
use crate::coords::{BrickIndex, BrickKey, ChunkCoord};
use crate::dims::{BRICK_VOXELS, OCCUPANCY_WORDS};
use crate::material::{MaterialId, MaterialParams, MaterialRegistry};
use crate::world::World;

pub const MAGIC: [u8; 8] = *b"NEWORLD\0";
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;

const META: [u8; 4] = *b"META";
const MATS: [u8; 4] = *b"MATS";
const BRKS: [u8; 4] = *b"BRKS";

/// Identifies the snapshot's place in the world's history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotMeta {
    /// Shared by a snapshot and the journal that continues it.
    pub lineage: u64,
    /// The last journal record already contained in this snapshot (0: none).
    pub journal_seq: u64,
}

pub(crate) fn section_crc(tag: [u8; 4], payload: &[u8]) -> u32 {
    let c = crc32_update(crc32(&tag), &(payload.len() as u64).to_le_bytes());
    crc32_update(c, payload)
}

pub(crate) fn encode_section(out: &mut Writer, tag: [u8; 4], payload: &[u8]) {
    out.bytes(&tag);
    out.u64(payload.len() as u64);
    out.u32(section_crc(tag, payload));
    out.bytes(payload);
}

pub fn encode(world: &World, meta: SnapshotMeta) -> Vec<u8> {
    let bricks: Vec<(BrickKey, &Brick)> = world.bricks().collect();

    let mut m = Writer::default();
    m.u64(meta.lineage);
    m.u64(world.version().raw());
    m.u64(meta.journal_seq);
    m.u64(bricks.len() as u64);

    let mut r = Writer::default();
    r.u32(world.materials().len() as u32);
    for (_, def) in world.materials().iter() {
        r.u32(def.name.len() as u32);
        r.bytes(def.name.as_bytes());
        for c in def.params.base_color.iter().chain(&def.params.emissive) {
            r.f32(*c);
        }
    }

    let mut b = Writer::default();
    b.u64(bricks.len() as u64);
    for (key, brick) in &bricks {
        b.i32(key.chunk.x);
        b.i32(key.chunk.y);
        b.i32(key.chunk.z);
        b.u8(key.brick.get() as u8);
        b.u64(brick.version().raw());
        for w in brick.occupancy_words() {
            b.u64(*w);
        }
        for (_, id) in brick.voxels() {
            b.u16(id.raw());
        }
    }

    let sections = [(META, m.buf), (MATS, r.buf), (BRKS, b.buf)];
    let mut out = Writer::default();
    out.bytes(&MAGIC);
    out.u16(FORMAT_MAJOR);
    out.u16(FORMAT_MINOR);
    out.u32(sections.len() as u32);
    let hc = crc32(&out.buf);
    out.u32(hc);
    for (tag, payload) in &sections {
        encode_section(&mut out, *tag, payload);
    }
    out.buf
}

pub(crate) struct Decoded {
    pub world: World,
    pub meta: SnapshotMeta,
    /// The registry exactly as stored in the file.
    pub file_registry: MaterialRegistry,
    /// File-local material ID (index) to the loaded world's ID.
    pub map: Vec<MaterialId>,
    pub materials: MaterialReport,
    pub skipped_sections: Vec<[u8; 4]>,
}

/// Checks the header, then returns the known sections (CRC-checked) and the tags of skipped ones.
/// Known sections by tag, and the tags of skipped (unknown) sections.
type Sections<'a> = (BTreeMap<[u8; 4], &'a [u8]>, Vec<[u8; 4]>);

fn split_sections(bytes: &[u8]) -> Result<Sections<'_>, PersistError> {
    let mut r = Reader::new(bytes, "snapshot header");
    if r.take(8)? != MAGIC {
        return Err(PersistError::BadMagic("snapshot"));
    }
    let major = r.u16()?;
    let _minor = r.u16()?;
    let count = r.u32()?;
    let stored = r.u32()?;
    if crc32(&bytes[..16]) != stored {
        return Err(PersistError::Checksum("snapshot header".into()));
    }
    // Only checked after the header CRC, so a flipped version byte reads as corruption.
    if major != FORMAT_MAJOR {
        return Err(PersistError::UnsupportedVersion { file: "snapshot", major });
    }
    let mut known = BTreeMap::new();
    let mut skipped = Vec::new();
    let mut r = Reader::new(&bytes[20..], "snapshot section");
    for _ in 0..count {
        let tag = r.tag()?;
        let len = r.u64()?;
        let stored = r.u32()?;
        let len = usize::try_from(len).map_err(|_| PersistError::Truncated("snapshot section"))?;
        let payload = r.take(len)?;
        if section_crc(tag, payload) != stored {
            return Err(PersistError::Checksum(format!("snapshot section {}", tag_str(tag))));
        }
        if [META, MATS, BRKS].contains(&tag) {
            if known.insert(tag, payload).is_some() {
                return Err(PersistError::Malformed(format!("duplicate section {}", tag_str(tag))));
            }
        } else {
            skipped.push(tag);
        }
    }
    r.finish()?;
    Ok((known, skipped))
}

pub(crate) fn tag_str(tag: [u8; 4]) -> String {
    tag.iter().map(|&c| if c.is_ascii_graphic() { c as char } else { '?' }).collect()
}

pub(crate) fn decode(bytes: &[u8], policy: &MaterialPolicy) -> Result<Decoded, PersistError> {
    let (sections, skipped_sections) = split_sections(bytes)?;
    let get = |tag| sections.get(&tag).copied().ok_or_else(|| PersistError::MissingSection(tag_str(tag)));

    let mut m = Reader::new(get(META)?, "META section");
    let lineage = m.u64()?;
    let version = ContentVersion(m.u64()?);
    let journal_seq = m.u64()?;
    let brick_count = m.u64()?;
    m.finish()?;

    let file_registry = decode_registry(get(MATS)?)?;
    let (registry, map, materials) = super::map_materials(&file_registry, policy)?;

    let mut b = Reader::new(get(BRKS)?, "BRKS section");
    let n = b.u64()?;
    if n != brick_count {
        return Err(PersistError::Malformed(format!("META lists {brick_count} bricks, BRKS {n}")));
    }
    let mut bricks = Vec::new();
    for _ in 0..n {
        let chunk = ChunkCoord { x: b.i32()?, y: b.i32()?, z: b.i32()? };
        let slot = b.u8()?;
        let brick = BrickIndex::from_raw(slot).ok_or_else(|| PersistError::Malformed(format!("brick slot {slot}")))?;
        let key = BrickKey { chunk, brick };
        let ver = ContentVersion(b.u64()?);
        let mut occupancy = [0u64; OCCUPANCY_WORDS];
        for w in &mut occupancy {
            *w = b.u64()?;
        }
        let mut mats = Box::new([MaterialId::from_raw(0); BRICK_VOXELS]);
        for (i, slot) in mats.iter_mut().enumerate() {
            if occupancy[i / 64] >> (i % 64) & 1 == 1 {
                let raw = b.u16()?;
                *slot = *map.get(raw as usize).ok_or_else(|| PersistError::Malformed(format!("brick {key:?}: material {raw} not in the file registry")))?;
            }
        }
        let brick = Brick::from_parts(occupancy, mats, ver).map_err(|e| PersistError::Malformed(format!("brick {key:?}: {e}")))?;
        bricks.push((key, brick));
    }
    b.finish()?;
    let world = World::from_parts(registry, version, bricks).map_err(PersistError::Malformed)?;
    Ok(Decoded { world, meta: SnapshotMeta { lineage, journal_seq }, file_registry, map, materials, skipped_sections })
}

fn decode_registry(payload: &[u8]) -> Result<MaterialRegistry, PersistError> {
    let mut r = Reader::new(payload, "MATS section");
    let count = r.u32()? as usize;
    if count > MaterialRegistry::CAPACITY {
        return Err(PersistError::Malformed(format!("{count} materials exceed the registry capacity")));
    }
    let mut reg = MaterialRegistry::new();
    for _ in 0..count {
        let len = r.u32()? as usize;
        let name = std::str::from_utf8(r.take(len)?).map_err(|_| PersistError::Malformed("material name is not UTF-8".into()))?;
        let mut c = [0f32; 6];
        for x in &mut c {
            *x = r.f32()?;
        }
        let params = MaterialParams { base_color: [c[0], c[1], c[2]], emissive: [c[3], c[4], c[5]] };
        reg.register(name, params).map_err(|e| PersistError::Malformed(format!("material registry: {e:?}")))?;
    }
    r.finish()?;
    Ok(reg)
}
