//! Journal file: an append-only log of committed transactions (format in ADR-0002).
//!
//! ```text
//! header   magic "NEJOURN\0" | major u16 | minor u16 | lineage u64 | base seq u64 | crc32(previous 28 bytes) u32
//! record   payload length u32 | !length u32 | payload | crc32(payload) u32
//! payload  seq u64 | world version before u64 | world version after u64 | op count u32 | ops
//! op       1 (set): x, y, z i32 | material
//!          2 (fill): min x, y, z i32 | max x, y, z i32 | material
//! material 0 (clear) | 1 then u16 file-local material ID
//! ```
//!
//! Records hold operations, not resulting bricks. The first record has sequence `base seq + 1`,
//! and each later one the next number. Material IDs refer to the registry of the snapshot with
//! the same lineage.
//!
//! Tail rule: a crash can leave the last append incomplete. An incomplete final record, a final
//! record with a bad payload checksum, or trailing zero bytes are dropped and reported as
//! [`Tail::Dropped`]. Anything wrong before the final record is a hard error, and so is a length
//! field that disagrees with its complement, since the length is what locates later records.

use super::codec::{crc32, Reader, Writer};
use super::PersistError;
use crate::brick::ContentVersion;
use crate::coords::VoxelCoord;
use crate::edit::{Op, Transaction};
use crate::material::MaterialId;

pub const MAGIC: [u8; 8] = *b"NEJOURN\0";
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;
pub const HEADER_LEN: usize = 32;
/// Upper bound on one record's payload, so a corrupt length cannot request a huge allocation.
pub const MAX_RECORD_PAYLOAD: u32 = 1 << 28;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub seq: u64,
    pub version_before: ContentVersion,
    pub version_after: ContentVersion,
    pub tx: Transaction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailReason {
    /// The file ends inside a record.
    Incomplete,
    /// The final record is complete but its payload checksum fails.
    Checksum,
    /// Only zero bytes follow the last valid record.
    ZeroFill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    Clean,
    /// `bytes` bytes starting at file offset `offset` were not loaded.
    Dropped { offset: u64, bytes: u64, reason: TailReason },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scanned {
    pub lineage: u64,
    pub base_seq: u64,
    pub records: Vec<Record>,
    pub tail: Tail,
    /// File length up to the end of the last valid record.
    pub valid_len: u64,
}

pub fn encode_header(lineage: u64, base_seq: u64) -> Vec<u8> {
    let mut w = Writer::default();
    w.bytes(&MAGIC);
    w.u16(FORMAT_MAJOR);
    w.u16(FORMAT_MINOR);
    w.u64(lineage);
    w.u64(base_seq);
    let c = crc32(&w.buf);
    w.u32(c);
    debug_assert_eq!(w.buf.len(), HEADER_LEN);
    w.buf
}

fn put_material(w: &mut Writer, m: Option<MaterialId>) {
    match m {
        None => w.u8(0),
        Some(id) => {
            w.u8(1);
            w.u16(id.raw());
        }
    }
}

fn put_coord(w: &mut Writer, v: VoxelCoord) {
    w.i32(v.x);
    w.i32(v.y);
    w.i32(v.z);
}

pub fn encode_record(rec: &Record) -> Vec<u8> {
    let mut p = Writer::default();
    p.u64(rec.seq);
    p.u64(rec.version_before.raw());
    p.u64(rec.version_after.raw());
    p.u32(rec.tx.ops.len() as u32);
    for op in &rec.tx.ops {
        match *op {
            Op::Set { at, material } => {
                p.u8(1);
                put_coord(&mut p, at);
                put_material(&mut p, material);
            }
            Op::Fill { min, max, material } => {
                p.u8(2);
                put_coord(&mut p, min);
                put_coord(&mut p, max);
                put_material(&mut p, material);
            }
        }
    }
    let len = p.buf.len() as u32;
    assert!(len <= MAX_RECORD_PAYLOAD, "transaction too large for one journal record");
    let mut w = Writer::default();
    w.u32(len);
    w.u32(!len);
    w.bytes(&p.buf);
    w.u32(crc32(&p.buf));
    w.buf
}

fn get_coord(r: &mut Reader) -> Result<VoxelCoord, PersistError> {
    Ok(VoxelCoord::new(r.i32()?, r.i32()?, r.i32()?))
}

fn get_material(r: &mut Reader) -> Result<Option<MaterialId>, PersistError> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(MaterialId::from_raw(r.u16()?))),
        t => Err(PersistError::Malformed(format!("journal material tag {t}"))),
    }
}

/// Decodes a payload whose checksum already passed. A failure here is a writer bug, not a torn
/// write, so it is always a hard error.
fn decode_payload(payload: &[u8]) -> Result<Record, PersistError> {
    let mut r = Reader::new(payload, "journal record");
    let seq = r.u64()?;
    let version_before = ContentVersion(r.u64()?);
    let version_after = ContentVersion(r.u64()?);
    let n = r.u32()?;
    let mut tx = Transaction::new();
    for _ in 0..n {
        match r.u8()? {
            1 => {
                let at = get_coord(&mut r)?;
                tx.set(at, get_material(&mut r)?);
            }
            2 => {
                let (min, max) = (get_coord(&mut r)?, get_coord(&mut r)?);
                tx.fill(min, max, get_material(&mut r)?);
            }
            t => return Err(PersistError::Malformed(format!("journal op tag {t} in record {seq}"))),
        }
    }
    r.finish()?;
    Ok(Record { seq, version_before, version_after, tx })
}

pub fn scan(bytes: &[u8]) -> Result<Scanned, PersistError> {
    let mut h = Reader::new(bytes, "journal header");
    if h.take(8)? != MAGIC {
        return Err(PersistError::BadMagic("journal"));
    }
    let major = h.u16()?;
    let _minor = h.u16()?;
    let lineage = h.u64()?;
    let base_seq = h.u64()?;
    if crc32(&bytes[..HEADER_LEN - 4]) != h.u32()? {
        return Err(PersistError::Checksum("journal header".into()));
    }
    if major != FORMAT_MAJOR {
        return Err(PersistError::UnsupportedVersion { file: "journal", major });
    }

    let mut records = Vec::new();
    let mut pos = HEADER_LEN;
    let dropped = |pos: usize, reason| Tail::Dropped { offset: pos as u64, bytes: (bytes.len() - pos) as u64, reason };
    let tail = loop {
        let rest = &bytes[pos..];
        if rest.is_empty() {
            break Tail::Clean;
        }
        if rest.iter().all(|&b| b == 0) {
            break dropped(pos, TailReason::ZeroFill);
        }
        if rest.len() < 8 {
            break dropped(pos, TailReason::Incomplete);
        }
        let len = u32::from_le_bytes(rest[0..4].try_into().unwrap());
        let comp = u32::from_le_bytes(rest[4..8].try_into().unwrap());
        if len != !comp || len > MAX_RECORD_PAYLOAD {
            return Err(PersistError::Checksum(format!("journal record length at offset {pos}")));
        }
        let total = 8 + len as usize + 4;
        if rest.len() < total {
            break dropped(pos, TailReason::Incomplete);
        }
        let payload = &rest[8..8 + len as usize];
        let stored = u32::from_le_bytes(rest[8 + len as usize..total].try_into().unwrap());
        if crc32(payload) != stored {
            if rest.len() == total {
                break dropped(pos, TailReason::Checksum);
            }
            return Err(PersistError::Checksum(format!("journal record at offset {pos}")));
        }
        let rec = decode_payload(payload)?;
        let expected = base_seq + records.len() as u64 + 1;
        if rec.seq != expected {
            return Err(PersistError::SequenceGap { expected, found: rec.seq });
        }
        records.push(rec);
        pos += total;
    };
    let valid_len = match tail {
        Tail::Clean => bytes.len() as u64,
        Tail::Dropped { offset, .. } => offset,
    };
    Ok(Scanned { lineage, base_seq, records, tail, valid_len })
}
