//! Phase 1B: exact save/reload, journal replay, torn tails, corruption detection, format pin and
//! material mapping (docs/changes/2026-09-23-phase1b-persistence.md).

use std::path::PathBuf;

use world::persist::store::{self, JOURNAL_FILE, SNAPSHOT_FILE};
use world::persist::{encode_journal_header, encode_record, encode_snapshot, load_bytes, scan_journal, MaterialPolicy, PersistError, Record, SnapshotMeta, Store, Tail, TailReason};
use world::{scene, MaterialId, MaterialParams, MaterialRegistry, Transaction, VoxelCoord, World};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

fn registry(names: &[&str]) -> (MaterialRegistry, Vec<MaterialId>) {
    let mut r = MaterialRegistry::new();
    let ids = names.iter().enumerate().map(|(i, n)| r.register(n, MaterialParams::diffuse(0.1 * i as f32, 0.5, 0.25)).unwrap()).collect();
    (r, ids)
}

/// A random transaction around the origin, crossing brick and chunk edges on the negative side too.
fn random_tx(rng: &mut Rng, ids: &[MaterialId]) -> Transaction {
    let mut tx = Transaction::new();
    for _ in 0..1 + rng.next() % 4 {
        let m = if rng.next() % 4 == 0 { None } else { Some(ids[(rng.next() % ids.len() as u64) as usize]) };
        let p = VoxelCoord::new(rng.range(-40, 40), rng.range(-40, 40), rng.range(-40, 40));
        if rng.next() % 3 == 0 {
            let q = VoxelCoord::new(p.x + rng.range(0, 10), p.y + rng.range(0, 6), p.z + rng.range(0, 10));
            tx.fill(p, q, m);
        } else {
            tx.set(p, m);
        }
    }
    tx
}

fn random_world(seed: u64, txs: usize) -> World {
    let (r, ids) = registry(&["stone", "wood", "glass", "lamp"]);
    let mut w = World::new(r);
    let mut rng = Rng(seed | 1);
    for _ in 0..txs {
        w.apply(&random_tx(&mut rng, &ids)).unwrap();
    }
    w
}

/// Applies `txs` to `w`, returning the journal bytes (header with base 0, one record per transaction).
fn journal_for(w: &mut World, lineage: u64, base: u64, txs: &[Transaction]) -> Vec<u8> {
    let mut bytes = encode_journal_header(lineage, base);
    for (i, tx) in txs.iter().enumerate() {
        let a = w.apply(tx).unwrap();
        bytes.extend(encode_record(&Record { seq: base + 1 + i as u64, version_before: a.version_before, version_after: a.version_after, tx: tx.clone() }));
    }
    bytes
}

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ne_persist_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

const META: SnapshotMeta = SnapshotMeta { lineage: 0xAB, journal_seq: 0 };

#[test]
fn street_block_round_trips_exactly() {
    let (w, _) = scene::street_block();
    let snap = encode_snapshot(&w, META);
    let loaded = load_bytes(&snap, &encode_journal_header(META.lineage, 0), &MaterialPolicy::Adopt).unwrap();
    assert_eq!(loaded.world, w, "content, versions and registry");
    assert_eq!(loaded.report.tail, Tail::Clean);
    assert_eq!((loaded.report.replayed, loaded.report.last_seq), (0, 0));
    // Encoding is deterministic.
    assert_eq!(encode_snapshot(&loaded.world, META), snap);
    eprintln!("street_block snapshot: {} bytes, {} bricks, {} voxels", snap.len(), w.stats().bricks, w.stats().occupied_voxels);
}

#[test]
fn random_and_empty_worlds_round_trip_exactly() {
    let mut worlds = vec![World::new(MaterialRegistry::new()), World::new(registry(&["only"]).0)];
    worlds.extend((1..=8).map(|s| random_world(s * 7919, 60)));
    for w in &worlds {
        let loaded = load_bytes(&encode_snapshot(w, META), &encode_journal_header(META.lineage, 0), &MaterialPolicy::Adopt).unwrap();
        assert_eq!(&loaded.world, w);
    }
    // Coverage: the random worlds reach negative chunks and several bricks.
    let w = &worlds[2];
    assert!(w.bricks().any(|(k, _)| k.chunk.x < 0 || k.chunk.y < 0 || k.chunk.z < 0));
    assert!(w.stats().bricks > 20, "{:?}", w.stats());
}

#[test]
fn snapshot_plus_journal_replay_equals_the_live_world() {
    for seed in 1..=6u64 {
        let mut rng = Rng(seed * 104_729);
        let (r, ids) = registry(&["stone", "wood", "glass", "lamp"]);
        let mut live = World::new(r);
        let pre = rng.next() % 30;
        for _ in 0..pre {
            live.apply(&random_tx(&mut rng, &ids)).unwrap();
        }
        let snap = encode_snapshot(&live, SnapshotMeta { lineage: seed, journal_seq: 5 });
        let txs: Vec<_> = (0..40).map(|_| random_tx(&mut rng, &ids)).collect();
        let jour = journal_for(&mut live, seed, 5, &txs);
        let loaded = load_bytes(&snap, &jour, &MaterialPolicy::Adopt).unwrap();
        assert_eq!(loaded.world, live, "seed {seed}");
        assert_eq!((loaded.report.replayed, loaded.report.last_seq, loaded.report.tail), (40, 45, Tail::Clean));
    }
}

#[test]
fn store_commits_snapshots_and_reopens_exactly() {
    let dir = temp_dir("store");
    let (r, ids) = registry(&["stone", "wood", "glass"]);
    let mut live = World::new(r);
    let mut rng = Rng(99);
    live.apply(&random_tx(&mut rng, &ids)).unwrap();
    let mut s = Store::create(&dir, &live, 42).unwrap();
    assert!(matches!(Store::create(&dir, &live, 42), Err(PersistError::AlreadyExists(_))));
    for i in 0..30 {
        let (seq, applied) = s.commit(&mut live, &random_tx(&mut rng, &ids)).unwrap();
        assert_eq!((seq, applied.version_after), (i + 1, live.version()));
        if i == 12 {
            s.snapshot(&live).unwrap();
        }
    }
    drop(s);
    let (w, rep) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(w, live);
    assert_eq!((rep.snapshot_seq, rep.replayed, rep.last_seq), (13, 17, 30));

    // Reopen, keep committing, reload again.
    let (mut w, mut s, _) = Store::open(&dir, &MaterialPolicy::Adopt).unwrap();
    for _ in 0..5 {
        s.commit(&mut w, &random_tx(&mut rng, &ids)).unwrap();
    }
    assert_eq!(s.seq(), 35);
    let (again, _) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(again, w);

    // An edit that bypasses the store is caught before the next commit or snapshot.
    w.set(VoxelCoord::new(0, 100, 0), Some(ids[0])).unwrap();
    assert!(matches!(s.commit(&mut w, &Transaction::new()), Err(PersistError::UnjournaledEdits { .. })));
    assert!(matches!(s.snapshot(&w), Err(PersistError::UnjournaledEdits { .. })));
    // A rejected transaction (unknown material) changes nothing and writes nothing.
    let (_, foreign) = registry(&["a", "b", "c", "d", "e"]);
    let (mut w2, mut s2, _) = Store::open(&dir, &MaterialPolicy::Adopt).unwrap();
    let before = w2.clone();
    let mut bad = Transaction::new();
    bad.set(VoxelCoord::new(1, 1, 1), Some(ids[0])).set(VoxelCoord::new(2, 2, 2), Some(foreign[4]));
    assert!(matches!(s2.commit(&mut w2, &bad), Err(PersistError::World(_))));
    assert_eq!(w2, before);
    assert_eq!(store::load(&dir, &MaterialPolicy::Adopt).unwrap().0, before);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn torn_tail_at_every_offset_drops_only_the_last_transaction() {
    let mut rng = Rng(5);
    let (r, ids) = registry(&["stone", "wood"]);
    let mut live = World::new(r);
    let snap = encode_snapshot(&live, META);
    let txs: Vec<_> = (0..6).map(|_| random_tx(&mut rng, &ids)).collect();
    let jour = journal_for(&mut live, META.lineage, 0, &txs[..5]);
    let before_last = live.clone();
    let mut full = jour.clone();
    full.extend(journal_for(&mut live, META.lineage, 5, &txs[5..])[32..].iter());
    let last_start = jour.len();

    let mut checked = 0;
    for cut in last_start + 1..full.len() {
        let loaded = load_bytes(&snap, &full[..cut], &MaterialPolicy::Adopt).unwrap_or_else(|e| panic!("cut {cut}: {e}"));
        assert_eq!(loaded.world, before_last, "cut {cut}");
        assert!(matches!(loaded.report.tail, Tail::Dropped { offset, .. } if offset == last_start as u64), "cut {cut}: {:?}", loaded.report.tail);
        assert_eq!(loaded.report.journal_valid_len, last_start as u64);
        checked += 1;
    }
    assert!(checked > 20);
    // Cut exactly at a record boundary: clean.
    let loaded = load_bytes(&snap, &full[..last_start], &MaterialPolicy::Adopt).unwrap();
    assert_eq!((loaded.world, loaded.report.tail), (before_last.clone(), Tail::Clean));
    let loaded = load_bytes(&snap, &full, &MaterialPolicy::Adopt).unwrap();
    assert_eq!((&loaded.world, loaded.report.tail), (&live, Tail::Clean));
    // Zero bytes after the last record (a file extended but not written).
    let mut zeros = full.clone();
    zeros.extend([0u8; 37]);
    let loaded = load_bytes(&snap, &zeros, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(loaded.world, live);
    assert!(matches!(loaded.report.tail, Tail::Dropped { reason: TailReason::ZeroFill, bytes: 37, .. }));
}

#[test]
fn store_open_repairs_a_torn_tail_and_continues() {
    let dir = temp_dir("torn");
    let (r, ids) = registry(&["stone"]);
    let mut live = World::new(r);
    let mut s = Store::create(&dir, &live, 7).unwrap();
    let mut rng = Rng(11);
    for _ in 0..3 {
        s.commit(&mut live, &random_tx(&mut rng, &ids)).unwrap();
    }
    let good = live.clone();
    s.commit(&mut live, &random_tx(&mut rng, &ids)).unwrap();
    drop(s);
    let jp = dir.join(JOURNAL_FILE);
    let len = std::fs::metadata(&jp).unwrap().len();
    std::fs::OpenOptions::new().write(true).open(&jp).unwrap().set_len(len - 3).unwrap();

    let (mut w, mut s, rep) = Store::open(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(w, good);
    assert!(matches!(rep.tail, Tail::Dropped { reason: TailReason::Incomplete, .. }));
    assert_eq!(s.seq(), 3);
    s.commit(&mut w, &random_tx(&mut rng, &ids)).unwrap();
    let (again, rep) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!((again, rep.tail, rep.last_seq), (w, Tail::Clean, 4));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_single_byte_flip_in_a_snapshot_is_detected() {
    let w = random_world(3, 4);
    let snap = encode_snapshot(&w, META);
    let jour = encode_journal_header(META.lineage, 0);
    assert!(load_bytes(&snap, &jour, &MaterialPolicy::Adopt).is_ok(), "control: the unmodified file loads");
    for i in 0..snap.len() {
        let mut bad = snap.clone();
        bad[i] ^= 0xFF;
        assert!(load_bytes(&bad, &jour, &MaterialPolicy::Adopt).is_err(), "flip at byte {i} of {} went undetected", snap.len());
    }
    // Truncation anywhere is detected too.
    for cut in 0..snap.len() {
        assert!(load_bytes(&snap[..cut], &jour, &MaterialPolicy::Adopt).is_err(), "cut at {cut}");
    }
    eprintln!("snapshot flips checked: {} bytes", snap.len());
}

#[test]
fn every_single_byte_flip_in_a_journal_is_detected_or_dropped_as_the_tail() {
    let mut rng = Rng(21);
    let (r, ids) = registry(&["stone", "wood"]);
    let mut live = World::new(r);
    let snap = encode_snapshot(&live, META);
    let txs: Vec<_> = (0..3).map(|_| random_tx(&mut rng, &ids)).collect();
    let first_two = journal_for(&mut live, META.lineage, 0, &txs[..2]);
    let before_last = live.clone();
    let mut jour = first_two.clone();
    jour.extend(journal_for(&mut live, META.lineage, 2, &txs[2..])[32..].iter());
    let last_start = first_two.len();
    let (mut hard, mut dropped) = (0, 0);
    for i in 0..jour.len() {
        let mut bad = jour.clone();
        bad[i] ^= 0xFF;
        match load_bytes(&snap, &bad, &MaterialPolicy::Adopt) {
            Err(_) => hard += 1,
            Ok(l) => {
                assert!(i >= last_start, "flip at byte {i} (before the final record) loaded");
                assert!(matches!(l.report.tail, Tail::Dropped { reason: TailReason::Checksum, .. }), "flip at {i}: {:?}", l.report.tail);
                assert_eq!(l.world, before_last, "flip at {i}");
                dropped += 1;
            }
        }
    }
    // The final record's payload and CRC can only be dropped; its length words are a hard error.
    assert_eq!(dropped, jour.len() - last_start - 8);
    assert_eq!(hard, last_start + 8);
}

#[test]
fn journal_must_continue_the_snapshot() {
    let mut rng = Rng(8);
    let (r, ids) = registry(&["stone"]);
    let live = World::new(r);
    let snap = encode_snapshot(&live, SnapshotMeta { lineage: 1, journal_seq: 0 });
    let txs: Vec<_> = (0..3).map(|_| random_tx(&mut rng, &ids)).collect();
    let jour = journal_for(&mut live.clone(), 1, 0, &txs);
    assert!(matches!(load_bytes(&snap, &journal_for(&mut live.clone(), 2, 0, &txs), &MaterialPolicy::Adopt), Err(PersistError::LineageMismatch { .. })));
    assert!(matches!(load_bytes(&snap, &journal_for(&mut live.clone(), 1, 3, &txs), &MaterialPolicy::Adopt), Err(PersistError::SequenceGap { .. })));

    // A record whose recorded versions disagree with replay.
    let scanned = scan_journal(&jour).unwrap();
    let mut forged = encode_journal_header(1, 0);
    for (i, rec) in scanned.records.iter().enumerate() {
        let mut rec = rec.clone();
        if i == 1 {
            rec.version_after = rec.version_before;
        }
        forged.extend(encode_record(&rec));
    }
    assert!(matches!(load_bytes(&snap, &forged, &MaterialPolicy::Adopt), Err(PersistError::Divergence { seq: 2, .. })));

    // A record that skips a sequence number.
    let mut gap = encode_journal_header(1, 0);
    for rec in scanned.records.iter().filter(|r| r.seq != 2) {
        gap.extend(encode_record(rec));
    }
    assert!(matches!(load_bytes(&snap, &gap, &MaterialPolicy::Adopt), Err(PersistError::SequenceGap { expected: 2, found: 3 })));
}

#[test]
fn unknown_major_version_is_refused_and_unknown_sections_are_skipped() {
    let w = random_world(4, 10);
    let snap = encode_snapshot(&w, META);
    let jour = encode_journal_header(META.lineage, 0);
    let rehead = |bytes: &mut Vec<u8>| {
        let c = world::persist::crc32(&bytes[..16]);
        bytes[16..20].copy_from_slice(&c.to_le_bytes());
    };
    let mut v2 = snap.clone();
    v2[8..10].copy_from_slice(&2u16.to_le_bytes());
    rehead(&mut v2);
    assert!(matches!(load_bytes(&v2, &jour, &MaterialPolicy::Adopt), Err(PersistError::UnsupportedVersion { major: 2, .. })));

    // Append a future section "XTRA" with a valid CRC (tag, length, payload).
    let mut extra = snap.clone();
    let count = u32::from_le_bytes(extra[12..16].try_into().unwrap()) + 1;
    extra[12..16].copy_from_slice(&count.to_le_bytes());
    rehead(&mut extra);
    let payload = b"future data";
    let mut crc_input = b"XTRA".to_vec();
    crc_input.extend((payload.len() as u64).to_le_bytes());
    crc_input.extend(payload);
    extra.extend(b"XTRA");
    extra.extend((payload.len() as u64).to_le_bytes());
    extra.extend(world::persist::crc32(&crc_input).to_le_bytes());
    extra.extend(payload);
    let loaded = load_bytes(&extra, &jour, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(loaded.world, w);
    assert_eq!(loaded.report.skipped_sections, vec!["XTRA".to_string()]);
}

#[test]
fn materials_map_by_name_and_missing_names_fail_loudly() {
    let (file_reg, f) = registry(&["a", "b", "c"]);
    let mut w = World::new(file_reg);
    let mut tx = Transaction::new();
    tx.fill(VoxelCoord::new(-9, 0, 0), VoxelCoord::new(9, 1, 1), Some(f[0])).set(VoxelCoord::new(0, 5, 0), Some(f[1]));
    w.apply(&tx).unwrap();
    let snap = encode_snapshot(&w, META);
    let mut tail_tx = Transaction::new();
    tail_tx.set(VoxelCoord::new(3, 3, 3), Some(f[2])).set(VoxelCoord::new(-9, 0, 0), None);
    let jour = journal_for(&mut w, META.lineage, 0, &[tail_tx]);

    // Running registry: different order, an extra material, and different parameters for "b".
    let mut running = MaterialRegistry::new();
    for (name, p) in [("c", 0.2), ("x", 0.5), ("a", 0.0), ("b", 0.9)] {
        running.register(name, MaterialParams::diffuse(p, 0.5, 0.25)).unwrap();
    }
    let loaded = load_bytes(&snap, &jour, &MaterialPolicy::MapByName(running.clone())).unwrap();
    assert!(loaded.report.materials.remapped);
    assert_eq!(loaded.report.materials.param_differences, vec!["b".to_string()]);
    assert_eq!(loaded.world.materials(), &running);
    let names = |w: &World| w.occupied().map(|(v, m)| (v, w.materials().get(m).unwrap().name.clone())).collect::<Vec<_>>();
    assert_eq!(names(&loaded.world), names(&w), "same material name at every voxel");
    assert_eq!(loaded.world.version(), w.version());
    assert_ne!(loaded.world.get(VoxelCoord::new(3, 3, 3)).unwrap().raw(), f[2].raw(), "IDs really were remapped");

    // Missing names: all of them are listed.
    let (partial, _) = registry(&["a"]);
    match load_bytes(&snap, &jour, &MaterialPolicy::MapByName(partial)) {
        Err(PersistError::MissingMaterials(missing)) => assert_eq!(missing, vec!["b".to_string(), "c".to_string()]),
        other => panic!("expected MissingMaterials, got {:?}", other.map(|l| l.report)),
    }
    // Mapping onto an identical registry is not a remap.
    let same = load_bytes(&snap, &jour, &MaterialPolicy::MapByName(w.materials().clone())).unwrap();
    assert_eq!((same.world, same.report.materials.remapped), (w, false));
}

#[test]
fn remapped_store_requires_a_snapshot_before_commits() {
    let dir = temp_dir("remap");
    let (r, ids) = registry(&["a", "b"]);
    let mut w = World::new(r);
    let mut s = Store::create(&dir, &w, 3).unwrap();
    let mut tx = Transaction::new();
    tx.set(VoxelCoord::new(1, 2, 3), Some(ids[1]));
    s.commit(&mut w, &tx).unwrap();
    drop(s);
    let (running, _) = registry(&["b", "a", "z"]);
    let (mut w, mut s, rep) = Store::open(&dir, &MaterialPolicy::MapByName(running.clone())).unwrap();
    assert!(rep.materials.remapped);
    assert!(matches!(s.commit(&mut w, &Transaction::new()), Err(PersistError::SnapshotRequired)));
    s.snapshot(&w).unwrap();
    let z = running.id_of("z").unwrap();
    let mut tx = Transaction::new();
    tx.set(VoxelCoord::new(0, 0, 0), Some(z));
    s.commit(&mut w, &tx).unwrap();
    let (again, rep) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(again, w);
    assert_eq!(rep.last_seq, 2);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn crash_between_snapshot_and_journal_restart_loads_and_continues() {
    let dir = temp_dir("crash");
    let (r, ids) = registry(&["stone", "wood"]);
    let mut live = World::new(r);
    let mut s = Store::create(&dir, &live, 9).unwrap();
    let mut rng = Rng(17);
    for _ in 0..6 {
        s.commit(&mut live, &random_tx(&mut rng, &ids)).unwrap();
    }
    drop(s);
    // Simulate: the new snapshot (seq 6) was renamed into place, the journal restart never happened.
    std::fs::write(dir.join(SNAPSHOT_FILE), encode_snapshot(&live, SnapshotMeta { lineage: 9, journal_seq: 6 })).unwrap();
    let (w, rep) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(w, live);
    assert_eq!((rep.skipped, rep.replayed, rep.journal_end_seq), (6, 0, 6));
    // Also the case where the snapshot is newer than the whole journal (journal lost its last records).
    std::fs::write(dir.join(SNAPSHOT_FILE), encode_snapshot(&live, SnapshotMeta { lineage: 9, journal_seq: 8 })).unwrap();
    let (mut w, mut s, rep) = Store::open(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!((rep.last_seq, rep.journal_end_seq), (8, 6));
    s.commit(&mut w, &random_tx(&mut rng, &ids)).unwrap();
    let (again, rep) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!((again, rep.last_seq, rep.replayed), (w, 9, 1));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Format pin: the encoder must reproduce the checked-in files byte for byte, and the decoder must
/// read them. Regenerate only for an intentional format change: `NE_WRITE_FIXTURES=1 cargo test`.
#[test]
fn format_matches_the_checked_in_fixture() {
    let (r, ids) = registry(&["stone", "wood"]);
    let mut w = World::new(r);
    let mut tx = Transaction::new();
    tx.fill(VoxelCoord::new(-2, 0, 0), VoxelCoord::new(2, 1, 1), Some(ids[0])).set(VoxelCoord::new(8, 0, 0), Some(ids[1]));
    w.apply(&tx).unwrap();
    let snap = encode_snapshot(&w, SnapshotMeta { lineage: 0x0123_4567_89AB_CDEF, journal_seq: 0 });
    let mut t2 = Transaction::new();
    t2.set(VoxelCoord::new(-2, 0, 0), None).fill(VoxelCoord::new(0, 1, 0), VoxelCoord::new(1, 2, 1), Some(ids[1]));
    let jour = journal_for(&mut w, 0x0123_4567_89AB_CDEF, 0, &[t2]);

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    if std::env::var_os("NE_WRITE_FIXTURES").is_some() {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("v1_small.snap"), &snap).unwrap();
        std::fs::write(dir.join("v1_small.journal"), &jour).unwrap();
    }
    let fsnap = std::fs::read(dir.join("v1_small.snap")).expect("fixture missing; see the test doc comment");
    let fjour = std::fs::read(dir.join("v1_small.journal")).expect("fixture missing; see the test doc comment");
    // Independent of the encoder: the header fields a reader looks at first.
    assert_eq!(&fsnap[..8], b"NEWORLD\0");
    assert_eq!(u16::from_le_bytes([fsnap[8], fsnap[9]]), 1);
    assert_eq!(&fjour[..8], b"NEJOURN\0");
    assert_eq!(u64::from_le_bytes(fjour[12..20].try_into().unwrap()), 0x0123_4567_89AB_CDEF);
    assert_eq!(fsnap, snap, "snapshot encoding changed");
    assert_eq!(fjour, jour, "journal encoding changed");
    let loaded = load_bytes(&fsnap, &fjour, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(loaded.world, w);
}
