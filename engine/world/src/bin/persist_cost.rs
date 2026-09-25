//! Records Phase 1B save/load costs for the street block. It records costs; it does not judge them.
//!
//! Usage: `persist_cost <scratch dir> [--trials N] [--commits N]`. The directory is created and
//! then removed. Prints one JSON line. Each timing is a list over trials; commit latency is
//! per commit, including `sync_data`.

use std::path::PathBuf;
use std::time::Instant;

use world::persist::store::{self, Store};
use world::persist::{encode_journal_header, encode_snapshot, load_bytes, MaterialPolicy, SnapshotMeta};
use world::{scene, Transaction, VoxelCoord};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).expect("usage: persist_cost <scratch dir> [--trials N] [--commits N]"));
    let flag = |name: &str, default: usize| args.iter().position(|a| a == name).map(|i| args[i + 1].parse().expect("number")).unwrap_or(default);
    let trials = flag("--trials", 5);
    let commits = flag("--commits", 50);

    let t = Instant::now();
    let (world0, _) = scene::street_block();
    let generate_ms = ms(t);
    let stats = world0.stats();
    let glass = world0.materials().id_of("glass").unwrap();

    let (mut encode, mut decode, mut create, mut load, mut commit_each, mut reload) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let meta = SnapshotMeta { lineage: 1, journal_seq: 0 };
    let empty_journal = encode_journal_header(1, 0);
    let mut snap_bytes = 0;
    let mut journal_bytes = 0;
    for trial in 0..trials {
        let t = Instant::now();
        let bytes = encode_snapshot(&world0, meta);
        encode.push(ms(t));
        snap_bytes = bytes.len();
        let t = Instant::now();
        let loaded = load_bytes(&bytes, &empty_journal, &MaterialPolicy::Adopt).unwrap();
        decode.push(ms(t));
        assert_eq!(loaded.world, world0);

        let d = dir.join(format!("trial{trial}"));
        let _ = std::fs::remove_dir_all(&d);
        let t = Instant::now();
        let mut s = Store::create(&d, &world0, 1).unwrap();
        create.push(ms(t));
        let t = Instant::now();
        let (w, _) = store::load(&d, &MaterialPolicy::Adopt).unwrap();
        load.push(ms(t));
        assert_eq!(w, world0);

        let mut live = w;
        for i in 0..commits as i32 {
            let mut tx = Transaction::new();
            tx.fill(VoxelCoord::new(i * 3, 40, 0), VoxelCoord::new(i * 3 + 2, 42, 2), Some(glass));
            let t = Instant::now();
            s.commit(&mut live, &tx).unwrap();
            commit_each.push(ms(t));
        }
        drop(s);
        journal_bytes = std::fs::metadata(d.join(store::JOURNAL_FILE)).unwrap().len() as usize;
        let t = Instant::now();
        let (w, rep) = store::load(&d, &MaterialPolicy::Adopt).unwrap();
        reload.push(ms(t));
        assert_eq!((w == live, rep.replayed), (true, commits as u64));
    }
    let _ = std::fs::remove_dir_all(&dir);

    commit_each.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| commit_each[((commit_each.len() - 1) as f64 * p).round() as usize];
    println!(
        "{{\"scene\":\"street_block\",\"bricks\":{},\"voxels\":{},\"payload_estimate_bytes\":{},\"snapshot_bytes\":{snap_bytes},\
\"generate_ms\":{generate_ms:.3},\"trials\":{trials},\"encode_ms\":{},\"decode_ms\":{},\"create_store_ms\":{},\"load_ms\":{},\
\"commits_per_trial\":{commits},\"journal_bytes_after_commits\":{journal_bytes},\"commit_ms_p50\":{:.3},\"commit_ms_p90\":{:.3},\"commit_ms_max\":{:.3},\
\"load_with_journal_ms\":{}}}",
        stats.bricks,
        stats.occupied_voxels,
        stats.payload_bytes,
        list(&encode),
        list(&decode),
        list(&create),
        list(&load),
        pct(0.5),
        pct(0.9),
        pct(1.0),
        list(&reload),
    );
}
