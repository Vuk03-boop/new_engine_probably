//! Phase 1B × 1C: a world reloaded from its files feeds the derived pipeline, and committed
//! transactions are the pipeline's edit batches.

mod common;

use common::{check_snapshot, run_to_idle};
use derived::{Config, Pipeline};
use world::persist::store::{self, Store};
use world::persist::MaterialPolicy;
use world::{scene, Transaction, VoxelCoord};

#[test]
fn reloaded_world_publishes_the_oracle_and_commits_feed_edit_batches() {
    let dir = std::env::temp_dir().join(format!("ne_persist_derived_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (mut live, _) = scene::street_block();
    let mut s = Store::create(&dir, &live, 1).unwrap();
    let mats = live.materials().clone();
    let glass = mats.id_of("glass").unwrap();
    let asphalt = mats.id_of("asphalt").unwrap();

    // Two committed transactions after the snapshot: a filled block and a hole, both across brick edges.
    let mut a = Transaction::new();
    a.fill(VoxelCoord::new(-5, 1, -5), VoxelCoord::new(5, 12, 5), Some(glass));
    let mut b = Transaction::new();
    b.fill(VoxelCoord::new(-2, 0, -2), VoxelCoord::new(2, 1, 2), None).set(VoxelCoord::new(0, 30, 0), Some(asphalt));
    s.commit(&mut live, &a).unwrap();
    s.commit(&mut live, &b).unwrap();
    drop(s);

    let (mut w, mut s, rep) = Store::open(&dir, &MaterialPolicy::MapByName(mats)).unwrap();
    assert_eq!(w, live);
    assert_eq!((rep.replayed, rep.materials.remapped), (2, false));

    let mut p = Pipeline::new(Config::default());
    p.mark_all(&w);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();

    // One commit is one edit batch.
    let mut c = Transaction::new();
    c.fill(VoxelCoord::new(6, 0, 6), VoxelCoord::new(10, 4, 10), None).set(VoxelCoord::new(7, 8, 7), Some(glass));
    let (_, applied) = s.commit(&mut w, &c).unwrap();
    assert!(!applied.changed.is_empty());
    p.notify_edits(&applied.changed);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();

    // The derived data after reload equals the derived data of the live world.
    let (reloaded, _) = store::load(&dir, &MaterialPolicy::Adopt).unwrap();
    assert_eq!(reloaded, w);
    check_snapshot(&p, &reloaded).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}
