#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use derived::{extract_world, BrickMesh, Merge, Pipeline};
use world::{BrickKey, MaterialId, MaterialParams, MaterialRegistry, World};

pub fn world_with(n: usize) -> (World, Vec<MaterialId>) {
    let mut r = MaterialRegistry::new();
    let ids = (0..n).map(|i| r.register(&format!("m{i}"), MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap()).collect();
    (World::new(r), ids)
}

/// Derived data recomputed from scratch for every brick of the world (default merge mode).
pub fn truth(w: &World) -> BTreeMap<BrickKey, BrickMesh> {
    w.bricks().map(|(k, _)| (k, extract_world(w, k, Merge::default()).expect("stored bricks are non-empty"))).collect()
}

/// The pipeline's current snapshot must equal `truth(w)` exactly: same keys, same meshes.
pub fn check_snapshot(p: &Pipeline, w: &World) -> Result<(), String> {
    let snap = p.current();
    let expected = truth(w);
    let got: BTreeSet<BrickKey> = snap.keys().collect();
    let want: BTreeSet<BrickKey> = expected.keys().copied().collect();
    if got != want {
        let extra: Vec<_> = got.difference(&want).collect();
        let missing: Vec<_> = want.difference(&got).collect();
        return Err(format!("snapshot {:?} key mismatch: extra {extra:?}, missing {missing:?}", snap.id));
    }
    for (k, s) in &expected {
        let h = snap.handle(*k).expect("key present");
        match p.resolve(h) {
            None => return Err(format!("snapshot {:?} entry {k:?} dangles", snap.id)),
            Some(g) if g != s => return Err(format!("snapshot {:?} entry {k:?} is stale: got {g:?}, want {s:?}", snap.id)),
            Some(_) => {}
        }
    }
    Ok(())
}

/// Every entry of the current snapshot, resolved.
pub fn snapshot_values(p: &Pipeline) -> BTreeMap<BrickKey, BrickMesh> {
    let snap = p.current();
    snap.keys().map(|k| (k, p.resolve(snap.handle(k).unwrap()).expect("current entries resolve").clone())).collect()
}

/// Check after a publication that may be partial. Keys that are not awaiting publication must equal
/// the truth now; keys still awaiting must keep their value from the previous snapshot `prev`.
pub fn check_published(p: &Pipeline, w: &World, prev: &BTreeMap<BrickKey, BrickMesh>) -> Result<(), String> {
    let now = snapshot_values(p);
    let t = truth(w);
    let keys: BTreeSet<BrickKey> = now.keys().chain(t.keys()).chain(prev.keys()).copied().collect();
    for k in keys {
        let got = now.get(&k);
        if p.is_awaiting(k) {
            if got != prev.get(&k) {
                return Err(format!("snapshot {:?}: awaiting key {k:?} changed before its group was ready", p.current().id));
            }
        } else if got != t.get(&k) {
            return Err(format!("snapshot {:?}: key {k:?} is stale: got {got:?}, want {:?}", p.current().id, t.get(&k)));
        }
    }
    Ok(())
}

/// Dispatches and completes everything in dispatch order, then publishes. Returns the number of jobs run.
pub fn run_to_idle(p: &mut Pipeline, w: &World) -> usize {
    let mut n = 0;
    loop {
        let jobs = p.dispatch(w);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            n += 1;
            p.complete(w, j.run());
        }
    }
    p.try_publish(w);
    assert!(p.is_idle(), "pipeline did not drain");
    n
}

/// xorshift64*.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
