//! Seeded random interleavings of edits, delayed notifications, dispatches, delayed out-of-order
//! completions, readers and publications, and a real multi-threaded run. Invariants are checked
//! after every step.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{check_published, check_snapshot, snapshot_values, world_with, Rng};
use derived::{affected_keys, Config, Faults, JobResult, Pipeline, Published, ReaderToken, SnapshotId, Stats, BrickMesh};
use world::{BrickKey, VoxelCoord, World};

const EXTENT: i32 = 24; // voxels [-24, 24) per axis: 6 bricks, crossing chunk edges at 0 and ±32

fn random_voxel(rng: &mut Rng) -> VoxelCoord {
    // Half of the coordinates are forced onto brick faces (local 0 or 7) to stress boundary edits.
    let mut c = || {
        let x = rng.below(2 * EXTENT as u64) as i32 - EXTENT;
        if rng.below(2) == 0 { x.div_euclid(8) * 8 + if rng.below(2) == 0 { 0 } else { 7 } } else { x }
    };
    VoxelCoord::new(c(), c(), c())
}

struct Outcome {
    stats: Stats,
    publishes_checked: usize,
    reader_reads_checked: usize,
    batches_checked: usize,
}

/// Tracks, for every edit batch, whether its keys have been published since. A batch must become
/// visible all at once: after any publication, a batch is fully published or not at all.
#[derive(Default)]
struct BatchTracker {
    open: Vec<BTreeMap<BrickKey, Option<SnapshotId>>>,
    closed: usize,
}

impl BatchTracker {
    fn add(&mut self, keys: impl IntoIterator<Item = BrickKey>) {
        let b: BTreeMap<_, _> = keys.into_iter().map(|k| (k, None)).collect();
        if !b.is_empty() {
            self.open.push(b);
        }
    }

    fn on_publish(&mut self, report: &Published) -> Result<(), String> {
        let keys: BTreeSet<BrickKey> = report.groups.iter().flatten().copied().collect();
        for b in &mut self.open {
            for (k, first) in b.iter_mut() {
                if first.is_none() && keys.contains(k) {
                    *first = Some(report.id);
                }
            }
            let done = b.values().filter(|f| f.is_some()).count();
            if done > 0 && done < b.len() {
                let missing: Vec<_> = b.iter().filter(|(_, f)| f.is_none()).map(|(k, _)| *k).collect();
                return Err(format!(
                    "snapshot {:?} published an edit batch half-way: {done} of {} keys; first unpublished {:?}",
                    report.id,
                    b.len(),
                    &missing[..missing.len().min(3)]
                ));
            }
        }
        let before = self.open.len();
        self.open.retain(|b| b.values().any(Option::is_none));
        self.closed += before - self.open.len();
        Ok(())
    }
}

/// Everything the harness tracks besides the pipeline and the world.
struct Harness {
    batches: BatchTracker,
    /// Edits applied to the world whose notification has not been delivered yet.
    unnotified: Vec<VoxelCoord>,
    /// Values of each snapshot at publication, for checking readers later.
    published: BTreeMap<SnapshotId, BTreeMap<BrickKey, BrickMesh>>,
    prev_values: BTreeMap<BrickKey, BrickMesh>,
    publishes_checked: usize,
}

impl Harness {
    fn notify(&mut self, p: &mut Pipeline) {
        if !self.unnotified.is_empty() {
            self.batches.add(self.unnotified.iter().flat_map(|&v| affected_keys(v)).collect::<BTreeSet<_>>());
            p.notify_edits(&self.unnotified);
            self.unnotified.clear();
        }
    }

    /// Publication happens at a frame boundary, after pending notifications are delivered.
    fn publish(&mut self, p: &mut Pipeline, w: &World) -> Result<(), String> {
        self.notify(p);
        if let Some(report) = p.try_publish(w) {
            check_published(p, w, &self.prev_values)?;
            self.batches.on_publish(&report)?;
            self.prev_values = snapshot_values(p);
            self.published.insert(report.id, self.prev_values.clone());
            self.publishes_checked += 1;
        }
        Ok(())
    }
}

fn stress(seed: u64, faults: Faults, steps: usize) -> Result<Outcome, String> {
    let mut rng = Rng(seed);
    let (mut w, m) = world_with(4);
    for _ in 0..3000 {
        let v = random_voxel(&mut rng);
        w.set(v, Some(m[rng.below(4) as usize])).unwrap();
    }
    let mut p = Pipeline::new(Config { queue_capacity: 8, max_in_flight: 6, faults, ..Config::default() });
    let mut h = Harness {
        batches: BatchTracker::default(),
        unnotified: Vec::new(),
        published: BTreeMap::from([(p.current().id, BTreeMap::new())]),
        prev_values: BTreeMap::new(),
        publishes_checked: 0,
    };
    h.batches.add(w.bricks().map(|(k, _)| k));
    p.mark_all(&w);

    let mut held: Vec<JobResult> = Vec::new();
    let mut readers: Vec<ReaderToken> = Vec::new();
    let mut reader_reads_checked = 0;

    for step in 0..steps {
        match rng.below(100) {
            // Edit rate is set so processing keeps up on average (with bursts); sustained overload is
            // tested separately in `scenarios::sustained_edits_defer_publication_and_recover`.
            0..=5 => {
                for _ in 0..1 + rng.below(4) {
                    let v = random_voxel(&mut rng);
                    let mat = if rng.below(3) == 0 { None } else { Some(m[rng.below(4) as usize]) };
                    if w.set(v, mat).unwrap() {
                        h.unnotified.push(v);
                    }
                }
            }
            6..=11 => h.notify(&mut p),
            12..=39 => {
                // Jobs run immediately against their captured copies; completion is delayed arbitrarily.
                for j in p.dispatch(&w) {
                    held.push(j.run());
                }
            }
            40..=79 => {
                if !held.is_empty() {
                    let r = held.swap_remove(rng.below(held.len() as u64) as usize);
                    p.complete(&w, r);
                }
            }
            80..=84 => readers.push(p.acquire()),
            85..=89 => {
                if !readers.is_empty() {
                    let r = readers.swap_remove(rng.below(readers.len() as u64) as usize);
                    p.release(r).map_err(|e| format!("release: {e:?}"))?;
                }
            }
            _ => h.publish(&mut p, &w).map_err(|e| format!("step {step}: {e}"))?,
        }
        p.check_live_handles().map_err(|e| format!("step {step}: {e}"))?;
        // Every reader sees exactly the data of the snapshot it acquired.
        for r in &readers {
            let expect = &h.published[&r.snapshot];
            for _ in 0..2 {
                let k = random_voxel(&mut rng).split().0;
                let got = p.read(r, k).map_err(|e| format!("step {step}: reader {:?}: {e:?}", r.id))?;
                if got != expect.get(&k) {
                    return Err(format!("step {step}: reader {:?} on {:?} saw {got:?}, want {:?}", r.id, r.snapshot, expect.get(&k)));
                }
                reader_reads_checked += 1;
            }
        }
    }

    // Drain: finish everything, release readers, deliver notifications and publish until idle.
    for r in held.drain(..) {
        p.complete(&w, r);
    }
    for r in readers.drain(..) {
        p.release(r).map_err(|e| format!("release: {e:?}"))?;
    }
    for _ in 0..1000 {
        for j in p.dispatch(&w) {
            p.complete(&w, j.run());
        }
        h.publish(&mut p, &w)?;
        if p.is_idle() {
            break;
        }
    }
    if !p.is_idle() {
        return Err("pipeline did not drain".into());
    }
    check_snapshot(&p, &w)?;
    if !h.batches.open.is_empty() {
        return Err(format!("{} edit batches never published", h.batches.open.len()));
    }
    if p.pool_stats().live != p.current().len() || p.retiring_len() != 0 || p.live_snapshots() != 1 {
        return Err(format!("leak: pool live {}, entries {}, retiring {}, snapshots {}", p.pool_stats().live, p.current().len(), p.retiring_len(), p.live_snapshots()));
    }
    Ok(Outcome { stats: p.stats(), publishes_checked: h.publishes_checked, reader_reads_checked, batches_checked: h.batches.closed })
}

const SEEDS: [u64; 4] = [1, 0xDEAD_BEEF, 42, 0x1234_5678_9ABC];

#[test]
fn random_interleavings_keep_every_invariant() {
    for seed in SEEDS {
        let o = stress(seed, Faults::default(), 6000).unwrap_or_else(|e| panic!("seed {seed:#x}: {e}"));
        let s = o.stats;
        println!(
            "seed {seed:#x}: {s:?}, publishes checked {}, reader reads checked {}, batches checked {}",
            o.publishes_checked, o.reader_reads_checked, o.batches_checked
        );
        // The run must exercise every mechanism it claims to test.
        assert!(s.rejected_cancelled > 0 && s.rejected_stale > 0 && s.accepted_by_content > 0, "seed {seed:#x}: {s:?}");
        assert!(s.overflow_deferred > 0 && s.coalesced > 0 && s.publish_deferred > 0 && s.groups_merged > 0, "seed {seed:#x}: {s:?}");
        assert!(s.freed > 0 && o.publishes_checked >= 20 && o.reader_reads_checked > 1000 && o.batches_checked >= 100, "seed {seed:#x}");
    }
}

#[test]
fn stress_is_deterministic() {
    let a = stress(7, Faults::default(), 3000).unwrap().stats;
    let b = stress(7, Faults::default(), 3000).unwrap().stats;
    assert_eq!(a, b);
}

fn detected(faults: Faults) -> Vec<String> {
    SEEDS.iter().filter_map(|&s| stress(s, faults, 6000).err()).collect()
}

#[test]
fn negative_controls_each_guard_is_needed() {
    let cases = [
        ("skip_neighbour_dirty", Faults { skip_neighbour_dirty: true, ..Faults::default() }),
        ("retire_without_readers", Faults { retire_without_readers: true, ..Faults::default() }),
        ("skip_cancel + skip_stale_check", Faults { skip_cancel: true, skip_stale_check: true, ..Faults::default() }),
        ("publish_partial_groups", Faults { publish_partial_groups: true, ..Faults::default() }),
    ];
    for (name, f) in cases {
        let errs = detected(f);
        println!("{name}: detected on {} of {} seeds; first: {:?}", errs.len(), SEEDS.len(), errs.first());
        assert!(!errs.is_empty(), "{name}: the invariants never failed, so they cannot detect this fault");
    }
}

#[test]
fn cancellation_and_stale_check_are_independent_guards() {
    // With either guard alone disabled, the other still keeps every invariant.
    for (name, f) in [
        ("skip_cancel", Faults { skip_cancel: true, ..Faults::default() }),
        ("skip_stale_check", Faults { skip_stale_check: true, ..Faults::default() }),
    ] {
        for seed in SEEDS {
            let o = stress(seed, f, 6000).unwrap_or_else(|e| panic!("{name}, seed {seed:#x}: {e}"));
            if name == "skip_cancel" {
                assert_eq!(o.stats.rejected_cancelled, 0);
                assert!(o.stats.rejected_stale > 0, "the stale check must be doing the cancel's work");
            }
        }
    }
}

#[test]
fn threaded_workers_with_random_delays() {
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    let mut rng = Rng(99);
    let (mut w, m) = world_with(4);
    for _ in 0..2000 {
        let v = random_voxel(&mut rng);
        w.set(v, Some(m[rng.below(4) as usize])).unwrap();
    }
    let mut p = Pipeline::new(Config { queue_capacity: 16, max_in_flight: 8, ..Config::default() });
    p.mark_all(&w);

    let (job_tx, job_rx) = mpsc::channel::<derived::Job>();
    let (res_tx, res_rx) = mpsc::channel::<JobResult>();
    let job_rx = Arc::new(Mutex::new(job_rx));
    let workers: Vec<_> = (0..3)
        .map(|wid| {
            let rx = Arc::clone(&job_rx);
            let tx = res_tx.clone();
            std::thread::spawn(move || {
                let mut r = Rng(1000 + wid);
                loop {
                    let job = match rx.lock().unwrap().recv() {
                        Ok(j) => j,
                        Err(_) => break,
                    };
                    std::thread::sleep(Duration::from_micros(r.below(1500)));
                    if tx.send(job.run()).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(res_tx);

    let mut checked = 0;
    let mut prev = snapshot_values(&p);
    for _ in 0..1500 {
        if rng.below(3) == 0 {
            let v = random_voxel(&mut rng);
            let mat = if rng.below(3) == 0 { None } else { Some(m[rng.below(4) as usize]) };
            if w.set(v, mat).unwrap() {
                p.notify_edits(&[v]);
            }
        }
        for j in p.dispatch(&w) {
            job_tx.send(j).unwrap();
        }
        while let Ok(r) = res_rx.try_recv() {
            p.complete(&w, r);
        }
        if p.try_publish(&w).is_some() {
            check_published(&p, &w, &prev).unwrap();
            prev = snapshot_values(&p);
            checked += 1;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    // Drain with the world frozen.
    while !p.is_idle() {
        for j in p.dispatch(&w) {
            job_tx.send(j).unwrap();
        }
        if p.in_flight_len() > 0 {
            let r = res_rx.recv_timeout(Duration::from_secs(10)).expect("worker results stopped arriving");
            p.complete(&w, r);
        }
        if p.try_publish(&w).is_some() {
            check_published(&p, &w, &prev).unwrap();
            prev = snapshot_values(&p);
            checked += 1;
        }
    }
    drop(job_tx);
    for h in workers {
        h.join().unwrap();
    }
    check_snapshot(&p, &w).unwrap();
    let s = p.stats();
    println!("threaded: {s:?}, publishes checked {checked}");
    assert!(checked >= 5 && s.accepted > 0);
    assert_eq!(p.pool_stats().live, p.current().len());
}
