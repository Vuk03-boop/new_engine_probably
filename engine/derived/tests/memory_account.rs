//! Phase 1D: `Pipeline::memory` must bracket the heap the pipeline really holds, counting every
//! simultaneous snapshot. One test in this binary, because the counting allocator is process-wide.

mod common;

use common::{check_snapshot, run_to_idle};
use derived::{Config, Pipeline};
use memory::{containers, Category, CountingAlloc, Report};
use world::{scene, Transaction, VoxelCoord};

#[global_allocator]
static HEAP: CountingAlloc = CountingAlloc::new();

/// Results are kept on the stack and printed after the last measurement: printing allocates
/// (the test harness captures output on the heap), which would pollute the measured windows.
struct Log {
    rows: [(&'static str, u64, u64, u64, u64); 8],
    n: usize,
}

impl Log {
    fn new() -> Self {
        Self { rows: [("", 0, 0, 0, 0); 8], n: 0 }
    }
    fn print(&self) {
        for &(label, idx, live, measured, reserved) in &self.rows[..self.n] {
            eprintln!("{label} {idx}: live {live} <= measured {measured} <= reserved {reserved} ({:.3} of reserved)", measured as f64 / reserved as f64);
        }
    }
}

fn check(log: &mut Log, label: &'static str, idx: u64, r: &Report, measured: u64) {
    let t = r.total();
    log.rows[log.n] = (label, idx, t.live, measured, t.reserved);
    log.n += 1;
    assert!(t.live <= measured, "{label} {idx}: reported live {} above measured {measured}", t.live);
    assert!(measured <= t.reserved, "{label} {idx}: measured {measured} above reported reserved {}", t.reserved);
}

#[test]
fn reported_pipeline_memory_brackets_measured_heap() {
    // Both worlds and the edit list are built before the measured window: only the pipeline is in it.
    let (w1, _) = scene::street_block();
    let mut w2 = w1.clone();
    let glass = w1.materials().id_of("glass").unwrap();
    let mut tx = Transaction::new();
    tx.fill(VoxelCoord::new(-40, 0, -40), VoxelCoord::new(40, 24, 40), Some(glass));
    let changed = w2.apply(&tx).unwrap().changed;

    let mut log = Log::new();
    let base = HEAP.counts().current;
    let mut p = Pipeline::new(Config::default());
    p.mark_all(&w1);

    // Jobs in flight are held by the caller: count them next to the pipeline's own report.
    let jobs = p.dispatch(&w1);
    let mut with_jobs = p.memory();
    with_jobs.add(Category::DerivedBookkeeping, containers::vec(&jobs));
    for j in &jobs {
        with_jobs.add(Category::DerivedBookkeeping, j.heap());
    }
    check(&mut log, "jobs in flight", 0, &with_jobs, HEAP.counts().current - base);
    for j in jobs {
        p.complete(&w1, j.run());
    }
    drop(with_jobs);

    run_to_idle(&mut p, &w1);
    check_snapshot(&p, &w1).unwrap();
    let one = p.memory();
    check(&mut log, "one snapshot", 0, &one, HEAP.counts().current - base);

    // A reader keeps the old snapshot while an edit publishes a new one: both are counted.
    let reader = p.acquire();
    p.notify_edits(&changed);
    HEAP.reset_peak();
    let before_publish = HEAP.counts().current;
    run_to_idle(&mut p, &w2);
    let transient = HEAP.counts().peak - before_publish;
    check_snapshot(&p, &w2).unwrap();
    let two = p.memory();
    let measured = HEAP.counts().current - base;
    check(&mut log, "two snapshots", 0, &two, measured);
    let snaps = |r: &Report| r.get(Category::DerivedSnapshots);
    // 2E sharing: the old snapshot's replaced shards are still counted, its shared shards only once.
    // (1C cloned the table, and this assertion was "two > 1.5 x one".)
    assert!(snaps(&two).live > snaps(&one).live, "the old snapshot is still counted: {:?} vs {:?}", snaps(&two), snaps(&one));
    assert!(snaps(&two).live < snaps(&one).live * 3 / 2, "shared shards are counted once: {:?} vs {:?}", snaps(&two), snaps(&one));
    assert!(two.total().reserved - snaps(&two).reserved < measured, "control: dropping the snapshot category must break the bound");

    // Releasing the reader retires the old snapshot and its replaced entries at the next publication.
    p.release(reader).unwrap();
    p.try_publish(&w2);
    let after = p.memory();
    check(&mut log, "after release", 0, &after, HEAP.counts().current - base);
    assert!(snaps(&after).live < snaps(&two).live, "old snapshot freed");
    assert!(after.get(Category::DerivedData).live < two.get(Category::DerivedData).live, "replaced entries freed");
    let (one_s, two_s) = (snaps(&one), snaps(&two));
    drop((one, two, after));
    drop(p);
    assert_eq!(HEAP.counts().current, base, "the pipeline freed everything it held");
    log.print();
    eprintln!("publish transient peak above the pre-publish heap: {transient} bytes; snapshot tables: one {one_s:?}, two {two_s:?}");
}
