//! Phase 1D: `World::memory` must bracket the heap the world really holds:
//! reported live <= measured <= reported reserved. One test in this binary, because the counting
//! allocator is process-wide.

use memory::{Category, CountingAlloc, Report};
use world::{scene, MaterialParams, MaterialRegistry, Transaction, VoxelCoord, World};

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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[test]
fn reported_world_memory_brackets_measured_heap() {
    // Street block: built inside the measured window; the generator's temporaries are gone by the end.
    let mut log = Log::new();
    let base = HEAP.counts().current;
    let (w, _) = scene::street_block();
    let measured = HEAP.counts().current - base;
    let r = w.memory();
    check(&mut log, "street_block", 0, &r, measured);
    // The bracket discriminates: without the bricks the reserved bound no longer covers the heap.
    assert!(r.total().reserved - r.get(Category::WorldBricks).reserved < measured, "control: dropping a category must break the bound");
    drop((w, r));
    assert_eq!(HEAP.counts().current, base, "the world freed everything it held");

    // Random worlds with many removals, so the chunk index is rebalanced and bricks are freed.
    for seed in 1..=4u64 {
        let base = HEAP.counts().current;
        let mut reg = MaterialRegistry::new();
        // A stack array, so the only heap in the window is the world's.
        let ids: [_; 20] = std::array::from_fn(|i| reg.register(&format!("material_with_a_longer_name_{i}"), MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap());
        let mut w = World::new(reg);
        let mut rng = Rng(seed * 6_364_136_223);
        for step in 0..3000 {
            let p = VoxelCoord::new((rng.next() % 400) as i32 - 200, (rng.next() % 64) as i32 - 32, (rng.next() % 400) as i32 - 200);
            let q = VoxelCoord::new(p.x + (rng.next() % 12) as i32, p.y + (rng.next() % 12) as i32, p.z + (rng.next() % 12) as i32);
            let m = if step % 3 == 0 { None } else { Some(ids[(rng.next() % 20) as usize]) };
            let mut tx = Transaction::new();
            tx.fill(p, q, m);
            w.apply(&tx).unwrap();
        }
        let measured = HEAP.counts().current - base;
        check(&mut log, "random seed", seed, &w.memory(), measured);
        drop(w);
        assert_eq!(HEAP.counts().current, base, "seed {seed}: freed everything");
    }
    log.print();
}
