//! Phase 1D memory accounting (VALIDATION-AND-BUDGETS §2: "memory accounting that must exist from day one").
//!
//! Two ways to account, sharing the same categories and units (bytes):
//!
//! - **Reported** usage: a subsystem computes what it holds from its own structures
//!   (`World::memory`, `Pipeline::memory`) as a [`Report`]. `live` counts exact payload bytes;
//!   `reserved` is an upper bound that includes container capacity and node overhead. The bounds
//!   are checked against measured heap by tests using [`CountingAlloc`].
//! - **Granted** usage: a pool asks a [`Ledger`] before it allocates. The ledger enforces a
//!   [`Budget`], refuses deterministically when a grant would exceed it (state unchanged, refusal
//!   counted), and tracks live, reserved, high-water and a resettable transient peak. Phase 2 GPU
//!   pools and staging are meant to allocate through it.
//!
//! [`Tracker`] keeps high-water marks over successive reports, so repeated sampling (for example
//! once per frame) records peaks of reported usage too. Sampling cannot see a peak that starts and
//! ends inside one call; [`CountingAlloc`] can, and is used for that in tests and tools.

mod counting;
mod ledger;

use std::collections::BTreeMap;
use std::fmt::Write;

pub use counting::{CountingAlloc, HeapCounts};
pub use ledger::{device_budget, Budget, Grant, Ledger, LedgerError, DEVICE_HEADROOM_PERCENT, P001_DEVICE_CAP_BYTES};

/// What memory is for. Categories stay separate so that duplicated representations are visible
/// and budgeted together (PROPOSITION §1, §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    /// Brick occupancy masks, material arrays and brick headers.
    WorldBricks,
    /// Chunk tables and the chunk index.
    WorldHierarchy,
    /// The material registry: definitions and the name index.
    Materials,
    /// Derived products held in slots (current and retiring).
    DerivedData,
    /// Snapshot tables, counting every snapshot alive at once, not only the current one.
    DerivedSnapshots,
    /// Queues, groups, in-flight and staged results, retirement lists.
    DerivedBookkeeping,
    // Device categories: none is used before Phase 2.
    /// Device voxel/brick payload.
    GpuWorld,
    /// Extracted meshes and index buffers.
    GpuMesh,
    /// BLAS/TLAS storage.
    GpuAccel,
    /// Acceleration-structure build and update scratch.
    GpuAccelScratch,
    /// Textures and material tables.
    GpuMaterial,
    /// Frame targets and temporal histories.
    GpuTemporal,
    /// Upload/readback staging, kept separate from primary storage (PROPOSITION §3).
    Staging,
}

impl Category {
    pub const ALL: [Category; 13] = [
        Category::WorldBricks,
        Category::WorldHierarchy,
        Category::Materials,
        Category::DerivedData,
        Category::DerivedSnapshots,
        Category::DerivedBookkeeping,
        Category::GpuWorld,
        Category::GpuMesh,
        Category::GpuAccel,
        Category::GpuAccelScratch,
        Category::GpuMaterial,
        Category::GpuTemporal,
        Category::Staging,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::WorldBricks => "world_bricks",
            Category::WorldHierarchy => "world_hierarchy",
            Category::Materials => "materials",
            Category::DerivedData => "derived_data",
            Category::DerivedSnapshots => "derived_snapshots",
            Category::DerivedBookkeeping => "derived_bookkeeping",
            Category::GpuWorld => "gpu_world",
            Category::GpuMesh => "gpu_mesh",
            Category::GpuAccel => "gpu_accel",
            Category::GpuAccelScratch => "gpu_accel_scratch",
            Category::GpuMaterial => "gpu_material",
            Category::GpuTemporal => "gpu_temporal",
            Category::Staging => "staging",
        }
    }

    pub fn is_device(self) -> bool {
        self >= Category::GpuWorld && self != Category::Staging
    }
}

/// Bytes in use (`live`) and bytes held (`reserved`, which includes capacity slack and container
/// overhead). `live <= reserved` always holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub live: u64,
    pub reserved: u64,
}

impl Usage {
    pub const ZERO: Usage = Usage { live: 0, reserved: 0 };

    pub fn new(live: u64, reserved: u64) -> Self {
        debug_assert!(live <= reserved, "live {live} above reserved {reserved}");
        Self { live, reserved }
    }

    /// Payload stored exactly, with no slack: live and reserved are equal.
    pub fn exact(bytes: u64) -> Self {
        Self { live: bytes, reserved: bytes }
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;
    fn add(self, o: Usage) -> Usage {
        Usage { live: self.live + o.live, reserved: self.reserved + o.reserved }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, o: Usage) {
        *self = *self + o;
    }
}

/// Usage per category. Adding a category twice sums it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    rows: BTreeMap<Category, Usage>,
}

impl Report {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, c: Category, u: Usage) -> &mut Self {
        *self.rows.entry(c).or_default() += u;
        self
    }

    pub fn merge(&mut self, other: &Report) -> &mut Self {
        for (&c, &u) in &other.rows {
            self.add(c, u);
        }
        self
    }

    pub fn get(&self, c: Category) -> Usage {
        self.rows.get(&c).copied().unwrap_or_default()
    }

    pub fn rows(&self) -> impl Iterator<Item = (Category, Usage)> + '_ {
        self.rows.iter().map(|(&c, &u)| (c, u))
    }

    pub fn total(&self) -> Usage {
        self.rows.values().fold(Usage::ZERO, |a, &b| a + b)
    }

    /// Host and device totals, kept apart: they come out of different budgets.
    pub fn split_totals(&self) -> (Usage, Usage) {
        let mut host = Usage::ZERO;
        let mut device = Usage::ZERO;
        for (&c, &u) in &self.rows {
            if c.is_device() {
                device += u;
            } else {
                host += u;
            }
        }
        (host, device)
    }

    /// One JSON object: `{"world_bricks":{"live":..,"reserved":..},..,"total":{..}}`.
    pub fn to_json(&self) -> String {
        let mut s = String::from("{");
        for (c, u) in self.rows() {
            let _ = write!(s, "\"{}\":{{\"live\":{},\"reserved\":{}}},", c.name(), u.live, u.reserved);
        }
        let t = self.total();
        let _ = write!(s, "\"total\":{{\"live\":{},\"reserved\":{}}}}}", t.live, t.reserved);
        s
    }
}

/// High-water marks over successive reports (per category and in total).
#[derive(Clone, Debug, Default)]
pub struct Tracker {
    high: BTreeMap<Category, u64>,
    total_high: u64,
    samples: u64,
}

impl Tracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sample(&mut self, r: &Report) {
        for (c, u) in r.rows() {
            let h = self.high.entry(c).or_default();
            *h = (*h).max(u.reserved);
        }
        self.total_high = self.total_high.max(r.total().reserved);
        self.samples += 1;
    }

    /// Highest reserved bytes seen for `c`.
    pub fn high_water(&self, c: Category) -> u64 {
        self.high.get(&c).copied().unwrap_or(0)
    }

    /// Highest total reserved bytes seen in one sample. This can be below the sum of per-category
    /// high-water marks, because the categories need not peak together.
    pub fn total_high_water(&self) -> u64 {
        self.total_high
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }
}

/// Size accounting for standard containers. `live` is the payload (`len × element size`);
/// `reserved` is an upper bound on what the container holds on the heap.
pub mod containers {
    use super::Usage;

    pub fn vec<T>(v: &Vec<T>) -> Usage {
        let e = std::mem::size_of::<T>() as u64;
        Usage::new(v.len() as u64 * e, v.capacity() as u64 * e)
    }

    pub fn vec_deque<T>(v: &std::collections::VecDeque<T>) -> Usage {
        let e = std::mem::size_of::<T>() as u64;
        Usage::new(v.len() as u64 * e, v.capacity() as u64 * e)
    }

    /// Upper bound for a `BTreeMap` (or `BTreeSet`, with a zero-sized value) of `n` entries.
    ///
    /// Based on the std layout (B = 6: at most 11 entries per node, at least 5 in every node except
    /// the root, internal nodes add 12 child pointers, and each node carries a parent pointer, a
    /// parent index and a length). This is an implementation detail of std, not a guarantee: the
    /// bound is tested against measured heap in `world` and `derived`, and fails loudly if std changes.
    pub fn btree_bound(n: usize, key: usize, value: usize) -> Usage {
        let entry = (key + value) as u64;
        let live = n as u64 * entry;
        if n == 0 {
            return Usage::new(0, 0);
        }
        let n = n as u64;
        let leaves = n / 5 + 1;
        // Each internal level has at most a fifth as many nodes as the one below, plus a root.
        let mut internal = 0;
        let mut level = leaves;
        while level > 1 {
            level = level / 5 + 1;
            internal += level;
        }
        let header = 16u64;
        let align = |x: u64| x.div_ceil(8) * 8;
        let leaf = align(header + 11 * entry);
        let inner = leaf + 12 * 8;
        Usage::new(live, (leaves * leaf + internal * inner).max(live))
    }

    pub fn btree_map<K, V>(m: &std::collections::BTreeMap<K, V>) -> Usage {
        btree_bound(m.len(), std::mem::size_of::<K>(), std::mem::size_of::<V>())
    }

    pub fn btree_set<K>(s: &std::collections::BTreeSet<K>) -> Usage {
        btree_bound(s.len(), std::mem::size_of::<K>(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_sums_rows_and_splits_host_from_device() {
        let mut r = Report::new();
        r.add(Category::WorldBricks, Usage::new(10, 16)).add(Category::WorldBricks, Usage::exact(4)).add(Category::GpuMesh, Usage::new(100, 128));
        assert_eq!(r.get(Category::WorldBricks), Usage::new(14, 20));
        assert_eq!(r.total(), Usage::new(114, 148));
        assert_eq!(r.split_totals(), (Usage::new(14, 20), Usage::new(100, 128)));
        assert!(!Category::Staging.is_device(), "staging is host-visible memory");
        assert_eq!(r.to_json(), "{\"world_bricks\":{\"live\":14,\"reserved\":20},\"gpu_mesh\":{\"live\":100,\"reserved\":128},\"total\":{\"live\":114,\"reserved\":148}}");
    }

    #[test]
    fn category_names_are_unique() {
        let names: std::collections::BTreeSet<_> = Category::ALL.iter().map(|c| c.name()).collect();
        assert_eq!(names.len(), Category::ALL.len());
    }

    #[test]
    fn tracker_keeps_high_water_per_category_and_total() {
        let mut t = Tracker::new();
        let mut a = Report::new();
        a.add(Category::WorldBricks, Usage::exact(50)).add(Category::DerivedData, Usage::exact(10));
        let mut b = Report::new();
        b.add(Category::WorldBricks, Usage::exact(20)).add(Category::DerivedData, Usage::exact(35));
        t.sample(&a);
        t.sample(&b);
        assert_eq!((t.high_water(Category::WorldBricks), t.high_water(Category::DerivedData)), (50, 35));
        assert_eq!(t.total_high_water(), 60);
        assert_eq!(t.samples(), 2);
    }

    #[test]
    fn container_bounds_hold_live_below_reserved() {
        for n in [0, 1, 5, 11, 12, 100, 10_000] {
            let u = containers::btree_bound(n, 12, 40);
            assert!(u.live <= u.reserved, "{n}");
        }
        let mut v: Vec<u32> = Vec::with_capacity(10);
        v.push(1);
        assert_eq!(containers::vec(&v), Usage::new(4, 40));
    }
}
