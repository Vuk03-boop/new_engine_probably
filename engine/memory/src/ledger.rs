//! Budgeted grants: a pool asks before it allocates, and releases when it frees.
//!
//! Contract:
//! - `reserve` succeeds only if the category's and the total reserved bytes stay within the
//!   budget. A refusal changes nothing except the refusal counter, and it depends only on the
//!   ledger state, so the fallback is deterministic.
//! - A [`Grant`] is not `Clone`; releasing it consumes it, so a double release cannot be written.
//!   Grants carry their ledger's id, and a foreign grant is rejected.
//! - `live` within a grant can move between 0 and its reserved size (a pool filling its capacity).
//! - High-water marks never decrease. The transient peak is the high-water since the last
//!   `reset_peak`, for measuring one operation such as a snapshot swap.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{Category, Report, Usage};

static NEXT_LEDGER: AtomicU64 = AtomicU64::new(1);

/// Byte limits on reserved memory. `None` means unlimited.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    pub total: Option<u64>,
    pub per_category: BTreeMap<Category, u64>,
}

impl Budget {
    pub fn unlimited() -> Self {
        Self::default()
    }

    pub fn with_total(mut self, bytes: u64) -> Self {
        self.total = Some(bytes);
        self
    }

    pub fn with_limit(mut self, c: Category, bytes: u64) -> Self {
        self.per_category.insert(c, bytes);
        self
    }
}

/// P-001: at most 3.5 GB of device memory (decimal gigabytes, the stricter reading).
pub const P001_DEVICE_CAP_BYTES: u64 = 3_500_000_000;
/// Headroom kept below the ceiling for the driver, other processes and fragmentation.
pub const DEVICE_HEADROOM_PERCENT: u64 = 10;

/// The device budget (ADR-0003): the smaller of the P-001 cap and the driver-reported heap budget
/// (`VK_EXT_memory_budget`; `None` if unavailable), minus the headroom. It limits the total; no
/// per-category split is imposed until measurements call for one.
pub fn device_budget(driver_heap_budget: Option<u64>) -> Budget {
    let ceiling = driver_heap_budget.map_or(P001_DEVICE_CAP_BYTES, |d| d.min(P001_DEVICE_CAP_BYTES));
    Budget::unlimited().with_total(ceiling - ceiling * DEVICE_HEADROOM_PERCENT / 100)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// Granting would exceed a limit. Nothing was reserved.
    OverBudget { category: Category, requested: u64, category_reserved: u64, total_reserved: u64, limit: u64, limit_is_total: bool },
    /// The grant belongs to another ledger.
    ForeignGrant,
    /// `live` above the grant's reserved size.
    LiveAboveReserved { live: u64, reserved: u64 },
}

/// A reservation. Release it with [`Ledger::release`].
#[derive(Debug, PartialEq, Eq)]
pub struct Grant {
    ledger: u64,
    id: u64,
    category: Category,
    reserved: u64,
}

impl Grant {
    pub fn category(&self) -> Category {
        self.category
    }
    pub fn reserved(&self) -> u64 {
        self.reserved
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Account {
    pub usage: Usage,
    pub high_water: u64,
    pub peak: u64,
    pub grants: u64,
    pub releases: u64,
    pub refusals: u64,
}

#[derive(Debug)]
pub struct Ledger {
    id: u64,
    budget: Budget,
    accounts: BTreeMap<Category, Account>,
    /// Outstanding grants: id → live bytes.
    live: BTreeMap<u64, u64>,
    next_grant: u64,
    total_reserved: u64,
    total_high_water: u64,
    total_peak: u64,
}

impl Ledger {
    pub fn new(budget: Budget) -> Self {
        Self {
            id: NEXT_LEDGER.fetch_add(1, Ordering::Relaxed),
            budget,
            accounts: BTreeMap::new(),
            live: BTreeMap::new(),
            next_grant: 0,
            total_reserved: 0,
            total_high_water: 0,
            total_peak: 0,
        }
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// Reserves `bytes` in `c`, with `live` starting at 0.
    pub fn reserve(&mut self, c: Category, bytes: u64) -> Result<Grant, LedgerError> {
        let acc = self.accounts.entry(c).or_default();
        let cat_after = acc.usage.reserved + bytes;
        let total_after = self.total_reserved + bytes;
        let refuse = |limit, limit_is_total| LedgerError::OverBudget {
            category: c,
            requested: bytes,
            category_reserved: acc.usage.reserved,
            total_reserved: self.total_reserved,
            limit,
            limit_is_total,
        };
        let err = match (self.budget.per_category.get(&c), self.budget.total) {
            (Some(&l), _) if cat_after > l => Some(refuse(l, false)),
            (_, Some(l)) if total_after > l => Some(refuse(l, true)),
            _ => None,
        };
        if let Some(e) = err {
            acc.refusals += 1;
            return Err(e);
        }
        acc.usage.reserved = cat_after;
        acc.high_water = acc.high_water.max(cat_after);
        acc.peak = acc.peak.max(cat_after);
        acc.grants += 1;
        self.total_reserved = total_after;
        self.total_high_water = self.total_high_water.max(total_after);
        self.total_peak = self.total_peak.max(total_after);
        let id = self.next_grant;
        self.next_grant += 1;
        self.live.insert(id, 0);
        Ok(Grant { ledger: self.id, id, category: c, reserved: bytes })
    }

    /// Sets how many of the grant's bytes are in use.
    pub fn set_live(&mut self, g: &Grant, live: u64) -> Result<(), LedgerError> {
        if g.ledger != self.id {
            return Err(LedgerError::ForeignGrant);
        }
        if live > g.reserved {
            return Err(LedgerError::LiveAboveReserved { live, reserved: g.reserved });
        }
        let old = self.live.insert(g.id, live).expect("outstanding grant");
        let acc = self.accounts.get_mut(&g.category).expect("account exists");
        acc.usage.live = acc.usage.live - old + live;
        Ok(())
    }

    pub fn release(&mut self, g: Grant) -> Result<(), LedgerError> {
        if g.ledger != self.id {
            return Err(LedgerError::ForeignGrant);
        }
        let live = self.live.remove(&g.id).expect("outstanding grant");
        let acc = self.accounts.get_mut(&g.category).expect("account exists");
        acc.usage.live -= live;
        acc.usage.reserved -= g.reserved;
        acc.releases += 1;
        self.total_reserved -= g.reserved;
        Ok(())
    }

    pub fn account(&self, c: Category) -> Account {
        self.accounts.get(&c).copied().unwrap_or_default()
    }

    pub fn total_reserved(&self) -> u64 {
        self.total_reserved
    }

    pub fn total_high_water(&self) -> u64 {
        self.total_high_water
    }

    /// Highest total reserved since the last [`Ledger::reset_peak`].
    pub fn total_peak(&self) -> u64 {
        self.total_peak
    }

    /// Starts a new transient-peak window at the current usage.
    pub fn reset_peak(&mut self) {
        for acc in self.accounts.values_mut() {
            acc.peak = acc.usage.reserved;
        }
        self.total_peak = self.total_reserved;
    }

    /// Grants not yet released. Non-zero at teardown means a leak.
    pub fn outstanding(&self) -> usize {
        self.live.len()
    }

    pub fn report(&self) -> Report {
        let mut r = Report::new();
        for (&c, acc) in &self.accounts {
            if acc.grants > 0 {
                r.add(c, acc.usage);
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grants_track_live_reserved_high_water_and_peak() {
        let mut l = Ledger::new(Budget::unlimited());
        let a = l.reserve(Category::GpuMesh, 1000).unwrap();
        let b = l.reserve(Category::GpuMesh, 500).unwrap();
        l.set_live(&a, 700).unwrap();
        assert_eq!(l.account(Category::GpuMesh).usage, Usage::new(700, 1500));
        l.release(b).unwrap();
        assert_eq!(l.account(Category::GpuMesh).usage, Usage::new(700, 1000));
        assert_eq!(l.account(Category::GpuMesh).high_water, 1500);
        l.reset_peak();
        assert_eq!(l.total_peak(), 1000);
        let c = l.reserve(Category::Staging, 300).unwrap();
        l.release(c).unwrap();
        assert_eq!((l.total_peak(), l.total_high_water(), l.total_reserved()), (1300, 1500, 1000));
        l.release(a).unwrap();
        assert_eq!((l.outstanding(), l.total_reserved()), (0, 0));
        assert_eq!(l.account(Category::GpuMesh).usage, Usage::ZERO);
    }

    #[test]
    fn refusal_is_deterministic_and_changes_nothing_but_the_counter() {
        let budget = Budget::unlimited().with_total(1000).with_limit(Category::GpuAccel, 600);
        let mut l = Ledger::new(budget);
        let a = l.reserve(Category::GpuAccel, 400).unwrap();
        let before = (l.account(Category::GpuAccel).usage, l.total_reserved(), l.outstanding());
        // Category limit.
        let e = l.reserve(Category::GpuAccel, 201).unwrap_err();
        assert!(matches!(e, LedgerError::OverBudget { limit: 600, limit_is_total: false, category_reserved: 400, .. }));
        // Total limit.
        let e = l.reserve(Category::GpuMesh, 601).unwrap_err();
        assert!(matches!(e, LedgerError::OverBudget { limit: 1000, limit_is_total: true, total_reserved: 400, .. }));
        assert_eq!((l.account(Category::GpuAccel).usage, l.total_reserved(), l.outstanding()), before);
        assert_eq!((l.account(Category::GpuAccel).refusals, l.account(Category::GpuMesh).refusals), (1, 1));
        // Exactly at the limit is allowed; the same request is refused again after that.
        let b = l.reserve(Category::GpuAccel, 200).unwrap();
        assert!(l.reserve(Category::GpuMesh, 401).is_err());
        let c = l.reserve(Category::GpuMesh, 400).unwrap();
        assert_eq!(l.total_reserved(), 1000);
        for g in [a, b, c] {
            l.release(g).unwrap();
        }
        // After releasing, the earlier refused request fits.
        assert!(l.reserve(Category::GpuMesh, 601).is_ok());
    }

    #[test]
    fn foreign_grants_and_overfull_live_are_rejected() {
        let mut a = Ledger::new(Budget::unlimited());
        let mut b = Ledger::new(Budget::unlimited());
        let g = a.reserve(Category::Staging, 10).unwrap();
        assert_eq!(b.set_live(&g, 1), Err(LedgerError::ForeignGrant));
        assert_eq!(a.set_live(&g, 11), Err(LedgerError::LiveAboveReserved { live: 11, reserved: 10 }));
        let g2 = b.reserve(Category::Staging, 5).unwrap();
        assert_eq!(a.release(g2), Err(LedgerError::ForeignGrant));
        assert_eq!(b.outstanding(), 1, "a rejected release leaves the grant outstanding in its own ledger");
        a.release(g).unwrap();
        assert_eq!(a.outstanding(), 0);
    }

    #[test]
    fn device_budget_takes_the_smaller_ceiling_minus_headroom() {
        // Phase 0 measured a 3,367.7 MiB driver budget on the RTX 3050 Laptop, about 3.53e9 bytes:
        // just above the P-001 cap, so the cap governs there.
        let laptop = (3367.7 * 1024.0 * 1024.0) as u64;
        assert!(laptop > P001_DEVICE_CAP_BYTES);
        assert_eq!(device_budget(Some(laptop)).total, Some(3_150_000_000));
        // A smaller driver budget governs when it is below the cap.
        let small = 2_000_000_000;
        assert_eq!(device_budget(Some(small)).total, Some(1_800_000_000));
        // A larger GPU is still capped at 3.5 GB; an unknown budget falls back to the cap.
        assert_eq!(device_budget(Some(24 << 30)).total, Some(3_150_000_000));
        assert_eq!(device_budget(None).total, Some(3_150_000_000));
        // The ledger refuses one byte over it.
        let mut l = Ledger::new(device_budget(None));
        let g = l.reserve(Category::GpuMesh, 3_150_000_000).unwrap();
        assert!(l.reserve(Category::GpuAccel, 1).is_err());
        l.release(g).unwrap();
    }

    #[test]
    fn report_lists_used_categories() {
        let mut l = Ledger::new(Budget::unlimited());
        let g = l.reserve(Category::GpuTemporal, 64).unwrap();
        l.set_live(&g, 32).unwrap();
        let r = l.report();
        assert_eq!(r.get(Category::GpuTemporal), Usage::new(32, 64));
        assert_eq!(r.rows().count(), 1);
    }
}
