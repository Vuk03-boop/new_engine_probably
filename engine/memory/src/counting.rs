//! A global allocator wrapper that counts heap bytes, for verifying reported usage in tests and
//! tools. It is not installed by any library; a binary or test opts in with
//! `#[global_allocator] static A: CountingAlloc = CountingAlloc::new();`.
//!
//! Counts are process-wide. Measure in a single-threaded section (one test per test binary) or
//! the deltas include other threads' allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct CountingAlloc {
    current: AtomicU64,
    peak: AtomicU64,
    allocations: AtomicU64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapCounts {
    /// Bytes currently allocated (requested sizes, as passed in the layouts).
    pub current: u64,
    /// Highest `current` since the last `reset_peak`.
    pub peak: u64,
    pub allocations: u64,
}

impl CountingAlloc {
    pub const fn new() -> Self {
        Self { current: AtomicU64::new(0), peak: AtomicU64::new(0), allocations: AtomicU64::new(0) }
    }

    pub fn counts(&self) -> HeapCounts {
        HeapCounts { current: self.current.load(Ordering::SeqCst), peak: self.peak.load(Ordering::SeqCst), allocations: self.allocations.load(Ordering::SeqCst) }
    }

    /// Starts a new peak window at the current usage.
    pub fn reset_peak(&self) {
        self.peak.store(self.current.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    fn grow(&self, n: u64) {
        let now = self.current.fetch_add(n, Ordering::SeqCst) + n;
        self.peak.fetch_max(now, Ordering::SeqCst);
    }

    fn shrink(&self, n: u64) {
        self.current.fetch_sub(n, Ordering::SeqCst);
    }
}

impl Default for CountingAlloc {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            self.grow(layout.size() as u64);
            self.allocations.fetch_add(1, Ordering::SeqCst);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            self.grow(layout.size() as u64);
            self.allocations.fetch_add(1, Ordering::SeqCst);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        self.shrink(layout.size() as u64);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            self.shrink(layout.size() as u64);
            self.grow(new_size as u64);
        }
        p
    }
}
