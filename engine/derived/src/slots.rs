//! Generational slot pool: the "allocation generation" identity (PROPOSITION §2).
//!
//! A [`Handle`] names a slot and the generation at which it was allocated. Freeing a slot bumps its
//! generation, so every handle to the old occupant stops resolving before the slot is reused.
//! A slot whose generation would wrap is retired permanently instead of being reused.

/// Physical allocation identity. Only valid for the pool that issued it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handle {
    index: u32,
    generation: u32,
}

impl Handle {
    pub fn index(self) -> u32 {
        self.index
    }
    pub fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PoolStats {
    pub live: usize,
    pub slots: usize,
    pub allocations: u64,
    pub frees: u64,
    /// Slots taken out of service because their generation counter would wrap.
    pub exhausted_slots: usize,
}

#[derive(Debug)]
pub struct SlotPool<T> {
    slots: Vec<Slot<T>>,
    /// Free slot indices, reused lowest-first so allocation order is deterministic.
    free: std::collections::BTreeSet<u32>,
    stats: PoolStats,
}

impl<T> Default for SlotPool<T> {
    fn default() -> Self {
        Self { slots: Vec::new(), free: Default::default(), stats: PoolStats::default() }
    }
}

impl<T> SlotPool<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Heap held by the pool (Phase 1D). `live` counts occupied slots; `reserved` counts the slot
    /// vector's capacity and the free list. `heap` gives each value's own heap.
    pub fn memory(&self, heap: impl Fn(&T) -> memory::Usage) -> memory::Usage {
        let slot = std::mem::size_of::<Slot<T>>() as u64;
        let occupied = self.slots.iter().filter(|s| s.value.is_some()).count() as u64;
        let mut u = memory::Usage::new(occupied * slot, self.slots.capacity() as u64 * slot);
        u += memory::Usage::new(0, memory::containers::btree_set(&self.free).reserved);
        for v in self.slots.iter().filter_map(|s| s.value.as_ref()) {
            u += heap(v);
        }
        u
    }

    pub fn alloc(&mut self, value: T) -> Handle {
        self.stats.allocations += 1;
        self.stats.live += 1;
        if let Some(index) = self.free.pop_first() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            return Handle { index, generation: slot.generation };
        }
        let index = u32::try_from(self.slots.len()).expect("slot pool exceeds u32 slots");
        self.slots.push(Slot { generation: 0, value: Some(value) });
        self.stats.slots = self.slots.len();
        Handle { index, generation: 0 }
    }

    /// Resolves a handle. `None` if the slot was freed or reused since the handle was issued.
    pub fn get(&self, h: Handle) -> Option<&T> {
        let slot = self.slots.get(h.index as usize)?;
        if slot.generation == h.generation { slot.value.as_ref() } else { None }
    }

    /// Frees the slot and invalidates every handle to it. Returns the value, or `None` for a stale handle
    /// (a double free is therefore harmless and detectable).
    pub fn free(&mut self, h: Handle) -> Option<T> {
        let slot = self.slots.get_mut(h.index as usize)?;
        if slot.generation != h.generation {
            return None;
        }
        let value = slot.value.take()?;
        self.stats.frees += 1;
        self.stats.live -= 1;
        match slot.generation.checked_add(1) {
            Some(g) => {
                slot.generation = g;
                self.free.insert(h.index);
            }
            None => self.stats.exhausted_slots += 1,
        }
        Some(value)
    }

    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    #[cfg(test)]
    fn force_generation(&mut self, index: u32, generation: u32) {
        self.slots[index as usize].generation = generation;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freed_handle_stops_resolving_and_reuse_gets_a_new_generation() {
        let mut p = SlotPool::new();
        let a = p.alloc("a");
        assert_eq!(p.get(a), Some(&"a"));
        assert_eq!(p.free(a), Some("a"));
        assert_eq!(p.get(a), None);
        let b = p.alloc("b");
        assert_eq!(b.index(), a.index(), "slot is reused");
        assert_ne!(b.generation(), a.generation());
        assert_eq!(p.get(a), None, "old handle must not see the new occupant");
        assert_eq!(p.get(b), Some(&"b"));
    }

    #[test]
    fn double_free_and_stale_free_are_rejected() {
        let mut p = SlotPool::new();
        let a = p.alloc(1);
        assert_eq!(p.free(a), Some(1));
        assert_eq!(p.free(a), None);
        let b = p.alloc(2);
        assert_eq!(p.free(a), None, "stale handle cannot free the new occupant");
        assert_eq!(p.get(b), Some(&2));
        assert_eq!(p.stats().live, 1);
    }

    #[test]
    fn exhausted_generation_retires_the_slot() {
        let mut p = SlotPool::new();
        let a = p.alloc(1);
        p.force_generation(a.index(), u32::MAX);
        let a = Handle { index: a.index(), generation: u32::MAX };
        assert_eq!(p.free(a), Some(1));
        let b = p.alloc(2);
        assert_ne!(b.index(), a.index(), "a wrapped generation would alias handle 0, so the slot is never reused");
        assert_eq!(p.stats().exhausted_slots, 1);
    }

    #[test]
    fn free_slots_are_reused_lowest_first() {
        let mut p = SlotPool::new();
        let hs: Vec<_> = (0..4).map(|i| p.alloc(i)).collect();
        p.free(hs[3]);
        p.free(hs[1]);
        assert_eq!(p.alloc(9).index(), 1);
        assert_eq!(p.alloc(9).index(), 3);
    }
}
