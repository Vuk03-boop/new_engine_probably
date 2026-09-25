//! Exact categorical material identity, kept separate from numerical parameters
//! (PROPOSITION §2) and from occupancy.
//!
//! The registry only grows: an ID, once assigned, always refers to the same material name.
//! No ID means "empty"; emptiness is stored in the brick occupancy mask.

use std::collections::BTreeMap;

/// Material identity: a registry index. Exact, never approximated or packed lossily.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialId(u16);

impl MaterialId {
    /// Crate-private: outside code gets IDs only from a registry.
    pub(crate) const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u16 {
        self.0
    }
}

/// Numerical material parameters. Linear RGB; emissive is in the same relative units as base colour
/// until a radiometric unit is decided with the lighting work (Phase 3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialParams {
    pub base_color: [f32; 3],
    pub emissive: [f32; 3],
}

impl MaterialParams {
    pub const fn diffuse(r: f32, g: f32, b: f32) -> Self {
        Self { base_color: [r, g, b], emissive: [0.0; 3] }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialDef {
    pub name: String,
    pub params: MaterialParams,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterialError {
    DuplicateName(String),
    RegistryFull,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialRegistry {
    defs: Vec<MaterialDef>,
    by_name: BTreeMap<String, MaterialId>,
}

impl MaterialRegistry {
    /// Maximum number of materials: every value of the `u16` ID.
    pub const CAPACITY: usize = u16::MAX as usize + 1;

    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a material and returns its new ID. Names are unique, and IDs are assigned in registration order.
    pub fn register(&mut self, name: &str, params: MaterialParams) -> Result<MaterialId, MaterialError> {
        if self.by_name.contains_key(name) {
            return Err(MaterialError::DuplicateName(name.to_owned()));
        }
        if self.defs.len() >= Self::CAPACITY {
            return Err(MaterialError::RegistryFull);
        }
        let id = MaterialId(self.defs.len() as u16);
        self.defs.push(MaterialDef { name: name.to_owned(), params });
        self.by_name.insert(name.to_owned(), id);
        Ok(id)
    }

    pub fn get(&self, id: MaterialId) -> Option<&MaterialDef> {
        self.defs.get(id.0 as usize)
    }

    pub fn contains(&self, id: MaterialId) -> bool {
        (id.0 as usize) < self.defs.len()
    }

    pub fn id_of(&self, name: &str) -> Option<MaterialId> {
        self.by_name.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Heap held by the registry: definitions, names and the name index (Phase 1D).
    pub fn memory(&self) -> memory::Usage {
        use memory::containers;
        let mut u = containers::vec(&self.defs) + containers::btree_map(&self.by_name);
        for d in &self.defs {
            u += memory::Usage::new(d.name.len() as u64, d.name.capacity() as u64);
        }
        for k in self.by_name.keys() {
            u += memory::Usage::new(k.len() as u64, k.capacity() as u64);
        }
        u
    }

    /// All materials in ID order.
    pub fn iter(&self) -> impl Iterator<Item = (MaterialId, &MaterialDef)> {
        self.defs.iter().enumerate().map(|(i, d)| (MaterialId(i as u16), d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GREY: MaterialParams = MaterialParams::diffuse(0.5, 0.5, 0.5);

    #[test]
    fn ids_are_stable_and_in_registration_order() {
        let mut r = MaterialRegistry::new();
        let a = r.register("stone", GREY).unwrap();
        let b = r.register("wood", GREY).unwrap();
        assert_eq!((a.raw(), b.raw()), (0, 1));
        assert_eq!(r.id_of("stone"), Some(a));
        assert_eq!(r.get(b).unwrap().name, "wood");
        // A later registration does not move earlier IDs.
        let _c = r.register("glass", GREY).unwrap();
        assert_eq!(r.id_of("stone"), Some(a));
        assert_eq!(r.id_of("wood"), Some(b));
    }

    #[test]
    fn duplicate_name_is_rejected_without_side_effects() {
        let mut r = MaterialRegistry::new();
        let a = r.register("stone", GREY).unwrap();
        assert_eq!(r.register("stone", MaterialParams::diffuse(1.0, 0.0, 0.0)), Err(MaterialError::DuplicateName("stone".into())));
        assert_eq!(r.len(), 1);
        assert_eq!(r.get(a).unwrap().params, GREY);
    }

    #[test]
    fn registry_capacity_is_the_full_u16_range() {
        let mut r = MaterialRegistry::new();
        for i in 0..MaterialRegistry::CAPACITY {
            r.register(&format!("m{i}"), GREY).unwrap();
        }
        assert_eq!(r.register("one_more", GREY), Err(MaterialError::RegistryFull));
        assert_eq!(r.id_of("m65535").map(MaterialId::raw), Some(u16::MAX));
    }

    #[test]
    fn unknown_id_is_not_contained() {
        let mut r = MaterialRegistry::new();
        assert!(!r.contains(MaterialId(0)));
        r.register("stone", GREY).unwrap();
        assert!(r.contains(MaterialId(0)));
        assert!(!r.contains(MaterialId(1)));
    }
}
