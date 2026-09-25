//! Edit transactions (Phase 1B): an ordered list of operations applied to the world atomically.
//!
//! A transaction is the unit that is journaled (one record per transaction, [`crate::persist`]) and
//! the unit that `derived` publishes (one `notify_edits` call per transaction), so persistence and
//! publication agree on what an edit batch is (ADR-0002).
//!
//! Atomicity: every material in the transaction is validated before the first write. Unknown
//! materials are the only way a write can fail, so a rejected transaction changes nothing.

use crate::brick::ContentVersion;
use crate::coords::VoxelCoord;
use crate::material::MaterialId;
use crate::world::{World, WorldError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// One voxel: `Some` fills it, `None` clears it.
    Set { at: VoxelCoord, material: Option<MaterialId> },
    /// Every voxel of the half-open box `[min, max)`, in `World::fill_box` order.
    Fill { min: VoxelCoord, max: VoxelCoord, material: Option<MaterialId> },
}

impl Op {
    pub fn material(&self) -> Option<MaterialId> {
        match *self {
            Op::Set { material, .. } | Op::Fill { material, .. } => material,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Transaction {
    pub ops: Vec<Op>,
}

impl Transaction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, at: VoxelCoord, material: Option<MaterialId>) -> &mut Self {
        self.ops.push(Op::Set { at, material });
        self
    }

    pub fn fill(&mut self, min: VoxelCoord, max: VoxelCoord, material: Option<MaterialId>) -> &mut Self {
        self.ops.push(Op::Fill { min, max, material });
        self
    }
}

/// What a transaction did. `changed` lists every voxel write that changed content, in write order;
/// a voxel written twice with two real changes appears twice. It is the input for
/// `derived::Pipeline::notify_edits`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    pub version_before: ContentVersion,
    pub version_after: ContentVersion,
    pub changed: Vec<VoxelCoord>,
}

impl World {
    /// Applies all operations in order, or none of them if any names an unknown material.
    pub fn apply(&mut self, tx: &Transaction) -> Result<Applied, WorldError> {
        for op in &tx.ops {
            if let Some(id) = op.material() {
                if !self.materials().contains(id) {
                    return Err(WorldError::UnknownMaterial(id));
                }
            }
        }
        let version_before = self.version();
        let mut changed = Vec::new();
        for op in &tx.ops {
            match *op {
                Op::Set { at, material } => {
                    if self.set(at, material).expect("materials validated") {
                        changed.push(at);
                    }
                }
                Op::Fill { min, max, material } => {
                    for z in min.z..max.z {
                        for y in min.y..max.y {
                            for x in min.x..max.x {
                                let at = VoxelCoord::new(x, y, z);
                                if self.set(at, material).expect("materials validated") {
                                    changed.push(at);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Applied { version_before, version_after: self.version(), changed })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{MaterialParams, MaterialRegistry};

    fn world_with(n: usize) -> (World, Vec<MaterialId>) {
        let mut r = MaterialRegistry::new();
        let ids = (0..n).map(|i| r.register(&format!("m{i}"), MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap()).collect();
        (World::new(r), ids)
    }

    fn v(x: i32, y: i32, z: i32) -> VoxelCoord {
        VoxelCoord::new(x, y, z)
    }

    #[test]
    fn apply_matches_direct_writes_including_versions() {
        let (mut a, m) = world_with(2);
        let mut b = a.clone();
        let mut tx = Transaction::new();
        tx.fill(v(-3, 0, 0), v(9, 2, 1), Some(m[0])).set(v(0, 0, 0), Some(m[1])).set(v(0, 0, 0), Some(m[1])).set(v(-3, 0, 0), None);
        let applied = a.apply(&tx).unwrap();
        b.fill_box(v(-3, 0, 0), v(9, 2, 1), Some(m[0])).unwrap();
        b.set(v(0, 0, 0), Some(m[1])).unwrap();
        b.set(v(-3, 0, 0), None).unwrap();
        assert_eq!(a, b);
        assert_eq!(applied.version_before, ContentVersion::NONE);
        assert_eq!(applied.version_after, a.version());
        // 24 fills, one overwrite, the repeated overwrite is a no-op, one clear.
        assert_eq!(applied.changed.len(), 24 + 1 + 1);
        assert_eq!(applied.changed.last(), Some(&v(-3, 0, 0)));
    }

    #[test]
    fn transaction_with_an_unknown_material_changes_nothing() {
        let (mut w, m) = world_with(1);
        let (_, foreign) = world_with(3);
        w.set(v(1, 1, 1), Some(m[0])).unwrap();
        let before = w.clone();
        let mut tx = Transaction::new();
        tx.fill(v(0, 0, 0), v(4, 4, 4), None).set(v(5, 5, 5), Some(foreign[2]));
        assert_eq!(w.apply(&tx), Err(WorldError::UnknownMaterial(foreign[2])));
        assert_eq!(w, before);
    }

    #[test]
    fn empty_and_no_op_transactions_keep_the_version() {
        let (mut w, m) = world_with(1);
        w.set(v(0, 0, 0), Some(m[0])).unwrap();
        let ver = w.version();
        let a = w.apply(&Transaction::new()).unwrap();
        let mut tx = Transaction::new();
        tx.set(v(0, 0, 0), Some(m[0])).fill(v(10, 10, 10), v(12, 12, 12), None);
        let b = w.apply(&tx).unwrap();
        for r in [a, b] {
            assert_eq!((r.version_before, r.version_after, r.changed.len()), (ver, ver, 0));
        }
    }
}
