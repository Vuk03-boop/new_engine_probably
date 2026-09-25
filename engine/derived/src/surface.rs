//! A brick's exposed-surface inputs: the dependency record shared by every per-brick surface
//! product, and the exposed-face summary (the 1C stand-in, now a cross-check for [`crate::mesh`]).
//!
//! A face is exposed when an occupied voxel's neighbour across that face is empty. A brick's result depends on
//! its own voxels **and** on the boundary layers of its six face-neighbour bricks, so edits on brick
//! boundaries invalidate neighbours.

use world::dims::{BRICK_EDGE, BRICK_VOXELS};
use world::{Brick, BrickKey, ContentVersion, MaterialId, VoxelCoord, World};

/// The target brick followed by its six face neighbours (-x, +x, -y, +y, -z, +z).
pub fn input_keys(key: BrickKey) -> [BrickKey; 7] {
    [key, key.offset(-1, 0, 0), key.offset(1, 0, 0), key.offset(0, -1, 0), key.offset(0, 1, 0), key.offset(0, 0, -1), key.offset(0, 0, 1)]
}

/// Bricks whose summary can change when voxel `v` changes: its own brick, plus each neighbour brick
/// across a brick face that `v` lies on.
pub fn affected_keys(v: VoxelCoord) -> Vec<BrickKey> {
    let (key, _) = v.split();
    let o = key.origin();
    let last = BRICK_EDGE - 1;
    let mut out = vec![key];
    for (local, axis) in [(v.x - o.x, 0), (v.y - o.y, 1), (v.z - o.z, 2)] {
        let mut d = [0; 3];
        if local == 0 {
            d[axis] = -1;
            out.push(key.offset(d[0], d[1], d[2]));
        } else if local == last {
            d[axis] = 1;
            out.push(key.offset(d[0], d[1], d[2]));
        }
    }
    out
}

/// Exposed faces per material, sorted by material ID; `total` is their sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceSummary {
    pub total: u32,
    pub by_material: Vec<(MaterialId, u32)>,
}

/// Computes the summary for `key` from a voxel lookup. Returns `None` when the brick is absent (empty).
pub fn summarize(key: BrickKey, target: Option<&Brick>, lookup: impl Fn(VoxelCoord) -> Option<MaterialId>) -> Option<SurfaceSummary> {
    let target = target?;
    let o = key.origin();
    let mut counts = std::collections::BTreeMap::<MaterialId, u32>::new();
    for (idx, m) in target.voxels() {
        let (x, y, z) = idx.xyz();
        let p = VoxelCoord::new(o.x + x, o.y + y, o.z + z);
        for (dx, dy, dz) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            if lookup(VoxelCoord::new(p.x + dx, p.y + dy, p.z + dz)).is_none() {
                *counts.entry(m).or_default() += 1;
            }
        }
    }
    Some(SurfaceSummary { total: counts.values().sum(), by_material: counts.into_iter().collect() })
}

/// Oracle: the summary computed directly from the world.
pub fn summarize_world(world: &World, key: BrickKey) -> Option<SurfaceSummary> {
    summarize(key, world.brick(key), |v| world.get(v))
}

/// The seven brick keys a result read and the version each had (`None` = the brick did not exist).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRecord {
    pub keys: [BrickKey; 7],
    pub versions: [Option<ContentVersion>; 7],
}

impl InputRecord {
    /// The first input whose version differs from the world's current one, if any.
    pub fn first_version_change(&self, world: &World) -> Option<BrickKey> {
        (0..7).find(|&i| world.brick(self.keys[i]).map(Brick::version) != self.versions[i]).map(|i| self.keys[i])
    }
}

const FACE: usize = (BRICK_EDGE * BRICK_EDGE) as usize;

/// The voxel layer just outside `key`'s face `n` (1..=6 in `input_keys` order), read through `lookup`.
fn face_layer(key: BrickKey, n: usize, lookup: impl Fn(VoxelCoord) -> Option<MaterialId>) -> [Option<MaterialId>; FACE] {
    let o = key.origin();
    let (axis, outside) = [(0, -1), (0, BRICK_EDGE), (1, -1), (1, BRICK_EDGE), (2, -1), (2, BRICK_EDGE)][n - 1];
    std::array::from_fn(|i| {
        let (a, b) = ((i as i32) % BRICK_EDGE, (i as i32) / BRICK_EDGE);
        let local = match axis {
            0 => [outside, a, b],
            1 => [a, outside, b],
            _ => [a, b, outside],
        };
        lookup(VoxelCoord::new(o.x + local[0], o.y + local[1], o.z + local[2]))
    })
}

/// Everything a result depends on, exactly: the target brick's content and the six neighbouring
/// face layers. The summary is a pure function of these, so "unchanged dependencies" means
/// "unchanged result", even when a neighbour's version changed through an edit elsewhere in that brick.
#[derive(Clone, Debug)]
pub struct Dependencies {
    pub record: InputRecord,
    target: Option<Brick>,
    faces: [[Option<MaterialId>; FACE]; 6],
}

impl Dependencies {
    /// The first input whose *relevant content* changed. Versions are the fast path; on a version
    /// mismatch the recorded content is compared with the world's.
    pub fn first_changed(&self, world: &World) -> Option<BrickKey> {
        let key = self.record.keys[0];
        (0..7)
            .find(|&i| {
                let k = self.record.keys[i];
                if world.brick(k).map(Brick::version) == self.record.versions[i] {
                    return false;
                }
                if i == 0 {
                    match (world.brick(k), &self.target) {
                        (None, None) => false,
                        (Some(a), Some(b)) => !a.same_content(b),
                        _ => true,
                    }
                } else {
                    face_layer(key, i, |v| world.get(v)) != self.faces[i - 1]
                }
            })
            .map(|i| self.record.keys[i])
    }
}

/// Immutable copies of a job's input bricks, taken at dispatch, plus their record.
#[derive(Clone, Debug)]
pub struct InputSnapshot {
    pub record: InputRecord,
    bricks: [Option<Brick>; 7],
}

impl InputSnapshot {
    pub fn capture(world: &World, key: BrickKey) -> Self {
        let keys = input_keys(key);
        let bricks = keys.map(|k| world.brick(k).cloned());
        let versions = std::array::from_fn(|i| bricks[i].as_ref().map(Brick::version));
        Self { record: InputRecord { keys, versions }, bricks }
    }

    fn lookup(&self, v: VoxelCoord) -> Option<MaterialId> {
        let (k, idx) = v.split();
        let i = self.record.keys.iter().position(|&x| x == k)?;
        self.bricks[i].as_ref()?.get(idx)
    }

    /// Recomputes the summary from the copies alone. The world is not touched, so this may run on any thread.
    pub fn summarize(&self) -> Option<SurfaceSummary> {
        summarize(self.record.keys[0], self.bricks[0].as_ref(), |v| self.lookup(v))
    }

    /// Extracts the target's mesh from the copies alone (Phase 2A). The world is not touched.
    pub fn extract(&self, merge: crate::mesh::Merge) -> Option<crate::mesh::BrickMesh> {
        crate::mesh::extract(self.record.keys[0], self.bricks[0].as_ref(), |v| self.lookup(v), merge)
    }

    /// The exact dependency record for the result computed from these copies.
    pub fn dependencies(&self) -> Dependencies {
        let key = self.record.keys[0];
        Dependencies {
            record: self.record,
            target: self.bricks[0].clone(),
            faces: std::array::from_fn(|n| face_layer(key, n + 1, |v| self.lookup(v))),
        }
    }
}

/// Heap held by a brick copy: its boxed material array (the brick header is inline in its owner).
fn brick_heap(b: &Option<Brick>) -> u64 {
    if b.is_some() {
        (BRICK_VOXELS * std::mem::size_of::<MaterialId>()) as u64
    } else {
        0
    }
}

impl SurfaceSummary {
    /// Heap outside the struct itself (Phase 1D).
    pub fn heap(&self) -> memory::Usage {
        memory::containers::vec(&self.by_material)
    }
}

impl Dependencies {
    /// Heap outside the struct itself (Phase 1D).
    pub fn heap(&self) -> memory::Usage {
        memory::Usage::exact(brick_heap(&self.target))
    }
}

impl InputSnapshot {
    /// Heap outside the struct itself (Phase 1D): the copied input bricks.
    pub fn heap(&self) -> memory::Usage {
        memory::Usage::exact(self.bricks.iter().map(brick_heap).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use world::{MaterialParams, MaterialRegistry};

    fn world() -> (World, MaterialId) {
        let mut r = MaterialRegistry::new();
        let m = r.register("stone", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
        (World::new(r), m)
    }

    #[test]
    fn single_and_adjacent_voxels() {
        let (mut w, m) = world();
        w.set(VoxelCoord::new(3, 3, 3), Some(m)).unwrap();
        let k = VoxelCoord::new(3, 3, 3).split().0;
        assert_eq!(summarize_world(&w, k).unwrap().total, 6);
        w.set(VoxelCoord::new(4, 3, 3), Some(m)).unwrap();
        assert_eq!(summarize_world(&w, k).unwrap().total, 10);
    }

    #[test]
    fn neighbour_brick_hides_boundary_faces() {
        let (mut w, m) = world();
        w.set(VoxelCoord::new(7, 0, 0), Some(m)).unwrap();
        let k = VoxelCoord::new(7, 0, 0).split().0;
        assert_eq!(summarize_world(&w, k).unwrap().total, 6);
        w.set(VoxelCoord::new(8, 0, 0), Some(m)).unwrap(); // in the +x neighbour brick
        assert_eq!(summarize_world(&w, k).unwrap().total, 5);
        assert!(affected_keys(VoxelCoord::new(8, 0, 0)).contains(&k), "boundary edit must mark the neighbour");
        // The job path sees the same thing from its copies.
        assert_eq!(InputSnapshot::capture(&w, k).summarize(), summarize_world(&w, k));
    }

    #[test]
    fn affected_keys_interior_edge_and_corner() {
        assert_eq!(affected_keys(VoxelCoord::new(3, 3, 3)).len(), 1);
        assert_eq!(affected_keys(VoxelCoord::new(0, 3, 3)).len(), 2);
        assert_eq!(affected_keys(VoxelCoord::new(7, 0, 15)).len(), 4);
        // Negative coordinates: -1 is local 7 of its brick, so +x is the neighbour.
        let a = affected_keys(VoxelCoord::new(-1, 3, 3));
        assert_eq!(a[1], VoxelCoord::new(0, 3, 3).split().0);
    }

    #[test]
    fn dependency_check_is_exact_not_brick_granular() {
        let (mut w, m) = world();
        w.set(VoxelCoord::new(3, 3, 3), Some(m)).unwrap();
        let k = VoxelCoord::new(3, 3, 3).split().0;
        let deps = InputSnapshot::capture(&w, k).dependencies();
        assert_eq!(deps.first_changed(&w), None);
        // Creating the -x neighbour brick away from the shared face changes its version, not the result.
        w.set(VoxelCoord::new(-3, 3, 3), Some(m)).unwrap();
        assert_eq!(deps.record.first_version_change(&w), Some(k.offset(-1, 0, 0)));
        assert_eq!(deps.first_changed(&w), None, "face layer x = -1 is unchanged");
        // A voxel on the shared face layer does change it.
        w.set(VoxelCoord::new(-1, 3, 3), Some(m)).unwrap();
        assert_eq!(deps.first_changed(&w), Some(k.offset(-1, 0, 0)));
        // A target change is always a change.
        let deps = InputSnapshot::capture(&w, k).dependencies();
        w.set(VoxelCoord::new(5, 5, 5), Some(m)).unwrap();
        assert_eq!(deps.first_changed(&w), Some(k));
    }
}
