//! The authoritative sparse world: a chunk hierarchy of sparse bricks.
//!
//! - Only chunks that contain at least one occupied voxel exist, and only their non-empty bricks.
//!   Clearing the last voxel frees the brick, and freeing a chunk's last brick frees the chunk.
//! - Storage is ordered (`BTreeMap`, fixed slot order), so iteration, and later snapshots, are
//!   deterministic.
//! - Every content change stamps the affected brick with the next world-wide [`ContentVersion`].
//!   A no-op write changes nothing and stamps nothing.

use std::collections::BTreeMap;

use crate::brick::{Brick, ContentVersion};
use crate::coords::{BrickIndex, BrickKey, ChunkCoord, VoxelCoord};
use crate::dims::{BRICK_VOXELS, CHUNK_BRICKS, CHUNK_EDGE_VOXELS};
use crate::material::{MaterialId, MaterialRegistry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorldError {
    /// The material ID is not registered in this world's registry.
    UnknownMaterial(MaterialId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Chunk {
    bricks: [Option<Box<Brick>>; CHUNK_BRICKS],
    live: u8,
}

impl Chunk {
    fn new() -> Self {
        Self { bricks: std::array::from_fn(|_| None), live: 0 }
    }
}

/// Storage counts, and an estimate of payload bytes (brick masks, material arrays and chunk
/// tables). The estimate excludes `BTreeMap` node overhead and the registry; [`World::memory`]
/// is the full per-category account (Phase 1D).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldStats {
    pub chunks: usize,
    pub bricks: usize,
    pub occupied_voxels: usize,
    pub payload_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct World {
    materials: MaterialRegistry,
    chunks: BTreeMap<ChunkCoord, Chunk>,
    /// The last stamp issued. `NONE` until the first content change.
    version: ContentVersion,
}

impl World {
    pub fn new(materials: MaterialRegistry) -> Self {
        Self { materials, chunks: BTreeMap::new(), version: ContentVersion::NONE }
    }

    /// Rebuilds a world from saved parts (Phase 1B), with the versions it was saved with. Bricks
    /// must come in `bricks()` order. Rejects anything the edit API could not have produced:
    /// unknown materials, versions above the world version, or two bricks sharing a version (every
    /// content change stamps exactly one brick with a fresh version).
    pub(crate) fn from_parts(materials: MaterialRegistry, version: ContentVersion, bricks: Vec<(BrickKey, Brick)>) -> Result<Self, String> {
        let mut w = World::new(materials);
        w.version = version;
        let mut seen_versions = std::collections::BTreeSet::new();
        let mut prev: Option<BrickKey> = None;
        for (key, brick) in bricks {
            if prev.is_some_and(|p| p >= key) {
                return Err(format!("brick {key:?} is out of order or duplicated"));
            }
            prev = Some(key);
            if brick.version() > version {
                return Err(format!("brick {key:?} version {} is above the world version {}", brick.version().0, version.0));
            }
            if !seen_versions.insert(brick.version()) {
                return Err(format!("brick {key:?} reuses version {}", brick.version().0));
            }
            if let Some((_, m)) = brick.voxels().find(|&(_, m)| !w.materials.contains(m)) {
                return Err(format!("brick {key:?} uses unknown material {}", m.raw()));
            }
            let chunk = w.chunks.entry(key.chunk).or_insert_with(Chunk::new);
            chunk.bricks[key.brick.get()] = Some(Box::new(brick));
            chunk.live += 1;
        }
        Ok(w)
    }

    pub fn materials(&self) -> &MaterialRegistry {
        &self.materials
    }

    /// The most recent content version stamped anywhere in the world.
    pub fn version(&self) -> ContentVersion {
        self.version
    }

    pub fn get(&self, v: VoxelCoord) -> Option<MaterialId> {
        let (key, idx) = v.split();
        self.brick(key)?.get(idx)
    }

    /// Sets one voxel: `Some(material)` fills it, `None` clears it. Returns whether the content changed.
    pub fn set(&mut self, v: VoxelCoord, m: Option<MaterialId>) -> Result<bool, WorldError> {
        if let Some(id) = m {
            if !self.materials.contains(id) {
                return Err(WorldError::UnknownMaterial(id));
            }
        }
        let (key, idx) = v.split();
        let next = ContentVersion(self.version.0 + 1);
        let slot = key.brick.get();
        match m {
            None => {
                let Some(chunk) = self.chunks.get_mut(&key.chunk) else { return Ok(false) };
                let Some(brick) = chunk.bricks[slot].as_mut() else { return Ok(false) };
                if !brick.put(idx, None) {
                    return Ok(false);
                }
                if brick.is_empty() {
                    chunk.bricks[slot] = None;
                    chunk.live -= 1;
                    if chunk.live == 0 {
                        self.chunks.remove(&key.chunk);
                    }
                } else {
                    brick.stamp(next);
                }
            }
            Some(_) => {
                let chunk = self.chunks.entry(key.chunk).or_insert_with(Chunk::new);
                let created = chunk.bricks[slot].is_none();
                let brick = chunk.bricks[slot].get_or_insert_with(|| Box::new(Brick::new(next)));
                if !brick.put(idx, m) {
                    // Only reachable for an existing brick: a new brick is empty, so a fill always changes it.
                    debug_assert!(!created);
                    return Ok(false);
                }
                if created {
                    chunk.live += 1;
                }
                brick.stamp(next);
            }
        }
        // Removing a brick is also a content change: the world version advances even though no brick carries the stamp.
        self.version = next;
        Ok(true)
    }

    /// Sets every voxel in the half-open box `[min, max)`. Returns the number of voxels that changed.
    pub fn fill_box(&mut self, min: VoxelCoord, max: VoxelCoord, m: Option<MaterialId>) -> Result<usize, WorldError> {
        let mut changed = 0;
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    changed += self.set(VoxelCoord::new(x, y, z), m)? as usize;
                }
            }
        }
        Ok(changed)
    }

    pub fn brick(&self, key: BrickKey) -> Option<&Brick> {
        self.chunks.get(&key.chunk)?.bricks[key.brick.get()].as_deref()
    }

    /// Every stored brick, in deterministic order: chunk coordinate, then slot.
    pub fn bricks(&self) -> impl Iterator<Item = (BrickKey, &Brick)> + '_ {
        self.chunks.iter().flat_map(|(&chunk, c)| {
            BrickIndex::all().filter_map(move |brick| c.bricks[brick.get()].as_deref().map(|b| (BrickKey { chunk, brick }, b)))
        })
    }

    /// Every occupied voxel and its material, in deterministic order.
    pub fn occupied(&self) -> impl Iterator<Item = (VoxelCoord, MaterialId)> + '_ {
        self.bricks().flat_map(|(key, b)| b.voxels().map(move |(idx, m)| (VoxelCoord::join(key, idx), m)))
    }

    /// Chunk-granular bounds of all stored data, as the half-open voxel box `[min, max)`. `None` if the world is empty.
    pub fn bounds(&self) -> Option<(VoxelCoord, VoxelCoord)> {
        let mut it = self.chunks.keys();
        let first = *it.next()?;
        let (mut lo, mut hi) = (first, first);
        for c in it {
            lo = ChunkCoord { x: lo.x.min(c.x), y: lo.y.min(c.y), z: lo.z.min(c.z) };
            hi = ChunkCoord { x: hi.x.max(c.x), y: hi.y.max(c.y), z: hi.z.max(c.z) };
        }
        let e = CHUNK_EDGE_VOXELS as i64;
        let clamp = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        Some((
            VoxelCoord::new(clamp(lo.x as i64 * e), clamp(lo.y as i64 * e), clamp(lo.z as i64 * e)),
            VoxelCoord::new(clamp((hi.x as i64 + 1) * e), clamp((hi.y as i64 + 1) * e), clamp((hi.z as i64 + 1) * e)),
        ))
    }

    pub fn stats(&self) -> WorldStats {
        let bricks = self.chunks.values().map(|c| c.live as usize).sum::<usize>();
        let occupied_voxels = self.bricks().map(|(_, b)| b.occupied_count()).sum();
        let brick_bytes = std::mem::size_of::<Brick>() + BRICK_VOXELS * std::mem::size_of::<MaterialId>();
        WorldStats {
            chunks: self.chunks.len(),
            bricks,
            occupied_voxels,
            payload_bytes: bricks * brick_bytes + self.chunks.len() * std::mem::size_of::<Chunk>(),
        }
    }

    /// Heap held by the world, by category (Phase 1D). Bricks are exact (two boxes each, no slack);
    /// the chunk index and the registry report container upper bounds as `reserved`.
    pub fn memory(&self) -> memory::Report {
        let bricks = self.chunks.values().map(|c| c.live as u64).sum::<u64>();
        let per_brick = (std::mem::size_of::<Brick>() + BRICK_VOXELS * std::mem::size_of::<MaterialId>()) as u64;
        let mut r = memory::Report::new();
        r.add(memory::Category::WorldBricks, memory::Usage::exact(bricks * per_brick));
        r.add(memory::Category::WorldHierarchy, memory::containers::btree_map(&self.chunks));
        r.add(memory::Category::Materials, self.materials.memory());
        r
    }

    /// Same registry and same voxels, ignoring version stamps.
    pub fn same_content(&self, other: &World) -> bool {
        self.materials == other.materials
            && self.chunks.len() == other.chunks.len()
            && self.bricks().count() == other.bricks().count()
            && self.bricks().zip(other.bricks()).all(|((ka, a), (kb, b))| ka == kb && a.same_content(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialParams;

    fn world_with(names: &[&str]) -> (World, Vec<MaterialId>) {
        let mut r = MaterialRegistry::new();
        let ids = names.iter().map(|n| r.register(n, MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap()).collect();
        (World::new(r), ids)
    }

    fn v(x: i32, y: i32, z: i32) -> VoxelCoord {
        VoxelCoord::new(x, y, z)
    }

    #[test]
    fn empty_world_reads_empty_and_stores_nothing() {
        let (w, _) = world_with(&["stone"]);
        assert_eq!(w.get(v(0, 0, 0)), None);
        assert_eq!(w.get(v(-1, -1, -1)), None);
        assert_eq!(w.bounds(), None);
        assert_eq!(w.stats(), WorldStats { chunks: 0, bricks: 0, occupied_voxels: 0, payload_bytes: 0 });
        assert_eq!(w.version(), ContentVersion::NONE);
    }

    #[test]
    fn set_get_round_trip_across_brick_and_chunk_edges() {
        let (mut w, m) = world_with(&["stone", "wood"]);
        let coords = [v(7, 0, 0), v(8, 0, 0), v(31, 31, 31), v(32, 32, 32), v(-1, -1, -1), v(-32, 0, 0), v(-33, 0, 0), v(0, -8, 0), v(0, -9, 0)];
        for (i, &c) in coords.iter().enumerate() {
            assert_eq!(w.set(c, Some(m[i % 2])), Ok(true));
        }
        for (i, &c) in coords.iter().enumerate() {
            assert_eq!(w.get(c), Some(m[i % 2]), "{c:?}");
        }
        // The six face neighbours of every written voxel stay empty unless they were written themselves.
        for &c in &coords {
            for d in [v(1, 0, 0), v(-1, 0, 0), v(0, 1, 0), v(0, -1, 0), v(0, 0, 1), v(0, 0, -1)] {
                let n = v(c.x + d.x, c.y + d.y, c.z + d.z);
                if !coords.contains(&n) {
                    assert_eq!(w.get(n), None, "neighbour {n:?} of {c:?}");
                }
            }
        }
        assert_eq!(w.stats().occupied_voxels, coords.len());
    }

    #[test]
    fn unknown_material_is_rejected_and_changes_nothing() {
        let (mut w, _) = world_with(&["stone"]);
        let (_, foreign) = world_with(&["a", "b", "c"]);
        let before = w.clone();
        assert_eq!(w.set(v(0, 0, 0), Some(foreign[2])), Err(WorldError::UnknownMaterial(foreign[2])));
        assert_eq!(w, before);
    }

    #[test]
    fn material_identity_is_exact_and_overwrite_replaces_it() {
        let (mut w, m) = world_with(&["stone", "wood", "glass"]);
        w.set(v(3, 3, 3), Some(m[2])).unwrap();
        assert_eq!(w.get(v(3, 3, 3)), Some(m[2]));
        assert_eq!(w.set(v(3, 3, 3), Some(m[0])), Ok(true));
        assert_eq!(w.get(v(3, 3, 3)), Some(m[0]));
        assert_eq!(w.stats().occupied_voxels, 1);
        // Material with raw ID 0 is a real material, not "empty".
        assert_eq!(m[0].raw(), 0);
        assert_eq!(w.get(v(3, 3, 4)), None);
    }

    #[test]
    fn clearing_last_voxel_frees_brick_then_chunk() {
        let (mut w, m) = world_with(&["stone"]);
        w.set(v(1, 1, 1), Some(m[0])).unwrap();
        w.set(v(9, 1, 1), Some(m[0])).unwrap(); // second brick, same chunk
        assert_eq!((w.stats().chunks, w.stats().bricks), (1, 2));
        w.set(v(9, 1, 1), None).unwrap();
        assert_eq!((w.stats().chunks, w.stats().bricks), (1, 1));
        w.set(v(1, 1, 1), None).unwrap();
        assert_eq!(w.stats(), WorldStats { chunks: 0, bricks: 0, occupied_voxels: 0, payload_bytes: 0 });
        assert_eq!(w.bounds(), None);
    }

    #[test]
    fn clearing_empty_space_allocates_nothing() {
        let (mut w, _) = world_with(&["stone"]);
        assert_eq!(w.fill_box(v(-20, -20, -20), v(20, 20, 20), None), Ok(0));
        assert_eq!(w.stats().chunks, 0);
        assert_eq!(w.version(), ContentVersion::NONE);
    }

    #[test]
    fn versions_advance_only_on_real_changes_and_only_for_the_touched_brick() {
        let (mut w, m) = world_with(&["stone", "wood"]);
        w.set(v(0, 0, 0), Some(m[0])).unwrap();
        w.set(v(8, 0, 0), Some(m[0])).unwrap();
        let (ka, _) = v(0, 0, 0).split();
        let (kb, _) = v(8, 0, 0).split();
        let a1 = w.brick(ka).unwrap().version();
        let b1 = w.brick(kb).unwrap().version();
        assert!(a1 < b1 && b1 == w.version());

        // No-op writes: same material, clearing an empty voxel.
        assert_eq!(w.set(v(0, 0, 0), Some(m[0])), Ok(false));
        assert_eq!(w.set(v(1, 0, 0), None), Ok(false));
        assert_eq!(w.brick(ka).unwrap().version(), a1);
        assert_eq!(w.version(), b1);

        // A real change stamps only its own brick.
        w.set(v(0, 0, 0), Some(m[1])).unwrap();
        let a2 = w.brick(ka).unwrap().version();
        assert!(a2 > b1);
        assert_eq!(w.brick(kb).unwrap().version(), b1);
    }

    #[test]
    fn recreated_brick_never_reuses_an_old_version() {
        let (mut w, m) = world_with(&["stone"]);
        let (key, _) = v(5, 5, 5).split();
        w.set(v(5, 5, 5), Some(m[0])).unwrap();
        let first = w.brick(key).unwrap().version();
        w.set(v(5, 5, 5), None).unwrap();
        assert!(w.brick(key).is_none());
        assert!(w.version() > first, "removal is a change");
        w.set(v(5, 5, 5), Some(m[0])).unwrap();
        let second = w.brick(key).unwrap().version();
        assert!(second > first);
        // Same content as before, but a different version: a job that read `first` can be rejected.
        assert_ne!(first, second);
    }

    #[test]
    fn same_edits_give_identical_worlds_and_order_independent_content() {
        let build = |order: &[usize]| {
            let (mut w, m) = world_with(&["stone", "wood"]);
            let edits = [(v(0, 0, 0), m[0]), (v(40, -3, 9), m[1]), (v(-7, 2, -70), m[0]), (v(1, 0, 0), m[1])];
            for &i in order {
                w.set(edits[i].0, Some(edits[i].1)).unwrap();
            }
            w
        };
        let a = build(&[0, 1, 2, 3]);
        let b = build(&[0, 1, 2, 3]);
        let c = build(&[3, 2, 1, 0]);
        assert_eq!(a, b, "same edit sequence is fully identical, versions included");
        assert_ne!(a, c, "different order gives different version stamps");
        assert!(a.same_content(&c), "but the same content");
        let listed: Vec<_> = a.occupied().collect();
        assert_eq!(listed, c.occupied().collect::<Vec<_>>());
        let mut sorted = listed.clone();
        sorted.sort_by_key(|&(c, _)| (c.split().0, c.split().1));
        assert_eq!(listed, sorted, "iteration is in (chunk, brick, voxel) order");
    }

    #[test]
    fn bounds_are_chunk_granular_and_cover_all_voxels() {
        let (mut w, m) = world_with(&["stone"]);
        w.set(v(-1, 0, 0), Some(m[0])).unwrap();
        w.set(v(40, 5, 0), Some(m[0])).unwrap();
        let (lo, hi) = w.bounds().unwrap();
        assert_eq!(lo, v(-32, 0, 0));
        assert_eq!(hi, v(64, 32, 32));
        for (c, _) in w.occupied() {
            assert!(lo.x <= c.x && c.x < hi.x && lo.y <= c.y && c.y < hi.y && lo.z <= c.z && c.z < hi.z);
        }
    }

    #[test]
    fn payload_estimate_scales_with_stored_bricks_only() {
        let (mut w, m) = world_with(&["stone"]);
        w.fill_box(v(0, 0, 0), v(8, 8, 8), Some(m[0])).unwrap();
        let one = w.stats();
        assert_eq!((one.bricks, one.occupied_voxels), (1, 512));
        // Fill a far-away brick: one more brick and one more chunk, not the space in between.
        w.set(v(1000, 1000, 1000), Some(m[0])).unwrap();
        let two = w.stats();
        assert_eq!((two.chunks, two.bricks), (2, 2));
        assert_eq!(two.payload_bytes, 2 * one.payload_bytes);
    }
}
