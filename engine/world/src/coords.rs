//! Integer coordinates and logical spatial identity.
//!
//! A voxel `v` occupies the half-open box `[v, v+1)` on each axis, in voxel units.
//! Chunk and brick coordinates use floor division, so negative coordinates work
//! the same as positive ones.

use crate::dims::{BRICK_EDGE, BRICK_VOXELS, CHUNK_BRICKS, CHUNK_EDGE_BRICKS, CHUNK_EDGE_VOXELS};

/// Absolute voxel coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Chunk coordinate: `floor(voxel / CHUNK_EDGE_VOXELS)`.
/// The derived ordering compares `x`, then `y`, then `z`, and fixes world iteration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Brick slot inside a chunk, `0..CHUNK_BRICKS`. Order: x fastest, then y, then z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BrickIndex(u8);

/// Voxel slot inside a brick, `0..BRICK_VOXELS`. Order: x fastest, then y, then z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoxelIndex(u16);

/// Logical spatial identity of a brick. It is independent of where the brick's
/// data is stored, now or on the GPU later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BrickKey {
    pub chunk: ChunkCoord,
    pub brick: BrickIndex,
}

impl VoxelCoord {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Splits into the brick that contains this voxel and the voxel's slot in it.
    pub fn split(self) -> (BrickKey, VoxelIndex) {
        let chunk = ChunkCoord {
            x: self.x.div_euclid(CHUNK_EDGE_VOXELS),
            y: self.y.div_euclid(CHUNK_EDGE_VOXELS),
            z: self.z.div_euclid(CHUNK_EDGE_VOXELS),
        };
        let (lx, ly, lz) = (
            self.x.rem_euclid(CHUNK_EDGE_VOXELS),
            self.y.rem_euclid(CHUNK_EDGE_VOXELS),
            self.z.rem_euclid(CHUNK_EDGE_VOXELS),
        );
        let brick = BrickIndex::from_xyz(lx / BRICK_EDGE, ly / BRICK_EDGE, lz / BRICK_EDGE);
        let voxel = VoxelIndex::from_xyz(lx % BRICK_EDGE, ly % BRICK_EDGE, lz % BRICK_EDGE);
        (BrickKey { chunk, brick }, voxel)
    }

    /// Inverse of [`VoxelCoord::split`].
    pub fn join(key: BrickKey, voxel: VoxelIndex) -> Self {
        let (bx, by, bz) = key.brick.xyz();
        let (vx, vy, vz) = voxel.xyz();
        Self {
            x: key.chunk.x * CHUNK_EDGE_VOXELS + bx * BRICK_EDGE + vx,
            y: key.chunk.y * CHUNK_EDGE_VOXELS + by * BRICK_EDGE + vy,
            z: key.chunk.z * CHUNK_EDGE_VOXELS + bz * BRICK_EDGE + vz,
        }
    }
}

impl BrickKey {
    /// The brick's minimum voxel corner.
    pub fn origin(self) -> VoxelCoord {
        let (bx, by, bz) = self.brick.xyz();
        VoxelCoord {
            x: self.chunk.x * CHUNK_EDGE_VOXELS + bx * BRICK_EDGE,
            y: self.chunk.y * CHUNK_EDGE_VOXELS + by * BRICK_EDGE,
            z: self.chunk.z * CHUNK_EDGE_VOXELS + bz * BRICK_EDGE,
        }
    }

    /// The brick `(dx, dy, dz)` bricks away, crossing chunk boundaries as needed. The caller keeps
    /// the result inside the i32 voxel range.
    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> BrickKey {
        let o = self.origin();
        VoxelCoord::new(o.x + dx * BRICK_EDGE, o.y + dy * BRICK_EDGE, o.z + dz * BRICK_EDGE).split().0
    }
}

impl BrickIndex {
    /// For decoding saved data; `None` outside `0..CHUNK_BRICKS`.
    pub(crate) fn from_raw(raw: u8) -> Option<Self> {
        ((raw as usize) < CHUNK_BRICKS).then_some(Self(raw))
    }

    fn from_xyz(x: i32, y: i32, z: i32) -> Self {
        debug_assert!((0..CHUNK_EDGE_BRICKS).contains(&x) && (0..CHUNK_EDGE_BRICKS).contains(&y) && (0..CHUNK_EDGE_BRICKS).contains(&z));
        Self((x + CHUNK_EDGE_BRICKS * (y + CHUNK_EDGE_BRICKS * z)) as u8)
    }

    pub fn xyz(self) -> (i32, i32, i32) {
        let i = self.0 as i32;
        (i % CHUNK_EDGE_BRICKS, (i / CHUNK_EDGE_BRICKS) % CHUNK_EDGE_BRICKS, i / (CHUNK_EDGE_BRICKS * CHUNK_EDGE_BRICKS))
    }

    pub fn get(self) -> usize {
        self.0 as usize
    }

    /// Every brick slot of a chunk, in storage order.
    pub fn all() -> impl Iterator<Item = BrickIndex> {
        (0..CHUNK_BRICKS).map(|i| BrickIndex(i as u8))
    }
}

impl VoxelIndex {
    fn from_xyz(x: i32, y: i32, z: i32) -> Self {
        debug_assert!((0..BRICK_EDGE).contains(&x) && (0..BRICK_EDGE).contains(&y) && (0..BRICK_EDGE).contains(&z));
        Self((x + BRICK_EDGE * (y + BRICK_EDGE * z)) as u16)
    }

    pub fn xyz(self) -> (i32, i32, i32) {
        let i = self.0 as i32;
        (i % BRICK_EDGE, (i / BRICK_EDGE) % BRICK_EDGE, i / (BRICK_EDGE * BRICK_EDGE))
    }

    pub fn get(self) -> usize {
        self.0 as usize
    }

    /// Every voxel slot of a brick, in storage order.
    pub fn all() -> impl Iterator<Item = VoxelIndex> {
        (0..BRICK_VOXELS).map(|i| VoxelIndex(i as u16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_join_round_trip_including_negatives_and_boundaries() {
        let edges = [i32::MIN, -1_000_001, -33, -32, -31, -9, -8, -7, -1, 0, 1, 7, 8, 9, 31, 32, 33, 1_000_000, i32::MAX];
        for &x in &edges {
            for &y in &[-1, 0, 8] {
                for &z in &edges {
                    let v = VoxelCoord::new(x, y, z);
                    let (key, idx) = v.split();
                    assert_eq!(VoxelCoord::join(key, idx), v, "{v:?}");
                }
            }
        }
    }

    #[test]
    fn negative_voxels_use_floor_division() {
        let (key, idx) = VoxelCoord::new(-1, -1, -1).split();
        assert_eq!(key.chunk, ChunkCoord { x: -1, y: -1, z: -1 });
        let last = CHUNK_EDGE_BRICKS - 1;
        assert_eq!(key.brick.xyz(), (last, last, last));
        assert_eq!(idx.xyz(), (BRICK_EDGE - 1, BRICK_EDGE - 1, BRICK_EDGE - 1));
        // -32 is the first voxel of chunk -1, and -33 the last voxel of chunk -2.
        assert_eq!(VoxelCoord::new(-32, 0, 0).split().0.chunk.x, -1);
        assert_eq!(VoxelCoord::new(-33, 0, 0).split().0.chunk.x, -2);
    }

    #[test]
    fn brick_origin_and_offset_cross_chunk_edges() {
        let (k, _) = VoxelCoord::new(-1, 5, 31).split();
        assert_eq!(k.origin(), VoxelCoord::new(-8, 0, 24));
        assert_eq!(k.offset(1, 0, 0).origin(), VoxelCoord::new(0, 0, 24));
        assert_eq!(k.offset(0, 0, 1).origin(), VoxelCoord::new(-8, 0, 32));
        assert_eq!(k.offset(0, 0, 1).chunk, ChunkCoord { x: -1, y: 0, z: 1 });
        assert_eq!(k.offset(1, 0, 0).offset(-1, 0, 0), k);
    }

    #[test]
    fn neighbouring_voxels_across_brick_and_chunk_edges_get_distinct_slots() {
        let a = VoxelCoord::new(7, 0, 0).split();
        let b = VoxelCoord::new(8, 0, 0).split();
        assert_eq!(a.0.chunk, b.0.chunk);
        assert_ne!(a.0.brick, b.0.brick);
        let c = VoxelCoord::new(31, 0, 0).split();
        let d = VoxelCoord::new(32, 0, 0).split();
        assert_ne!(c.0.chunk, d.0.chunk);
    }
}
