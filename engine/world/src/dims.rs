//! World dimensions. These are **starting tuning values**, not inherited constants
//! (PROPOSITION §2): change them here only, and every other module derives from them.

/// Brick edge in voxels, as a power of two.
pub const BRICK_EDGE_LOG2: u32 = 3;
/// Brick edge in voxels (8).
pub const BRICK_EDGE: i32 = 1 << BRICK_EDGE_LOG2;
/// Voxels per brick (512).
pub const BRICK_VOXELS: usize = (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize;
/// 64-bit words in a brick's occupancy mask.
pub const OCCUPANCY_WORDS: usize = BRICK_VOXELS / 64;

/// Chunk edge in bricks, as a power of two.
pub const CHUNK_EDGE_BRICKS_LOG2: u32 = 2;
/// Chunk edge in bricks (4).
pub const CHUNK_EDGE_BRICKS: i32 = 1 << CHUNK_EDGE_BRICKS_LOG2;
/// Bricks per chunk (64).
pub const CHUNK_BRICKS: usize = (CHUNK_EDGE_BRICKS * CHUNK_EDGE_BRICKS * CHUNK_EDGE_BRICKS) as usize;
/// Chunk edge in voxels (32).
pub const CHUNK_EDGE_VOXELS: i32 = BRICK_EDGE * CHUNK_EDGE_BRICKS;

/// Voxel edge in meters (1/16 m, the P-001 "fine blocky voxels" contract).
/// World axes are right-handed, Y up, matching the Phase 0 probes.
pub const VOXEL_SIZE_M: f64 = 1.0 / 16.0;

const _: () = assert!(BRICK_VOXELS % 64 == 0, "occupancy mask must be whole u64 words");
const _: () = assert!(BRICK_VOXELS <= u16::MAX as usize + 1, "VoxelIndex is u16");
const _: () = assert!(CHUNK_BRICKS <= u8::MAX as usize + 1, "BrickIndex is u8");
