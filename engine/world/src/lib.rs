//! Authoritative sparse voxel world (Phase 1A) and its CPU reference ray query (Phase 1E).
//!
//! Scope: the world data, material identity and content versions, plus edit transactions, the
//! edit journal and exact save/reload (1B, [`persist`]). Versioned jobs and retirement (1C) live in
//! the `derived` crate; allocation accounting (1D) is a later slice.

pub mod brick;
pub mod coords;
pub mod dims;
pub mod edit;
pub mod material;
pub mod persist;
pub mod reference;
pub mod scene;
pub mod world;

pub use brick::{Brick, ContentVersion};
pub use coords::{BrickIndex, BrickKey, ChunkCoord, VoxelCoord, VoxelIndex};
pub use edit::{Applied, Op, Transaction};
pub use material::{MaterialDef, MaterialError, MaterialId, MaterialParams, MaterialRegistry};
pub use world::{World, WorldError, WorldStats};
