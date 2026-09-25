//! Phase 1C: the lifetime substrate for data derived from the authoritative world, and Phase 2A:
//! its first real product, per-brick surface meshes.
//!
//! - [`slots`]: generational handles (allocation generations).
//! - [`surface`]: the exact input record of a per-brick surface product, and the face-count summary.
//! - [`mesh`]: surface extraction (quads with exact materials) and its CPU ray reference.
//! - [`pipeline`]: version-tagged jobs, a bounded queue, snapshot publication and retirement.

pub mod mesh;
pub mod pipeline;
pub mod slots;
pub mod surface;

pub use mesh::{exposed_faces, extract_world, trace_mesh, BrickMesh, Merge, MeshHit, Quad, Split, Triangles};
pub use pipeline::{Completion, Config, Faults, Job, JobId, JobResult, Pipeline, Published, ReadError, ReaderToken, Rejection, Snapshot, SnapshotId, Stats};
pub use slots::{Handle, SlotPool};
pub use surface::{affected_keys, summarize_world, Dependencies, InputRecord, SurfaceSummary};
