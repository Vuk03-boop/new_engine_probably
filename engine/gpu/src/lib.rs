//! Phase 2B: headless Vulkan bring-up for the engine.
//!
//! - [`context`]: instance, validation, capability-based device selection, driver memory budget.
//! - [`alloc`]: device memory in ledger-granted blocks (ADR-0003 budget), sub-allocated.
//! - [`timeline`]: one timeline semaphore; retirement of resources and 1C reader tokens after the
//!   GPU is done with them.
//! - [`submit`], [`staging`]: command submission, the upload ring, readback.
//! - [`layout`]: the ADR-0004 region mesh layout, built on the CPU.
//! - [`mesh`]: region meshes on the device (all-or-nothing upload).
//! - [`reflect`], [`decode`]: `slangc` reflection checks, and GPU decode of uploaded meshes.
//! - [`raster`] (2C-1): the camera conventions, the ADR-0003 G-buffer targets and the raster pass.
//! - [`equivalence`] (2C-1): the ADR-0003 Amendment 1 per-pixel check against the CPU voxel DDA.
//! - [`present`] (2C-2): the swapchain and frames in flight on the timeline.
//! - [`debug_view`] (2C-2): fullscreen debug views of the G-buffer.
//! - [`timing`] (2C-2): GPU timestamps per pass, percentiles.
//! - [`accel`] (2D): one BLAS per region from the uploaded meshes, one TLAS, the region table.
//! - [`ray`] (2D): primary visibility by ray query into the same values as the G-buffer.
//! - [`scene`] (2E): the device copy of one published snapshot, updated by edits (upload, BLAS and
//!   TLAS rebuild of the changed regions, swap, retirement).
//! - [`reference`] (3A): the reference path tracer of ADR-0005, checked against `light`.
//! - [`shade`] (3B, 3C): real-time lighting from the G-buffer (direct sun with one shadow ray, sky
//!   light with one visibility ray); 3F one bounce; 4B emitter light at both vertices.
//! - [`sky`] (3C): the sky tables on the device; the sky-view table rebuilt when the sun moves.
//! - [`sky_bake`] (S-020): bakes the reference sky that corrects the sky-view table (a tool).
//! - [`temporal`] (3D): reprojection, per-pixel history with age and rejection reasons (ADR-0006);
//!   3E: relight boxes and the sun-motion age cap (Amendment 1).
//! - [`denoise`] (3E): the native reconstruction (SVGF-style, on the exact guides).
//! - [`emitters`] (4A): the emitter table from region meshes, and its device upload.
//! - [`exposure`] (4B): the sums behind the viewer's automatic exposure.

pub mod accel;
pub mod alloc;
pub mod context;
pub mod debug_view;
pub mod decode;
pub mod denoise;
pub mod emitters;
pub mod equivalence;
pub mod exposure;
pub mod layout;
pub mod mesh;
pub mod present;
pub mod ray;
pub mod raster;
pub mod reference;
pub mod reflect;
pub mod scene;
pub mod shade;
pub mod sky;
pub mod sky_bake;
pub mod temporal;
pub mod staging;
pub mod submit;
pub mod timeline;
pub mod timing;

pub use alloc::{Allocator, Buffer, Image, Kind};
pub use context::{Gpu, GpuError, Result};
pub use layout::{build_regions, RegionKey, RegionMesh, RegionSize};
pub use mesh::GpuMeshes;
pub use raster::{Camera, Frame, Raster, Targets};
pub use timeline::{FrameReaders, Retirement, Timeline};
