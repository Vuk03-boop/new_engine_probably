//! Phase 2E: the device copy of one published 1C snapshot, updated by edits.
//!
//! - [`GpuScene`] holds the region meshes ([`GpuMeshes`]) and, on ray-tracing devices, their
//!   [`Accel`]. It shows exactly one published snapshot at a time ([`GpuScene::snapshot`]).
//! - **Edit path:** the brick keys a publication changed → [`affected_regions`] → those regions'
//!   meshes rebuilt from the new snapshot ([`region_meshes`]) → [`GpuScene::update`]: upload into
//!   new buffers, rebuild those regions' BLAS and the top level, then swap.
//! - **Coherence:** the swap happens only after every new resource was granted and built. On a
//!   refused grant the update frees what it made, leaves the scene on the previous snapshot, counts
//!   a deferral, and returns the error; the caller keeps the regions and retries.
//! - **Retirement:** replaced mesh buffers, BLAS and top level, and descriptor pools handed to
//!   [`GpuScene::retire`], wait for the timeline value of the last submission that may read them,
//!   and are freed by [`GpuScene::collect`] once it completes.
//! - The acceleration update waits for its build (as 2D's build does), which also waits for every
//!   earlier submission. The mesh-only path (no ray tracing) does not wait.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use ash::vk;
use derived::{Pipeline, ReadError, ReaderToken};
use world::dims::BRICK_EDGE;
use world::{BrickKey, VoxelCoord};

use crate::accel::{Accel, AccelFaults, AccelGarbage, BuildStats};
use crate::alloc::{Allocator, Buffer};
use crate::context::{Gpu, Result};
use crate::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use crate::mesh::{GpuMeshes, GpuRegion};
use crate::staging::Uploader;
use crate::timeline::{Retirement, Timeline};

/// Planted faults for negative controls. `Default` is correct.
#[derive(Clone, Copy, Debug, Default)]
pub struct SceneFaults {
    /// This region is left out of every update (a missed-dirty bug).
    pub skip_region: Option<RegionKey>,
    /// Replaced resources are freed at the swap instead of after retirement (a lifetime bug).
    pub free_at_swap: bool,
}

/// Something the GPU may still read.
pub enum Garbage {
    Buffer(Buffer),
    Accel(AccelGarbage),
    Pool(vk::DescriptorPool),
}

/// What one update did and what it cost on the host.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    /// Regions in the change set, uploaded (present in the new snapshot), and removed (now empty).
    pub regions_changed: usize,
    pub regions_uploaded: usize,
    pub regions_removed: usize,
    pub upload_bytes: u64,
    /// Buffer creation and copies into the staging ring, including the flush.
    pub upload_ms: f64,
    /// The acceleration update, including its wait for the build.
    pub accel_ms: f64,
    pub accel: Option<BuildStats>,
}

pub struct GpuScene {
    pub meshes: GpuMeshes,
    pub accel: Option<Accel>,
    /// The 1C snapshot (raw id) the scene shows.
    pub snapshot: u64,
    /// Per region: the snapshot it was last built from (the debug view's version view).
    built_at: BTreeMap<RegionKey, u64>,
    retiring: Retirement<Garbage>,
    pub faults: SceneFaults,
    pub accel_faults: AccelFaults,
    pub updates: u64,
    /// Updates refused (the previous snapshot stayed visible).
    pub deferred: u64,
}

/// The acceleration regions containing `keys`.
pub fn affected_regions(keys: impl IntoIterator<Item = BrickKey>, size: RegionSize) -> BTreeSet<RegionKey> {
    keys.into_iter().map(|k| RegionKey::of(k, size)).collect()
}

/// Every brick key inside region `r`.
pub fn region_bricks(r: RegionKey, size: RegionSize) -> impl Iterator<Item = BrickKey> {
    let o = r.origin(size);
    let e = size.bricks();
    (0..e).flat_map(move |z| (0..e).flat_map(move |y| (0..e).map(move |x| VoxelCoord::new(o.x + x * BRICK_EDGE, o.y + y * BRICK_EDGE, o.z + z * BRICK_EDGE).split().0)))
}

/// Rebuilds each of `regions` from the snapshot `token` reads. `None`: the region has no quads now.
pub fn region_meshes(pipeline: &Pipeline, token: &ReaderToken, regions: &BTreeSet<RegionKey>, size: RegionSize) -> std::result::Result<BTreeMap<RegionKey, Option<RegionMesh>>, ReadError> {
    let mut out = BTreeMap::new();
    for &r in regions {
        let mut bricks = Vec::new();
        for k in region_bricks(r, size) {
            if let Some(m) = pipeline.read(token, k)? {
                bricks.push((k, m));
            }
        }
        let mut built = build_regions(bricks, size);
        debug_assert!(built.keys().all(|&k| k == r));
        out.insert(r, built.remove(&r));
    }
    Ok(out)
}

impl GpuScene {
    /// Uploads every region, and builds the acceleration structure when `ray` is set (a ray-tracing
    /// device is needed then). All-or-nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn build(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, size: RegionSize, regions: &BTreeMap<RegionKey, RegionMesh>, snapshot: u64, ray: bool) -> Result<GpuScene> {
        let (meshes, uploaded) = GpuMeshes::upload(gpu, alloc, up, timeline, size, regions)?;
        let accel = if ray {
            match Accel::build(gpu, alloc, up, timeline, &meshes, AccelFaults::default()) {
                Ok(a) => Some(a),
                Err(e) => {
                    timeline.wait(gpu, uploaded, u64::MAX)?;
                    meshes.free_now(gpu, alloc);
                    return Err(e);
                }
            }
        } else {
            None
        };
        let built_at = meshes.regions.keys().map(|&k| (k, snapshot)).collect();
        Ok(GpuScene { meshes, accel, snapshot, built_at, retiring: Retirement::default(), faults: SceneFaults::default(), accel_faults: AccelFaults::default(), updates: 0, deferred: 0 })
    }

    pub fn size(&self) -> RegionSize {
        self.meshes.size
    }

    /// Applies `changed` (from [`region_meshes`]) as snapshot `snapshot`. On error nothing changed:
    /// the scene still shows its previous snapshot, and the deferral is counted.
    pub fn update(&mut self, gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, mut changed: BTreeMap<RegionKey, Option<RegionMesh>>, snapshot: u64) -> Result<UpdateStats> {
        if let Some(k) = self.faults.skip_region {
            changed.remove(&k);
        }
        let mut stats = UpdateStats { regions_changed: changed.len(), ..UpdateStats::default() };
        let keys: BTreeSet<RegionKey> = changed.keys().copied().collect();
        let present: BTreeMap<RegionKey, RegionMesh> = changed.into_iter().filter_map(|(k, m)| m.map(|m| (k, m))).collect();
        stats.regions_uploaded = present.len();
        stats.regions_removed = keys.iter().filter(|k| !present.contains_key(k) && self.meshes.regions.contains_key(k)).count();

        // 1. New buffers for the changed regions (all-or-nothing).
        let t = Instant::now();
        let (fresh, uploaded) = match GpuMeshes::upload_regions(gpu, alloc, up, timeline, &present) {
            Ok(x) => x,
            Err(e) => {
                self.deferred += 1;
                return Err(e);
            }
        };
        stats.upload_bytes = fresh.values().map(|r| r.sections.total).sum();
        stats.upload_ms = t.elapsed().as_secs_f64() * 1e3;

        // 2. The candidate region set: replaced regions come out, fresh ones go in.
        let mut old: Vec<GpuRegion> = keys.iter().filter_map(|k| self.meshes.regions.remove(k)).collect();
        self.meshes.regions.extend(fresh);

        // 3. The acceleration structure over the candidate set.
        let t = Instant::now();
        let mut garbage = None;
        if let Some(accel) = &mut self.accel {
            match accel.update(gpu, alloc, up, timeline, &self.meshes, &keys, self.accel_faults) {
                Ok((s, g)) => {
                    stats.accel = Some(s);
                    garbage = Some(g);
                }
                Err(e) => {
                    // Roll back: the fresh buffers were never visible; their copies may be in flight.
                    let wait = timeline.wait(gpu, uploaded, u64::MAX);
                    for k in &keys {
                        if let Some(r) = self.meshes.regions.remove(k) {
                            alloc.free(gpu, r.buffer);
                        }
                    }
                    for r in old.drain(..) {
                        self.meshes.regions.insert(r.key, r);
                    }
                    self.deferred += 1;
                    wait?;
                    return Err(e);
                }
            }
        }
        stats.accel_ms = t.elapsed().as_secs_f64() * 1e3;

        // 4. Swap done: retire what the new snapshot no longer uses.
        let last = timeline.last_signal();
        for r in old {
            self.retire(last, Garbage::Buffer(r.buffer));
        }
        if let Some(g) = garbage {
            self.retire(last, Garbage::Accel(g));
        }
        if self.faults.free_at_swap {
            self.collect(gpu, alloc, u64::MAX);
        }
        for k in &keys {
            if self.meshes.regions.contains_key(k) {
                self.built_at.insert(*k, snapshot);
            } else {
                self.built_at.remove(k);
            }
        }
        self.snapshot = snapshot;
        self.updates += 1;
        Ok(stats)
    }

    /// Hands `g` to retirement: freed once the timeline passes `value`.
    pub fn retire(&mut self, value: u64, g: Garbage) {
        self.retiring.push(value, g);
    }

    /// Frees everything retired at or before `completed`.
    pub fn collect(&mut self, gpu: &Gpu, alloc: &mut Allocator, completed: u64) {
        for g in self.retiring.collect(completed) {
            match g {
                Garbage::Buffer(b) => alloc.free(gpu, b),
                Garbage::Accel(a) => self.accel.as_ref().expect("accel garbage comes from the accel").free_garbage(gpu, alloc, a),
                Garbage::Pool(p) => unsafe { gpu.device.destroy_descriptor_pool(p, None) },
            }
        }
    }

    pub fn retiring_len(&self) -> usize {
        self.retiring.len()
    }

    /// (region key, snapshot it was built from), in region index order.
    pub fn region_rows(&self) -> Vec<(RegionKey, u64)> {
        self.meshes.regions.keys().map(|k| (*k, self.built_at[k])).collect()
    }

    /// Device bytes the scene holds now: mesh buffers (`GpuMesh`) and acceleration (`GpuAccel`).
    pub fn device_bytes(&self) -> (u64, u64) {
        let mesh = self.meshes.regions.values().map(|r| r.buffer.range().2).sum();
        (mesh, self.accel.as_ref().map_or(0, Accel::device_bytes))
    }

    /// Frees everything now, retired items included. Only when the GPU is idle.
    pub fn free_now(mut self, gpu: &Gpu, alloc: &mut Allocator) {
        self.collect(gpu, alloc, u64::MAX);
        if let Some(a) = self.accel {
            a.free_now(gpu, alloc);
        }
        self.meshes.free_now(gpu, alloc);
    }
}
