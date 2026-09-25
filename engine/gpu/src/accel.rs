//! Phase 2D: the ray-query representation of uploaded region meshes (ADR-0003, ADR-0004).
//!
//! - **One BLAS per region**, built straight from the region's device buffer: the same vertices
//!   (`R16G16B16A16_SFLOAT`, stride 8; `w` is ignored by the build) and indices the raster pass
//!   draws, so both paths see the same committed LoD. Geometry is opaque; builds prefer fast trace.
//! - **One TLAS** with one instance per region, in region key order: the transform translates by the
//!   region origin (world voxels), and the instance custom index is the region index, the first half
//!   of the ADR-0003 surface id. Instances carry no facing flag: Vulkan's default facing matches
//!   ADR-0004's winding (checked by the planted flipped-facing control).
//! - **Region table:** per region, the device addresses of its quad table and `tri_quad` map and
//!   their lengths, so the trace shader resolves (region, primitive) to the quad's material and face
//!   like raster does, and never reads past a table on a wrong id.
//! - **Memory:** BLAS, TLAS, instance and table buffers take `Category::GpuAccel` grants; build
//!   scratch takes `Category::GpuAccelScratch` and is freed when the build completes. Building is
//!   all-or-nothing: on a refused grant everything made so far is freed and nothing is recorded.
//! - BLAS builds are batched so that one batch's scratch stays under [`SCRATCH_BATCH_BYTES`]; batches
//!   reuse one scratch buffer, separated by a build-to-build barrier.
//! - **Updates (2E):** [`Accel::update`] rebuilds the BLAS of the changed regions only, and a new
//!   TLAS, instance buffer and region table over every region. It is all-or-nothing like a build.
//!   On success the replaced BLAS and the old top level come back as [`AccelGarbage`], to be freed
//!   after the last submission that may read them ([`Accel::free_garbage`]).
//!
//! [`Accel::free_now`] frees immediately: the caller must make sure the GPU is done with it.

use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;
use std::time::Instant;

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::layout::RegionKey;
use crate::mesh::GpuMeshes;
use crate::raster::VERTEX_FORMAT;
use crate::staging::Uploader;
use crate::submit::Submitter;
use crate::timeline::Timeline;
use crate::timing::GpuTimer;

/// Upper bound on one BLAS batch's scratch. A single build that needs more gets a batch of its own.
pub const SCRATCH_BATCH_BYTES: u64 = 32 << 20;

/// Host mirror of the trace shader's `RegionRef`: two device addresses and the bounds of both tables.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionRef {
    pub quads: u64,
    pub tri_quad: u64,
    pub quad_count: u32,
    pub tri_count: u32,
}

/// Planted faults for negative controls. `Default` builds correctly.
#[derive(Clone, Copy, Debug, Default)]
pub struct AccelFaults {
    /// Added to every instance translation (a transform bug).
    pub offset_error: [i32; 3],
    /// Region (by index) left out of the TLAS (a missing-instance bug).
    pub skip_region: Option<usize>,
    /// Instances with the front-counter-clockwise flag, which inverts the facing (a winding/culling bug).
    pub flip_facing: bool,
    /// Added to every instance custom index, modulo the region count (a wrong-id bug).
    pub custom_index_shift: u32,
}

/// What a build (or update) made and what it cost. BLAS figures cover the BLAS built by this call:
/// all of them for a build, the changed regions' for an update.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildStats {
    pub blas_count: u64,
    pub triangles: u64,
    /// Sum of the driver's sizes (`accelerationStructureSize`) of the BLAS built.
    pub blas_bytes: u64,
    pub tlas_bytes: u64,
    pub instance_bytes: u64,
    pub table_bytes: u64,
    /// Device bytes of the `GpuAccel` buffers this call made, as the allocator placed them (driver
    /// alignment included). For a build, that is all of them.
    pub accel_buffer_bytes: u64,
    /// Regions listed in the TLAS.
    pub instances: u64,
    /// Size of the scratch buffer used for the build (freed afterwards).
    pub scratch_bytes: u64,
    pub blas_batches: u64,
    /// GPU time of all BLAS builds, and of the TLAS build (timestamps; not nested, not summed).
    pub blas_gpu_ms: f64,
    pub tlas_gpu_ms: f64,
    /// Host time from the first size query to the recorded submission (size queries, allocation,
    /// instance and table uploads, recording).
    pub host_ms: f64,
}

struct Blas {
    accel: vk::AccelerationStructureKHR,
    buffer: Buffer,
}

/// The top level: rebuilt over every region by each build or update.
struct Top {
    tlas: vk::AccelerationStructureKHR,
    tlas_buffer: Buffer,
    instances: Buffer,
    table: Buffer,
    regions: u32,
}

pub struct Accel {
    loader: ash::khr::acceleration_structure::Device,
    blas: BTreeMap<RegionKey, Blas>,
    top: Top,
    pub stats: BuildStats,
}

/// Structures and buffers an update replaced. Free them with [`Accel::free_garbage`] once no
/// submission can read them.
#[derive(Default)]
pub struct AccelGarbage {
    structures: Vec<(vk::AccelerationStructureKHR, Buffer)>,
    buffers: Vec<Buffer>,
}

impl AccelGarbage {
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty() && self.buffers.is_empty()
    }
}

fn blas_info<'a>(g: &'a [vk::AccelerationStructureGeometryKHR<'a>]) -> vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
    vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(g)
}

fn align_up(x: u64, a: u64) -> u64 {
    x.next_multiple_of(a.max(1))
}

/// Things made so far during a build; freed together if a later step fails.
struct Made {
    buffers: Vec<Buffer>,
    structures: Vec<vk::AccelerationStructureKHR>,
}

impl Made {
    fn undo(self, gpu: &Gpu, alloc: &mut Allocator, loader: &ash::khr::acceleration_structure::Device) {
        for s in self.structures {
            unsafe { loader.destroy_acceleration_structure(s, None) };
        }
        for b in self.buffers {
            alloc.free(gpu, b);
        }
    }
}

/// Builds the BLAS of `rebuild` (keys of `meshes`) and a top level over every region of `meshes`,
/// taking the other regions' BLAS from `keep`. Waits for the build. All-or-nothing.
#[allow(clippy::too_many_arguments)]
fn build_set(
    gpu: &Gpu,
    alloc: &mut Allocator,
    up: &mut Uploader,
    timeline: &mut Timeline,
    loader: &ash::khr::acceleration_structure::Device,
    meshes: &GpuMeshes,
    rebuild: &BTreeSet<RegionKey>,
    keep: &BTreeMap<RegionKey, Blas>,
    faults: AccelFaults,
) -> Result<(BTreeMap<RegionKey, Blas>, Top, BuildStats)> {
    let t0 = Instant::now();
    let dev = &gpu.device;
    let mut asp = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut asp);
    unsafe { gpu.instance.get_physical_device_properties2(gpu.physical, &mut p2) };
    let scratch_align = asp.min_acceleration_structure_scratch_offset_alignment as u64;

    let regions: Vec<_> = meshes.regions.values().collect();
    let n = regions.len() as u32;
    let built: Vec<usize> = (0..regions.len()).filter(|&i| rebuild.contains(&regions[i].key)).collect();
    assert_eq!(built.len(), rebuild.len(), "every rebuilt region is in the meshes");
    assert!(regions.iter().all(|r| rebuild.contains(&r.key) || keep.contains_key(&r.key)), "every region has a BLAS");

    // 1. BLAS geometry and sizes, for the rebuilt regions.
    let geoms: Vec<vk::AccelerationStructureGeometryKHR> = built
        .iter()
        .map(|&i| {
            let r = regions[i];
            let a = r.buffer.address;
            let tri = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(VERTEX_FORMAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: a + r.sections.vertices })
                .vertex_stride(8)
                .max_vertex(r.vertex_count.saturating_sub(1) as u32)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR { device_address: a + r.sections.indices });
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri })
                .flags(vk::GeometryFlagsKHR::OPAQUE)
        })
        .collect();
    let sizes: Vec<vk::AccelerationStructureBuildSizesInfoKHR> = built
        .iter()
        .zip(&geoms)
        .map(|(&i, g)| {
            let mut s = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe { loader.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &blas_info(std::slice::from_ref(g)), &[regions[i].triangle_count as u32], &mut s) };
            s
        })
        .collect();

    // 2. Every buffer and structure, before anything is recorded (all-or-nothing).
    let mut made = Made { buffers: Vec::new(), structures: Vec::new() };
    macro_rules! attempt {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    made.undo(gpu, alloc, loader);
                    return Err(e);
                }
            }
        };
    }
    let as_usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
    let mut new_blas = Vec::with_capacity(built.len());
    for s in &sizes {
        let buffer = attempt!(alloc.create_buffer(gpu, s.acceleration_structure_size, as_usage, Category::GpuAccel, Kind::Device));
        let info = vk::AccelerationStructureCreateInfoKHR::default().buffer(buffer.buffer).size(s.acceleration_structure_size).ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
        let accel = unsafe { loader.create_acceleration_structure(&info, None) }.vk("vkCreateAccelerationStructureKHR");
        made.buffers.push(buffer);
        let accel = attempt!(accel);
        made.structures.push(accel);
        new_blas.push(accel);
    }
    let instance_bytes = (n.max(1) as u64) * size_of::<vk::AccelerationStructureInstanceKHR>() as u64;
    let instances = attempt!(alloc.create_buffer(gpu, instance_bytes, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_DST, Category::GpuAccel, Kind::Device));
    let instances_address = instances.address;
    made.buffers.push(instances);
    let table_bytes = (n.max(1) as u64) * size_of::<RegionRef>() as u64;
    let table = attempt!(alloc.create_buffer(gpu, table_bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Category::GpuAccel, Kind::Device));
    made.buffers.push(table);

    // TLAS sizes: one instance per region (fewer with a planted skip).
    let listed: Vec<usize> = (0..regions.len()).filter(|&i| faults.skip_region != Some(i)).collect();
    let tlas_geom = [vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().array_of_pointers(false).data(vk::DeviceOrHostAddressConstKHR { device_address: instances_address }),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE)];
    let tlas_info = || {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&tlas_geom)
    };
    let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe { loader.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &tlas_info(), &[listed.len() as u32], &mut tlas_sizes) };
    let tlas_buffer = attempt!(alloc.create_buffer(gpu, tlas_sizes.acceleration_structure_size, as_usage, Category::GpuAccel, Kind::Device));
    let info = vk::AccelerationStructureCreateInfoKHR::default().buffer(tlas_buffer.buffer).size(tlas_sizes.acceleration_structure_size).ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
    let tlas = unsafe { loader.create_acceleration_structure(&info, None) }.vk("vkCreateAccelerationStructureKHR");
    made.buffers.push(tlas_buffer);
    let tlas = attempt!(tlas);
    made.structures.push(tlas);

    // Scratch: BLAS batches, each within SCRATCH_BATCH_BYTES (or one oversized build).
    let mut batches: Vec<Vec<(usize, u64)>> = Vec::new(); // (index into `built`, scratch offset)
    let mut batch_bytes = 0u64;
    let mut scratch_need = align_up(tlas_sizes.build_scratch_size, scratch_align);
    for (j, s) in sizes.iter().enumerate() {
        let need = align_up(s.build_scratch_size, scratch_align);
        if batches.is_empty() || (batch_bytes + need > SCRATCH_BATCH_BYTES && batch_bytes > 0) {
            batches.push(Vec::new());
            batch_bytes = 0;
        }
        batches.last_mut().expect("a batch").push((j, batch_bytes));
        batch_bytes += need;
        scratch_need = scratch_need.max(batch_bytes);
    }
    let scratch_bytes = scratch_need + scratch_align;
    let scratch = attempt!(alloc.create_buffer(gpu, scratch_bytes, vk::BufferUsageFlags::STORAGE_BUFFER, Category::GpuAccelScratch, Kind::Device));
    let scratch_base = align_up(scratch.address, scratch_align);

    // 3. Instance and table contents.
    let address = |a: vk::AccelerationStructureKHR| unsafe { loader.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(a)) };
    let mut blas_address = vec![0u64; regions.len()];
    for (j, &i) in built.iter().enumerate() {
        blas_address[i] = address(new_blas[j]);
    }
    for (i, r) in regions.iter().enumerate() {
        if !rebuild.contains(&r.key) {
            blas_address[i] = address(keep[&r.key].accel);
        }
    }
    // No facing flag: Vulkan's default ray-triangle facing already treats ADR-0004's winding
    // (counter-clockwise from outside, right-handed Y-up) as front-facing. The first 2D run set
    // TRIANGLE_FRONT_COUNTERCLOCKWISE and hit only back faces; that is now the planted control.
    let flags = if faults.flip_facing { vk::GeometryInstanceFlagsKHR::TRIANGLE_FRONT_COUNTERCLOCKWISE } else { vk::GeometryInstanceFlagsKHR::empty() };
    let mut inst_bytes = Vec::with_capacity(listed.len() * size_of::<vk::AccelerationStructureInstanceKHR>());
    for &i in &listed {
        let o = regions[i].key.origin(meshes.size);
        let t = [o.x + faults.offset_error[0], o.y + faults.offset_error[1], o.z + faults.offset_error[2]].map(|v| v as f32);
        let inst = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR { matrix: [1.0, 0.0, 0.0, t[0], 0.0, 1.0, 0.0, t[1], 0.0, 0.0, 1.0, t[2]] },
            instance_custom_index_and_mask: vk::Packed24_8::new((i as u32 + faults.custom_index_shift) % n, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, flags.as_raw() as u8),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: blas_address[i] },
        };
        inst_bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(&inst as *const _ as *const u8, size_of::<vk::AccelerationStructureInstanceKHR>()) });
    }
    let mut table_data = Vec::with_capacity(regions.len() * size_of::<RegionRef>());
    for r in &regions {
        let rr = RegionRef { quads: r.buffer.address + r.sections.quads, tri_quad: r.buffer.address + r.sections.tri_quad, quad_count: r.quad_count as u32, tri_count: r.triangle_count as u32 };
        table_data.extend_from_slice(&rr.quads.to_le_bytes());
        table_data.extend_from_slice(&rr.tri_quad.to_le_bytes());
        table_data.extend_from_slice(&rr.quad_count.to_le_bytes());
        table_data.extend_from_slice(&rr.tri_count.to_le_bytes());
    }
    let instances = &made.buffers[made.buffers.len() - 3];
    let table_buf = &made.buffers[made.buffers.len() - 2];
    let uploaded = (|| {
        if !inst_bytes.is_empty() {
            up.upload(gpu, timeline, instances, 0, &inst_bytes)?;
        }
        if !table_data.is_empty() {
            up.upload(gpu, timeline, table_buf, 0, &table_data)?;
        }
        up.flush(gpu, timeline)
    })();
    if let Err(e) = uploaded {
        alloc.free(gpu, scratch);
        made.undo(gpu, alloc, loader);
        return Err(e);
    }

    // 4. Record: BLAS batches, then the TLAS; timestamps around each.
    let tools = GpuTimer::new(gpu, 1, 2).and_then(|t| Submitter::new(gpu).map(|s| (t, s)));
    let (mut timer, mut sub) = match tools {
        Ok(t) => t,
        Err(e) => {
            alloc.free(gpu, scratch);
            made.undo(gpu, alloc, loader);
            return Err(e);
        }
    };
    let mut host_ms = 0.0;
    let submitted = sub.begin(gpu, timeline).and_then(|cmd| {
        let build_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)];
        timer.reset(gpu, cmd, 0);
        timer.begin_pass(gpu, cmd, 0, 0);
        for batch in &batches {
            let infos: Vec<_> = batch
                .iter()
                .map(|&(j, off)| blas_info(std::slice::from_ref(&geoms[j])).dst_acceleration_structure(new_blas[j]).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_base + off }))
                .collect();
            let ranges: Vec<[vk::AccelerationStructureBuildRangeInfoKHR; 1]> =
                batch.iter().map(|&(j, _)| [vk::AccelerationStructureBuildRangeInfoKHR { primitive_count: regions[built[j]].triangle_count as u32, primitive_offset: 0, first_vertex: 0, transform_offset: 0 }]).collect();
            let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = ranges.iter().map(|r| &r[..]).collect();
            unsafe {
                loader.cmd_build_acceleration_structures(cmd, &infos, &range_refs);
                dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&build_barrier));
            }
        }
        timer.end_pass(gpu, cmd, 0, 0);
        timer.begin_pass(gpu, cmd, 0, 1);
        let info = [tlas_info().dst_acceleration_structure(tlas).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_base })];
        let range = [vk::AccelerationStructureBuildRangeInfoKHR { primitive_count: listed.len() as u32, primitive_offset: 0, first_vertex: 0, transform_offset: 0 }];
        unsafe { loader.cmd_build_acceleration_structures(cmd, &info, &[&range[..]]) };
        timer.end_pass(gpu, cmd, 0, 1);
        // The TLAS (and the BLAS it references) are read by ray queries in later submissions.
        let to_trace = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];
        unsafe { dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&to_trace)) };
        host_ms = t0.elapsed().as_secs_f64() * 1e3;
        sub.submit(gpu, timeline, cmd, &[])
    });
    let ms = submitted.and_then(|v| timeline.wait(gpu, v, u64::MAX)).and_then(|_| timer.read(gpu, 0));
    // On failure the GPU may still hold the scratch (a failed wait): drain it before freeing.
    let drained = if ms.is_err() { gpu.wait_idle() } else { Ok(()) };
    timer.destroy(gpu);
    sub.destroy(gpu);
    alloc.free(gpu, scratch);
    let ms = match ms.and_then(|m| drained.map(|()| m)) {
        Ok(m) => m.expect("slot written"),
        Err(e) => {
            made.undo(gpu, alloc, loader);
            return Err(e);
        }
    };

    let Made { mut buffers, .. } = made;
    let tlas_buffer = buffers.pop().expect("tlas buffer");
    let table = buffers.pop().expect("table buffer");
    let instances = buffers.pop().expect("instance buffer");
    let accel_buffer_bytes = buffers.iter().chain([&tlas_buffer, &table, &instances]).map(|b| b.range().2).sum();
    let stats = BuildStats {
        blas_count: built.len() as u64,
        triangles: built.iter().map(|&i| regions[i].triangle_count).sum(),
        blas_bytes: sizes.iter().map(|s| s.acceleration_structure_size).sum(),
        tlas_bytes: tlas_sizes.acceleration_structure_size,
        instance_bytes,
        table_bytes,
        accel_buffer_bytes,
        instances: listed.len() as u64,
        scratch_bytes,
        blas_batches: batches.len() as u64,
        blas_gpu_ms: ms[0],
        tlas_gpu_ms: ms[1],
        host_ms,
    };
    let blas = built.iter().zip(new_blas).zip(buffers).map(|((&i, accel), buffer)| (regions[i].key, Blas { accel, buffer })).collect();
    Ok((blas, Top { tlas, tlas_buffer, instances, table, regions: n }, stats))
}

impl Accel {
    /// Builds the BLAS of every region and the TLAS, and waits for the build to complete (the build
    /// waits for earlier uploads through queue order). Needs a ray-tracing device.
    pub fn build(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, meshes: &GpuMeshes, faults: AccelFaults) -> Result<Accel> {
        if !gpu.ray_tracing() {
            return Err(GpuError::NoDevice("acceleration structures need a ray-tracing device".into()));
        }
        let loader = ash::khr::acceleration_structure::Device::new(&gpu.instance, &gpu.device);
        let all: BTreeSet<RegionKey> = meshes.regions.keys().copied().collect();
        let (blas, top, stats) = build_set(gpu, alloc, up, timeline, &loader, meshes, &all, &BTreeMap::new(), faults)?;
        Ok(Accel { loader, blas, top, stats })
    }

    /// Rebuilds the BLAS of the `changed` regions that `meshes` holds (their buffers must be the new
    /// ones), drops the BLAS of `changed` regions it no longer holds, and rebuilds the top level over
    /// every region of `meshes`. Waits for the build. All-or-nothing: on error `self` is unchanged.
    /// On success returns the stats and the replaced resources.
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, meshes: &GpuMeshes, changed: &BTreeSet<RegionKey>, faults: AccelFaults) -> Result<(BuildStats, AccelGarbage)> {
        let rebuild: BTreeSet<RegionKey> = changed.iter().filter(|k| meshes.regions.contains_key(k)).copied().collect();
        let (blas, top, stats) = build_set(gpu, alloc, up, timeline, &self.loader, meshes, &rebuild, &self.blas, faults)?;
        let mut garbage = AccelGarbage::default();
        for k in changed {
            if let Some(old) = self.blas.remove(k) {
                garbage.structures.push((old.accel, old.buffer));
            }
        }
        self.blas.extend(blas);
        let old = std::mem::replace(&mut self.top, top);
        garbage.structures.push((old.tlas, old.tlas_buffer));
        garbage.buffers.push(old.instances);
        garbage.buffers.push(old.table);
        self.stats = stats;
        Ok((stats, garbage))
    }

    /// Destroys what an update replaced. Only when no submission can still read it.
    pub fn free_garbage(&self, gpu: &Gpu, alloc: &mut Allocator, garbage: AccelGarbage) {
        for (a, b) in garbage.structures {
            unsafe { self.loader.destroy_acceleration_structure(a, None) };
            alloc.free(gpu, b);
        }
        for b in garbage.buffers {
            alloc.free(gpu, b);
        }
    }

    pub fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.top.tlas
    }

    /// The region table (one [`RegionRef`] per region, in key order) and its size in bytes.
    pub fn table(&self) -> (vk::Buffer, u64) {
        (self.top.table.buffer, self.top.table.size)
    }

    pub fn region_count(&self) -> u32 {
        self.top.regions
    }

    /// Regions that have a BLAS, in key order.
    pub fn blas_keys(&self) -> impl Iterator<Item = RegionKey> + '_ {
        self.blas.keys().copied()
    }

    /// Device bytes of every `GpuAccel` buffer held now (BLAS and top level), as placed.
    pub fn device_bytes(&self) -> u64 {
        let t = &self.top;
        self.blas.values().map(|b| b.buffer.range().2).sum::<u64>() + [&t.tlas_buffer, &t.instances, &t.table].iter().map(|b| b.range().2).sum::<u64>()
    }

    /// Destroys every structure and frees every buffer now. Only when the GPU is idle or done with them.
    pub fn free_now(self, gpu: &Gpu, alloc: &mut Allocator) {
        let Accel { loader, blas, top, .. } = self;
        unsafe { loader.destroy_acceleration_structure(top.tlas, None) };
        for b in blas.into_values() {
            unsafe { loader.destroy_acceleration_structure(b.accel, None) };
            alloc.free(gpu, b.buffer);
        }
        alloc.free(gpu, top.tlas_buffer);
        alloc.free(gpu, top.instances);
        alloc.free(gpu, top.table);
    }
}
