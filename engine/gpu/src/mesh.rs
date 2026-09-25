//! Region meshes on the device (ADR-0004), one buffer per region holding its four sections.
//!
//! Upload is all-or-nothing: every region buffer is created (ledger grants taken) before any copy
//! is recorded. If a grant is refused, the buffers made so far are freed (the GPU never saw them)
//! and the error is returned, so the caller keeps showing the previous meshes (ADR-0003: a
//! refused grant defers publication, never evicts or overcommits).

use std::collections::BTreeMap;

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, Result};
use crate::layout::{RegionKey, RegionMesh, RegionSize, Sections};
use crate::staging::Uploader;
use crate::timeline::{Retirement, Timeline};

/// Region buffer usage; acceleration-structure build input is added on RT devices (2D builds BLAS
/// straight from these buffers).
pub fn mesh_usage(gpu: &Gpu) -> vk::BufferUsageFlags {
    let base = vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
    if gpu.ray_tracing() { base | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR } else { base }
}

pub struct GpuRegion {
    pub key: RegionKey,
    pub sections: Sections,
    pub quad_count: u64,
    pub vertex_count: u64,
    pub triangle_count: u64,
    pub buffer: Buffer,
}

pub struct GpuMeshes {
    pub size: RegionSize,
    pub align: u64,
    pub regions: BTreeMap<RegionKey, GpuRegion>,
}

/// Section alignment for this device: at least 16 and the storage-buffer offset alignment.
pub fn section_align(gpu: &Gpu) -> u64 {
    gpu.info.min_storage_buffer_offset_alignment.max(16).next_power_of_two()
}

impl GpuMeshes {
    /// Uploads every region. Returns the timeline value after which the data is on the device.
    pub fn upload(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, size: RegionSize, regions: &BTreeMap<RegionKey, RegionMesh>) -> Result<(GpuMeshes, u64)> {
        let (out, done) = Self::upload_regions(gpu, alloc, up, timeline, regions)?;
        Ok((GpuMeshes { size, align: section_align(gpu), regions: out }, done))
    }

    /// Uploads `regions` into new buffers (2E: the changed regions of an edit). All-or-nothing on a
    /// refused grant. Returns the regions and the timeline value after which the data is on the device.
    pub fn upload_regions(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, regions: &BTreeMap<RegionKey, RegionMesh>) -> Result<(BTreeMap<RegionKey, GpuRegion>, u64)> {
        let align = section_align(gpu);
        let images: Vec<(RegionKey, Sections, Vec<u8>)> = regions.iter().map(|(&k, r)| {
            let (s, b) = r.image(align);
            (k, s, b)
        }).collect();
        // 1. Reserve everything.
        let mut made: Vec<Buffer> = Vec::with_capacity(images.len());
        for (_, s, _) in &images {
            match alloc.create_buffer(gpu, s.total, mesh_usage(gpu), Category::GpuMesh, Kind::Device) {
                Ok(b) => made.push(b),
                Err(e) => {
                    for b in made {
                        alloc.free(gpu, b);
                    }
                    return Err(e);
                }
            }
        }
        // 2. Upload.
        let mut out = BTreeMap::new();
        for ((k, s, bytes), buffer) in images.into_iter().zip(made) {
            up.upload(gpu, timeline, &buffer, 0, &bytes)?;
            let r = &regions[&k];
            out.insert(k, GpuRegion { key: k, sections: s, quad_count: r.quad_count(), vertex_count: r.vertex_count(), triangle_count: r.triangle_count(), buffer });
        }
        let done = up.flush(gpu, timeline)?;
        Ok((out, done))
    }

    pub fn quad_count(&self) -> u64 {
        self.regions.values().map(|r| r.quad_count).sum()
    }

    pub fn triangle_count(&self) -> u64 {
        self.regions.values().map(|r| r.triangle_count).sum()
    }

    pub fn device_bytes(&self) -> u64 {
        self.regions.values().map(|r| r.sections.total).sum()
    }

    /// Hands every buffer to `retire` at `value` (the last submission that may read them).
    pub fn retire(self, value: u64, retire: &mut Retirement<Buffer>) {
        for r in self.regions.into_values() {
            retire.push(value, r.buffer);
        }
    }

    /// Frees now. Only when the GPU is idle or has never used these buffers.
    pub fn free_now(self, gpu: &Gpu, alloc: &mut Allocator) {
        for r in self.regions.into_values() {
            alloc.free(gpu, r.buffer);
        }
    }
}
