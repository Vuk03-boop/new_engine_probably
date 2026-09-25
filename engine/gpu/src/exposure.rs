//! Phase 4B: the sums behind the viewer's automatic exposure (`shaders/exposure.slang`; Phase 4
//! decision 3). [`GROUPS`] workgroups read every [`STRIDE`]-th pixel in x and y of the shown radiance (after
//! the filter), adds the material's emitted radiance on surface pixels when the lights are on (what the
//! light view shows), and writes `light::exposure::LogSums` into one slot of a small host-visible
//! buffer. The host reads a slot once the frame that wrote it has completed, and turns it into an
//! exposure with `light::exposure::exposure_from_sums` and `light::exposure::Adaptation`.

use std::mem::size_of;

use ash::vk;
use light::exposure::LogSums;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::debug_view::{MaterialRow, Tables};
use crate::raster::Targets;
use crate::reflect::{self, Field, Param};

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/exposure.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/exposure.json"));

/// Pixels read: every `STRIDE`-th in x and y (1080p: 480 x 270).
pub const STRIDE: u32 = 4;
/// Workgroups per frame, each writing a partial sum the host adds. One workgroup took 1.24 ms at
/// 1080p on the RTX 3050 (one multiprocessor doing every load in series).
pub const GROUPS: u32 = 64;

const F_EMISSION: u32 = 1;

/// Host mirror of the shader's push constants (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub slot: u32,
    pub floor: f32,
    pub flags: u32,
    pub material_count: u32,
    pub groups: u32,
}

const _: () = assert!(size_of::<Params>() == 32);

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "depth_tex", binding: 0, element: vec![] },
        Param::Descriptor { name: "material_tex", binding: 1, element: vec![] },
        Param::Descriptor { name: "radiance", binding: 2, element: vec![] },
        Param::Descriptor { name: "materials", binding: 3, element: vec![field!(MaterialRow, base), field!(MaterialRow, emissive)] },
        Param::Descriptor { name: "sums", binding: 4, element: vec![] },
        Param::PushConstants {
            name: "params",
            fields: vec![field!(Params, width), field!(Params, height), field!(Params, stride), field!(Params, slot), field!(Params, floor), field!(Params, flags), field!(Params, material_count), field!(Params, groups)],
        },
    ]
}

/// The pixels the pass reads for a `width` x `height` frame, row-major indices, in the shader's order.
pub fn sample_pixels(width: u32, height: u32) -> Vec<usize> {
    let s = STRIDE;
    let (nx, ny) = ((width + s / 2) / s, (height + s / 2) / s);
    (0..nx * ny).map(|j| ((j / nx) * s + s / 2) as usize * width as usize + ((j % nx) * s + s / 2) as usize).collect()
}

/// The host-visible sums, one slot per frame in flight ([`GROUPS`] partial sums of 16 B each,
/// `GpuTemporal`).
pub struct ExposureTargets {
    pub sums: Buffer,
    pub slots: u32,
}

impl ExposureTargets {
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, slots: u32) -> Result<ExposureTargets> {
        let mut sums = alloc.create_buffer(gpu, 16 * (GROUPS * slots.max(1)) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, Category::GpuTemporal, Kind::Host)?;
        sums.mapped().expect("host buffer").fill(0);
        Ok(ExposureTargets { sums, slots: slots.max(1) })
    }

    /// The sums of `slot` (its groups' partial sums added in f64). Only after the frame that wrote it
    /// has completed.
    pub fn read(&self, slot: u32) -> LogSums {
        let m = self.sums.mapped_ref().expect("host buffer");
        let mut s = LogSums::default();
        for g in 0..GROUPS as usize {
            let at = 16 * (slot as usize * GROUPS as usize + g);
            let f = |k: usize| f32::from_le_bytes(m[at + 4 * k..at + 4 * k + 4].try_into().unwrap()) as f64;
            s.sum += f(0);
            s.count += f(1);
            s.sum_log += f(2);
            s.lit += f(3);
        }
        s
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.sums);
    }
}

pub struct ExposureBindings {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl ExposureBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }

    pub fn into_pool(self) -> vk::DescriptorPool {
        self.pool
    }
}

pub struct Exposure {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Exposure {
    pub fn new(gpu: &Gpu) -> Result<Exposure> {
        Self::with_reflection(gpu, REFLECTION)
    }

    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Exposure> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [
            b(0, vk::DescriptorType::SAMPLED_IMAGE),
            b(1, vk::DescriptorType::SAMPLED_IMAGE),
            b(2, vk::DescriptorType::STORAGE_BUFFER),
            b(3, vk::DescriptorType::STORAGE_BUFFER),
            b(4, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let set_layout = unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.vk("vkCreateDescriptorSetLayout")?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(size_of::<Params>() as u32)];
        let sl = [set_layout];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc), None) }.vk("vkCreatePipelineLayout")?;
        let (chunks, rest) = SPIRV.as_chunks::<4>();
        assert!(rest.is_empty(), "SPIR-V is whole words");
        let words: Vec<u32> = chunks.iter().map(|&c| u32::from_le_bytes(c)).collect();
        let module = unsafe { dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.vk("vkCreateShaderModule")?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { dev.destroy_shader_module(module, None) };
        let pipeline = pipeline.map_err(|(_, e)| GpuError::Vk { call: "vkCreateComputePipelines", result: e })?[0];
        Ok(Exposure { set_layout, layout, pipeline })
    }

    /// Binds the G-buffer's depth and material, the shown radiance, the material rows (their emitted
    /// radiance) and the sums. Rebind after a resize.
    pub fn bind(&self, gpu: &Gpu, targets: &Targets, radiance: &Buffer, tables: &Tables, out: &ExposureTargets) -> Result<ExposureBindings> {
        let dev = &gpu.device;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 2 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = [self.set_layout];
        let set = match unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets") {
            Ok(s) => s[0],
            Err(e) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                return Err(e);
            }
        };
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        // Targets: depth, material (colour 1), as `DebugView::bind` orders them.
        let images = [img(targets.depth.view), img(targets.color[1].view)];
        let whole = |b: &Buffer| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let bufs = [whole(radiance), whole(&tables.materials), whole(&out.sums)];
        let mut writes: Vec<_> = images.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(info)).collect();
        writes.extend(bufs.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2 + i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)));
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(ExposureBindings { pool, set })
    }

    /// Records the pass into `slot`. The targets must be readable and the radiance written by earlier
    /// compute passes in this command buffer (made visible here); the sums are made visible to the host.
    /// `floor`: pixels below it are left out of the log sum (the previous frame's mean x 2⁻¹⁰; 0 at first).
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &ExposureBindings, out: &ExposureTargets, tables: &Tables, width: u32, height: u32, slot: u32, floor: f64, emission: bool) {
        assert!(slot < out.slots, "exposure slot {slot} of {}", out.slots);
        let p = Params { width, height, stride: STRIDE, slot, floor: floor as f32, flags: if emission { F_EMISSION } else { 0 }, material_count: tables.material_count, groups: GROUPS };
        let dev = &gpu.device;
        let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        unsafe {
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&before));
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.set], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, GROUPS, 1, 1);
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
        }
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4B C1: the exposure module's reflection matches the host, without a device; a moved field is
    /// refused.
    #[test]
    fn host_layout_matches_the_compiled_module() {
        reflect::check(REFLECTION, &host_layout()).unwrap();
        let mut bad = host_layout();
        if let Some(Param::PushConstants { fields, .. }) = bad.last_mut() {
            fields[4].offset += 4;
        }
        assert!(reflect::check(REFLECTION, &bad).is_err());
    }

    #[test]
    fn samples_are_every_stride_th_pixel() {
        let px = sample_pixels(1920, 1080);
        assert_eq!(px.len(), 480 * 270);
        assert_eq!(px[0], 2 * 1920 + 2);
        assert_eq!(px[1], 2 * 1920 + 6);
        assert_eq!(*px.last().unwrap(), 1078 * 1920 + 1918);
    }
}
