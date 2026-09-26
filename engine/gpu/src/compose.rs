//! Phase 4B: emission after reconstruction and the exposure meter (`shaders/compose.slang`,
//! docs/changes/2026-09-26-phase4b-many-lights.md).
//!
//! While the lights are on, one compute pass after the temporal pass and the filter (or after the
//! shade pass when they are off) adds the emitter table's per-material emitted radiance to every
//! surface pixel of the shown radiance (ADR-0005 Amendment 3: emission is never accumulated,
//! demodulated or filtered), and meters the shown image for the viewer's automatic exposure: per
//! 16×16 workgroup the sums of [`light::exposure::Meter`], written to a host-visible buffer of the
//! frame slot. [`ComposeTargets::reading`] adds them in f64 once that frame has completed.

use std::mem::size_of;

use ash::vk;
use light::exposure::Meter;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::emitters::RefEmitters;
use crate::raster::Targets;
use crate::reflect::{self, Field, Param};
use crate::shade::ShadeTargets;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/compose.json"));

/// Pixels per workgroup side.
pub const GROUP: u32 = 16;

mod flags {
    pub const EMISSION: u32 = 1;
    pub const METER: u32 = 2;
    pub const FAULT_METER_WITHOUT_EMISSION: u32 = 256;
}

/// Planted faults for negative controls; all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComposeFaults {
    /// The meter reads the radiance before emission is added.
    pub meter_without_emission: bool,
}

/// Host mirror of the shader's push constants (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub flags: u32,
    pub floor: f32,
    pub groups_x: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

const _: () = assert!(size_of::<Params>() == 32);

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

/// What the host code assumes about the compose module.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "radiance", binding: 0, element: vec![] },
        Param::Descriptor { name: "depth_tex", binding: 1, element: vec![] },
        Param::Descriptor { name: "material_tex", binding: 2, element: vec![] },
        Param::Descriptor { name: "emission", binding: 3, element: vec![] },
        Param::Descriptor { name: "partials", binding: 4, element: vec![] },
        Param::PushConstants {
            name: "params",
            fields: vec![
                field!(Params, width),
                field!(Params, height),
                field!(Params, flags),
                field!(Params, floor),
                field!(Params, groups_x),
                field!(Params, pad0),
                field!(Params, pad1),
                field!(Params, pad2),
            ],
        },
    ]
}

/// The meter's partial sums, one host-visible buffer per frame slot (`GpuTemporal`).
pub struct ComposeTargets {
    pub width: u32,
    pub height: u32,
    pub groups: (u32, u32),
    partials: Vec<Buffer>,
}

impl ComposeTargets {
    /// `slots` buffers (the viewer: one per frame in flight). All-or-nothing.
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, width: u32, height: u32, slots: usize) -> Result<ComposeTargets> {
        let groups = (width.div_ceil(GROUP), height.div_ceil(GROUP));
        let bytes = (groups.0 * groups.1) as u64 * 16;
        let mut partials = Vec::new();
        for _ in 0..slots {
            match alloc.create_buffer(gpu, bytes, vk::BufferUsageFlags::STORAGE_BUFFER, Category::GpuTemporal, Kind::Host) {
                Ok(b) => partials.push(b),
                Err(e) => {
                    for b in partials {
                        alloc.free(gpu, b);
                    }
                    return Err(e);
                }
            }
        }
        Ok(ComposeTargets { width, height, groups, partials })
    }

    pub fn device_bytes(&self) -> u64 {
        self.partials.iter().map(|b| b.size).sum()
    }

    /// The meter's reading of slot `slot`, added in f64. Only after the frame that wrote it has
    /// completed (and only when that frame metered).
    pub fn reading(&self, slot: usize) -> Meter {
        let bytes = self.partials[slot].mapped_ref().expect("host-visible");
        let n = (self.groups.0 * self.groups.1) as usize;
        let mut m = Meter { pixels: (self.width * self.height) as u64, ..Meter::default() };
        for c in bytes[..n * 16].as_chunks::<16>().0 {
            let f = |k: usize| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()) as f64;
            m.sum_y += f(0);
            m.sum_ln += f(1);
            m.count += f(2) as u64;
        }
        m
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        for b in self.partials {
            alloc.free(gpu, b);
        }
    }
}

pub struct ComposeBindings {
    pool: vk::DescriptorPool,
    /// One set per slot of the targets.
    sets: Vec<vk::DescriptorSet>,
}

impl ComposeBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }

    pub fn into_pool(self) -> vk::DescriptorPool {
        self.pool
    }
}

pub struct Compose {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Compose {
    pub fn new(gpu: &Gpu) -> Result<Compose> {
        reflect::check(REFLECTION, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [b(0, vk::DescriptorType::STORAGE_BUFFER), b(1, vk::DescriptorType::SAMPLED_IMAGE), b(2, vk::DescriptorType::SAMPLED_IMAGE), b(3, vk::DescriptorType::STORAGE_BUFFER), b(4, vk::DescriptorType::STORAGE_BUFFER)];
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
        Ok(Compose { set_layout, layout, pipeline })
    }

    /// Binds the shown radiance, the G-buffer's depth and material, the table's per-material emission
    /// and every slot of `out`. Rebind after a resize or a scene update (the table changes).
    pub fn bind(&self, gpu: &Gpu, targets: &Targets, radiance: &ShadeTargets, em: &RefEmitters, out: &ComposeTargets) -> Result<ComposeBindings> {
        let dev = &gpu.device;
        let n = out.partials.len() as u32;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 2 * n }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 * n }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = vec![self.set_layout; n as usize];
        let sets = match unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets") {
            Ok(s) => s,
            Err(e) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                return Err(e);
            }
        };
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        let images = [img(targets.depth.view), img(targets.color[1].view)];
        for (set, part) in sets.iter().zip(&out.partials) {
            let bufs = [
                [vk::DescriptorBufferInfo { buffer: radiance.radiance.buffer, offset: 0, range: radiance.radiance.size }],
                // Exactly the table's materials, so the shader's bounds check sees their count.
                [vk::DescriptorBufferInfo { buffer: em.emission.buffer, offset: 0, range: (em.materials as u64 * 16).max(16) }],
                [vk::DescriptorBufferInfo { buffer: part.buffer, offset: 0, range: part.size }],
            ];
            let writes = [
                vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bufs[0]),
                vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&images[0]),
                vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&images[1]),
                vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bufs[1]),
                vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bufs[2]),
            ];
            unsafe { dev.update_descriptor_sets(&writes, &[]) };
        }
        Ok(ComposeBindings { pool, sets })
    }

    /// Records the pass into slot `slot`: emission when `emission`, and the meter with `floor` (the
    /// previous reading's [`Meter::next_floor`]). The radiance writes of the pass before it (temporal,
    /// filter or shade) are made visible here; the caller makes this pass's writes visible to its
    /// consumer (`shade::radiance_to_fragment`), and waits for the frame before [`ComposeTargets::reading`].
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &ComposeBindings, out: &ComposeTargets, slot: usize, emission: bool, floor: f64, faults: ComposeFaults) {
        let mut f = flags::METER;
        if emission {
            f |= flags::EMISSION;
        }
        if faults.meter_without_emission {
            f |= flags::FAULT_METER_WITHOUT_EMISSION;
        }
        let p = Params { width: out.width, height: out.height, flags: f, floor: floor as f32, groups_x: out.groups.0, ..Params::default() };
        let dev = &gpu.device;
        let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
        unsafe {
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.sets[slot]], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, out.groups.0, out.groups.1, 1);
            // The partial sums are read by the host once the frame has completed.
            let h = [vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER).src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE).dst_stage_mask(vk::PipelineStageFlags2::HOST).dst_access_mask(vk::AccessFlags2::HOST_READ)];
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&h));
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

    /// C9 (4B): the compose module matches its host layout without a device; a moved field is refused.
    #[test]
    fn host_layout_matches_the_compiled_module() {
        reflect::check(REFLECTION, &host_layout()).unwrap();
        let mut bad = host_layout();
        if let Some(Param::PushConstants { fields, .. }) = bad.last_mut() {
            fields[3].offset += 4;
        }
        assert!(reflect::check(REFLECTION, &bad).is_err());
    }
}
