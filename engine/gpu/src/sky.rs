//! Phase 3C: the real-time sky tables on the device (`shaders/sky.slang`, `sky_common.slang`).
//!
//! - The transmittance, multiple-scattering and ground-irradiance tables do not depend on the sun:
//!   they are built once on the host by `light::sky::SkyLuts` (f64, the tested oracle) and uploaded
//!   ([`SkyTables::upload`]).
//! - The sky-view table depends on the sun: [`SkyView::record`] rebuilds it on the GPU (f32) when the
//!   sun moves. `tests/sky.rs` checks it against the host's `SkyLuts::view`.
//!
//! - The correction (S-020, `light::sky_ref`): the ratio reference ÷ table per sun elevation, view
//!   zenith and azimuth, uploaded from `SkyLuts::correction` (exactly 1 when uncorrected); the
//!   sky-view pass multiplies each texel by it.
//!
//! All five are RGBA32F buffers, row-major, under `Category::GpuTemporal` (about 1.2 MB together).
//! Their sizes are shader constants (`sky_common.slang`); [`SkyTables::upload`] refuses tables of any
//! other size.

use std::mem::size_of;

use ash::vk;
use light::sky::{SkyLuts, GROUND_W, MS_H, MS_W, TRANS_H, TRANS_W, VIEW_H, VIEW_W};
use light::sky_ref::{CORR_H, CORR_S, CORR_W};
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::reflect::{self, Field, Param};
use crate::staging::{download, Uploader};
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/sky.json"));

/// Host mirror of the shader's push constants.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub sun: [f32; 4],
}

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "trans", binding: 0, element: vec![] },
        Param::Descriptor { name: "ms", binding: 1, element: vec![] },
        Param::Descriptor { name: "ground_e", binding: 2, element: vec![] },
        Param::Descriptor { name: "view", binding: 3, element: vec![] },
        Param::Descriptor { name: "corr", binding: 4, element: vec![] },
        Param::PushConstants { name: "params", fields: vec![field!(Params, sun)] },
    ]
}

/// The tables on the device.
pub struct SkyTables {
    pub trans: Buffer,
    pub ms: Buffer,
    pub ground: Buffer,
    pub view: Buffer,
    /// The correction (1 when `SkyLuts` is uncorrected).
    pub corr: Buffer,
    pub corrected: bool,
}

fn texels(data: &[[f64; 3]]) -> Vec<u8> {
    data.iter().flat_map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, 0.0]).flat_map(f32::to_le_bytes).collect()
}

impl SkyTables {
    /// Uploads the sun-independent tables and the correction of `luts` and allocates the sky-view
    /// table. All-or-nothing.
    pub fn upload(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, luts: &SkyLuts) -> Result<(SkyTables, u64)> {
        let sizes = [(luts.transmittance.w, luts.transmittance.h, TRANS_W, TRANS_H), (luts.ms.w, luts.ms.h, MS_W, MS_H), (luts.ground.w, luts.ground.h, GROUND_W, 1), (luts.correction.len(), 1, CORR_W * CORR_H * CORR_S, 1)];
        if let Some(s) = sizes.iter().find(|s| (s.0, s.1) != (s.2, s.3)) {
            return Err(GpuError::Layout(vec![format!("sky table of {}x{}, the shader expects {}x{}", s.0, s.1, s.2, s.3)]));
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let data = [texels(&luts.transmittance.data), texels(&luts.ms.data), texels(&luts.ground.data), texels(&luts.correction)];
        let mut bufs: Vec<Buffer> = Vec::new();
        let lens = [data[0].len() as u64, data[1].len() as u64, data[2].len() as u64, data[3].len() as u64, (VIEW_W * VIEW_H * 16) as u64];
        for len in lens {
            match alloc.create_buffer(gpu, len, usage, Category::GpuTemporal, Kind::Device) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    for b in bufs {
                        alloc.free(gpu, b);
                    }
                    return Err(e);
                }
            }
        }
        let result = (|| {
            for (b, d) in bufs.iter().zip(&data) {
                up.upload(gpu, timeline, b, 0, d)?;
            }
            up.flush(gpu, timeline)
        })();
        match result {
            Ok(v) => {
                let mut it = bufs.into_iter();
                let (trans, ms, ground, corr, view) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
                Ok((SkyTables { trans, ms, ground, view, corr, corrected: luts.corrected }, v))
            }
            Err(e) => {
                for b in bufs {
                    alloc.free(gpu, b);
                }
                Err(e)
            }
        }
    }

    pub fn device_bytes(&self) -> u64 {
        self.trans.size + self.ms.size + self.ground.size + self.view.size + self.corr.size
    }

    /// Reads the sky-view table back (RGB per texel). Waits.
    pub fn read_view(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline) -> Result<Vec<[f32; 3]>> {
        let bytes = download(gpu, alloc, timeline, &[(&self.view, 0, (VIEW_W * VIEW_H * 16) as u64)])?.remove(0);
        Ok(bytes.as_chunks::<16>().0.iter().map(|c| [0, 1, 2].map(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))).collect())
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        for b in [self.trans, self.ms, self.ground, self.view, self.corr] {
            alloc.free(gpu, b);
        }
    }
}

/// The sky-view pass, bound to one [`SkyTables`].
pub struct SkyView {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl SkyView {
    pub fn new(gpu: &Gpu, tables: &SkyTables) -> Result<SkyView> {
        Self::with_reflection(gpu, tables, REFLECTION)
    }

    pub fn with_reflection(gpu: &Gpu, tables: &SkyTables, reflection: &str) -> Result<SkyView> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [b(0), b(1), b(2), b(3), b(4)];
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
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 5 }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let set = unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl)) }.vk("vkAllocateDescriptorSets")?[0];
        let infos = [tables.trans.buffer, tables.ms.buffer, tables.ground.buffer, tables.view.buffer, tables.corr.buffer].map(|buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }]);
        let writes: Vec<_> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(SkyView { set_layout, layout, pipeline, pool, set })
    }

    /// Rebuilds the sky-view table for the sun along unit `sun`, then makes it visible to later
    /// compute and fragment reads. Earlier reads of the table must be complete or ordered before
    /// this (the caller records it before the frame's shade pass).
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, sun: [f64; 3]) {
        let dev = &gpu.device;
        let p = Params { sun: [sun[0] as f32, sun[1] as f32, sun[2] as f32, 0.0] };
        let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
        // Earlier reads of the table (the previous frame's shade pass) before this write.
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)];
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
        unsafe {
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&before));
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, (VIEW_W as u32).div_ceil(8), (VIEW_H as u32).div_ceil(8), 1);
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
        }
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_descriptor_pool(self.pool, None);
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
