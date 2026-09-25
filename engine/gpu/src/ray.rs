//! Phase 2D: primary visibility by ray query (`shaders/ray_primary.slang`).
//!
//! A compute pass casts one ray per pixel centre through the TLAS of [`crate::accel::Accel`] and
//! writes the values the raster G-buffer holds: reverse-Z depth, R16G16_SNORM octahedral normal,
//! exact material, and the (region index, primitive) surface id. An id outside the region table
//! writes [`BAD_MATERIAL`] instead of reading out of bounds. [`RayPrimary::trace_to_host`]
//! returns them as a [`Frame`], so [`crate::equivalence::compare`] checks both paths with the same
//! declared contract (ADR-0003 Amendment 1).
//!
//! Outputs are device-local storage buffers under `Category::GpuTemporal` (frame targets), 16 + 8
//! bytes per pixel.

use std::mem::{offset_of, size_of};

use ash::vk;
use memory::Category;

use crate::accel::{Accel, RegionRef};
use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::raster::{Camera, Frame};
use crate::reflect::{self, Field, Param};
use crate::submit::{all_to_host, Submitter};
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ray_primary.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/ray_primary.json"));

/// Material written for a hit whose (region, primitive, quad) is outside the region table.
pub const BAD_MATERIAL: u16 = 0xFFFE;

/// Bytes per pixel of the two output buffers: `Hit` (16) and the surface id (8).
pub const HIT_BYTES: u64 = 16;
pub const SURFACE_BYTES: u64 = 8;

/// Host mirror of the shader's `Params` push constants.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub eye: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub width: u32,
    pub height: u32,
    pub near: f32,
    pub pad: u32,
}

/// Host mirror of the shader's `Hit`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Hit {
    pub depth: u32,
    pub normal: u32,
    pub material: u32,
    pub pad: u32,
}

impl Params {
    pub fn of(cam: &Camera) -> Params {
        let v = |a: [f64; 3], s: f64| [(a[0] * s) as f32, (a[1] * s) as f32, (a[2] * s) as f32, 0.0];
        Params {
            eye: v(cam.eye, 1.0),
            forward: v(cam.forward, 1.0),
            right: v(cam.right, cam.tan_half_x),
            up: v(cam.up, cam.tan_half_y),
            width: cam.width,
            height: cam.height,
            near: cam.near as f32,
            pad: 0,
        }
    }
}

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

/// What the host code assumes about the trace module.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "tlas", binding: 0, element: vec![] },
        Param::Descriptor { name: "regions", binding: 1, element: vec![field!(RegionRef, quads), field!(RegionRef, tri_quad), field!(RegionRef, quad_count), field!(RegionRef, tri_count)] },
        Param::Descriptor { name: "hits", binding: 2, element: vec![field!(Hit, depth), field!(Hit, normal), field!(Hit, material), field!(Hit, pad)] },
        Param::Descriptor { name: "surfaces", binding: 3, element: vec![] },
        Param::PushConstants {
            name: "params",
            fields: vec![field!(Params, eye), field!(Params, forward), field!(Params, right), field!(Params, up), field!(Params, width), field!(Params, height), field!(Params, near), field!(Params, pad)],
        },
    ]
}

/// Per-pixel outputs of the trace pass.
pub struct RayTargets {
    pub width: u32,
    pub height: u32,
    pub hits: Buffer,
    pub surfaces: Buffer,
}

impl RayTargets {
    /// All-or-nothing: on a refused grant nothing stays allocated.
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, width: u32, height: u32) -> Result<RayTargets> {
        let px = (width * height) as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let hits = alloc.create_buffer(gpu, px * HIT_BYTES, usage, Category::GpuTemporal, Kind::Device)?;
        match alloc.create_buffer(gpu, px * SURFACE_BYTES, usage, Category::GpuTemporal, Kind::Device) {
            Ok(surfaces) => Ok(RayTargets { width, height, hits, surfaces }),
            Err(e) => {
                alloc.free(gpu, hits);
                Err(e)
            }
        }
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.hits);
        alloc.free(gpu, self.surfaces);
    }
}

/// The descriptor set binding one TLAS, region table and output pair.
pub struct RayBindings {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl RayBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }

    /// The pool that owns the set, for deferred destruction (2E retirement).
    pub fn into_pool(self) -> vk::DescriptorPool {
        self.pool
    }
}

pub struct RayPrimary {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl RayPrimary {
    pub fn new(gpu: &Gpu) -> Result<RayPrimary> {
        Self::with_reflection(gpu, REFLECTION)
    }

    /// Checks `reflection` against [`host_layout`] first; a mismatch is `GpuError::Layout` and no
    /// Vulkan object is created.
    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<RayPrimary> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        if !gpu.ray_tracing() {
            return Err(GpuError::NoDevice("ray query needs a ray-tracing device".into()));
        }
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [
            b(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            b(1, vk::DescriptorType::STORAGE_BUFFER),
            b(2, vk::DescriptorType::STORAGE_BUFFER),
            b(3, vk::DescriptorType::STORAGE_BUFFER),
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
        Ok(RayPrimary { set_layout, layout, pipeline })
    }

    pub fn bind(&self, gpu: &Gpu, accel: &Accel, out: &RayTargets) -> Result<RayBindings> {
        let dev = &gpu.device;
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 },
        ];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = [self.set_layout];
        let set = match unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets") {
            Ok(s) => s[0],
            Err(e) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                return Err(e);
            }
        };
        let tlas = [accel.tlas()];
        let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlas);
        let (table, table_size) = accel.table();
        let infos = [
            [vk::DescriptorBufferInfo { buffer: table, offset: 0, range: table_size }],
            [vk::DescriptorBufferInfo { buffer: out.hits.buffer, offset: 0, range: out.hits.size }],
            [vk::DescriptorBufferInfo { buffer: out.surfaces.buffer, offset: 0, range: out.surfaces.size }],
        ];
        let mut writes = vec![vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).descriptor_count(1).push_next(&mut as_write)];
        for (k, info) in infos.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32 + 1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info));
        }
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(RayBindings { pool, set })
    }

    /// Records the trace dispatch. The outputs are written by the compute stage; the caller adds the
    /// barrier its consumer needs.
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &RayBindings, out: &RayTargets, camera: &Camera) {
        assert_eq!((out.width, out.height), (camera.width, camera.height), "camera and outputs must have the same size");
        let dev = &gpu.device;
        let p = Params::of(camera);
        let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
        unsafe {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.set], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, out.width.div_ceil(8), out.height.div_ceil(8), 1);
        }
    }

    /// Traces and reads the outputs back as a [`Frame`], waiting for completion.
    pub fn trace_to_host(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline, accel: &Accel, out: &RayTargets, camera: &Camera) -> Result<Frame> {
        let dev = &gpu.device;
        let px = (out.width * out.height) as u64;
        let tmp = alloc.create_buffer(gpu, px * (HIT_BYTES + SURFACE_BYTES), vk::BufferUsageFlags::TRANSFER_DST, Category::Staging, Kind::Host)?;
        let bindings = match self.bind(gpu, accel, out) {
            Ok(b) => b,
            Err(e) => {
                alloc.free(gpu, tmp);
                return Err(e);
            }
        };
        let mut sub = Submitter::new(gpu)?;
        let result = (|| {
            let cmd = sub.begin(gpu, timeline)?;
            self.record(gpu, cmd, &bindings, out, camera);
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
            unsafe {
                dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
                dev.cmd_copy_buffer(cmd, out.hits.buffer, tmp.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: px * HIT_BYTES }]);
                dev.cmd_copy_buffer(cmd, out.surfaces.buffer, tmp.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: px * HIT_BYTES, size: px * SURFACE_BYTES }]);
            }
            all_to_host(gpu, cmd);
            let v = sub.submit(gpu, timeline, cmd, &[])?;
            timeline.wait(gpu, v, u64::MAX)?;
            let m = tmp.mapped_ref().expect("host buffer");
            let word = |o: usize| u32::from_le_bytes(m[o..o + 4].try_into().unwrap());
            let n = px as usize;
            let s0 = n * HIT_BYTES as usize;
            Ok(Frame {
                width: out.width,
                height: out.height,
                depth: (0..n).map(|i| f32::from_bits(word(16 * i))).collect(),
                normal: (0..n)
                    .map(|i| {
                        let w = word(16 * i + 4);
                        [(w & 0xFFFF) as u16 as i16, (w >> 16) as u16 as i16]
                    })
                    .collect(),
                material: (0..n).map(|i| word(16 * i + 8) as u16).collect(),
                surface: (0..n).map(|i| [word(s0 + 8 * i), word(s0 + 8 * i + 4)]).collect(),
            })
        })();
        gpu.wait_idle()?;
        sub.destroy(gpu);
        bindings.destroy(gpu);
        alloc.free(gpu, tmp);
        result
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
