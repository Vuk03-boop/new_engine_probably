//! Phase 3B: real-time lighting from the G-buffer (`shaders/shade.slang`, ADR-0005 conventions).
//!
//! One compute pass reads the raster targets (in `SHADER_READ_ONLY_OPTIMAL`, see
//! [`crate::debug_view::targets_to_read`]) and the TLAS, and writes linear HDR radiance per pixel
//! into [`ShadeTargets`] (RGBA32F, 16 B/px under `Category::GpuTemporal`), which
//! [`crate::debug_view::View::Light`] tone-maps.
//!
//! 3B term: direct sun, one shadow ray per pixel to a uniform point on the disk (or the centre with
//! `point_sun`). The pixel's PCG32 stream is the reference's (`gpu::reference`, `light::reference`)
//! for the same frame and seed, so frame `f` equals the reference's sun-only sample `f` up to the
//! surface position (depth-reconstructed here, ray-hit there).
//!
//! 3C term: sky light, one cosine-sampled visibility ray per pixel (the reference's continuation ray
//! at the primary vertex, same stream), taking the sky-view table ([`crate::sky`]) when it escapes.
//! Pixels without a surface show the table's sky and the sun disk. `uniform_sky` is the exact control
//! (sky 1, sun off, as in the reference).
//!
//! 3F term (`bounce`, on by default): one bounce of diffuse light. The sky ray becomes the
//! reference's continuation ray; where it hits a surface, that point gets the sun (its own shadow
//! ray) and the sky (one more visibility ray) times both albedos, exactly as the reference's second
//! vertex with `max_bounces` 1. Up to 4 rays per pixel (3C: 2). The hit's material comes from the
//! region table ([`Accel::table`]), bound next to the TLAS.

use std::mem::size_of;

use ash::vk;
use light::reference::Lighting;
use light::sun;
use memory::Category;

use crate::accel::{Accel, RegionRef};
use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::raster::{Camera, Targets};
use crate::reference::RefMaterials;
use crate::reflect::{self, Field, Param};
use crate::staging::download;
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shade.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/shade.json"));

/// Bytes per pixel of the radiance output.
pub const RADIANCE_BYTES: u64 = 16;

/// `Params::flags` bits, as in the shader.
pub mod flags {
    pub const SUN: u32 = 1;
    pub const SKY: u32 = 2;
    pub const POINT_SUN: u32 = 4;
    pub const UNIFORM_SKY: u32 = 8;
    pub const BOUNCE: u32 = 16;
    pub const FAULT_FLIP_X: u32 = 256;
    pub const FAULT_DEPTH: u32 = 512;
}

/// What the pass computes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadeSettings {
    pub sun: bool,
    pub sky: bool,
    /// Sample the sun's centre only (hard shadows): the exact control.
    pub point_sun: bool,
    /// Control: the sky is 1 in every direction and the sun is off.
    pub uniform_sky: bool,
    /// 3F: one bounce of diffuse light (sun and sky at the continuation ray's hit).
    pub bounce: bool,
}

impl Default for ShadeSettings {
    fn default() -> ShadeSettings {
        ShadeSettings { sun: true, sky: true, point_sun: false, uniform_sky: false, bounce: true }
    }
}

/// Planted faults for negative controls; all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShadeFaults {
    /// Flip the normal of ±x faces.
    pub flip_x: bool,
    /// Rebuild the surface point 1% too far along the pixel ray.
    pub depth: bool,
}

/// Host mirror of the shader's push constants (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub eye: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_ground: [f32; 4],
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub seed: u32,
    pub flags: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

const _: () = assert!(size_of::<Params>() == 128);

impl Params {
    pub fn new(cam: &Camera, light: &Lighting, s: ShadeSettings, faults: ShadeFaults, frame: u32, seed: u32) -> Params {
        let v = |a: [f64; 3], sc: f64, w: f64| [(a[0] * sc) as f32, (a[1] * sc) as f32, (a[2] * sc) as f32, w as f32];
        let mut f = 0;
        for (on, bit) in [(s.sun, flags::SUN), (s.sky, flags::SKY), (s.point_sun, flags::POINT_SUN), (s.uniform_sky, flags::UNIFORM_SKY), (s.bounce, flags::BOUNCE), (faults.flip_x, flags::FAULT_FLIP_X), (faults.depth, flags::FAULT_DEPTH)] {
            if on {
                f |= bit;
            }
        }
        Params {
            eye: v(cam.eye, 1.0, sun::sun_cos_max()),
            forward: v(cam.forward, 1.0, cam.near),
            right: v(cam.right, cam.tan_half_x, 0.0),
            up: v(cam.up, cam.tan_half_y, 0.0),
            sun_dir: v(light.sun_dir, 1.0, 0.0),
            sun_ground: v(light.sun_at_ground, 1.0, 1.0 / sun::sun_solid_angle()),
            width: cam.width,
            height: cam.height,
            frame,
            seed,
            flags: f,
            ..Params::default()
        }
    }
}

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

/// What the host code assumes about the shade module.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "depth_tex", binding: 0, element: vec![] },
        Param::Descriptor { name: "normal_tex", binding: 1, element: vec![] },
        Param::Descriptor { name: "material_tex", binding: 2, element: vec![] },
        Param::Descriptor { name: "tlas", binding: 3, element: vec![] },
        Param::Descriptor { name: "albedo", binding: 4, element: vec![] },
        Param::Descriptor { name: "radiance", binding: 5, element: vec![] },
        Param::Descriptor { name: "sky_view", binding: 6, element: vec![] },
        Param::Descriptor { name: "regions", binding: 7, element: vec![field!(RegionRef, quads), field!(RegionRef, tri_quad), field!(RegionRef, quad_count), field!(RegionRef, tri_count)] },
        Param::PushConstants {
            name: "params",
            fields: vec![
                field!(Params, eye),
                field!(Params, forward),
                field!(Params, right),
                field!(Params, up),
                field!(Params, sun_dir),
                field!(Params, sun_ground),
                field!(Params, width),
                field!(Params, height),
                field!(Params, frame),
                field!(Params, seed),
                field!(Params, flags),
                field!(Params, pad0),
                field!(Params, pad1),
                field!(Params, pad2),
            ],
        },
    ]
}

/// The radiance output, one RGBA32F per pixel, row-major.
pub struct ShadeTargets {
    pub width: u32,
    pub height: u32,
    pub radiance: Buffer,
}

impl ShadeTargets {
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, width: u32, height: u32) -> Result<ShadeTargets> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let radiance = alloc.create_buffer(gpu, (width * height) as u64 * RADIANCE_BYTES, usage, Category::GpuTemporal, Kind::Device)?;
        Ok(ShadeTargets { width, height, radiance })
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.radiance);
    }

    /// Reads the radiance back (RGB and the bad-material flag in w). Waits.
    pub fn read(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline) -> Result<Vec<[f32; 4]>> {
        let n = (self.width * self.height) as u64;
        let bytes = download(gpu, alloc, timeline, &[(&self.radiance, 0, n * RADIANCE_BYTES)])?.remove(0);
        Ok(bytes.as_chunks::<16>().0.iter().map(|c| [0, 1, 2, 3].map(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))).collect())
    }
}

pub struct ShadeBindings {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl ShadeBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }

    /// The pool that owns the set, for deferred destruction.
    pub fn into_pool(self) -> vk::DescriptorPool {
        self.pool
    }
}

pub struct Shade {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Shade {
    pub fn new(gpu: &Gpu) -> Result<Shade> {
        Self::with_reflection(gpu, REFLECTION)
    }

    /// Checks `reflection` against [`host_layout`] first; a mismatch is `GpuError::Layout`.
    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Shade> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        if !gpu.ray_tracing() {
            return Err(GpuError::NoDevice("real-time lighting needs a ray-tracing device".into()));
        }
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [
            b(0, vk::DescriptorType::SAMPLED_IMAGE),
            b(1, vk::DescriptorType::SAMPLED_IMAGE),
            b(2, vk::DescriptorType::SAMPLED_IMAGE),
            b(3, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            b(4, vk::DescriptorType::STORAGE_BUFFER),
            b(5, vk::DescriptorType::STORAGE_BUFFER),
            b(6, vk::DescriptorType::STORAGE_BUFFER),
            b(7, vk::DescriptorType::STORAGE_BUFFER),
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
        Ok(Shade { set_layout, layout, pipeline })
    }

    /// Binds the G-buffer targets, the TLAS, the albedo table, the output, the sky-view table and the
    /// region table (3F).
    /// Rebind after a resize or a scene update (the TLAS changes), when the GPU no longer uses the old
    /// set. The sky-view table is rewritten in place when the sun moves; no rebind is needed.
    pub fn bind(&self, gpu: &Gpu, targets: &Targets, accel: &Accel, albedo: &RefMaterials, out: &ShadeTargets, sky_view: &Buffer) -> Result<ShadeBindings> {
        let dev = &gpu.device;
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 3 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 },
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
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        // Targets: depth, normal (colour 0), material (colour 1), as `DebugView::bind` orders them.
        let images = [img(targets.depth.view), img(targets.color[0].view), img(targets.color[1].view)];
        let tlas = [accel.tlas()];
        let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlas);
        let (table, table_size) = accel.table();
        let bufs = [
            [vk::DescriptorBufferInfo { buffer: albedo.buffer.buffer, offset: 0, range: (albedo.count as u64 * 16).max(16) }],
            [vk::DescriptorBufferInfo { buffer: out.radiance.buffer, offset: 0, range: out.radiance.size }],
            [vk::DescriptorBufferInfo { buffer: sky_view.buffer, offset: 0, range: sky_view.size }],
            [vk::DescriptorBufferInfo { buffer: table, offset: 0, range: table_size }],
        ];
        let mut writes: Vec<_> = images.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(info)).collect();
        writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).descriptor_count(1).push_next(&mut as_write));
        for (k, info) in bufs.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4 + k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info));
        }
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(ShadeBindings { pool, set })
    }

    /// Records the pass. The targets must already be readable ([`crate::debug_view::targets_to_read`]);
    /// the caller makes the radiance writes visible to its consumer ([`radiance_to_fragment`]).
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &ShadeBindings, out: &ShadeTargets, params: &Params) {
        assert_eq!((out.width, out.height), (params.width, params.height), "camera and output must have the same size");
        let dev = &gpu.device;
        let bytes = unsafe { std::slice::from_raw_parts(params as *const Params as *const u8, size_of::<Params>()) };
        unsafe {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.set], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, out.width.div_ceil(8), out.height.div_ceil(8), 1);
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

/// Makes the radiance writes visible to the fragment stage (the light view).
pub fn radiance_to_fragment(gpu: &Gpu, cmd: vk::CommandBuffer) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
}
