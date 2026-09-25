//! Phase 3A: the GPU reference path tracer (`shaders/reference.slang`, ADR-0005).
//!
//! The same estimator as `light::reference` (CPU, f64): sun by next-event estimation, sky by cosine
//! sampling into the Monte Carlo atmosphere, `max_bounces` surface bounces, emission off. One sample
//! per pixel per dispatch, accumulated in a [`RefAccum`] (sum and sum of squares, 32 B/px under
//! `Category::GpuTemporal`, reference mode only). Frame `f` of pixel `i` draws from the same PCG32
//! stream as the CPU's sample `f`, so both estimate the same quantity with independent-looking noise
//! and are compared statistically (`tests/reference.rs`).
//!
//! Albedos are uploaded once per material registry ([`RefMaterials`], `Category::GpuMaterial`), after
//! `light::reference::albedos` has refused anything outside [0, 1].
//!
//! 4A (ADR-0005 Amendment 3): emission and emitter next-event estimation, as `light::reference`.
//! [`Reference::bind_lit`] binds an emitter table ([`RefEmitters`]); [`Reference::bind`] binds none,
//! and [`Reference::accumulate`] then refuses settings that use emitters.

use std::mem::{offset_of, size_of};
use std::ops::Range;

use ash::vk;
use light::reference::{Accum, Lighting, Settings};
use light::sun;
use memory::Category;

use crate::accel::{Accel, RegionRef};
use crate::alloc::{Allocator, Buffer, Kind};
use crate::emitters::{GpuEmitter, RefEmitters};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::raster::Camera;
use crate::reflect::{self, Field, Param};
use crate::staging::{download, Uploader};
use crate::submit::Submitter;
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reference.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/reference.json"));

/// Bytes per pixel of the accumulation buffer: sum and sum of squares, each RGBA32F.
pub const ACC_BYTES: u64 = 32;

/// `Params::flags` bits, as in the shader.
pub mod flags {
    pub const SUN: u32 = 1;
    pub const SKY: u32 = 2;
    pub const POINT_SUN: u32 = 4;
    pub const UNIFORM_SKY: u32 = 8;
    pub const ALBEDO_ONE: u32 = 16;
    pub const RESET: u32 = 32;
    pub const PROBE_T: u32 = 64;
    pub const EMISSION: u32 = 128;
    pub const FAULT_PDF: u32 = 256;
    pub const FAULT_NO_COS: u32 = 512;
    pub const EMITTERS_DIRECT: u32 = 1024;
    pub const EMITTERS_INDIRECT: u32 = 2048;
    pub const EMITTER_AREA: u32 = 4096;
    pub const FAULT_NO_SOLID_ANGLE: u32 = 8192;
    pub const FAULT_DOUBLE_EMISSION: u32 = 16384;
    pub const FAULT_NO_EMITTER_COS: u32 = 32768;
}

/// Planted faults for negative controls; all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefFaults {
    /// Continuation weight × 1.1: a wrong BSDF-sampling PDF.
    pub wrong_pdf: bool,
    /// The sun term without its cosine.
    pub no_cosine: bool,
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
    pub max_bounces: u32,
    pub steps: u32,
    pub sun_steps: u32,
}

/// Host mirror of the shader's accumulation element.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Acc {
    pub sum: [f32; 4],
    pub sum_sq: [f32; 4],
}

impl Params {
    pub fn new(cam: &Camera, light: &Lighting, s: &Settings, faults: RefFaults, frame: u32, seed: u32) -> Params {
        let v = |a: [f64; 3], sc: f64, w: f64| [(a[0] * sc) as f32, (a[1] * sc) as f32, (a[2] * sc) as f32, w as f32];
        let mut f = 0;
        for (on, bit) in [
            (s.sun, flags::SUN),
            (s.sky, flags::SKY),
            (s.point_sun, flags::POINT_SUN),
            (s.uniform_sky, flags::UNIFORM_SKY),
            (s.albedo_one, flags::ALBEDO_ONE),
            (faults.wrong_pdf, flags::FAULT_PDF),
            (faults.no_cosine, flags::FAULT_NO_COS),
            (s.emission, flags::EMISSION),
            (s.emitters_direct, flags::EMITTERS_DIRECT),
            (s.emitters_indirect, flags::EMITTERS_INDIRECT),
            (s.emitter_area_sampling, flags::EMITTER_AREA),
            (s.emitter_faults.no_solid_angle, flags::FAULT_NO_SOLID_ANGLE),
            (s.emitter_faults.double_emission, flags::FAULT_DOUBLE_EMISSION),
            (s.emitter_faults.no_emitter_cosine, flags::FAULT_NO_EMITTER_COS),
        ] {
            if on {
                f |= bit;
            }
        }
        let so = &s.sky_options;
        // The shader's event cap and roulette start are constants; the host settings must match them.
        assert_eq!((so.max_events, so.rr_from), (32, 3), "reference.slang hard-codes MAX_EVENTS 32 and RR_FROM 3");
        assert!(so.steps >= 2 && so.steps.is_multiple_of(2), "march steps must be even (split at the tangent point)");
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
            max_bounces: s.max_bounces,
            steps: so.steps,
            sun_steps: so.sun_steps,
        }
    }
}

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

/// What the host code assumes about the reference module.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "tlas", binding: 0, element: vec![] },
        Param::Descriptor { name: "regions", binding: 1, element: vec![field!(RegionRef, quads), field!(RegionRef, tri_quad), field!(RegionRef, quad_count), field!(RegionRef, tri_count)] },
        Param::Descriptor { name: "albedo", binding: 2, element: vec![] },
        Param::Descriptor { name: "acc", binding: 3, element: vec![field!(Acc, sum), field!(Acc, sum_sq)] },
        Param::Descriptor { name: "emitters", binding: 4, element: GpuEmitter::fields() },
        Param::Descriptor { name: "emission", binding: 5, element: vec![] },
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
                field!(Params, max_bounces),
                field!(Params, steps),
                field!(Params, sun_steps),
            ],
        },
    ]
}

/// Per-material albedo (RGBA32F, alpha unused).
pub struct RefMaterials {
    pub buffer: Buffer,
    pub count: u32,
}

impl RefMaterials {
    /// Uploads `albedo` (from `light::reference::albedos`); returns the timeline value after which it
    /// is on the device.
    pub fn upload(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, albedo: &[[f64; 3]]) -> Result<(RefMaterials, u64)> {
        let bytes: Vec<u8> = albedo.iter().flat_map(|a| [a[0] as f32, a[1] as f32, a[2] as f32, 0.0]).flat_map(f32::to_le_bytes).collect();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = alloc.create_buffer(gpu, (bytes.len() as u64).max(16), usage, Category::GpuMaterial, Kind::Device)?;
        let v = up.upload(gpu, timeline, &buffer, 0, &bytes).and_then(|_| up.flush(gpu, timeline));
        match v {
            Ok(v) => Ok((RefMaterials { buffer, count: albedo.len() as u32 }, v)),
            Err(e) => {
                alloc.free(gpu, buffer);
                Err(e)
            }
        }
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.buffer);
    }
}

/// The accumulation target.
pub struct RefAccum {
    pub width: u32,
    pub height: u32,
    pub buffer: Buffer,
}

impl RefAccum {
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, width: u32, height: u32) -> Result<RefAccum> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let buffer = alloc.create_buffer(gpu, (width * height) as u64 * ACC_BYTES, usage, Category::GpuTemporal, Kind::Device)?;
        Ok(RefAccum { width, height, buffer })
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.buffer);
    }
}

pub struct RefBindings {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// The bound emitter table's size, or `None` when [`Reference::bind`] bound placeholders.
    emitters: Option<u32>,
}

impl RefBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// What a readback holds: the per-pixel sums, and how many samples hit a bad surface id.
pub struct RefImage {
    pub accum: Accum,
    pub bad_samples: u64,
}

pub struct Reference {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Reference {
    pub fn new(gpu: &Gpu) -> Result<Reference> {
        Self::with_reflection(gpu, REFLECTION)
    }

    /// Checks `reflection` against [`host_layout`] first; a mismatch is `GpuError::Layout`.
    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Reference> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        if !gpu.ray_tracing() {
            return Err(GpuError::NoDevice("the reference renderer needs a ray-tracing device".into()));
        }
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [
            b(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            b(1, vk::DescriptorType::STORAGE_BUFFER),
            b(2, vk::DescriptorType::STORAGE_BUFFER),
            b(3, vk::DescriptorType::STORAGE_BUFFER),
            b(4, vk::DescriptorType::STORAGE_BUFFER),
            b(5, vk::DescriptorType::STORAGE_BUFFER),
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
        Ok(Reference { set_layout, layout, pipeline })
    }

    /// Binds without emitters: the emitter slots hold the albedo buffer as a placeholder that the
    /// shader never reads (the emitter flags must stay off; `accumulate` checks).
    pub fn bind(&self, gpu: &Gpu, accel: &Accel, materials: &RefMaterials, out: &RefAccum) -> Result<RefBindings> {
        let alb = vk::DescriptorBufferInfo { buffer: materials.buffer.buffer, offset: 0, range: (materials.count as u64 * 16).max(16) };
        self.bind_with(gpu, accel, materials, out, alb, alb, None)
    }

    /// Binds with the emitter table `em` (built from the same registry as `materials`).
    pub fn bind_lit(&self, gpu: &Gpu, accel: &Accel, materials: &RefMaterials, em: &RefEmitters, out: &RefAccum) -> Result<RefBindings> {
        assert_eq!(em.materials, materials.count, "the emitter table and the albedos come from one registry");
        let rows = vk::DescriptorBufferInfo { buffer: em.emitters.buffer, offset: 0, range: (em.count as u64 * size_of::<GpuEmitter>() as u64).max(size_of::<GpuEmitter>() as u64) };
        let emission = vk::DescriptorBufferInfo { buffer: em.emission.buffer, offset: 0, range: (em.materials as u64 * 16).max(16) };
        self.bind_with(gpu, accel, materials, out, rows, emission, Some(em.count))
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_with(&self, gpu: &Gpu, accel: &Accel, materials: &RefMaterials, out: &RefAccum, rows: vk::DescriptorBufferInfo, emission: vk::DescriptorBufferInfo, emitters: Option<u32>) -> Result<RefBindings> {
        let dev = &gpu.device;
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 5 },
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
            // Exactly the albedo rows, so the shader's bounds check sees the material count.
            [vk::DescriptorBufferInfo { buffer: materials.buffer.buffer, offset: 0, range: (materials.count as u64 * 16).max(16) }],
            [vk::DescriptorBufferInfo { buffer: out.buffer.buffer, offset: 0, range: out.buffer.size }],
            [rows],
            [emission],
        ];
        let mut writes = vec![vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).descriptor_count(1).push_next(&mut as_write)];
        for (k, info) in infos.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32 + 1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info));
        }
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(RefBindings { pool, set, emitters })
    }

    /// Records one dispatch with `params`. Consecutive dispatches read-modify-write the accumulation:
    /// the caller separates them with [`compute_to_compute`].
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &RefBindings, out: &RefAccum, params: &Params) {
        assert_eq!((out.width, out.height), (params.width, params.height), "camera and accumulation must have the same size");
        let dev = &gpu.device;
        let bytes = unsafe { std::slice::from_raw_parts(params as *const Params as *const u8, size_of::<Params>()) };
        unsafe {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.set], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, out.width.div_ceil(8), out.height.div_ceil(8), 1);
        }
    }

    /// Accumulates frames `frames` (the first one resets the accumulation when `frames.start` is
    /// `reset_at`), `per_submit` dispatches per submission, waiting for each submission so no single
    /// one runs long enough to trip the OS GPU timeout. Returns the GPU milliseconds of the dispatches
    /// as measured by host wall time per submission (a coarse figure; see `timing` for pass timing).
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(&self, gpu: &Gpu, timeline: &mut Timeline, bindings: &RefBindings, out: &RefAccum, cam: &Camera, light: &Lighting, s: &Settings, faults: RefFaults, seed: u32, frames: Range<u32>, reset_at: u32, per_submit: u32) -> Result<f64> {
        assert!(!s.uses_emitters() || bindings.emitters.is_some(), "emitter settings need bindings from bind_lit");
        let mut sub = Submitter::new(gpu)?;
        let t = std::time::Instant::now();
        let result = (|| {
            let mut f = frames.start;
            while f < frames.end {
                let cmd = sub.begin(gpu, timeline)?;
                let end = (f + per_submit.max(1)).min(frames.end);
                for frame in f..end {
                    let mut p = Params::new(cam, light, s, faults, frame, seed);
                    p.right[3] = f32::from_bits(bindings.emitters.unwrap_or(0));
                    if frame == reset_at {
                        p.flags |= flags::RESET;
                    }
                    compute_to_compute(gpu, cmd);
                    self.record(gpu, cmd, bindings, out, &p);
                }
                let v = sub.submit(gpu, timeline, cmd, &[])?;
                timeline.wait(gpu, v, u64::MAX)?;
                f = end;
            }
            Ok(())
        })();
        gpu.wait_idle()?;
        sub.destroy(gpu);
        result.map(|_| t.elapsed().as_secs_f64() * 1e3)
    }

    /// Writes, instead of samples, the transmittance from the observer along each primary direction
    /// with `steps` march steps (the constant-agreement probe).
    #[allow(clippy::too_many_arguments)]
    pub fn probe_transmittance(&self, gpu: &Gpu, timeline: &mut Timeline, bindings: &RefBindings, out: &RefAccum, cam: &Camera, light: &Lighting, steps: u32) -> Result<()> {
        let mut s = Settings::default();
        s.sky_options.steps = steps;
        let mut p = Params::new(cam, light, &s, RefFaults::default(), 0, 0);
        p.flags = flags::PROBE_T;
        let mut sub = Submitter::new(gpu)?;
        let result = (|| {
            let cmd = sub.begin(gpu, timeline)?;
            self.record(gpu, cmd, bindings, out, &p);
            let v = sub.submit(gpu, timeline, cmd, &[])?;
            timeline.wait(gpu, v, u64::MAX)
        })();
        gpu.wait_idle()?;
        sub.destroy(gpu);
        result.map(|_| ())
    }

    /// Reads the accumulation back; `samples` is the number of frames accumulated since the reset.
    pub fn read(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline, out: &RefAccum, samples: u32) -> Result<RefImage> {
        let n = (out.width * out.height) as usize;
        let bytes = download(gpu, alloc, timeline, &[(&out.buffer, 0, n as u64 * ACC_BYTES)])?.remove(0);
        let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
        let mut sum = Vec::with_capacity(n);
        let mut sum_sq = Vec::with_capacity(n);
        let mut bad = 0.0;
        for i in 0..n {
            let o = i * ACC_BYTES as usize;
            sum.push([f(o), f(o + 4), f(o + 8)]);
            bad += f(o + 12);
            sum_sq.push([f(o + 16), f(o + 20), f(o + 24)]);
        }
        Ok(RefImage { accum: Accum { width: out.width, height: out.height, samples, sum, sum_sq }, bad_samples: bad as u64 })
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Makes earlier compute writes visible to the next dispatch.
pub fn compute_to_compute(gpu: &Gpu, cmd: vk::CommandBuffer) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled module's reflection matches the host layout (4A: with the emitter bindings and
    /// the 80-byte emitter), without a device.
    #[test]
    fn host_layout_matches_the_compiled_module() {
        reflect::check(REFLECTION, &host_layout()).unwrap();
        assert_eq!(size_of::<Params>(), 128);
        // Negative control: the emitter's `meta` field moved by 4 bytes is refused.
        let mut bad = host_layout();
        for p in &mut bad {
            if let Param::Descriptor { name: "emitters", element, .. } = p {
                element.last_mut().unwrap().offset += 4;
            }
        }
        assert!(reflect::check(REFLECTION, &bad).is_err());
    }
}
