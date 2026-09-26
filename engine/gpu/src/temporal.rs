//! Phase 3D: the temporal foundation (ADR-0006; `shaders/temporal.slang`).
//!
//! After the shade pass, one compute pass per frame reprojects every surface pixel into the previous
//! frame, validates the four bilinear history taps against the stored guides (face, integer plane,
//! material, region key and snapshot), and accumulates: accepted pixels blend the fresh sample into
//! the history with weight 1 / age (age capped at `max_age`), rejected ones restart from it. The
//! resolved colour and the state (age, reason) are also written back into the shade pass's radiance
//! buffer, which the light view and the age / reason debug views read.
//!
//! [`History`] owns the frame-sized buffers (`GpuTemporal`: guides, colour and state ping-pong, plus
//! motion; 72 B/px) and the previous frame's camera and sun. Global resets are decided here: the first
//! frame, a camera cut (eye moved more than [`CUT_VOXELS`]), a light jump (sun moved more than
//! [`LIGHT_JUMP_DEG`]) or a forced reset. A resize makes a new `History`, so it starts with a reset.
//!
//! 3E (ADR-0006 Amendment 1), lighting that changes under an accepted history: an edit lists its
//! voxel boxes ([`Relight`]) for the frame after it is published, and pixels whose sun ray crosses a
//! box or that lie within its sky radius restart as "relit"; while the sun moves, the age cap shrinks
//! so that the history spans at most `sun_tolerance_deg` of sun motion.
//!
//! 4B (ADR-0006 Amendment 2): the street's lights switching on or off is a light jump too. The caller
//! tells the history whether the lights are on ([`History::set_lights`]); a change from the previous
//! frame resets every pixel.

use std::mem::size_of;

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::raster::{Camera, Targets};
use crate::reflect::{self, Field, Param};
use crate::staging::download;
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/temporal.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/temporal.json"));

/// A camera move longer than this in one frame is a cut (ADR-0006), voxels (4 m).
pub const CUT_VOXELS: f64 = 64.0;
/// A sun move larger than this in one frame is a light jump (ADR-0006), degrees.
pub const LIGHT_JUMP_DEG: f64 = 1.0;
/// Relight boxes per frame; more are merged into the last one.
pub const MAX_RELIGHT_BOXES: usize = 16;
const RELIGHT_BYTES: u64 = 16 * (1 + 2 * MAX_RELIGHT_BOXES as u64);

/// History reasons (ADR-0006), as in the shader.
pub mod reason {
    pub const ACCEPTED: u32 = 0;
    pub const SKY: u32 = 1;
    pub const OFFSCREEN: u32 = 2;
    pub const NO_PREV: u32 = 3;
    pub const MATERIAL: u32 = 4;
    pub const NORMAL: u32 = 5;
    pub const DISOCCLUDED: u32 = 6;
    pub const EDITED: u32 = 7;
    pub const RESET: u32 = 8;
    pub const RELIT: u32 = 9;
    pub const NAMES: [&str; 10] = ["accepted", "sky", "off-screen", "no previous surface", "material", "normal", "disoccluded", "edited", "reset", "relit"];
}

mod flags {
    pub const RESET: u32 = 1;
    pub const BILINEAR: u32 = 2;
    pub const FAULT_RESET_ALWAYS: u32 = 256;
    pub const FAULT_NEVER_REJECT: u32 = 512;
    pub const FAULT_NO_FRESH: u32 = 1024;
}

/// The history length cap (a slider, ADR-0006; default 64) and the sun motion one history may span
/// (3E; default 0.5°: the age cap is `sun_tolerance_deg` / the sun's motion this frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalSettings {
    pub max_age: u32,
    pub sun_tolerance_deg: f64,
    /// 3E: resample the history bilinearly (the 3D behaviour) instead of with Catmull-Rom.
    pub bilinear: bool,
}

impl Default for TemporalSettings {
    fn default() -> TemporalSettings {
        TemporalSettings { max_age: 64, sun_tolerance_deg: 0.5, bilinear: false }
    }
}

impl TemporalSettings {
    /// The age cap for a frame in which the sun moved `sun_deg`.
    pub fn age_cap(&self, sun_deg: f64) -> u32 {
        let max = self.max_age.max(1);
        if sun_deg > 0.0 {
            ((self.sun_tolerance_deg / sun_deg).floor().clamp(1.0, max as f64)) as u32
        } else {
            max
        }
    }
}

/// Voxels an edit changed this frame: `lo` inclusive, `hi` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relight {
    pub lo: [i32; 3],
    pub hi: [i32; 3],
}

impl Relight {
    /// The smallest box holding both.
    pub fn union(self, o: Relight) -> Relight {
        Relight { lo: std::array::from_fn(|a| self.lo[a].min(o.lo[a])), hi: std::array::from_fn(|a| self.hi[a].max(o.hi[a])) }
    }

    /// The box as the shader tests it: grown by a 1-voxel margin (the sun disk's penumbra up to about
    /// 400 voxels away), with sky radius 4 x its largest side + 1 voxel (outside it, removing or
    /// adding the box changes a point's sky irradiance by less than about 2%).
    pub fn rows(&self) -> [[f32; 4]; 2] {
        let side = (0..3).map(|a| self.hi[a] - self.lo[a]).max().unwrap_or(0).max(1) as f32;
        let (l, h) = (self.lo.map(|v| v as f32 - 1.0), self.hi.map(|v| v as f32 + 1.0));
        [[l[0], l[1], l[2], 4.0 * side + 1.0], [h[0], h[1], h[2], 0.0]]
    }
}

/// The relight buffer's contents: sun direction and count, then two rows per box (at most
/// [`MAX_RELIGHT_BOXES`]; the rest are merged into the last one).
pub fn relight_rows(sun: [f64; 3], boxes: &[Relight]) -> Vec<[f32; 4]> {
    let mut merged: Vec<Relight> = boxes.iter().take(MAX_RELIGHT_BOXES).copied().collect();
    if boxes.len() > MAX_RELIGHT_BOXES {
        merged[MAX_RELIGHT_BOXES - 1] = boxes[MAX_RELIGHT_BOXES - 1..].iter().copied().reduce(Relight::union).unwrap();
    }
    let mut rows = vec![[sun[0] as f32, sun[1] as f32, sun[2] as f32, merged.len() as f32]];
    for b in &merged {
        rows.extend(b.rows());
    }
    rows
}

/// Planted faults for negative controls; all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TemporalFaults {
    /// Reset every frame (the old engine's failure).
    pub reset_always: bool,
    /// Accept every tap that has a surface.
    pub never_reject: bool,
    /// Accepted pixels keep the history and ignore the fresh sample.
    pub no_fresh: bool,
}

/// Host mirror of the shader's push constants (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub eye: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub prev_eye: [f32; 4],
    pub prev_fwd: [f32; 4],
    pub prev_right: [f32; 4],
    pub prev_up: [f32; 4],
}

const _: () = assert!(size_of::<Params>() == 128);

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

pub fn host_layout() -> Vec<Param> {
    let names = ["depth_tex", "normal_tex", "material_tex", "surface_tex", "regions", "radiance", "prev_guides", "next_guides", "prev_hist", "next_hist", "prev_state", "next_state", "motion", "relight"];
    let mut v: Vec<Param> = names.iter().enumerate().map(|(i, n)| Param::Descriptor { name: n, binding: i as u64, element: vec![] }).collect();
    v.push(Param::PushConstants {
        name: "params",
        fields: vec![
            field!(Params, eye),
            field!(Params, forward),
            field!(Params, right),
            field!(Params, up),
            field!(Params, prev_eye),
            field!(Params, prev_fwd),
            field!(Params, prev_right),
            field!(Params, prev_up),
        ],
    });
    v
}

#[derive(Clone, Copy, Debug)]
struct Prev {
    cam: Camera,
    sun: [f64; 3],
    lights: bool,
}

/// Why this frame was a global reset (all false: history carried over).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResetCause {
    pub first: bool,
    pub cut: bool,
    pub light: bool,
    /// 4B: the lights switched on or off since the previous frame.
    pub lights: bool,
    pub forced: bool,
}

impl ResetCause {
    pub fn any(&self) -> bool {
        self.first || self.cut || self.light || self.lights || self.forced
    }
}

/// Per pixel: (age, reason), and motion in pixels.
pub type Readback = (Vec<(u32, u32)>, Vec<[f32; 2]>);

/// The history of one frame size.
pub struct History {
    pub width: u32,
    pub height: u32,
    guides: [Buffer; 2],
    hist: [Buffer; 2],
    state: [Buffer; 2],
    pub motion: Buffer,
    /// 3E: this frame's sun direction and relight boxes (`relight_rows`), written in the command stream.
    relight: Buffer,
    /// The buffers read this frame are `[parity]`, written `[1 - parity]`.
    parity: usize,
    prev: Option<Prev>,
    pub frames: u64,
    /// The age cap of the last frame recorded (`TemporalSettings::age_cap`).
    pub age_cap: u32,
    /// 4B: whether the lights are on for the next frame recorded ([`History::set_lights`]).
    lights: bool,
}

impl History {
    /// All-or-nothing.
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, width: u32, height: u32) -> Result<History> {
        let px = (width * height) as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let sizes = [16, 16, 16, 16, 4, 4, 8];
        let mut bufs = Vec::new();
        for s in sizes {
            match alloc.create_buffer(gpu, px * s, usage, Category::GpuTemporal, Kind::Device) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    for b in bufs {
                        alloc.free(gpu, b);
                    }
                    return Err(e);
                }
            }
        }
        let relight = match alloc.create_buffer(gpu, RELIGHT_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Category::GpuTemporal, Kind::Device) {
            Ok(b) => b,
            Err(e) => {
                for b in bufs {
                    alloc.free(gpu, b);
                }
                return Err(e);
            }
        };
        let mut it = bufs.into_iter();
        let mut n = || it.next().unwrap();
        Ok(History { width, height, guides: [n(), n()], hist: [n(), n()], state: [n(), n()], motion: n(), relight, parity: 0, prev: None, frames: 0, age_cap: 0, lights: false })
    }

    /// 4B: whether the street's lights are on for the next frame recorded (off by default). A change
    /// from the previous frame is a light jump: that frame resets every pixel.
    pub fn set_lights(&mut self, on: bool) {
        self.lights = on;
    }

    pub fn device_bytes(&self) -> u64 {
        self.guides.iter().chain(&self.hist).chain(&self.state).map(|b| b.size).sum::<u64>() + self.motion.size + self.relight.size
    }

    /// 3E: the guide and history buffers (`[parity()]` holds the last frame written) for the denoiser.
    pub(crate) fn guides(&self) -> &[Buffer; 2] {
        &self.guides
    }

    pub(crate) fn hist(&self) -> &[Buffer; 2] {
        &self.hist
    }

    /// Which of the ping-pong buffers the last recorded frame wrote.
    pub fn parity(&self) -> usize {
        self.parity
    }

    /// Reads back the state (age, reason) and motion of the last frame written. Waits.
    pub fn read(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline) -> Result<Readback> {
        let px = (self.width * self.height) as u64;
        let w = self.parity; // after record, the last written buffers are [parity]
        let out = download(gpu, alloc, timeline, &[(&self.state[w], 0, px * 4), (&self.motion, 0, px * 8)])?;
        let st = out[0].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).map(|s| (s & 0xFFFF, (s >> 16) & 0xFF)).collect();
        let mo = out[1].as_chunks::<8>().0.iter().map(|c| [f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap())]).collect();
        Ok((st, mo))
    }

    /// 4B: reads back the history colour of the last frame written: the resolved running mean (xyz)
    /// and the luminance moment (w). Waits.
    pub fn read_colour(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline) -> Result<Vec<[f32; 4]>> {
        let px = (self.width * self.height) as u64;
        let out = download(gpu, alloc, timeline, &[(&self.hist[self.parity], 0, px * 16)])?.remove(0);
        Ok(out.as_chunks::<16>().0.iter().map(|c| [0, 1, 2, 3].map(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))).collect())
    }

    /// Reads back the guides of the last frame written (face | material << 3, plane, region key,
    /// snapshot). Waits.
    pub fn read_guides(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline) -> Result<Vec<[u32; 4]>> {
        let px = (self.width * self.height) as u64;
        let out = download(gpu, alloc, timeline, &[(&self.guides[self.parity], 0, px * 16)])?.remove(0);
        Ok(out.as_chunks::<16>().0.iter().map(|c| [0, 1, 2, 3].map(|k| u32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))).collect())
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        for b in self.guides.into_iter().chain(self.hist).chain(self.state) {
            alloc.free(gpu, b);
        }
        alloc.free(gpu, self.motion);
        alloc.free(gpu, self.relight);
    }
}

pub struct TemporalBindings {
    pool: vk::DescriptorPool,
    /// `[p]` reads the `[p]` buffers and writes the `[1 - p]` ones.
    sets: [vk::DescriptorSet; 2],
}

impl TemporalBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }

    pub fn into_pool(self) -> vk::DescriptorPool {
        self.pool
    }
}

pub struct Temporal {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Temporal {
    pub fn new(gpu: &Gpu) -> Result<Temporal> {
        Self::with_reflection(gpu, REFLECTION)
    }

    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Temporal> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let mut bindings: Vec<_> = (0..4).map(|i| b(i, vk::DescriptorType::SAMPLED_IMAGE)).collect();
        bindings.extend((4..14).map(|i| b(i, vk::DescriptorType::STORAGE_BUFFER)));
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
        Ok(Temporal { set_layout, layout, pipeline })
    }

    /// Binds the G-buffer targets, the region table (`debug_view::Tables::regions`), the shade
    /// radiance and `history`. Rebind after a resize or a new region table.
    pub fn bind(&self, gpu: &Gpu, targets: &Targets, regions: &Buffer, radiance: &Buffer, history: &History) -> Result<TemporalBindings> {
        let dev = &gpu.device;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 8 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 20 }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = [self.set_layout, self.set_layout];
        let sets = match unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets") {
            Ok(s) => [s[0], s[1]],
            Err(e) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                return Err(e);
            }
        };
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        let images = [img(targets.depth.view), img(targets.color[0].view), img(targets.color[1].view), img(targets.color[2].view)];
        let whole = |b: &Buffer| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        for (p, &set) in sets.iter().enumerate() {
            let (r, w) = (p, 1 - p);
            let bufs = [whole(regions), whole(radiance), whole(&history.guides[r]), whole(&history.guides[w]), whole(&history.hist[r]), whole(&history.hist[w]), whole(&history.state[r]), whole(&history.state[w]), whole(&history.motion), whole(&history.relight)];
            let mut writes: Vec<_> = images.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(info)).collect();
            writes.extend(bufs.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4 + i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)));
            unsafe { dev.update_descriptor_sets(&writes, &[]) };
        }
        Ok(TemporalBindings { pool, sets })
    }

    /// Records the pass for this frame and advances `history` (parity, previous camera and sun).
    /// `relight` lists the voxel boxes edited since the previous frame (3E). The shade pass must
    /// precede it in the same queue (its radiance writes are made visible here); the caller makes the
    /// outputs visible to their consumer (`shade::radiance_to_fragment`).
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &TemporalBindings, history: &mut History, cam: &Camera, sun: [f64; 3], s: TemporalSettings, faults: TemporalFaults, force_reset: bool, relight: &[Relight]) -> ResetCause {
        assert_eq!((history.width, history.height), (cam.width, cam.height), "camera and history must have the same size");
        let mut sun_deg = 0.0;
        let cause = match history.prev {
            None => ResetCause { first: true, forced: force_reset, ..ResetCause::default() },
            Some(p) => {
                let moved = (0..3).map(|a| (cam.eye[a] - p.cam.eye[a]).powi(2)).sum::<f64>().sqrt();
                let cos = (0..3).map(|a| sun[a] * p.sun[a]).sum::<f64>().clamp(-1.0, 1.0);
                sun_deg = cos.acos().to_degrees();
                ResetCause { first: false, cut: moved > CUT_VOXELS, light: sun_deg > LIGHT_JUMP_DEG, lights: p.lights != history.lights, forced: force_reset }
            }
        };
        history.age_cap = s.age_cap(sun_deg);
        let prev = history.prev.map_or(*cam, |p| p.cam);
        let mut f = 0;
        for (on, bit) in [(cause.any(), flags::RESET), (s.bilinear, flags::BILINEAR), (faults.reset_always, flags::FAULT_RESET_ALWAYS), (faults.never_reject, flags::FAULT_NEVER_REJECT), (faults.no_fresh, flags::FAULT_NO_FRESH)] {
            if on {
                f |= bit;
            }
        }
        let v = |a: [f64; 3], sc: f64, w: f32| [(a[0] * sc) as f32, (a[1] * sc) as f32, (a[2] * sc) as f32, w];
        let p = Params {
            eye: v(cam.eye, 1.0, cam.near as f32),
            forward: v(cam.forward, 1.0, 0.0),
            right: v(cam.right, cam.tan_half_x, f32::from_bits(cam.width)),
            up: v(cam.up, cam.tan_half_y, f32::from_bits(cam.height)),
            prev_eye: v(prev.eye, 1.0, prev.near as f32),
            prev_fwd: v(prev.forward, 1.0, f32::from_bits(f)),
            prev_right: v(prev.right, 1.0 / prev.tan_half_x, f32::from_bits(history.age_cap)),
            prev_up: v(prev.up, 1.0 / prev.tan_half_y, 0.0),
        };
        let dev = &gpu.device;
        let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
        let rows: Vec<u8> = relight_rows(sun, relight).iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        // The previous frame's relight reads finish before this frame's update (write after read).
        let war = [vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER).dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)];
        // The shade pass's radiance writes, the relight update, and the previous frame's history reads and writes.
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
        unsafe {
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&war));
            dev.cmd_update_buffer(cmd, history.relight.buffer, 0, &rows);
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.sets[history.parity]], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            dev.cmd_dispatch(cmd, cam.width.div_ceil(8), cam.height.div_ceil(8), 1);
        }
        history.parity = 1 - history.parity;
        history.prev = Some(Prev { cam: *cam, sun, lights: history.lights });
        history.frames += 1;
        cause
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
