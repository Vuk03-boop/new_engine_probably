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
//! so that the history spans at most `sun_tolerance_deg` of sun motion. 4B (Amendment 3): not below
//! `sun_cap_min_elevation_deg` (−12°), where the sky gives no light that matters.
//!
//! 4B (ADR-0006 Amendment 2), emitters: while the lights are on, a box also carries a bound on the
//! emitter power the edit changed ([`changed_power`], rule E1) and the emitters it may shadow most
//! ([`Relight::with_lights`], rule E2); a pixel restarts as relit when either bound on the change of
//! its direct emitter light is at least [`EMITTER_TOLERANCE`] of its history's luminance. A change of
//! the lights' state ([`History::set_lights`]) is a light jump: a global reset.

use std::mem::size_of;

use ash::vk;
use memory::Category;

use light::emitters::{luminance, EmitterTable};
use world::{MaterialId, VoxelCoord};

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
/// 4B: emitters listed per box for rule E2.
pub const BOX_LIGHTS: usize = 8;
/// 4B: a pixel restarts when a bound on the change of its direct emitter light is at least this
/// fraction of its history's luminance (as 3E's sky radius, about 2%).
pub const EMITTER_TOLERANCE: f64 = 0.02;
/// Rows per box: lo, hi, then two per listed emitter.
const BOX_ROWS: usize = 2 + 2 * BOX_LIGHTS;
/// Header rows: sun and box count; the emitter tolerance.
const HEAD_ROWS: usize = 2;
const RELIGHT_BYTES: u64 = 16 * (HEAD_ROWS + BOX_ROWS * MAX_RELIGHT_BOXES) as u64;

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
    /// 4B (ADR-0006 Amendment 3): below this sun elevation (degrees; default −12°) the sun-motion
    /// cap is off, since the sky's light there is at most 1.1 × 10⁻⁸ units on an albedo-0.3 surface
    /// (`light::sky` `diagnostic_skylight_below_horizon`). The light-jump reset still applies.
    pub sun_cap_min_elevation_deg: f64,
    /// 3E: resample the history bilinearly (the 3D behaviour) instead of with Catmull-Rom.
    pub bilinear: bool,
}

impl Default for TemporalSettings {
    fn default() -> TemporalSettings {
        TemporalSettings { max_age: 64, sun_tolerance_deg: 0.5, sun_cap_min_elevation_deg: -12.0, bilinear: false }
    }
}

impl TemporalSettings {
    /// The age cap for a frame in which the sun moved `sun_deg` and ended at `elevation_deg`.
    pub fn age_cap(&self, sun_deg: f64, elevation_deg: f64) -> u32 {
        let max = self.max_age.max(1);
        if sun_deg > 0.0 && elevation_deg >= self.sun_cap_min_elevation_deg {
            ((self.sun_tolerance_deg / sun_deg).floor().clamp(1.0, max as f64)) as u32
        } else {
            max
        }
    }
}

/// 4B: an emitter a box may shadow (rule E2): its centre, half-diagonal and luminance, and the
/// potential it was chosen by, Φ / (d² + 1) with d from the centre to the box (voxels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxLight {
    pub centre: [f64; 3],
    pub radius: f64,
    pub luminance: f64,
    pub potential: f64,
}

/// Voxels an edit changed this frame: `lo` inclusive, `hi` exclusive. 4B: with the lights on, also
/// the bound on the emitter power it changed (E1) and the emitters it may shadow (E2).
#[derive(Clone, Debug, PartialEq)]
pub struct Relight {
    pub lo: [i32; 3],
    pub hi: [i32; 3],
    /// Upper bound on the emitter power added or removed (π · area · luminance, voxels², ADR-0005
    /// units; [`changed_power`]). 0: no light changed, or the lights are off.
    pub power: f64,
    /// At most [`BOX_LIGHTS`], largest potential first. Empty: the lights are off.
    pub lights: Vec<BoxLight>,
}

impl Relight {
    /// A box without emitter terms (the 3E rules only).
    pub fn new(lo: [i32; 3], hi: [i32; 3]) -> Relight {
        Relight { lo, hi, power: 0.0, lights: Vec::new() }
    }

    /// The smallest box holding both; powers add, and the lights keep the [`BOX_LIGHTS`] largest.
    pub fn union(self, o: Relight) -> Relight {
        let mut lights = self.lights;
        lights.extend(o.lights);
        lights.sort_by(|a, b| b.potential.total_cmp(&a.potential));
        lights.truncate(BOX_LIGHTS);
        Relight { lo: std::array::from_fn(|a| self.lo[a].min(o.lo[a])), hi: std::array::from_fn(|a| self.hi[a].max(o.hi[a])), power: self.power + o.power, lights }
    }

    /// 4B: this box with the emitter terms of `table` (the lights are on): `power` from
    /// [`changed_power`], and the [`BOX_LIGHTS`] emitters of largest Φ / (d² + 1), d from the emitter's
    /// centre to the box.
    pub fn with_lights(mut self, power: f64, table: &EmitterTable) -> Relight {
        self.power = power;
        let mut all: Vec<BoxLight> = table
            .emitters
            .iter()
            .map(|e| {
                let centre: [f64; 3] = std::array::from_fn(|a| e.p0[a] + 0.5 * (e.eu[a] + e.ev[a]));
                let radius = 0.5 * (0..3).map(|a| (e.eu[a] + e.ev[a]).powi(2)).sum::<f64>().sqrt();
                let y = luminance(e.radiance);
                let d2: f64 = (0..3).map(|a| (self.lo[a] as f64 - centre[a]).max(centre[a] - self.hi[a] as f64).max(0.0).powi(2)).sum();
                BoxLight { centre, radius, luminance: y, potential: std::f64::consts::PI * e.area * y / (d2 + 1.0) }
            })
            .filter(|l| l.potential > 0.0)
            .collect();
        all.sort_by(|a, b| b.potential.total_cmp(&a.potential));
        all.truncate(BOX_LIGHTS);
        self.lights = all;
        self
    }

    /// The box as the shader tests it: grown by a 1-voxel margin (the sun disk's penumbra up to about
    /// 400 voxels away), with sky radius 4 x its largest side + 1 voxel (outside it, removing or
    /// adding the box changes a point's sky irradiance by less than about 2%); 4B: the hi row's w is
    /// the E1 coefficient ΔΦ / π².
    pub fn rows(&self) -> [[f32; 4]; 2] {
        let side = (0..3).map(|a| self.hi[a] - self.lo[a]).max().unwrap_or(0).max(1) as f32;
        let (l, h) = (self.lo.map(|v| v as f32 - 1.0), self.hi.map(|v| v as f32 + 1.0));
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        [[l[0], l[1], l[2], 4.0 * side + 1.0], [h[0], h[1], h[2], (self.power / pi2) as f32]]
    }

    /// 4B: the two rows of each listed emitter (centre and grown radius; luminance), [`BOX_LIGHTS`]
    /// pairs, unused ones zero (luminance 0 is skipped).
    pub fn light_rows(&self) -> Vec<[f32; 4]> {
        let mut rows = Vec::with_capacity(2 * BOX_LIGHTS);
        for k in 0..BOX_LIGHTS {
            match self.lights.get(k) {
                Some(l) => {
                    rows.push([l.centre[0] as f32, l.centre[1] as f32, l.centre[2] as f32, (l.radius + 1.0) as f32]);
                    rows.push([l.luminance as f32, 0.0, 0.0, 0.0]);
                }
                None => rows.extend([[0.0; 4]; 2]),
            }
        }
        rows
    }
}

/// 4B rule E1: an upper bound on the emitter power (π · area · luminance, voxels², ADR-0005 units) that
/// setting `edits` adds or removes. Every face of an edited voxel whose material emits before
/// (`before`) or after, and the face of each emitting face neighbour that the edit may cover or
/// expose. `emission` is per material index (`light::emitters::emission`).
pub fn changed_power(edits: &[(VoxelCoord, Option<MaterialId>)], before: impl Fn(VoxelCoord) -> Option<MaterialId>, emission: &[[f64; 3]]) -> f64 {
    let y = |m: Option<MaterialId>| m.and_then(|m| emission.get(m.raw() as usize)).map_or(0.0, |l| luminance(*l));
    let mut sum = 0.0;
    for &(v, m) in edits {
        sum += 6.0 * (y(before(v)) + y(m));
        for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
            sum += y(before(VoxelCoord::new(v.x + dx, v.y + dy, v.z + dz)));
        }
    }
    std::f64::consts::PI * sum
}

/// The relight buffer's contents: sun direction and count; the emitter tolerance; then per box its
/// two rows and [`BOX_LIGHTS`] light row pairs (at most [`MAX_RELIGHT_BOXES`] boxes; the rest are
/// merged into the last one).
pub fn relight_rows(sun: [f64; 3], boxes: &[Relight]) -> Vec<[f32; 4]> {
    let mut merged: Vec<Relight> = boxes.iter().take(MAX_RELIGHT_BOXES).cloned().collect();
    if boxes.len() > MAX_RELIGHT_BOXES {
        merged[MAX_RELIGHT_BOXES - 1] = boxes[MAX_RELIGHT_BOXES - 1..].iter().cloned().reduce(Relight::union).unwrap();
    }
    let mut rows = vec![[sun[0] as f32, sun[1] as f32, sun[2] as f32, merged.len() as f32], [EMITTER_TOLERANCE as f32, 0.0, 0.0, 0.0]];
    for b in &merged {
        rows.extend(b.rows());
        rows.extend(b.light_rows());
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
    /// 4B: whether the lights were on.
    lights: bool,
}

/// Why this frame was a global reset (all false: history carried over).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResetCause {
    pub first: bool,
    pub cut: bool,
    pub light: bool,
    pub forced: bool,
    /// 4B: the lights were switched on or off (a light jump).
    pub lights: bool,
}

impl ResetCause {
    pub fn any(&self) -> bool {
        self.first || self.cut || self.light || self.forced || self.lights
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

    /// 4B: the lights' state for the next frame recorded. A change from the previous frame's state is a
    /// light jump (ADR-0006 Amendment 2): that frame resets every history.
    pub fn set_lights(&mut self, on: bool) {
        self.lights = on;
    }

    pub fn lights(&self) -> bool {
        self.lights
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
    /// `relight` lists the voxel boxes edited since the previous frame (3E; 4B: with their emitter
    /// terms while the lights are on). The shade pass must
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
                ResetCause { first: false, cut: moved > CUT_VOXELS, light: sun_deg > LIGHT_JUMP_DEG, forced: force_reset, lights: p.lights != history.lights }
            }
        };
        history.age_cap = s.age_cap(sun_deg, light::sun::elevation_deg(sun));
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::emitters::table;
    use crate::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
    use derived::{Config, Merge, Pipeline};
    use world::scene::{street_night, Dressing};
    use world::{BrickKey, Transaction, World};

    fn rows_of(rows: &[[f32; 4]], b: usize) -> &[[f32; 4]] {
        &rows[HEAD_ROWS + BOX_ROWS * b..HEAD_ROWS + BOX_ROWS * (b + 1)]
    }

    fn light(p: f64) -> BoxLight {
        BoxLight { centre: [p, 2.0 * p, 3.0], radius: 0.5, luminance: p, potential: p }
    }

    /// 4B (ADR-0006 Amendment 3): the sun-motion cap holds down to −12° and is off below; the light
    /// jump still resets. Viewer speeds: 0.0625° per frame (60 fps) and 0.03° (about 120 fps).
    #[test]
    fn sun_cap_is_off_when_the_sun_is_well_below_the_horizon() {
        let s = TemporalSettings::default();
        let path = light::sun::SunPath::default();
        let (blue_hour, night) = (light::sun::elevation_deg(path.direction(18.5)), light::sun::elevation_deg(path.direction(21.0)));
        assert!(blue_hour > -12.0 && night < -12.0, "{blue_hour} {night}");
        // Day, blue hour and the cutoff itself keep the 3E cap.
        for e in [45.0, blue_hour, -12.0] {
            assert_eq!(s.age_cap(0.0625, e), 8, "elevation {e}");
            assert_eq!(s.age_cap(0.03, e), 16, "elevation {e}");
        }
        // Below it the history reaches max_age.
        for e in [-12.001, night] {
            for d in [0.0, 0.03, 0.0625, 0.9] {
                assert_eq!(s.age_cap(d, e), 64, "elevation {e}, {d}°/frame");
            }
        }
        // Planted fault: with the cutoff at −90° the night keeps the day's cap.
        let never = TemporalSettings { sun_cap_min_elevation_deg: -90.0, ..s };
        assert_eq!(never.age_cap(0.0625, night), 8);
    }

    /// 4B C2: the header, the box rows and the light rows are where the shader reads them; more than
    /// 16 boxes merge into the last (powers add, the 8 largest lights stay).
    #[test]
    fn relight_rows_follow_the_shader_layout() {
        let mut boxes: Vec<Relight> = (0..20)
            .map(|i| {
                let mut b = Relight::new([i, 0, 0], [i + 2, 1, 1]);
                b.power = 1.0 + i as f64;
                b.lights = (0..3).map(|k| light((10 * i + k) as f64 + 1.0)).collect();
                b
            })
            .collect();
        boxes[0].lights.clear();
        let rows = relight_rows([0.0, 1.0, 0.0], &boxes);
        assert_eq!(rows.len(), HEAD_ROWS + BOX_ROWS * MAX_RELIGHT_BOXES);
        assert_eq!((rows.len() * 16) as u64, RELIGHT_BYTES);
        assert_eq!(rows[0], [0.0, 1.0, 0.0, 16.0]);
        assert_eq!(rows[1][0], EMITTER_TOLERANCE as f32);
        let pi2 = (std::f64::consts::PI * std::f64::consts::PI) as f32;
        let b1 = rows_of(&rows, 1);
        assert_eq!(b1[0], [0.0, -1.0, -1.0, 4.0 * 2.0 + 1.0]);
        assert_eq!(b1[1], [4.0, 2.0, 2.0, (2.0 / (std::f64::consts::PI * std::f64::consts::PI)) as f32]);
        assert_eq!(b1[2], [11.0, 22.0, 3.0, 1.5]);
        assert_eq!(b1[3], [11.0, 0.0, 0.0, 0.0]);
        assert_eq!(b1[2 + 2 * 3], [0.0; 4], "unused light rows are zero");
        assert!(rows_of(&rows, 0)[2..].iter().all(|r| *r == [0.0; 4]), "no lights: zero rows");
        // Boxes 15..19 merged: the union box, powers 16 + 17 + 18 + 19 + 20, the 8 largest lights.
        let last = rows_of(&rows, 15);
        assert_eq!(last[0][0], 14.0);
        assert_eq!(last[1][0], 21.0 + 1.0);
        assert!((last[1][3] - (90.0 / pi2)).abs() < 1e-5);
        let lum: Vec<f32> = (0..BOX_LIGHTS).map(|k| last[3 + 2 * k][0]).collect();
        assert_eq!(lum, vec![193.0, 192.0, 191.0, 183.0, 182.0, 181.0, 173.0, 172.0]);
    }

    fn drain(p: &mut Pipeline, w: &World) -> BTreeSet<BrickKey> {
        loop {
            let jobs = p.dispatch(w);
            if jobs.is_empty() {
                break;
            }
            for j in jobs {
                p.complete(w, j.run());
            }
        }
        p.try_publish(w).map(|x| x.groups.into_iter().flatten().collect()).unwrap_or_default()
    }

    fn full(p: &mut Pipeline) -> BTreeMap<RegionKey, RegionMesh> {
        let t = p.acquire();
        let keys: Vec<BrickKey> = p.current().keys().collect();
        let meshes: Vec<_> = keys.iter().map(|&k| (k, p.read(&t, k).unwrap().unwrap().clone())).collect();
        p.release(t).unwrap();
        build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk)
    }

    const DIRS: [(i32, i32, i32); 6] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

    /// The emitting unit faces (voxel, direction) around `voxels` and their luminance: the exact
    /// face-level oracle.
    fn emitting_faces(w: &World, voxels: &[VoxelCoord], emission: &[[f64; 3]]) -> BTreeMap<(i32, i32, i32, usize), f64> {
        let mut out = BTreeMap::new();
        let mut near = BTreeSet::new();
        for v in voxels {
            near.insert((v.x, v.y, v.z));
            for (dx, dy, dz) in DIRS {
                near.insert((v.x + dx, v.y + dy, v.z + dz));
            }
        }
        for &(x, y, z) in &near {
            let Some(m) = w.get(VoxelCoord::new(x, y, z)) else { continue };
            let l = luminance(emission[m.raw() as usize]);
            if l <= 0.0 {
                continue;
            }
            for (k, (dx, dy, dz)) in DIRS.iter().enumerate() {
                if w.get(VoxelCoord::new(x + dx, y + dy, z + dz)).is_none() {
                    out.insert((x, y, z, k), l);
                }
            }
        }
        out
    }

    /// 4B C2: `changed_power` bounds the gross emitter power an edit adds or removes (the unit faces
    /// that stop or start emitting, which is also >= the net change of the table's total power, from
    /// from-scratch tables through the 1C pipeline), within 20 x; and is 0 for an edit that touches no
    /// emitting voxel or neighbour. The listed lights are the 8 largest by Φ / (d² + 1).
    #[test]
    fn changed_power_bounds_the_edit() {
        let (mut w, _) = street_night(Dressing::Full);
        let emission = light::emitters::emission(w.materials()).unwrap();
        let mats = w.materials().clone();
        let id = |n: &str| mats.id_of(n).unwrap();
        let mut p = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
        p.mark_all(&w);
        drain(&mut p, &w);
        let total = |p: &mut Pipeline| {
            let snap = p.current().id.raw();
            table(&full(p), &mats, snap).unwrap().emitters.iter().map(|e| e.power).sum::<f64>()
        };
        let first = |w: &World, m| w.occupied().find(|&(_, x)| x == m).map(|(v, _)| v).unwrap();
        let lamp = first(&w, id("lamp"));
        let bulb = first(&w, id("bulb"));
        let air = VoxelCoord::new(136, 40, 64);
        assert_eq!(w.get(air), None);
        let quiet = w
            .occupied()
            .find(|&(v, m)| luminance(emission[m.raw() as usize]) == 0.0 && DIRS.iter().all(|(dx, dy, dz)| w.get(VoxelCoord::new(v.x + dx, v.y + dy, v.z + dz)).is_none_or(|n| luminance(emission[n.raw() as usize]) == 0.0)))
            .unwrap()
            .0;
        let steps: Vec<(&str, VoxelCoord, Option<MaterialId>)> = vec![("lamp", lamp, None), ("bulb", bulb, None), ("neon in open air", air, Some(id("neon_pink"))), ("quiet", quiet, None)];
        for (label, v, m) in steps {
            let edits = [(v, m)];
            let before_total = total(&mut p);
            let faces_before = emitting_faces(&w, &[v], &emission);
            let bound = changed_power(&edits, |c| w.get(c), &emission);
            let mut tx = Transaction::new();
            tx.set(v, m);
            let applied = w.apply(&tx).unwrap();
            p.notify_edits(&applied.changed);
            drain(&mut p, &w);
            let after_total = total(&mut p);
            let faces_after = emitting_faces(&w, &[v], &emission);
            let keys: BTreeSet<_> = faces_before.keys().chain(faces_after.keys()).copied().collect();
            let gross = std::f64::consts::PI * keys.iter().filter(|k| faces_before.get(k) != faces_after.get(k)).map(|k| faces_before.get(k).unwrap_or(&0.0) + faces_after.get(k).unwrap_or(&0.0)).sum::<f64>();
            let net = (after_total - before_total).abs();
            eprintln!("C2 {label}: bound {bound:.4e}, gross {gross:.4e}, net table change {net:.4e}");
            if label == "quiet" {
                assert_eq!(bound, 0.0);
                assert_eq!(gross, 0.0);
                assert!(net <= 1e-12 * before_total);
            } else {
                assert!(gross > 0.0, "{label}: the edit changes emitting faces");
                assert!(net <= gross * (1.0 + 1e-9), "{label}: the face oracle covers the table's net change");
                assert!(bound >= gross * (1.0 - 1e-12), "{label}: the bound holds");
                assert!(bound <= 20.0 * gross, "{label}: the bound is within 20x");
            }
        }
        // The listed lights: the 8 largest potentials by brute force.
        let t = table(&full(&mut p), &mats, 1).unwrap();
        let b = Relight::new([lamp.x - 2, lamp.y - 2, lamp.z - 2], [lamp.x + 2, lamp.y + 2, lamp.z + 2]).with_lights(1.0, &t);
        let mut all: Vec<f64> = t
            .emitters
            .iter()
            .map(|e| {
                let c: [f64; 3] = std::array::from_fn(|a| e.p0[a] + 0.5 * (e.eu[a] + e.ev[a]));
                let d2: f64 = (0..3).map(|a| (b.lo[a] as f64 - c[a]).max(c[a] - b.hi[a] as f64).max(0.0).powi(2)).sum();
                e.power / (d2 + 1.0)
            })
            .collect();
        all.sort_by(|a, b| b.total_cmp(a));
        assert_eq!(b.lights.len(), BOX_LIGHTS);
        for (l, want) in b.lights.iter().zip(&all) {
            assert!((l.potential - want).abs() <= 1e-12 * want, "{} vs {want}", l.potential);
        }
        assert_eq!(b.power, 1.0);
    }
}
