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
//!
//! 4B part 2 (ADR-0006 Amendment 2), relight for lights: on the frame after an edit the caller also
//! hands the history the light rows of the edit ([`light_rows`], [`History::set_light_rows`]): the
//! emitters it changed, and per edit box the unchanged emitters that could cast new shadows through
//! it. With the lights on, the lit module (`-D EMITTERS`) rejects an otherwise accepted pixel as
//! "relit by a light" (reason 10) when a row's light matters there and the edit can change it
//! ([`relit_by_light`] is the same rule on the host). With the lights off the M3 module runs.

use std::mem::size_of;

use ash::vk;
use memory::Category;

use light::emitters::{luminance, Emitter, EmitterTable};

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::raster::{Camera, Targets};
use crate::reflect::{self, Field, Param};
use crate::staging::download;
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/temporal.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/temporal.json"));
/// 4B: the lit module (relight for lights).
pub const SPIRV_LIT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/temporal_lit.spv"));
pub const REFLECTION_LIT: &str = include_str!(concat!(env!("OUT_DIR"), "/temporal_lit.json"));

/// A camera move longer than this in one frame is a cut (ADR-0006), voxels (4 m).
pub const CUT_VOXELS: f64 = 64.0;
/// A sun move larger than this in one frame is a light jump (ADR-0006), degrees.
pub const LIGHT_JUMP_DEG: f64 = 1.0;
/// Relight boxes per frame; more are merged into the last one.
pub const MAX_RELIGHT_BOXES: usize = 16;
const RELIGHT_BYTES: u64 = 16 * (1 + 2 * MAX_RELIGHT_BOXES as u64);
/// 4B: light rows per frame (ADR-0006 Amendment 2), of which at most [`MAX_CHANGED_ROWS`] changed
/// emitters (more are merged into the last one) and [`SHADOW_ROWS_PER_BOX`] per edit box.
pub const MAX_LIGHT_ROWS: usize = 64;
pub const MAX_CHANGED_ROWS: usize = 16;
pub const SHADOW_ROWS_PER_BOX: usize = 4;
/// 4B: a light matters at a pixel when its irradiance bound over π is at least this fraction of the
/// history's luminance.
pub const LIGHT_FRACTION: f32 = 0.02;
/// The light rows follow the box rows: a head (count) and three rows each; about 3 KB.
const LIGHT_BYTES: u64 = 16 * (1 + 3 * MAX_LIGHT_ROWS as u64);

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
    /// 4B (ADR-0006 Amendment 2).
    pub const RELIT_LIGHT: u32 = 10;
    pub const NAMES: [&str; 11] = ["accepted", "sky", "off-screen", "no previous surface", "material", "normal", "disoccluded", "edited", "reset", "relit", "relit by a light"];
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

/// The boxes as the relight buffer holds them: at most [`MAX_RELIGHT_BOXES`], the rest merged into
/// the last one.
pub fn merged_boxes(boxes: &[Relight]) -> Vec<Relight> {
    let mut merged: Vec<Relight> = boxes.iter().take(MAX_RELIGHT_BOXES).copied().collect();
    if boxes.len() > MAX_RELIGHT_BOXES {
        merged[MAX_RELIGHT_BOXES - 1] = boxes[MAX_RELIGHT_BOXES - 1..].iter().copied().reduce(Relight::union).unwrap();
    }
    merged
}

/// The relight buffer's contents: sun direction and count, then two rows per box (at most
/// [`MAX_RELIGHT_BOXES`]; the rest are merged into the last one).
pub fn relight_rows(sun: [f64; 3], boxes: &[Relight]) -> Vec<[f32; 4]> {
    let merged = merged_boxes(boxes);
    let mut rows = vec![[sun[0] as f32, sun[1] as f32, sun[2] as f32, merged.len() as f32]];
    for b in &merged {
        rows.extend(b.rows());
    }
    rows
}

/// 4B: one light row (ADR-0006 Amendment 2): an emitter's bounding sphere, its side, and either
/// "changed" or the edit box it may be shadowed through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightRow {
    pub centre: [f64; 3],
    /// Half the diagonal (the bounding sphere's radius), voxels.
    pub radius: f64,
    /// The outward normal of a one-sided emitter; zero for a row lit from both sides (merged).
    pub normal: [f64; 3],
    /// Y(L_e): the emitted luminance (the largest one of a merged row).
    pub y: f64,
    /// Σ Y(L_e) · A, voxels².
    pub ya: f64,
    /// `None`: the emitter changed; `Some(b)`: unchanged, tested against box `b` of [`merged_boxes`].
    pub shadow_box: Option<usize>,
}

impl LightRow {
    /// Emitter `e`'s row.
    pub fn of(e: &Emitter, shadow_box: Option<usize>) -> LightRow {
        let centre = std::array::from_fn(|a| e.p0[a] + 0.5 * (e.eu[a] + e.ev[a]));
        let radius = 0.5 * (0..3).map(|a| (e.eu[a] + e.ev[a]).powi(2)).sum::<f64>().sqrt();
        let y = luminance(e.radiance);
        LightRow { centre, radius, normal: e.normal, y, ya: y * e.area, shadow_box }
    }

    /// One conservative row for several changed ones: a bounding sphere, both sides, the largest Y
    /// and the summed Y · A.
    pub fn merge(rows: &[LightRow]) -> LightRow {
        let lo: [f64; 3] = std::array::from_fn(|a| rows.iter().map(|r| r.centre[a] - r.radius).fold(f64::INFINITY, f64::min));
        let hi: [f64; 3] = std::array::from_fn(|a| rows.iter().map(|r| r.centre[a] + r.radius).fold(f64::NEG_INFINITY, f64::max));
        let centre: [f64; 3] = std::array::from_fn(|a| 0.5 * (lo[a] + hi[a]));
        let radius = rows.iter().map(|r| (0..3).map(|a| (r.centre[a] - centre[a]).powi(2)).sum::<f64>().sqrt() + r.radius).fold(0.0, f64::max);
        LightRow { centre, radius, normal: [0.0; 3], y: rows.iter().map(|r| r.y).fold(0.0, f64::max), ya: rows.iter().map(|r| r.ya).sum(), shadow_box: None }
    }

    /// The unoccluded irradiance bound at `p`: Y · min(2π, A / d²), d the distance to the sphere.
    pub fn bound(&self, p: [f64; 3]) -> f64 {
        let d = ((0..3).map(|a| (p[a] - self.centre[a]).powi(2)).sum::<f64>().sqrt() - self.radius).max(0.0);
        let two_pi = 2.0 * std::f64::consts::PI * self.y;
        if d > 0.0 {
            two_pi.min(self.ya / (d * d))
        } else {
            two_pi
        }
    }

    /// The three rows the shader reads.
    pub fn rows(&self) -> [[f32; 4]; 3] {
        let (c, n) = (self.centre.map(|x| x as f32), self.normal.map(|x| x as f32));
        let b = self.shadow_box.map_or(-1.0, |b| b as f32);
        [[c[0], c[1], c[2], self.radius as f32], [n[0], n[1], n[2], self.y as f32], [self.ya as f32, b, 0.0, 0.0]]
    }
}

/// 4B: the light rows of an edit (ADR-0006 Amendment 2), at most [`MAX_LIGHT_ROWS`]:
/// - `changed` (the emitters in only one of the old and new tables, `emitters::changed`), at most
///   [`MAX_CHANGED_ROWS`], the rest merged into the last one;
/// - per box of [`merged_boxes`]`(boxes)`, the [`SHADOW_ROWS_PER_BOX`] emitters of `table` not in
///   `changed` with the largest irradiance bound at the box's centre (ties in table order);
/// - rows beyond the cap are left out (the declared approximation, measured by R4).
pub fn light_rows(changed: &[Emitter], table: &EmitterTable, boxes: &[Relight]) -> Vec<LightRow> {
    let mut rows: Vec<LightRow> = changed.iter().take(MAX_CHANGED_ROWS).map(|e| LightRow::of(e, None)).collect();
    if changed.len() > MAX_CHANGED_ROWS {
        let rest: Vec<LightRow> = changed[MAX_CHANGED_ROWS - 1..].iter().map(|e| LightRow::of(e, None)).collect();
        rows[MAX_CHANGED_ROWS - 1] = LightRow::merge(&rest);
    }
    let key = |e: &Emitter| (e.face, e.p0.map(f64::to_bits), e.eu.map(f64::to_bits), e.ev.map(f64::to_bits));
    let gone: std::collections::HashSet<_> = changed.iter().map(key).collect();
    for (b, bx) in merged_boxes(boxes).iter().enumerate() {
        let c: [f64; 3] = std::array::from_fn(|a| 0.5 * (bx.lo[a] + bx.hi[a]) as f64);
        let mut best: Vec<(f64, usize)> = Vec::new();
        for (i, e) in table.emitters.iter().enumerate() {
            if gone.contains(&key(e)) {
                continue;
            }
            let bound = LightRow::of(e, None).bound(c);
            // Keep the SHADOW_ROWS_PER_BOX largest, first in table order on ties.
            let at = best.iter().position(|&(v, _)| bound > v).unwrap_or(best.len());
            if at < SHADOW_ROWS_PER_BOX {
                best.insert(at, (bound, i));
                best.truncate(SHADOW_ROWS_PER_BOX);
            }
        }
        rows.extend(best.iter().map(|&(_, i)| LightRow::of(&table.emitters[i], Some(b))));
    }
    rows.truncate(MAX_LIGHT_ROWS);
    rows
}

/// 4B: the lit module's rule on the host, in the shader's f32 order: whether a pixel at surface
/// point `p` whose (bilinear) history has luminance `yh` is relit by a light, with `boxes` the rows
/// of [`relight_rows`] and `lights` the light rows.
pub fn relit_by_light(p: [f32; 3], yh: f32, boxes: &[[f32; 4]], lights: &[LightRow]) -> bool {
    const TWO_PI: f32 = std::f32::consts::TAU;
    const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    lights.iter().any(|l| {
        let [a, b, c] = l.rows();
        let (centre, normal) = ([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
        let v: [f32; 3] = std::array::from_fn(|k| p[k] - centre[k]);
        if dot(normal, normal) > 0.0 && dot(v, normal) <= 0.0 {
            return false;
        }
        let d = (dot(v, v).sqrt() - a[3]).max(0.0);
        let bound = if d > 0.0 { (TWO_PI * b[3]).min(c[0] / (d * d)) } else { TWO_PI * b[3] };
        if bound * INV_PI < LIGHT_FRACTION * yh {
            return false;
        }
        let Ok(bx) = usize::try_from(c[1] as i32) else { return true };
        let lo: [f32; 3] = std::array::from_fn(|k| boxes[1 + 2 * bx][k] - a[3]);
        let hi: [f32; 3] = std::array::from_fn(|k| boxes[2 + 2 * bx][k] + a[3]);
        let s: [f32; 3] = std::array::from_fn(|k| centre[k] - p[k]);
        let (mut enter, mut leave) = (0.0f32, 1.0f32);
        for k in 0..3 {
            if s[k] == 0.0 {
                if p[k] < lo[k] || p[k] > hi[k] {
                    leave = -1.0;
                }
            } else {
                let (t0, t1) = ((lo[k] - p[k]) / s[k], (hi[k] - p[k]) / s[k]);
                enter = enter.max(t0.min(t1));
                leave = leave.min(t0.max(t1));
            }
        }
        enter <= leave
    })
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
    /// 4B: the light rows of the next frame recorded ([`History::set_light_rows`]).
    light_rows: Vec<LightRow>,
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
        let relight = match alloc.create_buffer(gpu, RELIGHT_BYTES + LIGHT_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, Category::GpuTemporal, Kind::Device) {
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
        Ok(History { width, height, guides: [n(), n()], hist: [n(), n()], state: [n(), n()], motion: n(), relight, parity: 0, prev: None, frames: 0, age_cap: 0, lights: false, light_rows: Vec::new() })
    }

    /// 4B: whether the street's lights are on for the next frame recorded (off by default). A change
    /// from the previous frame is a light jump: that frame resets every pixel.
    pub fn set_lights(&mut self, on: bool) {
        self.lights = on;
    }

    /// 4B: the light rows of the next frame recorded (the frame after an edit is shown; at most
    /// [`MAX_LIGHT_ROWS`] are kept). They apply to that frame only, and only with the lights on.
    pub fn set_light_rows(&mut self, rows: &[LightRow]) {
        self.light_rows = rows.iter().take(MAX_LIGHT_ROWS).copied().collect();
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
    /// The M3 module, and (4B) the lit one, used while the lights are on.
    pipeline: vk::Pipeline,
    lit: vk::Pipeline,
}

impl Temporal {
    pub fn new(gpu: &Gpu) -> Result<Temporal> {
        Self::with_reflection(gpu, REFLECTION)
    }

    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Temporal> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        reflect::check(REFLECTION_LIT, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let mut bindings: Vec<_> = (0..4).map(|i| b(i, vk::DescriptorType::SAMPLED_IMAGE)).collect();
        bindings.extend((4..14).map(|i| b(i, vk::DescriptorType::STORAGE_BUFFER)));
        let set_layout = unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.vk("vkCreateDescriptorSetLayout")?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(size_of::<Params>() as u32)];
        let sl = [set_layout];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc), None) }.vk("vkCreatePipelineLayout")?;
        let pipeline = Self::pipeline(gpu, layout, SPIRV)?;
        let lit = Self::pipeline(gpu, layout, SPIRV_LIT)?;
        Ok(Temporal { set_layout, layout, pipeline, lit })
    }

    fn pipeline(gpu: &Gpu, layout: vk::PipelineLayout, spirv: &[u8]) -> Result<vk::Pipeline> {
        let dev = &gpu.device;
        let (chunks, rest) = spirv.as_chunks::<4>();
        assert!(rest.is_empty(), "SPIR-V is whole words");
        let words: Vec<u32> = chunks.iter().map(|&c| u32::from_le_bytes(c)).collect();
        let module = unsafe { dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.vk("vkCreateShaderModule")?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { dev.destroy_shader_module(module, None) };
        Ok(pipeline.map_err(|(_, e)| GpuError::Vk { call: "vkCreateComputePipelines", result: e })?[0])
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
        // 4B: with the lights on, the lit module and this frame's light rows after the box rows.
        let lights = history.lights.then(|| {
            let mut v = vec![[history.light_rows.len() as f32, 0.0, 0.0, 0.0]];
            v.extend(history.light_rows.iter().flat_map(LightRow::rows));
            v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        });
        history.light_rows.clear();
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
            if let Some(l) = &lights {
                dev.cmd_update_buffer(cmd, history.relight.buffer, RELIGHT_BYTES, l);
            }
            dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, if lights.is_some() { self.lit } else { self.pipeline });
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
            gpu.device.destroy_pipeline(self.lit, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use light::emitters::{EmitterId, EmitterQuad};
    use world::{MaterialParams, MaterialRegistry};

    fn table(quads: &[(u32, i32, i32, i32)]) -> EmitterTable {
        // (quad, plane, u0, v0): 1 × 1 quads facing +y (face 3) in region 0.
        let mut r = MaterialRegistry::new();
        let hot = r.register("hot", MaterialParams { base_color: [0.5; 3], emissive: [10.0, 5.0, 1.0] }).unwrap();
        let q = quads.iter().map(|&(i, plane, u0, v0)| EmitterQuad { id: EmitterId { key: [0; 3], quad: i, snapshot: 1 }, material: hot, face: 3, plane, u0, v0, u1: u0 + 1, v1: v0 + 1 });
        EmitterTable::build(&r, 1, q).unwrap()
    }

    /// The lit module matches the host layout without a device, and the shader's light head sits
    /// after the box rows.
    #[test]
    fn lit_module_matches_the_host_layout() {
        reflect::check(REFLECTION, &host_layout()).unwrap();
        reflect::check(REFLECTION_LIT, &host_layout()).unwrap();
        let src = include_str!("../shaders/temporal.slang");
        assert!(src.contains(&format!("LIGHT_HEAD = {}u;", 1 + 2 * MAX_RELIGHT_BOXES)));
        assert!(src.contains(&format!("LIGHT_FRACTION = {LIGHT_FRACTION};")));
        assert_eq!(reason::NAMES.len() as u32, reason::RELIT_LIGHT + 1);
    }

    /// The rows: changed emitters first (merged beyond the cap), then per box the unchanged emitters
    /// with the largest bounds at its centre, capped at `MAX_LIGHT_ROWS`.
    #[test]
    fn light_rows_follow_the_rule() {
        let t = table(&(0..40).map(|i| (i, 10, 4 * i as i32, 0)).collect::<Vec<_>>());
        let b = Relight { lo: [0, 20, 0], hi: [3, 23, 3] };
        // Two changed: two rows without a box, then the 4 nearest unchanged ones for the box.
        let changed = [t.emitters[0], t.emitters[1]];
        let rows = light_rows(&changed, &t, &[b]);
        assert_eq!(rows.len(), 2 + SHADOW_ROWS_PER_BOX);
        assert!(rows[..2].iter().all(|r| r.shadow_box.is_none() && r.normal == [0.0, 1.0, 0.0]));
        let near: Vec<[f64; 3]> = rows[2..].iter().map(|r| r.centre).collect();
        assert!(rows[2..].iter().all(|r| r.shadow_box == Some(0)));
        assert!(!near.iter().any(|c| *c == rows[0].centre || *c == rows[1].centre), "a changed emitter is not a shadow row");
        assert_eq!(near.iter().map(|c| c[1] as i32).collect::<Vec<_>>(), [10; 4]);
        let xs: Vec<f64> = near.iter().map(|c| c[2]).collect();
        assert_eq!(xs, [8.5, 12.5, 16.5, 20.5], "the nearest to the box, largest bound first");
        let r = LightRow::of(&t.emitters[0], None);
        assert!((r.radius - 0.5f64.sqrt()).abs() < 1e-12 && (r.ya - r.y).abs() < 1e-12);
        // 20 changed: 16 rows, the last one merged (both sides, the summed Y·A, a sphere around all).
        let changed: Vec<Emitter> = t.emitters[..20].to_vec();
        let rows = light_rows(&changed, &t, &[]);
        assert_eq!(rows.len(), MAX_CHANGED_ROWS);
        let m = rows[MAX_CHANGED_ROWS - 1];
        assert_eq!(m.normal, [0.0; 3]);
        assert!((m.ya - 5.0 * r.ya).abs() < 1e-9);
        for e in &changed[MAX_CHANGED_ROWS - 1..] {
            let c = LightRow::of(e, None);
            assert!((0..3).map(|a| (c.centre[a] - m.centre[a]).powi(2)).sum::<f64>().sqrt() + c.radius <= m.radius + 1e-9);
        }
        // 20 boxes (merged to 16): 16 changed + 16 × 4 shadow rows, capped at 64.
        let boxes: Vec<Relight> = (0..20).map(|i| Relight { lo: [0, 20 + i, 0], hi: [1, 21 + i, 1] }).collect();
        assert_eq!(light_rows(&changed, &t, &boxes).len(), MAX_LIGHT_ROWS);
    }

    /// The host rule: in front, the 2% threshold, changed, and the grown box on the segment.
    #[test]
    fn host_rule_decides_as_designed() {
        let t = table(&[(0, 10, 0, 0)]);
        let e = t.emitters[0];
        let changed = [LightRow::of(&e, None)];
        let c = changed[0].centre.map(|x| x as f32);
        let none = relight_rows([0.0, 1.0, 0.0], &[]);
        // In front and bright relative to the history: relit; behind: not.
        assert!(relit_by_light([c[0], 12.0, c[2]], 0.01, &none, &changed));
        assert!(!relit_by_light([c[0], 8.0, c[2]], 0.01, &none, &changed));
        // Far away the bound falls below 2% of the history.
        let p = [c[0], 10.0 + 400.0, c[2]];
        let bound = changed[0].bound(p.map(|x| x as f64));
        assert!(relit_by_light(p, (bound / std::f64::consts::PI / 0.02 * 0.99) as f32, &none, &changed));
        assert!(!relit_by_light(p, (bound / std::f64::consts::PI / 0.02 * 1.01) as f32, &none, &changed));
        // A shadow row: relit only when the segment to the centre crosses the grown box.
        let bx = Relight { lo: [-2, 13, -2], hi: [3, 14, 3] };
        let boxes = relight_rows([0.0, 1.0, 0.0], &[bx]);
        let shadow = [LightRow::of(&e, Some(0))];
        assert!(relit_by_light([c[0], 20.0, c[2]], 0.0, &boxes, &shadow), "through the box");
        assert!(!relit_by_light([c[0], 11.0, c[2]], 0.0, &boxes, &shadow), "below the grown box (margin 1 + radius)");
        assert!(relit_by_light([c[0], 11.5, c[2]], 0.0, &boxes, &shadow), "inside the grown box");
        assert!(!relit_by_light([c[0] + 60.0, 20.0, c[2]], 0.0, &boxes, &shadow), "beside the box");
    }
}
