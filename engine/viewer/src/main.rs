//! The street block, rasterized into the ADR-0003 G-buffer and shown through the debug views, with
//! a walking camera (M1), a free-fly camera, 2 frames in flight and per-pass GPU timing. M2 (2E):
//! voxel edits reach both the raster G-buffer and the ray-query representation, and R shows either.
//!
//! usage: viewer [--merge greedy|none] [--region brick|2x2x2_bricks|chunk|2x2x2_chunks]
//!               [--present mailbox|fifo|immediate] [--view 0-12] [--hour H] [--run-day] [--max-age N] [--no-denoise] [--no-bounce] [--no-sky-correction] [--source raster|ray]
//!               [--camera street|low] [--prefilter-age N] [--scene street|night [--dressing lamps|windows|full|dense]]
//!               [--size WxH] [--frames N [--walk] [--cycle-views] [--edit-script [--edit-size N]] [--resize-at FRAME WxH]] [--log PATH]
//!               [--trace CSV] [--present-wait N]
//!
//! The present mode defaults to MAILBOX (user decision, 2026-09-24; FIFO failed the 60 fps gate on
//! sky-heavy views, see the M1 record).
//!
//! Controls: right mouse button + move to look; F switches between walking and flying; key 0 is
//! the light view (the start: real-time lighting of `gpu::shade` in ADR-0005 units: the direct sun
//! with real shadows (3B), the physically based sky and its light (3C) and one bounce of both (3F)),
//! key 9 the M1 lit view (sun and ambient, no shadows), keys 1-7 the debug views (material, normal,
//! region, brick, surface id, depth, snapshot); - and = change exposure; Esc quits.
//! - Time of day (3B): [ and ] move the sun a quarter hour back / forward along `light::SunPath`
//!   (45° N, equinox); T runs the day (one hour per 4 s). `--hour H` sets the start (default 9).
//! - Temporal accumulation (3D, ADR-0006): on by default, H toggles it; key 8 cycles the history age,
//!   history reason and motion views (`--view 10`, `11`, `12`); `--max-age N` sets the history cap.
//! - Reconstruction (3E): the light view is filtered by `gpu::denoise` (on by default while
//!   accumulating; N toggles it, `--no-denoise` starts with it off). Edits relight the pixels whose
//!   sun or sky light they may change, and the history shortens while the sun moves (ADR-0006
//!   Amendment 1).
//! - One bounce (3F): on by default; B toggles it (the history restarts), `--no-bounce` starts with
//!   it off. Light from two or more bounces is not computed.
//! - Q2 comparison (3G): P switches the filter between the 3E filter (guide prefilter at every age,
//!   the default) and `DenoiseSettings::prefilter_age` = `--prefilter-age N` (default 8: pixels at
//!   least that old compare their own values); the title shows which. `--camera low` starts flying
//!   at the 3A low camera, where the Q2 miss is (the far shadow edge across the road at midday).
//! - Scenes (4A): `--scene street` (the default) is the M3 street block; `--scene night` is
//!   `world::scene::street_night` with `--dressing` (default `full`). The night scene's `GpuScene`
//!   publishes the emitter table with its meshes, and every edit rebuilds it inside the update
//!   (ADR-0003 Amendment 3); the JSON line reports the emitter count and the table's time per edit.
//! - The lights (4B): on a scene with an emitter table (`--scene night`) the lights follow the sun
//!   (on below the horizon); L switches them by hand until the sun next crosses the horizon, and
//!   `--lights auto|on|off` sets the start. While they are on, the shade pass takes `--emitter-spp`
//!   emitter samples at the primary and bounce hits (the lit module), `gpu::compose` adds emission
//!   after reconstruction, and the light view's exposure is automatic (metered log-average, 1 s
//!   adaptation, `light::exposure::adapt`; - and = still offset it). While they are off the frame is
//!   M3's exactly, exposure included. A lights switch resets the history (a light jump). The JSON
//!   reports `lights`, `exposure`, `emitter_spp` and `stale_table_frames`.
//! - Sky correction (S-020): the sky-view table is corrected from the reference baked in
//!   `light/data/sky_reference_v1.bin`; `--no-sky-correction` runs without it. A missing or refused
//!   file prints a warning and runs uncorrected; the JSON line reports `"sky_correction"`.
//! - Frame rate: the title shows the average fps and the 10% / 1% lows (fps over the slowest 10% /
//!   1% of frames) of the last 1000 frames; on exit the same for the whole run is printed and logged.
//! - Editing (M2): left click removes the voxel under the cursor; E or the middle button places a
//!   voxel of the same material against the face under the cursor (not inside the walker). R
//!   switches the views between the raster G-buffer and the ray-query result.
//! - `--edit-script` (with `--frames`): every 40 frames, remove the voxel at the screen centre, and
//!   20 frames later place one against the face then at the centre. Each edit's stages and its
//!   edit-to-visible time (edit start to the completion of the first frame showing it) are logged.
//!   `--edit-size N` (3G) removes a box of N³ voxels around the voxel at the centre instead, and
//!   restores its previous contents half a period later; the JSON line lists every edit's
//!   edit-to-visible time (`visible_all_ms`).
//! - Walking (the start): W A S D to walk, Shift to run, Space to jump. Collision, gravity and
//!   step-up come from the `walk` crate, against the authoritative world. Falling far below the
//!   world puts you back at the start.
//! - Flying: W A S D to move; Space / Ctrl up / down; Shift x4 speed; the mouse wheel sets speed.
//!
//! `--frames N` is the scripted run, indexed by frame (not time), so runs are repeatable: the camera
//! flies a fixed path down the street, or with `--walk` the walker follows a fixed input script at
//! 1/60 s of simulated time per frame (up the curb, along the shop fronts, back over the curb with
//! jumps). After N frames one JSON summary line is printed (and appended to `--log`), and the
//! process exits non-zero if validation reported an error or a warning, or the walker ever
//! overlapped a solid voxel.
//! `--resize-at` asks the window for a new size at that frame, so swapchain and target recreation
//! run under the same checks.
//!
//! The meshes come from a 1C `derived::Pipeline` snapshot. Every frame acquires a reader token on
//! that snapshot and `FrameReaders` holds it until the frame's timeline value completes. An edit
//! commits to the world, runs the jobs inline, publishes, and updates the `GpuScene` (changed
//! regions' meshes and BLAS, the TLAS); a refused update leaves the previous snapshot on screen and
//! retries next frame. After a successful update the frames in flight are waited for (the
//! acceleration update has already waited for them) and the descriptor sets are rebuilt.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write as _;
use std::time::Instant;

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::compose::{Compose, ComposeBindings, ComposeFaults, ComposeTargets};
use gpu::alloc::Allocator;
use gpu::debug_view::{targets_to_read, DebugView, Lighting, Source, Tables, View};
use gpu::layout::{build_regions, RegionSize};
use gpu::present::{image_to_attachment, image_to_present, Acquired, Frames, Swapchain};
use gpu::raster::{Camera, Faults, Raster, Targets};
use gpu::ray::{RayBindings, RayPrimary, RayTargets};
use gpu::reference::RefMaterials;
use gpu::scene::{affected_regions, region_meshes, Garbage, GpuScene};
use gpu::shade::{self, radiance_to_fragment, Shade, ShadeBindings, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::denoise::{Denoise, DenoiseBindings, DenoiseFaults, DenoiseSettings, DenoiseTargets};
use gpu::temporal::{History, Relight, Temporal, TemporalBindings, TemporalFaults, TemporalSettings};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::timing::{percentile, GpuTimer};
use gpu::{FrameReaders, Gpu, Timeline};
use memory::Category;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};
use world::dims::VOXEL_SIZE_M;
use light::emitters::{Lights, LightsMode};
use world::scene::Dressing;
use world::{scene, BrickKey, MaterialId, Transaction, VoxelCoord, World};

const FRAMES_IN_FLIGHT: usize = 2;
const NEAR: f64 = 0.1;
/// Passes timed on the GPU, in order. `ray_trace` is empty unless the ray-query source is shown;
/// `shade` (3B) is empty unless the light view is shown.
const PASSES: [&str; 4] = ["gbuffer", "debug_view", "ray_trace", "shade"];
/// Seed of the real-time lighting's random streams (ADR-0005 PCG32).
const SHADE_SEED: u32 = 0x3B;
/// Longest edit reach, voxels.
const REACH: f64 = 1024.0;
/// `--edit-script` period in frames.
const EDIT_PERIOD: u64 = 40;

#[derive(Clone, Debug)]
struct Args {
    merge: Merge,
    region: RegionSize,
    present: vk::PresentModeKHR,
    view: View,
    size: (u32, u32),
    frames: Option<u64>,
    walk_script: bool,
    cycle_views: bool,
    resize_at: Option<(u64, u32, u32)>,
    log: Option<String>,
    /// Per-frame CSV: interval and phases.
    trace: Option<String>,
    /// Before each frame, wait until at most this many presents are not yet on the display
    /// (needs present wait). Off by default.
    present_wait: Option<u64>,
    edit_script: bool,
    /// `--edit-script` box side in voxels (1: single voxels, the M2 script).
    edit_size: i32,
    source: Source,
    /// Local solar time of the start (3B).
    hour: f64,
    /// Start with the day running (T), e.g. for scripted runs of the sky updates (3C).
    run_day: bool,
    /// 3D: the history cap (ADR-0006 `max_age`).
    max_age: u32,
    /// 3E: start with the filter off.
    no_denoise: bool,
    no_bounce: bool,
    no_sky_correction: bool,
    /// 3G: start flying at the 3A low camera.
    camera_low: bool,
    /// 3G: the `prefilter_age` P switches to (the 3E filter is `u32::MAX`).
    prefilter_age: u32,
    /// 4A: `--scene night` with its dressing; `None` is the M3 street block.
    night: Option<Dressing>,
    /// 4B: the lights' start state and the emitter samples per vertex.
    lights: LightsMode,
    emitter_spp: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { merge: Merge::Greedy, region: RegionSize::Chunk, present: vk::PresentModeKHR::MAILBOX, view: View::Light, size: (1280, 720), frames: None, walk_script: false, cycle_views: false, resize_at: None, log: None, trace: None, present_wait: None, edit_script: false, edit_size: 1, source: Source::Raster, hour: 9.0, run_day: false, max_age: 64, no_denoise: false, no_bounce: false, no_sky_correction: false, camera_low: false, prefilter_age: 8, night: None, lights: LightsMode::Auto, emitter_spp: 1 };
    let mut dressing = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--merge" => {
                let v = val()?;
                a.merge = *Merge::ALL.iter().find(|m| m.name() == v).ok_or(format!("unknown merge {v}"))?;
            }
            "--region" => {
                let v = val()?;
                a.region = *RegionSize::ALL.iter().find(|r| r.name() == v).ok_or(format!("unknown region size {v}"))?;
            }
            "--present" => {
                a.present = match val()?.as_str() {
                    "fifo" => vk::PresentModeKHR::FIFO,
                    "mailbox" => vk::PresentModeKHR::MAILBOX,
                    "immediate" => vk::PresentModeKHR::IMMEDIATE,
                    v => return Err(format!("unknown present mode {v}")),
                }
            }
            "--view" => {
                let v: usize = val()?.parse().map_err(|e| format!("--view: {e}"))?;
                a.view = match v {
                    0 => View::Light,
                    9 => View::Lit,
                    10 => View::HistoryAge,
                    11 => View::HistoryReason,
                    12 => View::Motion,
                    _ => *View::ALL[..7].get(v.wrapping_sub(1)).ok_or("--view is 0 (light), 1 to 7 (debug), 9 (M1 lit), 10 to 12 (history age, reason, motion)")?,
                };
            }
            "--run-day" => a.run_day = true,
            "--no-denoise" => a.no_denoise = true,
            "--no-bounce" => a.no_bounce = true,
            "--no-sky-correction" => a.no_sky_correction = true,
            "--camera" => {
                a.camera_low = match val()?.as_str() {
                    "street" => false,
                    "low" => true,
                    v => return Err(format!("--camera: {v}")),
                }
            }
            "--scene" => {
                a.night = match val()?.as_str() {
                    "street" => None,
                    "night" => Some(Dressing::default()),
                    v => return Err(format!("--scene: {v}")),
                }
            }
            "--lights" => {
                a.lights = match val()?.as_str() {
                    "auto" => LightsMode::Auto,
                    "on" => LightsMode::On,
                    "off" => LightsMode::Off,
                    v => return Err(format!("--lights: {v}")),
                }
            }
            "--emitter-spp" => {
                a.emitter_spp = val()?.parse().map_err(|e| format!("--emitter-spp: {e}"))?;
                if ![1, 2, 4].contains(&a.emitter_spp) {
                    return Err("--emitter-spp is 1, 2 or 4".into());
                }
            }
            "--dressing" => {
                let v = val()?;
                dressing = Some(*Dressing::ALL.iter().find(|d| d.name() == v).ok_or(format!("unknown dressing {v}"))?);
            }
            "--prefilter-age" => {
                a.prefilter_age = val()?.parse().map_err(|e| format!("--prefilter-age: {e}"))?;
                if a.prefilter_age == 0 {
                    return Err("--prefilter-age is at least 1".into());
                }
            }
            "--max-age" => a.max_age = val()?.parse().map_err(|e| format!("--max-age: {e}"))?,
            "--hour" => {
                a.hour = val()?.parse().map_err(|e| format!("--hour: {e}"))?;
            }
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or("--size is WxH")?;
                a.size = (w.parse().map_err(|e| format!("--size: {e}"))?, h.parse().map_err(|e| format!("--size: {e}"))?);
            }
            "--frames" => a.frames = Some(val()?.parse().map_err(|e| format!("--frames: {e}"))?),
            "--cycle-views" => a.cycle_views = true,
            "--walk" => a.walk_script = true,
            "--edit-script" => a.edit_script = true,
            "--edit-size" => {
                a.edit_size = val()?.parse().map_err(|e| format!("--edit-size: {e}"))?;
                if !(1..=64).contains(&a.edit_size) {
                    return Err("--edit-size is 1 to 64".into());
                }
            }
            "--source" => {
                a.source = match val()?.as_str() {
                    "raster" => Source::Raster,
                    "ray" => Source::Ray,
                    v => return Err(format!("unknown source {v}")),
                }
            }
            "--resize-at" => {
                let f = val()?.parse().map_err(|e| format!("--resize-at: {e}"))?;
                let v = val()?;
                let (w, h) = v.split_once('x').ok_or("--resize-at FRAME WxH")?;
                a.resize_at = Some((f, w.parse().map_err(|e| format!("--resize-at: {e}"))?, h.parse().map_err(|e| format!("--resize-at: {e}"))?));
            }
            "--log" => a.log = Some(val()?),
            "--trace" => a.trace = Some(val()?),
            "--present-wait" => a.present_wait = Some(val()?.parse().map_err(|e| format!("--present-wait: {e}"))?),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if let Some(d) = dressing {
        match &mut a.night {
            Some(n) => *n = d,
            None => return Err("--dressing needs --scene night".into()),
        }
    }
    Ok(a)
}

/// Free-fly camera state, voxel units, yaw/pitch in radians (yaw 0 looks along +x).
#[derive(Clone, Copy, Debug)]
struct Fly {
    eye: [f64; 3],
    yaw: f64,
    pitch: f64,
    fov_deg: f64,
    speed: f64,
}

impl Fly {
    fn forward(&self) -> [f64; 3] {
        [self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos()]
    }

    fn camera(&self, extent: vk::Extent2D) -> Camera {
        let f = self.forward();
        let target = [self.eye[0] + f[0], self.eye[1] + f[1], self.eye[2] + f[2]];
        Camera::look_at(self.eye, target, self.fov_deg, extent.width, extent.height, NEAR)
    }

    /// The scripted path for frame `i` of `n`: down the street at eye height, sweeping the view.
    fn scripted(i: u64, n: u64, fov_deg: f64) -> Fly {
        let t = i as f64 / n.max(1) as f64;
        let tau = std::f64::consts::TAU;
        Fly {
            eye: [8.0 + 360.0 * t, 27.0 + 6.0 * (tau * t).sin(), 64.0 + 24.0 * (2.0 * tau * t).sin()],
            yaw: 1.0 * (3.0 * tau * t).sin(),
            pitch: -0.15 + 0.1 * (tau * t).cos(),
            fov_deg,
            speed: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Walk,
    Fly,
}

/// Simulated seconds per frame in the scripted walk.
const SCRIPT_DT: f64 = 1.0 / 60.0;

/// Length of one scripted-walk loop in frames (21.8 s). Each loop starts again at the spawn point,
/// so the walk stays inside the street however long the run is.
const SCRIPT_LOOP_FRAMES: u64 = 1308;

/// The scripted walk's input at simulated time `t` into a loop: (intent, yaw). The loop is closed: it ends
/// about where it started, so the restart at the spawn point is a small jump.
fn walk_script(t: f64) -> (walk::Intent, f64) {
    use std::f64::consts::{FRAC_PI_2, PI};
    let go = |x: f64, z: f64, run: bool, jump: bool| walk::Intent { dir: [x, z], run, jump };
    match t {
        t if t < 0.5 => (go(0.0, 0.0, false, false), FRAC_PI_2),
        // Across the road, up the curb, to the shop-window sill.
        t if t < 7.0 => (go(0.0, 1.0, false, false), FRAC_PI_2),
        // Run along the shop fronts.
        t if t < 11.5 => (go(1.0, 0.0, true, false), 0.0),
        // Back toward the road, jumping about every 1.5 s, over and down the curb.
        t if t < 16.0 => (go(0.0, -1.0, false, (t % 1.5) < 0.05), -FRAC_PI_2),
        // Run back along the road, the way the shop-front run came.
        t if t < 20.5 => (go(-1.0, 0.0, true, false), PI),
        // Back to the start (the first leg stops at the shop-window sill, so this one is shorter).
        _ => (go(0.0, -1.0, false, false), -FRAC_PI_2),
    }
}

#[derive(Default)]
struct Input {
    keys: BTreeMap<KeyCode, bool>,
    looking: bool,
    /// Cursor position in window pixels.
    cursor: Option<(f64, f64)>,
}

/// An edit asked for this frame, at a pixel (x, y from the top left).
#[derive(Clone, Copy, Debug)]
enum EditRequest {
    Remove(f64, f64),
    Place(f64, f64),
    /// Remove an N³ box around the voxel under the pixel (`--edit-size`).
    RemoveBox(f64, f64, i32),
    /// Put back what the last `RemoveBox` removed.
    RestoreBox,
}

/// Host milliseconds of one edit's stages, and its edit-to-visible time.
#[derive(Clone, Copy, Debug, Default)]
struct EditTiming {
    voxels: usize,
    /// World commit, dirty marking, jobs (inline) and publication.
    commit_ms: f64,
    /// Changed regions' meshes built from the new snapshot.
    layout_ms: f64,
    upload_ms: f64,
    /// Acceleration update, including its wait for earlier work.
    accel_ms: f64,
    /// Waiting for the frames in flight, the region table, and the rebuilt descriptor sets.
    rebind_ms: f64,
    /// 4A: the emitter table's host build and upload inside the update (night scene; else 0).
    emitters_ms: f64,
    regions: usize,
    /// Edit start to the completion of the first frame showing it (polled once per frame).
    visible_ms: f64,
}

#[derive(Default)]
struct Edits {
    requests: Vec<EditRequest>,
    /// Published brick keys not yet on the device (a refused update keeps them for the retry).
    dirty: BTreeSet<BrickKey>,
    /// Edits applied to the world but not yet shown: (start, timing).
    unshown: Vec<(Instant, EditTiming)>,
    /// Shown by a submitted frame, waiting for it: (frame timeline value, start, timing).
    in_flight: Vec<(u64, Instant, EditTiming)>,
    done: Vec<EditTiming>,
    applied: u64,
    /// Nothing under the cursor, no face to place against, or the voxel would overlap the walker.
    rejected: u64,
    /// Updates refused by the budget (the previous snapshot stayed on screen).
    deferred: u64,
    /// 3E: voxels edited but not yet on the device, and those the next shaded frame relights.
    relight_pending: Vec<Relight>,
    relight: Vec<Relight>,
    /// The voxels the last `RemoveBox` changed, with their previous contents, and its box.
    removed_box: Vec<(VoxelCoord, Option<MaterialId>)>,
    removed_bounds: Option<([i32; 3], [i32; 3])>,
}

impl Input {
    fn held(&self, k: KeyCode) -> bool {
        self.keys.get(&k).copied().unwrap_or(false)
    }
}

/// Samples in milliseconds.
#[derive(Default)]
struct Series {
    frame: Vec<f64>,
    passes: Vec<Vec<f64>>,
    /// CPU time of the walker update per frame, walking only.
    walk: Vec<f64>,
}

/// CPU time inside one frame, in milliseconds, split where the frame can block.
#[derive(Clone, Copy, Debug, Default)]
struct Phases {
    /// `Frames::begin`: waiting for this slot's previous frame on the GPU.
    begin: f64,
    /// `Frames::acquire`: submitting the scene, then `vkAcquireNextImageKHR`.
    acquire: f64,
    /// `Frames::submit_present`: submitting the view pass, then `vkQueuePresentKHR`.
    present: f64,
    /// All of `frame()`, including the above.
    inside: f64,
    /// `--present-wait`: waiting for earlier presents to reach the display (before `begin`).
    pwait: f64,
    /// GPU pass times (G-buffer, view, ray trace, shade) of the frame that last used this slot, read in this frame.
    gpu: [f64; 4],
    /// Start of this frame, milliseconds after the run's first frame.
    at: f64,
}

/// Frame intervals over one 60 Hz frame (16.67 ms) are listed with the previous frame's phases,
/// so a slow interval shows whether the time went into the frame or outside it (the event loop).
const SLOW_MS: f64 = 1000.0 / 60.0;

/// Frame intervals kept for the title's recent fps lows (about 7 s at 144 fps).
const RECENT_FRAMES: usize = 1000;

/// Average fps and the 10% / 1% lows: the frame rate over the slowest 10% / 1% of the intervals (ms).
fn fps_lows(v: impl IntoIterator<Item = f64>) -> Option<(f64, f64, f64)> {
    let mut s: Vec<f64> = v.into_iter().collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(|a, b| b.total_cmp(a));
    let low = |f: f64| {
        let k = ((s.len() as f64 * f).ceil() as usize).max(1);
        1000.0 * k as f64 / s[..k].iter().sum::<f64>()
    };
    Some((1000.0 * s.len() as f64 / s.iter().sum::<f64>(), low(0.10), low(0.01)))
}

impl Series {
    fn summary(v: &[f64]) -> String {
        match (percentile(v, 50.0), percentile(v, 99.0), v.iter().copied().reduce(f64::max)) {
            (Some(a), Some(b), Some(c)) => format!("{{\"n\":{},\"mean\":{:.3},\"p50\":{a:.3},\"p99\":{b:.3},\"max\":{c:.3}}}", v.len(), v.iter().sum::<f64>() / v.len() as f64),
            _ => "null".into(),
        }
    }
}

struct State {
    gpu: Gpu,
    alloc: Allocator,
    timeline: Timeline,
    up: Uploader,
    raster: Raster,
    scene: GpuScene,
    bindings: gpu::raster::Bindings,
    /// Ray-query pass, its outputs (sized like the targets) and bindings (ray-tracing devices).
    ray: Option<RayPrimary>,
    ray_out: Option<RayTargets>,
    ray_bindings: Option<RayBindings>,
    tables: Tables,
    /// 3B real-time lighting (ray-tracing devices): the pass, its output and bindings, the albedos.
    shade: Option<Shade>,
    shade_out: Option<ShadeTargets>,
    shade_bindings: Option<ShadeBindings>,
    albedo: Option<RefMaterials>,
    /// 3C: the sky tables and the sky-view pass (ray-tracing devices).
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
    /// 3D: the temporal pass, its history (sized like the targets) and bindings.
    temporal: Option<Temporal>,
    history: Option<History>,
    temporal_bindings: Option<TemporalBindings>,
    /// 3E: the filter, its buffers (sized like the targets) and bindings.
    denoise: Option<Denoise>,
    denoise_targets: Option<DenoiseTargets>,
    denoise_bindings: Option<DenoiseBindings>,
    /// 4B: the lit shade set (the scene's emitter table), and emission and the meter after
    /// reconstruction (scenes with a table only).
    shade_lit_bindings: Option<ShadeBindings>,
    compose: Option<Compose>,
    compose_out: Option<ComposeTargets>,
    compose_bindings: Option<ComposeBindings>,
    swapchain: Option<Swapchain>,
    targets: Option<Targets>,
    debug: DebugView,
    frames: Frames,
    timer: GpuTimer,
    readers: FrameReaders,
    pipeline: Pipeline,
    // Last: the window must outlive the surface.
    window: Window,
}

impl State {
    /// 4B: destroys the lit shade set and the compose sets (the GPU no longer uses them).
    fn unbind_lights(&mut self) {
        if let Some(b) = self.shade_lit_bindings.take() {
            b.destroy(&self.gpu);
        }
        if let Some(b) = self.compose_bindings.take() {
            b.destroy(&self.gpu);
        }
    }

    /// 4B: binds the lit shade set and the compose sets to the scene's current emitter table, after a
    /// start, a resize or an update (the old sets must no longer be in use). A table the scene refuses
    /// as stale is not bound: the lights are then not rendered (`stale_table_frames`).
    fn bind_lights(&mut self) -> Result<(), String> {
        let e = |x: gpu::GpuError| x.to_string();
        self.unbind_lights();
        let (Some(sh), Some(o), Some(a), Some(al), Some(sky), Some(t), Some(c), Some(co)) = (&self.shade, &self.shade_out, &self.scene.accel, &self.albedo, &self.sky, &self.targets, &self.compose, &self.compose_out) else {
            return Ok(());
        };
        match self.scene.emitters() {
            Ok(Some(em)) => {
                self.shade_lit_bindings = Some(sh.bind_lit(&self.gpu, t, a, al, o, &sky.view, &em.device).map_err(e)?);
                self.compose_bindings = Some(c.bind(&self.gpu, t, o, &em.device, co).map_err(e)?);
            }
            Ok(None) => {}
            Err(x) => eprintln!("emitter table not bound: {x:?}"),
        }
        Ok(())
    }
}

/// 4B: the light view's exposure before the - and = offset: M3's (sun and sky) while the lights are
/// off; automatic while they are on, starting from the exposure shown last (at start-up, from the
/// first reading; until then M3's, bounded) and adapting to the meter's target
/// (`light::exposure::adapt`).
#[derive(Default)]
struct Exposure {
    auto: Option<f64>,
    seed: Option<f64>,
    /// The meter's last target and the floor for the next reading.
    target: Option<f64>,
    floor: f64,
    /// The exposure shown last frame.
    last: Option<f64>,
}

impl Exposure {
    /// The lights switched: the automatic exposure starts again (from the last shown one, unless this
    /// is the first frame) and waits for a new reading.
    fn lights_switched(&mut self, first_frame: bool) {
        self.auto = None;
        self.seed = if first_frame { None } else { self.last };
        self.target = None;
        self.floor = 0.0;
    }

    fn shown(&mut self, lights: bool, m3: f64, dt: f64) -> f64 {
        use light::exposure::{adapt, MAX_EXPOSURE, MIN_EXPOSURE};
        let e = if !lights {
            m3
        } else {
            let a = match (self.auto, self.seed, self.target) {
                (Some(a), _, t) => Some(adapt(a, t, dt)),
                (None, Some(seed), _) => Some(adapt(seed, None, 0.0)),
                (None, None, Some(t)) => Some(adapt(t, None, 0.0)),
                (None, None, None) => None,
            };
            self.auto = a;
            a.unwrap_or_else(|| m3.clamp(MIN_EXPOSURE, MAX_EXPOSURE))
        };
        self.last = Some(e);
        e
    }
}

struct App {
    args: Args,
    world: World,
    fly: Fly,
    mode: Mode,
    lighting: Lighting,
    /// 3B: time of day, the sun it gives (ADR-0005 units), and whether the day runs.
    hour: f64,
    sun: light::reference::Lighting,
    run_day: bool,
    /// Exposure compensation of the light view, in stops (- and =).
    light_stops: f32,
    /// 3C: the host sky tables (for the sky-view table's oracle and the display exposure), the hour
    /// the device's sky-view table was built for, and the sky's zenith luminance at this hour.
    luts: light::sky::SkyLuts,
    sky_hour: Option<f64>,
    sky_zenith: f64,
    /// 3D: accumulate the lighting over frames (H), and the last frame the shade pass ran (a gap in
    /// shading resets the history).
    accumulate: bool,
    last_shaded: Option<u64>,
    /// 3E: filter the accumulated lighting (N).
    denoise: bool,
    /// 3G: the filter's `prefilter_age` (P switches between `u32::MAX`, the 3E filter, and
    /// `--prefilter-age`).
    prefilter_age: u32,
    /// 3F: one bounce of sun and sky light (B).
    bounce: bool,
    /// 4B: the lights (L), whether they were rendered last frame, and how often they switched.
    lights: Lights,
    lights_shown: bool,
    lights_switches: u64,
    /// 4B: the light view's exposure, and which frame slots metered.
    exposure: Exposure,
    metered: [bool; FRAMES_IN_FLIGHT],
    /// 4B: frames that wanted the lights but had no current emitter table (must stay 0).
    stale_table_frames: u64,
    walker: walk::Walker,
    spawn: [f64; 3],
    kill_y: f64,
    respawns: u64,
    /// Scripted walk: restarts at the spawn point at the start of each loop.
    script_loops: u64,
    /// Frames on which the walker's box overlapped a solid voxel (must stay 0).
    overlaps: u64,
    input: Input,
    view: View,
    source: Source,
    edits: Edits,
    state: Option<State>,
    resize: bool,
    recreates: u64,
    frame_index: u64,
    last: Option<Instant>,
    all: Series,
    phases: Vec<Phases>,
    /// (frame index, interval ms, the phases of the frame the interval covers).
    slow: Vec<(u64, f64, Phases)>,
    /// Per frame: the interval ending at its start (0 for the first) and its phases.
    trace: Vec<(f64, Phases)>,
    /// Window events that change presentation: focus lost, occluded, and the refresh rate seen.
    focus_lost: u64,
    /// Key presses, mouse-button presses and wheel events received (3G: a scripted run with any is
    /// disturbed; keys change views and settings without a focus change).
    input_events: u64,
    occluded: u64,
    refresh_mhz: Option<u32>,
    /// The first frame's start: a monotonic instant and Unix milliseconds, to line traces up with
    /// external logs (nvidia-smi).
    run_start: Option<(Instant, u128)>,
    present_wait_timeouts: u64,
    window_series: Series,
    /// The last `RECENT_FRAMES` frame intervals, for the title's fps lows.
    recent: VecDeque<f64>,
    window_start: Instant,
    exit_code: i32,
    error: Option<String>,
}

fn prepare_meshes(world: &World, merge: Merge) -> (Pipeline, Vec<(world::BrickKey, derived::BrickMesh)>) {
    let mut p = Pipeline::new(Config { merge, ..Config::default() });
    p.mark_all(world);
    loop {
        let jobs = p.dispatch(world);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            p.complete(world, j.run());
        }
    }
    p.try_publish(world);
    assert!(p.is_idle(), "pipeline did not drain");
    let token = p.acquire();
    let keys: Vec<_> = p.current().keys().collect();
    let meshes = keys.into_iter().filter_map(|k| p.read(&token, k).expect("token is live").map(|m| (k, m.clone()))).collect();
    p.release(token).expect("issued here");
    (p, meshes)
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<State, String> {
        let e = |x: gpu::GpuError| x.to_string();
        let attrs = Window::default_attributes().with_title("new-engine viewer").with_inner_size(PhysicalSize::new(self.args.size.0, self.args.size.1));
        let window = el.create_window(attrs).map_err(|x| x.to_string())?;
        let display = window.display_handle().map_err(|x| x.to_string())?.as_raw();
        let handle = window.window_handle().map_err(|x| x.to_string())?.as_raw();
        let exts = ash_window::enumerate_required_extensions(display).map_err(|x| format!("surface extensions: {x:?}"))?;
        let gpu = Gpu::with_surface(exts, |entry, instance| unsafe { ash_window::create_surface(entry, instance, display, handle, None) }).map_err(e)?;
        eprintln!("device {:?}, validation {}", gpu.info.name, gpu.validation_enabled());
        let mut alloc = Allocator::new(gpu.device_budget());
        let mut timeline = Timeline::new(&gpu).map_err(e)?;
        let mut up = Uploader::new(&gpu, &mut alloc, DEFAULT_RING_BYTES).map_err(e)?;

        let t = Instant::now();
        let (pipeline, brick_meshes) = prepare_meshes(&self.world, self.args.merge);
        let snapshot = pipeline.current().id;
        let regions = build_regions(brick_meshes.iter().map(|(k, m)| (*k, m)), self.args.region);
        let snap_n = snapshot.raw();
        let rt = gpu.ray_tracing();
        let scene = match self.args.night {
            Some(_) => GpuScene::build_lit(&gpu, &mut alloc, &mut up, &mut timeline, self.args.region, &regions, snap_n, rt, self.world.materials()).map_err(e)?,
            None => GpuScene::build(&gpu, &mut alloc, &mut up, &mut timeline, self.args.region, &regions, snap_n, rt).map_err(e)?,
        };
        if let Ok(Some(em)) = scene.emitters() {
            eprintln!("emitters: {} ({} B device, GpuMaterial)", em.table.len(), em.device.device_bytes());
        }
        let meshes = &scene.meshes;
        let mats: Vec<world::MaterialParams> = self.world.materials().iter().map(|(_, d)| d.params).collect();
        let (tables, _) = Tables::upload(&gpu, &mut alloc, &mut up, &mut timeline, &mats, meshes, snap_n).map_err(e)?;
        eprintln!(
            "meshes: merge {} region {}: {} regions, {} quads, {} triangles, {} B device, snapshot {snap_n}, built in {:.2} s",
            self.args.merge.name(),
            self.args.region.name(),
            meshes.regions.len(),
            meshes.quad_count(),
            meshes.triangle_count(),
            meshes.device_bytes(),
            t.elapsed().as_secs_f64()
        );
        let raster = Raster::new(&gpu).map_err(e)?;
        let bindings = raster.bind(&gpu, &scene.meshes).map_err(e)?;
        let size = window.inner_size();
        let swapchain = Swapchain::new(&gpu, vk::Extent2D { width: size.width, height: size.height }, self.args.present, None).map_err(e)?.ok_or("window has zero size at start")?;
        eprintln!("swapchain {:?} {:?}, {} images {}x{}, present {:?} (asked {:?}); ~{} B driver-owned, not in the ledger", swapchain.format.format, swapchain.format.color_space, swapchain.images.len(), swapchain.extent.width, swapchain.extent.height, swapchain.present_mode, self.args.present, swapchain.estimated_bytes());
        let targets = Targets::new(&gpu, &mut alloc, swapchain.extent).map_err(e)?;
        let ray = if rt { Some(RayPrimary::new(&gpu).map_err(e)?) } else { None };
        let ray_out = if rt { Some(RayTargets::new(&gpu, &mut alloc, swapchain.extent.width, swapchain.extent.height).map_err(e)?) } else { None };
        let ray_bindings = match (&ray, &ray_out, &scene.accel) {
            (Some(r), Some(o), Some(a)) => Some(r.bind(&gpu, a, o).map_err(e)?),
            _ => None,
        };
        let debug = DebugView::new(&gpu, swapchain.format.format).map_err(e)?;
        debug.bind(&gpu, &targets, &tables, ray_out.as_ref());
        // 3B: real-time lighting on ray-tracing devices.
        let (shade, shade_out, shade_bindings, albedo, sky, sky_pass) = if let Some(a) = &scene.accel {
            let al = light::reference::albedos(self.world.materials())?;
            let (albedo, v) = RefMaterials::upload(&gpu, &mut alloc, &mut up, &mut timeline, &al).map_err(e)?;
            timeline.wait(&gpu, v, u64::MAX).map_err(e)?;
            let (sky, v) = SkyTables::upload(&gpu, &mut alloc, &mut up, &mut timeline, &self.luts).map_err(e)?;
            timeline.wait(&gpu, v, u64::MAX).map_err(e)?;
            let sky_pass = SkyView::new(&gpu, &sky).map_err(e)?;
            let sh = Shade::new(&gpu).map_err(e)?;
            let out = ShadeTargets::new(&gpu, &mut alloc, swapchain.extent.width, swapchain.extent.height).map_err(e)?;
            let b = sh.bind(&gpu, &targets, a, &albedo, &out, &sky.view).map_err(e)?;
            debug.bind_radiance(&gpu, &out.radiance);
            (Some(sh), Some(out), Some(b), Some(albedo), Some(sky), Some(sky_pass))
        } else {
            (None, None, None, None, None, None)
        };
        // 3D: the temporal pass and history, when the lighting exists.
        let (temporal, history, temporal_bindings) = if let Some(out) = &shade_out {
            let t = Temporal::new(&gpu).map_err(e)?;
            let h = History::new(&gpu, &mut alloc, swapchain.extent.width, swapchain.extent.height).map_err(e)?;
            let b = t.bind(&gpu, &targets, &tables.regions, &out.radiance, &h).map_err(e)?;
            debug.bind_motion(&gpu, &h.motion);
            (Some(t), Some(h), Some(b))
        } else {
            (None, None, None)
        };
        // 3E: the filter reads the history's guides and moments.
        let (denoise, denoise_targets, denoise_bindings) = if let (Some(out), Some(h), Some(al)) = (&shade_out, &history, &albedo) {
            let d = Denoise::new(&gpu).map_err(e)?;
            let t = DenoiseTargets::new(&gpu, &mut alloc, swapchain.extent.width, swapchain.extent.height).map_err(e)?;
            let b = d.bind(&gpu, &out.radiance, h, al, &t).map_err(e)?;
            (Some(d), Some(t), Some(b))
        } else {
            (None, None, None)
        };
        // 4B: emission and the meter, for a scene with an emitter table.
        let (compose, compose_out) = if let (Some(_), Ok(Some(_))) = (&shade_out, scene.emitters()) {
            (Some(Compose::new(&gpu).map_err(e)?), Some(ComposeTargets::new(&gpu, &mut alloc, swapchain.extent.width, swapchain.extent.height, FRAMES_IN_FLIGHT).map_err(e)?))
        } else {
            (None, None)
        };
        let frames = Frames::new(&gpu, FRAMES_IN_FLIGHT).map_err(e)?;
        let timer = GpuTimer::new(&gpu, FRAMES_IN_FLIGHT as u32, PASSES.len() as u32).map_err(e)?;
        let l = alloc.ledger();
        eprintln!(
            "ledger live: GpuMesh {} B, GpuAccel {} B, GpuMaterial {} B, GpuTemporal {} B (targets {} B), Staging {} B",
            l.account(Category::GpuMesh).usage.live,
            l.account(Category::GpuAccel).usage.live,
            l.account(Category::GpuMaterial).usage.live,
            l.account(Category::GpuTemporal).usage.live,
            targets.device_bytes(),
            l.account(Category::Staging).usage.live
        );
        let mut state = State { gpu, alloc, timeline, up, raster, scene, bindings, ray, ray_out, ray_bindings, tables, shade, shade_out, shade_bindings, albedo, sky, sky_pass, temporal, history, temporal_bindings, denoise, denoise_targets, denoise_bindings, shade_lit_bindings: None, compose, compose_out, compose_bindings: None, swapchain: Some(swapchain), targets: Some(targets), debug, frames, timer, readers: FrameReaders::default(), pipeline, window };
        state.bind_lights()?;
        Ok(state)
    }

    fn recreate(&mut self) -> Result<(), String> {
        let e = |x: gpu::GpuError| x.to_string();
        let s = self.state.as_mut().expect("state");
        s.frames.wait_all(&s.gpu, &s.timeline).map_err(e)?;
        s.gpu.wait_idle().map_err(e)?;
        let size = s.window.inner_size();
        let old = s.swapchain.take();
        s.swapchain = Swapchain::new(&s.gpu, vk::Extent2D { width: size.width, height: size.height }, self.args.present, old).map_err(e)?;
        if let Some(t) = s.targets.take() {
            t.free(&s.gpu, &mut s.alloc);
        }
        if let Some(b) = s.ray_bindings.take() {
            b.destroy(&s.gpu);
        }
        if let Some(o) = s.ray_out.take() {
            o.free(&s.gpu, &mut s.alloc);
        }
        if let Some(b) = s.shade_bindings.take() {
            b.destroy(&s.gpu);
        }
        if let Some(o) = s.shade_out.take() {
            o.free(&s.gpu, &mut s.alloc);
        }
        if let Some(b) = s.temporal_bindings.take() {
            b.destroy(&s.gpu);
        }
        if let Some(h) = s.history.take() {
            h.free(&s.gpu, &mut s.alloc);
        }
        if let Some(b) = s.denoise_bindings.take() {
            b.destroy(&s.gpu);
        }
        if let Some(t) = s.denoise_targets.take() {
            t.free(&s.gpu, &mut s.alloc);
        }
        s.unbind_lights();
        let had_compose = s.compose_out.is_some();
        if let Some(c) = s.compose_out.take() {
            c.free(&s.gpu, &mut s.alloc);
        }
        if let Some(sw) = &s.swapchain {
            assert_eq!(sw.format.format, s.debug.format, "surface format changed");
            let t = Targets::new(&s.gpu, &mut s.alloc, sw.extent).map_err(e)?;
            if let (Some(r), Some(a)) = (&s.ray, &s.scene.accel) {
                let o = RayTargets::new(&s.gpu, &mut s.alloc, sw.extent.width, sw.extent.height).map_err(e)?;
                s.ray_bindings = Some(r.bind(&s.gpu, a, &o).map_err(e)?);
                s.ray_out = Some(o);
            }
            s.debug.bind(&s.gpu, &t, &s.tables, s.ray_out.as_ref());
            if let (Some(sh), Some(a), Some(al), Some(sky)) = (&s.shade, &s.scene.accel, &s.albedo, &s.sky) {
                let o = ShadeTargets::new(&s.gpu, &mut s.alloc, sw.extent.width, sw.extent.height).map_err(e)?;
                s.shade_bindings = Some(sh.bind(&s.gpu, &t, a, al, &o, &sky.view).map_err(e)?);
                s.debug.bind_radiance(&s.gpu, &o.radiance);
                if let Some(tp) = &s.temporal {
                    let h = History::new(&s.gpu, &mut s.alloc, sw.extent.width, sw.extent.height).map_err(e)?;
                    s.temporal_bindings = Some(tp.bind(&s.gpu, &t, &s.tables.regions, &o.radiance, &h).map_err(e)?);
                    s.debug.bind_motion(&s.gpu, &h.motion);
                    if let Some(d) = &s.denoise {
                        let dt = DenoiseTargets::new(&s.gpu, &mut s.alloc, sw.extent.width, sw.extent.height).map_err(e)?;
                        s.denoise_bindings = Some(d.bind(&s.gpu, &o.radiance, &h, al, &dt).map_err(e)?);
                        s.denoise_targets = Some(dt);
                    }
                    s.history = Some(h);
                }
                s.shade_out = Some(o);
            }
            if had_compose {
                s.compose_out = Some(ComposeTargets::new(&s.gpu, &mut s.alloc, sw.extent.width, sw.extent.height, FRAMES_IN_FLIGHT).map_err(e)?);
            }
            s.targets = Some(t);
            s.bind_lights()?;
        }
        // The meter's slots were recreated: no reading is pending.
        self.metered = [false; FRAMES_IN_FLIGHT];
        self.resize = false;
        self.recreates += 1;
        Ok(())
    }

    /// Moves the sun to `hour` (wrapped to a day); the M1 lit view follows the same direction.
    fn set_hour(&mut self, hour: f64) {
        self.hour = hour.rem_euclid(24.0);
        self.sun = sun_at(self.hour);
        self.lighting.sun_dir = self.sun.sun_dir;
        self.sky_zenith = sky_zenith(&self.luts, &self.sun);
    }

    /// Starts walking with the feet at `feet`, lifted out of solid voxels. False if no free space.
    fn start_walking(&mut self, feet: [f64; 3]) -> bool {
        let mut w = walk::Walker::new(self.walker.params, feet);
        w.stats = self.walker.stats; // counters span respawns and mode switches
        if !w.resolve_start(&self.world, 1024) {
            return false;
        }
        self.walker = w;
        self.mode = Mode::Walk;
        true
    }

    fn step_camera(&mut self, dt: f64) {
        if let Some(n) = self.args.frames {
            if self.args.cycle_views {
                self.view = View::ALL[((self.frame_index * View::ALL.len() as u64) / n.max(1)) as usize % View::ALL.len()];
            }
            if !self.args.walk_script {
                self.fly = Fly::scripted(self.frame_index, n, self.fly.fov_deg);
                return;
            }
            let in_loop = self.frame_index % SCRIPT_LOOP_FRAMES;
            if self.frame_index > 0 && in_loop == 0 {
                self.script_loops += 1;
                let spawn = self.spawn;
                assert!(self.start_walking(spawn), "the spawn point is free");
            }
            let (intent, yaw) = walk_script(in_loop as f64 * SCRIPT_DT);
            self.fly.yaw = yaw;
            self.fly.pitch = -0.1;
            self.step_walker(intent, SCRIPT_DT);
            return;
        }
        let i = &self.input;
        let flat = [self.fly.yaw.cos(), 0.0, self.fly.yaw.sin()];
        let right = [-flat[2], 0.0, flat[0]];
        if self.mode == Mode::Walk {
            let mut d = [0.0f64; 2];
            for (k, v, s) in [(KeyCode::KeyW, flat, 1.0), (KeyCode::KeyS, flat, -1.0), (KeyCode::KeyD, right, 1.0), (KeyCode::KeyA, right, -1.0)] {
                if i.held(k) {
                    d[0] += v[0] * s;
                    d[1] += v[2] * s;
                }
            }
            let intent = walk::Intent { dir: d, run: i.held(KeyCode::ShiftLeft), jump: i.held(KeyCode::Space) };
            self.step_walker(intent, dt);
            return;
        }
        let f = self.fly.forward();
        let mut m = [0.0f64; 3];
        let mut add = |v: [f64; 3], s: f64| (0..3).for_each(|a| m[a] += v[a] * s);
        if i.held(KeyCode::KeyW) {
            add(f, 1.0);
        }
        if i.held(KeyCode::KeyS) {
            add(f, -1.0);
        }
        if i.held(KeyCode::KeyD) {
            add(right, 1.0);
        }
        if i.held(KeyCode::KeyA) {
            add(right, -1.0);
        }
        if i.held(KeyCode::Space) {
            add([0.0, 1.0, 0.0], 1.0);
        }
        if i.held(KeyCode::ControlLeft) {
            add([0.0, 1.0, 0.0], -1.0);
        }
        let boost = if i.held(KeyCode::ShiftLeft) { 4.0 } else { 1.0 };
        for (e, d) in self.fly.eye.iter_mut().zip(m) {
            *e += d * self.fly.speed * boost * dt;
        }
    }

    /// Advances the walker and puts the camera at its eye.
    fn step_walker(&mut self, intent: walk::Intent, dt: f64) {
        let t = Instant::now();
        self.walker.update(&self.world, intent, dt);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.all.walk.push(ms);
        if self.walker.overlaps(&self.world) {
            self.overlaps += 1;
        }
        if self.walker.feet[1] < self.kill_y {
            self.respawns += 1;
            let spawn = self.spawn;
            assert!(self.start_walking(spawn), "the spawn point is free");
        }
        self.fly.eye = self.walker.render_eye();
    }

    fn frame(&mut self, el: &ActiveEventLoop) -> Result<(), String> {
        let e = |x: gpu::GpuError| x.to_string();
        let now = Instant::now();
        let dt = self.last.map_or(0.0, |l| (now - l).as_secs_f64());
        if self.resize {
            self.recreate()?;
        }
        self.step_camera(dt);
        if self.run_day {
            self.set_hour(self.hour + dt / 4.0);
        }
        self.script_edits();
        self.apply_edits()?;
        let s = self.state.as_mut().expect("state");
        let (Some(sw), Some(targets)) = (&s.swapchain, &s.targets) else {
            return Ok(()); // minimized
        };
        let mut ph = Phases::default();
        let (t0, _) = *self.run_start.get_or_insert_with(|| (now, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())));
        ph.at = (now - t0).as_secs_f64() * 1e3;
        if let Some(n) = self.args.present_wait {
            let t = Instant::now();
            if sw.wait_presented(n, 100_000_000) == Some(false) {
                self.present_wait_timeouts += 1;
            }
            ph.pwait = t.elapsed().as_secs_f64() * 1e3;
        }
        let t = Instant::now();
        let scene = s.frames.begin(&s.gpu, &s.timeline).map_err(e)?;
        ph.begin = t.elapsed().as_secs_f64() * 1e3;
        let slot = scene.slot;
        // The slot's previous frame has completed: its timestamps are ready and its token can go.
        if let Some(ms) = s.timer.read(&s.gpu, slot).map_err(e)? {
            ph.gpu = [ms[0], ms[1], ms[2], ms[3]];
            for (i, v) in ms.iter().enumerate() {
                for series in [&mut self.all, &mut self.window_series] {
                    series.passes.resize(PASSES.len(), Vec::new());
                    series.passes[i].push(*v);
                }
            }
        }
        // 4B: the meter's reading of this slot's previous frame (it has completed).
        if std::mem::take(&mut self.metered[slot as usize]) {
            if let Some(co) = &s.compose_out {
                let r = co.reading(slot as usize);
                self.exposure.floor = r.next_floor();
                self.exposure.target = r.target();
            }
        }
        let completed = s.timeline.completed(&s.gpu).map_err(e)?;
        s.readers.release_completed(&mut s.pipeline, completed);
        s.scene.collect(&s.gpu, &mut s.alloc, completed);
        let edits = &mut self.edits;
        edits.in_flight.retain(|&(v, start, mut t)| {
            if v <= completed {
                t.visible_ms = start.elapsed().as_secs_f64() * 1e3;
                edits.done.push(t);
                false
            } else {
                true
            }
        });

        // Scene: the G-buffer, submitted before the swapchain image is acquired.
        let camera = self.fly.camera(targets.extent);
        let token = s.pipeline.acquire();
        s.timer.reset(&s.gpu, scene.cmd, slot);
        s.timer.begin_pass(&s.gpu, scene.cmd, slot, 0);
        s.raster.record(&s.gpu, scene.cmd, targets, &s.scene.meshes, &s.bindings, &camera, Faults::default());
        s.timer.end_pass(&s.gpu, scene.cmd, slot, 0);
        // The ray pass runs only when shown; its timestamps are always written (0 ms when off).
        let source = if s.ray_bindings.is_some() { self.source } else { Source::Raster };
        s.timer.begin_pass(&s.gpu, scene.cmd, slot, 2);
        if let (Source::Ray, Some(r), Some(b), Some(o)) = (source, &s.ray, &s.ray_bindings, &s.ray_out) {
            r.record(&s.gpu, scene.cmd, b, o, &camera);
            let to_view = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
            unsafe { s.gpu.device.cmd_pipeline_barrier2(scene.cmd, &vk::DependencyInfo::default().memory_barriers(&to_view)) };
        }
        s.timer.end_pass(&s.gpu, scene.cmd, slot, 2);
        // 3B: the real-time lighting runs only when shown; its timestamps are always written.
        let shaded = matches!(self.view, View::Light | View::HistoryAge | View::HistoryReason | View::Motion) && s.shade_bindings.is_some();
        // 4B: the lights, on a scene with an emitter table; rendered only with a current table.
        let lights_wanted = self.lights.update(self.sun.sun_dir) && s.compose.is_some();
        if lights_wanted && s.shade_lit_bindings.is_none() {
            self.stale_table_frames += 1;
        }
        let lit = lights_wanted && s.shade_lit_bindings.is_some() && s.compose_bindings.is_some();
        if lit != self.lights_shown {
            // A switch after the first frame (the start state is not one).
            self.lights_switches += u64::from(self.frame_index > 0);
            self.lights_shown = lit;
            self.exposure.lights_switched(self.frame_index == 0);
        }
        s.timer.begin_pass(&s.gpu, scene.cmd, slot, 3);
        if let (true, Some(sh), Some(b), Some(o)) = (shaded, &s.shade, &s.shade_bindings, &s.shade_out) {
            // 3C: the sky-view table follows the sun (rebuilt only when the hour changed).
            if self.sky_hour != Some(self.hour) {
                if let Some(sp) = &s.sky_pass {
                    sp.record(&s.gpu, scene.cmd, self.sun.sun_dir);
                    self.sky_hour = Some(self.hour);
                }
            }
            targets_to_read(&s.gpu, scene.cmd, targets);
            let set = ShadeSettings { bounce: self.bounce, emitters: lit, emitter_spp: self.args.emitter_spp, ..ShadeSettings::default() };
            let p = shade::Params::new(&camera, &self.sun, set, ShadeFaults::default(), self.frame_index as u32, SHADE_SEED);
            let b = if lit { s.shade_lit_bindings.as_ref().unwrap_or(b) } else { b };
            sh.record(&s.gpu, scene.cmd, b, o, &p);
            // 3D: accumulate. A gap in shading (another view was shown) or turning it back on resets.
            if let (true, Some(tp), Some(tb), Some(h)) = (self.accumulate, &s.temporal, &s.temporal_bindings, s.history.as_mut()) {
                let gap = self.last_shaded.is_none_or(|f| f + 1 != self.frame_index);
                let ts = TemporalSettings { max_age: self.args.max_age, ..TemporalSettings::default() };
                h.set_lights(lit);
                tp.record(&s.gpu, scene.cmd, tb, h, &camera, self.sun.sun_dir, ts, TemporalFaults::default(), gap, &self.edits.relight);
                // 3E: the filter shows the accumulated lighting reconstructed; the history stays raw.
                if let (true, Some(d), Some(db), Some(dt)) = (self.denoise, &s.denoise, &s.denoise_bindings, &s.denoise_targets) {
                    let ds = DenoiseSettings { prefilter_age: self.prefilter_age, ..DenoiseSettings::default() };
                    d.record(&s.gpu, scene.cmd, db, h, dt, ds, DenoiseFaults::default());
                }
            }
            // 4B: emission after reconstruction, and the meter.
            if let (true, Some(c), Some(cb), Some(co)) = (lit, &s.compose, &s.compose_bindings, &s.compose_out) {
                c.record(&s.gpu, scene.cmd, cb, co, slot as usize, true, self.exposure.floor, ComposeFaults::default());
                self.metered[slot as usize] = true;
            }
            self.edits.relight.clear();
            self.last_shaded = Some(self.frame_index);
            radiance_to_fragment(&s.gpu, scene.cmd);
        }
        s.timer.end_pass(&s.gpu, scene.cmd, slot, 3);
        let t = Instant::now();
        let acquired = s.frames.acquire(&s.gpu, &mut s.timeline, sw, scene).map_err(e)?;
        ph.acquire = t.elapsed().as_secs_f64() * 1e3;
        let frame = match acquired {
            Acquired::Frame { frame, .. } => frame,
            Acquired::OutOfDate { scene_value } => {
                s.readers.hold(scene_value, token);
                self.resize = true;
                return Ok(());
            }
        };
        // Present: the debug view into the acquired image.
        let cmd = frame.cmd;
        let image = sw.images[frame.image as usize];
        s.timer.begin_pass(&s.gpu, cmd, slot, 1);
        image_to_attachment(&s.gpu, cmd, image);
        let mut lighting = self.lighting;
        let m3 = light_exposure(&self.sun, self.sky_zenith, 0.0) as f64;
        lighting.light_exposure = (self.exposure.shown(self.lights_shown, m3, dt) * 2f64.powf(self.light_stops as f64)) as f32;
        if shaded {
            // The targets were made readable before the shade pass.
            s.debug.draw(&s.gpu, cmd, targets, &s.tables, sw.views[frame.image as usize], &camera, self.view, &lighting, source);
        } else {
            s.debug.record(&s.gpu, cmd, targets, &s.tables, sw.views[frame.image as usize], &camera, self.view, &lighting, source);
        }
        image_to_present(&s.gpu, cmd, image);
        s.timer.end_pass(&s.gpu, cmd, slot, 1);
        let t = Instant::now();
        let (value, recreate) = s.frames.submit_present(&s.gpu, &mut s.timeline, sw, frame).map_err(e)?;
        ph.present = t.elapsed().as_secs_f64() * 1e3;
        s.readers.hold(value, token);
        // Edits count as shown only once the device holds everything published (a refused update
        // keeps them unshown until its retry succeeds).
        if self.edits.dirty.is_empty() {
            for (start, t) in self.edits.unshown.drain(..) {
                self.edits.in_flight.push((value, start, t));
            }
        }
        if recreate {
            self.resize = true;
        }
        // Begin waited for this slot's previous frame, so at most one token per frame in flight.
        assert!(s.readers.len() <= FRAMES_IN_FLIGHT, "reader tokens outlive their frames: {}", s.readers.len());

        let interval = self.last.map_or(0.0, |l| (now - l).as_secs_f64() * 1e3);
        if let Some(l) = self.last {
            let ms = (now - l).as_secs_f64() * 1e3;
            self.all.frame.push(ms);
            self.window_series.frame.push(ms);
            if self.recent.len() == RECENT_FRAMES {
                self.recent.pop_front();
            }
            self.recent.push_back(ms);
            if ms > SLOW_MS {
                self.slow.push((self.frame_index, ms, self.phases.last().copied().unwrap_or_default()));
            }
        }
        self.last = Some(now);
        ph.inside = now.elapsed().as_secs_f64() * 1e3;
        self.phases.push(ph);
        if self.args.trace.is_some() {
            self.trace.push((interval, ph));
        }
        self.frame_index += 1;
        if let Some((f, w, h)) = self.args.resize_at {
            if self.frame_index == f {
                let _ = s.window.request_inner_size(PhysicalSize::new(w, h));
            }
        }

        if self.window_start.elapsed().as_secs_f64() >= 1.0 {
            let w = &self.window_series;
            let p = |v: &[f64], q| percentile(v, q).unwrap_or(f64::NAN);
            let gpu_ms: Vec<String> = PASSES.iter().zip(&w.passes).map(|(n, v)| format!("{n} {:.2}", p(v, 50.0))).collect();
            let (avg, low10, low1) = fps_lows(self.recent.iter().copied()).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
            s.window.set_title(&format!(
                "new-engine viewer | {} | {} ({}, R) | filter {} (P) | sun {:.2} h, {:.1}° ([ ] T) | {:.0} fps (last {} frames: avg {avg:.0}, 10% low {low10:.0}, 1% low {low1:.0}), frame p50 {:.2} p99 {:.2} ms | GPU p50 ms: {} | {:?} {}x{} | snapshot {}, edits {} (last visible in {:.1} ms){}",
                match self.mode {
                    Mode::Walk => "walk (F: fly)",
                    Mode::Fly => "fly (F: walk)",
                },
                self.view.name(),
                if source == Source::Ray { "ray query" } else { "raster" },
                if !self.denoise { "off".to_string() } else if self.prefilter_age == u32::MAX { "3E".to_string() } else { format!("prefilter_age {}", self.prefilter_age) },
                self.hour,
                light::sun::elevation_deg(self.sun.sun_dir),
                w.frame.len() as f64 / self.window_start.elapsed().as_secs_f64(),
                self.recent.len(),
                p(&w.frame, 50.0),
                p(&w.frame, 99.0),
                gpu_ms.join(", "),
                sw.present_mode,
                sw.extent.width,
                sw.extent.height,
                s.scene.snapshot,
                self.edits.applied,
                self.edits.done.last().map_or(f64::NAN, |t| t.visible_ms),
                match (self.args.night, s.scene.emitters()) {
                    (Some(d), Ok(Some(em))) => format!(
                        " | night ({}), {} emitters | lights {} ({}, L) | exposure {} {:.3e}",
                        d.name(),
                        em.table.len(),
                        if self.lights_shown { "on" } else { "off" },
                        if self.lights.manual() { "by hand" } else { "auto" },
                        if self.lights_shown { "auto" } else { "M3" },
                        self.exposure.last.unwrap_or(f64::NAN)
                    ),
                    (Some(d), _) => format!(" | night ({}), emitter table not current", d.name()),
                    _ => String::new(),
                }
            ));
            self.window_series = Series::default();
            self.window_start = Instant::now();
        }
        if self.args.frames.is_some_and(|n| self.frame_index >= n) {
            el.exit();
        }
        Ok(())
    }

    /// The pixel an interactive edit aims at: the cursor, or the centre while looking or without one.
    fn edit_pixel(&self) -> (f64, f64) {
        let extent = self.state.as_ref().and_then(|s| s.targets.as_ref()).map_or((1.0, 1.0), |t| (t.extent.width as f64, t.extent.height as f64));
        match self.input.cursor {
            Some(c) if !self.input.looking => c,
            _ => (extent.0 / 2.0, extent.1 / 2.0),
        }
    }

    /// `--edit-script`: remove at the centre, then place at the centre half a period later.
    fn script_edits(&mut self) {
        if !self.args.edit_script || self.args.frames.is_none() {
            return;
        }
        let Some(t) = self.state.as_ref().and_then(|s| s.targets.as_ref()) else { return };
        let c = (t.extent.width as f64 / 2.0, t.extent.height as f64 / 2.0);
        let n = self.args.edit_size;
        match (self.frame_index % EDIT_PERIOD, n) {
            (10, 1) => self.edits.requests.push(EditRequest::Remove(c.0, c.1)),
            (30, 1) => self.edits.requests.push(EditRequest::Place(c.0, c.1)),
            (10, _) => self.edits.requests.push(EditRequest::RemoveBox(c.0, c.1, n)),
            (30, _) => self.edits.requests.push(EditRequest::RestoreBox),
            _ => {}
        }
    }

    /// The voxels an edit request sets (None removes) and its box (`hi` exclusive), or None when it
    /// is rejected: nothing under the pixel, no face to place against, a voxel inside the walker, or
    /// nothing to restore.
    #[allow(clippy::type_complexity)]
    fn edit_voxels(&mut self, req: EditRequest, camera: &Camera, extent: vk::Extent2D) -> Option<(Vec<(VoxelCoord, Option<MaterialId>)>, [i32; 3], [i32; 3])> {
        let (x, y) = match req {
            EditRequest::RestoreBox => {
                let (lo, hi) = self.edits.removed_bounds.take()?;
                let voxels = std::mem::take(&mut self.edits.removed_box);
                return (!voxels.is_empty()).then_some((voxels, lo, hi));
            }
            EditRequest::Remove(x, y) | EditRequest::Place(x, y) | EditRequest::RemoveBox(x, y, _) => (x, y),
        };
        let (px, py) = ((x.max(0.0) as u32).min(extent.width - 1), (y.max(0.0) as u32).min(extent.height - 1));
        let hit = world::reference::trace(&self.world, &camera.ray(px, py), REACH)?;
        let v = hit.voxel;
        match req {
            EditRequest::Remove(..) => Some((vec![(v, None)], [v.x, v.y, v.z], [v.x + 1, v.y + 1, v.z + 1])),
            EditRequest::Place(..) => {
                let n = hit.face?.normal();
                let at = VoxelCoord::new(v.x + n[0] as i32, v.y + n[1] as i32, v.z + n[2] as i32);
                let (lo, hi) = self.walker.aabb();
                let p = [at.x as f64, at.y as f64, at.z as f64];
                if self.mode == Mode::Walk && (0..3).all(|a| lo[a] < p[a] + 1.0 && hi[a] > p[a]) {
                    return None;
                }
                Some((vec![(at, Some(hit.material))], [at.x, at.y, at.z], [at.x + 1, at.y + 1, at.z + 1]))
            }
            EditRequest::RemoveBox(_, _, n) => {
                let lo = [v.x - n / 2, v.y - n / 2, v.z - n / 2];
                let hi = [lo[0] + n, lo[1] + n, lo[2] + n];
                let mut saved = Vec::new();
                for x in lo[0]..hi[0] {
                    for y in lo[1]..hi[1] {
                        for z in lo[2]..hi[2] {
                            let c = VoxelCoord::new(x, y, z);
                            if let Some(m) = self.world.get(c) {
                                saved.push((c, Some(m)));
                            }
                        }
                    }
                }
                let removed = saved.iter().map(|&(c, _)| (c, None)).collect();
                self.edits.removed_box = saved;
                self.edits.removed_bounds = Some((lo, hi));
                Some((removed, lo, hi))
            }
            EditRequest::RestoreBox => unreachable!(),
        }
    }

    /// Commits this frame's edit requests, publishes, and updates the device snapshot.
    fn apply_edits(&mut self) -> Result<(), String> {
        let e = |x: gpu::GpuError| x.to_string();
        let requests = std::mem::take(&mut self.edits.requests);
        let Some(extent) = self.state.as_ref().and_then(|s| s.targets.as_ref()).map(|t| t.extent) else {
            return Ok(());
        };
        let camera = self.fly.camera(extent);
        for req in requests {
            let Some((voxels, lo, hi)) = self.edit_voxels(req, &camera, extent) else {
                self.edits.rejected += 1;
                continue;
            };
            let start = Instant::now();
            let mut tx = Transaction::new();
            for &(c, m) in &voxels {
                tx.set(c, m);
            }
            let applied = self.world.apply(&tx).map_err(|x| format!("edit: {x:?}"))?;
            if applied.changed.is_empty() {
                self.edits.rejected += 1;
                continue;
            }
            let s = self.state.as_mut().expect("state");
            s.pipeline.notify_edits(&applied.changed);
            loop {
                let jobs = s.pipeline.dispatch(&self.world);
                if jobs.is_empty() {
                    break;
                }
                for j in jobs {
                    s.pipeline.complete(&self.world, j.run());
                }
            }
            if let Some(p) = s.pipeline.try_publish(&self.world) {
                self.edits.dirty.extend(p.groups.into_iter().flatten());
            }
            self.edits.relight_pending.push(Relight { lo, hi });
            let t = EditTiming { voxels: applied.changed.len(), commit_ms: start.elapsed().as_secs_f64() * 1e3, ..EditTiming::default() };
            self.edits.unshown.push((start, t));
            self.edits.applied += 1;
        }
        if self.edits.dirty.is_empty() {
            return Ok(());
        }
        // One device update for everything published since the last successful one.
        let s = self.state.as_mut().expect("state");
        let size = s.scene.size();
        let t = Instant::now();
        let token = s.pipeline.acquire();
        let changed = region_meshes(&s.pipeline, &token, &affected_regions(self.edits.dirty.iter().copied(), size), size).map_err(|x| format!("snapshot read: {x:?}"))?;
        s.pipeline.release(token).map_err(|x| format!("{x:?}"))?;
        let layout_ms = t.elapsed().as_secs_f64() * 1e3;
        let regions = changed.len();
        let snapshot = s.pipeline.current().id.raw();
        let u = match s.scene.update(&s.gpu, &mut s.alloc, &mut s.up, &mut s.timeline, changed, snapshot) {
            Ok(u) => u,
            Err(gpu::GpuError::OverBudget(x)) => {
                self.edits.deferred += 1;
                eprintln!("edit update refused, previous snapshot stays on screen: {x:?}");
                return Ok(());
            }
            Err(x) => return Err(x.to_string()),
        };
        self.edits.dirty.clear();
        let pending = std::mem::take(&mut self.edits.relight_pending);
        self.edits.relight.extend(pending);
        // Rebuild what points at replaced buffers. The frames in flight have finished (the
        // acceleration update waited for them; the wait here covers the mesh-only path).
        let t = Instant::now();
        s.frames.wait_all(&s.gpu, &s.timeline).map_err(e)?;
        let old = std::mem::replace(&mut s.bindings, s.raster.bind(&s.gpu, &s.scene.meshes).map_err(e)?);
        old.destroy(&s.gpu);
        if let (Some(r), Some(o), Some(a)) = (&s.ray, &s.ray_out, &s.scene.accel) {
            if let Some(old) = s.ray_bindings.replace(r.bind(&s.gpu, a, o).map_err(e)?) {
                old.destroy(&s.gpu);
            }
        }
        let (old_table, _) = s.tables.replace_regions(&s.gpu, &mut s.alloc, &mut s.up, &mut s.timeline, &s.scene.region_rows()).map_err(e)?;
        s.scene.retire(s.timeline.last_signal(), Garbage::Buffer(old_table));
        if let Some(targets) = &s.targets {
            s.debug.bind(&s.gpu, targets, &s.tables, s.ray_out.as_ref());
            // 3B: the shade pass reads the new TLAS.
            if let (Some(sh), Some(o), Some(a), Some(al), Some(sky)) = (&s.shade, &s.shade_out, &s.scene.accel, &s.albedo, &s.sky) {
                if let Some(old) = s.shade_bindings.replace(sh.bind(&s.gpu, targets, a, al, o, &sky.view).map_err(e)?) {
                    old.destroy(&s.gpu);
                }
                s.debug.bind_radiance(&s.gpu, &o.radiance);
                // 3D: the temporal pass reads the new region table.
                if let (Some(tp), Some(h)) = (&s.temporal, &s.history) {
                    if let Some(old) = s.temporal_bindings.replace(tp.bind(&s.gpu, targets, &s.tables.regions, &o.radiance, h).map_err(e)?) {
                        old.destroy(&s.gpu);
                    }
                    s.debug.bind_motion(&s.gpu, &h.motion);
                }
            }
        }
        // 4B: the lit shade set and compose read the new emitter table.
        s.bind_lights()?;
        let rebind_ms = t.elapsed().as_secs_f64() * 1e3;
        for (_, t) in self.edits.unshown.iter_mut() {
            t.layout_ms = layout_ms;
            t.upload_ms = u.upload_ms;
            t.accel_ms = u.accel_ms;
            t.rebind_ms = rebind_ms;
            t.emitters_ms = u.emitters_ms;
            t.regions = regions;
        }
        Ok(())
    }

    /// At exit: waits for the GPU, so every shown edit gets its visible time.
    fn finish_edits(&mut self) {
        let Some(s) = &self.state else { return };
        let _ = s.frames.wait_all(&s.gpu, &s.timeline);
        let done = s.timeline.completed(&s.gpu).unwrap_or(0);
        let edits = &mut self.edits;
        edits.in_flight.retain(|&(v, start, mut t)| {
            if v <= done {
                t.visible_ms = start.elapsed().as_secs_f64() * 1e3;
                edits.done.push(t);
                false
            } else {
                true
            }
        });
    }

    fn edits_summary(&self) -> String {
        let e = &self.edits;
        let col = |f: fn(&EditTiming) -> f64| Series::summary(&e.done.iter().map(f).collect::<Vec<_>>());
        format!(
            "{{\"applied\":{},\"shown\":{},\"rejected\":{},\"deferred\":{},\"not_shown_at_exit\":{},\"voxels\":{},\"regions\":{},\"commit_ms\":{},\"layout_ms\":{},\"upload_ms\":{},\"accel_ms\":{},\"rebind_ms\":{},\"emitters_ms\":{},\"visible_ms\":{},\"visible_all_ms\":[{}]}}",
            e.applied,
            e.done.len(),
            e.rejected,
            e.deferred,
            e.in_flight.len() + e.unshown.len(),
            e.done.iter().map(|t| t.voxels).sum::<usize>(),
            col(|t| t.regions as f64),
            col(|t| t.commit_ms),
            col(|t| t.layout_ms),
            col(|t| t.upload_ms),
            col(|t| t.accel_ms),
            col(|t| t.rebind_ms),
            col(|t| t.emitters_ms),
            col(|t| t.visible_ms),
            e.done.iter().map(|t| format!("{:.2}", t.visible_ms)).collect::<Vec<_>>().join(",")
        )
    }

    fn walk_summary(&self) -> String {
        let w = &self.walker;
        let st = w.stats;
        format!(
            "{{\"feet\":[{:.4},{:.4},{:.4}],\"grounded\":{},\"substeps\":{},\"steps_up\":{},\"snaps_down\":{},\"jumps\":{},\"voxels_tested\":{},\"overlap_frames\":{},\"respawns\":{},\"script_loops\":{},\"update_ms\":{}}}",
            w.feet[0],
            w.feet[1],
            w.feet[2],
            w.grounded,
            st.substeps,
            st.steps_up,
            st.snaps_down,
            st.jumps,
            st.voxels,
            self.overlaps,
            self.respawns,
            self.script_loops,
            Series::summary(&self.all.walk)
        )
    }

    fn phases_summary(&self) -> String {
        let col = |f: fn(&Phases) -> f64| Series::summary(&self.phases.iter().map(f).collect::<Vec<_>>());
        let slow: Vec<String> = self
            .slow
            .iter()
            .take(40)
            .map(|(i, ms, p)| format!("[{i},{ms:.2},{:.2},{:.2},{:.2},{:.2},{:.2}]", p.begin, p.acquire, p.present, p.inside, ms - p.inside))
            .collect();
        let over = |t: f64| self.all.frame.iter().filter(|&&v| v > t).count();
        format!(
            "{{\"pwait\":{},\"begin\":{},\"acquire\":{},\"present\":{},\"inside\":{},\"over_16_7\":{},\"over_33_3\":{},\"slow_cols\":\"frame,interval,begin,acquire,present,inside,outside\",\"slow\":[{}]}}",
            col(|p| p.pwait),
            col(|p| p.begin),
            col(|p| p.acquire),
            col(|p| p.present),
            col(|p| p.inside),
            over(SLOW_MS),
            over(2.0 * SLOW_MS),
            slow.join(",")
        )
    }

    fn summary(&self) -> String {
        let s = self.state.as_ref().expect("state");
        let sw = s.swapchain.as_ref();
        let (errors, warnings) = s.gpu.validation_counts();
        let passes: Vec<String> = PASSES.iter().zip(&self.all.passes).map(|(n, v)| format!("\"{n}\":{}", Series::summary(v))).collect();
        format!(
            "{{\"run\":\"viewer\",\"device\":{:?},\"validation\":{},\"scene\":\"{}\",\"dressing\":{},\"emitters\":{},\"merge\":\"{}\",\"region\":\"{}\",\"source\":\"{}\",\"edits\":{},\"present\":\"{:?}\",\"extent\":[{},{}],\"frames\":{},\"frames_in_flight\":{},\"swapchain_recreates\":{},\"scripted\":{},\"cycle_views\":{},\"triangles\":{},\"snapshot\":{},\"frame_ms\":{},\"fps\":{},\"gpu_ms\":{{{}}},\"mode\":\"{}\",\"view\":\"{}\",\"hour\":{:.3},\"accumulate\":{},\"denoise\":{},\"prefilter_age\":{},\"bounce\":{},\"sky_correction\":{},\"max_age\":{},\"present_wait\":{},\"present_wait_supported\":{},\"present_wait_timeouts\":{},\"start_unix_ms\":{},\"refresh_mhz\":{},\"focus_lost\":{},\"occluded\":{},\"input_events\":{},\"phases_ms\":{},\"walk\":{},\"lights\":{{\"on\":{},\"manual\":{},\"switches\":{}}},\"exposure\":{{\"mode\":\"{}\",\"value\":{},\"target\":{}}},\"emitter_spp\":{},\"stale_table_frames\":{},\"validation_errors\":{errors},\"validation_warnings\":{warnings},\"readers_held_at_exit\":{}}}",
            s.gpu.info.name,
            s.gpu.validation_enabled(),
            if self.args.night.is_some() { "night" } else { "street" },
            self.args.night.map_or("null".to_string(), |d| format!("\"{}\"", d.name())),
            match s.scene.emitters() {
                Ok(Some(em)) => em.table.len().to_string(),
                Ok(None) => "null".to_string(),
                Err(x) => format!("\"stale: {x:?}\""),
            },
            self.args.merge.name(),
            self.args.region.name(),
            if self.source == Source::Ray { "ray" } else { "raster" },
            self.edits_summary(),
            sw.map(|x| x.present_mode).unwrap_or(self.args.present),
            sw.map_or(0, |x| x.extent.width),
            sw.map_or(0, |x| x.extent.height),
            self.frame_index,
            FRAMES_IN_FLIGHT,
            self.recreates,
            self.args.frames.is_some(),
            self.args.cycle_views,
            s.scene.meshes.triangle_count(),
            s.scene.snapshot,
            Series::summary(&self.all.frame),
            fps_lows(self.all.frame.iter().copied()).map_or("null".into(), |(a, l10, l1)| format!("{{\"avg\":{a:.1},\"low10\":{l10:.1},\"low1\":{l1:.1}}}")),
            passes.join(","),
            if self.mode == Mode::Walk { "walk" } else { "fly" },
            self.view.name(),
            self.hour,
            self.accumulate,
            self.accumulate && self.denoise,
            self.prefilter_age,
            self.bounce,
            self.luts.corrected,
            self.args.max_age,
            self.args.present_wait.map_or("null".to_string(), |n| n.to_string()),
            s.gpu.present_wait(),
            self.present_wait_timeouts,
            self.run_start.map_or(0, |(_, u)| u),
            self.refresh_mhz.map_or("null".to_string(), |r| r.to_string()),
            self.focus_lost,
            self.occluded,
            self.input_events,
            self.phases_summary(),
            self.walk_summary(),
            self.lights_shown,
            self.lights.manual(),
            self.lights_switches,
            if self.lights_shown { "auto" } else { "m3" },
            self.exposure.last.map_or("null".to_string(), |x| format!("{x:.6e}")),
            self.exposure.target.map_or("null".to_string(), |x| format!("{x:.6e}")),
            self.args.emitter_spp,
            self.stale_table_frames,
            s.readers.len()
        )
    }

    fn shutdown(&mut self) {
        let Some(s) = self.state.take() else { return };
        let State { gpu, mut alloc, timeline, up, raster, scene, bindings, ray, ray_out, ray_bindings, tables, shade, shade_out, shade_bindings, albedo, sky, sky_pass, temporal, history, temporal_bindings, denoise, denoise_targets, denoise_bindings, shade_lit_bindings, compose, compose_out, compose_bindings, swapchain, targets, debug, frames, timer, mut readers, mut pipeline, window, .. } = s;
        let _ = frames.wait_all(&gpu, &timeline);
        let _ = gpu.wait_idle();
        let done = timeline.completed(&gpu).unwrap_or(0);
        readers.release_completed(&mut pipeline, done);
        assert!(readers.is_empty(), "every frame's reader token is released after the GPU finished");
        frames.destroy(&gpu);
        timer.destroy(&gpu);
        debug.destroy(&gpu);
        if let Some(t) = targets {
            t.free(&gpu, &mut alloc);
        }
        if let Some(sw) = swapchain {
            sw.destroy(&gpu);
        }
        bindings.destroy(&gpu);
        raster.destroy(&gpu);
        if let Some(b) = ray_bindings {
            b.destroy(&gpu);
        }
        if let Some(o) = ray_out {
            o.free(&gpu, &mut alloc);
        }
        if let Some(r) = ray {
            r.destroy(&gpu);
        }
        if let Some(b) = shade_bindings {
            b.destroy(&gpu);
        }
        if let Some(o) = shade_out {
            o.free(&gpu, &mut alloc);
        }
        if let Some(sh) = shade {
            sh.destroy(&gpu);
        }
        if let Some(a) = albedo {
            a.free(&gpu, &mut alloc);
        }
        if let Some(p) = sky_pass {
            p.destroy(&gpu);
        }
        if let Some(b) = temporal_bindings {
            b.destroy(&gpu);
        }
        if let Some(h) = history {
            h.free(&gpu, &mut alloc);
        }
        if let Some(t) = temporal {
            t.destroy(&gpu);
        }
        if let Some(b) = denoise_bindings {
            b.destroy(&gpu);
        }
        if let Some(t) = denoise_targets {
            t.free(&gpu, &mut alloc);
        }
        if let Some(d) = denoise {
            d.destroy(&gpu);
        }
        if let Some(b) = shade_lit_bindings {
            b.destroy(&gpu);
        }
        if let Some(b) = compose_bindings {
            b.destroy(&gpu);
        }
        if let Some(c) = compose_out {
            c.free(&gpu, &mut alloc);
        }
        if let Some(c) = compose {
            c.destroy(&gpu);
        }
        if let Some(t) = sky {
            t.free(&gpu, &mut alloc);
        }
        tables.free_now(&gpu, &mut alloc);
        scene.free_now(&gpu, &mut alloc);
        up.destroy(&gpu, &mut alloc);
        timeline.destroy(&gpu);
        let leaked = alloc.destroy(&gpu);
        if leaked != 0 {
            eprintln!("LEAK: {leaked} buffers or images still live at exit");
            self.exit_code = 1;
        }
        drop(gpu);
        drop(window);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match self.init(el) {
            Ok(s) => {
                self.refresh_mhz = s.window.current_monitor().and_then(|m| m.refresh_rate_millihertz());
                self.state = Some(s);
                self.window_start = Instant::now();
            }
            Err(msg) => {
                self.error = Some(msg);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let pressed = match &event {
            WindowEvent::KeyboardInput { event, .. } => event.state == ElementState::Pressed,
            WindowEvent::MouseInput { state, .. } => *state == ElementState::Pressed,
            WindowEvent::MouseWheel { .. } => true,
            _ => false,
        };
        if pressed {
            self.input_events += 1;
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(_) => self.resize = true,
            WindowEvent::Focused(false) => self.focus_lost += 1,
            WindowEvent::Occluded(true) => self.occluded += 1,
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let down = event.state == ElementState::Pressed;
                    self.input.keys.insert(code, down);
                    if down {
                        let digits = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5, KeyCode::Digit6, KeyCode::Digit7];
                        if let Some(i) = digits.iter().position(|&d| d == code) {
                            self.view = View::ALL[i];
                        }
                        if code == KeyCode::Digit0 {
                            self.view = View::Light;
                        }
                        if code == KeyCode::Digit9 {
                            self.view = View::Lit;
                        }
                        // Exposure: - and =, in steps of a quarter stop (the light view keeps its own).
                        if code == KeyCode::Minus || code == KeyCode::Equal {
                            let q = if code == KeyCode::Minus { -0.25f32 } else { 0.25 };
                            if self.view == View::Light {
                                self.light_stops = (self.light_stops + q).clamp(-8.0, 8.0);
                            } else {
                                self.lighting.exposure = (self.lighting.exposure * 2.0f32.powf(q)).clamp(1.0 / 64.0, 64.0);
                            }
                        }
                        // Time of day: [ and ] a quarter hour; T runs the day.
                        if code == KeyCode::BracketLeft || code == KeyCode::BracketRight {
                            let q = if code == KeyCode::BracketLeft { -0.25 } else { 0.25 };
                            self.set_hour(self.hour + q);
                        }
                        if code == KeyCode::KeyT {
                            self.run_day = !self.run_day;
                        }
                        if code == KeyCode::KeyH {
                            self.accumulate = !self.accumulate;
                        }
                        if code == KeyCode::KeyN {
                            self.denoise = !self.denoise;
                            self.last_shaded = None;
                        }
                        if code == KeyCode::KeyP {
                            self.prefilter_age = if self.prefilter_age == u32::MAX { self.args.prefilter_age } else { u32::MAX };
                        }
                        if code == KeyCode::KeyL {
                            self.lights.toggle();
                        }
                        if code == KeyCode::KeyB {
                            self.bounce = !self.bounce;
                            self.last_shaded = None;
                        }
                        if code == KeyCode::Digit8 {
                            self.view = match self.view {
                                View::HistoryAge => View::HistoryReason,
                                View::HistoryReason => View::Motion,
                                _ => View::HistoryAge,
                            };
                        }
                        if code == KeyCode::Escape {
                            el.exit();
                        }
                        if code == KeyCode::KeyR {
                            self.source = if self.source == Source::Raster { Source::Ray } else { Source::Raster };
                        }
                        if code == KeyCode::KeyE && self.args.frames.is_none() {
                            let (x, y) = self.edit_pixel();
                            self.edits.requests.push(EditRequest::Place(x, y));
                        }
                        if code == KeyCode::KeyF && self.args.frames.is_none() {
                            if self.mode == Mode::Walk {
                                self.mode = Mode::Fly;
                            } else {
                                let e = self.fly.eye;
                                let feet = [e[0], e[1] - self.walker.params.eye_height, e[2]];
                                if !self.start_walking(feet) {
                                    eprintln!("no free space to walk above {feet:?}; still flying");
                                }
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => self.input.looking = state == ElementState::Pressed,
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } if self.args.frames.is_none() && matches!(button, MouseButton::Left | MouseButton::Middle) => {
                let (x, y) = self.edit_pixel();
                self.edits.requests.push(if button == MouseButton::Left { EditRequest::Remove(x, y) } else { EditRequest::Place(x, y) });
            }
            WindowEvent::CursorMoved { position, .. } => self.input.cursor = Some((position.x, position.y)),
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                self.fly.speed = (self.fly.speed * 1.25f64.powf(steps)).clamp(1.0, 4096.0);
            }
            WindowEvent::RedrawRequested => {
                if let Err(msg) = self.frame(el) {
                    self.error = Some(msg);
                    el.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.input.looking && self.args.frames.is_none() {
                self.fly.yaw += delta.0 * 0.003;
                self.fly.pitch = (self.fly.pitch - delta.1 * 0.003).clamp(-1.55, 1.55);
            }
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\nusage: viewer [--merge greedy|none] [--region brick|2x2x2_bricks|chunk|2x2x2_chunks] [--present mailbox|fifo|immediate] [--view 0-12] [--hour H] [--run-day] [--max-age N] [--no-denoise] [--no-bounce] [--no-sky-correction] [--source raster|ray] [--camera street|low] [--prefilter-age N] [--scene street|night [--dressing lamps|windows|full|dense]] [--lights auto|on|off] [--emitter-spp 1|2|4] [--size WxH] [--frames N [--walk] [--cycle-views] [--edit-script [--edit-size N]] [--resize-at FRAME WxH]] [--log PATH] [--trace CSV] [--present-wait N]");
            std::process::exit(2);
        }
    };
    let (world, view) = match args.night {
        Some(d) => scene::street_night(d),
        None => scene::street_block(),
    };
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    let (eye, target) = (vox(view.eye_m), vox(view.target_m));
    let d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    let mut fly = Fly { eye, yaw: d[2].atan2(d[0]), pitch: (d[1] / (d[0] * d[0] + d[2] * d[2]).sqrt()).atan(), fov_deg: view.vertical_fov_deg, speed: 64.0 };
    if args.camera_low {
        // The 3A low camera (voxel units), as in `gpu/tests/gate.rs`.
        let (e, t) = ([4.3, 30.0, 20.2], [380.0, 60.0, 30.0]);
        let d = [t[0] - e[0], t[1] - e[1], t[2] - e[2]];
        fly = Fly { eye: e, yaw: d[2].atan2(d[0]), pitch: (d[1] / (d[0] * d[0] + d[2] * d[2]).sqrt()).atan(), fov_deg: 70.0, speed: 64.0 };
    }
    let initial_view = args.view;
    let start_hour = args.hour;
    let run_day_at_start = args.run_day;
    let denoise_at_start = !args.no_denoise;
    let bounce_at_start = !args.no_bounce;
    let lights_at_start = args.lights;
    // 3C: the sun-independent sky tables (about 0.3 s), corrected from the baked reference (S-020).
    let mut luts = light::sky::SkyLuts::new(light::Atmosphere::default());
    if !args.no_sky_correction {
        match light::sky_ref::SkyReference::load_default(&luts.atmosphere).and_then(|r| luts.apply_reference(&r)) {
            Ok(st) => eprintln!("sky correction applied (ratio {:.3} to {:.3}, {} clamped)", st.min, st.max, st.clamped),
            Err(e) => eprintln!("WARNING: running without the sky correction: {e}"),
        }
    }
    let initial_source = args.source;
    let params = walk::Params::default();
    let spawn = [eye[0], eye[1] - params.eye_height, eye[2]];
    let kill_y = world.bounds().map_or(0, |(lo, _)| lo.y) as f64 - 256.0;
    let mut app = App {
        args,
        world,
        fly,
        mode: Mode::Fly,
        lighting: Lighting::new(sun_at(start_hour).sun_dir),
        hour: start_hour,
        sun: sun_at(start_hour),
        run_day: run_day_at_start,
        light_stops: 0.0,
        sky_zenith: sky_zenith(&luts, &sun_at(start_hour)),
        luts,
        sky_hour: None,
        accumulate: true,
        denoise: denoise_at_start,
        prefilter_age: u32::MAX,
        bounce: bounce_at_start,
        lights: Lights::new(lights_at_start),
        lights_shown: false,
        lights_switches: 0,
        exposure: Exposure::default(),
        metered: [false; FRAMES_IN_FLIGHT],
        stale_table_frames: 0,
        last_shaded: None,
        walker: walk::Walker::new(params, spawn),
        spawn,
        kill_y,
        respawns: 0,
        script_loops: 0,
        overlaps: 0,
        input: Input::default(),
        view: initial_view,
        source: initial_source,
        edits: Edits::default(),
        state: None,
        resize: false,
        recreates: 0,
        frame_index: 0,
        last: None,
        all: Series::default(),
        phases: Vec::new(),
        slow: Vec::new(),
        trace: Vec::new(),
        focus_lost: 0,
        input_events: 0,
        occluded: 0,
        refresh_mhz: None,
        run_start: None,
        present_wait_timeouts: 0,
        window_series: Series::default(),
        recent: VecDeque::new(),
        window_start: Instant::now(),
        exit_code: 0,
        error: None,
    };
    // Walking is the start, except for the scripted fly path.
    if (app.args.frames.is_none() || app.args.walk_script) && !app.args.camera_low {
        assert!(app.start_walking(spawn), "the reference view's feet are free");
        app.fly.eye = app.walker.render_eye();
    }
    let el = EventLoop::new().expect("event loop");
    el.set_control_flow(ControlFlow::Poll);
    if let Err(e) = el.run_app(&mut app) {
        eprintln!("event loop: {e}");
        app.exit_code = 1;
    }
    if let Some(msg) = &app.error {
        eprintln!("FAIL: {msg}");
        app.exit_code = 1;
    }
    if app.state.is_some() {
        app.finish_edits();
        let line = app.summary();
        if let Some((avg, low10, low1)) = fps_lows(app.all.frame.iter().copied()) {
            eprintln!("fps over the run ({} frames): average {avg:.1}, 10% low {low10:.1}, 1% low {low1:.1}", app.all.frame.len());
        }
        println!("{line}");
        if let Some(path) = &app.args.log {
            let f = std::fs::OpenOptions::new().create(true).append(true).open(path);
            if let Err(e) = f.and_then(|mut f| writeln!(f, "{line}")) {
                eprintln!("cannot append to {path}: {e}");
                app.exit_code = 1;
            }
        }
        if let Some(path) = &app.args.trace {
            let start = app.run_start.map_or(0, |(_, u)| u);
            let mut csv = format!("# start_unix_ms {start}\nframe,at_ms,interval_ms,pwait_ms,begin_ms,acquire_ms,present_ms,inside_ms,gpu_gbuffer_ms,gpu_view_ms,gpu_ray_ms\n");
            for (i, (iv, p)) in app.trace.iter().enumerate() {
                csv.push_str(&format!("{i},{:.3},{iv:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n", p.at, p.pwait, p.begin, p.acquire, p.present, p.inside, p.gpu[0], p.gpu[1], p.gpu[2]));
            }
            if let Err(e) = std::fs::write(path, csv) {
                eprintln!("cannot write {path}: {e}");
                app.exit_code = 1;
            }
        }
        // Warnings count as failures too (a sampled-image format mismatch is only a warning).
        let (errors, warnings) = app.state.as_ref().map(|s| s.gpu.validation_counts()).unwrap_or((0, 0));
        if errors > 0 || warnings > 0 {
            eprintln!("FAIL: {errors} validation errors, {warnings} warnings: {:?}", app.state.as_ref().map(|s| s.gpu.first_validation_errors()));
            app.exit_code = 1;
        }
        if app.args.edit_script && (app.edits.done.is_empty() || !app.edits.in_flight.is_empty() || !app.edits.unshown.is_empty() || !app.edits.dirty.is_empty()) {
            eprintln!("FAIL: the edit script shows {} edits, {} not shown, {} regions' bricks still dirty", app.edits.done.len(), app.edits.in_flight.len() + app.edits.unshown.len(), app.edits.dirty.len());
            app.exit_code = 1;
        }
        if app.overlaps > 0 {
            eprintln!("FAIL: the walker overlapped a solid voxel on {} frames", app.overlaps);
            app.exit_code = 1;
        }
        // The scripted walk must stay in the street: falling off the world means the run measured sky.
        if app.args.walk_script && app.respawns > 0 {
            eprintln!("FAIL: the scripted walk fell off the world {} times", app.respawns);
            app.exit_code = 1;
        }
        app.shutdown();
    }
    std::process::exit(app.exit_code);
}

/// The sun at `hour` (ADR-0005 path: 45° N, equinox).
fn sun_at(hour: f64) -> light::reference::Lighting {
    light::reference::Lighting::new(light::Atmosphere::default(), light::SunPath::default().direction(hour))
}

/// Display exposure of the light view: a white Lambertian surface facing the sun, lit by the sun and
/// by the sky, maps near 1, shifted by `stops`. Display only.
fn light_exposure(sun: &light::reference::Lighting, sky: f64, stops: f32) -> f32 {
    let g = sun.sun_at_ground;
    let y = 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2];
    (2.0f64.powf(stops as f64) / (y / std::f64::consts::PI + sky + 1e-7)) as f32
}

/// The sky's mean radiance over the upper hemisphere (the ground-irradiance table / pi), luminance,
/// for the display exposure. The zenith alone is far darker than the horizon glow after sunset.
fn sky_zenith(luts: &light::sky::SkyLuts, sun: &light::reference::Lighting) -> f64 {
    let e = luts.ground_irradiance(sun.sun_dir[1]);
    (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]) / std::f64::consts::PI
}
