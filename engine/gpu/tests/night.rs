//! Phase 4B GPU tests: the street's lights in the real-time path (the lit shade module, emission and
//! the meter after reconstruction, the lights as a light jump) against the reference, and the
//! measurements that set up 4C and 4D. Criteria are frozen in
//! `docs/changes/2026-09-26-phase4b-many-lights.md` (G10, G11, G13, G14, Q1–Q4, M1–M3; part 2: R3,
//! R4). Like the
//! other GPU tests they need the Vulkan SDK and an RT GPU and fail (never skip) without them.
//!
//! With `NE_4B_DIR` set, the 1080p references and converged images are cached there (keyed by every
//! parameter) and the display images for FLIP are written there (`results/phase4b/flip.py`).
//! `equal_time_curve_cost` is a timing run: set `NE_NO_VALIDATION=1` for it.
//! Run: `cargo test --release -j 2 -p gpu --test night -- --test-threads=1 --nocapture`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::alloc::Allocator;
use gpu::compose::{Compose, ComposeBindings, ComposeFaults, ComposeTargets};
use gpu::debug_view::{targets_to_read, Tables};
use gpu::denoise::{Denoise, DenoiseBindings, DenoiseFaults, DenoiseSettings, DenoiseTargets};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::raster::{Bindings as RasterBindings, Camera, Faults, Raster, Targets};
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::scene::{affected_regions, region_meshes, GpuScene};
use gpu::shade::{self, Shade, ShadeBindings, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::temporal::{light_rows, reason, relight_rows, relit_by_light, History, LightRow, Relight, Temporal, TemporalBindings, TemporalFaults, TemporalSettings};
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, Timeline};
use light::exposure::{metric_exposure, Meter};
use light::reference::{albedos, Accum, Lighting, Settings};
use light::sky::SkyLuts;
use light::sky_ref::SkyReference;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::reference::trace;
use world::scene::{self, street_night, Dressing};
use world::{BrickKey, MaterialId, Transaction, VoxelCoord, World};

const NEAR: f64 = 0.1;
/// Shade and reference seed of the exactness, convergence and emission checks.
const SEED: u32 = 0x4B;
/// Q: the shade seeds are `Q_SEED + k`, k < `Q_SEEDS`; the references use `REF_SEED`.
const Q_SEED: u32 = 0x4B00;
const Q_SEEDS: u32 = 8;
const REF_SEED: u32 = 0x4C;
const W: u32 = 1920;
const H: u32 = 1080;
const REF_SPP: u32 = 8192;
/// Frames of a converged real-time image (s = 4, `max_age` = this).
const TARGET_FRAMES: u32 = 8192;
const TARGET_SPP: u32 = 4;
/// Reference frames per submission (short enough for the display driver's watchdog at 1080p).
const REF_PER_SUBMIT: u32 = 4;
/// Bump when a camera or a setting of the cached images changes.
const CACHE_VERSION: u32 = 1;
const NIGHT: f64 = 21.0;
const BLUE_HOUR: f64 = 18.5;
const STILL_TIMES: [(&str, f64); 2] = [("blue_hour", BLUE_HOUR), ("night", NIGHT)];
const MOTION_FRAMES: [u32; 3] = [8, 16, 32];
const FACE_NONE: u32 = 7;

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "4B needs the ray-tracing device (P-001)");
    g
}

/// For timing runs: validation may be off (`NE_NO_VALIDATION`).
fn gpu_for_timing() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device");
    assert!(g.ray_tracing());
    g
}

fn lighting(hour: f64) -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(hour))
}

/// The 3A reference cameras (street, low), plus B1's overhead camera when `overhead`.
fn cameras(w: u32, h: u32, overhead: bool) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    let mut v = vec![
        ("street", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)),
        ("low", Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, NEAR)),
    ];
    if overhead {
        v.push(("overhead", Camera::look_at([-60.5, 300.25, -80.75], [192.0, 0.0, 150.0], 50.0, w, h, NEAR)));
    }
    v
}

/// The 3E D4 path: the street camera moved sideways 0.25 voxel and turned 0.2° per frame.
fn motion_camera(k: u32, w: u32, h: u32) -> Camera {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    let (eye, target) = (vox(view.eye_m), vox(view.target_m));
    let f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    let (s, c) = (0.2 * k as f64).to_radians().sin_cos();
    let f2 = [f[0] * c - f[2] * s, f[1], f[0] * s + f[2] * c];
    let e2 = [eye[0] + 0.25 * k as f64, eye[1], eye[2]];
    Camera::look_at(e2, [e2[0] + f2[0], e2[1] + f2[1], e2[2] + f2[2]], view.vertical_fov_deg, w, h, NEAR)
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn lum4(c: &[f32; 4]) -> f64 {
    lum([c[0] as f64, c[1] as f64, c[2] as f64])
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

fn full_regions(p: &mut Pipeline, size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    let t = p.acquire();
    let keys: Vec<BrickKey> = p.current().keys().collect();
    let meshes: Vec<_> = keys.iter().map(|&k| (k, p.read(&t, k).unwrap().unwrap().clone())).collect();
    p.release(t).unwrap();
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
}

/// Makes every earlier command's writes visible to every later command (between batched frames).
fn full_barrier(g: &Gpu, cmd: vk::CommandBuffer) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)];
    unsafe { g.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

/// One dressing of `street_night` on the device, with its emitter table, region table and albedos.
struct Night {
    dressing: Dressing,
    /// Part 2: the world and its 1C pipeline, for edits.
    world: World,
    pipeline: Pipeline,
    scene: GpuScene,
    tables: Tables,
    mats: RefMaterials,
    /// The table's per-material emitted radiance (host copy).
    emission: Vec<[f64; 3]>,
    emitters: usize,
}

/// Where the compose pass runs in a frame.
#[derive(Clone, Copy, PartialEq)]
enum ComposeAt {
    Off,
    /// After the temporal pass and the filter (the viewer's order).
    After,
    /// Planted fault: between the shade pass and the temporal pass.
    BeforeTemporal,
}

/// One frame's settings.
#[derive(Clone, Copy)]
struct Step {
    set: ShadeSettings,
    faults: ShadeFaults,
    /// The temporal pass's settings; `None`: no temporal pass (the raw shade output).
    temporal: Option<TemporalSettings>,
    filter: bool,
    lights: bool,
    compose: ComposeAt,
    compose_faults: ComposeFaults,
    floor: f64,
    /// Part 2: a forced reset of the temporal pass.
    reset: bool,
    frame: u32,
    seed: u32,
}

impl Step {
    fn raw(set: ShadeSettings, frame: u32, seed: u32) -> Step {
        Step { set, faults: ShadeFaults::default(), temporal: None, filter: false, lights: set.emitters, compose: ComposeAt::Off, compose_faults: ComposeFaults::default(), floor: 0.0, reset: false, frame, seed }
    }

    /// The viewer's frame with the lights on: lit shade, temporal pass, filter (no compose).
    fn viewer(spp: u32, frame: u32, seed: u32) -> Step {
        Step { temporal: Some(TemporalSettings::default()), filter: true, ..Step::raw(lit(spp), frame, seed) }
    }
}

/// The viewer's lighting with the lights on and `spp` emitter samples per vertex.
fn lit(spp: u32) -> ShadeSettings {
    ShadeSettings { emitters: true, emitter_spp: spp, ..ShadeSettings::default() }
}

/// The per-size buffers and bindings of one scene.
struct Frame {
    scene: usize,
    w: u32,
    h: u32,
    targets: Targets,
    out: ShadeTargets,
    history: History,
    dn: DenoiseTargets,
    co: ComposeTargets,
    rb: RasterBindings,
    sb: ShadeBindings,
    sbl: ShadeBindings,
    tb: TemporalBindings,
    db: DenoiseBindings,
    cb: ComposeBindings,
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
    sky_sun: Option<[f64; 3]>,
    nights: Vec<Night>,
    raster: Raster,
    shade: Shade,
    temporal: Temporal,
    denoise: Denoise,
    compose: Compose,
    reference: Reference,
    sub: Option<Submitter>,
    /// Part 2: the relight boxes and light rows of the next frame recorded (then cleared).
    next_boxes: Vec<Relight>,
    next_rows: Vec<LightRow>,
}

impl Rig {
    fn new(dressings: &[Dressing]) -> Rig {
        Self::with_gpu(gpu(), dressings)
    }

    fn with_gpu(g: Gpu, dressings: &[Dressing]) -> Rig {
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let mut nights = Vec::new();
        for &d in dressings {
            let (world, _) = street_night(d);
            let mut pipeline = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
            pipeline.mark_all(&world);
            drain(&mut pipeline, &world);
            let snap = pipeline.current().id.raw();
            let rs = full_regions(&mut pipeline, RegionSize::Chunk);
            let scene = GpuScene::build_lit(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs, snap, true, world.materials()).unwrap();
            let params: Vec<world::MaterialParams> = world.materials().iter().map(|(_, d)| d.params).collect();
            let (tables, v) = Tables::upload(&g, &mut alloc, &mut up, &mut tl, &params, &scene.meshes, snap).unwrap();
            tl.wait(&g, v, u64::MAX).unwrap();
            let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedos(world.materials()).unwrap()).unwrap();
            tl.wait(&g, v, u64::MAX).unwrap();
            let em = scene.emitters().unwrap().expect("street_night has an emitter table");
            let (emission, emitters) = (em.table.emission.clone(), em.table.len());
            eprintln!("scene {}: {emitters} emitters", d.name());
            nights.push(Night { dressing: d, world, pipeline, scene, tables, mats, emission, emitters });
        }
        // The viewer's sky: corrected from the baked reference (S-020).
        let mut luts = SkyLuts::new(Atmosphere::default());
        let r = SkyReference::load_default(&luts.atmosphere).expect("the baked sky reference (run the sky_bake tool)");
        luts.apply_reference(&r).unwrap();
        let (sky, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &luts).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let sky_pass = SkyView::new(&g, &sky).unwrap();
        let raster = Raster::new(&g).unwrap();
        let shade = Shade::new(&g).unwrap();
        let temporal = Temporal::new(&g).unwrap();
        let denoise = Denoise::new(&g).unwrap();
        let compose = Compose::new(&g).unwrap();
        let reference = Reference::new(&g).unwrap();
        let sub = Submitter::new(&g).unwrap();
        Rig { g, alloc, tl, up, sky: Some(sky), sky_pass: Some(sky_pass), sky_sun: None, nights, raster, shade, temporal, denoise, compose, reference, sub: Some(sub), next_boxes: Vec::new(), next_rows: Vec::new() }
    }

    /// The buffers and bindings of scene `scene` at w × h.
    fn frame_set(&mut self, scene: usize, w: u32, h: u32) -> Frame {
        let g = &self.g;
        let n = &self.nights[scene];
        let em = n.scene.emitters().unwrap().unwrap();
        let targets = Targets::new(g, &mut self.alloc, vk::Extent2D { width: w, height: h }).unwrap();
        let out = ShadeTargets::new(g, &mut self.alloc, w, h).unwrap();
        let history = History::new(g, &mut self.alloc, w, h).unwrap();
        let dn = DenoiseTargets::new(g, &mut self.alloc, w, h).unwrap();
        let co = ComposeTargets::new(g, &mut self.alloc, w, h, 1).unwrap();
        let accel = n.scene.accel.as_ref().unwrap();
        let sky = &self.sky.as_ref().unwrap().view;
        let rb = self.raster.bind(g, &n.scene.meshes).unwrap();
        let sb = self.shade.bind(g, &targets, accel, &n.mats, &out, sky).unwrap();
        let sbl = self.shade.bind_lit(g, &targets, accel, &n.mats, &out, sky, &em.device).unwrap();
        let tb = self.temporal.bind(g, &targets, &n.tables.regions, &out.radiance, &history).unwrap();
        let db = self.denoise.bind(g, &out.radiance, &history, &n.mats, &dn).unwrap();
        let cb = self.compose.bind(g, &targets, &out, &em.device, &co).unwrap();
        Frame { scene, w, h, targets, out, history, dn, co, rb, sb, sbl, tb, db, cb }
    }

    /// Rebuilds the frame's bindings after an edit (new meshes, TLAS, emitter table, region table).
    fn rebind(&mut self, f: &mut Frame) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let n = &self.nights[f.scene];
        let em = n.scene.emitters().unwrap().unwrap();
        let accel = n.scene.accel.as_ref().unwrap();
        let sky = &self.sky.as_ref().unwrap().view;
        std::mem::replace(&mut f.rb, self.raster.bind(g, &n.scene.meshes).unwrap()).destroy(g);
        std::mem::replace(&mut f.sb, self.shade.bind(g, &f.targets, accel, &n.mats, &f.out, sky).unwrap()).destroy(g);
        std::mem::replace(&mut f.sbl, self.shade.bind_lit(g, &f.targets, accel, &n.mats, &f.out, sky, &em.device).unwrap()).destroy(g);
        std::mem::replace(&mut f.tb, self.temporal.bind(g, &f.targets, &n.tables.regions, &f.out.radiance, &f.history).unwrap()).destroy(g);
        std::mem::replace(&mut f.cb, self.compose.bind(g, &f.targets, &f.out, &em.device, &f.co).unwrap()).destroy(g);
    }

    /// The voxels of a box (`hi` exclusive) of scene `scene` and their materials.
    fn voxels(&self, scene: usize, lo: [i32; 3], hi: [i32; 3]) -> Voxels {
        let w = &self.nights[scene].world;
        let mut v = Vec::new();
        for x in lo[0]..hi[0] {
            for y in lo[1]..hi[1] {
                for z in lo[2]..hi[2] {
                    let c = VoxelCoord::new(x, y, z);
                    v.push((c, w.get(c)));
                }
            }
        }
        v
    }

    /// Part 2: sets voxels of the frame's scene (None removes), publishes, updates the device scene
    /// (meshes, TLAS, emitter table) and the region table, and rebinds. Returns the emitters the
    /// update changed (`GpuScene::take_changed_emitters`).
    fn edit(&mut self, f: &mut Frame, voxels: &[(VoxelCoord, Option<MaterialId>)]) -> Vec<light::emitters::Emitter> {
        let g = &self.g;
        let n = &mut self.nights[f.scene];
        let mut tx = Transaction::new();
        for &(c, m) in voxels {
            tx.set(c, m);
        }
        let applied = n.world.apply(&tx).unwrap();
        assert!(!applied.changed.is_empty(), "the edit must change something");
        n.pipeline.notify_edits(&applied.changed);
        let keys = drain(&mut n.pipeline, &n.world);
        let size = RegionSize::Chunk;
        let t = n.pipeline.acquire();
        let changed = region_meshes(&n.pipeline, &t, &affected_regions(keys.iter().copied(), size), size).unwrap();
        n.pipeline.release(t).unwrap();
        g.wait_idle().unwrap();
        n.scene.update(g, &mut self.alloc, &mut self.up, &mut self.tl, changed, n.pipeline.current().id.raw()).unwrap();
        g.wait_idle().unwrap();
        let (old, _) = n.tables.replace_regions(g, &mut self.alloc, &mut self.up, &mut self.tl, &n.scene.region_rows()).unwrap();
        g.wait_idle().unwrap();
        self.alloc.free(g, old);
        let done = self.tl.completed(g).unwrap();
        n.scene.collect(g, &mut self.alloc, done);
        let em = n.scene.take_changed_emitters();
        self.rebind(f);
        em
    }

    fn free_frame(&mut self, f: Frame) {
        let g = &self.g;
        g.wait_idle().unwrap();
        f.rb.destroy(g);
        f.sb.destroy(g);
        f.sbl.destroy(g);
        f.tb.destroy(g);
        f.db.destroy(g);
        f.cb.destroy(g);
        f.co.free(g, &mut self.alloc);
        f.dn.free(g, &mut self.alloc);
        f.history.free(g, &mut self.alloc);
        f.out.free(g, &mut self.alloc);
        f.targets.free(g, &mut self.alloc);
    }

    /// A fresh history (the next frame is a first-frame reset).
    fn new_history(&mut self, f: &mut Frame) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let old = std::mem::replace(&mut f.history, History::new(g, &mut self.alloc, f.w, f.h).unwrap());
        old.free(g, &mut self.alloc);
        let n = &self.nights[f.scene];
        let tb = self.temporal.bind(g, &f.targets, &n.tables.regions, &f.out.radiance, &f.history).unwrap();
        let db = self.denoise.bind(g, &f.out.radiance, &f.history, &n.mats, &f.dn).unwrap();
        std::mem::replace(&mut f.tb, tb).destroy(g);
        std::mem::replace(&mut f.db, db).destroy(g);
    }

    fn record(&mut self, cmd: vk::CommandBuffer, f: &mut Frame, cam: &Camera, light: &Lighting, st: &Step) {
        if self.sky_sun != Some(light.sun_dir) {
            self.sky_pass.as_ref().unwrap().record(&self.g, cmd, light.sun_dir);
            self.sky_sun = Some(light.sun_dir);
        }
        let g = &self.g;
        let n = &self.nights[f.scene];
        self.raster.record(g, cmd, &f.targets, &n.scene.meshes, &f.rb, cam, Faults::default());
        targets_to_read(g, cmd, &f.targets);
        let lit_faults = st.faults.emitter_after_continuation || st.faults.no_bounce_emitters || st.faults.emission_in_shade || st.faults.emitter_sum;
        let sb = if st.set.emitters || lit_faults { &f.sbl } else { &f.sb };
        self.shade.record(g, cmd, sb, &f.out, &shade::Params::new(cam, light, st.set, st.faults, st.frame, st.seed));
        if st.compose == ComposeAt::BeforeTemporal {
            self.compose.record(g, cmd, &f.cb, &f.co, 0, true, st.floor, st.compose_faults);
        }
        if let Some(ts) = st.temporal {
            f.history.set_lights(st.lights);
            f.history.set_light_rows(&std::mem::take(&mut self.next_rows));
            let boxes = std::mem::take(&mut self.next_boxes);
            self.temporal.record(g, cmd, &f.tb, &mut f.history, cam, light.sun_dir, ts, TemporalFaults::default(), st.reset, &boxes);
            if st.filter {
                self.denoise.record(g, cmd, &f.db, &f.history, &f.dn, DenoiseSettings::default(), DenoiseFaults::default());
            }
        }
        if st.compose == ComposeAt::After {
            self.compose.record(g, cmd, &f.cb, &f.co, 0, true, st.floor, st.compose_faults);
        }
        full_barrier(g, cmd);
    }

    /// Frames `frames` of `st` (frame index and, from `st.frame`, the camera per frame), `per_submit`
    /// per submission, waiting for the last.
    fn run(&mut self, f: &mut Frame, cam: &dyn Fn(u32) -> Camera, light: &Lighting, st: Step, frames: std::ops::RangeInclusive<u32>, per_submit: u32) {
        let mut sub = self.sub.take().unwrap();
        let mut k = *frames.start();
        while k <= *frames.end() {
            let cmd = sub.begin(&self.g, &self.tl).unwrap();
            let end = (k + per_submit - 1).min(*frames.end());
            for i in k..=end {
                self.record(cmd, f, &cam(i), light, &Step { frame: i, ..st });
            }
            let v = sub.submit(&self.g, &mut self.tl, cmd, &[]).unwrap();
            self.tl.wait(&self.g, v, u64::MAX).unwrap();
            k = end + 1;
        }
        self.sub = Some(sub);
    }

    fn one(&mut self, f: &mut Frame, cam: &Camera, light: &Lighting, st: Step) {
        let c = *cam;
        self.run(f, &move |_| c, light, st, st.frame..=st.frame, 1);
    }

    fn radiance(&mut self, f: &Frame) -> Vec<[f32; 4]> {
        f.out.read(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    fn history_colour(&mut self, f: &Frame) -> Vec<[f32; 4]> {
        f.history.read_colour(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    fn guides(&mut self, f: &Frame) -> Vec<[u32; 4]> {
        f.history.read_guides(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    fn state(&mut self, f: &Frame) -> Vec<(u32, u32)> {
        f.history.read(&self.g, &mut self.alloc, &mut self.tl).unwrap().0
    }

    /// Reference samples `frames` of scene `scene` (emitter table bound).
    #[allow(clippy::too_many_arguments)]
    fn reference(&mut self, scene: usize, cam: &Camera, light: &Lighting, s: &Settings, seed: u32, frames: std::ops::Range<u32>, per_submit: u32) -> Accum {
        let out = RefAccum::new(&self.g, &mut self.alloc, cam.width, cam.height).unwrap();
        let n = &self.nights[scene];
        let b = self.reference.bind_lit(&self.g, n.scene.accel.as_ref().unwrap(), &n.mats, &n.scene.emitters().unwrap().unwrap().device, &out).unwrap();
        let count = frames.len() as u32;
        let first = frames.start;
        self.reference.accumulate(&self.g, &mut self.tl, &b, &out, cam, light, s, RefFaults::default(), seed, frames, first, per_submit).unwrap();
        let img = self.reference.read(&self.g, &mut self.alloc, &mut self.tl, &out, count).unwrap();
        assert_eq!(img.bad_samples, 0);
        self.g.wait_idle().unwrap();
        b.destroy(&self.g);
        out.free(&self.g, &mut self.alloc);
        img.accum
    }

    fn finish(mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let (errors, warnings) = g.validation_counts();
        eprintln!("validation: {errors} errors, {warnings} warnings");
        assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
        self.sub.take().unwrap().destroy(g);
        self.sky_pass.take().unwrap().destroy(g);
        self.sky.take().unwrap().free(g, &mut self.alloc);
        for n in self.nights.drain(..) {
            n.mats.free(g, &mut self.alloc);
            n.tables.free_now(g, &mut self.alloc);
            n.scene.free_now(g, &mut self.alloc);
        }
        self.raster.destroy(g);
        self.shade.destroy(g);
        self.temporal.destroy(g);
        self.denoise.destroy(g);
        self.compose.destroy(g);
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

/// The reference settings of the real-time transport (one bounce, emission off).
fn ref_settings(sun: bool, sky: bool, bounce: bool) -> Settings {
    Settings { sun, sky, max_bounces: u32::from(bounce), emitters_direct: true, emitters_indirect: true, ..Settings::default() }
}

// ---- G10: exact per pixel ----

struct Agreement {
    mismatch: usize,
    bad: usize,
}

fn agreement(rt: &[[f32; 4]], rf: &Accum, tol: f64) -> Agreement {
    let mut a = Agreement { mismatch: 0, bad: 0 };
    for (x, y) in rt.iter().zip(&rf.sum) {
        a.bad += usize::from(x[3] != 0.0);
        let same = (0..3).all(|k| if y[k] == 0.0 { x[k] == 0.0 } else { (x[k] as f64 / y[k] - 1.0).abs() <= tol });
        a.mismatch += usize::from(!same);
    }
    a
}

/// G10: frame f of the lit shade pass equals the reference's one-bounce sample f per pixel
/// (relative 1e-3 on >= 99.9% of pixels), in five settings on three cameras; the controls (emitters
/// off, three planted faults) are caught and a different frame differs.
#[test]
fn lit_shade_equals_the_reference_sample_per_pixel() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let (w, h) = (480, 270);
    let n = (w * h) as usize;
    let frame = 29;
    let mut f = rig.frame_set(0, w, h);
    let none = ShadeSettings { sun: false, sky: false, point_sun: false, uniform_sky: false, bounce: true, emitters: true, emitter_spp: 1 };
    let arms: Vec<(&str, f64, ShadeSettings, Settings)> = vec![
        ("emitters only", NIGHT, none, ref_settings(false, false, true)),
        ("uniform sky", NIGHT, ShadeSettings { uniform_sky: true, ..none }, Settings { uniform_sky: true, ..ref_settings(false, false, true) }),
        ("point sun 12 h", 12.0, ShadeSettings { sun: true, point_sun: true, ..none }, Settings { point_sun: true, ..ref_settings(true, false, true) }),
        ("disk sun 12 h", 12.0, ShadeSettings { sun: true, ..none }, ref_settings(true, false, true)),
        ("bounce off", NIGHT, ShadeSettings { bounce: false, ..none }, ref_settings(false, false, false)),
    ];
    let controls: [(&str, ShadeFaults, bool); 4] = [
        ("emitters off", ShadeFaults::default(), false),
        ("emitter_after_continuation", ShadeFaults { emitter_after_continuation: true, ..ShadeFaults::default() }, true),
        ("no_bounce_emitters", ShadeFaults { no_bounce_emitters: true, ..ShadeFaults::default() }, true),
        ("emission_in_shade", ShadeFaults { emission_in_shade: true, ..ShadeFaults::default() }, true),
    ];
    let mut failed = Vec::new();
    // Per kind: (arms, arms failed, total mismatches).
    let mut tally: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (cname, cam) in cameras(w, h, true) {
        for (label, hour, set, rs) in &arms {
            let light = lighting(*hour);
            let rf = rig.reference(0, &cam, &light, rs, SEED, frame..frame + 1, 1);
            rig.one(&mut f, &cam, &light, Step::raw(*set, frame, SEED));
            let rt = rig.radiance(&f);
            rig.one(&mut f, &cam, &light, Step::raw(ShadeSettings { emitters: false, ..*set }, frame, SEED));
            let off = rig.radiance(&f);
            let changed = rt.iter().zip(&off).filter(|(a, b)| a[..3] != b[..3]).count();
            let a = agreement(&rt, &rf, 1e-3);
            let pass = a.mismatch * 1000 <= n && a.bad == 0 && changed >= 1000;
            eprintln!("G10 {cname} {label}: {} mismatched ({:.3}%), bad {}, {changed} px changed by the emitters", a.mismatch, 100.0 * a.mismatch as f64 / n as f64, a.bad);
            let t = tally.entry("ok").or_default();
            t.0 += 1;
            t.1 += usize::from(!pass);
            t.2 += a.mismatch;
            if !pass {
                failed.push(format!("{cname}/{label}"));
            }
            for (kind, faults, emitters) in controls {
                let x = if emitters {
                    rig.one(&mut f, &cam, &light, Step { faults, ..Step::raw(*set, frame, SEED) });
                    rig.radiance(&f)
                } else {
                    off.clone()
                };
                let c = agreement(&x, &rf, 1e-3);
                eprintln!("    control {kind}: {} mismatched ({:.2}%)", c.mismatch, 100.0 * c.mismatch as f64 / n as f64);
                let t = tally.entry(kind).or_default();
                t.0 += 1;
                t.1 += usize::from(c.mismatch * 1000 > n);
                t.2 += c.mismatch;
            }
        }
    }
    let ok_total = tally["ok"].2;
    for (k, (arms, failing, total)) in &tally {
        eprintln!("G10 tally {k}: {failing} of {arms} arms fail, {total} mismatched pixels in total");
        if *k != "ok" && (failing * 2 < *arms || *total < 10 * ok_total.max(1)) {
            failed.push(format!("control {k} not caught: {failing} of {arms} arms, {total} px (ok arms {ok_total} px)"));
        }
    }
    let (_, cam) = cameras(w, h, false).remove(0);
    let light = lighting(NIGHT);
    let rf = rig.reference(0, &cam, &light, &arms[0].3, SEED, frame..frame + 1, 1);
    rig.one(&mut f, &cam, &light, Step::raw(arms[0].2, frame + 1, SEED));
    let rt = rig.radiance(&f);
    let c = agreement(&rt, &rf, 1e-3).mismatch;
    eprintln!("G10 control: frame {} against reference frame {frame}: {c} mismatched", frame + 1);
    if c == 0 {
        failed.push("frame control: different frames agree".into());
    }
    rig.free_frame(f);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---- G11: convergence ----

struct Stat {
    mean_z: [f64; 3],
    frac_over_4: f64,
    rel_mean: [f64; 3],
}

/// The 3A metric: image-mean z per channel and the fraction of pixel-channels with |z| > 4.
fn compare(a: &Accum, b: &Accum) -> Stat {
    let n = a.sum.len();
    let (mut over, mut total) = (0usize, 0usize);
    let mut mean_z = [0.0; 3];
    let mut rel_mean = [0.0; 3];
    for k in 0..3 {
        let (mut ma, mut mb, mut va, mut vb) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let (x, y) = (a.mean(i)[k], b.mean(i)[k]);
            let (sx, sy) = (a.std_error(i)[k], b.std_error(i)[k]);
            ma += x;
            mb += y;
            va += sx * sx;
            vb += sy * sy;
            let se = (sx * sx + sy * sy).sqrt();
            if se > 0.0 {
                total += 1;
                if ((x - y) / se).abs() > 4.0 {
                    over += 1;
                }
            } else if x != y {
                total += 1;
                over += 1;
            }
        }
        mean_z[k] = (ma - mb) / (va + vb).sqrt();
        rel_mean[k] = ma / mb - 1.0;
    }
    Stat { mean_z, frac_over_4: over as f64 / total.max(1) as f64, rel_mean }
}

/// The raw shade output over `frames`, summed per pixel on the host.
fn shade_accum(rig: &mut Rig, f: &mut Frame, cam: &Camera, light: &Lighting, set: ShadeSettings, faults: ShadeFaults, frames: u32) -> Accum {
    let n = (cam.width * cam.height) as usize;
    let mut a = Accum { width: cam.width, height: cam.height, samples: frames, sum: vec![[0.0; 3]; n], sum_sq: vec![[0.0; 3]; n] };
    for k in 0..frames {
        rig.one(f, cam, light, Step { faults, ..Step::raw(set, k, SEED) });
        for (i, p) in rig.radiance(f).iter().enumerate() {
            assert_eq!(p[3], 0.0, "bad surface id");
            for (c, x) in p[..3].iter().enumerate() {
                let x = *x as f64;
                a.sum[i][c] += x;
                a.sum_sq[i][c] += x * x;
            }
        }
    }
    a
}

/// G11: the raw accumulation with 1, 2 and 4 emitter samples converges to the reference (4A G4's
/// metric); the planted `emitter_sum` is caught at 2 and 4.
#[test]
fn lit_shade_converges_to_the_reference() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let (w, h) = (240, 135);
    let frames = 4096;
    let mut f = rig.frame_set(0, w, h);
    let light = lighting(NIGHT);
    let set = |spp: u32| ShadeSettings { sun: false, sky: false, emitters: true, emitter_spp: spp, ..ShadeSettings::default() };
    let rs = ref_settings(false, false, true);
    let mut failed = Vec::new();
    for (cname, cam) in cameras(w, h, false) {
        let r = rig.reference(0, &cam, &light, &rs, SEED, 1_000_000..1_000_000 + frames, 64);
        let null_img = rig.reference(0, &cam, &light, &rs, SEED, 3_000_000..3_000_000 + frames, 64);
        let null = compare(&r, &null_img).frac_over_4;
        let limit = (1.5 * null + 0.002).max(0.01);
        eprintln!("G11 null {cname}: |z|>4 {:.3}%, limit {:.3}%", null * 100.0, limit * 100.0);
        for spp in [1u32, 2, 4] {
            for fault in [false, true] {
                if fault && spp == 1 {
                    continue;
                }
                let faults = ShadeFaults { emitter_sum: fault, ..ShadeFaults::default() };
                let a = shade_accum(&mut rig, &mut f, &cam, &light, set(spp), faults, frames);
                let st = compare(&a, &r);
                let pass = st.mean_z.iter().all(|z| z.abs() < 4.0) && st.frac_over_4 <= limit;
                eprintln!(
                    "G11 {cname} s={spp}{}: rel mean {:?}, z {:?}, |z|>4 {:.3}% ({})",
                    if fault { " emitter_sum" } else { "" },
                    st.rel_mean.map(|x| (x * 1e4).round() / 1e4),
                    st.mean_z.map(|x| (x * 100.0).round() / 100.0),
                    st.frac_over_4 * 100.0,
                    if pass { "pass" } else { "FAIL" }
                );
                match (fault, pass) {
                    (false, false) => failed.push(format!("{cname} s={spp}")),
                    (true, true) => failed.push(format!("control emitter_sum not caught {cname} s={spp}")),
                    _ => {}
                }
            }
        }
    }
    rig.free_frame(f);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---- G13: emission and the light jump ----

fn surface_mask(guides: &[[u32; 4]]) -> Vec<bool> {
    guides.iter().map(|g| g[0] & 7 != FACE_NONE).collect()
}

/// G13: compose adds exactly the table's emission per surface pixel and nothing else; the history
/// never holds emission (the planted `compose_before_temporal` is caught); a lights change resets
/// every surface pixel and unchanged lights accept them.
#[test]
fn emission_after_reconstruction_and_the_light_jump() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let (w, h) = (480, 270);
    let light = lighting(NIGHT);
    let mut f = rig.frame_set(0, w, h);
    let mut failed = Vec::new();
    for (cname, cam) in cameras(w, h, false) {
        // Emission: the viewer's frame, read before and after compose.
        rig.new_history(&mut f);
        rig.one(&mut f, &cam, &light, Step::viewer(1, 1, SEED));
        let reflected = rig.radiance(&f);
        let guides = rig.guides(&f);
        // Compose alone on that frame's shown radiance.
        let mut sub = rig.sub.take().unwrap();
        let cmd = sub.begin(&rig.g, &rig.tl).unwrap();
        rig.compose.record(&rig.g, cmd, &f.cb, &f.co, 0, true, 0.0, ComposeFaults::default());
        let v = sub.submit(&rig.g, &mut rig.tl, cmd, &[]).unwrap();
        rig.tl.wait(&rig.g, v, u64::MAX).unwrap();
        rig.sub = Some(sub);
        let shown = rig.radiance(&f);
        let em = rig.nights[0].emission.clone();
        let (mut wrong, mut emitting, mut w_changed) = (0usize, 0usize, 0usize);
        for i in 0..shown.len() {
            let g = guides[i];
            let e = if g[0] & 7 == FACE_NONE { [0.0; 3] } else { em[(g[0] >> 3) as usize] };
            emitting += usize::from(lum(e) > 0.0);
            let ok = (0..3).all(|c| {
                let want = reflected[i][c] + e[c] as f32;
                (shown[i][c] - want).abs() <= 1e-6 * want.abs().max(1e-30)
            });
            wrong += usize::from(!ok);
            w_changed += usize::from(shown[i][3].to_bits() != reflected[i][3].to_bits());
        }
        eprintln!("G13 {cname}: emission wrong on {wrong} px ({emitting} emitting px), w changed on {w_changed}");
        if wrong > 0 || w_changed > 0 || emitting < 100 {
            failed.push(format!("{cname} emission"));
        }
        // The history holds no emission: 8 frames with compose after, without it, and before temporal.
        let mut hist = BTreeMap::new();
        for (name, at) in [("with compose", ComposeAt::After), ("without", ComposeAt::Off), ("compose_before_temporal", ComposeAt::BeforeTemporal)] {
            rig.new_history(&mut f);
            let c = cam;
            rig.run(&mut f, &move |_| c, &light, Step { compose: at, ..Step::viewer(1, 1, SEED) }, 1..=8, 8);
            hist.insert(name, rig.history_colour(&f).iter().map(|p| p.map(f32::to_bits)).collect::<Vec<_>>());
        }
        let same = hist["with compose"] == hist["without"];
        let fault = hist["compose_before_temporal"] != hist["without"];
        eprintln!("G13 {cname}: history with compose == without: {same}; compose_before_temporal differs: {fault}");
        if !same || !fault {
            failed.push(format!("{cname} history"));
        }
        // The light jump.
        rig.new_history(&mut f);
        let c = cam;
        rig.run(&mut f, &move |_| c, &light, Step::viewer(1, 1, SEED), 1..=3, 3);
        let m = surface_mask(&rig.guides(&f));
        let count = |st: &[(u32, u32)], r: u32| st.iter().zip(&m).filter(|(s, m)| **m && s.1 == r).count();
        let surf = m.iter().filter(|&&b| b).count();
        let mut jumps = Vec::new();
        for (k, lights) in [(4u32, false), (5, false), (6, true), (7, true)] {
            rig.one(&mut f, &cam, &light, Step { lights, set: ShadeSettings { emitters: lights, ..lit(1) }, ..Step::viewer(1, k, SEED) });
            let st = rig.state(&f);
            jumps.push((k, lights, count(&st, reason::RESET), count(&st, reason::ACCEPTED)));
        }
        eprintln!("G13 {cname}: {surf} surface px; (frame, lights, reset, accepted): {jumps:?}");
        let want = |i: usize, reset: bool| if reset { jumps[i].2 == surf } else { jumps[i].3 == surf };
        if !(want(0, true) && want(1, false) && want(2, true) && want(3, false)) {
            failed.push(format!("{cname} light jump"));
        }
    }
    rig.free_frame(f);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---- G14: the meter ----

/// G14: the meter's sums equal the host's over the shown image at 1920×1080 and 1001×563; the
/// planted `meter_without_emission` is caught.
#[test]
fn meter_matches_the_host() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let light = lighting(NIGHT);
    let mut failed = Vec::new();
    for (w, h) in [(1920u32, 1080u32), (1001, 563)] {
        let mut f = rig.frame_set(0, w, h);
        let (_, cam) = cameras(w, h, false).remove(0);
        // A first reading gives the floor, so the floor rule is exercised.
        rig.one(&mut f, &cam, &light, Step { compose: ComposeAt::After, ..Step::viewer(1, 1, SEED) });
        let floor = f.co.reading(0).next_floor();
        for fault in [false, true] {
            rig.new_history(&mut f);
            rig.one(&mut f, &cam, &light, Step { compose: ComposeAt::After, floor, compose_faults: ComposeFaults { meter_without_emission: fault }, ..Step::viewer(1, 1, SEED) });
            let gpu = f.co.reading(0);
            let y: Vec<f64> = rig.radiance(&f).iter().map(lum4).collect();
            let host = Meter::of(&y, floor);
            let rel = |a: f64, b: f64| ((a - b) / b).abs();
            let ok = rel(gpu.sum_y, host.sum_y) <= 1e-4 && rel(gpu.sum_ln, host.sum_ln) <= 1e-4 && (gpu.count as f64 - host.count as f64).abs() <= 1e-4 * y.len() as f64;
            eprintln!("G14 {w}x{h}{}: floor {floor:.3e}; sum Y {:.6e} / {:.6e}, sum ln Y {:.6e} / {:.6e}, count {} / {} ({})", if fault { " meter_without_emission" } else { "" }, gpu.sum_y, host.sum_y, gpu.sum_ln, host.sum_ln, gpu.count, host.count, if ok { "equal" } else { "DIFFER" });
            if ok == fault {
                failed.push(format!("{w}x{h} fault {fault}"));
            }
        }
        rig.free_frame(f);
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---- Q: quality at 1080p ----

fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("NE_4B_DIR").map(PathBuf::from)
}

/// A comparison image: per pixel the mean and its standard error (0 for converged real-time images).
struct Target {
    mean: Vec<[f64; 3]>,
    se: Vec<[f64; 3]>,
}

impl Target {
    fn save(&self, key: &str, header: &str) {
        let Some(dir) = cache_dir() else { return };
        let mut b = header.as_bytes().to_vec();
        for (m, s) in self.mean.iter().zip(&self.se) {
            for v in m.iter().chain(s) {
                b.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{key}.bin")), b).unwrap();
    }

    fn load(key: &str, header: &str) -> Option<Target> {
        let b = std::fs::read(cache_dir()?.join(format!("{key}.bin"))).ok()?;
        let body = b.strip_prefix(header.as_bytes())?;
        if body.len() % 24 != 0 {
            return None;
        }
        let f = |i: usize| f32::from_le_bytes(body[4 * i..4 * i + 4].try_into().unwrap()) as f64;
        let n = body.len() / 24;
        Some(Target { mean: (0..n).map(|p| [f(6 * p), f(6 * p + 1), f(6 * p + 2)]).collect(), se: (0..n).map(|p| [f(6 * p + 3), f(6 * p + 4), f(6 * p + 5)]).collect() })
    }
}

/// The one-bounce reference (reflected light), cached.
fn reference_1080(rig: &mut Rig, key: &str, cam: &Camera, light: &Lighting) -> Target {
    let header = format!("NE4BREF v{CACHE_VERSION} {key} {}x{} spp {REF_SPP} seed {REF_SEED} sun {:?}\n", cam.width, cam.height, light.sun_dir);
    if let Some(t) = Target::load(key, &header) {
        eprintln!("reference {key}: cached");
        return t;
    }
    let t0 = std::time::Instant::now();
    let a = rig.reference(0, cam, light, &ref_settings(true, true, true), REF_SEED, 0..REF_SPP, REF_PER_SUBMIT);
    let t = Target { mean: (0..a.sum.len()).map(|i| a.mean(i)).collect(), se: (0..a.sum.len()).map(|i| a.std_error(i)).collect() };
    eprintln!("reference {key}: {REF_SPP} spp in {:.1} s", t0.elapsed().as_secs_f64());
    t.save(key, &header);
    t
}

/// The real-time estimator converged at a still camera (`TARGET_FRAMES` frames, s = `TARGET_SPP`,
/// no filter), cached.
fn converged_1080(rig: &mut Rig, f: &mut Frame, key: &str, cam: &Camera, light: &Lighting, bounce: bool) -> Target {
    let header = format!("NE4BCONV v{CACHE_VERSION} {key} {}x{} frames {TARGET_FRAMES} spp {TARGET_SPP} bounce {bounce} eye {:?} fwd {:?} sun {:?}\n", cam.width, cam.height, cam.eye, cam.forward, light.sun_dir);
    if let Some(t) = Target::load(key, &header) {
        eprintln!("converged {key}: cached");
        return t;
    }
    let t0 = std::time::Instant::now();
    rig.new_history(f);
    let set = ShadeSettings { bounce, ..lit(TARGET_SPP) };
    let st = Step { set, temporal: Some(TemporalSettings { max_age: TARGET_FRAMES, ..TemporalSettings::default() }), filter: false, ..Step::raw(set, 0, 0x4BCC) };
    let c = *cam;
    rig.run(f, &move |_| c, light, st, 1..=TARGET_FRAMES, 16);
    let h = rig.history_colour(f);
    let t = Target { mean: h.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect(), se: vec![[0.0; 3]; h.len()] };
    eprintln!("converged {key}: {TARGET_FRAMES} frames in {:.1} s", t0.elapsed().as_secs_f64());
    t.save(key, &header);
    t
}

/// Relative MSE (per channel, ε = (0.1 × the channel's mean)²) over the masked pixels: the 3E metric.
fn rel_mse(x: &[[f32; 4]], t: &[[f64; 3]], mask: &[bool]) -> f64 {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| t[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-30));
    idx.iter().map(|&i| (0..3).map(|c| (x[i][c] as f64 - t[i][c]).powi(2) / (t[i][c].powi(2) + eps[c])).sum::<f64>()).sum::<f64>() / (3.0 * n)
}

/// Mean luminance over the masked pixels.
fn mean_lum(x: &[[f32; 4]], mask: &[bool]) -> f64 {
    let (s, n) = x.iter().zip(mask).filter(|(_, m)| **m).fold((0.0, 0usize), |(s, n), (p, _)| (s + lum4(p), n + 1));
    s / n.max(1) as f64
}

fn mean_lum_t(t: &[[f64; 3]], mask: &[bool]) -> f64 {
    let (s, n) = t.iter().zip(mask).filter(|(_, m)| **m).fold((0.0, 0usize), |(s, n), (p, _)| (s + lum(*p), n + 1));
    s / n.max(1) as f64
}

/// The display image (reflected light + the table's emission, `exposure`, the ACES fit, sRGB) as a
/// PPM in `NE_4B_DIR`.
fn write_display(name: &str, w: u32, h: u32, px: &dyn Fn(usize) -> [f64; 3], exposure: f64) {
    let Some(dir) = cache_dir() else { return };
    let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
    let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
    for i in 0..(w * h) as usize {
        for c in px(i) {
            bytes.push((srgb(aces(c.max(0.0) * exposure)) * 255.0 + 0.5) as u8);
        }
    }
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{name}.ppm")), bytes).unwrap();
}

/// Per pixel: the table's emitted radiance of its material (0 on sky pixels).
fn emission_image(em: &[[f64; 3]], guides: &[[u32; 4]]) -> Vec<[f64; 3]> {
    guides.iter().map(|g| if g[0] & 7 == FACE_NONE { [0.0; 3] } else { em[(g[0] >> 3) as usize] }).collect()
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn f64of(p: &[f32; 4]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

/// Per seed and age: (filtered, raw) images of a still camera from a fresh history.
struct StillRun {
    filtered: BTreeMap<u32, Vec<[f32; 4]>>,
    raw: BTreeMap<u32, Vec<[f32; 4]>>,
}

/// Q1, Q2 (s = 1) and the still images for Q4, and M2 (s = 2, 4): both cameras at blue hour and
/// night, 8 seed sequences, ages 1, 4, 16, 64. Against the reference at night and the converged
/// real-time image at blue hour (per pixel); energy against the reference at both.
#[test]
fn stills_at_blue_hour_and_night() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let mut f = rig.frame_set(0, W, H);
    let em = rig.nights[0].emission.clone();
    let mut failed = Vec::new();
    let filt_ages = [1u32, 4, 16, 64];
    let raw_ages = [1u32, 4, 8, 16, 64];
    for (time, hour) in STILL_TIMES {
        let light = lighting(hour);
        for (cname, cam) in cameras(W, H, false) {
            let key = format!("{cname}_{time}");
            let r = reference_1080(&mut rig, &format!("ref_{key}"), &cam, &light);
            let per_pixel = if time == "night" { r.mean.clone() } else { converged_1080(&mut rig, &mut f, &format!("conv_{key}"), &cam, &light, true).mean };
            for spp in [1u32, 2, 4] {
                // Per age: sums over seeds of (filtered bias vs reference, filtered vs raw, rel MSE filtered, rel MSE raw).
                let mut q1: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
                let mut q2f: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
                let mut q2r: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
                let mut mask = Vec::new();
                for k in 0..Q_SEEDS {
                    rig.new_history(&mut f);
                    let mut run = StillRun { filtered: BTreeMap::new(), raw: BTreeMap::new() };
                    for age in 1..=64u32 {
                        rig.one(&mut f, &cam, &light, Step::viewer(spp, age, Q_SEED + k));
                        if filt_ages.contains(&age) {
                            run.filtered.insert(age, rig.radiance(&f));
                        }
                        if raw_ages.contains(&age) {
                            run.raw.insert(age, rig.history_colour(&f));
                        }
                        if age == 1 && mask.is_empty() {
                            mask = surface_mask(&rig.guides(&f));
                        }
                    }
                    let lr = mean_lum_t(&r.mean, &mask);
                    for &age in &filt_ages {
                        let (fl, rl) = (mean_lum(&run.filtered[&age], &mask), mean_lum(&run.raw[&age], &mask));
                        let e = q1.entry(age).or_default();
                        e.0.push(fl / lr - 1.0);
                        e.1.push(fl / rl - 1.0);
                        q2f.entry(age).or_default().push(rel_mse(&run.filtered[&age], &per_pixel, &mask));
                    }
                    for &age in &raw_ages {
                        q2r.entry(age).or_default().push(rel_mse(&run.raw[&age], &per_pixel, &mask));
                    }
                    if k == 0 && spp == 1 {
                        let guides = rig.guides(&f);
                        let emi = emission_image(&em, &guides);
                        let cmp: Vec<[f64; 3]> = per_pixel.iter().zip(&emi).map(|(a, b)| add(*a, *b)).collect();
                        let exposure = metric_exposure(&cmp.iter().map(|c| lum(*c)).collect::<Vec<_>>()).unwrap_or(1.0);
                        write_display(&format!("still_{key}_ref"), W, H, &|i| cmp[i], exposure);
                        for age in [1u32, 16, 64] {
                            let (fi, ra) = (&run.filtered[&age], &run.raw[&age]);
                            write_display(&format!("still_{key}_filt_{age}"), W, H, &|i| add(f64of(&fi[i]), emi[i]), exposure);
                            write_display(&format!("still_{key}_raw_{age}"), W, H, &|i| add(f64of(&ra[i]), emi[i]), exposure);
                        }
                    }
                }
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                let spread = |v: &[f64]| (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
                for (age, gain) in [(1u32, 8u32), (4, 4), (16, 4), (64, 1)] {
                    let (b, fr) = (&q1[&age].0, &q1[&age].1);
                    let (ef, er, same) = (mean(&q2f[&age]), mean(&q2r[&(age * gain)]), mean(&q2r[&age]));
                    let ok1 = mean(b).abs() <= 0.02 && mean(fr).abs() <= 0.01;
                    let ok2 = ef <= er;
                    eprintln!(
                        "{key} s={spp} age {age:2}: bias vs reference {:+.4} (seeds {:+.4}..{:+.4}), filtered vs raw {:+.4} | rel_mse filtered {:.4} (seeds {:.4}..{:.4}), raw at age {} {:.4}, raw at age {age} {:.4} | Q1 {} Q2 {}",
                        mean(b),
                        spread(b).0,
                        spread(b).1,
                        mean(fr),
                        ef,
                        spread(&q2f[&age]).0,
                        spread(&q2f[&age]).1,
                        age * gain,
                        er,
                        same,
                        if ok1 { "pass" } else { "FAIL" },
                        if ok2 { "pass" } else { "FAIL" }
                    );
                    if spp == 1 && !ok1 {
                        failed.push(format!("Q1 {key} age {age}"));
                    }
                    if spp == 1 && !ok2 {
                        failed.push(format!("Q2 {key} age {age}"));
                    }
                }
            }
        }
    }
    rig.free_frame(f);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Q3 and the motion images for Q4: the 3E D4 path at night, frames 8, 16, 32, 8 seeds: filtered
/// within 1% of raw in the same frame and within 2% of the converged real-time image at that pose.
#[test]
fn motion_at_night() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let mut f = rig.frame_set(0, W, H);
    let em = rig.nights[0].emission.clone();
    let light = lighting(NIGHT);
    let targets: Vec<Target> = MOTION_FRAMES.iter().map(|&k| converged_1080(&mut rig, &mut f, &format!("conv_motion_night_{k}"), &motion_camera(k, W, H), &light, true)).collect();
    let mut failed = Vec::new();
    for spp in [1u32, 2, 4] {
        // Per motion frame: per seed (filtered vs raw, filtered vs target).
        let mut acc: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for k in 0..Q_SEEDS {
            rig.new_history(&mut f);
            for i in 1..=*MOTION_FRAMES.last().unwrap() {
                rig.one(&mut f, &motion_camera(i, W, H), &light, Step::viewer(spp, i, Q_SEED + k));
                if let Some(j) = MOTION_FRAMES.iter().position(|&m| m == i) {
                    let (fi, ra, g) = (rig.radiance(&f), rig.history_colour(&f), rig.guides(&f));
                    let m = surface_mask(&g);
                    let (fl, rl, tl) = (mean_lum(&fi, &m), mean_lum(&ra, &m), mean_lum_t(&targets[j].mean, &m));
                    let e = acc.entry(i).or_default();
                    e.0.push(fl / rl - 1.0);
                    e.1.push(fl / tl - 1.0);
                    if k == 0 && spp == 1 {
                        let emi = emission_image(&em, &g);
                        let cmp: Vec<[f64; 3]> = targets[j].mean.iter().zip(&emi).map(|(a, b)| add(*a, *b)).collect();
                        let exposure = metric_exposure(&cmp.iter().map(|c| lum(*c)).collect::<Vec<_>>()).unwrap_or(1.0);
                        let key = format!("motion_night_{i}");
                        write_display(&format!("{key}_ref"), W, H, &|p| cmp[p], exposure);
                        write_display(&format!("{key}_filt"), W, H, &|p| add(f64of(&fi[p]), emi[p]), exposure);
                        write_display(&format!("{key}_raw"), W, H, &|p| add(f64of(&ra[p]), emi[p]), exposure);
                    }
                }
            }
        }
        for (i, (fr, ft)) in &acc {
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let ok = mean(fr).abs() <= 0.01 && mean(ft).abs() <= 0.02;
            eprintln!("motion night s={spp} frame {i:2}: filtered vs raw {:+.4}, vs converged {:+.4} (seeds {:+.4}..{:+.4}) | Q3 {}", mean(fr), mean(ft), ft.iter().cloned().fold(f64::INFINITY, f64::min), ft.iter().cloned().fold(f64::NEG_INFINITY, f64::max), if ok { "pass" } else { "FAIL" });
            if spp == 1 && !ok {
                failed.push(format!("Q3 frame {i}"));
            }
        }
    }
    rig.free_frame(f);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// M3 (b), data: at night, the filtered relative MSE at ages 16 and 64 with the bounce off (against
/// its own converged image) beside the bounce on (against its converged image), 8 seeds.
#[test]
fn indirect_noise_at_night() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let mut f = rig.frame_set(0, W, H);
    let light = lighting(NIGHT);
    for (cname, cam) in cameras(W, H, false) {
        for bounce in [true, false] {
            let t = converged_1080(&mut rig, &mut f, &format!("conv_{cname}_night_bounce_{bounce}"), &cam, &light, bounce);
            let mut e: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
            for k in 0..Q_SEEDS {
                rig.new_history(&mut f);
                let mut mask = Vec::new();
                for age in 1..=64u32 {
                    let st = Step::viewer(1, age, Q_SEED + k);
                    rig.one(&mut f, &cam, &light, Step { set: ShadeSettings { bounce, ..st.set }, ..st });
                    if age == 1 {
                        mask = surface_mask(&rig.guides(&f));
                    }
                    if age == 16 || age == 64 {
                        let (fi, ra) = (rig.radiance(&f), rig.history_colour(&f));
                        let x = e.entry(age).or_default();
                        x.0.push(rel_mse(&fi, &t.mean, &mask));
                        x.1.push(rel_mse(&ra, &t.mean, &mask));
                    }
                }
            }
            for (age, (fe, re)) in &e {
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                eprintln!("M3b {cname} night bounce {bounce} age {age:2}: rel_mse filtered {:.4}, raw {:.4}", mean(fe), mean(re));
            }
        }
    }
    rig.free_frame(f);
    rig.finish();
}

// ---- M1: the equal-time curve ----

const CURVE_W: u32 = 480;
const CURVE_H: u32 = 270;
const CURVE_REF_SPP: u32 = 16_384;
const CURVE_FRAMES: u32 = 64;

/// M1 (error): per dressing, camera and s, the one-frame relative MSE of the raw shade output against
/// the dressing's reference (480×270, one bounce, 16,384 spp) over 64 frames, with its standard error
/// and the share of the error in the worst 1% of pixels.
#[test]
fn equal_time_curve_error() {
    let mut rig = Rig::new(&Dressing::ALL);
    let light = lighting(NIGHT);
    for d in 0..rig.nights.len() {
        let mut f = rig.frame_set(d, CURVE_W, CURVE_H);
        for (cname, cam) in cameras(CURVE_W, CURVE_H, false) {
            let t0 = std::time::Instant::now();
            let r = rig.reference(d, &cam, &light, &ref_settings(true, true, true), REF_SEED, 0..CURVE_REF_SPP, 64);
            let rmean: Vec<[f64; 3]> = (0..r.sum.len()).map(|i| r.mean(i)).collect();
            eprintln!("M1 reference {} {cname}: {CURVE_REF_SPP} spp in {:.1} s", rig.nights[d].dressing.name(), t0.elapsed().as_secs_f64());
            rig.one(&mut f, &cam, &light, Step::viewer(1, 1, SEED));
            let mask = surface_mask(&rig.guides(&f));
            for spp in [1u32, 2, 4] {
                let idx: Vec<usize> = (0..rmean.len()).filter(|&i| mask[i]).collect();
                let n = idx.len().max(1) as f64;
                let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| rmean[i][c]).sum::<f64>() / n);
                let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-30));
                let mut per_frame = Vec::new();
                let mut per_pixel = vec![0.0f64; rmean.len()];
                for k in 0..CURVE_FRAMES {
                    rig.one(&mut f, &cam, &light, Step::raw(lit(spp), k, SEED));
                    let x = rig.radiance(&f);
                    per_frame.push(rel_mse(&x, &rmean, &mask));
                    for &i in &idx {
                        per_pixel[i] += (0..3).map(|c| (x[i][c] as f64 - rmean[i][c]).powi(2) / (rmean[i][c].powi(2) + eps[c])).sum::<f64>() / 3.0;
                    }
                }
                let m = per_frame.iter().sum::<f64>() / per_frame.len() as f64;
                let var = per_frame.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (per_frame.len() - 1) as f64;
                let mut pp: Vec<f64> = per_pixel.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
                pp.sort_by(|a, b| b.total_cmp(a));
                let top = pp.iter().take(pp.len().div_ceil(100)).sum::<f64>() / pp.iter().sum::<f64>().max(1e-300);
                eprintln!("M1 error {} ({} emitters) {cname} s={spp}: one-frame rel_mse {m:.4} +- {:.4}, worst 1% of pixels carry {:.1}%", rig.nights[d].dressing.name(), rig.nights[d].emitters, (var / per_frame.len() as f64).sqrt(), 100.0 * top);
            }
        }
        rig.free_frame(f);
    }
    rig.finish();
}

/// M1 (time) and M2's compose cost: the shade pass at 1920×1080 at night for every dressing and s,
/// all arms of a camera in one command buffer in alternating order, 40 repetitions (median, p90);
/// the compose pass on Full. Validation off gives the timing run: `NE_NO_VALIDATION=1`.
#[test]
fn equal_time_curve_cost() {
    const REPS: usize = 40;
    let g = gpu_for_timing();
    let validation = g.validation_enabled();
    let mut rig = Rig::with_gpu(g, &Dressing::ALL);
    let light = lighting(NIGHT);
    let mut frames: Vec<Frame> = (0..rig.nights.len()).map(|d| rig.frame_set(d, W, H)).collect();
    let arms: Vec<(usize, u32)> = (0..rig.nights.len()).flat_map(|d| [1u32, 2, 4].map(|s| (d, s))).collect();
    let passes = arms.len() as u32 + 1;
    for (cname, cam) in cameras(W, H, false) {
        let mut timer = GpuTimer::new(&rig.g, 1, passes).unwrap();
        let mut ms = vec![Vec::new(); passes as usize];
        for rep in 0..=REPS {
            let mut sub = rig.sub.take().unwrap();
            let cmd = sub.begin(&rig.g, &rig.tl).unwrap();
            timer.reset(&rig.g, cmd, 0);
            let order: Vec<usize> = if rep % 2 == 0 { (0..arms.len()).collect() } else { (0..arms.len()).rev().collect() };
            if let (Some(sp), false) = (&rig.sky_pass, rig.sky_sun == Some(light.sun_dir)) {
                sp.record(&rig.g, cmd, light.sun_dir);
                rig.sky_sun = Some(light.sun_dir);
            }
            for &a in &order {
                let (d, spp) = arms[a];
                let f = &frames[d];
                let n = &rig.nights[d];
                rig.raster.record(&rig.g, cmd, &f.targets, &n.scene.meshes, &f.rb, &cam, Faults::default());
                targets_to_read(&rig.g, cmd, &f.targets);
                full_barrier(&rig.g, cmd);
                timer.begin_pass(&rig.g, cmd, 0, a as u32);
                rig.shade.record(&rig.g, cmd, &f.sbl, &f.out, &shade::Params::new(&cam, &light, lit(spp), ShadeFaults::default(), rep as u32, SEED));
                timer.end_pass(&rig.g, cmd, 0, a as u32);
                full_barrier(&rig.g, cmd);
            }
            // Compose (emission and the meter) on Full, after the arms.
            let full = Dressing::ALL.iter().position(|&d| d == Dressing::Full).unwrap();
            let f = &frames[full];
            timer.begin_pass(&rig.g, cmd, 0, passes - 1);
            rig.compose.record(&rig.g, cmd, &f.cb, &f.co, 0, true, 0.0, ComposeFaults::default());
            timer.end_pass(&rig.g, cmd, 0, passes - 1);
            let v = sub.submit(&rig.g, &mut rig.tl, cmd, &[]).unwrap();
            rig.tl.wait(&rig.g, v, u64::MAX).unwrap();
            rig.sub = Some(sub);
            let t = timer.read(&rig.g, 0).unwrap().unwrap();
            if rep > 0 {
                for (p, v) in t.iter().enumerate() {
                    ms[p].push(*v);
                }
            }
        }
        for (a, &(d, spp)) in arms.iter().enumerate() {
            eprintln!("M1 time {} ({} emitters) {cname} s={spp}: shade {:.3} ms median, p90 {:.3} (validation {validation}, {REPS} reps)", rig.nights[d].dressing.name(), rig.nights[d].emitters, percentile(&ms[a], 50.0).unwrap(), percentile(&ms[a], 90.0).unwrap());
        }
        let c = &ms[passes as usize - 1];
        eprintln!("M2 compose {cname} (Full, 1080p): {:.3} ms median, p90 {:.3}", percentile(c, 50.0).unwrap(), percentile(c, 90.0).unwrap());
        timer.destroy(&rig.g);
    }
    for f in frames.drain(..) {
        rig.free_frame(f);
    }
    rig.finish();
}

/// Part 2 (R3, R4): the 3E rig's size.
const R_W: u32 = 320;
const R_H: u32 = 180;
/// Frames of R4's converged images (still camera, temporal pass, no filter).
const R_CONV: u32 = 4096;
/// Frames before the edit in each R arm (the edit's frame is the next one).
const R_BEFORE: u32 = 32;

/// The lit shade of the 3E rig with the bounce (the viewer's lighting, s = 1).
fn r_step(frame: u32) -> Step {
    Step::viewer(1, frame, SEED)
}

/// The still camera converged: a fresh history, `R_CONV` frames, no filter.
fn converged_small(rig: &mut Rig, f: &mut Frame, cam: &Camera, light: &Lighting) -> Vec<[f64; 3]> {
    rig.new_history(f);
    let st = Step { temporal: Some(TemporalSettings { max_age: R_CONV, ..TemporalSettings::default() }), filter: false, ..Step::raw(lit(1), 0, 0x4BCC) };
    let c = *cam;
    rig.run(f, &move |_| c, light, st, 1..=R_CONV, 64);
    rig.history_colour(f).iter().map(f64of).collect()
}

/// The lamp heads of the scene: the bounding boxes (`hi` exclusive) of the `lamp` voxels, one per
/// run of consecutive x (the heads are 8 m apart).
fn lamp_heads(w: &World) -> Vec<VoxelBox> {
    let lamp = w.materials().id_of("lamp").unwrap();
    let mut voxels: Vec<[i32; 3]> = w.occupied().filter(|&(_, m)| m == lamp).map(|(v, _)| [v.x, v.y, v.z]).collect();
    voxels.sort();
    let mut heads: Vec<VoxelBox> = Vec::new();
    for c in voxels {
        match heads.last_mut() {
            Some(h) if c[0] <= h.1[0] => {
                h.0 = std::array::from_fn(|a| h.0[a].min(c[a]));
                h.1 = std::array::from_fn(|a| h.1[a].max(c[a] + 1));
            }
            _ => heads.push((c, c.map(|x| x + 1))),
        }
    }
    heads
}

/// A voxel box: `lo` inclusive, `hi` exclusive.
type VoxelBox = ([i32; 3], [i32; 3]);
/// Edit (b)'s choice: the lamp head, the box, and the pixels in reach (every 4th).
type BoxChoice = (VoxelBox, VoxelBox, usize);
/// Voxels to set (None removes).
type Voxels = Vec<(VoxelCoord, Option<MaterialId>)>;

/// Edit (b)'s box: 3 × 3 × 3 in open air on the segment from a road point the camera sees to a lamp
/// head's lower face, a fraction of the way up. Of the candidates (every 8th pixel on the road,
/// fractions 0.15, 0.3, 0.5, every head), the one whose light rows reach the most pixels (the
/// geometric part of the rule, on every 4th pixel); with its head and that count.
fn choose_box(w: &World, table: &light::emitters::EmitterTable, cam: &Camera, points: &[Option<[f32; 3]>], sun: [f64; 3]) -> Option<BoxChoice> {
    let sample: Vec<[f32; 3]> = points.iter().step_by(4).flatten().copied().collect();
    let mut best: Option<BoxChoice> = None;
    for head in lamp_heads(w) {
        let top = [0.5 * (head.0[0] + head.1[0]) as f64, head.0[1] as f64, 0.5 * (head.0[2] + head.1[2]) as f64];
        for y in (0..cam.height).step_by(8) {
            for x in (0..cam.width).step_by(8) {
                let Some(p) = points[(y * cam.width + x) as usize] else { continue };
                if p[1].abs() > 1e-3 {
                    continue; // the road's surface is y = 0
                }
                for t in [0.15, 0.3, 0.5] {
                    let c: [i32; 3] = std::array::from_fn(|a| (p[a] as f64 + t * (top[a] - p[a] as f64)).floor() as i32);
                    let (lo, hi) = (c.map(|v| v - 1), c.map(|v| v + 2));
                    let free = (lo[0]..hi[0]).all(|x| (lo[1]..hi[1]).all(|y| (lo[2]..hi[2]).all(|z| w.get(VoxelCoord::new(x, y, z)).is_none())));
                    if !free {
                        continue;
                    }
                    let boxes = [Relight { lo, hi }];
                    let (rows, br) = (light_rows(&[], table, &boxes), relight_rows(sun, &boxes));
                    let n = sample.iter().filter(|p| relit_by_light(**p, 0.0, &br, &rows)).count();
                    if best.is_none_or(|b| n > b.2) {
                        best = Some((head, (lo, hi), n));
                    }
                }
            }
        }
    }
    best
}

/// Surface points of the camera's pixels by the exact DDA (`None`: sky), as f32 like the shader's.
fn surface_points(w: &World, cam: &Camera) -> Vec<Option<[f32; 3]>> {
    (0..cam.width * cam.height)
        .map(|i| {
            let r = cam.ray(i % cam.width, i / cam.width);
            trace(w, &r, 1e9).map(|hit| std::array::from_fn(|a| (r.origin[a] + hit.t * r.dir[a]) as f32))
        })
        .collect()
}

fn lum32(c: &[f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// R3, R4 (4B part 2, ADR-0006 Amendment 2): lights on, 21 h, 320 × 180, the 3E rig with the bounce.
/// Two edits: (a) a whole lamp head removed; (b) a 3 × 3 × 3 box placed in open air between a lamp
/// head and the road. R3: after each edit the pixels relit by a light equal the host rule on every
/// pixel whose history is otherwise accepted (<= 0.05% differ, >= 100 relit by a light); the arm
/// without the light rows must differ on >= 100. R4: 4 frames after the edit, the pixels whose
/// converged value changed by > 25% outside the rebuilt regions have a filtered relative MSE <= 3 x
/// that of an arm reset at the edit; the arm without the light rows must fail it.
#[test]
fn edits_relight_what_they_change_of_the_lights() {
    let mut rig = Rig::new(&[Dressing::Full]);
    let mut f = rig.frame_set(0, R_W, R_H);
    let cam = cameras(R_W, R_H, false)[0].1;
    let light = lighting(NIGHT);
    let sun = light.sun_dir;
    let before = surface_points(&rig.nights[0].world, &cam);

    // Edit (b)'s box and its lamp head (edit (a) removes the same head): the placement whose light
    // rows can change the light of the most pixels (the rule's geometric part; no threshold).
    let (head, bx) = {
        let n = &rig.nights[0];
        let (head, b, count) = choose_box(&n.world, &n.scene.emitters().unwrap().unwrap().table, &cam, &before, sun).expect("a box in open air under a lamp head");
        eprintln!("chosen: head {head:?}, box {b:?} ({count} of every 4th px in reach)");
        (head, b)
    };
    let pole = rig.nights[0].world.materials().id_of("metal_pole").unwrap();
    let head_saved = rig.voxels(0, head.0, head.1);
    let box_saved = rig.voxels(0, bx.0, bx.1);
    let edits: [(&str, Voxels, Relight); 2] = [
        ("a_lamp_head_removed", head_saved.iter().map(|&(c, _)| (c, None)).collect(), Relight { lo: head.0, hi: head.1 }),
        ("b_box_placed", box_saved.iter().map(|&(c, _)| (c, Some(pole))).collect(), Relight { lo: bx.0, hi: bx.1 }),
    ];
    let saved = [head_saved.clone(), box_saved.clone()];

    let t_old = converged_small(&mut rig, &mut f, &cam, &light);
    let mut failed = Vec::new();
    for ((name, edit, b), undo) in edits.iter().zip(&saved) {
        // The edit's converged image, surface points and light rows (as the viewer lists them).
        let changed = rig.edit(&mut f, edit);
        let t_new = converged_small(&mut rig, &mut f, &cam, &light);
        let points = surface_points(&rig.nights[0].world, &cam);
        let rows = light_rows(&changed, &rig.nights[0].scene.emitters().unwrap().unwrap().table, &[*b]);
        let restored = rig.edit(&mut f, undo);
        let boxes = [*b];
        let br = relight_rows(sun, &boxes);
        eprintln!("{name}: {} changed emitters, {} light rows ({} on restoring)", changed.len(), rows.len(), restored.len());

        let mut arms = BTreeMap::new();
        for arm in ["lights", "none", "reset"] {
            rig.new_history(&mut f);
            let c = cam;
            rig.run(&mut f, &move |_| c, &light, r_step(1), 1..=R_BEFORE, 8);
            let hist = rig.history_colour(&f);
            rig.edit(&mut f, edit);
            rig.next_boxes = boxes.to_vec();
            if arm == "lights" {
                rig.next_rows = rows.clone();
            }
            rig.one(&mut f, &cam, &light, Step { reset: arm == "reset", ..r_step(R_BEFORE + 1) });
            let st = rig.state(&f);
            rig.run(&mut f, &move |_| c, &light, r_step(R_BEFORE + 2), R_BEFORE + 2..=R_BEFORE + 4, 3);
            let x = rig.radiance(&f);
            rig.edit(&mut f, undo);
            arms.insert(arm, (st, x, hist));
        }

        // R3: the host rule on the pixels otherwise accepted, with the history before the edit frame.
        let hist = &arms["lights"].2;
        for arm in ["lights", "none"] {
            let st = &arms[arm].0;
            let (mut relit, mut differ, mut decided) = (0usize, 0usize, 0usize);
            for (i, p) in points.iter().enumerate() {
                let (Some(p), r) = (*p, st[i].1) else { continue };
                if r != reason::ACCEPTED && r != reason::RELIT_LIGHT {
                    continue;
                }
                decided += 1;
                let h = relit_by_light(p, lum32(&hist[i]), &br, &rows);
                if r == reason::RELIT_LIGHT {
                    relit += 1;
                }
                if (r == reason::RELIT_LIGHT) != h {
                    differ += 1;
                }
            }
            let ok = differ * 2000 <= (R_W * R_H) as usize && relit >= 100;
            eprintln!("R3 {name} {arm}: {relit} px relit by a light, {differ} of {decided} decisions differ from the host rule ({} px) {}", R_W * R_H, if ok { "(passes)" } else { "(fails)" });
            match arm {
                "lights" if !ok => failed.push(format!("R3 {name}")),
                "none" if differ < 100 => failed.push(format!("R3 {name}: the control without light rows differs on only {differ} px")),
                _ => {}
            }
        }
        // R4: changed pixels with a history outside the rebuilt regions.
        let st = &arms["lights"].0;
        let mask: Vec<bool> = (0..(R_W * R_H) as usize)
            .map(|i| {
                let (a, b) = (lum(t_old[i]), lum(t_new[i]));
                matches!(st[i].1, reason::ACCEPTED | reason::RELIT | reason::RELIT_LIGHT) && (a - b).abs() > 0.25 * a.max(b)
            })
            .collect();
        let n = mask.iter().filter(|&&m| m).count();
        let e: BTreeMap<&str, f64> = arms.iter().map(|(k, (_, x, _))| (*k, rel_mse(x, &t_new, &mask))).collect();
        let by_reason = |arm: &str| {
            let mut c = [0usize; 11];
            for (i, m) in mask.iter().enumerate() {
                if *m {
                    c[arms[arm].0[i].1 as usize] += 1;
                }
            }
            c
        };
        eprintln!("R4 {name}: {n} changed px (reasons in the lights arm {:?}); 4 frames after the edit: lights {:.4}, none {:.4}, reset {:.4}", by_reason("lights"), e["lights"], e["none"], e["reset"]);
        if n < 100 {
            failed.push(format!("R4 {name}: needs >= 100 changed px to be exercised, got {n}"));
        }
        if e["lights"] > 3.0 * e["reset"] {
            failed.push(format!("R4 {name}"));
        }
        if e["none"] <= 3.0 * e["reset"] {
            failed.push(format!("R4 {name}: the control without light rows is not caught"));
        }
    }
    rig.free_frame(f);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}
