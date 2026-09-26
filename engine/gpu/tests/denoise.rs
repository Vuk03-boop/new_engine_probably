//! Phase 3E GPU tests: the native reconstruction (`gpu::denoise`) and the temporal pass's response to
//! lighting change (relight boxes, the sun-motion age cap; ADR-0006 Amendment 1). Errors are measured
//! against the same estimator converged on a still camera (thousands of frames), so they isolate the
//! reconstruction from the estimator's own bias (3B, 3C). Criteria are declared in
//! `docs/changes/2026-09-24-phase3e-denoise.md`; `edits_relight_what_they_change_with_the_bounce` is
//! 3G E2 (`docs/changes/2026-09-25-phase3g-gate.md`). Like the other GPU tests they need the Vulkan SDK
//! and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test denoise -- --test-threads=1 --nocapture`.

use std::collections::{BTreeMap, BTreeSet};

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::alloc::Allocator;
use gpu::debug_view::{targets_to_read, Tables};
use gpu::denoise::{Denoise, DenoiseBindings, DenoiseFaults, DenoiseSettings, DenoiseTargets};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::raster::{Bindings as RasterBindings, Camera, Faults, Raster, Targets};
use gpu::reference::RefMaterials;
use gpu::scene::{affected_regions, region_meshes, GpuScene};
use gpu::shade::{self, Shade, ShadeBindings, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::temporal::{reason, History, Relight, Temporal, TemporalBindings, TemporalFaults, TemporalSettings};
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, GpuError, Timeline};
use light::reference::{albedos, Lighting};
use light::sky::SkyLuts;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::reference::{trace, Ray};
use world::{scene, BrickKey, MaterialId, Transaction, VoxelCoord, World};

/// Sun and sky without the 3F bounce: the lighting this file's criteria were measured with.
const DIRECT: ShadeSettings = ShadeSettings { sun: true, sky: true, point_sun: false, uniform_sky: false, bounce: false, ..ShadeSettings::NONE };
const NEAR: f64 = 0.1;
const SEED: u32 = 0x3E;
const W: u32 = 320;
const H: u32 = 180;
/// Frames accumulated for a converged target (still camera, `max_age` above it).
const TARGET_FRAMES: u32 = 2048;
/// The shade frame index where target sequences start (independent of the measured sequences).
const TARGET_FRAME0: u32 = 1 << 24;
/// Hours measured on stills: morning (long shadows), midday, dusk, twilight (sky only).
const HOURS: [f64; 4] = [8.0, 12.0, 17.75, 18.25];

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3E needs the ray-tracing device (P-001)");
    g
}

fn lighting(hour: f64) -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(hour))
}

fn street_camera(w: u32, h: u32) -> Camera {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)
}

/// The street camera moved sideways by `dx` voxels and turned by `yaw_deg`.
fn moved_camera(dx: f64, yaw_deg: f64, w: u32, h: u32) -> Camera {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    let (eye, target) = (vox(view.eye_m), vox(view.target_m));
    let f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    let (s, c) = yaw_deg.to_radians().sin_cos();
    let f2 = [f[0] * c - f[2] * s, f[1], f[0] * s + f[2] * c];
    let e2 = [eye[0] + dx, eye[1], eye[2]];
    Camera::look_at(e2, [e2[0] + f2[0], e2[1] + f2[1], e2[2] + f2[2]], view.vertical_fov_deg, w, h, NEAR)
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

/// What one frame runs after the shade pass.
#[derive(Clone, Copy)]
struct Step {
    ts: TemporalSettings,
    shade: ShadeSettings,
    /// None: the resolved (accumulated) radiance is shown unfiltered.
    denoise: Option<(DenoiseSettings, DenoiseFaults)>,
    force_reset: bool,
}

impl Step {
    fn raw(max_age: u32) -> Step {
        Step { ts: TemporalSettings { max_age, ..TemporalSettings::default() }, shade: DIRECT, denoise: None, force_reset: false }
    }

    fn filtered(max_age: u32, d: DenoiseSettings) -> Step {
        Step { denoise: Some((d, DenoiseFaults::default())), ..Step::raw(max_age) }
    }
}

/// The filter under test: `NE_FILTER` (`DenoiseSettings::parse`, e.g. `conservative:4`), else the
/// default. The G4 filter record (`docs/changes/2026-09-26-4b-filter-energy.md`) runs the criteria
/// with both; the first use prints which.
fn filter_under_test() -> DenoiseSettings {
    let d = std::env::var("NE_FILTER").map_or_else(|_| DenoiseSettings::default(), |s| DenoiseSettings::parse(&s).expect("NE_FILTER"));
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| eprintln!("filter under test: {}{}", d.tag(), if std::env::var_os("NE_FILTER").is_some() { " (NE_FILTER)" } else { " (default)" }));
    d
}

struct Bound {
    rb: RasterBindings,
    sb: ShadeBindings,
    tb: TemporalBindings,
    db: DenoiseBindings,
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    world: World,
    pipeline: Pipeline,
    scene: Option<GpuScene>,
    tables: Option<Tables>,
    mats: Option<RefMaterials>,
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
    targets: Option<Targets>,
    out: Option<ShadeTargets>,
    history: Option<History>,
    dn: Option<DenoiseTargets>,
    bound: Option<Bound>,
    raster: Raster,
    shade: Shade,
    temporal: Temporal,
    denoise: Denoise,
    sub: Option<Submitter>,
    sky_sun: Option<[f64; 3]>,
    w: u32,
    h: u32,
}

impl Rig {
    fn new() -> Rig {
        Self::with_gpu(gpu(), W, H)
    }

    /// For timing runs: validation may be off (`NE_NO_VALIDATION`).
    fn for_timing(w: u32, h: u32) -> Rig {
        let g = Gpu::new().expect("an RT-capable Vulkan device");
        assert!(g.ray_tracing());
        Self::with_gpu(g, w, h)
    }

    fn with_gpu(g: Gpu, w: u32, h: u32) -> Rig {
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let (world, _) = scene::street_block();
        let mut pipeline = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
        pipeline.mark_all(&world);
        drain(&mut pipeline, &world);
        let rs = full_regions(&mut pipeline, RegionSize::Chunk);
        let scene = GpuScene::build(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs, pipeline.current().id.raw(), true).unwrap();
        let params: Vec<world::MaterialParams> = world.materials().iter().map(|(_, d)| d.params).collect();
        let (tables, _) = Tables::upload(&g, &mut alloc, &mut up, &mut tl, &params, &scene.meshes, pipeline.current().id.raw()).unwrap();
        let (mats, _) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedos(world.materials()).unwrap()).unwrap();
        let (sky, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &SkyLuts::new(Atmosphere::default())).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let sky_pass = SkyView::new(&g, &sky).unwrap();
        let targets = Targets::new(&g, &mut alloc, vk::Extent2D { width: w, height: h }).unwrap();
        let out = ShadeTargets::new(&g, &mut alloc, w, h).unwrap();
        let history = History::new(&g, &mut alloc, w, h).unwrap();
        let dn = DenoiseTargets::new(&g, &mut alloc, w, h).unwrap();
        let raster = Raster::new(&g).unwrap();
        let shade = Shade::new(&g).unwrap();
        let temporal = Temporal::new(&g).unwrap();
        let denoise = Denoise::new(&g).unwrap();
        let sub = Submitter::new(&g).unwrap();
        let mut rig = Rig {
            g,
            alloc,
            tl,
            up,
            world,
            pipeline,
            scene: Some(scene),
            tables: Some(tables),
            mats: Some(mats),
            sky: Some(sky),
            sky_pass: Some(sky_pass),
            targets: Some(targets),
            out: Some(out),
            history: Some(history),
            dn: Some(dn),
            bound: None,
            raster,
            shade,
            temporal,
            denoise,
            sub: Some(sub),
            sky_sun: None,
            w,
            h,
        };
        rig.rebind();
        rig
    }

    /// (Re)creates every descriptor binding (after an edit or a new history).
    fn rebind(&mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        if let Some(b) = self.bound.take() {
            b.rb.destroy(g);
            b.sb.destroy(g);
            b.tb.destroy(g);
            b.db.destroy(g);
        }
        let (targets, out, s, h) = (self.targets.as_ref().unwrap(), self.out.as_ref().unwrap(), self.scene.as_ref().unwrap(), self.history.as_ref().unwrap());
        let rb = self.raster.bind(g, &s.meshes).unwrap();
        let sb = self.shade.bind(g, targets, s.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), out, &self.sky.as_ref().unwrap().view).unwrap();
        let tb = self.temporal.bind(g, targets, &self.tables.as_ref().unwrap().regions, &out.radiance, h).unwrap();
        let db = self.denoise.bind(g, &out.radiance, h, self.mats.as_ref().unwrap(), self.dn.as_ref().unwrap()).unwrap();
        self.bound = Some(Bound { rb, sb, tb, db });
    }

    /// A fresh history (the next frame is a first-frame reset).
    fn new_history(&mut self) {
        self.g.wait_idle().unwrap();
        self.history.take().unwrap().free(&self.g, &mut self.alloc);
        self.history = Some(History::new(&self.g, &mut self.alloc, self.w, self.h).unwrap());
        self.rebind();
    }

    /// Records one frame: sky table (when the sun moved), raster, shade, temporal, and the filter.
    #[allow(clippy::too_many_arguments)]
    fn record(&mut self, cmd: vk::CommandBuffer, cam: &Camera, light: &Lighting, step: Step, relight: &[Relight], frame: u32, timer: Option<(&GpuTimer, u32)>) {
        let (g, targets, out, s, b) = (&self.g, self.targets.as_ref().unwrap(), self.out.as_ref().unwrap(), self.scene.as_ref().unwrap(), self.bound.as_ref().unwrap());
        if self.sky_sun != Some(light.sun_dir) {
            self.sky_pass.as_ref().unwrap().record(g, cmd, light.sun_dir);
            self.sky_sun = Some(light.sun_dir);
        }
        self.raster.record(g, cmd, targets, &s.meshes, &b.rb, cam, Faults::default());
        targets_to_read(g, cmd, targets);
        self.shade.record(g, cmd, &b.sb, out, &shade::Params::new(cam, light, step.shade, ShadeFaults::default(), frame, SEED));
        let h = self.history.as_mut().unwrap();
        if let Some((t, slot)) = timer {
            t.begin_pass(g, cmd, slot, 0);
        }
        self.temporal.record(g, cmd, &b.tb, h, cam, light.sun_dir, step.ts, TemporalFaults::default(), step.force_reset, relight);
        if let Some((t, slot)) = timer {
            t.end_pass(g, cmd, slot, 0);
            t.begin_pass(g, cmd, slot, 1);
        }
        if let Some((d, f)) = step.denoise {
            self.denoise.record(g, cmd, &b.db, h, self.dn.as_ref().unwrap(), d, f);
        }
        if let Some((t, slot)) = timer {
            t.end_pass(g, cmd, slot, 1);
        }
    }

    /// Runs one frame; returns the shown radiance when `read`.
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, cam: &Camera, light: &Lighting, step: Step, relight: &[Relight], frame: u32, read: bool) -> Option<Vec<[f32; 4]>> {
        let mut sub = self.sub.take().unwrap();
        let cmd = sub.begin(&self.g, &self.tl).unwrap();
        self.record(cmd, cam, light, step, relight, frame, None);
        let v = sub.submit(&self.g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(&self.g, v, u64::MAX).unwrap();
        self.sub = Some(sub);
        read.then(|| self.out.as_ref().unwrap().read(&self.g, &mut self.alloc, &mut self.tl).unwrap())
    }

    /// The estimator converged on a still camera: a fresh history, `TARGET_FRAMES` frames, no filter.
    fn target(&mut self, cam: &Camera, light: &Lighting) -> Vec<[f64; 3]> {
        self.target_with(cam, light, DIRECT, TARGET_FRAMES)
    }

    fn target_with(&mut self, cam: &Camera, light: &Lighting, shade: ShadeSettings, frames: u32) -> Vec<[f64; 3]> {
        self.new_history();
        let step = Step { shade, ..Step::raw(frames * 2) };
        let mut last = None;
        for k in 0..frames {
            last = self.frame(cam, light, step, &[], TARGET_FRAME0 + k, k + 1 == frames);
        }
        self.new_history();
        last.unwrap().iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect()
    }

    fn guides(&mut self) -> Vec<[u32; 4]> {
        self.history.as_ref().unwrap().read_guides(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    fn state(&mut self) -> Vec<(u32, u32)> {
        self.history.as_ref().unwrap().read(&self.g, &mut self.alloc, &mut self.tl).unwrap().0
    }

    /// The voxels of a box (`hi` exclusive) and their materials.
    fn voxels(&self, lo: [i32; 3], hi: [i32; 3]) -> Vec<(VoxelCoord, Option<MaterialId>)> {
        let mut v = Vec::new();
        for x in lo[0]..hi[0] {
            for y in lo[1]..hi[1] {
                for z in lo[2]..hi[2] {
                    let c = VoxelCoord::new(x, y, z);
                    v.push((c, self.world.get(c)));
                }
            }
        }
        v
    }

    /// Sets voxels (None removes), publishes, updates the device scene and the region table, and
    /// rebinds. Returns the region keys whose snapshot changed.
    fn edit(&mut self, voxels: &[(VoxelCoord, Option<MaterialId>)]) -> BTreeSet<RegionKey> {
        let mut tx = Transaction::new();
        for &(c, m) in voxels {
            tx.set(c, m);
        }
        let applied = self.world.apply(&tx).unwrap();
        assert!(!applied.changed.is_empty(), "the edit must change something");
        self.pipeline.notify_edits(&applied.changed);
        let keys = drain(&mut self.pipeline, &self.world);
        let size = RegionSize::Chunk;
        let t = self.pipeline.acquire();
        let changed = region_meshes(&self.pipeline, &t, &affected_regions(keys.iter().copied(), size), size).unwrap();
        self.pipeline.release(t).unwrap();
        let regions: BTreeSet<RegionKey> = changed.keys().copied().collect();
        let s = self.scene.as_mut().unwrap();
        s.update(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, changed, self.pipeline.current().id.raw()).unwrap();
        self.g.wait_idle().unwrap();
        let (old, _) = self.tables.as_mut().unwrap().replace_regions(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, &s.region_rows()).unwrap();
        self.g.wait_idle().unwrap();
        self.alloc.free(&self.g, old);
        let done = self.tl.completed(&self.g).unwrap();
        s.collect(&self.g, &mut self.alloc, done);
        self.rebind();
        regions
    }

    fn finish(mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let (errors, warnings) = g.validation_counts();
        eprintln!("validation: {errors} errors, {warnings} warnings");
        assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
        if let Some(b) = self.bound.take() {
            b.rb.destroy(g);
            b.sb.destroy(g);
            b.tb.destroy(g);
            b.db.destroy(g);
        }
        self.sub.take().unwrap().destroy(g);
        self.dn.take().unwrap().free(g, &mut self.alloc);
        self.history.take().unwrap().free(g, &mut self.alloc);
        self.out.take().unwrap().free(g, &mut self.alloc);
        self.targets.take().unwrap().free(g, &mut self.alloc);
        self.sky_pass.take().unwrap().destroy(g);
        self.sky.take().unwrap().free(g, &mut self.alloc);
        self.mats.take().unwrap().free(g, &mut self.alloc);
        self.tables.take().unwrap().free_now(g, &mut self.alloc);
        self.scene.take().unwrap().free_now(g, &mut self.alloc);
        self.raster.destroy(g);
        self.shade.destroy(g);
        self.temporal.destroy(g);
        self.denoise.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

const FACE_NONE: u32 = 7;

fn surface_mask(guides: &[[u32; 4]]) -> Vec<bool> {
    guides.iter().map(|g| g[0] & 7 != FACE_NONE).collect()
}

/// Surface pixels with a different surface (face, material or plane, or sky) within `r` pixels.
fn edge_mask(guides: &[[u32; 4]], w: u32, h: u32, r: i32) -> Vec<bool> {
    let (w, h) = (w as i32, h as i32);
    (0..guides.len())
        .map(|i| {
            let g = guides[i];
            if g[0] & 7 == FACE_NONE {
                return false;
            }
            let (x, y) = (i as i32 % w, i as i32 / w);
            (-r..=r).any(|dy| {
                (-r..=r).any(|dx| {
                    let (qx, qy) = (x + dx, y + dy);
                    qx >= 0 && qy >= 0 && qx < w && qy < h && {
                        let q = guides[(qy * w + qx) as usize];
                        q[0] != g[0] || q[1] != g[1]
                    }
                })
            })
        })
        .collect()
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Relative MSE (per channel, epsilon = (0.1 x the channel's mean)^2) and the relative bias of the
/// mean luminance, over the masked pixels.
#[derive(Clone, Copy, Debug)]
struct Error {
    rel_mse: f64,
    bias: f64,
    pixels: usize,
}

fn error(x: &[[f32; 4]], t: &[[f64; 3]], mask: &[bool]) -> Error {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| t[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-12));
    let mut se = 0.0;
    let (mut lx, mut lt) = (0.0, 0.0);
    for &i in &idx {
        for c in 0..3 {
            se += (x[i][c] as f64 - t[i][c]).powi(2) / (t[i][c].powi(2) + eps[c]);
        }
        lx += lum([x[i][0] as f64, x[i][1] as f64, x[i][2] as f64]);
        lt += lum(t[i]);
    }
    Error { rel_mse: se / (3.0 * n), bias: (lx - lt) / lt.max(1e-30), pixels: idx.len() }
}

/// Per-pixel squared relative error (the `error` metric's terms; 0 outside the mask).
fn rel_terms(x: &[[f32; 4]], t: &[[f64; 3]], mask: &[bool]) -> Vec<f64> {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| t[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-12));
    (0..t.len())
        .map(|i| if mask[i] { (0..3).map(|c| (x[i][c] as f64 - t[i][c]).powi(2) / (t[i][c].powi(2) + eps[c])).sum::<f64>() / 3.0 } else { 0.0 })
        .collect()
}

/// Diagnostic (run before the filter's criteria were declared): the unfiltered estimator's error
/// against the converged target at history ages 1 to 64, per hour, on the street camera.
#[test]
#[ignore]
fn measure_raw_input() {
    let mut rig = Rig::new();
    let cam = street_camera(W, H);
    for hour in HOURS {
        let light = lighting(hour);
        let t = rig.target(&cam, &light);
        // The target's own noise: a second, independent target.
        let t2: Vec<[f32; 4]> = {
            rig.new_history();
            let mut last = None;
            for k in 0..TARGET_FRAMES {
                last = rig.frame(&cam, &light, Step::raw(TARGET_FRAMES * 2), &[], 2 * TARGET_FRAME0 + k, k + 1 == TARGET_FRAMES);
            }
            last.unwrap()
        };
        rig.new_history();
        let mut line = Vec::new();
        for k in 1..=64u32 {
            let x = rig.frame(&cam, &light, Step::raw(64), &[], k, k.is_power_of_two());
            if let Some(x) = x {
                let g = rig.guides();
                let (m, e) = (surface_mask(&g), edge_mask(&g, W, H, 2));
                let (a, b) = (error(&x, &t, &m), error(&x, &t, &e));
                line.push(format!("age {k}: rel_mse {:.4} bias {:+.4} edge rel_mse {:.4}", a.rel_mse, a.bias, b.rel_mse));
                if k == 1 {
                    let tt = error(&t2, &t, &m);
                    eprintln!("hour {hour}: {} surface px, {} edge px; target vs target rel_mse {:.2e} bias {:+.1e}", a.pixels, b.pixels, tt.rel_mse, tt.bias);
                }
            }
        }
        for l in line {
            eprintln!("  raw {l}");
        }
    }
    rig.finish();
}

/// Runs a still camera from a fresh history (frame index = age); returns the shown radiance at `ages`.
fn run_still(rig: &mut Rig, cam: &Camera, light: &Lighting, step: Step, ages: &[u32]) -> BTreeMap<u32, Vec<[f32; 4]>> {
    rig.new_history();
    let last = *ages.iter().max().unwrap();
    let mut out = BTreeMap::new();
    for k in 1..=last {
        if let Some(x) = rig.frame(cam, light, step, &[], k, ages.contains(&k)) {
            out.insert(k, x);
        }
    }
    out
}

fn filtered_with(max_age: u32, d: DenoiseSettings, faults: DenoiseFaults) -> Step {
    Step { denoise: Some((d, faults)), ..Step::raw(max_age) }
}

/// The display exposure of the 3D images: 1 / (sun + sky irradiance at the ground, as radiance).
fn exposure(light: &Lighting, luts: &SkyLuts) -> f64 {
    let g = light.sun_at_ground;
    let e = luts.ground_irradiance(light.sun_dir[1]);
    1.0 / ((lum(g) + lum(e)) / std::f64::consts::PI + 1e-7)
}

/// Writes images side by side as a PPM (ACES fit, sRGB).
fn write_strip(name: &str, w: u32, h: u32, images: &[Vec<[f32; 4]>], exposure: f64) {
    let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
    let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    let mut bytes = format!("P6\n{} {h}\n255\n", w * images.len() as u32).into_bytes();
    for y in 0..h {
        for img in images {
            for x in 0..w {
                for &c in &img[(y * w + x) as usize][..3] {
                    bytes.push((srgb(aces(c as f64 * exposure)) * 255.0 + 0.5) as u8);
                }
            }
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results").join(name);
    std::fs::write(&path, bytes).unwrap();
    eprintln!("wrote {}", path.display());
}

fn as_image(t: &[[f64; 3]]) -> Vec<[f32; 4]> {
    t.iter().map(|c| [c[0] as f32, c[1] as f32, c[2] as f32, 0.0]).collect()
}

/// D1-D3: stills. Equivalent-sample gain >= 8 at age 1, >= 4 at ages 4 and 16, >= 1 at age 64; the
/// filtered mean luminance within 1% of the raw one in the same frame and 2% of the target; at age
/// 64 on edge pixels filtered <= raw, which the planted `ignore_guides` must fail. Layout checked.
#[test]
fn stills_meet_the_budget() {
    let mut rig = Rig::new();
    match Denoise::with_reflection(&rig.g, gpu::temporal::REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("denoise pipeline with the temporal reflection refused: {} errors", e.len()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }
    let luts = SkyLuts::new(Atmosphere::default());
    let cam = street_camera(W, H);
    let d = filter_under_test();
    let mut failed = Vec::new();
    for hour in HOURS {
        let light = lighting(hour);
        let t = rig.target(&cam, &light);
        let raw = run_still(&mut rig, &cam, &light, Step::raw(64), &[1, 4, 8, 16, 64]);
        let g = rig.guides();
        let (m, e) = (surface_mask(&g), edge_mask(&g, W, H, 2));
        let filt = run_still(&mut rig, &cam, &light, Step::filtered(64, d), &[1, 4, 16, 64]);
        let fault = run_still(&mut rig, &cam, &light, filtered_with(64, d, DenoiseFaults { ignore_guides: true }), &[64]);
        for (age, gain) in [(1u32, 8u32), (4, 4), (16, 4), (64, 1)] {
            let (f, r, rg) = (error(&filt[&age], &t, &m), error(&raw[&age], &t, &m), error(&raw[&(age * gain)], &t, &m));
            let d1 = f.rel_mse <= rg.rel_mse;
            let d2 = (f.bias - r.bias).abs() <= 0.01 && f.bias.abs() <= 0.02;
            eprintln!(
                "hour {hour} age {age}: filtered rel_mse {:.4} (raw {:.4}, raw at age {} {:.4}: gain {}) bias {:+.4} (raw {:+.4}) {}",
                f.rel_mse,
                r.rel_mse,
                age * gain,
                rg.rel_mse,
                if d1 { "met" } else { "MISSED" },
                f.bias,
                r.bias,
                if d2 { "energy ok" } else { "ENERGY OFF" }
            );
            if !d1 {
                failed.push(format!("D1 hour {hour} age {age}"));
            }
            if !d2 {
                failed.push(format!("D2 hour {hour} age {age}"));
            }
        }
        let (fe, re, xe) = (error(&filt[&64], &t, &e), error(&raw[&64], &t, &e), error(&fault[&64], &t, &e));
        let xa = error(&fault[&64], &t, &m);
        eprintln!("hour {hour} edges at 64: filtered {:.4} raw {:.4} | ignore_guides {:.4} (all surfaces {:.4}, bias {:+.4})", fe.rel_mse, re.rel_mse, xe.rel_mse, xa.rel_mse, xa.bias);
        if fe.rel_mse > re.rel_mse {
            failed.push(format!("D3 hour {hour}"));
        }
        if xe.rel_mse <= re.rel_mse {
            failed.push(format!("D3 fault ignore_guides not caught at hour {hour}"));
        }
        if hour == 8.0 || hour == 17.75 {
            write_strip(&format!("denoise_3e_{hour}.ppm"), W, H, &[raw[&1].clone(), filt[&1].clone(), filt[&64].clone(), as_image(&t)], exposure(&light, &luts));
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// The shown radiance and the guides of one frame.
type Shown = (Vec<[f32; 4]>, Vec<[u32; 4]>);

/// D4: a moving camera (0.25 voxel sideways and 0.2 deg of yaw per frame, 8 h): at frames 8, 16, 32
/// the filtered relative MSE is at most 1/4 of the raw one of the same frame, energy within 1%.
#[test]
fn motion_path_meets_the_budget() {
    let mut rig = Rig::new();
    let light = lighting(8.0);
    let cam_at = |k: u32| moved_camera(0.25 * k as f64, 0.2 * k as f64, W, H);
    let checks = [8u32, 16, 32];
    let targets: Vec<_> = checks.iter().map(|&k| rig.target(&cam_at(k), &light)).collect();
    let mut shown: BTreeMap<(bool, u32), Shown> = BTreeMap::new();
    for filtered in [false, true] {
        rig.new_history();
        let step = if filtered { Step::filtered(64, filter_under_test()) } else { Step::raw(64) };
        for k in 1..=32u32 {
            if let Some(x) = rig.frame(&cam_at(k), &light, step, &[], k, checks.contains(&k)) {
                let g = rig.guides();
                shown.insert((filtered, k), (x, g));
            }
        }
    }
    let mut failed = Vec::new();
    for (i, &k) in checks.iter().enumerate() {
        let (r, g) = &shown[&(false, k)];
        let (f, _) = &shown[&(true, k)];
        let m = surface_mask(g);
        let (fe, re) = (error(f, &targets[i], &m), error(r, &targets[i], &m));
        let ok = fe.rel_mse * 4.0 <= re.rel_mse && (fe.bias - re.bias).abs() <= 0.01;
        eprintln!("motion frame {k}: filtered rel_mse {:.4} bias {:+.4} | raw {:.4} bias {:+.4} {}", fe.rel_mse, fe.bias, re.rel_mse, re.bias, if ok { "ok" } else { "MISSED" });
        // Diagnostic: the error's share by the raw arm's history age and edge proximity.
        let (tf, tr) = (rel_terms(f, &targets[i], &m), rel_terms(r, &targets[i], &m));
        let edges = edge_mask(g, W, H, 2);
        let total = m.iter().filter(|&&b| b).count() as f64;
        for (label, lo_age, hi_age) in [("age 1-3", 1u32, 4u32), ("age 4-15", 4, 16), ("age 16+", 16, u32::MAX)] {
            for edge in [false, true] {
                let sel: Vec<usize> = (0..m.len()).filter(|&j| m[j] && edges[j] == edge && { let a = (r[j][3] as u32) & 0xFFFF; a >= lo_age && a < hi_age }).collect();
                let (sf, sr): (f64, f64) = (sel.iter().map(|&j| tf[j]).sum(), sel.iter().map(|&j| tr[j]).sum());
                eprintln!("  {label} {}: {} px, filtered {:.4} raw {:.4} (shares of the frame's mean)", if edge { "edge    " } else { "interior" }, sel.len(), sf / total, sr / total);
            }
        }
        if !ok {
            failed.push(k);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "D4 missed at frames {failed:?}");
}

/// A labelled camera path and step of `diagnose_motion_edges`.
type Arm = (&'static str, Box<dyn Fn(u32) -> Camera>, Step);

/// Diagnostic for D4: the edge error at frame 32 of the motion path, split by what moves (the same
/// view held still, translation only, rotation only) and by the filter's settings.
#[test]
#[ignore]
fn diagnose_motion_edges() {
    let mut rig = Rig::new();
    let light = lighting(8.0);
    let k = 32u32;
    let path = |dx: f64, yaw: f64| move |j: u32| moved_camera(dx * j as f64, yaw * j as f64, W, H);
    let end = moved_camera(0.25 * k as f64, 0.2 * k as f64, W, H);
    let t = rig.target(&end, &light);
    let d = DenoiseSettings::default();
    let raw = run_still(&mut rig, &end, &light, Step::raw(64), &[32]);
    let g = rig.guides();
    let (m, e) = (surface_mask(&g), edge_mask(&g, W, H, 2));
    let show = |label: &str, x: &[[f32; 4]]| {
        let (a, b) = (error(x, &t, &m), error(x, &t, &e));
        eprintln!("{label}: all {:.4} (bias {:+.4}) | edge {:.4} (bias {:+.4})", a.rel_mse, a.bias, b.rel_mse, b.bias);
    };
    show("still at the end view, raw age 32", &raw[&32]);
    let filt = run_still(&mut rig, &end, &light, Step::filtered(64, d), &[32]);
    show("still at the end view, filtered age 32", &filt[&32]);
    let arms: Vec<Arm> = vec![
        ("motion raw", Box::new(path(0.25, 0.2)), Step::raw(64)),
        ("motion filtered", Box::new(path(0.25, 0.2)), Step::filtered(64, d)),
        ("translation only raw", Box::new(move |j| moved_camera(0.25 * j as f64, 0.2 * k as f64, W, H)), Step::raw(64)),
        ("translation only filtered", Box::new(move |j| moved_camera(0.25 * j as f64, 0.2 * k as f64, W, H)), Step::filtered(64, d)),
        ("rotation only raw", Box::new(move |j| moved_camera(0.25 * k as f64, 0.2 * j as f64, W, H)), Step::raw(64)),
        ("rotation only filtered", Box::new(move |j| moved_camera(0.25 * k as f64, 0.2 * j as f64, W, H)), Step::filtered(64, d)),
        ("motion bilinear raw", Box::new(path(0.25, 0.2)), Step { ts: TemporalSettings { bilinear: true, ..Step::raw(64).ts }, ..Step::raw(64) }),
        ("motion bilinear filtered", Box::new(path(0.25, 0.2)), Step { ts: TemporalSettings { bilinear: true, ..Step::raw(64).ts }, ..Step::filtered(64, d) }),
        ("motion max_age 4 filtered", Box::new(path(0.25, 0.2)), Step::filtered(4, d)),
        ("motion max_age 8 filtered", Box::new(path(0.25, 0.2)), Step::filtered(8, d)),
        ("motion max_age 16 filtered", Box::new(path(0.25, 0.2)), Step::filtered(16, d)),
        ("motion prefilter off", Box::new(path(0.25, 0.2)), Step::filtered(64, DenoiseSettings { prefilter_levels: 0, ..d })),
        ("motion levels 1", Box::new(path(0.25, 0.2)), Step::filtered(64, DenoiseSettings { levels: 1, ..d })),
        ("motion levels 3", Box::new(path(0.25, 0.2)), Step::filtered(64, DenoiseSettings { levels: 3, ..d })),
        ("motion sigma 1", Box::new(path(0.25, 0.2)), Step::filtered(64, DenoiseSettings { sigma_l: 1.0, ..d })),
        ("motion sigma 16", Box::new(path(0.25, 0.2)), Step::filtered(64, DenoiseSettings { sigma_l: 16.0, ..d })),
    ];
    // NE_DIAG_HARD_ONLY=1 skips the noisy arms.
    let noisy = std::env::var_os("NE_DIAG_HARD_ONLY").is_none();
    // NE_DIAG_ARM=<text> runs only the noisy arms whose label contains it.
    let only = std::env::var("NE_DIAG_ARM").ok();
    for (label, cam, step) in arms.into_iter().filter(|a| noisy && only.as_ref().is_none_or(|o| a.0.contains(o.as_str()))) {
        rig.new_history();
        let mut x = None;
        for j in 1..=k {
            x = rig.frame(&cam(j), &light, step, &[], j, j == k);
        }
        show(label, &x.unwrap());
    }
    // Deterministic lighting (the sun's centre only, no sky): no noise, so any error left on the
    // motion path is the history's reprojection (or the filter's), not variance.
    if std::env::var_os("NE_DIAG_NO_HARD").is_some() {
        rig.finish();
        return;
    }
    let hard = ShadeSettings { sky: false, point_sun: true, ..DIRECT };
    let th = rig.target_with(&end, &light, hard, 4);
    let show_h = |label: &str, x: &[[f32; 4]]| {
        let (a, b) = (error(x, &th, &m), error(x, &th, &e));
        eprintln!("hard sun {label}: all {:.4} (bias {:+.4}) | edge {:.4} (bias {:+.4})", a.rel_mse, a.bias, b.rel_mse, b.bias);
    };
    let still = run_still(&mut rig, &end, &light, Step { shade: hard, ..Step::raw(64) }, &[32]);
    show_h("still raw", &still[&32]);
    let hard_arms: Vec<(&str, Step)> = vec![
        ("motion raw", Step { shade: hard, ..Step::raw(64) }),
        ("motion raw bilinear", Step { shade: hard, ts: TemporalSettings { bilinear: true, ..Step::raw(64).ts }, ..Step::raw(64) }),
        ("motion filtered", Step { shade: hard, ..Step::filtered(64, d) }),
    ];
    for (label, step) in hard_arms {
        rig.new_history();
        let mut x = None;
        for j in 1..=k {
            x = rig.frame(&moved_camera(0.25 * j as f64, 0.2 * j as f64, W, H), &light, step, &[], j, j == k);
        }
        let x = x.unwrap();
        show_h(label, &x);
        // Where the raw motion error is: by the pixel's age and a large relative error.
        if label == "motion raw" {
            let terms = rel_terms(&x, &th, &m);
            let mut worst: Vec<(f64, usize)> = terms.iter().enumerate().filter(|(i, _)| m[*i]).map(|(i, &v)| (v, i)).collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            let total: f64 = terms.iter().sum();
            let top: f64 = worst.iter().take(worst.len() / 100).map(|w| w.0).sum();
            eprintln!("  worst 1% of pixels carry {:.0}% of the error; examples (x, y, age, reason, shown, target):", 100.0 * top / total);
            let lit = th.iter().map(|t| lum(*t)).fold(0.0f64, f64::max);
            let (mut sl, mut n) = (0.0, 0.0);
            for (i, t) in th.iter().enumerate() {
                if m[i] {
                    sl += lum(*t);
                    n += 1.0;
                }
            }
            eprintln!("  target luminance: max {lit:.4}, mean {:.4}", sl / n);
            let (cx, cy) = ((worst[0].1 as u32 % W) as i32, (worst[0].1 as u32 / W) as i32);
            for (name, f) in [("target lum x1000", 0), ("shown lum x1000", 1), ("surface (face|mat, plane)", 2)] {
                eprintln!("  {name} around ({cx}, {cy}):");
                for y in cy - 3..=cy + 3 {
                    let row: Vec<String> = (cx - 3..=cx + 3)
                        .map(|px| {
                            let j = (y as u32 * W + px as u32) as usize;
                            match f {
                                0 => format!("{:6.1}", 1000.0 * lum(th[j])),
                                1 => format!("{:6.1}", 1000.0 * lum([x_at(&x, j, 0), x_at(&x, j, 1), x_at(&x, j, 2)])),
                                _ => format!("{:>5}|{:<4}", g[j][0], g[j][1] as i32),
                            }
                        })
                        .collect();
                    eprintln!("    {}", row.join(" "));
                }
            }
            for &(_, i) in worst.iter().take(8) {
                let st = x[i][3] as u32;
                eprintln!("    ({}, {}) age {} reason {} edge {} shown {:.4?} target {:.4?}", i as u32 % W, i as u32 / W, st & 0xFFFF, (st >> 16) & 0xFF, e[i], &x[i][..3], th[i]);
            }
        }
    }
    rig.finish();
}

fn x_at(x: &[[f32; 4]], j: usize, c: usize) -> f64 {
    x[j][c] as f64
}

/// Runs the D4 camera path (0.25 voxel sideways and 0.2 deg of yaw per frame) from a fresh history
/// and returns the shown radiance at frame `k`.
fn run_path(rig: &mut Rig, light: &Lighting, step: Step, k: u32) -> Vec<[f32; 4]> {
    rig.new_history();
    let mut x = None;
    for j in 1..=k {
        x = rig.frame(&moved_camera(0.25 * j as f64, 0.2 * j as f64, W, H), light, step, &[], j, j == k);
    }
    x.unwrap()
}

/// Writes per-pixel relative errors side by side as a heat map PPM: black 0, red, yellow, white at
/// `full` (the relative error, not squared).
fn write_heat(name: &str, w: u32, h: u32, maps: &[Vec<f64>], full: f64) {
    let mut bytes = format!("P6\n{} {h}\n255\n", w * maps.len() as u32).into_bytes();
    for y in 0..h {
        for m in maps {
            for x in 0..w {
                let v = (m[(y * w + x) as usize].sqrt() / full).clamp(0.0, 1.0) * 3.0;
                let c = [v.min(1.0), (v - 1.0).clamp(0.0, 1.0), (v - 2.0).clamp(0.0, 1.0)];
                bytes.extend(c.map(|c| (c * 255.0 + 0.5) as u8));
            }
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results").join(name);
    std::fs::write(&path, bytes).unwrap();
    eprintln!("wrote {}", path.display());
}

/// Illustration for D4 (images only): frame 32 of the motion path at 8 h. `motion_3e_noisy.ppm`:
/// target, the same view held still (filtered, 32 frames), moving raw, moving filtered; and their
/// relative-error heat maps. `motion_3e_hard.ppm`: the same with the hard sun only (no noise): target,
/// still, moving with bilinear history, moving with Catmull-Rom.
#[test]
#[ignore]
fn motion_images() {
    let mut rig = Rig::new();
    let luts = SkyLuts::new(Atmosphere::default());
    let light = lighting(8.0);
    let k = 32u32;
    let end = moved_camera(0.25 * k as f64, 0.2 * k as f64, W, H);
    let d = DenoiseSettings::default();
    let ex = exposure(&light, &luts);
    let t = rig.target(&end, &light);
    let still = run_still(&mut rig, &end, &light, Step::filtered(64, d), &[k]).remove(&k).unwrap();
    let m = surface_mask(&rig.guides());
    let raw = run_path(&mut rig, &light, Step::raw(64), k);
    let filt = run_path(&mut rig, &light, Step::filtered(64, d), k);
    for (label, x) in [("still filtered", &still), ("moving raw", &raw), ("moving filtered", &filt)] {
        eprintln!("{label}: rel_mse {:.4}", error(x, &t, &m).rel_mse);
    }
    write_strip("motion_3e_noisy.ppm", W, H, &[as_image(&t), still.clone(), raw.clone(), filt.clone()], ex);
    let heat: Vec<Vec<f64>> = [&still, &raw, &filt].iter().map(|x| rel_terms(x, &t, &m)).collect();
    write_heat("motion_3e_noisy_error.ppm", W, H, &heat, 0.5);
    let hard = ShadeSettings { sky: false, point_sun: true, ..DIRECT };
    let th = rig.target_with(&end, &light, hard, 4);
    let hs = run_still(&mut rig, &end, &light, Step { shade: hard, ..Step::raw(64) }, &[k]).remove(&k).unwrap();
    let hb = run_path(&mut rig, &light, Step { shade: hard, ts: TemporalSettings { bilinear: true, ..Step::raw(64).ts }, ..Step::raw(64) }, k);
    let hc = run_path(&mut rig, &light, Step { shade: hard, ..Step::raw(64) }, k);
    write_strip("motion_3e_hard.ppm", W, H, &[as_image(&th), hs, hb, hc], ex);
    rig.finish();
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The ADR-0006 Amendment 1 rule on the host, with the rows the GPU receives.
fn host_relit(p: [f64; 3], sun: [f64; 3], boxes: &[Relight]) -> bool {
    boxes.iter().any(|b| {
        let [lo, hi] = b.rows().map(|r| r.map(|v| v as f64));
        let d: [f64; 3] = std::array::from_fn(|a| (lo[a] - p[a]).max(p[a] - hi[a]).max(0.0));
        if dot3(d, d) <= lo[3] * lo[3] {
            return true;
        }
        let (mut enter, mut leave) = (0.0f64, f64::INFINITY);
        for a in 0..3 {
            let (t0, t1) = ((lo[a] - p[a]) / sun[a], (hi[a] - p[a]) / sun[a]);
            enter = enter.max(t0.min(t1));
            leave = leave.min(t0.max(t1));
        }
        enter <= leave
    })
}

/// R1, R2: an edit that changes sunlight outside its rebuilt regions (8 h). The relit pixels equal
/// the host's rule (<= 0.05% differ, >= 100 relit; an arm without boxes must fail the check), and 4
/// frames after the edit the pixels whose converged value changed by > 25% have a filtered relative
/// MSE <= 3 x that of an arm reset at the edit (the arm without boxes must fail it).
#[test]
fn edits_relight_what_they_change() {
    relight_check(DIRECT);
}

/// 3G E2: R1 and R2 with the 3F bounce on (the viewer's lighting), same edit and thresholds.
#[test]
fn edits_relight_what_they_change_with_the_bounce() {
    relight_check(ShadeSettings::default());
}

fn relight_check(shade: ShadeSettings) {
    let mut rig = Rig::new();
    eprintln!("lighting: {shade:?}");
    let cam = street_camera(W, H);
    let light = lighting(8.0);
    let sun = light.sun_dir;
    // An occluder far from the shadow it casts: sun rays from visible lit-facing surfaces.
    let mut occ: Vec<(f64, VoxelCoord, [f64; 3])> = Vec::new();
    for y in (0..H).step_by(4) {
        for x in (0..W).step_by(4) {
            let r = cam.ray(x, y);
            let Some(hit) = trace(&rig.world, &r, 1e9) else { continue };
            let Some(face) = hit.face else { continue };
            let n = face.normal();
            if dot3(n, sun) <= 0.0 {
                continue;
            }
            let p: [f64; 3] = std::array::from_fn(|a| r.origin[a] + hit.t * r.dir[a] + n[a] * 1e-3);
            if let Some(o) = trace(&rig.world, &Ray { origin: p, dir: sun }, 1e9) {
                if o.t > 64.0 {
                    occ.push((o.t, o.voxel, p));
                }
            }
        }
    }
    occ.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(!occ.is_empty(), "no distant occluder found");
    // The box around an occluder must actually let the sun in: a thick wall still shadows after an
    // 8^3 hole. Choose the occluder whose box frees the most samples (the sun ray, restarted where it
    // leaves the box, escapes).
    let box_of = |v: VoxelCoord| ([v.x - 4, v.y - 4, v.z - 4], [v.x + 4, v.y + 4, v.z + 4]);
    let inside = |c: VoxelCoord, lo: [i32; 3], hi: [i32; 3]| {
        let c = [c.x, c.y, c.z];
        (0..3).all(|a| c[a] >= lo[a] && c[a] < hi[a])
    };
    let freed = |lo: [i32; 3], hi: [i32; 3]| {
        occ.iter()
            .filter(|(_, o, p)| {
                inside(*o, lo, hi) && {
                    let leave = (0..3)
                        .map(|a| {
                            let (t0, t1) = ((lo[a] as f64 - p[a]) / sun[a], (hi[a] as f64 - p[a]) / sun[a]);
                            t0.max(t1)
                        })
                        .fold(f64::INFINITY, f64::min);
                    let origin: [f64; 3] = std::array::from_fn(|a| p[a] + sun[a] * (leave + 1e-3));
                    trace(&rig.world, &Ray { origin, dir: sun }, 1e9).is_none()
                }
            })
            .count()
    };
    let mut best: Option<(usize, VoxelCoord, f64)> = None;
    let mut seen = BTreeSet::new();
    for &(t, v, _) in &occ {
        if !seen.insert((v.x, v.y, v.z)) {
            continue;
        }
        let (lo, hi) = box_of(v);
        let n = freed(lo, hi);
        if best.is_none_or(|b| n > b.0) {
            best = Some((n, v, t));
        }
    }
    let (nfreed, v, dist) = best.unwrap();
    let (lo, hi) = box_of(v);
    eprintln!("edit: {} shadowed samples with a distant occluder; removing [{lo:?}, {hi:?}) around {v:?} at distance {dist:.0} frees {nfreed} of them", occ.len());
    let boxes = [Relight::new(lo, hi)];
    let saved = rig.voxels(lo, hi);
    let removed: Vec<_> = saved.iter().map(|&(c, _)| (c, None)).collect();
    let t_old = rig.target_with(&cam, &light, shade, TARGET_FRAMES);
    rig.edit(&removed);
    let t_new = rig.target_with(&cam, &light, shade, TARGET_FRAMES);
    // The host rule on the edited world (surface points by the exact DDA).
    let host: Vec<Option<bool>> = (0..W * H)
        .map(|i| {
            let r = cam.ray(i % W, i / W);
            trace(&rig.world, &r, 1e9).map(|hit| host_relit(std::array::from_fn(|a| r.origin[a] + hit.t * r.dir[a]), sun, &boxes))
        })
        .collect();
    rig.edit(&saved);

    let d = filter_under_test();
    let mut arms = BTreeMap::new();
    for (name, bx, reset) in [("relight", &boxes[..], false), ("none", &[][..], false), ("reset", &[][..], true)] {
        rig.new_history();
        for k in 1..=32u32 {
            rig.frame(&cam, &light, Step { shade, ..Step::filtered(64, d) }, &[], k, false);
        }
        rig.edit(&removed);
        rig.frame(&cam, &light, Step { force_reset: reset, shade, ..Step::filtered(64, d) }, bx, 33, false);
        let st = rig.state();
        let mut x = None;
        for k in 34..=36u32 {
            x = rig.frame(&cam, &light, Step { shade, ..Step::filtered(64, d) }, &[], k, k == 36);
        }
        rig.edit(&saved);
        arms.insert(name, (st, x.unwrap()));
    }

    let mut failed = Vec::new();
    // R1.
    for name in ["relight", "none"] {
        let st = &arms[name].0;
        let (mut relit, mut differ) = (0usize, 0usize);
        for (i, h) in host.iter().enumerate() {
            let (Some(h), r) = (*h, st[i].1) else { continue };
            if r == reason::RELIT {
                relit += 1;
            }
            if (r == reason::RELIT && !h) || (r == reason::ACCEPTED && h) {
                differ += 1;
            }
        }
        let ok = differ * 2000 <= (W * H) as usize && relit >= 100;
        eprintln!("R1 {name}: {relit} px relit, {differ} of {} decisions differ from the host rule {}", W * H, if ok { "(passes)" } else { "(fails)" });
        match (name, ok) {
            ("relight", false) => failed.push("R1".to_string()),
            ("none", true) => failed.push("R1 control without boxes not caught".to_string()),
            _ => {}
        }
    }
    // R2: changed pixels with a history outside the rebuilt regions.
    let st = &arms["relight"].0;
    let changed: Vec<bool> = (0..(W * H) as usize)
        .map(|i| {
            let (a, b) = (lum(t_old[i]), lum(t_new[i]));
            matches!(st[i].1, reason::ACCEPTED | reason::RELIT) && (a - b).abs() > 0.25 * a.max(b)
        })
        .collect();
    let n = changed.iter().filter(|&&c| c).count();
    let e: BTreeMap<&str, Error> = arms.iter().map(|(k, (_, x))| (*k, error(x, &t_new, &changed))).collect();
    eprintln!("R2: {n} changed px; 4 frames after the edit: relight {:.4}, none {:.4}, reset {:.4}", e["relight"].rel_mse, e["none"].rel_mse, e["reset"].rel_mse);
    if n < 100 {
        failed.push(format!("R2 needs >= 100 changed px, got {n}"));
    }
    if e["relight"].rel_mse > 3.0 * e["reset"].rel_mse {
        failed.push("R2".to_string());
    }
    if e["none"].rel_mse <= 3.0 * e["reset"].rel_mse {
        failed.push("R2 control without boxes not caught".to_string());
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// S1: the sun moving at the viewer's day speed (0.0625 deg per frame at 60 fps) from 8 h for 128
/// frames: at the last frame, the filtered error with the age cap is below the one without it, and
/// at most 2 x the still filtered error at age 8 (at the last sun position).
#[test]
fn a_moving_sun_does_not_lag() {
    let mut rig = Rig::new();
    let cam = street_camera(W, H);
    let hour_at = |k: u32| 8.0 + k as f64 * 0.0625 / 15.0;
    let last = 128u32;
    let end = lighting(hour_at(last));
    let t = rig.target(&cam, &end);
    let d = filter_under_test();
    let mut e = BTreeMap::new();
    for (name, tol) in [("cap", TemporalSettings::default().sun_tolerance_deg), ("no cap", f64::INFINITY)] {
        rig.new_history();
        let step = Step { ts: TemporalSettings { max_age: 64, sun_tolerance_deg: tol, ..TemporalSettings::default() }, ..Step::filtered(64, d) };
        let mut x = None;
        for k in 1..=last {
            x = rig.frame(&cam, &lighting(hour_at(k)), step, &[], k, k == last);
        }
        let g = rig.guides();
        let err = error(&x.unwrap(), &t, &surface_mask(&g));
        eprintln!("sun motion {name}: age cap {}, filtered rel_mse {:.4} bias {:+.4}", rig.history.as_ref().unwrap().age_cap, err.rel_mse, err.bias);
        e.insert(name, err.rel_mse);
    }
    let still = run_still(&mut rig, &cam, &end, Step::filtered(64, d), &[8]);
    let g = rig.guides();
    let s8 = error(&still[&8], &t, &surface_mask(&g)).rel_mse;
    eprintln!("still sun, filtered at age 8: {s8:.4}");
    rig.finish();
    assert!(e["cap"] < e["no cap"], "S1: the cap does not beat the lag");
    assert!(e["cap"] <= 2.0 * s8, "S1: the capped error is over 2 x the still one at age 8");
}

/// C1 (and data): GPU time of the temporal pass and of the filter at 1080p, turning camera (validation
/// off gives the timing run: `set NE_NO_VALIDATION=1`).
#[test]
fn denoise_cost_at_1080p() {
    let mut rig = Rig::for_timing(COST_W, COST_H);
    let d = filter_under_test();
    let (tp, dn) = time_filter(&mut rig, d);
    let p = |v: &[f64], q| percentile(v, q).unwrap();
    eprintln!("cost 1080p (validation {}): temporal median {:.3} p90 {:.3} ms | denoise ({} levels) median {:.3} p90 {:.3} ms ({} reps)", rig.g.validation_enabled(), p(&tp, 50.0), p(&tp, 90.0), d.levels, p(&dn, 50.0), p(&dn, 90.0), dn.len());
    let median = p(&dn, 50.0);
    rig.finish();
    assert!(median <= 3.0, "C1: denoise median {median:.3} ms over 3.0 ms");
}

/// The G4 filter record's F5 (`docs/changes/2026-09-26-4b-filter-energy.md`): the filter under test
/// (`NE_FILTER`) against the default at 1080p, interleaved: 7 repetitions of `time_filter` per arm in
/// alternating order (the first pair is warm-up and dropped). Pass: the median of the candidate's
/// per-repetition medians is at most the default's plus 0.05 ms. Run with validation off.
#[test]
#[ignore]
fn filter_cost_against_the_default() {
    const REPS: usize = 7;
    let mut rig = Rig::for_timing(COST_W, COST_H);
    let (d, c) = (DenoiseSettings::default(), filter_under_test());
    let (mut md, mut mc) = (Vec::new(), Vec::new());
    for rep in 0..REPS {
        let order = if rep % 2 == 0 { [(d, true), (c, false)] } else { [(c, false), (d, true)] };
        for (s, is_default) in order {
            let (_, dn) = time_filter(&mut rig, s);
            let m = percentile(&dn, 50.0).unwrap();
            if rep > 0 {
                if is_default { md.push(m) } else { mc.push(m) }
            }
        }
    }
    let (pd, pc) = (percentile(&md, 50.0).unwrap(), percentile(&mc, 50.0).unwrap());
    eprintln!("F5 (validation {}): default {} per-repetition medians {md:.3?} -> {pd:.3} ms | {} {mc:.3?} -> {pc:.3} ms | difference {:+.3} ms", rig.g.validation_enabled(), d.tag(), c.tag(), pc - pd);
    rig.finish();
    assert!(pc <= pd + 0.05, "F5: {} median {pc:.3} ms over the default's {pd:.3} + 0.05 ms", c.tag());
}

const COST_W: u32 = 1920;
const COST_H: u32 = 1080;

/// GPU times (temporal pass, filter) of frames 9-68 of a turning camera at 1080p, from a fresh history.
fn time_filter(rig: &mut Rig, d: DenoiseSettings) -> (Vec<f64>, Vec<f64>) {
    rig.new_history();
    let light = lighting(8.0);
    let mut timer = GpuTimer::new(&rig.g, 1, 2).unwrap();
    let step = Step::filtered(64, d);
    let (mut tp, mut dn) = (Vec::new(), Vec::new());
    for k in 1..=68u32 {
        let cam = moved_camera(0.0, 0.2 * k as f64, COST_W, COST_H);
        let mut sub = rig.sub.take().unwrap();
        let cmd = sub.begin(&rig.g, &rig.tl).unwrap();
        timer.reset(&rig.g, cmd, 0);
        rig.record(cmd, &cam, &light, step, &[], k, Some((&timer, 0)));
        let v = sub.submit(&rig.g, &mut rig.tl, cmd, &[]).unwrap();
        rig.tl.wait(&rig.g, v, u64::MAX).unwrap();
        rig.sub = Some(sub);
        let ms = timer.read(&rig.g, 0).unwrap().unwrap();
        if k > 8 {
            tp.push(ms[0]);
            dn.push(ms[1]);
        }
    }
    timer.destroy(&rig.g);
    (tp, dn)
}

/// Diagnostic for C1: the filter's median time with stages added one at a time.
#[test]
#[ignore]
fn denoise_cost_breakdown() {
    let mut rig = Rig::for_timing(COST_W, COST_H);
    let d = DenoiseSettings::default();
    let arms = [
        ("init + remodulate (levels 0)", DenoiseSettings { levels: 0, prefilter_levels: 0, ..d }),
        ("+ level 1 (no prefilter)", DenoiseSettings { levels: 1, prefilter_levels: 0, ..d }),
        ("+ prefilter on level 1", DenoiseSettings { levels: 1, ..d }),
        ("+ level 2 (the default)", d),
        ("default, moments from age 1 (no 7x7 variance)", DenoiseSettings { moments_age: 1, ..d }),
        ("default without the variance pre-blur", DenoiseSettings { variance_blur: false, ..d }),
        ("default again (drift check)", d),
    ];
    for (label, s) in arms {
        let (_, dn) = time_filter(&mut rig, s);
        eprintln!("{label}: median {:.3} ms, p90 {:.3} ms", percentile(&dn, 50.0).unwrap(), percentile(&dn, 90.0).unwrap());
    }
    rig.finish();
}

/// Diagnostic: the sliders swept on stills (8 h and 18.25 h) and the ablations: unfiltered, temporal
/// only (the raw accumulation), spatial only (max_age 1 + filter), both; max_age 16 to 128.
#[test]
#[ignore]
fn sweep_sliders() {
    let mut rig = Rig::new();
    let cam = street_camera(W, H);
    for hour in [8.0, 18.25] {
        let light = lighting(hour);
        let t = rig.target(&cam, &light);
        let g = {
            run_still(&mut rig, &cam, &light, Step::raw(64), &[1]);
            rig.guides()
        };
        let (m, e) = (surface_mask(&g), edge_mask(&g, W, H, 2));
        let report = |label: String, x: &BTreeMap<u32, Vec<[f32; 4]>>| {
            let cols: Vec<String> = x.iter().map(|(k, v)| {
                let (a, b) = (error(v, &t, &m), error(v, &t, &e));
                format!("@{k} {:.4} (edge {:.4}, bias {:+.4})", a.rel_mse, b.rel_mse, a.bias)
            }).collect();
            eprintln!("hour {hour} {label}: {}", cols.join(" | "));
        };
        let ages = [1u32, 4, 64];
        report("unfiltered / temporal only".into(), &run_still(&mut rig, &cam, &light, Step::raw(64), &ages));
        report("spatial only (max_age 1)".into(), &run_still(&mut rig, &cam, &light, Step::filtered(1, DenoiseSettings::default()), &ages));
        for levels in 0..=6 {
            report(format!("levels {levels} sigma 4"), &run_still(&mut rig, &cam, &light, Step::filtered(64, DenoiseSettings { levels, ..DenoiseSettings::default() }), &ages));
        }
        for sigma_l in [1.0, 2.0, 8.0, 16.0, 1e6] {
            report(format!("levels 5 sigma {sigma_l}"), &run_still(&mut rig, &cam, &light, Step::filtered(64, DenoiseSettings { sigma_l, ..DenoiseSettings::default() }), &ages));
        }
        for prefilter_levels in [0, 1, 2, 3] {
            report(format!("prefilter_levels {prefilter_levels}"), &run_still(&mut rig, &cam, &light, Step::filtered(64, DenoiseSettings { prefilter_levels, ..DenoiseSettings::default() }), &ages));
        }
        for moments_age in [2, 8, 16] {
            report(format!("moments_age {moments_age}"), &run_still(&mut rig, &cam, &light, Step::filtered(64, DenoiseSettings { moments_age, ..DenoiseSettings::default() }), &[1, 4, 8, 16, 64]));
        }
        for levels in [2, 3, 4] {
            for moments_age in [4, 8] {
                for sigma_l in [3.0, 4.0, 6.0] {
                    report(format!("levels {levels} moments_age {moments_age} sigma {sigma_l}"), &run_still(&mut rig, &cam, &light, Step::filtered(64, DenoiseSettings { levels, sigma_l, moments_age, ..DenoiseSettings::default() }), &[1, 4, 8, 16, 64]));
                }
            }
        }
        for max_age in [16, 32, 128] {
            report(format!("max_age {max_age}"), &run_still(&mut rig, &cam, &light, Step::filtered(max_age, DenoiseSettings::default()), &[16, 64, 128]));
        }
    }
    rig.finish();
}
