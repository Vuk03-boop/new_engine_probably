//! Phase 3G GPU tests (the M3 gate): the real-time path as the viewer runs it (sun, the corrected
//! sky, one bounce, the temporal pass and the filter) against the one-bounce reference at 1080p, and
//! the cost of the frames after a history reset. Criteria are declared in
//! `docs/changes/2026-09-25-phase3g-gate.md`. Like the other GPU tests they need the Vulkan SDK and an
//! RT GPU and fail (never skip) without them.
//!
//! With `NE_GATE_DIR` set, the references are cached there (keyed by every parameter; a mismatch
//! renders again) and the display images for FLIP are written there as PPM
//! (`engine/results/phase3g/flip.py` compares them).
//! Run: `cargo test --release -j 2 -p gpu --test gate -- --test-threads=1 --nocapture`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::alloc::Allocator;
use gpu::debug_view::{targets_to_read, Tables};
use gpu::denoise::{Denoise, DenoiseBindings, DenoiseFaults, DenoiseSettings, DenoiseTargets};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::raster::{Bindings as RasterBindings, Camera, Faults, Raster, Targets};
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::scene::GpuScene;
use gpu::shade::{self, Shade, ShadeBindings, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::temporal::{History, Temporal, TemporalBindings, TemporalFaults, TemporalSettings};
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, Timeline};
use light::reference::{albedos, Lighting, Settings};
use light::sky::SkyLuts;
use light::sky_ref::SkyReference;
use light::sun::REFERENCE_TIMES;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::{scene, BrickKey, World};

/// A shown image and its valid-pixel mask.
type Shown = (Vec<[f32; 4]>, Vec<bool>);

const NEAR: f64 = 0.1;
/// The viewer's shade seed.
const SEED: u32 = 0x3B;
/// The reference's seed (independent of the real-time streams).
const REF_SEED: u32 = 0x3C;
const W: u32 = 1920;
const H: u32 = 1080;
/// Reference samples per pixel (twilight: `REF_SPP_TWILIGHT`).
const REF_SPP: u32 = 4096;
const REF_SPP_TWILIGHT: u32 = 16_384;
/// Reference frames per submission (short enough for the display driver's watchdog at 1080p).
const REF_PER_SUBMIT: u32 = 8;
/// Bump when a camera or the reference settings change, so cached references are not reused.
const CACHE_VERSION: u32 = 1;
/// Motion paths: hours and the frames compared.
const MOTION_HOURS: [(&str, f64); 2] = [("morning", 8.0), ("dusk", 17.75)];
const MOTION_FRAMES: [u32; 3] = [8, 16, 32];

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3G needs the ray-tracing device (P-001)");
    g
}

fn lighting(hour: f64) -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(hour))
}

/// The 3A reference cameras.
fn cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        ("street", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)),
        ("low", Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, NEAR)),
    ]
}

/// The 3E D4 path: the street camera moved sideways by `dx` voxels and turned by `yaw_deg`.
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

fn motion_camera(k: u32) -> Camera {
    moved_camera(0.25 * k as f64, 0.2 * k as f64, W, H)
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

struct Bound {
    rb: RasterBindings,
    sb: ShadeBindings,
    tb: TemporalBindings,
    db: DenoiseBindings,
}

/// The viewer's frame (sky table when the sun moved, G-buffer, shade, temporal pass, filter) at one
/// size, plus the reference renderer on the same TLAS.
struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    luts: SkyLuts,
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
    reference: Reference,
    sub: Option<Submitter>,
    sky_sun: Option<[f64; 3]>,
    /// The filter's settings (the default, except in diagnostics).
    dn_settings: DenoiseSettings,
    w: u32,
    h: u32,
}

impl Rig {
    fn new() -> Rig {
        Self::with_gpu(gpu(), W, H)
    }

    /// For timing runs: validation may be off (`NE_NO_VALIDATION`).
    fn for_timing() -> Rig {
        let g = Gpu::new().expect("an RT-capable Vulkan device");
        assert!(g.ray_tracing());
        Self::with_gpu(g, W, H)
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
        // The viewer's sky: corrected from the baked reference (S-020).
        let mut luts = SkyLuts::new(Atmosphere::default());
        let r = SkyReference::load_default(&luts.atmosphere).expect("the baked sky reference (run the sky_bake tool)");
        luts.apply_reference(&r).unwrap();
        let (sky, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &luts).unwrap();
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
        let reference = Reference::new(&g).unwrap();
        let sub = Submitter::new(&g).unwrap();
        let mut rig = Rig {
            g,
            alloc,
            tl,
            up,
            luts,
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
            reference,
            sub: Some(sub),
            sky_sun: None,
            dn_settings: DenoiseSettings::default(),
            w,
            h,
        };
        rig.rebind();
        rig
    }

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

    /// Records one viewer frame; `filtered` runs the filter (the viewer's default). Timer passes:
    /// 0 shade, 1 temporal, 2 filter.
    fn record(&mut self, cmd: vk::CommandBuffer, cam: &Camera, light: &Lighting, filtered: bool, frame: u32, timer: Option<&GpuTimer>) {
        let (g, targets, out, s, b) = (&self.g, self.targets.as_ref().unwrap(), self.out.as_ref().unwrap(), self.scene.as_ref().unwrap(), self.bound.as_ref().unwrap());
        if self.sky_sun != Some(light.sun_dir) {
            self.sky_pass.as_ref().unwrap().record(g, cmd, light.sun_dir);
            self.sky_sun = Some(light.sun_dir);
        }
        self.raster.record(g, cmd, targets, &s.meshes, &b.rb, cam, Faults::default());
        targets_to_read(g, cmd, targets);
        let stamp = |pass: u32, begin: bool| {
            if let Some(t) = timer {
                if begin {
                    t.begin_pass(g, cmd, 0, pass);
                } else {
                    t.end_pass(g, cmd, 0, pass);
                }
            }
        };
        stamp(0, true);
        self.shade.record(g, cmd, &b.sb, out, &shade::Params::new(cam, light, ShadeSettings::default(), ShadeFaults::default(), frame, SEED));
        stamp(0, false);
        let h = self.history.as_mut().unwrap();
        stamp(1, true);
        self.temporal.record(g, cmd, &b.tb, h, cam, light.sun_dir, TemporalSettings::default(), TemporalFaults::default(), false, &[]);
        stamp(1, false);
        stamp(2, true);
        if filtered {
            self.denoise.record(g, cmd, &b.db, h, self.dn.as_ref().unwrap(), self.dn_settings, DenoiseFaults::default());
        }
        stamp(2, false);
    }

    /// Runs one frame; returns the shown radiance when `read`.
    fn frame(&mut self, cam: &Camera, light: &Lighting, filtered: bool, frame: u32, read: bool) -> Option<Vec<[f32; 4]>> {
        let mut sub = self.sub.take().unwrap();
        let cmd = sub.begin(&self.g, &self.tl).unwrap();
        self.record(cmd, cam, light, filtered, frame, None);
        let v = sub.submit(&self.g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(&self.g, v, u64::MAX).unwrap();
        self.sub = Some(sub);
        read.then(|| self.out.as_ref().unwrap().read(&self.g, &mut self.alloc, &mut self.tl).unwrap())
    }

    fn guides(&mut self) -> Vec<[u32; 4]> {
        self.history.as_ref().unwrap().read_guides(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    /// The one-bounce reference (mean and standard error per pixel), from the cache when present.
    fn reference(&mut self, key: &str, cam: &Camera, light: &Lighting, spp: u32) -> Reference1 {
        let header = format!("NE3GREF v{CACHE_VERSION} {key} {}x{} spp {spp} seed {REF_SEED} bounces 1 sun {:?}\n", cam.width, cam.height, light.sun_dir);
        let path = gate_dir().map(|d| d.join(format!("ref_{key}.bin")));
        if let Some(r) = path.as_ref().and_then(|p| Reference1::load(p, &header)) {
            eprintln!("reference {key}: cached");
            return r;
        }
        let out = RefAccum::new(&self.g, &mut self.alloc, cam.width, cam.height).unwrap();
        let s = self.scene.as_ref().unwrap();
        let b = self.reference.bind(&self.g, s.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &out).unwrap();
        let t = std::time::Instant::now();
        let s = Settings { max_bounces: 1, ..Settings::default() };
        self.reference.accumulate(&self.g, &mut self.tl, &b, &out, cam, light, &s, RefFaults::default(), REF_SEED, 0..spp, 0, REF_PER_SUBMIT).unwrap();
        let img = self.reference.read(&self.g, &mut self.alloc, &mut self.tl, &out, spp).unwrap();
        assert_eq!(img.bad_samples, 0);
        self.g.wait_idle().unwrap();
        b.destroy(&self.g);
        out.free(&self.g, &mut self.alloc);
        let a = img.accum;
        let r = Reference1 { mean: (0..a.sum.len()).map(|i| a.mean(i)).collect(), se: (0..a.sum.len()).map(|i| a.std_error(i)).collect() };
        eprintln!("reference {key}: {spp} spp in {:.1} s", t.elapsed().as_secs_f64());
        if let Some(p) = path {
            r.save(&p, &header);
        }
        r
    }

    /// The viewer's display exposure (`light_exposure` at 0 stops): 1 / (sun + mean sky radiance).
    fn exposure(&self, light: &Lighting) -> f64 {
        let e = self.luts.ground_irradiance(light.sun_dir[1]);
        1.0 / ((lum(light.sun_at_ground) + lum(e)) / std::f64::consts::PI + 1e-7)
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
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

fn gate_dir() -> Option<PathBuf> {
    std::env::var_os("NE_GATE_DIR").map(PathBuf::from)
}

/// A reference image: the per-pixel mean and its standard error.
struct Reference1 {
    mean: Vec<[f64; 3]>,
    se: Vec<[f64; 3]>,
}

impl Reference1 {
    /// The header line, then per pixel the mean and the standard error as 6 f32 (little-endian).
    fn save(&self, path: &std::path::Path, header: &str) {
        let mut b = header.as_bytes().to_vec();
        for (m, s) in self.mean.iter().zip(&self.se) {
            for v in m.iter().chain(s) {
                b.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b).unwrap();
    }

    fn load(path: &std::path::Path, header: &str) -> Option<Reference1> {
        let b = std::fs::read(path).ok()?;
        let body = b.strip_prefix(header.as_bytes())?;
        if body.len() % 24 != 0 {
            return None;
        }
        let f = |i: usize| f32::from_le_bytes(body[4 * i..4 * i + 4].try_into().unwrap()) as f64;
        let n = body.len() / 24;
        Some(Reference1 { mean: (0..n).map(|p| [f(6 * p), f(6 * p + 1), f(6 * p + 2)]).collect(), se: (0..n).map(|p| [f(6 * p + 3), f(6 * p + 4), f(6 * p + 5)]).collect() })
    }
}

const FACE_NONE: u32 = 7;

fn surface_mask(guides: &[[u32; 4]]) -> Vec<bool> {
    guides.iter().map(|g| g[0] & 7 != FACE_NONE).collect()
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Relative MSE (per channel, epsilon = (0.1 x the channel's mean)^2) and the relative bias of the
/// mean luminance over the masked pixels: the 3E metric.
#[derive(Clone, Copy, Debug)]
struct Error {
    rel_mse: f64,
    bias: f64,
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
    Error { rel_mse: se / (3.0 * n), bias: (lx - lt) / lt.max(1e-30) }
}

/// The reference's own noise in the same metric (expected to add equally to every arm's error).
fn ref_noise(r: &Reference1, mask: &[bool]) -> f64 {
    let idx: Vec<usize> = (0..r.mean.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| r.mean[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-12));
    idx.iter().map(|&i| (0..3).map(|c| r.se[i][c].powi(2) / (r.mean[i][c].powi(2) + eps[c])).sum::<f64>()).sum::<f64>() / (3.0 * n)
}

/// Writes the display image (exposure, the ACES fit, sRGB: the viewer's light view) as a PPM into
/// `NE_GATE_DIR`, for FLIP.
fn write_display(name: &str, w: u32, h: u32, px: &dyn Fn(usize) -> [f64; 3], exposure: f64) {
    let Some(dir) = gate_dir() else { return };
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

fn shown(x: &[[f32; 4]]) -> impl Fn(usize) -> [f64; 3] + '_ {
    move |i| [x[i][0] as f64, x[i][1] as f64, x[i][2] as f64]
}

/// Runs a still camera from a fresh history (frame index = age); returns the shown radiance at `ages`.
fn run_still(rig: &mut Rig, cam: &Camera, light: &Lighting, filtered: bool, ages: &[u32]) -> BTreeMap<u32, Vec<[f32; 4]>> {
    rig.new_history();
    let last = *ages.iter().max().unwrap();
    let mut out = BTreeMap::new();
    for k in 1..=last {
        if let Some(x) = rig.frame(cam, light, filtered, k, ages.contains(&k)) {
            out.insert(k, x);
        }
    }
    out
}

/// Q1, Q2 and the still images for Q4: both reference cameras at the five reference times, the
/// shown image (filtered, the viewer's default) and the unfiltered accumulation against the
/// one-bounce reference at ages 1, 4, 16, 64.
#[test]
fn stills_against_the_reference() {
    let mut rig = Rig::new();
    let mut failed = Vec::new();
    for (time, hour) in REFERENCE_TIMES {
        let light = lighting(hour);
        let exposure = rig.exposure(&light);
        let spp = if time == "twilight" { REF_SPP_TWILIGHT } else { REF_SPP };
        for (cname, cam) in cameras(W, H) {
            let key = format!("{cname}_{time}");
            let r = rig.reference(&key, &cam, &light, spp);
            let raw = run_still(&mut rig, &cam, &light, false, &[1, 4, 8, 16, 64]);
            let m = surface_mask(&rig.guides());
            let filt = run_still(&mut rig, &cam, &light, true, &[1, 4, 16, 64]);
            eprintln!("{key}: {} surface px, reference noise rel_mse {:.2e}", m.iter().filter(|&&b| b).count(), ref_noise(&r, &m));
            for (age, gain) in [(1u32, 8u32), (4, 4), (16, 4), (64, 1)] {
                let (f, a, rg) = (error(&filt[&age], &r.mean, &m), error(&raw[&age], &r.mean, &m), error(&raw[&(age * gain)], &r.mean, &m));
                let q1 = f.bias.abs() <= 0.02;
                let q2 = f.rel_mse <= rg.rel_mse;
                eprintln!(
                    "  age {age:2}: filtered rel_mse {:.4} bias {:+.4} | raw {:.4} bias {:+.4} | raw at age {:2} {:.4} | Q1 {} Q2 {}",
                    f.rel_mse,
                    f.bias,
                    a.rel_mse,
                    a.bias,
                    age * gain,
                    rg.rel_mse,
                    if q1 { "pass" } else { "FAIL" },
                    if q2 { "pass" } else { "FAIL" }
                );
                if !q1 {
                    failed.push(format!("Q1 {key} age {age}"));
                }
                if !q2 {
                    failed.push(format!("Q2 {key} age {age}"));
                }
            }
            for age in [1u32, 16, 64] {
                write_display(&format!("still_{key}_raw_{age}"), W, H, &shown(&raw[&age]), exposure);
                write_display(&format!("still_{key}_filt_{age}"), W, H, &shown(&filt[&age]), exposure);
            }
            write_display(&format!("still_{key}_ref"), W, H, &|i| r.mean[i], exposure);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Q3 and the motion images for Q4: the 3E D4 path at 8 h and 17.75 h, frames 8, 16, 32: filtered
/// mean luminance within 1% of the raw one in the same frame and within 2% of the reference.
#[test]
fn motion_against_the_reference() {
    let mut rig = Rig::new();
    let mut failed = Vec::new();
    for (time, hour) in MOTION_HOURS {
        let light = lighting(hour);
        let exposure = rig.exposure(&light);
        let refs: Vec<Reference1> = MOTION_FRAMES.iter().map(|&k| rig.reference(&format!("motion_{time}_{k}"), &motion_camera(k), &light, REF_SPP)).collect();
        // (filtered, frame) -> (the shown image, its valid-pixel mask).
        let mut shown_at: BTreeMap<(bool, u32), Shown> = BTreeMap::new();
        for filtered in [false, true] {
            rig.new_history();
            for k in 1..=*MOTION_FRAMES.last().unwrap() {
                if let Some(x) = rig.frame(&motion_camera(k), &light, filtered, k, MOTION_FRAMES.contains(&k)) {
                    let m = surface_mask(&rig.guides());
                    shown_at.insert((filtered, k), (x, m));
                }
            }
        }
        for (i, &k) in MOTION_FRAMES.iter().enumerate() {
            let r = &refs[i];
            let (x, m) = &shown_at[&(false, k)];
            let (f, _) = &shown_at[&(true, k)];
            let (fe, re) = (error(f, &r.mean, m), error(x, &r.mean, m));
            let ok = (fe.bias - re.bias).abs() <= 0.01 && fe.bias.abs() <= 0.02;
            eprintln!(
                "motion {time} frame {k:2}: filtered rel_mse {:.4} bias {:+.4} | raw {:.4} bias {:+.4} | reference noise {:.2e} | Q3 {}",
                fe.rel_mse,
                fe.bias,
                re.rel_mse,
                re.bias,
                ref_noise(r, m),
                if ok { "pass" } else { "FAIL" }
            );
            if !ok {
                failed.push(format!("Q3 {time} frame {k}"));
            }
            let key = format!("motion_{time}_{k}");
            write_display(&format!("{key}_raw"), W, H, &shown(x), exposure);
            write_display(&format!("{key}_filt"), W, H, &shown(f), exposure);
            write_display(&format!("{key}_ref"), W, H, &|j| r.mean[j], exposure);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// T (data, the reset-frame tail of S-019): GPU time of the shade pass, the temporal pass and the
/// filter at 1080p on frames 1-8 after a fresh history against later frames, street camera at 8 h,
/// 6 histories of 40 frames. Validation off gives the timing run: `$env:NE_NO_VALIDATION = "1"`.
#[test]
fn reset_frame_cost_at_1080p() {
    const RUNS: usize = 6;
    const FRAMES: u32 = 40;
    let mut rig = Rig::for_timing();
    let validation = rig.g.validation_enabled();
    let light = lighting(8.0);
    let cam = &cameras(W, H)[0].1;
    let mut timer = GpuTimer::new(&rig.g, 1, 3).unwrap();
    // by_age[a][pass]: samples at age a (1-based, 0 unused).
    let mut by_age = vec![[Vec::new(), Vec::new(), Vec::new()]; FRAMES as usize + 1];
    for run in 0..RUNS + 1 {
        rig.new_history();
        for k in 1..=FRAMES {
            let mut sub = rig.sub.take().unwrap();
            let cmd = sub.begin(&rig.g, &rig.tl).unwrap();
            timer.reset(&rig.g, cmd, 0);
            rig.record(cmd, cam, &light, true, k, Some(&timer));
            let v = sub.submit(&rig.g, &mut rig.tl, cmd, &[]).unwrap();
            rig.tl.wait(&rig.g, v, u64::MAX).unwrap();
            rig.sub = Some(sub);
            let ms = timer.read(&rig.g, 0).unwrap().unwrap();
            // Run 0 warms the pipelines and caches.
            if run > 0 {
                for p in 0..3 {
                    by_age[k as usize][p].push(ms[p]);
                }
            }
        }
    }
    let p = |v: &[f64], q| percentile(v, q).unwrap();
    let pool = |ages: std::ops::RangeInclusive<usize>, pass: usize| ages.flat_map(|a| by_age[a][pass].clone()).collect::<Vec<f64>>();
    for (a, [s, t, d]) in by_age.iter().enumerate().take(11).skip(1) {
        eprintln!("age {a:2} (validation {validation}): shade {:.3} temporal {:.3} filter {:.3} ms (medians of {})", p(s, 50.0), p(t, 50.0), p(d, 50.0), s.len());
    }
    let steady: Vec<Vec<f64>> = (0..3).map(|pass| pool(9..=FRAMES as usize, pass)).collect();
    let young: Vec<Vec<f64>> = (0..3).map(|pass| pool(1..=8, pass)).collect();
    for (name, v) in [("ages 1-8", &young), ("ages 9-40", &steady)] {
        eprintln!(
            "{name}: shade p50 {:.3} p90 {:.3} | temporal p50 {:.3} p90 {:.3} | filter p50 {:.3} p90 {:.3} ms ({} frames)",
            p(&v[0], 50.0),
            p(&v[0], 90.0),
            p(&v[1], 50.0),
            p(&v[1], 90.0),
            p(&v[2], 50.0),
            p(&v[2], 90.0),
            v[0].len()
        );
    }
    timer.destroy(&rig.g);
    rig.finish();
}

/// Per-pixel relative squared error (the `error` metric's terms; 0 outside the mask).
fn rel_terms(x: &[[f32; 4]], t: &[[f64; 3]], mask: &[bool]) -> Vec<f64> {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| t[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-12));
    (0..t.len())
        .map(|i| if mask[i] { (0..3).map(|c| (x[i][c] as f64 - t[i][c]).powi(2) / (t[i][c].powi(2) + eps[c])).sum::<f64>() / 3.0 } else { 0.0 })
        .collect()
}

/// Diagnostic for the Q2 miss on the low camera at midday (`NE_DIAG_ARM` = camera_time, default
/// low_midday): the error at ages 1, 4, 16, 64 split into lighting-edge pixels (the reference's 3x3
/// luminance max / min > 1.5) and the rest, with the mean-luminance bias, for the unfiltered
/// accumulation and for filter settings varied one at a time (`NE_DIAG_SWEEP=prefilter_age`: the
/// 3G prefilter age instead).
#[test]
#[ignore]
fn diagnose_q2() {
    let arm = std::env::var("NE_DIAG_ARM").unwrap_or_else(|_| "low_midday".into());
    let (cname, time) = arm.split_once('_').unwrap();
    let hour = REFERENCE_TIMES.iter().find(|t| t.0 == time).unwrap().1;
    let mut rig = Rig::new();
    let light = lighting(hour);
    let cam = cameras(W, H).into_iter().find(|c| c.0 == cname).unwrap().1;
    let spp = if time == "twilight" { REF_SPP_TWILIGHT } else { REF_SPP };
    let r = rig.reference(&arm, &cam, &light, spp);
    let ages = [1u32, 4, 16, 64];
    let raw = run_still(&mut rig, &cam, &light, false, &ages);
    let m = surface_mask(&rig.guides());
    let (w, h) = (W as i32, H as i32);
    let edge: Vec<bool> = (0..(W * H) as usize)
        .map(|i| {
            let (x, y) = (i as i32 % w, i as i32 / w);
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (qx, qy) = ((x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1));
                    let l = lum(r.mean[(qy * w + qx) as usize]);
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
            m[i] && hi > 1.5 * lo.max(1e-12)
        })
        .collect();
    let total = m.iter().filter(|&&b| b).count() as f64;
    let ne = edge.iter().filter(|&&b| b).count();
    eprintln!("{arm}: {total} surface px, {ne} lighting-edge px");
    let report = |name: &str, x: &BTreeMap<u32, Vec<[f32; 4]>>| {
        for age in ages {
            let t = rel_terms(&x[&age], &r.mean, &m);
            let all = t.iter().sum::<f64>() / total;
            let on_edge = (0..t.len()).filter(|&i| edge[i]).map(|i| t[i]).sum::<f64>() / total;
            let bias = error(&x[&age], &r.mean, &m).bias;
            eprintln!("  {name:28} age {age:2}: rel_mse {all:.4} = edges {on_edge:.4} + rest {:.4}, bias {bias:+.4}", all - on_edge);
        }
    };
    report("raw", &raw);
    let d = DenoiseSettings::default();
    let arms: Vec<(String, DenoiseSettings)> = if std::env::var("NE_DIAG_SWEEP").as_deref() == Ok("prefilter_age") {
        [2u32, 4, 8, 16, 32, u32::MAX].iter().map(|&a| (format!("prefilter_age {a}"), DenoiseSettings { prefilter_age: a, ..d })).collect()
    } else {
        [
        ("filter default", d),
        ("prefilter_levels 0", DenoiseSettings { prefilter_levels: 0, ..d }),
        ("variance_blur off", DenoiseSettings { variance_blur: false, ..d }),
        ("sigma_l 2", DenoiseSettings { sigma_l: 2.0, ..d }),
        ("sigma_l 1", DenoiseSettings { sigma_l: 1.0, ..d }),
        ("levels 1", DenoiseSettings { levels: 1, ..d }),
        ("levels 0 (control: = raw)", DenoiseSettings { levels: 0, ..d }),
        ]
        .iter()
        .map(|(n, s)| (n.to_string(), *s))
        .collect()
    };
    for (name, s) in arms {
        rig.dn_settings = s;
        let f = run_still(&mut rig, &cam, &light, true, &ages);
        report(&name, &f);
    }
    rig.finish();
}
