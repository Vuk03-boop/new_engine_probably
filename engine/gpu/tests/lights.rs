//! Phase 4B GPU tests: the street's emitters in the real-time path (`gpu::shade` with emitters, the
//! light view's emission, the lights switch, the automatic exposure's sums and the emitter relight
//! rules) against the reference. Criteria G1–G6 and M1 of
//! `docs/changes/2026-09-25-phase4b-many-lights.md`. Like the other GPU tests they need the Vulkan
//! SDK and an RT GPU and fail (never skip) without them.
//!
//! With `NE_GATE_DIR` set, G4's and G5's references are cached there (keyed by every parameter) and
//! G4's display images for FLIP are written there as PPM (`engine/results/phase4b/flip.py`).
//! Run: `cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture`; M1 (the
//! equal-time curve) is ignored there and runs with validation off:
//! `set NE_NO_VALIDATION=1 && cargo test --release -j 2 -p gpu --test lights -- --ignored --test-threads=1 --nocapture equal_time_curve`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::alloc::{Allocator, Kind};
use gpu::debug_view::{targets_to_read, DebugView, Lighting as ViewLighting, Source, Tables, View};
use gpu::denoise::{Denoise, DenoiseBindings, DenoiseFaults, DenoiseSettings, DenoiseTargets};
use gpu::exposure::{sample_pixels, Exposure, ExposureBindings, ExposureTargets};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::raster::{Bindings as RasterBindings, Camera, Faults, Raster, Targets};
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::scene::{affected_regions, region_meshes, GpuScene};
use gpu::shade::{self, Shade, ShadeBindings, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::temporal::{changed_power, reason, History, Relight, Temporal, TemporalBindings, TemporalFaults, TemporalSettings, EMITTER_TOLERANCE};
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, Timeline};
use light::emitters::{luminance, EmitterTable};
use light::exposure::{exposure_from_sums, log_sums, metric_exposure, DARK_FRACTION};
use light::reference::{albedos, Accum, Lighting, Settings};
use light::sky::SkyLuts;
use light::sky_ref::SkyReference;
use light::{Atmosphere, SunPath};
use memory::Category;
use world::dims::VOXEL_SIZE_M;
use world::reference::trace;
use world::scene::{street_night, Dressing};
use world::{BrickKey, MaterialId, Transaction, VoxelCoord, World};

const NEAR: f64 = 0.1;
/// The real-time streams' seed.
const SEED: u32 = 0x4B;
/// The reference's seed (independent of the real-time streams), and a second one for the null.
const REF_SEED: u32 = 0x4C;
const NULL_SEED: u32 = 0x4D;
const NIGHT: f64 = 21.0;
const BLUE_HOUR: f64 = 18.5;
/// Bump when a camera or the reference settings change, so cached references are not reused.
const CACHE_VERSION: u32 = 1;
const FACE_NONE: u32 = 7;

fn gpu_validated() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "4B needs the ray-tracing device (P-001)");
    g
}

fn lighting(hour: f64) -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(hour))
}

/// 4A's cameras: the street view and the 3A low camera.
fn cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = street_night(Dressing::Lamps);
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![("street", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)), ("low", Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, NEAR))]
}

fn street_camera(w: u32, h: u32) -> Camera {
    cameras(w, h).remove(0).1
}

/// The reference transport of the real-time path at night: sun, sky, one bounce, emitters at both
/// vertices, emission off (the real-time radiance is reflected light only).
fn ref_settings() -> Settings {
    Settings { max_bounces: 1, emitters_direct: true, emitters_indirect: true, ..Settings::default() }
}

/// The viewer's lighting with the lights on.
fn lit() -> ShadeSettings {
    ShadeSettings { emitters: true, ..ShadeSettings::default() }
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

fn lum(c: [f64; 3]) -> f64 {
    luminance(c)
}

fn lum4(c: [f32; 4]) -> f64 {
    lum([c[0] as f64, c[1] as f64, c[2] as f64])
}

/// One frame of the viewer's path.
#[derive(Clone, Copy, Debug)]
struct Step {
    shade: ShadeSettings,
    faults: ShadeFaults,
    ts: TemporalSettings,
    /// Run the temporal pass (else the shade pass's radiance is read as is).
    temporal: bool,
    filter: bool,
    lights: bool,
    force_reset: bool,
    /// Bind the shade pass without the table (`Shade::bind`): the M3-unchanged control.
    plain: bool,
}

impl Step {
    /// The viewer's frame with the lights on (bounce, emitters, temporal defaults, the filter).
    fn viewer() -> Step {
        Step { shade: lit(), faults: ShadeFaults::default(), ts: TemporalSettings::default(), temporal: true, filter: true, lights: true, force_reset: false, plain: false }
    }

    /// The shade pass alone.
    fn shade(shade: ShadeSettings) -> Step {
        Step { shade, temporal: false, filter: false, ..Step::viewer() }
    }
}

struct Bound {
    rb: RasterBindings,
    sb: ShadeBindings,
    sb_plain: ShadeBindings,
    tb: TemporalBindings,
    db: DenoiseBindings,
    xb: ExposureBindings,
}

/// The viewer's frame on `street_night` at one size (sky table when the sun moved, G-buffer, shade
/// with the scene's emitter table, temporal pass, filter, exposure sums), plus the reference on the
/// same TLAS and table, and edits through the 1C pipeline.
struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    world: World,
    pipeline: Pipeline,
    emission: Vec<[f64; 3]>,
    scene: Option<GpuScene>,
    tables: Option<Tables>,
    mats: Option<RefMaterials>,
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
    targets: Option<Targets>,
    out: Option<ShadeTargets>,
    history: Option<History>,
    dn: Option<DenoiseTargets>,
    xo: Option<ExposureTargets>,
    bound: Option<Bound>,
    raster: Raster,
    shade: Shade,
    temporal: Temporal,
    denoise: Denoise,
    exposure: Exposure,
    reference: Reference,
    sub: Option<Submitter>,
    sky_sun: Option<[f64; 3]>,
    w: u32,
    h: u32,
}

impl Rig {
    fn new(dressing: Dressing, w: u32, h: u32) -> Rig {
        Self::with_gpu(gpu_validated(), dressing, w, h)
    }

    /// For timing runs: validation may be off (`NE_NO_VALIDATION`).
    fn for_timing(dressing: Dressing, w: u32, h: u32) -> Rig {
        let g = Gpu::new().expect("an RT-capable Vulkan device");
        assert!(g.ray_tracing());
        Self::with_gpu(g, dressing, w, h)
    }

    fn with_gpu(g: Gpu, dressing: Dressing, w: u32, h: u32) -> Rig {
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let (world, _) = street_night(dressing);
        let emission = light::emitters::emission(world.materials()).unwrap();
        let mut pipeline = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
        pipeline.mark_all(&world);
        drain(&mut pipeline, &world);
        let rs = full_regions(&mut pipeline, RegionSize::Chunk);
        let snap = pipeline.current().id.raw();
        let scene = GpuScene::build_lit(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs, snap, true, world.materials()).unwrap();
        eprintln!("rig: {} ({} emitters), {w}x{h}", dressing.name(), scene.emitters().unwrap().unwrap().table.len());
        let params: Vec<world::MaterialParams> = world.materials().iter().map(|(_, d)| d.params).collect();
        let (tables, _) = Tables::upload(&g, &mut alloc, &mut up, &mut tl, &params, &scene.meshes, snap).unwrap();
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
        let xo = ExposureTargets::new(&g, &mut alloc, 1).unwrap();
        let raster = Raster::new(&g).unwrap();
        let shade = Shade::new(&g).unwrap();
        let temporal = Temporal::new(&g).unwrap();
        let denoise = Denoise::new(&g).unwrap();
        let exposure = Exposure::new(&g).unwrap();
        let reference = Reference::new(&g).unwrap();
        let sub = Submitter::new(&g).unwrap();
        let mut rig = Rig {
            g,
            alloc,
            tl,
            up,
            world,
            pipeline,
            emission,
            scene: Some(scene),
            tables: Some(tables),
            mats: Some(mats),
            sky: Some(sky),
            sky_pass: Some(sky_pass),
            targets: Some(targets),
            out: Some(out),
            history: Some(history),
            dn: Some(dn),
            xo: Some(xo),
            bound: None,
            raster,
            shade,
            temporal,
            denoise,
            exposure,
            reference,
            sub: Some(sub),
            sky_sun: None,
            w,
            h,
        };
        rig.rebind();
        rig
    }

    fn table(&self) -> &EmitterTable {
        &self.scene.as_ref().unwrap().emitters().unwrap().unwrap().table
    }

    /// (Re)creates every descriptor binding (after an edit or a new history).
    fn rebind(&mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        if let Some(b) = self.bound.take() {
            b.rb.destroy(g);
            b.sb.destroy(g);
            b.sb_plain.destroy(g);
            b.tb.destroy(g);
            b.db.destroy(g);
            b.xb.destroy(g);
        }
        let (targets, out, s, h) = (self.targets.as_ref().unwrap(), self.out.as_ref().unwrap(), self.scene.as_ref().unwrap(), self.history.as_ref().unwrap());
        let (a, mats, sky) = (s.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &self.sky.as_ref().unwrap().view);
        let em = s.emitters().unwrap().unwrap();
        let rb = self.raster.bind(g, &s.meshes).unwrap();
        let sb = self.shade.bind_lit(g, targets, a, mats, out, sky, &em.device).unwrap();
        assert_eq!(sb.emitters(), Some(em.table.len() as u32));
        let sb_plain = self.shade.bind(g, targets, a, mats, out, sky).unwrap();
        let tables = self.tables.as_ref().unwrap();
        let tb = self.temporal.bind(g, targets, &tables.regions, &out.radiance, h).unwrap();
        let db = self.denoise.bind(g, &out.radiance, h, mats, self.dn.as_ref().unwrap()).unwrap();
        let xb = self.exposure.bind(g, targets, &out.radiance, tables, self.xo.as_ref().unwrap()).unwrap();
        self.bound = Some(Bound { rb, sb, sb_plain, tb, db, xb });
    }

    /// A fresh history (the next frame is a first-frame reset).
    fn new_history(&mut self) {
        self.g.wait_idle().unwrap();
        self.history.take().unwrap().free(&self.g, &mut self.alloc);
        self.history = Some(History::new(&self.g, &mut self.alloc, self.w, self.h).unwrap());
        self.rebind();
    }

    /// Records one frame. Timer passes (slot 0): 0 shade, 1 temporal, 2 filter, 3 exposure.
    #[allow(clippy::too_many_arguments)]
    fn record(&mut self, cmd: vk::CommandBuffer, cam: &Camera, light: &Lighting, step: Step, relight: &[Relight], frame: u32, exposure_floor: Option<f64>, timer: Option<&GpuTimer>) {
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
        let sb = if step.plain { &b.sb_plain } else { &b.sb };
        self.shade.record(g, cmd, sb, out, &shade::Params::new(cam, light, step.shade, step.faults, frame, SEED));
        stamp(0, false);
        let h = self.history.as_mut().unwrap();
        stamp(1, true);
        if step.temporal {
            h.set_lights(step.lights);
            self.temporal.record(g, cmd, &b.tb, h, cam, light.sun_dir, step.ts, TemporalFaults::default(), step.force_reset, relight);
        }
        stamp(1, false);
        stamp(2, true);
        if step.temporal && step.filter {
            self.denoise.record(g, cmd, &b.db, h, self.dn.as_ref().unwrap(), DenoiseSettings::default(), DenoiseFaults::default());
        }
        stamp(2, false);
        stamp(3, true);
        if let Some(floor) = exposure_floor {
            self.exposure.record(g, cmd, &b.xb, self.xo.as_ref().unwrap(), self.tables.as_ref().unwrap(), self.w, self.h, 0, floor, step.lights);
        }
        stamp(3, false);
    }

    /// Runs one frame; returns the radiance buffer when `read`.
    fn frame(&mut self, cam: &Camera, light: &Lighting, step: Step, relight: &[Relight], frame: u32, read: bool) -> Option<Vec<[f32; 4]>> {
        self.frame_with(cam, light, step, relight, frame, read, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn frame_with(&mut self, cam: &Camera, light: &Lighting, step: Step, relight: &[Relight], frame: u32, read: bool, exposure_floor: Option<f64>) -> Option<Vec<[f32; 4]>> {
        let mut sub = self.sub.take().unwrap();
        let cmd = sub.begin(&self.g, &self.tl).unwrap();
        self.record(cmd, cam, light, step, relight, frame, exposure_floor, None);
        let v = sub.submit(&self.g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(&self.g, v, u64::MAX).unwrap();
        self.sub = Some(sub);
        read.then(|| self.out.as_ref().unwrap().read(&self.g, &mut self.alloc, &mut self.tl).unwrap())
    }

    /// The shade pass alone (no temporal pass), frame `frame`.
    fn shade_frame(&mut self, cam: &Camera, light: &Lighting, set: ShadeSettings, faults: ShadeFaults, frame: u32, plain: bool) -> Vec<[f32; 4]> {
        self.frame(cam, light, Step { faults, plain, ..Step::shade(set) }, &[], frame, true).unwrap()
    }

    /// The reference on the scene's TLAS and table: samples `first..first + frames` of `seed`.
    fn reference(&mut self, cam: &Camera, light: &Lighting, s: &Settings, seed: u32, first: u32, frames: u32) -> Accum {
        let out = RefAccum::new(&self.g, &mut self.alloc, cam.width, cam.height).unwrap();
        let sc = self.scene.as_ref().unwrap();
        let em = sc.emitters().unwrap().unwrap();
        let b = self.reference.bind_lit(&self.g, sc.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &em.device, &out).unwrap();
        let per = if cam.width * cam.height > 1_000_000 { 8 } else { 64 };
        self.reference.accumulate(&self.g, &mut self.tl, &b, &out, cam, light, s, RefFaults::default(), seed, first..first + frames, first, per).unwrap();
        let img = self.reference.read(&self.g, &mut self.alloc, &mut self.tl, &out, frames).unwrap();
        assert_eq!(img.bad_samples, 0);
        self.g.wait_idle().unwrap();
        b.destroy(&self.g);
        out.free(&self.g, &mut self.alloc);
        img.accum
    }

    /// The reference, from the cache in `NE_GATE_DIR` when present.
    fn cached_reference(&mut self, key: &str, cam: &Camera, light: &Lighting, spp: u32) -> Accum {
        let header = format!("NE4BREF v{CACHE_VERSION} {key} {}x{} spp {spp} seed {REF_SEED} snapshot {} emitters {} sun {:?}\n", cam.width, cam.height, self.scene.as_ref().unwrap().snapshot, self.table().len(), light.sun_dir);
        let path = gate_dir().map(|d| d.join(format!("ref4b_{key}.bin")));
        if let Some(a) = path.as_ref().and_then(|p| load_accum(p, &header, cam.width, cam.height, spp)) {
            eprintln!("reference {key}: cached");
            return a;
        }
        let t = std::time::Instant::now();
        let a = self.reference(cam, light, &ref_settings(), REF_SEED, 0, spp);
        eprintln!("reference {key}: {spp} spp in {:.1} s", t.elapsed().as_secs_f64());
        if let Some(p) = path {
            save_accum(&p, &header, &a);
        }
        a
    }

    fn guides(&mut self) -> Vec<[u32; 4]> {
        self.history.as_ref().unwrap().read_guides(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    fn state(&mut self) -> Vec<(u32, u32)> {
        self.history.as_ref().unwrap().read(&self.g, &mut self.alloc, &mut self.tl).unwrap().0
    }

    /// Sets voxels (None removes), publishes, updates the device scene (meshes, TLAS, emitter table)
    /// and the region table, and rebinds. Returns the regions rebuilt and the E1 power bound (from the
    /// world before the edit).
    fn edit(&mut self, voxels: &[(VoxelCoord, Option<MaterialId>)]) -> (BTreeSet<RegionKey>, f64) {
        let power = changed_power(voxels, |c| self.world.get(c), &self.emission);
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
        (regions, power)
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

    /// Per pixel of the last frame: the emitted radiance the light view adds (surface pixels, by the
    /// history guides' material).
    fn emission_of(&self, guides: &[[u32; 4]]) -> Vec<[f64; 3]> {
        guides.iter().map(|g| if g[0] & 7 == FACE_NONE { [0.0; 3] } else { self.emission.get((g[0] >> 3) as usize).copied().unwrap_or([0.0; 3]) }).collect()
    }

    fn finish(mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let (errors, warnings) = g.validation_counts();
        eprintln!("validation: {errors} errors, {warnings} warnings");
        if g.validation_enabled() {
            assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
        }
        if let Some(b) = self.bound.take() {
            b.rb.destroy(g);
            b.sb.destroy(g);
            b.sb_plain.destroy(g);
            b.tb.destroy(g);
            b.db.destroy(g);
            b.xb.destroy(g);
        }
        self.sub.take().unwrap().destroy(g);
        self.xo.take().unwrap().free(g, &mut self.alloc);
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
        self.exposure.destroy(g);
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

fn gate_dir() -> Option<PathBuf> {
    std::env::var_os("NE_GATE_DIR").map(PathBuf::from)
}

/// The header line, then per pixel the sum and the sum of squares as 6 f64 (little-endian).
fn save_accum(path: &std::path::Path, header: &str, a: &Accum) {
    let mut b = header.as_bytes().to_vec();
    for (s, q) in a.sum.iter().zip(&a.sum_sq) {
        for v in s.iter().chain(q) {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b).unwrap();
}

fn load_accum(path: &std::path::Path, header: &str, w: u32, h: u32, samples: u32) -> Option<Accum> {
    let b = std::fs::read(path).ok()?;
    let body = b.strip_prefix(header.as_bytes())?;
    let n = (w * h) as usize;
    if body.len() != n * 48 {
        return None;
    }
    let f = |i: usize| f64::from_le_bytes(body[8 * i..8 * i + 8].try_into().unwrap());
    Some(Accum { width: w, height: h, samples, sum: (0..n).map(|p| [f(6 * p), f(6 * p + 1), f(6 * p + 2)]).collect(), sum_sq: (0..n).map(|p| [f(6 * p + 3), f(6 * p + 4), f(6 * p + 5)]).collect() })
}

/// Surface pixels: the primary ray hits (a uniform sky of 1 shows exactly 1 where it misses).
fn surface_mask(rig: &mut Rig, cam: &Camera) -> Vec<bool> {
    let u = rig.reference(cam, &lighting(12.0), &Settings { uniform_sky: true, max_bounces: 0, ..Settings::default() }, REF_SEED, 0, 1);
    u.sum.iter().map(|v| *v != [1.0; 3]).collect()
}


// ---------------------------------------------------------------------------------------------
// G1

struct Agreement {
    mismatch: usize,
    /// Data: pixels outside 3B–3F's tolerance of 10⁻⁵ (see the record's G1 correction).
    over_1e5: usize,
    bad: usize,
}

/// G1's tolerance (corrected before the first run): the emitter terms depend continuously on the
/// surface point, which the real-time pass rebuilds from depth and the reference takes from its ray
/// hit, and the f32 solid angle carries about 10⁻⁶ sr of rounding (10⁻⁴ relative at 10⁻² sr).
const G1_TOLERANCE: f64 = 1e-3;

fn agreement(rt: &[[f32; 4]], rf: &Accum) -> Agreement {
    let mut a = Agreement { mismatch: 0, over_1e5: 0, bad: 0 };
    for (x, y) in rt.iter().zip(&rf.sum) {
        a.bad += usize::from(x[3] != 0.0);
        let within = |tol: f64| (0..3).all(|k| if y[k] == 0.0 { x[k] == 0.0 } else { (x[k] as f64 / y[k] - 1.0).abs() <= tol });
        a.mismatch += usize::from(!within(G1_TOLERANCE));
        a.over_1e5 += usize::from(!within(1e-5));
    }
    a
}

/// G1: frame f of the shade pass with emitters equals the reference's sample f per pixel (emitters
/// only with and without the bounce, uniform sky, point sun), with the controls (emitters off, a
/// different frame, two emitter samples) and the M3-unchanged check (emitters off: the pass bound with
/// the table equals the pass bound without it, bit for bit).
#[test]
fn emitters_equal_the_reference_sample_per_pixel() {
    let (w, h) = (480, 270);
    let mut rig = Rig::new(Dressing::Full, w, h);
    let n = (w * h) as usize;
    let frame = 29;
    let mut failed = Vec::new();
    let none = ShadeSettings { emitters: true, ..ShadeSettings::NONE };
    let r1 = Settings { sun: false, sky: false, max_bounces: 1, emitters_direct: true, emitters_indirect: true, ..Settings::default() };
    let arms: Vec<(&str, f64, ShadeSettings, Settings)> = vec![
        ("emitters, bounce", NIGHT, ShadeSettings { bounce: true, ..none }, r1),
        ("uniform sky + emitters, bounce", NIGHT, ShadeSettings { uniform_sky: true, bounce: true, ..none }, Settings { uniform_sky: true, ..r1 }),
        ("point sun 8 h + emitters, bounce", 8.0, ShadeSettings { sun: true, point_sun: true, bounce: true, ..none }, Settings { sun: true, point_sun: true, ..r1 }),
        ("emitters, no bounce", NIGHT, none, Settings { max_bounces: 0, emitters_indirect: false, ..r1 }),
    ];
    // Per kind: (arms, arms failed, total mismatches).
    let mut tally: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (cname, cam) in cameras(w, h) {
        for (label, hour, set, rs) in &arms {
            let light = lighting(*hour);
            let rf = rig.reference(&cam, &light, rs, SEED, frame, 1);
            let rt = rig.shade_frame(&cam, &light, *set, ShadeFaults::default(), frame, false);
            let off = rig.shade_frame(&cam, &light, ShadeSettings { emitters: false, ..*set }, ShadeFaults::default(), frame, false);
            let changed = rt.iter().zip(&off).filter(|(a, b)| a[..3] != b[..3]).count();
            let a = agreement(&rt, &rf);
            let ctl = agreement(&off, &rf);
            let pass = a.mismatch * 1000 <= n && a.bad == 0 && changed >= 1000;
            eprintln!(
                "G1 {cname} {label}: {} mismatched ({:.3}%; data: {} over 1e-5), bad {}, {changed} px changed by the emitters | control emitters off: {} mismatched ({:.2}%) {}",
                a.mismatch,
                100.0 * a.mismatch as f64 / n as f64,
                a.over_1e5,
                a.bad,
                ctl.mismatch,
                100.0 * ctl.mismatch as f64 / n as f64,
                if pass { "pass" } else { "FAIL" }
            );
            for (kind, m, ok) in [("ok", a.mismatch, pass), ("emitters off", ctl.mismatch, ctl.mismatch * 1000 <= n)] {
                let t = tally.entry(kind).or_default();
                t.0 += 1;
                t.1 += usize::from(!ok);
                t.2 += m;
            }
            if !pass {
                failed.push(format!("G1 {cname}/{label}"));
            }
        }
    }
    let ok_total = tally["ok"].2;
    let (arms_off, failing_off, total_off) = tally["emitters off"];
    eprintln!("G1 tally: correct arms {ok_total} mismatched px; emitters off fails {failing_off} of {arms_off} arms, {total_off} px");
    if failing_off < arms_off || total_off < 10 * ok_total.max(1) {
        failed.push(format!("G1 control emitters off not caught: {failing_off} of {arms_off} arms, {total_off} px (ok arms {ok_total} px)"));
    }
    let cam = street_camera(w, h);
    let light = lighting(NIGHT);
    let (_, _, set, rs) = arms[0];
    let rf = rig.reference(&cam, &light, &rs, SEED, frame, 1);
    let next = agreement(&rig.shade_frame(&cam, &light, set, ShadeFaults::default(), frame + 1, false), &rf).mismatch;
    let two = agreement(&rig.shade_frame(&cam, &light, ShadeSettings { emitter_samples: 2, ..set }, ShadeFaults::default(), frame, false), &rf).mismatch;
    eprintln!("G1 controls: frame {} against reference frame {frame}: {next} mismatched; 2 emitter samples: {two} mismatched", frame + 1);
    if next == 0 || two == 0 {
        failed.push("G1 frame / samples control not caught".into());
    }
    // M3 unchanged: with emitters off, bound with the table or without it gives the same bits.
    for hour in [8.0, NIGHT] {
        let light = lighting(hour);
        for set in [ShadeSettings::default(), ShadeSettings { bounce: false, ..ShadeSettings::default() }] {
            let a = rig.shade_frame(&cam, &light, set, ShadeFaults::default(), frame, false);
            let b = rig.shade_frame(&cam, &light, set, ShadeFaults::default(), frame, true);
            let differ = a.iter().zip(&b).filter(|(x, y)| x.map(f32::to_bits) != y.map(f32::to_bits)).count();
            eprintln!("G1 M3 unchanged at {hour} h, bounce {}: {differ} px differ between bind_lit and bind", set.bounce);
            if differ != 0 {
                failed.push(format!("G1 M3 unchanged at {hour} h"));
            }
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---------------------------------------------------------------------------------------------
// G2

struct Stat {
    mean_z: [f64; 3],
    frac_over_4: f64,
    rel_mean: [f64; 3],
}

/// 4A's G4 metric over the masked pixels: image-mean z per channel and the fraction of pixel-channels
/// with |z| > 4.
fn compare(a: &Accum, b: &Accum, mask: &[bool]) -> Stat {
    let (mut over, mut total) = (0usize, 0usize);
    let mut mean_z = [0.0; 3];
    let mut rel_mean = [0.0; 3];
    for k in 0..3 {
        let (mut ma, mut mb, mut va, mut vb) = (0.0, 0.0, 0.0, 0.0);
        for i in (0..a.sum.len()).filter(|&i| mask[i]) {
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
        mean_z[k] = (ma - mb) / (va + vb).sqrt().max(1e-300);
        rel_mean[k] = ma / mb - 1.0;
    }
    Stat { mean_z, frac_over_4: over as f64 / total.max(1) as f64, rel_mean }
}

/// The shade pass's frames `0..frames` as an `Accum` (sum and sum of squares per pixel).
fn shade_accum(rig: &mut Rig, cam: &Camera, light: &Lighting, set: ShadeSettings, faults: ShadeFaults, frames: u32) -> Accum {
    let n = (cam.width * cam.height) as usize;
    let mut a = Accum { width: cam.width, height: cam.height, samples: frames, sum: vec![[0.0; 3]; n], sum_sq: vec![[0.0; 3]; n] };
    for f in 0..frames {
        let x = rig.shade_frame(cam, light, set, faults, f, false);
        for (i, p) in x.iter().enumerate() {
            assert_eq!(p[3], 0.0, "bad surface id");
            for (c, &v) in p[..3].iter().enumerate() {
                let v = v as f64;
                a.sum[i][c] += v;
                a.sum_sq[i][c] += v * v;
            }
        }
    }
    a
}

/// G2: the shade pass with emitters and the bounce (k = 1, 2, 4 emitter samples at the primary
/// vertex), averaged over 1024 frames, converges to the reference at night (both cameras, Full and
/// Dense); the fault "no solid angle" fails; blue hour at k = 1 is data.
#[test]
fn emitters_converge_to_the_reference() {
    let (w, h) = (320, 180);
    const FRAMES: u32 = 1024;
    const SPP: u32 = 16_384;
    let mut failed = Vec::new();
    for dressing in [Dressing::Full, Dressing::Dense] {
        let mut rig = Rig::new(dressing, w, h);
        for (cname, cam) in cameras(w, h) {
            let mask = surface_mask(&mut rig, &cam);
            let mut arms: Vec<(String, f64, u32, ShadeFaults, bool)> = [1u32, 2, 4].iter().map(|&k| (format!("k {k}"), NIGHT, k, ShadeFaults::default(), true)).collect();
            if dressing == Dressing::Full {
                arms.push(("fault no_solid_angle".into(), NIGHT, 1, ShadeFaults { no_solid_angle: true, ..ShadeFaults::default() }, true));
                arms.push(("blue hour (data)".into(), BLUE_HOUR, 1, ShadeFaults::default(), false));
            }
            let mut refs: BTreeMap<u64, (Accum, f64)> = BTreeMap::new();
            for (label, hour, k, faults, judged) in arms {
                let light = lighting(hour);
                let key = hour.to_bits();
                if let std::collections::btree_map::Entry::Vacant(e) = refs.entry(key) {
                    let r = rig.reference(&cam, &light, &ref_settings(), REF_SEED, 0, SPP);
                    let null = rig.reference(&cam, &light, &ref_settings(), NULL_SEED, 0, SPP / 16);
                    let nf = compare(&null, &r, &mask).frac_over_4;
                    e.insert((r, nf));
                }
                let (r, null) = &refs[&key];
                let limit = (1.5 * null + 0.002).max(0.01);
                let set = ShadeSettings { emitter_samples: k, ..lit() };
                let a = shade_accum(&mut rig, &cam, &light, set, faults, FRAMES);
                let st = compare(&a, r, &mask);
                let fault = faults.no_solid_angle;
                let pass = st.mean_z.iter().all(|z| z.abs() < 4.0) && st.frac_over_4 <= limit;
                let caught = st.mean_z.iter().any(|z| z.abs() > 10.0);
                eprintln!(
                    "G2 {} {cname} {label}: rel mean {:?}, z {:?}, |z|>4 {:.3}% (null {:.3}%, limit {:.3}%) {}",
                    dressing.name(),
                    st.rel_mean.map(|x| (x * 1e4).round() / 1e4),
                    st.mean_z.map(|x| (x * 100.0).round() / 100.0),
                    st.frac_over_4 * 100.0,
                    null * 100.0,
                    limit * 100.0,
                    if !judged { "(data)" } else if fault { if caught { "caught" } else { "NOT CAUGHT" } } else if pass { "pass" } else { "FAIL" }
                );
                if judged && !fault && !pass {
                    failed.push(format!("G2 {} {cname} {label}", dressing.name()));
                }
                if fault && !caught {
                    failed.push(format!("G2 fault not caught {cname}"));
                }
            }
        }
        rig.finish();
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---------------------------------------------------------------------------------------------
// G3

fn aces(x: f64) -> f64 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

/// G3: the light view with a zero radiance buffer and the lights on shows aces(exposure · L_e) on
/// every surface pixel (the material from the G-buffer), and with the lights off exactly the
/// lights-off image; at least 1000 emissive pixels.
#[test]
fn the_light_view_adds_emission() {
    let (w, h) = (480, 270);
    let mut rig = Rig::new(Dressing::Full, w, h);
    let cam = street_camera(w, h);
    let light = lighting(NIGHT);
    let exposure = 30.0f32;
    let s = rig.scene.as_ref().unwrap();
    let rb = rig.raster.bind(&rig.g, &s.meshes).unwrap();
    let gframe = rig.raster.render_to_host(&rig.g, &mut rig.alloc, &mut rig.tl, rig.targets.as_ref().unwrap(), &s.meshes, &rb, &cam, Faults::default()).unwrap();
    let dv = DebugView::new(&rig.g, vk::Format::R8G8B8A8_UNORM).unwrap();
    dv.bind(&rig.g, rig.targets.as_ref().unwrap(), rig.tables.as_ref().unwrap(), None);
    dv.bind_radiance(&rig.g, &rig.out.as_ref().unwrap().radiance);
    let img = rig.alloc.create_image(&rig.g, vk::Format::R8G8B8A8_UNORM, vk::Extent2D { width: w, height: h }, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::ImageAspectFlags::COLOR, Category::GpuTemporal).unwrap();
    let px = (w * h) as usize;
    let host = rig.alloc.create_buffer(&rig.g, px as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST, Category::Staging, Kind::Host).unwrap();
    let mut images = Vec::new();
    for emission in [true, false] {
        let mut sub = rig.sub.take().unwrap();
        let g = &rig.g;
        let cmd = sub.begin(g, &rig.tl).unwrap();
        let (targets, out, b) = (rig.targets.as_ref().unwrap(), rig.out.as_ref().unwrap(), rig.bound.as_ref().unwrap());
        rig.raster.record(g, cmd, targets, &rig.scene.as_ref().unwrap().meshes, &b.rb, &cam, Faults::default());
        targets_to_read(g, cmd, targets);
        // Every term off: the radiance buffer is 0 everywhere.
        rig.shade.record(g, cmd, &b.sb, out, &shade::Params::new(&cam, &light, ShadeSettings::NONE, ShadeFaults::default(), 0, SEED));
        shade::radiance_to_fragment(g, cmd);
        gpu::present::image_to_attachment(g, cmd, img.image);
        let mut l = ViewLighting::new(light.sun_dir);
        l.light_exposure = exposure;
        l.emission = emission;
        dv.draw(g, cmd, targets, rig.tables.as_ref().unwrap(), img.view, &cam, View::Light, &l, Source::Raster);
        let to_copy = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })];
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
        };
        unsafe {
            g.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_copy));
            g.device.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.buffer, &[copy]);
        }
        gpu::submit::all_to_host(g, cmd);
        let v = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
        rig.tl.wait(g, v, u64::MAX).unwrap();
        rig.sub = Some(sub);
        images.push(host.mapped_ref().unwrap().to_vec());
    }
    let (on, off) = (&images[0], &images[1]);
    let (mut emissive, mut bad_on, mut bad_off) = (0usize, 0usize, 0usize);
    for i in 0..px {
        let surface = gframe.depth[i] > 0.0;
        let le = if surface { rig.emission[gframe.material[i] as usize] } else { [0.0; 3] };
        if lum(le) > 0.0 {
            emissive += 1;
        }
        for c in 0..3 {
            let want = aces(exposure as f64 * le[c]);
            if (on[4 * i + c] as f64 / 255.0 - want).abs() > 1.0 / 255.0 + 1e-3 {
                bad_on += 1;
            }
            if off[4 * i + c] != 0 {
                bad_off += 1;
            }
        }
    }
    eprintln!("G3: {emissive} emissive px; lights on: {bad_on} channel values off by more than 1/255 + 1e-3; lights off: {bad_off} nonzero");
    let g = &rig.g;
    g.wait_idle().unwrap();
    dv.destroy(g);
    rb.destroy(g);
    rig.alloc.free_image(g, img);
    rig.alloc.free(g, host);
    rig.finish();
    assert!(emissive >= 1000 && bad_on == 0 && bad_off == 0, "G3 failed");
}

// ---------------------------------------------------------------------------------------------
// G4

/// Relative MSE (per channel, epsilon = (0.1 x the channel's mean)^2) and the relative bias of the
/// mean luminance over the masked pixels: the 3E / 3G metric.
#[derive(Clone, Copy, Debug)]
struct Error {
    rel_mse: f64,
    bias: f64,
}

fn error(x: &[[f32; 4]], t: &[[f64; 3]], mask: &[bool]) -> Error {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| t[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-30));
    let mut se = 0.0;
    let (mut lx, mut lt) = (0.0, 0.0);
    for &i in &idx {
        for c in 0..3 {
            se += (x[i][c] as f64 - t[i][c]).powi(2) / (t[i][c].powi(2) + eps[c]);
        }
        lx += lum4(x[i]);
        lt += lum(t[i]);
    }
    Error { rel_mse: se / (3.0 * n), bias: (lx - lt) / lt.max(1e-300) }
}

/// The reference's own noise in the same metric.
fn ref_noise(r: &Accum, mask: &[bool]) -> f64 {
    let idx: Vec<usize> = (0..r.sum.len()).filter(|&i| mask[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| r.mean(i)[c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|m| (0.1 * m).powi(2).max(1e-30));
    idx.iter().map(|&i| (0..3).map(|c| r.std_error(i)[c].powi(2) / (r.mean(i)[c].powi(2) + eps[c])).sum::<f64>()).sum::<f64>() / (3.0 * n)
}

/// Writes a display image (emission added, exposure, the ACES fit, sRGB: the viewer's light view) as a
/// PPM into `NE_GATE_DIR`, for FLIP.
fn write_display(name: &str, w: u32, h: u32, px: &dyn Fn(usize) -> [f64; 3], exposure: f64) {
    let Some(dir) = gate_dir() else { return };
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

/// Runs a still camera from a fresh history (frame index = age); returns the shown radiance at `ages`.
fn run_still(rig: &mut Rig, cam: &Camera, light: &Lighting, filter: bool, ages: &[u32]) -> BTreeMap<u32, Vec<[f32; 4]>> {
    rig.new_history();
    let last = *ages.iter().max().unwrap();
    let mut out = BTreeMap::new();
    for k in 1..=last {
        if let Some(x) = rig.frame(cam, light, Step { filter, ..Step::viewer() }, &[], k, ages.contains(&k)) {
            out.insert(k, x);
        }
    }
    out
}

/// G4 (Q1, Q2 and the images for Q4): the viewer's frame with the lights on at 1080p against the
/// one-bounce reference with emitters (16,384 spp), both cameras; judged at night, blue hour as data.
/// Q1 is judged as Q1a (the filter's energy change against the raw history of the same frames,
/// ±2%) and Q1b (the raw history's bias within max(2%, 4 sigma)); corrected before any run.
#[test]
fn night_against_the_reference() {
    let (w, h) = (1920, 1080);
    let mut rig = Rig::new(Dressing::Full, w, h);
    let mut failed = Vec::new();
    for (time, hour, judged) in [("night", NIGHT, true), ("blue_hour", BLUE_HOUR, false)] {
        let light = lighting(hour);
        for (cname, cam) in cameras(w, h) {
            let key = format!("{cname}_{time}");
            let r = rig.cached_reference(&key, &cam, &light, 16_384);
            let mean: Vec<[f64; 3]> = (0..r.sum.len()).map(|i| r.mean(i)).collect();
            let raw = run_still(&mut rig, &cam, &light, false, &[1, 4, 8, 16, 64]);
            let guides = rig.guides();
            let m: Vec<bool> = guides.iter().map(|g| g[0] & 7 != FACE_NONE).collect();
            let em = rig.emission_of(&guides);
            let filt = run_still(&mut rig, &cam, &light, true, &[1, 4, 16, 64]);
            // The 1-sigma of one real-time frame's mean luminance over the surface, from the
            // reference's per-sample variance (the same estimator): Q1's noise scale at age 1.
            let (mut var, mut tot) = (0.0, 0.0);
            for i in (0..r.sum.len()).filter(|&i| m[i]) {
                var += (lum(r.std_error(i)) * (r.samples as f64).sqrt()).powi(2);
                tot += lum(r.mean(i));
            }
            let floor = var.sqrt() / tot.max(1e-300);
            eprintln!(
                "G4 {key}: {} surface px, reference noise rel_mse {:.2e}, one frame's mean luminance 1-sigma {:.2}%{}",
                m.iter().filter(|&&b| b).count(),
                ref_noise(&r, &m),
                100.0 * floor,
                if judged { "" } else { " (blue hour: data)" }
            );
            for (age, gain) in [(1u32, 8u32), (4, 4), (16, 4), (64, 1)] {
                let (f, a, rg) = (error(&filt[&age], &mean, &m), error(&raw[&age], &mean, &m), error(&raw[&(age * gain)], &mean, &m));
                // Q1, corrected before any run (4B record, "G4 correction"): one night frame's mean
                // is too noisy for ±2% (1-sigma about 7.5% at 1080p, CPU-measured). Q1a: the filter
                // keeps energy, paired with the raw history of the same frames (the filter writes
                // only the shown radiance), within ±2%. Q1b: the raw history is unbiased, within
                // max(2%, 4 sigma) at its age (sigma = the frame's 1-sigma / sqrt(age)).
                let sigma = floor / (age as f64).sqrt();
                let q1a = (f.bias - a.bias).abs() <= 0.02;
                let q1b = a.bias.abs() <= (4.0 * sigma).max(0.02);
                let q1 = q1a && q1b;
                let q2 = f.rel_mse <= rg.rel_mse;
                eprintln!(
                    "  age {age:2}: filtered rel_mse {:.4} bias {:+.4} | raw {:.4} bias {:+.4} (z {:+.2}) | raw at age {:2} {:.4} | filter energy {:+.4} | Q1 {} (a {} b {}) Q2 {}",
                    f.rel_mse,
                    f.bias,
                    a.rel_mse,
                    a.bias,
                    a.bias / sigma.max(1e-300),
                    age * gain,
                    rg.rel_mse,
                    f.bias - a.bias,
                    if q1 { "pass" } else { "FAIL" },
                    if q1a { "pass" } else { "FAIL" },
                    if q1b { "pass" } else { "FAIL" },
                    if q2 { "pass" } else { "FAIL" }
                );
                if judged && !q1 {
                    failed.push(format!("Q1 {key} age {age}"));
                }
                if judged && !q2 {
                    failed.push(format!("Q2 {key} age {age}"));
                }
            }
            // Display images: emission added on surface pixels; the reference's metric exposure.
            let shown_ref = |i: usize| std::array::from_fn(|c| mean[i][c] + em[i][c]);
            let ex = metric_exposure(&(0..mean.len()).map(|i| lum(shown_ref(i))).collect::<Vec<f64>>()).unwrap_or(1.0);
            eprintln!("  display exposure (metric, from the reference with emission): {ex:.4e}");
            for age in [1u32, 16, 64] {
                for (tag, x) in [("raw", &raw[&age]), ("filt", &filt[&age])] {
                    write_display(&format!("still4b_{key}_{tag}_{age}"), w, h, &|i| std::array::from_fn(|c| x[i][c] as f64 + em[i][c]), ex);
                }
            }
            write_display(&format!("still4b_{key}_ref"), w, h, &shown_ref, ex);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---------------------------------------------------------------------------------------------
// G5

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// An edit: label, box (`hi` exclusive), and the material set (None removes).
type EditBox = (&'static str, [i32; 3], [i32; 3], Option<MaterialId>);

/// One R2 arm: the state after the edit frame, and the shown radiance 3 frames later.
type Arm = (Vec<(u32, u32)>, Vec<[f32; 4]>);

/// Slab test of the segment p + t seg, t in [0, 1], against the box [lo, hi].
fn segment_hits(p: [f64; 3], seg: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> bool {
    let (mut enter, mut leave) = (0.0f64, 1.0f64);
    for a in 0..3 {
        let (t0, t1) = ((lo[a] - p[a]) / seg[a], (hi[a] - p[a]) / seg[a]);
        enter = enter.max(t0.min(t1));
        leave = leave.min(t0.max(t1));
    }
    enter <= leave
}

/// The ADR-0006 Amendments 1 and 2 rule on the host, with the rows the GPU receives: `y` is the
/// history's luminance.
fn host_relit(p: [f64; 3], y: f64, sun: [f64; 3], boxes: &[Relight]) -> bool {
    let thr = EMITTER_TOLERANCE as f32 as f64 * y;
    boxes.iter().any(|b| {
        let [lo, hi] = b.rows().map(|r| r.map(|v| v as f64));
        let d: [f64; 3] = std::array::from_fn(|a| (lo[a] - p[a]).max(p[a] - hi[a]).max(0.0));
        let d2 = dot3(d, d);
        if d2 <= lo[3] * lo[3] {
            return true;
        }
        let (mut enter, mut leave) = (0.0f64, f64::INFINITY);
        for a in 0..3 {
            let (t0, t1) = ((lo[a] - p[a]) / sun[a], (hi[a] - p[a]) / sun[a]);
            enter = enter.max(t0.min(t1));
            leave = leave.min(t0.max(t1));
        }
        if enter <= leave {
            return true;
        }
        if hi[3] > 0.0 && hi[3] >= thr * d2 {
            return true;
        }
        let area = (hi[0] - lo[0]) * (hi[1] - lo[1]) + (hi[1] - lo[1]) * (hi[2] - lo[2]) + (hi[2] - lo[2]) * (hi[0] - lo[0]);
        let rows = b.light_rows();
        (0..rows.len() / 2).any(|j| {
            let (c, ye) = (rows[2 * j].map(|v| v as f64), rows[2 * j + 1][0] as f64);
            if ye <= 0.0 || ye * area / std::f64::consts::PI < thr * d2 {
                return false;
            }
            let r = c[3];
            segment_hits(p, [c[0] - p[0], c[1] - p[1], c[2] - p[2]], [lo[0] - r, lo[1] - r, lo[2] - r], [hi[0] + r, hi[1] + r, hi[2] + r])
        })
    })
}

/// G5: edits that change emitter light (remove part of a lamp head; add neon in open air; put a stone
/// box between a lamp head and the road). R1: the relit pixels equal the host's rule (3E + E1 + E2);
/// the 3E-only boxes must fail R1 in at least one edit. R2: 4 frames after the edit, the pixels whose
/// converged value changed by > 25% (and by > 4 standard errors) have a filtered relative MSE
/// <= 3 x that of an arm reset at the edit; the 3E-only arm must fail R2 in at least one edit.
#[test]
fn edits_relight_emitter_light() {
    let (w, h) = (960, 540);
    const SPP: u32 = 16_384;
    let mut rig = Rig::new(Dressing::Full, w, h);
    let cam = street_camera(w, h);
    let light = lighting(NIGHT);
    let sun = light.sun_dir;
    let mats = rig.world.materials().clone();
    let id = |n: &str| mats.id_of(n).unwrap();
    // The lamp head nearest to the camera: its lowest, smallest-x, smallest-z voxel.
    let lamp = rig
        .world
        .occupied()
        .filter(|&(_, m)| m == id("lamp"))
        .map(|(v, _)| v)
        .min_by(|a, b| {
            let d = |v: &VoxelCoord| (0..3).map(|k| ([v.x, v.y, v.z][k] as f64 - cam.eye[k]).powi(2)).sum::<f64>();
            d(a).total_cmp(&d(b)).then((a.y, a.x, a.z).cmp(&(b.y, b.x, b.z)))
        })
        .unwrap();
    let head: Vec<[i32; 3]> = rig.world.occupied().filter(|&(v, m)| m == id("lamp") && (v.x - lamp.x).abs() < 8 && (v.z - lamp.z).abs() < 8).map(|(v, _)| [v.x, v.y, v.z]).collect();
    let head_lo: [i32; 3] = std::array::from_fn(|a| head.iter().map(|v| v[a]).min().unwrap());
    let head_hi: [i32; 3] = std::array::from_fn(|a| head.iter().map(|v| v[a] + 1).max().unwrap());
    eprintln!("G5: lamp head [{head_lo:?}, {head_hi:?})");
    // Neon in open air 1 m in front of a façade, found along the street at 2 m height.
    let neon_at = (0..64)
        .map(|k| [120 + 8 * k, 32, 160])
        .find(|&[x, y, z]| (0..2).all(|dx| (0..2).all(|dy| (0..2).all(|dz| rig.world.get(VoxelCoord::new(x + dx, y + dy, z + dz)).is_none()))) && rig.world.get(VoxelCoord::new(x, y, z + 16)).is_some())
        .expect("open air 1 m from a façade");
    let edits: Vec<EditBox> = vec![
        ("remove lamp voxels", head_lo, [head_lo[0] + 2, head_lo[1] + 2, head_lo[2] + 2], None),
        ("add neon in open air", neon_at, [neon_at[0] + 2, neon_at[1] + 2, neon_at[2] + 2], Some(id("neon_pink"))),
        // Between the lamp head and the shop façades across the road (+z), 1 voxel from the head.
        ("stone beside a lamp head", [head_lo[0], head_lo[1] - 1, head_hi[2] + 1], [head_lo[0] + 4, head_lo[1] + 3, head_hi[2] + 5], Some(id("sidewalk"))),
    ];
    let mut failed = Vec::new();
    let (mut r1_e3_caught, mut r2_e3_caught, mut r2_judged) = (false, false, 0usize);
    for (label, lo, hi, mat) in edits {
        let saved = rig.voxels(lo, hi);
        let applied: Vec<_> = match mat {
            None => saved.iter().map(|&(c, _)| (c, None)).collect(),
            // Place only into empty voxels.
            Some(m) => saved.iter().filter(|(_, old)| old.is_none()).map(|&(c, _)| (c, Some(m))).collect(),
        };
        let restore: Vec<_> = saved.iter().filter(|(c, _)| applied.iter().any(|(a, _)| a == c)).copied().collect();
        eprintln!("G5 {label}: box [{lo:?}, {hi:?}), {} voxels set", applied.len());
        let t_old = rig.reference(&cam, &light, &ref_settings(), REF_SEED, 0, SPP);
        let (regions, power) = rig.edit(&applied);
        let table = rig.table().clone();
        let full = Relight { power, ..Relight::new(lo, hi) }.with_lights(power, &table);
        let e3 = Relight::new(lo, hi);
        eprintln!("  E1 power bound {power:.4e}; E2 lights {:?}", full.lights.iter().map(|l| (l.luminance * 1e3).round() / 1e3).collect::<Vec<_>>());
        let t_new = rig.reference(&cam, &light, &ref_settings(), REF_SEED, 0, SPP);
        // Host surface points on the edited world.
        let points: Vec<Option<[f64; 3]>> = (0..w * h)
            .map(|i| {
                let r = cam.ray(i % w, i / w);
                trace(&rig.world, &r, 1e9).map(|hit| std::array::from_fn(|a| r.origin[a] + hit.t * r.dir[a]))
            })
            .collect();
        rig.edit(&restore);

        // R1: unfiltered (the history is the previous frame's radiance on a still camera).
        let mut r1: BTreeMap<&str, bool> = BTreeMap::new();
        for (name, bx) in [("full", &full), ("e3", &e3)] {
            rig.new_history();
            let mut prev = None;
            for k in 1..=32u32 {
                prev = rig.frame(&cam, &light, Step { filter: false, ..Step::viewer() }, &[], k, k == 32);
            }
            let prev = prev.unwrap();
            rig.edit(&applied);
            rig.frame(&cam, &light, Step { filter: false, ..Step::viewer() }, std::slice::from_ref(bx), 33, false);
            let st = rig.state();
            rig.edit(&restore);
            let (mut relit, mut differ) = (0usize, 0usize);
            for (i, p) in points.iter().enumerate() {
                let (Some(p), r) = (*p, st[i].1) else { continue };
                if !matches!(r, reason::ACCEPTED | reason::RELIT) {
                    continue;
                }
                let hr = host_relit(p, lum4(prev[i]), sun, std::slice::from_ref(&full));
                relit += usize::from(r == reason::RELIT);
                differ += usize::from((r == reason::RELIT) != hr);
            }
            let ok = differ * 2000 <= (w * h) as usize && relit >= 100;
            eprintln!("  R1 {name}: {relit} px relit, {differ} decisions differ from the host rule {}", if ok { "(passes)" } else { "(fails)" });
            r1.insert(name, ok);
        }
        if !r1["full"] {
            failed.push(format!("R1 {label}"));
        }
        r1_e3_caught |= !r1["e3"];

        // R2: filtered arms.
        let mut arms: BTreeMap<&str, Arm> = BTreeMap::new();
        for (name, bx, reset) in [("full", Some(&full), false), ("e3", Some(&e3), false), ("reset", None, true)] {
            rig.new_history();
            for k in 1..=32u32 {
                rig.frame(&cam, &light, Step::viewer(), &[], k, false);
            }
            rig.edit(&applied);
            let boxes: Vec<Relight> = bx.into_iter().cloned().collect();
            rig.frame(&cam, &light, Step { force_reset: reset, ..Step::viewer() }, &boxes, 33, false);
            let st = rig.state();
            let mut x = None;
            for k in 34..=36u32 {
                x = rig.frame(&cam, &light, Step::viewer(), &[], k, k == 36);
            }
            rig.edit(&restore);
            arms.insert(name, (st, x.unwrap()));
        }
        let st = &arms["full"].0;
        let target: Vec<[f64; 3]> = (0..t_new.sum.len()).map(|i| t_new.mean(i)).collect();
        let changed: Vec<bool> = (0..(w * h) as usize)
            .map(|i| {
                let (a, b) = (lum(t_old.mean(i)), lum(t_new.mean(i)));
                let se = (lum(t_old.std_error(i)).powi(2) + lum(t_new.std_error(i)).powi(2)).sqrt();
                matches!(st[i].1, reason::ACCEPTED | reason::RELIT) && (a - b).abs() > 0.25 * a.max(b) && (a - b).abs() > 4.0 * se
            })
            .collect();
        let n = changed.iter().filter(|&&c| c).count();
        let missed = (0..changed.len()).filter(|&i| changed[i] && st[i].1 == reason::ACCEPTED).count();
        let calm = (0..changed.len())
            .filter(|&i| {
                st[i].1 == reason::RELIT && {
                    let (a, b) = (lum(t_old.mean(i)), lum(t_new.mean(i)));
                    (a - b).abs() <= 0.1 * a.max(b)
                }
            })
            .count();
        let e: BTreeMap<&str, Error> = arms.iter().map(|(k, (_, x))| (*k, error(x, &target, &changed))).collect();
        eprintln!(
            "  R2: {n} changed px ({missed} of them not relit: the approximation's misses; {calm} relit px changed by <= 10%); 4 frames after the edit: full {:.4}, 3E only {:.4}, reset {:.4}; rebuilt regions {}",
            e["full"].rel_mse,
            e["e3"].rel_mse,
            e["reset"].rel_mse,
            regions.len()
        );
        if n >= 100 {
            r2_judged += 1;
            if e["full"].rel_mse > 3.0 * e["reset"].rel_mse {
                failed.push(format!("R2 {label}"));
            }
            r2_e3_caught |= e["e3"].rel_mse > 3.0 * e["reset"].rel_mse;
        }
    }
    if !r1_e3_caught {
        failed.push("R1 control (3E rules only) not caught in any edit".into());
    }
    if r2_judged > 0 && !r2_e3_caught {
        failed.push("R2 control (3E rules only) not caught in any edit".into());
    }
    if r2_judged == 0 {
        failed.push("R2 judged no edit (< 100 changed px in each)".into());
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---------------------------------------------------------------------------------------------
// G6

/// G6: switching the lights resets every surface pixel in that frame only; the exposure pass's sums
/// over a 1080p night frame equal the host's over the read-back radiance plus emission.
#[test]
fn lights_switch_and_exposure_sums() {
    let (w, h) = (960, 540);
    let mut rig = Rig::new(Dressing::Full, w, h);
    let cam = street_camera(w, h);
    let light = lighting(NIGHT);
    let mut failed = Vec::new();
    rig.new_history();
    let plan: Vec<(u32, bool)> = (1..=14).map(|k| (k, (9..12).contains(&k))).collect();
    let mut prev_lights = None;
    for (k, on) in plan {
        rig.frame(&cam, &light, Step { lights: on, shade: ShadeSettings { emitters: on, ..lit() }, ..Step::viewer() }, &[], k, false);
        let st = rig.state();
        let surf: Vec<&(u32, u32)> = st.iter().filter(|s| s.1 != reason::SKY).collect();
        let reset = surf.iter().filter(|s| s.1 == reason::RESET && s.0 == 1).count();
        let accepted = surf.iter().filter(|s| s.1 == reason::ACCEPTED).count();
        let switched = prev_lights.is_some_and(|p| p != on);
        let first = prev_lights.is_none();
        let ok = if switched || first { reset == surf.len() } else { accepted * 1000 >= surf.len() * 999 };
        if switched || k == 10 || k == 13 {
            eprintln!("G6 frame {k} lights {on}: {reset} reset, {accepted} accepted of {} surface px {}", surf.len(), if ok { "" } else { "FAIL" });
        }
        if !ok {
            failed.push(format!("G6 switch frame {k}"));
        }
        prev_lights = Some(on);
    }
    rig.finish();
    // Exposure sums at 1080p.
    let (w, h) = (1920, 1080);
    let mut rig = Rig::new(Dressing::Full, w, h);
    let cam = street_camera(w, h);
    rig.new_history();
    for k in 1..=4u32 {
        rig.frame(&cam, &light, Step::viewer(), &[], k, false);
    }
    // The floor from frame 5's image (the viewer takes it from the previous frame's mean); the pass
    // runs on frame 6 and is compared with the host's sums over frame 6's read-back image.
    let x = rig.frame(&cam, &light, Step::viewer(), &[], 5, true).unwrap();
    let guides = rig.guides();
    let em = rig.emission_of(&guides);
    let pix = sample_pixels(w, h);
    let ys: Vec<f64> = pix.iter().map(|&i| lum4(x[i]) + lum(em[i])).collect();
    let floor = ys.iter().sum::<f64>() / ys.len() as f64 * DARK_FRACTION;
    let host = log_sums(&ys, floor);
    let x6 = rig.frame_with(&cam, &light, Step::viewer(), &[], 6, true, Some(floor)).unwrap();
    let ys6: Vec<f64> = pix.iter().map(|&i| lum4(x6[i]) + lum(em[i])).collect();
    let host6 = log_sums(&ys6, floor);
    let gpu6 = rig.xo.as_ref().unwrap().read(0);
    let rel = |a: f64, b: f64| ((a - b) / b.abs().max(1e-300)).abs();
    let (e_gpu, e_host) = (exposure_from_sums(&gpu6), exposure_from_sums(&host6));
    eprintln!(
        "G6 exposure sums: gpu sum {:.6e} n {} log {:.6e} lit {} | host sum {:.6e} n {} log {:.6e} lit {} | exposure gpu {e_gpu:?} host {e_host:?} (frame 5 host {:?})",
        gpu6.sum, gpu6.count, gpu6.sum_log, gpu6.lit, host6.sum, host6.count, host6.sum_log, host6.lit, exposure_from_sums(&host)
    );
    let ok = rel(gpu6.sum, host6.sum) <= 1e-3 && gpu6.count == host6.count && rel(gpu6.sum_log, host6.sum_log) <= 1e-3 && e_gpu.zip(e_host).is_some_and(|(a, b)| rel(a, b) <= 1e-3);
    if !ok {
        failed.push("G6 exposure sums".into());
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

// ---------------------------------------------------------------------------------------------
// M1

/// M1 (data): the equal-time curve. For each dressing and k = 1, 2, 4: the shade pass's GPU time at
/// 1080p (median of 64 frames after 16 warm-up frames, emitters and the bounce on) and the one-frame
/// relative MSE of the emitter-direct term alone (16 frames, 960x540) against the reference's
/// emitter-direct term (4,096 spp); and the whole frame's passes at Full, k = 1. Validation off:
/// `set NE_NO_VALIDATION=1`.
#[test]
#[ignore]
fn equal_time_curve() {
    let light = lighting(NIGHT);
    let direct = ShadeSettings { emitters: true, ..ShadeSettings::NONE };
    let rd = Settings { sun: false, sky: false, max_bounces: 0, emitters_direct: true, ..Settings::default() };
    let p = |v: &[f64], q| percentile(v, q).unwrap();
    for dressing in Dressing::ALL {
        // Error at 960x540.
        let (w, h) = (960, 540);
        let mut rig = Rig::for_timing(dressing, w, h);
        let emitters = rig.table().len();
        let cam = street_camera(w, h);
        let r = rig.reference(&cam, &light, &rd, REF_SEED, 0, 4096);
        let mean: Vec<[f64; 3]> = (0..r.sum.len()).map(|i| r.mean(i)).collect();
        let mask = surface_mask(&mut rig, &cam);
        let mut errors = BTreeMap::new();
        for k in [1u32, 2, 4] {
            let mut e = 0.0;
            for f in 0..16 {
                let x = rig.shade_frame(&cam, &light, ShadeSettings { emitter_samples: k, ..direct }, ShadeFaults::default(), 1000 + f, false);
                e += error(&x, &mean, &mask).rel_mse / 16.0;
            }
            errors.insert(k, e);
        }
        rig.finish();
        // Time at 1080p.
        let (w, h) = (1920, 1080);
        let mut rig = Rig::for_timing(dressing, w, h);
        let cam = street_camera(w, h);
        let mut timer = GpuTimer::new(&rig.g, 1, 4).unwrap();
        rig.new_history();
        for k in [1u32, 2, 4] {
            let full_frame = dressing == Dressing::Full && k == 1;
            let mut ms: Vec<Vec<f64>> = vec![Vec::new(); 4];
            for f in 0..80u32 {
                let step = if full_frame { Step { shade: ShadeSettings { emitter_samples: k, ..lit() }, ..Step::viewer() } } else { Step::shade(ShadeSettings { emitter_samples: k, ..lit() }) };
                let mut sub = rig.sub.take().unwrap();
                let cmd = sub.begin(&rig.g, &rig.tl).unwrap();
                timer.reset(&rig.g, cmd, 0);
                rig.record(cmd, &cam, &light, step, &[], f, full_frame.then_some(0.0), Some(&timer));
                let v = sub.submit(&rig.g, &mut rig.tl, cmd, &[]).unwrap();
                rig.tl.wait(&rig.g, v, u64::MAX).unwrap();
                rig.sub = Some(sub);
                let t = timer.read(&rig.g, 0).unwrap().unwrap();
                if f >= 16 {
                    for (i, x) in t.iter().enumerate() {
                        ms[i].push(*x);
                    }
                }
            }
            let shade_ms = p(&ms[0], 50.0);
            let e = errors[&k];
            eprintln!(
                "M1 {} ({emitters} emitters) k {k}: shade {shade_ms:.3} ms (p90 {:.3}), one-frame emitter-direct rel_mse {e:.4}, rel_mse x ms {:.4}{}",
                dressing.name(),
                p(&ms[0], 90.0),
                e * shade_ms,
                if full_frame { format!(" | whole frame: temporal {:.3}, filter {:.3}, exposure {:.3} ms", p(&ms[1], 50.0), p(&ms[2], 50.0), p(&ms[3], 50.0)) } else { String::new() }
            );
        }
        timer.destroy(&rig.g);
        rig.finish();
    }
}
