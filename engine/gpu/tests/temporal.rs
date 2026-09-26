//! Phase 3D GPU tests: the temporal foundation (`gpu::temporal`, ADR-0006) against host
//! re-derivations: the running mean, the reprojection and the per-tap decisions, edits and global
//! resets, with planted faults. Criteria are declared in `docs/changes/2026-09-24-phase3d-temporal.md`.
//! Like the other GPU tests they need the Vulkan SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test temporal -- --test-threads=1 --nocapture`.

use std::collections::{BTreeMap, BTreeSet};

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::alloc::Allocator;
use gpu::debug_view::{targets_to_read, Tables};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::raster::{Camera, Faults, Frame, Raster, Targets, BACKGROUND_SURFACE};
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::scene::{affected_regions, region_meshes, GpuScene};
use gpu::shade::{self, Shade, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::timing::{percentile, GpuTimer};
use gpu::temporal::{reason, History, ResetCause, Temporal, TemporalFaults, TemporalSettings};
use gpu::{Gpu, GpuError, Timeline};
use light::reference::{albedos, Lighting, Settings};
use light::sky::SkyLuts;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::{scene, BrickKey, Transaction, World};

const NEAR: f64 = 0.1;
const SEED: u32 = 0x3D;
/// Sun and sky without the 3F bounce: the lighting this file's criteria were measured with.
const DIRECT: ShadeSettings = ShadeSettings { sun: true, sky: true, point_sun: false, uniform_sky: false, bounce: false, emitters: false, emitter_spp: 1 };
const SUN_ONLY: ShadeSettings = ShadeSettings { sun: true, sky: false, point_sun: false, uniform_sky: false, bounce: false, emitters: false, emitter_spp: 1 };
const W: u32 = 320;
const H: u32 = 180;

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3D needs the ray-tracing device (P-001)");
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

/// One frame's outputs read back: the fresh shade radiance, the resolved radiance and state, motion,
/// and the G-buffer (for the host re-derivation).
struct Out {
    fresh: Vec<[f32; 4]>,
    resolved: Vec<[f32; 4]>,
    state: Vec<(u32, u32)>,
    motion: Vec<[f32; 2]>,
    gbuffer: Frame,
    cause: ResetCause,
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
    raster: Raster,
    shade: Shade,
    temporal: Temporal,
}

impl Rig {
    fn new() -> Rig {
        Self::with_gpu(gpu())
    }

    /// For timing runs: validation may be off (`NE_NO_VALIDATION`).
    fn for_timing() -> Rig {
        let g = Gpu::new().expect("an RT-capable Vulkan device");
        assert!(g.ray_tracing());
        Self::with_gpu(g)
    }

    fn with_gpu(g: Gpu) -> Rig {
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
        let targets = Targets::new(&g, &mut alloc, vk::Extent2D { width: W, height: H }).unwrap();
        let out = ShadeTargets::new(&g, &mut alloc, W, H).unwrap();
        let history = History::new(&g, &mut alloc, W, H).unwrap();
        let raster = Raster::new(&g).unwrap();
        let shade = Shade::new(&g).unwrap();
        let temporal = Temporal::new(&g).unwrap();
        Rig {
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
            raster,
            shade,
            temporal,
        }
    }

    /// Raster, sky table, shade (read back: fresh), then temporal (read back: resolved, state, motion).
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, cam: &Camera, light: &Lighting, set: ShadeSettings, ts: TemporalSettings, faults: TemporalFaults, force_reset: bool, frame: u32) -> Out {
        let (g, targets, out, s) = (&self.g, self.targets.as_ref().unwrap(), self.out.as_ref().unwrap(), self.scene.as_ref().unwrap());
        let tables = self.tables.as_ref().unwrap();
        let rb = self.raster.bind(g, &s.meshes).unwrap();
        let sb = self.shade.bind(g, targets, s.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), out, &self.sky.as_ref().unwrap().view).unwrap();
        let tb = self.temporal.bind(g, targets, &tables.regions, &out.radiance, self.history.as_ref().unwrap()).unwrap();
        let mut sub = Submitter::new(g).unwrap();
        let cmd = sub.begin(g, &self.tl).unwrap();
        self.sky_pass.as_ref().unwrap().record(g, cmd, light.sun_dir);
        self.raster.record(g, cmd, targets, &s.meshes, &rb, cam, Faults::default());
        targets_to_read(g, cmd, targets);
        self.shade.record(g, cmd, &sb, out, &shade::Params::new(cam, light, set, ShadeFaults::default(), frame, SEED));
        let v = sub.submit(g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(g, v, u64::MAX).unwrap();
        let fresh = out.read(g, &mut self.alloc, &mut self.tl).unwrap();
        let cmd = sub.begin(g, &self.tl).unwrap();
        let cause = self.temporal.record(g, cmd, &tb, self.history.as_mut().unwrap(), cam, light.sun_dir, ts, faults, force_reset, &[]);
        let v = sub.submit(g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(g, v, u64::MAX).unwrap();
        let resolved = out.read(g, &mut self.alloc, &mut self.tl).unwrap();
        let (state, motion) = self.history.as_ref().unwrap().read(g, &mut self.alloc, &mut self.tl).unwrap();
        g.wait_idle().unwrap();
        sub.destroy(g);
        rb.destroy(g);
        sb.destroy(g);
        tb.destroy(g);
        // The G-buffer of this frame, for the host (a separate raster readback of the same camera).
        let b2 = self.raster.bind(g, &s.meshes).unwrap();
        let gbuffer = self.raster.render_to_host(g, &mut self.alloc, &mut self.tl, targets, &s.meshes, &b2, cam, Faults::default()).unwrap();
        b2.destroy(g);
        Out { fresh, resolved, state, motion, gbuffer, cause }
    }

    /// Removes the voxels of a small box, publishes, updates the device scene and the region table.
    /// Returns the region keys whose snapshot changed.
    fn edit(&mut self, lo: [i32; 3], hi: [i32; 3]) -> BTreeSet<RegionKey> {
        let mut tx = Transaction::new();
        for x in lo[0]..hi[0] {
            for y in lo[1]..hi[1] {
                for z in lo[2]..hi[2] {
                    tx.set(world::VoxelCoord::new(x, y, z), None);
                }
            }
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
        regions
    }

    fn reference_mean(&mut self, cam: &Camera, light: &Lighting, frames: u32) -> Vec<[f64; 3]> {
        let r = Reference::new(&self.g).unwrap();
        let out = RefAccum::new(&self.g, &mut self.alloc, cam.width, cam.height).unwrap();
        let s = self.scene.as_ref().unwrap();
        let b = r.bind(&self.g, s.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &out).unwrap();
        let set = Settings { sky: false, max_bounces: 0, ..Settings::default() };
        r.accumulate(&self.g, &mut self.tl, &b, &out, cam, light, &set, RefFaults::default(), SEED, 1_000_000..1_000_000 + frames, 1_000_000, 64).unwrap();
        let img = r.read(&self.g, &mut self.alloc, &mut self.tl, &out, frames).unwrap();
        self.g.wait_idle().unwrap();
        b.destroy(&self.g);
        out.free(&self.g, &mut self.alloc);
        r.destroy(&self.g);
        let a = &img.accum;
        (0..a.sum.len()).map(|i| a.mean(i)).collect()
    }

    fn finish(mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let (errors, warnings) = g.validation_counts();
        eprintln!("validation: {errors} errors, {warnings} warnings");
        assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
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
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

fn is_surface(f: &Frame, i: usize) -> bool {
    f.depth[i] > 0.0 && f.surface[i] != BACKGROUND_SURFACE
}

/// Criterion 1: layout checked; with a static camera the resolved colour after frame k equals the
/// host's running mean of the fresh samples (within 1e-4 relative on at least 99.9% of surface
/// pixels) and the age is k on every surface pixel (frame 1 is the reset). Planted "reset every
/// frame" and "no fresh sample" must fail it.
#[test]
fn static_history_is_the_running_mean() {
    let mut rig = Rig::new();
    match Temporal::with_reflection(&rig.g, gpu::shade::REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("temporal pipeline with the shade reflection refused: {} errors", e.len()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }
    let cam = street_camera(W, H);
    let light = lighting(8.0);
    let frames = 24u32;
    let mut failed = Vec::new();
    for (fname, faults) in [("ok", TemporalFaults::default()), ("reset_always", TemporalFaults { reset_always: true, ..TemporalFaults::default() }), ("no_fresh", TemporalFaults { no_fresh: true, ..TemporalFaults::default() })] {
        // A fresh history per arm.
        rig.history.take().unwrap().free(&rig.g, &mut rig.alloc);
        rig.history = Some(History::new(&rig.g, &mut rig.alloc, W, H).unwrap());
        let mut mean: Vec<[f64; 3]> = vec![[0.0; 3]; (W * H) as usize];
        let (mut worst_bad, mut age_bad, mut surfaces) = (0usize, 0usize, 0usize);
        for k in 1..=frames {
            let o = rig.frame(&cam, &light, ShadeSettings::default(), TemporalSettings { max_age: 64, ..TemporalSettings::default() }, faults, false, k);
            assert_eq!(o.cause.first, k == 1);
            if k == frames {
                surfaces = (0..mean.len()).filter(|&i| is_surface(&o.gbuffer, i)).count();
            }
            for (i, m) in mean.iter_mut().enumerate() {
                for (mc, &x) in m.iter_mut().zip(&o.fresh[i][..3]) {
                    *mc += (x as f64 - *mc) / k as f64;
                }
                if k == frames && is_surface(&o.gbuffer, i) {
                    let r = o.resolved[i];
                    let e = (0..3).map(|c| (r[c] as f64 - m[c]).abs() / m[c].abs().max(1e-6)).fold(0.0, f64::max);
                    if e > 1e-4 {
                        worst_bad += 1;
                    }
                    if o.state[i] != (k, reason::ACCEPTED) {
                        age_bad += 1;
                    }
                }
            }
        }
        let pass = worst_bad * 1000 <= surfaces && age_bad == 0;
        eprintln!("static {fname}: {surfaces} surface px; {worst_bad} off the running mean by more than 1e-4; {age_bad} with state other than (age {frames}, accepted)");
        match (fname, pass) {
            ("ok", false) => failed.push("ok".to_string()),
            (f, true) if f != "ok" => failed.push(format!("fault {f} not caught")),
            _ => {}
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Criterion 2: the accumulated sun term (static camera, 256 frames, max_age 1024) converges to the
/// reference's sun component: image mean per channel within 1%.
#[test]
fn accumulation_converges_to_the_reference() {
    let mut rig = Rig::new();
    let cam = street_camera(W, H);
    let light = lighting(8.0);
    let mut last = None;
    for k in 1..=256u32 {
        last = Some(rig.frame(&cam, &light, SUN_ONLY, TemporalSettings { max_age: 1024, ..TemporalSettings::default() }, TemporalFaults::default(), false, k));
    }
    let o = last.unwrap();
    let rf = rig.reference_mean(&cam, &light, 4096);
    let n = rf.len() as f64;
    let rt: [f64; 3] = [0, 1, 2].map(|c| o.resolved.iter().map(|p| p[c] as f64).sum::<f64>() / n);
    let re: [f64; 3] = [0, 1, 2].map(|c| rf.iter().map(|p| p[c]).sum::<f64>() / n);
    let rel = [0, 1, 2].map(|c| rt[c] / re[c] - 1.0);
    eprintln!("convergence: accumulated {rt:?} reference {re:?} relative {rel:?}");
    rig.finish();
    assert!(rel.iter().all(|r| r.abs() < 0.01), "{rel:?}");
}

/// Host re-derivation of one pixel's decision (ADR-0006), from the two frames' G-buffers.
struct HostGuide {
    face: u32,
    plane: i64,
    material: u16,
    key: RegionKey,
    snapshot: u64,
}

fn host_guides(f: &Frame, cam: &Camera, rows: &[(RegionKey, u64)]) -> Vec<Option<HostGuide>> {
    (0..f.depth.len())
        .map(|i| {
            if !is_surface(f, i) {
                return None;
            }
            let (x, y) = ((i as u32) % f.width, (i as u32) / f.width);
            let r = cam.ray(x, y);
            let t = cam.distance(f.depth[i]);
            let p = [0, 1, 2].map(|a| r.origin[a] + t * r.dir[a]);
            let e = f.normal[i].map(|b| (b as f64 / 32767.0).max(-1.0));
            let mut n = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
            if n[2] < 0.0 {
                let sg = |v: f64| if v >= 0.0 { 1.0 } else { -1.0 };
                let (a, b) = (n[0], n[1]);
                n[0] = (1.0 - b.abs()) * sg(a);
                n[1] = (1.0 - a.abs()) * sg(b);
            }
            let axis = (0..3).max_by(|&a, &b| n[a].abs().partial_cmp(&n[b].abs()).unwrap()).unwrap();
            let face = axis as u32 * 2 + u32::from(n[axis] >= 0.0);
            let (key, snapshot) = rows[f.surface[i][0] as usize];
            Some(HostGuide { face, plane: p[axis].round() as i64, material: f.material[i], key, snapshot })
        })
        .collect()
}

/// The host's expected reason for pixel `i` of the current frame, and its previous position.
fn host_decision(i: usize, cur: &Frame, cam: &Camera, prev_cam: &Camera, cg: &[Option<HostGuide>], pg: &[Option<HostGuide>]) -> (u32, Option<[f64; 2]>) {
    let Some(c) = &cg[i] else { return (reason::SKY, None) };
    let (x, y) = ((i as u32) % cur.width, (i as u32) / cur.width);
    let r = cam.ray(x, y);
    let t = cam.distance(cur.depth[i]);
    let p = [0, 1, 2].map(|a| r.origin[a] + t * r.dir[a]);
    let v = [p[0] - prev_cam.eye[0], p[1] - prev_cam.eye[1], p[2] - prev_cam.eye[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let z = dot(v, prev_cam.forward);
    if z <= prev_cam.near {
        return (reason::OFFSCREEN, None);
    }
    let nx = dot(v, prev_cam.right) / (z * prev_cam.tan_half_x);
    let ny = dot(v, prev_cam.up) / (z * prev_cam.tan_half_y);
    if !(-1.0..=1.0).contains(&nx) || !(-1.0..=1.0).contains(&ny) {
        return (reason::OFFSCREEN, None);
    }
    let (w, h) = (cur.width as i64, cur.height as i64);
    let pp = [(nx + 1.0) * 0.5 * w as f64, (1.0 - ny) * 0.5 * h as f64];
    let f = [pp[0] - 0.5, pp[1] - 0.5];
    let base = [f[0].floor() as i64, f[1].floor() as i64];
    let snap = |t: f64| if t < 1e-3 { 0.0 } else if t > 1.0 - 1e-3 { 1.0 } else { t };
    let tt = [snap(f[0] - base[0] as f64), snap(f[1] - base[1] as f64)];
    let (mut any, mut nearest, mut nearest_w) = (false, reason::NO_PREV, -1.0);
    for k in 0..4 {
        let o = [k & 1, k >> 1];
        let q = [base[0] + o[0], base[1] + o[1]];
        let wgt = (if o[0] == 1 { tt[0] } else { 1.0 - tt[0] }) * (if o[1] == 1 { tt[1] } else { 1.0 - tt[1] });
        let mut tr = reason::OFFSCREEN;
        if q[0] >= 0 && q[1] >= 0 && q[0] < w && q[1] < h {
            tr = match &pg[(q[1] * w + q[0]) as usize] {
                None => reason::NO_PREV,
                Some(pv) if pv.key == c.key && pv.snapshot != c.snapshot => reason::EDITED,
                Some(pv) if pv.material != c.material => reason::MATERIAL,
                Some(pv) if pv.face != c.face => reason::NORMAL,
                Some(pv) if pv.plane != c.plane => reason::DISOCCLUDED,
                Some(_) => reason::ACCEPTED,
            };
            if tr == reason::ACCEPTED && wgt > 0.0 {
                any = true;
            }
        }
        if wgt > nearest_w {
            nearest_w = wgt;
            nearest = tr;
        }
    }
    let rsn = if any { reason::ACCEPTED } else if nearest == reason::ACCEPTED { reason::NO_PREV } else { nearest };
    (rsn, Some([pp[0] - (x as f64 + 0.5), pp[1] - (y as f64 + 0.5)]))
}

/// Criteria 3 and 4: after a camera move, the GPU's motion equals the host's reprojection (within
/// 0.01 px) and its per-pixel reasons equal the host's decisions: at most 0.05% of pixels and 5% of
/// the host's rejections differ, with a non-trivial number of rejections. Planted "never reject"
/// must fail it. (The first run allowed 0.5% of pixels, looser than the fault's own footprint of
/// about 0.35%: the control could not fail. The correct arms agreed on every pixel.)
#[test]
fn moving_camera_decisions_match_the_host() {
    let mut rig = Rig::new();
    let light = lighting(8.0);
    let rows = rig.scene.as_ref().unwrap().region_rows();
    let mut failed = Vec::new();
    for (fname, faults) in [("ok", TemporalFaults::default()), ("never_reject", TemporalFaults { never_reject: true, ..TemporalFaults::default() })] {
        for (dx, yaw) in [(4.0, 0.0), (-6.0, 3.0), (0.0, 8.0)] {
            rig.history.take().unwrap().free(&rig.g, &mut rig.alloc);
            rig.history = Some(History::new(&rig.g, &mut rig.alloc, W, H).unwrap());
            let (a, b) = (street_camera(W, H), moved_camera(dx, yaw, W, H));
            let oa = rig.frame(&a, &light, ShadeSettings::default(), TemporalSettings::default(), faults, false, 1);
            let ob = rig.frame(&b, &light, ShadeSettings::default(), TemporalSettings::default(), faults, false, 2);
            let (ga, gb) = (host_guides(&oa.gbuffer, &a, &rows), host_guides(&ob.gbuffer, &b, &rows));
            let n = gb.len();
            let (mut agree, mut rejected, mut motion_bad, mut on_screen) = (0usize, 0usize, 0usize, 0usize);
            let mut hist: BTreeMap<(u32, u32), usize> = BTreeMap::new();
            for i in 0..n {
                let (want, mv) = host_decision(i, &ob.gbuffer, &b, &a, &gb, &ga);
                let got = ob.state[i].1;
                if got == want {
                    agree += 1;
                } else {
                    *hist.entry((want, got)).or_default() += 1;
                }
                if want != reason::ACCEPTED && want != reason::SKY {
                    rejected += 1;
                }
                if let Some(m) = mv {
                    on_screen += 1;
                    let g = ob.motion[i];
                    if (g[0] as f64 - m[0]).abs() > 0.01 || (g[1] as f64 - m[1]).abs() > 0.01 {
                        motion_bad += 1;
                    }
                }
            }
            let differ = n - agree;
            let pass = differ * 2000 <= n && differ * 20 <= rejected && motion_bad * 1000 <= on_screen && rejected > 100;
            let top: Vec<String> = hist.iter().rev().take(4).map(|((w, g), c)| format!("host {} gpu {}: {c}", reason::NAMES[*w as usize], reason::NAMES[*g as usize])).collect();
            eprintln!(
                "move {fname} dx {dx} yaw {yaw}: {agree} of {n} reasons agree ({:.3}%), host rejects {rejected}, motion off by > 0.01 px on {motion_bad} of {on_screen}; disagreements {top:?}",
                100.0 * agree as f64 / n as f64
            );
            match (fname, pass) {
                ("ok", false) => failed.push(format!("ok dx {dx} yaw {yaw}")),
                ("never_reject", true) => failed.push(format!("fault never_reject not caught dx {dx} yaw {yaw}")),
                _ => {}
            }
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Criterion 5: after an edit, the GPU's decisions equal the host's re-derivation from the two
/// frames' G-buffers and region tables on every pixel (at most 0.05% differ), "edited" occurs, and
/// only in rebuilt regions. (The first version required 98% of the rebuilt region's pixels to be
/// "edited"; pixels whose taps lie across a region border get the other region's verdict by the
/// ADR-0006 rule, 9 of 295 there, so that criterion did not match the contract.)
#[test]
fn an_edit_resets_only_its_regions() {
    let mut rig = Rig::new();
    let cam = street_camera(W, H);
    let light = lighting(8.0);
    rig.frame(&cam, &light, ShadeSettings::default(), TemporalSettings::default(), TemporalFaults::default(), false, 1);
    let rows_before = rig.scene.as_ref().unwrap().region_rows();
    let o1 = rig.frame(&cam, &light, ShadeSettings::default(), TemporalSettings::default(), TemporalFaults::default(), false, 2);
    let c = (W / 2 + (H / 2) * W) as usize;
    assert!(is_surface(&o1.gbuffer, c), "the screen centre shows a surface");
    let hit = world::reference::trace(&rig.world, &cam.ray(W / 2, H / 2), 1e9).unwrap();
    let v = hit.voxel;
    let changed = rig.edit([v.x - 2, v.y - 2, v.z - 2], [v.x + 2, v.y + 2, v.z + 2]);
    let rows = rig.scene.as_ref().unwrap().region_rows();
    let o3 = rig.frame(&cam, &light, ShadeSettings::default(), TemporalSettings::default(), TemporalFaults::default(), false, 3);
    let (g1, g3) = (host_guides(&o1.gbuffer, &cam, &rows_before), host_guides(&o3.gbuffer, &cam, &rows));
    let n = o3.state.len();
    let (mut differ, mut edited, mut edited_out, mut in_changed) = (0usize, 0usize, 0usize, 0usize);
    let mut hist: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for i in 0..n {
        let (want, _) = host_decision(i, &o3.gbuffer, &cam, &cam, &g3, &g1);
        let got = o3.state[i].1;
        if got != want {
            differ += 1;
            *hist.entry((want, got)).or_default() += 1;
        }
        if is_surface(&o3.gbuffer, i) {
            let key = rows[o3.gbuffer.surface[i][0] as usize].0;
            let inside = changed.contains(&key);
            in_changed += usize::from(inside);
            if got == reason::EDITED {
                edited += 1;
                edited_out += usize::from(!inside);
            }
        }
    }
    let top: Vec<String> = hist.iter().map(|((w, g), c)| format!("host {} gpu {}: {c}", reason::NAMES[*w as usize], reason::NAMES[*g as usize])).collect();
    eprintln!("edit: {} regions rebuilt, {in_changed} px in them; {edited} px edited ({edited_out} outside them); {differ} of {n} decisions differ from the host {top:?}", changed.len());
    rig.finish();
    assert!(differ * 2000 <= n, "{differ} decisions differ");
    assert!(edited > 0 && edited_out == 0, "edited {edited}, outside {edited_out}");
}

/// Criterion 6: a camera cut and a light jump reset every surface pixel; slow sun motion does not.
#[test]
fn global_resets() {
    let mut rig = Rig::new();
    let light = lighting(8.0);
    let cam = street_camera(W, H);
    let run = |rig: &mut Rig, cam: &Camera, light: &Lighting, k: u32| rig.frame(cam, light, ShadeSettings::default(), TemporalSettings::default(), TemporalFaults::default(), false, k);
    run(&mut rig, &cam, &light, 1);
    run(&mut rig, &cam, &light, 2);
    let count = |o: &Out, r: u32| (0..o.state.len()).filter(|&i| is_surface(&o.gbuffer, i) && o.state[i].1 == r).count();
    let surfaces = |o: &Out| (0..o.state.len()).filter(|&i| is_surface(&o.gbuffer, i)).count();
    // Slow sun motion: 0.25 minutes of the day, about 0.06 degrees.
    let slow = lighting(8.0 + 0.25 / 60.0);
    let o = run(&mut rig, &cam, &slow, 3);
    eprintln!("slow sun: cause {:?}, {} of {} accepted", o.cause, count(&o, reason::ACCEPTED), surfaces(&o));
    let slow_ok = !o.cause.any() && count(&o, reason::ACCEPTED) == surfaces(&o);
    // Light jump: 10 minutes, 2.5 degrees.
    let jump = lighting(8.0 + 10.0 / 60.0);
    let o = run(&mut rig, &cam, &jump, 4);
    eprintln!("light jump: cause {:?}, {} of {} reset", o.cause, count(&o, reason::RESET), surfaces(&o));
    let jump_ok = o.cause.light && count(&o, reason::RESET) == surfaces(&o);
    // Camera cut: 100 voxels.
    let far = moved_camera(100.0, 0.0, W, H);
    let o = run(&mut rig, &far, &jump, 5);
    eprintln!("camera cut: cause {:?}, {} of {} reset", o.cause, count(&o, reason::RESET), surfaces(&o));
    let cut_ok = o.cause.cut && count(&o, reason::RESET) == surfaces(&o);
    rig.finish();
    assert!(slow_ok && jump_ok && cut_ok, "slow {slow_ok} jump {jump_ok} cut {cut_ok}");
}

/// Data only: the street at 960x540 after 1 and 64 accumulated frames (sun and sky, still camera),
/// tone-mapped as the viewer does, written to `results/temporal_3d_<hour>_<frames>.ppm`.
#[test]
fn accumulated_images() {
    let (w, h) = (960u32, 540u32);
    let mut rig = Rig::new();
    rig.history.take().unwrap().free(&rig.g, &mut rig.alloc);
    rig.out.take().unwrap().free(&rig.g, &mut rig.alloc);
    rig.targets.take().unwrap().free(&rig.g, &mut rig.alloc);
    rig.targets = Some(Targets::new(&rig.g, &mut rig.alloc, vk::Extent2D { width: w, height: h }).unwrap());
    rig.out = Some(ShadeTargets::new(&rig.g, &mut rig.alloc, w, h).unwrap());
    let luts = SkyLuts::new(Atmosphere::default());
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results");
    let cam = street_camera(w, h);
    for hour in [8.0, 17.75] {
        rig.history = Some(History::new(&rig.g, &mut rig.alloc, w, h).unwrap());
        let light = lighting(hour);
        let g = light.sun_at_ground;
        let y = 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2];
        let e = luts.ground_irradiance(light.sun_dir[1]);
        let exposure = 1.0 / (y / std::f64::consts::PI + (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]) / std::f64::consts::PI + 1e-7);
        for k in 1..=64u32 {
            let o = rig.frame(&cam, &light, DIRECT, TemporalSettings::default(), TemporalFaults::default(), false, k);
            if k == 1 || k == 64 {
                let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
                let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
                let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
                for px in &o.resolved {
                    for &c in &px[..3] {
                        bytes.push((srgb(aces(c as f64 * exposure)) * 255.0 + 0.5) as u8);
                    }
                }
                let path = dir.join(format!("temporal_3d_{hour}_{k}.ppm"));
                std::fs::write(&path, bytes).unwrap();
                eprintln!("wrote {}", path.display());
            }
        }
        rig.history.take().unwrap().free(&rig.g, &mut rig.alloc);
    }
    rig.history = Some(History::new(&rig.g, &mut rig.alloc, w, h).unwrap());
    rig.finish();
}

/// Data only: GPU time of the shade pass and the temporal pass at 1080p, separately (validation off
/// gives the timing run: `set NE_NO_VALIDATION=1`).
#[test]
fn temporal_cost_at_1080p() {
    const FW: u32 = 1920;
    const FH: u32 = 1080;
    let mut rig = Rig::for_timing();
    let validation = rig.g.validation_enabled();
    rig.history.take().unwrap().free(&rig.g, &mut rig.alloc);
    rig.out.take().unwrap().free(&rig.g, &mut rig.alloc);
    rig.targets.take().unwrap().free(&rig.g, &mut rig.alloc);
    let targets = Targets::new(&rig.g, &mut rig.alloc, vk::Extent2D { width: FW, height: FH }).unwrap();
    let out = ShadeTargets::new(&rig.g, &mut rig.alloc, FW, FH).unwrap();
    let mut history = History::new(&rig.g, &mut rig.alloc, FW, FH).unwrap();
    let light = lighting(8.0);
    let s = rig.scene.as_ref().unwrap();
    let rb = rig.raster.bind(&rig.g, &s.meshes).unwrap();
    let sb = rig.shade.bind(&rig.g, &targets, s.accel.as_ref().unwrap(), rig.mats.as_ref().unwrap(), &out, &rig.sky.as_ref().unwrap().view).unwrap();
    let tb = rig.temporal.bind(&rig.g, &targets, &rig.tables.as_ref().unwrap().regions, &out.radiance, &history).unwrap();
    let mut timer = GpuTimer::new(&rig.g, 1, 2).unwrap();
    let mut sub = Submitter::new(&rig.g).unwrap();
    let (mut sh, mut tp) = (Vec::new(), Vec::new());
    for rep in 0..62u32 {
        // A slowly turning camera, so the history is reprojected.
        let cam = moved_camera(0.0, rep as f64 * 0.1, FW, FH);
        let g = &rig.g;
        let cmd = sub.begin(g, &rig.tl).unwrap();
        rig.sky_pass.as_ref().unwrap().record(g, cmd, light.sun_dir);
        timer.reset(g, cmd, 0);
        rig.raster.record(g, cmd, &targets, &s.meshes, &rb, &cam, Faults::default());
        targets_to_read(g, cmd, &targets);
        timer.begin_pass(g, cmd, 0, 0);
        rig.shade.record(g, cmd, &sb, &out, &shade::Params::new(&cam, &light, DIRECT, ShadeFaults::default(), rep, SEED));
        timer.end_pass(g, cmd, 0, 0);
        timer.begin_pass(g, cmd, 0, 1);
        rig.temporal.record(g, cmd, &tb, &mut history, &cam, light.sun_dir, TemporalSettings::default(), TemporalFaults::default(), false, &[]);
        timer.end_pass(g, cmd, 0, 1);
        let v = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
        rig.tl.wait(g, v, u64::MAX).unwrap();
        let ms = timer.read(g, 0).unwrap().unwrap();
        if rep >= 2 {
            sh.push(ms[0]);
            tp.push(ms[1]);
        }
    }
    let p = |v: &[f64], q| percentile(v, q).unwrap();
    eprintln!(
        "cost 1080p (validation {validation}): shade (sun + sky) median {:.3} p90 {:.3} ms | temporal median {:.3} p90 {:.3} ms (60 reps, turning camera)",
        p(&sh, 50.0),
        p(&sh, 90.0),
        p(&tp, 50.0),
        p(&tp, 90.0)
    );
    let g = &rig.g;
    g.wait_idle().unwrap();
    sub.destroy(g);
    timer.destroy(g);
    rb.destroy(g);
    sb.destroy(g);
    tb.destroy(g);
    history.free(g, &mut rig.alloc);
    out.free(g, &mut rig.alloc);
    targets.free(g, &mut rig.alloc);
    rig.targets = Some(Targets::new(&rig.g, &mut rig.alloc, vk::Extent2D { width: W, height: H }).unwrap());
    rig.out = Some(ShadeTargets::new(&rig.g, &mut rig.alloc, W, H).unwrap());
    rig.history = Some(History::new(&rig.g, &mut rig.alloc, W, H).unwrap());
    rig.finish();
}

