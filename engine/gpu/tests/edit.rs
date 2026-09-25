//! Phase 2E GPU integration tests: edits end to end (world commit → 1C jobs → publication → region
//! meshes → upload → BLAS/TLAS update → swap → retirement), checked per pixel against the CPU
//! reference of the edited world for raster and ray query; the granularity sweep's measurements;
//! a refused grant; planted faults. Criteria: `docs/changes/2026-09-24-phase2e-edits.md`.
//! Like the other GPU tests, they need the Vulkan SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test edit -- --test-threads=1 --nocapture`.
//! Timing run: the same with `NE_NO_VALIDATION=1`. The validation layer's host overhead dominates
//! the host stages (thousands of draws and descriptor writes), so latency is measured with it off;
//! such a run still makes every correctness check but cannot check validation (reported, not passed).

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use ash::vk;
use derived::{Config, Merge, Pipeline};
use gpu::alloc::{Allocator, Kind};
use gpu::debug_view::{DebugView, Lighting, Source, Tables, View};
use gpu::equivalence::{self, RefPixel};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::raster::{Camera, Faults, Frame, Raster, Targets};
use gpu::ray::{RayPrimary, RayTargets};
use gpu::scene::{affected_regions, region_meshes, GpuScene, UpdateStats};
use gpu::staging::{download, Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, GpuError, Timeline};
use memory::{Budget, Category};
use world::dims::VOXEL_SIZE_M;
use world::{scene, BrickKey, MaterialId, Transaction, VoxelCoord, World};

const W: u32 = 640;
const H: u32 = 360;
const FW: u32 = 1920;
const FH: u32 = 1080;
const NEAR: f64 = 0.1;
const REF_THREADS: usize = 2;
/// Latency repetitions per edit kind (criterion 6), then one clearing edit.
const REPS: usize = 10;
/// Interleaved repetitions of the trace-cost measurement (criterion 6).
const COST_REPS: usize = 30;

/// The `perf_phase1` edits: one voxel, an 8^3 box straddling 8 bricks, a 32^3 box (half open).
const EDITS: [(&str, [i32; 3], [i32; 3]); 3] = [("voxel", [100, 4, 60], [101, 5, 61]), ("box8", [100, 4, 60], [108, 12, 68]), ("box32", [96, 0, 56], [128, 32, 88])];

fn timing_run() -> bool {
    std::env::var_os("NE_NO_VALIDATION").is_some()
}

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    if timing_run() {
        eprintln!("TIMING RUN: validation off (NE_NO_VALIDATION); validation checks NOT RUN");
    } else {
        assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    }
    assert!(g.ray_tracing(), "2E needs the ray-tracing device (P-001)");
    g
}

fn assert_clean(g: &Gpu) {
    if timing_run() {
        eprintln!("validation: NOT RUN (timing run)");
        return;
    }
    let (errors, warnings) = g.validation_counts();
    eprintln!("validation: {errors} errors, {warnings} warnings");
    assert_eq!((errors, warnings), (0, 0), "validation: {:?}", g.first_validation_errors());
}

fn v(p: [i32; 3]) -> VoxelCoord {
    VoxelCoord::new(p[0], p[1], p[2])
}

/// The edit close-up and the street view (criterion 2), and the 2C-1 cameras for trace cost.
fn check_cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![("edit_close", Camera::look_at([70.5, 26.25, 28.75], [108.0, 6.0, 68.0], 60.0, w, h, NEAR)), ("street_view", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR))]
}

fn cost_cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        ("street_view", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)),
        ("chunk_corner", Camera::look_at([128.0, 32.0, 64.0], [250.0, 20.0, 190.0], 70.0, w, h, NEAR)),
        ("overhead", Camera::look_at([-60.5, 300.25, -80.75], [192.0, 0.0, 150.0], 50.0, w, h, NEAR)),
        ("grazing", Camera::look_at([4.3, 8.6, 20.2], [380.0, 3.0, 30.0], 60.0, w, h, NEAR)),
    ]
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Runs every job inline and publishes. Returns the published brick keys.
fn drain(p: &mut Pipeline, w: &World) -> (BTreeSet<BrickKey>, f64, f64) {
    let t = Instant::now();
    loop {
        let jobs = p.dispatch(w);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            p.complete(w, j.run());
        }
    }
    let jobs_ms = ms(t);
    let t = Instant::now();
    let published = p.try_publish(w);
    let publish_ms = ms(t);
    assert!(p.is_idle(), "pipeline did not drain");
    (published.map(|x| x.groups.into_iter().flatten().collect()).unwrap_or_default(), jobs_ms, publish_ms)
}

fn populated(w: &World, merge: Merge) -> Pipeline {
    let mut p = Pipeline::new(Config { merge, ..Config::default() });
    p.mark_all(w);
    drain(&mut p, w);
    p
}

/// Every region of the pipeline's current snapshot, built from scratch.
fn full_regions(p: &mut Pipeline, size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    let t = p.acquire();
    let keys: Vec<BrickKey> = p.current().keys().collect();
    let meshes: Vec<_> = keys.iter().map(|&k| (k, p.read(&t, k).unwrap().unwrap().clone())).collect();
    p.release(t).unwrap();
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
}

/// Region meshes of the brick keys a publication changed, from the current snapshot.
fn changed_regions(p: &mut Pipeline, keys: &BTreeSet<BrickKey>, size: RegionSize) -> BTreeMap<RegionKey, Option<RegionMesh>> {
    let t = p.acquire();
    let r = region_meshes(p, &t, &affected_regions(keys.iter().copied(), size), size).unwrap();
    p.release(t).unwrap();
    r
}

fn edit_tx(lo: [i32; 3], hi: [i32; 3], m: Option<MaterialId>) -> Transaction {
    let mut tx = Transaction::new();
    tx.fill(v(lo), v(hi), m);
    tx
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    /// Readbacks for checks: a separate, unlimited ledger, so a test budget governs only the scene.
    check: Allocator,
    tl: Timeline,
    up: Uploader,
    raster: Raster,
    ray: RayPrimary,
    small: Targets,
    small_ray: RayTargets,
    full: Targets,
    full_ray: RayTargets,
}

impl Rig {
    fn new() -> Rig {
        Self::with_budget(None)
    }

    fn with_budget(budget: Option<Budget>) -> Rig {
        let g = gpu();
        let mut alloc = match budget {
            Some(b) => Allocator::with_block_bytes(b, 1 << 20),
            None => Allocator::new(g.device_budget()),
        };
        let tl = Timeline::new(&g).unwrap();
        let up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let raster = Raster::new(&g).unwrap();
        let ray = RayPrimary::new(&g).unwrap();
        let small = Targets::new(&g, &mut alloc, vk::Extent2D { width: W, height: H }).unwrap();
        let small_ray = RayTargets::new(&g, &mut alloc, W, H).unwrap();
        let full = Targets::new(&g, &mut alloc, vk::Extent2D { width: FW, height: FH }).unwrap();
        let full_ray = RayTargets::new(&g, &mut alloc, FW, FH).unwrap();
        let check = Allocator::new(Budget::unlimited());
        Rig { g, alloc, check, tl, up, raster, ray, small, small_ray, full, full_ray }
    }

    fn scene(&mut self, p: &mut Pipeline, size: RegionSize, ray: bool) -> GpuScene {
        let rs = full_regions(p, size);
        GpuScene::build(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, size, &rs, p.current().id.raw(), ray).unwrap()
    }

    /// Raster and (with an accel) ray frames at the check size.
    fn frames(&mut self, s: &GpuScene, cam: &Camera) -> (Frame, Option<Frame>) {
        let b = self.raster.bind(&self.g, &s.meshes).unwrap();
        let r = self.raster.render_to_host(&self.g, &mut self.check, &mut self.tl, &self.small, &s.meshes, &b, cam, Faults::default()).unwrap();
        b.destroy(&self.g);
        let y = s.accel.as_ref().map(|a| self.ray.trace_to_host(&self.g, &mut self.check, &mut self.tl, a, &self.small_ray, cam).unwrap());
        (r, y)
    }

    fn finish(self) {
        let Rig { g, mut alloc, check, tl, up, raster, ray, small, small_ray, full, full_ray } = self;
        g.wait_idle().unwrap();
        small.free(&g, &mut alloc);
        small_ray.free(&g, &mut alloc);
        full.free(&g, &mut alloc);
        full_ray.free(&g, &mut alloc);
        raster.destroy(&g);
        ray.destroy(&g);
        up.destroy(&g, &mut alloc);
        tl.destroy(&g);
        assert_eq!(alloc.destroy(&g), 0, "leaked buffers or images");
        assert_eq!(check.destroy(&g), 0, "leaked readback buffers");
    }
}

/// Reference images, computed once per world state and shared by every setting (the edit sequence
/// is the same for each; the stored world proves it).
#[derive(Default)]
struct RefCache {
    by_label: BTreeMap<String, (World, Vec<Vec<RefPixel>>)>,
}

impl RefCache {
    fn get(&mut self, label: &str, w: &World, cams: &[(&str, Camera)]) -> &Vec<Vec<RefPixel>> {
        if !self.by_label.contains_key(label) {
            let t = Instant::now();
            let refs = cams.iter().map(|(_, c)| equivalence::reference(w, c, REF_THREADS)).collect();
            eprintln!("reference {label}: {} cameras, {:.2} s", cams.len(), t.elapsed().as_secs_f64());
            self.by_label.insert(label.to_string(), (w.clone(), refs));
        }
        let (stored, refs) = &self.by_label[label];
        assert!(stored.same_content(w), "{label}: the world differs from the one the reference was made for");
        refs
    }
}

/// Criterion 2 for one state: raster and ray against the reference, and against each other.
fn check_state(rig: &mut Rig, s: &GpuScene, p: &mut Pipeline, refs: &[Vec<RefPixel>], cams: &[(&str, Camera)], label: &str) -> u64 {
    let regions = full_regions(p, s.size());
    let mut failures = 0;
    for ((name, cam), r) in cams.iter().zip(refs) {
        let (ras, ray) = rig.frames(s, cam);
        let ray = ray.expect("ray scene");
        let rr = equivalence::compare(&ras, r, cam, &regions);
        let yr = equivalence::compare(&ray, r, cam, &regions);
        let ag = equivalence::agreement(&ras, &ray, r, cam, &regions);
        eprintln!("check {label} {name}: raster {}", rr.summary());
        eprintln!("check {label} {name}: ray    {}", yr.summary());
        eprintln!("check {label} {name}: agree  {}", ag.summary());
        for e in rr.examples.iter().chain(&yr.examples).chain(&ag.examples).take(4) {
            eprintln!("  example: {e}");
        }
        failures += rr.failures() + yr.failures() + ag.failures();
    }
    failures
}

/// Criterion 2: the device meshes equal a from-scratch build byte for byte; the ledger holds
/// exactly the scene after collection; scratch is gone.
fn check_incremental(rig: &mut Rig, s: &mut GpuScene, p: &mut Pipeline, label: &str) {
    rig.g.wait_idle().unwrap();
    let done = rig.tl.completed(&rig.g).unwrap();
    s.collect(&rig.g, &mut rig.alloc, done);
    assert_eq!(s.retiring_len(), 0, "{label}: everything retired was collected");
    let full = full_regions(p, s.size());
    let have: Vec<RegionKey> = s.meshes.regions.keys().copied().collect();
    let want: Vec<RegionKey> = full.keys().copied().collect();
    assert_eq!(have, want, "{label}: region set differs from a from-scratch build");
    if let Some(a) = &s.accel {
        assert_eq!(a.blas_keys().collect::<Vec<_>>(), want, "{label}: BLAS set differs");
    }
    let mut mismatched = 0;
    for chunk in want.chunks(256) {
        let ranges: Vec<_> = chunk.iter().map(|k| (&s.meshes.regions[k].buffer, 0, s.meshes.regions[k].sections.total)).collect();
        let got = download(&rig.g, &mut rig.check, &mut rig.tl, &ranges).unwrap();
        for (k, bytes) in chunk.iter().zip(got) {
            let (sec, img) = full[k].image(s.meshes.align);
            if sec != s.meshes.regions[k].sections || img != bytes {
                mismatched += 1;
            }
        }
    }
    assert_eq!(mismatched, 0, "{label}: {mismatched} regions differ from a from-scratch build");
    let l = rig.alloc.ledger();
    let (mesh, accel) = s.device_bytes();
    let live = |c| l.account(c).usage.live;
    assert_eq!(live(Category::GpuMesh), mesh, "{label}: GpuMesh ledger");
    assert_eq!(live(Category::GpuAccel), accel, "{label}: GpuAccel ledger");
    assert_eq!(live(Category::GpuAccelScratch), 0, "{label}: scratch freed");
}

#[derive(Default)]
struct Stages {
    apply: Vec<f64>,
    notify: Vec<f64>,
    jobs: Vec<f64>,
    publish: Vec<f64>,
    layout: Vec<f64>,
    upload: Vec<f64>,
    accel: Vec<f64>,
    blas_gpu: Vec<f64>,
    tlas_gpu: Vec<f64>,
    bind: Vec<f64>,
    frame: Vec<f64>,
    total: Vec<f64>,
    regions: Vec<f64>,
    blas_tris: Vec<f64>,
    upload_kb: Vec<f64>,
}

fn json_series(name: &str, v: &[f64]) -> String {
    let p = |q| percentile(v, q).unwrap_or(f64::NAN);
    format!("\"{name}_median\":{:.4},\"{name}_p95\":{:.4},\"{name}_max\":{:.4}", p(50.0), p(95.0), v.iter().cloned().fold(f64::NAN, f64::max))
}

impl Stages {
    fn json(&self) -> String {
        [
            json_series("apply_ms", &self.apply),
            json_series("notify_ms", &self.notify),
            json_series("jobs_ms", &self.jobs),
            json_series("publish_ms", &self.publish),
            json_series("layout_ms", &self.layout),
            json_series("upload_ms", &self.upload),
            json_series("accel_ms", &self.accel),
            json_series("blas_gpu_ms", &self.blas_gpu),
            json_series("tlas_gpu_ms", &self.tlas_gpu),
            json_series("bind_ms", &self.bind),
            json_series("frame_ms", &self.frame),
            json_series("total_ms", &self.total),
            json_series("regions", &self.regions),
            json_series("blas_triangles", &self.blas_tris),
            json_series("upload_kib", &self.upload_kb),
        ]
        .join(",")
    }
}

/// One edit end to end, timed: commit, jobs, publication, region meshes, update, rebind, and one
/// full-size frame (raster and ray) that must complete. Returns the update's stats.
#[allow(clippy::too_many_arguments)]
fn timed_edit(rig: &mut Rig, s: &mut GpuScene, p: &mut Pipeline, w: &mut World, tx: &Transaction, cam: &Camera, timer: &mut GpuTimer, sub: &mut Submitter, st: &mut Stages) -> UpdateStats {
    let t0 = Instant::now();
    let applied = w.apply(tx).unwrap();
    assert!(!applied.changed.is_empty(), "the edit changes content");
    let t_apply = ms(t0);
    let t = Instant::now();
    p.notify_edits(&applied.changed);
    let t_notify = ms(t);
    let (keys, t_jobs, t_publish) = drain(p, w);
    let t = Instant::now();
    let changed = changed_regions(p, &keys, s.size());
    let t_layout = ms(t);
    let u = s.update(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, changed, p.current().id.raw()).unwrap();
    let t = Instant::now();
    let rb = rig.raster.bind(&rig.g, &s.meshes).unwrap();
    let yb = rig.ray.bind(&rig.g, s.accel.as_ref().unwrap(), &rig.full_ray).unwrap();
    let t_bind = ms(t);
    let t = Instant::now();
    let g = &rig.g;
    let cmd = sub.begin(g, &rig.tl).unwrap();
    timer.reset(g, cmd, 0);
    rig.raster.record(g, cmd, &rig.full, &s.meshes, &rb, cam, Faults::default());
    rig.ray.record(g, cmd, &yb, &rig.full_ray, cam);
    let v = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
    rig.tl.wait(g, v, u64::MAX).unwrap();
    let t_frame = ms(t);
    let total = ms(t0);
    // Retire the bindings of this frame (their pools) and collect: the frame is complete.
    s.retire(v, gpu::scene::Garbage::Pool(rb.into_pool()));
    s.retire(v, gpu::scene::Garbage::Pool(yb.into_pool()));
    s.collect(g, &mut rig.alloc, rig.tl.completed(g).unwrap());
    let a = u.accel.expect("accel stats");
    st.apply.push(t_apply);
    st.notify.push(t_notify);
    st.jobs.push(t_jobs);
    st.publish.push(t_publish);
    st.layout.push(t_layout);
    st.upload.push(u.upload_ms);
    st.accel.push(u.accel_ms);
    st.blas_gpu.push(a.blas_gpu_ms);
    st.tlas_gpu.push(a.tlas_gpu_ms);
    st.bind.push(t_bind);
    st.frame.push(t_frame);
    st.total.push(total);
    st.regions.push(u.regions_changed as f64);
    st.blas_tris.push(a.triangles as f64);
    st.upload_kb.push(u.upload_bytes as f64 / 1024.0);
    u
}

/// Criteria 2, 5 and 6: every setting, every edit, both paths, plus the sweep's measurements.
#[test]
fn edits_reach_raster_and_ray_for_every_setting() {
    let (w0, _) = scene::street_block();
    let glass = w0.materials().id_of("glass").unwrap();
    let asphalt = w0.materials().id_of("asphalt").unwrap();
    let mut rig = Rig::new();
    eprintln!("device {}", rig.g.info.name);
    let cams = check_cameras(W, H);
    let edit_cam = check_cameras(FW, FH).remove(0).1;
    let costs = cost_cameras(FW, FH);
    let mut refs = RefCache::default();
    let mut failed = Vec::new();
    for merge in Merge::ALL {
        for size in RegionSize::ALL {
            let label = format!("{}/{}", merge.name(), size.name());
            let mut w = w0.clone();
            let mut p = populated(&w, merge);
            let t = Instant::now();
            let mut s = rig.scene(&mut p, size, true);
            let build_ms = ms(t);
            let b = s.accel.as_ref().unwrap().stats;
            let (mesh_bytes, accel_bytes) = s.device_bytes();
            check_incremental(&mut rig, &mut s, &mut p, &format!("{label} initial"));

            // Trace and raster cost at 1080p, interleaved (as 2D).
            let rb = rig.raster.bind(&rig.g, &s.meshes).unwrap();
            let yb = rig.ray.bind(&rig.g, s.accel.as_ref().unwrap(), &rig.full_ray).unwrap();
            let mut timer = GpuTimer::new(&rig.g, 1, 2).unwrap();
            let mut sub = Submitter::new(&rig.g).unwrap();
            let (mut ras_med, mut ray_med) = (Vec::new(), Vec::new());
            for (_, cam) in &costs {
                let (mut ras, mut ray) = (Vec::new(), Vec::new());
                for rep in 0..COST_REPS + 2 {
                    let g = &rig.g;
                    let cmd = sub.begin(g, &rig.tl).unwrap();
                    timer.reset(g, cmd, 0);
                    let order = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
                    for pass in order {
                        timer.begin_pass(g, cmd, 0, pass);
                        if pass == 0 {
                            rig.raster.record(g, cmd, &rig.full, &s.meshes, &rb, cam, Faults::default());
                        } else {
                            rig.ray.record(g, cmd, &yb, &rig.full_ray, cam);
                        }
                        timer.end_pass(g, cmd, 0, pass);
                    }
                    let v = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
                    rig.tl.wait(g, v, u64::MAX).unwrap();
                    let m = timer.read(g, 0).unwrap().unwrap();
                    if rep >= 2 {
                        ras.push(m[0]);
                        ray.push(m[1]);
                    }
                }
                ras_med.push(percentile(&ras, 50.0).unwrap());
                ray_med.push(percentile(&ray, 50.0).unwrap());
            }
            rig.g.wait_idle().unwrap();
            rb.destroy(&rig.g);
            yb.destroy(&rig.g);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let l = rig.alloc.ledger();
            println!(
                "SWEEP {{\"section\":\"setting\",\"merge\":\"{}\",\"region\":\"{}\",\"regions\":{},\"quads\":{},\"triangles\":{},\"mesh_bytes\":{mesh_bytes},\"accel_bytes\":{accel_bytes},\"ledger_gpu_mesh_reserved\":{},\"ledger_gpu_accel_reserved\":{},\"scratch_bytes\":{},\"blas_bytes\":{},\"tlas_bytes\":{},\"build_blas_gpu_ms\":{:.3},\"build_tlas_gpu_ms\":{:.3},\"build_host_ms\":{:.1},\"scene_build_wall_ms\":{build_ms:.1},\"trace_ms_by_camera\":{:?},\"raster_ms_by_camera\":{:?},\"trace_ms_mean\":{:.4},\"raster_ms_mean\":{:.4}}}",
                merge.name(),
                size.name(),
                s.meshes.regions.len(),
                s.meshes.quad_count(),
                s.meshes.triangle_count(),
                l.account(Category::GpuMesh).usage.reserved,
                l.account(Category::GpuAccel).usage.reserved,
                b.scratch_bytes,
                b.blas_bytes,
                b.tlas_bytes,
                b.blas_gpu_ms,
                b.tlas_gpu_ms,
                b.host_ms,
                ray_med.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(),
                ras_med.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(),
                mean(&ray_med),
                mean(&ras_med)
            );

            // Edits: REPS alternating glass / asphalt, then one clear; checks after the first and the clear.
            for (kind, lo, hi) in EDITS {
                let mut st = Stages::default();
                for r in 0..=REPS {
                    let m = if r == REPS { None } else if r % 2 == 0 { Some(glass) } else { Some(asphalt) };
                    let mut probe = Stages::default();
                    let target = if r < REPS { &mut st } else { &mut probe };
                    timed_edit(&mut rig, &mut s, &mut p, &mut w, &edit_tx(lo, hi, m), &edit_cam, &mut timer, &mut sub, target);
                    if r == 0 || r == REPS {
                        let state = if r == 0 { "placed" } else { "cleared" };
                        let lab = format!("{kind}_{state}_after_{}", EDITS.iter().position(|e| e.0 == kind).unwrap());
                        let rr = refs.get(&lab, &w, &cams).clone();
                        let f = check_state(&mut rig, &s, &mut p, &rr, &cams, &format!("{label} {lab}"));
                        if f > 0 {
                            failed.push(format!("{label} {lab}: {f} failures"));
                        }
                        check_incremental(&mut rig, &mut s, &mut p, &format!("{label} {lab}"));
                    }
                    if r == REPS {
                        println!("SWEEP {{\"section\":\"clear\",\"merge\":\"{}\",\"region\":\"{}\",\"edit\":\"{kind}\",{}}}", merge.name(), size.name(), probe.json());
                    }
                }
                println!("SWEEP {{\"section\":\"edit\",\"merge\":\"{}\",\"region\":\"{}\",\"edit\":\"{kind}\",\"reps\":{REPS},{}}}", merge.name(), size.name(), st.json());
            }
            timer.destroy(&rig.g);
            sub.destroy(&rig.g);
            rig.g.wait_idle().unwrap();
            s.free_now(&rig.g, &mut rig.alloc);
        }
    }
    assert!(failed.is_empty(), "failed: {failed:#?}");
    assert_clean(&rig.g);
    rig.finish();
}

/// The ray-query source of the debug views shows the same colours as the raster source wherever
/// the two frames hold identical values (criterion 9's view, checked headless).
#[test]
fn debug_views_read_ray_outputs_like_the_gbuffer() {
    let (w, _) = scene::street_block();
    let mut rig = Rig::new();
    let mut p = populated(&w, Merge::Greedy);
    let s = rig.scene(&mut p, RegionSize::Chunk, true);
    let mats: Vec<world::MaterialParams> = w.materials().iter().map(|(_, d)| d.params).collect();
    let (tables, _) = Tables::upload(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, &mats, &s.meshes, s.snapshot).unwrap();
    let dv = DebugView::new(&rig.g, vk::Format::R8G8B8A8_UNORM).unwrap();
    dv.bind(&rig.g, &rig.small, &tables, Some(&rig.small_ray));
    let out = rig.alloc.create_image(&rig.g, vk::Format::R8G8B8A8_UNORM, vk::Extent2D { width: W, height: H }, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::ImageAspectFlags::COLOR, Category::GpuTemporal).unwrap();
    let rb = rig.raster.bind(&rig.g, &s.meshes).unwrap();
    let yb = rig.ray.bind(&rig.g, s.accel.as_ref().unwrap(), &rig.small_ray).unwrap();
    let light = Lighting::new([0.4, 0.8, 0.3]);
    let px = (W * H) as u64;
    let host = rig.alloc.create_buffer(&rig.g, px * 4, vk::BufferUsageFlags::TRANSFER_DST, Category::Staging, Kind::Host).unwrap();
    let mut sub = Submitter::new(&rig.g).unwrap();
    let mut total_same = 0u64;
    let mut total_diff = 0u64;
    for (name, cam) in check_cameras(W, H) {
        let (ras, ray) = rig.frames(&s, &cam);
        let ray = ray.unwrap();
        // Views that read depth need identical depth bits; the others need identical normal,
        // material and surface id on pixels both paths hit (depth only decides hit or miss there).
        let exact: Vec<bool> = (0..px as usize).map(|i| ras.depth[i].to_bits() == ray.depth[i].to_bits() && ras.normal[i] == ray.normal[i] && ras.material[i] == ray.material[i] && ras.surface[i] == ray.surface[i]).collect();
        let both_hit: Vec<bool> = (0..px as usize).map(|i| ras.depth[i] > 0.0 && ray.depth[i] > 0.0 && ras.normal[i] == ray.normal[i] && ras.material[i] == ray.material[i] && ras.surface[i] == ray.surface[i]).collect();
        // Lighting and temporal views show gpu::shade / gpu::temporal output, checked in their own tests.
        for view in View::ALL.into_iter().filter(|v| v.reads_gbuffer()) {
            let same = if matches!(view, View::Brick | View::Depth) { &exact } else { &both_hit };
            let mut images = Vec::new();
            for source in [Source::Raster, Source::Ray] {
                let g = &rig.g;
                let cmd = sub.begin(g, &rig.tl).unwrap();
                rig.raster.record(g, cmd, &rig.small, &s.meshes, &rb, &cam, Faults::default());
                rig.ray.record(g, cmd, &yb, &rig.small_ray, &cam);
                let b = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
                unsafe { g.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
                gpu::present::image_to_attachment(g, cmd, out.image);
                dv.record(g, cmd, &rig.small, &tables, out.view, &cam, view, &light, source);
                let to_copy = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(out.image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })];
                let copy = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: W, height: H, depth: 1 },
                };
                unsafe {
                    g.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_copy));
                    g.device.cmd_copy_image_to_buffer(cmd, out.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.buffer, &[copy]);
                }
                gpu::submit::all_to_host(g, cmd);
                let val = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
                rig.tl.wait(g, val, u64::MAX).unwrap();
                images.push(host.mapped_ref().unwrap().to_vec());
            }
            let (mut n_same, mut n_diff) = (0u64, 0u64);
            for i in 0..px as usize {
                if !same[i] {
                    continue;
                }
                if images[0][4 * i..4 * i + 4] == images[1][4 * i..4 * i + 4] {
                    n_same += 1;
                } else {
                    n_diff += 1;
                }
            }
            eprintln!("view {name} {}: {n_same} identical-value pixels agree, {n_diff} differ ({} pixels hold different values)", view.name(), same.iter().filter(|x| !**x).count());
            total_same += n_same;
            total_diff += n_diff;
        }
    }
    assert!(total_same > 0);
    assert_eq!(total_diff, 0, "the ray source shows the same colour for the same values");
    let g = &rig.g;
    g.wait_idle().unwrap();
    sub.destroy(g);
    rb.destroy(g);
    yb.destroy(g);
    dv.destroy(g);
    rig.alloc.free(g, host);
    rig.alloc.free_image(g, out);
    tables.free_now(g, &mut rig.alloc);
    s.free_now(g, &mut rig.alloc);
    assert_clean(&rig.g);
    rig.finish();
}

/// Criterion 3: a refused update leaves the previous snapshot visible and correct, frees what it
/// made and counts the deferral; after space is freed the retry shows the new snapshot.
/// Scenario A refuses the mesh upload (total budget, with a ballast freed for the retry);
/// scenario B refuses the acceleration update after the mesh upload succeeded (rollback).
#[test]
fn refused_update_keeps_the_previous_snapshot() {
    let (w0, _) = scene::street_block();
    let glass = w0.materials().id_of("glass").unwrap();
    let (merge, size) = (Merge::None, RegionSize::Chunks2);
    let (_, lo, hi) = EDITS[2];
    let cams = check_cameras(W, H);
    let old_refs: Vec<Vec<RefPixel>> = cams.iter().map(|(_, c)| equivalence::reference(&w0, c, REF_THREADS)).collect();
    let mut w1 = w0.clone();
    w1.apply(&edit_tx(lo, hi, Some(glass))).unwrap();
    let new_refs: Vec<Vec<RefPixel>> = cams.iter().map(|(_, c)| equivalence::reference(&w1, c, REF_THREADS)).collect();

    // A probe run measures what setup reserves in total and in GpuAccel, so the budgets are exact.
    let (total0, accel0) = {
        let mut rig = Rig::with_budget(Some(Budget::unlimited()));
        let mut p = populated(&w0, merge);
        let s = rig.scene(&mut p, size, true);
        let l = rig.alloc.ledger();
        let r = (l.total_reserved(), l.account(Category::GpuAccel).usage.reserved);
        rig.g.wait_idle().unwrap();
        s.free_now(&rig.g, &mut rig.alloc);
        rig.finish();
        r
    };
    const BALLAST: u64 = 256 << 20;
    for scenario in ["A_total_budget", "B_accel_limit"] {
        let budget = if scenario.starts_with('A') { Budget::unlimited().with_total(total0 + BALLAST) } else { Budget::unlimited().with_limit(Category::GpuAccel, accel0) };
        let mut rig = Rig::with_budget(Some(budget));
        let mut w = w0.clone();
        let mut p = populated(&w, merge);
        let mut s = rig.scene(&mut p, size, true);
        assert_eq!(rig.alloc.ledger().total_reserved(), total0, "{scenario}: setup is deterministic");
        let ballast = scenario.starts_with('A').then(|| rig.alloc.create_buffer(&rig.g, BALLAST, vk::BufferUsageFlags::STORAGE_BUFFER, Category::GpuTemporal, Kind::Device).unwrap());
        let shown = s.snapshot;
        let applied = w.apply(&edit_tx(lo, hi, Some(glass))).unwrap();
        p.notify_edits(&applied.changed);
        let (keys, _, _) = drain(&mut p, &w);
        let before = (rig.alloc.ledger().account(Category::GpuMesh), rig.alloc.ledger().account(Category::GpuAccel), rig.alloc.stats().buffers_live);
        let changed = changed_regions(&mut p, &keys, size);
        match s.update(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, changed, p.current().id.raw()) {
            Err(GpuError::OverBudget(e)) => eprintln!("{scenario}: update refused: {e:?}"),
            other => panic!("{scenario}: expected OverBudget, got {:?}", other.map(|u| u.regions_changed)),
        }
        rig.g.wait_idle().unwrap();
        let after = (rig.alloc.ledger().account(Category::GpuMesh), rig.alloc.ledger().account(Category::GpuAccel), rig.alloc.stats().buffers_live);
        assert_eq!((before.0.usage.live, before.1.usage.live, before.2), (after.0.usage.live, after.1.usage.live, after.2), "{scenario}: nothing new stays allocated");
        assert_eq!(s.snapshot, shown, "{scenario}: the previous snapshot is still shown");
        assert_eq!(s.deferred, 1, "{scenario}: deferral counted");
        assert!(after.0.refusals + after.1.refusals > before.0.refusals + before.1.refusals, "{scenario}: refusal counted by the ledger");
        // The device still shows the old world: check against the old world's reference. The CPU
        // regions for resolving surface ids are the ones the scene was built from.
        let mut p_old = populated(&w0, merge);
        let failures = check_state(&mut rig, &s, &mut p_old, &old_refs, &cams, &format!("{scenario} refused"));
        assert_eq!(failures, 0, "{scenario}: the previous snapshot is shown correctly");
        if let Some(b) = ballast {
            rig.alloc.free(&rig.g, b);
            let changed = changed_regions(&mut p, &keys, size);
            s.update(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, changed, p.current().id.raw()).unwrap();
            assert_eq!(s.snapshot, p.current().id.raw());
            let failures = check_state(&mut rig, &s, &mut p, &new_refs, &cams, &format!("{scenario} retried"));
            assert_eq!(failures, 0, "{scenario}: the retry shows the new snapshot");
            check_incremental(&mut rig, &mut s, &mut p, &format!("{scenario} retried"));
        }
        rig.g.wait_idle().unwrap();
        s.free_now(&rig.g, &mut rig.alloc);
        assert_clean(&rig.g);
        rig.finish();
    }
}

/// Criterion 4: a region left out of the update is caught by the reference check; replaced buffers
/// freed at the swap while a frame is in flight are reported by validation (the correct run of the
/// same sequence is clean).
#[test]
fn planted_edit_faults_are_caught() {
    let (w0, _) = scene::street_block();
    let glass = w0.materials().id_of("glass").unwrap();
    let (_, lo, hi) = EDITS[2];
    let cams = check_cameras(W, H);
    let mut w1 = w0.clone();
    w1.apply(&edit_tx(lo, hi, Some(glass))).unwrap();
    let new_refs: Vec<Vec<RefPixel>> = cams.iter().map(|(_, c)| equivalence::reference(&w1, c, REF_THREADS)).collect();

    // (a) Missed region.
    {
        let mut rig = Rig::new();
        let mut w = w0.clone();
        let mut p = populated(&w, Merge::Greedy);
        let mut s = rig.scene(&mut p, RegionSize::Chunk, true);
        let skipped = RegionKey::of(v(lo).split().0, RegionSize::Chunk);
        s.faults.skip_region = Some(skipped);
        let applied = w.apply(&edit_tx(lo, hi, Some(glass))).unwrap();
        p.notify_edits(&applied.changed);
        let (keys, _, _) = drain(&mut p, &w);
        let changed = changed_regions(&mut p, &keys, RegionSize::Chunk);
        s.update(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, changed, p.current().id.raw()).unwrap();
        let failures = check_state(&mut rig, &s, &mut p, &new_refs[..1], &cams[..1], "planted skip_region");
        eprintln!("planted skip_region {skipped:?}: {failures} failures (edit close-up)");
        assert!(failures > 0, "a missed region must fail the reference check");
        rig.g.wait_idle().unwrap();
        s.free_now(&rig.g, &mut rig.alloc);
        assert_clean(&rig.g);
        rig.finish();
    }

    // (b) Early free, mesh-only path (no wait): a raster frame is held pending by a gate semaphore
    // while the update swaps; the correct run first, then the planted one.
    for planted in [false, true] {
        let mut rig = Rig::new();
        let mut w = w0.clone();
        let mut p = populated(&w, Merge::Greedy);
        let mut s = rig.scene(&mut p, RegionSize::Chunk, false);
        s.faults.free_at_swap = planted;
        let gate = Timeline::new(&rig.g).unwrap();
        let rb = rig.raster.bind(&rig.g, &s.meshes).unwrap();
        let mut sub = Submitter::new(&rig.g).unwrap();
        let cam = &cams[0].1;
        let cmd = sub.begin(&rig.g, &rig.tl).unwrap();
        rig.raster.record(&rig.g, cmd, &rig.small, &s.meshes, &rb, cam, Faults::default());
        let frame = sub.submit(&rig.g, &mut rig.tl, cmd, &[(gate.semaphore, 1)]).unwrap();
        let (e0, _) = rig.g.validation_counts();
        let applied = w.apply(&edit_tx(lo, hi, Some(glass))).unwrap();
        p.notify_edits(&applied.changed);
        let (keys, _, _) = drain(&mut p, &w);
        let changed = changed_regions(&mut p, &keys, RegionSize::Chunk);
        s.update(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, changed, p.current().id.raw()).unwrap();
        s.retire(rig.tl.last_signal(), gpu::scene::Garbage::Pool(rb.into_pool()));
        let (e1, _) = rig.g.validation_counts();
        gate.signal_from_host(&rig.g, 1).unwrap();
        rig.tl.wait(&rig.g, frame, u64::MAX).unwrap();
        rig.g.wait_idle().unwrap();
        s.collect(&rig.g, &mut rig.alloc, rig.tl.completed(&rig.g).unwrap());
        eprintln!("early free planted={planted}: validation errors during the swap {}; first: {:?}", e1 - e0, rig.g.first_validation_errors().first());
        if planted {
            assert!(e1 > e0, "freeing in-use buffers at the swap must be reported");
        } else {
            assert_eq!(e1, e0, "the retired path is clean");
            let failures = check_state_raster_only(&mut rig, &s, &mut p, &new_refs, &cams);
            assert_eq!(failures, 0, "mesh-only path shows the edit");
        }
        sub.destroy(&rig.g);
        gate.destroy(&rig.g);
        s.free_now(&rig.g, &mut rig.alloc);
        if !planted {
            assert_clean(&rig.g);
        }
        rig.finish();
    }
}

fn check_state_raster_only(rig: &mut Rig, s: &GpuScene, p: &mut Pipeline, refs: &[Vec<RefPixel>], cams: &[(&str, Camera)]) -> u64 {
    let regions = full_regions(p, s.size());
    let mut failures = 0;
    for ((name, cam), r) in cams.iter().zip(refs) {
        let (ras, _) = rig.frames(s, cam);
        let rr = equivalence::compare(&ras, r, cam, &regions);
        eprintln!("mesh-only {name}: {}", rr.summary());
        failures += rr.failures();
    }
    failures
}
