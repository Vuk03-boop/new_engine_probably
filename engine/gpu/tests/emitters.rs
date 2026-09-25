//! Phase 4A GPU tests: emission and emitter next-event estimation in the GPU reference against the
//! CPU reference (ADR-0005 Amendment 3), and the emitter table published by `GpuScene` through
//! edits (ADR-0003 Amendment 3). Criteria G1–G6 of `docs/changes/2026-09-25-phase4a-emitters.md`.
//! Like the other GPU tests they need the Vulkan SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test emitters -- --test-threads=1 --nocapture`.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use derived::{extract_world, Config, Merge, Pipeline};
use gpu::accel::{Accel, AccelFaults};
use gpu::alloc::Allocator;
use gpu::emitters::{self, GpuEmitter, RefEmitters};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::Camera;
use gpu::reference::{RefAccum, RefBindings, RefFaults, RefImage, RefMaterials, Reference};
use gpu::scene::{affected_regions, region_meshes, GpuScene};
use gpu::staging::{download, Uploader, DEFAULT_RING_BYTES};
use gpu::{Gpu, GpuError, Timeline};
use light::emitters::{EmitterTable, StaleTable};
use light::reference::{self as cpu, Accum, EmitterFaults, Lighting, Pinhole, Settings};
use light::{Atmosphere, SunPath};
use memory::{Budget, Category};
use world::dims::VOXEL_SIZE_M;
use world::scene::{street_night, Dressing};
use world::{BrickKey, MaterialId, MaterialParams, MaterialRegistry, Transaction, VoxelCoord, World};

const NEAR: f64 = 0.1;
const SEED: u32 = 0x4A;
const SNAPSHOT: u64 = 1;

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "4A needs the ray-tracing device (P-001)");
    g
}

fn street_camera(w: u32, h: u32) -> Camera {
    let (_, view) = street_night(Dressing::Lamps);
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)
}

/// The 3A second view: along the street, low.
fn low_camera(w: u32, h: u32) -> Camera {
    Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, NEAR)
}

fn pinhole(c: &Camera) -> Pinhole {
    let s = |a: [f64; 3], k: f64| a.map(|x| x * k);
    Pinhole { eye: c.eye, forward: c.forward, right: s(c.right, c.tan_half_x), up: s(c.up, c.tan_half_y), width: c.width, height: c.height }
}

fn lighting(hour: f64) -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(hour))
}

fn regions(world: &World) -> BTreeMap<RegionKey, RegionMesh> {
    let meshes: Vec<_> = world.bricks().map(|(k, _)| (k, extract_world(world, k, Merge::Greedy).unwrap())).collect();
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk)
}

/// A closed 16³ box of albedo 0.5 whose every face emits radiance 1 (the emissive furnace).
fn furnace() -> World {
    let mut r = MaterialRegistry::new();
    let wall = r.register("wall", MaterialParams { base_color: [0.5; 3], emissive: [1.0; 3] }).unwrap();
    let mut w = World::new(r);
    let v = VoxelCoord::new;
    w.fill_box(v(0, 0, 0), v(16, 16, 16), Some(wall)).unwrap();
    w.fill_box(v(1, 1, 1), v(15, 15, 15), None).unwrap();
    w
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    world: World,
    albedo: Vec<[f64; 3]>,
    table: EmitterTable,
    gm: Option<GpuMeshes>,
    accel: Option<Accel>,
    mats: Option<RefMaterials>,
    em: Option<RefEmitters>,
    reference: Reference,
}

impl Rig {
    fn new(world: World) -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let albedo = cpu::albedos(world.materials()).unwrap();
        let rs = regions(&world);
        let table = emitters::table(&rs, world.materials(), SNAPSHOT).unwrap();
        let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs).unwrap();
        let accel = Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()).unwrap();
        let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedo).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let (em, v) = RefEmitters::upload(&g, &mut alloc, &mut up, &mut tl, &table).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let reference = Reference::new(&g).unwrap();
        eprintln!("rig: {} emitters, {} materials", table.len(), albedo.len());
        Rig { g, alloc, tl, up, world, albedo, table, gm: Some(gm), accel: Some(accel), mats: Some(mats), em: Some(em), reference }
    }

    fn target(&mut self, w: u32, h: u32) -> (RefAccum, RefBindings) {
        let out = RefAccum::new(&self.g, &mut self.alloc, w, h).unwrap();
        let b = self.reference.bind_lit(&self.g, self.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), self.em.as_ref().unwrap(), &out).unwrap();
        (out, b)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_from(&mut self, out: &RefAccum, b: &RefBindings, cam: &Camera, light: &Lighting, s: &Settings, first: u32, frames: u32, per_submit: u32) -> RefImage {
        let ms = self.reference.accumulate(&self.g, &mut self.tl, b, out, cam, light, s, RefFaults::default(), SEED, first..first + frames, first, per_submit).unwrap();
        eprintln!("  gpu: {frames} frames at {}x{} in {ms:.0} ms", cam.width, cam.height);
        self.reference.read(&self.g, &mut self.alloc, &mut self.tl, out, frames).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu(&self, cam: &Camera, light: &Lighting, s: &Settings, first: u32, count: u32) -> Accum {
        cpu::render_lit(&self.world, &self.albedo, &self.table, light, &pinhole(cam), s, first, count, SEED, 2)
    }

    fn release(&mut self, out: RefAccum, b: RefBindings) {
        self.g.wait_idle().unwrap();
        b.destroy(&self.g);
        out.free(&self.g, &mut self.alloc);
    }

    fn finish(mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let (errors, warnings) = g.validation_counts();
        eprintln!("validation: {errors} errors, {warnings} warnings");
        assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
        self.em.take().unwrap().free(g, &mut self.alloc);
        self.mats.take().unwrap().free(g, &mut self.alloc);
        self.accel.take().unwrap().free_now(g, &mut self.alloc);
        self.gm.take().unwrap().free_now(g, &mut self.alloc);
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

fn dark(s: Settings) -> Settings {
    Settings { sun: false, sky: false, ..s }
}

/// G1 + G2: the module binds with the emitter table (layout from reflection, 0 validation errors),
/// and emission at the primary hit, which is deterministic, equals the CPU per pixel.
#[test]
fn emission_at_the_primary_hit_matches_the_cpu_per_pixel() {
    let (world, _) = street_night(Dressing::Full);
    let mut rig = Rig::new(world);
    let (w, h) = (320, 180);
    let s = dark(Settings { emission: true, max_bounces: 0, ..Settings::default() });
    let light = lighting(21.0);
    let mut failed = Vec::new();
    for (cname, cam) in [("street_view", street_camera(w, h)), ("low", low_camera(w, h))] {
        let (out, b) = rig.target(w, h);
        let c = rig.cpu(&cam, &light, &s, 0, 1);
        let g = rig.render_from(&out, &b, &cam, &light, &s, 0, 1, 1);
        let (mut emissive, mut mismatch, mut worst) = (0, 0, 0.0f64);
        for i in 0..c.sum.len() {
            let (cv, gv) = (c.sum[i], g.accum.sum[i]);
            if cv[1] > 0.0 || gv[1] > 0.0 {
                emissive += 1;
                let e = (0..3).map(|k| (gv[k] / cv[k] - 1.0).abs()).fold(0.0, f64::max);
                worst = worst.max(if e.is_nan() { f64::INFINITY } else { e });
                if e.is_nan() || e > 1e-6 {
                    mismatch += 1;
                }
            }
        }
        eprintln!("G2 {cname}: {emissive} emissive pixels, {mismatch} over 1e-6 (worst {worst:.2e}), bad {}", g.bad_samples);
        if mismatch > 0 || emissive < 500 || g.bad_samples > 0 {
            failed.push(cname);
        }
        rig.release(out, b);
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// G3: the emissive furnace on the GPU equals L_e Σ₀^{B+1} aᵏ; "emission counted twice" is caught.
#[test]
fn emissive_furnace_on_the_gpu() {
    let mut rig = Rig::new(furnace());
    let cam = Camera::look_at([8.0, 8.0, 8.0], [8.0, 8.0, 0.0], 80.0, 64, 64, NEAR);
    let b_max = 16u32;
    let expect: f64 = (0..=b_max + 1).map(|k| 0.5f64.powi(k as i32)).sum();
    let s = dark(Settings { emission: true, emitters_direct: true, emitters_indirect: true, max_bounces: b_max, ..Settings::default() });
    let (out, b) = rig.target(64, 64);
    let mut results = Vec::new();
    for f in [EmitterFaults::default(), EmitterFaults { double_emission: true, ..EmitterFaults::default() }] {
        let img = rig.render_from(&out, &b, &cam, &light_night(), &Settings { emitter_faults: f, ..s }, 0, 1024, 64);
        let n = img.accum.sum.len();
        let m = (0..n).map(|i| img.accum.mean(i)[1]).sum::<f64>() / n as f64;
        let se = (0..n).map(|i| img.accum.std_error(i)[1].powi(2)).sum::<f64>().sqrt() / n as f64;
        eprintln!("G3 {f:?}: image mean {m:.6} ± {se:.6} (expect {expect:.6}), ratio {:.5}, bad {}", m / expect, img.bad_samples);
        results.push((m, se, img.bad_samples));
    }
    rig.release(out, b);
    rig.finish();
    let (m, se, bad) = results[0];
    assert_eq!(bad, 0);
    assert!((m - expect).abs() < 3.0 * se && (m / expect - 1.0).abs() < 0.005, "furnace {m} ± {se}, expect {expect}");
    assert!((results[1].0 - expect).abs() > 10.0 * results[1].1, "double emission not caught");
}

fn light_night() -> Lighting {
    lighting(21.0)
}

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

/// G4: the GPU and CPU references estimate the same night image (sun, sky, emission, emitters direct
/// and indirect, one bounce) on `street_night` (Full) at 21 h; "area PDF without solid angle" fails.
#[test]
fn night_street_matches_the_cpu_statistically() {
    let (world, _) = street_night(Dressing::Full);
    let mut rig = Rig::new(world);
    let (w, h) = (96, 54);
    let (cpu_spp, gpu_spp) = (256, 4096);
    let s = Settings { max_bounces: 1, emission: true, emitters_direct: true, emitters_indirect: true, ..Settings::default() };
    let light = light_night();
    let mut failed = Vec::new();
    for (cname, cam) in [("street_view", street_camera(w, h)), ("low", low_camera(w, h))] {
        let (out, b) = rig.target(w, h);
        let t = Instant::now();
        let c = rig.cpu(&cam, &light, &s, 1_000_000, cpu_spp);
        let cpu_s = t.elapsed().as_secs_f64();
        let big = rig.render_from(&out, &b, &cam, &light, &s, 0, gpu_spp, 256);
        let null_img = rig.render_from(&out, &b, &cam, &light, &s, 2_000_000, cpu_spp, 256);
        let null = compare(&big.accum, &null_img.accum).frac_over_4;
        let limit = (1.5 * null + 0.002).max(0.01);
        eprintln!("G4 null {cname}: gpu vs gpu |z|>4 {:.3}%, limit {:.3}%", null * 100.0, limit * 100.0);
        for f in [EmitterFaults::default(), EmitterFaults { no_solid_angle: true, ..EmitterFaults::default() }] {
            let g = if f == EmitterFaults::default() { big.accum.clone() } else { rig.render_from(&out, &b, &cam, &light, &Settings { emitter_faults: f, ..s }, 0, gpu_spp, 256).accum };
            let st = compare(&g, &c);
            eprintln!(
                "G4 {cname} {f:?}: image mean gpu {} cpu {}, rel {:?}, z {:?}, |z|>4 {:.3}%, cpu {cpu_s:.1} s",
                fmt3(image_mean(&g)),
                fmt3(image_mean(&c)),
                st.rel_mean.map(|x| (x * 1e4).round() / 1e4),
                st.mean_z.map(|x| (x * 100.0).round() / 100.0),
                st.frac_over_4 * 100.0
            );
            let pass = st.mean_z.iter().all(|z| z.abs() < 4.0) && st.frac_over_4 <= limit && big.bad_samples == 0;
            match (f.no_solid_angle, pass) {
                (false, false) => failed.push(cname.to_string()),
                (true, true) => failed.push(format!("control no_solid_angle not caught {cname}")),
                _ => {}
            }
        }
        rig.release(out, b);
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

fn fmt3(v: [f64; 3]) -> String {
    format!("[{:.4e} {:.4e} {:.4e}]", v[0], v[1], v[2])
}

fn image_mean(a: &Accum) -> [f64; 3] {
    let n = a.sum.len() as f64;
    [0, 1, 2].map(|k| (0..a.sum.len()).map(|i| a.mean(i)[k]).sum::<f64>() / n)
}

/// Diagnostic (on demand, `--ignored`; CPU only, no device): is G4's colour pattern CPU noise? The
/// first laptop run showed R +0.6%, G −3.0%, B −4.6% (GPU against CPU, z ≈ −2) on the street view.
/// Here the CPU at G4's 256 samples is compared with the CPU at 4096 independent samples: if the
/// better CPU estimate moves by the same amounts, the GPU agreed with it.
#[test]
#[ignore]
fn diagnostic_g4_cpu_convergence() {
    let (world, _) = street_night(Dressing::Full);
    let albedo = cpu::albedos(world.materials()).unwrap();
    let table = emitters::table(&regions(&world), world.materials(), SNAPSHOT).unwrap();
    let s = Settings { max_bounces: 1, emission: true, emitters_direct: true, emitters_indirect: true, ..Settings::default() };
    let light = light_night();
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
    for (cname, cam) in [("street_view", street_camera(96, 54)), ("low", low_camera(96, 54))] {
        let t = Instant::now();
        let a = cpu::render_lit(&world, &albedo, &table, &light, &pinhole(&cam), &s, 1_000_000, 256, SEED, threads);
        let b = cpu::render_lit(&world, &albedo, &table, &light, &pinhole(&cam), &s, 3_000_000, 4096, SEED, threads);
        let st = compare(&b, &a);
        eprintln!(
            "G4 diagnostic {cname}: cpu 256 {}, cpu 4096 {}; 4096 / 256 − 1 = {:?}, z {:?} ({:.0} s, {threads} threads)",
            fmt3(image_mean(&a)),
            fmt3(image_mean(&b)),
            st.rel_mean.map(|x| (x * 1e4).round() / 1e4),
            st.mean_z.map(|x| (x * 100.0).round() / 100.0),
            t.elapsed().as_secs_f64()
        );
    }
}

/// G5: building the emitter table of `street_night` (Dense) from region meshes on the host takes
/// ≤ 5 ms (median of 20), 10% of the 50 ms edit budget. Host only; no device needed.
#[test]
fn emitter_table_build_fits_the_edit_budget() {
    let (world, _) = street_night(Dressing::Dense);
    let rs = regions(&world);
    let mut ms: Vec<f64> = (0..20)
        .map(|_| {
            let t = Instant::now();
            let table = emitters::table(&rs, world.materials(), SNAPSHOT).unwrap();
            std::hint::black_box(&table);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    let n = emitters::table(&rs, world.materials(), SNAPSHOT).unwrap().len();
    eprintln!("G5: {n} emitters, table build median {:.3} ms, max {:.3} ms", ms[10], ms[19]);
    assert!(ms[10] <= 5.0, "median {} ms", ms[10]);
}

/// Runs every job inline and publishes; the published brick keys.
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
    let published = p.try_publish(w);
    assert!(p.is_idle(), "pipeline did not drain");
    published.map(|x| x.groups.into_iter().flatten().collect()).unwrap_or_default()
}

/// Every region of the pipeline's current snapshot, built from scratch.
fn full_regions(p: &mut Pipeline, size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    let t = p.acquire();
    let keys: Vec<BrickKey> = p.current().keys().collect();
    let meshes: Vec<_> = keys.iter().map(|&k| (k, p.read(&t, k).unwrap().unwrap().clone())).collect();
    p.release(t).unwrap();
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
}

/// Commits `edits`, runs the jobs, publishes; the regions to rebuild.
fn edit(w: &mut World, p: &mut Pipeline, edits: &[(VoxelCoord, Option<MaterialId>)], size: RegionSize) -> BTreeSet<RegionKey> {
    let mut tx = Transaction::new();
    for &(v, m) in edits {
        tx.set(v, m);
    }
    let applied = w.apply(&tx).unwrap();
    assert!(!applied.changed.is_empty(), "the edit changes the world");
    p.notify_edits(&applied.changed);
    affected_regions(drain(p, w), size)
}

/// The meshes of `regions` in the pipeline's current snapshot (read again for a retry, as the viewer does).
fn meshes(p: &mut Pipeline, regions: &BTreeSet<RegionKey>, size: RegionSize) -> BTreeMap<RegionKey, Option<RegionMesh>> {
    let token = p.acquire();
    let changed = region_meshes(p, &token, regions, size).unwrap();
    p.release(token).unwrap();
    changed
}

/// One emitter without the snapshot of its id: geometry, radiance, sampling, region and quad.
type Row = (u8, [f64; 3], [f64; 3], [f64; 3], [f64; 3], f64, u32, u32, [i32; 3], u32);

/// Voxel edits: (voxel, new material).
type Edits = Vec<(VoxelCoord, Option<MaterialId>)>;

/// Everything but the snapshot in the ids, in table order.
fn content(t: &EmitterTable) -> Vec<Row> {
    t.emitters.iter().map(|e| (e.face, e.p0, e.eu, e.ev, e.radiance, e.pdf, e.threshold, e.alias, e.id.key, e.id.quad)).collect()
}

fn material_live(alloc: &Allocator) -> u64 {
    alloc.ledger().account(Category::GpuMaterial).usage.live
}

/// G6 (a)–(c): the published table is the snapshot's, equals a from-scratch build, is on the device
/// byte for byte, and after collection the ledger holds exactly it beyond `before`.
#[allow(clippy::too_many_arguments)]
fn check_published(g: &Gpu, alloc: &mut Allocator, check: &mut Allocator, tl: &mut Timeline, s: &mut GpuScene, p: &mut Pipeline, w: &World, before: u64, label: &str) {
    g.wait_idle().unwrap();
    let done = tl.completed(g).unwrap();
    s.collect(g, alloc, done);
    assert_eq!(s.retiring_len(), 0, "{label}: everything retired was collected");
    let e = s.emitters().unwrap_or_else(|x| panic!("{label}: stale table {x:?}")).expect("a lit scene");
    assert_eq!((e.table.snapshot, e.device.snapshot), (s.snapshot, s.snapshot), "{label}: snapshots");
    let scratch = emitters::table(&full_regions(p, s.size()), w.materials(), s.snapshot).unwrap();
    assert_eq!(content(&e.table), content(&scratch), "{label}: the published table differs from a from-scratch build");
    let rows = GpuEmitter::bytes(&GpuEmitter::of(&e.table));
    let em = emitters::emission_bytes(&e.table);
    assert_eq!(e.device.count as usize, e.table.len());
    let got = download(g, check, tl, &[(&e.device.emitters, 0, rows.len() as u64), (&e.device.emission, 0, em.len() as u64)]).unwrap();
    assert!(got[0] == rows && got[1] == em, "{label}: the device holds another table");
    assert_eq!(material_live(alloc), before + s.emitter_bytes(), "{label}: GpuMaterial ledger");
    eprintln!("G6 {label}: {} emitters at snapshot {}, device bytes equal ({} + {} B)", e.table.len(), s.snapshot, rows.len(), em.len());
}

/// G6 (4A part 2): the emitter table in `GpuScene` through edits, on the device: equal to a
/// from-scratch build after every update, byte-equal on the device, retired when replaced; a stale
/// table is refused; a refused table upload leaves the whole previous snapshot.
#[test]
fn the_scene_publishes_the_table_with_its_meshes() {
    let g = gpu();
    let mut alloc = Allocator::new(g.device_budget());
    let mut check = Allocator::new(Budget::unlimited());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let size = RegionSize::Chunk;
    let (mut w, _) = street_night(Dressing::Full);
    let mut p = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
    p.mark_all(&w);
    drain(&mut p, &w);
    let before = material_live(&alloc);
    let mut s = GpuScene::build_lit(&g, &mut alloc, &mut up, &mut tl, size, &full_regions(&mut p, size), p.current().id.raw(), true, w.materials()).unwrap();
    check_published(&g, &mut alloc, &mut check, &mut tl, &mut s, &mut p, &w, before, "build");

    // The C6 sequence.
    let mats = w.materials().clone();
    let id = |n: &str| mats.id_of(n).unwrap();
    let first = |w: &World, m| w.occupied().find(|&(_, x)| x == m).map(|(v, _)| v).unwrap();
    let lamp = first(&w, id("lamp"));
    let bulb = first(&w, id("bulb"));
    let air = VoxelCoord::new(136, 40, 64);
    assert_eq!(w.get(air), None, "open air above the road");
    let emits: Vec<bool> = light::emitters::emission(&mats).unwrap().iter().map(|l| light::emitters::luminance(*l) > 0.0).collect();
    let lit_regions: BTreeSet<RegionKey> = s.emitters().unwrap().unwrap().table.emitters.iter().map(|e| RegionKey { x: e.id.key[0], y: e.id.key[1], z: e.id.key[2] }).collect();
    let inside = |c: i32| (1..7).contains(&c.rem_euclid(8));
    let (quiet, quiet_mat) = w.occupied().find(|&(v, m)| !emits[m.raw() as usize] && inside(v.x) && inside(v.y) && inside(v.z) && !lit_regions.contains(&RegionKey::of(v.split().0, size))).unwrap();
    let steps: Vec<(&str, Edits)> = vec![
        ("remove a lamp voxel", vec![(lamp, None)]),
        ("remove a bulb voxel", vec![(bulb, None)]),
        ("add an emissive voxel in open air", vec![(air, Some(id("neon_pink")))]),
        ("remove a voxel in a region without emitters", vec![(quiet, None)]),
        ("restore everything", vec![(lamp, Some(id("lamp"))), (bulb, Some(id("bulb"))), (air, None), (quiet, Some(quiet_mat))]),
    ];
    for (label, edits) in &steps {
        let regions = edit(&mut w, &mut p, edits, size);
        let changed = meshes(&mut p, &regions, size);
        let old_bytes = s.emitter_bytes();
        let u = s.update(&g, &mut alloc, &mut up, &mut tl, changed, p.current().id.raw()).unwrap();
        // (c) The replaced table is retired, not freed at the swap.
        assert!(s.retiring_len() >= 2, "{label}: the old table is retired");
        assert_eq!(material_live(&alloc), before + s.emitter_bytes() + old_bytes, "{label}: old and new tables both live until retirement");
        eprintln!("G6 {label}: update with {} emitters, table {:.3} ms (host build and upload)", u.emitters.unwrap(), u.emitters_ms);
        check_published(&g, &mut alloc, &mut check, &mut tl, &mut s, &mut p, &w, before, label);
    }

    // (e) A refused table upload leaves meshes, TLAS and table on the previous snapshot; the retry works.
    let (snap, rows) = (s.snapshot, s.region_rows());
    let blas: Vec<RegionKey> = s.accel.as_ref().unwrap().blas_keys().collect();
    let regions = edit(&mut w, &mut p, &[(lamp, None)], size);
    s.faults.refuse_emitters = true;
    match s.update(&g, &mut alloc, &mut up, &mut tl, meshes(&mut p, &regions, size), p.current().id.raw()) {
        Err(GpuError::OverBudget(_)) => {}
        other => panic!("the planted refusal must be an over-budget error, got {:?}", other.map(|u| u.regions_changed)),
    }
    assert_eq!((s.snapshot, s.region_rows(), s.deferred), (snap, rows, 1), "refused: the previous snapshot stays");
    assert_eq!(s.accel.as_ref().unwrap().blas_keys().collect::<Vec<_>>(), blas, "refused: the previous BLAS set stays");
    assert_eq!(s.emitters().unwrap().unwrap().table.snapshot, snap, "refused: the previous table stays");
    g.wait_idle().unwrap();
    assert_eq!(material_live(&alloc), before + s.emitter_bytes(), "refused: nothing of the new table is left");
    s.faults.refuse_emitters = false;
    s.update(&g, &mut alloc, &mut up, &mut tl, meshes(&mut p, &regions, size), p.current().id.raw()).unwrap();
    check_published(&g, &mut alloc, &mut check, &mut tl, &mut s, &mut p, &w, before, "retry after a refusal");

    // (d) A stale table is refused.
    let published = s.snapshot;
    s.faults.stale_emitters = true;
    let regions = edit(&mut w, &mut p, &[(lamp, Some(id("lamp")))], size);
    s.update(&g, &mut alloc, &mut up, &mut tl, meshes(&mut p, &regions, size), p.current().id.raw()).unwrap();
    match s.emitters() {
        Err(StaleTable { table, expected }) => {
            assert_eq!((table, expected), (published, s.snapshot));
            eprintln!("G6 stale table: refused (table {table}, scene {expected})");
        }
        Ok(_) => panic!("a stale table must be refused"),
    }

    g.wait_idle().unwrap();
    s.free_now(&g, &mut alloc);
    up.destroy(&g, &mut alloc);
    let (errors, warnings) = g.validation_counts();
    eprintln!("validation: {errors} errors, {warnings} warnings");
    assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
    tl.destroy(&g);
    assert_eq!(alloc.destroy(&g), 0, "leaked buffers");
    assert_eq!(check.destroy(&g), 0, "leaked readback buffers");
}
