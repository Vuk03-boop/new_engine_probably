//! Phase 2D GPU integration tests: the ray-query representation (one BLAS per region, one TLAS)
//! against the CPU voxel reference and against raster, per pixel (ADR-0003 Amendment 1); planted
//! faults; ledger accounting; build and trace cost (data only). The criteria are declared in
//! `docs/changes/2026-09-24-phase2d-ray-query.md`. Like the other GPU tests, they need the Vulkan
//! SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test ray -- --test-threads=1 --nocapture`.

use std::collections::BTreeMap;
use std::time::Instant;

use ash::vk;
use derived::{extract_world, BrickMesh, Merge};
use gpu::accel::{Accel, AccelFaults};
use gpu::alloc::Allocator;
use gpu::equivalence::{self, RefPixel, Report};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::{Camera, Faults, Frame, Raster, Targets, BACKGROUND_SURFACE, FS_REFLECTION};
use gpu::ray::{RayPrimary, RayTargets};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, GpuError, Timeline};
use memory::{Budget, Category};
use world::dims::VOXEL_SIZE_M;
use world::{scene, BrickKey, World};

const W: u32 = 640;
const H: u32 = 360;
const NEAR: f64 = 0.1;
const REF_THREADS: usize = 2;

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "2D needs the ray-tracing device (P-001); NE_GPU_ALLOW_NO_RT does not apply");
    g
}

fn assert_clean(g: &Gpu) {
    let (errors, warnings) = g.validation_counts();
    eprintln!("validation: {errors} errors, {warnings} warnings");
    assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
}

/// The 2C-1 street cameras (same values as `tests/raster.rs`).
fn cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        ("street_view", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)),
        ("chunk_corner", Camera::look_at([128.0, 32.0, 64.0], [250.0, 20.0, 190.0], 70.0, w, h, NEAR)),
        ("overhead", Camera::look_at([-60.5, 300.25, -80.75], [192.0, 0.0, 150.0], 50.0, w, h, NEAR)),
        ("grazing", Camera::look_at([4.3, 8.6, 20.2], [380.0, 3.0, 30.0], 60.0, w, h, NEAR)),
    ]
}

fn street_meshes(w: &World, merge: Merge) -> Vec<(BrickKey, BrickMesh)> {
    w.bricks().map(|(k, _)| (k, extract_world(w, k, merge).unwrap())).collect()
}

fn regions(meshes: &[(BrickKey, BrickMesh)], size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
}

fn reference(w: &World, name: &str, cam: &Camera) -> Vec<RefPixel> {
    let t = Instant::now();
    let r = equivalence::reference(w, cam, REF_THREADS);
    eprintln!("reference {name}: {}x{}, {:.2} s on {REF_THREADS} threads", cam.width, cam.height, t.elapsed().as_secs_f64());
    r
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    raster: Raster,
    targets: Targets,
    ray: RayPrimary,
    ray_out: RayTargets,
}

impl Rig {
    fn new(w: u32, h: u32) -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let tl = Timeline::new(&g).unwrap();
        let up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let raster = Raster::new(&g).unwrap();
        let targets = Targets::new(&g, &mut alloc, vk::Extent2D { width: w, height: h }).unwrap();
        let ray = RayPrimary::new(&g).unwrap();
        let ray_out = RayTargets::new(&g, &mut alloc, w, h).unwrap();
        Rig { g, alloc, tl, up, raster, targets, ray, ray_out }
    }

    fn upload(&mut self, size: RegionSize, rs: &BTreeMap<RegionKey, RegionMesh>) -> GpuMeshes {
        GpuMeshes::upload(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, size, rs).unwrap().0
    }

    fn build(&mut self, gm: &GpuMeshes, faults: AccelFaults) -> Accel {
        Accel::build(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, gm, faults).unwrap()
    }

    fn raster(&mut self, gm: &GpuMeshes, cam: &Camera) -> Frame {
        let b = self.raster.bind(&self.g, gm).unwrap();
        let f = self.raster.render_to_host(&self.g, &mut self.alloc, &mut self.tl, &self.targets, gm, &b, cam, Faults::default()).unwrap();
        b.destroy(&self.g);
        f
    }

    fn trace(&mut self, accel: &Accel, cam: &Camera) -> Frame {
        self.ray.trace_to_host(&self.g, &mut self.alloc, &mut self.tl, accel, &self.ray_out, cam).unwrap()
    }

    fn finish(self) {
        let Rig { g, mut alloc, tl, up, raster, targets, ray, ray_out } = self;
        g.wait_idle().unwrap();
        targets.free(&g, &mut alloc);
        ray_out.free(&g, &mut alloc);
        raster.destroy(&g);
        ray.destroy(&g);
        up.destroy(&g, &mut alloc);
        tl.destroy(&g);
        assert_eq!(alloc.destroy(&g), 0, "leaked buffers or images");
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

/// Criteria 1, 2 and 4 (ledger after a build), with the build cost per setting as data.
#[test]
fn ray_query_matches_the_reference_and_raster_for_every_setting() {
    let (w, _) = scene::street_block();
    let cams = cameras(W, H);
    let refs: Vec<Vec<RefPixel>> = cams.iter().map(|(n, c)| reference(&w, n, c)).collect();
    let mut rig = Rig::new(W, H);
    eprintln!("device {}", rig.g.info.name);
    let mut failed = Vec::new();
    for merge in Merge::ALL {
        let meshes = street_meshes(&w, merge);
        for size in RegionSize::ALL {
            let rs = regions(&meshes, size);
            let gm = rig.upload(size, &rs);
            let accel = rig.build(&gm, AccelFaults::default());
            let s = accel.stats;
            // Criterion 4: the ledger holds exactly the build's buffers; scratch is gone.
            let acc = rig.alloc.ledger().account(Category::GpuAccel).usage.live;
            let scratch = rig.alloc.ledger().account(Category::GpuAccelScratch).usage.live;
            eprintln!(
                "build merge {} region {}: {} BLAS, {} triangles, {} batches | BLAS {:.2} MiB TLAS {:.3} MiB instances {:.3} MiB table {:.3} MiB, buffers {:.2} MiB (ledger GpuAccel live {:.2} MiB), scratch {:.2} MiB (live after {}) | GPU BLAS {:.3} ms TLAS {:.3} ms, host {:.1} ms | mesh {:.2} MiB",
                merge.name(),
                size.name(),
                s.blas_count,
                s.triangles,
                s.blas_batches,
                mib(s.blas_bytes),
                mib(s.tlas_bytes),
                mib(s.instance_bytes),
                mib(s.table_bytes),
                mib(s.accel_buffer_bytes),
                mib(acc),
                mib(s.scratch_bytes),
                scratch,
                s.blas_gpu_ms,
                s.tlas_gpu_ms,
                s.host_ms,
                mib(gm.device_bytes())
            );
            if acc != s.accel_buffer_bytes || scratch != 0 {
                failed.push(format!("ledger {}/{}: GpuAccel live {acc} vs buffers {}, scratch live {scratch}", merge.name(), size.name(), s.accel_buffer_bytes));
            }
            for ((name, cam), rf) in cams.iter().zip(&refs) {
                let ray = rig.trace(&accel, cam);
                let rep = equivalence::compare(&ray, rf, cam, &rs);
                eprintln!("ray    {name} merge {} region {}: {}", merge.name(), size.name(), rep.summary());
                for e in &rep.examples {
                    eprintln!("    {e}");
                }
                if !rep.passes() || rep.checked * 4 < rep.ref_hits {
                    failed.push(format!("ray/{name}/{}/{}", merge.name(), size.name()));
                }
                let ras = rig.raster(&gm, cam);
                let ag = equivalence::agreement(&ras, &ray, rf, cam, &rs);
                eprintln!("agree  {name} merge {} region {}: {}", merge.name(), size.name(), ag.summary());
                for e in &ag.examples {
                    eprintln!("    {e}");
                }
                if ag.failures() > 0 {
                    failed.push(format!("agree/{name}/{}/{}", merge.name(), size.name()));
                }
            }
            rig.g.wait_idle().unwrap();
            accel.free_now(&rig.g, &mut rig.alloc);
            assert_eq!(rig.alloc.ledger().account(Category::GpuAccel).usage.live, 0, "accel buffers returned");
            gm.free_now(&rig.g, &mut rig.alloc);
        }
    }
    assert_clean(&rig.g);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

fn check_control(name: &str, rep: &Report, caught: bool) {
    eprintln!("control {name}: {}", rep.summary());
    for e in rep.examples.iter().take(3) {
        eprintln!("    {e}");
    }
    assert!(caught, "planted fault '{name}' was not caught: {}", rep.summary());
}

/// Criterion 3.
#[test]
fn planted_accel_faults_are_caught() {
    let (w, _) = scene::street_block();
    let (name, cam) = cameras(W, H).remove(0);
    let rf = reference(&w, name, &cam);
    let mut rig = Rig::new(W, H);

    match RayPrimary::with_reflection(&rig.g, FS_REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("trace pipeline with the raster reflection refused: {} errors, first {:?}", e.len(), e.first()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }

    let meshes = street_meshes(&w, Merge::Greedy);
    let size = RegionSize::Chunk;
    let rs = regions(&meshes, size);
    let gm = rig.upload(size, &rs);

    let accel = rig.build(&gm, AccelFaults::default());
    let good = rig.trace(&accel, &cam);
    accel.free_now(&rig.g, &mut rig.alloc);
    let base = equivalence::compare(&good, &rf, &cam, &rs);
    eprintln!("control baseline: {}", base.summary());
    assert!(base.passes(), "the unfaulted trace must pass before the controls mean anything");

    let run = |rig: &mut Rig, faults: AccelFaults| {
        let accel = rig.build(&gm, faults);
        let f = rig.trace(&accel, &cam);
        rig.g.wait_idle().unwrap();
        accel.free_now(&rig.g, &mut rig.alloc);
        (equivalence::compare(&f, &rf, &cam, &rs), f)
    };

    let (rep, _) = run(&mut rig, AccelFaults { offset_error: [1, 0, 0], ..AccelFaults::default() });
    check_control("instance_offset_off_by_one", &rep, rep.face + rep.depth > 0);

    let mut count: BTreeMap<u32, u64> = BTreeMap::new();
    for s in &good.surface {
        if *s != BACKGROUND_SURFACE {
            *count.entry(s[0]).or_default() += 1;
        }
    }
    let (&busiest, &px) = count.iter().max_by_key(|(_, &n)| n).unwrap();
    let (rep, _) = run(&mut rig, AccelFaults { skip_region: Some(busiest as usize), ..AccelFaults::default() });
    eprintln!("dropped instance {busiest} covered {px} px");
    check_control("dropped_instance", &rep, rep.crack > 0);

    let (rep, _) = run(&mut rig, AccelFaults { flip_facing: true, ..AccelFaults::default() });
    check_control("flipped_facing", &rep, rep.failures() * 2 > rep.ref_hits);

    let (rep, _) = run(&mut rig, AccelFaults { custom_index_shift: 1, ..AccelFaults::default() });
    check_control("custom_index_shifted", &rep, rep.material + rep.face + rep.bad_id > 0);

    rig.g.wait_idle().unwrap();
    gm.free_now(&rig.g, &mut rig.alloc);
    assert_clean(&rig.g);
    rig.finish();
}

/// Criterion 4: a refused grant leaves nothing allocated and is counted.
#[test]
fn accel_build_is_all_or_nothing_on_refusal() {
    let (w, _) = scene::street_block();
    let g = gpu();
    let meshes = street_meshes(&w, Merge::Greedy);
    let size = RegionSize::Chunk;
    let rs = regions(&meshes, size);
    // Room for the meshes, the staging ring and 4 MiB more (1 MiB blocks), not for the BLAS.
    let align = gpu::mesh::section_align(&g);
    let mesh_bytes: u64 = rs.values().map(|r| r.sections(align).total).sum();
    let total = mesh_bytes.next_multiple_of(1 << 20) + (4 << 20) + (4 << 20);
    let mut alloc = Allocator::with_block_bytes(Budget::unlimited().with_total(total), 1 << 20);
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, 4 << 20).unwrap();
    let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, size, &rs).unwrap();
    let before = (alloc.ledger().account(Category::GpuAccel), alloc.ledger().account(Category::GpuAccelScratch), alloc.stats());
    match Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()) {
        Err(GpuError::OverBudget(e)) => eprintln!("build refused: {e:?}"),
        other => panic!("expected OverBudget, got {:?}", other.map(|a| a.stats)),
    }
    let after = (alloc.ledger().account(Category::GpuAccel), alloc.ledger().account(Category::GpuAccelScratch), alloc.stats());
    eprintln!("mesh {:.2} MiB; before {:?}; after {:?}", mib(gm.device_bytes()), before, after);
    assert_eq!(before.0.usage, after.0.usage, "no GpuAccel bytes left reserved");
    assert_eq!(before.1.usage, after.1.usage, "no scratch left reserved");
    assert_eq!(before.2.buffers_live, after.2.buffers_live, "all-or-nothing");
    assert!(after.0.refusals + after.1.refusals > before.0.refusals + before.1.refusals, "refusal counted");
    g.wait_idle().unwrap();
    gm.free_now(&g, &mut alloc);
    up.destroy(&g, &mut alloc);
    tl.destroy(&g);
    assert_eq!(alloc.destroy(&g), 0);
    assert_clean(&g);
}

/// Criterion 6, data only: raster G-buffer and ray-query trace at 1080p, interleaved.
#[test]
fn trace_and_raster_cost_at_1080p() {
    const FW: u32 = 1920;
    const FH: u32 = 1080;
    const REPS: usize = 30;
    let (w, _) = scene::street_block();
    let mut rig = Rig::new(FW, FH);
    let cams = cameras(FW, FH);
    for (merge, size) in [(Merge::Greedy, RegionSize::Chunk), (Merge::Greedy, RegionSize::Brick), (Merge::None, RegionSize::Chunk)] {
        let meshes = street_meshes(&w, merge);
        let rs = regions(&meshes, size);
        let gm = rig.upload(size, &rs);
        let accel = rig.build(&gm, AccelFaults::default());
        let rb = rig.raster.bind(&rig.g, &gm).unwrap();
        let yb = rig.ray.bind(&rig.g, &accel, &rig.ray_out).unwrap();
        let mut timer = GpuTimer::new(&rig.g, 1, 2).unwrap();
        let mut sub = Submitter::new(&rig.g).unwrap();
        for (name, cam) in &cams {
            let (mut ras, mut ray) = (Vec::new(), Vec::new());
            for rep in 0..REPS + 2 {
                let g = &rig.g;
                let cmd = sub.begin(g, &rig.tl).unwrap();
                timer.reset(g, cmd, 0);
                // Alternate which pass goes first, so neither always runs on a warmer GPU.
                let order = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
                for pass in order {
                    timer.begin_pass(g, cmd, 0, pass);
                    if pass == 0 {
                        rig.raster.record(g, cmd, &rig.targets, &gm, &rb, cam, Faults::default());
                    } else {
                        rig.ray.record(g, cmd, &yb, &rig.ray_out, cam);
                    }
                    timer.end_pass(g, cmd, 0, pass);
                }
                let v = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
                rig.tl.wait(g, v, u64::MAX).unwrap();
                let ms = timer.read(g, 0).unwrap().unwrap();
                if rep >= 2 {
                    ras.push(ms[0]);
                    ray.push(ms[1]);
                }
            }
            let p = |v: &[f64], q| percentile(v, q).unwrap();
            eprintln!(
                "cost {name} merge {} region {} {FW}x{FH}: raster gbuffer median {:.3} p90 {:.3} ms | ray trace median {:.3} p90 {:.3} ms ({REPS} reps, interleaved)",
                merge.name(),
                size.name(),
                p(&ras, 50.0),
                p(&ras, 90.0),
                p(&ray, 50.0),
                p(&ray, 90.0)
            );
        }
        let g = &rig.g;
        g.wait_idle().unwrap();
        sub.destroy(g);
        timer.destroy(g);
        rb.destroy(g);
        yb.destroy(g);
        accel.free_now(g, &mut rig.alloc);
        gm.free_now(g, &mut rig.alloc);
    }
    assert_clean(&rig.g);
    rig.finish();
}
