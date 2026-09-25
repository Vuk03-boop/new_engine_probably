//! Phase 3A GPU tests: the reference path tracer (`gpu::reference`) against the CPU reference
//! (`light`), per ADR-0005. Criteria are declared in `docs/changes/2026-09-24-phase3a-reference.md`.
//! Like the other GPU tests they need the Vulkan SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test reference -- --test-threads=1 --nocapture`.

use std::time::Instant;

use derived::{extract_world, Merge};
use gpu::accel::{Accel, AccelFaults};
use gpu::alloc::Allocator;
use gpu::layout::{build_regions, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::Camera;
use gpu::reference::{RefAccum, RefBindings, RefFaults, RefImage, RefMaterials, Reference};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::{Gpu, GpuError, Timeline};
use light::reference::{self as cpu, Accum, Lighting, Pinhole, Settings};
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::{scene, World};

const NEAR: f64 = 0.1;
const SEED: u32 = 0x3A;

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3A needs the ray-tracing device (P-001)");
    g
}

fn street_camera(w: u32, h: u32) -> Camera {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)
}

/// A second view: along the street, low, with sky and shadowed façades.
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

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    world: World,
    albedo: Vec<[f64; 3]>,
    gm: Option<GpuMeshes>,
    accel: Option<Accel>,
    mats: Option<RefMaterials>,
    reference: Reference,
}

impl Rig {
    fn new() -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let (world, _) = scene::street_block();
        let albedo = cpu::albedos(world.materials()).unwrap();
        let meshes: Vec<_> = world.bricks().map(|(k, _)| (k, extract_world(&world, k, Merge::Greedy).unwrap())).collect();
        let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk);
        let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs).unwrap();
        let accel = Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()).unwrap();
        let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedo).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let reference = Reference::new(&g).unwrap();
        Rig { g, alloc, tl, up, world, albedo, gm: Some(gm), accel: Some(accel), mats: Some(mats), reference }
    }

    fn target(&mut self, w: u32, h: u32) -> (RefAccum, RefBindings) {
        let out = RefAccum::new(&self.g, &mut self.alloc, w, h).unwrap();
        let b = self.reference.bind(&self.g, self.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &out).unwrap();
        (out, b)
    }

    #[allow(clippy::too_many_arguments)]
    fn render(&mut self, out: &RefAccum, b: &RefBindings, cam: &Camera, light: &Lighting, s: &Settings, faults: RefFaults, frames: u32, per_submit: u32) -> RefImage {
        self.render_from(out, b, cam, light, s, faults, 0, frames, per_submit)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_from(&mut self, out: &RefAccum, b: &RefBindings, cam: &Camera, light: &Lighting, s: &Settings, faults: RefFaults, first: u32, frames: u32, per_submit: u32) -> RefImage {
        let ms = self.reference.accumulate(&self.g, &mut self.tl, b, out, cam, light, s, faults, SEED, first..first + frames, first, per_submit).unwrap();
        eprintln!("  gpu: {frames} frames at {}x{} in {ms:.0} ms", cam.width, cam.height);
        self.reference.read(&self.g, &mut self.alloc, &mut self.tl, out, frames).unwrap()
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
        self.mats.take().unwrap().free(g, &mut self.alloc);
        self.accel.take().unwrap().free_now(g, &mut self.alloc);
        self.gm.take().unwrap().free_now(g, &mut self.alloc);
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

/// Criterion 1: the module's layout is checked from reflection (a wrong one is refused), and the
/// shader's atmosphere constants agree with `light::Atmosphere`: GPU and CPU transmittance along
/// the same directions with the same step count.
#[test]
fn layout_is_checked_and_atmosphere_constants_agree() {
    let mut rig = Rig::new();
    match Reference::with_reflection(&rig.g, gpu::ray::REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("reference pipeline with the ray-primary reflection refused: {} errors", e.len()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }
    // A wide view straddling the horizon: transmittance from nearly 0 to the zenith.
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.3, 1.0], 150.0, 160, 160, NEAR);
    let (out, b) = rig.target(160, 160);
    let atm = Atmosphere::default();
    let light = lighting(12.0);
    let steps = 64;
    rig.reference.probe_transmittance(&rig.g, &mut rig.tl, &b, &out, &cam, &light, steps).unwrap();
    let img = rig.reference.read(&rig.g, &mut rig.alloc, &mut rig.tl, &out, 1).unwrap();
    let ph = pinhole(&cam);
    let (mut worst, mut zero_mismatch, mut compared) = (0.0f64, 0, 0);
    for y in 0..cam.height {
        for x in 0..cam.width {
            let d = light::sample::normalize(ph.dir(x, y));
            let c = atm.transmittance(atm.observer(), d, steps);
            let g = img.accum.sum[(y * cam.width + x) as usize];
            if (c[0] == 0.0) != (g[0] == 0.0) {
                zero_mismatch += 1;
                continue;
            }
            for k in 0..3 {
                if c[k] > 1e-3 {
                    compared += 1;
                    worst = worst.max((g[k] / c[k] - 1.0).abs());
                }
            }
        }
    }
    eprintln!("transmittance probe: {compared} channel values compared, worst relative error {worst:.2e}, ground/sky disagreements {zero_mismatch}");
    rig.release(out, b);
    rig.finish();
    assert!(compared > 30_000);
    assert!(worst < 2e-3, "GPU and CPU transmittance differ by {worst}");
    assert!(zero_mismatch <= 2, "ground-hit decisions differ on {zero_mismatch} pixels");
}

/// Criterion 2: white furnace. Albedo 1 under a uniform sky of 1: every path that escapes is exactly 1.
/// Control: a 10% PDF error must be caught.
#[test]
fn white_furnace_is_one_and_a_wrong_pdf_is_caught() {
    let mut rig = Rig::new();
    let cam = street_camera(320, 180);
    let (out, b) = rig.target(320, 180);
    let s = Settings { uniform_sky: true, albedo_one: true, max_bounces: 64, ..Settings::default() };
    let light = lighting(12.0);
    let mut stats = Vec::new();
    for faults in [RefFaults::default(), RefFaults { wrong_pdf: true, ..RefFaults::default() }] {
        let img = rig.render(&out, &b, &cam, &light, &s, faults, 16, 16);
        let n = img.accum.sum.len();
        let exact = (0..n).filter(|&i| img.accum.mean(i).iter().all(|&v| v == 1.0)).count();
        let mean: f64 = (0..n).map(|i| img.accum.mean(i)[1]).sum::<f64>() / n as f64;
        eprintln!("furnace {faults:?}: image mean {mean:.6}, {exact} of {n} pixels exactly 1, bad samples {}", img.bad_samples);
        stats.push((mean, exact, n, img.bad_samples));
    }
    rig.release(out, b);
    rig.finish();
    let (mean, exact, n, bad) = stats[0];
    assert_eq!(bad, 0);
    assert!((mean - 1.0).abs() < 1e-3, "furnace mean {mean}");
    assert!(exact * 1000 >= n * 999, "only {exact} of {n} pixels are exactly 1");
    assert!(stats[1].0 > 1.05, "the planted PDF fault was not caught: mean {}", stats[1].0);
}

/// Criterion 3: point sun, no sky, no bounce is deterministic: per pixel the GPU equals the CPU
/// closed form ρ/π × E × T × cos θ (or 0 in shadow). Shadow decisions may differ only on a thin set
/// of edge pixels. Control: dropping the cosine must be caught.
#[test]
fn point_sun_matches_the_cpu_per_pixel() {
    let mut rig = Rig::new();
    let (w, h) = (320, 180);
    let mut failed = Vec::new();
    for (cname, cam) in [("street_view", street_camera(w, h)), ("low", low_camera(w, h))] {
        let (out, b) = rig.target(w, h);
        for hour in [8.0, 12.0, 17.75] {
            let light = lighting(hour);
            let s = Settings { sky: false, point_sun: true, max_bounces: 0, ..Settings::default() };
            let t = Instant::now();
            let c = cpu::render(&rig.world, &rig.albedo, &light, &pinhole(&cam), &s, 0, 1, SEED, 2);
            let cpu_s = t.elapsed().as_secs_f64();
            for faults in [RefFaults::default(), RefFaults { no_cosine: true, ..RefFaults::default() }] {
                let g = rig.render(&out, &b, &cam, &light, &s, faults, 1, 1);
                let (mut lit, mut shadow_mismatch, mut value_mismatch, mut worst) = (0, 0, 0, 0.0f64);
                for i in 0..c.sum.len() {
                    let (cv, gv) = (c.sum[i], g.accum.sum[i]);
                    if (cv[1] > 0.0) != (gv[1] > 0.0) {
                        shadow_mismatch += 1;
                    } else if cv[1] > 0.0 {
                        lit += 1;
                        let e = (0..3).map(|k| (gv[k] / cv[k] - 1.0).abs()).fold(0.0, f64::max);
                        worst = worst.max(e);
                        if e > 1e-4 {
                            value_mismatch += 1;
                        }
                    }
                }
                let n = c.sum.len();
                eprintln!(
                    "point sun {cname} {hour} h {faults:?}: {lit} lit pixels agree to {worst:.2e} ({value_mismatch} over 1e-4), shadow disagreements {shadow_mismatch} of {n}, bad {}, cpu {cpu_s:.1} s",
                    g.bad_samples
                );
                // At least 1000 lit pixels, so the comparison is not vacuous (the first run's 10% guard
                // was too strict for the noon street view: 2588 lit pixels, 4.5%).
                let pass = value_mismatch == 0 && shadow_mismatch * 500 <= n && g.bad_samples == 0 && lit >= 1000;
                match (faults.no_cosine, pass) {
                    (false, false) => failed.push(format!("{cname}/{hour}")),
                    (true, true) => failed.push(format!("control no_cosine not caught {cname}/{hour}")),
                    _ => {}
                }
            }
        }
        rig.release(out, b);
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

struct Stat {
    mean_z: [f64; 3],
    frac_over_4: f64,
    rel_mean: [f64; 3],
}

/// Compares two independent estimates of the same image: the image mean per channel (z against the
/// combined standard error of the means) and the fraction of pixel-channels with |z| > 4.
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
        // Pixel means are independent, so the variance of the image mean is the sum of pixel variances / n².
        mean_z[k] = (ma - mb) / (va + vb).sqrt();
        rel_mean[k] = ma / mb - 1.0;
    }
    Stat { mean_z, frac_over_4: over as f64 / total.max(1) as f64, rel_mean }
}

/// Diagnostic (on demand, `--ignored`): the per-pixel |z| > 4 rate of the metric itself. A GPU image
/// of `spp` samples from independent frames against the 4096-sample GPU image is a null comparison
/// (same estimator): its rate is the metric's false-positive rate for heavy-tailed pixels. The CPU at
/// the same `spp` is then compared with it.
#[test]
#[ignore]
fn diagnostic_tail_rate_of_the_pixel_metric() {
    let mut rig = Rig::new();
    let (w, h) = (96, 54);
    let s = Settings { max_bounces: 2, ..Settings::default() };
    let cam = street_camera(w, h);
    let (out, b) = rig.target(w, h);
    for (tname, hour) in [("midday", 12.0), ("twilight", 18.25)] {
        let light = lighting(hour);
        let big = rig.render(&out, &b, &cam, &light, &s, RefFaults::default(), 4096, 256);
        for spp in [256, 1024] {
            let null = rig.render_from(&out, &b, &cam, &light, &s, RefFaults::default(), 2_000_000, spp, 256);
            let c = cpu::render(&rig.world, &rig.albedo, &light, &pinhole(&cam), &s, 1_000_000, spp, SEED, 2);
            let (sn, sc) = (compare(&big.accum, &null.accum), compare(&big.accum, &c));
            eprintln!(
                "tail {tname} spp {spp}: null (gpu vs gpu) |z|>4 {:.3}% mean z {:?} | cpu vs gpu |z|>4 {:.3}% mean z {:?}",
                sn.frac_over_4 * 100.0,
                sn.mean_z.map(|x| (x * 100.0).round() / 100.0),
                sc.frac_over_4 * 100.0,
                sc.mean_z.map(|x| (x * 100.0).round() / 100.0)
            );
        }
    }
    rig.release(out, b);
    rig.finish();
}

/// Criterion 4: the GPU reference and the CPU reference estimate the same image, at dawn, midday,
/// dusk and after sunset, with sun, sky and two bounces. Control: the PDF fault must fail it.
#[test]
fn gpu_reference_matches_the_cpu_statistically() {
    let mut rig = Rig::new();
    let (w, h) = (96, 54);
    let (cpu_spp, gpu_spp) = (256, 4096);
    let s = Settings { max_bounces: 2, ..Settings::default() };
    let mut failed = Vec::new();
    for (cname, cam) in [("street_view", street_camera(w, h)), ("low", low_camera(w, h))] {
        let (out, b) = rig.target(w, h);
        for (tname, hour) in [("dawn", 6.25), ("midday", 12.0), ("dusk", 17.75), ("twilight", 18.25)] {
            let light = lighting(hour);
            let t = Instant::now();
            // CPU samples come from frames far from the GPU's, so the two estimates are independent.
            let c = cpu::render(&rig.world, &rig.albedo, &light, &pinhole(&cam), &s, 1_000_000, cpu_spp, SEED, 2);
            let cpu_s = t.elapsed().as_secs_f64();
            // The metric's own |z| > 4 rate here: an independent GPU image at the CPU's sample count
            // (heavy-tailed twilight pixels underestimate their error at 256 samples; see the 3A record).
            let big = rig.render(&out, &b, &cam, &light, &s, RefFaults::default(), gpu_spp, 256);
            let null_img = rig.render_from(&out, &b, &cam, &light, &s, RefFaults::default(), 2_000_000, cpu_spp, 256);
            let null = compare(&big.accum, &null_img.accum).frac_over_4;
            let limit = (1.5 * null + 0.002).max(0.01);
            eprintln!("null {cname} {tname}: gpu vs gpu |z|>4 {:.3}%, limit {:.3}%", null * 100.0, limit * 100.0);
            for faults in [RefFaults::default(), RefFaults { wrong_pdf: true, ..RefFaults::default() }] {
                let g = rig.render(&out, &b, &cam, &light, &s, faults, gpu_spp, 256);
                let st = compare(&g.accum, &c);
                let m = |a: &Accum, k: usize| (0..a.sum.len()).map(|i| a.mean(i)[k]).sum::<f64>() / a.sum.len() as f64;
                eprintln!(
                    "stat {cname} {tname} {faults:?}: image mean gpu [{:.4e} {:.4e} {:.4e}] cpu [{:.4e} {:.4e} {:.4e}], rel {:?}, z {:?}, |z|>4 {:.3}%, bad {}, cpu {cpu_s:.1} s",
                    m(&g.accum, 0),
                    m(&g.accum, 1),
                    m(&g.accum, 2),
                    m(&c, 0),
                    m(&c, 1),
                    m(&c, 2),
                    st.rel_mean.map(|x| (x * 1e4).round() / 1e4),
                    st.mean_z.map(|x| (x * 100.0).round() / 100.0),
                    st.frac_over_4 * 100.0,
                    g.bad_samples
                );
                let pass = st.mean_z.iter().all(|z| z.abs() < 4.0) && st.frac_over_4 <= limit && g.bad_samples == 0;
                match (faults.wrong_pdf, pass) {
                    (false, false) => failed.push(format!("{cname}/{tname}")),
                    (true, true) => failed.push(format!("control wrong_pdf not caught {cname}/{tname}")),
                    _ => {}
                }
            }
        }
        rig.release(out, b);
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}
