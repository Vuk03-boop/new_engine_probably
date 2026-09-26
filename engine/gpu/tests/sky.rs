//! Phase 3C GPU tests: the real-time sky (`gpu::sky`, the sky term of `gpu::shade`) against the host
//! tables (`light::sky`) and the Monte Carlo reference (`gpu::reference`). Criteria are declared in
//! `docs/changes/2026-09-24-phase3c-sky.md`. The `corrected_*` tests re-run criteria 1, 3 and 4 with
//! the sky correction baked from the reference (S-020, `docs/changes/2026-09-24-phase3c-sky-correction.md`);
//! the uncorrected ones are unchanged and serve as its controls. Like the other GPU tests they need the
//! Vulkan SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test sky -- --test-threads=1 --nocapture`.

use ash::vk;
use derived::{extract_world, Merge};
use gpu::accel::{Accel, AccelFaults};
use gpu::alloc::Allocator;
use gpu::debug_view::targets_to_read;
use gpu::layout::{build_regions, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::{Camera, Faults, Raster, Targets};
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::shade::{self, Shade, ShadeFaults, ShadeSettings, ShadeTargets};
use gpu::sky::{SkyTables, SkyView};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::timing::{percentile, GpuTimer};
use gpu::submit::Submitter;
use gpu::{Gpu, GpuError, Timeline};
use light::reference::{albedos, Accum, Lighting, Settings};
use light::sky::{SkyLuts, VIEW_H, VIEW_W};
use light::sky_ref::SkyReference;
use light::sun::REFERENCE_TIMES;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::scene;

const NEAR: f64 = 0.1;
const SEED: u32 = 0x3C;
const SKY_ONLY: ShadeSettings = ShadeSettings { sun: false, sky: true, point_sun: false, uniform_sky: false, bounce: false, emitters: false, emitter_spp: 1 };

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3C needs the ray-tracing device (P-001)");
    g
}

fn lighting(hour: f64) -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(hour))
}

fn street_cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        ("street", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)),
        ("low", Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, NEAR)),
        ("overhead", Camera::look_at([-60.5, 300.25, -80.75], [192.0, 0.0, 150.0], 50.0, w, h, NEAR)),
    ]
}

/// Cameras that see only sky and planet ground: 200,000 voxels (12.5 km) from the street, looking
/// away from it, toward the sun's azimuth, away from it and across, pitched 10° down so the
/// ground below the horizon is in view; plus one looking up.
fn sky_cameras(sun: [f64; 3], w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let centre = [192.0, 32.0, 128.0];
    let az = sun[2].atan2(sun[0]);
    let mut v = Vec::new();
    for (name, a) in [("toward_sun", az), ("away", az + std::f64::consts::PI), ("across", az + std::f64::consts::FRAC_PI_2)] {
        let f = [a.cos(), 0.0, a.sin()];
        let eye = [centre[0] - 200_000.0 * f[0], centre[1], centre[2] - 200_000.0 * f[2]];
        let pitch = (-10f64).to_radians();
        let target = [eye[0] + f[0], eye[1] + pitch.tan(), eye[2] + f[2]];
        v.push((name, Camera::look_at(eye, target, 60.0, w, h, NEAR)));
    }
    let eye = [centre[0] - 200_000.0, 32.0, centre[2]];
    v.push(("up", Camera::look_at(eye, [eye[0] + 0.01, eye[1] + 1.0, eye[2]], 120.0, w, h, NEAR)));
    v
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    luts: SkyLuts,
    mats: Option<RefMaterials>,
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
    gm: Option<GpuMeshes>,
    accel: Option<Accel>,
    raster: Raster,
    shade: Shade,
    reference: Reference,
}

/// The sky tables, corrected from the baked reference (S-020) or not.
fn sky_luts(corrected: bool) -> SkyLuts {
    let t = std::time::Instant::now();
    let mut luts = SkyLuts::new(Atmosphere::default());
    eprintln!("host sky tables built in {:.2} s", t.elapsed().as_secs_f64());
    if corrected {
        let t = std::time::Instant::now();
        let r = SkyReference::load_default(&luts.atmosphere).expect("the baked sky reference (run the sky_bake tool)");
        let st = luts.apply_reference(&r).unwrap();
        eprintln!("sky correction applied in {:.2} s: {st:?}", t.elapsed().as_secs_f64());
    }
    luts
}

impl Rig {
    fn new(corrected: bool) -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let (world, _) = scene::street_block();
        let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedos(world.materials()).unwrap()).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let luts = sky_luts(corrected);
        let (sky, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &luts).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let sky_pass = SkyView::new(&g, &sky).unwrap();
        let meshes: Vec<_> = world.bricks().map(|(k, _)| (k, extract_world(&world, k, Merge::Greedy).unwrap())).collect();
        let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk);
        let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs).unwrap();
        let accel = Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()).unwrap();
        let raster = Raster::new(&g).unwrap();
        let shade = Shade::new(&g).unwrap();
        let reference = Reference::new(&g).unwrap();
        Rig { g, alloc, tl, up, luts, mats: Some(mats), sky: Some(sky), sky_pass: Some(sky_pass), gm: Some(gm), accel: Some(accel), raster, shade, reference }
    }

    fn update_sky(&mut self, sun: [f64; 3]) {
        let mut sub = Submitter::new(&self.g).unwrap();
        let cmd = sub.begin(&self.g, &self.tl).unwrap();
        self.sky_pass.as_ref().unwrap().record(&self.g, cmd, sun);
        let v = sub.submit(&self.g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(&self.g, v, u64::MAX).unwrap();
        self.g.wait_idle().unwrap();
        sub.destroy(&self.g);
    }

    /// The shade pass's frame `frame` (the sky table must be current).
    fn shade_frame(&mut self, cam: &Camera, light: &Lighting, set: ShadeSettings, frame: u32) -> Vec<[f32; 4]> {
        let (w, h) = (cam.width, cam.height);
        let targets = Targets::new(&self.g, &mut self.alloc, vk::Extent2D { width: w, height: h }).unwrap();
        let out = ShadeTargets::new(&self.g, &mut self.alloc, w, h).unwrap();
        let gm = self.gm.as_ref().unwrap();
        let rb = self.raster.bind(&self.g, gm).unwrap();
        let sb = self.shade.bind(&self.g, &targets, self.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &out, &self.sky.as_ref().unwrap().view).unwrap();
        let mut sub = Submitter::new(&self.g).unwrap();
        let g = &self.g;
        let cmd = sub.begin(g, &self.tl).unwrap();
        self.raster.record(g, cmd, &targets, gm, &rb, cam, Faults::default());
        targets_to_read(g, cmd, &targets);
        self.shade.record(g, cmd, &sb, &out, &shade::Params::new(cam, light, set, ShadeFaults::default(), frame, SEED));
        let v = sub.submit(g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(g, v, u64::MAX).unwrap();
        g.wait_idle().unwrap();
        sub.destroy(g);
        rb.destroy(g);
        sb.destroy(g);
        let r = out.read(&self.g, &mut self.alloc, &mut self.tl).unwrap();
        out.free(&self.g, &mut self.alloc);
        targets.free(&self.g, &mut self.alloc);
        r
    }

    fn reference_image(&mut self, cam: &Camera, light: &Lighting, s: &Settings, first: u32, frames: u32) -> Accum {
        let out = RefAccum::new(&self.g, &mut self.alloc, cam.width, cam.height).unwrap();
        let b = self.reference.bind(&self.g, self.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &out).unwrap();
        self.reference.accumulate(&self.g, &mut self.tl, &b, &out, cam, light, s, RefFaults::default(), SEED, first..first + frames, first, 64).unwrap();
        let img = self.reference.read(&self.g, &mut self.alloc, &mut self.tl, &out, frames).unwrap();
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
        self.sky_pass.take().unwrap().destroy(g);
        self.sky.take().unwrap().free(g, &mut self.alloc);
        self.mats.take().unwrap().free(g, &mut self.alloc);
        self.accel.take().unwrap().free_now(g, &mut self.alloc);
        self.gm.take().unwrap().free_now(g, &mut self.alloc);
        self.raster.destroy(g);
        self.shade.destroy(g);
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Criterion 1: layout checked; the GPU sky-view table (f32) equals the host's (f64) per texel at
/// the five reference times, within 0.5% where the host value is above 1e-7.
#[test]
fn gpu_sky_table_equals_the_host_table() {
    sky_table_equals_the_host_table(false);
}

/// S-020 K3: criterion 1 with the correction applied on both sides.
#[test]
fn corrected_gpu_sky_table_equals_the_host_table() {
    sky_table_equals_the_host_table(true);
}

fn sky_table_equals_the_host_table(corrected: bool) {
    let mut rig = Rig::new(corrected);
    assert_eq!(rig.sky.as_ref().unwrap().corrected, corrected);
    match SkyView::with_reflection(&rig.g, rig.sky.as_ref().unwrap(), gpu::shade::REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("sky pipeline with the shade reflection refused: {} errors", e.len()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }
    let mut failed = Vec::new();
    for (name, hour) in REFERENCE_TIMES {
        let sun = SunPath::default().direction(hour);
        rig.update_sky(sun);
        let gpu_view = rig.sky.as_ref().unwrap().read_view(&rig.g, &mut rig.alloc, &mut rig.tl).unwrap();
        let host = rig.luts.view(sun);
        assert_eq!(host.data.len(), VIEW_W * VIEW_H);
        let (mut worst, mut over, mut compared, mut at) = (0.0f64, 0, 0, 0);
        for (i, (g, h)) in gpu_view.iter().zip(&host.data).enumerate() {
            for c in 0..3 {
                if h[c] > 1e-7 {
                    compared += 1;
                    let e = (g[c] as f64 / h[c] - 1.0).abs();
                    if e > worst {
                        worst = e;
                        at = i;
                    }
                    if e > 5e-3 {
                        over += 1;
                    }
                }
            }
        }
        eprintln!("sky table {name} (corrected {corrected}): {compared} values, worst relative {worst:.2e} at texel ({}, {}), {over} over 0.5%", at % VIEW_W, at / VIEW_W);
        if over > 0 {
            failed.push(name);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Criterion 2: with a uniform sky of 1 (sun off), frame f of the shade pass equals the reference's
/// direct-sky sample f per pixel (the albedo if the cosine ray escapes, else 0): the visibility ray
/// and its stream are the reference's. Mismatches at most 0.1% of pixels (the declared edge set).
#[test]
fn uniform_sky_term_equals_the_reference_per_pixel() {
    let mut rig = Rig::new(false);
    let (w, h) = (480, 270);
    let light = lighting(12.0);
    let mut failed = Vec::new();
    for (name, cam) in street_cameras(w, h) {
        let frame = 23;
        let rt = rig.shade_frame(&cam, &light, ShadeSettings { sun: false, sky: false, point_sun: false, uniform_sky: true, bounce: false, emitters: false, emitter_spp: 1 }, frame);
        let rf = rig.reference_image(&cam, &light, &Settings { uniform_sky: true, max_bounces: 0, ..Settings::default() }, frame, 1);
        let (mut mismatch, mut lit, mut bad) = (0, 0, 0);
        for (x, y) in rt.iter().zip(&rf.sum) {
            bad += usize::from(x[3] != 0.0);
            let same = (0..3).all(|k| if y[k] == 0.0 { x[k] == 0.0 } else { (x[k] as f64 / y[k] - 1.0).abs() <= 1e-5 });
            if same {
                lit += usize::from(y[1] > 0.0);
            } else {
                mismatch += 1;
            }
        }
        let n = rt.len();
        eprintln!("uniform sky {name}: {mismatch} of {n} mismatched ({:.3}%), {lit} lit, bad {bad}", 100.0 * mismatch as f64 / n as f64);
        if mismatch * 1000 > n || bad > 0 || lit < 1000 {
            failed.push(name);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Criterion 3: the table's sky against the Monte Carlo sky, per pixel of sky-only cameras, at the
/// five reference times. The error counted is what exceeds 3 standard errors of the reference.
/// Budget: luminance 95th percentile within 5% (dawn to dusk) and 15% (twilight); per channel
/// within 10% (dawn to dusk).
#[test]
fn table_sky_matches_the_monte_carlo_sky() {
    table_sky_against_the_monte_carlo_sky(false);
}

/// S-020 K5: criterion 3 with the correction.
#[test]
fn corrected_table_sky_matches_the_monte_carlo_sky() {
    table_sky_against_the_monte_carlo_sky(true);
}

fn table_sky_against_the_monte_carlo_sky(corrected: bool) {
    let mut rig = Rig::new(corrected);
    let (w, h) = (96, 54);
    let mut failed = Vec::new();
    for (name, hour) in REFERENCE_TIMES {
        let light = lighting(hour);
        rig.update_sky(light.sun_dir);
        let spp = if name == "twilight" { 65_536 } else { 4096 };
        let (mut errs, mut chan, mut raw) = (Vec::new(), Vec::new(), Vec::new());
        for (cname, cam) in sky_cameras(light.sun_dir, w, h) {
            let rt = rig.shade_frame(&cam, &light, SKY_ONLY, 0);
            let rf = rig.reference_image(&cam, &light, &Settings { sun: false, max_bounces: 0, ..Settings::default() }, 0, spp);
            let mut geometry = 0;
            for (i, x) in rt.iter().enumerate() {
                let (m, se) = (rf.mean(i), rf.std_error(i));
                let t = [x[0] as f64, x[1] as f64, x[2] as f64];
                if m[1] <= 0.0 {
                    geometry += 1;
                    continue;
                }
                let beyond = |a: f64, b: f64, e: f64| ((a - b).abs() - 3.0 * e).max(0.0) / b;
                let lse = lum(se);
                errs.push(beyond(lum(t), lum(m), lse));
                raw.push((lum(t) / lum(m) - 1.0).abs());
                chan.push((0..3).map(|c| beyond(t[c], m[c], se[c])).fold(0.0, f64::max));
            }
            eprintln!("  {name} camera {cname}: {} px compared, {geometry} skipped (no sky)", rt.len() - geometry);
        }
        for v in [&mut errs, &mut chan, &mut raw] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let p = |v: &Vec<f64>, q: usize| v[(v.len() * q) / 100];
        eprintln!(
            "sky {name} ({spp} spp, corrected {corrected}): luminance raw mean {:.4} p50 {:.4} p95 {:.4} max {:.4}; beyond 3 se p95 {:.4}; channel beyond 3 se p95 {:.4}",
            raw.iter().sum::<f64>() / raw.len() as f64,
            p(&raw, 50),
            p(&raw, 95),
            raw.last().unwrap(),
            p(&errs, 95),
            p(&chan, 95)
        );
        let (lim, clim) = if name == "twilight" { (0.15, f64::INFINITY) } else { (0.05, 0.10) };
        if p(&errs, 95) > lim || p(&chan, 95) > clim {
            failed.push(format!("{name}: p95 {:.4}, channel p95 {:.4}", p(&errs, 95), p(&chan, 95)));
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Criterion 4: the sky term on the street's surfaces, averaged over 256 frames, against the
/// reference's direct-sky component (sun off, no bounce, Monte Carlo sky, 4096 spp): image mean per
/// channel within 5% (dawn to dusk) and 15% (twilight), the table-sky budget.
///
/// KNOWN FAILURE of the uncorrected table (2026-09-24, 3C record): dawn and dusk blue -5.7%,
/// twilight blue +15.5%, just over the declared budget; the other arms pass. The cause is the
/// table's isotropic multiple-scattering approximation. The fix chosen by the user (S-020) is the
/// correction baked from the reference: `corrected_surface_sky_term_converges_to_the_reference`
/// passes this criterion. This uncorrected run is kept, unloosened, as its control (it must still
/// fail with the recorded numbers). Run it with `--ignored`.
#[test]
#[ignore = "control: the uncorrected table fails criterion 4 as recorded (3C record); the corrected test passes it (S-020)"]
fn surface_sky_term_converges_to_the_reference() {
    surface_sky_term_against_the_reference(false);
}

/// S-020 K4: criterion 4 with the correction, at the declared budget (5% dawn to dusk, 15%
/// twilight). The uncorrected test above is its control.
#[test]
fn corrected_surface_sky_term_converges_to_the_reference() {
    surface_sky_term_against_the_reference(true);
}

fn surface_sky_term_against_the_reference(corrected: bool) {
    let mut rig = Rig::new(corrected);
    let (w, h) = (160, 90);
    let frames = 256u32;
    let mut failed = Vec::new();
    for (name, hour) in REFERENCE_TIMES {
        let light = lighting(hour);
        rig.update_sky(light.sun_dir);
        for (cname, cam) in street_cameras(w, h).into_iter().take(2) {
            let mut acc = vec![[0.0f64; 3]; (w * h) as usize];
            for f in 0..frames {
                for (a, x) in acc.iter_mut().zip(rig.shade_frame(&cam, &light, SKY_ONLY, f)) {
                    for c in 0..3 {
                        a[c] += x[c] as f64 / frames as f64;
                    }
                }
            }
            let spp = if name == "twilight" { 16_384 } else { 4096 };
            let rf = rig.reference_image(&cam, &light, &Settings { sun: false, max_bounces: 0, ..Settings::default() }, 1_000_000, spp);
            let n = acc.len() as f64;
            let rt_mean = [0, 1, 2].map(|c| acc.iter().map(|a| a[c]).sum::<f64>() / n);
            let rf_mean = [0, 1, 2].map(|c| (0..acc.len()).map(|i| rf.mean(i)[c]).sum::<f64>() / n);
            let rel = [0, 1, 2].map(|c| rt_mean[c] / rf_mean[c] - 1.0);
            eprintln!("surface sky {name} {cname} (corrected {corrected}): real time {:.4e} {:.4e} {:.4e} reference {:.4e} {:.4e} {:.4e} relative {:?}", rt_mean[0], rt_mean[1], rt_mean[2], rf_mean[0], rf_mean[1], rf_mean[2], rel.map(|r| (r * 1e4).round() / 1e4));
            let lim = if name == "twilight" { 0.15 } else { 0.05 };
            if rel.iter().any(|r| r.abs() > lim) {
                failed.push(format!("{name}/{cname}"));
            }
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// S-020 K7, data: the sky-view pass (the only pass the correction touches) with the uncorrected
/// and the corrected tables, interleaved, 40 reps after 2 warm-up reps. Validation off gives the
/// timing run: `set NE_NO_VALIDATION=1`.
#[test]
fn sky_view_pass_cost() {
    const REPS: usize = 40;
    let g = Gpu::new().expect("an RT-capable Vulkan device");
    let validation = g.validation_enabled();
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let mut tables = Vec::new();
    for corrected in [false, true] {
        let (t, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &sky_luts(corrected)).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        eprintln!("sky tables (corrected {corrected}): {} bytes on the device", t.device_bytes());
        let pass = SkyView::new(&g, &t).unwrap();
        tables.push((t, pass));
    }
    let mut timer = GpuTimer::new(&g, 1, 2).unwrap();
    let mut sub = Submitter::new(&g).unwrap();
    for (name, hour) in REFERENCE_TIMES {
        let sun = SunPath::default().direction(hour);
        let mut ms = [Vec::new(), Vec::new()];
        for rep in 0..REPS + 2 {
            let cmd = sub.begin(&g, &tl).unwrap();
            timer.reset(&g, cmd, 0);
            for k in 0..2 {
                let arm = (k + rep) % 2;
                timer.begin_pass(&g, cmd, 0, arm as u32);
                tables[arm].1.record(&g, cmd, sun);
                timer.end_pass(&g, cmd, 0, arm as u32);
            }
            let v = sub.submit(&g, &mut tl, cmd, &[]).unwrap();
            tl.wait(&g, v, u64::MAX).unwrap();
            let t = timer.read(&g, 0).unwrap().unwrap();
            if rep >= 2 {
                ms[0].push(t[0]);
                ms[1].push(t[1]);
            }
        }
        let p = |v: &[f64], q| percentile(v, q).unwrap();
        eprintln!(
            "sky-view pass {name} (validation {validation}): uncorrected p50 {:.4} p90 {:.4} ms, corrected p50 {:.4} p90 {:.4} ms",
            p(&ms[0], 50.0),
            p(&ms[0], 90.0),
            p(&ms[1], 50.0),
            p(&ms[1], 90.0)
        );
    }
    g.wait_idle().unwrap();
    sub.destroy(&g);
    timer.destroy(&g);
    for (t, pass) in tables {
        pass.destroy(&g);
        t.free(&g, &mut alloc);
    }
    up.destroy(&g, &mut alloc);
    tl.destroy(&g);
    assert_eq!(alloc.destroy(&g), 0, "leaked buffers");
}
