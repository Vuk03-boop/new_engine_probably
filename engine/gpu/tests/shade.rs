//! Phase 3B GPU tests: real-time lighting (`gpu::shade`) against the 3A reference, per pixel and
//! per frame, plus its cost. Criteria are declared in `docs/changes/2026-09-24-phase3b-sun.md`.
//! Like the other GPU tests they need the Vulkan SDK and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test shade -- --test-threads=1 --nocapture`.

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
use gpu::submit::Submitter;
use gpu::timing::{percentile, GpuTimer};
use gpu::{Gpu, GpuError, Timeline};
use light::reference::{albedos, Lighting, Settings};
use light::sky::SkyLuts;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::{scene, World};

const NEAR: f64 = 0.1;
const SEED: u32 = 0x3B;
/// Sun and sky without the 3F bounce: the lighting this file's criteria were measured with.
const DIRECT: ShadeSettings = ShadeSettings { sun: true, sky: true, point_sun: false, uniform_sky: false, bounce: false, emitters: false, emitter_spp: 1 };
/// The 3B term alone.
const SUN_ONLY: ShadeSettings = ShadeSettings { sun: true, sky: false, point_sun: false, uniform_sky: false, bounce: false, emitters: false, emitter_spp: 1 };

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3B needs the ray-tracing device (P-001)");
    g
}

fn cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        ("street", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, NEAR)),
        ("low", Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, NEAR)),
        ("overhead", Camera::look_at([-60.5, 300.25, -80.75], [192.0, 0.0, 150.0], 50.0, w, h, NEAR)),
    ]
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
    mats: Option<RefMaterials>,
    raster: Raster,
    shade: Shade,
    reference: Reference,
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
}

struct Scene {
    gm: GpuMeshes,
    accel: Accel,
}

impl Rig {
    fn new() -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let (world, _) = scene::street_block();
        let al = albedos(world.materials()).unwrap();
        let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &al).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let raster = Raster::new(&g).unwrap();
        let shade = Shade::new(&g).unwrap();
        let reference = Reference::new(&g).unwrap();
        let (sky, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &SkyLuts::new(Atmosphere::default())).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let sky_pass = SkyView::new(&g, &sky).unwrap();
        Rig { g, alloc, tl, up, world, mats: Some(mats), raster, shade, reference, sky: Some(sky), sky_pass: Some(sky_pass) }
    }

    fn scene(&mut self, merge: Merge, size: RegionSize) -> Scene {
        let meshes: Vec<_> = self.world.bricks().map(|(k, _)| (k, extract_world(&self.world, k, merge).unwrap())).collect();
        let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), size);
        let (gm, _) = GpuMeshes::upload(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, size, &rs).unwrap();
        let accel = Accel::build(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, &gm, AccelFaults::default()).unwrap();
        Scene { gm, accel }
    }

    fn free_scene(&mut self, s: Scene) {
        self.g.wait_idle().unwrap();
        s.accel.free_now(&self.g, &mut self.alloc);
        s.gm.free_now(&self.g, &mut self.alloc);
    }

    /// Raster G-buffer, then the shade pass for `frame`; returns the radiance.
    #[allow(clippy::too_many_arguments)]
    fn shade_frame(&mut self, s: &Scene, targets: &Targets, out: &ShadeTargets, cam: &Camera, light: &Lighting, set: ShadeSettings, faults: ShadeFaults, frame: u32) -> Vec<[f32; 4]> {
        let rb = self.raster.bind(&self.g, &s.gm).unwrap();
        let sb = self.shade.bind(&self.g, targets, &s.accel, self.mats.as_ref().unwrap(), out, &self.sky.as_ref().unwrap().view).unwrap();
        let mut sub = Submitter::new(&self.g).unwrap();
        let g = &self.g;
        let cmd = sub.begin(g, &self.tl).unwrap();
        self.sky_pass.as_ref().unwrap().record(g, cmd, light.sun_dir);
        self.raster.record(g, cmd, targets, &s.gm, &rb, cam, Faults::default());
        targets_to_read(g, cmd, targets);
        self.shade.record(g, cmd, &sb, out, &shade::Params::new(cam, light, set, faults, frame, SEED));
        let v = sub.submit(g, &mut self.tl, cmd, &[]).unwrap();
        self.tl.wait(g, v, u64::MAX).unwrap();
        g.wait_idle().unwrap();
        sub.destroy(g);
        rb.destroy(g);
        sb.destroy(g);
        out.read(&self.g, &mut self.alloc, &mut self.tl).unwrap()
    }

    /// The reference's sun-only sample of `frame` (no sky, no bounce).
    fn reference_frame(&mut self, s: &Scene, cam: &Camera, light: &Lighting, point_sun: bool, frame: u32) -> Vec<[f64; 3]> {
        let out = RefAccum::new(&self.g, &mut self.alloc, cam.width, cam.height).unwrap();
        let b = self.reference.bind(&self.g, &s.accel, self.mats.as_ref().unwrap(), &out).unwrap();
        let set = Settings { sky: false, max_bounces: 0, point_sun, ..Settings::default() };
        self.reference.accumulate(&self.g, &mut self.tl, &b, &out, cam, light, &set, RefFaults::default(), SEED, frame..frame + 1, frame, 1).unwrap();
        let img = self.reference.read(&self.g, &mut self.alloc, &mut self.tl, &out, 1).unwrap();
        assert_eq!(img.bad_samples, 0);
        self.g.wait_idle().unwrap();
        b.destroy(&self.g);
        out.free(&self.g, &mut self.alloc);
        img.accum.sum
    }

    fn finish(mut self) {
        let g = &self.g;
        g.wait_idle().unwrap();
        let (errors, warnings) = g.validation_counts();
        eprintln!("validation: {errors} errors, {warnings} warnings");
        assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
        self.mats.take().unwrap().free(g, &mut self.alloc);
        self.sky_pass.take().unwrap().destroy(g);
        self.sky.take().unwrap().free(g, &mut self.alloc);
        self.raster.destroy(g);
        self.shade.destroy(g);
        self.reference.destroy(g);
        self.up.destroy(g, &mut self.alloc);
        self.tl.destroy(g);
        assert_eq!(self.alloc.destroy(g), 0, "leaked buffers");
    }
}

struct Agreement {
    equal: usize,
    lit: usize,
    mismatch: usize,
    worst: f64,
    bad: usize,
    example: Option<String>,
}

/// Per pixel: equal when both are 0 or all channels agree to 1e-4 relative.
fn agreement(rt: &[[f32; 4]], rf: &[[f64; 3]], w: u32) -> Agreement {
    let mut a = Agreement { equal: 0, lit: 0, mismatch: 0, worst: 0.0, bad: 0, example: None };
    for (i, (x, y)) in rt.iter().zip(rf).enumerate() {
        if x[3] != 0.0 {
            a.bad += 1;
        }
        let zero = (x[0] == 0.0 && x[1] == 0.0 && x[2] == 0.0, y[0] == 0.0 && y[1] == 0.0 && y[2] == 0.0);
        match zero {
            (true, true) => a.equal += 1,
            (false, false) => {
                let e = (0..3).map(|k| if y[k] == 0.0 { if x[k] == 0.0 { 0.0 } else { f64::INFINITY } } else { (x[k] as f64 / y[k] - 1.0).abs() }).fold(0.0, f64::max);
                a.worst = a.worst.max(e);
                if e <= 1e-4 {
                    a.equal += 1;
                    a.lit += 1;
                } else {
                    a.mismatch += 1;
                    a.example.get_or_insert_with(|| format!("({}, {}) rt {:?} ref {:?}", i as u32 % w, i as u32 / w, x, y));
                }
            }
            _ => {
                a.mismatch += 1;
                a.example.get_or_insert_with(|| format!("({}, {}) rt {:?} ref {:?} (one is 0)", i as u32 % w, i as u32 / w, x, y));
            }
        }
    }
    a
}

/// Criterion 1: the module layout is checked from reflection.
#[test]
fn shade_layout_is_checked() {
    let rig = Rig::new();
    match Shade::with_reflection(&rig.g, gpu::reference::REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("shade pipeline with the reference reflection refused: {} errors", e.len()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }
    rig.finish();
}

/// Criteria 2 and 3: frame f of the real-time sun equals the reference's sun-only sample f per
/// pixel (disk and point sun), at dawn, morning, midday and dusk, for three cameras. Mismatches
/// (surface position rebuilt from depth versus the ray hit) at most 0.1% of pixels per arm.
/// Planted faults must fail at least half of the arms and mismatch at least 10x as many pixels in
/// total as the correct arms: a fault can be invisible in one configuration (flipped x normals
/// under the noon point sun, whose x component is 0), so "every arm" is not a valid requirement.
#[test]
fn real_time_sun_equals_the_reference_sample_per_pixel() {
    let mut rig = Rig::new();
    let (w, h) = (480, 270);
    let s = rig.scene(Merge::Greedy, RegionSize::Chunk);
    let targets = Targets::new(&rig.g, &mut rig.alloc, vk::Extent2D { width: w, height: h }).unwrap();
    let out = ShadeTargets::new(&rig.g, &mut rig.alloc, w, h).unwrap();
    let n = (w * h) as usize;
    let mut failed = Vec::new();
    // Per fault: (arms, arms failed, total mismatches).
    let mut tally: std::collections::BTreeMap<&str, (usize, usize, usize)> = Default::default();
    for (cname, cam) in cameras(w, h) {
        for hour in [6.25, 8.0, 12.0, 17.75] {
            let light = lighting(hour);
            for point_sun in [false, true] {
                let frame = 17;
                let rf = rig.reference_frame(&s, &cam, &light, point_sun, frame);
                let arms = [("ok", ShadeFaults::default()), ("flip_x", ShadeFaults { flip_x: true, ..ShadeFaults::default() }), ("depth", ShadeFaults { depth: true, ..ShadeFaults::default() })];
                for (fname, faults) in arms {
                    let rt = rig.shade_frame(&s, &targets, &out, &cam, &light, ShadeSettings { sun: true, sky: false, point_sun, uniform_sky: false, bounce: false, emitters: false, emitter_spp: 1 }, faults, frame);
                    let a = agreement(&rt, &rf, w);
                    let pass = a.mismatch * 1000 <= n && a.bad == 0 && a.lit >= 1000;
                    eprintln!(
                        "sun {cname} {hour} h {} {fname}: {} equal ({} lit, worst {:.1e}), {} mismatched ({:.3}%), bad {} {}",
                        if point_sun { "point" } else { "disk" },
                        a.equal,
                        a.lit,
                        a.worst,
                        a.mismatch,
                        100.0 * a.mismatch as f64 / n as f64,
                        a.bad,
                        a.example.as_deref().unwrap_or("")
                    );
                    let t = tally.entry(fname).or_default();
                    t.0 += 1;
                    t.1 += usize::from(!pass);
                    t.2 += a.mismatch;
                    if fname == "ok" && !pass {
                        failed.push(format!("{cname}/{hour}/{point_sun}"));
                    }
                }
            }
        }
    }
    let ok_total = tally["ok"].2;
    for (f, (arms, failing, total)) in &tally {
        eprintln!("tally {f}: {failing} of {arms} arms fail, {total} mismatched pixels in total");
        if *f != "ok" && (failing * 2 < *arms || *total < 10 * ok_total.max(1)) {
            failed.push(format!("fault {f} not caught: {failing} of {arms} arms, {total} px (ok arms {ok_total} px)"));
        }
    }
    // A different frame must differ (the streams are per frame), so the equality is not a constant.
    let (_, cam) = cameras(w, h).remove(0);
    let light = lighting(8.0);
    let rf = rig.reference_frame(&s, &cam, &light, false, 17);
    let rt = rig.shade_frame(&s, &targets, &out, &cam, &light, SUN_ONLY, ShadeFaults::default(), 18);
    let a = agreement(&rt, &rf, w);
    eprintln!("control: frame 18 against reference frame 17: {} mismatched", a.mismatch);
    if a.mismatch == 0 {
        failed.push("frame control: different frames agree".into());
    }
    rig.g.wait_idle().unwrap();
    out.free(&rig.g, &mut rig.alloc);
    targets.free(&rig.g, &mut rig.alloc);
    rig.free_scene(s);
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Criterion 4, data only: G-buffer and shade pass at 1080p per granularity setting, interleaved
/// (ADR-0003 Amendment 2 asks for this when secondary rays arrive). Validation off gives the timing
/// run: `set NE_NO_VALIDATION=1`; with validation on the numbers include its overhead.
#[test]
fn shade_cost_at_1080p() {
    const FW: u32 = 1920;
    const FH: u32 = 1080;
    const REPS: usize = 40;
    let g = Gpu::new().expect("an RT-capable Vulkan device");
    assert!(g.ray_tracing());
    let validation = g.validation_enabled();
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let (world, _) = scene::street_block();
    let al = albedos(world.materials()).unwrap();
    let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &al).unwrap();
    tl.wait(&g, v, u64::MAX).unwrap();
    let raster = Raster::new(&g).unwrap();
    let shade_pass = Shade::new(&g).unwrap();
    let (sky, v) = SkyTables::upload(&g, &mut alloc, &mut up, &mut tl, &SkyLuts::new(Atmosphere::default())).unwrap();
    tl.wait(&g, v, u64::MAX).unwrap();
    let sky_pass = SkyView::new(&g, &sky).unwrap();
    let targets = Targets::new(&g, &mut alloc, vk::Extent2D { width: FW, height: FH }).unwrap();
    let out = ShadeTargets::new(&g, &mut alloc, FW, FH).unwrap();
    let light = lighting(8.0);
    for (merge, size) in [(Merge::Greedy, RegionSize::Chunk), (Merge::Greedy, RegionSize::Chunks2), (Merge::Greedy, RegionSize::Brick), (Merge::None, RegionSize::Chunk)] {
        let meshes: Vec<_> = world.bricks().map(|(k, _)| (k, extract_world(&world, k, merge).unwrap())).collect();
        let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), size);
        let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, size, &rs).unwrap();
        let accel = Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()).unwrap();
        let rb = raster.bind(&g, &gm).unwrap();
        let sb = shade_pass.bind(&g, &targets, &accel, &mats, &out, &sky.view).unwrap();
        let mut timer = GpuTimer::new(&g, 1, 3).unwrap();
        let mut sub = Submitter::new(&g).unwrap();
        for (name, cam) in cameras(FW, FH) {
            let (mut ras, mut sh, mut bo) = (Vec::new(), Vec::new(), Vec::new());
            for rep in 0..REPS + 2 {
                let cmd = sub.begin(&g, &tl).unwrap();
                sky_pass.record(&g, cmd, light.sun_dir);
                timer.reset(&g, cmd, 0);
                timer.begin_pass(&g, cmd, 0, 0);
                raster.record(&g, cmd, &targets, &gm, &rb, &cam, Faults::default());
                timer.end_pass(&g, cmd, 0, 0);
                targets_to_read(&g, cmd, &targets);
                // 3F: the pinned 3C lighting (slot 1) and the default with the bounce (slot 2), in
                // alternating order.
                let arms = [(1, DIRECT), (2, ShadeSettings::default())];
                for k in 0..2 {
                    let (slot, set) = arms[(k + rep) % 2];
                    timer.begin_pass(&g, cmd, 0, slot);
                    shade_pass.record(&g, cmd, &sb, &out, &shade::Params::new(&cam, &light, set, ShadeFaults::default(), rep as u32, SEED));
                    timer.end_pass(&g, cmd, 0, slot);
                    gpu::reference::compute_to_compute(&g, cmd);
                }
                let v = sub.submit(&g, &mut tl, cmd, &[]).unwrap();
                tl.wait(&g, v, u64::MAX).unwrap();
                let ms = timer.read(&g, 0).unwrap().unwrap();
                if rep >= 2 {
                    ras.push(ms[0]);
                    sh.push(ms[1]);
                    bo.push(ms[2]);
                }
            }
            let p = |v: &[f64], q| percentile(v, q).unwrap();
            eprintln!(
                "cost {name} merge {} region {} {FW}x{FH} (validation {validation}): gbuffer median {:.3} p90 {:.3} ms | shade (sun + sky, 2 rays) median {:.3} p90 {:.3} ms | with the bounce (up to 4 rays) median {:.3} p90 {:.3} ms ({REPS} reps, interleaved)",
                merge.name(),
                size.name(),
                p(&ras, 50.0),
                p(&ras, 90.0),
                p(&sh, 50.0),
                p(&sh, 90.0),
                p(&bo, 50.0),
                p(&bo, 90.0)
            );
        }
        g.wait_idle().unwrap();
        sub.destroy(&g);
        timer.destroy(&g);
        rb.destroy(&g);
        sb.destroy(&g);
        accel.free_now(&g, &mut alloc);
        gm.free_now(&g, &mut alloc);
    }
    g.wait_idle().unwrap();
    out.free(&g, &mut alloc);
    targets.free(&g, &mut alloc);
    mats.free(&g, &mut alloc);
    sky_pass.destroy(&g);
    sky.free(&g, &mut alloc);
    raster.destroy(&g);
    shade_pass.destroy(&g);
    up.destroy(&g, &mut alloc);
    tl.destroy(&g);
    let (errors, _) = g.validation_counts();
    assert_eq!(errors, 0);
    assert_eq!(alloc.destroy(&g), 0);
}

/// Data only: the light view as the viewer shows it (sun and sky; exposure: a white Lambertian surface
/// lit by the sun and the mean sky near 1, ACES fit, sRGB), street camera, 960x540, for looking at.
/// Written to `results/shade_3c_<hour>.ppm`.
#[test]
fn light_view_images() {
    let mut rig = Rig::new();
    let (w, h) = (960, 540);
    let s = rig.scene(Merge::Greedy, RegionSize::Chunk);
    let targets = Targets::new(&rig.g, &mut rig.alloc, vk::Extent2D { width: w, height: h }).unwrap();
    let out = ShadeTargets::new(&rig.g, &mut rig.alloc, w, h).unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results");
    std::fs::create_dir_all(&dir).unwrap();
    let (_, cam) = cameras(w, h).remove(0);
    let luts = SkyLuts::new(Atmosphere::default());
    for hour in [6.25, 8.0, 12.0, 17.75, 18.25] {
        let light = lighting(hour);
        let rt = rig.shade_frame(&s, &targets, &out, &cam, &light, DIRECT, ShadeFaults::default(), 0);
        let g = light.sun_at_ground;
        let y = 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2];
        let e = luts.ground_irradiance(light.sun_dir[1]);
        let sky = (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]) / std::f64::consts::PI;
        let exposure = 1.0 / (y / std::f64::consts::PI + sky + 1e-7);
        let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
        let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
        for px in &rt {
            for &c in &px[..3] {
                bytes.push((srgb(aces(c as f64 * exposure)) * 255.0 + 0.5) as u8);
            }
        }
        let path = dir.join(format!("shade_3c_{hour}.ppm"));
        std::fs::write(&path, bytes).unwrap();
        eprintln!("wrote {} (exposure {exposure:.2})", path.display());
    }
    rig.g.wait_idle().unwrap();
    out.free(&rig.g, &mut rig.alloc);
    targets.free(&rig.g, &mut rig.alloc);
    rig.free_scene(s);
    rig.finish();
}
