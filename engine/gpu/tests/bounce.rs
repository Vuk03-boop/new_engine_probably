//! Phase 3F GPU tests: one bounce of diffuse light in `gpu::shade` against the 3A reference with
//! `max_bounces` 1 (`gpu::reference`). Criteria are declared in
//! `docs/changes/2026-09-24-phase3f-bounce.md`; `corrected_one_bounce_converges_to_the_reference`
//! re-runs B2 with the sky correction (S-020, data). Like the other GPU tests they need the Vulkan SDK
//! and an RT GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test bounce -- --test-threads=1 --nocapture`.

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
use gpu::{Gpu, Timeline};
use light::reference::{albedos, Accum, Lighting, Settings};
use light::sky::SkyLuts;
use light::sky_ref::SkyReference;
use light::sun::REFERENCE_TIMES;
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::scene;

const NEAR: f64 = 0.1;
const SEED: u32 = 0x3F;
/// Sun and sky with the bounce (the default).
const BOUNCE: ShadeSettings = ShadeSettings { sun: true, sky: true, point_sun: false, uniform_sky: false, bounce: true };

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    assert!(g.ray_tracing(), "3F needs the ray-tracing device (P-001)");
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

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    mats: Option<RefMaterials>,
    sky: Option<SkyTables>,
    sky_pass: Option<SkyView>,
    gm: Option<GpuMeshes>,
    accel: Option<Accel>,
    raster: Raster,
    shade: Shade,
    reference: Reference,
}

impl Rig {
    /// `corrected`: the sky-view table corrected from the baked reference (S-020).
    fn new(corrected: bool) -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let mut tl = Timeline::new(&g).unwrap();
        let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let (world, _) = scene::street_block();
        let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedos(world.materials()).unwrap()).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        let mut luts = SkyLuts::new(Atmosphere::default());
        if corrected {
            let r = SkyReference::load_default(&luts.atmosphere).expect("the baked sky reference (run the sky_bake tool)");
            luts.apply_reference(&r).unwrap();
        }
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
        Rig { g, alloc, tl, up, mats: Some(mats), sky: Some(sky), sky_pass: Some(sky_pass), gm: Some(gm), accel: Some(accel), raster, shade, reference }
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

    /// The shade pass's frames `frames`, each read back and handed to `each` (the sky table must be
    /// current). One set of targets for all frames.
    fn shade_frames(&mut self, cam: &Camera, light: &Lighting, set: ShadeSettings, frames: std::ops::Range<u32>, mut each: impl FnMut(u32, &[[f32; 4]])) {
        let (w, h) = (cam.width, cam.height);
        let targets = Targets::new(&self.g, &mut self.alloc, vk::Extent2D { width: w, height: h }).unwrap();
        let out = ShadeTargets::new(&self.g, &mut self.alloc, w, h).unwrap();
        let gm = self.gm.as_ref().unwrap();
        let rb = self.raster.bind(&self.g, gm).unwrap();
        let sb = self.shade.bind(&self.g, &targets, self.accel.as_ref().unwrap(), self.mats.as_ref().unwrap(), &out, &self.sky.as_ref().unwrap().view).unwrap();
        let mut sub = Submitter::new(&self.g).unwrap();
        for frame in frames {
            let g = &self.g;
            let cmd = sub.begin(g, &self.tl).unwrap();
            self.raster.record(g, cmd, &targets, self.gm.as_ref().unwrap(), &rb, cam, Faults::default());
            targets_to_read(g, cmd, &targets);
            self.shade.record(g, cmd, &sb, &out, &shade::Params::new(cam, light, set, ShadeFaults::default(), frame, SEED));
            let v = sub.submit(g, &mut self.tl, cmd, &[]).unwrap();
            self.tl.wait(g, v, u64::MAX).unwrap();
            let r = out.read(&self.g, &mut self.alloc, &mut self.tl).unwrap();
            each(frame, &r);
        }
        let g = &self.g;
        g.wait_idle().unwrap();
        sub.destroy(g);
        rb.destroy(g);
        sb.destroy(g);
        out.free(&self.g, &mut self.alloc);
        targets.free(&self.g, &mut self.alloc);
    }

    fn shade_frame(&mut self, cam: &Camera, light: &Lighting, set: ShadeSettings, frame: u32) -> Vec<[f32; 4]> {
        let mut r = Vec::new();
        self.shade_frames(cam, light, set, frame..frame + 1, |_, x| r = x.to_vec());
        r
    }

    /// The mean of frames `0..frames` per pixel, with the bad-id count.
    fn shade_mean(&mut self, cam: &Camera, light: &Lighting, set: ShadeSettings, frames: u32) -> (Vec<[f64; 3]>, usize) {
        let mut acc = vec![[0.0f64; 3]; (cam.width * cam.height) as usize];
        let mut bad = 0;
        self.shade_frames(cam, light, set, 0..frames, |_, x| {
            for (a, p) in acc.iter_mut().zip(x) {
                bad += usize::from(p[3] != 0.0);
                for c in 0..3 {
                    a[c] += p[c] as f64 / frames as f64;
                }
            }
        });
        (acc, bad)
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

    /// Surface pixels: the primary ray hits (a uniform sky of 1 shows exactly 1 where it misses).
    fn surface_mask(&mut self, cam: &Camera) -> Vec<bool> {
        let u = self.reference_image(cam, &lighting(12.0), &Settings { uniform_sky: true, max_bounces: 0, ..Settings::default() }, 0, 1);
        u.sum.iter().map(|v| *v != [1.0; 3]).collect()
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

/// Per-pixel agreement of one shade frame with one reference sample.
struct Agreement {
    mismatch: usize,
    bad: usize,
}

fn agreement(rt: &[[f32; 4]], rf: &Accum) -> Agreement {
    let mut a = Agreement { mismatch: 0, bad: 0 };
    for (x, y) in rt.iter().zip(&rf.sum) {
        a.bad += usize::from(x[3] != 0.0);
        let same = (0..3).all(|k| if y[k] == 0.0 { x[k] == 0.0 } else { (x[k] as f64 / y[k] - 1.0).abs() <= 1e-5 });
        a.mismatch += usize::from(!same);
    }
    a
}

/// B1: frame f of the shade pass with the bounce equals the reference's one-bounce sample f per
/// pixel (uniform sky with the sun off; point and disk sun with the sky off). Controls: the bounce
/// off, and a different frame.
#[test]
fn one_bounce_equals_the_reference_sample_per_pixel() {
    let mut rig = Rig::new(false);
    let (w, h) = (480, 270);
    let n = (w * h) as usize;
    let frame = 29;
    let mut failed = Vec::new();
    // Per arm kind: (arms, arms failed, total mismatches).
    let mut tally: std::collections::BTreeMap<&str, (usize, usize, usize)> = Default::default();
    let none = ShadeSettings { sun: false, sky: false, point_sun: false, uniform_sky: false, bounce: true };
    let mut arms: Vec<(String, f64, ShadeSettings, Settings)> = vec![("uniform sky".into(), 12.0, ShadeSettings { uniform_sky: true, ..none }, Settings { uniform_sky: true, max_bounces: 1, ..Settings::default() })];
    for hour in [8.0, 12.0, 17.75] {
        for point_sun in [true, false] {
            let label = format!("{} sun {hour} h", if point_sun { "point" } else { "disk" });
            arms.push((label, hour, ShadeSettings { sun: true, point_sun, ..none }, Settings { sky: false, point_sun, max_bounces: 1, ..Settings::default() }));
        }
    }
    for (cname, cam) in street_cameras(w, h) {
        for (label, hour, set, rs) in &arms {
            let light = lighting(*hour);
            let rf = rig.reference_image(&cam, &light, rs, frame, 1);
            let rt = rig.shade_frame(&cam, &light, *set, frame);
            let direct = rig.shade_frame(&cam, &light, ShadeSettings { bounce: false, ..*set }, frame);
            let changed = rt.iter().zip(&direct).filter(|(a, b)| a[..3] != b[..3]).count();
            let a = agreement(&rt, &rf);
            let ctl = agreement(&direct, &rf);
            let pass = a.mismatch * 1000 <= n && a.bad == 0 && changed >= 1000;
            eprintln!(
                "{cname} {label}: {} mismatched ({:.3}%), bad {}, {changed} px changed by the bounce | control bounce off: {} mismatched ({:.2}%)",
                a.mismatch,
                100.0 * a.mismatch as f64 / n as f64,
                a.bad,
                ctl.mismatch,
                100.0 * ctl.mismatch as f64 / n as f64
            );
            for (kind, m, ok) in [("ok", a.mismatch, pass), ("bounce off", ctl.mismatch, ctl.mismatch * 1000 <= n)] {
                let t = tally.entry(kind).or_default();
                t.0 += 1;
                t.1 += usize::from(!ok);
                t.2 += m;
            }
            if !pass {
                failed.push(format!("{cname}/{label}"));
            }
        }
    }
    let ok_total = tally["ok"].2;
    for (k, (arms, failing, total)) in &tally {
        eprintln!("tally {k}: {failing} of {arms} arms fail, {total} mismatched pixels in total");
        if *k != "ok" && (failing * 2 < *arms || *total < 10 * ok_total.max(1)) {
            failed.push(format!("control {k} not caught: {failing} of {arms} arms, {total} px (ok arms {ok_total} px)"));
        }
    }
    let (_, cam) = street_cameras(w, h).remove(0);
    let light = lighting(8.0);
    let rf = rig.reference_image(&cam, &light, &arms[1].3, frame, 1);
    let rt = rig.shade_frame(&cam, &light, arms[1].2, frame + 1);
    let c = agreement(&rt, &rf).mismatch;
    eprintln!("control: frame {} against reference frame {frame}: {c} mismatched", frame + 1);
    if c == 0 {
        failed.push("frame control: different frames agree".into());
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// A pixel selection.
type Mask<'a> = Box<dyn Fn(usize) -> bool + 'a>;

/// Channel means over the pixels where `keep` holds.
fn masked_mean(n: usize, keep: &dyn Fn(usize) -> bool, v: &dyn Fn(usize) -> [f64; 3]) -> [f64; 3] {
    let mut s = [0.0; 3];
    let mut k = 0.0f64;
    for i in (0..n).filter(|&i| keep(i)) {
        let x = v(i);
        for c in 0..3 {
            s[c] += x[c];
        }
        k += 1.0;
    }
    s.map(|x| x / k.max(1.0))
}

fn rel(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|c| a[c] / b[c] - 1.0)
}

fn fmt3(v: [f64; 3]) -> String {
    format!("{:+.4} {:+.4} {:+.4}", v[0], v[1], v[2])
}

/// B2 (morning and midday; the other times as data) and B4 (data): the shade pass with the bounce,
/// averaged over 256 frames, against the reference with one bounce, over surface pixels; the
/// indirect component on both sides from the same streams; and the 1-sample noise with and without
/// the bounce.
#[test]
fn one_bounce_converges_to_the_reference() {
    converges_to_the_reference(false);
}

/// S-020 K6, data: B2 with the sky correction (no B4). Morning and midday are still judged.
#[test]
fn corrected_one_bounce_converges_to_the_reference() {
    converges_to_the_reference(true);
}

fn converges_to_the_reference(corrected: bool) {
    let mut rig = Rig::new(corrected);
    let (w, h) = (160, 90);
    let n = (w * h) as usize;
    let frames = 256u32;
    let mut failed = Vec::new();
    for (name, hour) in REFERENCE_TIMES {
        let light = lighting(hour);
        rig.update_sky(light.sun_dir);
        let judged = name == "morning" || name == "midday";
        for (cname, cam) in street_cameras(w, h).into_iter().take(2) {
            let m = rig.surface_mask(&cam);
            let (b, bad_b) = rig.shade_mean(&cam, &light, BOUNCE, frames);
            let (d, bad_d) = rig.shade_mean(&cam, &light, ShadeSettings { bounce: false, ..BOUNCE }, frames);
            let spp = if name == "twilight" { 16_384 } else { 4096 };
            let r1 = rig.reference_image(&cam, &light, &Settings { max_bounces: 1, ..Settings::default() }, 1_000_000, spp);
            let r0 = rig.reference_image(&cam, &light, &Settings { max_bounces: 0, ..Settings::default() }, 1_000_000, spp);
            let all = |i: usize| m[i];
            let surf = |i: usize| r1.mean(i) != r0.mean(i) || b[i] != d[i];
            let img = rel(masked_mean(n, &all, &|i| b[i]), masked_mean(n, &all, &|i| r1.mean(i)));
            let ind_rt = masked_mean(n, &all, &|i| [0, 1, 2].map(|c| b[i][c] - d[i][c]));
            let ind_rf = masked_mean(n, &all, &|i| [0, 1, 2].map(|c| r1.mean(i)[c] - r0.mean(i)[c]));
            let ind = rel(ind_rt, ind_rf);
            let share = lum(ind_rf) / lum(masked_mean(n, &all, &|i| r1.mean(i)));
            let pass = bad_b + bad_d == 0 && img.iter().chain(&ind).all(|r| r.abs() <= 0.05);
            eprintln!(
                "{name} {cname} (sky corrected {corrected}): image {} | indirect {} (bounce share of the reference {:.1}%, {} px with indirect) bad {} {}",
                fmt3(img),
                fmt3(ind),
                100.0 * share,
                (0..n).filter(|&i| surf(i)).count(),
                bad_b + bad_d,
                if judged { if pass { "pass" } else { "FAIL" } } else { "(data)" }
            );
            if judged && !pass {
                failed.push(format!("{name}/{cname}"));
            }
        }
        // B4: the 1-sample noise against the 256-frame mean, street camera.
        if corrected {
            continue;
        }
        let (_, cam) = street_cameras(w, h).remove(0);
        for set in [ShadeSettings { bounce: false, ..BOUNCE }, BOUNCE] {
            let (m, _) = rig.shade_mean(&cam, &light, set, frames);
            let (mut num, mut k) = (0.0f64, 0.0f64);
            let mean_l = m.iter().map(|x| lum(*x)).sum::<f64>() / n as f64;
            let eps = (0.1 * mean_l).powi(2);
            rig.shade_frames(&cam, &light, set, 5000..5008, |_, x| {
                for (p, t) in x.iter().zip(&m) {
                    let (a, b) = (lum([p[0] as f64, p[1] as f64, p[2] as f64]), lum(*t));
                    if b > 0.0 {
                        num += (a - b).powi(2) / (b * b + eps);
                        k += 1.0;
                    }
                }
            });
            eprintln!("  noise {name} street {}: 1-sample relMSE {:.3}, mean luminance {:.4e}", if set.bounce { "bounce" } else { "direct" }, num / k.max(1.0), mean_l);
        }
    }
    rig.finish();
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// B3, data only: the energy of bounces >= 2 that 3F does not add: the reference with 8 bounces
/// against 1 (and 0), per camera and time, for the whole image and for pixels not directly sunlit
/// (the point sun's direct term is 0).
#[test]
fn missing_energy_of_higher_bounces() {
    let mut rig = Rig::new(false);
    let (w, h) = (160, 90);
    let n = (w * h) as usize;
    for (name, hour) in REFERENCE_TIMES {
        let light = lighting(hour);
        for (cname, cam) in street_cameras(w, h).into_iter().take(2) {
            let spp = if name == "twilight" { 8192 } else { 2048 };
            let m = rig.surface_mask(&cam);
            let r = [0u32, 1, 8].map(|b| rig.reference_image(&cam, &light, &Settings { max_bounces: b, ..Settings::default() }, 2_000_000, spp));
            let sun = rig.reference_image(&cam, &light, &Settings { sky: false, point_sun: true, max_bounces: 0, ..Settings::default() }, 0, 1);
            let rows: [(&str, Mask); 2] = [("surfaces", Box::new(|i| m[i])), ("not sunlit", Box::new(|i| m[i] && sun.mean(i)[1] == 0.0))];
            for (label, keep) in rows {
                let l = [0, 1, 2].map(|k| lum(masked_mean(n, &keep, &|i| r[k].mean(i))));
                eprintln!(
                    "{name} {cname} {label}: luminance 0 bounces {:.4e}, 1 bounce {:.4e}, 8 bounces {:.4e}; 1 bounce reaches {:.1}% (direct {:.1}%), missing {:.1}%",
                    l[0],
                    l[1],
                    l[2],
                    100.0 * l[1] / l[2],
                    100.0 * l[0] / l[2],
                    100.0 * (1.0 - l[1] / l[2])
                );
            }
        }
    }
    rig.finish();
}

/// Data only: the street camera at 8 h and 17.75 h, 256 frames averaged, without and with the
/// bounce, tone-mapped as the viewer's light view (white Lambertian surface under sun and mean sky
/// near 1, ACES fit, sRGB). Written to `results/bounce_3f_<hour>_<direct|bounce>.ppm`.
#[test]
#[ignore]
fn bounce_images() {
    let mut rig = Rig::new(false);
    let (w, h) = (480, 270);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results");
    std::fs::create_dir_all(&dir).unwrap();
    let (_, cam) = street_cameras(w, h).remove(0);
    let luts = SkyLuts::new(Atmosphere::default());
    for hour in [8.0, 17.75] {
        let light = lighting(hour);
        rig.update_sky(light.sun_dir);
        let g = light.sun_at_ground;
        let e = luts.ground_irradiance(light.sun_dir[1]);
        let exposure = 1.0 / (lum(g) / std::f64::consts::PI + lum(e) / std::f64::consts::PI + 1e-7);
        for (label, set) in [("direct", ShadeSettings { bounce: false, ..BOUNCE }), ("bounce", BOUNCE)] {
            let (m, _) = rig.shade_mean(&cam, &light, set, 256);
            let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
            let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
            let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
            for px in &m {
                for &c in px {
                    bytes.push((srgb(aces(c * exposure)) * 255.0 + 0.5) as u8);
                }
            }
            let path = dir.join(format!("bounce_3f_{hour}_{label}.ppm"));
            std::fs::write(&path, bytes).unwrap();
            eprintln!("wrote {}", path.display());
        }
    }
    rig.finish();
}
