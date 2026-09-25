//! Phase 4A: emitters in the CPU reference (ADR-0005 Amendment 3). Criteria C2–C4 of the
//! [4A record](../../docs/changes/2026-09-25-phase4a-emitters.md), and the part 2 magnitudes
//! (M1–M2, `diagnostic_magnitudes`, ignored: about an hour on 4 threads).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use derived::{extract_world, Merge};
use light::emitters::{EmitterId, EmitterQuad, EmitterTable};
use light::reference::{albedos, render_lit, Accum, EmitterFaults, Lighting, Pinhole, Settings, RAY_OFFSET};
use light::sample::{add, cross, dot, normalize, scale, sub, V3};
use light::{Atmosphere, SunPath};
use world::{MaterialParams, MaterialRegistry, VoxelCoord, World};

/// Every emitter quad of `world`, grouped into cubic sources of `group` bricks per edge (1 = bricks;
/// 4 = chunk regions, as the GPU groups them), with local coordinates and per-source quad indices.
fn quads(world: &World, group: i32) -> Vec<EmitterQuad> {
    let edge = 8 * group;
    let mut by_source: BTreeMap<[i32; 3], Vec<(VoxelCoord, derived::Quad)>> = BTreeMap::new();
    for (k, _) in world.bricks() {
        let o = k.origin();
        let key = [o.x.div_euclid(edge), o.y.div_euclid(edge), o.z.div_euclid(edge)];
        if let Some(m) = extract_world(world, k, Merge::Greedy) {
            by_source.entry(key).or_default().extend(m.quads.iter().map(|&q| (o, q)));
        }
    }
    let mut out = Vec::new();
    for (key, qs) in by_source {
        let so = key.map(|c| c * edge);
        for (i, (o, q)) in qs.into_iter().enumerate() {
            let off = [o.x - so[0], o.y - so[1], o.z - so[2]];
            let a = q.axis();
            let (ua, va) = q.uv_axes();
            let l = |x: u8, ax: usize| (x as i32 + off[ax]) as u8;
            let id = EmitterId { key, quad: i as u32, snapshot: 1 };
            out.push(EmitterQuad::from_local(id, q.material, q.face, l(q.plane, a), l(q.u0, ua), l(q.v0, va), l(q.u1, ua), l(q.v1, va), so));
        }
    }
    out
}

fn table(world: &World, group: i32) -> EmitterTable {
    EmitterTable::build(world.materials(), 1, quads(world, group)).unwrap()
}

fn night() -> Lighting {
    Lighting::new(Atmosphere::default(), SunPath::default().direction(0.0))
}

fn dark(s: Settings) -> Settings {
    Settings { sun: false, sky: false, ..s }
}

/// Image mean of `acc / expect` and its standard error; per-pixel worst |z|.
fn ratio(acc: &Accum, expect: impl Fn(usize) -> f64, c: usize) -> (f64, f64, f64) {
    let n = acc.sum.len();
    let (mut m, mut v, mut zmax) = (0.0, 0.0, 0.0f64);
    for i in 0..n {
        let e = expect(i);
        let (mi, se) = (acc.mean(i)[c], acc.std_error(i)[c]);
        m += mi / e;
        v += (se / e).powi(2);
        zmax = zmax.max(((mi - e) / se.max(1e-300)).abs());
    }
    (m / n as f64, v.sqrt() / n as f64, zmax)
}

/// A 64 × 64 floor, and a 16 × 16 emissive panel at height 12 whose only exposed emissive faces are
/// its bottom faces (a non-emissive ring and cap cover the rest).
fn panel_world() -> (World, V3) {
    let mut r = MaterialRegistry::new();
    let grey = r.register("grey", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
    let le = [2.0f32, 1.5, 0.5];
    let panel = r.register("panel", MaterialParams { base_color: [0.5; 3], emissive: le }).unwrap();
    let mut w = World::new(r);
    let v = VoxelCoord::new;
    w.fill_box(v(0, 0, 0), v(64, 1, 64), Some(grey)).unwrap();
    w.fill_box(v(23, 12, 23), v(41, 14, 41), Some(grey)).unwrap();
    w.fill_box(v(24, 12, 24), v(40, 13, 40), Some(panel)).unwrap();
    (w, le.map(|x| x as f64))
}

/// Lambert's irradiance from a uniform Lambertian polygon of unit radiance at `x` (normal `n`).
fn polygon_irradiance(x: V3, n: V3, verts: &[V3]) -> f64 {
    let mut e = 0.0;
    for i in 0..verts.len() {
        let (a, b) = (normalize(sub(verts[i], x)), normalize(sub(verts[(i + 1) % verts.len()], x)));
        let g = dot(a, b).clamp(-1.0, 1.0).acos();
        e += g * dot(n, normalize(cross(a, b)));
    }
    (e / 2.0).abs()
}

/// C2: an emissive rectangle over a plane against the closed form; two planted faults are caught.
#[test]
fn emissive_rectangle_matches_lamberts_formula() {
    let (w, le) = panel_world();
    let al = albedos(w.materials()).unwrap();
    let em = table(&w, 1);
    assert_eq!(em.emitters.iter().map(|e| e.area).sum::<f64>(), 256.0, "only the panel's bottom faces emit");
    assert!(em.emitters.iter().all(|e| e.face == 2));
    let cam = Pinhole { eye: [32.0, 6.0, 32.0], forward: [0.0, -1.0, 0.0], right: [0.8, 0.0, 0.0], up: [0.0, 0.0, -0.8], width: 8, height: 8 };
    let verts = [[24.0, 12.0, 24.0], [40.0, 12.0, 24.0], [40.0, 12.0, 40.0], [24.0, 12.0, 40.0]];
    let geo: Vec<f64> = (0..64u32)
        .map(|i| {
            let d = cam.dir(i % 8, i / 8);
            let t = (1.0 - cam.eye[1]) / d[1];
            let mut p = add(cam.eye, scale(d, t));
            p[1] = 1.0 + RAY_OFFSET;
            0.5 / PI * polygon_irradiance(p, [0.0, 1.0, 0.0], &verts)
        })
        .collect();
    let s = dark(Settings { emitters_direct: true, max_bounces: 0, ..Settings::default() });
    let run = |area: bool, f: EmitterFaults| render_lit(&w, &al, &em, &night(), &cam, &Settings { emitter_area_sampling: area, emitter_faults: f, ..s }, 0, 1024, 11, 4);
    // Both strategies: by solid angle (the default) and the area control.
    for area in [false, true] {
        let acc = run(area, EmitterFaults::default());
        for (c, &lc) in le.iter().enumerate() {
            let (m, se, zmax) = ratio(&acc, |i| geo[i] * lc, c);
            eprintln!("C2 area {area} channel {c}: mean ratio {m:.5} ± {se:.5}, worst pixel |z| {zmax:.2}");
            assert!(zmax < 5.0, "pixel |z| {zmax}");
            assert!((m - 1.0).abs() < 3.0 * se && (m - 1.0).abs() < 0.005, "mean ratio {m} ± {se}");
        }
    }
    let faults = [
        (false, EmitterFaults { no_solid_angle: true, ..EmitterFaults::default() }),
        (true, EmitterFaults { no_solid_angle: true, ..EmitterFaults::default() }),
        (true, EmitterFaults { no_emitter_cosine: true, ..EmitterFaults::default() }),
    ];
    for (area, f) in faults {
        let (m, se, _) = ratio(&run(area, f), |i| geo[i] * le[1], 1);
        eprintln!("C2 fault (area {area}) {f:?}: mean ratio {m:.4} ± {se:.5}");
        assert!((m - 1.0).abs() > 10.0 * se, "fault {f:?} not caught: {m} ± {se}");
    }
}

/// C3: the emissive furnace, L_e Σ₀^{B+1} aᵏ; "emission counted twice" is caught.
#[test]
fn emissive_furnace_sums_the_bounces() {
    let mut r = MaterialRegistry::new();
    let wall = r.register("wall", MaterialParams { base_color: [0.5; 3], emissive: [1.0; 3] }).unwrap();
    let mut w = World::new(r);
    let v = VoxelCoord::new;
    w.fill_box(v(0, 0, 0), v(16, 16, 16), Some(wall)).unwrap();
    w.fill_box(v(1, 1, 1), v(15, 15, 15), None).unwrap();
    let al = albedos(w.materials()).unwrap();
    let em = table(&w, 1);
    let cam = Pinhole { eye: [8.0, 8.0, 8.0], forward: [0.0, 0.0, -1.0], right: [0.9, 0.0, 0.0], up: [0.0, 0.9, 0.0], width: 8, height: 8 };
    let b = 16;
    let expect: f64 = (0..=b + 1).map(|k| 0.5f64.powi(k)).sum();
    let s = dark(Settings { emission: true, emitters_direct: true, emitters_indirect: true, max_bounces: b as u32, ..Settings::default() });
    // At 256 samples with area sampling the image mean's standard error (0.9%) could not resolve the
    // 0.5% bound, and more samples did not shrink it (first runs, 4A record).
    let spp = 4096;
    let run = |f: EmitterFaults, spp: u32| render_lit(&w, &al, &em, &night(), &cam, &Settings { emitter_faults: f, ..s }, 0, spp, 12, 4);
    let (m, se, zmax) = ratio(&run(EmitterFaults::default(), spp), |_| expect, 0);
    eprintln!("C3: mean ratio {m:.5} ± {se:.5} (expect {expect:.6}), worst pixel |z| {zmax:.2}");
    // Standard error against sample count: ∝ 1/√N by solid angle; by area it does not fall (the
    // 1/d² term next to a wall edge has infinite variance; first runs, 4A record).
    for area in [false, true] {
        let ses: Vec<f64> = [1024, 4096, 16_384].iter().map(|&n| ratio(&render_lit(&w, &al, &em, &night(), &cam, &Settings { emitter_area_sampling: area, ..s }, 0, n, 13, 4), |_| expect, 0).1).collect();
        eprintln!("C3 standard error at 1k / 4k / 16k spp, area {area}: {ses:.5?}");
    }
    assert!((m - 1.0).abs() < 3.0 * se && (m - 1.0).abs() < 0.005, "mean ratio {m} ± {se}");
    let (m, se, _) = ratio(&run(EmitterFaults { double_emission: true, ..EmitterFaults::default() }, 256), |_| expect, 0);
    eprintln!("C3 fault double_emission: mean ratio {m:.4} ± {se:.5}");
    assert!((m - 1.0).abs() > 10.0 * se);
}

/// Without emitter terms, the table does not matter: the same numbers are drawn and the same image
/// comes out (C5's in-test half; the fingerprint is the other).
#[test]
fn emitters_off_ignores_the_table() {
    let (w, view) = world::scene::street_block();
    let al = albedos(w.materials()).unwrap();
    let em = table(&w, 4);
    assert!(!em.is_empty(), "the street registers lamps and signs as emissive");
    let eye = view.eye_m.map(|x| x / world::dims::VOXEL_SIZE_M);
    let cam = Pinhole { eye, forward: [0.8, -0.1, 0.6], right: [0.3, 0.0, -0.4], up: [0.0, 0.3, 0.0], width: 12, height: 8 };
    let light = Lighting::new(Atmosphere::default(), SunPath::default().direction(12.0));
    let s = Settings { max_bounces: 1, ..Settings::default() };
    let a = light::reference::render(&w, &al, &light, &cam, &s, 0, 2, 5, 2);
    let b = render_lit(&w, &al, &em, &light, &cam, &s, 0, 2, 5, 2);
    assert_eq!(a.sum, b.sum);
    assert_eq!(a.sum_sq, b.sum_sq);
}

/// C4: grouping does not change the table; an edit does; a stale table is refused.
#[test]
fn table_identity_follows_geometry_and_snapshots() {
    let (mut w, _) = world::scene::street_block();
    let (bricks, regions) = (table(&w, 1), table(&w, 4));
    assert_eq!(bricks.len(), regions.len());
    for (a, b) in bricks.emitters.iter().zip(&regions.emitters) {
        assert_eq!((a.face, a.p0, a.eu, a.ev, a.radiance, a.pdf, a.threshold, a.alias), (b.face, b.p0, b.eu, b.ev, b.radiance, b.pdf, b.threshold, b.alias));
    }
    assert_ne!(bricks.emitters[0].id, regions.emitters.last().unwrap().id);
    // Remove one lamp-head voxel (the first lamp: x = 64..68, y = 69..72, z = -10 + 10..15).
    let lamp = w.materials().id_of("lamp").unwrap();
    let v = w.occupied().find(|&(_, m)| m == lamp).map(|(v, _)| v).unwrap();
    w.set(v, None).unwrap();
    let edited = EmitterTable::build(w.materials(), 2, quads(&w, 4)).unwrap();
    assert_ne!(edited.emitters.iter().map(|e| (e.p0, e.eu, e.ev)).collect::<Vec<_>>(), regions.emitters.iter().map(|e| (e.p0, e.eu, e.ev)).collect::<Vec<_>>());
    assert!(regions.check(2).is_err(), "a table from snapshot 1 must be refused at snapshot 2");
    assert!(edited.check(2).is_ok());
}

/// The night dressing's light counts (the 4A sweep): emitter quads and total power per level.
#[test]
fn night_dressing_light_counts() {
    use world::scene::{street_night, Dressing};
    let mut last = 0;
    for d in Dressing::ALL {
        let (w, _) = street_night(d);
        let t = table(&w, 4);
        let by = |name: &str| {
            let id = w.materials().id_of(name);
            t.emitters.iter().filter(|e| Some(e.material) == id).count()
        };
        eprintln!(
            "dressing {:8}: {:5} emitter quads (lamp {}, sign {}, neon {}, window {}, bulb {}), total power {:.4}",
            d.name(),
            t.len(),
            by("lamp"),
            by("shop_sign"),
            by("neon_pink") + by("neon_cyan") + by("neon_green"),
            by("window_lit"),
            by("bulb"),
            t.total_power
        );
        assert!(t.len() > last, "{d:?} adds lights");
        last = t.len();
    }
}

/// Smoke: the night street with every term on (sun, sky, emission, emitters direct and indirect,
/// 2 bounces) renders finite, non-negative values, and the emitters light it.
#[test]
fn night_street_renders_finite() {
    use world::scene::{street_night, Dressing};
    let (w, view) = street_night(Dressing::Full);
    let al = albedos(w.materials()).unwrap();
    let em = table(&w, 4);
    let eye = view.eye_m.map(|x| x / world::dims::VOXEL_SIZE_M);
    let target = view.target_m.map(|x| x / world::dims::VOXEL_SIZE_M);
    let f = normalize(sub(target, eye));
    let r = normalize(cross(f, [0.0, 1.0, 0.0]));
    let u = cross(r, f);
    let cam = Pinhole { eye, forward: f, right: scale(r, 0.8), up: scale(u, 0.45), width: 32, height: 18 };
    let light = Lighting::new(Atmosphere::default(), SunPath::default().direction(21.0));
    let s = Settings { max_bounces: 2, emission: true, emitters_direct: true, emitters_indirect: true, ..Settings::default() };
    let lit = render_lit(&w, &al, &em, &light, &cam, &s, 0, 8, 3, 4);
    let off = light::reference::render(&w, &al, &light, &cam, &Settings { max_bounces: 2, ..Settings::default() }, 0, 8, 3, 4);
    assert!(lit.sum.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
    let mean = |a: &Accum| a.sum.iter().map(|p| p[1]).sum::<f64>() / (a.sum.len() as f64 * a.samples as f64);
    eprintln!("night street at 21 h: mean G with lights {:.3e}, without {:.3e}", mean(&lit), mean(&off));
    assert!(mean(&lit) > 100.0 * mean(&off), "the lights dominate the night street");
}

// ---- 4A part 2: the magnitudes (M1–M2), measurements with a fixed method. ----

/// Luminance sums of one pixel's paired samples: `a` at `max_bounces` 1, `b` at 8, `d = b − a`
/// (exactly the light after 2 or more bounces: the same random stream, so `a` is `b`'s prefix).
#[derive(Clone, Copy, Debug, Default)]
struct Paired {
    a: f64,
    aa: f64,
    b: f64,
    bb: f64,
    d: f64,
    dd: f64,
    db: f64,
}

/// A pinhole with the `gpu::raster::Camera::look_at` conventions.
fn look_at(eye: V3, target: V3, fov_deg: f64, w: u32, h: u32) -> Pinhole {
    let f = normalize(sub(target, eye));
    let r = normalize(cross(f, [0.0, 1.0, 0.0]));
    let u = cross(r, f);
    let ty = (fov_deg.to_radians() / 2.0).tan();
    Pinhole { eye, forward: f, right: scale(r, ty * w as f64 / h as f64), up: scale(u, ty), width: w, height: h }
}

/// The two reference cameras of `ref_light` (3A).
fn reference_cameras(w: u32, h: u32) -> [(&'static str, Pinhole); 2] {
    let (_, view) = world::scene::street_block();
    let vox = |m: V3| m.map(|x| x / world::dims::VOXEL_SIZE_M);
    [("street", look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h)), ("low", look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h))]
}

/// `f(x, y, acc)` for every pixel on `threads` threads (rows interleaved), accumulating into `px`.
fn each_pixel<T: Send>(px: &mut [T], w: u32, threads: usize, f: impl Fn(u32, u32, &mut T) + Sync) {
    let rows: Vec<&mut [T]> = px.chunks_mut(w as usize).collect();
    let mut by_thread: Vec<Vec<(u32, &mut [T])>> = (0..threads).map(|_| Vec::new()).collect();
    for (y, row) in rows.into_iter().enumerate() {
        by_thread[y % threads].push((y as u32, row));
    }
    std::thread::scope(|s| {
        for rows in by_thread {
            let f = &f;
            s.spawn(move || {
                for (y, row) in rows {
                    for (x, p) in row.iter_mut().enumerate() {
                        f(x as u32, y, p);
                    }
                }
            });
        }
    });
}

/// The `q` quantile (nearest rank) of `v`; NaN when empty.
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(((v.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(f64::NAN)
}

/// σ/μ of one sample per pixel from `n` samples' sums; `None` where μ = 0.
fn one_sample_noise(sum: f64, sum_sq: f64, n: f64) -> Option<f64> {
    let mu = sum / n;
    (mu > 0.0).then(|| ((sum_sq - sum * sum / n).max(0.0) / (n - 1.0)).sqrt() / mu)
}

/// M1 and M2 (4A part 2): the share of bounces ≥ 2 and the one-sample noise on `street_night`
/// (Full) at the M4 times with the lights by the rule, and the direct one-sample noise per dressing
/// at night. Measurements, not pass/fail; the method is frozen in the 4A record. CPU only.
/// Run: `cargo test --release -j 2 -p light --test emitters diagnostic_magnitudes -- --ignored --nocapture`.
#[test]
#[ignore]
fn diagnostic_magnitudes() {
    use light::emitters::{lights_on, luminance};
    use light::exposure::DARK_FRACTION;
    use light::reference::sample_pixel_lit;
    use light::sun::{elevation_deg, NIGHT_TIMES};
    use world::scene::{street_night, Dressing};

    const W: u32 = 128;
    const H: u32 = 72;
    const SEED: u32 = 0x4A2;
    /// M1's precision target (percentage points of the image) and the sample schedule.
    const TARGET_PP: f64 = 0.5;
    const FIRST: u32 = 1024;
    const MAX_SPP: u32 = 16_384;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (world, _) = street_night(Dressing::Full);
    let al = albedos(world.materials()).unwrap();
    let em = table(&world, 4);
    let n_px = (W * H) as usize;
    eprintln!("magnitudes: street_night (full), {} emitters, {W}x{H}, {threads} threads, seed {SEED:#x}", em.len());

    for (cname, cam) in reference_cameras(W, H) {
        for (tname, hour) in NIGHT_TIMES {
            let sun = SunPath::default().direction(hour);
            let light = Lighting::new(Atmosphere::default(), sun);
            let on = lights_on(sun);
            // Emission at the primary hit is left out: bounces do not change it.
            let s1 = Settings { max_bounces: 1, emitters_direct: on, emitters_indirect: on, ..Settings::default() };
            let s8 = Settings { max_bounces: 8, ..s1 };
            let t = std::time::Instant::now();
            let mut px = vec![Paired::default(); n_px];
            let (mut done, mut batch) = (0u32, FIRST);
            loop {
                let first = done;
                each_pixel(&mut px, W, threads, |x, y, p| {
                    for f in first..first + batch {
                        let a = luminance(sample_pixel_lit(&world, &al, &em, &light, &cam, &s1, x, y, f, SEED));
                        let b = luminance(sample_pixel_lit(&world, &al, &em, &light, &cam, &s8, x, y, f, SEED));
                        let d = b - a;
                        *p = Paired { a: p.a + a, aa: p.aa + a * a, b: p.b + b, bb: p.bb + b * b, d: p.d + d, dd: p.dd + d * d, db: p.db + d * b };
                    }
                });
                done += batch;
                let n = done as f64;
                let (sa, sb, sd) = px.iter().fold((0.0, 0.0, 0.0), |(x, y, z), p| (x + p.a, y + p.b, z + p.d));
                let share = sd / sb;
                // Ratio estimator: Var(Σd − R Σb) over independent pixels, from each pixel's samples.
                let var: f64 = px
                    .iter()
                    .map(|p| {
                        let (sz, szz) = (p.d - share * p.b, p.dd - 2.0 * share * p.db + share * share * p.bb);
                        n * (szz - sz * sz / n).max(0.0) / (n - 1.0)
                    })
                    .sum();
                let se = var.sqrt() / sb;
                if 100.0 * se <= TARGET_PP || done >= MAX_SPP {
                    let mean_b = sb / (n_px as f64 * n);
                    let lit: Vec<&Paired> = px.iter().filter(|p| p.b / n >= DARK_FRACTION * mean_b && p.b > 0.0).collect();
                    let shares: Vec<f64> = lit.iter().map(|p| p.d / p.b).collect();
                    let noise: Vec<f64> = px.iter().filter_map(|p| one_sample_noise(p.a, p.aa, n)).collect();
                    let noise_lit: Vec<f64> = lit.iter().filter_map(|p| one_sample_noise(p.a, p.aa, n)).collect();
                    // Emission at the primary hit, deterministic (one sample), for context only.
                    let mut e = vec![0.0f64; n_px];
                    if on {
                        let es = Settings { sun: false, sky: false, max_bounces: 0, emission: true, ..Settings::default() };
                        each_pixel(&mut e, W, threads, |x, y, v| *v = luminance(sample_pixel_lit(&world, &al, &em, &light, &cam, &es, x, y, 0, SEED)));
                    }
                    let mean_e = e.iter().sum::<f64>() / n_px as f64;
                    eprintln!(
                        "M1 {cname} {tname} ({:+.2}°, lights {}): bounces >= 2 share {:.2}% ± {:.2} pp (image L1 {:.4e}, L8 {:.4e}); per lit pixel ({} of {n_px}) median {:.2}%, p90 {:.2}%; {done} spp, {:.0} s",
                        elevation_deg(sun),
                        if on { "on" } else { "off" },
                        100.0 * share,
                        100.0 * se,
                        sa / (n_px as f64 * n),
                        mean_b,
                        lit.len(),
                        100.0 * quantile(&shares, 0.5),
                        100.0 * quantile(&shares, 0.9),
                        t.elapsed().as_secs_f64()
                    );
                    eprintln!(
                        "M2a {cname} {tname}: one-sample σ/μ of the one-bounce estimator: median {:.2}, p90 {:.2} over {} pixels with μ > 0 (lit pixels: median {:.2}, p90 {:.2}); emission at the primary hit {:.1}% of the full image mean",
                        quantile(&noise, 0.5),
                        quantile(&noise, 0.9),
                        noise.len(),
                        quantile(&noise_lit, 0.5),
                        quantile(&noise_lit, 0.9),
                        100.0 * mean_e / (mean_e + mean_b)
                    );
                    break;
                }
                batch = done;
            }
        }
    }

    // M2b: emitter light alone at the primary hit, one emitter sample, at night, per dressing.
    const SPP_B: u32 = 256;
    let light = Lighting::new(Atmosphere::default(), SunPath::default().direction(21.0));
    let s = Settings { sun: false, sky: false, max_bounces: 0, emitters_direct: true, ..Settings::default() };
    for d in Dressing::ALL {
        let (world, _) = street_night(d);
        let al = albedos(world.materials()).unwrap();
        let em = table(&world, 4);
        for (cname, cam) in reference_cameras(W, H) {
            let t = std::time::Instant::now();
            let mut px = vec![(0.0f64, 0.0f64); n_px];
            each_pixel(&mut px, W, threads, |x, y, p| {
                for f in 0..SPP_B {
                    let v = luminance(sample_pixel_lit(&world, &al, &em, &light, &cam, &s, x, y, f, SEED));
                    *p = (p.0 + v, p.1 + v * v);
                }
            });
            let noise: Vec<f64> = px.iter().filter_map(|p| one_sample_noise(p.0, p.1, SPP_B as f64)).collect();
            let mean = px.iter().map(|p| p.0).sum::<f64>() / (n_px as f64 * SPP_B as f64);
            eprintln!(
                "M2b {} {cname}: {} emitters; direct emitter light at the primary hit, one sample: σ/μ median {:.2}, p90 {:.2} over {} pixels with μ > 0; image mean {mean:.4e}; {SPP_B} spp, {:.0} s",
                d.name(),
                em.len(),
                quantile(&noise, 0.5),
                quantile(&noise, 0.9),
                noise.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }
}

/// 4B, before the laptop run: the noise floor of G4's Q1 (the shown mean luminance within ±2% of
/// the reference). One real-time frame is one sample per pixel of the one-bounce estimator with
/// emitters at both vertices (G4's reference transport); the relative 1-sigma of its image mean is
/// sqrt(Σ Var_i) / Σ μ_i, and at `a` frames of history it is that over sqrt(a). Measured at a small
/// size and scaled to 1920x1080 by sqrt(pixels), since the per-pixel statistics do not depend on
/// the resolution. Night, both cameras, Full. Measurement, not pass/fail. CPU only.
/// Run: `cargo test --release -j 2 -p light --test emitters diagnostic_g4_noise_floor -- --ignored --nocapture`.
#[test]
#[ignore]
fn diagnostic_g4_noise_floor() {
    use light::emitters::luminance;
    use light::reference::sample_pixel_lit;
    use world::scene::{street_night, Dressing};

    const W: u32 = 192;
    const H: u32 = 108;
    const SPP: u32 = 2048;
    const SEED: u32 = 0x4B1;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (world, _) = street_night(Dressing::Full);
    let al = albedos(world.materials()).unwrap();
    let em = table(&world, 4);
    let light = Lighting::new(Atmosphere::default(), SunPath::default().direction(21.0));
    let s = Settings { max_bounces: 1, emitters_direct: true, emitters_indirect: true, ..Settings::default() };
    let scale_to_1080p = ((W * H) as f64 / (1920.0 * 1080.0)).sqrt();
    for (cname, cam) in reference_cameras(W, H) {
        let t = std::time::Instant::now();
        let mut px = vec![(0.0f64, 0.0f64); (W * H) as usize];
        each_pixel(&mut px, W, threads, |x, y, p| {
            for f in 0..SPP {
                let v = luminance(sample_pixel_lit(&world, &al, &em, &light, &cam, &s, x, y, f, SEED));
                *p = (p.0 + v, p.1 + v * v);
            }
        });
        let n = SPP as f64;
        let (mut mu, mut var) = (0.0, 0.0);
        let mut contrib: Vec<f64> = Vec::new();
        for &(sum, sq) in &px {
            let v = (sq - sum * sum / n).max(0.0) / (n - 1.0);
            mu += sum / n;
            var += v;
            contrib.push(v);
        }
        contrib.sort_by(|a, b| b.total_cmp(a));
        let top = |k: usize| 100.0 * contrib.iter().take(k).sum::<f64>() / var.max(1e-300);
        let floor = var.sqrt() / mu.max(1e-300) * scale_to_1080p;
        eprintln!(
            "G4 floor {cname} night: one frame's mean luminance 1-sigma at 1080p {:.2}% (at {W}x{H}: {:.2}%); ages 1 / 4 / 16 / 64: {:.2}% / {:.2}% / {:.2}% / {:.2}%; the noisiest 1% of pixels carry {:.0}% of the variance; {SPP} spp, {:.0} s",
            100.0 * floor,
            100.0 * floor / scale_to_1080p,
            100.0 * floor,
            100.0 * floor / 2.0,
            100.0 * floor / 4.0,
            100.0 * floor / 8.0,
            top(contrib.len() / 100),
            t.elapsed().as_secs_f64()
        );
    }
}
