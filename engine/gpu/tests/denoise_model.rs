//! A CPU mirror of the filter (`shaders/denoise.slang`, every mode and flag of `Denoise::record`) fed
//! with CPU frames of the viewer's transport, to screen filter settings without a GPU (the G4 filter
//! record, `docs/changes/2026-09-26-4b-filter-energy.md`). Device-free; ignored; no criterion.
//!
//! The frames are the CPU reference's samples with the real-time streams' seed: 4B's G1 showed that
//! frame f of the shade pass equals the reference's sample f per pixel, so the input is the viewer's
//! own radiance (the guides come from a CPU primary ray instead of the raster, so a few edge pixels
//! differ). A still camera from a fresh history is the exact running mean of frames 1..age with
//! its luminance second moment (the temporal pass below `max_age` 64), kept in f32 like the GPU.
//! The filter arithmetic is f32 like the shader. This is a model: the GPU tests judge.
//!
//! Frames and the reference are cached in `NE_MODEL_DIR` (default: the system temp dir), so sweeps
//! only rerun the filter. Size: `NE_MODEL_SIZE=WxH` (default 960x540); reference samples
//! `NE_MODEL_REF_SPP` (default 256, another seed).
//! Run: `cargo test --release -j 2 -p gpu --test denoise_model -- --ignored --nocapture`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use derived::{Config, Merge, Pipeline};
use gpu::denoise::{DenoiseSettings, Weights};
use gpu::layout::{build_regions, RegionSize};
use light::emitters::{luminance, EmitterTable};
use light::reference::{albedos, sample_pixel_lit, Lighting, Pinhole, Settings};
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::reference::{trace, Ray};
use world::scene::{street_night, Dressing};
use world::{BrickKey, World};

/// The real-time streams' seed (`gpu/tests/lights.rs`) and the reference's.
const SEED: u32 = 0x4B;
const REF_SEED: u32 = 0x4C;
const FACE_NONE: u32 = 7;
const AGES: [u32; 5] = [1, 4, 8, 16, 64];
/// Bump when the frames' transport, cameras or the cache layout change.
const CACHE_VERSION: u32 = 1;

fn size() -> (u32, u32) {
    let s = std::env::var("NE_MODEL_SIZE").unwrap_or_else(|_| "960x540".into());
    let (w, h) = s.split_once('x').expect("NE_MODEL_SIZE=WxH");
    (w.parse().unwrap(), h.parse().unwrap())
}

fn ref_spp() -> u32 {
    std::env::var("NE_MODEL_REF_SPP").ok().map_or(256, |s| s.parse().unwrap())
}

fn cache_dir() -> PathBuf {
    std::env::var_os("NE_MODEL_DIR").map_or_else(|| std::env::temp_dir().join("ne_denoise_model"), PathBuf::from)
}

fn look_at(eye: [f64; 3], target: [f64; 3], fov_deg: f64, w: u32, h: u32) -> Pinhole {
    use light::sample::{cross, normalize, scale, sub};
    let f = normalize(sub(target, eye));
    let r = normalize(cross(f, [0.0, 1.0, 0.0]));
    let u = cross(r, f);
    let ty = (fov_deg.to_radians() / 2.0).tan();
    Pinhole { eye, forward: f, right: scale(r, ty * w as f64 / h as f64), up: scale(u, ty), width: w, height: h }
}

/// 4A's cameras (`gpu/tests/lights.rs`): the street view and the 3A low camera.
fn cameras(w: u32, h: u32) -> Vec<(&'static str, Pinhole)> {
    let (_, view) = street_night(Dressing::Lamps);
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![("street", look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h)), ("low", look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h))]
}

/// The GPU's emitter table for the world (chunk regions through the 1C pipeline, as `GpuScene`).
fn gpu_table(world: &World) -> EmitterTable {
    let mut p = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
    p.mark_all(world);
    loop {
        let jobs = p.dispatch(world);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            p.complete(world, j.run());
        }
    }
    p.try_publish(world);
    let t = p.acquire();
    let keys: Vec<BrickKey> = p.current().keys().collect();
    let meshes: Vec<_> = keys.iter().map(|&k| (k, p.read(&t, k).unwrap().unwrap().clone())).collect();
    p.release(t).unwrap();
    let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk);
    gpu::emitters::table(&rs, world.materials(), p.current().id.raw()).unwrap()
}

/// `f(i, x, y)` for every pixel on all threads (rows interleaved).
fn par_pixels<T: Send + Default + Clone>(w: u32, h: u32, f: impl Fn(u32, u32) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut out = vec![T::default(); (w * h) as usize];
    let mut by_thread: Vec<Vec<(u32, &mut [T])>> = (0..threads).map(|_| Vec::new()).collect();
    for (y, row) in out.chunks_mut(w as usize).enumerate() {
        by_thread[y % threads].push((y as u32, row));
    }
    std::thread::scope(|s| {
        for rows in by_thread {
            let f = &f;
            s.spawn(move || {
                for (y, row) in rows {
                    for (x, p) in row.iter_mut().enumerate() {
                        *p = f(x as u32, y);
                    }
                }
            });
        }
    });
    out
}

/// One camera's input: the guides, the history at `AGES` and the reference.
struct Scene {
    w: u32,
    h: u32,
    /// (material << 3 | face, plane); face `FACE_NONE` for the sky.
    keys: Vec<[u32; 2]>,
    /// Per age: (the history's colour, its luminance second moment).
    hist: BTreeMap<u32, Vec<[f32; 4]>>,
    /// Reference mean and squared standard error per channel.
    ref_mean: Vec<[f64; 3]>,
    ref_se2: Vec<[f64; 3]>,
}

fn write_f32(path: &std::path::Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn read_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    Some(b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect())
}

#[derive(Clone, Default)]
struct Px {
    key: [u32; 2],
    hist: Vec<[f32; 4]>,
    ref_sum: [f64; 3],
    ref_sq: [f64; 3],
}

#[allow(clippy::too_many_arguments)]
fn scene(tag: &str, world: &World, table: &EmitterTable, light: &Lighting, s: &Settings, cam: &Pinhole, w: u32, h: u32) -> Scene {
    let spp = ref_spp();
    let dir = cache_dir();
    let path = dir.join(format!("v{CACHE_VERSION}_{tag}_{w}x{h}_ref{spp}.bin"));
    let n = (w * h) as usize;
    let per = 2 + 4 * AGES.len() + 6;
    if let Some(v) = read_f32(&path).filter(|v| v.len() == n * per) {
        let mut sc = Scene { w, h, keys: Vec::with_capacity(n), hist: BTreeMap::new(), ref_mean: Vec::with_capacity(n), ref_se2: Vec::with_capacity(n) };
        for a in AGES {
            sc.hist.insert(a, Vec::with_capacity(n));
        }
        for p in v.chunks_exact(per) {
            sc.keys.push([p[0].to_bits(), p[1].to_bits()]);
            for (k, a) in AGES.iter().enumerate() {
                let o = 2 + 4 * k;
                sc.hist.get_mut(a).unwrap().push([p[o], p[o + 1], p[o + 2], p[o + 3]]);
            }
            let o = 2 + 4 * AGES.len();
            sc.ref_mean.push([p[o] as f64, p[o + 1] as f64, p[o + 2] as f64]);
            sc.ref_se2.push([p[o + 3] as f64, p[o + 4] as f64, p[o + 5] as f64]);
        }
        eprintln!("model {tag}: cached {}", path.display());
        return sc;
    }
    let t = std::time::Instant::now();
    let al = albedos(world.materials()).unwrap();
    let last = *AGES.iter().max().unwrap();
    let px = par_pixels(w, h, |x, y| {
        let d = cam.dir(x, y);
        let key = match trace(world, &Ray { origin: cam.eye, dir: d }, f64::INFINITY) {
            None => [FACE_NONE, 0],
            Some(hit) => {
                let f = hit.face.expect("the camera is outside solid voxels");
                let a = f.axis as usize;
                let plane = (cam.eye[a] + d[a] * hit.t).round() as i32;
                [(hit.material.raw() as u32) << 3 | (2 * f.axis as u32 + f.positive as u32), plane as u32]
            }
        };
        let mut hist = Vec::with_capacity(AGES.len());
        let (mut c, mut m) = ([0.0f32; 3], 0.0f32);
        for k in 1..=last {
            let x3 = sample_pixel_lit(world, &al, table, light, cam, s, x, y, k, SEED).map(|v| v as f32);
            let l = luminance(x3.map(|v| v as f64)) as f32;
            for ch in 0..3 {
                c[ch] += (x3[ch] - c[ch]) / k as f32;
            }
            m += (l * l - m) / k as f32;
            if AGES.contains(&k) {
                hist.push([c[0], c[1], c[2], m]);
            }
        }
        let (mut rs, mut rq) = ([0.0; 3], [0.0; 3]);
        for f in 0..spp {
            let v = sample_pixel_lit(world, &al, table, light, cam, s, x, y, f, REF_SEED);
            for ch in 0..3 {
                rs[ch] += v[ch];
                rq[ch] += v[ch] * v[ch];
            }
        }
        Px { key, hist, ref_sum: rs, ref_sq: rq }
    });
    let mut v = Vec::with_capacity(n * per);
    for p in &px {
        v.push(f32::from_bits(p.key[0]));
        v.push(f32::from_bits(p.key[1]));
        for hh in &p.hist {
            v.extend_from_slice(hh);
        }
        let k = spp as f64;
        let mean = p.ref_sum.map(|s| s / k);
        for m in mean {
            v.push(m as f32);
        }
        for ch in 0..3 {
            let var = (p.ref_sq[ch] - p.ref_sum[ch] * p.ref_sum[ch] / k).max(0.0) / (k - 1.0);
            v.push((var / k) as f32);
        }
    }
    std::fs::create_dir_all(&dir).unwrap();
    write_f32(&path, &v);
    eprintln!("model {tag}: rendered {} frames + {spp} reference spp at {w}x{h} in {:.0} s", last, t.elapsed().as_secs_f64());
    scene(tag, world, table, light, s, cam, w, h)
}

// ---------------------------------------------------------------------------------------------
// The filter, mode for mode as `shaders/denoise.slang` and `Denoise::record`.

const F_PREFILTER: u32 = 1;
const F_INIT_GUIDE: u32 = 2;
const F_REMODULATE: u32 = 4;
const F_NO_VARIANCE_BLUR: u32 = 8;
const F_SYMMETRIC: u32 = 16;
const F_CONSERVATIVE: u32 = 32;
/// Model only (not in the shader): the conservative exchange without the border normaliser (W = 1).
const F_MODEL_PLAIN: u32 = 1 << 30;

/// The shader's border normaliser for a level: the kernel's sum over the pixel's same-surface taps at
/// `step`, rounded up to 1/64 (the value packed into the surface key); 1 from the third level on.
fn border_sum(m: &Model, i: usize, step: i32, level: u32) -> f32 {
    const H: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];
    if level >= 2 {
        return 1.0;
    }
    let (x, y) = ((i as i32) % m.sc.w as i32, (i as i32) / m.sc.w as i32);
    let mut t = 0.0f32;
    for dy in -2..=2i32 {
        for dx in -2..=2i32 {
            if let Some(q) = m.neighbour(x + dx * step, y + dy * step) {
                if m.same(i, q) {
                    t += H[(dx + 2) as usize] * H[(dy + 2) as usize];
                }
            }
        }
    }
    (t * 64.0).ceil().min(64.0) / 64.0
}

fn lum3(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

struct Model<'a> {
    sc: &'a Scene,
    albedo: &'a [[f32; 3]],
    /// The radiance buffer: colour and age.
    radiance: Vec<[f32; 4]>,
    hist: &'a [[f32; 4]],
    filt: [Vec<[f32; 4]>; 2],
    lum_guide: Vec<f32>,
}

impl Model<'_> {
    fn same(&self, i: usize, j: usize) -> bool {
        self.sc.keys[i] == self.sc.keys[j]
    }

    fn sky(&self, i: usize) -> bool {
        self.sc.keys[i][0] & 7 == FACE_NONE
    }

    fn albedo_of(&self, i: usize) -> [f32; 3] {
        self.albedo.get((self.sc.keys[i][0] >> 3) as usize).copied().unwrap_or([1.0; 3])
    }

    fn neighbour(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && x < self.sc.w as i32 && y < self.sc.h as i32).then(|| (y as u32 * self.sc.w + x as u32) as usize)
    }

    fn demodulate(c: [f32; 3], a: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|k| if a[k] > 1e-4 { c[k] / a[k] } else { 0.0 })
    }

    fn init(&mut self, flags: u32, s: &DenoiseSettings) {
        let (w, h) = (self.sc.w as i32, self.sc.h as i32);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let r = self.radiance[i];
                if self.sky(i) {
                    self.filt[0][i] = [r[0], r[1], r[2], -1.0];
                    if flags & F_INIT_GUIDE != 0 {
                        self.lum_guide[i] = 0.0;
                    }
                    continue;
                }
                let a = self.albedo_of(i);
                let illum = Self::demodulate([r[0], r[1], r[2]], a);
                let age = (r[3] as u32 & 0xFFFF).max(1);
                if flags & F_INIT_GUIDE != 0 && age >= s.prefilter_age {
                    self.lum_guide[i] = -1.0 - lum3(illum);
                } else if flags & F_INIT_GUIDE != 0 {
                    let (mut gs, mut gn) = (0.0f32, 0.0f32);
                    for gy in -1..=1 {
                        for gx in -1..=1 {
                            let Some(q) = self.neighbour(x + gx, y + gy) else { continue };
                            if !self.same(i, q) {
                                continue;
                            }
                            let rq = self.radiance[q];
                            gs += lum3(Self::demodulate([rq[0], rq[1], rq[2]], self.albedo_of(q)));
                            gn += 1.0;
                        }
                    }
                    self.lum_guide[i] = gs / gn;
                }
                let la = lum3(a).max(1e-4);
                let var = if age >= s.moments_age.max(1) {
                    let l = lum3([r[0], r[1], r[2]]);
                    (self.hist[i][3] - l * l).max(0.0)
                } else {
                    let (mut s1, mut s2, mut n) = (0.0f32, 0.0f32, 0.0f32);
                    for dy in -3..=3 {
                        for dx in -3..=3 {
                            let Some(q) = self.neighbour(x + dx, y + dy) else { continue };
                            if !self.same(i, q) {
                                continue;
                            }
                            let rq = self.radiance[q];
                            let l = lum3([rq[0], rq[1], rq[2]]);
                            s1 += l;
                            s2 += l * l;
                            n += 1.0;
                        }
                    }
                    s1 /= n;
                    (s2 / n - s1 * s1).max(0.0) * age as f32
                };
                self.filt[0][i] = [illum[0], illum[1], illum[2], var / (age as f32 * la * la)];
            }
        }
    }

    fn prefilter(&mut self, src: usize) {
        let (w, h) = (self.sc.w as i32, self.sc.h as i32);
        let mut out = vec![0.0f32; self.lum_guide.len()];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if self.sky(i) {
                    continue;
                }
                let (mut s, mut n) = (0.0f32, 0.0f32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(q) = self.neighbour(x + dx, y + dy) else { continue };
                        if !self.same(i, q) {
                            continue;
                        }
                        let c = self.filt[src][q];
                        s += lum3([c[0], c[1], c[2]]);
                        n += 1.0;
                    }
                }
                out[i] = s / n;
            }
        }
        self.lum_guide = out;
    }

    fn guide_value(g: f32) -> f32 {
        if g >= 0.0 {
            g
        } else {
            -1.0 - g
        }
    }

    fn level(&mut self, step: i32, src: usize, flags: u32, sigma_l: f32) {
        const H: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];
        let (w, h) = (self.sc.w as i32, self.sc.h as i32);
        let dst = 1 - src;
        let last = flags & F_REMODULATE != 0;
        let pre = flags & F_PREFILTER != 0;
        let sym = flags & (F_SYMMETRIC | F_CONSERVATIVE) != 0;
        let cons = flags & F_CONSERVATIVE != 0;
        let geo = cons;
        let level = step.trailing_zeros();
        // The border normaliser per pixel for this level (1 everywhere for the plain model arm).
        let wgeo: Vec<f32> = if geo { (0..(w * h) as usize).map(|i| if flags & F_MODEL_PLAIN != 0 || self.sky(i) { 1.0 } else { border_sum(self, i, step, level) }).collect() } else { Vec::new() };
        let mut out = self.filt[dst].clone();
        // A pixel's edge-stopping variance in its guide's units (var / 9 for a 3x3-mean guide).
        let gvar = |m: &Model, q: usize| -> f32 {
            let v = m.filt[src][q][3];
            if pre && m.lum_guide[q] >= 0.0 {
                v * (1.0 / 9.0)
            } else {
                v
            }
        };
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let c = self.filt[src][i];
                if self.sky(i) {
                    if !last {
                        out[i] = c;
                    }
                    continue;
                }
                let gp = if pre { self.lum_guide[i] } else { 0.0 };
                let lp = if pre { Self::guide_value(gp) } else { lum3([c[0], c[1], c[2]]) };
                let mean_guide = pre && gp >= 0.0;
                let mut sigma = 0.0f32;
                let vi = gvar(self, i);
                if !sym {
                    let (mut gv, mut gw) = (c[3], 1.0f32);
                    if flags & F_NO_VARIANCE_BLUR == 0 {
                        gv = 0.0;
                        gw = 0.0;
                        for vy in -1..=1 {
                            for vx in -1..=1 {
                                let Some(q) = self.neighbour(x + vx, y + vy) else { continue };
                                if !self.same(i, q) {
                                    continue;
                                }
                                let k = (if vx == 0 { 0.5 } else { 0.25 }) * (if vy == 0 { 0.5 } else { 0.25 });
                                gv += k * self.filt[src][q][3];
                                gw += k;
                            }
                        }
                    }
                    sigma = sigma_l * (gv / gw * if mean_guide { 1.0 / 9.0 } else { 1.0 }).max(1e-12).sqrt() + 1e-12;
                }
                let (mut sum, mut vsum, mut wsum) = ([0.0f32; 3], 0.0f32, 0.0f32);
                // Geo: the exchange sum_q c (x_q - x) and its coefficients' squares on the taps.
                let (mut ex, mut csum) = ([0.0f32; 3], 0.0f32);
                for dy in -2..=2i32 {
                    for dx in -2..=2i32 {
                        let Some(q) = self.neighbour(x + dx * step, y + dy * step) else { continue };
                        if !self.same(i, q) {
                            continue;
                        }
                        let cq = self.filt[src][q];
                        let lq = if pre { Self::guide_value(self.lum_guide[q]) } else { lum3([cq[0], cq[1], cq[2]]) };
                        let s = if sym { sigma_l * (vi + gvar(self, q)).max(1e-12).sqrt() + 1e-12 } else { sigma };
                        let wt = H[(dx + 2) as usize] * H[(dy + 2) as usize] * (-(lq - lp).abs() / s).exp();
                        if geo {
                            if q != i {
                                let cq_ = wt / wgeo[i].max(wgeo[q]);
                                for k in 0..3 {
                                    ex[k] += cq_ * (cq[k] - c[k]);
                                }
                                vsum += cq_ * cq_ * cq[3];
                                csum += cq_;
                            }
                            continue;
                        }
                        for k in 0..3 {
                            sum[k] += wt * cq[k];
                        }
                        vsum += wt * wt * cq[3];
                        wsum += wt;
                    }
                }
                let (o, v): ([f32; 3], f32) = if geo {
                    // The pixel keeps 1 - csum of its own value.
                    let cii = 1.0 - csum;
                    (std::array::from_fn(|k| c[k] + ex[k]), vsum + cii * cii * c[3])
                } else {
                    (std::array::from_fn(|k| sum[k] / wsum), vsum / (wsum * wsum))
                };
                if last {
                    let a = self.albedo_of(i);
                    let r = self.radiance[i];
                    self.radiance[i] = [o[0] * a[0], o[1] * a[1], o[2] * a[2], r[3]];
                } else {
                    out[i] = [o[0], o[1], o[2], v];
                }
            }
        }
        if !last {
            self.filt[dst] = out;
        }
    }

    fn remodulate(&mut self, src: usize) {
        for i in 0..self.radiance.len() {
            if self.sky(i) {
                continue;
            }
            let a = self.albedo_of(i);
            let c = self.filt[src][i];
            let r = self.radiance[i];
            self.radiance[i] = [c[0] * a[0], c[1] * a[1], c[2] * a[2], r[3]];
        }
    }
}

fn weight_flags(s: &DenoiseSettings) -> u32 {
    (if s.variance_blur { 0 } else { F_NO_VARIANCE_BLUR })
        | match s.weights {
            Weights::Svgf => 0,
            Weights::Symmetric => F_SYMMETRIC,
            Weights::Conservative => F_CONSERVATIVE,
        }
}

/// The filter on the history at `age`; returns the shown radiance.
fn filter(sc: &Scene, albedo: &[[f32; 3]], age: u32, s: &DenoiseSettings) -> Vec<[f32; 4]> {
    filter_with(sc, albedo, age, s, 0)
}

/// `filter` with model-only flags added to every level.
fn filter_with(sc: &Scene, albedo: &[[f32; 3]], age: u32, s: &DenoiseSettings, extra: u32) -> Vec<[f32; 4]> {
    let hist = &sc.hist[&age];
    let n = hist.len();
    let mut m = Model { sc, albedo, radiance: hist.iter().map(|c| [c[0], c[1], c[2], age as f32]).collect(), hist, filt: [vec![[0.0; 4]; n], vec![[0.0; 4]; n]], lum_guide: vec![0.0; n] };
    let base = weight_flags(s) | extra;
    let init_guide = s.levels > 0 && s.prefilter_levels > 0;
    m.init(base | if init_guide { F_INIT_GUIDE } else { 0 }, s);
    for level in 0..s.levels {
        let pre = level < s.prefilter_levels;
        if pre && level > 0 {
            m.prefilter((level % 2) as usize);
        }
        let mut flags = base | if pre { F_PREFILTER } else { 0 };
        if level + 1 == s.levels {
            flags |= F_REMODULATE;
        }
        m.level(1 << level, (level % 2) as usize, flags, s.sigma_l);
    }
    if s.levels == 0 {
        m.remodulate(0);
    }
    m.radiance
}

// ---------------------------------------------------------------------------------------------
// Metrics: G4's (`gpu/tests/lights.rs`).

fn lum4(c: [f32; 4]) -> f64 {
    luminance([c[0] as f64, c[1] as f64, c[2] as f64])
}

fn energy_change(f: &[[f32; 4]], r: &[[f32; 4]], m: &[bool]) -> f64 {
    let (mut sf, mut sr) = (0.0, 0.0);
    for i in (0..m.len()).filter(|&i| m[i]) {
        sf += lum4(f[i]);
        sr += lum4(r[i]);
    }
    (sf - sr) / sr.max(1e-300)
}

/// G4's relative MSE against the reference, with the reference's own noise (its squared standard
/// error) subtracted per pixel, so a finite reference does not hide differences between arms.
fn rel_mse(x: &[[f32; 4]], sc: &Scene, m: &[bool]) -> f64 {
    let idx: Vec<usize> = (0..m.len()).filter(|&i| m[i]).collect();
    let n = idx.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|c| idx.iter().map(|&i| sc.ref_mean[i][c]).sum::<f64>() / n);
    let eps: [f64; 3] = mean.map(|v| (0.1 * v).powi(2).max(1e-30));
    let mut se = 0.0;
    for &i in &idx {
        for c in 0..3 {
            let t = sc.ref_mean[i][c];
            se += ((x[i][c] as f64 - t).powi(2) - sc.ref_se2[i][c]) / (t * t + eps[c]);
        }
    }
    se / (3.0 * n)
}

fn arms() -> Vec<(String, DenoiseSettings, u32)> {
    let d = DenoiseSettings::default();
    let mut v = vec![("default (Svgf, sigma_l 4)".to_string(), d, 0), ("levels 0 (control)".to_string(), DenoiseSettings { levels: 0, prefilter_levels: 0, ..d }, 0), ("sigma_l 1e6".to_string(), DenoiseSettings { sigma_l: 1e6, ..d }, 0)];
    for weights in [Weights::Symmetric, Weights::Conservative] {
        for sl in [1.0f32, 2.0, 4.0, 8.0, 16.0] {
            v.push((format!("{weights:?} sigma_l {sl}"), DenoiseSettings { weights, sigma_l: sl, ..d }, 0));
        }
    }
    for sl in [4.0f32, 8.0] {
        v.push((format!("plain conservative (model) sigma_l {sl}"), DenoiseSettings { weights: Weights::Conservative, sigma_l: sl, ..d }, F_MODEL_PLAIN));
    }
    v
}

/// The model's screen: each arm's energy change against the raw history of the same frames, and
/// G4's Q2 (filtered relative MSE against the raw history at gain x age), night, both cameras,
/// Full; dusk (M3 transport) as the control. Also checks that the model's `levels 0` keeps energy
/// (the instrument) and prints how far the conservative arms' energy change is from 0.
#[test]
#[ignore]
fn model_filter_energy() {
    let (w, h) = size();
    let (world, _) = street_night(Dressing::Full);
    let table = gpu_table(&world);
    let albedo: Vec<[f32; 3]> = albedos(world.materials()).unwrap().iter().map(|a| a.map(|x| x as f32)).collect();
    eprintln!("model: {w}x{h}, {} emitters, reference {} spp", table.len(), ref_spp());
    let night = Lighting::new(Atmosphere::default(), SunPath::default().direction(21.0));
    let dusk = Lighting::new(Atmosphere::default(), SunPath::default().direction(17.5));
    let lit = Settings { max_bounces: 1, emitters_direct: true, emitters_indirect: true, ..Settings::default() };
    let m3 = Settings { max_bounces: 1, ..Settings::default() };
    let none = EmitterTable { snapshot: 0, emitters: Vec::new(), emission: Vec::new(), total_power: 0.0 };
    let only = std::env::var("NE_MODEL_ARMS").ok();
    let mut failed = Vec::new();
    for (cname, cam) in cameras(w, h) {
        for (time, light, s, tab) in [("night", &night, &lit, &table), ("dusk", &dusk, &m3, &none)] {
            let sc = scene(&format!("{cname}_{time}"), &world, tab, light, s, &cam, w, h);
            let m: Vec<bool> = sc.keys.iter().map(|k| k[0] & 7 != FACE_NONE).collect();
            eprintln!("{cname} {time}: {} surface px", m.iter().filter(|&&b| b).count());
            let raw: BTreeMap<u32, f64> = AGES.iter().map(|&a| (a, rel_mse(&sc.hist[&a], &sc, &m))).collect();
            eprintln!("  raw rel_mse at ages {AGES:?}: {:?}", raw.values().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
            for (name, dn, extra) in arms() {
                if only.as_deref().is_some_and(|o| !name.contains(o) && !name.starts_with("levels 0")) {
                    continue;
                }
                let t = std::time::Instant::now();
                let mut line = format!("  {name:28}");
                for (age, gain) in [(1u32, 8u32), (4, 4), (16, 4), (64, 1)] {
                    let raw_px: Vec<[f32; 4]> = sc.hist[&age].iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
                    let f = filter_with(&sc, &albedo, age, &dn, extra);
                    let e = energy_change(&f, &raw_px, &m);
                    let q = rel_mse(&f, &sc, &m);
                    let limit = raw[&(age * gain).min(64)];
                    line += &format!(" | a{age}: E {:+.4} mse {:.3} {}", e, q, if q <= limit { "≤" } else { ">" });
                    if name.starts_with("levels 0") && e.abs() > 1e-4 {
                        failed.push(format!("{cname} {time} levels 0 age {age}: {e}"));
                    }
                }
                eprintln!("{line}  ({:.1} s)", t.elapsed().as_secs_f64());
            }
        }
    }
    assert!(failed.is_empty(), "the model's instrument failed: {failed:?}");
}

/// C2 (device-free, runs by default): the conservative weights on synthetic histories. A dark
/// surface with rare bright samples (moments from the history, age 16): `Conservative` keeps each
/// surface's energy to f32 rounding, where `Svgf` loses it (the negative control: the mirror shows
/// the fault); a flat image stays flat; nothing crosses from one surface to its neighbour.
#[test]
fn conservative_weights_keep_energy() {
    let (w, h) = (48u32, 32u32);
    let n = (w * h) as usize;
    // Two surfaces (left and right halves, different materials); a few pixels hold a bright history.
    let keys: Vec<[u32; 2]> = (0..n).map(|i| if (i as u32 % w) < w / 2 { [1 << 3, 5] } else { [2 << 3, 5] }).collect();
    let albedo = vec![[1.0f32; 3], [0.5, 0.4, 0.3], [0.7, 0.7, 0.7]];
    let mut state = 0x9E37_79B9u32;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    let mut hist = Vec::with_capacity(n);
    for _ in 0..n {
        // Mostly dark (0.01), 2% of pixels drew one sample of 50 among 16 (history mean 3.1).
        let bright = rnd() < 0.02;
        let a = albedo[1][0];
        let mean = if bright { (15.0 * 0.01 + 50.0) / 16.0 } else { 0.01 + 0.001 * rnd() };
        let m2 = if bright { (15.0 * 0.0001 + 2500.0) / 16.0 } else { mean * mean };
        hist.push([mean * a, mean * 0.8 * a, mean * 0.6 * a, m2 * a * a]);
    }
    let flat: Vec<[f32; 4]> = (0..n).map(|_| [0.2, 0.2, 0.2, 0.04]).collect();
    let mk = |x: Vec<[f32; 4]>| Scene { w, h, keys: keys.clone(), hist: [(16u32, x)].into_iter().collect(), ref_mean: vec![[0.0; 3]; n], ref_se2: vec![[0.0; 3]; n] };
    let (spiky, flat) = (mk(hist), mk(flat));
    let half = |left: bool| -> Vec<bool> { (0..n).map(|i| ((i as u32 % w) < w / 2) == left).collect() };
    let d = DenoiseSettings::default();
    let raw: Vec<[f32; 4]> = spiky.hist[&16].iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
    for (weights, keeps) in [(Weights::Svgf, false), (Weights::Symmetric, false), (Weights::Conservative, true)] {
        for sigma_l in [1.0f32, 4.0, 16.0] {
            let s = DenoiseSettings { weights, sigma_l, ..d };
            let f = filter(&spiky, &albedo, 16, &s);
            let (el, er) = (energy_change(&f, &raw, &half(true)), energy_change(&f, &raw, &half(false)));
            eprintln!("{weights:?} sigma_l {sigma_l}: energy change left {el:+.2e}, right {er:+.2e}");
            if keeps {
                assert!(el.abs() < 1e-5 && er.abs() < 1e-5, "{weights:?} sigma_l {sigma_l}: {el} / {er}");
                let g = filter(&flat, &albedo, 16, &s);
                let worst = g.iter().zip(&flat.hist[&16]).map(|(a, b)| (a[0] - b[0]).abs()).fold(0.0f32, f32::max);
                assert!(worst < 1e-6, "flat image changed by {worst}");
            } else if weights == Weights::Svgf && sigma_l == 4.0 {
                assert!(el < -0.05, "the negative control: Svgf should lose energy here, got {el}");
            }
        }
    }
}
