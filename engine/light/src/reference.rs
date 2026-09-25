//! The CPU reference path tracer (ADR-0005), over the exact voxel DDA of `world::reference`.
//!
//! It is the statistical oracle for `gpu::reference`: the same estimator, in f64, drawing its random
//! numbers in the same order from the same [`Rng`] streams (sample `s` of pixel `i` uses
//! `Rng::new(i, s, seed)`).
//!
//! Per sample:
//! 1. A primary ray through the pixel centre. A miss sees the sky (sky on) and the sun disk (sun on).
//!    A hit sees the surface's emitted radiance (`emission`, 4A).
//! 2. At each surface vertex: the sun by next-event estimation (a uniform point on the disk, or the
//!    centre with `point_sun`, and one shadow ray); then the emitters by next-event estimation
//!    (`emitters_direct` at the primary vertex, `emitters_indirect` after it; ADR-0005 Amendment 3);
//!    then one cosine-sampled continuation ray. A continuation that escapes takes the sky radiance;
//!    one that hits a surface continues while bounces remain (`max_bounces` = surface-to-surface
//!    bounces) and never adds that surface's emission.
//!
//! Emitter terms are off by default, so every M3 image is unchanged; with them off, no emitter random
//! numbers are drawn.

use std::f64::consts::PI;

use world::reference::{trace, Ray};
use world::{MaterialRegistry, World};

use crate::atmosphere::{Atmosphere, SkyOptions};
use crate::emitters::{EmitterTable, SphericalRect, SOLID_ANGLE_MIN};
use crate::sample::{self, add, dot, mul, normalize, scale, sub, Rng, V3};
use crate::sun::{self, E_SUN};

/// Offset of secondary-ray origins along the face normal, voxels (ADR-0005).
pub const RAY_OFFSET: f64 = 1.0 / 256.0;

/// A pinhole camera in voxel units with the `gpu::raster::Camera` conventions: the ray through pixel
/// centre (x, y) is `forward + sx * right + sy * up`, with `right` and `up` pre-scaled by the
/// half-angle tangents.
#[derive(Clone, Copy, Debug)]
pub struct Pinhole {
    pub eye: V3,
    pub forward: V3,
    pub right: V3,
    pub up: V3,
    pub width: u32,
    pub height: u32,
}

impl Pinhole {
    pub fn dir(&self, x: u32, y: u32) -> V3 {
        let sx = 2.0 * (x as f64 + 0.5) / self.width as f64 - 1.0;
        let sy = 1.0 - 2.0 * (y as f64 + 0.5) / self.height as f64;
        add(self.forward, add(scale(self.right, sx), scale(self.up, sy)))
    }
}

/// What the estimator includes. The defaults are the full M3 image (sun, sky, 8 bounces).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub sun: bool,
    pub sky: bool,
    pub max_bounces: u32,
    /// Sample the sun's centre only (hard shadows): the exact control of 3B.
    pub point_sun: bool,
    /// Control: the sky is 1 in every direction and the sun is off.
    pub uniform_sky: bool,
    /// Control: every albedo is 1.
    pub albedo_one: bool,
    pub sky_options: SkyOptions,
    /// Emitted radiance where the primary ray hits (4A).
    pub emission: bool,
    /// Emitter next-event estimation at the primary vertex (4A).
    pub emitters_direct: bool,
    /// Emitter next-event estimation at vertices 1..=`max_bounces` (4A).
    pub emitters_indirect: bool,
    /// Control: sample every emitter by area (uniform on the quad) instead of by solid angle.
    pub emitter_area_sampling: bool,
    /// Planted emitter faults for negative controls.
    pub emitter_faults: EmitterFaults,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            sun: true,
            sky: true,
            max_bounces: 8,
            point_sun: false,
            uniform_sky: false,
            albedo_one: false,
            sky_options: SkyOptions::default(),
            emission: false,
            emitters_direct: false,
            emitters_indirect: false,
            emitter_area_sampling: false,
            emitter_faults: EmitterFaults::default(),
        }
    }
}

impl Settings {
    /// Whether any emitter term is on (the estimator then needs an [`EmitterTable`]).
    pub fn uses_emitters(&self) -> bool {
        self.emission || self.emitters_direct || self.emitters_indirect
    }
}

/// Planted faults for the emitter terms (4A negative controls); all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmitterFaults {
    /// The area PDF used as if it were a solid-angle PDF: A / P(i) in place of the geometry term
    /// (by area) or of S / P(i) (by solid angle).
    pub no_solid_angle: bool,
    /// A continuation that hits an emitter adds its emission (counted twice with next-event estimation).
    pub double_emission: bool,
    /// The emitter's cosine is dropped (by area; by solid angle there is none to drop).
    pub no_emitter_cosine: bool,
}

/// The sun for one time of day.
#[derive(Clone, Copy, Debug)]
pub struct Lighting {
    pub atmosphere: Atmosphere,
    /// Unit vector toward the sun.
    pub sun_dir: V3,
    /// Direct-sun irradiance at the street, perpendicular to the sun: `E_SUN × T_obs` (ADR-0005).
    pub sun_at_ground: V3,
}

impl Lighting {
    /// Steps of the host transmittance to the sun (fine quadrature, error far below 1e-6).
    pub const SUN_STEPS: u32 = 4096;

    pub fn new(atmosphere: Atmosphere, sun_dir: V3) -> Lighting {
        let sun_dir = normalize(sun_dir);
        let t = atmosphere.transmittance(atmosphere.observer(), sun_dir, Self::SUN_STEPS);
        Lighting { atmosphere, sun_dir, sun_at_ground: scale(t, E_SUN) }
    }

    /// Radiance of the sun disk seen from the street.
    pub fn sun_radiance(&self) -> V3 {
        scale(self.sun_at_ground, 1.0 / sun::sun_solid_angle())
    }
}

/// Lambertian albedo per material id (ADR-0005: each channel in [0, 1], finite).
pub fn albedos(reg: &MaterialRegistry) -> Result<Vec<V3>, String> {
    reg.iter()
        .map(|(id, def)| {
            let c = def.params.base_color;
            if c.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)) {
                Ok(c.map(|x| x as f64))
            } else {
                Err(format!("material {} ({}) has base colour {c:?} outside [0, 1]", id.raw(), def.name))
            }
        })
        .collect()
}

struct Surface {
    point: V3,
    normal: V3,
    albedo: V3,
    material: world::MaterialId,
}

fn hit_surface(world: &World, albedo: &[V3], origin: V3, dir: V3, s: &Settings) -> Option<Surface> {
    let h = trace(world, &Ray { origin, dir }, f64::INFINITY)?;
    let face = h.face.expect("ray origins lie outside solid voxels");
    let mut point = add(origin, scale(dir, h.t));
    let normal = face.normal();
    let a = face.axis as usize;
    // Faces lie on integer voxel planes: snap, then offset (ADR-0005).
    point[a] = point[a].round() + normal[a] * RAY_OFFSET;
    let albedo = if s.albedo_one { [1.0; 3] } else { albedo[h.material.raw() as usize] };
    Some(Surface { point, normal, albedo, material: h.material })
}

fn sky(light: &Lighting, dir: V3, rng: &mut Rng, s: &Settings) -> V3 {
    if s.uniform_sky {
        [1.0; 3]
    } else if s.sky {
        light.atmosphere.sky_sample(dir, light.sun_dir, rng, &s.sky_options)
    } else {
        [0.0; 3]
    }
}

/// Emitter next-event estimation at `surf` (ADR-0005 Amendment 3), without β and albedo: one emitter
/// by power, then a point uniform in its solid angle (by area when the solid angle is below
/// [`SOLID_ANGLE_MIN`], or with `area`), and one visibility segment. Always draws its numbers first.
fn emitter_sample(world: &World, em: &EmitterTable, surf: &Surface, rng: &mut Rng, area: bool, f: &EmitterFaults) -> V3 {
    let e = &em.emitters[em.select(rng)];
    let (u, v) = (rng.uniform(), rng.uniform());
    // Behind or in the emitter's plane: every point of it has cos θ_y ≤ 0.
    if dot(e.normal, sub(surf.point, e.p0)) <= 0.0 {
        return [0.0; 3];
    }
    let sr = if area { None } else { Some(SphericalRect::new(e, surf.point)).filter(|r| r.solid_angle >= SOLID_ANGLE_MIN) };
    let y = match &sr {
        Some(r) => r.sample(u, v),
        None => e.point(u, v),
    };
    let d = sub(y, surf.point);
    let d2 = dot(d, d);
    let w = scale(d, 1.0 / d2.sqrt());
    let (cx, cy) = (dot(surf.normal, w), -dot(e.normal, w));
    if cx <= 0.0 || cy <= 0.0 {
        return [0.0; 3];
    }
    // The segment ends RAY_OFFSET in front of the emitter's face, so the face itself never occludes.
    let end = add(y, scale(e.normal, RAY_OFFSET));
    if trace(world, &Ray { origin: surf.point, dir: sub(end, surf.point) }, 1.0).is_some() {
        return [0.0; 3];
    }
    // Radiance × cos θ_x / pdf in solid angle, over π.
    let k = if f.no_solid_angle {
        e.area
    } else if let Some(r) = &sr {
        r.solid_angle
    } else if f.no_emitter_cosine {
        e.area / d2
    } else {
        e.area * cy / d2
    };
    scale(e.radiance, cx * k / (e.pdf * PI))
}

/// One sample of pixel (x, y) for frame `frame`, without emitters (`s` must not use them).
#[allow(clippy::too_many_arguments)]
pub fn sample_pixel(world: &World, albedo: &[V3], light: &Lighting, cam: &Pinhole, s: &Settings, x: u32, y: u32, frame: u32, seed: u32) -> V3 {
    assert!(!s.uses_emitters(), "emitter terms need an EmitterTable: use sample_pixel_lit");
    sample_pixel_lit(world, albedo, &NO_EMITTERS, light, cam, s, x, y, frame, seed)
}

static NO_EMITTERS: EmitterTable = EmitterTable { snapshot: 0, emitters: Vec::new(), emission: Vec::new(), total_power: 0.0 };

/// One sample of pixel (x, y) for frame `frame`, with the emitters of `em`.
#[allow(clippy::too_many_arguments)]
pub fn sample_pixel_lit(world: &World, albedo: &[V3], em: &EmitterTable, light: &Lighting, cam: &Pinhole, s: &Settings, x: u32, y: u32, frame: u32, seed: u32) -> V3 {
    let mut rng = Rng::new(y * cam.width + x, frame, seed);
    let d = cam.dir(x, y);
    let Some(mut surf) = hit_surface(world, albedo, cam.eye, d, s) else {
        let dn = normalize(d);
        let mut l = sky(light, dn, &mut rng, s);
        if s.sun && !s.uniform_sky && dot(dn, light.sun_dir) >= sun::sun_cos_max() {
            l = add(l, light.sun_radiance());
        }
        return l;
    };
    let sun_on = s.sun && !s.uniform_sky;
    let mut l = if s.emission { em.emission_of(surf.material) } else { [0.0; 3] };
    let mut beta = [1.0; 3];
    for bounce in 0..=s.max_bounces {
        // Sun: always draw two numbers, so the streams stay aligned with the GPU.
        let (u1, u2) = (rng.uniform(), rng.uniform());
        if sun_on {
            let ws = if s.point_sun { light.sun_dir } else { sample::uniform_cone(light.sun_dir, sun::sun_cos_max(), u1, u2) };
            let c = dot(surf.normal, ws);
            if c > 0.0 && trace(world, &Ray { origin: surf.point, dir: ws }, f64::INFINITY).is_none() {
                l = add(l, mul(mul(beta, surf.albedo), scale(light.sun_at_ground, c / PI)));
            }
        }
        let nee = if bounce == 0 { s.emitters_direct } else { s.emitters_indirect };
        if nee && !em.is_empty() {
            l = add(l, mul(mul(beta, surf.albedo), emitter_sample(world, em, &surf, &mut rng, s.emitter_area_sampling, &s.emitter_faults)));
        }
        let wi = sample::cosine_hemisphere(surf.normal, rng.uniform(), rng.uniform());
        beta = mul(beta, surf.albedo);
        match hit_surface(world, albedo, surf.point, wi, s) {
            None => {
                l = add(l, mul(beta, sky(light, wi, &mut rng, s)));
                break;
            }
            Some(next) => {
                if s.emitter_faults.double_emission {
                    l = add(l, mul(beta, em.emission_of(next.material)));
                }
                if bounce == s.max_bounces {
                    break;
                }
                surf = next;
            }
        }
    }
    l
}

/// Per-pixel sums over the samples of an image.
#[derive(Clone, Debug)]
pub struct Accum {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub sum: Vec<V3>,
    pub sum_sq: Vec<V3>,
}

impl Accum {
    pub fn mean(&self, i: usize) -> V3 {
        scale(self.sum[i], 1.0 / self.samples as f64)
    }

    /// Standard error of the per-pixel mean, per channel.
    pub fn std_error(&self, i: usize) -> V3 {
        let n = self.samples as f64;
        let m = self.mean(i);
        [0, 1, 2].map(|c| ((self.sum_sq[i][c] / n - m[c] * m[c]).max(0.0) / (n - 1.0).max(1.0)).sqrt())
    }
}

/// Renders samples `first..first + count` of every pixel on `threads` threads (rows interleaved),
/// without emitters (`s` must not use them).
#[allow(clippy::too_many_arguments)]
pub fn render(world: &World, albedo: &[V3], light: &Lighting, cam: &Pinhole, s: &Settings, first: u32, count: u32, seed: u32, threads: usize) -> Accum {
    assert!(!s.uses_emitters(), "emitter terms need an EmitterTable: use render_lit");
    render_lit(world, albedo, &NO_EMITTERS, light, cam, s, first, count, seed, threads)
}

/// [`render`] with the emitters of `em`.
#[allow(clippy::too_many_arguments)]
pub fn render_lit(world: &World, albedo: &[V3], em: &EmitterTable, light: &Lighting, cam: &Pinhole, s: &Settings, first: u32, count: u32, seed: u32, threads: usize) -> Accum {
    let (w, h) = (cam.width, cam.height);
    let rows: Vec<Vec<(V3, V3)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for y in (t as u32..h).step_by(threads) {
                        let row: Vec<(V3, V3)> = (0..w)
                            .map(|x| {
                                let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
                                for f in first..first + count {
                                    let v = sample_pixel_lit(world, albedo, em, light, cam, s, x, y, f, seed);
                                    a = add(a, v);
                                    b = add(b, mul(v, v));
                                }
                                (a, b)
                            })
                            .collect();
                        out.push((y, row));
                    }
                    out
                })
            })
            .collect();
        let mut rows = vec![Vec::new(); h as usize];
        for hd in handles {
            for (y, row) in hd.join().expect("reference thread") {
                rows[y as usize] = row;
            }
        }
        rows
    });
    let flat: Vec<(V3, V3)> = rows.into_iter().flatten().collect();
    Accum { width: w, height: h, samples: count, sum: flat.iter().map(|p| p.0).collect(), sum_sq: flat.iter().map(|p| p.1).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sun::SunPath;
    use world::{MaterialParams, VoxelCoord};

    /// A 32 x 32 floor with a 4-voxel wall, albedo 0.5.
    fn floor_world() -> World {
        let mut r = MaterialRegistry::new();
        let m = r.register("grey", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
        let mut w = World::new(r);
        for x in 0..32 {
            for z in 0..32 {
                w.set(VoxelCoord::new(x, 0, z), Some(m)).unwrap();
            }
        }
        for y in 1..5 {
            for z in 0..32 {
                w.set(VoxelCoord::new(20, y, z), Some(m)).unwrap();
            }
        }
        w
    }

    fn down_camera() -> Pinhole {
        Pinhole { eye: [10.5, 12.0, 16.0], forward: [0.0, -1.0, 0.0], right: [0.3, 0.0, 0.0], up: [0.0, 0.0, -0.3], width: 8, height: 8 }
    }

    #[test]
    fn albedos_outside_the_unit_range_are_refused() {
        let mut r = MaterialRegistry::new();
        r.register("ok", MaterialParams::diffuse(0.2, 1.0, 0.0)).unwrap();
        assert!(albedos(&r).is_ok());
        r.register("hot", MaterialParams::diffuse(1.2, 0.0, 0.0)).unwrap();
        assert!(albedos(&r).unwrap_err().contains("hot"));
    }

    /// White furnace: albedo 1 under a uniform sky of 1 gives exactly 1 for every path that escapes.
    #[test]
    fn white_furnace_is_one() {
        let w = floor_world();
        let al = albedos(w.materials()).unwrap();
        let light = Lighting::new(Atmosphere::default(), SunPath::default().direction(12.0));
        let s = Settings { uniform_sky: true, albedo_one: true, max_bounces: 64, ..Settings::default() };
        let acc = render(&w, &al, &light, &down_camera(), &s, 0, 64, 1, 2);
        for i in 0..acc.sum.len() {
            let m = acc.mean(i);
            assert!(m.iter().all(|&v| (v - 1.0).abs() < 1e-12), "pixel {i}: {m:?}");
        }
    }

    /// Point sun, no sky, no bounce: an open floor pixel is ρ/π × E × T × cos θ, exactly.
    #[test]
    fn point_sun_on_a_plane_matches_the_closed_form() {
        let w = floor_world();
        let al = albedos(w.materials()).unwrap();
        let sun = SunPath::default().direction(12.0);
        let light = Lighting::new(Atmosphere::default(), sun);
        let s = Settings { sky: false, point_sun: true, max_bounces: 0, ..Settings::default() };
        let cam = down_camera();
        let acc = render(&w, &al, &light, &cam, &s, 0, 1, 1, 1);
        let expect = scale(light.sun_at_ground, 0.5 / PI * sun[1]);
        // The camera looks at the floor west of the wall; the noon sun is due south, so nothing shades it.
        for i in 0..acc.sum.len() {
            let m = acc.mean(i);
            for c in 0..3 {
                assert!((m[c] / expect[c] - 1.0).abs() < 1e-12, "pixel {i}: {m:?} vs {expect:?}");
            }
        }
        assert!(expect[0] > expect[2], "the noon sun at 45° is slightly warm after the atmosphere");
    }
}

#[cfg(test)]
mod fingerprint {
    use super::*;
    use crate::sun::SunPath;

    /// Diagnostic (4A): prints a bit-level fingerprint of a small street render with the default
    /// settings, so a change can show that emitters-off images are unchanged on the same machine.
    #[test]
    #[ignore]
    fn diagnostic_street_fingerprint() {
        let (w, view) = world::scene::street_block();
        let al = albedos(w.materials()).unwrap();
        let vox = |m: [f64; 3]| m.map(|x| x / world::dims::VOXEL_SIZE_M);
        let (eye, target) = (vox(view.eye_m), vox(view.target_m));
        let f = normalize(sample::sub(target, eye));
        let r = normalize(sample::cross(f, [0.0, 1.0, 0.0]));
        let u = sample::cross(r, f);
        let th = (view.vertical_fov_deg.to_radians() / 2.0).tan();
        let cam = Pinhole { eye, forward: f, right: scale(r, th * 16.0 / 9.0), up: scale(u, th), width: 32, height: 18 };
        for hour in [6.25, 12.0, 18.25] {
            let light = Lighting::new(Atmosphere::default(), SunPath::default().direction(hour));
            let acc = render(&w, &al, &light, &cam, &Settings { max_bounces: 2, ..Settings::default() }, 0, 4, 7, 2);
            let mut h: u64 = 0xcbf29ce484222325;
            for v in acc.sum.iter().chain(acc.sum_sq.iter()).flatten() {
                h = (h ^ v.to_bits()).wrapping_mul(0x100000001b3);
            }
            eprintln!("fingerprint hour {hour}: {h:016x}");
        }
    }
}
