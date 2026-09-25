//! The CPU reference path tracer (ADR-0005), over the exact voxel DDA of `world::reference`.
//!
//! It is the statistical oracle for `gpu::reference`: the same estimator, in f64, drawing its random
//! numbers in the same order from the same [`Rng`] streams (sample `s` of pixel `i` uses
//! `Rng::new(i, s, seed)`). Emission is off (ADR-0005, M3).
//!
//! Per sample:
//! 1. A primary ray through the pixel centre. A miss sees the sky (sky on) and the sun disk (sun on).
//! 2. At each surface vertex: the sun by next-event estimation (a uniform point on the disk, or the
//!    centre with `point_sun`, and one shadow ray), then one cosine-sampled continuation ray. A
//!    continuation that escapes takes the sky radiance; one that hits a surface continues while
//!    bounces remain (`max_bounces` = surface-to-surface bounces).

use std::f64::consts::PI;

use world::reference::{trace, Ray};
use world::{MaterialRegistry, World};

use crate::atmosphere::{Atmosphere, SkyOptions};
use crate::sample::{self, add, dot, mul, normalize, scale, Rng, V3};
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
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { sun: true, sky: true, max_bounces: 8, point_sun: false, uniform_sky: false, albedo_one: false, sky_options: SkyOptions::default() }
    }
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
    Some(Surface { point, normal, albedo })
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

/// One sample of pixel (x, y) for frame `frame`.
#[allow(clippy::too_many_arguments)]
pub fn sample_pixel(world: &World, albedo: &[V3], light: &Lighting, cam: &Pinhole, s: &Settings, x: u32, y: u32, frame: u32, seed: u32) -> V3 {
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
    let mut l = [0.0; 3];
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
        let wi = sample::cosine_hemisphere(surf.normal, rng.uniform(), rng.uniform());
        beta = mul(beta, surf.albedo);
        match hit_surface(world, albedo, surf.point, wi, s) {
            None => {
                l = add(l, mul(beta, sky(light, wi, &mut rng, s)));
                break;
            }
            Some(next) if bounce < s.max_bounces => surf = next,
            Some(_) => break,
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

/// Renders samples `first..first + count` of every pixel on `threads` threads (rows interleaved).
#[allow(clippy::too_many_arguments)]
pub fn render(world: &World, albedo: &[V3], light: &Lighting, cam: &Pinhole, s: &Settings, first: u32, count: u32, seed: u32, threads: usize) -> Accum {
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
                                    let v = sample_pixel(world, albedo, light, cam, s, x, y, f, seed);
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
