//! Random numbers and direction sampling shared with `gpu/shaders/reference.slang` (ADR-0005).
//!
//! - [`Rng`]: PCG32 RXS-M-XS with a 32-bit state; floats are the top 24 bits of the output, so the
//!   GPU (f32) and the CPU draw the same `u` values from the same seed.
//! - Directions are unit vectors; every sampler also states its PDF in solid angle.

use std::f64::consts::PI;

pub type V3 = [f64; 3];

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn mul(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / dot(a, a).sqrt())
}

/// The PCG hash (Jarzynski and Olano 2020), used only to derive seeds.
pub fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// PCG32 RXS-M-XS, 32-bit state.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u32,
}

impl Rng {
    /// The per-pixel, per-frame stream: `pcg_hash(pixel + pcg_hash(frame + pcg_hash(seed)))`.
    pub fn new(pixel: u32, frame: u32, seed: u32) -> Rng {
        Rng { state: pcg_hash(pixel.wrapping_add(pcg_hash(frame.wrapping_add(pcg_hash(seed))))) }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(747796405).wrapping_add(2891336453);
        let s = self.state;
        let word = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
        (word >> 22) ^ word
    }

    /// Uniform in [0, 1) with 24 bits, exactly representable in f32.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u32() >> 8) as f64 * (1.0 / 16_777_216.0)
    }
}

/// An orthonormal basis (t, b, n) around unit `n` (Duff et al. 2017).
pub fn basis(n: V3) -> (V3, V3) {
    let s = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n[2]);
    let b = n[0] * n[1] * a;
    ([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]], [b, s + n[1] * n[1] * a, -n[1]])
}

fn local(n: V3, x: f64, y: f64, z: f64) -> V3 {
    let (t, b) = basis(n);
    [t[0] * x + b[0] * y + n[0] * z, t[1] * x + b[1] * y + n[1] * z, t[2] * x + b[2] * y + n[2] * z]
}

/// Cosine-weighted hemisphere around `n`; PDF cos θ / π.
pub fn cosine_hemisphere(n: V3, u1: f64, u2: f64) -> V3 {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    local(n, r * phi.cos(), r * phi.sin(), (1.0 - u1).max(0.0).sqrt())
}

/// Uniform in the cone of half-angle acos(`cos_max`) around `axis`; PDF 1 / (2π(1 − cos_max)).
pub fn uniform_cone(axis: V3, cos_max: f64, u1: f64, u2: f64) -> V3 {
    let c = 1.0 - u1 * (1.0 - cos_max);
    let s = (1.0 - c * c).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    local(axis, s * phi.cos(), s * phi.sin(), c)
}

/// Uniform on the sphere; PDF 1 / 4π.
pub fn uniform_sphere(u1: f64, u2: f64) -> V3 {
    let z = 1.0 - 2.0 * u1;
    let s = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    [s * phi.cos(), s * phi.sin(), z]
}

/// Henyey–Greenstein phase for the scattering angle with cosine `cos_theta` (between the
/// propagation directions before and after scattering).
pub fn hg_phase(cos_theta: f64, g: f64) -> f64 {
    let d = 1.0 + g * g - 2.0 * g * cos_theta;
    (1.0 - g * g) / (4.0 * PI * d * d.sqrt())
}

/// Samples the new propagation direction from HG around the old one, `dir`.
pub fn sample_hg(dir: V3, g: f64, u1: f64, u2: f64) -> V3 {
    let c = if g.abs() < 1e-4 {
        1.0 - 2.0 * u1
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        (1.0 + g * g - s * s) / (2.0 * g)
    };
    let c = c.clamp(-1.0, 1.0);
    let sn = (1.0 - c * c).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    local(dir, sn * phi.cos(), sn * phi.sin(), c)
}

pub fn rayleigh_phase(cos_theta: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_uniform_and_deterministic() {
        let mut a = Rng::new(7, 3, 1);
        let mut b = Rng::new(7, 3, 1);
        let mut bins = [0u32; 16];
        for _ in 0..160_000 {
            let x = a.uniform();
            assert_eq!(x, b.uniform());
            assert!((0.0..1.0).contains(&x));
            bins[(x * 16.0) as usize] += 1;
        }
        // Expected 10,000 per bin, standard deviation about 97.
        assert!(bins.iter().all(|&n| (n as i64 - 10_000).abs() < 500), "{bins:?}");
        assert_ne!(Rng::new(7, 3, 1).next_u32(), Rng::new(8, 3, 1).next_u32());
        assert_ne!(Rng::new(7, 3, 1).next_u32(), Rng::new(7, 4, 1).next_u32());
    }

    #[test]
    fn basis_is_orthonormal() {
        for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], normalize([0.3, -0.8, 0.2])] {
            let (t, b) = basis(n);
            for (x, y) in [(t, b), (t, n), (b, n)] {
                assert!(dot(x, y).abs() < 1e-12);
            }
            for x in [t, b] {
                assert!((dot(x, x) - 1.0).abs() < 1e-12);
            }
        }
    }

    /// The sampled mean cosine against each PDF's analytic moment.
    #[test]
    fn samplers_match_their_pdfs() {
        let n = normalize([0.2, 0.9, -0.1]);
        let mut r = Rng::new(1, 2, 3);
        let k = 200_000;
        let (mut cos_h, mut cos_c, mut cos_g, mut z_s) = (0.0, 0.0, 0.0, 0.0);
        let cmax = 0.9f64;
        for _ in 0..k {
            cos_h += dot(cosine_hemisphere(n, r.uniform(), r.uniform()), n);
            let c = dot(uniform_cone(n, cmax, r.uniform(), r.uniform()), n);
            assert!(c >= cmax - 1e-12);
            cos_c += c;
            cos_g += dot(sample_hg(n, 0.8, r.uniform(), r.uniform()), n);
            z_s += uniform_sphere(r.uniform(), r.uniform())[2];
        }
        let k = k as f64;
        // E[cos] = 2/3 for the cosine lobe, (1 + cmax)/2 for the cone, g for HG, 0 for the sphere.
        assert!((cos_h / k - 2.0 / 3.0).abs() < 3e-3, "{}", cos_h / k);
        assert!((cos_c / k - (1.0 + cmax) / 2.0).abs() < 1e-3);
        assert!((cos_g / k - 0.8).abs() < 3e-3, "{}", cos_g / k);
        assert!((z_s / k).abs() < 5e-3);
    }

    #[test]
    fn phases_integrate_to_one() {
        let n = 20_000;
        let (mut r, mut h) = (0.0, 0.0);
        for i in 0..n {
            let c = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
            r += rayleigh_phase(c);
            h += hg_phase(c, 0.8);
        }
        let w = 2.0 * PI * 2.0 / n as f64;
        assert!((r * w - 1.0).abs() < 1e-6);
        assert!((h * w - 1.0).abs() < 1e-3, "{}", h * w);
    }
}
