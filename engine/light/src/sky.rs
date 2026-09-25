//! Phase 3C: the real-time sky model (Hillaire 2020, "A Scalable and Production Ready Sky and
//! Atmosphere Rendering Technique"), built from the ADR-0005 [`Atmosphere`]. This CPU version (f64)
//! is the oracle for `gpu/shaders/sky.slang`; both are measured against the Monte Carlo sky of
//! [`Atmosphere::sky_sample`], which is the reference.
//!
//! Three look-up tables, all RGB, sampled bilinearly at texel centres with clamped edges:
//! - **Transmittance** ([`TRANS_W`] × [`TRANS_H`]): from radius r and view-zenith cosine μ to the top
//!   of the atmosphere, for rays that miss the ground (Bruneton's (r, μ) mapping). 128 midpoint steps.
//! - **Ground irradiance** ([`GROUND_W`] × 1): the skylight falling on the planet ground, E_sky(μ_sun),
//!   from the sky seen at ground level (256 cosine-stratified directions). Bruneton 2008 has it;
//!   Hillaire 2020 lights the ground by the direct sun only, which the 3C comparison with the Monte
//!   Carlo reference showed is too dark: at dawn the ground below the horizon came out at half the
//!   reference, and the ground-reflected light missing from the sky cost up to 15% of blue.
//! - **Multiple scattering** ([`MS_W`] × [`MS_H`]): Ψ_ms(r, μ_sun), the isotropic multiple-scattering
//!   source per unit scattering coefficient: L₂ / (1 − f_ms) from 64 directions, 20 steps each, with
//!   the lit Lambertian ground (direct sun plus the ground irradiance). It depends on the sun's zenith
//!   angle at the point only, so it is built once. Ground irradiance and multiple scattering depend on
//!   each other: both are built twice (first with a sunlit-only ground).
//! - **Sky view** ([`VIEW_W`] × [`VIEW_H`]): the in-scattered radiance seen from the observer, by
//!   view azimuth relative to the sun (u, concentrated towards the sun's side) and view zenith (v,
//!   concentrated at the horizon, which lies just below 0° at the observer's 2 m). 32 steps per ray,
//!   single scattering with the Rayleigh and HG phases plus σ_s × Ψ_ms, and the directly lit ground.
//!   Rebuilt when the sun moves. The sun disk is not in it (ADR-0005).
//! - **Correction** (S-020, [`crate::sky_ref`]): each sky-view texel is multiplied by the ratio
//!   reference ÷ table, from the Monte Carlo sky baked per direction and sun elevation, interpolated
//!   bilinearly in the view coordinates and linearly in sun elevation. It removes the bias of the
//!   isotropic multiple-scattering approximation. Exactly 1 until [`SkyLuts::apply_reference`].
//!
//! Steps follow [`step_bound`] (split at the tangent point, dense at the lower end), as in the
//! reference; the first version's uniform steps missed the 1.2 km haze layer on slanted paths (1.2%
//! transmittance error). They are integrated with Hillaire's energy-conserving form: over a step of
//! extinction σ_t and length dt, a source S contributes T × S × (1 − e^{−σ_t dt}) / σ_t, where T is
//! the transmittance at the step's start.

use std::f64::consts::PI;

use crate::atmosphere::{step_bound, Atmosphere};
use crate::sky_ref::{self, SkyReference, CORR_H, CORR_S, CORR_W};
use crate::sample::{self, V3};
use crate::sun::E_SUN;

pub const TRANS_W: usize = 256;
pub const TRANS_H: usize = 64;
pub const MS_W: usize = 32;
pub const MS_H: usize = 32;
pub const VIEW_W: usize = 192;
pub const VIEW_H: usize = 108;
pub const GROUND_W: usize = 64;
pub const GROUND_DIRS: u32 = 16;
/// Built once, so more than Hillaire's 40: at 40 a grazing ray at 1.5 km was 0.9% off.
pub const TRANS_STEPS: u32 = 128;
pub const MS_STEPS: u32 = 20;
pub const MS_DIRS: u32 = 8;
pub const VIEW_STEPS: u32 = 32;

/// A table of RGB texels, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    pub w: usize,
    pub h: usize,
    pub data: Vec<V3>,
}

impl Lut {
    fn build(w: usize, h: usize, f: impl Fn(f64, f64) -> V3) -> Lut {
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                data.push(f((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64));
            }
        }
        Lut { w, h, data }
    }

    /// Bilinear at (u, v) in [0, 1], texel centres at (i + 0.5) / n, edges clamped.
    pub fn sample(&self, u: f64, v: f64) -> V3 {
        let fx = (u * self.w as f64 - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = (v * self.h as f64 - 0.5).clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let at = |x: usize, y: usize| self.data[y * self.w + x];
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        [0, 1, 2].map(|k| (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty)
    }
}

/// Distance from radius `r` along view-zenith cosine `mu` to the sphere of radius `radius`
/// (the far root). `None` if the ray misses it.
fn to_sphere_far(r: f64, mu: f64, radius: f64) -> Option<f64> {
    let disc = r * r * (mu * mu - 1.0) + radius * radius;
    (disc >= 0.0).then(|| -r * mu + disc.sqrt())
}

/// Whether a ray from radius `r` with view-zenith cosine `mu` hits the ground.
fn hits_ground(a: &Atmosphere, r: f64, mu: f64) -> bool {
    let rg = a.ground_radius_m;
    mu < 0.0 && r * r * (mu * mu - 1.0) + rg * rg >= 0.0
}

fn ground_distance(a: &Atmosphere, r: f64, mu: f64) -> f64 {
    let rg = a.ground_radius_m;
    let disc = (r * r * (mu * mu - 1.0) + rg * rg).max(0.0);
    (-r * mu - disc.sqrt()).max(0.0)
}

/// Radius and zenith cosine at distance `t` along a ray from (r, μ); `nu` is the cosine between the
/// ray and the sun, `mu_s` the sun's zenith cosine at the start. Returns (r_t, μ_s at the point).
fn along(r: f64, mu: f64, mu_s: f64, nu: f64, t: f64) -> (f64, f64) {
    let r_t = (t * t + 2.0 * r * mu * t + r * r).sqrt();
    (r_t, ((r * mu_s + t * nu) / r_t).clamp(-1.0, 1.0))
}

fn sub_uv(x: f64, n: usize) -> f64 {
    0.5 / n as f64 + x * (1.0 - 1.0 / n as f64)
}

fn unit_from_sub(u: f64, n: usize) -> f64 {
    (u - 0.5 / n as f64) / (1.0 - 1.0 / n as f64)
}

pub struct SkyLuts {
    pub atmosphere: Atmosphere,
    pub transmittance: Lut,
    pub ms: Lut,
    pub ground: Lut,
    /// Steps per sky-view ray (default [`VIEW_STEPS`]; diagnostics vary it).
    pub view_steps: u32,
    /// Multiple-scattering table: size, directions per axis and steps (defaults [`MS_W`], [`MS_H`],
    /// [`MS_DIRS`], [`MS_STEPS`]; diagnostics vary them).
    pub ms_quality: (usize, usize, u32, u32),
    /// Ratio reference ÷ table per [`sky_ref`] texel ([`sky_ref::index`]); exactly 1 when uncorrected.
    pub correction: Vec<V3>,
    /// Whether [`SkyLuts::apply_reference`] has set the correction.
    pub corrected: bool,
}

/// What [`SkyLuts::apply_reference`] found, per channel value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectionStats {
    pub min: f64,
    pub max: f64,
    /// Ratios outside [0.5, 2], clamped.
    pub clamped: usize,
    /// Values where the table is below 1e-9 (ratio set to 1).
    pub dark: usize,
}

impl SkyLuts {
    /// Builds the sun-independent tables.
    pub fn new(atmosphere: Atmosphere) -> SkyLuts {
        Self::with_ms_quality(atmosphere, (MS_W, MS_H, MS_DIRS, MS_STEPS))
    }

    pub fn with_ms_quality(atmosphere: Atmosphere, ms_quality: (usize, usize, u32, u32)) -> SkyLuts {
        let (ms_w, ms_h, _, _) = ms_quality;
        let a = atmosphere;
        let transmittance = Lut::build(TRANS_W, TRANS_H, |u, v| {
            let (r, mu) = trans_rmu(&a, u, v);
            trans_integral(&a, r, mu)
        });
        let zero = |w| Lut { w, h: 1, data: vec![[0.0; 3]; w] };
        let mut luts = SkyLuts { atmosphere, transmittance, ms: zero(1), ground: zero(GROUND_W), view_steps: VIEW_STEPS, ms_quality, correction: vec![[1.0; 3]; CORR_W * CORR_H * CORR_S], corrected: false };
        // Twice: multiple scattering needs the ground irradiance and the ground irradiance needs the sky.
        for _ in 0..2 {
            luts.ms = Lut::build(ms_w, ms_h, |u, v| {
                let mu_s = sub_to_unit_clamped(u, ms_w) * 2.0 - 1.0;
                let r = a.ground_radius_m + sub_to_unit_clamped(v, ms_h) * (a.top_radius_m - a.ground_radius_m);
                luts.ms_texel(r, mu_s)
            });
            luts.ground = Lut::build(GROUND_W, 1, |u, _| luts.ground_texel(sub_to_unit_clamped(u, GROUND_W) * 2.0 - 1.0));
        }
        luts
    }

    /// E_sky at the ground for the sun zenith cosine μ_s.
    pub fn ground_irradiance(&self, mu_s: f64) -> V3 {
        self.ground.sample(sub_uv((mu_s * 0.5 + 0.5).clamp(0.0, 1.0), GROUND_W), 0.5)
    }

    /// Radiance of the Lambertian planet ground at a point whose sun zenith cosine is μ_s.
    fn ground_radiance(&self, r: f64, mu_s: f64) -> V3 {
        let a = &self.atmosphere;
        let ts = self.transmittance(r, mu_s);
        let e = self.ground_irradiance(mu_s);
        [0, 1, 2].map(|c| a.ground_albedo / PI * (ts[c] * E_SUN * mu_s.max(0.0) + e[c]))
    }

    fn ground_texel(&self, mu_s: f64) -> V3 {
        let r = self.atmosphere.ground_radius_m;
        let sun: V3 = [(1.0 - mu_s * mu_s).max(0.0).sqrt(), mu_s, 0.0];
        let mut e = [0.0; 3];
        let n = GROUND_DIRS * GROUND_DIRS;
        for i in 0..GROUND_DIRS {
            for j in 0..GROUND_DIRS {
                // Cosine-stratified upper hemisphere: E = π × mean(L).
                let u1 = (i as f64 + 0.5) / GROUND_DIRS as f64;
                let phi = 2.0 * PI * (j as f64 + 0.5) / GROUND_DIRS as f64;
                let s = u1.sqrt();
                let w: V3 = [s * phi.cos(), (1.0 - u1).sqrt(), s * phi.sin()];
                let l = self.integrate_view(r, w[1], mu_s, sample::dot(w, sun));
                e = [0, 1, 2].map(|c| e[c] + PI * l[c] / n as f64);
            }
        }
        e
    }

    /// Transmittance from (r, μ) to the top of the atmosphere, 0 if the ray hits the ground.
    pub fn transmittance(&self, r: f64, mu: f64) -> V3 {
        if hits_ground(&self.atmosphere, r, mu) {
            return [0.0; 3];
        }
        let (u, v) = trans_uv(&self.atmosphere, r, mu);
        self.transmittance.sample(u, v)
    }

    /// Ψ_ms at radius r for the sun zenith cosine μ_s.
    pub fn multiple_scattering(&self, r: f64, mu_s: f64) -> V3 {
        let a = &self.atmosphere;
        let u = sub_uv((mu_s * 0.5 + 0.5).clamp(0.0, 1.0), self.ms.w);
        let v = sub_uv(((r - a.ground_radius_m) / (a.top_radius_m - a.ground_radius_m)).clamp(0.0, 1.0), self.ms.h);
        self.ms.sample(u, v)
    }

    fn ms_texel(&self, r: f64, mu_s: f64) -> V3 {
        let a = &self.atmosphere;
        let sun: V3 = [(1.0 - mu_s * mu_s).max(0.0).sqrt(), mu_s, 0.0];
        let (_, _, dirs, steps) = self.ms_quality;
        let (mut l2, mut fms) = ([0.0; 3], [0.0; 3]);
        let n = dirs * dirs;
        for i in 0..dirs {
            for j in 0..dirs {
                // Stratified uniform sphere; +Y is the local zenith.
                let cos_t = 1.0 - 2.0 * (i as f64 + 0.5) / dirs as f64;
                let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
                let phi = 2.0 * PI * (j as f64 + 0.5) / dirs as f64;
                let w: V3 = [sin_t * phi.cos(), cos_t, sin_t * phi.sin()];
                let mu = w[1];
                let nu = sample::dot(w, sun);
                let ground = hits_ground(a, r, mu);
                let t_end = if ground { ground_distance(a, r, mu) } else { to_sphere_far(r, mu, a.top_radius_m).unwrap_or(0.0) };
                let mut tr = [1.0; 3];
                let (mut lw, mut fw) = ([0.0; 3], [0.0; 3]);
                let pu = 1.0 / (4.0 * PI);
                for k in 0..steps {
                    let t0 = step_bound(k, steps, t_end, -r * mu);
                    let t1 = step_bound(k + 1, steps, t_end, -r * mu);
                    let (rt, mst) = along(r, mu, mu_s, nu, 0.5 * (t0 + t1));
                    let m = a.medium(rt - a.ground_radius_m);
                    let ss = [0, 1, 2].map(|c| m.rayleigh[c] + m.mie);
                    let ts = self.transmittance(rt, mst);
                    let dt = t1 - t0;
                    for c in 0..3 {
                        let e = m.extinction[c];
                        let int = (1.0 - (-e * dt).exp()) / e;
                        lw[c] += tr[c] * ss[c] * pu * ts[c] * E_SUN * int;
                        fw[c] += tr[c] * ss[c] * int;
                        tr[c] *= (-e * dt).exp();
                    }
                }
                if ground {
                    let (rg, msg) = along(r, mu, mu_s, nu, t_end);
                    let g = self.ground_radiance(rg, msg);
                    for c in 0..3 {
                        lw[c] += tr[c] * g[c];
                    }
                }
                for c in 0..3 {
                    l2[c] += lw[c] / n as f64;
                    fms[c] += fw[c] / n as f64;
                }
            }
        }
        // L₂ and f_ms integrate over the sphere with the isotropic phase: the mean over directions.
        [0, 1, 2].map(|c| l2[c] / (1.0 - fms[c]))
    }

    /// The sky-view table for the sun along unit `sun` (+Y up at the observer), times the correction.
    pub fn view(&self, sun: V3) -> Lut {
        let a = &self.atmosphere;
        let r = a.ground_radius_m + a.observer_altitude_m;
        let mu_s = sun[1];
        Lut::build(VIEW_W, VIEW_H, |u, v| {
            let (x, y) = (sub_to_unit_clamped(u, VIEW_W), sub_to_unit_clamped(v, VIEW_H));
            let (mu, cos_phi) = view_from_unit(a, x, y);
            let sin_v = (1.0 - mu * mu).max(0.0).sqrt();
            let nu = (mu * mu_s + sin_v * (1.0 - mu_s * mu_s).max(0.0).sqrt() * cos_phi).clamp(-1.0, 1.0);
            let l = self.integrate_view(r, mu, mu_s, nu);
            let k = self.correction_at(x, y, mu_s);
            [0, 1, 2].map(|c| l[c] * k[c])
        })
    }

    /// Sets the correction from a baked reference: per texel and channel, reference ÷ table (this
    /// table code, f64), clamped to [0.5, 2]; 1 where the table is below 1e-9. Refuses a reference
    /// baked for another atmosphere.
    pub fn apply_reference(&mut self, r: &SkyReference) -> Result<CorrectionStats, String> {
        if r.fingerprint != sky_ref::fingerprint(&self.atmosphere, &r.options) {
            return Err("sky reference: baked for a different atmosphere (fingerprint mismatch)".into());
        }
        let a = self.atmosphere;
        let r_obs = a.ground_radius_m + a.observer_altitude_m;
        let mut st = CorrectionStats { min: f64::INFINITY, max: 0.0, clamped: 0, dark: 0 };
        let mut corr = vec![[1.0; 3]; CORR_W * CORR_H * CORR_S];
        for k in 0..CORR_S {
            for y in 0..CORR_H {
                for x in 0..CORR_W {
                    let (d, sun) = sky_ref::texel(&a, x, y, k);
                    let t = self.integrate_view(r_obs, d[1], sun[1], sample::dot(d, sun));
                    let i = sky_ref::index(x, y, k);
                    for c in 0..3 {
                        if t[c] < 1e-9 {
                            st.dark += 1;
                            continue;
                        }
                        let q = r.mean[i][c] / t[c];
                        st.min = st.min.min(q);
                        st.max = st.max.max(q);
                        st.clamped += usize::from(!(0.5..=2.0).contains(&q));
                        corr[i][c] = q.clamp(0.5, 2.0);
                    }
                }
            }
        }
        self.correction = corr;
        self.corrected = true;
        Ok(st)
    }

    /// The correction at sky-view unit coordinates (x, y) for the sun zenith cosine μ_s: bilinear in
    /// (x, y) with texels at i / (n − 1), linear between the two nearest sun elevations.
    pub fn correction_at(&self, x: f64, y: f64, mu_s: f64) -> V3 {
        let ks = sky_ref::slice_coord(mu_s.clamp(-1.0, 1.0).asin().to_degrees());
        let k0 = ks.floor() as usize;
        let k1 = (k0 + 1).min(CORR_S - 1);
        let tk = ks - k0 as f64;
        let fx = (x * (CORR_W - 1) as f64).clamp(0.0, (CORR_W - 1) as f64);
        let fy = (y * (CORR_H - 1) as f64).clamp(0.0, (CORR_H - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(CORR_W - 1), (y0 + 1).min(CORR_H - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let slice = |k: usize| {
            let at = |x: usize, y: usize| self.correction[sky_ref::index(x, y, k)];
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            [0, 1, 2].map(|i| (a[i] * (1.0 - tx) + b[i] * tx) * (1.0 - ty) + (c[i] * (1.0 - tx) + d[i] * tx) * ty)
        };
        let (s0, s1) = (slice(k0), slice(k1));
        [0, 1, 2].map(|i| s0[i] * (1.0 - tk) + s1[i] * tk)
    }

    /// In-scattered radiance along a ray from (r, μ) with sun zenith cosine μ_s and view–sun cosine ν.
    pub fn integrate_view(&self, r: f64, mu: f64, mu_s: f64, nu: f64) -> V3 {
        let a = &self.atmosphere;
        let ground = hits_ground(a, r, mu);
        let t_end = if ground { ground_distance(a, r, mu) } else { to_sphere_far(r, mu, a.top_radius_m).unwrap_or(0.0) };
        let pr = sample::rayleigh_phase(nu);
        let pm = sample::hg_phase(nu, a.mie_g);
        let mut tr = [1.0; 3];
        let mut l = [0.0; 3];
        let mut t0 = 0.0;
        for k in 0..self.view_steps {
            let t1 = step_bound(k + 1, self.view_steps, t_end, -r * mu);
            let (rt, mst) = along(r, mu, mu_s, nu, 0.5 * (t0 + t1));
            let m = a.medium(rt - a.ground_radius_m);
            let ts = self.transmittance(rt, mst);
            let psi = self.multiple_scattering(rt, mst);
            let dt = t1 - t0;
            for c in 0..3 {
                let s = (m.rayleigh[c] * pr + m.mie * pm) * ts[c] * E_SUN + (m.rayleigh[c] + m.mie) * psi[c] * E_SUN;
                let e = m.extinction[c];
                l[c] += tr[c] * s * (1.0 - (-e * dt).exp()) / e;
                tr[c] *= (-e * dt).exp();
            }
            t0 = t1;
        }
        if ground {
            let (rg, msg) = along(r, mu, mu_s, nu, t_end);
            let g = self.ground_radiance(rg, msg);
            for c in 0..3 {
                l[c] += tr[c] * g[c];
            }
        }
        l
    }
}

/// The sky radiance of `view` (built for the sun along `sun`) in unit direction `d`.
pub fn sky_lookup(a: &Atmosphere, view: &Lut, sun: V3, d: V3) -> V3 {
    let (u, v) = view_uv(a, sun, d);
    view.sample(u, v)
}

fn sub_to_unit_clamped(u: f64, n: usize) -> f64 {
    unit_from_sub(u, n).clamp(0.0, 1.0)
}

/// Bruneton's transmittance mapping: texel (u, v) to (r, μ).
fn trans_rmu(a: &Atmosphere, u: f64, v: f64) -> (f64, f64) {
    let (rg, rt) = (a.ground_radius_m, a.top_radius_m);
    let (xm, xr) = (sub_to_unit_clamped(u, TRANS_W), sub_to_unit_clamped(v, TRANS_H));
    let h = (rt * rt - rg * rg).sqrt();
    let rho = h * xr;
    let r = (rho * rho + rg * rg).sqrt();
    let d_min = rt - r;
    let d_max = rho + h;
    let d = d_min + xm * (d_max - d_min);
    let mu = if d == 0.0 { 1.0 } else { ((h * h - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0) };
    (r, mu)
}

fn trans_uv(a: &Atmosphere, r: f64, mu: f64) -> (f64, f64) {
    let (rg, rt) = (a.ground_radius_m, a.top_radius_m);
    let h = (rt * rt - rg * rg).sqrt();
    let rho = (r * r - rg * rg).max(0.0).sqrt();
    let d = to_sphere_far(r, mu, rt).unwrap_or(0.0).max(0.0);
    let d_min = rt - r;
    let d_max = rho + h;
    let xm = if d_max > d_min { (d - d_min) / (d_max - d_min) } else { 0.0 };
    (sub_uv(xm.clamp(0.0, 1.0), TRANS_W), sub_uv((rho / h).clamp(0.0, 1.0), TRANS_H))
}

fn trans_integral(a: &Atmosphere, r: f64, mu: f64) -> V3 {
    let t_end = to_sphere_far(r, mu, a.top_radius_m).unwrap_or(0.0).max(0.0);
    let mut tau = [0.0; 3];
    for k in 0..TRANS_STEPS {
        let (t0, t1) = (step_bound(k, TRANS_STEPS, t_end, -r * mu), step_bound(k + 1, TRANS_STEPS, t_end, -r * mu));
        let t = 0.5 * (t0 + t1);
        let rt = (t * t + 2.0 * r * mu * t + r * r).sqrt();
        let m = a.medium(rt - a.ground_radius_m);
        for (t, e) in tau.iter_mut().zip(m.extinction) {
            *t += e * (t1 - t0);
        }
    }
    tau.map(|x| (-x).exp())
}

/// The observer's horizon: the zenith angle of the ray grazing the ground, and the angle between the
/// horizon and the nadir.
fn horizon(a: &Atmosphere) -> (f64, f64) {
    let r = a.ground_radius_m + a.observer_altitude_m;
    let beta = ((r * r - a.ground_radius_m * a.ground_radius_m).sqrt() / r).acos();
    (PI - beta, beta)
}

/// Sky-view texel (u, v) to (view-zenith cosine, cosine of the azimuth relative to the sun).
#[cfg(test)]
fn view_from_uv(a: &Atmosphere, u: f64, v: f64) -> (f64, f64) {
    view_from_unit(a, sub_to_unit_clamped(u, VIEW_W), sub_to_unit_clamped(v, VIEW_H))
}

/// Sky-view unit coordinates (x azimuth, y zenith, both in [0, 1]) to (view-zenith cosine, cosine of
/// the azimuth relative to the sun).
pub fn view_from_unit(a: &Atmosphere, x: f64, y: f64) -> (f64, f64) {
    let (zh, beta) = horizon(a);
    let zenith = if y < 0.5 {
        let c = 1.0 - 2.0 * y;
        (1.0 - c * c) * zh
    } else {
        let c = 2.0 * y - 1.0;
        zh + c * c * beta
    };
    let cos_phi = -(x * x * 2.0 - 1.0);
    (zenith.cos(), cos_phi)
}

/// Unit direction `d` (+Y up) to sky-view (u, v) for the sun along `sun`.
pub fn view_uv(a: &Atmosphere, sun: V3, d: V3) -> (f64, f64) {
    let (zh, beta) = horizon(a);
    let zenith = d[1].clamp(-1.0, 1.0).acos();
    let y = if zenith < zh {
        let c = (1.0 - zenith / zh).max(0.0).sqrt();
        (1.0 - c) * 0.5
    } else {
        let c = ((zenith - zh) / beta).clamp(0.0, 1.0).sqrt();
        0.5 + 0.5 * c
    };
    let (dh, sh) = ((d[0] * d[0] + d[2] * d[2]).sqrt(), (sun[0] * sun[0] + sun[2] * sun[2]).sqrt());
    let cos_phi = if dh > 1e-12 && sh > 1e-12 { ((d[0] * sun[0] + d[2] * sun[2]) / (dh * sh)).clamp(-1.0, 1.0) } else { 1.0 };
    let x = ((1.0 - cos_phi) * 0.5).sqrt();
    (sub_uv(x, VIEW_W), sub_uv(y, VIEW_H))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{normalize, Rng};
    use crate::sun::{SunPath, REFERENCE_TIMES};
    use crate::SkyOptions;

    #[test]
    fn view_mapping_round_trips() {
        let a = Atmosphere::default();
        for (v, u) in [(0.1, 0.2), (0.49, 0.9), (0.51, 0.5), (0.95, 0.05)] {
            let (mu, cos_phi) = view_from_uv(&a, u, v);
            // A direction with that zenith and azimuth relative to a sun along +x.
            let s = (1.0 - mu * mu).max(0.0).sqrt();
            let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
            let d = [s * cos_phi, mu, s * sin_phi];
            let (u2, v2) = view_uv(&a, normalize([1.0, 0.3, 0.0]), d);
            assert!((u2 - u).abs() < 1e-9 && (v2 - v).abs() < 1e-9, "({u}, {v}) -> ({u2}, {v2})");
        }
    }

    /// Transmittance table against the fine direct integral, at points between texels.
    #[test]
    fn transmittance_table_matches_the_integral() {
        let a = Atmosphere::default();
        let luts = SkyLuts::new(a);
        let mut worst = 0.0f64;
        for h in [2.0, 1500.0, 12_000.0, 60_000.0] {
            for mu in [1.0, 0.6, 0.2, 0.05, 0.01, -0.004] {
                let r = a.ground_radius_m + h;
                if hits_ground(&a, r, mu) {
                    continue;
                }
                let p = [0.0, r, 0.0];
                let d = [(1.0 - mu * mu).sqrt(), mu, 0.0];
                let exact = a.transmittance(p, d, 4096);
                let lut = luts.transmittance(r, mu);
                let direct = trans_integral(&a, r, mu);
                for c in 0..3 {
                    if exact[c] > 1e-3 {
                        let e = (lut[c] / exact[c] - 1.0).abs();
                        if e > 0.005 {
                            eprintln!("  h {h} mu {mu} ch {c}: table {:.5} integral(40 steps) {:.5} exact {:.5}", lut[c], direct[c], exact[c]);
                        }
                        worst = worst.max(e);
                    }
                }
            }
        }
        eprintln!("transmittance table worst relative error {worst:.2e}");
        assert!(worst < 0.01, "{worst}");
    }

    /// The sky-view table against the Monte Carlo reference sky (ADR-0005), per direction, from
    /// dawn to dusk. Budget declared before measuring: luminance error within 5% at the 95th
    /// percentile of directions, per channel within 10%. The error counted is what exceeds 3
    /// standard errors of the reference. Twilight is checked on the GPU (`tests/sky.rs`), where the
    /// reference can afford the samples its heavy tail needs (3A record).
    #[test]
    fn sky_table_matches_the_monte_carlo_sky() {
        let a = Atmosphere::default();
        let luts = SkyLuts::new(a);
        let n = 6000u32;
        let lum = |c: V3| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let mut failed = Vec::new();
        for (name, hour) in REFERENCE_TIMES.into_iter().filter(|(n, _)| *n != "twilight") {
            let sun = SunPath::default().direction(hour);
            let view = luts.view(sun);
            let (mut errs, mut chan, mut raw) = (Vec::new(), Vec::new(), Vec::new());
            let mut k = 0u32;
            for el in [-20.0f64, -3.0, 0.5, 3.0, 10.0, 30.0, 60.0, 89.0] {
                for az in [0.0f64, 45.0, 90.0, 135.0, 180.0, 270.0] {
                    let (e, z) = (el.to_radians(), az.to_radians());
                    let d = normalize([e.cos() * z.cos(), e.sin(), e.cos() * z.sin()]);
                    let t = sky_lookup(&a, &view, sun, d);
                    let (mut s, mut s2) = ([0.0; 3], [0.0; 3]);
                    let (mut ls, mut ls2) = (0.0, 0.0);
                    for i in 0..n {
                        let x = a.sky_sample(d, sun, &mut Rng::new(i, k, 7), &SkyOptions::default());
                        s = [0, 1, 2].map(|c| s[c] + x[c]);
                        s2 = [0, 1, 2].map(|c| s2[c] + x[c] * x[c]);
                        ls += lum(x);
                        ls2 += lum(x) * lum(x);
                    }
                    k += 1;
                    let nf = n as f64;
                    let m = s.map(|v| v / nf);
                    let se = [0, 1, 2].map(|c| ((s2[c] / nf - m[c] * m[c]).max(0.0) / nf).sqrt());
                    let (lm, lse) = (ls / nf, ((ls2 / nf - (ls / nf).powi(2)).max(0.0) / nf).sqrt());
                    let beyond = |x: f64, y: f64, e: f64| ((x - y).abs() - 3.0 * e).max(0.0) / y;
                    let rel = beyond(lum(t), lm, lse);
                    raw.push((lum(t) / lm - 1.0).abs());
                    errs.push(rel);
                    chan.push((0..3).map(|c| beyond(t[c], m[c], se[c])).fold(0.0, f64::max));
                    if rel > 0.05 {
                        eprintln!("  {name} el {el} az {az}: table {t:?} mc {m:?} (se {se:?}) excess {rel:.3}");
                    }
                }
            }
            for v in [&mut errs, &mut chan, &mut raw] {
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            }
            let p95 = |v: &Vec<f64>| v[(v.len() * 95) / 100];
            eprintln!(
                "sky table vs MC {name}: luminance |rel| raw mean {:.4} p95 {:.4} max {:.4}; beyond 3 se: p95 {:.4}; channel beyond 3 se p95 {:.4}",
                raw.iter().sum::<f64>() / raw.len() as f64,
                p95(&raw),
                raw.last().unwrap(),
                p95(&errs),
                p95(&chan)
            );
            if p95(&errs) > 0.05 || p95(&chan) > 0.10 {
                failed.push(format!("{name}: luminance p95 {:.4}, channel p95 {:.4}", p95(&errs), p95(&chan)));
            }
        }
        assert!(failed.is_empty(), "{failed:?}");
    }

    /// Diagnostic (on demand, 4B's night history cap): the skylight falling on the ground,
    /// E_sky(sun elevation), from 0° to −20°, computed directly (not the 64-texel table), with its
    /// fall per degree and the radiance it gives a Lambertian surface of albedo 0.3.
    #[test]
    #[ignore]
    fn diagnostic_skylight_below_horizon() {
        let luts = SkyLuts::new(Atmosphere::default());
        let mut prev: Option<f64> = None;
        for k in 0..=20 {
            let el = -(k as f64);
            let e = luts.ground_texel(el.to_radians().sin());
            let y = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
            let fall = prev.map_or(String::from("-"), |p| format!("{:.2}", p / y));
            eprintln!("elevation {el:>5.1}: E_sky Y {y:.3e}  radiance at albedo 0.3 {:.3e}  fall per degree {fall}", 0.3 / PI * y);
            prev = Some(y);
        }
    }

    /// Diagnostic (on demand): the table against the Monte Carlo sky per elevation (azimuth 90° from
    /// the sun), with the default ground (albedo 0.3) and with a black ground, which separates the
    /// ground's contribution from the multiple-scattering approximation in the air.
    #[test]
    #[ignore]
    fn diagnostic_sky_error_by_elevation_and_ground() {
        let n = 20_000u32;
        let fine = (128, 128, 32, 80);
        for (albedo, steps, q) in [(0.3, VIEW_STEPS, None), (0.0, VIEW_STEPS, None), (0.0, VIEW_STEPS, Some(fine)), (0.3, VIEW_STEPS, Some(fine))] {
            let a = Atmosphere { ground_albedo: albedo, ..Atmosphere::default() };
            let mut luts = q.map_or_else(|| SkyLuts::new(a), |q| SkyLuts::with_ms_quality(a, q));
            luts.view_steps = steps;
            for (name, hour) in [("dawn", 6.25), ("morning", 8.0), ("midday", 12.0)] {
                let sun = SunPath::default().direction(hour);
                let view = luts.view(sun);
                let mut line = format!("albedo {albedo} steps {steps} ms {:?} {name}:", luts.ms_quality);
                for el in [-20.0f64, -3.0, 0.5, 3.0, 10.0, 30.0, 60.0, 89.0] {
                    let (e, z) = (el.to_radians(), 90f64.to_radians());
                    let d = normalize([e.cos() * z.cos(), e.sin(), e.cos() * z.sin()]);
                    let t = sky_lookup(&a, &view, sun, d);
                    let mut m = [0.0; 3];
                    for i in 0..n {
                        let x = a.sky_sample(d, sun, &mut Rng::new(i, 3, 9), &SkyOptions::default());
                        m = [0, 1, 2].map(|c| m[c] + x[c] / n as f64);
                    }
                    let r: Vec<String> = (0..3).map(|c| format!("{:+.3}", t[c] / m[c] - 1.0)).collect();
                    line += &format!(" | {el}: {}", r.join(" "));
                }
                eprintln!("{line}");
            }
        }
    }
}
