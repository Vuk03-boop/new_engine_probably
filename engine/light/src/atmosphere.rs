//! The physical atmosphere of ADR-0005 and its Monte Carlo sky estimator (the sky reference).
//!
//! # Conventions
//!
//! - Positions are in meters from the planet centre; +Y is up at the observer, who stands at
//!   `(0, ground_radius + observer_altitude, 0)`.
//! - The medium is **discretised** along each ray segment: extinction and scattering are constant
//!   within a march step and taken at the step's midpoint altitude ([`step_bound`] gives the layout).
//!   Free-flight sampling and transmittance are exact for that discretised medium; the step count is
//!   a parameter whose convergence the tests measure.
//! - [`Atmosphere::sky_sample`] estimates the sky radiance seen from the observer in a direction,
//!   **excluding the sun disk** (ADR-0005: the sun is sampled only by next-event estimation), with all
//!   scattering orders and the Lambertian planet ground. Radiance is relative to `E_SUN = 1`.
//! - `gpu/shaders/reference.slang` implements the same functions in f32; the GPU tests compare them.

use std::f64::consts::PI;

use crate::sample::{self, add, dot, mul, normalize, scale, Rng, V3};
use crate::sun::E_SUN;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    pub ground_radius_m: f64,
    pub top_radius_m: f64,
    pub rayleigh_scattering: V3,
    pub rayleigh_scale_height_m: f64,
    pub mie_scattering: f64,
    pub mie_extinction: f64,
    pub mie_scale_height_m: f64,
    pub mie_g: f64,
    pub ozone_absorption: V3,
    pub ozone_center_m: f64,
    pub ozone_half_width_m: f64,
    pub ground_albedo: f64,
    pub observer_altitude_m: f64,
}

impl Default for Atmosphere {
    fn default() -> Atmosphere {
        Atmosphere {
            ground_radius_m: 6_360_000.0,
            top_radius_m: 6_460_000.0,
            rayleigh_scattering: [5.802e-6, 13.558e-6, 33.1e-6],
            rayleigh_scale_height_m: 8000.0,
            mie_scattering: 3.996e-6,
            mie_extinction: 4.440e-6,
            mie_scale_height_m: 1200.0,
            mie_g: 0.8,
            ozone_absorption: [0.650e-6, 1.881e-6, 0.085e-6],
            ozone_center_m: 25_000.0,
            ozone_half_width_m: 15_000.0,
            ground_albedo: 0.3,
            observer_altitude_m: 2.0,
        }
    }
}

/// Coefficients at one altitude, m⁻¹.
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    pub rayleigh: V3,
    pub mie: f64,
    pub extinction: V3,
}

/// Estimator settings for the sky reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyOptions {
    /// March steps per free-flight segment.
    pub steps: u32,
    /// March steps per sun transmittance.
    pub sun_steps: u32,
    /// Hard cap on scattering and ground events per path (ADR-0005: 32).
    pub max_events: u32,
    /// Russian roulette from this event on.
    pub rr_from: u32,
}

impl Default for SkyOptions {
    fn default() -> SkyOptions {
        SkyOptions { steps: 64, sun_steps: 32, max_events: 32, rr_from: 3 }
    }
}

/// Boundary `i` (0..=n) of the march layout for a segment of length `t_end` whose tangent point
/// (closest approach to the planet centre) is at `t_c`. Steps are split at the tangent point
/// (n even) and concentrated quadratically towards the lower, denser end of each part.
pub fn step_bound(i: u32, n: u32, t_end: f64, t_c: f64) -> f64 {
    if t_c > 0.0 && t_c < t_end {
        let half = n / 2;
        if i <= half {
            let u = i as f64 / half as f64;
            t_c * (1.0 - (1.0 - u) * (1.0 - u))
        } else {
            let u = (i - half) as f64 / half as f64;
            t_c + (t_end - t_c) * u * u
        }
    } else if t_c <= 0.0 {
        let u = i as f64 / n as f64;
        t_end * u * u
    } else {
        let u = i as f64 / n as f64;
        t_end * (1.0 - (1.0 - u) * (1.0 - u))
    }
}

fn exp3(t: V3) -> V3 {
    [(-t[0]).exp(), (-t[1]).exp(), (-t[2]).exp()]
}

fn mean(v: V3) -> f64 {
    (v[0] + v[1] + v[2]) / 3.0
}

fn max3(v: V3) -> f64 {
    v[0].max(v[1]).max(v[2])
}

impl Atmosphere {
    pub fn observer(&self) -> V3 {
        [0.0, self.ground_radius_m + self.observer_altitude_m, 0.0]
    }

    pub fn altitude(&self, p: V3) -> f64 {
        dot(p, p).sqrt() - self.ground_radius_m
    }

    pub fn medium(&self, h: f64) -> Medium {
        let r = (-h / self.rayleigh_scale_height_m).exp();
        let m = (-h / self.mie_scale_height_m).exp();
        let o = (1.0 - (h - self.ozone_center_m).abs() / self.ozone_half_width_m).max(0.0);
        let rayleigh = scale(self.rayleigh_scattering, r);
        let mie_ext = self.mie_extinction * m;
        Medium {
            rayleigh,
            mie: self.mie_scattering * m,
            extinction: [0, 1, 2].map(|c| rayleigh[c] + mie_ext + self.ozone_absorption[c] * o),
        }
    }

    /// Distance to the planet ground along `d` from `p`, if the ray goes down into it. A point on the
    /// ground hits it only when moving downwards.
    pub fn ground_hit(&self, p: V3, d: V3) -> Option<f64> {
        let b = dot(p, d);
        let c = dot(p, p) - self.ground_radius_m * self.ground_radius_m;
        let disc = b * b - c;
        (b < 0.0 && disc >= 0.0).then(|| (c / (-b + disc.sqrt())).max(0.0))
    }

    /// Distance to the top of the atmosphere from a point inside it.
    pub fn top_exit(&self, p: V3, d: V3) -> f64 {
        let b = dot(p, d);
        let c = dot(p, p) - self.top_radius_m * self.top_radius_m;
        -b + (b * b - c).max(0.0).sqrt()
    }

    /// The segment a ray from `p` travels inside the atmosphere: (length, ends on the ground).
    pub fn segment(&self, p: V3, d: V3) -> (f64, bool) {
        match self.ground_hit(p, d) {
            Some(t) => (t, true),
            None => (self.top_exit(p, d), false),
        }
    }

    /// Transmittance from `p` along `d` to the top of the atmosphere; 0 when the ground blocks it.
    pub fn transmittance(&self, p: V3, d: V3, steps: u32) -> V3 {
        let (t_end, ground) = self.segment(p, d);
        if ground {
            return [0.0; 3];
        }
        exp3(self.optical_depth(p, d, t_end, steps))
    }

    fn optical_depth(&self, p: V3, d: V3, t_end: f64, steps: u32) -> V3 {
        let t_c = -dot(p, d);
        let mut tau = [0.0; 3];
        let mut t0 = 0.0;
        for i in 1..=steps {
            let t1 = step_bound(i, steps, t_end, t_c);
            let m = self.medium(self.altitude(add(p, scale(d, 0.5 * (t0 + t1)))));
            tau = add(tau, scale(m.extinction, t1 - t0));
            t0 = t1;
        }
        tau
    }

    /// Scattering towards `wo` of light travelling along `wi` per unit length (m⁻¹ sr⁻¹).
    fn in_scatter(&self, m: &Medium, cos_theta: f64) -> V3 {
        let pr = sample::rayleigh_phase(cos_theta);
        let pm = sample::hg_phase(cos_theta, self.mie_g);
        [0, 1, 2].map(|c| m.rayleigh[c] * pr + m.mie * pm)
    }

    /// One Monte Carlo sample of the sky radiance seen from the observer along unit `d`, with the
    /// sun along unit `sun` (sun disk excluded).
    pub fn sky_sample(&self, d: V3, sun: V3, rng: &mut Rng, o: &SkyOptions) -> V3 {
        self.path_sample(self.observer(), d, sun, rng, o)
    }

    /// The same estimator from any point inside the atmosphere.
    pub fn path_sample(&self, start: V3, d: V3, sun: V3, rng: &mut Rng, o: &SkyOptions) -> V3 {
        let mut p = start;
        let mut d = d;
        let mut beta = [1.0; 3];
        let mut l = [0.0; 3];
        for event in 0..o.max_events {
            let (t_end, ground) = self.segment(p, d);
            let t_c = -dot(p, d);
            // Spectral MIS: pick a channel, sample its free-flight distance in the discretised medium.
            let c = ((rng.uniform() * 3.0) as usize).min(2);
            let target = -(1.0 - rng.uniform()).ln();
            let mut tau = [0.0; 3];
            let mut t0 = 0.0;
            let mut hit: Option<(f64, Medium)> = None;
            for i in 1..=o.steps {
                let t1 = step_bound(i, o.steps, t_end, t_c);
                let m = self.medium(self.altitude(add(p, scale(d, 0.5 * (t0 + t1)))));
                let next = tau[c] + m.extinction[c] * (t1 - t0);
                if next >= target {
                    let t = t0 + (target - tau[c]) / m.extinction[c];
                    tau = add(tau, scale(m.extinction, t - t0));
                    hit = Some((t, m));
                    break;
                }
                tau = add(tau, scale(m.extinction, t1 - t0));
                t0 = t1;
            }
            let tr = exp3(tau);
            match hit {
                Some((t, m)) => {
                    let pdf = mean(mul(m.extinction, tr));
                    beta = scale(mul(beta, tr), 1.0 / pdf);
                    p = add(p, scale(d, t));
                    // Next-event estimation of the sun (a point direction in the medium).
                    let ts = self.transmittance(p, sun, o.sun_steps);
                    l = add(l, mul(mul(beta, self.in_scatter(&m, dot(d, sun))), scale(ts, E_SUN)));
                    // New direction: uniform sphere for the Rayleigh lobe, HG for Mie, one-sample mixture.
                    let wr = mean(m.rayleigh) / (mean(m.rayleigh) + m.mie);
                    let nd = if rng.uniform() < wr { sample::uniform_sphere(rng.uniform(), rng.uniform()) } else { sample::sample_hg(d, self.mie_g, rng.uniform(), rng.uniform()) };
                    let cos = dot(d, nd);
                    let pdf_dir = wr / (4.0 * PI) + (1.0 - wr) * sample::hg_phase(cos, self.mie_g);
                    beta = scale(mul(beta, self.in_scatter(&m, cos)), 1.0 / pdf_dir);
                    d = nd;
                }
                None => {
                    beta = scale(mul(beta, tr), 1.0 / mean(tr));
                    if !ground {
                        break;
                    }
                    p = add(p, scale(d, t_end));
                    let n = normalize(p);
                    let rho = self.ground_albedo;
                    let cos_s = dot(n, sun);
                    if cos_s > 0.0 {
                        let ts = self.transmittance(p, sun, o.sun_steps);
                        l = add(l, mul(beta, scale(ts, E_SUN * rho / PI * cos_s)));
                    }
                    d = sample::cosine_hemisphere(n, rng.uniform(), rng.uniform());
                    beta = scale(beta, rho);
                }
            }
            if event + 1 >= o.rr_from {
                let q = max3(beta).min(1.0);
                if q <= 0.0 || rng.uniform() >= q {
                    break;
                }
                beta = scale(beta, 1.0 / q);
            }
        }
        l
    }

    /// Single scattering (plus the directly lit ground) seen from the observer, by deterministic
    /// midpoint quadrature with `steps` steps: the test oracle for `sky_sample` with one event.
    pub fn single_scattering(&self, d: V3, sun: V3, steps: u32, sun_steps: u32) -> V3 {
        let p = self.observer();
        let (t_end, ground) = self.segment(p, d);
        let t_c = -dot(p, d);
        let mut tau = [0.0; 3];
        let mut l = [0.0; 3];
        let mut t0 = 0.0;
        let cos = dot(d, sun);
        for i in 1..=steps {
            let t1 = step_bound(i, steps, t_end, t_c);
            let dt = t1 - t0;
            let q = add(p, scale(d, 0.5 * (t0 + t1)));
            let m = self.medium(self.altitude(q));
            let tv = exp3(add(tau, scale(m.extinction, 0.5 * dt)));
            let ts = self.transmittance(q, sun, sun_steps);
            l = add(l, scale(mul(mul(tv, self.in_scatter(&m, cos)), ts), E_SUN * dt));
            tau = add(tau, scale(m.extinction, dt));
            t0 = t1;
        }
        if ground {
            let g = add(p, scale(d, t_end));
            let n = normalize(g);
            let cos_s = dot(n, sun);
            if cos_s > 0.0 {
                let ts = self.transmittance(g, sun, sun_steps);
                l = add(l, scale(mul(exp3(tau), ts), E_SUN * self.ground_albedo / PI * cos_s));
            }
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::normalize;
    use crate::sun::SunPath;

    /// Straight up from the observer the altitude is the distance, so the optical depth has a closed
    /// form: σH(e^{−a/H} − e^{−top/H}) per exponential layer, plus the whole ozone tent (σ × half-width).
    #[test]
    fn zenith_transmittance_matches_the_closed_form() {
        let a = Atmosphere::default();
        let h0 = a.observer_altitude_m;
        let top = a.top_radius_m - a.ground_radius_m;
        let layer = |s: f64, hs: f64| s * hs * ((-h0 / hs).exp() - (-top / hs).exp());
        let exact: V3 = [0, 1, 2].map(|c| {
            (-(layer(a.rayleigh_scattering[c], a.rayleigh_scale_height_m) + layer(a.mie_extinction, a.mie_scale_height_m) + a.ozone_absorption[c] * a.ozone_half_width_m)).exp()
        });
        for steps in [32, 64, 4096] {
            let t = a.transmittance(a.observer(), [0.0, 1.0, 0.0], steps);
            let err = (0..3).map(|c| (t[c] / exact[c] - 1.0).abs()).fold(0.0, f64::max);
            eprintln!("zenith transmittance, {steps} steps: {t:?} vs {exact:?}, max relative error {err:.2e}");
            assert!(err < if steps >= 4096 { 1e-6 } else { 2e-3 }, "{steps} steps: {err}");
        }
    }

    #[test]
    fn step_layout_covers_the_segment_in_order() {
        for (t_end, t_c) in [(100.0, -5.0), (100.0, 40.0), (100.0, 150.0)] {
            let n = 16;
            assert_eq!(step_bound(0, n, t_end, t_c), 0.0);
            assert!((step_bound(n, n, t_end, t_c) - t_end).abs() < 1e-12);
            for i in 1..=n {
                assert!(step_bound(i, n, t_end, t_c) > step_bound(i - 1, n, t_end, t_c));
            }
        }
    }

    #[test]
    fn ground_blocks_a_set_sun() {
        let a = Atmosphere::default();
        let set = SunPath::default().direction(18.25);
        assert_eq!(a.transmittance(a.observer(), set, 64), [0.0; 3]);
        assert!(a.transmittance(a.observer(), SunPath::default().direction(6.25), 64)[0] > 0.0);
    }

    /// The Monte Carlo estimator limited to one event must converge to the quadrature.
    #[test]
    fn one_event_monte_carlo_matches_single_scattering_quadrature() {
        let a = Atmosphere::default();
        let o = SkyOptions { steps: 128, sun_steps: 64, max_events: 1, rr_from: 1000 };
        let sun = SunPath::default().direction(8.0);
        for d in [normalize([0.0, 1.0, 0.0]), normalize([0.3, 0.2, 0.8]), normalize([-0.9, 0.03, 0.1]), normalize([0.2, -0.3, 0.9])] {
            let q = a.single_scattering(d, sun, 4096, 256);
            let n = 40_000u32;
            let (mut s, mut s2) = ([0.0; 3], [0.0; 3]);
            let mut rng = Rng::new(11, 0, 5);
            for _ in 0..n {
                let x = a.sky_sample(d, sun, &mut rng, &o);
                s = add(s, x);
                s2 = add(s2, mul(x, x));
            }
            for c in 0..3 {
                let m = s[c] / n as f64;
                let se = ((s2[c] / n as f64 - m * m) / n as f64).sqrt();
                let z = (m - q[c]) / se;
                eprintln!("dir {d:?} ch {c}: MC {m:.6e} ± {se:.1e}, quadrature {:.6e}, z {z:.2}, rel {:.2e}", q[c], m / q[c] - 1.0);
                // 4.5 standard errors, plus 0.5% for the 128-step discretisation.
                assert!((m - q[c]).abs() < 4.5 * se + 5e-3 * q[c], "dir {d:?} channel {c}");
            }
        }
    }

    /// The discretisation error of the default steps (64 view, 32 sun), per ADR-0005, two ways:
    /// - noise-free: the single-scattering quadrature at the default steps against 4096 / 512, at
    ///   every reference time including twilight;
    /// - Monte Carlo on the same random streams against 512 / 256 steps, at dawn and midday only. At
    ///   twilight the paths decorrelate and independent streams differ by 8–14% even at 40,000
    ///   samples (`diagnostic_step_dependence_at_twilight`), so that comparison measures noise there.
    #[test]
    fn default_step_count_is_converged_within_one_percent() {
        let a = Atmosphere::default();
        let o = SkyOptions::default();
        let dirs = [normalize([0.0, 1.0, 0.0]), normalize([0.5, 0.1, 0.8]), normalize([-0.9, 0.02, 0.1]), normalize([0.1, -0.2, 0.9])];
        let mut worst_q = 0.0f64;
        for (_, hour) in crate::sun::REFERENCE_TIMES {
            let sun = SunPath::default().direction(hour);
            for d in dirs {
                let (c, f) = (a.single_scattering(d, sun, o.steps, o.sun_steps), a.single_scattering(d, sun, 4096, 512));
                // Both zero (a set sun blocked along the whole path) agree; one zero is an infinite error.
                let rel: V3 = [0, 1, 2].map(|k| if f[k] == 0.0 { if c[k] == 0.0 { 0.0 } else { f64::INFINITY } } else { c[k] / f[k] - 1.0 });
                assert!(rel.iter().all(|r| !r.is_nan()), "NaN at hour {hour} dir {d:?}");
                eprintln!("quadrature hour {hour} dir {d:?}: default vs fine {rel:?}");
                worst_q = rel.iter().fold(worst_q, |w, r| w.max(r.abs()));
            }
        }
        assert!(worst_q < 0.01, "single-scattering discretisation error {worst_q}");
        let fine = SkyOptions { steps: 512, sun_steps: 256, ..SkyOptions::default() };
        let mut worst = 0.0f64;
        for hour in [6.25, 12.0] {
            let sun = SunPath::default().direction(hour);
            for d in [normalize([0.0, 1.0, 0.0]), normalize([0.5, 0.1, 0.8]), normalize([-0.9, 0.02, 0.1]), normalize([0.1, -0.2, 0.9])] {
                let n = 20_000u32;
                let (mut s, mut f) = ([0.0; 3], [0.0; 3]);
                for i in 0..n {
                    s = add(s, a.sky_sample(d, sun, &mut Rng::new(i, 9, 1), &SkyOptions::default()));
                    f = add(f, a.sky_sample(d, sun, &mut Rng::new(i, 9, 1), &fine));
                }
                let rel: V3 = [0, 1, 2].map(|c| s[c] / f[c] - 1.0);
                eprintln!("hour {hour} dir {d:?}: 64 vs 512 steps relative difference {rel:?}");
                worst = rel.iter().fold(worst, |w, r| w.max(r.abs()));
            }
        }
        assert!(worst < 0.01, "64 steps differ from 512 by {worst}");
    }

    /// Diagnostic (on demand): which march drives the twilight step dependence. Same streams, the
    /// view-march steps and the sun-march steps varied separately against a 1024/512 run.
    #[test]
    #[ignore]
    fn diagnostic_step_dependence_at_twilight() {
        let a = Atmosphere::default();
        let sun = SunPath::default().direction(18.25);
        let d = normalize([0.0, 1.0, 0.0]);
        let n = 40_000u32;
        let run = |steps, sun_steps, stream: u32| {
            let o = SkyOptions { steps, sun_steps, ..SkyOptions::default() };
            let mut s = [0.0; 3];
            for i in 0..n {
                s = add(s, a.sky_sample(d, sun, &mut Rng::new(i, stream, 1), &o));
            }
            s
        };
        let fine = run(1024, 512, 9);
        let noise = run(1024, 512, 10);
        eprintln!("noise (other stream, 1024/512): {:?}", [0, 1, 2].map(|c| noise[c] / fine[c] - 1.0));
        for (steps, sun_steps) in [(64, 32), (64, 512), (1024, 32), (128, 64), (128, 128), (256, 64), (64, 128)] {
            let x = run(steps, sun_steps, 9);
            eprintln!("steps {steps} sun_steps {sun_steps}: relative to 1024/512 {:?}", [0, 1, 2].map(|c| x[c] / fine[c] - 1.0));
        }
    }

    /// Multiple scattering adds light, and the midday sky is blue-dominant overhead.
    #[test]
    fn multiple_scattering_adds_light_and_the_sky_is_blue() {
        let a = Atmosphere::default();
        let sun = SunPath::default().direction(12.0);
        let d = normalize([0.0, 1.0, -0.3]);
        let n = 20_000;
        let mut rng = Rng::new(3, 1, 4);
        let (mut one, mut all) = ([0.0; 3], [0.0; 3]);
        let o1 = SkyOptions { max_events: 1, ..SkyOptions::default() };
        for _ in 0..n {
            one = add(one, a.sky_sample(d, sun, &mut rng, &o1));
            all = add(all, a.sky_sample(d, sun, &mut rng, &SkyOptions::default()));
        }
        eprintln!("single {:?}, all orders {:?}", scale(one, 1.0 / n as f64), scale(all, 1.0 / n as f64));
        assert!(all[2] > one[2] * 1.05, "multiple scattering adds more than 5% in blue");
        assert!(all[2] > all[1] && all[1] > all[0], "blue sky");
    }
}
