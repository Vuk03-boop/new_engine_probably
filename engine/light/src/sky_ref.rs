//! The sky correction's data (S-020): the Monte Carlo sky of [`Atmosphere::sky_sample`] (the
//! reference) averaged per direction and sun elevation by the `sky_bake` tool (`gpu` crate) and
//! stored in a versioned file. [`crate::sky::SkyLuts::apply_reference`] turns it into the ratio
//! reference ÷ table that the sky-view table is multiplied by.
//!
//! - **Grid:** [`CORR_W`] azimuths × [`CORR_H`] view zeniths at the sky-view table's unit coordinates
//!   (texel i at i / (n − 1): azimuth relative to the sun, concentrated on its side; zenith
//!   concentrated at the horizon), × [`CORR_S`] sun elevations ([`elevation_deg`]).
//! - **Samples** per texel differ by slice ([`bake_samples`]): the reference's noise grows as the sun
//!   sets (a 1,024-sample pilot: per-texel relative error p95 10–17% above the horizon, 25–100% from
//!   −2° to −6°, and mostly zero estimates below −7°, which is why the grid stops at −6°).
//! - **File** (little-endian): magic `NESKYREF`, version, W, H, S, seed, the four [`SkyOptions`]
//!   fields, the [`fingerprint`] (u64), S elevations (f32), S sample counts (u32), then per texel the
//!   mean and the standard error of the mean (RGB, f32), and an FNV-1a 64 checksum of everything
//!   before it. Any mismatch refuses the file.
//! - The mean is stored, not the ratio, so a change to the table code needs no new bake; a change to
//!   the atmosphere or the estimator settings changes the fingerprint and does.

use std::path::Path;

use crate::atmosphere::{Atmosphere, SkyOptions};
use crate::sample::V3;
use crate::sky::view_from_unit;

pub const CORR_W: usize = 32;
pub const CORR_H: usize = 32;
pub const CORR_S: usize = 33;
pub const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"NESKYREF";
/// The baked file, next to this crate's sources.
pub const DEFAULT_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sky_reference_v1.bin");

/// Sun elevation of slice `k`: −6° to 10° every 1°, then every 5° to 90°.
pub fn elevation_deg(k: usize) -> f64 {
    if k <= 16 {
        -6.0 + k as f64
    } else {
        10.0 + 5.0 * (k - 16) as f64
    }
}

/// The fractional slice of a sun elevation, clamped to the grid (below −6° the −6° slice).
pub fn slice_coord(e_deg: f64) -> f64 {
    let k = if e_deg < 10.0 { e_deg + 6.0 } else { 16.0 + (e_deg - 10.0) / 5.0 };
    k.clamp(0.0, (CORR_S - 1) as f64)
}

/// The bake's samples per texel for slice `k`, from the noise of two pilots (1,024 samples, then
/// 1/512 of an earlier schedule): 2^20 from 5° up, 2^21 from −2° to 4°, then 2^22, 2^24, 2^25 and
/// 2^26 at −3°, −4°, −5° and −6°.
pub fn bake_samples(k: usize) -> u32 {
    match elevation_deg(k) {
        e if e >= 5.0 => 1 << 20,
        e if e >= -2.0 => 1 << 21,
        e if e >= -3.0 => 1 << 22,
        e if e >= -4.0 => 1 << 24,
        e if e >= -5.0 => 1 << 25,
        _ => 1 << 26,
    }
}

pub fn index(x: usize, y: usize, k: usize) -> usize {
    (k * CORR_H + y) * CORR_W + x
}

/// Texel (x, y, k): the unit view direction (+Y up) and the sun direction (azimuth along +X).
pub fn texel(a: &Atmosphere, x: usize, y: usize, k: usize) -> (V3, V3) {
    let (mu, cos_phi) = view_from_unit(a, x as f64 / (CORR_W - 1) as f64, y as f64 / (CORR_H - 1) as f64);
    let sin_v = (1.0 - mu * mu).max(0.0).sqrt();
    let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
    let e = elevation_deg(k).to_radians();
    ([sin_v * cos_phi, mu, sin_v * sin_phi], [e.cos(), e.sin(), 0.0])
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// FNV-1a 64 over every field of the atmosphere and the estimator settings.
pub fn fingerprint(a: &Atmosphere, o: &SkyOptions) -> u64 {
    let mut b = Vec::new();
    let f = [
        a.ground_radius_m,
        a.top_radius_m,
        a.rayleigh_scattering[0],
        a.rayleigh_scattering[1],
        a.rayleigh_scattering[2],
        a.rayleigh_scale_height_m,
        a.mie_scattering,
        a.mie_extinction,
        a.mie_scale_height_m,
        a.mie_g,
        a.ozone_absorption[0],
        a.ozone_absorption[1],
        a.ozone_absorption[2],
        a.ozone_center_m,
        a.ozone_half_width_m,
        a.ground_albedo,
        a.observer_altitude_m,
    ];
    for x in f {
        b.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    for x in [o.steps, o.sun_steps, o.max_events, o.rr_from] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fnv(&b)
}

/// The baked reference sky.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyReference {
    /// Samples per texel, per slice.
    pub samples: Vec<u32>,
    pub seed: u32,
    pub options: SkyOptions,
    pub fingerprint: u64,
    /// Per texel ([`index`]): the mean radiance and its standard error.
    pub mean: Vec<V3>,
    pub se: Vec<V3>,
}

impl SkyReference {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        let o = &self.options;
        for x in [VERSION, CORR_W as u32, CORR_H as u32, CORR_S as u32, self.seed, o.steps, o.sun_steps, o.max_events, o.rr_from] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&self.fingerprint.to_le_bytes());
        for k in 0..CORR_S {
            b.extend_from_slice(&(elevation_deg(k) as f32).to_le_bytes());
        }
        assert_eq!(self.samples.len(), CORR_S);
        for n in &self.samples {
            b.extend_from_slice(&n.to_le_bytes());
        }
        for (m, s) in self.mean.iter().zip(&self.se) {
            for x in m.iter().chain(s) {
                b.extend_from_slice(&(*x as f32).to_le_bytes());
            }
        }
        let sum = fnv(&b);
        b.extend_from_slice(&sum.to_le_bytes());
        b
    }

    /// Parses a file and checks it against the atmosphere and estimator settings in use.
    pub fn from_bytes(b: &[u8], a: &Atmosphere, o: &SkyOptions) -> Result<SkyReference, String> {
        let n = CORR_W * CORR_H * CORR_S;
        let len = 8 + 9 * 4 + 8 + CORR_S * 8 + n * 24 + 8;
        if b.len() != len {
            return Err(format!("sky reference: {} bytes, expected {len}", b.len()));
        }
        if &b[..8] != MAGIC {
            return Err("sky reference: wrong magic".into());
        }
        let (body, tail) = b.split_at(len - 8);
        if fnv(body) != u64::from_le_bytes(tail.try_into().unwrap()) {
            return Err("sky reference: checksum mismatch".into());
        }
        let u = |i: usize| u32::from_le_bytes(b[8 + 4 * i..12 + 4 * i].try_into().unwrap());
        if u(0) != VERSION {
            return Err(format!("sky reference: version {}, expected {VERSION}", u(0)));
        }
        if (u(1), u(2), u(3)) != (CORR_W as u32, CORR_H as u32, CORR_S as u32) {
            return Err(format!("sky reference: grid {}x{}x{}, expected {CORR_W}x{CORR_H}x{CORR_S}", u(1), u(2), u(3)));
        }
        let options = SkyOptions { steps: u(5), sun_steps: u(6), max_events: u(7), rr_from: u(8) };
        if options != *o {
            return Err(format!("sky reference: estimator {options:?}, in use {o:?}"));
        }
        let fp = u64::from_le_bytes(b[44..52].try_into().unwrap());
        if fp != fingerprint(a, o) {
            return Err("sky reference: baked for a different atmosphere (fingerprint mismatch)".into());
        }
        let f = |at: usize| f32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as f64;
        for k in 0..CORR_S {
            if f(52 + 4 * k) != elevation_deg(k) as f32 as f64 {
                return Err(format!("sky reference: slice {k} at {}°, expected {}°", f(52 + 4 * k), elevation_deg(k)));
            }
        }
        let samples: Vec<u32> = (0..CORR_S).map(|k| u(11 + CORR_S + k)).collect();
        if samples.iter().any(|&n| n < 2) {
            return Err("sky reference: fewer than 2 samples in a slice".into());
        }
        let base = 52 + 8 * CORR_S;
        let (mut mean, mut se) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let t = base + 24 * i;
            mean.push([0, 1, 2].map(|c| f(t + 4 * c)));
            se.push([0, 1, 2].map(|c| f(t + 12 + 4 * c)));
        }
        if mean.iter().chain(&se).flatten().any(|x| !x.is_finite() || *x < 0.0) {
            return Err("sky reference: a negative or non-finite value".into());
        }
        Ok(SkyReference { samples, seed: u(4), options, fingerprint: fp, mean, se })
    }

    pub fn load(path: &Path, a: &Atmosphere, o: &SkyOptions) -> Result<SkyReference, String> {
        let b = std::fs::read(path).map_err(|e| format!("sky reference {}: {e}", path.display()))?;
        Self::from_bytes(&b, a, o)
    }

    /// The baked file ([`DEFAULT_FILE`]) for the default estimator settings.
    pub fn load_default(a: &Atmosphere) -> Result<SkyReference, String> {
        Self::load(Path::new(DEFAULT_FILE), a, &SkyOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::{view_uv, SkyLuts, VIEW_H, VIEW_W};

    fn synthetic(a: &Atmosphere) -> SkyReference {
        let n = CORR_W * CORR_H * CORR_S;
        let o = SkyOptions::default();
        let mean = (0..n).map(|i| [1e-3 * (1 + i % 7) as f64, 2e-3, 3e-3]).collect();
        let se = (0..n).map(|i| [1e-6 * (i % 3) as f64, 0.0, 1e-6]).collect();
        SkyReference { samples: (0..CORR_S).map(|k| 64 + k as u32).collect(), seed: 7, options: o, fingerprint: fingerprint(a, &o), mean, se }
    }

    /// K2: the format round-trips (values are stored as f32) and every kind of mismatch is refused.
    #[test]
    fn file_round_trips_and_refuses_mismatches() {
        let a = Atmosphere::default();
        let o = SkyOptions::default();
        let r = synthetic(&a);
        let b = r.to_bytes();
        let back = SkyReference::from_bytes(&b, &a, &o).unwrap();
        let f32s = |v: &Vec<V3>| v.iter().map(|c| c.map(|x| x as f32 as f64)).collect::<Vec<_>>();
        assert_eq!(back.mean, f32s(&r.mean));
        assert_eq!(back.se, f32s(&r.se));
        assert_eq!((&back.samples, back.seed, back.options, back.fingerprint), (&r.samples, r.seed, r.options, r.fingerprint));

        let refused = |b: &[u8], a: &Atmosphere, o: &SkyOptions| SkyReference::from_bytes(b, a, o).unwrap_err();
        let mut m = b.clone();
        m[0] = b'X';
        eprintln!("{}", refused(&m, &a, &o));
        eprintln!("{}", refused(&b[..b.len() - 1], &a, &o));
        let mut c = b.clone();
        let mid = c.len() / 2;
        c[mid] ^= 1;
        assert!(refused(&c, &a, &o).contains("checksum"));
        // A version or grid change with a valid checksum.
        let reseal = |mut v: Vec<u8>| {
            let n = v.len() - 8;
            let s = fnv(&v[..n]);
            v[n..].copy_from_slice(&s.to_le_bytes());
            v
        };
        let mut v = b.clone();
        v[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(refused(&reseal(v), &a, &o).contains("version"));
        let mut g = b.clone();
        g[12..16].copy_from_slice(&31u32.to_le_bytes());
        assert!(refused(&reseal(g), &a, &o).contains("grid"));
        let other = Atmosphere { mie_g: 0.76, ..a };
        assert!(refused(&b, &other, &o).contains("fingerprint"));
        let fewer = SkyOptions { steps: 32, ..o };
        assert!(refused(&b, &a, &fewer).contains("estimator"));
        let mut neg = r.clone();
        neg.mean[5][1] = -1.0;
        assert!(refused(&neg.to_bytes(), &a, &o).contains("negative"));
    }

    /// The grid's directions land on the grid's coordinates in the sky-view lookup: texel (x, y, k)
    /// maps to unit (x / (W − 1), y / (H − 1)) and slice k, so the bake and the correction agree.
    #[test]
    fn grid_directions_map_to_their_texels() {
        let a = Atmosphere::default();
        let unit = |u: f64, n: usize| (u - 0.5 / n as f64) / (1.0 - 1.0 / n as f64);
        for k in 0..CORR_S {
            for y in 0..CORR_H {
                for x in 0..CORR_W {
                    let (d, sun) = texel(&a, x, y, k);
                    assert!((crate::sample::dot(d, d) - 1.0).abs() < 1e-12);
                    let (u, v) = view_uv(&a, sun, d);
                    let (ux, uy) = (unit(u, VIEW_W), unit(v, VIEW_H));
                    // With the view or the sun vertical, azimuth has no meaning (the lookup takes x = 0).
                    let pole = d[1].abs() > 1.0 - 1e-12 || sun[1] > 1.0 - 1e-12;
                    assert!(pole || (ux - x as f64 / (CORR_W - 1) as f64).abs() < 1e-6, "x {x} y {y} k {k}: {ux}");
                    assert!((uy - y as f64 / (CORR_H - 1) as f64).abs() < 1e-6, "x {x} y {y} k {k}: {uy}");
                    assert!((slice_coord(sun[1].asin().to_degrees()) - k as f64).abs() < 1e-9);
                }
            }
        }
    }

    /// K2: uncorrected tables have a ratio of exactly 1; a reference equal to the table gives ratio 1
    /// at the grid points, and the correction interpolates between them.
    #[test]
    fn correction_is_one_when_uncorrected_and_matches_at_grid_points() {
        let a = Atmosphere::default();
        let mut luts = SkyLuts::new(a);
        assert!(!luts.corrected);
        assert!(luts.correction.iter().flatten().all(|&x| x == 1.0));
        // A reference twice the table value at every texel (except where the table is ~0).
        let r_obs = a.ground_radius_m + a.observer_altitude_m;
        let mut r = synthetic(&a);
        for k in 0..CORR_S {
            for y in 0..CORR_H {
                for x in 0..CORR_W {
                    let (d, sun) = texel(&a, x, y, k);
                    let t = luts.integrate_view(r_obs, d[1], sun[1], crate::sample::dot(d, sun));
                    r.mean[index(x, y, k)] = t.map(|c| 2.0 * c);
                }
            }
        }
        let stats = luts.apply_reference(&r).unwrap();
        eprintln!("{stats:?}");
        assert!(luts.corrected);
        for k in [0, 12, 32] {
            for (x, y) in [(0, 0), (5, 14), (31, 31)] {
                let (d, sun) = texel(&a, x, y, k);
                let c = luts.correction_at(x as f64 / 31.0, y as f64 / 31.0, sun[1]);
                let t = luts.integrate_view(r_obs, d[1], sun[1], crate::sample::dot(d, sun));
                for ch in 0..3 {
                    // 2 within the f32 rounding of the stored mean; 1 where the table is ~0.
                    let want = if t[ch] < 1e-9 { 1.0 } else { 2.0 };
                    assert!((c[ch] - want).abs() < 1e-6, "({x},{y},{k}) ch {ch}: {}", c[ch]);
                }
            }
        }
        let other = Atmosphere { ground_albedo: 0.2, ..a };
        let mut wrong = SkyLuts::new(other);
        assert!(wrong.apply_reference(&r).unwrap_err().contains("fingerprint"));
    }

    /// K1: the baked file's noise. Relative standard error of the mean per texel and channel (mean
    /// above 0): p95 ≤ 0.5% for sun elevations ≥ 0°, ≤ 1% below. Fails (does not skip) without the
    /// file.
    #[test]
    fn baked_reference_noise_is_within_budget() {
        let a = Atmosphere::default();
        let r = SkyReference::load_default(&a).expect("the baked sky reference (run the sky_bake tool)");
        let (mut up, mut down) = (Vec::new(), Vec::new());
        for k in 0..CORR_S {
            for i in 0..CORR_W * CORR_H {
                let t = k * CORR_W * CORR_H + i;
                for c in 0..3 {
                    if r.mean[t][c] > 0.0 {
                        let e = r.se[t][c] / r.mean[t][c];
                        if elevation_deg(k) >= 0.0 { up.push(e) } else { down.push(e) }
                    }
                }
            }
        }
        for v in [&mut up, &mut down] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let p = |v: &Vec<f64>, q: f64| v[((v.len() - 1) as f64 * q) as usize];
        eprintln!("sun >= 0°: {} values, relative se p50 {:.5} p95 {:.5} max {:.5}", up.len(), p(&up, 0.5), p(&up, 0.95), up.last().unwrap());
        eprintln!("sun < 0°: {} values, relative se p50 {:.5} p95 {:.5} max {:.5}", down.len(), p(&down, 0.5), p(&down, 0.95), down.last().unwrap());
        eprintln!("samples per texel by slice {:?}, seed {:#x}", r.samples, r.seed);
        assert!(p(&up, 0.95) <= 0.005 && p(&down, 0.95) <= 0.01);
    }
}
