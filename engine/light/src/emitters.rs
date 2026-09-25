//! Phase 4A: the emitter table (ADR-0005 Amendment 3, ADR-0003 Amendment 3).
//!
//! - One emitter is one surface-mesh quad whose material emits: a one-sided Lambertian rectangle on
//!   an integer voxel plane, emitting along the face's outward normal.
//! - The table is sorted by geometry (face, plane, u0, v0), so the same quads give the same table
//!   however they were grouped (bricks, regions of any size) and the CPU and the GPU agree on every
//!   index.
//! - Selection is proportional to power Φ = π · A · Y(L_e) through an alias table with 24-bit
//!   thresholds. [`Emitter::pdf`] is the probability the sampler *realizes*, computed exactly from the
//!   quantized table, so the estimator stays unbiased.
//! - Each emitter keeps its identity ([`EmitterId`]: source key, quad index, source snapshot); the
//!   table keeps the snapshot it was built for.

use std::f64::consts::PI;

use world::{MaterialId, MaterialRegistry};

use crate::sample::{Rng, V3};

/// Rec.709 luminance weights (Y of linear RGB).
pub const LUMINANCE: V3 = [0.2126, 0.7152, 0.0722];

/// Coin resolution of the alias table: thresholds are integers in `0..=COIN_ONE`, compared with the
/// top 24 bits of `next_u32`.
pub const COIN_ONE: u32 = 1 << 24;

pub fn luminance(c: V3) -> f64 {
    LUMINANCE[0] * c[0] + LUMINANCE[1] * c[1] + LUMINANCE[2] * c[2]
}

/// Where an emitter came from: the source's key (a region or brick key, in its own units), the
/// quad's index in that source, and the source's snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EmitterId {
    pub key: [i32; 3],
    pub quad: u32,
    pub snapshot: u64,
}

/// A quad in world voxel coordinates, as a source hands it to the table (the ADR-0004 convention:
/// `face = axis * 2 + positive`, the quad lies on `axis = plane` and covers `[u0, u1) × [v0, v1)` on
/// axes `(axis + 1) % 3` and `(axis + 2) % 3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmitterQuad {
    pub id: EmitterId,
    pub material: MaterialId,
    pub face: u8,
    pub plane: i32,
    pub u0: i32,
    pub v0: i32,
    pub u1: i32,
    pub v1: i32,
}

impl EmitterQuad {
    /// A quad given in source-local coordinates (`u8`, as `derived::Quad` and the GPU region quads
    /// store them) whose source starts at voxel `origin`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_local(id: EmitterId, material: MaterialId, face: u8, plane: u8, u0: u8, v0: u8, u1: u8, v1: u8, origin: [i32; 3]) -> EmitterQuad {
        let a = (face / 2) as usize;
        let (ua, va) = ((a + 1) % 3, (a + 2) % 3);
        EmitterQuad {
            id,
            material,
            face,
            plane: plane as i32 + origin[a],
            u0: u0 as i32 + origin[ua],
            v0: v0 as i32 + origin[va],
            u1: u1 as i32 + origin[ua],
            v1: v1 as i32 + origin[va],
        }
    }
}

/// One emitter, ready to sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    pub id: EmitterId,
    pub material: MaterialId,
    pub face: u8,
    /// Corner (u0, v0) on the plane, voxels.
    pub p0: V3,
    /// Edge along u, and along v (axis-aligned, lengths u1 − u0 and v1 − v0).
    pub eu: V3,
    pub ev: V3,
    /// Outward unit normal.
    pub normal: V3,
    /// Voxel units².
    pub area: f64,
    pub radiance: V3,
    /// π · A · Y(L_e), relative.
    pub power: f64,
    /// Realized selection probability.
    pub pdf: f64,
    /// Alias entry: this index is kept when the coin is below `threshold`, else `alias` is chosen.
    pub threshold: u32,
    pub alias: u32,
}

impl Emitter {
    /// The point at (u, v) ∈ [0, 1)² on the quad.
    pub fn point(&self, u: f64, v: f64) -> V3 {
        [0, 1, 2].map(|c| self.p0[c] + u * self.eu[c] + v * self.ev[c])
    }

    fn geometry_key(&self) -> (u8, i64, i64, i64) {
        let a = (self.face / 2) as usize;
        let (ua, va) = ((a + 1) % 3, (a + 2) % 3);
        (self.face, self.p0[a] as i64, self.p0[ua] as i64, self.p0[va] as i64)
    }
}

/// The emitters of one scene snapshot, plus the emitted radiance of every material (for emission at
/// the primary hit).
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterTable {
    pub snapshot: u64,
    pub emitters: Vec<Emitter>,
    /// Emitted radiance per material id.
    pub emission: Vec<V3>,
    pub total_power: f64,
}

/// A table read against a snapshot it was not built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaleTable {
    pub table: u64,
    pub expected: u64,
}

/// Emitted radiance per material id; refuses a channel that is negative or not finite.
pub fn emission(reg: &MaterialRegistry) -> Result<Vec<V3>, String> {
    reg.iter()
        .map(|(id, def)| {
            let e = def.params.emissive;
            if e.iter().all(|x| x.is_finite() && *x >= 0.0) {
                Ok(e.map(|x| x as f64))
            } else {
                Err(format!("material {} ({}) has emissive {e:?}: every channel must be finite and >= 0", id.raw(), def.name))
            }
        })
        .collect()
}

impl EmitterTable {
    /// No emitters; emission of every material still known (for the primary hit).
    pub fn empty(reg: &MaterialRegistry, snapshot: u64) -> Result<EmitterTable, String> {
        Ok(EmitterTable { snapshot, emitters: Vec::new(), emission: emission(reg)?, total_power: 0.0 })
    }

    /// Builds the table from every quad of a snapshot; quads of non-emissive materials are skipped.
    pub fn build(reg: &MaterialRegistry, snapshot: u64, quads: impl IntoIterator<Item = EmitterQuad>) -> Result<EmitterTable, String> {
        let emission = emission(reg)?;
        let mut emitters = Vec::new();
        for q in quads {
            let Some(&l) = emission.get(q.material.raw() as usize) else {
                return Err(format!("quad {:?} has unknown material {}", q.id, q.material.raw()));
            };
            let y = luminance(l);
            if y <= 0.0 {
                continue;
            }
            if q.face > 5 || q.u1 <= q.u0 || q.v1 <= q.v0 {
                return Err(format!("degenerate quad {q:?}"));
            }
            let a = (q.face / 2) as usize;
            let (ua, va) = ((a + 1) % 3, (a + 2) % 3);
            let mut p0 = [0.0; 3];
            p0[a] = q.plane as f64;
            p0[ua] = q.u0 as f64;
            p0[va] = q.v0 as f64;
            let (mut eu, mut ev, mut normal) = ([0.0; 3], [0.0; 3], [0.0; 3]);
            eu[ua] = (q.u1 - q.u0) as f64;
            ev[va] = (q.v1 - q.v0) as f64;
            normal[a] = if q.face % 2 == 1 { 1.0 } else { -1.0 };
            let area = eu[ua] * ev[va];
            emitters.push(Emitter { id: q.id, material: q.material, face: q.face, p0, eu, ev, normal, area, radiance: l, power: PI * area * y, pdf: 0.0, threshold: 0, alias: 0 });
        }
        emitters.sort_by_key(Emitter::geometry_key);
        if let Some(w) = emitters.windows(2).find(|w| w[0].geometry_key() == w[1].geometry_key()) {
            return Err(format!("two emitters share a face position: {:?} and {:?}", w[0].id, w[1].id));
        }
        let total_power = emitters.iter().map(|e| e.power).sum();
        let mut t = EmitterTable { snapshot, emitters, emission, total_power };
        t.build_alias();
        Ok(t)
    }

    /// Vose's alias method, then thresholds quantized to 24 bits and the realized probabilities.
    fn build_alias(&mut self) {
        let n = self.emitters.len();
        if n == 0 {
            return;
        }
        let mut q: Vec<f64> = self.emitters.iter().map(|e| e.power * n as f64 / self.total_power).collect();
        let mut alias: Vec<u32> = (0..n as u32).collect();
        let (mut small, mut large): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| q[i] < 1.0);
        while let (Some(s), Some(&l)) = (small.pop(), large.last()) {
            alias[s] = l as u32;
            q[l] -= 1.0 - q[s];
            if q[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        // Leftovers (only rounding) keep themselves.
        for i in small.into_iter().chain(large) {
            q[i] = 1.0;
            alias[i] = i as u32;
        }
        let one = COIN_ONE as f64;
        let mut count = vec![0u64; n];
        for i in 0..n {
            // At least 1, so an emitter with power > 0 is never impossible to pick.
            let th = if alias[i] as usize == i { COIN_ONE } else { ((q[i] * one).round() as u32).clamp(1, COIN_ONE) };
            self.emitters[i].threshold = th;
            self.emitters[i].alias = alias[i];
            count[i] += th as u64;
            count[alias[i] as usize] += (COIN_ONE - th) as u64;
        }
        let denom = n as f64 * one;
        for (e, c) in self.emitters.iter_mut().zip(count) {
            e.pdf = c as f64 / denom;
        }
    }

    pub fn len(&self) -> usize {
        self.emitters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }

    /// Refuses a table built for another snapshot.
    pub fn check(&self, snapshot: u64) -> Result<(), StaleTable> {
        if self.snapshot == snapshot { Ok(()) } else { Err(StaleTable { table: self.snapshot, expected: snapshot }) }
    }

    /// Emitted radiance of a material (0 for an unknown id).
    pub fn emission_of(&self, m: MaterialId) -> V3 {
        self.emission.get(m.raw() as usize).copied().unwrap_or([0.0; 3])
    }

    /// Picks an emitter index with probability [`Emitter::pdf`]. Draws at least two `next_u32`.
    pub fn select(&self, rng: &mut Rng) -> usize {
        let i = uniform_index(self.emitters.len() as u32, rng) as usize;
        let e = &self.emitters[i];
        if (rng.next_u32() >> 8) < e.threshold { i } else { e.alias as usize }
    }
}

/// Whether the street's lights are on with the sun at `sun_dir` (Phase 4 decision 3): on while the
/// sun is below the horizon (elevation < 0°), so daytime stays exactly as accepted in M3.
pub fn lights_on(sun_dir: V3) -> bool {
    sun_dir[1] < 0.0
}

/// Below this solid angle (sr) an emitter is sampled by area instead of by solid angle. In f32 (the
/// GPU) the solid angle Σgᵢ − 2π carries about 10⁻⁶ sr of cancellation error, ≤ 10⁻⁴ relative here;
/// and a quad this small in solid angle is far enough away that area sampling's 1/d² is bounded.
pub const SOLID_ANGLE_MIN: f64 = 1e-2;

/// A rectangle as seen from a point, for solid-angle sampling (Ureña, Fajardo and King 2013, "An
/// area-preserving parametrization for spherical rectangles"). `gpu/shaders/reference.slang` has the
/// same code in f32.
#[derive(Clone, Copy, Debug)]
pub struct SphericalRect {
    o: V3,
    x: V3,
    y: V3,
    z: V3,
    z0: f64,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    b0: f64,
    b1: f64,
    k: f64,
    /// The solid angle.
    pub solid_angle: f64,
}

impl SphericalRect {
    /// `e` seen from `o`; `o` must lie strictly in front of the emitter.
    pub fn new(e: &Emitter, o: V3) -> SphericalRect {
        use crate::sample::{cross, dot, normalize, scale, sub};
        let (exl, eyl) = (dot(e.eu, e.eu).sqrt(), dot(e.ev, e.ev).sqrt());
        let (x, y) = (scale(e.eu, 1.0 / exl), scale(e.ev, 1.0 / eyl));
        let mut z = cross(x, y);
        let d = sub(e.p0, o);
        let mut z0 = dot(d, z);
        if z0 > 0.0 {
            z = scale(z, -1.0);
            z0 = -z0;
        }
        let (x0, y0) = (dot(d, x), dot(d, y));
        let (x1, y1) = (x0 + exl, y0 + eyl);
        let n0 = normalize([0.0, z0, -y0]);
        let n1 = normalize([-z0, 0.0, x1]);
        let n2 = normalize([0.0, -z0, y1]);
        let n3 = normalize([z0, 0.0, -x0]);
        let g = |a: V3, b: V3| (-dot(a, b)).clamp(-1.0, 1.0).acos();
        let (g0, g1, g2, g3) = (g(n0, n1), g(n1, n2), g(n2, n3), g(n3, n0));
        let k = 2.0 * PI - g2 - g3;
        SphericalRect { o, x, y, z, z0, x0, y0, x1, y1, b0: n0[2], b1: n2[2], k, solid_angle: g0 + g1 - k }
    }

    /// The point for (u, v) ∈ [0, 1)²; uniform in solid angle.
    pub fn sample(&self, u: f64, v: f64) -> V3 {
        let au = u * self.solid_angle + self.k;
        let fu = (au.cos() * self.b0 - self.b1) / au.sin();
        let cu = ((fu * fu + self.b0 * self.b0).sqrt().recip() * if fu > 0.0 { 1.0 } else { -1.0 }).clamp(-1.0, 1.0);
        let xu = (-(cu * self.z0) / (1.0 - cu * cu).max(0.0).sqrt()).clamp(self.x0, self.x1);
        let d = (xu * xu + self.z0 * self.z0).sqrt();
        let h0 = self.y0 / (d * d + self.y0 * self.y0).sqrt();
        let h1 = self.y1 / (d * d + self.y1 * self.y1).sqrt();
        let hv = h0 + v * (h1 - h0);
        let yv = if hv * hv < 1.0 - 1e-6 { hv * d / (1.0 - hv * hv).sqrt() } else { self.y1 };
        [0, 1, 2].map(|c| self.o[c] + xu * self.x[c] + yv * self.y[c] + self.z0 * self.z[c])
    }
}

/// Uniform in `0..n` (`n ≥ 1`), exactly: Lemire's multiply-shift with rejection. The GPU does the
/// same with a 32 × 32 → 64-bit product built from 16-bit halves.
pub fn uniform_index(n: u32, rng: &mut Rng) -> u32 {
    assert!(n > 0, "uniform_index needs n >= 1");
    let mut m = rng.next_u32() as u64 * n as u64;
    if (m as u32) < n {
        let t = n.wrapping_neg() % n;
        while (m as u32) < t {
            m = rng.next_u32() as u64 * n as u64;
        }
    }
    (m >> 32) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sun::{SunPath, NIGHT_TIMES, REFERENCE_TIMES};
    use world::MaterialParams;

    /// C7 (4A part 2): the lights are off whenever the sun is up and on once it has set.
    #[test]
    fn lights_follow_the_sun() {
        let path = SunPath::default();
        let on = |h: f64| lights_on(path.direction(h));
        for (name, hour) in REFERENCE_TIMES {
            assert_eq!(on(hour), name == "twilight", "{name}");
        }
        let got: Vec<(&str, bool)> = NIGHT_TIMES.iter().map(|&(n, h)| (n, on(h))).collect();
        assert_eq!(got, [("dusk", false), ("blue_hour", true), ("night", true)]);
        // The switch sits at sunset (elevation 0°, 18 h at the equinox).
        assert!(!on(17.99) && on(18.01));
    }

    fn reg() -> (MaterialRegistry, MaterialId, MaterialId, MaterialId) {
        let mut r = MaterialRegistry::new();
        let dark = r.register("dark", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
        let dim = r.register("dim", MaterialParams { base_color: [0.5; 3], emissive: [0.001, 0.001, 0.001] }).unwrap();
        let hot = r.register("hot", MaterialParams { base_color: [0.5; 3], emissive: [10.0, 5.0, 1.0] }).unwrap();
        (r, dark, dim, hot)
    }

    fn quad(i: u32, m: MaterialId, face: u8, plane: i32, u0: i32, v0: i32, w: i32) -> EmitterQuad {
        EmitterQuad { id: EmitterId { key: [0; 3], quad: i, snapshot: 1 }, material: m, face, plane, u0, v0, u1: u0 + w, v1: v0 + 1 }
    }

    /// C1: a 50-emitter table whose powers span 10⁴ : 1.
    #[test]
    fn alias_selection_matches_the_realized_probabilities() {
        let (r, _, dim, hot) = reg();
        let quads: Vec<_> = (0..50).map(|i| quad(i, if i % 3 == 0 { dim } else { hot }, 3, 0, 0, 2 * i as i32, 1 + (i as i32 % 7))).collect();
        let t = EmitterTable::build(&r, 1, quads).unwrap();
        assert_eq!(t.len(), 50);
        let psum: f64 = t.emitters.iter().map(|e| e.pdf).sum();
        assert!((psum - 1.0).abs() < 1e-12, "{psum}");
        let (pmin, pmax) = t.emitters.iter().fold((1.0f64, 0.0f64), |(a, b), e| (a.min(e.power), b.max(e.power)));
        assert!(pmax / pmin > 1e4, "power range {}", pmax / pmin);
        for e in &t.emitters {
            let ideal = e.power / t.total_power;
            assert!(e.pdf > 0.0);
            if ideal >= 1e-3 {
                assert!((e.pdf / ideal - 1.0).abs() < 1e-5, "{:?}: {} vs {ideal}", e.id, e.pdf);
            }
        }
        let mut rng = Rng::new(3, 4, 5);
        let k = 1_000_000;
        let mut counts = vec![0u64; t.len()];
        for _ in 0..k {
            counts[t.select(&mut rng)] += 1;
        }
        let chi2: f64 = counts.iter().zip(&t.emitters).map(|(&c, e)| (c as f64 - k as f64 * e.pdf).powi(2) / (k as f64 * e.pdf)).sum();
        // Negative control: the index draw alone (no coin, no alias) must fail the same test.
        let mut bad = vec![0u64; t.len()];
        for _ in 0..k {
            bad[uniform_index(t.len() as u32, &mut rng) as usize] += 1;
        }
        let chi2_bad: f64 = bad.iter().zip(&t.emitters).map(|(&c, e)| (c as f64 - k as f64 * e.pdf).powi(2) / (k as f64 * e.pdf)).sum();
        eprintln!("alias chi2 = {chi2:.2} (df 49, limit 85.35); negative control {chi2_bad:.0}");
        assert!(chi2 < 85.35, "chi2 {chi2}");
        assert!(chi2_bad > 85.35, "the control must fail: {chi2_bad}");
    }

    #[test]
    fn tiny_emitters_are_never_impossible() {
        let (mut r, _, _, hot) = reg();
        let faint = r.register("faint", MaterialParams { base_color: [0.5; 3], emissive: [1e-12, 1e-12, 1e-12] }).unwrap();
        let t = EmitterTable::build(&r, 1, [quad(0, hot, 3, 0, 0, 0, 64), quad(1, faint, 3, 0, 0, 2, 1)]).unwrap();
        assert!(t.emitters.iter().all(|e| e.pdf > 0.0), "{:?}", t.emitters.iter().map(|e| e.pdf).collect::<Vec<_>>());
    }

    #[test]
    fn uniform_index_is_uniform_and_in_range() {
        let mut rng = Rng::new(1, 1, 1);
        let n = 7u32;
        let mut c = [0u32; 7];
        for _ in 0..70_000 {
            let i = uniform_index(n, &mut rng);
            c[i as usize] += 1;
        }
        // Expected 10,000 each, sd ~93.
        assert!(c.iter().all(|&x| (x as i64 - 10_000).abs() < 500), "{c:?}");
        assert_eq!(uniform_index(1, &mut rng), 0);
    }

    #[test]
    fn geometry_and_order_are_canonical() {
        let (r, dark, _, hot) = reg();
        let a = quad(0, hot, 2, 5, 1, 1, 3);
        let b = quad(1, hot, 3, 4, 0, 0, 2);
        let c = quad(2, dark, 3, 4, 0, 5, 2);
        let t1 = EmitterTable::build(&r, 9, [a, b, c]).unwrap();
        let t2 = EmitterTable::build(&r, 9, [b, c, a]).unwrap();
        assert_eq!(t1, t2);
        assert_eq!(t1.len(), 2, "the dark quad is not an emitter");
        let down = t1.emitters.iter().find(|e| e.face == 2).unwrap();
        assert_eq!(down.normal, [0.0, -1.0, 0.0]);
        assert_eq!(down.area, 3.0);
        assert_eq!(down.point(0.0, 0.0), [1.0, 5.0, 1.0]);
        // For a y face, u runs along z (u0 = 1, 3 long) and v along x (v0 = 1, 1 long).
        assert_eq!(down.point(1.0, 1.0), [2.0, 5.0, 4.0]);
    }

    /// The spherical rectangle's solid angle against the exact triangle formula (Van Oosterom and
    /// Strackee 1983), and its samples stay on the quad and are uniform in solid angle: their mean
    /// direction matches the exact ∫ω dω of the polygon (½ Σ θᵢ ûᵢ over its edges).
    #[test]
    fn spherical_rectangle_is_uniform_in_solid_angle() {
        use crate::sample::{cross, dot, normalize, sub};
        let (r, _, _, hot) = reg();
        let t = EmitterTable::build(&r, 1, [EmitterQuad { id: EmitterId { key: [0; 3], quad: 0, snapshot: 1 }, material: hot, face: 2, plane: 10, u0: 2, v0: -3, u1: 7, v1: 1 }]).unwrap();
        let e = &t.emitters[0];
        let corners = [e.point(0.0, 0.0), e.point(1.0, 0.0), e.point(1.0, 1.0), e.point(0.0, 1.0)];
        for o in [[0.5, 3.0, 0.2], [-4.0, 9.5, 12.0], [0.0, 9.99, 2.0]] {
            let sr = SphericalRect::new(e, o);
            let v: Vec<V3> = corners.iter().map(|&c| sub(c, o)).collect();
            let tri = |a: V3, b: V3, c: V3| {
                let l = |x: V3| dot(x, x).sqrt();
                let num = dot(a, cross(b, c)).abs();
                let den = l(a) * l(b) * l(c) + dot(a, b) * l(c) + dot(a, c) * l(b) + dot(b, c) * l(a);
                2.0 * num.atan2(den)
            };
            let omega = tri(v[0], v[1], v[2]) + tri(v[0], v[2], v[3]);
            assert!((sr.solid_angle / omega - 1.0).abs() < 1e-9, "{o:?}: {} vs {omega}", sr.solid_angle);
            let mut mean = [0.0; 3];
            for i in 0..4 {
                let (a, b) = (normalize(v[i]), normalize(v[(i + 1) % 4]));
                let th = dot(a, b).clamp(-1.0, 1.0).acos();
                let u = normalize(cross(a, b));
                mean = [0, 1, 2].map(|c| mean[c] + 0.5 * th * u[c]);
            }
            // The winding's sign: the mean direction points at the quad.
            let sg = if dot(mean, v[0]) > 0.0 { 1.0 } else { -1.0 };
            let mut rng = Rng::new(9, 9, 9);
            let k = 200_000;
            let mut m = [0.0; 3];
            for _ in 0..k {
                let p = sr.sample(rng.uniform(), rng.uniform());
                assert!((p[1] - 10.0).abs() < 1e-9 && (2.0..=7.0 + 1e-9).contains(&p[2]) && (-3.0 - 1e-9..=1.0).contains(&p[0]), "{p:?}");
                let w = normalize(sub(p, o));
                m = [0, 1, 2].map(|c| m[c] + w[c] / k as f64);
            }
            let q = mean.map(|c| sg * c / omega);
            assert!((0..3).all(|c| (m[c] - q[c]).abs() < 5e-3), "{o:?}: {m:?} vs {q:?}");
        }
    }

    #[test]
    fn bad_emission_and_stale_snapshots_are_refused() {
        let (mut r, _, _, hot) = reg();
        let t = EmitterTable::build(&r, 4, [quad(0, hot, 3, 0, 0, 0, 1)]).unwrap();
        assert_eq!(t.check(4), Ok(()));
        assert_eq!(t.check(5), Err(StaleTable { table: 4, expected: 5 }));
        r.register("neg", MaterialParams { base_color: [0.5; 3], emissive: [-1.0, 0.0, 0.0] }).unwrap();
        assert!(EmitterTable::build(&r, 4, []).unwrap_err().contains("neg"));
    }
}
