//! CPU reference ray query (Phase 1E): the ground truth later GPU paths are compared against.
//!
//! # Convention
//!
//! - Units are voxels: voxel `v` is the half-open box `[v, v+1)` on each axis.
//! - A ray `o + t·d` is inside voxel `v` at time `t` when `floor(o + t·d) = v`.
//! - The hit is the smallest `t` in `[0, t_max]` at which the ray enters an occupied voxel. That is
//!   the voxel's entry time, clamped to 0 when the origin is already inside.
//! - A voxel the ray only touches for zero length (it passes exactly along an edge or corner, or
//!   starts on the face it is leaving) is **not** hit.
//! - Crossing times are always computed as `(plane - o) / d` in f64, the same expression in both
//!   implementations. [`trace`] and [`trace_brute`] must therefore agree **exactly** (bit-equal `t`,
//!   same voxel, material and face), not merely within a tolerance.
//!
//! [`trace`] is a voxel DDA (visit cells in ray order). [`trace_brute`] tests every occupied voxel;
//! it is the oracle, not meant for speed.

use crate::coords::{BrickKey, VoxelCoord};
use crate::material::MaterialId;
use crate::world::World;
use crate::brick::Brick;

/// Largest supported coordinate magnitude (voxel units) for ray origins. Crossing times of
/// neighbouring planes must stay distinct in f64, which is comfortably true below 2^40.
pub const MAX_COORD: f64 = (1u64 << 40) as f64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: [f64; 3],
    pub dir: [f64; 3],
}

/// The face through which the ray entered: `axis` 0/1/2 = x/y/z; `positive` means the face's outward
/// normal points along +axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Face {
    pub axis: u8,
    pub positive: bool,
}

impl Face {
    pub fn normal(self) -> [f64; 3] {
        let mut n = [0.0; 3];
        n[self.axis as usize] = if self.positive { 1.0 } else { -1.0 };
        n
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Ray parameter in voxel units (times |dir|).
    pub t: f64,
    pub voxel: VoxelCoord,
    pub material: MaterialId,
    /// `None` when the origin is strictly inside the hit voxel.
    pub face: Option<Face>,
}

fn check_ray(ray: &Ray) {
    assert!(ray.origin.iter().all(|o| o.is_finite() && o.abs() < MAX_COORD), "ray origin out of range: {ray:?}");
    assert!(ray.dir.iter().all(|d| d.is_finite()) && ray.dir.iter().any(|&d| d != 0.0), "ray direction must be finite and non-zero: {ray:?}");
}

/// Entry and exit time of cell `c` along one axis with `d != 0`.
fn axis_span(o: f64, d: f64, c: i64) -> (f64, f64) {
    let near = ((c as f64) - o) / d;
    let far = ((c + 1) as f64 - o) / d;
    if d > 0.0 { (near, far) } else { (far, near) }
}

/// The ray's time interval inside voxel `c`: `Some((t_in, t_out))` with `t_in < t_out`, `t_in >= 0`.
fn cell_interval(ray: &Ray, c: [i64; 3]) -> Option<(f64, f64)> {
    let (mut t_in, mut t_out) = (0.0f64, f64::INFINITY);
    for a in 0..3 {
        let (o, d) = (ray.origin[a], ray.dir[a]);
        if d == 0.0 {
            if o.floor() as i64 != c[a] {
                return None;
            }
        } else {
            let (e, x) = axis_span(o, d, c[a]);
            t_in = t_in.max(e);
            t_out = t_out.min(x);
        }
    }
    (t_in < t_out).then_some((t_in, t_out))
}

/// Entry face of cell `c` for a ray whose interval starts at `t_in`: the lowest axis whose entry
/// plane is crossed exactly at `t_in`. `None` if every entry plane is behind the origin (inside).
fn entry_face(ray: &Ray, c: [i64; 3], t_in: f64) -> Option<Face> {
    (0..3).find_map(|a| {
        let d = ray.dir[a];
        (d != 0.0 && axis_span(ray.origin[a], d, c[a]).0 == t_in).then_some(Face { axis: a as u8, positive: d < 0.0 })
    })
}

fn coord(c: [i64; 3]) -> VoxelCoord {
    VoxelCoord::new(c[0] as i32, c[1] as i32, c[2] as i32)
}

/// Oracle: tests every occupied voxel. Ties at equal `t` (not expected for disjoint cells, see the
/// module docs) resolve to the smallest voxel coordinate.
pub fn trace_brute(world: &World, ray: &Ray, t_max: f64) -> Option<Hit> {
    check_ray(ray);
    let mut best: Option<(f64, VoxelCoord, MaterialId)> = None;
    for (v, m) in world.occupied() {
        if let Some((t_in, _)) = cell_interval(ray, [v.x as i64, v.y as i64, v.z as i64]) {
            let better = match best {
                None => true,
                Some((bt, bv, _)) => t_in < bt || (t_in == bt && v < bv),
            };
            if t_in <= t_max && better {
                best = Some((t_in, v, m));
            }
        }
    }
    best.map(|(t, v, material)| Hit { t, voxel: v, material, face: entry_face(ray, [v.x as i64, v.y as i64, v.z as i64], t) })
}

/// Voxel DDA. See the module docs for the convention it shares with [`trace_brute`].
pub fn trace(world: &World, ray: &Ray, t_max: f64) -> Option<Hit> {
    check_ray(ray);
    let (lo, hi) = world.bounds()?;
    // Expand by one voxel: the first cell visited is then outside the stored data, and rounding in
    // the box test cannot skip an occupied cell.
    let lo = [lo.x as i64 - 1, lo.y as i64 - 1, lo.z as i64 - 1];
    let hi = [hi.x as i64 + 1, hi.y as i64 + 1, hi.z as i64 + 1];

    let (mut t0, mut t1) = (0.0f64, t_max);
    for a in 0..3 {
        let (o, d) = (ray.origin[a], ray.dir[a]);
        if d == 0.0 {
            if o < lo[a] as f64 || o >= hi[a] as f64 {
                return None;
            }
        } else {
            let ta = (lo[a] as f64 - o) / d;
            let tb = (hi[a] as f64 - o) / d;
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
        }
    }
    if t0 > t1 {
        return None;
    }

    // Starting cell: the cell whose axis interval contains t0 as `entry <= t0 < exit`, i.e. the cell
    // the ray occupies just after t0. Found by rounding, then corrected with the exact crossing times.
    let mut c = [0i64; 3];
    let mut step = [0i64; 3];
    for a in 0..3 {
        let (o, d) = (ray.origin[a], ray.dir[a]);
        if d == 0.0 {
            c[a] = o.floor() as i64;
            continue;
        }
        step[a] = if d > 0.0 { 1 } else { -1 };
        let mut k = (o + t0 * d).floor() as i64;
        while axis_span(o, d, k).1 <= t0 {
            k += step[a];
        }
        while axis_span(o, d, k).0 > t0 {
            k -= step[a];
        }
        c[a] = k;
    }

    // Each step leaves the cell on at least one axis; the box spans (hi - lo) cells per axis.
    let max_steps = (0..3).map(|a| (hi[a] - lo[a]) as u64).sum::<u64>() + 3;
    let mut cache: Option<(BrickKey, Option<&Brick>)> = None;
    let mut t = t0;
    for _ in 0..=max_steps {
        if t > t1 {
            return None;
        }
        let exits = [0, 1, 2].map(|a| if step[a] == 0 { f64::INFINITY } else { axis_span(ray.origin[a], ray.dir[a], c[a]).1 });
        let t_exit = exits[0].min(exits[1]).min(exits[2]);
        if t < t_exit {
            let v = coord(c);
            let (key, idx) = v.split();
            let brick = match cache {
                Some((k, b)) if k == key => b,
                _ => {
                    let b = world.brick(key);
                    cache = Some((key, b));
                    b
                }
            };
            if let Some(material) = brick.and_then(|b| b.get(idx)) {
                return Some(Hit { t, voxel: v, material, face: entry_face(ray, c, t) });
            }
        }
        if t_exit == f64::INFINITY {
            return None;
        }
        for a in 0..3 {
            if exits[a] == t_exit {
                c[a] += step[a];
            }
        }
        t = t_exit;
    }
    panic!("DDA exceeded {max_steps} steps: {ray:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{MaterialParams, MaterialRegistry};

    fn one_voxel(at: VoxelCoord) -> (World, MaterialId) {
        let mut r = MaterialRegistry::new();
        let m = r.register("stone", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
        let mut w = World::new(r);
        w.set(at, Some(m)).unwrap();
        (w, m)
    }

    fn ray(o: [f64; 3], d: [f64; 3]) -> Ray {
        Ray { origin: o, dir: d }
    }

    fn both(w: &World, r: &Ray, t_max: f64) -> Option<Hit> {
        let a = trace(w, r, t_max);
        let b = trace_brute(w, r, t_max);
        assert_eq!(a, b, "DDA and brute force disagree for {r:?}, t_max {t_max}");
        a
    }

    #[test]
    fn hits_from_each_axis_direction_with_correct_face() {
        let (w, m) = one_voxel(VoxelCoord::new(0, 0, 0));
        for a in 0..3 {
            for s in [1.0, -1.0] {
                let mut o = [0.5; 3];
                o[a] = 0.5 - 5.0 * s;
                let mut d = [0.0; 3];
                d[a] = s;
                let h = both(&w, &ray(o, d), f64::INFINITY).expect("hit");
                assert_eq!(h.t, 4.5);
                assert_eq!(h.material, m);
                assert_eq!(h.face, Some(Face { axis: a as u8, positive: s < 0.0 }));
            }
        }
    }

    #[test]
    fn negative_coordinates_hit() {
        let (w, _) = one_voxel(VoxelCoord::new(-1, -1, -1));
        let h = both(&w, &ray([-0.5, -0.5, 3.0], [0.0, 0.0, -1.0]), f64::INFINITY).unwrap();
        assert_eq!((h.t, h.voxel), (3.0, VoxelCoord::new(-1, -1, -1)));
        assert_eq!(h.face, Some(Face { axis: 2, positive: true }));
    }

    #[test]
    fn origin_inside_hits_at_zero_without_face() {
        let (w, _) = one_voxel(VoxelCoord::new(2, 2, 2));
        let h = both(&w, &ray([2.5, 2.25, 2.75], [0.3, -1.0, 0.2]), f64::INFINITY).unwrap();
        assert_eq!((h.t, h.face), (0.0, None));
    }

    #[test]
    fn half_open_faces() {
        let (w, _) = one_voxel(VoxelCoord::new(0, 0, 0));
        // Along the plane y = 0 (inside the voxel's closed-open range): hit.
        assert!(both(&w, &ray([-3.0, 0.0, 0.5], [1.0, 0.0, 0.0]), f64::INFINITY).is_some());
        // Along the plane y = 1 (the next cell up): miss.
        assert!(both(&w, &ray([-3.0, 1.0, 0.5], [1.0, 0.0, 0.0]), f64::INFINITY).is_none());
        // Starting on the voxel's -x face and moving away: zero-length contact, miss.
        assert!(both(&w, &ray([0.0, 0.5, 0.5], [-1.0, 0.0, 0.0]), f64::INFINITY).is_none());
        // Starting on the voxel's +x face (x = 1 is the next cell) and moving in: hit at t = 0 through that face.
        let h = both(&w, &ray([1.0, 0.5, 0.5], [-1.0, 0.0, 0.0]), f64::INFINITY).unwrap();
        assert_eq!((h.t, h.face), (0.0, Some(Face { axis: 0, positive: true })));
    }

    #[test]
    fn exact_edge_graze_is_not_a_hit() {
        let (w, _) = one_voxel(VoxelCoord::new(0, 0, 0));
        // Passes exactly through the edge x = 1, y = 1 heading up-left: touches the voxel's corner line only.
        let r = ray([2.0, 0.0, 0.5], [-1.0, 1.0, 0.0]);
        assert!(both(&w, &r, f64::INFINITY).is_none());
    }

    #[test]
    fn t_max_is_inclusive() {
        let (w, _) = one_voxel(VoxelCoord::new(0, 0, 0));
        let r = ray([-4.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        assert!(both(&w, &r, 3.999).is_none());
        assert_eq!(both(&w, &r, 4.0).unwrap().t, 4.0);
    }

    #[test]
    fn misses_and_empty_world() {
        let (w, _) = one_voxel(VoxelCoord::new(0, 0, 0));
        assert!(both(&w, &ray([-4.0, 5.5, 0.5], [1.0, 0.0, 0.0]), f64::INFINITY).is_none());
        assert!(both(&w, &ray([-4.0, 0.5, 0.5], [-1.0, 0.0, 0.0]), f64::INFINITY).is_none());
        let empty = World::new(MaterialRegistry::new());
        assert!(both(&empty, &ray([0.0; 3], [1.0, 0.0, 0.0]), f64::INFINITY).is_none());
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_direction_is_rejected() {
        let (w, _) = one_voxel(VoxelCoord::new(0, 0, 0));
        trace(&w, &ray([0.0; 3], [0.0; 3]), 1.0);
    }
}
