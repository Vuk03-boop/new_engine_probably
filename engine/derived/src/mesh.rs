//! Phase 2A: per-brick surface extraction, the first real derived product (ADR-0004).
//!
//! # Contract
//!
//! - A **unit face** is one face of an occupied voxel whose neighbour across that face is empty.
//!   A brick's mesh covers exactly its own voxels' unit faces, each **exactly once**, and nothing
//!   else. Every quad carries the exact `MaterialId` of the voxels behind it.
//! - [`Merge::None`] emits one quad per unit face. [`Merge::Greedy`] merges unit faces of one brick
//!   that share a face direction, a plane and a material into rectangles. Merging never crosses a
//!   material, a plane or the brick. Both are exact: they cover the same unit faces.
//! - Coordinates are brick-local integers. A quad lies in the plane `axis = plane` (0..=8) and
//!   covers the half-open rectangle `[u0, u1) × [v0, v1)` on the other two axes, with
//!   `u = (axis + 1) % 3` and `v = (axis + 2) % 3`.
//! - [`Quad::corners`] lists the corners counter-clockwise seen from outside, so
//!   `(c1 - c0) × (c2 - c0)` points along the outward normal. Triangles are `(c0, c1, c2)` and
//!   `(c0, c2, c3)`, in that order.
//! - Output order is deterministic: face direction, then plane, then scan order.
//! - [`BrickMesh::triangulate`] turns quads into a **watertight** triangle mesh (ADR-0004
//!   amendment 1): greedy quads meet at T-junctions, and raster and ray–triangle tests are only
//!   watertight across shared edges. See [`Split`].
//!
//! A mesh depends on the brick's voxels and on the occupancy of the six face layers just outside
//! it, which is exactly what [`crate::surface::Dependencies`] records.

use world::dims::BRICK_EDGE;
use world::reference::{Face, Ray};
use world::{Brick, BrickKey, MaterialId, VoxelCoord, World};

use crate::surface::SurfaceSummary;

/// Mesh merge extent: one of the two granularity parameters of ADR-0003 (the "slider").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Merge {
    /// One quad per exposed voxel face. No T-junctions.
    None,
    /// Greedy rectangles within a brick, per face direction, plane and material.
    #[default]
    Greedy,
}

impl Merge {
    pub const ALL: [Merge; 2] = [Merge::None, Merge::Greedy];

    pub fn name(self) -> &'static str {
        match self {
            Merge::None => "none",
            Merge::Greedy => "greedy",
        }
    }
}

/// One axis-aligned rectangle of exposed surface. 8 bytes; see the module docs for the convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Quad {
    pub material: MaterialId,
    /// `axis * 2 + positive`: 0 = -x, 1 = +x, 2 = -y, 3 = +y, 4 = -z, 5 = +z.
    pub face: u8,
    /// Brick-local plane coordinate on the face axis, `0..=8`.
    pub plane: u8,
    pub u0: u8,
    pub v0: u8,
    pub u1: u8,
    pub v1: u8,
}

const _: () = assert!(std::mem::size_of::<Quad>() == 8, "ADR-0004: a quad record is 8 bytes");
const _: () = assert!(BRICK_EDGE <= u8::MAX as i32, "brick-local coordinates fit u8");

const E: usize = BRICK_EDGE as usize;

impl Quad {
    pub fn axis(self) -> usize {
        (self.face / 2) as usize
    }

    pub fn positive(self) -> bool {
        self.face % 2 == 1
    }

    /// The reference module's face type (outward normal).
    pub fn to_face(self) -> Face {
        Face { axis: self.face / 2, positive: self.positive() }
    }

    /// The (u, v) axes for this quad's face axis.
    pub fn uv_axes(self) -> (usize, usize) {
        let a = self.axis();
        ((a + 1) % 3, (a + 2) % 3)
    }

    pub fn area(self) -> u32 {
        (self.u1 - self.u0) as u32 * (self.v1 - self.v0) as u32
    }

    /// Brick-local corners, counter-clockwise seen from outside (along the outward normal).
    pub fn corners(self) -> [[u8; 3]; 4] {
        let (ua, va) = self.uv_axes();
        let at = |u: u8, v: u8| {
            let mut p = [0u8; 3];
            p[self.axis()] = self.plane;
            p[ua] = u;
            p[va] = v;
            p
        };
        let (a, b, c, d) = (at(self.u0, self.v0), at(self.u1, self.v0), at(self.u1, self.v1), at(self.u0, self.v1));
        // e_u × e_v = +e_axis (cyclic axes), so this order faces +axis; reverse it for -axis faces.
        if self.positive() { [a, b, c, d] } else { [a, d, c, b] }
    }

    /// Every unit face this quad covers, as (voxel behind the face, face).
    pub fn unit_faces(self, key: BrickKey) -> impl Iterator<Item = (VoxelCoord, Face)> {
        let o = key.origin();
        let (ua, va) = self.uv_axes();
        let depth = if self.positive() { self.plane as i32 - 1 } else { self.plane as i32 };
        let face = self.to_face();
        (self.v0..self.v1).flat_map(move |v| {
            (self.u0..self.u1).map(move |u| {
                let mut l = [0i32; 3];
                l[self.axis()] = depth;
                l[ua] = u as i32;
                l[va] = v as i32;
                (VoxelCoord::new(o.x + l[0], o.y + l[1], o.z + l[2]), face)
            })
        })
    }
}

/// A brick's extracted surface. An occupied brick whose faces are all hidden has an empty mesh.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BrickMesh {
    pub quads: Vec<Quad>,
}

impl BrickMesh {
    /// Unit faces per material, the same numbers [`crate::surface::summarize`] counts directly.
    pub fn summary(&self) -> SurfaceSummary {
        let mut counts = std::collections::BTreeMap::<MaterialId, u32>::new();
        for q in &self.quads {
            *counts.entry(q.material).or_default() += q.area();
        }
        SurfaceSummary { total: counts.values().sum(), by_material: counts.into_iter().collect() }
    }

    /// Heap outside the struct itself (Phase 1D).
    pub fn heap(&self) -> memory::Usage {
        memory::containers::vec(&self.quads)
    }
}

/// Local grid with a one-voxel border: index `(x+1) + 10 (y+1) + 100 (z+1)` for x, y, z in -1..=8.
const G: usize = E + 2;

fn gi(x: i32, y: i32, z: i32) -> usize {
    (x + 1) as usize + G * ((y + 1) as usize + G * (z + 1) as usize)
}

/// Extracts `key`'s mesh from its brick and a voxel lookup (used for the face layers outside the
/// brick only). Returns `None` when the brick is absent.
pub fn extract(key: BrickKey, target: Option<&Brick>, lookup: impl Fn(VoxelCoord) -> Option<MaterialId>, merge: Merge) -> Option<BrickMesh> {
    let target = target?;
    let o = key.origin();
    let mut grid = [None::<MaterialId>; G * G * G];
    for (idx, m) in target.voxels() {
        let (x, y, z) = idx.xyz();
        grid[gi(x, y, z)] = Some(m);
    }
    // The six face layers just outside the brick (edges and corners are never read).
    for a in 0..3 {
        for outside in [-1, BRICK_EDGE] {
            for i in 0..BRICK_EDGE {
                for j in 0..BRICK_EDGE {
                    let mut l = [0; 3];
                    l[a] = outside;
                    l[(a + 1) % 3] = i;
                    l[(a + 2) % 3] = j;
                    grid[gi(l[0], l[1], l[2])] = lookup(VoxelCoord::new(o.x + l[0], o.y + l[1], o.z + l[2]));
                }
            }
        }
    }

    let mut quads = Vec::new();
    for face in 0..6u8 {
        let (a, positive) = ((face / 2) as usize, face % 2 == 1);
        let (ua, va) = ((a + 1) % 3, (a + 2) % 3);
        let step = if positive { 1 } else { -1 };
        for depth in 0..BRICK_EDGE {
            // mask[v][u]: material of an exposed unit face in this layer.
            let mut mask = [[None::<MaterialId>; E]; E];
            for (v, row) in mask.iter_mut().enumerate() {
                for (u, cell) in row.iter_mut().enumerate() {
                    let mut l = [0i32; 3];
                    l[a] = depth;
                    l[ua] = u as i32;
                    l[va] = v as i32;
                    if let Some(m) = grid[gi(l[0], l[1], l[2])] {
                        l[a] += step;
                        if grid[gi(l[0], l[1], l[2])].is_none() {
                            *cell = Some(m);
                        }
                    }
                }
            }
            let plane = (if positive { depth + 1 } else { depth }) as u8;
            emit(&mut mask, merge, |material, u0, v0, u1, v1| quads.push(Quad { material, face, plane, u0, v0, u1, v1 }));
        }
    }
    Some(BrickMesh { quads })
}

/// Turns one layer's mask into rectangles (consuming it) in row-major scan order.
fn emit(mask: &mut [[Option<MaterialId>; E]; E], merge: Merge, mut out: impl FnMut(MaterialId, u8, u8, u8, u8)) {
    for v in 0..E {
        let mut u = 0;
        while u < E {
            let Some(m) = mask[v][u] else {
                u += 1;
                continue;
            };
            let (mut w, mut h) = (1, 1);
            if merge == Merge::Greedy {
                while u + w < E && mask[v][u + w] == Some(m) {
                    w += 1;
                }
                while v + h < E && mask[v + h][u..u + w].iter().all(|&c| c == Some(m)) {
                    h += 1;
                }
            }
            for row in mask.iter_mut().skip(v).take(h) {
                for c in &mut row[u..u + w] {
                    *c = None;
                }
            }
            out(m, u as u8, v as u8, (u + w) as u8, (v + h) as u8);
            u += w;
        }
    }
}

/// Oracle: the mesh extracted directly from the world.
pub fn extract_world(world: &World, key: BrickKey, merge: Merge) -> Option<BrickMesh> {
    extract(key, world.brick(key), |v| world.get(v), merge)
}

/// Every exposed unit face of the world with its material, computed directly from voxels (the
/// coverage oracle, independent of [`extract`]).
pub fn exposed_faces(world: &World) -> Vec<(VoxelCoord, Face, MaterialId)> {
    let mut out = Vec::new();
    for (p, m) in world.occupied() {
        for a in 0..3u8 {
            for positive in [false, true] {
                let mut n = [p.x, p.y, p.z];
                n[a as usize] += if positive { 1 } else { -1 };
                if world.get(VoxelCoord::new(n[0], n[1], n[2])).is_none() {
                    out.push((p, Face { axis: a, positive }, m));
                }
            }
        }
    }
    out
}

/// How quad edges are split before triangulation.
///
/// [`Split::Watertight`] (the product) makes every triangle edge shared, using only the brick's
/// own quads, so a brick's triangles still depend on nothing but its mesh:
/// - An edge lying on the brick's boundary (one of its two fixed coordinates is 0 or 8) is split
///   at **every lattice point**. The brick across that boundary does the same, so both sides
///   produce the same unit segments whatever their merging.
/// - Any other edge is split at every corner of this brick's quads that lies strictly inside it.
///   Its interior is inside the open brick, so no other brick's vertex can lie on it.
///
/// A quad with no split points keeps two triangles `(c0, c1, c2), (c0, c2, c3)`. A quad with split
/// points becomes a fan around its centre (a half-integer point): one triangle per outline
/// segment, none degenerate, all counter-clockwise from outside.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Split {
    #[default]
    Watertight,
    /// Two triangles per quad, no splitting: the pre-amendment layout. Negative controls only.
    CornersOnly,
}

/// A brick's triangles. Coordinates are brick-local in **half-voxel units** (×2), `0..=16`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Triangles {
    pub vertices: Vec<[u8; 3]>,
    /// Vertex indices, counter-clockwise from outside.
    pub indices: Vec<[u32; 3]>,
    /// For each triangle, the index of its quad in [`BrickMesh::quads`].
    pub quad: Vec<u32>,
}

impl BrickMesh {
    /// See [`Split`].
    pub fn triangulate(&self, split: Split) -> Triangles {
        let n = E + 1;
        let at = |p: [u8; 3]| p[0] as usize + n * (p[1] as usize + n * p[2] as usize);
        let mut corner = vec![false; n * n * n];
        for q in &self.quads {
            for c in q.corners() {
                corner[at(c)] = true;
            }
        }
        let edge = BRICK_EDGE as u8;
        let mut t = Triangles::default();
        let mut outline = Vec::new();
        for (qi, q) in self.quads.iter().enumerate() {
            let cs = q.corners();
            outline.clear();
            for k in 0..4 {
                let (p, r) = (cs[k], cs[(k + 1) % 4]);
                outline.push(p);
                if split == Split::CornersOnly {
                    continue;
                }
                let e = (0..3).find(|&a| p[a] != r[a]).expect("a quad edge has length");
                let boundary = (0..3).filter(|&a| a != e).any(|a| p[a] == 0 || p[a] == edge);
                let (lo, hi) = (p[e].min(r[e]), p[e].max(r[e]));
                let mut inner: Vec<[u8; 3]> = (lo + 1..hi)
                    .map(|c| {
                        let mut x = p;
                        x[e] = c;
                        x
                    })
                    .filter(|&x| boundary || corner[at(x)])
                    .collect();
                if p[e] > r[e] {
                    inner.reverse();
                }
                outline.extend(inner);
            }
            let base = t.vertices.len() as u32;
            let dbl = |p: [u8; 3]| p.map(|c| 2 * c);
            t.vertices.extend(outline.iter().map(|&p| dbl(p)));
            if outline.len() == 4 {
                t.indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
                t.quad.extend([qi as u32; 2]);
            } else {
                let m = outline.len() as u32;
                let (ua, va) = q.uv_axes();
                let mut c = [0u8; 3];
                c[q.axis()] = 2 * q.plane;
                c[ua] = q.u0 + q.u1;
                c[va] = q.v0 + q.v1;
                t.vertices.push(c);
                let centre = base + m;
                for i in 0..m {
                    t.indices.push([centre, base + i, base + (i + 1) % m]);
                    t.quad.push(qi as u32);
                }
            }
        }
        t
    }
}

/// A ray–mesh hit, in the same terms as [`world::reference::Hit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshHit {
    pub t: f64,
    pub voxel: VoxelCoord,
    pub material: MaterialId,
    pub face: Face,
}

/// The cell `c` in `[lo, hi)` the ray occupies just after time `t` on one axis, using the reference
/// crossing expression `(plane - o) / d`. `None` if the ray is outside the range then.
fn cell_after(o: f64, d: f64, t: f64, lo: i64, hi: i64) -> Option<i64> {
    if d == 0.0 {
        let c = o.floor() as i64;
        return (lo..hi).contains(&c).then_some(c);
    }
    (lo..hi).find(|&c| {
        let (near, far) = (((c as f64) - o) / d, (((c + 1) as f64) - o) / d);
        let (enter, exit) = if d > 0.0 { (near, far) } else { (far, near) };
        enter <= t && t < exit
    })
}

/// CPU ray–quad reference: the nearest quad the ray enters from outside (front faces only).
///
/// It follows the [`world::reference`] convention: crossing times are `(plane - o) / d` in f64,
/// in-quad tests are half-open and use crossing times rather than hit points, and a tie at equal
/// `t` picks the lowest face axis (as the DDA's entry face does). For rays in general position it
/// must agree exactly with `world::reference::trace` from an origin outside occupied voxels.
///
/// Known difference on a measure-zero set: a ray passing *exactly* through the shared edge of two
/// diagonal voxels enters the far voxel for zero length in the DDA's view, but crosses no quad.
pub fn trace_mesh<'a>(meshes: impl IntoIterator<Item = (BrickKey, &'a BrickMesh)>, ray: &Ray, t_max: f64) -> Option<MeshHit> {
    let mut best: Option<MeshHit> = None;
    for (key, mesh) in meshes {
        let o = key.origin();
        let origin = [o.x as i64, o.y as i64, o.z as i64];
        // Conservative brick-box prefilter (half a voxel of slack); the exact test is per quad.
        let (mut t0, mut t1) = (0.0f64, t_max);
        for (a, &corner) in origin.iter().enumerate() {
            let (lo, hi) = (corner as f64 - 0.5, (corner + BRICK_EDGE as i64) as f64 + 0.5);
            let (ro, d) = (ray.origin[a], ray.dir[a]);
            if d == 0.0 {
                if ro < lo || ro > hi {
                    t0 = f64::INFINITY;
                }
            } else {
                let (ta, tb) = ((lo - ro) / d, (hi - ro) / d);
                t0 = t0.max(ta.min(tb));
                t1 = t1.min(ta.max(tb));
            }
        }
        if t0 > t1 || best.is_some_and(|b| t0 > b.t) {
            continue;
        }
        for &q in &mesh.quads {
            let a = q.axis();
            let d = ray.dir[a];
            // Front faces only: the ray must move against the outward normal.
            if (q.positive() && d >= 0.0) || (!q.positive() && d <= 0.0) {
                continue;
            }
            let t = ((origin[a] + q.plane as i64) as f64 - ray.origin[a]) / d;
            if !(0.0..=t_max).contains(&t) {
                continue;
            }
            if let Some(b) = best {
                if t > b.t || (t == b.t && q.axis() >= b.face.axis as usize) {
                    continue;
                }
            }
            let (ua, va) = q.uv_axes();
            let Some(cu) = cell_after(ray.origin[ua], ray.dir[ua], t, origin[ua] + q.u0 as i64, origin[ua] + q.u1 as i64) else { continue };
            let Some(cv) = cell_after(ray.origin[va], ray.dir[va], t, origin[va] + q.v0 as i64, origin[va] + q.v1 as i64) else { continue };
            let mut c = [0i64; 3];
            c[a] = origin[a] + q.plane as i64 - if q.positive() { 1 } else { 0 };
            c[ua] = cu;
            c[va] = cv;
            best = Some(MeshHit { t, voxel: VoxelCoord::new(c[0] as i32, c[1] as i32, c[2] as i32), material: q.material, face: q.to_face() });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use world::{MaterialParams, MaterialRegistry};

    fn world(n: usize) -> (World, Vec<MaterialId>) {
        let mut r = MaterialRegistry::new();
        let ids = (0..n).map(|i| r.register(&format!("m{i}"), MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap()).collect();
        (World::new(r), ids)
    }

    fn key_of(x: i32, y: i32, z: i32) -> BrickKey {
        VoxelCoord::new(x, y, z).split().0
    }

    #[test]
    fn single_voxel_gives_six_unit_quads_facing_out() {
        let (mut w, m) = world(1);
        w.set(VoxelCoord::new(3, 4, 5), Some(m[0])).unwrap();
        for merge in Merge::ALL {
            let mesh = extract_world(&w, key_of(3, 4, 5), merge).unwrap();
            assert_eq!(mesh.quads.len(), 6);
            for q in &mesh.quads {
                assert_eq!(q.area(), 1);
                let [c0, c1, c2, _] = q.corners().map(|c| c.map(|x| x as i32));
                let (e1, e2) = ([c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]], [c2[0] - c0[0], c2[1] - c0[1], c2[2] - c0[2]]);
                let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                let mut want = [0; 3];
                want[q.axis()] = if q.positive() { 1 } else { -1 };
                assert_eq!(n, want, "winding of {q:?}");
                let faces: Vec<_> = q.unit_faces(key_of(3, 4, 5)).collect();
                assert_eq!(faces, vec![(VoxelCoord::new(3, 4, 5), q.to_face())]);
            }
        }
    }

    #[test]
    fn greedy_merges_a_slab_but_never_across_materials() {
        let (mut w, m) = world(2);
        // A 8×1×8 floor at y = 2, left half material 0, right half material 1.
        w.fill_box(VoxelCoord::new(0, 2, 0), VoxelCoord::new(4, 3, 8), Some(m[0])).unwrap();
        w.fill_box(VoxelCoord::new(4, 2, 0), VoxelCoord::new(8, 3, 8), Some(m[1])).unwrap();
        let k = key_of(0, 0, 0);
        let g = extract_world(&w, k, Merge::Greedy).unwrap();
        let top: Vec<_> = g.quads.iter().filter(|q| q.face == 3).collect();
        assert_eq!(top.len(), 2, "one top quad per material: {top:?}");
        assert!(top.iter().all(|q| q.area() == 32 && q.plane == 3));
        let none = extract_world(&w, k, Merge::None).unwrap();
        assert_eq!(none.summary(), g.summary());
        assert_eq!(g.summary(), crate::surface::summarize_world(&w, k).unwrap());
    }

    #[test]
    fn faces_hidden_by_a_neighbour_brick_are_not_emitted() {
        let (mut w, m) = world(1);
        w.set(VoxelCoord::new(7, 0, 0), Some(m[0])).unwrap();
        w.set(VoxelCoord::new(8, 0, 0), Some(m[0])).unwrap();
        let mesh = extract_world(&w, key_of(7, 0, 0), Merge::Greedy).unwrap();
        assert_eq!(mesh.quads.len(), 5);
        assert!(!mesh.quads.iter().any(|q| q.face == 1), "+x face is hidden by the neighbour brick");
    }

    #[test]
    fn diagonal_edge_ray_is_the_documented_difference() {
        use world::reference::trace;
        let (mut w, m) = world(1);
        // Voxels (1,0,0) and (0,1,0) occupied; (1,1,0) occupied behind their shared edge x = 1, y = 1.
        for p in [(1, 0, 0), (0, 1, 0), (1, 1, 0)] {
            w.set(VoxelCoord::new(p.0, p.1, p.2), Some(m[0])).unwrap();
        }
        let meshes: Vec<_> = w.bricks().map(|(k, _)| (k, extract_world(&w, k, Merge::None).unwrap())).collect();
        let r = Ray { origin: [0.0, 0.0, 0.5], dir: [1.0, 1.0, 0.0] };
        assert!(trace(&w, &r, 10.0).is_some(), "the DDA enters (1,1,0) through the edge");
        assert!(trace_mesh(meshes.iter().map(|(k, m)| (*k, m)), &r, 10.0).is_none(), "no quad is crossed");
    }
}
