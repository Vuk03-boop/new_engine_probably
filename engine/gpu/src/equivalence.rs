//! ADR-0003 Amendment 1: the per-pixel equivalence check between a rendered G-buffer and the CPU
//! voxel reference (`world::reference::trace`, the 1E DDA). Pure CPU; no device needed.
//!
//! Declared before any measurement (2C decisions 2 and 6); a breach is a bug to diagnose, never a
//! reason to change these numbers:
//! - **Depth:** distance along the pixel ray must satisfy
//!   `|t_gpu − t_ref|·|d| ≤ DEPTH_REL_TOL·t_ref·|d| + DEPTH_ABS_TOL` (voxels).
//! - **Near-edge exclusion:** besides the pixel-centre ray, four probe rays at ±[`EDGE_PROBE_PX`]
//!   in x and y are traced. If any probe resolves to a different (voxel, face), or hit/miss, the
//!   pixel is *near an edge*: a projected voxel edge passes within `EDGE_PROBE_PX/√2` of its centre,
//!   where raster and ray may legitimately resolve to neighbouring voxels. Such pixels skip the
//!   exact checks and are counted, never hidden.
//! - **Cracks:** a pixel whose five reference rays all hit the same face plane (same axis,
//!   direction and plane coordinate) lies inside one continuous planar surface. There, no surface
//!   drawn, or only a *different* surface farther than the depth tolerance, is a crack, *whether or
//!   not the pixel is near an edge*: T-junction cracks sit exactly on voxel edges, so the edge
//!   exclusion must not hide them.
//! - A surface on the reference's own plane, but farther than the depth tolerance, is not a crack
//!   (the surface is there); it is counted as `plane_depth` and reported. On non-edge pixels the
//!   depth check proper also counts it. (First 2C-1 run: at grazing incidence the rasterizer's
//!   sub-pixel vertex snapping moves depth along the ray far more than position on the plane.)
//! - **Exact fields** (all other hit pixels): material, octahedral normal (R16G16_SNORM bits), and
//!   the resolved (voxel, face): the surface id's quad (`tri_quad[primitive]`), and the unit face of it the GPU depth puts
//!   the sample on. The raw primitive index is not compared.

use std::collections::BTreeMap;

use world::reference::{trace, Face, Ray};
use world::{MaterialId, VoxelCoord, World};

use crate::layout::{RegionKey, RegionMesh, RegionQuad};
use crate::raster::{Camera, Frame, BACKGROUND_SURFACE};

pub const DEPTH_REL_TOL: f64 = 1e-4;
pub const DEPTH_ABS_TOL: f64 = 1e-3;
pub const EDGE_PROBE_PX: f64 = 1.0 / 16.0;
/// Farther than any street-block distance.
pub const T_MAX: f64 = 1e6;

/// One reference ray's result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub t: f64,
    pub voxel: VoxelCoord,
    pub material: MaterialId,
    pub face: Face,
}

impl Sample {
    /// (axis, positive, plane coordinate) of the face's plane.
    pub fn plane(&self) -> (u8, bool, i32) {
        plane_of(self.voxel, self.face)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefPixel {
    pub centre: Option<Sample>,
    pub near_edge: bool,
    /// All five rays hit the same face plane.
    pub same_plane: bool,
}

fn sample(world: &World, ray: &Ray) -> (Option<Sample>, bool) {
    match trace(world, ray, T_MAX) {
        None => (None, true),
        Some(h) => match h.face {
            Some(face) => (Some(Sample { t: h.t, voxel: h.voxel, material: h.material, face }), true),
            // The origin is inside a voxel: no face to compare.
            None => (None, false),
        },
    }
}

fn reference_pixel(world: &World, cam: &Camera, x: u32, y: u32) -> RefPixel {
    let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
    let ray = |dx: f64, dy: f64| Ray { origin: cam.eye, dir: cam.dir(fx + dx, fy + dy) };
    let (centre, centre_ok) = sample(world, &ray(0.0, 0.0));
    let e = EDGE_PROBE_PX;
    let probes = [(e, 0.0), (-e, 0.0), (0.0, e), (0.0, -e)].map(|(dx, dy)| sample(world, &ray(dx, dy)));
    let key = |s: &Option<Sample>| s.map(|s| (s.voxel, s.face));
    let near_edge = !centre_ok || probes.iter().any(|(p, ok)| !ok || key(p) != key(&centre));
    let same_plane = centre.is_some_and(|c| probes.iter().all(|(p, _)| p.is_some_and(|p| p.plane() == c.plane())));
    RefPixel { centre, near_edge, same_plane }
}

/// The reference image, row-major from the top-left pixel, computed on `threads` threads.
pub fn reference(world: &World, cam: &Camera, threads: usize) -> Vec<RefPixel> {
    let (w, h) = (cam.width, cam.height);
    let rows_per = (h as usize).div_ceil(threads.max(1));
    let mut out = vec![RefPixel { centre: None, near_edge: false, same_plane: false }; (w * h) as usize];
    std::thread::scope(|s| {
        for (k, part) in out.chunks_mut(rows_per * w as usize).enumerate() {
            s.spawn(move || {
                for (i, p) in part.iter_mut().enumerate() {
                    let idx = k * rows_per * w as usize + i;
                    *p = reference_pixel(world, cam, idx as u32 % w, idx as u32 / w);
                }
            });
        }
    });
    out
}

/// Octahedral encoding, the same expression as `oct_encode` in `shaders/raster.slang`.
pub fn oct_encode(n: [f64; 3]) -> [f64; 2] {
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    let (x, y, z) = (n[0] / l, n[1] / l, n[2] / l);
    if z < 0.0 {
        let s = |v: f64| if v >= 0.0 { 1.0 } else { -1.0 };
        [(1.0 - y.abs()) * s(x), (1.0 - x.abs()) * s(y)]
    } else {
        [x, y]
    }
}

/// The R16G16_SNORM bits the G-buffer must hold for a face's normal.
pub fn normal_bits(face: Face) -> [i16; 2] {
    oct_encode(face.normal()).map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
}

/// Counts from one comparison. Every count except `near_edge` must be zero.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub pixels: u64,
    pub ref_hits: u64,
    pub gpu_hits: u64,
    pub near_edge: u64,
    /// Hit pixels that went through every exact check.
    pub checked: u64,
    pub crack: u64,
    /// The reference plane was drawn, but farther than the depth tolerance (reported, see module docs).
    pub plane_depth: u64,
    pub material: u64,
    pub normal: u64,
    pub face: u64,
    pub depth: u64,
    /// The GPU drew a surface where every reference ray misses.
    pub extra: u64,
    /// A surface id that names no region or quad.
    pub bad_id: u64,
    /// Largest |depth error| / tolerance over checked pixels.
    pub max_depth_ratio: f64,
    pub examples: Vec<String>,
}

impl Report {
    pub fn failures(&self) -> u64 {
        self.crack + self.material + self.normal + self.face + self.depth + self.extra + self.bad_id
    }

    pub fn passes(&self) -> bool {
        self.failures() == 0
    }

    fn note(&mut self, s: String) {
        if self.examples.len() < 8 {
            self.examples.push(s);
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "pixels {} ref_hits {} gpu_hits {} checked {} near_edge {} ({:.2}% of hits) | crack {} material {} normal {} face {} depth {} extra {} bad_id {} | plane_depth {} | max depth err/tol {:.4}",
            self.pixels,
            self.ref_hits,
            self.gpu_hits,
            self.checked,
            self.near_edge,
            100.0 * self.near_edge as f64 / self.ref_hits.max(1) as f64,
            self.crack,
            self.material,
            self.normal,
            self.face,
            self.depth,
            self.extra,
            self.bad_id,
            self.plane_depth,
            self.max_depth_ratio
        )
    }
}

/// (axis, positive, plane coordinate), as [`Sample::plane`].
fn plane_of(voxel: VoxelCoord, face: Face) -> (u8, bool, i32) {
    let c = [voxel.x, voxel.y, voxel.z][face.axis as usize];
    (face.axis, face.positive, if face.positive { c + 1 } else { c })
}

/// The (voxel, face) a GPU sample resolves to: its quad's face, and the unit cell of the quad's
/// plane that the depth-reconstructed point lies in.
fn resolve(q: RegionQuad, origin: VoxelCoord, point: [f64; 3]) -> (VoxelCoord, Face) {
    let a = (q.face / 2) as usize;
    let positive = q.face % 2 == 1;
    let o = [origin.x, origin.y, origin.z];
    let plane = o[a] + q.plane as i32;
    let mut v = [0i32; 3];
    for (k, c) in v.iter_mut().enumerate() {
        *c = if k == a { if positive { plane - 1 } else { plane } } else { point[k].floor() as i32 };
    }
    (VoxelCoord::new(v[0], v[1], v[2]), Face { axis: a as u8, positive })
}

/// Compares `frame` with the reference. `regions` must be the CPU regions the GPU meshes were
/// uploaded from; region index `i` in the surface id is the `i`-th in key order.
pub fn compare(frame: &Frame, refs: &[RefPixel], cam: &Camera, regions: &BTreeMap<RegionKey, RegionMesh>) -> Report {
    let list: Vec<&RegionMesh> = regions.values().collect();
    let mut r = Report { pixels: refs.len() as u64, ..Report::default() };
    for (i, rp) in refs.iter().enumerate() {
        let (x, y) = (i as u32 % frame.width, i as u32 / frame.width);
        let d = cam.dir(x as f64 + 0.5, y as f64 + 0.5);
        let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let tol = |t: f64| DEPTH_REL_TOL * t * dl + DEPTH_ABS_TOL;
        if rp.centre.is_some() {
            r.ref_hits += 1;
        }
        let gpu = if frame.surface[i] == BACKGROUND_SURFACE {
            None
        } else {
            r.gpu_hits += 1;
            let [ri, prim] = frame.surface[i];
            let Some(q) = list.get(ri as usize).and_then(|m| m.tri_quad.get(prim as usize).and_then(|&qi| m.quads.get(qi as usize)).map(|q| (m, *q))) else {
                r.bad_id += 1;
                r.note(format!("({x},{y}): surface id {:?} names no quad", frame.surface[i]));
                continue;
            };
            let t = cam.distance(frame.depth[i]);
            let point = [0, 1, 2].map(|k| cam.eye[k] + t * d[k]);
            let (voxel, face) = resolve(q.1, q.0.key.origin(q.0.size), point);
            Some((t, voxel, face, q.1.material))
        };
        // Cracks: checked on every pixel inside one continuous plane, edge or not.
        if let (true, Some(c)) = (rp.same_plane, rp.centre) {
            match gpu {
                None => {
                    r.crack += 1;
                    r.note(format!("({x},{y}): crack (nothing drawn), reference {:?} {:?} t {:.4}", c.voxel, c.face, c.t));
                }
                Some(g) if g.0 > c.t + tol(c.t) / dl => {
                    if plane_of(g.1, g.2) == c.plane() {
                        r.plane_depth += 1;
                    } else {
                        r.crack += 1;
                        r.note(format!("({x},{y}): crack (farther surface {:?} {:?} t {:.4}), reference {:?} {:?} t {:.4}", g.1, g.2, g.0, c.voxel, c.face, c.t));
                    }
                }
                Some(_) => {}
            }
        }
        if rp.near_edge {
            r.near_edge += 1;
            continue;
        }
        match (rp.centre, gpu) {
            (None, None) => {}
            (None, Some(g)) => {
                r.extra += 1;
                r.note(format!("({x},{y}): extra surface {:?} {:?} where the reference misses", g.1, g.2));
            }
            (Some(_), None) => {} // counted as a crack above (a non-edge hit is always same-plane)
            (Some(c), Some((t, voxel, face, qmat))) => {
                r.checked += 1;
                if frame.material[i] != c.material.raw() || qmat != c.material.raw() {
                    r.material += 1;
                    r.note(format!("({x},{y}): material {} (quad {qmat}), reference {}", frame.material[i], c.material.raw()));
                }
                if frame.normal[i] != normal_bits(c.face) {
                    r.normal += 1;
                    r.note(format!("({x},{y}): normal {:?}, reference {:?}", frame.normal[i], normal_bits(c.face)));
                }
                if (voxel, face) != (c.voxel, c.face) {
                    r.face += 1;
                    r.note(format!("({x},{y}): resolved {voxel:?} {face:?}, reference {:?} {:?}", c.voxel, c.face));
                }
                let err = (t - c.t).abs() * dl;
                let ratio = err / tol(c.t);
                r.max_depth_ratio = r.max_depth_ratio.max(ratio);
                if ratio > 1.0 {
                    r.depth += 1;
                    r.note(format!("({x},{y}): depth t {t:.6}, reference {:.6} (err {err:.2e} > tol {:.2e})", c.t, tol(c.t)));
                }
            }
        }
    }
    r
}

/// Direct per-pixel agreement of two rendered frames of the same camera and meshes (2D: raster
/// against ray query). Both are resolved to (voxel, face) as in [`compare`]. Pixels the reference
/// marks near an edge are counted separately: there the two paths may legitimately resolve to
/// neighbouring voxels (ADR-0003 Amendment 1), so only the non-edge counts must be zero.
#[derive(Clone, Debug, Default)]
pub struct Agreement {
    pub pixels: u64,
    pub both_hit: u64,
    /// One frame hit and the other missed.
    pub hit_miss: u64,
    pub hit_miss_edge: u64,
    /// Both hit, but material, normal bits or resolved (voxel, face) differ.
    pub differ: u64,
    pub differ_edge: u64,
    /// Both hit with the same resolved face, but depth differs by more than twice the declared
    /// tolerance (each is allowed that tolerance from the reference).
    pub depth: u64,
    /// Identical surface id (region, primitive).
    pub same_id: u64,
    /// Largest |depth difference| / (twice the tolerance) over pixels with the same resolved face.
    pub max_depth_ratio: f64,
    pub examples: Vec<String>,
}

impl Agreement {
    /// Non-edge disagreements; must be zero.
    pub fn failures(&self) -> u64 {
        self.hit_miss + self.differ + self.depth
    }

    pub fn summary(&self) -> String {
        format!(
            "pixels {} both_hit {} same_id {} | non-edge: hit_miss {} differ {} depth {} | near-edge: hit_miss {} differ {} | max depth diff/tol {:.4}",
            self.pixels, self.both_hit, self.same_id, self.hit_miss, self.differ, self.depth, self.hit_miss_edge, self.differ_edge, self.max_depth_ratio
        )
    }
}

pub fn agreement(a: &Frame, b: &Frame, refs: &[RefPixel], cam: &Camera, regions: &BTreeMap<RegionKey, RegionMesh>) -> Agreement {
    let list: Vec<&RegionMesh> = regions.values().collect();
    let mut r = Agreement { pixels: refs.len() as u64, ..Agreement::default() };
    let resolve_px = |f: &Frame, i: usize, d: [f64; 3]| -> Option<(f64, VoxelCoord, Face)> {
        if f.surface[i] == BACKGROUND_SURFACE {
            return None;
        }
        let [ri, prim] = f.surface[i];
        let m = list.get(ri as usize)?;
        let q = *m.quads.get(*m.tri_quad.get(prim as usize)? as usize)?;
        let t = cam.distance(f.depth[i]);
        let (v, face) = resolve(q, m.key.origin(m.size), [0, 1, 2].map(|k| cam.eye[k] + t * d[k]));
        Some((t, v, face))
    };
    for (i, rp) in refs.iter().enumerate() {
        let (x, y) = (i as u32 % a.width, i as u32 / a.width);
        let d = cam.dir(x as f64 + 0.5, y as f64 + 0.5);
        let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        match (resolve_px(a, i, d), resolve_px(b, i, d)) {
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                if rp.near_edge {
                    r.hit_miss_edge += 1;
                } else {
                    r.hit_miss += 1;
                    if r.examples.len() < 8 {
                        r.examples.push(format!("({x},{y}): hit/miss: {:?} vs {:?}", a.surface[i], b.surface[i]));
                    }
                }
            }
            (Some((ta, va, fa)), Some((tb, vb, fb))) => {
                r.both_hit += 1;
                if a.surface[i] == b.surface[i] {
                    r.same_id += 1;
                }
                if a.material[i] != b.material[i] || a.normal[i] != b.normal[i] || (va, fa) != (vb, fb) {
                    if rp.near_edge {
                        r.differ_edge += 1;
                    } else {
                        r.differ += 1;
                        if r.examples.len() < 8 {
                            r.examples.push(format!("({x},{y}): {va:?} {fa:?} mat {} vs {vb:?} {fb:?} mat {}", a.material[i], b.material[i]));
                        }
                    }
                    continue;
                }
                // Each frame is within the declared tolerance of the reference, so of each other within twice it.
                let tol = 2.0 * (DEPTH_REL_TOL * ta.min(tb) * dl + DEPTH_ABS_TOL);
                let ratio = (ta - tb).abs() * dl / tol;
                r.max_depth_ratio = r.max_depth_ratio.max(ratio);
                if ratio > 1.0 && !rp.near_edge {
                    r.depth += 1;
                    if r.examples.len() < 8 {
                        r.examples.push(format!("({x},{y}): depth t {ta:.6} vs {tb:.6}"));
                    }
                }
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_normals_encode_to_exact_snorm_bits() {
        let f = |axis, positive| normal_bits(Face { axis, positive });
        assert_eq!(f(0, true), [32767, 0]);
        assert_eq!(f(0, false), [-32767, 0]);
        assert_eq!(f(1, true), [0, 32767]);
        assert_eq!(f(1, false), [0, -32767]);
        assert_eq!(f(2, true), [0, 0]);
        assert_eq!(f(2, false), [32767, 32767]);
        // All six are distinct.
        let mut all: Vec<_> = (0..3).flat_map(|a| [f(a, true), f(a, false)]).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn camera_rays_have_unit_forward_component() {
        let cam = Camera::look_at([1.5, 2.25, -3.0], [40.0, 7.0, 30.0], 60.0, 64, 36, 0.1);
        for (x, y) in [(0, 0), (63, 35), (31, 17)] {
            let r = cam.ray(x, y);
            let f: f64 = (0..3).map(|k| r.dir[k] * cam.forward[k]).sum();
            assert!((f - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn resolve_picks_the_voxel_behind_the_face() {
        let q = RegionQuad { material: 3, face: 3, plane: 5, u0: 0, v0: 0, u1: 4, v1: 4 };
        // +y face at y = 5 (world 13 with origin y 8): the voxel is below the plane.
        let (v, f) = resolve(q, VoxelCoord::new(16, 8, 0), [17.5, 13.0, 2.25]);
        assert_eq!((v, f), (VoxelCoord::new(17, 12, 2), Face { axis: 1, positive: true }));
        let q = RegionQuad { face: 2, ..q };
        let (v, _) = resolve(q, VoxelCoord::new(16, 8, 0), [17.5, 13.0, 2.25]);
        assert_eq!(v, VoxelCoord::new(17, 13, 2));
    }
}
