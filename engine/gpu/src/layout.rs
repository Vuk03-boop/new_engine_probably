//! ADR-0004 GPU mesh layout, built on the CPU (pure; no device needed).
//!
//! Bricks are grouped into **acceleration regions** (the second ADR-0003 granularity parameter).
//! Each region gets one byte image with four aligned sections (ADR-0004 amendment 1):
//!
//! - **vertices:** `R16G16B16A16_SFLOAT`, region-local voxel units, `w = 1`. Values are multiples
//!   of 1/2 (fan centres of split quads), exact in f16. From [`derived::BrickMesh::triangulate`]
//!   with [`Split::Watertight`], so every triangle edge is shared (no T-junction cracks).
//! - **indices:** `u32`, three per triangle, counter-clockwise from outside;
//! - **triangle quads:** one `u32` per triangle, the index of its quad: primitive `p` belongs to
//!   quad `tri_quad[p]` (replaces the old `p / 2`);
//! - **quads:** one `u64` per quad, region-local (bit layout below; the shader constants in
//!   `shaders/mesh_decode.slang` must match, which the GPU decode test checks by execution).
//!
//! Quad bits: material `[0,16)`, face `[16,19)`, plane `[19,26)`, u0 `[26,33)`, v0 `[33,40)`,
//! u1 `[40,47)`, v1 `[47,54)`. Region-local coordinates are `0..=64`, so 7 bits each.
//! Within a region, bricks are in `BrickKey` order and each brick's quads keep their order.

use std::collections::BTreeMap;

use derived::{BrickMesh, Quad, Split};
use world::dims::BRICK_EDGE;
use world::{BrickKey, VoxelCoord};

pub const MATERIAL_SHIFT: u32 = 0;
pub const FACE_SHIFT: u32 = 16;
pub const PLANE_SHIFT: u32 = 19;
pub const U0_SHIFT: u32 = 26;
pub const V0_SHIFT: u32 = 33;
pub const U1_SHIFT: u32 = 40;
pub const V1_SHIFT: u32 = 47;
pub const COORD_BITS: u32 = 7;

/// Device bytes: 8 per vertex, 16 per triangle (12 of indices, 4 of triangle quad), 8 per quad.
pub const BYTES_PER_VERTEX: u64 = 8;
pub const BYTES_PER_TRIANGLE: u64 = 16;
pub const BYTES_PER_QUAD: u64 = 8;

/// Acceleration region edge (ADR-0003 slider).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegionSize {
    Brick,
    Bricks2,
    Chunk,
    Chunks2,
}

impl RegionSize {
    pub const ALL: [RegionSize; 4] = [RegionSize::Brick, RegionSize::Bricks2, RegionSize::Chunk, RegionSize::Chunks2];

    /// Edge in bricks.
    pub fn bricks(self) -> i32 {
        match self {
            RegionSize::Brick => 1,
            RegionSize::Bricks2 => 2,
            RegionSize::Chunk => 4,
            RegionSize::Chunks2 => 8,
        }
    }

    pub fn voxels(self) -> i32 {
        self.bricks() * BRICK_EDGE
    }

    pub fn name(self) -> &'static str {
        match self {
            RegionSize::Brick => "brick",
            RegionSize::Bricks2 => "2x2x2_bricks",
            RegionSize::Chunk => "chunk",
            RegionSize::Chunks2 => "2x2x2_chunks",
        }
    }
}

const _: () = assert!(8 * BRICK_EDGE < (1 << COORD_BITS), "the largest region's coordinates (0..=64) fit 7 bits");

/// Region identity: region coordinates in units of the region edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl RegionKey {
    pub fn of(brick: BrickKey, size: RegionSize) -> Self {
        let o = brick.origin();
        let e = size.voxels();
        Self { x: o.x.div_euclid(e), y: o.y.div_euclid(e), z: o.z.div_euclid(e) }
    }

    pub fn origin(self, size: RegionSize) -> VoxelCoord {
        let e = size.voxels();
        VoxelCoord::new(self.x * e, self.y * e, self.z * e)
    }
}

/// A quad in region-local coordinates, as the device stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionQuad {
    pub material: u16,
    pub face: u8,
    pub plane: u8,
    pub u0: u8,
    pub v0: u8,
    pub u1: u8,
    pub v1: u8,
}

impl RegionQuad {
    fn from_brick(q: Quad, offset: [u8; 3]) -> Self {
        let a = q.axis();
        let (ua, va) = q.uv_axes();
        Self {
            material: q.material.raw(),
            face: q.face,
            plane: q.plane + offset[a],
            u0: q.u0 + offset[ua],
            v0: q.v0 + offset[va],
            u1: q.u1 + offset[ua],
            v1: q.v1 + offset[va],
        }
    }

    pub fn pack(self) -> u64 {
        (self.material as u64) << MATERIAL_SHIFT
            | (self.face as u64) << FACE_SHIFT
            | (self.plane as u64) << PLANE_SHIFT
            | (self.u0 as u64) << U0_SHIFT
            | (self.v0 as u64) << V0_SHIFT
            | (self.u1 as u64) << U1_SHIFT
            | (self.v1 as u64) << V1_SHIFT
    }

    pub fn unpack(bits: u64) -> Self {
        let f = |shift: u32, n: u32| ((bits >> shift) & ((1u64 << n) - 1)) as u8;
        Self {
            material: (bits >> MATERIAL_SHIFT) as u16,
            face: f(FACE_SHIFT, 3),
            plane: f(PLANE_SHIFT, COORD_BITS),
            u0: f(U0_SHIFT, COORD_BITS),
            v0: f(V0_SHIFT, COORD_BITS),
            u1: f(U1_SHIFT, COORD_BITS),
            v1: f(V1_SHIFT, COORD_BITS),
        }
    }

    /// Corners in ADR-0004 order, region-local (the same rule as `derived::Quad::corners`).
    pub fn corners(self) -> [[u8; 3]; 4] {
        let a = (self.face / 2) as usize;
        let (ua, va) = ((a + 1) % 3, (a + 2) % 3);
        let at = |u: u8, v: u8| {
            let mut p = [0u8; 3];
            p[a] = self.plane;
            p[ua] = u;
            p[va] = v;
            p
        };
        let (c0, c1, c2, c3) = (at(self.u0, self.v0), at(self.u1, self.v0), at(self.u1, self.v1), at(self.u0, self.v1));
        if self.face % 2 == 1 { [c0, c1, c2, c3] } else { [c0, c3, c2, c1] }
    }
}

/// Exact half-float bits of a small non-negative integer (`n < 2048`).
pub fn f16_of_int(n: u32) -> u16 {
    assert!(n < 2048, "{n} is not exact in f16");
    if n == 0 {
        return 0;
    }
    let e = 31 - n.leading_zeros();
    let mantissa = (n << (10 - e)) & 0x3FF;
    (((e + 15) << 10) | mantissa) as u16
}

/// Exact half-float bits of a non-negative multiple of 1/2 below 1024, given in half units
/// (`halves = 2 × value`).
pub fn f16_of_halves(halves: u32) -> u16 {
    assert!(halves < 2048, "{halves}/2 is not exact in f16");
    if halves == 0 {
        return 0;
    }
    let e = 31 - halves.leading_zeros();
    let mantissa = (halves << (10 - e)) & 0x3FF;
    (((e + 14) << 10) | mantissa) as u16
}

/// `1.0` in f16.
pub const F16_ONE: u16 = 0x3C00;

/// Section offsets within a region's byte image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sections {
    pub vertices: u64,
    pub indices: u64,
    pub tri_quad: u64,
    pub quads: u64,
    pub total: u64,
}

impl Sections {
    pub fn new(vertices: u64, triangles: u64, quads: u64, align: u64) -> Self {
        assert!(align.is_power_of_two());
        let up = |x: u64| (x + align - 1) & !(align - 1);
        let v = 0;
        let i = up(v + vertices * BYTES_PER_VERTEX);
        let t = up(i + triangles * 12);
        let q = up(t + triangles * 4);
        Self { vertices: v, indices: i, tri_quad: t, quads: q, total: up(q + quads * BYTES_PER_QUAD).max(align) }
    }
}

/// One region's device image, built on the CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionMesh {
    pub key: RegionKey,
    pub size: RegionSize,
    pub quads: Vec<RegionQuad>,
    /// Region-local, in half-voxel units (×2), `0..=128`.
    pub vertices: Vec<[u8; 3]>,
    /// Counter-clockwise from outside.
    pub indices: Vec<[u32; 3]>,
    /// For each triangle (primitive), the index of its quad.
    pub tri_quad: Vec<u32>,
}

impl RegionMesh {
    pub fn quad_count(&self) -> u64 {
        self.quads.len() as u64
    }

    pub fn vertex_count(&self) -> u64 {
        self.vertices.len() as u64
    }

    pub fn triangle_count(&self) -> u64 {
        self.indices.len() as u64
    }

    pub fn sections(&self, align: u64) -> Sections {
        Sections::new(self.vertex_count(), self.triangle_count(), self.quad_count(), align)
    }

    /// The full byte image with sections aligned to `align` (≥ the device's storage-buffer offset alignment).
    pub fn image(&self, align: u64) -> (Sections, Vec<u8>) {
        let s = self.sections(align);
        let mut b = vec![0u8; s.total as usize];
        for (i, p) in self.vertices.iter().enumerate() {
            let o = s.vertices as usize + i * 8;
            for (k, &x) in p.iter().enumerate() {
                b[o + 2 * k..o + 2 * k + 2].copy_from_slice(&f16_of_halves(x as u32).to_le_bytes());
            }
            b[o + 6..o + 8].copy_from_slice(&F16_ONE.to_le_bytes());
        }
        for (i, tri) in self.indices.iter().enumerate() {
            for (k, idx) in tri.iter().enumerate() {
                let o = s.indices as usize + (3 * i + k) * 4;
                b[o..o + 4].copy_from_slice(&idx.to_le_bytes());
            }
            let o = s.tri_quad as usize + i * 4;
            b[o..o + 4].copy_from_slice(&self.tri_quad[i].to_le_bytes());
        }
        for (i, q) in self.quads.iter().enumerate() {
            let o = s.quads as usize + i * 8;
            b[o..o + 8].copy_from_slice(&q.pack().to_le_bytes());
        }
        (s, b)
    }
}

/// Groups brick meshes into regions with the watertight triangulation. Regions without quads are omitted.
pub fn build_regions<'a>(meshes: impl IntoIterator<Item = (BrickKey, &'a BrickMesh)>, size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    build_regions_with(meshes, size, Split::Watertight)
}

/// [`build_regions`] with an explicit split rule; [`Split::CornersOnly`] is for negative controls.
pub fn build_regions_with<'a>(meshes: impl IntoIterator<Item = (BrickKey, &'a BrickMesh)>, size: RegionSize, split: Split) -> BTreeMap<RegionKey, RegionMesh> {
    let mut by_region: BTreeMap<RegionKey, Vec<(BrickKey, &BrickMesh)>> = BTreeMap::new();
    for (k, m) in meshes {
        if !m.quads.is_empty() {
            by_region.entry(RegionKey::of(k, size)).or_default().push((k, m));
        }
    }
    by_region
        .into_iter()
        .map(|(rk, mut bricks)| {
            bricks.sort_by_key(|(k, _)| *k);
            let ro = rk.origin(size);
            let mut r = RegionMesh { key: rk, size, quads: Vec::new(), vertices: Vec::new(), indices: Vec::new(), tri_quad: Vec::new() };
            for (k, m) in bricks {
                let o = k.origin();
                let off = [(o.x - ro.x) as u8, (o.y - ro.y) as u8, (o.z - ro.z) as u8];
                let t = m.triangulate(split);
                let (vbase, qbase) = (r.vertices.len() as u32, r.quads.len() as u32);
                r.vertices.extend(t.vertices.iter().map(|v| [0, 1, 2].map(|a| v[a] + 2 * off[a])));
                r.indices.extend(t.indices.iter().map(|tri| tri.map(|i| i + vbase)));
                r.tri_quad.extend(t.quad.iter().map(|&q| q + qbase));
                r.quads.extend(m.quads.iter().map(|&q| RegionQuad::from_brick(q, off)));
            }
            (rk, r)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use derived::{extract_world, Merge};
    use world::{scene, MaterialParams, MaterialRegistry, World};

    #[test]
    fn f16_integers_are_exact() {
        // Reference values: 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200, 64.0 = 0x5400, 1024 = 0x6400.
        assert_eq!([0, 1, 2, 3, 64, 1024].map(f16_of_int), [0, 0x3C00, 0x4000, 0x4200, 0x5400, 0x6400]);
        assert_eq!(f16_of_int(1), F16_ONE);
        // Halves: 0.5 = 0x3800, 1.5 = 0x3E00, 64.5 = 0x5408; integers agree with f16_of_int.
        assert_eq!([1, 3, 129].map(f16_of_halves), [0x3800, 0x3E00, 0x5408]);
        for n in 0..1024 {
            assert_eq!(f16_of_halves(2 * n), f16_of_int(n));
        }
        // Round-trip every value a region can hold through an independent decoder.
        for n in 0..=64u32 {
            let h = f16_of_int(n) as u32;
            let (e, m) = ((h >> 10) & 0x1F, h & 0x3FF);
            let v = if e == 0 { m as f64 / 1024.0 * 2f64.powi(-14) } else { (1.0 + m as f64 / 1024.0) * 2f64.powi(e as i32 - 15) };
            assert_eq!(v, n as f64);
        }
    }

    #[test]
    fn pack_round_trips_and_fields_do_not_overlap() {
        let q = RegionQuad { material: 0xFFFF, face: 5, plane: 64, u0: 63, v0: 1, u1: 64, v1: 64 };
        assert_eq!(RegionQuad::unpack(q.pack()), q);
        let widths = [(MATERIAL_SHIFT, 16), (FACE_SHIFT, 3), (PLANE_SHIFT, 7), (U0_SHIFT, 7), (V0_SHIFT, 7), (U1_SHIFT, 7), (V1_SHIFT, 7)];
        let mut used = 0u64;
        for (s, n) in widths {
            let m = ((1u64 << n) - 1) << s;
            assert_eq!(used & m, 0, "field at {s} overlaps");
            used |= m;
        }
        assert_eq!(used, (1u64 << 54) - 1, "contiguous 54 bits");
    }

    #[test]
    fn regions_preserve_every_quad_in_world_position() {
        let (w, _) = scene::street_block();
        let meshes: Vec<_> = w.bricks().map(|(k, _)| (k, extract_world(&w, k, Merge::Greedy).unwrap())).collect();
        let total: usize = meshes.iter().map(|(_, m)| m.quads.len()).sum();
        for size in RegionSize::ALL {
            let regions = build_regions(meshes.iter().map(|(k, m)| (*k, m)), size);
            assert_eq!(regions.values().map(|r| r.quads.len()).sum::<usize>(), total, "{size:?}");
            // World-space corners agree with the brick-local ones.
            let mut from_regions: Vec<[i32; 3]> = Vec::new();
            for r in regions.values() {
                let o = r.key.origin(size);
                for q in &r.quads {
                    assert!([q.plane, q.u0, q.v0, q.u1, q.v1].iter().all(|&x| x as i32 <= size.voxels()));
                    for c in q.corners() {
                        from_regions.push([o.x + c[0] as i32, o.y + c[1] as i32, o.z + c[2] as i32]);
                    }
                }
            }
            let mut from_bricks: Vec<[i32; 3]> = Vec::new();
            for (k, m) in &meshes {
                let o = k.origin();
                for q in &m.quads {
                    for c in q.corners() {
                        from_bricks.push([o.x + c[0] as i32, o.y + c[1] as i32, o.z + c[2] as i32]);
                    }
                }
            }
            from_regions.sort();
            from_bricks.sort();
            assert_eq!(from_regions, from_bricks, "{size:?}");
            // Every triangle lies on its quad's rectangle, region-local (half-voxel units).
            for r in regions.values() {
                assert_eq!(r.indices.len(), r.tri_quad.len());
                for (tri, &qi) in r.indices.iter().zip(&r.tri_quad) {
                    let q = r.quads[qi as usize];
                    let a = (q.face / 2) as usize;
                    let (ua, va) = ((a + 1) % 3, (a + 2) % 3);
                    for &i in tri {
                        let p = r.vertices[i as usize];
                        assert_eq!(p[a], 2 * q.plane, "{size:?}");
                        assert!((2 * q.u0..=2 * q.u1).contains(&p[ua]) && (2 * q.v0..=2 * q.v1).contains(&p[va]), "{size:?}: {p:?} outside {q:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn negative_coordinates_group_into_the_right_region() {
        let mut r = MaterialRegistry::new();
        let m = r.register("m", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
        let mut w = World::new(r);
        w.set(VoxelCoord::new(-1, -1, -1), Some(m)).unwrap();
        let (k, _) = VoxelCoord::new(-1, -1, -1).split();
        for size in RegionSize::ALL {
            let rk = RegionKey::of(k, size);
            assert_eq!((rk.x, rk.y, rk.z), (-1, -1, -1));
            let mesh = extract_world(&w, k, Merge::Greedy).unwrap();
            let regions = build_regions([(k, &mesh)], size);
            let q = regions[&rk].quads[0];
            // The voxel at -1 is the last voxel of its region.
            let top = size.voxels() as u8;
            assert!(q.plane == top || q.plane == top - 1, "{size:?}: {q:?}");
        }
    }

    #[test]
    fn image_sections_are_aligned_and_hold_the_triangulation() {
        // One 2x1 slab on the ground: its greedy top quad has a split point on its boundary edges.
        let mut reg = MaterialRegistry::new();
        let m = reg.register("m", MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap();
        let mut w = World::new(reg);
        w.fill_box(VoxelCoord::new(0, 0, 0), VoxelCoord::new(2, 1, 1), Some(m)).unwrap();
        let (k, _) = VoxelCoord::new(0, 0, 0).split();
        let mesh = extract_world(&w, k, Merge::Greedy).unwrap();
        let rs = build_regions([(k, &mesh)], RegionSize::Chunk);
        let r = rs.values().next().unwrap();
        assert!(r.triangle_count() > 2 * r.quad_count(), "split quads are fans");
        let (s, b) = r.image(64);
        assert!([s.indices, s.tri_quad, s.quads, s.total].iter().all(|x| x % 64 == 0));
        assert_eq!(b.len() as u64, s.total);
        let u32_at = |o: u64| u32::from_le_bytes(b[o as usize..][..4].try_into().unwrap());
        for (i, tri) in r.indices.iter().enumerate() {
            for (k, &idx) in tri.iter().enumerate() {
                assert_eq!(u32_at(s.indices + 4 * (3 * i as u64 + k as u64)), idx);
            }
            assert_eq!(u32_at(s.tri_quad + 4 * i as u64), r.tri_quad[i]);
        }
        assert_eq!(u64::from_le_bytes(b[s.quads as usize..][..8].try_into().unwrap()), r.quads[0].pack());
        // Unmerged: two triangles per quad, tri_quad = p / 2 as before the amendment.
        let none = extract_world(&w, k, Merge::None).unwrap();
        let r = build_regions([(k, &none)], RegionSize::Chunk).into_values().next().unwrap();
        assert!(r.tri_quad.iter().enumerate().all(|(p, &q)| q as usize == p / 2));
    }
}
