//! Deterministic procedural test scene: one street block with shops (P-001: "a street with shops").
//!
//! It is a generator of world data only, and writes through the public `World` API. It has no
//! lighting semantics: emissive and glass materials are registered with parameters, but how they
//! transport light is Phase 3 work.

use crate::coords::VoxelCoord;
use crate::dims::VOXEL_SIZE_M;
use crate::material::{MaterialId, MaterialParams, MaterialRegistry};
use crate::world::World;

/// Camera and sun for the scene's reference view. Positions are in meters, right-handed, Y up.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye_m: [f64; 3],
    pub target_m: [f64; 3],
    pub vertical_fov_deg: f64,
    /// Unit vector pointing toward the sun.
    pub sun_dir: [f64; 3],
}

/// Voxels per meter.
fn m(meters: f64) -> i32 {
    (meters / VOXEL_SIZE_M).round() as i32
}

fn v(x: i32, y: i32, z: i32) -> VoxelCoord {
    VoxelCoord::new(x, y, z)
}

struct Mats {
    asphalt: MaterialId,
    road_line: MaterialId,
    sidewalk: MaterialId,
    curb: MaterialId,
    walls: [MaterialId; 3],
    trim: MaterialId,
    glass: MaterialId,
    door: MaterialId,
    awnings: [MaterialId; 3],
    sign: MaterialId,
    roof: MaterialId,
    pole: MaterialId,
    lamp: MaterialId,
}

fn registry() -> (MaterialRegistry, Mats) {
    let mut r = MaterialRegistry::new();
    let mut d = |name: &str, p: MaterialParams| r.register(name, p).expect("unique scene material names");
    let e = |r: f32, g: f32, b: f32, er: f32, eg: f32, eb: f32| MaterialParams { base_color: [r, g, b], emissive: [er, eg, eb] };
    let mats = Mats {
        asphalt: d("asphalt", MaterialParams::diffuse(0.06, 0.06, 0.065)),
        road_line: d("road_line", MaterialParams::diffuse(0.75, 0.72, 0.6)),
        sidewalk: d("sidewalk", MaterialParams::diffuse(0.42, 0.41, 0.39)),
        curb: d("curb", MaterialParams::diffuse(0.55, 0.55, 0.53)),
        walls: [
            d("brick_red", MaterialParams::diffuse(0.38, 0.12, 0.08)),
            d("plaster_cream", MaterialParams::diffuse(0.72, 0.62, 0.45)),
            d("plaster_blue", MaterialParams::diffuse(0.25, 0.36, 0.5)),
        ],
        trim: d("trim_white", MaterialParams::diffuse(0.8, 0.8, 0.78)),
        glass: d("glass", MaterialParams::diffuse(0.1, 0.14, 0.16)),
        door: d("wood_door", MaterialParams::diffuse(0.22, 0.12, 0.06)),
        awnings: [
            d("awning_red", MaterialParams::diffuse(0.55, 0.05, 0.05)),
            d("awning_green", MaterialParams::diffuse(0.05, 0.3, 0.1)),
            d("awning_stripe", MaterialParams::diffuse(0.7, 0.55, 0.1)),
        ],
        sign: d("shop_sign", e(0.9, 0.8, 0.5, 1.0, 0.75, 0.35)),
        roof: d("roof", MaterialParams::diffuse(0.18, 0.17, 0.17)),
        pole: d("metal_pole", MaterialParams::diffuse(0.08, 0.09, 0.08)),
        lamp: d("lamp", e(1.0, 0.9, 0.7, 1.0, 0.85, 0.55)),
    };
    (r, mats)
}

/// Writes boxes into the world; scene materials are always registered, so failures are bugs.
struct Builder {
    w: World,
}

impl Builder {
    fn fill(&mut self, a: VoxelCoord, b: VoxelCoord, mat: MaterialId) {
        self.w.fill_box(a, b, Some(mat)).expect("scene materials are registered");
    }

    fn clear(&mut self, a: VoxelCoord, b: VoxelCoord) {
        self.w.fill_box(a, b, None).expect("clearing never fails");
    }
}

/// Builds the street block. The street runs along +x; buildings stand on the +z side.
pub fn street_block() -> (World, View) {
    let (reg, k) = registry();
    let mut s = Builder { w: World::new(reg) };
    let len = m(24.0); // street length along x

    // Ground: road z in [0, 8) m, sidewalks on both sides, 0.5 m of ground below y = 0.
    let (road_z0, road_z1) = (m(0.0), m(8.0));
    s.fill(v(0, -m(0.5), road_z0 - m(3.0)), v(len, 0, road_z1 + m(3.0)), k.asphalt);
    // Centre line: 12 cm wide dashes, 1.5 m long with 1.5 m gaps, painted as a 1-voxel inlay.
    let mut x = m(0.75);
    while x < len {
        s.fill(v(x, -1, m(3.94)), v((x + m(1.5)).min(len), 0, m(4.06)), k.road_line);
        x += m(3.0);
    }
    // Sidewalks 3 m wide, raised 19 cm (3 voxels), with a curb course along the road edge.
    for (z0, z1, curb_z) in [(road_z0 - m(3.0), road_z0, road_z0 - 2), (road_z1, road_z1 + m(3.0), road_z1)] {
        s.fill(v(0, 0, z0), v(len, 3, z1), k.sidewalk);
        s.fill(v(0, 0, curb_z), v(len, 3, curb_z + 2), k.curb);
    }

    // Three shops on the +z side, 8 m wide, 5 m deep, hollow shells with 25 cm walls.
    let front = road_z1 + m(3.0);
    let depth = m(5.0);
    let wall = 4;
    let heights = [m(7.0), m(9.5), m(6.0)];
    for (i, &h) in heights.iter().enumerate() {
        let x0 = i as i32 * m(8.0);
        let x1 = x0 + m(8.0);
        s.fill(v(x0, 0, front), v(x1, h, front + depth), k.walls[i]);
        s.clear(v(x0 + wall, 0, front + wall), v(x1 - wall, h - wall, front + depth - wall));
        // Flat roof slab with a parapet lip along the front.
        s.fill(v(x0, h, front), v(x1, h + 2, front + depth), k.roof);
        s.fill(v(x0, h + 2, front), v(x1, h + 6, front + 2), k.trim);

        // Shop window: 4.5 m x 2.2 m opening at 0.6 m, glass pane set 3 voxels in, mullion and sill.
        let (wx0, wx1) = (x0 + m(0.6), x0 + m(5.1));
        let (wy0, wy1) = (m(0.6), m(2.8));
        s.clear(v(wx0, wy0, front), v(wx1, wy1, front + wall));
        s.fill(v(wx0, wy0, front + 3), v(wx1, wy1, front + 4), k.glass);
        let mid = (wx0 + wx1) / 2;
        s.fill(v(mid - 1, wy0, front + 2), v(mid + 1, wy1, front + 3), k.trim);
        s.fill(v(wx0 - 2, wy0 - 2, front - 2), v(wx1 + 2, wy0, front), k.trim);

        // Door: 1.2 m x 2.3 m, recessed 3 voxels.
        let (dx0, dx1) = (x0 + m(5.9), x0 + m(7.1));
        s.clear(v(dx0, 0, front), v(dx1, m(2.3), front + wall));
        s.fill(v(dx0, 0, front + 3), v(dx1, m(2.3), front + 4), k.door);

        // Upper-floor windows on the taller shops.
        if h > m(6.5) {
            for j in 0..3 {
                let ux0 = x0 + m(0.9) + j * m(2.4);
                s.clear(v(ux0, m(4.0), front), v(ux0 + m(1.2), m(5.6), front + 3));
                s.fill(v(ux0, m(4.0), front + 3), v(ux0 + m(1.2), m(5.6), front + 4), k.glass);
                s.fill(v(ux0 - 1, m(3.9), front - 1), v(ux0 + m(1.2) + 1, m(4.0), front), k.trim);
            }
        }

        // Awning over the window: 1.2 m deep, sloping down 3 voxels toward the street.
        let aw = m(1.2);
        for d in 0..aw {
            let drop = d * 3 / aw;
            s.fill(v(wx0 - 4, m(3.1) - drop, front - 1 - d), v(wx1 + 4, m(3.1) - drop + 1, front - d), k.awnings[i]);
        }
        // Emissive sign board above the awning.
        s.fill(v(x0 + m(1.0), m(3.3), front - 2), v(x0 + m(6.0), m(3.8), front), k.sign);
    }

    // Street lamps on the far sidewalk every 8 m: 12 cm pole, 4.5 m tall, arm and lamp head over the road.
    for i in 0..3 {
        let px = m(4.0) + i * m(8.0);
        let pz = road_z0 - m(0.6);
        s.fill(v(px, 3, pz), v(px + 2, m(4.5), pz + 2), k.pole);
        s.fill(v(px, m(4.5), pz), v(px + 2, m(4.5) + 2, pz + m(0.8)), k.pole);
        s.fill(v(px - 1, m(4.5) - 3, pz + m(0.6)), v(px + 3, m(4.5), pz + m(0.9)), k.lamp);
    }

    let view = View {
        eye_m: [2.0, 1.7, 2.5],
        target_m: [14.0, 2.6, 12.5],
        vertical_fov_deg: 60.0,
        sun_dir: normalize([-0.45, 0.75, -0.35]),
    };
    (s.w, view)
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_deterministic() {
        let (a, _) = street_block();
        let (b, _) = street_block();
        assert_eq!(a, b);
    }

    #[test]
    fn scene_contains_every_registered_material() {
        let (w, _) = street_block();
        let mut seen = vec![false; w.materials().len()];
        for (_, mat) in w.occupied() {
            seen[mat.raw() as usize] = true;
        }
        let missing: Vec<_> = w.materials().iter().filter(|(id, _)| !seen[id.raw() as usize]).map(|(_, d)| d.name.clone()).collect();
        assert!(missing.is_empty(), "materials never placed: {missing:?}");
    }
}
