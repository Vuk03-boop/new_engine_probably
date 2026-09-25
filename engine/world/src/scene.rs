//! Deterministic procedural test scenes: one street block with shops (P-001: "a street with shops"),
//! and the same street at night with its lights (4A).
//!
//! They are generators of world data only, and write through the public `World` API.
//! [`street_block`] is unchanged since Phase 2 (every M3 test and reference depends on it).
//! [`street_night`] is the same geometry and material ids plus the night dressing; its emitted
//! radiance is in ADR-0005 units (Amendment 3), authored in cd/m².

use crate::coords::VoxelCoord;
use crate::dims::VOXEL_SIZE_M;
use crate::material::{MaterialId, MaterialParams, MaterialRegistry, CANDELA_PER_UNIT};
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

/// Emitted radiance for a colour (normalized to luminance 1) at `cd_per_m2` (ADR-0005 Amendment 3).
pub fn nits(rgb: [f64; 3], cd_per_m2: f64) -> [f32; 3] {
    let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    rgb.map(|c| (c / y * cd_per_m2 / CANDELA_PER_UNIT) as f32)
}

/// The night lights: colour (linear RGB, before normalizing) and luminance in cd/m². Proposed in the
/// 4A record, for the user to adjust.
pub mod night {
    pub const LAMP: ([f64; 3], f64) = ([1.0, 0.80, 0.60], 7_000.0);
    pub const SIGN: ([f64; 3], f64) = ([1.0, 0.75, 0.35], 400.0);
    pub const NEON: [([f64; 3], f64); 3] = [([1.0, 0.10, 0.45], 300.0), ([0.10, 0.75, 1.0], 300.0), ([0.20, 1.0, 0.30], 300.0)];
    pub const WINDOW: ([f64; 3], f64) = ([1.0, 0.72, 0.42], 60.0);
    pub const BULB: ([f64; 3], f64) = ([1.0, 0.55, 0.20], 800.0);
}

/// Night dressing for [`street_night`]; the levels drive the 4A light-count sweep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dressing {
    /// The street's own lamps and signs only.
    Lamps,
    /// + lit upper windows.
    Windows,
    /// + neon outlines on the shop windows + 4 string lights across the street, a bulb every 0.5 m.
    #[default]
    Full,
    /// + string lights every 1 m along the street, a bulb every 0.25 m (measurement only).
    Dense,
}

impl Dressing {
    pub const ALL: [Dressing; 4] = [Dressing::Lamps, Dressing::Windows, Dressing::Full, Dressing::Dense];

    pub fn name(self) -> &'static str {
        match self {
            Dressing::Lamps => "lamps",
            Dressing::Windows => "windows",
            Dressing::Full => "full",
            Dressing::Dense => "dense",
        }
    }
}

/// Materials only the night street registers (after every `street_block` material, so shared ids agree).
struct NightMats {
    neon: [MaterialId; 3],
    window: MaterialId,
    bulb: MaterialId,
    wire: MaterialId,
}

fn registry() -> (MaterialRegistry, Mats) {
    registry_with(false)
}

/// The street's registry; with `night`, the lamps and signs carry their night luminance.
fn registry_with(night: bool) -> (MaterialRegistry, Mats) {
    let mut r = MaterialRegistry::new();
    let mut d = |name: &str, p: MaterialParams| r.register(name, p).expect("unique scene material names");
    let e = |r: f32, g: f32, b: f32, er: f32, eg: f32, eb: f32| MaterialParams { base_color: [r, g, b], emissive: [er, eg, eb] };
    let lit = |base: [f32; 3], (rgb, cd): ([f64; 3], f64)| MaterialParams { base_color: base, emissive: nits(rgb, cd) };
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
        sign: d("shop_sign", if night { lit([0.9, 0.8, 0.5], night::SIGN) } else { e(0.9, 0.8, 0.5, 1.0, 0.75, 0.35) }),
        roof: d("roof", MaterialParams::diffuse(0.18, 0.17, 0.17)),
        pole: d("metal_pole", MaterialParams::diffuse(0.08, 0.09, 0.08)),
        lamp: d("lamp", if night { lit([1.0, 0.9, 0.7], night::LAMP) } else { e(1.0, 0.9, 0.7, 1.0, 0.85, 0.55) }),
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
    let view = build_street(&mut s, &k);
    (s.w, view)
}

/// The street block at night (4A): the same geometry and material ids as [`street_block`], the
/// lamps and signs at their night luminance, plus the `dressing`.
pub fn street_night(dressing: Dressing) -> (World, View) {
    let (mut reg, k) = registry_with(true);
    let mut d = |name: &str, p: MaterialParams| reg.register(name, p).expect("unique scene material names");
    let lit = |base: [f32; 3], (rgb, cd): ([f64; 3], f64)| MaterialParams { base_color: base, emissive: nits(rgb, cd) };
    let n = NightMats {
        neon: [0, 1, 2].map(|i| d(["neon_pink", "neon_cyan", "neon_green"][i], lit([0.8, 0.8, 0.8], night::NEON[i]))),
        window: d("window_lit", lit([0.1, 0.14, 0.16], night::WINDOW)),
        bulb: d("bulb", lit([0.9, 0.9, 0.85], night::BULB)),
        wire: d("wire", MaterialParams::diffuse(0.04, 0.04, 0.04)),
    };
    let mut s = Builder { w: World::new(reg) };
    let view = build_street(&mut s, &k);
    dress(&mut s, &k, &n, dressing);
    (s.w, view)
}

// The street's layout, shared by `build_street` and `dress` (voxels).
const LEN: f64 = 24.0;
const ROAD_Z1: f64 = 8.0;
const SHOP_WIDTH: f64 = 8.0;
const SHOP_HEIGHTS: [f64; 3] = [7.0, 9.5, 6.0];
const LAMP_XS: [f64; 3] = [4.0, 12.0, 20.0];

fn dress(s: &mut Builder, k: &Mats, n: &NightMats, dressing: Dressing) {
    let front = m(ROAD_Z1) + m(3.0);
    if dressing >= Dressing::Windows {
        // Lit upper windows: the same panes as the glass, now emissive.
        for (i, &h) in SHOP_HEIGHTS.iter().enumerate() {
            let x0 = i as i32 * m(SHOP_WIDTH);
            if m(h) > m(6.5) {
                for j in 0..3 {
                    let ux0 = x0 + m(0.9) + j * m(2.4);
                    s.fill(v(ux0, m(4.0), front + 3), v(ux0 + m(1.2), m(5.6), front + 4), n.window);
                }
            }
        }
    }
    if dressing >= Dressing::Full {
        // Neon outline around each shop window, one voxel in front of the façade, under the awning
        // and around the sill.
        for i in 0..3 {
            let x0 = i * m(SHOP_WIDTH);
            let (wx0, wx1) = (x0 + m(0.6), x0 + m(5.1));
            let (y0, y1) = (m(0.6) - 4, m(2.8) + 1);
            let z = front - 1;
            let neon = n.neon[i as usize];
            s.fill(v(wx0 - 3, y0, z), v(wx1 + 3, y0 + 1, z + 1), neon);
            s.fill(v(wx0 - 3, y1, z), v(wx1 + 3, y1 + 1, z + 1), neon);
            s.fill(v(wx0 - 3, y0, z), v(wx0 - 2, y1 + 1, z + 1), neon);
            s.fill(v(wx1 + 2, y0, z), v(wx1 + 3, y1 + 1, z + 1), neon);
        }
    }
    let (xs, spacing): (Vec<i32>, i32) = match dressing {
        Dressing::Lamps | Dressing::Windows => (Vec::new(), 0),
        Dressing::Full => ([2.0, 8.0, 16.0, 22.0].iter().map(|&x| m(x)).collect(), 8),
        Dressing::Dense => ((0..24).map(|i| m(0.5) + i * m(1.0)).collect(), 4),
    };
    for x in xs {
        // Keep clear of the lamp heads and arms.
        if LAMP_XS.iter().any(|&lx| (x - m(lx)).abs() <= 4) {
            continue;
        }
        string_light(s, k, n, x, front - 1, spacing);
    }
}

/// A string light across the street at `x`: a thin pole on the far sidewalk, a sagging wire to the
/// façade (a parabola, 0.8 m sag between anchors at 5 m) and a bulb below it every `spacing` voxels.
fn string_light(s: &mut Builder, k: &Mats, n: &NightMats, x: i32, z_facade: i32, spacing: i32) {
    let (za, zb) = (-m(2.5), z_facade);
    let (ya, sag) = (m(5.0), m(0.8) as f64);
    s.fill(v(x, 3, za - 2), v(x + 1, ya, za), k.pole);
    let y_at = |z: i32| {
        let t = (z - za) as f64 / (zb - za) as f64;
        (ya as f64 - 4.0 * sag * t * (1.0 - t)).round() as i32
    };
    for z in za..=zb {
        let (y, y_next) = (y_at(z), y_at((z + 1).min(zb)));
        s.fill(v(x, y.min(y_next), z), v(x + 1, y.max(y_next) + 1, z + 1), n.wire);
    }
    let mut z = za + spacing;
    while z <= zb - spacing / 2 {
        let y = (y_at(z - 1).min(y_at(z)).min(y_at(z + 1))) - 1;
        s.fill(v(x, y, z), v(x + 1, y + 1, z + 1), n.bulb);
        z += spacing;
    }
}

/// The street's geometry; returns its reference view.
fn build_street(s: &mut Builder, k: &Mats) -> View {
    let len = m(LEN); // street length along x

    // Ground: road z in [0, 8) m, sidewalks on both sides, 0.5 m of ground below y = 0.
    let (road_z0, road_z1) = (m(0.0), m(ROAD_Z1));
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
    let heights = SHOP_HEIGHTS.map(m);
    for (i, &h) in heights.iter().enumerate() {
        let x0 = i as i32 * m(SHOP_WIDTH);
        let x1 = x0 + m(SHOP_WIDTH);
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
    for lx in LAMP_XS {
        let px = m(lx);
        let pz = road_z0 - m(0.6);
        s.fill(v(px, 3, pz), v(px + 2, m(4.5), pz + 2), k.pole);
        s.fill(v(px, m(4.5), pz), v(px + 2, m(4.5) + 2, pz + m(0.8)), k.pole);
        s.fill(v(px - 1, m(4.5) - 3, pz + m(0.6)), v(px + 3, m(4.5), pz + m(0.9)), k.lamp);
    }

    View {
        eye_m: [2.0, 1.7, 2.5],
        target_m: [14.0, 2.6, 12.5],
        vertical_fov_deg: 60.0,
        sun_dir: normalize([-0.45, 0.75, -0.35]),
    }
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
    fn night_street_keeps_the_day_street_and_adds_lights() {
        let (day, _) = street_block();
        for d in Dressing::ALL {
            let (night, _) = street_night(d);
            assert_eq!(night, street_night(d).0, "{d:?} is deterministic");
            // Shared ids have the same names; lamps and signs changed only their emission.
            for (id, def) in day.materials().iter() {
                let nd = night.materials().get(id).unwrap();
                assert_eq!((&nd.name, nd.params.base_color), (&def.name, def.params.base_color));
            }
            // Every day voxel is still there with the same material, except the lit window panes.
            let window = night.materials().id_of("window_lit").unwrap();
            for (v, m) in day.occupied() {
                let nm = night.get(v).unwrap();
                assert!(nm == m || nm == window, "{d:?}: {v:?} changed");
            }
        }
        let lamp = street_night(Dressing::Lamps).0;
        let l = lamp.materials().get(lamp.materials().id_of("lamp").unwrap()).unwrap().params.emissive;
        let y = 0.2126 * l[0] as f64 + 0.7152 * l[1] as f64 + 0.0722 * l[2] as f64;
        assert!((y * CANDELA_PER_UNIT / night::LAMP.1 - 1.0).abs() < 1e-6, "lamp luminance {y}");
        let count = |d| street_night(d).0.occupied().count();
        assert!(count(Dressing::Lamps) < count(Dressing::Windows) + 1 && count(Dressing::Windows) < count(Dressing::Full) && count(Dressing::Full) < count(Dressing::Dense));
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
