//! Walking collision: resting, stepping, sliding, tunnelling, ceilings, and a seeded random walk
//! through the street block that checks the no-overlap invariant after every substep.

use std::collections::HashSet;

use walk::{Intent, Params, Solid, Walker};
use world::dims::VOXEL_SIZE_M;
use world::scene;

/// A set of solid voxels, for small hand-made scenes.
#[derive(Default)]
struct Grid(HashSet<(i32, i32, i32)>);

impl Grid {
    /// Fills the half-open box `[a, b)`.
    fn fill(&mut self, a: [i32; 3], b: [i32; 3]) -> &mut Self {
        for x in a[0]..b[0] {
            for y in a[1]..b[1] {
                for z in a[2]..b[2] {
                    self.0.insert((x, y, z));
                }
            }
        }
        self
    }

    /// A floor whose top face is y = 0, over x and z in [-64, 64).
    fn floor() -> Grid {
        let mut g = Grid::default();
        g.fill([-64, -1, -64], [64, 0, 64]);
        g
    }
}

impl Solid for Grid {
    fn solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.0.contains(&(x, y, z))
    }
}

fn walk_for<S: Solid>(w: &mut Walker, s: &S, intent: Intent, seconds: f64) {
    let n = (seconds / w.params.substep_s).round() as u64;
    for _ in 0..n {
        w.substep(s, intent);
        assert!(!w.overlaps(s), "box overlaps a solid voxel at {:?}", w.feet);
    }
}

fn toward(x: f64, z: f64) -> Intent {
    Intent { dir: [x, z], ..Intent::default() }
}

fn m(x: f64) -> f64 {
    x / VOXEL_SIZE_M
}

#[test]
fn lands_and_rests_exactly_on_the_floor() {
    let g = Grid::floor();
    let mut w = Walker::new(Params::default(), [0.5, 20.25, 0.5]);
    walk_for(&mut w, &g, Intent::default(), 2.0);
    assert!(w.grounded);
    assert_eq!(w.feet[1], 0.0, "flush on the floor's top face");
    let rest = w.feet;
    walk_for(&mut w, &g, Intent::default(), 10.0);
    assert_eq!(w.feet, rest, "no drift while standing");
    assert!(w.grounded);
}

/// The step-height slider: which obstacle heights are climbed without jumping.
#[test]
fn step_height_sweep() {
    let mut table = Vec::new();
    for step in [0.0, 2.0, 3.0, 4.0, 6.0] {
        let mut row = Vec::new();
        for h in 1..=8 {
            let mut g = Grid::floor();
            g.fill([16, 0, -64], [64, h, 64]);
            let p = Params { step_height: step, ..Params::default() };
            let mut w = Walker::new(p, [0.0, 0.0, 0.0]);
            w.grounded = true;
            walk_for(&mut w, &g, toward(1.0, 0.0), 2.0);
            let climbed = w.feet[0] > 16.0;
            assert_eq!(climbed, (h as f64) <= step, "step {step}, obstacle {h}: feet {:?}", w.feet);
            if climbed {
                assert_eq!(w.feet[1], h as f64, "stands flush on the obstacle");
                assert_eq!(w.stats.steps_up, 1);
            } else {
                assert_eq!(w.feet[0], 16.0 - w.params.half_width, "stopped flush against it");
                assert_eq!(w.feet[1], 0.0);
            }
            row.push(if climbed { "up" } else { "--" });
        }
        table.push(format!("step {step}: {}", row.join(" ")));
    }
    println!("step height (voxels) vs obstacle height 1..=8:\n{}", table.join("\n"));
}

#[test]
fn step_up_needs_headroom() {
    // A 3-voxel ledge with a ceiling above it: the gap must fit the 28.8-voxel box.
    for (gap, fits) in [(28, false), (29, true)] {
        let mut g = Grid::floor();
        g.fill([16, 0, -64], [64, 3, 64]);
        g.fill([16, 3 + gap, -64], [64, 3 + gap + 1, 64]);
        let mut w = Walker::new(Params::default(), [0.0, 0.0, 0.0]);
        w.grounded = true;
        walk_for(&mut w, &g, toward(1.0, 0.0), 2.0);
        assert_eq!(w.feet[0] > 16.0, fits, "gap {gap}: feet {:?}", w.feet);
    }
}

#[test]
fn snaps_down_a_curb_instead_of_falling() {
    let mut g = Grid::floor();
    g.fill([-64, 0, -64], [16, 3, 64]);
    let mut w = Walker::new(Params::default(), [0.0, 3.0, 0.0]);
    w.grounded = true;
    let n = (2.0 / w.params.substep_s) as u64;
    for _ in 0..n {
        w.substep(&g, toward(1.0, 0.0));
        assert!(w.grounded, "airborne walking off a 3-voxel curb at {:?}", w.feet);
    }
    assert_eq!(w.feet[1], 0.0);
    assert_eq!(w.stats.snaps_down, 1);
    // A drop deeper than the step height is a fall.
    let mut g = Grid::floor();
    g.fill([-64, 0, -64], [16, 8, 64]);
    let mut w = Walker::new(Params::default(), [0.0, 8.0, 0.0]);
    w.grounded = true;
    walk_for(&mut w, &g, toward(1.0, 0.0), 2.0);
    assert_eq!(w.stats.snaps_down, 0);
    assert_eq!(w.feet[1], 0.0);
}

#[test]
fn slides_along_walls_without_snagging() {
    // A wall at x = 10 built from separate 1-voxel-thick slabs (seams every 3 voxels in z and y).
    let mut g = Grid::floor();
    for z in (-64..64).step_by(3) {
        for y in (0..40).step_by(3) {
            g.fill([10, y, z], [11, y + 3, z + 3]);
        }
    }
    let mut w = Walker::new(Params::default(), [0.0, 0.0, -40.0]);
    w.grounded = true;
    let d = std::f64::consts::FRAC_1_SQRT_2;
    walk_for(&mut w, &g, toward(d, d), 0.5); // reaches the wall
    assert_eq!(w.feet[0], 10.0 - w.params.half_width);
    let z0 = w.feet[2];
    let t = 1.5;
    walk_for(&mut w, &g, toward(d, d), t);
    let expected = w.params.walk_speed * d * t;
    assert!((w.feet[2] - z0 - expected).abs() < 1e-6, "slid {} of {expected}", w.feet[2] - z0);
    assert_eq!(w.feet[1], 0.0);
    assert_eq!(w.feet[0], 10.0 - w.params.half_width);
    // Walking on a floor also has no seams to catch on: exact distance over many voxel edges.
    let mut w = Walker::new(Params::default(), [-50.0, 0.0, 0.3]);
    w.grounded = true;
    walk_for(&mut w, &Grid::floor(), toward(1.0, 0.0), 2.0);
    assert!((w.feet[0] + 50.0 - w.params.walk_speed * 2.0).abs() < 1e-6);
}

#[test]
fn inside_corner_holds() {
    let mut g = Grid::floor();
    g.fill([10, 0, -64], [11, 40, 64]).fill([-64, 0, 10], [64, 40, 11]);
    let mut w = Walker::new(Params::default(), [0.0, 0.0, 0.0]);
    w.grounded = true;
    walk_for(&mut w, &g, Intent { dir: [1.0, 1.0], run: true, jump: false }, 3.0);
    let e = 10.0 - w.params.half_width;
    assert_eq!((w.feet[0], w.feet[2]), (e, e));
}

#[test]
fn thin_walls_and_floors_do_not_tunnel() {
    // 1-voxel wall, at 5000 voxels/s with 1/30 s substeps: 167 voxels per substep.
    let mut g = Grid::floor();
    g.fill([30, 0, -64], [31, 40, 64]);
    let p = Params { walk_speed: 5000.0, substep_s: 1.0 / 30.0, max_frame_s: 1.0, ..Params::default() };
    let mut w = Walker::new(p, [0.0, 0.0, 0.0]);
    w.grounded = true;
    walk_for(&mut w, &g, toward(1.0, 0.0), 1.0);
    assert_eq!(w.feet[0], 30.0 - p.half_width);
    // Falling 20000 voxels onto a 1-voxel slab at up to 10000 voxels/s.
    let mut g = Grid::default();
    g.fill([-8, 100, -8], [8, 101, 8]);
    let p = Params { max_fall_speed: 10000.0, gravity: 50000.0, substep_s: 1.0 / 30.0, ..Params::default() };
    let mut w = Walker::new(p, [0.0, 20000.0, 0.0]);
    walk_for(&mut w, &g, Intent::default(), 5.0);
    assert!(w.grounded);
    assert_eq!(w.feet[1], 101.0);
}

#[test]
fn ceilings_stop_a_jump() {
    let mut g = Grid::floor();
    g.fill([-64, 34, -64], [64, 35, 64]);
    let mut w = Walker::new(Params::default(), [0.0, 0.0, 0.0]);
    w.grounded = true;
    let mut top: f64 = 0.0;
    let n = (2.0 / w.params.substep_s) as u64;
    for i in 0..n {
        w.substep(&g, Intent { jump: i == 0, ..Intent::default() });
        assert!(!w.overlaps(&g));
        top = top.max(w.feet[1]);
    }
    assert_eq!(w.stats.jumps, 1);
    assert!((top + w.params.height - 34.0).abs() < 1e-9, "head flush against the ceiling: {}", top + w.params.height);
    assert!(w.grounded);
    assert_eq!(w.feet[1], 0.0);
    // Without the ceiling the same jump reaches about 0.45 m.
    let mut w = Walker::new(Params::default(), [0.0, 0.0, 0.0]);
    w.grounded = true;
    let mut top: f64 = 0.0;
    for i in 0..n {
        w.substep(&Grid::floor(), Intent { jump: i == 0, ..Intent::default() });
        top = top.max(w.feet[1]);
    }
    assert!((top - m(0.45)).abs() < 0.1, "apex {top} voxels");
}

#[test]
fn frame_rate_does_not_change_the_path() {
    let (world, _) = scene::street_block();
    let run = |fps: f64| {
        let mut w = Walker::new(Params::default(), [m(2.0), 0.0, m(2.5)]);
        let frames = (4.0 * fps).round() as u64;
        for _ in 0..frames {
            w.update(&world, toward(0.3, 1.0), 1.0 / fps);
        }
        w
    };
    let (a, b) = (run(30.0), run(144.0));
    let step = Params::default().walk_speed * Params::default().substep_s;
    for i in 0..3 {
        assert!((a.feet[i] - b.feet[i]).abs() <= step * 1.01, "axis {i}: {:?} vs {:?}", a.feet, b.feet);
    }
}

#[test]
fn resolve_start_lifts_out_of_solid() {
    let mut g = Grid::floor();
    g.fill([-4, 0, -4], [4, 10, 4]);
    let mut w = Walker::new(Params::default(), [0.0, 2.5, 0.0]);
    assert!(w.overlaps(&g));
    assert!(w.resolve_start(&g, 64));
    assert_eq!(w.feet[1], 10.0);
    let mut w = Walker::new(Params::default(), [0.0, 2.5, 0.0]);
    assert!(!w.resolve_start(&g, 3), "cannot rise 7.5 voxels in 3");
}

/// The street block: from the reference view's feet, across the road and up the curb onto the
/// sidewalk, until the shop-window sill stops the walker.
#[test]
fn street_block_curb_and_shop_front() {
    let (world, view) = scene::street_block();
    let eye = view.eye_m.map(m);
    let p = Params::default();
    let feet = [eye[0], eye[1] - p.eye_height, eye[2]];
    assert!(feet[1].abs() < 1e-9, "the reference eye stands on the road: feet {feet:?}");
    let mut w = Walker::new(p, [feet[0], 0.0, feet[2]]);
    walk_for(&mut w, &world, Intent::default(), 0.5);
    assert!(w.grounded && w.feet[1] == 0.0);
    walk_for(&mut w, &world, toward(0.0, 1.0), 10.0);
    // Curb and sidewalk tops are 3 voxels up; the window sill (trim at y 8..10, z 174..176) stops us.
    assert_eq!(w.feet[1], 3.0);
    assert_eq!(w.feet[2], 174.0 - p.half_width);
    assert_eq!(w.stats.steps_up, 1);
    // With a 2-voxel step the curb (z from 128) blocks.
    let mut w = Walker::new(Params { step_height: 2.0, ..p }, [feet[0], 0.0, feet[2]]);
    walk_for(&mut w, &world, toward(0.0, 1.0), 10.0);
    assert_eq!((w.feet[1], w.feet[2]), (0.0, 128.0 - p.half_width));
}

/// A tiny deterministic generator (no dependencies).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn random_walks_never_overlap_the_street_block() {
    let (world, _) = scene::street_block();
    let p = Params::default();
    let mut total = walk::Stats::default();
    let mut respawns = 0;
    let mut busy = std::time::Duration::ZERO;
    for seed in 0..8u64 {
        let mut rng = Lcg(seed * 7919 + 1);
        let spawn = [m(1.0) + rng.next() * m(22.0), 0.0, m(-2.5) + rng.next() * m(13.0)];
        let mut w = Walker::new(p, spawn);
        assert!(w.resolve_start(&world, 256));
        let mut intent = Intent::default();
        for i in 0..(20.0 / p.substep_s) as u64 {
            if i % 120 == 0 {
                let a = rng.next() * std::f64::consts::TAU;
                intent = Intent { dir: [a.cos(), a.sin()], run: rng.next() < 0.4, jump: rng.next() < 0.3 };
            }
            let t = std::time::Instant::now();
            w.substep(&world, intent);
            busy += t.elapsed();
            assert!(!w.overlaps(&world), "seed {seed} substep {i}: overlap at {:?}", w.feet);
            if w.feet[1] < -256.0 {
                w = Walker::new(p, spawn);
                assert!(w.resolve_start(&world, 256));
                respawns += 1;
            }
        }
        let s = w.stats;
        total.substeps += s.substeps;
        total.steps_up += s.steps_up;
        total.snaps_down += s.snaps_down;
        total.jumps += s.jumps;
        total.layers += s.layers;
        total.voxels += s.voxels;
    }
    let secs = busy.as_secs_f64();
    println!(
        "random walks: {} substeps, {} steps up, {} snaps down, {} jumps, {respawns} respawns; {:.1} layers and {:.0} voxels tested per substep; {:.2} us per substep",
        total.substeps,
        total.steps_up,
        total.snaps_down,
        total.jumps,
        total.layers as f64 / total.substeps as f64,
        total.voxels as f64 / total.substeps as f64,
        secs * 1e6 / total.substeps as f64
    );
    // The walks must have exercised what they claim to check.
    assert!(total.steps_up > 0 && total.snaps_down > 0 && total.jumps > 0, "{total:?}");
}

/// Negative control: the overlap check the random walk relies on does detect penetration.
#[test]
fn overlap_check_detects_penetration() {
    let (world, _) = scene::street_block();
    let mut w = Walker::new(Params::default(), [m(2.0), 0.0, m(2.5)]);
    assert!(!w.overlaps(&world));
    w.feet[1] -= 0.5; // half a voxel into the road
    assert!(w.overlaps(&world));
    let mut w = Walker::new(Params::default(), [m(2.0), 3.0, 175.0]); // inside the shop-window sill
    assert!(w.overlaps(&world));
    w.feet[2] = 174.0 - w.params.half_width; // flush against it
    assert!(!w.overlaps(&world));
}
