//! DDA (`trace`) against brute force (`trace_brute`) on seeded random worlds and rays.
//! Agreement must be exact: same `t` bits, voxel, material and face.

use world::reference::{trace, trace_brute, Ray};
use world::{MaterialParams, MaterialRegistry, VoxelCoord, World};

/// xorshift64*: deterministic and dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_world(rng: &mut Rng, extent: i32, density: f64) -> World {
    let mut r = MaterialRegistry::new();
    let mats: Vec<_> = (0..5).map(|i| r.register(&format!("m{i}"), MaterialParams::diffuse(0.5, 0.5, 0.5)).unwrap()).collect();
    let mut w = World::new(r);
    for z in -extent..extent {
        for y in -extent..extent {
            for x in -extent..extent {
                if rng.unit() < density {
                    let m = mats[(rng.next() % mats.len() as u64) as usize];
                    w.set(VoxelCoord::new(x, y, z), Some(m)).unwrap();
                }
            }
        }
    }
    w
}

/// Mix of generic rays and the degenerate cases the convention is about: axis-aligned and diagonal
/// directions, integer (on-plane) origins, and zero components.
fn random_ray(rng: &mut Rng, reach: f64) -> Ray {
    let kind = rng.next() % 6;
    let mut origin = [0.0; 3];
    for o in &mut origin {
        *o = (rng.unit() * 2.0 - 1.0) * reach;
    }
    if kind == 1 || kind == 3 {
        for o in &mut origin {
            *o = o.round();
        }
    }
    let mut dir = [0.0; 3];
    match kind {
        2 => dir[(rng.next() % 3) as usize] = if rng.next() % 2 == 0 { 1.0 } else { -1.0 },
        3 => {
            for d in &mut dir {
                *d = [-1.0, 0.0, 1.0][(rng.next() % 3) as usize];
            }
            if dir == [0.0; 3] {
                dir[0] = 1.0;
            }
        }
        4 => {
            for d in &mut dir {
                *d = rng.unit() * 2.0 - 1.0;
            }
            dir[(rng.next() % 3) as usize] = 0.0;
            if dir == [0.0; 3] {
                dir[1] = -1.0;
            }
        }
        _ => {
            for d in &mut dir {
                *d = rng.unit() * 2.0 - 1.0;
            }
            if dir == [0.0; 3] {
                dir[2] = 1.0;
            }
        }
    }
    Ray { origin, dir }
}

#[test]
fn dda_matches_brute_force_exactly() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut hits = 0;
    let mut rays = 0;
    let mut t_limited = 0;
    for (extent, density) in [(6, 0.15), (12, 0.04), (20, 0.015), (40, 0.003)] {
        // A 40-voxel extent spans chunk boundaries in every direction around the origin.
        for _world in 0..3 {
            let w = random_world(&mut rng, extent, density);
            for _ in 0..3000 {
                let ray = random_ray(&mut rng, extent as f64 * 1.5);
                let t_max = if rng.next() % 4 == 0 {
                    t_limited += 1;
                    rng.unit() * extent as f64 * 3.0
                } else {
                    f64::INFINITY
                };
                let a = trace(&w, &ray, t_max);
                let b = trace_brute(&w, &ray, t_max);
                assert_eq!(a, b, "disagreement for {ray:?} t_max {t_max}");
                if let (Some(a), Some(b)) = (a, b) {
                    // `==` treats 0.0 and -0.0 as equal; beyond that sign, the bits must match.
                    assert!(a.t.to_bits() == b.t.to_bits() || a.t == 0.0, "t bits differ: {} vs {}", a.t, b.t);
                    hits += 1;
                }
                rays += 1;
            }
        }
    }
    // The comparison must exercise hits, misses and limited rays, or it proves little.
    println!("rays {rays}, hits {hits}, t-limited {t_limited}");
    assert!(hits >= 2000 && rays - hits >= 2000, "hits {hits} of {rays}");
    assert!(t_limited > 1000);
}

/// Negative control: the comparison must detect a difference when the two worlds differ by one voxel.
#[test]
fn comparison_detects_a_single_missing_voxel() {
    let mut rng = Rng(12345);
    let full = random_world(&mut rng, 8, 0.03);
    let (victim, _) = full.occupied().nth(10).expect("enough voxels");
    let mut holed = full.clone();
    holed.set(victim, None).unwrap();

    let mut mismatches = 0;
    let mut tried = 0;
    // Rays aimed at the removed voxel's centre from random directions.
    for _ in 0..500 {
        let dir = [rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0];
        let target = [victim.x as f64 + 0.5, victim.y as f64 + 0.5, victim.z as f64 + 0.5];
        let origin = [target[0] - dir[0] * 30.0, target[1] - dir[1] * 30.0, target[2] - dir[2] * 30.0];
        let ray = Ray { origin, dir };
        tried += 1;
        if trace(&full, &ray, f64::INFINITY) != trace_brute(&holed, &ray, f64::INFINITY) {
            mismatches += 1;
        }
    }
    assert!(mismatches > 0, "no mismatch found in {tried} rays: the comparison cannot fail");
}
