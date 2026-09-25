//! Phase 2A: surface extraction checked against two independent oracles.
//!
//! - **Coverage:** every exposed unit face of the world (computed directly from voxels) is covered
//!   by exactly one quad, with the right material, and nothing else is covered.
//! - **Ray agreement:** the CPU ray–quad reference agrees exactly with the 1E voxel DDA (same `t`
//!   bit for bit, voxel, material and face) for rays in general position, including rays that cross
//!   chunk boundaries.
//!
//! Each checker has a negative control: a planted mesh defect it must report.

mod common;

use std::collections::BTreeMap;

use common::{run_to_idle, world_with, Rng};
use derived::{exposed_faces, extract_world, summarize_world, trace_mesh, BrickMesh, Config, Merge, Pipeline, Quad, Split};
use world::dims::VOXEL_SIZE_M;
use world::reference::{trace, Face, Ray};
use world::{scene, BrickKey, MaterialId, VoxelCoord, World};

type Meshes = BTreeMap<BrickKey, BrickMesh>;

fn meshes(w: &World, merge: Merge) -> Meshes {
    w.bricks().map(|(k, _)| (k, extract_world(w, k, merge).expect("stored brick"))).collect()
}

/// Coverage check. Returns every problem found (empty = pass).
fn coverage_errors(w: &World, m: &Meshes) -> Vec<String> {
    let mut errors = Vec::new();
    let mut covered: BTreeMap<(VoxelCoord, u8, bool), (MaterialId, u32)> = BTreeMap::new();
    for (&k, mesh) in m {
        for q in &mesh.quads {
            if !(q.u0 < q.u1 && q.v0 < q.v1 && q.u1 <= 8 && q.v1 <= 8 && q.plane <= 8 && q.face < 6) {
                errors.push(format!("{k:?}: malformed quad {q:?}"));
                continue;
            }
            for (v, f) in q.unit_faces(k) {
                if v.split().0 != k {
                    errors.push(format!("{k:?}: quad {q:?} covers {v:?} outside its brick"));
                }
                covered.entry((v, f.axis, f.positive)).or_insert((q.material, 0)).1 += 1;
                let e = covered.get_mut(&(v, f.axis, f.positive)).unwrap();
                if e.0 != q.material {
                    errors.push(format!("{v:?} {f:?}: covered with {:?} and {:?}", e.0, q.material));
                }
            }
        }
    }
    let want: BTreeMap<(VoxelCoord, u8, bool), MaterialId> = exposed_faces(w).into_iter().map(|(v, f, m)| ((v, f.axis, f.positive), m)).collect();
    for (key, &(mat, n)) in &covered {
        match want.get(key) {
            None => errors.push(format!("{key:?}: covered but not an exposed face")),
            Some(&wm) if wm != mat => errors.push(format!("{key:?}: material {mat:?}, voxel has {wm:?}")),
            _ => {}
        }
        if n != 1 {
            errors.push(format!("{key:?}: covered {n} times"));
        }
    }
    for key in want.keys() {
        if !covered.contains_key(key) {
            errors.push(format!("{key:?}: exposed face not covered"));
        }
    }
    errors
}

/// A random world spanning chunk boundaries (voxels in [-24, 24)³, chunk edges at 0), with a few
/// materials in blobs so greedy merging has both mergeable areas and material boundaries.
fn random_world(seed: u64) -> World {
    let (mut w, mats) = world_with(3);
    let mut rng = Rng(seed);
    for _ in 0..160 {
        let c = |r: &mut Rng| r.below(48) as i32 - 24;
        let (x, y, z) = (c(&mut rng), c(&mut rng), c(&mut rng));
        let s = |r: &mut Rng| 1 + r.below(9) as i32;
        let (sx, sy, sz) = (s(&mut rng), s(&mut rng), s(&mut rng));
        let m = if rng.below(5) == 0 { None } else { Some(mats[rng.below(3) as usize]) };
        w.fill_box(VoxelCoord::new(x, y, z), VoxelCoord::new((x + sx).min(24), (y + sy).min(24), (z + sz).min(24)), m).unwrap();
    }
    w
}

fn street() -> (World, [f64; 3]) {
    let (w, view) = scene::street_block();
    (w, view.eye_m.map(|x| x / VOXEL_SIZE_M))
}

fn random_dir(rng: &mut Rng) -> [f64; 3] {
    loop {
        let d = [rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 - 1.0];
        let n = d.iter().map(|x| x * x).sum::<f64>();
        if n > 1e-6 && n <= 1.0 {
            return d;
        }
    }
}

/// Rays with origins in empty space: half from `eye` (if given), half from random points inside
/// the world's bounding box.
fn rays(w: &World, eye: Option<[f64; 3]>, n: usize, seed: u64) -> Vec<Ray> {
    let (lo, hi) = w.bounds().unwrap();
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let origin = match eye {
            Some(e) if out.len() % 2 == 0 => e,
            _ => {
                let (l, h) = ([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
                [0, 1, 2].map(|a| l[a] as f64 + rng.unit() * (h[a] - l[a]) as f64)
            }
        };
        let cell = VoxelCoord::new(origin[0].floor() as i32, origin[1].floor() as i32, origin[2].floor() as i32);
        if w.get(cell).is_none() {
            out.push(Ray { origin, dir: random_dir(&mut rng) });
        }
    }
    out
}

/// Ray agreement. Returns (rays that hit, disagreements).
fn ray_disagreements(w: &World, m: &Meshes, rays: &[Ray]) -> (usize, Vec<String>) {
    let mut hits = 0;
    let mut errors = Vec::new();
    for r in rays {
        let dda = trace(w, r, 1e6);
        let mesh = trace_mesh(m.iter().map(|(k, v)| (*k, v)), r, 1e6);
        hits += dda.is_some() as usize;
        let same = match (dda, mesh) {
            (None, None) => true,
            (Some(a), Some(b)) => a.t == b.t && a.voxel == b.voxel && a.material == b.material && a.face == Some(b.face),
            _ => false,
        };
        if !same {
            errors.push(format!("{r:?}: DDA {dda:?}, mesh {mesh:?}"));
        }
    }
    (hits, errors)
}

#[test]
fn coverage_is_exact_on_the_street_and_random_worlds() {
    let (street, _) = street();
    let mut worlds = vec![("street".to_string(), street)];
    for seed in [1u64, 2, 3, 4] {
        worlds.push((format!("random seed {seed}"), random_world(seed)));
    }
    for (name, w) in &worlds {
        for merge in Merge::ALL {
            let m = meshes(w, merge);
            let errors = coverage_errors(w, &m);
            assert!(errors.is_empty(), "{name}, merge {}: {} problems, first: {:?}", merge.name(), errors.len(), &errors[..errors.len().min(5)]);
            for (&k, mesh) in &m {
                assert_eq!(mesh.summary(), summarize_world(w, k).unwrap(), "{name} {k:?}: face counts per material");
            }
        }
        let (none, greedy) = (meshes(w, Merge::None), meshes(w, Merge::Greedy));
        let count = |m: &Meshes| m.values().map(|x| x.quads.len()).sum::<usize>();
        assert!(count(&greedy) < count(&none), "{name}: greedy must merge something");
    }
}

#[test]
fn negative_controls_coverage_checker_reports_planted_defects() {
    let w = random_world(7);
    let good = meshes(&w, Merge::Greedy);
    assert!(coverage_errors(&w, &good).is_empty());
    let (&k, _) = good.iter().find(|(_, m)| m.quads.len() >= 2).unwrap();

    let plant = |f: &dyn Fn(&mut Vec<Quad>)| {
        let mut bad = good.clone();
        f(&mut bad.get_mut(&k).unwrap().quads);
        coverage_errors(&w, &bad)
    };
    assert!(!plant(&|q| { q.remove(0); }).is_empty(), "dropped face");
    assert!(!plant(&|q| { let d = q[0]; q.push(d); }).is_empty(), "duplicated face");
    assert!(!plant(&|q| { q[0].plane ^= 1; }).is_empty(), "shifted plane");

    // Cross-material merge: two touching voxels of different materials; extend A's top quad over
    // B's and drop B's, as a mesher that ignored materials would.
    let (mut w2, mats) = world_with(2);
    w2.set(VoxelCoord::new(2, 2, 2), Some(mats[0])).unwrap();
    w2.set(VoxelCoord::new(2, 2, 3), Some(mats[1])).unwrap();
    let mut bad = meshes(&w2, Merge::Greedy);
    assert!(coverage_errors(&w2, &bad).is_empty());
    let quads = &mut bad.values_mut().next().unwrap().quads;
    // +y faces: axis 1, so u = z and v = x. A's top covers z in [2, 3), B's z in [3, 4).
    let a = quads.iter().position(|q| q.face == 3 && q.material == mats[0]).unwrap();
    quads[a].u1 = 4;
    quads.retain(|q| !(q.face == 3 && q.material == mats[1]));
    let errors = coverage_errors(&w2, &bad);
    assert!(errors.iter().any(|e| e.contains("material")), "cross-material merge: {errors:?}");
}

#[test]
fn rays_agree_with_the_voxel_dda_on_the_street() {
    let (w, eye) = street();
    let n = if cfg!(debug_assertions) { 300 } else { 3000 };
    let rs = rays(&w, Some(eye), n, 0xA11CE);
    for merge in Merge::ALL {
        let m = meshes(&w, merge);
        let (hits, errors) = ray_disagreements(&w, &m, &rs);
        assert!(hits > n / 3, "merge {}: only {hits} of {n} rays hit; the test must exercise hits", merge.name());
        assert!(errors.is_empty(), "merge {}: {} of {n} disagree, first: {:?}", merge.name(), errors.len(), &errors[..errors.len().min(3)]);
        eprintln!("street, merge {}: {n} rays, {hits} hits, 0 disagreements", merge.name());
    }
}

#[test]
fn rays_agree_with_the_voxel_dda_across_chunk_boundaries() {
    let n = if cfg!(debug_assertions) { 300 } else { 2000 };
    for seed in [11u64, 12, 13] {
        let w = random_world(seed);
        let rs = rays(&w, None, n, seed);
        for merge in Merge::ALL {
            let m = meshes(&w, merge);
            let (hits, errors) = ray_disagreements(&w, &m, &rs);
            // The hit floor only proves the comparison is exercised (sparse random worlds let many rays escape).
            assert!(hits > n / 5, "seed {seed}: only {hits} hits ({} disagreements)", errors.len());
            assert!(errors.is_empty(), "seed {seed} merge {}: {} disagree, first: {:?}", merge.name(), errors.len(), &errors[..errors.len().min(3)]);
            eprintln!("random seed {seed}, merge {}: {n} rays, {hits} hits, 0 disagreements", merge.name());
        }
    }
    // Rays along the chunk planes x = 0 and z = 0 themselves (half-open: they belong to the cells above).
    let w = random_world(11);
    let m = meshes(&w, Merge::Greedy);
    let mut rng = Rng(99);
    let mut along = Vec::new();
    while along.len() < 200 {
        let d = random_dir(&mut rng);
        let o = [0.0, rng.unit() * 48.0 - 24.0, -40.0];
        along.push(Ray { origin: o, dir: [0.0, d[1], d[2].abs() + 0.1] });
    }
    let (hits, errors) = ray_disagreements(&w, &m, &along);
    assert!(hits > 0 && errors.is_empty(), "rays in the plane x = 0: {hits} hits, {} disagree: {:?}", errors.len(), errors.first());
    eprintln!("rays in the chunk plane x = 0: {} rays, {hits} hits, 0 disagreements", along.len());
}

#[test]
fn negative_control_ray_checker_reports_a_shifted_face() {
    let (w, eye) = street();
    let good = meshes(&w, Merge::Greedy);
    let rs = rays(&w, Some(eye), 400, 5);
    // Find the brick that the first eye ray hits and shift the plane of every quad facing that way.
    let hit = rs.iter().find_map(|r| trace(&w, r, 1e6)).expect("some ray hits");
    let k = hit.voxel.split().0;
    let face: Face = hit.face.unwrap();
    let mut bad = good.clone();
    for q in &mut bad.get_mut(&k).unwrap().quads {
        if q.to_face() == face {
            q.plane = if face.positive { q.plane - 1 } else { q.plane + 1 };
        }
    }
    let (_, errors) = ray_disagreements(&w, &bad, &rs);
    assert!(!errors.is_empty(), "a shifted face must change some hit");
}

#[test]
fn pipeline_publishes_meshes_equal_to_direct_extraction_in_both_modes() {
    let w = random_world(21);
    for merge in Merge::ALL {
        let mut p = Pipeline::new(Config { merge, ..Config::default() });
        p.mark_all(&w);
        run_to_idle(&mut p, &w);
        let snap = p.current();
        let want = meshes(&w, merge);
        assert_eq!(snap.len(), want.len());
        for (k, m) in &want {
            assert_eq!(p.resolve(snap.handle(*k).unwrap()), Some(m), "{k:?} merge {}", merge.name());
        }
    }
}

/// World triangles in half-voxel units (×2), with the outward normal of each triangle's quad.
fn world_triangles(m: &Meshes, split: Split) -> Vec<([[i64; 3]; 3], Quad)> {
    let mut out = Vec::new();
    for (&k, mesh) in m {
        let o = k.origin();
        let o2 = [2 * o.x as i64, 2 * o.y as i64, 2 * o.z as i64];
        let t = mesh.triangulate(split);
        assert_eq!(t.indices.len(), t.quad.len());
        for (tri, &qi) in t.indices.iter().zip(&t.quad) {
            let p = tri.map(|i| {
                let v = t.vertices[i as usize];
                [o2[0] + v[0] as i64, o2[1] + v[1] as i64, o2[2] + v[2] as i64]
            });
            out.push((p, mesh.quads[qi as usize]));
        }
    }
    out
}

fn cross(a: [i64; 3], b: [i64; 3], c: [i64; 3]) -> [i64; 3] {
    let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
    [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
}

/// Watertightness of a closed oriented surface: every directed edge a→b is matched by as many
/// edges b→a. A T-junction leaves a long edge whose reverse exists only in pieces. Also checks
/// that no triangle is degenerate, each faces its quad's outward normal, and each quad's
/// triangles have exactly its area. Returns the problems found (empty = pass).
fn watertight_errors(tris: &[([[i64; 3]; 3], Quad)]) -> Vec<String> {
    let mut errors = Vec::new();
    let mut edges: BTreeMap<([i64; 3], [i64; 3]), i64> = BTreeMap::new();
    for (p, q) in tris {
        let n = cross(p[0], p[1], p[2]);
        let along = n[q.axis()] * if q.positive() { 1 } else { -1 };
        let off_axis = (0..3).any(|a| a != q.axis() && n[a] != 0);
        if along <= 0 || off_axis {
            errors.push(format!("triangle {p:?} of {q:?}: normal {n:?}"));
        }
        for i in 0..3 {
            *edges.entry((p[i], p[(i + 1) % 3])).or_default() += 1;
        }
    }
    for (&(a, b), &n) in &edges {
        if edges.get(&(b, a)).copied().unwrap_or(0) != n {
            errors.push(format!("edge {a:?}→{b:?} ×{n} has no matching reverse"));
        }
    }
    errors
}

/// Twice the triangle area in half-voxel units² is 8 per unit face.
fn area_errors(m: &Meshes, split: Split) -> Vec<String> {
    let mut errors = Vec::new();
    for (&k, mesh) in m {
        let t = mesh.triangulate(split);
        let mut per_quad = vec![0i64; mesh.quads.len()];
        for (tri, &qi) in t.indices.iter().zip(&t.quad) {
            let p = tri.map(|i| t.vertices[i as usize].map(|c| c as i64));
            let n = cross(p[0], p[1], p[2]);
            per_quad[qi as usize] += n.iter().map(|c| c.abs()).sum::<i64>();
        }
        for (q, got) in mesh.quads.iter().zip(per_quad) {
            if got != 8 * q.area() as i64 {
                errors.push(format!("{k:?} {q:?}: triangle area {got}, want {}", 8 * q.area()));
            }
        }
    }
    errors
}

#[test]
fn watertight_triangulation_shares_every_edge() {
    let (street, _) = street();
    let mut worlds = vec![("street".to_string(), street)];
    for seed in [1u64, 2, 3, 4] {
        worlds.push((format!("random seed {seed}"), random_world(seed)));
    }
    for (name, w) in &worlds {
        for merge in Merge::ALL {
            let m = meshes(w, merge);
            let tris = world_triangles(&m, Split::Watertight);
            let errors = watertight_errors(&tris);
            assert!(errors.is_empty(), "{name}, merge {}: {} problems, first: {:?}", merge.name(), errors.len(), &errors[..errors.len().min(5)]);
            let errors = area_errors(&m, Split::Watertight);
            assert!(errors.is_empty(), "{name}, merge {}: {:?}", merge.name(), &errors[..errors.len().min(5)]);
            let quads: usize = m.values().map(|x| x.quads.len()).sum();
            eprintln!("{name}, merge {}: {quads} quads, {} watertight triangles, {} vertices", merge.name(), tris.len(), m.values().map(|x| x.triangulate(Split::Watertight).vertices.len()).sum::<usize>());
        }
        // Unmerged quads have unit edges, so splitting changes nothing.
        let none = meshes(w, Merge::None);
        for mesh in none.values() {
            assert_eq!(mesh.triangulate(Split::Watertight), mesh.triangulate(Split::CornersOnly), "{name}");
        }
    }
}

#[test]
fn negative_control_unsplit_greedy_has_t_junctions() {
    // The checker must see the pre-amendment layout's T-junctions, and pass unmerged quads.
    let (w, _) = street();
    let greedy = meshes(&w, Merge::Greedy);
    let errors = watertight_errors(&world_triangles(&greedy, Split::CornersOnly));
    eprintln!("street greedy, corners only: {} unmatched directed edges, e.g. {:?}", errors.len(), errors.first());
    assert!(!errors.is_empty(), "unsplit greedy quads must show T-junctions");
    assert!(area_errors(&greedy, Split::CornersOnly).is_empty(), "the area check is independent of splitting");
    assert!(watertight_errors(&world_triangles(&meshes(&w, Merge::None), Split::CornersOnly)).is_empty());
    // A planted flipped triangle and a dropped triangle are reported.
    let mut tris = world_triangles(&greedy, Split::Watertight);
    tris[0].0.swap(1, 2);
    assert!(!watertight_errors(&tris).is_empty(), "flipped triangle");
    let mut tris = world_triangles(&greedy, Split::Watertight);
    tris.remove(0);
    assert!(!watertight_errors(&tris).is_empty(), "dropped triangle");
}
