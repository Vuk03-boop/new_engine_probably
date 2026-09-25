//! Headless CPU reference render of the procedural street block (Phase 1E).
//!
//! usage: ref_render <out.ppm> [--size W H] [--threads N]
//!
//! Shading is a debug view, not lighting: base colour × (ambient + sun·N·L × shadow), plus the
//! material's own emissive colour; sky gradient on miss. One primary and at most one shadow ray per
//! pixel, both through `reference::trace`. The output does not depend on the thread count.
//! Prints one JSON line with scene statistics and timings.

use std::io::Write;
use std::time::Instant;

use world::dims::VOXEL_SIZE_M;
use world::reference::{trace, Ray};
use world::scene::street_block;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ref_render <out.ppm> [--size W H] [--threads N]");
        std::process::exit(2);
    }
    let out_path = &args[1];
    let (mut width, mut height, mut threads) = (640usize, 360usize, 2usize);
    let mut i = 2;
    while i < args.len() {
        let num = |k: usize| -> usize {
            args.get(k).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!("bad number after {}", args[i]);
                std::process::exit(2)
            })
        };
        match args[i].as_str() {
            "--size" => {
                width = num(i + 1);
                height = num(i + 2);
                i += 3;
            }
            "--threads" => {
                threads = num(i + 1).max(1);
                i += 2;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let t_build = Instant::now();
    let (world, view) = street_block();
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let stats = world.stats();

    // Camera in voxel units.
    let to_vox = |p: [f64; 3]| [p[0] / VOXEL_SIZE_M, p[1] / VOXEL_SIZE_M, p[2] / VOXEL_SIZE_M];
    let eye = to_vox(view.eye_m);
    let fwd = norm(sub(to_vox(view.target_m), eye));
    let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    let tan_half = (view.vertical_fov_deg.to_radians() * 0.5).tan();
    let aspect = width as f64 / height as f64;
    let sun = view.sun_dir;

    let shade_row = |y: usize, row: &mut [u8], counts: &mut [u64; 4]| {
        for x in 0..width {
            let u = (2.0 * (x as f64 + 0.5) / width as f64 - 1.0) * tan_half * aspect;
            let v = (1.0 - 2.0 * (y as f64 + 0.5) / height as f64) * tan_half;
            let dir = norm([fwd[0] + u * right[0] + v * up[0], fwd[1] + u * right[1] + v * up[1], fwd[2] + u * right[2] + v * up[2]]);
            counts[0] += 1;
            let rgb = match trace(&world, &Ray { origin: eye, dir }, f64::INFINITY) {
                None => {
                    let s = dir[1].max(0.0);
                    [0.55 - 0.25 * s, 0.7 - 0.2 * s, 0.9]
                }
                Some(hit) => {
                    counts[1] += 1;
                    let p = world.materials().get(hit.material).expect("hit material is registered").params;
                    let n = hit.face.map(|f| f.normal()).unwrap_or([0.0, 0.0, 0.0]);
                    let ndl = dot(n, sun).max(0.0);
                    let mut lit = 0.0;
                    if ndl > 0.0 {
                        counts[2] += 1;
                        let hp = [eye[0] + hit.t * dir[0] + 1e-4 * n[0], eye[1] + hit.t * dir[1] + 1e-4 * n[1], eye[2] + hit.t * dir[2] + 1e-4 * n[2]];
                        if trace(&world, &Ray { origin: hp, dir: sun }, f64::INFINITY).is_none() {
                            lit = ndl;
                        } else {
                            counts[3] += 1;
                        }
                    }
                    // Ambient varies by face so unlit faces stay distinguishable.
                    let amb = 0.12 + 0.06 * n[1] + 0.02 * n[0];
                    let k = amb + 0.9 * lit;
                    [0, 1, 2].map(|c| p.base_color[c] as f64 * k + p.emissive[c] as f64)
                }
            };
            for c in 0..3 {
                row[x * 3 + c] = (rgb[c].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
            }
        }
    };

    let t_render = Instant::now();
    let mut img = vec![0u8; width * height * 3];
    let mut totals = [0u64; 4];
    {
        // Interleave rows over threads; each row is computed independently, so the image is thread-count independent.
        let mut rows: Vec<(usize, &mut [u8])> = img.chunks_mut(width * 3).enumerate().collect();
        let mut buckets: Vec<Vec<(usize, &mut [u8])>> = (0..threads).map(|_| Vec::new()).collect();
        for (k, r) in rows.drain(..).enumerate() {
            buckets[k % threads].push(r);
        }
        let shade_row = &shade_row;
        let results: Vec<[u64; 4]> = std::thread::scope(|s| {
            let handles: Vec<_> = buckets
                .into_iter()
                .map(|bucket| {
                    s.spawn(move || {
                        let mut counts = [0u64; 4];
                        for (y, row) in bucket {
                            shade_row(y, row, &mut counts);
                        }
                        counts
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("render thread panicked")).collect()
        });
        for r in results {
            for c in 0..4 {
                totals[c] += r[c];
            }
        }
    }
    let render_ms = t_render.elapsed().as_secs_f64() * 1e3;

    let mut out = Vec::with_capacity(img.len() + 32);
    write!(out, "P6\n{width} {height}\n255\n").unwrap();
    out.extend_from_slice(&img);
    std::fs::write(out_path, out).unwrap_or_else(|e| {
        eprintln!("cannot write {out_path}: {e}");
        std::process::exit(1)
    });

    let (lo, hi) = world.bounds().expect("scene is not empty");
    println!(
        "{{\"scene\":\"street_block\",\"voxel_m\":{},\"materials\":{},\"chunks\":{},\"bricks\":{},\"occupied_voxels\":{},\"payload_mib\":{:.2},\
\"bounds_min\":[{},{},{}],\"bounds_max\":[{},{},{}],\"world_version\":{},\"width\":{},\"height\":{},\"threads\":{},\
\"primary_rays\":{},\"primary_hits\":{},\"shadow_rays\":{},\"shadow_blocked\":{},\"build_ms\":{:.1},\"render_ms\":{:.1}}}",
        VOXEL_SIZE_M,
        world.materials().len(),
        stats.chunks,
        stats.bricks,
        stats.occupied_voxels,
        stats.payload_bytes as f64 / (1024.0 * 1024.0),
        lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
        world.version().raw(),
        width,
        height,
        threads,
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        build_ms,
        render_ms
    );
}
