//! Phase 3A: renders the ADR-0005 reference images of the street (GPU reference path tracer).
//!
//! `ref_light <out dir> [--size W H] [--spp N] [--bounces N] [--times name,name]`
//!
//! For each reference camera and time of day it writes `<camera>_<time>.pfm` (linear HDR, the data)
//! and `<camera>_<time>.ppm` (for looking: exposure from the image's log-average luminance, key
//! 0.18, then the ACES fit and sRGB encoding; display only). One JSON line per image goes to stdout:
//! time, sun elevation, sun irradiance at the street, mean radiance, the mean relative standard error
//! of the pixels, and the render time.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use derived::{extract_world, Merge};
use gpu::accel::{Accel, AccelFaults};
use gpu::alloc::Allocator;
use gpu::layout::{build_regions, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::Camera;
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::{Gpu, Timeline};
use light::reference::{albedos, Accum, Lighting, Settings};
use light::sun::{elevation_deg, REFERENCE_TIMES};
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::scene;

const SEED: u32 = 0x3A;

fn cameras(w: u32, h: u32) -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        ("street", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, w, h, 0.1)),
        ("low", Camera::look_at([4.3, 30.0, 20.2], [380.0, 60.0, 30.0], 70.0, w, h, 0.1)),
    ]
}

fn write_pfm(path: &Path, a: &Accum) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "PF\n{} {}\n-1.0\n", a.width, a.height)?;
    // PFM rows run bottom to top.
    for y in (0..a.height).rev() {
        for x in 0..a.width {
            for c in a.mean((y * a.width + x) as usize) {
                f.write_all(&(c as f32).to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn luminance(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn write_ppm(path: &Path, a: &Accum) -> std::io::Result<f64> {
    let n = a.sum.len();
    let log_avg = ((0..n).map(|i| (luminance(a.mean(i)) + 1e-6).ln()).sum::<f64>() / n as f64).exp();
    let exposure = 0.18 / log_avg;
    let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
    let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{} {}\n255\n", a.width, a.height)?;
    for i in 0..n {
        let m = a.mean(i);
        f.write_all(&m.map(|c| (srgb(aces(c * exposure)) * 255.0 + 0.5) as u8))?;
    }
    Ok(exposure)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).expect("usage: ref_light <out dir> [--size W H] [--spp N] [--bounces N] [--times a,b]"));
    let (mut w, mut h, mut spp, mut bounces) = (960u32, 540u32, 2048u32, 8u32);
    let mut times: Vec<(&str, f64)> = REFERENCE_TIMES.to_vec();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                w = args[i + 1].parse().unwrap();
                h = args[i + 2].parse().unwrap();
                i += 3;
            }
            "--spp" => {
                spp = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--bounces" => {
                bounces = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--times" => {
                let want: Vec<&str> = args[i + 1].split(',').collect();
                times.retain(|(n, _)| want.contains(n));
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    let g = Gpu::new().expect("an RT-capable Vulkan device");
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let (world, _) = scene::street_block();
    let albedo = albedos(world.materials()).expect("material albedos in [0, 1]");
    let meshes: Vec<_> = world.bricks().map(|(k, _)| (k, extract_world(&world, k, Merge::Greedy).unwrap())).collect();
    let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk);
    let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs).unwrap();
    let accel = Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()).unwrap();
    let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedo).unwrap();
    tl.wait(&g, v, u64::MAX).unwrap();
    let reference = Reference::new(&g).unwrap();
    let out = RefAccum::new(&g, &mut alloc, w, h).unwrap();
    let b = reference.bind(&g, &accel, &mats, &out).unwrap();
    let s = Settings { max_bounces: bounces, ..Settings::default() };

    for (cname, cam) in cameras(w, h) {
        for &(tname, hour) in &times {
            let sun = SunPath::default().direction(hour);
            let light = Lighting::new(Atmosphere::default(), sun);
            let t = Instant::now();
            reference.accumulate(&g, &mut tl, &b, &out, &cam, &light, &s, RefFaults::default(), SEED, 0..spp, 0, 8).unwrap();
            let img = reference.read(&g, &mut alloc, &mut tl, &out, spp).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let a = &img.accum;
            let n = a.sum.len();
            let mean = (0..n).fold([0.0; 3], |m, i| {
                let p = a.mean(i);
                [m[0] + p[0] / n as f64, m[1] + p[1] / n as f64, m[2] + p[2] / n as f64]
            });
            let rse = (0..n)
                .map(|i| {
                    let (m, e) = (luminance(a.mean(i)), luminance(a.std_error(i)));
                    if m > 0.0 { e / m } else { 0.0 }
                })
                .sum::<f64>()
                / n as f64;
            let stem = format!("{cname}_{tname}");
            write_pfm(&out_dir.join(format!("{stem}.pfm")), a).unwrap();
            let exposure = write_ppm(&out_dir.join(format!("{stem}.ppm")), a).unwrap();
            println!(
                "{{\"image\":\"{stem}\",\"hour\":{hour},\"sun_elevation_deg\":{:.2},\"sun_at_street\":[{:.5},{:.5},{:.5}],\"mean\":[{:.5e},{:.5e},{:.5e}],\"mean_relative_std_error\":{rse:.4},\"exposure\":{exposure:.2},\"size\":[{w},{h}],\"spp\":{spp},\"bounces\":{bounces},\"bad_samples\":{},\"seconds\":{secs:.1}}}",
                elevation_deg(sun),
                light.sun_at_ground[0],
                light.sun_at_ground[1],
                light.sun_at_ground[2],
                mean[0],
                mean[1],
                mean[2],
                img.bad_samples
            );
        }
    }

    g.wait_idle().unwrap();
    b.destroy(&g);
    out.free(&g, &mut alloc);
    mats.free(&g, &mut alloc);
    accel.free_now(&g, &mut alloc);
    gm.free_now(&g, &mut alloc);
    reference.destroy(&g);
    up.destroy(&g, &mut alloc);
    tl.destroy(&g);
    let (errors, warnings) = g.validation_counts();
    eprintln!("validation: {errors} errors, {warnings} warnings");
    assert_eq!(alloc.destroy(&g), 0, "leaked buffers");
    assert_eq!(errors, 0);
}
