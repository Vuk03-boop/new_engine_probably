//! Phase 3A: renders the ADR-0005 reference images of the street (GPU reference path tracer).
//!
//! `ref_light <out dir> [--size W H] [--spp N] [--bounces N] [--times name,name] [--scene street|night [--dressing lamps|windows|full|dense]]`
//!
//! For each reference camera and time of day it writes `<camera>_<time>.pfm` (linear HDR, the data)
//! and `<camera>_<time>.ppm` (for looking: the metric exposure `light::exposure::metric_exposure`,
//! key 0.18 over the log-average luminance of the pixels at least 2⁻¹⁰ of the mean, then the ACES
//! fit and sRGB encoding; display only). One JSON line per image goes to stdout: time, sun elevation,
//! sun irradiance at the street, whether the lights are on, mean radiance, the mean relative standard
//! error of the pixels, the exposure and the render time.
//!
//! 4A: `--scene night` renders `world::scene::street_night` (`--dressing`, default `full`) at the M4
//! times (`light::sun::NIGHT_TIMES`) with the emitter table bound; the lights follow
//! `light::emitters::lights_on` (on below the horizon: emission and emitter light at every vertex).
//! The default `--scene street` is the 3A set, unchanged apart from the display exposure.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use derived::{extract_world, Merge};
use gpu::accel::{Accel, AccelFaults};
use gpu::alloc::Allocator;
use gpu::emitters::{self, RefEmitters};
use gpu::layout::{build_regions, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::Camera;
use gpu::reference::{RefAccum, RefFaults, RefMaterials, Reference};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::{Gpu, Timeline};
use light::emitters::lights_on;
use light::exposure::metric_exposure;
use light::reference::{albedos, Accum, Lighting, Settings};
use light::sun::{elevation_deg, NIGHT_TIMES, REFERENCE_TIMES};
use light::{Atmosphere, SunPath};
use world::dims::VOXEL_SIZE_M;
use world::scene::{self, Dressing};

const SEED: u32 = 0x3A;
/// The night set's seed (4A).
const SEED_NIGHT: u32 = 0x4A;

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

/// Writes the display image with the metric exposure; returns it (`None`: the image is black, shown
/// with exposure 1).
fn write_ppm(path: &Path, a: &Accum) -> std::io::Result<Option<f64>> {
    let n = a.sum.len();
    let metric = metric_exposure(&(0..n).map(|i| luminance(a.mean(i))).collect::<Vec<f64>>());
    let exposure = metric.unwrap_or(1.0);
    let aces = |x: f64| ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
    let srgb = |x: f64| if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{} {}\n255\n", a.width, a.height)?;
    for i in 0..n {
        let m = a.mean(i);
        f.write_all(&m.map(|c| (srgb(aces(c * exposure)) * 255.0 + 0.5) as u8))?;
    }
    Ok(metric)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).expect("usage: ref_light <out dir> [--size W H] [--spp N] [--bounces N] [--times a,b] [--scene street|night [--dressing lamps|windows|full|dense]]"));
    let (mut w, mut h, mut spp, mut bounces) = (960u32, 540u32, 2048u32, 8u32);
    let mut want: Option<Vec<String>> = None;
    let (mut night, mut dressing) = (false, None);
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
                want = Some(args[i + 1].split(',').map(str::to_string).collect());
                i += 2;
            }
            "--scene" => {
                night = match args[i + 1].as_str() {
                    "street" => false,
                    "night" => true,
                    v => panic!("--scene: {v}"),
                };
                i += 2;
            }
            "--dressing" => {
                dressing = Some(*Dressing::ALL.iter().find(|d| d.name() == args[i + 1]).unwrap_or_else(|| panic!("unknown dressing {}", args[i + 1])));
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(night || dressing.is_none(), "--dressing needs --scene night");
    let dressing = dressing.unwrap_or_default();
    let mut times: Vec<(&str, f64)> = if night { NIGHT_TIMES.to_vec() } else { REFERENCE_TIMES.to_vec() };
    if let Some(want) = &want {
        times.retain(|(n, _)| want.iter().any(|x| x == n));
    }
    let seed = if night { SEED_NIGHT } else { SEED };
    std::fs::create_dir_all(&out_dir).unwrap();

    let g = Gpu::new().expect("an RT-capable Vulkan device");
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let (world, _) = if night { scene::street_night(dressing) } else { scene::street_block() };
    let albedo = albedos(world.materials()).expect("material albedos in [0, 1]");
    let meshes: Vec<_> = world.bricks().map(|(k, _)| (k, extract_world(&world, k, Merge::Greedy).unwrap())).collect();
    let rs = build_regions(meshes.iter().map(|(k, m)| (*k, m)), RegionSize::Chunk);
    let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs).unwrap();
    let accel = Accel::build(&g, &mut alloc, &mut up, &mut tl, &gm, AccelFaults::default()).unwrap();
    let (mats, v) = RefMaterials::upload(&g, &mut alloc, &mut up, &mut tl, &albedo).unwrap();
    tl.wait(&g, v, u64::MAX).unwrap();
    // 4A: the night scene binds its emitter table (snapshot 1: the scene as built).
    let em = if night {
        let table = emitters::table(&rs, world.materials(), 1).expect("the night scene's emitter table");
        eprintln!("street_night ({}): {} emitters", dressing.name(), table.len());
        let (em, v) = RefEmitters::upload(&g, &mut alloc, &mut up, &mut tl, &table).unwrap();
        tl.wait(&g, v, u64::MAX).unwrap();
        Some(em)
    } else {
        None
    };
    let reference = Reference::new(&g).unwrap();
    let out = RefAccum::new(&g, &mut alloc, w, h).unwrap();
    let b = match &em {
        Some(em) => reference.bind_lit(&g, &accel, &mats, em, &out).unwrap(),
        None => reference.bind(&g, &accel, &mats, &out).unwrap(),
    };

    let mut bad = 0;
    for (cname, cam) in cameras(w, h) {
        for &(tname, hour) in &times {
            let sun = SunPath::default().direction(hour);
            let light = Lighting::new(Atmosphere::default(), sun);
            // The lights follow the sun (night scene only; the street block has none).
            let lights = night && lights_on(sun);
            let s = Settings { max_bounces: bounces, emission: lights, emitters_direct: lights, emitters_indirect: lights, ..Settings::default() };
            eprintln!("rendering {cname}_{tname}: {spp} spp, lights {}", if lights { "on" } else { "off" });
            let t = Instant::now();
            reference.accumulate(&g, &mut tl, &b, &out, &cam, &light, &s, RefFaults::default(), seed, 0..spp, 0, 8).unwrap();
            let img = reference.read(&g, &mut alloc, &mut tl, &out, spp).unwrap();
            let secs = t.elapsed().as_secs_f64();
            bad += img.bad_samples;
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
            let exposure = write_ppm(&out_dir.join(format!("{stem}.ppm")), a).unwrap().map_or("null".to_string(), |x| format!("{x:.6e}"));
            println!(
                "{{\"image\":\"{stem}\",\"scene\":\"{}\",\"dressing\":{},\"lights\":{lights},\"emitters\":{},\"hour\":{hour},\"sun_elevation_deg\":{:.2},\"sun_at_street\":[{:.5},{:.5},{:.5}],\"mean\":[{:.5e},{:.5e},{:.5e}],\"mean_relative_std_error\":{rse:.4},\"exposure\":{exposure},\"size\":[{w},{h}],\"spp\":{spp},\"bounces\":{bounces},\"seed\":{seed},\"bad_samples\":{},\"seconds\":{secs:.1}}}",
                if night { "night" } else { "street" },
                if night { format!("\"{}\"", dressing.name()) } else { "null".to_string() },
                em.as_ref().map_or(0, |e| e.count),
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
    if let Some(em) = em {
        em.free(&g, &mut alloc);
    }
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
    assert_eq!(bad, 0, "bad samples (NaN, infinite or negative)");
}
