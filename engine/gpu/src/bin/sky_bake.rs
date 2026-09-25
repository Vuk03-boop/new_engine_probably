//! S-020: bakes the reference sky for the sky correction and writes `light::sky_ref`'s file.
//!
//! `sky_bake [--scale D] [--per-dispatch M] [--seed S] [--out PATH]`
//!
//! Samples per texel by slice are `light::sky_ref::bake_samples` divided by D (default 1; a smoke
//! run uses e.g. 512 with `--per-dispatch 64`); 1,024 per dispatch; seed 0x5C0; the file `light/data/sky_reference_v1.bin`.
//! Slices with the same sample count are baked together. Rerun it when the atmosphere or the sky estimator's settings
//! change (the file's fingerprint then no longer matches and it is refused). JSON lines go to
//! stdout: the noise of the stored mean and the correction it gives against the current table code.

use std::path::PathBuf;

use gpu::alloc::Allocator;
use gpu::sky_bake::bake;
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::{Gpu, Timeline};
use light::sky::SkyLuts;
use light::sky_ref::{self, elevation_deg, fingerprint, SkyReference, CORR_H, CORR_S, CORR_W};
use light::{Atmosphere, SkyOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut scale, mut per_dispatch, mut seed) = (1u32, 1024u32, 0x5C0u32);
    let mut out = PathBuf::from(sky_ref::DEFAULT_FILE);
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).expect("missing value");
        match args[i].as_str() {
            "--scale" => scale = v.parse().expect("--scale D"),
            "--per-dispatch" => per_dispatch = v.parse().expect("--per-dispatch M"),
            "--seed" => seed = v.parse().expect("--seed S"),
            "--out" => out = PathBuf::from(v),
            a => panic!("unknown argument {a}; usage: sky_bake [--scale D] [--per-dispatch M] [--seed S] [--out PATH]"),
        }
        i += 2;
    }
    let a = Atmosphere::default();
    let o = SkyOptions::default();
    let mut jobs = Vec::with_capacity(CORR_W * CORR_H * CORR_S);
    for k in 0..CORR_S {
        for y in 0..CORR_H {
            for x in 0..CORR_W {
                debug_assert_eq!(jobs.len(), sky_ref::index(x, y, k));
                jobs.push(sky_ref::texel(&a, x, y, k));
            }
        }
    }

    let g = Gpu::new().expect("an RT-capable Vulkan device");
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let samples: Vec<u32> = (0..CORR_S).map(|k| sky_ref::bake_samples(k) / scale).collect();
    let per_slice = CORR_W * CORR_H;
    let (mut mean, mut se) = (vec![[0.0; 3]; jobs.len()], vec![[0.0; 3]; jobs.len()]);
    let t0 = std::time::Instant::now();
    let mut k = 0;
    while k < CORR_S {
        let mut end = k + 1;
        while end < CORR_S && samples[end] == samples[k] {
            end += 1;
        }
        let (a0, a1) = (k * per_slice, end * per_slice);
        let n = samples[k];
        eprintln!("slices {k}..{end} ({}° to {}°): {} texels x {n} samples, {} replicas, seed {seed:#x}", elevation_deg(k), elevation_deg(end - 1), a1 - a0, gpu::sky_bake::replicas(a1 - a0));
        let every = (n / 8).max(1);
        let baked = bake(&g, &mut alloc, &mut up, &mut tl, &jobs[a0..a1], a0 as u32, &o, seed, n, per_dispatch, |done| {
            if done.is_multiple_of(every) {
                eprintln!("  {done} / {n}, {:.0} s", t0.elapsed().as_secs_f64());
            }
        })
        .unwrap();
        mean[a0..a1].copy_from_slice(&baked.mean);
        se[a0..a1].copy_from_slice(&baked.se);
        eprintln!("  {:.1} s ({:.0} M samples/s)", baked.seconds, (a1 - a0) as f64 * n as f64 / baked.seconds / 1e6);
        k = end;
    }
    let seconds = t0.elapsed().as_secs_f64();
    g.wait_idle().unwrap();
    let (errors, warnings) = g.validation_counts();
    up.destroy(&g, &mut alloc);
    tl.destroy(&g);
    assert_eq!(alloc.destroy(&g), 0, "leaked buffers");

    let r = SkyReference { samples: samples.clone(), seed, options: o, fingerprint: fingerprint(&a, &o), mean, se };
    let bytes = r.to_bytes();
    // The file is written once, then re-read and checked exactly as the engine loads it.
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let tmp = out.with_extension("tmp");
    std::fs::write(&tmp, &bytes).unwrap();
    std::fs::rename(&tmp, &out).unwrap();
    let back = SkyReference::load(&out, &a, &o).expect("re-read the baked file");

    // Noise of the stored mean, by sun elevation band.
    let mut bands: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for k in 0..CORR_S {
        for t in k * CORR_W * CORR_H..(k + 1) * CORR_W * CORR_H {
            for c in 0..3 {
                if back.mean[t][c] > 0.0 {
                    bands[usize::from(elevation_deg(k) >= 0.0)].push(back.se[t][c] / back.mean[t][c]);
                }
            }
        }
    }
    let mut luts = SkyLuts::new(a);
    let stats = luts.apply_reference(&back).unwrap();
    println!(
        "{{\"file\":{:?},\"bytes\":{},\"texels\":{},\"samples_by_slice\":{samples:?},\"seed\":{seed},\"seconds\":{seconds:.1},\"validation_errors\":{errors},\"validation_warnings\":{warnings},\"ratio_min\":{:.4},\"ratio_max\":{:.4},\"clamped\":{},\"dark\":{}}}",
        out.display().to_string(),
        bytes.len(),
        jobs.len(),
        stats.min,
        stats.max,
        stats.clamped,
        stats.dark
    );
    for (name, v) in ["sun_below_0", "sun_at_or_above_0"].into_iter().zip(bands.iter_mut()) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
        println!("{{\"noise\":\"{name}\",\"values\":{},\"rel_se_p50\":{:.5},\"rel_se_p95\":{:.5},\"rel_se_max\":{:.5}}}", v.len(), p(0.5), p(0.95), p(1.0));
    }
    // The correction per slice: its mean and range over the texels, per channel.
    for k in 0..CORR_S {
        let s = &luts.correction[k * CORR_W * CORR_H..(k + 1) * CORR_W * CORR_H];
        let stat = |c: usize| {
            let (mut lo, mut hi, mut sum) = (f64::INFINITY, 0.0f64, 0.0);
            for t in s {
                lo = lo.min(t[c]);
                hi = hi.max(t[c]);
                sum += t[c];
            }
            format!("[{:.4},{:.4},{:.4}]", sum / s.len() as f64, lo, hi)
        };
        println!("{{\"slice\":{k},\"sun_elevation_deg\":{},\"ratio_mean_min_max\":{{\"r\":{},\"g\":{},\"b\":{}}}}}", elevation_deg(k), stat(0), stat(1), stat(2));
    }
}
