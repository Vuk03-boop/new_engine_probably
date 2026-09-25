//! Phase 2C-1 GPU integration tests: raster G-buffer against the CPU voxel reference
//! (ADR-0003 Amendment 1), planted-fault controls, and the targets' budget; 2C-2 adds the debug
//! views (every pixel re-derived on the CPU from the G-buffer) and the GPU timer; M1 adds the lit
//! view to the same check. Like `device.rs`, they
//! need the Vulkan SDK and a GPU and fail (never skip) without them.
//! Run: `cargo test --release -j 2 -p gpu --test raster -- --test-threads=1 --nocapture`.

use std::collections::BTreeMap;
use std::time::Instant;

use ash::vk;
use derived::{extract_world, BrickMesh, Merge, Split};
use gpu::alloc::Allocator;
use gpu::equivalence::{self, RefPixel, Report};
use gpu::layout::{build_regions, build_regions_with, RegionKey, RegionMesh, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::raster::{Camera, Faults, Frame, Raster, Targets, BACKGROUND_SURFACE, FS_REFLECTION, VS_REFLECTION};
use gpu::staging::{Uploader, DEFAULT_RING_BYTES};
use gpu::{Gpu, GpuError, Timeline};
use memory::{Budget, Category};
use world::dims::VOXEL_SIZE_M;
use world::{scene, BrickKey, World};

const W: u32 = 640;
const H: u32 = 360;
const NEAR: f64 = 0.1;
const REF_THREADS: usize = 2;

fn gpu() -> Gpu {
    let g = Gpu::new().expect("an RT-capable Vulkan device is required for gpu tests");
    assert!(g.validation_enabled(), "gpu tests require the Khronos validation layer (Vulkan SDK)");
    if !g.ray_tracing() {
        eprintln!("SUPPLEMENTAL RUN on a non-RT device ({}): not the P-001 device gate", g.info.name);
    }
    g
}

fn assert_clean(g: &Gpu) {
    let (errors, _) = g.validation_counts();
    assert_eq!(errors, 0, "validation errors: {:?}", g.first_validation_errors());
}

/// Street cameras, voxel units. Declared before the first measurement.
fn cameras() -> Vec<(&'static str, Camera)> {
    let (_, view) = scene::street_block();
    let vox = |m: [f64; 3]| m.map(|x| x / VOXEL_SIZE_M);
    vec![
        // The scene's reference view.
        ("street_view", Camera::look_at(vox(view.eye_m), vox(view.target_m), view.vertical_fov_deg, W, H, NEAR)),
        // Eye exactly on a chunk corner (x, y, z multiples of 32): rays start on region boundaries.
        ("chunk_corner", Camera::look_at([128.0, 32.0, 64.0], [250.0, 20.0, 190.0], 70.0, W, H, NEAR)),
        // High oblique view over roofs and many region boundaries.
        ("overhead", Camera::look_at([-60.5, 300.25, -80.75], [192.0, 0.0, 150.0], 50.0, W, H, NEAR)),
        // Low and down the street: the ground at grazing angles, long T-junction edges.
        ("grazing", Camera::look_at([4.3, 8.6, 20.2], [380.0, 3.0, 30.0], 60.0, W, H, NEAR)),
    ]
}

fn street_meshes(w: &World, merge: Merge) -> Vec<(BrickKey, BrickMesh)> {
    w.bricks().map(|(k, _)| (k, extract_world(w, k, merge).unwrap())).collect()
}

fn regions(meshes: &[(BrickKey, BrickMesh)], size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
}

struct Rig {
    g: Gpu,
    alloc: Allocator,
    tl: Timeline,
    up: Uploader,
    raster: Raster,
    targets: Targets,
}

impl Rig {
    fn new() -> Rig {
        let g = gpu();
        let mut alloc = Allocator::new(g.device_budget());
        let tl = Timeline::new(&g).unwrap();
        let up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
        let raster = Raster::new(&g).unwrap();
        let targets = Targets::new(&g, &mut alloc, vk::Extent2D { width: W, height: H }).unwrap();
        Rig { g, alloc, tl, up, raster, targets }
    }

    fn upload(&mut self, size: RegionSize, rs: &BTreeMap<RegionKey, RegionMesh>) -> GpuMeshes {
        GpuMeshes::upload(&self.g, &mut self.alloc, &mut self.up, &mut self.tl, size, rs).unwrap().0
    }

    fn render_with(&mut self, raster: Option<&Raster>, gm: &GpuMeshes, cam: &Camera, faults: Faults) -> Frame {
        let raster = raster.unwrap_or(&self.raster);
        let b = raster.bind(&self.g, gm).unwrap();
        let f = raster.render_to_host(&self.g, &mut self.alloc, &mut self.tl, &self.targets, gm, &b, cam, faults).unwrap();
        b.destroy(&self.g);
        f
    }

    fn render(&mut self, gm: &GpuMeshes, cam: &Camera, faults: Faults) -> Frame {
        self.render_with(None, gm, cam, faults)
    }

    fn finish(self) {
        let Rig { g, mut alloc, tl, up, raster, targets } = self;
        g.wait_idle().unwrap();
        targets.free(&g, &mut alloc);
        raster.destroy(&g);
        up.destroy(&g, &mut alloc);
        tl.destroy(&g);
        assert_eq!(alloc.destroy(&g), 0, "leaked buffers or images");
    }
}

fn reference(w: &World, name: &str, cam: &Camera) -> Vec<RefPixel> {
    let t = Instant::now();
    let r = equivalence::reference(w, cam, REF_THREADS);
    let hits = r.iter().filter(|p| p.centre.is_some()).count();
    eprintln!("reference {name}: {W}x{H}, 5 rays/px, {hits} hits, {:.2} s on {REF_THREADS} threads", t.elapsed().as_secs_f64());
    r
}

/// 24-bit BMP of a linear RGBA8 readback, sRGB-encoded as the swapchain would, for visual
/// inspection only.
fn write_rgba_bmp(path: &std::path::Path, px: &[u8], fw: u32, fh: u32) {
    let (fw, fh) = (fw as usize, fh as usize);
    let row = (3 * fw).next_multiple_of(4);
    let mut b = Vec::with_capacity(54 + row * fh);
    b.extend_from_slice(b"BM");
    b.extend_from_slice(&((54 + row * fh) as u32).to_le_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&54u32.to_le_bytes());
    b.extend_from_slice(&40u32.to_le_bytes());
    b.extend_from_slice(&(fw as i32).to_le_bytes());
    b.extend_from_slice(&(fh as i32).to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24u16.to_le_bytes());
    b.extend_from_slice(&[0; 24]);
    let srgb = |v: u8| {
        let l = v as f32 / 255.0;
        let s = if l <= 0.0031308 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round() as u8
    };
    for y in (0..fh).rev() {
        let start = b.len();
        for x in 0..fw {
            let i = 4 * (y * fw + x);
            b.extend_from_slice(&[srgb(px[i + 2]), srgb(px[i + 1]), srgb(px[i])]);
        }
        b.resize(start + row, 0);
    }
    std::fs::write(path, b).expect("write bmp");
}

/// 24-bit BMP of the material target, shaded by face axis, for visual inspection (not evidence of
/// correctness: the equivalence counts are).
fn write_bmp(path: &std::path::Path, frame: &Frame, w: &World) {
    let colors: BTreeMap<u16, [f32; 3]> = w.materials().iter().map(|(id, d)| (id.raw(), d.params.base_color)).collect();
    let (fw, fh) = (frame.width as usize, frame.height as usize);
    let row = (3 * fw).next_multiple_of(4);
    let mut b = Vec::with_capacity(54 + row * fh);
    let size = (54 + row * fh) as u32;
    b.extend_from_slice(b"BM");
    b.extend_from_slice(&size.to_le_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&54u32.to_le_bytes());
    b.extend_from_slice(&40u32.to_le_bytes());
    b.extend_from_slice(&(fw as i32).to_le_bytes());
    b.extend_from_slice(&(fh as i32).to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24u16.to_le_bytes());
    b.extend_from_slice(&[0; 24]);
    for y in (0..fh).rev() {
        let start = b.len();
        for x in 0..fw {
            let i = y * fw + x;
            let rgb = if frame.surface[i] == BACKGROUND_SURFACE {
                [0.45, 0.6, 0.85]
            } else {
                let c = colors.get(&frame.material[i]).copied().unwrap_or([1.0, 0.0, 1.0]);
                // Normal bits: +y brightest, x faces mid, z faces darker.
                let shade = match frame.normal[i] {
                    [0, 32767] => 1.0,
                    [0, -32767] => 0.35,
                    [32767, 0] | [-32767, 0] => 0.75,
                    _ => 0.6,
                };
                c.map(|v| v * shade)
            };
            let srgb = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            b.extend_from_slice(&[srgb(rgb[2]), srgb(rgb[1]), srgb(rgb[0])]);
        }
        b.resize(start + row, 0);
    }
    std::fs::write(path, b).unwrap();
}

fn results_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../results")
}

#[test]
fn raster_matches_the_voxel_reference_for_every_setting() {
    let (w, _) = scene::street_block();
    let cams = cameras();
    let refs: Vec<Vec<RefPixel>> = cams.iter().map(|(n, c)| reference(&w, n, c)).collect();
    let mut rig = Rig::new();
    eprintln!("device {}; targets {W}x{H}: {} B driver-required", rig.g.info.name, rig.targets.device_bytes());
    let mut failed = Vec::new();
    for merge in Merge::ALL {
        let meshes = street_meshes(&w, merge);
        for size in RegionSize::ALL {
            let rs = regions(&meshes, size);
            let gm = rig.upload(size, &rs);
            eprintln!("merge {} region {}: {} quads, {} triangles, {} B device images", merge.name(), size.name(), gm.quad_count(), gm.triangle_count(), gm.device_bytes());
            for ((name, cam), rf) in cams.iter().zip(&refs) {
                let frame = rig.render(&gm, cam, Faults::default());
                let rep = equivalence::compare(&frame, rf, cam, &rs);
                eprintln!("camera {name} merge {} region {}: {}", merge.name(), size.name(), rep.summary());
                for e in &rep.examples {
                    eprintln!("    {e}");
                }
                // Declared before measuring: the exact checks must cover at least a quarter of the
                // reference hits, so the edge exclusion cannot quietly swallow the image.
                if !rep.passes() || rep.checked * 4 < rep.ref_hits {
                    failed.push(format!("{name}/{}/{}", merge.name(), size.name()));
                }
                if merge == Merge::Greedy && size == RegionSize::Chunk {
                    write_bmp(&results_dir().join(format!("raster_2c1_{name}.bmp")), &frame, &w);
                }
            }
            rig.g.wait_idle().unwrap();
            gm.free_now(&rig.g, &mut rig.alloc);
        }
    }
    assert_clean(&rig.g);
    rig.finish();
    assert!(failed.is_empty(), "settings that break ADR-0003 Amendment 1: {failed:?}");
}

fn check_control(name: &str, rep: &Report, caught: bool) {
    eprintln!("control {name}: {}", rep.summary());
    for e in rep.examples.iter().take(3) {
        eprintln!("    {e}");
    }
    assert!(caught, "planted fault '{name}' was not caught: {}", rep.summary());
}

#[test]
fn planted_faults_are_caught() {
    let (w, _) = scene::street_block();
    let (name, cam) = cameras().remove(0);
    let rf = reference(&w, name, &cam);
    let mut rig = Rig::new();
    let meshes = street_meshes(&w, Merge::Greedy);
    let size = RegionSize::Chunk;
    let rs = regions(&meshes, size);
    let gm = rig.upload(size, &rs);

    let good = rig.render(&gm, &cam, Faults::default());
    let base = equivalence::compare(&good, &rf, &cam, &rs);
    eprintln!("control baseline: {}", base.summary());
    assert!(base.passes(), "the unfaulted render must pass before the controls mean anything");

    // 1. Reversed winding: front faces culled, back faces of far walls drawn instead.
    let reversed = Raster::with_front_face(&rig.g, vk::FrontFace::CLOCKWISE, VS_REFLECTION, FS_REFLECTION).unwrap();
    let f = rig.render_with(Some(&reversed), &gm, &cam, Faults::default());
    reversed.destroy(&rig.g);
    let rep = equivalence::compare(&f, &rf, &cam, &rs);
    check_control("reversed_winding", &rep, rep.failures() * 2 > rep.ref_hits);

    // 2. Camera offset off by one voxel in x.
    let f = rig.render(&gm, &cam, Faults { offset_error: [1, 0, 0], ..Faults::default() });
    let rep = equivalence::compare(&f, &rf, &cam, &rs);
    check_control("offset_off_by_one", &rep, rep.face + rep.depth > 0);

    // 3. The material (and face) read from the wrong primitive: every region's quad table is
    //    rotated by one record on the device, as if the shader read quad q + 1.
    for (k, r) in &gm.regions {
        let (_, image) = rs[k].image(gm.align);
        let s = r.sections;
        let table = &image[s.quads as usize..(s.quads + r.quad_count * 8) as usize];
        let mut rotated = table[8..].to_vec();
        rotated.extend_from_slice(&table[..8]);
        rig.up.upload(&rig.g, &mut rig.tl, &r.buffer, s.quads, &rotated).unwrap();
    }
    rig.up.flush(&rig.g, &mut rig.tl).unwrap();
    let f = rig.render(&gm, &cam, Faults::default());
    let rep = equivalence::compare(&f, &rf, &cam, &rs);
    check_control("material_from_wrong_primitive", &rep, rep.material > 0);
    // Restore the tables.
    for (k, r) in &gm.regions {
        let (_, image) = rs[k].image(gm.align);
        rig.up.upload(&rig.g, &mut rig.tl, &r.buffer, 0, &image).unwrap();
    }
    rig.up.flush(&rig.g, &mut rig.tl).unwrap();
    let f = rig.render(&gm, &cam, Faults::default());
    assert!(equivalence::compare(&f, &rf, &cam, &rs).passes(), "restored tables pass again");

    // 4. A dropped region: the one covering the most pixels in the good render.
    let mut count: BTreeMap<u32, u64> = BTreeMap::new();
    for s in &good.surface {
        if *s != BACKGROUND_SURFACE {
            *count.entry(s[0]).or_default() += 1;
        }
    }
    let (&busiest, &px) = count.iter().max_by_key(|(_, &n)| n).unwrap();
    let f = rig.render(&gm, &cam, Faults { skip_region: Some(busiest as usize), ..Faults::default() });
    let rep = equivalence::compare(&f, &rf, &cam, &rs);
    eprintln!("dropped region {busiest} covered {px} px");
    check_control("dropped_region", &rep, rep.crack > 0);

    rig.g.wait_idle().unwrap();
    gm.free_now(&rig.g, &mut rig.alloc);
    assert_clean(&rig.g);
    rig.finish();
}

#[test]
fn negative_control_unsplit_greedy_cracks_again() {
    // The pre-amendment layout (two triangles per greedy quad) must still show the T-junction
    // crack the first 2C-1 run found, so the watertight layout is what closes it.
    let (w, _) = scene::street_block();
    let (name, cam) = cameras().remove(2);
    assert_eq!(name, "overhead");
    let rf = reference(&w, name, &cam);
    let mut rig = Rig::new();
    let meshes = street_meshes(&w, Merge::Greedy);
    let size = RegionSize::Chunk;
    for (split, want_crack) in [(Split::CornersOnly, true), (Split::Watertight, false)] {
        let rs = build_regions_with(meshes.iter().map(|(k, m)| (*k, m)), size, split);
        let gm = rig.upload(size, &rs);
        let f = rig.render(&gm, &cam, Faults::default());
        let rep = equivalence::compare(&f, &rf, &cam, &rs);
        eprintln!("overhead greedy chunk, split {split:?}: {} triangles: {}", gm.triangle_count(), rep.summary());
        for e in rep.examples.iter().take(3) {
            eprintln!("    {e}");
        }
        assert_eq!(rep.crack > 0, want_crack, "split {split:?}");
        rig.g.wait_idle().unwrap();
        gm.free_now(&rig.g, &mut rig.alloc);
    }
    assert_clean(&rig.g);
    rig.finish();
}

#[test]
fn targets_take_ledger_grants_and_a_refusal_changes_nothing() {
    let g = gpu();
    let mut alloc = Allocator::new(g.device_budget());
    let extent = vk::Extent2D { width: 1920, height: 1080 };
    let t = Targets::new(&g, &mut alloc, extent).unwrap();
    let acc = alloc.ledger().account(Category::GpuTemporal);
    let texels = 18 * 1920 * 1080u64;
    eprintln!("1080p targets: {} B texels (18 B/px), {} B driver-required, ledger GpuTemporal live {} reserved {}", texels, t.device_bytes(), acc.usage.live, acc.usage.reserved);
    assert!(t.device_bytes() >= texels && acc.usage.live == t.device_bytes());
    assert_eq!(alloc.stats().images_live, 4);
    t.free(&g, &mut alloc);
    assert_eq!(alloc.ledger().account(Category::GpuTemporal).usage.live, 0);
    assert_eq!(alloc.stats().blocks_live, 0, "empty blocks are returned");
    assert_eq!(alloc.destroy(&g), 0);

    // Refusal: a budget of 8 MiB cannot hold a 1080p G-buffer (37 MB).
    let mut small = Allocator::with_block_bytes(Budget::unlimited().with_total(8 << 20), 4 << 20);
    let before = (small.ledger().account(Category::GpuTemporal), small.stats());
    match Targets::new(&g, &mut small, extent) {
        Err(GpuError::OverBudget(_)) => {}
        other => panic!("expected OverBudget, got {:?}", other.map(|_| ())),
    }
    let after = (small.ledger().account(Category::GpuTemporal), small.stats());
    assert_eq!(before.0.usage, after.0.usage, "no bytes left reserved");
    assert_eq!((after.1.images_live, after.1.blocks_live), (0, 0), "all-or-nothing");
    assert!(small.ledger().account(Category::GpuTemporal).refusals > before.0.refusals, "refusal counted");
    assert_eq!(small.destroy(&g), 0);
    assert_clean(&g);
}

// ---- 2C-2: debug views and GPU timing, headless (the window itself is exercised by the viewer)

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

fn hash_colour(h: u32) -> [f32; 3] {
    let h = hash(h);
    [h & 255, (h >> 8) & 255, (h >> 16) & 255].map(|c| c as f32 / 255.0 * 0.8 + 0.2)
}

fn hash3(k: [i32; 3]) -> u32 {
    hash((k[0] as u32).wrapping_mul(73856093) ^ hash((k[1] as u32).wrapping_mul(19349663) ^ hash((k[2] as u32).wrapping_mul(83492791))))
}

/// The colour `debug_view.slang` must write for pixel `i` of `frame`, re-derived on the CPU from
/// the read-back G-buffer (linear, before UNORM rounding).
/// The lit view's tone curve (Narkowicz's ACES fit), as the shader's `aces`.
fn aces(x: f32) -> f32 {
    (x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn expected_view(view: gpu::debug_view::View, frame: &Frame, i: usize, mats: &[world::MaterialParams], keys: &[RegionKey], snapshot: u32, cam: &Camera, light: &gpu::debug_view::Lighting) -> [f32; 3] {
    use gpu::debug_view::View;
    let (d, s) = (frame.depth[i], frame.surface[i]);
    let lerp = |a: [f32; 3], b: [f32; 3], t: f32| [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * t);
    if d <= 0.0 || s[0] as usize >= keys.len() {
        if view == View::Lit {
            let r = cam.ray((i as u32) % frame.width, (i as u32) / frame.width);
            let dy = (r.dir[1] / (r.dir[0] * r.dir[0] + r.dir[1] * r.dir[1] + r.dir[2] * r.dir[2]).sqrt()) as f32;
            return lerp([0.7, 0.8, 0.92], [0.25, 0.45, 0.85], dy.clamp(0.0, 1.0)).map(|v| aces(v * light.exposure));
        }
        return [0.45, 0.62, 0.85];
    }
    let e = frame.normal[i].map(|b| (b as f32 / 32767.0).max(-1.0));
    let mut n = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
    if n[2] < 0.0 {
        let sg = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
        let (x, y) = (n[0], n[1]);
        n[0] = (1.0 - y.abs()) * sg(x);
        n[1] = (1.0 - x.abs()) * sg(y);
    }
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let n = n.map(|v| v / l);
    let shade = n[0].abs() * 0.75 + n[1].abs() * 1.0 + n[2].abs() * 0.6;
    let key = keys[s[0] as usize];
    match view {
        View::Material => {
            let m = frame.material[i] as usize;
            if m < mats.len() { mats[m].base_color.map(|c| c * shade) } else { [1.0, 0.0, 1.0] }
        }
        View::Lit => {
            let m = frame.material[i] as usize;
            if m >= mats.len() {
                return [1.0, 0.0, 1.0];
            }
            let sd = light.sun_dir;
            let sl = (sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]).sqrt();
            let sun = sd.map(|v| (v / sl) as f32);
            let ndl = (n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]).max(0.0);
            let ambient = lerp([0.35, 0.3, 0.25], [0.55, 0.7, 1.0], n[1] * 0.5 + 0.5).map(|v| v * light.ambient);
            let direct = [1.0, 0.95, 0.88].map(|v: f32| v * light.sun_intensity * ndl);
            let p = mats[m];
            [0, 1, 2].map(|c| aces((p.base_color[c] * (direct[c] + ambient[c]) + p.emissive[c]) * light.exposure))
        }
        View::Normal => n.map(|v| v * 0.5 + 0.5),
        View::Region => hash_colour(hash3([key.x, key.y, key.z])).map(|c| c * shade),
        View::Brick => {
            // The brick of the voxel behind the surface, from the f64 pixel ray and the stored depth.
            let (x, y) = ((i as u32) % frame.width, (i as u32) / frame.width);
            let r = cam.ray(x, y);
            let t = cam.distance(d);
            let p = [0, 1, 2].map(|a| (r.origin[a] + t * r.dir[a] - 0.5 * n[a] as f64).floor() as i32);
            hash_colour(hash3(p.map(|v| v.div_euclid(8)))).map(|c| c * shade)
        }
        View::Surface => hash_colour(hash(s[0].wrapping_mul(0x9E3779B9)) ^ s[1]).map(|c| c * shade),
        View::Depth => {
            let t = cam.near as f32 / d;
            let g = 1.0 - (t.max(1.0).log2() / 12.0).clamp(0.0, 1.0);
            [g; 3]
        }
        View::Snapshot => hash_colour(snapshot + 1).map(|c| c * shade),
        View::Light | View::HistoryAge | View::HistoryReason | View::Motion => unreachable!("not a G-buffer view"),
    }
}

#[test]
fn debug_views_show_the_stored_gbuffer_values_and_the_timer_measures() {
    use gpu::debug_view::{DebugView, Lighting, Tables, View};
    use gpu::present::image_to_attachment;
    use gpu::submit::{all_to_host, Submitter};
    use gpu::timing::GpuTimer;

    let (w, scene_view) = scene::street_block();
    let (_, cam) = cameras().remove(0);
    let light = Lighting::new(scene_view.sun_dir);
    let mut rig = Rig::new();
    let meshes = street_meshes(&w, Merge::Greedy);
    let size = RegionSize::Chunk;
    let rs = regions(&meshes, size);
    let gm = rig.upload(size, &rs);
    let frame = rig.render(&gm, &cam, Faults::default());
    let mats: Vec<world::MaterialParams> = w.materials().iter().map(|(_, d)| d.params).collect();
    let keys: Vec<RegionKey> = gm.regions.keys().copied().collect();
    let snapshot = 7u32;
    let (tables, _) = Tables::upload(&rig.g, &mut rig.alloc, &mut rig.up, &mut rig.tl, &mats, &gm, snapshot as u64).unwrap();

    // Planted reflection mismatch: the raster modules' reflection is refused before any Vulkan object.
    let fmt = vk::Format::R8G8B8A8_UNORM;
    match DebugView::with_reflection(&rig.g, fmt, VS_REFLECTION, FS_REFLECTION) {
        Err(GpuError::Layout(e)) => eprintln!("debug view with the raster reflection refused: {} errors, first {:?}", e.len(), e.first()),
        other => panic!("expected a layout error, got {:?}", other.map(|_| ())),
    }

    let dv = DebugView::new(&rig.g, fmt).unwrap();
    dv.bind(&rig.g, &rig.targets, &tables, None);
    let extent = vk::Extent2D { width: W, height: H };
    let out = rig.alloc.create_image(&rig.g, fmt, extent, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::ImageAspectFlags::COLOR, Category::GpuTemporal).unwrap();
    let rb = rig.alloc.create_buffer(&rig.g, (W * H * 4) as u64, vk::BufferUsageFlags::TRANSFER_DST, Category::Staging, gpu::Kind::Host).unwrap();
    let mut timer = GpuTimer::new(&rig.g, 1, 2).unwrap();
    assert!(timer.read(&rig.g, 0).unwrap().is_none(), "no timings before the slot is written");
    let bindings = rig.raster.bind(&rig.g, &gm).unwrap();
    let mut sub = Submitter::new(&rig.g).unwrap();
    let mut failed = Vec::new();
    // Lighting and temporal views show gpu::shade / gpu::temporal output, checked in their own tests.
    for view in View::ALL.into_iter().filter(|v| v.reads_gbuffer()) {
        let g = &rig.g;
        let cmd = sub.begin(g, &rig.tl).unwrap();
        timer.reset(g, cmd, 0);
        timer.begin_pass(g, cmd, 0, 0);
        rig.raster.record(g, cmd, &rig.targets, &gm, &bindings, &cam, Faults::default());
        timer.end_pass(g, cmd, 0, 0);
        timer.begin_pass(g, cmd, 0, 1);
        image_to_attachment(g, cmd, out.image);
        dv.record(g, cmd, &rig.targets, &tables, out.view, &cam, view, &light, gpu::debug_view::Source::Raster);
        timer.end_pass(g, cmd, 0, 1);
        let b = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(out.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })];
        let region = [vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: W, height: H, depth: 1 },
        }];
        unsafe {
            g.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b));
            g.device.cmd_copy_image_to_buffer(cmd, out.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rb.buffer, &region);
        }
        all_to_host(g, cmd);
        let v = sub.submit(g, &mut rig.tl, cmd, &[]).unwrap();
        rig.tl.wait(g, v, u64::MAX).unwrap();
        let ms = timer.read(g, 0).unwrap().expect("slot written");
        assert!(ms.iter().all(|&t| t > 0.0 && t < 1000.0), "pass times {ms:?}");

        let px = rb.mapped_ref().unwrap();
        let (mut bad, mut drawn, mut example) = (0u64, 0u64, None);
        // UNORM rounding of the same float math: 1 LSB; the depth view uses the GPU's log2: 2 LSB.
        let tol = if view == View::Depth { 2 } else { 1 };
        for i in 0..(W * H) as usize {
            if frame.surface[i] != BACKGROUND_SURFACE {
                drawn += 1;
            }
            let got = [px[4 * i], px[4 * i + 1], px[4 * i + 2]];
            let want = expected_view(view, &frame, i, &mats, &keys, snapshot, &cam, &light).map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as i32);
            if (0..3).any(|c| (got[c] as i32 - want[c]).abs() > tol) {
                bad += 1;
                if example.is_none() {
                    example = Some(format!("first at ({}, {}): GPU {got:?}, CPU {want:?}", i as u32 % W, i as u32 / W));
                }
            }
        }
        if view == View::Lit {
            write_rgba_bmp(&results_dir().join("lit_m1_street_view.bmp"), px, W, H);
        }
        eprintln!("view {:<16} {bad} of {} px differ from the CPU re-derivation ({drawn} drawn); GPU ms gbuffer {:.3} view {:.3} {}", view.name(), W * H, ms[0], ms[1], example.unwrap_or_default());
        // Declared before measuring. Exact views: every pixel. The brick view reconstructs a
        // position in f32 on the GPU (f64 here), so a tangential coordinate within rounding of a
        // voxel edge may floor differently: at most 0.1% of drawn pixels.
        let allowed = if view == View::Brick { drawn / 1000 } else { 0 };
        if bad > allowed {
            failed.push(format!("{}: {bad} px", view.name()));
        }
    }
    // Planted control: the lit readback (the last view) against a CPU with the sun 2% brighter
    // must differ, so the lit check can fail.
    {
        let px = rb.mapped_ref().unwrap();
        let brighter = Lighting { sun_intensity: light.sun_intensity * 1.02, ..light };
        let differ = (0..(W * H) as usize)
            .filter(|&i| {
                let want = expected_view(View::Lit, &frame, i, &mats, &keys, snapshot, &cam, &brighter).map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as i32);
                (0..3).any(|c| (px[4 * i + c] as i32 - want[c]).abs() > 1)
            })
            .count();
        eprintln!("planted: lit view against a 2% brighter sun on the CPU: {differ} px differ");
        assert!(differ > 1000, "the lit check cannot tell a 2% sun change: {differ} px");
    }
    // The lighting sliders, for choosing defaults by eye: the CPU lit view (exact to the GPU, as
    // checked above) at half size, rows exposure 0.4 / 0.6 / 0.8, columns sun 2 / 3 / 4, per ambient.
    for ambient in [0.4f32, 0.6, 0.9] {
        let (cw, ch) = (W / 2, H / 2);
        let mut sheet = vec![0u8; (3 * cw * 3 * ch * 4) as usize];
        for (r, exposure) in [0.4f32, 0.6, 0.8].into_iter().enumerate() {
            for (c, sun) in [2.0f32, 3.0, 4.0].into_iter().enumerate() {
                let l = Lighting { sun_intensity: sun, ambient, exposure, ..light };
                for y in 0..ch {
                    for x in 0..cw {
                        let i = (2 * y * W + 2 * x) as usize;
                        let v = expected_view(View::Lit, &frame, i, &mats, &keys, snapshot, &cam, &l).map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
                        let o = (4 * ((r as u32 * ch + y) * 3 * cw + c as u32 * cw + x)) as usize;
                        sheet[o..o + 3].copy_from_slice(&v);
                    }
                }
            }
        }
        write_rgba_bmp(&results_dir().join(format!("lit_m1_sweep_ambient_{ambient}.bmp")), &sheet, 3 * cw, 3 * ch);
    }
    let g = &rig.g;
    g.wait_idle().unwrap();
    sub.destroy(g);
    timer.destroy(g);
    bindings.destroy(g);
    dv.destroy(g);
    rig.alloc.free_image(g, out);
    rig.alloc.free(g, rb);
    tables.free_now(g, &mut rig.alloc);
    gm.free_now(g, &mut rig.alloc);
    assert_clean(&rig.g);
    // Warnings too: a sampled-image format mismatch (undefined loads) is only a warning, and it
    // occurred in the first 2C-2 run while every pixel still matched.
    assert_eq!(rig.g.validation_counts().1, 0, "validation warnings in the debug-view path");
    rig.finish();
    assert!(failed.is_empty(), "debug views that do not show the stored values: {failed:?}");
}
