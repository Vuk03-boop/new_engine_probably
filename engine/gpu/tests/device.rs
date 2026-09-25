//! Phase 2B GPU integration tests. They need the Vulkan SDK (pinned `slangc`, validation layer)
//! and an RT-capable GPU; without them they fail (they are never skipped, so an unavailable GPU
//! can not pass as green). Run: `cargo test --release -j 2 -p gpu`.
//!
//! Every test that is not a planted-fault control ends with zero validation errors.

use ash::vk;
use derived::{extract_world, BrickMesh, Config, Merge, Pipeline};
use gpu::alloc::{Allocator, Buffer, Kind};
use gpu::decode::{host_layout, Decoder, REFLECTION};
use gpu::layout::{build_regions, RegionKey, RegionMesh, RegionSize};
use gpu::mesh::GpuMeshes;
use gpu::staging::{download, Uploader, DEFAULT_RING_BYTES};
use gpu::submit::Submitter;
use gpu::{FrameReaders, Gpu, GpuError, Retirement, Timeline};
use memory::{Budget, Category};
use std::collections::BTreeMap;
use world::{scene, BrickKey, World};

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

fn street_meshes(merge: Merge) -> (World, Vec<(BrickKey, BrickMesh)>) {
    let (w, _) = scene::street_block();
    let m = w.bricks().map(|(k, _)| (k, extract_world(&w, k, merge).unwrap())).collect();
    (w, m)
}

fn regions(meshes: &[(BrickKey, BrickMesh)], size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
    build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
}

#[test]
fn device_budget_follows_adr_0003() {
    let g = gpu();
    let driver = g.info.driver_heap_budget;
    let total = g.device_budget().total.unwrap();
    let ceiling = driver.map_or(memory::P001_DEVICE_CAP_BYTES, |d| d.min(memory::P001_DEVICE_CAP_BYTES));
    assert_eq!(total, ceiling - ceiling / 10);
    eprintln!("device {:?}, driver device-local budget {:?} B, engine budget {total} B", g.info.name, driver);
    assert_clean(&g);
}

#[test]
fn mesh_uploads_read_back_bit_exact_for_every_setting() {
    let g = gpu();
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    for merge in Merge::ALL {
        let (_, meshes) = street_meshes(merge);
        for size in RegionSize::ALL {
            let rs = regions(&meshes, size);
            let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, size, &rs).unwrap();
            let ranges: Vec<(&Buffer, u64, u64)> = gm.regions.values().map(|r| (&r.buffer, 0, r.sections.total)).collect();
            let back = download(&g, &mut alloc, &mut tl, &ranges).unwrap();
            let mut mismatched = 0;
            for ((k, r), got) in rs.iter().zip(&back) {
                let (_, want) = r.image(gm.align);
                if &want != got {
                    mismatched += 1;
                    eprintln!("{k:?}: readback differs");
                }
            }
            assert_eq!(mismatched, 0, "merge {} region {}", merge.name(), size.name());
            let acc = alloc.ledger().account(Category::GpuMesh);
            assert!(acc.usage.live >= gm.device_bytes() && acc.usage.reserved >= acc.usage.live);
            eprintln!(
                "merge {} region {}: {} regions, {} quads, {} triangles, {} B images, ledger GpuMesh live {} reserved {}, blocks {}",
                merge.name(),
                size.name(),
                gm.regions.len(),
                gm.quad_count(),
                gm.triangle_count(),
                gm.device_bytes(),
                acc.usage.live,
                acc.usage.reserved,
                alloc.stats().blocks_live
            );
            g.wait_idle().unwrap();
            gm.free_now(&g, &mut alloc);
            assert_eq!(alloc.ledger().account(Category::GpuMesh).usage.reserved, 0, "all mesh blocks returned");
        }
    }
    up.destroy(&g, &mut alloc);
    assert_eq!(alloc.ledger().outstanding(), 0, "no grants left");
    assert_eq!(alloc.destroy(&g), 0, "no buffers leaked");
    tl.destroy(&g);
    assert_clean(&g);
}

fn expect_decoded(rs: &BTreeMap<RegionKey, RegionMesh>, decoded: &[gpu::decode::Decoded]) -> Vec<String> {
    let mut errors = Vec::new();
    for ((k, r), d) in rs.iter().zip(decoded) {
        assert_eq!((d.quads.len(), d.triangles.len()), (r.quads.len(), r.indices.len()));
        for (i, (q, got)) in r.quads.iter().zip(&d.quads).enumerate() {
            let want = (q.material as u32, q.face as u32, q.plane as u32, q.u0 as u32, q.v0 as u32, q.u1 as u32, q.v1 as u32, 0);
            let g = (got.material, got.face, got.plane, got.u0, got.v0, got.u1, got.v1, got.pad);
            if want != g && errors.len() < 10 {
                errors.push(format!("{k:?} quad record {i}: GPU {g:?}, CPU {want:?}"));
            }
        }
        for (t, c) in d.triangles.iter().enumerate() {
            if (c.quad, c.ok) != (r.tri_quad[t], gpu::decode::TRI_OK) && errors.len() < 10 {
                errors.push(format!("{k:?} triangle {t}: GPU quad {} ok {:#x}, CPU quad {}", c.quad, c.ok, r.tri_quad[t]));
            }
        }
    }
    errors
}

#[test]
fn gpu_decode_agrees_with_the_host_layout() {
    let g = gpu();
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let dec = Decoder::new(&g).unwrap();
    for merge in Merge::ALL {
        let (_, meshes) = street_meshes(merge);
        for size in [RegionSize::Brick, RegionSize::Chunks2] {
            let rs = regions(&meshes, size);
            let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, size, &rs).unwrap();
            let decoded = dec.run(&g, &mut alloc, &mut tl, &gm).unwrap();
            let errors = expect_decoded(&rs, &decoded);
            assert!(errors.is_empty(), "merge {} region {}: {errors:?}", merge.name(), size.name());
            eprintln!("merge {} region {}: {} quads decoded and {} triangles checked on the GPU, all equal", merge.name(), size.name(), gm.quad_count(), gm.triangle_count());
            gm.free_now(&g, &mut alloc);
        }
    }
    dec.destroy(&g);
    up.destroy(&g, &mut alloc);
    alloc.destroy(&g);
    tl.destroy(&g);
    assert_clean(&g);
}

#[test]
fn negative_controls_layout_mismatches_are_reported() {
    // Reflection side: a host declaration that disagrees with the shader is refused.
    let mut wrong = host_layout();
    if let gpu::reflect::Param::Descriptor { element, .. } = &mut wrong[4] {
        element.swap(1, 2); // face and plane swapped
    }
    let e = gpu::reflect::check(REFLECTION, &wrong).unwrap_err();
    assert!(e.iter().any(|m| m.contains("decoded")), "{e:?}");
    let e = gpu::reflect::check(REFLECTION, &host_layout()[..6]).unwrap_err();
    assert!(e.iter().any(|m| m.contains("params") && m.contains("not declared")), "undeclared push constants: {e:?}");
    let g = gpu();
    let moved = REFLECTION.replacen("\"index\": 2", "\"index\": 5", 1);
    assert!(matches!(Decoder::with_reflection(&g, &moved), Err(GpuError::Layout(_))), "a moved binding stops pipeline creation");

    // Execution side: corrupted bytes on the device are caught by the GPU decode.
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let dec = Decoder::new(&g).unwrap();
    let (_, meshes) = street_meshes(Merge::Greedy);
    let rs = regions(&meshes, RegionSize::Chunk);
    let (gm, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &rs).unwrap();
    let (k, r) = gm.regions.iter().next().unwrap();
    // Vertex 1 (on quad 0's outline) moved one voxel off quad 0's plane; the record of quad 1
    // packed with u0 and v0 swapped.
    let mut v = rs[k].vertices[1];
    v[(rs[k].quads[0].face / 2) as usize] += 2;
    let bad_vertex: Vec<u8> = [v[0] as u32, v[1] as u32, v[2] as u32, 2].iter().flat_map(|&x| gpu::layout::f16_of_halves(x).to_le_bytes()).collect();
    up.upload(&g, &mut tl, &r.buffer, r.sections.vertices + 8, &bad_vertex).unwrap();
    let mut q1 = rs[k].quads[1];
    std::mem::swap(&mut q1.u0, &mut q1.v0);
    let swapped = if q1.u0 == q1.v0 { q1.pack() ^ (1 << gpu::layout::U0_SHIFT) } else { q1.pack() };
    up.upload(&g, &mut tl, &r.buffer, r.sections.quads + 8, &swapped.to_le_bytes()).unwrap();
    up.flush(&g, &mut tl).unwrap();
    let decoded = dec.run(&g, &mut alloc, &mut tl, &gm).unwrap();
    let errors = expect_decoded(&rs, &decoded);
    assert!(errors.iter().any(|e| e.contains("triangle") && e.contains("CPU quad 0")), "corrupt vertex caught: {errors:?}");
    assert!(errors.iter().any(|e| e.contains("quad record 1")), "corrupt record caught: {errors:?}");
    gm.free_now(&g, &mut alloc);
    dec.destroy(&g);
    up.destroy(&g, &mut alloc);
    alloc.destroy(&g);
    tl.destroy(&g);
    assert_clean(&g);
}

#[test]
fn a_refused_grant_changes_nothing_and_old_meshes_stay_valid() {
    let g = gpu();
    // Budget, sized from the greedy images (they grew with the watertight layout): the staging
    // ring, enough 4 MiB mesh blocks for the greedy meshes plus one, and room for the final
    // readback. The unmerged meshes (~37 MB) cannot fit in what is left.
    let block = 4 << 20;
    let (_, greedy) = street_meshes(Merge::Greedy);
    let small = regions(&greedy, RegionSize::Chunk);
    let images: u64 = small.values().map(|r| r.sections(gpu::mesh::section_align(&g)).total).sum();
    let mesh_room = (images.div_ceil(block) + 1) * block;
    let readback = images.next_multiple_of(1 << 20).max(block);
    let limit = DEFAULT_RING_BYTES + mesh_room + readback;
    let mut alloc = Allocator::with_block_bytes(Budget::unlimited().with_total(limit), block);
    let mut tl = Timeline::new(&g).unwrap();
    let mut up = Uploader::new(&g, &mut alloc, DEFAULT_RING_BYTES).unwrap();
    let (old, _) = GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &small).unwrap();
    let before = (alloc.ledger().total_reserved(), alloc.ledger().outstanding(), alloc.stats().blocks_live, alloc.stats().buffers_live);
    let refusals_before = alloc.ledger().account(Category::GpuMesh).refusals;

    let (_, unmerged) = street_meshes(Merge::None);
    let big = regions(&unmerged, RegionSize::Chunk); // ~37 MB: cannot fit
    match GpuMeshes::upload(&g, &mut alloc, &mut up, &mut tl, RegionSize::Chunk, &big) {
        Err(GpuError::OverBudget(_)) => {}
        Err(e) => panic!("expected a budget refusal, got {e:?}"),
        Ok(_) => panic!("the unmerged meshes must not fit in {mesh_room} B of mesh room"),
    }
    let after = (alloc.ledger().total_reserved(), alloc.ledger().outstanding(), alloc.stats().blocks_live, alloc.stats().buffers_live);
    assert_eq!(after, before, "(reserved, grants, blocks, buffers) unchanged by the refusal");
    assert!(alloc.ledger().account(Category::GpuMesh).refusals > refusals_before, "refusal counted");
    assert!(alloc.ledger().total_high_water() <= limit, "never over budget");

    // The previous meshes are still intact on the device.
    let ranges: Vec<_> = old.regions.values().map(|r| (&r.buffer, 0, r.sections.total)).collect();
    let back = download(&g, &mut alloc, &mut tl, &ranges).unwrap();
    for ((_, r), got) in small.iter().zip(&back) {
        assert_eq!(&r.image(old.align).1, got);
    }
    old.free_now(&g, &mut alloc);
    up.destroy(&g, &mut alloc);
    alloc.destroy(&g);
    tl.destroy(&g);
    assert_clean(&g);
}

/// Submits a fill of `dst` with `pattern` that waits for `gate` to reach 1. Returns its value.
fn gated_fill(g: &Gpu, sub: &mut Submitter, tl: &mut Timeline, gate: &Timeline, dst: &Buffer, pattern: u32) -> u64 {
    let cmd = sub.begin(g, tl).unwrap();
    unsafe { g.device.cmd_fill_buffer(cmd, dst.buffer, 0, vk::WHOLE_SIZE, pattern) };
    gpu::submit::all_to_host(g, cmd);
    sub.submit(g, tl, cmd, &[(gate.semaphore, 1)]).unwrap()
}

/// Returns true if the in-flight GPU write landed in `later`, i.e. memory was reused too early.
fn retirement_scenario(g: &Gpu, free_early: bool) -> bool {
    let mut alloc = Allocator::new(g.device_budget());
    let mut tl = Timeline::new(g).unwrap();
    let gate = Timeline::new(g).unwrap();
    let mut sub = Submitter::new(g).unwrap();
    let usage = vk::BufferUsageFlags::TRANSFER_DST;
    // `keeper` keeps the block alive so the early free cannot release device memory under the GPU.
    let keeper = alloc.create_buffer(g, 4096, usage, Category::GpuMesh, Kind::Host).unwrap();
    let old = alloc.create_buffer(g, 4096, usage, Category::GpuMesh, Kind::Host).unwrap();
    let old_range = old.range();
    let v = gated_fill(g, &mut sub, &mut tl, &gate, &old, 0xDEAD_BEEF);
    let mut retire = Retirement::default();
    if free_early {
        alloc.free(g, old);
    } else {
        retire.push(v, old);
        assert!(retire.collect(tl.completed(g).unwrap()).is_empty(), "the GPU has not run yet: nothing retires");
    }
    let mut later = alloc.create_buffer(g, 4096, usage, Category::GpuMesh, Kind::Host).unwrap();
    assert_eq!(later.range() == old_range, free_early, "first-fit reuses the range only if it was freed");
    later.mapped().unwrap().fill(0x11);
    gate.signal_from_host(g, 1).unwrap();
    assert!(tl.wait(g, v, 5_000_000_000).unwrap(), "gated submission completes");
    let corrupted = later.mapped_ref().unwrap().iter().any(|&b| b != 0x11);
    for b in retire.collect(tl.completed(g).unwrap()) {
        alloc.free(g, b);
    }
    alloc.free(g, later);
    alloc.free(g, keeper);
    g.wait_idle().unwrap();
    sub.destroy(g);
    alloc.destroy(g);
    tl.destroy(g);
    gate.destroy(g);
    corrupted
}

#[test]
fn retirement_waits_for_the_gpu() {
    let g = gpu();
    assert!(!retirement_scenario(&g, false), "correct retirement: the in-flight write stays in the old buffer");
    assert_clean(&g);
}

#[test]
fn negative_control_early_free_is_caught() {
    let g = gpu();
    assert!(retirement_scenario(&g, true), "freeing before the GPU is done lets its write corrupt the reused memory");
    let (errors, _) = g.validation_counts();
    eprintln!("planted early free: corruption detected; validation reported {errors} error(s)");
}

#[test]
fn frame_readers_keep_1c_snapshots_alive_until_the_gpu_is_done() {
    let g = gpu();
    let (mut w, _) = scene::street_block();
    let mut p = Pipeline::new(Config::default());
    p.mark_all(&w);
    loop {
        let jobs = p.dispatch(&w);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            p.complete(&w, j.run());
        }
    }
    p.try_publish(&w).expect("initial publication");
    let (key, _) = w.bricks().next().unwrap();
    let old_handle = p.current().handle(key).unwrap();

    let mut tl = Timeline::new(&g).unwrap();
    let gate = Timeline::new(&g).unwrap();
    let mut sub = Submitter::new(&g).unwrap();
    let mut frames = FrameReaders::default();
    let cmd = sub.begin(&g, &tl).unwrap();
    let v = sub.submit(&g, &mut tl, cmd, &[(gate.semaphore, 1)]).unwrap();
    frames.hold(v, p.acquire());

    // Edit the brick and publish its new mesh while the frame is still in flight.
    let o = key.origin();
    let voxel = w.bricks().next().unwrap().1.voxels().next().unwrap().0;
    let (x, y, z) = voxel.xyz();
    let at = world::VoxelCoord::new(o.x + x, o.y + y, o.z + z);
    w.set(at, None).unwrap();
    p.notify_edits(&[at]);
    loop {
        let jobs = p.dispatch(&w);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            p.complete(&w, j.run());
        }
    }
    p.try_publish(&w).expect("edit publication");
    assert_ne!(p.current().handle(key), Some(old_handle));

    assert_eq!(frames.release_completed(&mut p, tl.completed(&g).unwrap()), 0, "frame not done");
    assert!(p.resolve(old_handle).is_some(), "the in-flight frame's mesh is still allocated");
    assert!(p.retiring_len() > 0);

    gate.signal_from_host(&g, 1).unwrap();
    assert!(tl.wait(&g, v, 5_000_000_000).unwrap());
    assert_eq!(frames.release_completed(&mut p, tl.completed(&g).unwrap()), 1);
    assert!(p.resolve(old_handle).is_none(), "freed once the GPU finished the frame");
    assert_eq!(p.retiring_len(), 0);

    g.wait_idle().unwrap();
    sub.destroy(&g);
    tl.destroy(&g);
    gate.destroy(&g);
    assert_clean(&g);
}

/// 2C/2D prep: the formats ADR-0003 (G-buffer) and ADR-0004 (vertices) assume must be supported
/// by the selected device. Storage-image support is reported (2D may write its G-buffer from compute).
#[test]
fn formats_assumed_by_adr_0003_and_0004_are_supported() {
    let g = gpu();
    let props = |f: vk::Format| unsafe { g.instance.get_physical_device_format_properties(g.physical, f) };
    let color = vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    let storage = vk::FormatFeatureFlags::STORAGE_IMAGE;
    let mut missing = Vec::new();
    for (name, f, need) in [
        ("depth D32_SFLOAT", vk::Format::D32_SFLOAT, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
        ("normal R16G16_SNORM", vk::Format::R16G16_SNORM, color),
        ("material R16_UINT", vk::Format::R16_UINT, color),
        ("surface id R32G32_UINT", vk::Format::R32G32_UINT, color),
    ] {
        let p = props(f).optimal_tiling_features;
        eprintln!("{name}: attachment {} storage image {}", p.contains(need), p.contains(storage));
        if !p.contains(need) {
            missing.push(name);
        }
    }
    let v = props(vk::Format::R16G16B16A16_SFLOAT).buffer_features;
    let as_vertex = v.contains(vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR);
    eprintln!("vertex R16G16B16A16_SFLOAT: vertex buffer {} acceleration-structure vertex {as_vertex}", v.contains(vk::FormatFeatureFlags::VERTEX_BUFFER));
    if !v.contains(vk::FormatFeatureFlags::VERTEX_BUFFER) {
        missing.push("vertex buffer R16G16B16A16_SFLOAT");
    }
    if g.ray_tracing() && !as_vertex {
        missing.push("acceleration-structure vertex R16G16B16A16_SFLOAT");
    }
    assert!(missing.is_empty(), "unsupported: {missing:?}");
    assert_clean(&g);
}
