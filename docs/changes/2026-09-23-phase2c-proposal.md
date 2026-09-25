# Change (planned): Phase 2C — raster visibility and the first window

Status: **2C done.** 2C-1 passes since ADR-0004 Amendment 1 ([2C-1](2026-09-23-phase2c1-raster-equivalence.md), [watertight greedy](2026-09-23-phase2c1-watertight-greedy.md)); 2C-2 implemented ([2C-2](2026-09-23-phase2c2-window-frame-loop.md), S-011). Earlier status: 2C-1 stopped at the greedy crack. Authorized after the compact (S-009, user: "continue"), with the recommended decisions except decision 3.
Corrections from 2C-1:
- Decision 3 is not needed: `memory::Category::GpuTemporal` is already documented as "Frame targets and temporal histories".
- Reversed winding does not make the image empty. Back faces of far surfaces are drawn instead. The control is caught by the equivalence counts.
Date: 2026-09-23. Builds on 2A (meshes) and 2B (GPU layer, passed on the RTX 3050).
Parent plan: [Phase 2 proposal](2026-09-23-phase2-proposal.md). Contracts: [ADR-0003](../adr/ADR-0003-device-budget-and-visibility-buffers.md) (G-buffer, Amendment 1), [ADR-0004](../adr/ADR-0004-surface-mesh-layout.md) (mesh layout).

## Facts already established (prep evidence)

- **Formats:** the RTX 3050 supports every G-buffer format as an attachment and as a storage image: `D32_SFLOAT`, `R16G16_SNORM`, `R16_UINT`, `R32G32_UINT`. `R16G16B16A16_SFLOAT` is supported as a vertex buffer and as acceleration-structure input. Test: `formats_assumed_by_adr_0003_and_0004_are_supported`.
- **Crates:** `winit =0.30.13`, `ash-window =0.13.0` and `raw-window-handle 0.6.2` are in the local cargo cache (Phase 0). They are approved (A-006) but not yet in the engine.
- **Street block on the RTX:** 10,503 greedy quads (0.68 MB at chunk regions) or 520,540 unmerged (33.3 MB). The device budget is 3.15 × 10⁹ B.
- **CPU reference:** the voxel DDA runs at about 0.17–0.29 M rays/s on one thread. A 640×360 reference image is 230 k rays, a few seconds with threads. 1080p references are for occasional runs, not every test.

## Proposed slices

**2C-1: headless raster and the equivalence test** (first; no window).
- One graphics pipeline draws every region: vertices and indices from the region buffers, with a per-region push constant for the camera-relative offset. It writes the four ADR-0003 targets:
  - depth
  - octahedral normal
  - `MaterialId`, read from the quad table by `PrimitiveID ÷ 2`
  - (region index, `PrimitiveID`)
- Read the targets back and compare per pixel with a CPU reference, using `world::reference::trace` from the same camera through pixel centres. This applies the Amendment 1 contract:
  - material, normal and resolved voxel face exact
  - depth within the declared tolerance (below)
  - zero crack pixels
  - near-edge pixels counted and reported
- **Matrix:** both merge modes × the 4 region sizes, with street and chunk-boundary cameras. This is where the **T-junction crack risk** gets measured.
- **Planted controls:** reversed winding (the image must go empty under back-face culling); an off-by-one camera offset (depth/face mismatches); a material read from the wrong primitive (material mismatches); a dropped region (crack pixels).

**2C-2: window, frame loop and debug views.**
- `winit` window, swapchain (FIFO by default; MAILBOX as an option), 2 frames in flight on the 2B timeline.
- `FrameReaders` holds the 1C snapshot for each frame in flight.
- Debug views in a fullscreen pass: material colour (from `MaterialParams`), normal, region/brick hash, surface-ID hash, linear depth, snapshot version.
- A free-fly camera: enough to look around. The walking camera with collision is a later M1 step.
- GPU timestamps per pass, with frame-time p50/p99 logged. This is not yet the 60 fps gate.

## Decisions for the user at authorization (recommendation first)

1. **Order:** 2C-1 before 2C-2 (recommended). The equivalence test catches geometry and ID bugs before anything is judged by eye.
2. **Depth tolerance**, declared now, before any measurement, so it cannot drift to fit results. Distance along the pixel ray, reconstructed from `D32_SFLOAT` reverse-Z (infinite far plane), must satisfy |t_gpu − t_ref| ≤ 1 × 10⁻⁴ · t_ref + 1 × 10⁻³ voxel. That is about three orders above float32 resolution, and far below a voxel (1/16 m). A breach is a bug to diagnose, never a tolerance to raise.
3. **New memory category `GpuTargets`** for the G-buffer and swapchain-size images (37.3 MB at 1080p). None of the current categories fits; `GpuTemporal` is for history. This is a small public-interface change to `memory::Category`.
4. **Image support in `gpu::alloc`:** add `create_image` on the same blocks and ledger. Images and buffers share blocks only if the driver's granularity rules allow it; otherwise they use separate pools.
5. **Conventions** (recommended; checked by the planted-winding control):
   - world right-handed, Y up, voxel units
   - camera-relative rendering, with an integer camera origin subtracted per region
   - reverse-Z infinite perspective
   - Vulkan's Y-down handled by a negative viewport height
   - front face set to match ADR-0004's counter-clockwise-from-outside after that flip
   - back-face culling on
6. **Pixel-centre rule:** the reference ray goes through (x + 0.5, y + 0.5), the rasterizer's sample point, with no MSAA in 2C.

## Stop conditions

- Crack pixels > 0 in greedy mode → stop. Report the count and where the cracks are, and propose a crack-free triangulation or unmerged regions. Do not accept the cracks.
- Any exact field (material, normal, resolved face) differs outside the near-edge set → stop and diagnose. Never rebaseline.
- Validation errors → fix them before measuring anything.

## Not in 2C

- ray query and BLAS (2D)
- edits end to end (2E)
- lighting beyond the debug views
- the walking camera and collision, simple shading, and the 60 fps gate (M1 steps after 2C)
