# Change: Phase 2C-1 — headless raster G-buffer and the per-pixel equivalence test

Status: **implemented. The greedy stop was resolved by [ADR-0004 Amendment 1](../adr/ADR-0004-surface-mesh-layout.md) (see the [watertight greedy record](2026-09-23-phase2c1-watertight-greedy.md)), and the gate now passes for every setting.** What follows is the original 2C-1 run: the gate was stopped on greedy meshes. Unmerged meshes pass every setting. Greedy meshes leave 1 real T-junction crack pixel, so the [2C proposal](2026-09-23-phase2c-proposal.md)'s stop condition applies. The test is left failing; nothing was rebaselined. 2C-2 (the window) is not started.
Date and baseline: 2026-09-23, after 2B. No Git repository.
Authorization (S-009), user: "continue", after "no 2c simply prep for it i will compact then we shall do it". Taken as authorizing 2C with the six recommended decisions. Decision 3 was changed; see below.

## What was built (`engine/gpu`)

**Shader and build:**
- `shaders/raster.slang`: a vertex stage (camera-relative, reverse-Z infinite projection) and a fragment stage. The fragment stage writes:
  - an octahedral normal
  - the exact material, from the quad table by `PrimitiveID ÷ 2`
  - the surface id (region index, `PrimitiveID`)
- `build.rs` compiles several entry points per file. The fragment `SV_PrimitiveID` makes Slang declare SPIR-V `Geometry` (capability 2). The allow-list stopped the build until `geometryShader` was enabled on the device and 2 was added to the list.

**Device:**
- `context`: device selection now also requires `geometryShader` and `dynamicRendering`, and enables both.

**Memory and reflection:**
- `alloc`: `create_image` / `free_image`. Images use the same 64 MiB blocks and ledger as buffers, but never share a block with them, so `bufferImageGranularity` cannot apply.
- `reflect::varyings`: checks vertex-input and fragment-output locations and types from `slangc` reflection, alongside the existing binding and push-constant check.

**`raster` module:**
- `Camera` (look-at; `ray(x, y)` gives the matching CPU ray through the pixel centre)
- `Targets` (the four ADR-0003 images, all-or-nothing)
- `Raster` (reflection checked before any Vulkan object is created; dynamic rendering; one indexed draw per region)
- `Faults`, for the planted controls

**`equivalence` module:** the ADR-0003 Amendment 1 check against `world::reference::trace`, with the tolerances declared before measuring:
- **Depth:** |t_gpu − t_ref|·|d| ≤ 1e-4·t_ref·|d| + 1e-3 voxel.
- **Near-edge:** four probe rays at ±1/16 px. If any resolves to a different (voxel, face), the pixel is near an edge. Such pixels are excluded from the exact checks and counted.
- **Crack:** checked on every pixel whose five reference rays hit the same face plane, edge or not. A crack is nothing drawn, or only a different, farther surface.

**Decision 3 changed.** `memory::Category::GpuTemporal` is already documented as "Frame targets and temporal histories", so the G-buffer uses it. No new category and no public-interface change were needed. The proposal's claim that `GpuTemporal` "is for history" was wrong.

## Results (RTX 3050, validation on)

Command: `cargo test --release -j 2 -p gpu --test raster -- --test-threads=1 --nocapture`. Exit 101, 2 of 3 tests pass. Log: `engine/results/test_gpu_2c1_rtx.log`.

**Matrix:** 4 cameras × 2 merge modes × 4 region sizes, 640×360, 5 reference rays per pixel. Across all 32 settings:
- material, normal, resolved face and depth mismatches on checked pixels: **0**
- extra surfaces and bad ids: **0**
- validation errors: **0**

| Camera | Reference hits | Checked exactly | Near-edge | Cracks: none / greedy | Max depth err/tol: none / greedy | Plane-depth outliers |
|---|---|---|---|---|---|---|
| street_view (the scene's view) | 173,137 | 142,318 | 17.8% | 0 / 0 | 0.51 / 0.62 | 0 |
| chunk_corner (eye on a chunk corner) | 180,972 | 156,594 | 13.5% | 0 / 0 | 0.52 / 0.59 | 0 |
| overhead | 102,337 | 75,032 | 26.7% | 0 / **1** | 0.41 / 0.37 | 0 |
| grazing (low, down the street) | 147,065 | 123,704 | 15.9% | 0 / 0 | 0.88 / 0.95 | 268 / 258 |

- The counts are identical across the 4 region sizes, so region size never changed a pixel.
- **The greedy crack:** pixel (331, 279) of `overhead`. The reference hits the +y face of voxel (83, −1, 64), the ground at z = 64, which is a brick boundary. The greedy mesh draws nothing there, at every region size; the unmerged mesh covers it.
  - Rasterization is watertight only across shared edges. Greedy quads meet at T-junctions, and after projection and sub-pixel snapping, two collinear edges can leave a gap of a fraction of a pixel.
  - This is the ADR-0004 risk, now observed: 1 pixel in about 600k greedy hit pixels.
- **Plane-depth outliers** (the diagnosis of the first run): at the grazing camera, the GPU draws the reference's own plane, but up to about 0.1 voxel farther along the ray than the tolerance allows.
  - All of these are near-edge pixels; checked pixels stay within tolerance, at up to 0.95 of it.
  - Cause: at grazing incidence the depth along the ray changes about 65 voxels per pixel row. Sub-pixel vertex snapping therefore moves depth a lot while moving the hit point on the plane very little (about 0.001 voxel).
  - The first run counted these as cracks; the split classifier (reported, never hidden) is in `gpu::equivalence`. Log of the first run: `engine/results/test_gpu_2c1_rtx_attempt1.log`.
- **Reference cost:** 1.9–5.3 s per camera on 2 threads (1.15 M rays).

**Planted controls** (street_view, greedy, chunk regions; the unfaulted baseline passes first):

| Fault | Caught as |
|---|---|
| Reversed winding (front face clockwise) | 172,048 cracks plus normal, face and depth mismatches. The image is not empty: the back faces of far walls are drawn instead. The proposal said it "must go empty", which was wrong. |
| Region offset off by one voxel in x | 4,613 face and 4,896 depth mismatches, and 2,446 cracks |
| Material read from the wrong primitive (quad tables rotated by one record on the device) | 25,933 material mismatches. After restoring the tables, the render passes again. |
| Dropped region (the busiest one, 8,866 px) | 8,866 cracks, exactly its pixels |

**Targets:**
- A 1080p G-buffer is 37,324,800 B of texels. The driver requires 39,813,120 B, and ledger `GpuTemporal` live is exactly that.
- Freeing returns every block.
- An 8 MiB budget refuses the targets with `OverBudget`, leaving nothing allocated and counting the refusal.

**Other checks:**
- `cargo test --release -j 2 -p gpu --lib --test device -- --test-threads=1 --nocapture`: exit 0, 11 unit tests and 9 device tests (the device requirements changed). The only validation error is the planted early-free control. Log: `engine/results/test_gpu_2c1_rtx_device.log`.
- `cargo clippy --release -j 2 --workspace --all-targets`: exit 0, the same 6 pre-existing lints (`results/clippy.log`).
- Pure crates: unchanged, not rerun (99 tests at the last run).

**Visual output** (not evidence; the counts are): `engine/results/raster_2c1_{street_view,chunk_corner,overhead,grazing}.bmp`, greedy meshes with chunk regions, material colour shaded by face axis.

## Stop: the greedy T-junction crack

Per the proposal's stop condition, the cracks are not accepted. Options for the user:

1. **Unmerged meshes for rendering now** (recommended to unblock 2C-2).
   - Crack-free on every camera and region size here.
   - Cost: 520,540 quads, 33.3 MB, against 0.68 MB greedy. That is about 1% of the 3.15 × 10⁹ B budget, and about 1 M triangles, which is ordinary for this GPU.
   - Greedy stays in the code, and stays swept as a slider, but is not used for display until fixed.
2. **T-junction-free greedy:**
   - Split quad edges at every vertex of coplanar neighbours, so every edge is shared.
   - Quads then become polygons with a varying triangle count, so `primitive ÷ 2 = quad` breaks. A per-triangle quad index (4 B per triangle) would replace it. This is an ADR-0004 amendment.
   - The ray-query BLAS (2D) has the same T-junction exposure, and this fix covers both.
3. **Expanding quads by an epsilon in the vertex shader:**
   - Raster only. The overlap is within the near-edge set.
   - It does not fix the BLAS, and the half-float vertices can't hold the offset (the half-float step at 64 is 1/16 voxel). Not recommended.

The measured crack rate (1 in about 600k pixels) does not change the rule; ADR-0004 already says accepting cracks is not an option.

## Risks seen

- Depth error on checked pixels reaches 0.95 of the tolerance at grazing incidence. Higher resolutions or lower cameras may breach it. The raster error there is screen-space (sub-pixel snapping), which a ray-distance tolerance models poorly. Per the rule, a breach is diagnosed first; the tolerance is not raised to fit. Ray query (2D) will not have this error.
- Not tested: a planted reflection mismatch for the raster modules (the mechanism is the one tested in 2B for `decode`). Also not tested: the debug build of these tests.

## NOT RUN

- 2C-2: window, swapchain, frame loop, debug views, free-fly camera, GPU timing.
- 1080p equivalence runs, and cold-process runs.

## Closeout

- **Docs:**
  - the [2C proposal](2026-09-23-phase2c-proposal.md): status, decision 3, the winding correction
  - [ADR-0004](../adr/ADR-0004-surface-mesh-layout.md): the crack measured
  - [DECISIONS](../DECISIONS.md) (S-009), `engine/README.md`, `docs/NOW.md`
- **Revert:** delete these files and changes:
  - `gpu/shaders/raster.slang`, `gpu/src/raster.rs`, `gpu/src/equivalence.rs`, `gpu/tests/raster.rs`
  - the image functions in `alloc.rs`, `varyings` in `reflect.rs`
  - the feature requirements in `context.rs`
  - the multi-entry `SHADERS` table and capability 2 in `build.rs`
