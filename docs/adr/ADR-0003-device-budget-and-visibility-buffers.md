# ADR 0003 — Device memory budget and primary-visibility buffers

Status: accepted (budget: user decision; buffer formats: delegated to the model). Amended 2026-09-23: the image-identity rule of decision 3 is corrected (Amendment 1). Amended 2026-09-24: the granularity default (Amendment 2).
Date: 2026-09-23
Decision authority:
- **Budget:** the user, "gpu memory budget sure max is 3.5 so make sure it fits there".
- **Buffers:** delegated, "screne buffers i will leave that up to you".
- **Granularity:** the user asked for it to be a tested parameter ("cant we make a slider for it"). That is recorded here as a contract, not as a chosen value.

Related: [Phase 2 proposal](../changes/2026-09-23-phase2-proposal.md), [Phase 1 perf and Phase 2 prep](../changes/2026-09-23-phase1-perf-and-phase2-prep.md), P-001.

## Context and actual constraint

- P-001 allows at most 3.5 GB of VRAM on the RTX 3050 Laptop (4 GB).
- Phase 0 measured a driver heap budget (`VK_EXT_memory_budget`) of 3,367.7 MiB, which is about 3.53 × 10⁹ bytes. On this laptop that is just *above* the 3.5 GB cap.
- Phase 2 needs:
  - a device budget before its first allocation
  - G-buffer formats before raster (2C)
  - an ID scheme that the raster and ray paths share, so they can be compared exactly (2D)

## Decision

1. **Device budget** (`memory::device_budget`):
   - Total = min(3.5 × 10⁹ bytes, driver heap budget) − 10% headroom. Decimal GB is used, the stricter reading.
   - On this laptop that is **3.15 × 10⁹ bytes (about 3,004 MiB)**.
   - Every device and staging allocation takes a `Ledger` grant first. A refused grant defers the affected publication: the previous snapshot stays visible, and the refusal is counted. It never evicts silently, and never overcommits.
   - No per-category split yet; the categories are reported. A split is added only when a measured contention needs one.
2. **Primary-visibility G-buffer** (rendered by raster in 2C, and reproduced by ray query in 2D):

   | Target | Format | Bytes/px | Content |
   |---|---|---|---|
   | depth | `D32_SFLOAT` | 4 | reverse-Z depth |
   | normal | `R16G16_SNORM` | 4 | octahedral unit normal; exact for today's axis-aligned faces, and ready for smooth voxels later |
   | material | `R16_UINT` | 2 | exact `MaterialId`; never filtered or blended |
   | surface id | `R32G32_UINT` | 8 | (instance index, primitive index): the same pair ray query returns (`InstanceCustomIndex`, `PrimitiveID`). It maps to (chunk, brick, quad) through the published snapshot's tables |

   - That is 18 bytes/px: 37.3 MB at 1920×1080, 66.4 MB at 2560×1440.
   - The snapshot version is per frame (a push constant), not per pixel.
   - Motion vectors and history buffers come with the temporal work (Phase 3+), and will be budgeted then.
3. **Granularity is a parameter, not a constant.**
   - Mesh merge extent: none, or greedy within a brick.
   - Acceleration region size: 1 brick, 2³ bricks, 1 chunk (4³ bricks), or 2³ chunks.
   - Both are runtime configuration, swept by a benchmark (the "slider").
   - ~~At LoD 0 the geometry is exact, so the rendered image **must be bit-identical across every setting**.~~ **Corrected, see Amendment 1:** every setting must show the same voxel faces. Checked per pixel, material, normal and resolved voxel face are exact, depth is within a declared tolerance, and crack pixels are zero. This is a correctness check, not an appearance tradeoff.
   - The sweep chooses the default by memory, BLAS/TLAS build time, trace time and edit-to-visible latency.
   - Appearance-versus-cost sliders apply to approximations, such as LoD distance, resolution scale and sample counts, and arrive with those features.

## Contracts and consequences

- The ID pair is the contract that makes raster-versus-ray comparison exact. The per-pixel rule is in Amendment 1: the resolved voxel face and material must match, and depth must be within a declared tolerance.
- The host/shader layouts of these targets and the per-instance tables get layout-agreement checks from `slangc` reflection when 2B/2C start. Mesh vertex layout is decided in 2A, as an amendment or ADR-0004.
- Costs accepted:
  - 18 bytes/px of G-buffer
  - 10% of the budget held back
  - a deferred publication is visible to the user as a delayed edit, and is reported
- Not decided here: new engine dependencies (`ash`, `winit`, `ash-window`) and the authorization of Phase 2 slices.

## Validation and revisiting

- Implemented and tested now: `memory::device_budget`. Unit tests cover the laptop value, a smaller driver budget, a larger GPU, a missing budget, and the ledger refusing one byte over.
- Revisit:
  - if a larger GPU tier is added (the cap then becomes per tier)
  - if the sweep shows a per-category split is needed
  - if a visibility buffer beats the thin G-buffer at matched quality

## Implementation status

- Budget: implemented.
- Buffers and sweep: specified, not implemented (2B–2D). The mesh layout is in [ADR-0004](ADR-0004-surface-mesh-layout.md).

## Amendment 1 (2026-09-23): what "exact at LoD 0" requires

**Trigger:** the user asked, "LoD 0 wont doing it bit exact prevent improvements?" Working through the 2A mesh layout ([ADR-0004](ADR-0004-surface-mesh-layout.md)) showed that the original wording was wrong in two ways.

**Correction 1: "bit-identical image" cannot hold as written.**
- The surface ID differs by construction: a merged quad has a different primitive index than the unit quads it replaces.
- Depth can differ by a rounding step (ulp) between different triangulations of the same plane.
- Greedy quads create T-junctions, so rasterization and ray-triangle tests can leave single-pixel cracks. Those are only watertight across shared edges.
- A literal bit-identity gate would therefore either block greedy merging, or invite rebaselining the image until the test passes. CLAUDE.md forbids the second.

**Correction 2: what the check is actually for.**
- It compares granularity settings with each other at the same LoD, not against a frozen golden image. It exists to catch meshing and acceleration-structure bugs.
- It does not limit improvements:
  - Lighting, shading, anti-aliasing and denoising change every setting equally.
  - LoD > 0, smoothing and other approximate representations are separate representations. Each gets its own reference and error budget, measured against LoD 0.
- LoD 0 stays the exact reference that approximations are measured against.

**Replacement contract.** For the same camera, snapshot and LoD 0, every mesh-merge × region setting, and the ray-query path, must agree per pixel:
- **material ID:** exact
- **normal:** exact (axis-aligned faces)
- **resolved voxel face:** exact. The surface ID is resolved through the snapshot tables to (voxel, face); the raw primitive index is not compared.
- **depth:** within a declared tolerance, set in 2C from the depth format and projection
- **crack pixels** (no surface where the unmerged mesh or voxel DDA shows one): zero
- **excluded:** pixels whose centre lies within the declared depth/position tolerance of a projected voxel edge. There the raster sample and the ray sample may resolve to adjacent voxels. The excluded count is reported, not hidden, and must stay a thin edge set, not whole regions.

The CPU side of this contract is already met in 2A: both merge modes cover the same unit faces, and ray–quad results agree exactly with the DDA.

## Amendment 2 (2026-09-24): the granularity default

- **Decision (user, 2026-09-24, accepting the 2E recommendation):** the default is **greedy merge within the brick, one acceleration region per chunk** (`--merge greedy --region chunk`).
  - **2³ chunks** (`--region 2x2x2_chunks`) is kept as the documented alternative for larger, less dense scenes. The user asked for both to stay options.
  - All four region sizes and both merge modes remain runtime settings.
- **Why:** in the 2E sweep ([record](../changes/2026-09-24-phase2e-edits.md)) trace time was a plateau (0.91–1.0 ms at 1080p for all 8 settings, within run-to-run noise). Greedy merge used about a quarter of the unmerged memory and had 40% cheaper raster. Chunk regions build in 3 ms and meet the edit budget with a wide margin.
- **Superseded rule:** "the smallest region whose trace time is within 5% of the best" gave no stable answer. It picked brick regions, which cost more memory and a slower full build, and did not edit faster.
  - For a future re-sweep: among settings on the trace plateau (within the measured noise) that fit the edit budget, prefer the least memory, then the cheaper full build.
- **Edit budget: accepted by the user on 2026-09-24** ("Confirming it", S-016): edit-to-visible p95 ≤ 50 ms for the 1-voxel and 8³ edits (3 frames at 60 Hz), ≤ 100 ms for the 32³ edit. Every 2E setting meets it with a wide margin.
- **Revisit:** for a denser scene or a higher edit rate, or when secondary rays (Phase 3) make trace time weigh more.

## Amendment 3 (2026-09-25, 4A): the emitter table joins the publication set

- **Authority:** S-024 (4A authorized); ADR-0005 Amendment 3 defines the emitters. Record: [4A](../changes/2026-09-25-phase4a-emitters.md).
- **Decision.** The emitter table (`light::emitters::EmitterTable`) is a derived product of the same snapshot as the meshes and the acceleration structure. It is built from the published region meshes, uploaded and **swapped with them**: a frame reads meshes, TLAS, region table and emitter table of one snapshot, never a mix.
- **Identity.** Each emitter carries (region key, quad index in the region, the region's snapshot). An edited region gets a new snapshot, so its emitters are new emitters; later reuse (4C, 4D) must treat an emitter id from another snapshot as gone.
- **Order.** The table is sorted by geometry (face, plane, u0, v0), so the same set of quads gives the same table whatever the region size, and the CPU reference and the GPU agree on every index.
- **Staleness.** The table records the snapshot it was built for; a reader that finds a different snapshot refuses it (a planted stale table must be caught).
- **Budget.** 80 B per emitter on the device, plus 16 B per material for emitted radiance, under `Category::GpuMaterial`. The table build is part of the edit-to-visible latency and must keep the S-016 edit budget.
- **Implementation status (2026-09-25):** the table is built from region meshes (`gpu::emitters::table`) and uploaded for the reference (`RefEmitters`); building it inside `GpuScene::update`, the swap with the meshes and the edit-latency measurement are 4A part 2.
