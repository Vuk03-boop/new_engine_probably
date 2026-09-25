# Change (planned): Phase 2 — coherent visibility and the first visual slice

Status: **in progress.**
- All decisions are settled: 1 is accepted (A-006), 2–4 are in [ADR-0003](../adr/ADR-0003-device-budget-and-visibility-buffers.md), and 5 is 2A authorized (S-006).
- **2A is done:** [record](2026-09-23-phase2a-surface-extraction.md), [ADR-0004](../adr/ADR-0004-surface-mesh-layout.md).
- **2B passed on the RTX 3050:** [record](2026-09-23-phase2b-gpu-bring-up.md).
- **2C passed:** [2C proposal](2026-09-23-phase2c-proposal.md), [2C-1](2026-09-23-phase2c1-raster-equivalence.md), [2C-2](2026-09-23-phase2c2-window-frame-loop.md). M1 was accepted on 2026-09-24.
- **2D passed on the RTX 3050 (2026-09-24):** [record](2026-09-24-phase2d-ray-query.md).
- **2E done (2026-09-24, S-014):** [record](2026-09-24-phase2e-edits.md). Edits reach both paths in every setting; the default rule of §5 gave no stable answer, so the granularity default awaits the user, with the edit budget.
Prep done: [Phase 1 perf and Phase 2 prep](2026-09-23-phase1-perf-and-phase2-prep.md).
Date: 2026-09-23. Builds on Phase 1 (1A–1E, all complete).

## What Phase 2 is for

- **Roadmap** ([BUILD-ROADMAP](../../BUILD-ROADMAP.md) Phase 2):
  - exposed-surface extraction and material-aware merging
  - a simple LoD/residency policy
  - opaque raster visibility, plus a matching ray-query representation
  - extra mesh/BLAS/TLAS memory tracked explicitly
- **Gate:**
  - Raster and ray paths see the committed scene.
  - Silhouettes, material behaviour and chunk boundaries agree within the declared geometric convention.
  - Edit latency and acceleration rebuild cost fit the chosen budget.
- **Architecture fork:** if high-edit workloads make surface extraction plus RT structures untenable, compare a direct-voxel alternative *here*, before lighting depends on the losing path.

## Proposed slices (in order)

1. **2A — Surface extraction (CPU only, no new dependencies).** *Done 2026-09-23.*
   - Replaces the 1C stand-in with the real derived product: per-brick quads for exposed voxel faces, greedy-merged within the brick per (face direction, material).
   - Every quad carries its exact `MaterialId` and brick key (guide/provenance record); merging never crosses materials.
   - Runs through the existing 1C pipeline (jobs, staleness, publication, retirement) and reports its size to 1D.
   - **Tests:**
     - *Face coverage:* every exposed voxel face is covered exactly once, and nothing else is.
     - *Ray agreement:* a CPU ray–quad reference agrees with the 1E voxel DDA on hit, distance, face and material, under the half-open convention, including chunk-boundary rays.
     - *Negative controls* for a dropped face, a cross-material merge, and a missed neighbour-dirty.
2. **2B — GPU bring-up inside `engine/`, headless.** *Done 2026-09-23 (passed on the RTX 3050).*
   - A `gpu` crate: instance and device, validation.
   - Device memory through the 1D `Ledger` with a device budget; an upload staging ring.
   - Fence-based retirement, replacing 1C's simulated reader tokens with real frames in flight.
   - Shader build with pinned `slangc`, plus host/shader layout agreement checks (sizes, offsets, flags) from `slangc` reflection.
   - Uploads 2A meshes and reads them back bit-exact.
3. **2C — Raster visibility.**
   - Renders the committed snapshot into a thin G-buffer: depth, normal, exact material ID (integer target), brick/quad ID, snapshot version.
   - Debug views: material, normal, brick, version/age.
   - Headless image tests first, then a window (`winit`) for viewing.
4. **2D — Ray-query representation.**
   - One BLAS per chunk, built from the same 2A meshes (the same committed LoD, per PROPOSITION §3), plus a TLAS.
   - A compute pass casts primary rays with ray query.
   - **Per-pixel raster-versus-ray comparison** of depth (declared tolerance), material ID (exact) and brick ID (exact), with chunk-boundary and silhouette cameras.
   - BLAS/TLAS memory and build scratch go into the ledger.
5. **2E — Edits end to end.**
   - Edit → 1B commit → 1C jobs → mesh publication → upload → BLAS rebuild of the touched chunks → atomic snapshot swap after the frames using the old one retire.
   - Measures: per-stage timings and edit-to-visible latency.
   - Publication shares unchanged entries instead of cloning the table, fixing the 1D observation of a 4.75 MB transient.
   - Runs the **granularity sweep** ("slider", ADR-0003): mesh merge (none / within brick) × acceleration region (1 brick, 2³ bricks, 1 chunk, 2³ chunks). It reports device memory per category, BLAS/TLAS build ms, trace ms, and edit-to-visible ms for the 1-voxel, 8³ and 32³ edits from `perf_phase1`.
     - Every setting must pass the per-pixel equivalence of ADR-0003 Amendment 1: resolved voxel face, material and normal exact, depth within tolerance, zero crack pixels. This replaces the earlier "identical image hash", which was wrong. Merged quads have different primitive IDs and create T-junctions.
     - The default is the smallest region whose trace time is within 5% of the best, and whose edit latency fits the budget.
   - Runs the **direct-voxel fork check**: if BLAS rebuild or extraction dominates edit latency at the chosen edit rate, prototype and compare the alternative.

LoD/residency stays at "everything resident, LoD 0" until a scene exceeds the budget; the street block is about 6 MB of world data. That trigger is recorded rather than built.

## Decisions (status 2026-09-23)

- Decisions 2–4 are settled in ADR-0003.
- Decision 1 was accepted by the user on 2026-09-23 ("You may add them"), with the dependencies to be added with the 2B `gpu` crate.
- Decision 5: the user authorized 2A ("for 2A sure").

- **2 (budget):** accepted. **Correction:** this proposal first said the driver figure is the real ceiling on this laptop. It is not: 3,367.7 MiB is about 3.53 × 10⁹ bytes, above the 3.5 GB cap, so the cap governs. The budget is 3.15 × 10⁹ bytes (about 3,004 MiB). The unit test in `memory::ledger` caught this.
- **3 (granularity):** turned into a parameter swept in 2E (above), as the user asked.
- **4 (buffers):** delegated. A thin G-buffer, 18 bytes/px, with an (instance, primitive) surface ID shared with ray query.

The original text follows.

1. **Engine dependencies for 2B/2C.**
   - Recommended: `ash =0.38.0` (already in ADR-0001), plus `winit =0.30.13` and `ash-window =0.13.0` (the versions proven by the Phase 0 window probe).
   - GPU memory uses our own simple pool sub-allocator behind the ledger, not VMA or `gpu-allocator`. The roadmap says to start with simple allocation.
2. **Device memory budget.**
   - Recommended: the total device budget is the smaller of P-001's 3.5 GB and the driver-reported `VK_EXT_memory_budget` heap budget, minus 10% headroom.
   - Phase 0 measured a driver budget of 3,367.7 MiB, so the driver figure is the real ceiling on this laptop (about 3.0 GiB after headroom).
   - A grant over budget is refused and the edit's publication is deferred: the old snapshot stays visible, and the refusal is counted.
3. **Mesh and BLAS granularity.**
   - Recommended: mesh per brick (keeps the 1C pipeline keyed as it is), BLAS per chunk (291 BLAS for the street block).
   - An edit rebuilds the touched chunks' BLAS.
   - Alternatives are a BLAS per brick (5,103 instances) or larger regions; measure in 2E before changing.
4. **Visibility output.**
   - Recommended: a thin G-buffer with an exact integer material ID and a brick/quad ID.
   - A pure visibility buffer is the alternative. It saves bandwidth but needs a material-resolve pass.
5. **Order and authorization.**
   - Recommended: authorize 2A now. It is CPU only, with no new dependencies, and it is the prerequisite for everything else.
   - Authorize 2B–2E once you approve decisions 1–2. Each slice ends with its own record, as in Phase 1.

The mesh/vertex layout and the G-buffer formats become host/shader interfaces. They get an ADR when 2B/2C starts (ADR-0003), including the layout-agreement checks CLAUDE.md requires.

## Excluded from Phase 2

- lighting (Phase 3)
- transparency and glass behaviour beyond opaque visibility
- LoD beyond level 0
- streaming
- DLSS/RR
- multi-queue overlap
