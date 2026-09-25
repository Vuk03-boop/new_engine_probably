# ADR 0004 — Surface mesh layout and the surface-ID mapping

Status: accepted, **amended 2026-09-23 (Amendment 1: watertight triangulation, per-triangle quad index)**. The CPU half is implemented (2A). The GPU half is implemented in 2B and verified on the RTX 3050 (see the [2B record](../changes/2026-09-23-phase2b-gpu-bring-up.md)). Where decisions 2–4 below disagree with Amendment 1, the amendment governs.
Date: 2026-09-23
Decision authority: delegated within the approved 2A slice (user: "for 2A sure"). The Phase 2 plan named the mesh/vertex layout as the first 2A decision.
Related: [2A record](../changes/2026-09-23-phase2a-surface-extraction.md), [ADR-0003](ADR-0003-device-budget-and-visibility-buffers.md) (G-buffer, surface ID, granularity parameter).

## Context and actual constraint

- 2A turns voxels into quads. 2B uploads them, 2C rasterizes them and 2D builds BLAS from them. The layout crosses the host/shader boundary, so it needs one written contract.
- ADR-0003 fixes the surface ID as (instance index, primitive index), the pair ray query returns. The mesh layout must make that pair resolve to exactly one quad and material.
- Measured on the street block (`engine/results/perf_phase2a.jsonl`): 520,540 quads without merging and 10,503 with greedy merging. The most in one brick is 240 unmerged and 11 greedy.

## Options and tradeoffs

- **CPU record:** brick-local integer rectangles (chosen), or float vertices. Integers are exact and 8 bytes each, and a unit-face test can check them.
- **GPU vertex position:**
  - `R32G32B32_SFLOAT`: 12 B per vertex, the simplest option.
  - `R16G16B16A16_SFLOAT`: 8 B per vertex (chosen). Integers up to 2048 are exact in half floats, and a region is at most 64 voxels on a side (2³ chunks), so positions stay exact.
  - Both formats are on Vulkan's required list for acceleration-structure vertex buffers. 2B still queries support before relying on it.
- **Material per vertex or per primitive:** per primitive, through the quad table (chosen). A vertex attribute would be interpolated or need `flat`, and it duplicates data.
- **Index width:** 32-bit (chosen for now). 16-bit indices would limit a region to 16,384 quads; the granularity sweep can revisit this.

## Decision

1. **CPU quad** (`derived::mesh::Quad`, 8 bytes, checked at compile time):
   - Fields: `material: u16`, `face: u8` (axis × 2 + positive), `plane: u8` (0..=8), and `u0, v0, u1, v1: u8`.
   - Coordinates are brick-local. The quad covers the half-open rectangle [u0, u1) × [v0, v1), with u = (axis + 1) mod 3 and v = (axis + 2) mod 3.
   - Corners run counter-clockwise seen from outside. Triangles are (c0, c1, c2) and (c0, c2, c3).
   - Quads are ordered by face direction, then plane, then scan order.
   - Merge modes (the ADR-0003 slider): `None`, one quad per exposed voxel face; and `Greedy`, rectangles within one brick, face direction, plane and material. Merging never crosses a material, a plane or a brick.
2. **GPU layout, one set per acceleration region** (implemented in 2B; the region size is the other ADR-0003 slider):
   - **Vertices:** `R16G16B16A16_SFLOAT`, region-local voxel units, w = 1. Four vertices per quad, in the order given by `Quad::corners`.
   - **Indices:** `u32`, six per quad: (4q, 4q+1, 4q+2) and (4q, 4q+2, 4q+3).
   - **Quad table:** a storage buffer of 8-byte records, the quad re-encoded in region-local coordinates. Fixed in 2B at 54 bits: material [0,16), face [16,19), plane [19,26), u0 [26,33), v0 [33,40), u1 [40,47), v1 [47,54). Coordinates are 0..=64, so 7 bits each; the 52-bit estimate here was wrong. The shader constants are checked by executing a GPU decode against the CPU unpack, not only by reading the source.
   - **Instances:** one per region. `InstanceCustomIndex` is the region index; the instance transform translates to the region origin in voxels, and a per-frame camera-relative offset keeps the numbers small.
3. **Surface ID mapping:** (instance, primitive) → region = instance, quad = primitive ÷ 2. Raster and BLAS draw from the same index buffer in the same order, so the raster `PrimitiveID` equals the ray-query `PrimitiveID`.
4. **Projected GPU size:** 64 B per quad (32 B vertices, 24 B indices, 8 B table), excluding the BLAS itself. For the street block that is 0.67 MB greedy and 33.3 MB unmerged. The BLAS is measured in 2D.

## Contracts and consequences

- **Exactness:** both merge modes cover exactly the same exposed unit faces, each once, with the voxel's exact material. This is tested against a direct voxel scan on the street and on random worlds that cross chunk boundaries.
- **Ray reference:** `derived::trace_mesh` is the CPU oracle for the later ray-query path. It uses the 1E crossing expression and half-open rule. For rays in general position it agrees exactly with the voxel DDA on t (bit-equal), voxel, material and face.
  - Known measure-zero difference: a ray passing exactly through the shared edge of two diagonal voxels. The DDA reports a hit; no quad is crossed. A unit test documents this case.
- **T-junctions (accepted risk, measured later):** greedy quads meet at T-junctions, where one quad's corner lies partway along another quad's edge. Rasterization and ray-triangle tests are only watertight across *shared* edges, so single-pixel cracks are possible.
  - 2C/2D count crack pixels, meaning background visible where the unmerged mesh shows a surface, as part of the sweep.
  - If the count is not zero, the options are T-junction-free triangulation, or the unmerged mode for the affected regions. Accepting cracks is not an option.
- **Units:** voxel units throughout. World meters = voxels × `VOXEL_SIZE_M`, applied only in the camera transform.
- **Memory:** CPU quads are counted under `DerivedData` in 1D. GPU buffers take `Ledger` grants under `GpuMesh`.

## Validation and revisiting

- **Done in 2A:** size assertion, winding test, coverage and ray-agreement tests with negative controls, and pipeline equality in both modes.
- **2B** (implemented; results in the 2B record):
  - uploads read back bit-exact for all 8 settings
  - GPU decode equals the CPU for every quad
  - host/shader layouts are checked from `slangc` reflection before any pipeline is created
  - SPIR-V capabilities are checked against an allow-list at build time
  - The RTX supports `R16G16B16A16_SFLOAT` as a vertex buffer and as acceleration-structure vertex input (format test, 2026-09-23).
- **2C-1** ([record](../changes/2026-09-23-phase2c1-raster-equivalence.md)):
  - Raster `PrimitiveID ÷ 2` resolves to the right quad for every checked pixel, in 32 settings (4 cameras × both merges × 4 region sizes). Material, normal and resolved face are exact.
  - **T-junction cracks observed:** greedy meshes leave 1 crack pixel, on the `overhead` camera at the ground's z = 64 brick boundary, at every region size. Unmerged meshes leave 0.
- **Revisit:**
  - if crack pixels are not zero. **Triggered by 2C-1, resolved by Amendment 1** (the user chose T-junction-free greedy with a per-triangle quad index).
  - if the sweep favours 16-bit indices or another region size
  - when smooth or non-axis-aligned geometry arrives, which needs a new representation with its own error budget

## Implementation status

- CPU quad, merge modes, extraction through the 1C pipeline, and `trace_mesh`: implemented and tested.
- GPU layout: implemented in `engine/gpu` (`layout`, `mesh`, `decode`).
  - Sections are aligned to max(16, `minStorageBufferOffsetAlignment`), so device bytes depend on the device. For example, on the Intel iGPU the greedy mesh at 1-brick regions takes 1.15 MB, against 0.67 MB at chunk regions.
  - Acceleration-structure input usage is set only on RT devices.

## Amendment 1 (2026-09-23): watertight triangulation and a per-triangle quad index

**Trigger:** 2C-1 found 1 T-junction crack pixel with greedy meshes (see the revisit list above). User decision: "Fix the merged meshes properly. Split rectangle edges so every edge is shared ... green light for that only for now."

**Decision (replaces decision 2's vertex and index rules, and decision 3's mapping):**
- **Split rule** (`derived::Split::Watertight`, in `BrickMesh::triangulate`). It uses only the brick's own quads, so a brick's triangles still depend on nothing but its mesh, and the 1C dependency record is unchanged:
  - An edge lying on the brick boundary (one of its fixed coordinates is 0 or 8) is split at every lattice point. The neighbouring brick does the same on its side, so both produce identical unit segments.
  - Any other edge is split at every corner of the brick's own quads strictly inside it. Its interior lies inside the open brick, so no other brick's vertex can touch it.
- **Triangulation:**
  - A quad with no split points keeps 2 triangles, (c0, c1, c2) and (c0, c2, c3). Every unmerged quad is such a quad.
  - A split quad becomes a fan around its centre, one triangle per outline segment. None is degenerate, and all are counter-clockwise from outside.
  - Centres are half-integers. Vertices are therefore multiples of 1/2, still exact in `R16G16B16A16_SFLOAT` (`gpu::layout::f16_of_halves`).
- **GPU sections per region** (`gpu::layout::Sections`):
  - vertices, 8 B each
  - indices, 3 × `u32` per triangle
  - **triangle quads**, a `u32` per triangle: the index of its quad
  - quads, 8 B each; the 54-bit record is unchanged
- **Surface ID mapping:** (instance, primitive) → region = instance, quad = `tri_quad[primitive]`. The old `primitive ÷ 2` no longer holds for split quads. Raster and BLAS still draw from the same index buffer, so the pair still agrees between them.
- `Split::CornersOnly`, the pre-amendment triangulation, is kept for negative controls only.

**Evidence** ([record](../changes/2026-09-23-phase2c1-watertight-greedy.md)):
- **CPU, every directed edge matched by its reverse:** holds on the street and 4 random chunk-crossing worlds, in both merge modes. Every triangle faces its quad's normal, and the triangle area per quad is exact.
  - Control: unsplit greedy leaves 1,170 unmatched edges on the street.
- **GPU decode:** every triangle's vertices lie on its `tri_quad` quad, with outward winding, for all street triangles, checked by execution.
- **Raster equivalence:** 0 cracks in all 32 settings. Control: the unsplit layout shows the crack again (1 pixel on `overhead`).

**Cost, street block:**

| Merge | Quads | Triangles | Vertices | Device images |
|---|---|---|---|---|
| none | 520,540 | 1,041,080 | 2,082,160 | 37.48 MB (was 33.3) |
| greedy, watertight | 10,503 | 275,890 | 286,546 | 6.79 MB (was 0.67 MB) |
| greedy, unsplit (control only) | 10,503 | 21,006 | 42,012 | not a product |

- Splitting boundary edges at every lattice point is most of the greedy cost.
- A cheaper rule would need to know the neighbours' vertices, which would break brick-local dependencies. That tradeoff is for the 2E sweep, not decided here.
- The per-triangle index adds 4 B per triangle in both modes.

**Revisit** if the 2E sweep shows triangle count or BLAS size limiting, or if smooth or non-axis-aligned geometry arrives.
