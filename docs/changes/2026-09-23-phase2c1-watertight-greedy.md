# Change: watertight greedy meshes (ADR-0004 Amendment 1)

Status: **done; the 2C-1 equivalence gate now passes for every setting.**
Date and baseline: 2026-09-23, after the [2C-1 record](2026-09-23-phase2c1-raster-equivalence.md) stopped on 1 greedy T-junction crack. No Git repository.
Authorization (S-010), user: "Fix the merged meshes properly. Split rectangle edges so every edge is shared. This changes the mesh layout in ADR-0004 and needs an extra index per triangle ... green light for that only for now". 2C-2 is not included.

## What changed

**`derived::mesh`:**
- `Split` (`Watertight`, and `CornersOnly` for controls), `Triangles`, and `BrickMesh::triangulate`.
- The rule is in [ADR-0004 Amendment 1](../adr/ADR-0004-surface-mesh-layout.md). It is brick-local: boundary edges are split at every lattice point, inner edges at the brick's own quad corners. Split quads become centre fans.

**`gpu::layout`:**
- `RegionMesh` carries vertices, in half-voxel units, plus indices and `tri_quad`.
- `Sections` has four sections: vertices, indices, triangle quads, quads.
- New: `f16_of_halves`, and `build_regions_with` for choosing the split rule.

**Shaders:**
- `shaders/raster.slang` resolves quad = `tri_quad[PrimitiveID]` (binding 1).
- `shaders/mesh_decode.slang` still decodes every quad record, and now also checks every triangle on the device: its vertices lie on its quad's plane and rectangle, its quad index is in range, and it winds outward.

**Host code:** `raster`, `decode`, `mesh` and `equivalence` follow the new sections.

## Results

**CPU, pure crates:**
- `cargo test --release -j 2 --no-fail-fast` exits 0: **101 tests**, 99 old and 2 new. Log: `engine/results/test_release.log`.
- **New tests** (`derived/tests/mesh.rs`):
  - `watertight_triangulation_shares_every_edge`: on the street and 4 random chunk-crossing worlds, in both merge modes, every directed edge a→b is matched by an equal count of b→a. Every triangle faces its quad's outward normal, and the triangle area per quad equals the quad's.
  - Unmerged quads triangulate identically with and without splitting.
  - `negative_control_unsplit_greedy_has_t_junctions`: the unsplit greedy mesh leaves 1,170 unmatched directed edges on the street. A flipped and a dropped triangle are also reported.

**GPU, RTX 3050, validation on:**
- `cargo test --release -j 2 -p gpu -- --test-threads=1 --nocapture` exits 0: **11 unit, 9 device, 4 raster.** Log: `engine/results/test_gpu_2c1_watertight_rtx.log`.
- **Equivalence, all 32 settings** (4 cameras × 2 merges × 4 region sizes):
  - cracks, and material, normal, face and depth mismatches: 0
  - extra surfaces and bad ids: 0
- **Greedy, per camera:**
  - `overhead`: 0 cracks, down from 1.
  - Max depth err/tol: 0.53 street_view, 0.64 chunk_corner, 0.31 overhead, 0.95 grazing.
  - Plane-depth outliers, reported but not failures: 2 street_view, 235 grazing.
- **New GPU control** (`negative_control_unsplit_greedy_cracks_again`): the `CornersOnly` layout still leaves the crack (1 pixel, 21,006 triangles), and the watertight layout leaves 0 (275,890 triangles). The layout is what closes it.
- **GPU decode:** all 520,540 unmerged quads and 1,041,080 triangles, and all 10,503 greedy quads and 275,890 triangles, agree with the CPU.
  - Controls still caught: a vertex moved off its quad's plane (its triangles fail), and a swapped quad record.
- **Earlier controls unchanged:** reversed winding, offset, wrong-primitive material, dropped region.

**Test fixture change (not a rebaseline):** `a_refused_grant_changes_nothing_and_old_meshes_stay_valid` had a fixed budget of the ring plus 2 × 4 MiB, sized for the old 0.67 MB greedy images.
- With 6.8 MB images, the final readback no longer fitted, and it failed after the refusal itself had behaved correctly.
- The budget is now sized from the measured greedy images: the ring, the mesh blocks plus one, and room for the readback.
- It still refuses the 37 MB unmerged upload with nothing changed.

**Clippy:** exit 0, the 6 pre-existing lints only.

**Cost, street block:**

| Setting | Triangles | Device images |
|---|---|---|
| none (all region sizes) | 1,041,080 | 37.48 MB (was 33.32 MB) |
| greedy, 1-brick regions | 275,890 | 6.85 MB (was 0.715 MB) |
| greedy, chunk regions | 275,890 | 6.79 MB (was 0.673 MB) |

- Greedy is still 3.8× fewer triangles and 5.5× fewer bytes than unmerged. Most of its cost is splitting brick-boundary edges at every lattice point.
- A neighbour-aware split would be cheaper, but a brick's mesh would then depend on its neighbours' meshes. That is left for the 2E sweep to judge.

## Risks

- The worst checked depth error at grazing incidence is still 0.95 of the tolerance. That is unchanged by the fix, and it remains a risk for 1080p.
- Fan triangles are thin slivers from the quad centre. Nothing wrong was measured, but their BLAS build and trace cost is unmeasured until 2D.

## NOT RUN

- the debug build of the GPU tests
- 1080p equivalence
- GPU timing
- 2C-2
- the 2D BLAS path (the same buffers and `tri_quad` are meant for it)

## Closeout

- **Docs:** ADR-0004 (Amendment 1, status), [DECISIONS](../DECISIONS.md) (S-010, A-007), the [2C-1 record](2026-09-23-phase2c1-raster-equivalence.md) (status), `engine/README.md`, `docs/NOW.md`.
- **Revert:**
  - remove `Split`, `Triangles` and `triangulate` from `derived::mesh`
  - restore the 3-section layout in `gpu::layout` and `primitive ÷ 2` in `raster.slang` and `equivalence`
  - restore the old decode shader and host code, and the old fixture budget
