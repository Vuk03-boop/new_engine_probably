# Proposition: a new voxel-world rendering engine

**Revision 1 · 2026-09-22 · Proposed, not implemented or performance-validated**

## 1. Objective and central decision

Build for **appearance per unit of total cost**, not maximum algorithm count. The objective includes convincing illumination, clean materials, reflections/transmission, fine contact detail, motion stability, responsive edits, bounded memory and predictable frame latency.

Appearance and performance cannot both be maximized without constraints. We will maintain a quality/performance frontier across explicit presets, comparing alternatives at matched quality or matched cost. The target hardware, resolution, frame budget, scene scale and edit rate must be chosen before acceptance targets are fixed. See [context](CONTEXT-AND-CONSTRAINTS.md) and [validation](VALIDATION-AND-BUDGETS.md).

### Recommended architecture

> Keep voxels authoritative for the world; derive the representation most efficient for each rendering job. Rasterize ordinary opaque visibility, trace lighting against coherent surface geometry, and retain direct voxel traversal where its properties justify it.

The world should not be forced through one universal marcher. Equally, meshes must not become an unbounded second copy that destroys the memory benefits of sparse voxels. **All duplicated representations must be budgeted together.**

```text
Persistent world, edits and material registry
                  |
     Sparse editable bricks / spatial hierarchy
                  |
     Versioned jobs and immutable frame snapshots
                  |
       +----------+------------+----------------+
       |                       |                |
  Surface meshes + LoDs    Ray structures    Conservative bounds
       |                       |                |
       +----------+------------+----------------+
                  |
      Visibility + material/geometry guides
                  |
       Explicit sampled lighting reference
                  |
     ReSTIR direct/indirect reuse where useful
                  |
     Optional layers / LoD mapping / diffuse NRC
                  |
      Denoising and temporal reconstruction
                  |
        HDR composition, tone mapping, display
```

This is an architectural recommendation, not a finding that hybrid rendering beats direct voxel traversal on every workload. Edit-heavy destruction is a deliberate comparison case and can change the representation choice.

## 2. World, identity and update model

### Authoritative sparse world

Use a chunk hierarchy containing smaller sparse bricks. Keep occupancy, exact categorical material identity and numerical material parameters logically separate. Chunk and brick dimensions are tuning decisions, not inherited constants.

Store persistent edits in a journal or equivalent transactional world store. Generation is a producer of world data, not a hidden substitute for edited geometry. CPU gameplay queries and render extraction must have a declared world-version relationship.

### Identity is a first-class interface

Distinguish four identities:

- **Logical spatial identity:** region/brick/object identity independent of GPU placement.
- **Content version:** changes when relevant geometry/material data changes.
- **Allocation generation:** changes when a physical slot is reused.
- **Surface identity:** enough information to relate actual surfaces across frames; not merely a temporary chunk or triangle index.

A stable identifier does not prove a stable point correspondence. A merged face, deleted voxel, remesh or changed LoD may still require rejection or an explicit map.

### Transactional derived data

Every mesh, ray structure, bound, light-sampling distribution and cache result records its input versions. Reject stale job results. Upload data and complete required builds before publishing descriptors; retire old resources only after their last readers finish.

For edits, the initial correctness policy is **a coherent committed render snapshot**, not partially updated raster and ray worlds. Keep the prior snapshot until the replacement resources for the affected publication set are ready. Report and budget edit-to-visible latency; gameplay must not silently assume the displayed snapshot is newer than it is. If immediate rendering is required, add a separately validated direct-voxel overlay/fallback, not an unversioned mixture.

Chunk boundaries, neighbor faces, coarse ancestors and emitter tables are part of that publication set. Cancellation, overload and eviction all have explicit behavior.

## 3. Sharing, allocation and residency

### Geometry sharing

Begin with simple sparse bricks and exact deduplication. Add reflection/permutation-aware immutable sharing only after a representative census shows a net benefit. Occurrences retain transforms and material association; edits use copy-on-write. Start without destructive translations.

Count geometry dictionaries, handles, transforms, attributes, allocation slack, extraction cost and derived meshes/acceleration structures. Compressing voxel occupancy does not automatically compress those derived representations. Prefer object instancing for reusable surface geometry where possible; arbitrary occupancy-node sharing need not become a hardware acceleration-structure instance per node.

### GPU memory

Use budgeted pools with stable logical handles and explicit physical placement. Separate geometry, material, mesh and temporal allocations; do not force every payload into one page class. Keep upload staging and primary GPU storage distinct unless a measured device-specific path justifies otherwise.

Resolve page/handle indirection at task or brick granularity where possible. Bindless descriptors, buffer device addresses and cooperative operations are capability choices, not universal assumptions. Address stability never removes lifetime or synchronization requirements.

### Visibility and demand

The GPU performs frustum/occlusion tests, screen-space-error selection, work compaction and indirect execution. It can emit compact residency requests. The CPU manages generation, persistent storage, edit authority and bounded asynchronous jobs.

All request queues have capacity limits, deduplication, overflow counters and a fallback. Unknown data is not transparent by default. Keep a coarse resident representation or explicit conservative boundary behavior; missing offscreen casters cannot silently become guaranteed visibility.

## 4. Representations and traversal

| Query | Initial choice | Qualification |
|---|---|---|
| Ordinary opaque primary visibility | Rasterized exposed-surface meshes | Material-aware merging must preserve UVs, normals and exact face identity lookup. |
| Surface shadow/reflection/GI rays | Hardware ray tracing over matching surface geometry | Include BLAS/TLAS build/update cost, memory and publication latency. |
| World edits, occupancy and gameplay queries | Sparse voxel bricks | Separate render approximation from authoritative simulation semantics. |
| Highly dynamic or specialized sparse queries | Direct voxel traversal | Must agree on materials/opacity and coexist through an explicit visibility policy. |
| Distant world | Coarser geometry selected by screen-space error | Keep a validity policy for primary versus secondary-ray LoD. |
| Terrain-only empty intervals | Conservative height/min-max hierarchy | Excludes unrepresented overhangs/edits; fall back when uncertified. |

Raster and traced geometry initially use the same committed surface/LoD selection. Do not introduce different shadow/GI LoDs until their approximation and temporal consequences have a contract. Camera-invisible geometry can still cast shadows or appear in reflections; primary visibility is not the whole residency demand.

An independent simple ray/intersection reference is required. If maintaining extracted meshes and hardware ray structures loses on the chosen edit workload, compare a direct-voxel ray backend before adding further optimizations. Do not maintain two fully optimized backends prematurely.

### Conservative empty-space metadata

NAADF contributes certified empty volumes. Build explicit bounds with a documented distance/interval convention, rounding policy, material policy and invalidation footprint. Insertion or newly resident geometry invalidates unsafe certificates before use; unknown space uses fallback. A bound for opaque shadows is not automatically valid for transmission or collision.

Speculative height estimates may order searches, but skipping needs a proof over the skipped interval. Endpoint verification is insufficient. JFA’s approximate nearest-seed values are not automatically safe skip distances.

## 5. Shading and lighting

### Establish a reference before reuse

Build explicit BSDF/material evaluation, light sampling, PDFs/MIS, emissive-surface selection, environment/sun lighting and path evaluation. Provide a high-sample reference mode for controlled scenes before introducing reservoirs or learned continuation. These are foundational rendering requirements, not implementations supplied by the reviewed papers.

Keep linear HDR quantities, units and exposure conventions documented. Diffuse, specular and transmission contributions must be distinguishable. Water/glass require defined interface orientation, absorption and roughness behavior; not every expensive caustic needs to be in the first real-time tier.

Do not stack baked probes, voxel light, AO and traced indirect illumination as if all were independent physical energy. A stylized ambient/AO mode may exist, but its artistic approximation and quality comparator must be explicit.

### Reservoir-based direct illumination

Use a dedicated direct-light reservoir method where numerous emissive surfaces or dynamic lights justify it. A single sun can have a simpler specialized query. A direct-light implementation still needs its own primary-source/algorithm review; the supplied indirect/path papers do not substitute for it.

### Indirect lighting: ReSTIR PT family

After the sampled reference is correct, introduce the actual estimator machinery: candidates, reservoirs, valid path shifts/reconnection, weights, visibility, canonical support and temporal/spatial resampling. ReSTIR PT Enhanced contributes its specific work-reduction mechanisms, including reciprocal pairing where duplicated bidirectional shift evaluation really exists.

Compatibility-guided selection chooses better neighbor domains. It does not replace visibility, weight calculations or support-covering candidates. Any biased decorrelation/filtering option must be labeled and evaluated separately; do not call the entire real-time pipeline unbiased by association with a paper.

### Temporal layers and LoD correspondence

Start with a working single-layer temporal system whose producer, persistence, reset and rejection counters are observable. Add sparse extra receiver layers only in scenes where disocclusion quality improves enough to justify memory and work. Layer collisions, ownership, overflow and fresh fallback are specified—not left to write order.

For authored objects, consistent LoD charts and partial reciprocal correspondence can support the LoD ReSTIR approach. For generated voxel surfaces, extraction may preserve provenance for surviving patches; changed topology remains unmatched. Jacobians belong to the estimator transporting measure, not automatically to TAA color blending. Mapping cost and retained old geometry count against the budget.

### Optional neural radiance cache

NRC is a research-tier diffuse-path continuation accelerator, not a universal AO/visibility/specular replacement. Retain explicitly traced camera-near/high-frequency interactions. Train against defined radiance targets and account for path generation, encoding, optimizer/EMA state, updates, warm-up and scene-change adaptation.

Combine it with reservoir reuse only after specifying where cached continuation enters the estimator and what bias/dependence is introduced. First evaluate each technique independently against the reference; never count the same removed tracing work twice. Uncertain or rapidly changed regions need fresh tracing or reset behavior, not a presumed neural safety guarantee.

Cooperative matrices accelerate this workload only when actual device properties and validated kernels support a useful configuration. FP16 is a reasonable candidate, not a commitment to a specific tile/network or latency.

## 6. Textures, materials and reconstruction

Keep categorical IDs exact. Adaptive palettes and conventional GPU texture compression/filtering are baseline choices. SmoothQuant/MX become candidates for numerical neural inputs/weights where their accuracy and execution cost justify them; they are not material-ID codecs.

Use collaborative texture filtering only when source texel production is sufficiently expensive and its filter assumptions fit the workload. Each lane reconstructs its own result; equal geometry is not equal texture lookup. Cheap native textures remain the control.

Use an explicit guide schema for reconstruction: depth convention, world/shading normals, motion convention and units, jitter, material albedo, roughness, specular distance where required, and history validity. The final owning surface writes its guides. Secondary work must not accidentally overwrite them with another receiver’s attributes.

Keep a native reconstruction path for reference/debugging. Vendor reconstruction is optional and integrated according to its own contract; do not stack two temporal reconstructors without a supported design. Appearance acceptance includes motion, not just attractive still images.

## 7. Execution and proposed technology direction

**Proposed starting direction:** Rust for world/host systems, a Vulkan-first native backend for fine-grained ray-tracing/resource control, and a modular shader toolchain evaluated in an initial compiler/capability spike. Slang-to-SPIR-V is a candidate, not an already validated dependency choice. Do not simultaneously undertake a full DX12 port or broad renderer abstraction.

The initial product hardware floor is a discrete GPU with the selected ray-tracing capability. A lower non-RT tier is optional scope to approve, not promised for free. Neural/cooperative-matrix features are a separate optional capability tier. Freeze exact toolchain/dependency versions after the feasibility gate.

Use a declarative pass/resource graph with explicit accesses and subresource lifetimes. Keep bindings narrow enough for meaningful hazard analysis; validate real barriers and queue ownership. Default to one graphics/compute execution plan, then introduce asynchronous transfer/compute only where a measured independent window exists.

Use hybrid execution: coherent wavefront queues where material/path divergence matters, bounded fused stages where intermediate traffic dominates. FlashAttention inspires working-set/IO accounting; it does not specify a traversal kernel. SGLang inspires exact-key immutable-data reuse; it does not make neighboring rays equivalent. Continuous batching inspires refillable work, not arbitrary per-lane stealing at the expense of locality.

Shader compilation runs off the event thread, with bounded specialization growth, progress, cancellation and a compatible pipeline-cache policy. Generated modules, compiler versions, selected features and keys are recorded for every capture. Smaller source is not assumed to compile faster.

## 8. Deliberate exclusions and success definition

Do not make the following baseline requirements: destructive-translation DAG search, dense multi-layer reservoirs at every pixel, neural lighting everywhere, JFA block-light replacement, persistent mapped shader storage, blanket FP32 conversion, universal ray-axis normalization, arbitrary native multi-queue use, or mandatory compressed neural textures.

A useful technique can be absent from a successful engine. Admit each by evidence of net appearance/cost benefit under the chosen workload. The [technique map](TECHNIQUE-MAP.md) accounts for every supplied original idea, including exclusions.

**The first success is not the full stack.** It is a coherent editable world, stable raster/ray visibility, correct materials, measured resource lifetimes and an honest lighting reference. The [roadmap](BUILD-ROADMAP.md) then adds frontier techniques one at a time with independent controls.

No performance number in this proposition is a forecast. No new engine exists yet.
