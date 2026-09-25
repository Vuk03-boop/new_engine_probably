# Technique map — what each idea is for in the new engine

## Reading rule

**Core direction** means proposed architecture, not a validated speedup. **Conditional** means useful only after a workload/capability/equivalence gate. **Research tier** means substantial prerequisites and quality risk. **Excluded transfer** means the original mapping is still wrong even in a new engine.

Proposal IDs below are the original research IDs, retained for traceability. Their historical verdicts are not rewritten. The sources linked in the final column are our owning reviews, containing the paper identity, evidence and limitations.

## All 21 original proposals

| ID / source idea | Proposed role | Needs first | Do not assume | Evidence |
|---|---|---|---|---|
| **P01 — NAADF** | Conditional conservative empty-volume acceleration for direct voxel rays | Explicit bound convention, material policy, insertion invalidation, unknown-space fallback | Spare packed bits, arbitrary-ray safety from samples, or universal faster traversal | [Representation review](../research-review/batch-01-representation.md) |
| **P02 — Aokana** | Core direction: GPU visibility/LoD/work compaction and budgeted demand-driven residency | CPU world authority, asynchronous requests, overflow handling and coarse fallbacks | GPU rendering implies CPU-free planning, generation or editing | [Streaming review](../research-review/batch-02-streaming-memory.md) |
| **P03 — transform-aware DAGs** | Conditional immutable occupancy sharing; copy-on-write occurrences | Separate attributes/metadata, transforms, exact reconstruction and ownership | Occupancy compression automatically compresses meshes/BLAS or improves ray time; translations are cheap | [Representation review](../research-review/batch-01-representation.md) |
| **P04 — collaborative texture filtering** | Conditional shared source-texel production for expensive neural/procedural/compressed textures | Appropriate filter footprint, per-lane reconstruction and supported collaboration | Same plane means same sampled value; paper filtering is identical to any chosen sampler | [Texture review](../research-review/batch-05-textures-shader-organization.md) |
| **P05 — compatibility-guided neighbors** | ReSTIR candidate-domain selection after a correct estimator exists | Reservoirs, valid shifts, weighting, canonical support and visibility | Geometric compatibility proves equal blocker distance or visibility | [Spatial reuse review](../research-review/batch-06-restir-spatial-reuse.md) |
| **P06 — multi-layer splatting** | Research tier: sparse hidden-receiver history for disocclusion | Live single-layer history, layer ownership/collision rules, budget, fallback, estimator-specific reverse operations | Two temporal parities are two layers; retained receiver means valid lighting; unseen surfaces have history | [Temporal review](../research-review/batch-07-temporal-lod-continuity.md) |
| **P07 — LoD ReSTIR** | Research tier: partial cross-LoD path correspondence | Stable instance identity, charts/provenance, reciprocal matching, Jacobians, unmatched rejection | Dissolve is mapping; every voxel remesh is invertible; Jacobians belong in generic color TAA | [Temporal review](../research-review/batch-07-temporal-lod-continuity.md) |
| **P08 — ReSTIR PT Enhanced** | Main advanced indirect-lighting path; reciprocal pairing where duplicated shift work exists | Explicit path sampling, reconnection, reservoirs, MIS/weights and visibility | Half all gather ALU/registers; all algorithm options are unbiased; paper frame gains transfer | [Spatial reuse review](../research-review/batch-06-restir-spatial-reuse.md) |
| **P09 — FlashAttention / FlashAttention-2** | Execution principle: bounded working sets, deliberate fusion, fewer intermediates | Measured traffic, coherent batches, uniform synchronization and occupancy analysis | Attention kernels transfer directly; copying ancestors eliminates leaves/materials | [Traversal review](../research-review/batch-03-coherent-traversal.md) |
| **P10 — SmoothQuant / MX** | Conditional numerical compression for neural weights/activations or suitable features | Actual neural workload, precision/quality tests and efficient supported kernels | Categorical material IDs can be approximated; quantization beats FP16 on a small online network | [Representation review](../research-review/batch-01-representation.md) |
| **P11 — speculative decoding analogy** | Bounded search prediction; terrain-specific conservative interval tests | Verification of the whole skipped interval and a fallback on unrepresented geometry | Endpoint-only checking preserves first-hit geometry; probabilistic language sampling proof transfers | [Lighting/heightfield review](../research-review/batch-08-lighting-neural-heightfield.md) |
| **P12 — NRC + cooperative matrices** | Research tier: learned diffuse-tail continuation and optional accelerated inference/training | Traced radiance targets, estimator boundary, training state, scene-change policy and verified device/kernel | One tiny network replaces AO, visibility, probes and specular transport; a feature flag establishes execution | [Lighting review](../research-review/batch-08-lighting-neural-heightfield.md), [correction](../research-review/cross-audit-other-ai.md) |
| **P13 — PagedAttention** | Core logical/physical separation; physical paging/suballocation sized by traces | Budget, generation-tagged handles, overflow, upload ordering and reader retirement | 4 KiB is universally right; page tables remove suballocation waste or every binding cost | [Memory review](../research-review/batch-02-streaming-memory.md) |
| **P14 — Orca / continuous batching** | Conditional coherent wavefront queue refill and work balancing | Actual task-length/locality data; bounded queues and termination rules | Per-lane stealing removes SIMT divergence; CPU serving scheduler determines GPU residency | [Scheduling review](../research-review/batch-04-scheduling-queues.md) |
| **P15 — sparse MoE analogy** | No MoE port. Use explicit material/BSDF dispatch and bounded static variants on engineering grounds | Modular semantics, tested guide ownership, generated-code and compile-cost measurement | Learned routing proves shader merging or final instruction deduplication; fewer source lines mean faster compilation | [Shader organization review](../research-review/batch-05-textures-shader-organization.md) |
| **P16 — SGLang prefix caching** | Conditional exact-key immutable geometry/material-data reuse | Correct key, lifetime/generation, active participation and independent ray state | Nearby rays have an identical computational prefix or can copy a leader’s hit | [Traversal review](../research-review/batch-03-coherent-traversal.md) |
| **P17 — JFA** | Optional approximate spatial fields, effects masks and preprocessing | Error-tolerant consumer and a specified metric/domain | Exact obstacle-aware per-channel lighting, safe collision distances or conservative empty-space bounds | [Lighting review](../research-review/batch-08-lighting-neural-heightfield.md) |
| **P18 — async compute** | Conditional overlap of genuinely independent uploads/builds/cache work | Pass graph, actual queues, synchronization, ownership and measured spare capacity | More queues mean more throughput; serial scan dependencies disappear; pass times add under overlap | [Queue review](../research-review/batch-04-scheduling-queues.md) |
| **P19 — FP32/INT32 balancing** | Late, device-specific exact-domain kernel optimization | Generated-code bottleneck, exact arithmetic domain and full differential tests | Half the machine is idle; masks/addresses may become float; an arithmetic model predicts frame speed | [Execution review](../research-review/batch-04-scheduling-queues.md) |
| **P20 — host-visible/device-local uploads** | Measured staging rings and device-specific upload placement | Actual memory types, map/coherency rules, queue visibility and retirement | Host visibility guarantees zero-copy or bandwidth; persistent mapping is legal through every API | [API/memory review](../research-review/batch-02-streaming-memory.md) |
| **P21 — axis-normalized ray-box tests** | Conditional optimization for repeated box tests where setup amortizes | Signed parameter conversion, zero-direction handling, tie/rounding policy | One FMA is the entire test; normalization removes guard requirements or always fits a DDA | [Numeric review](../research-review/batch-03-coherent-traversal.md) |

## All 13 retained paper groups

The [canonical coverage register](../research-review/sources/batch-09/coverage.csv) records the 16 actual files and their hashes; duplicates are not extra research evidence. This folder links the originals rather than duplicating PDFs.

| Group | Role / decision |
|---|---|
| NAADF | Explicit conservative bounds for selected voxel paths; not mandatory for mesh visibility |
| Transform-Aware Sparse Voxel DAGs | Conditional geometry memory sharing |
| Aokana | Streaming/LoD/render-work organization, with editable-world extensions designed separately |
| Collaborative Texture Filtering | Conditional expensive-texel collaboration |
| FlashAttention | IO/working-set organization, not a direct traversal algorithm |
| FlashAttention-2 | Work partitioning principles, not assumed SIMT equivalence |
| SmoothQuant | Possible neural numerical quantization, not material identity |
| Microscaling formats | Possible numerical storage/compute formats, capability dependent |
| Compatibility-Guided Neighbor Selection | Neighbor-domain choice within a real ReSTIR estimator |
| ReSTIR PT Enhanced | Advanced indirect path reuse and specific work reductions |
| Multi-Layer Reservoir Splatting | Budgeted temporal disocclusion research |
| LoD ReSTIR | Valid partial correspondence, not automatic voxel morphing |
| Speculative Decoding | Verification discipline; no direct probabilistic-to-geometric guarantee |

External literature already reviewed includes PagedAttention, SGLang, Orca, sparse MoE, axis-normalized intersections, NRC and JFA; hardware/API sources cover the remaining feasibility questions. See the [source register](../research-review/sources/INDEX.md). Original JFA DOI is `10.1145/1111411.1111431`; NRC DOI is `10.1145/3450626.3459812`.

## Combinations that require a new proof or experiment

- **Voxel sharing + meshes + ray structures:** account for all derived copies, transforms and instance granularity. A compressed voxel tree alone does not settle GPU memory use.
- **ReSTIR + NRC:** define the estimator boundary and cache-dependent target; test each alone before a combined arm. Their theoretical properties do not combine automatically.
- **Extra temporal layers + LoD mapping:** map/reject all retained identities, not only today’s visible layer; budget old representations and candidate lookup.
- **Paging + async queues:** a new mapping must not expose stale or repurposed physical storage to another reader or unsubmitted command buffer.
- **Cooperative filtering/data caches + dynamic scheduling:** shared scratch ownership and coherence depend on scheduling. Arbitrary ray mixing may erase the saving.
- **Raster LoD + ray LoD:** differing silhouettes/casters are a rendering approximation; do not pass them off as exact reuse.
- **AO/probes + traced diffuse:** separate approximation modes or avoid double-counting ambient energy.

**Intentionally absent from the baseline:** destructive translations, blanket four-bit materials, learned shader routing, JFA-based exact lighting, universal forward splatting, always-on neural caches, unconditional multi-queue execution and advertised paper speedup multipliers.
