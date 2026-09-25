# Build roadmap — dependency order, not an implementation authorization

**No phase has been implemented in this new project.** This is a proposed sequence. Do not create a code scaffold or pick dependency versions until the scope/capability gate is approved. Avoid a calendar promise before hardware, workload and staffing are known.

Each phase produces a runnable vertical slice only after implementation is separately authorized. Keep a reference mode and a control-plus-one comparison throughout.

## Phase 0 — Product contract and capability/compiler feasibility

**Decide:** target devices/OS, native/internal resolution, frame and memory budgets, edit rate, visual style, startup tolerance and RT floor. Decide whether a non-RT tier is in scope.

**Future work:** exercise the proposed native API and shader toolchain on the real device; verify ray-tracing support, resource limits, pipeline creation, debug/capture tools, precision/subgroup properties and optional matrix tuples. Inventory SDK/code licenses. Evaluate modular Slang-to-SPIR-V against a minimal alternative before adopting it as a project dependency.

**Deliver:** signed-off target matrix, deterministic scene/edit traces, exact/approximate contract, capability report and toolchain lock.

**Gate:** representative shader/API paths execute correctly and can be profiled; main event loop remains responsive during cold compilation. Stop or reduce scope when capability or tooling fails. A hardware feature listing alone does not pass.

## Phase 1 — Authoritative world and lifetime substrate

**Build later:** simple sparse bricks, exact material registry, persistent edits, logical IDs, content versions, allocation generations, version-tagged jobs, bounded queues and resource retirement. Start with simple allocation behind logical handles; do not implement complex paging before evidence.

**Deliver:** headless world/edit replay; snapshots; cancellation/stale-result diagnostics; allocation accounting; simple intersection/occupancy reference.

**Gate:** exact save/reload and material identity; boundary edits and reused handles cannot expose stale results; overload has deterministic fallback. Test the publication/retirement contract with artificial delays and out-of-order jobs.

**Do not add yet:** neural caches, reservoirs, transform canonicalization, elaborate queue overlap.

## Phase 2 — Coherent visibility and the first visual slice

**Build later:** exposed-surface extraction, material-aware merging, a simple LoD/residency policy, opaque raster visibility and a matching ray-query representation on the selected backend. Track the extra mesh/BLAS/TLAS memory explicitly.

**Deliver:** visible editable scenes, stable guide records, debug surface/version views, raster-versus-ray intersection comparison and per-stage timings. Material IDs/face mapping must survive mesh merging.

**Gate:** raster and ray paths see the intended committed scene; silhouettes, alpha/cutout/material behavior and chunk boundaries agree within the declared geometric convention. Edit latency and acceleration rebuild cost fit the chosen budget.

**Architecture fork:** if high-edit workloads make surface extraction/RT structures untenable, compare a simple direct-voxel alternative here. Do not postpone this decision until a large lighting system depends on a losing geometry path.

## Phase 3 — Honest sampled lighting and reconstruction

**Build later:** BSDFs, sun/environment/emitter sampling, PDFs/MIS, visibility, a high-sample reference for controlled scenes and an initial low-sample real-time path. Add water/glass in a staged material test suite, not by burying secondary semantics in opaque shading.

Define the guide schema and native reconstruction path. Integrate a vendor reconstructor only after validating its feature/device requirements, units, motion, reset and material guides. Vendor and native paths have distinct capture configurations.

**Deliver:** reference images and motion sequences; energy/material tests; decomposition views; noisy-versus-reconstructed comparisons; measured tracing/reconstruction cost.

**Gate:** basic transport and guide/reset behavior are understood before reuse. Do not hide missing light or wrong motion under more history/clamping. The reference is expensive but trustworthy; the real-time target is a separate acceptance result.

## Phase 4 — Add reservoir reuse in controlled steps

1. Introduce a reviewed direct-light reservoir method where a many-light workload warrants it; retain a simpler sun path.
2. Build indirect path reuse with required path state, shifts, weights and canonical support.
3. Add compatibility-guided candidate selection.
4. Add applicable ReSTIR PT Enhanced work reductions, with reciprocal pairing only where both evaluations exist.

**Deliver:** one independent arm per mechanism, reservoirs/weights/validity diagnostics, visibility-query and shift-evaluation counts, reference quality/time curves.

**Gate:** useful temporal/spatial reuse is observed; correctness/quality matches the selected estimator contract; net time/memory benefit is demonstrated at matched quality. Biased options are labeled, isolated and judged under an explicit approximation budget.

**Stop:** a cache that never becomes valid, a reset every frame, unexplained energy loss, invalid shifts, queue overflow without fallback, or a “win” that is only blur hiding lost detail.

## Phase 5 — Improve temporal robustness, one mechanism at a time

**Branch A:** sparse additional receiver layers for identified disocclusion failures. Include scatter ownership, collisions, compaction, overflow and fresh sampling.

**Branch B:** partial LoD correspondence for supported surfaces/instances. Keep unmatched rejection. Count retained geometry, mappings, candidate search and Jacobian work.

**Deliver:** motion/disocclusion/LoD torture sequences and layer/mapping allocation traces.

**Gate:** fewer temporal artifacts at an acceptable frame/memory cost. Test cold start, camera cuts, edits, lighting changes, evictions and repeated LoD toggling. Neither branch is required to ship the earlier functioning renderer.

## Phase 6 — Scale representation and residency from traces

This phase can overlap conceptually with earlier profiling, but do not combine implementation experiments before their controls exist.

- Measure immutable geometry symmetry; introduce reflection/permutation sharing only if total resident bytes improve after all metadata/derived data.
- Compare simple preallocation, size-class allocation and trace-designed paging behind the same logical interface.
- Add conservative empty-space bounds only to direct-voxel queries where traversal profiles justify them.
- Improve demand selection and LoD budgets without dropping offscreen ray contributors silently.

**Gate:** an actual memory/stream/edit bottleneck improves with exact reconstruction and lifetime safety. Retain a simple allocator/uncompressed/reference traversal control. Destructive translation search is not a requirement.

## Phase 7 — Optional neural/material research tier

**NRC branch:** select diffuse-tail targets, implement reference training data and a primitive validated against CPU outputs, then test learnability and net cost. Compare reference, ReSTIR-only, NRC-only and the explicitly designed combined estimator. Include training/warm-up/edit adaptation and fallback tracing.

**Texture branch:** test expensive-texel collaboration against native sampling; numerical quantization against FP16 only for actual supported neural workloads.

**Gate:** better appearance at matched frame/memory cost or lower total cost at a predeclared matched quality. Require moving/editing scenes, not just steady-state static averages. Unsupported or unhelpful neural features remain absent from the shipping tier.

## Phase 8 — Kernel and queue specialization

Only after profiling, test wavefront versus fused stages, coherent dynamic scheduling, immutable-data caching, exact-domain arithmetic, box-test variants, upload placement and asynchronous overlap. One change per arm initially.

**Gate:** generated-code/resource data explains a repeatable end-to-end gain without violating exact contracts or memory/latency tails. Include compiler startup cost and thermal/power state. Do not infer success from instruction counts or sum overlapping pass intervals.

## Minimum credible release versus full research build

A credible first engine stops after **coherent world/visibility + reference-backed lighting/reconstruction**, with modest residency and clearly selected quality settings. Reservoir reuse is the first major advanced-lighting expansion. Additional layers, LoD transport, neural continuation and heavy compression are optional milestones, not a prerequisite to seeing the first correct frame.

## Handoff rule for every phase

Record: input versions, decision/ADR, implementation scope, control, scene/feature/pack/device configuration, commands actually run, raw outputs, known failures, quality/performance result, state-reset procedure and rollback method. A failed gate means revise or stop—not add another technique to compensate blindly.
