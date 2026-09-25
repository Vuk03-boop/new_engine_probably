# Context, prerequisites and scope boundaries

## 1. Why this folder exists

We reviewed 21 proposed transfers into the existing voxel engine, then cross-audited the results against code and another AI’s reports. The completed register is **18 rejected as stated, 3 ADAPT**. Most rejections concern missing prerequisites, incorrect analogies or unsupported performance promises—not worthless papers.

The user then requested a ground-up proposal prioritizing appearance and performance, without preserving the old architecture. This folder answers that request. The user separately selected **workspace-only cleanup**: preserve engine code, papers, research evidence and history.

The new proposal may choose a different renderer, backend, data layout and feature policy. It must not pretend that hypothetical compatibility is an implemented or measured result.

## 2. Working assumptions, not hidden requirements

- An editable, large voxel world is still the product. Terrain, buildings, emissive materials, water/glass and moving-camera scenes are representative.
- A modern discrete GPU is the initial performance target. The proposed main renderer needs hardware ray tracing; a non-RT product tier requires a separate scope decision.
- Fidelity includes temporal stability, contact detail, reflections, transmission and edit response—not just a high-sample screenshot.
- World identity and material semantics should survive representation changes. Visual blockiness versus smooth surface extraction is an art/product choice still to approve.
- Native host/API control is acceptable if the maintenance cost is accepted. Vulkan-first/Rust and the candidate shader toolchain are proposals, not inherited dependencies.
- No existing engine source is modified or imported into a new executable in this task.

## 3. Decisions required before implementation

| Decision | Why it changes the design | Required artifact |
|---|---|---|
| GPU models, OS, driver/API floor, VRAM | Determines ray tracing, precision, subgroup/matrix options and realistic memory capacity | Capability/limits matrix from actual target devices |
| Resolution, refresh/frame budget, input latency | Determines tracing/reconstruction allocation and performance acceptance | Native/internal resolutions and preset targets |
| Visible world scale, content density and motion | Determines streaming, acceleration structures and temporal state | Representative scene/camera traces |
| Edit rate and worst-case edit pattern | Determines mesh/BLAS rebuild viability and commit latency | Boundary edits, burst destruction, emissive changes and reload traces |
| Art direction | Defines block silhouettes, textures, water, lighting and acceptable approximations | Fixed reference views plus motion sequences |
| Memory and startup budgets | Limits mesh/voxel duplication, reservoirs, cold compilation and warm-up | Per-pool budgets and loading-latency targets |
| Determinism requirements | Separates authoritative gameplay from visual sampling and asynchronous rendering | World/render version and reproducibility contract |
| Supported reconstruction vendors | Changes guide requirements, licensing, testing and deployment | SDK/backend feature matrix with a native fallback policy |
| Team/time/platform capacity | Native API + PT + streaming is a major project | Explicitly approved first vertical slice and excluded scope |

Do not infer any of these from the old laptop alone. The old machine is a valuable stress target if retained, not proof it can support every proposed quality tier.

## 4. What the new engine needs

### Data and reference material

- Deterministic voxel/material scenes with transparent, cutout, emissive, thin and repeated geometry.
- Replayable edit streams and camera/light paths, including reloads and interrupted jobs.
- A material/light specification with units, opacity rules and reference evaluation.
- An intersection oracle and a high-sample lighting reference for controlled scenes.
- The reviewed papers, their actual algorithm assumptions, and any separately reviewed direct-light/denoising foundations needed for implementation.

### Tooling and access

- Actual target hardware; Rust/native build toolchain; chosen shader compiler; API validation and GPU capture/profiling tools.
- Compiler intermediate/native-code and resource-usage inspection where available.
- A repeatable cold/warm compilation test, frame-timeline capture and paired image/sequence comparison.
- License review for third-party code, SDKs, assets and datasets. Paper availability does not grant permission to copy an implementation.
- A pinned dependency/source provenance record. No dependency upgrade is implicitly authorized by this document.

### Design contracts before optimizations

Persistent identity; versioned jobs; atomic publication; GPU resource retirement; material and opacity parity across raster/ray paths; history reset/production; queue overflow; unknown-residency behavior; and an observable pass graph.

## 5. What transfers from the current engine

**Transfer:** engineering lessons, deterministic scene ideas, domain knowledge, material requirements where still desired, and carefully audited algorithms/tests as reference concepts.

**Do not inherit automatically:** the 64³ chunk layout, packed water nibble, atlas dimensions, fixed water settings, shader-body duplication, wgpu feature limits, old flag names, old timing thresholds or renderer-specific bit-exact image baselines.

The four-body no-merge rule continues to protect the frozen old engine. In the new engine, we should design modular shading with tested per-call receiver semantics rather than reproduce that constraint. We still preserve the underlying lesson: code paths that look similar may have different outputs, ordering or guide ownership.

New exact controls will be declared for the new engine. Approximate rendering changes require separate image/temporal quality acceptance; voxel/material IDs and resource lifetimes remain exact.

## 6. Evidence hierarchy and limitations

Read [Batch 09](../research-review/batch-09-synthesis.md) with its [correction addendum](../research-review/cross-audit-other-ai.md). In particular, default-off AO still has a producer dispatch in the old split path. Older default-cost wording is superseded.

The retained corpus has 16 PDFs / 13 groups / 199 canonical pages. Historical complete reading is recorded in the research folder. Current integrity/model/source checks are not Rust/shader/API/GPU validation. No new hardware research or implementation benchmark was performed for this proposition, and it is not an exhaustive survey of all graphics research available today.

Some historical external PDFs, rendered images and original API archives are unavailable; extracted texts/unpacked sources survive. Do not equate them with newly verified archives. The engine’s 295 files match a historical baseline, but Git metadata is absent. Old handoff “next action” instructions are historical, not instructions to start modifying that tree.
