# Validation, budgets and acceptance contracts

## 1. The optimization objective

Maintain a measured **quality / frame-time / memory / edit-latency frontier**. Compare either quality at a matched cost or cost at a predeclared matched quality. Reject claims based only on rays/second, compressed node count, sampler-call count or paper percentages.

No target is locked yet. Before implementation fill in:

| Budget | Required decision |
|---|---|
| Display and internal resolution | Per preset, including dynamic-resolution policy |
| Frame/latency | Target rate, CPU/GPU critical-path headroom, allowed tail latency and interaction latency |
| Memory | Total device/process budget; explicit voxel, mesh, acceleration, texture, reservoir, history, staging and optional neural allocations |
| Updates | Normal/burst edit rate, edit-to-visible latency, queue capacity and overload fallback |
| Startup | Cold/warm compilation, pipeline/streaming progress, cancellation and first useful frame |
| Quality | Reference scenes/sequences, material/art direction, permitted bias/noise/ghosting and failure thresholds |

Frame duration implied by a target rate is `1000 / target_fps` milliseconds; that is a budget, not a performance forecast. CPU and GPU overlap, so their stage sums are not automatically frame latency.

## 2. Memory accounting that must exist from day one

Record **live, reserved, high-water and transient peak** bytes separately. Count all simultaneous snapshots, not only the new one after a swap.

- World data: occupancy + materials + hierarchy + sharing dictionaries/handles/transforms.
- Derived data: extracted meshes/indices + BLAS/TLAS and build scratch + correspondence maps.
- Temporal data: active pixels × records/layers × temporal versions × record stride, plus compaction/ownership/scratch metadata.
- Uploads: staging capacity, pending copies, retirement queues and duplicated old/new resources.
- Neural tier: encoded features, weights/EMA, optimizer state, targets, activations/gradients, batch buffers and tracing/training scratch.

Report storage savings against the entire relevant resident system. Sparse layers/paging must count allocation metadata and worst-case occupancy, not just the ideal sparse scene. Define a deterministic overflow response for every pool.

## 3. Correctness classes

### Exact invariants

Material identity, world edits/save-load, handle generation, queue/resource lifetime, conservative skip safety, categorical lookups and declared off-control output must preserve their contracts. The raster/ray representation must match its committed scene version. Define floating-point/tie conventions for intersection comparisons rather than silently treating different primitives as bit-identical.

### Controlled approximation

LoD geometry, finite-sample lighting, denoising, biased reuse options, neural continuation and approximate effects fields have explicit reference/error contracts. Judge spatial error **and** time-dependent artifacts. A material ID is never a controlled approximation merely because a numeric format is smaller.

### Evidence levels

Keep separate: mathematical proof, CPU model, source pin, generated-shader validation, API validation, device correctness, image/sequence comparison and performance measurement. Passing one does not imply the next.

## 4. Minimum scene and event matrix

| Scene/event | Main failure it must reveal |
|---|---|
| Sparse outdoor terrain with long empty intervals | Traversal/build amortization, conservative bound failures |
| Dense interior with thin walls and tiny emitters | Visibility leaks, missed light support, unstable indirect illumination |
| Repeated structures with differing materials | Incorrect sharing, transformed attribute rank, hidden derived-data cost |
| Foliage/cutouts and moving occluders | Raster/ray opacity mismatch and temporal disocclusion |
| Water/glass with rough and smooth reflection/transmission | Wrong normals/interfaces/guides, excessive denoising, path cost |
| LoD switches with topology changes | Invalid correspondence, silhouette mismatch, history persistence |
| Boundary insertion/deletion and emitter removal | Neighbor invalidation, stale lighting, obsolete jobs |
| Burst destruction, job cancellation and immediate reload | Queue overflow, frame spikes, allocation generation mistakes |
| Camera cut, resize and feature disable/re-enable | Invalid initialization/reset and stale target consumption |
| Bright/dark light changes and cold neural cache | Training lag, energy bias, fallback behavior |
| VRAM pressure and offscreen reflection/caster demand | Missing-space policy and insufficient residency coverage |

Each scene has a reproducible config/input trace and a positive-control mutation demonstrating that the test can fail. No single vantage or still image certifies an engine feature.

## 5. Measurement protocol

1. Freeze scene, camera, light/animation, edits, seed policy, pack, build features, shader keys, device/driver and resolution.
2. Separate cold-start from warm steady-state and reset candidate-owned history/queues/weights consistently. Use paired seeds where appropriate and additional independent seeds to avoid overfitting stochastic results.
3. Run A/A controls, then repeated paired/interleaved or randomized A/B trials. Record power/thermal state and dispersion, medians and relevant tails.
4. Capture total frame critical path and constituent spans without double-counting nested/overlapping intervals. Include CPU, uploads, builds, compiler time and edit-to-visible latency.
5. Capture images and sequences against the declared reference. Use meaningful error metrics plus motion/ghosting/contact-detail review; a blur that lowers one error score is not automatically better appearance.
6. Inspect generated code/resource use when claiming arithmetic, occupancy, cache or synchronization gains. Logical loads are not measured DRAM transactions.
7. Predeclare the meaningful improvement/error thresholds and trial count before inspecting candidate results. Do not substitute a universal guessed millisecond threshold.

## 6. Acceptance card required for each future change

- **Hypothesis:** what specific bottleneck/quality issue is addressed?
- **Prerequisites:** capabilities, identities, estimator assumptions and supported materials/LoDs.
- **Control and candidate:** exactly what differs; which state is reset and how.
- **Invariants:** exact outputs/lifetimes and admitted approximation.
- **Workload:** normal and adversarial scene/event traces.
- **Measurements:** setup/runtime/memory/quality/latency, plus expected counter changes.
- **Pass and stop rules:** numerical budgets selected in advance; any unsafe lifetime or false-empty certificate stops immediately.
- **Fallback:** unsupported device, missing residency, bad correspondence, overflow or untrained cache.
- **Revert:** remove the isolated path or restore the reference configuration; preserve evidence and negative results.

A candidate that reduces kernel time but worsens the selected total frame/edit/quality budget does not pass. A sound negative result is useful and should remain in the record.

## 7. Proposed operational safeguards

Pure world/math tests remain device-free; GPU tests use a separately identified integration harness. Record one full test run and its true exit status before filtering logs. Exercise default product features and matched windowed/headless configurations. If the existing laptop is used, retain the practical `-j 2` build-concurrency limit until re-evaluated.

No benchmarks, GPU tests or engine builds were performed while creating these documents. The cleanup preservation/link checks are document/workspace verification only.
