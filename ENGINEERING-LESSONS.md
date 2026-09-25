# Engineering lessons from building and auditing the current engine

These are practical lessons for the new engine, not instructions to copy every old workaround. Historical measurements below are project records, **not rerun results**. When a diagnosis was superseded, use the corrected explanation.

Primary reading: [old lessons log](../recovered-codebase/docs/lessons.md), [old behavioral contract](../recovered-codebase/AGENTS.md), [cross-audit corrections](../research-review/cross-audit-other-ai.md).

## 1. A feature flag has three jobs: produce, consume and retain

**What happened:** our research initially treated directional AO’s default-off appearance as zero producer work. The split renderer still dispatched the five-tap producer; only `ao_sample` bypassed consumption.

**New rule:** every optional feature declares producer scheduling, consumers, resource allocation and history lifetime separately. Disabling the appearance is not proof that computation or VRAM disappears.

**Acceptance exercise:** toggle the feature and observe scheduled dispatches, resource bytes, output and re-enable behavior. Show that the diagnostic can detect a deliberately ungated producer. Source: [A09-01](../research-review/cross-audit-other-ai.md) and [AO ADR](../recovered-codebase/docs/adr/0011-directional-ambient-occlusion.md). The ADR’s historical noise-floor result is not a new measurement or proof that today’s dispatch is free.

## 2. Temporal memory must first be allowed to exist

**What happened:** the historical DLSS reset bug discarded history every frame despite plausible buffers. Separately, the current inspected shadow history has no fresh producer into its temporal textures: cleared guide depths cannot bootstrap through reprojection alone.

**New rule:** instrument production, age, reset reason, valid population, accepted reuse and fresh fallback before tuning spatial/temporal sampling.

**Acceptance exercise:** cold start, static camera, camera cut, disable/re-enable, moving caster, edit, resize and reload. Show valid state can appear and survive when appropriate, and disappear when required. Do not assume static camera means maximum valid reuse. Sources: [lessons log](../recovered-codebase/docs/lessons.md), [shadow dataflow review](../research-review/batch-06-restir-spatial-reuse.md).

## 3. Test a reachable effect against the same scene

**What happened:** zero-pixel comparisons sometimes used a view that never reached the feature. Another test compared a changed vantage against the default camera, guaranteeing a meaningless large difference.

**New rule:** a capture records the full input snapshot: camera, animation, light state, saved settings, content pack, build features, specialization key, internal/output resolution and temporal reset. Baseline and candidate differ only in the declared variable.

**Acceptance exercise:** prove the feature-on/off fork is reachable in its positive-control scene, then test whether the intended exact optimization preserves that result. For an intentionally output-identical scheduling feature, use execution/cost evidence rather than demanding an image difference. Source: [lessons log](../recovered-codebase/docs/lessons.md).

## 4. A source-string test is not a behavior test

**What happened:** pins matched comments rather than the mechanism; a “no jitter token” assertion missed jitter inside a called helper. Missing-input early returns could make tests vacuous.

**New rule:** test semantic relationships, actual selected/generated shader modules and runtime behavior at the appropriate layer. Use source pins only as narrow regression guards, clearly labeled.

**Acceptance exercise:** mutation/negative controls must cause failures: wrong guide convention, absent resource, stale generation, broken producer. Keep pure CPU tests separate from shader/API/device integration tests. Source: [lessons log](../recovered-codebase/docs/lessons.md).

## 5. Compile time is a resource budget

**What happened:** a same-key historical one-line equivalent shader edit changed cold compilation from 4.8 s to 133 s. The visible failure was diagnosed as a shader-size/driver cliff before logs identified an unresponsive-window timeout; moving compilation off the message thread addressed the hang.

**New rule:** asynchronous pipeline compilation from the beginning, bounded variants, progress/cancellation and cold/warm timing. Record the exact generated module, selected entry, constants, toolchain and key. A live process/title screen is not proof of rendered frames.

**Important qualification:** [ADR 0010](../recovered-codebase/docs/adr/0010-compile-pipelines-off-the-main-thread.md) contains historical compiler-mechanism prose. The [pinned compiler audit](../research-review/batch-05-textures-shader-organization.md) found entry compaction/override processing; do not copy the blanket claim that every specialization necessarily compiles the complete original module unchanged. Nor is a universal five-second OS termination rule established.

**Acceptance exercise:** a deliberately slow cold compile must not freeze the event loop; confirm actual frames for the selected key after completion. Smaller source does not guarantee less compiler work.

## 6. Read the layer that actually failed

**What happened:** a native/vendor failure had no Rust panic; API validation identified missing device capability setup. Other symptoms were OS AppHang events, not shader validation errors.

**New rule:** route failures to the right evidence: API validation, GPU fault/capture tooling, OS event records, compiler diagnostics and application errors. Read the vendor integration contract before inferring units from wrapper code.

**Acceptance exercise:** intentionally misconfigure a supported test case and confirm the right diagnostic is captured. Device capability availability, device enablement, legal shader usage and successful execution are four distinct gates. Source: [lessons log](../recovered-codebase/docs/lessons.md).

## 7. Headless, default-feature and windowed are different configurations

**What happened:** native headless captures and a windowed run with saved preferences exercised different settings, devices or shader keys. A green non-default build did not validate the user’s actual path.

**New rule:** one serialized render configuration usable by tests, headless rendering and windowed runs, with explicit device/vendor differences. Test the actual product feature matrix as well as the simplest reference path.

**Acceptance exercise:** log effective settings and generated keys; compare matched modes only. A vendor reconstruction result is not expected to be pixel-identical to a native path. Source: [lessons log](../recovered-codebase/docs/lessons.md) and [old AGENTS](../recovered-codebase/AGENTS.md).

## 8. Thermal drift can reverse an apparent result

**What happened:** sequential laptop benchmarks showed unrelated traversal time changing with run order. Paired context and interleaving changed the interpretation.

**New rule:** repeated interleaved/randomized arms, matched warm-up, recorded power/thermal state and uncertainty. Monitor negative-control stages that should not be affected. A changed negative control prompts investigation; it does not alone prove a particular cause.

**Acceptance exercise:** run A/A as well as A/B, report dispersion and tails, preserve raw logs. Fewer source operations do not make a slower measurement impossible: clocks, scheduling, caches and variance can change it. Source: [lessons log](../recovered-codebase/docs/lessons.md).

## 9. Optimize the total producer/consumer system

**What happened:** proposed reuse added guide reconstruction, passes, scratch or histories while claiming only the skipped work. Source/code models also showed that roots/L1 caching left leaf/material traffic intact.

**New rule:** include preprocessing, new loads, barriers, allocations, uploads, invalidation, warm-up and fallback. Measure critical path rather than summing overlapping timers. A timestamp family and its children must not both be charged to the same total.

**Qualification:** historical shadow penalties do not prove the current missing-producer gap caused them. Preserve observations while separating unverified causality. Sources: [scheduling review](../research-review/batch-04-scheduling-queues.md), [shadow review](../research-review/batch-06-restir-spatial-reuse.md).

## 10. Guide ownership and mathematical quantities are interfaces

**What happened:** reconstruction needed actual material guides and appropriate reflection information, not an unrelated surface-motion guess. The old review also found that radiance, probe factors and AO were being treated as interchangeable.

**New rule:** specify each quantity’s meaning, units, coordinate frame, producer, owner and consumer. Secondary hits cannot overwrite primary-surface guides accidentally. Define roughness rather than reuse an unrelated variance term by name.

**Acceptance exercise:** analytic motion/depth scenes and diagnostic albedo/normal/roughness/specular-distance captures; verify each store’s ownership and dispatch order. Source: [RR guide ADR](../recovered-codebase/docs/adr/0008-ray-reconstruction-material-guides.md). Its open roughness/convention questions remain historical uncertainties, not settled facts to copy.

## 11. Smooth inputs can become discontinuous downstream

**What happened:** smooth foam noise was multiplied by cell-local animated gain, reintroducing square boundaries. Interpolating gains at shared corners addressed the identified mechanism; interpolating wrapped phase angles is a different operation.

**New rule:** test the whole composition, derivatives/continuity where relevant, negative coordinates, boundaries and time. Do not remove animation or detail merely to make a defect disappear unless that is the intended art change.

**Acceptance exercise:** neighboring-cell edges/vertices over time plus fixed-pose and animated GPU captures. CPU continuity is not GPU bit-identity or performance proof. Source: [ADR 0012](../recovered-codebase/docs/adr/0012-continuous-shore-foam.md), explicitly marked GPU acceptance pending in the retained record.

## 12. Preserve evidence; retire instructions, not knowledge

**What happened:** historical prose became stale, some reconstructed binaries were unavailable, and archive links outlived their original paths. Repeatedly “cleaning” evidence would have made the mistakes harder to understand.

**New rule:** immutable experiment inputs/results and superseding decisions; a short current entry point; paths/symbols rather than line numbers. Generated caches are disposable. Original papers, negative results, captured compiler artifacts and meaningful patches are not disposable merely because they are old.

**Acceptance exercise:** hashes and a deletion manifest before/after cleanup; identify missing evidence honestly. Run expensive suites once, preserve the full exit status/log, then analyze the log rather than invoking the suite again just to count output. Sources: [cross-audit](../research-review/cross-audit-other-ai.md), [reconstruction records](../reconstruction/README.md).

## What we deliberately do not copy

The new engine does not inherit fixed old water flags, exact bit layouts, four literal shading bodies, one-line shader formatting folklore or old numerical timing thresholds. It inherits **explicit semantics, controlled experiments, real lifecycle tests and truthful evidence boundaries**. New modular shading may share implementation where receiver semantics and generated-code behavior are actually preserved.
