# What worked, what did not, and where effort paid poorly

**Evidence review for a single-model new-engine project · 2026-09-22**

This is the practical supplement to the original [engineering lessons](../ENGINEERING-LESSONS.md). It is based on the old engine's records and targeted source inspection, not generic “best practices.” Portable source snapshots are in [reference/](reference/README.md).

## Read the labels first

- **Historical measurement:** a recorded result from the old project/hardware. It was not rerun here; stated scene/configuration limitations still apply.
- **Current source finding:** supported by the preserved code or completed audit, not proof of device execution.
- **Pending:** acceptance or benefit has not been demonstrated. Pending does not mean impossible or zero benefit.
- **Recommendation:** a rule for future work, not an already measured outcome.

There is no complete timesheet, token bill or development-cost ledger. We can identify documented expensive detours and poor runtime returns, but cannot honestly name the largest total monetary cost or compute a complete ROI ranking. Do not multiply or rank incomparable paper percentages.

## 1. What was worth keeping

| Mechanism/practice | What worked and how we know | Transfer to the new engine |
|---|---|---|
| Hierarchical skipping, local traversal caches, exact run sharing and adaptive material palettes | The code/audit establishes these mechanisms already exist; they invalidate claims that the baseline was naive. No isolated GPU benefit for each was newly measured. [Audit](reference/audit-corrections.txt), disposition audit. | Establish an efficient simple reference before adding a replacement. Measure the real baseline, including existing reuse. |
| Worker-thread pipeline compilation | Historical same-key cold compilation could grow from 4.8 s to 133 s; moving compilation off the message thread allowed a slow compile to finish without blocking normal window event handling. Current `Renderer::ensure_spec` uses a named worker thread. [ADR 0010](reference/old-adr-0010-compilation.txt). | High-value responsiveness architecture, not a claim that compilation itself became fast. Separate startup, UI liveness and frame-time budgets. |
| Correct temporal-history lifecycle | The old DLSS shimmer investigation found history being reset every frame. Current code records `taa_on || dlss_ran || rr_ran`, rather than `taa_on` alone. [Lessons](reference/old-lessons.txt), “A temporal algorithm with no temporal benefit is being reset.” | Make fresh production/reset/age visible before testing clever reuse. Good buffers cannot help an algorithm forced to forget them. |
| DLSS Super Resolution on the historical laptop workload | Recorded Quality evaluate cost 2.66 ms; recorded GPU total 8.24 ms versus native 10.45 ms. Lower internal-resolution work more than offset evaluation in those runs. The three compared modes used different vantages, so this is not a clean paired universal speedup. [DLSS record](reference/old-dlss-open-questions.txt), “Already measured.” | Reconstruction can buy useful tracing/shading budget. Re-measure matched scenes and moving quality on the new targets; do not import the percentage or use the old numbers as a product target. |
| Reduced-resolution water secondary legs | Historical full resolution added about 0.15 ms; differences reached 4/255 at a reflection vantage and 15/255 near shallows. The old project kept reduced-resolution legs as its cost/quality choice. [DLSS record](reference/old-dlss-open-questions.txt); [water ADR](reference/old-adr-0002-water.txt). | Spend samples where they affect the chosen image. Near-field refraction and distant reflections can justify different budgets; do not claim all full-resolution work is wasted. |
| Paired scenes, counters, ablations and actual error logs | These corrected wrong-camera diffs, detected temporal work doing the opposite of its goal, isolated foam, and distinguished a window timeout from a shader limit. [Lessons](reference/old-lessons.txt). | Usually cheaper and more decisive than a speculative rewrite. Invest early in the ability to observe the exact pipeline being tested. |

## 2. The most instructive poor-return cases

### R01 — Ray Reconstruction: large observed runtime cost, benefit still unjudged

**Attempt:** integrate RR to improve reflection reconstruction/stability.

**Recorded cost:** on the RTX 3050 Laptop, Quality evaluate was **12.16 ms RR versus 2.66 ms SR**, about **4.6× the evaluate cost**, not 4.6× whole-frame cost. Recorded GPU totals were 17.56 ms RR, 8.24 ms SR and 10.45 ms native.

**Benefit status:** the required motion/reflection-quality judgment was still open. Integration work and cheaper ray/filter stages did not establish a net product win. This is **high cost with unproven benefit**, not proof that RR provides no benefit on every engine/GPU.

**How learned:** adding `q::DLSS` exposed neural evaluation that the earlier timing display omitted; examining guide assumptions showed further open issues. The recorded mode runs used different vantages/chunk counts, so precise paired frame comparisons remain limited.

**Next-time rule:** instrument vendor evaluation and whole frame first; validate guides and motion; require a matched reference-quality case before making an expensive reconstruction tier default. Keep capability/guide infrastructure separate from endorsement of its current algorithm cost.

**Source:** [old DLSS open questions](reference/old-dlss-open-questions.txt), items 1–6 and “Already measured”; [RR guide ADR](reference/old-adr-0008-rr-guides.txt). These costs are historical, not a new benchmark or a ban on future RR.

### R02 — Screen-space reflected legs: implementation complexity, no useful measured gain, visual regression

**Attempt:** screen-space ray marching for glass/water reflection, then fall back to the existing world trace on misses.

**Outcome:** the old ADR records **“0.0 ms”** benefit in the tested setup and upright ghost terrain over water. The visibility buffer described the camera-visible surface, not necessarily the surface the reflected ray needed. Misses paid both the screen-space attempt and the original trace. The targeted glass premium also shrank from an invalid cross-scene claim of 17 ms to about 1 ms under controlled same-seed comparison.

**How learned:** on-hardware A/B and image inspection; the paths were subsequently retired behind literal false enables, which remain visible in current source. A string/call-site test could still pass even with the feature intentionally inactive.

**Next-time rule:** first prove the representation contains the information the query needs. Estimate fallback frequency and full cost before building the accelerator. Do not re-enable a retired path just because its flag still parses.

**Source:** [ADR 0005](reference/old-adr-0005-temporal.txt), “Measured outcome” and “Retirement.” This condemns that implementation/transfer, not every modern SSR algorithm.

### R03 — Shader “cliff” debugging: many tests answering the wrong question

**Recorded effort:** the lessons log describes **roughly 30 windowed bisect runs** before the parked AO path compiled in 79 s and rendered after compilation was moved off the main thread. A separate equivalent one-line edit had a 4.8 s → 133 s cold-compile change at a matched key.

**What failed:** treating an unresponsive window as proof of a shader-size ceiling; treating a live title-screen process as proof of rendered frames; sometimes comparing different specialization keys.

**How learned:** OS AppHang records, timed matched-key headless compilation and instrumentation at the actual pipeline factory. Some earlier intermittent failures also involved machine state; they must not all be assigned one cause from a shared exit code.

**Next-time rule:** after a reproducible silent failure, identify the failing layer before repeated shader surgery. Time cold compilation, inspect OS/API diagnostics, log generated modules/keys and confirm actual frames. Recheck controls when a revert fails to restore behavior.

**Important correction:** old ADR prose about all overrides reaching the driver unchanged/full-module compilation is not generally accurate for the inspected pinned Naga route. The later audit found entry compaction and override evaluation. A universal five-second OS kill rule and a universal one-line-only formatting rule are not established.

**Sources:** [lessons](reference/old-lessons.txt), final compile-cliff entries; [ADR 0010](reference/old-adr-0010-compilation.txt); [compiler audit](reference/shader-organization-review.txt), §4.3–4.4.

### R04 — Temporal shadow reuse: added work without demonstrated reuse value

**Historical observation:** paired testing recorded **+0.15 / +0.23 ms** at the reflection vantage. Separate guide reconstruction in reprojection and primary tracing added overhead.

**Later source finding:** the inspected temporal textures lack a fresh-history producer. Reprojection only carries previously valid entries; fresh blocker distances go to another texture. Cleared histories cannot bootstrap through those paths.

**How learned:** first timing/cost inspection, then tracing actual host resource bindings and all relevant producers—not treating a shader variable name as a resource identity.

**Next-time rule:** validate live cache population and hit/miss/fallback counters before optimizing the cache or adding spatial/layer reuse. Price unconditional work against conditional savings.

**Do not conflate:** the later source finding does not establish which historical revision ran or prove the cause of those old timings. Static camera alone does not prove maximal valid reuse.

**Sources:** [testing record](reference/old-testing.txt), “Both batch-103 temporal flags”; [cross-audit](reference/audit-corrections.txt), P05/P06 and agreement limits.

### R05 — Temporal Hi-Z: extra pass cost, worse stated work counter

**Historical observation:** **+0.06 / +0.10 ms**. Baseline marched 68,896 pairs and deferred 104,413; the temporal arm marched 69,200 and deferred 104,109—**304 fewer deferrals** in that test.

**How learned:** deterministic purpose-specific counters, not a subjective image difference. The later testing record is more informative than the earlier ADR’s suggested motion/LoD win case.

**Next-time rule:** state which counter should improve before timing. If it moves the wrong way, investigate or reject that workload before investing in micro-optimization. The record does not settle a universal failure or prove a sign bug.

**Sources:** [testing record](reference/old-testing.txt); [historical ADR](reference/old-adr-0005-temporal.txt).

### R06 — The “65% lost reflection” plan: a large premise from a small effective term

**Attempt:** plan work around the literal `0.65` as if it removed 65% of reflection detail.

**Actual evidence:** the coefficient was `roughness × 0.65`; recorded roughness peaked near 0.036, so the effective coefficient was below 0.03. A/B across six vantages recorded a maximum channel difference of 10/255. The reference describes it as sub-perceptual in those views; that is not a universal perceptual threshold.

**How learned:** measure the multiplier and compare the actual output of the suspect expression, not only a compelling-looking upstream debug view.

**Next-time rule:** validate magnitude and visual contribution before scoping a rewrite. Correctly choosing not to optimize can be the best outcome. Do not confuse removing this mix with the separate variance-based normal filtering decision.

**Sources:** [lessons](reference/old-lessons.txt), “A coefficient is the literal times the multiplier”; [RR guide ADR](reference/old-adr-0008-rr-guides.txt).

### R07 — Invalid A/B experiments: expensive green results with no information

**Examples:** comparing different vantages/worlds; flags whose prerequisites were off; feature-bit collisions that made both arms enabled; title-screen-only liveness; filtering a flaky test and removing the shared Rayon-pool contention that triggered it.

**How learned:** inspect effective flags/configs and full command records; run known positive/negative controls; reproduce the original unfiltered configuration. The bit-collision correction voided earlier tint/probe measurements; those numbers cannot be salvaged by averaging them.

**Next-time rule:** one serialized effective configuration, uniqueness/layout checks, reachable feature scenes, explicit intentional-inert classification and reproducible scheduling conditions. A zero image difference is not proof of either success or dead code by itself.

**Sources:** [lessons](reference/old-lessons.txt); [ADR 0005 bit-collision correction](reference/old-adr-0005-temporal.txt).

### R08 — Repeating the full test suite merely to format output

**Recorded effort:** the old log reports **3 min 36 s** for a twice-invoked suite pattern versus **53 s** for one invocation captured and analyzed. These are historical elapsed observations, not a universal speed ratio attributable only to one factor.

**Next-time rule:** run once, preserve exit code/stdout/stderr, then count/filter the file. Focused tests during iteration; broader/configuration tests at the appropriate acceptance point. Do not save cost by hiding test failures or skipping essential gates.

**Source:** [lessons](reference/old-lessons.txt), “Do not invoke the suite twice in one command.”

### R09 — Re-deriving SDK conventions and guessing at native failures

**Effort:** several failed rounds inferring depth/motion/jitter conventions from wrapper source while the SDK guide was already available; repeated Rust-side guesses at a native failure.

**How learned:** vendor documentation resolved convention questions; API validation named missing device-address enablement. A wrapper convenience getter also need not be the vendor’s recommended resolution.

**Next-time rule:** read the relevant vendor contract, query actual returned settings, and validate the native API path before changing shader math. “Feature exposed,” “feature enabled,” “legal kernel” and “correct device execution” are separate facts.

**Source:** [lessons](reference/old-lessons.txt), vendor spec/native crash/convenience-getter entries.

### R10 — Polishing a fix that had met its goal (new engine, 4B filter fix, 2026-09-26)

**Attempt:** stop the filter deleting night light (4B's G4). The first laptop run met that goal: energy change 0.0000 where the old filter lost 5–25%, and M3's day gate 40 of 40, one arm better than at 3G.

**What followed:** two guards missed. One day check (D1, noon, first frame after a reset) came in 0.2% over its absolute limit, and the cost check read +0.68 ms against a limit of "today + 0.05 ms". The proposal was another round: change σ, cut the cost and rerun three checks, about 2 hours of the user's laptop plus a cloud session, while no plausible result would have changed whether the fix was worth keeping.

**What the criteria hid:** D1 was judged against an absolute limit, not against today's filter. Against today's filter the first frame after a reset is 20–36% noisier at every day hour, and the converged history is 11–22% cleaner (`f4_3e_criteria.log` against `test_gpu_3g_denoise.log`). That is a real trade for the user to judge by eye in the viewer in a minute; a 0.2% threshold miss is not.

**Why it went wrong:** no split between the goal and the guards; guards with no tolerance, judged against fixed limits instead of today's default; a correctness fix given a zero cost budget; a decision rule that said what to do on a pass but not on a small miss; no count of the user's laptop hours.

**Next-time rule:** CLAUDE.md "Good enough beats perfect" (O-005). Goal and guards frozen apart, guards against today's default with a tolerance, a cost budget from the milestone's headroom. When the goal passes, propose keeping the change with the misses recorded. Rerun only if the result could flip keep / drop; bundle laptop checks; one fix round at most.

**Sources:** [filter record](changes/2026-09-26-4b-filter-energy.md); `engine/results/local-run/2026-09-26_1125-4b-filter/`.

## 3. Cases that are not honest “wins” or “waste” yet

### AO: useful appearance option, unresolved precise runtime value

The ADR records a visible terraces effect and an exact off capture for that historical comparison. Its four-round timing difference, **+0.12 ms with standard error 0.153 ms**, was not a demonstrated cost difference. Later code audit showed off disables sampling, **not producer dispatch**.

Do not say AO is free, that disabling it saves nothing, or that skipping its producer is already a proven win. Separate appearance value, scheduled work, allocation and measured cost. [AO ADR](reference/old-adr-0011-ao.txt); [A09-01 correction](reference/audit-corrections.txt).

### Foam: strong localization, implementation acceptance still incomplete

The recorded fixed-pose ablation removed the squares while keeping base noise; a slant-to-vertical depth change did not. Source analysis identified an unsmoothed animated gain multiplying smooth noise. Shared-corner gain interpolation addresses that mathematical discontinuity while retaining animation.

The test reads the assembled shader and validates with Naga, but that test’s existence is not evidence it was run here. The retained ADR still says GPU acceptance pending; the change adds three phase hashes and three sine evaluations per call. No net timing or exact off-arm acceptance should be invented. [Foam ADR](reference/old-adr-0012-foam.txt).

### Transform sharing, paging, neural lighting and the rest of the paper stack

Most were reviewed proposals, **not implemented experiments with known development cost**. Rejecting an inappropriate transfer avoids potential work; it does not prove “we spent X implementing it” or that the method cannot work in a new architecture. Even the three ADAPT outcomes remain hypotheses, not measured engine wins. [Audit](reference/audit-corrections.txt).

## 4. Practical prioritization for the next model

1. **First:** expose the right counters/configuration, correct units/identity/reset/lifetime, and reproduce the user’s actual frame. These enabled most successful diagnoses.
2. **Then:** fix a localized correctness error or remove demonstrated unnecessary work under a tested contract.
3. **Then:** optimize the measured bottleneck against the current efficient baseline; include total cost and quality.
4. **Last:** new temporal layers, neural integration, cross-LoD transport, paging/queue complexity or compiler/body consolidation. Require prerequisites and a bounded falsification test before committing to integration.
5. **Do not pay twice:** one active experiment, one result record, one current handoff, one full log per run. No swarms or parallel model reviews.
6. **Stop at good enough:** once a task's goal passes, propose keeping it with the guard misses recorded; another round needs a result that could flip keep / drop (R10).

For future ROI, record development sessions/iterations and expensive run durations when available, then report runtime/memory/quality gains separately. Never pretend saved GPU milliseconds measure developer effort, or that missing effort records justify made-up totals.

## 5. Corrections that must travel with these lessons

- Off contribution is not off production: AO correction governs older cost prose.
- Shadow history source liveness and historical timing causality are different questions.
- RR evaluate cost is not whole-frame cost; the recorded three-mode comparison used different vantages; quality benefit remained pending.
- Compile-time AppHang is not a universal shader-size limit; the pinned compiler route was more nuanced than the old ADR explanation.
- The old lesson saying `src/bin/` was absent is historical; this preserved snapshot contains it. Old “mixed-EOL” prose was later corrected by byte inspection. Do not treat the raw log as a timeless specification.
- No new GPU, Rust, Naga or performance run occurred while authoring this starter. Fresh lexical checks and document preservation are the only new verification claimed here.
