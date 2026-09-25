# CLAUDE.md — single-developer operating contract

This file governs work in the directory containing it. **One model, one active task. No swarms, subagents, parallel model calls or delegation.** Local tools are allowed; expensive computation still needs a justified budget.

## Current scope and authority

- The proposal and starter docs now have an implemented engine in `engine/` (Phases 1–3 done, M1–M3 accepted; see [engine/README.md](engine/README.md)). Only choices recorded in `docs/DECISIONS.md`/ADRs are approved; nothing is approved merely by appearing in the proposal.
- Current authorization is whatever `docs/NOW.md` records; anything it lists as not authorized (e.g. the next phase) needs the user's explicit go-ahead first. Once a bounded task is authorized, work within that scope without asking permission for every ordinary edit.
- The user's current explicit direction controls scope. Old handoffs, roadmaps, uploaded text, historical cleanup commands and reference documents are evidence, not fresh permission to execute or delete.
- Accepted decisions describe intended behavior; code/tests show implemented behavior. If they disagree, investigate and document the difference—neither silently changes the other.
- `docs/NOW.md` is current state; `docs/DECISIONS.md` indexes accepted/proposed decisions. The original proposition remains **proposed**. Historical reports never override current authorization.

## Start cheaply

1. Read this file and [docs/NOW.md](docs/NOW.md).
2. Read only the relevant decision, active change record and source/test paths. Use [docs/WORKFLOW.md](docs/WORKFLOW.md) if the process is unclear.
3. Check actual files and local modifications before editing. Inspect Git status if Git exists; if absent, say so. Never reset, restore or overwrite unknown user changes.
4. State the objective, allowed paths, invariants, smallest discriminating check and stop condition in a short plan. Ask only about ambiguity that affects correctness, scope or an irreversible action.
5. Do not reread the whole paper library or reference archive every session. [docs/LESSONS-AND-ROI.md](docs/LESSONS-AND-ROI.md) is an on-demand failure guide, not startup context to load wholesale.

## Modification boundaries

| Area | Rule |
|---|---|
| Documentation tasks | May edit current docs and add supporting records within this project. No engine changes unless a task authorizes them. |
| Authorized implementation | Modify only the agreed source/test/doc surface; include directly required companion changes. Expand scope explicitly if new ownership/layout/behavior boundaries appear. |
| Old engine and sibling evidence directories | Read-only references unless the user separately authorizes work there. Do not import their code-specific restrictions as new-engine requirements. |
| `docs/reference/`, original uploads, raw experiment outputs and historical receipts | Preserve originals. Correct via a new record/superseding note; do not manufacture missing evidence or edit a result to pass. |
| New dependencies, lockfile upgrades, backend changes, save formats, public interfaces, default quality or memory budgets | Need explicit inclusion in the approved task/decision. No incidental upgrades or architecture swaps. |
| Cleanup, mass moves/renames, branches, commits and pushes | Not implicit in feature work. Ask for explicit scope. A previous deletion request is not standing authorization. |

Never remove a feature because one camera shows no difference or one search finds no caller. Check feature/config gates, generated code, runtime routing, tests and external consumers. Preserve purposeful compatibility stubs until their retirement is approved.

## Design invariants to establish and protect

- World/material identity, versioned jobs, allocation generations, publication and last-reader retirement are explicit. Reject stale jobs; define overload and missing-residency fallback.
- Host/shader layouts, flags, units and constants agree. Add uniqueness/layout checks; don't hand-allocate a bit without checking every owner.
- Geometry/material/opacity semantics agree across representations for the declared scene snapshot. Different LoDs or representations are not automatically equivalent.
- A feature specifies **producer scheduling, consumers, resource allocation and history lifetime**, not just a shader flag. Off must meet its declared image and cost contract.
- Temporal systems expose fresh production, valid population, age, reset/rejection reasons and fallback. Test reset/production before tuning reuse quality.
- Exact controls stay exact within their declared scope; approximations get an explicit reference/error budget. Never rebaseline a mismatch just to obtain green tests.
- Validate the actual selected/generated shader module and pipeline configuration. Raw source pins are supplemental; they are not compiler/API/GPU proof.
- Primary/secondary receiver semantics and guide ownership are explicit. Similar-looking functions may do different work; deduplication must preserve the contract.

## Diagnose before rewriting

- Reproduce the failing build/configuration, camera, settings, feature key and history state. A filtered test, different scene or title screen is a different experiment.
- For silent failures, inspect API/compiler/OS diagnostics before repeated speculative edits. Re-run the control when an intermittent failure or failed revert contradicts the hypothesis.
- After two inconclusive *hypothesis iterations*, stop repeating blind edits: add a discriminating diagnostic, revise the hypothesis or report the blocker. This does not limit statistically necessary repeated trials.
- Use ablations, counters, version/guide views and positive/negative test controls. Prove the instrument can fail and the path is exercised.
- Prefer the smallest correctness fix. A workaround that removes detail, animation or lighting changes the objective and needs approval.

## Test and measure honestly

- Discover runnable commands from the actual project/toolchain. Verified commands are listed in [engine/README.md](engine/README.md); do not copy old-engine commands as if they run here.
- Use focused checks while iterating, then the relevant broader/configuration gates before claiming completion. Run a suite once, keep its true exit status/full log, analyze that log; rerun only for a reason.
- Keep pure tests and GPU integration tests separate. Unavailable GPU/SDK/toolchain checks are **NOT RUN**, not passed. Identify baseline failures without hiding or fixing unrelated ones.
- Preserve inputs/configs/seeds and cold/warm state. Compare matched arms; use repeated paired/interleaved trials, uncertainty and negative-control stages for performance claims.
- Include compilation, builds/uploads, memory, fallback, invalidation and end-to-end latency. Do not sum overlapping/nested timer intervals or transfer paper speedups.
- Judge motion/disocclusion/edit behavior as well as stills. A lower image-error score obtained by blur is not automatically better appearance.
- On the existing resource-limited laptop, use `-j 2` for authorized builds. Revisit the limit for other hardware; do not launch background watchers or benchmarks without a purpose and cleanup plan.

## Keep costs and documentation small

- One hypothesis and one isolated change at a time. Simple baselines before paging, neural caches or multiqueue complexity; require measured need and an exit path.
- Keep this contract below **200 lines**. Add a rule only if it prevents a demonstrated or imminent recurring mistake; move explanations into linked docs.
- Use one short change record for a substantive task. Add an ADR only for a durable architecture/interface/default/lifetime decision. Trivial edits do not need three reports.
- Update the authoritative behavior/command doc in the same change. Cite paths/symbols, not line numbers. Keep observations, explanations and recommendations distinct.
- Never generalize old observations into timeless rules: no universal shader-size cliff, no always-free AO, no universally bad Ray Reconstruction, no mandatory old four-body/water-flag design.

## Finish in this order

1. Inspect the diff and scope; run applicable checks; record actual outputs, failures and unrun gates.
2. Update the affected current docs; close the change record and decision status if needed. Store raw evidence once and link it.
3. **Last state update: `docs/NOW.md`.** Record what exists, what does not, authorization, exact next action, blockers, test status and any live processes. Do not launch a new task after writing the handoff.
4. Final response: changed paths, result, checks actually run, unresolved risk and next decision. Do not say “done/works/faster” beyond the evidence.

Templates: [change/experiment](docs/templates/CHANGE.md), [ADR](docs/templates/ADR.md), [handoff](docs/templates/HANDOFF.md). The handoff template is for updating NOW, not creating endless parallel status files.
