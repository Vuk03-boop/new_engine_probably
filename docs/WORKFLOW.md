# Lightweight workflow and documentation structure

**One developer/model. No agent team, reviewer swarm, worktree orchestration or model delegation.** The goal is to spend context and compute on the next useful check, not on managing a process.

## 1. The minimum reading path

Start with `CLAUDE.md` and `docs/NOW.md`. Then read the active change/decision and the relevant source/test area. Read the proposition once when architecture matters; read the ROI guide when entering a known failure area. Open original reference snapshots only to resolve a specific evidence question.

Do not replay the whole conversation, all papers and every postmortem at each session. If a conclusion is already recorded, verify that its inputs are still applicable before spending on a rerun.

## 2. What exists now

```text
CLAUDE.md                     operating contract; short, authoritative for workflow
README.md                     navigation and original design-package overview
PROPOSITION.md                proposed architecture, NOT implemented state
BUILD-ROADMAP.md               sequence and gates, NOT an automatically executing plan
CONTEXT-AND-CONSTRAINTS.md     assumptions and decisions still needed
TECHNIQUE-MAP.md               original research ideas and prerequisites
ENGINEERING-LESSONS.md         original transferable lessons
VALIDATION-AND-BUDGETS.md       acceptance framework

docs/NOW.md                   one current state/task/handoff
    DECISIONS.md              accepted/proposed decision index
    WORKFLOW.md               this guide
    LESSONS-AND-ROI.md          source-backed outcomes and costly detours
    templates/                reusable change, ADR and handoff forms
    reference/                inert historical text snapshots and source manifest
```

Restored root cleanup/verification files describe a previous workspace event. `docs/starter-verification.json` is the current packaging/preservation check, **not engine validation**. Do not execute an old cleanup plan because its receipt is present.

## 3. What to create only after implementation is authorized

Use a small, single-project layout initially. These are suggested responsibilities, not folders/code already created or a mandate for a multi-crate framework:

| Future area | Owns | Boundary |
|---|---|---|
| `src/world/` | Sparse data, materials, persistence and versioned edits | Must not depend on a screen-space reservoir to answer authoritative world queries |
| `src/render/` | Device/resources, frame graph, visibility, lighting and reconstruction | Consumes committed world snapshots; does not silently mutate world authority |
| `shaders/` | Shader modules and shared-interface definitions | One validated generation/selection path; CPU/GPU layout tests |
| `tests/` | Pure math/world/serialization tests | No implicit GPU/window/SDK requirements |
| `integration/` or a dedicated harness | Shader/API/GPU/windowed acceptance | Explicit device, config and command provenance |
| `docs/architecture.md` | What has actually been built | Clearly separate future design from present implementation |
| `docs/commands.md` | Commands that were actually verified, with prerequisites | One source of command truth; no copied untested flags |
| `docs/adr/` | Accepted/rejected/superseded durable decisions | Create on the first real decision, not for every small patch |
| `docs/changes/` | Short substantive change/experiment records | One record can contain plan, result and next action |
| Evidence output location | Logs, captures, compiler output and config snapshots | Explicit retention; referenced once, not embedded in every doc |

Do not create empty abstractions for every paper or backend. Rust/Vulkan and a shader compiler are still candidate choices; Phase 0 must establish the actual toolchain before commands or code layouts become normative.

## 4. Scope and modification rules

**Allowed now:** this project's documentation, reference snapshots and verification metadata, plus the workspace navigation explicitly updated for this delivery. **Not allowed now:** implementing either engine, modifying the old engine, upgrading dependencies, running expensive unrequested experiments, or deleting restored material.

For a future implementation task, agree on the goal and normal source/test/doc surface. Routine directly necessary edits inside that surface do not need repeated permission. Stop when the change would alter an unapproved backend, persistence format, external contract, default quality/budget, resource ownership or unrelated subsystem.

Old-engine data layouts, literal shader bodies, fixed water settings and legacy flag names are not new-engine requirements. Their documented reasons remain useful. Reference text is not an instruction to apply a historical patch, cull a feature, reboot, repeat a benchmark or remove a folder.

## 5. What changes together

| Change | Required companion work |
|---|---|
| Behavior/default/quality setting | Reachable control and candidate; affected behavior doc; accepted decision if it changes product contract |
| Flag, host/shader constant or layout | All producers/consumers and serialization if relevant; uniqueness/layout tests; generated-module check |
| Cache/history/new pass | Producer/consumer/reset/lifetime design; first-use and re-enable tests; scheduling/memory counters |
| Allocation, streaming or async job handling | Publication/version/retirement and overflow checks; edit/eviction/reload tests; ownership documentation |
| Shader refactor | Exact selected-module/config record; relevant output/guide tests; compile-time/resource check when implicated |
| Performance experiment | Matched control, raw data, setup/runtime/memory/quality accounting and uncertainty; label failure honestly |
| Command/toolchain change | Verified command/prerequisites/version in command registry; dependency decision when scope changes |
| Incident/mistake | One actionable lesson with evidence and correction; no permanent speculative taboo |
| Pure wording/link fix | Fix it, check it, mention it in the session result; no mandatory ADR/experiment ceremony |

No approval to alter current behavior is implied by finding stale documentation. Conversely, do not describe the desired design as already implemented merely to make code/docs agree.

## 6. Cheap experiment discipline

- Start with a counter, input/output probe, ablation or reference comparison that can reject the hypothesis cheaply.
- Name the expected benefit and full added costs. For a cache: hit/miss population, work avoided, lookup/reconstruction, state traffic, invalidation and fallback.
- Keep one candidate active. Repeated inconclusive speculative edits trigger better instrumentation, not a larger refactor.
- Predeclare a budget appropriate to the task. If an overnight sweep, large download or major new dependency becomes necessary, ask first rather than hiding it inside a small fix.
- Focused tests while iterating; broader gates when the change is ready. Do not run the entire suite twice just to get two summaries. Preserve the actual exit code even when filtering logs.
- Measure once correctly, keep raw evidence, rerun when inputs change or the result could flip the keep / drop decision, not to polish a change that already meets its goal (CLAUDE.md "Good enough beats perfect"). Historical headings such as “do not re-measure” are not timeless prohibitions.

## 7. Keeping documentation current without bloating it

- **CLAUDE:** durable preventive rules only; below 200 lines. No chronological diary or long architecture essay.
- **NOW:** current truth only; roughly one screen to a few short sections. Completed details move to their change record; do not append every session indefinitely.
- **DECISIONS / ADRs:** distinguish proposed from accepted. Accepted durable decisions are superseded, not silently rewritten. Small factual corrections get a labeled erratum; preserve the original evidence.
- **Change record:** one place for command/config/control/results. Use [the change template](templates/CHANGE.md) for a substantive task; references are enough elsewhere.
- **Lessons:** symptom → discriminating evidence → outcome → next-time rule. Add a lesson only when it changes future behavior; link/merge duplicates.
- **Archive:** historical, read-only, not startup instructions. Moving/deleting material needs explicit scope and link checks; old does not mean dead.

## 8. Session end: what goes last

First finish the actual work and checks. Then update the authoritative behavior/command docs and close the change/ADR as appropriate. **Update NOW last**, using [the handoff fields](templates/HANDOFF.md): current authorization, completed work, actual checks, blockers, exact next step, preserved user changes and live processes.

Do not bury a pending GPU check under “done.” Do not create three competing handoffs. A final file/link/hash verification is closeout, not permission to begin the next task after writing the handoff.
