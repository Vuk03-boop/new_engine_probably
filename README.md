# New engine — ground-up proposal

**Revision 1 · 2026-09-22 · Status: design proposition only**

## Single-model starter — added after restoration

**Working on this project? Read [CLAUDE.md](CLAUDE.md) and [docs/NOW.md](docs/NOW.md) first.** One model/developer, no swarms or subagent delegation. The original proposal was restored from the user's uploads. **Since then the new engine has been built in [`engine/`](engine/README.md) (Phases 1–3, M1–M3 accepted, 2026-09-25); the proposal text below is the original design proposition, kept as written, and its status lines are historical.**

- [Workflow and modification boundaries](docs/WORKFLOW.md): what to read/change, what to protect, and what to update last.
- [What worked, what did not, and ROI](docs/LESSONS-AND-ROI.md): actual old-engine cases, how we learned, historical numbers and limitations.
- [Decision index](docs/DECISIONS.md): operating scope versus still-proposed architecture.
- [Change/experiment template](docs/templates/CHANGE.md), [ADR template](docs/templates/ADR.md), [handoff fields](docs/templates/HANDOFF.md).
- [Portable reference evidence](docs/reference/README.md): on-demand text snapshots, not active instructions. Do not load all of them every session.
- [Current starter verification](docs/starter-verification.json): restoration/preservation/document checks only.

The original reading order below is for understanding the **design**, not a requirement to reread every file on each session. Root `cleanup-receipt.json`, `preservation-baseline.json`, `verification.json` and `WORKSPACE-CLEANUP.md` are **restored historical records**; they do not verify this expanded package or authorize another cleanup. Some original links lead to the larger workspace; the new operating guide and ROI evidence are self-contained in this folder.


This folder proposes a new engine, not a patch series for the recovered one. The objective is **the best stable appearance within explicit frame-time, latency and memory budgets**. No engine code, project scaffold, dependency installation or build has been created.

## Read in this order

1. **[PROPOSITION.md](PROPOSITION.md)** — architecture, what renders what, core decisions and optional research.
2. [CONTEXT-AND-CONSTRAINTS.md](CONTEXT-AND-CONSTRAINTS.md) — assumptions, required inputs, what transfers from the old project and what does not.
3. [TECHNIQUE-MAP.md](TECHNIQUE-MAP.md) — every original proposal mapped to an appropriate new-engine role, prerequisites and exclusions.
4. [BUILD-ROADMAP.md](BUILD-ROADMAP.md) — dependency-ordered construction, deliverables and stop/go gates.
5. [ENGINEERING-LESSONS.md](ENGINEERING-LESSONS.md) — practical lessons from the current engine, including corrected historical diagnoses.
6. [VALIDATION-AND-BUDGETS.md](VALIDATION-AND-BUDGETS.md) — correctness, quality, performance, memory and acceptance contracts.
7. [WORKSPACE-CLEANUP.md](WORKSPACE-CLEANUP.md) — actual cleanup, retained evidence and preservation checks.

## One-sentence design

**An editable sparse-voxel world with versioned derived surface geometry, GPU-driven visibility/residency, hybrid raster/ray-traced rendering, and a measured progression from a trustworthy sampled-lighting reference to ReSTIR reuse and selective neural acceleration.**

## Authority and evidence

- This is the active **new-engine design** entry point, not authorization to implement.
- [Research synthesis](../research-review/batch-09-synthesis.md) and [cross-audit corrections](../research-review/cross-audit-other-ai.md) remain the evidence base. A paper’s successful mechanism does not establish this proposed engine’s performance.
- [Current engine](../recovered-codebase/) is frozen reference material. Its exact layouts, fixed settings and four-body shader constraint are **not inherited requirements for the new architecture**.
- [Historical archive](../archive/) and [reconstruction records](../reconstruction/) are retained provenance, not active execution instructions.
- The old register remains **18 original transfers rejected / 3 ADAPT**. Redesigning the prerequisites creates new hypotheses; it does not retroactively accept those old transfers.

**Next action:** approve or revise the target hardware/workload/quality contract and architecture tradeoffs. Implementation requires a separate explicit go-ahead.
