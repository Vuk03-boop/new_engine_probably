# Change / experiment: <short name>

Status: planned / in progress / blocked / passed / failed / not run
Date and baseline: <revision or file/config hashes; do not invent Git state>
Authorization: <user request or accepted delegated scope>

## Objective and smallest useful test

- Problem/hypothesis:
- Expected counter, quality or cost change:
- What cheap observation could falsify it before implementation?
- Time/compute budget (expensive runs are counted in the stop rule below):

## Criteria and stop rule (frozen before the first run; CLAUDE.md "Good enough beats perfect")

- Goal criteria (what the task is for; a miss means fix, rescope or drop):
- Guards (don't break what works), each judged against today's default with a tolerance:
- Cost budget (ms / MB, taken from the milestone's headroom):
- Stop rule: goal met → propose keeping it, guard misses recorded for the user to accept. Goal missed → diagnose, then at most one fix round, or rescope / drop.
- Run budget: one judged run plus at most one fix round; pending GPU checks bundled into it; which result would flip keep / drop:

## Scope and contracts

- Allowed source/test/doc paths:
- Explicit exclusions:
- Exact invariants and/or permitted approximation:
- Producer, consumer, identity/version, reset and lifetime changes:
- Dependent callers/configurations/representations:

## Control and candidate

- What single variable differs?
- Scene/camera/seed/edits/light/animation/config/pack:
- Device/driver/API/features/resolution/compiler/specialization key:
- Cold/warm state; histories/queues/weights reset:
- Positive/negative control proving the path and diagnostic work:
- For performance: matched workload, repetitions/order and uncertainty plan:

## Commands and evidence actually produced

| Command/config or check | Exit/result | Evidence path | Scope/limitation |
|---|---|---|---|
| <fill only after running; otherwise NOT RUN> | | | |

Record full logs once; link rather than duplicate them. Distinguish CPU model, selected shader validation, API validation, device correctness, image/sequence result and performance. A command that was planned but never run is not evidence.

## Result and full cost

- Exact/quality result, including motion/edits:
- Runtime, setup/compile/upload and memory/latency cost:
- Development iterations/expensive-run duration if known; otherwise unknown:
- Confidence/uncertainty and workload limits:
- Decision: keep / reject / revise / blocked; why:
- Guard misses for the user to accept (value, limit, today's value):
- Known failures and checks not run:

## Closeout

- Minimal diff and affected current docs updated:
- ADR needed? If yes, link; if no, no ceremonial record:
- Fallback and revert:
- Lesson worth retaining / reopening condition if rejected:
- Next action and authorization needed:

Update `docs/NOW.md` last. Use one record for this task, not separate plan/result/handoff files with conflicting status.
