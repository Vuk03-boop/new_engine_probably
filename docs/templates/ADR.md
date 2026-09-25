# ADR <id> — <durable decision>

Status: proposed / accepted / rejected / superseded by <id>
Date:
Decision authority: <explicit user approval or delegated scope>
Related change/evidence:

## Context and actual constraint

What requires a durable decision? Which facts are measured/source-verified, and which are assumptions? What remains unknown?

## Options and tradeoffs

Include the simplest valid baseline and “do not do this yet.” Compare correctness, maintenance/developer effort, runtime, memory, latency, compilation and quality. Do not fill absent measurements with predicted paper percentages.

## Decision

State the boundary precisely: affected interface/ownership/representation/default and scope. Approval of a design is not evidence implementation or acceptance is complete.

## Contracts and consequences

- Exact invariants, units/layout/identity and admitted approximation:
- Update/reset/publication/retirement behavior:
- Supported/unsupported capabilities and fallback:
- Costs, risks and complexity deliberately accepted:
- Current docs/tests/commands affected:

## Validation and revisiting

- Required acceptance evidence and status:
- Stop/reopen conditions:
- Migration/rollback if applicable:

## Implementation status

Not implemented / partial / implemented with pending gates / accepted implementation.
Link the actual evidence; retain unresolved checks.

## Supersession or errata

Preserve historical reasoning/results. Change an accepted direction through a superseding decision; label factual corrections rather than silently rewriting what was observed.
