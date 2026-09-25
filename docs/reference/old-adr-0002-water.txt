# 0002. Water variance filtering (WA01)

Status: Accepted

## Context

Procedural waves carry sub-pixel slope detail that aliases into sparkling,
flickering highlights. The textbook fix — clamping wave-slope variance —
removes the artifact but also the high-frequency detail that gives the
surface its optical normals and reflections, leaving flat, dead water.

## Decision

Preserve the removed variance and compensate in the normal. As unresolved
variance grows, the final shading normal is blended toward the geometric
(flat) normal, which caps extreme reflection vectors without smoothing the
surface itself. Because this model assumes a flat atlas layer for that
blendpoint reference, the `water-mottle` parameter is pinned to `0.0`: the
`--water-mottle` flag still parses for backward compatibility, but its value
is discarded with a stderr notice ("water mottle is fixed at zero to keep
the atlas layer flat"). See `config.rs` (`water_mottle`, the flag arm) and
the water suites.

## Alternatives considered

- **Clamp variance** — rejected: loses detail; the aliasing/detail trade was
  the original problem.
- **Keep `--water-mottle` user-tunable** — rejected: mottle warps the layer
  the variance blend assumes is flat; a tunable known to break the water
  model is worse than a pinned one that says so.
- **Full-res traced secondary water legs** — available as the measurement
  arm (`--no-water-sec`) but not the default; half-res legs
  (`--water-sec-scale`, default 2) won on frame cost.

## Consequences

- Stable, artifact-free water with detail retained; a hard-coded runtime
  behavior replace a user knob, deliberately.
- `--water-mottle` becomes a compatibility stub that must keep its stderr
  notice; do not "repair" its plumbing (`docs/lessons.md`).
- `tests/water.rs`, `tests/water_sec.rs` and
  `validation/check_glass_transport.py`-family history pin the behavior.
