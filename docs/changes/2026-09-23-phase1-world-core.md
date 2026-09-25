# Change: Phase 1A world core + 1E reference ray query

Status: passed (within the limits below)
Date and baseline: 2026-09-23. New tree `engine/`; no Git repository. Phase 0 probes are unchanged, except the throwaway `PROBE_HOLD_MS` switch recorded in [the Phase 0 record](2026-09-22-phase0-prep.md).
Authorization: user, "If so greenlight 1A and 1E". The condition: the defaults must follow the original plan.

## Objective and smallest useful test

- **Objective:** the first engine code.
  - The authoritative sparse world: chunk → brick hierarchy, exact material identity, content versions.
  - A CPU reference ray/occupancy query, to serve as the ground truth for later GPU paths.
- **Falsifiers:**
  - Split/join round-trips fail at negative coordinates or chunk edges.
  - A no-op write or a recreated brick reuses a version.
  - The DDA and brute force disagree on any ray.
  - The comparison cannot detect a removed voxel.
- **Stop condition:** all tests pass in debug and release, the negative control fails as it should, and a reference image is rendered and inspected.

## Scope and contracts

- **Allowed paths:** `engine/` (new), plus this record, `docs/NOW.md` and `docs/DECISIONS.md`.
- **Excluded:** 1B journal/save, 1C jobs/retirement, 1D accounting, GPU work, external crates.
- **Plan alignment check** against [PROPOSITION.md](../../PROPOSITION.md) §2 and [CONTEXT-AND-CONSTRAINTS.md](../../CONTEXT-AND-CONSTRAINTS.md) §5. It changed two proposed defaults before implementation:
  - A chunk hierarchy of sparse bricks, not a flat brick map.
  - Occupancy separate from material ID, with no "0 = empty" material. Numerical parameters are kept separate from identity.

  Dimensions stay tuning constants in `engine/world/src/dims.rs`: 8³ bricks, 4³-brick chunks, 1/16 m voxels.
- **Invariants implemented:**
  - Material IDs are stable and exact. Writing an unknown ID is rejected, with no side effects.
  - Empty space allocates nothing. Clearing the last voxel frees the brick, then the chunk.
  - Every content change stamps the touched brick with the next world-wide version. No-op writes stamp nothing, and a freed-then-recreated brick gets a strictly newer version.
  - Deterministic ordered storage and iteration.
  - Reference-query convention: half-open voxels, hit = entry time clamped at 0, zero-length contact is not a hit, `t_max` inclusive. Crossing times use one shared f64 expression, so the DDA and brute force must agree **bit for bit**, not within a tolerance.
- No ADR: dimensions are declared tuning values, and the separation of occupancy and identity follows the already-proposed design. A later change to the save format or to public interfaces would need one (1B).

## Commands and evidence actually produced

| Check | Result | Evidence |
|---|---|---|
| `cargo test --release -j 2` | exit 0; 28 unit + 2 integration tests pass | `engine/results/test_release.log` |
| `cargo test -j 2` (debug: overflow checks, debug asserts) | exit 0; same 30 tests; the randomized test takes 240 s unoptimized | `engine/results/test_debug.log` |
| DDA vs brute force, seeded (4 world sizes × 3 worlds × 3000 rays) | 36,000 rays, **all exactly equal**. 5,752 hits, 9,065 with finite `t_max`. The mix covers axis-aligned, diagonal, on-plane-origin and zero-component rays, and extents up to ±40 voxels across chunk edges. | same log |
| Negative control: one voxel removed from the oracle's world | mismatches detected (the test requires > 0) | same log |
| `ref_render` street block, 640×360 | exit 0. 17 materials, 291 chunks, 5,103 bricks, 1,515,582 voxels, 5.56 MiB payload estimate. Build 109 ms, render 806 ms (2 threads). 230,400 primary rays, 173,137 hits; 171,196 shadow rays, 34,900 blocked. | `engine/results/street_ref.json`, `street_ref.png` |
| Thread-count determinism (1 vs 2 threads) | byte-identical | `compare.py`: 0 differing px |

Development iterations:
- The first agreement run failed only its coverage floor, because random worlds were too sparse (2.6% hits). Density was raised, and the floor changed from a hit fraction to an absolute ≥ 2,000 hits and ≥ 2,000 misses. Every run had zero disagreements.
- `cargo` auto-installed the pinned toolchain name `1.98.1-x86_64-pc-windows-msvc`. It is the same version as the existing default.

## Result and full cost

- **Image** (`engine/results/street_ref.png`): three shopfronts with recessed windows and doors, emissive signs, sidewalk and curb, dashed centre line, and lamp-post shadows on the road. Inspected visually. The shading is a debug view (N·L sun, one shadow ray, ambient, flat emissive colour), not lighting.
- **Limits:**
  - One camera.
  - The payload estimate excludes allocator and `BTreeMap` overhead.
  - The DDA does per-voxel lookups with a one-brick cache and no empty-brick skipping. That is correct by construction, but slow, and deliberately so for a reference.
  - Glass is opaque and emissive surfaces light nothing.
  - Timings are single observations, not performance claims.
- **Decision:** keep.

## Closeout

- Docs: `engine/README.md` (verified commands), this record, `docs/NOW.md`.
- Revert: delete `engine/`; nothing else depends on it.
- Next: slice 1C (versioned jobs, bounded queues, retirement; tests with artificial delays and out-of-order completion) or 1B (edit journal, save/reload; save format ADR). Needs user authorization.
