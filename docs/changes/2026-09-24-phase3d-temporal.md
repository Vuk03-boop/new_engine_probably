# Change: Phase 3D — the temporal foundation

Status: **passed on the RTX 3050 (2026-09-24).**
Date and baseline: 2026-09-24, after 3C (no Git).
Authorization: S-017 (slices 3A–3G).

## Objective

A history of earlier frames that knows when it is wrong, before any filter is tuned ([Phase 3 proposal](2026-09-24-phase3-proposal.md) §3D; LESSONS: an old temporal path was reset every frame and had no fresh producer). The contract is [ADR-0006](../adr/ADR-0006-guides-and-history.md).

## What was built

- **ADR-0006:** guides (face, **integer plane coordinate**, material, region key and snapshot: an exact surface test, no depth tolerance, because voxel faces lie on integer planes), reprojection with per-tap bilinear validation (snapped within 1/1000 px of a centre), per-pixel age and one of nine reasons, fresh production every frame, global resets (first frame, camera cut > 64 voxels, sun jump > 1°, forced), `max_age` (default 64). 72 B/px, 149 MB at 1080p.
- **`gpu::temporal`** + `shaders/temporal.slang`: `History` (guides, colour and state ping-pong, motion), `Temporal` (two descriptor sets for the ping-pong; reset decisions on the host; planted faults `reset_always`, `never_reject`, `no_fresh`). The resolved colour and the state go back into the shade radiance buffer, so the light view needs no rebinding.
- **`debug_view`:** views 9–11 (history age, history reason, motion; `bind_motion`), and `View::reads_gbuffer` for the G-buffer view tests.
- **Viewer:** accumulation on by default (H toggles), key 8 cycles age / reason / motion (`--view 10`–`12`), `--max-age N`; the history is rebuilt on resize, rebound after edits (new region table), and reset when shading resumes after a gap.

## Criteria and results

Logs: `engine/results/test_gpu_3d_temporal.log` (the first run: `test_gpu_3d_temporal_first.log`), `test_gpu_3d_cost.log`, `test_gpu_3d_rtx.log`, `viewer_3d.jsonl`. Images: `engine/results/temporal_3d_sheet.png` (street, 8 h and 17.75 h; left one frame, right 64 accumulated).

| # | Criterion | Result |
|---|---|---|
| 1 | Layout checked; still camera: after frame k the resolved colour equals the host's running mean of the fresh samples (1e-4) on ≥ 99.9% of surface pixels, and every surface pixel has age k, accepted. Planted `reset_always` and `no_fresh` fail it | 0 of 43,288 off; all ages 24 and accepted. `reset_always`: 43,089 off, every age wrong; `no_fresh`: 43,089 off |
| 2 | Accumulation (256 frames, max_age 1024) converges to the reference's sun component, image mean within 1% | within 2.9 × 10⁻⁵ |
| 3, 4 | After a camera move (3 moves), motion equals the host's reprojection (0.01 px) and the reasons equal the host's per-tap re-derivation: ≤ 0.05% of pixels and ≤ 5% of the host's rejections differ, ≥ 100 rejections. Planted `never_reject` fails it | correct arms: 57,600 of 57,600 reasons agree, 0 motion errors, 202 / 7,964 / 8,603 rejections; `never_reject`: 194–204 px differ, fails every arm |
| 5 | After an edit, the decisions equal the host's on every pixel (≤ 0.05% differ), "edited" occurs, only in rebuilt regions | 0 of 57,600 differ; all 295 px of the rebuilt region edited, 0 outside |
| 6 | Camera cut and sun jump reset every surface pixel; a 0.06° sun move does not | cut 39,918 / 39,918 reset, jump 43,288 / 43,288, slow 43,288 / 43,288 accepted |
| — | Full `-p gpu` release | exit 0: 12 + 9 + 4 + 5 + 4 (+1) + 4 + 3 (+1) + 7 |
| — | Pure suite; clippy | exit 0, 133 tests; only the 6 pre-existing `world` lints |
| — | Viewer: all 12 views with a resize; 40 edits at 1080p; a walk with the day running through sunrise | all exit 0, validation 0 / 0, no leaks, 0 overlaps; edits p99 16.4 ms (budget 50) |

**Cost at 1080p** (validation off, turning camera, 60 reps): shade (sun + sky) 1.10 ms median, **temporal 1.26 ms** median (1.35 p90). The GPU frame so far is about 2.9 ms (G-buffer 0.3, shade 1.1, temporal 1.3, view 0.2). The temporal pass is memory-bound (four taps of guides, colour and state); RGBA16F history and packed guides can halve it if 3G needs the time.

## Failures on the way (diagnosed)

1. **Still camera off the running mean on 14% of pixels** (ages all correct). Hypothesis: f32 reprojection of a still camera lands about 1e-5 px off the pixel centre, so the bilinear fetch mixes ~1e-5 of a neighbour's history, which exceeds 1e-4 relative next to 1-sample sky noise. Fix: snap within 1/1000 px (ADR-0006). Result: 0 pixels off, confirming the cause; it also removes accumulation blur for a still camera.
2. **`never_reject` not caught** by the first criterion (≤ 0.5% of pixels may differ). The fault's footprint is about 0.35%, so the control could not fail. The correct arms agree on every pixel, so the limit became 0.05% of pixels and 5% of the host's rejections: stricter, not looser.
3. **The first edit criterion** ("98% of the rebuilt region's pixels edited") did not match the ADR-0006 rule: 9 of 295 pixels take their history from taps across a region border and get that region's verdict. Replaced by the exact check (every decision equals the host's), which passes on all pixels.
4. Python patching on Windows had silently converted the files it wrote to CRLF; they were converted back to LF (files that were CRLF before this session were left as they were).

## Known limitation (for 3E)

An edit changes lighting outside the rebuilt regions (a removed voxel lets the sun in elsewhere), and the sun moving changes shadows everywhere. That history is not rejected; it follows the new light at the 1/age rate (about a second at `max_age` 64). Detecting lighting change (temporal gradients, colour-space clamping) is 3E's job; the reason view makes the lag visible.

## Not run

- The debug-build `-p gpu` run (last after 2E).
- Motion sequences against the reference (3E, with the filter).
- Another GPU.

## Closeout

- Docs: ADR-0006 (new), this record, `engine/README.md`, NOW.
- Next: 3E, the native reconstruction: temporal accumulation with lighting-change handling and an edge-aware spatial filter on demodulated light, measured with relative MSE, mean-luminance bias and FLIP against the reference, on stills and motion paths.
