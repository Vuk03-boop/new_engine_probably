# ADR 0006 — Guides and the temporal history contract

Status: accepted (delegated within S-017: lighting technique delegated; the proposal §3D is the approved plan).
Date: 2026-09-24
Decision authority: S-017.
Related: [Phase 3 proposal](../changes/2026-09-24-phase3-proposal.md) §3D, [3D record](../changes/2026-09-24-phase3d-temporal.md), ADR-0003 (G-buffer, surface ids, region table), ADR-0005 (light units), LESSONS "Correct temporal-history lifecycle" (evidence: an old temporal path was reset every frame and had no fresh producer).

## Context and actual constraint

- The real-time lighting (3B/3C) is one sample per pixel per frame; 3E's reconstruction and later reuse need a history of earlier frames, and a way to know when that history is wrong.
- The world is static between edits; the camera moves; the sun moves slowly (time of day); edits rebuild whole regions (ADR-0003 region table: region key and the snapshot each region was uploaded from).
- The raster G-buffer is discarded every frame (`UNDEFINED` load), so nothing of the previous frame survives today.

## Decision

**Guides.** The current frame's guides are the ADR-0003 G-buffer (depth, normal, material, surface id) plus, per pixel, the pixel's **region key and region snapshot** (from the region table). After each frame a temporal pass stores the guides the next frame needs in a guide buffer (ping-pong, 16 B/px): the face (axis and sign) and material, the face's **integer plane coordinate** (voxel faces lie on integer planes, so the same surface has exactly the same plane and a different parallel surface differs by at least one voxel: the surface test is exact, with no depth tolerance), the packed region key and the region snapshot. The previous camera is kept on the host.

**Reprojection.** The surface point is rebuilt from depth and projected through the previous camera. Motion (previous pixel position − current pixel position, in pixels) is exact for static geometry and is stored (8 B/px) for later consumers. History is fetched bilinearly from the four previous pixels around the reprojected position; each tap is validated on its own, and the weights of valid taps are renormalised. A bilinear coordinate within 1/1000 px of a pixel centre is snapped to it, so a still camera reuses exactly its own pixels (f32 reprojection lands about 1e-5 px off the centre).

**Per-pixel history state** (ping-pong, 4 B/px): the history length (age, in samples, capped at `max_age`) and the reason of the last decision:

| Reason | Meaning |
|---|---|
| 0 accepted | at least one valid tap |
| 1 sky | no surface in this pixel (the sky is deterministic, not accumulated) |
| 2 off-screen | the point was outside the previous frame or behind its near plane |
| 3 no previous surface | every tap was sky or invalid in the previous frame |
| 4 material | the taps' material differs |
| 5 normal | the taps' face (axis and sign) differs |
| 6 disoccluded | the taps' face lies on another plane (a different integer plane coordinate) |
| 7 edited | the tap is in the same region, but the region was rebuilt since (snapshot changed) |
| 8 reset | a global reset: first frame, resize, camera cut, light jump |

When several taps fail for different reasons, the nearest tap's reason is reported.

**Fresh production.** Every visible pixel receives the frame's new sample, whatever its history: an accepted pixel becomes `prev + (x − prev) / age`, a rejected one becomes `x` with age 1. A history that is never refreshed, or reset every frame, is a failure the tests must catch.

**Global resets:** the first frame; a resize; a camera cut (the eye moved more than 64 voxels, 4 m, in one frame); a light jump (the sun direction changed by more than 1° in one frame). Slow sun motion (running the day: about 0.06° per frame) is not a reset; the age cap bounds how long old light lingers.

**Moments (added at the start of 3E):** the history colour's fourth channel holds the running mean of the fresh sample's squared luminance, blended like the colour, for the denoiser's variance estimate. The 3D tests pass unchanged with it (`test_gpu_3e_moments_temporal.log`).

**`max_age`** is a parameter (a "slider", swept in 3E), default 64.

## Contracts and consequences

- Consumers read the resolved history (the accumulated radiance), the age and the reason; 3E's filter uses the age to weigh its spatial pass, and the reason views are the debugging instrument.
- The guide, state and history buffers are frame-sized, under `GpuTemporal`: guides 2 × 16, state 2 × 4, history 2 × 16 (RGBA32F), motion 8 → **72 B/px, 149 MB at 1080p** (the proposal estimated 74 B/px). RGBA16F history can halve the history later if memory matters.
- **Display:** the temporal pass writes the resolved radiance, and the pixel's age and reason, back into the shade pass's radiance buffer (after reading that pixel's fresh sample), so the light view and the age / reason / motion debug views need no per-frame rebinding.
- Edits reset the pixels whose history comes from a rebuilt region (plus genuine disocclusions); a pixel whose valid taps all lie in an untouched neighbouring region keeps them.
- **Known limitation:** an edit also changes lighting outside the rebuilt regions (a removed voxel lets the sun in elsewhere). There the history is not rejected; it follows the new light at the 1/age rate, about a second at `max_age` 64. Detecting lighting change (temporal gradients, colour clamping) belongs to 3E. *Addressed by Amendment 1 (relight boxes and the sun-motion cap).*
- Not decided here: the spatial filter and its guides' weights (3E), bounce-light history (3F), reservoirs (Phase 4).

## Amendment 1 (3E, 2026-09-24): lighting change and reconstruction

Status: **accepted 2026-09-24 with 3E closing (S-018 for the D4 gap, S-019 for the C1 cost).** This amendment closes the known limitation above.

- **Reason 9, relit.** An edit lists its changed voxels as boxes (`temporal::Relight`, at most 16 per frame, the rest merged into the last) for the first frame that shows it. A pixel whose history would otherwise be accepted restarts (age 1, reason relit) in two cases:
  - the ray from its surface point towards the sun crosses a box grown by 1 voxel, so its sun shadow may have changed (the margin covers the sun disk's penumbra up to about 400 voxels away);
  - the point lies within the box's **sky radius**: 4 × the largest side + 1 voxel. Outside that radius, removing or adding the box changes sky irradiance by less than about 2%, a declared approximation measured by R2.
  - The boxes and the sun direction reach the shader in a 528 B buffer written in the command stream (`vkCmdUpdateBuffer`), so no host-visible ring is needed.
  - Pixels inside rebuilt regions keep reason 7, edited.
- **Sun-motion age cap.** When the sun moves Δ° in a frame, that frame's cap is `clamp(floor(sun_tolerance_deg / Δ), 1, max_age)`. The default tolerance is 0.5°, so one history spans at most half a degree of sun path. At the viewer's day speed (0.0625° per frame at 60 fps) the cap is 8. The jump reset (> 1° in one frame) stays.
- **History resampling.** When the bilinear taps accept, the colour and moment are resampled with Catmull-Rom over the 4×4 previous pixels, if every tap with a weight is valid, clamped to the 2×2 taps' range; otherwise bilinear as before. The reasons, the age and the still-camera snap are unchanged (a snapped pixel's Catmull-Rom weights are (0, 1, 0, 0)). The moment is resampled as each tap's variance (moment − luminance²) and rebuilt from the resampled mean: resampling the raw second moment adds the taps' spread of means to the variance every frame of motion. `TemporalSettings::bilinear` restores the 3D resampling. Measured: motion error −11% together (3E record); a moving history still blurs hard shadow edges by 2–3 px after 32 frames.
- **Reconstruction** (`gpu::denoise`):
  - an SVGF-style filter after the temporal pass, on albedo-demodulated light;
  - its variance comes from the moments from age 8, before that from the 7×7 same-surface neighbours;
  - its à-trous taps must be on the **same surface** by the exact guides (face, material, integer plane), with luminance edge-stopping; the first level compares 3×3 same-surface means (removes the energy loss of value-dependent weights at young ages);
  - it writes only the shown radiance, so the history stays the exact running mean;
  - two frame-sized buffers of 16 B/px, a 4 B/px luminance guide and an 8 B/px compact surface key (44 B/px) under `GpuTemporal`; the first level's guide is built in init and the last level remodulates (memory-bound: 3.47 ms at 1080p, 3E record);
  - defaults, from the measured sweep (3E record): 2 levels, σ_l = 4, prefilter on level 1 of 2, moments from age 8.
- Criteria and results: the [3E record](../changes/2026-09-24-phase3e-denoise.md).

## Validation

3D tests: a static camera's history equals the host's running mean; accumulation converges to the reference; planted "reset every frame", "never reject" and "no fresh sample" are caught; an edit resets only its regions; motion equals the host's reprojection; global resets reset everything.

## Implementation status

See the [3D record](../changes/2026-09-24-phase3d-temporal.md).

## Amendment 2 (4B, 2026-09-26): the lights

Status: **accepted as design, both parts** (delegated within S-024: lighting technique; the [4B record](../changes/2026-09-26-phase4b-many-lights.md) froze it before any code). Part 2 (relight for lights) is built; its GPU validation (R3–R5, E4) is not run yet.

- **The lights switching is a light jump** (Phase 4 decision 3). The caller tells the history whether the street's lights are on for the next frame (`History::set_lights`). A change from the previous frame resets every pixel (reason 8, `ResetCause::lights`), like the sun's 1° jump. The M3 callers never set it, so their behaviour is unchanged.
- **Emission never enters the history** (ADR-0005 Amendment 3): `gpu::compose` adds it to the shown radiance after the temporal pass and the filter.
- **Validation:** 4B G13 (a lights change resets every surface pixel; unchanged lights accept them; the history is bit-identical with and without compose, and a compose before the temporal pass is caught).
- **Part 2, relight for lights** (a third relight rule, after 3E's; reason 10, **relit by a light**):
  - **Light rows.** On the frame that shows an edit, the caller hands the history up to 64 rows (`temporal::light_rows`, `History::set_light_rows`; about 3 KB after the 3E boxes in the relight buffer, written in the command stream):
    - the emitters the update changed (`GpuScene::take_changed_emitters`: in only one of the old and new tables, within the rebuilt regions; a quad split differently counts as changed), at most 16, the rest merged into the 16th as one conservative row (a bounding sphere, both sides, the largest Y, the summed Y·A);
    - per edit box, the 4 unchanged emitters with the largest unoccluded irradiance bound at the box's centre;
    - rows beyond 64 are left out.
  - **The test per pixel**, outside rebuilt regions, on a pixel whose history is otherwise accepted after the 3E rules: relit when for some row the pixel is in front of the emitter, Y(L_e)·min(2π, A/d²)/π ≥ 2% of the history's luminance (d to the bounding sphere; albedo counted as 1), and either the emitter changed or the segment from the pixel to the emitter's centre crosses the edit box grown by 1 voxel plus the emitter's radius. `temporal::relit_by_light` is the same rule on the host, in the shader's f32 order.
  - **Only while the lights are on:** the lit module (`temporal_lit`, `-D EMITTERS`) runs then; without the lights the M3 module (byte-identical to main's) runs, so relighting with the lights off is exactly 3E's.
  - **Declared approximation:** the 2% threshold, 4 shadow rows per box and the 64-row cap. R4 measures it.
  - **Validation:** 4B R3 (decisions against the host rule), R4 (lag against a full reset), R5 (3E R1/R2 unchanged), E4 (edit budget).
- **Numbering:** the Phase 4 proposal named this amendment for 4C's reservoir history; that becomes Amendment 3.
