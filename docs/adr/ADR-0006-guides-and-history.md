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

## Amendment 2 (4B, 2026-09-25): emitters

Status: **accepted 2026-09-26 (S-026):** G5's R1 and R2 pass in all 3 corrected edits on the RTX 3050; R2's control is accepted as underpowered at night (first proposed with 4B, S-024). Record: [4B](../changes/2026-09-25-phase4b-many-lights.md).

- **The lights' switch is a light jump.** `History::set_lights` carries whether the lights are on; a change from the previous frame resets every history (`ResetCause::lights`), like the sun's > 1° jump.
- **Relight rules for emitter light**, applied while the lights are on, besides Amendment 1's sun and sky rules. A pixel whose history would otherwise be accepted restarts as *relit* (reason 9) when a bound on the change of its **direct** emitter light is at least **2%** of its history's luminance Y_h (the bilinear taps' mean; `temporal::EMITTER_TOLERANCE`). With d the distance from the surface point to the box grown by 1 voxel:
  - **E1, a light changed:** the box carries ΔΦ, an upper bound on the emitter power the edit adds or removes (`temporal::changed_power`: π × luminance for every face of every edited voxel whose material emits before or after, and for the face of each emitting neighbour the edit may cover or expose). A Lambertian emitter set of power ΔΦ inside the box changes the reflected radiance at distance d by at most ΔΦ / (π² d²) (albedo ≤ 1). Restart when ΔΦ / (π² d²) ≥ 0.02 Y_h.
  - **E2, a light shadowed or unshadowed:** the box lists up to 8 emitters of the scene's current table with the largest Φ / (d_e² + 1), d_e from the emitter's centre to the box (`Relight::with_lights`). Restart when the segment from the point to a listed emitter's centre crosses the box grown by that emitter's half-diagonal (plus the 1-voxel margin), and L_Y · A / (π d²) ≥ 0.02 Y_h, with A = ab + bc + ca of the grown box (a bound on the solid angle it can block, times the emitter's luminance).
  - **Declared approximation:** emitters not listed, and indirect (bounce) changes, restart nothing. G5 measures the misses (changed pixels not relit) and the cost (relit pixels that did not change).
- **Buffer:** the relight buffer grows to a header of 2 rows (sun and box count; the tolerance) plus 18 rows per box (lo and sky radius; hi and ΔΦ / π²; 8 × (centre and grown radius, luminance)): 4,640 B for 16 boxes, still written in the command stream (`vkCmdUpdateBuffer`).
- With the lights off (and on the M3 street, which has no emitters) boxes carry no emitter terms, so the 3E behaviour is unchanged.

## Amendment 3 (4B, 2026-09-25): no sun-motion cap at night

Status: **accepted 2026-09-25 (S-025).** Record: [4B](../changes/2026-09-25-phase4b-many-lights.md).

- **Rule.** Amendment 1's sun-motion age cap applies only while the sun's elevation is at least `TemporalSettings::sun_cap_min_elevation_deg` (default **−12°**). Below it the cap is `max_age`. The light-jump reset (> 1° in one frame) is unchanged.
- **Why.** Below the horizon the sun casts no shadows, and the sky is its only light. The cap held a night history at 8–16 frames while the day ran, which made the first night look noisier than necessary (4B record, first look). Measured (`light::sky` `diagnostic_skylight_below_horizon`, computed directly from the real-time sky model, uncorrected): the skylight radiance on an albedo-0.3 surface falls about 2× per degree, from 8.0 × 10⁻⁴ units at 0° to 5.4 × 10⁻⁶ at −6°, 2.4 × 10⁻⁷ at −9°, **1.1 × 10⁻⁸ at −12°** and 3.2 × 10⁻¹¹ at −18°. The night street's mean reflected emitter light is about 2 × 10⁻⁵ (4A record: image mean 1.6 × 10⁻⁴, 86–90% directly seen emission), so at −12° the sky is about 0.05% of it.
- **Declared approximation.** Below −12° a history may lag the sky's dimming by up to `max_age` frames (about 2° of sun at the viewer's day speed, at most a few times the sky's own value in that range). That is small next to the lamps' light but not bounded per pixel: a surface no lamp reaches shows that lag in full, and it is black in any case. Blue hour (−5.3°) keeps the cap.
- **Test.** `temporal::tests::sun_cap_is_off_when_the_sun_is_well_below_the_horizon` (pure, `gpu --lib`), with a planted cutoff of −90° that keeps the day's cap.


3D tests: a static camera's history equals the host's running mean; accumulation converges to the reference; planted "reset every frame", "never reject" and "no fresh sample" are caught; an edit resets only its regions; motion equals the host's reprojection; global resets reset everything.

## Implementation status

See the [3D record](../changes/2026-09-24-phase3d-temporal.md). Amendment 2 (4B): implemented in `gpu::temporal` (`changed_power`, `Relight::with_lights`, `History::set_lights`) and `shaders/temporal.slang`; the row layout passes C2 (cloud); G5 (R1, R2; its R2 control accepted as underpowered at night, S-026) and G6 pass on the RTX 3050 ([4B record](../changes/2026-09-25-phase4b-many-lights.md)).

Amendment 3 (4B): `TemporalSettings::sun_cap_min_elevation_deg`, applied in `Temporal::record` through `TemporalSettings::age_cap(sun_deg, elevation_deg)`; the pure test passes in the cloud; the viewer's night look with the day running is NOT RUN until the user looks again.
