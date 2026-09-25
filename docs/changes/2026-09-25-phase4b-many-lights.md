# Change: Phase 4B — many lights without reuse (the control)

Status: **in progress.** Criteria frozen 2026-09-25, before any code or run (below). Written in a cloud session (no GPU; [CLOUD.md](../CLOUD.md)).
Date and baseline: 2026-09-25, branch `claude/hopeful-bell-972dbs` from `main` at `eb17538` (4A merged).
Authorization: S-024 (4A and 4B authorized); the user's "go ahead then you are on high greenlight to do 4b" (2026-09-25).

## Objective

- Light the street at night in real time with its own emitters, without reuse: one emitter sample per pixel chosen by power, with one shadow ray, at the primary hit and at the bounce hit, then the existing temporal pass and filter ([Phase 4 proposal](2026-09-25-phase4-proposal.md) §4B).
- It is the **control** every reuse arm (4C, 4D) is compared with, so its estimator must be exactly the reference's and its cost and error curve must be measured, not tuned.
- It also brings what 4A moved here: L, the automatic lights switch and the night exposure, and emitter relight rules for edits.

## Scope

**Builds:**

1. **Emitters in `gpu::shade`** (ADR-0005 Amendment 3, now in real time):
   - `ShadeSettings::emitters`: emitter next-event estimation at the primary vertex and at the bounce hit (with `bounce`), the reference's `emitters_direct` and `emitters_indirect` with `max_bounces` 1. Same function, same random-number order (after the sun's two numbers, before the continuation's), so frame f equals the reference's sample f per pixel.
   - The sampling function moves from `reference.slang` into a shared include, so the reference and the real-time pass run one definition.
   - `ShadeSettings::emitter_samples` (default 1): k emitter samples at the primary vertex, averaged (the swept "samples per pixel" of the proposal; the bounce hit keeps one). k = 1 is the estimator above; k > 1 draws its extra samples right after the first.
   - `Shade::bind_lit` binds the scene's emitter table; `Shade::bind` binds none and the pass refuses emitter settings with it. The emitter count comes from the bound table, never from the caller.
   - With `emitters` off no emitter random numbers are drawn, so every M3 image is unchanged.
   - Planted fault for the controls: `ShadeFaults::no_solid_angle` (the emitter pdf without the solid-angle conversion).
2. **Emission at the primary hit, after reconstruction** (ADR-0005 Amendment 3): the light view adds the material's emitted radiance to surface pixels when the lights are on (`debug_view::Lighting::emission`). It is never in the shaded, accumulated or filtered radiance, so the filter's guides and demodulation stay the surface's own, and every test on that radiance measures reflected light only (4A's recommendation).
3. **Lights** (Phase 4 decision 3):
   - On while the sun is below the horizon (`light::emitters::lights_on`), only for a scene with an emitter table (`--scene night`); the M3 street never lights.
   - L flips the automatic state by hand; `--lights auto|on|off` (default `auto`).
   - A switch is a light jump: the history resets (`History::set_lights`, `ResetCause::lights`).
4. **Night exposure** (decision 3): automatic exposure in the viewer.
   - `gpu::exposure`: one small compute pass after the filter, over every 4th pixel in x and y: per frame the sum of displayed luminance (reflected + emission), and the sum of its log and the count over the pixels at least 2⁻¹⁰ of the previous frame's mean (the 4A metric exposure's rule, one frame late). Read back two frames later (frames in flight), host-visible, 32 B.
   - `light::exposure::Adaptation`: the target is 0.18 over the log-average; the log exposure follows it with a 1 s time constant; bounded to [2⁻², 2¹⁶]; − and = still offset it in stops; the first measurement is taken at once.
   - `--exposure auto|sky`: `auto` is the default for `--scene night`, `sky` (M3's sun-and-sky exposure, unchanged) for `--scene street`.
5. **Relight rules for emitters** (ADR-0006 Amendment 2), for edits while the lights are on. A pixel whose history would otherwise be accepted restarts as *relit* when, besides 3E's sun and sky rules:
   - **E1, a light changed:** `Relight::power` bounds the emitter power (π · area · luminance) the edit adds or removes: every face of every edited voxel whose material emits before or after, and the face of each emitting neighbour that the edit may cover or expose. A Lambertian emitter set of power ΔΦ inside the box changes the reflected radiance at a point at distance d from the box by at most ΔΦ / (π² d²) (albedo ≤ 1). The pixel restarts when that bound is at least **2%** of its history's luminance.
   - **E2, a light shadowed or unshadowed:** each box lists up to 8 emitters of the table with the largest Φ / (d² + 1), d from the emitter's centre to the box. A pixel restarts when the segment from its surface point to a listed emitter's centre crosses the box grown by that emitter's half-diagonal (plus 3E's 1-voxel margin), and the most that box can block of that emitter, L_Y · A / (π d²) (A: the grown box's ab + bc + ca, d from the point to the box), is at least 2% of its history's luminance.
   - **Declared approximation:** emitters not listed, and indirect (bounce) changes, restart nothing. Measured by G5's data, like 3E's R2.
   - Both terms are 0 when the lights are off.
6. **Viewer:** `--scene night` renders its lights (emitters in the shade pass, emission in the light view, the exposure pass), `--emitter-samples k`, L, `--lights`, `--exposure`; the title shows the lights and the exposure; the JSON line gains `lights`, `emitter_samples`, `exposure` and the exposure pass's GPU time.
7. **`run-local.cmd`** rewritten for 4B's GPU checks.

**Excluded:** reservoirs (4C+), path reuse, any change of estimator, default or cost of the M3 street (`--scene street`), any change of the filter or temporal defaults, bloom or glare, the night sky.

## Criteria (frozen 2026-09-25, before any code or run)

Surface pixels only unless stated. "Reference" is `gpu::reference` with `emitters_direct` and `emitters_indirect`, `max_bounces` 1, **emission off** (the real-time radiance holds reflected light only). Night is 21 h, blue hour 18.5 h; `street_night` Full unless stated; cameras are 4A's (street, low).

**Pure and device-free (cloud and laptop):**

| # | Criterion |
|---|---|
| C1 | Layout: the shade module's reflection matches the host (with the emitter binding and the 80-byte emitter), and the exposure module's; a moved field is refused (negative control). The debug-view layout is unchanged. |
| C2 | Relight rows (`gpu --lib`): the header, the box rows and the 8 light rows per box are where the shader reads them; more than 16 boxes merge into the last (powers add, the merged list keeps the 8 largest). `changed_power`: 0 for an edit that touches no emitting voxel or neighbour; on `street_night` through the real 1C pipeline, for removing a lamp voxel, removing a bulb voxel and adding a neon voxel in open air, the bound is ≥ the exact change of the table's total power (from-scratch tables before and after), and ≤ 20 × it. The listed lights are the 8 largest by Φ / (d² + 1) (checked against a brute-force sort). |
| C3 | Exposure (`light` unit tests): `Adaptation` takes the first target at once; after 1 s of 60 fps frames toward a new target the log-exposure has closed 1 − 1/e of the gap (within 1%); it stays within [2⁻², 2¹⁶]; stops multiply it by 2^stops. The sums the GPU writes, combined on the host (`exposure_from_sums`), equal `metric_exposure` on the same pixels when the floor is the image's own mean (within 10⁻⁹ relative, f64). |
| C4 | Builds and lints: `gpu` (every test binary), `viewer`, `ref_light`; clippy on the workspace with only the 6 pre-existing `world` lints; the pure suite passes (153 + the new tests). |

**GPU (laptop, `run-local.cmd`; validation on unless stated; 0 validation errors in every test):**

| # | Criterion |
|---|---|
| G1 | **Exact, per pixel.** Frame f of the shade pass with emitters equals the reference's sample f per pixel (relative 10⁻⁵; 0 where the reference is 0), 480×270, both cameras. Arms: (a) emitters only, bounce on (sun and sky off); (b) uniform sky of 1 plus emitters, bounce on; (c) the point sun at 8 h plus emitters, bounce on (the lights forced on); (d) emitters only, bounce off, against `max_bounces` 0. Each arm: ≤ 0.1% of pixels mismatched (the 3B–3F edge set), 0 bad ids, ≥ 1000 pixels changed by the emitters. **Controls:** emitters off fails every arm with ≥ 10× the correct arms' mismatches; frame f + 1 differs; `emitter_samples` 2 differs. **M3 unchanged:** with `emitters` off, the pass bound with `bind_lit` equals the pass bound with `bind`, bit for bit (street camera, day and night). |
| G2 | **Convergence** (the k > 1 arms and the real transport): the shade pass with emitters and the bounce, averaged over 1024 frames at 320×180, against the reference at 16,384 spp, night, both cameras, Full and Dense, k = 1, 2, 4. Pass: the image mean per channel |z| < 4, and the pixel |z| > 4 rate ≤ max(1%, 1.5 × null + 0.2%), null the same rate of the reference against a second reference of other seeds (4A's G4 metric). **Control:** `no_solid_angle` fails (image-mean |z| > 10) in every arm it runs (k = 1, both cameras, Full). Blue hour at k = 1 is data (the table sky against the Monte Carlo sky, and the heavy tail 4A found). |
| G3 | **Emission in the light view:** drawing the light view (R8G8B8A8_UNORM) with a zero radiance buffer and the lights on gives aces(exposure · L_e[material]) on every surface pixel within 1/255 + 10⁻³, and exactly the lights-off image with the lights off; ≥ 1000 emissive pixels on the street camera at night. |
| G4 | **Night quality after the temporal pass and the filter** (M3's Q1, Q2, Q4 on reflected light): 1920×1080, the viewer's frame (bounce, temporal defaults, filter defaults, emitters, k = 1) from a fresh history, against the reference at 16,384 spp. **Q1** energy: the shown mean luminance within ±2% of the reference at ages 1, 4, 16, 64. **Q2** error: filtered relative MSE ≤ raw at gain × age (gains 8 / 4 / 4 / 1 at ages 1 / 4 / 16 / 64). **Q4** FLIP (`results/phase4b/flip.py`, the display with emission added and the reference's metric exposure): filtered ≤ raw at ages 1, 16, 64. Judged at **night**, both cameras; **blue hour is data** (4A's recommendation). Also data: the reference's own noise, the filtered error at night against 3G's at dusk (the night noise 4C must beat). |
| G5 | **Relight with emitters** (960×540, street camera, night, the viewer's frame), three edits: remove the nearest lamp head's voxels in a 2³ box (E1), add a 2³ neon box in open air 1 m from a façade (E1), place a 4³ box of stone next to a lamp head between it and the road (E2). **R1:** after each edit the pixels the GPU marks relit equal the host's rule (3E + E1 + E2 on the same rows) with ≤ 0.05% of pixels differing and ≥ 100 relit; the arm with 3E's rule only (the boxes without power and lights) must fail R1 in at least one edit. **R2:** the pixels whose converged value (4096 frames) changed by > 25% outside the rebuilt regions: 4 frames after the edit, their filtered relative MSE ≤ 3 × that of an arm reset at the edit, in each edit with ≥ 100 such pixels; the 3E-only arm must fail R2 in at least one edit. **Data:** per edit, the changed pixels not relit (the declared approximation's misses) and the relit pixels that did not change (its cost). |
| G6 | **Lights switch and exposure pass:** switching the lights on (and off) with a still camera resets every surface pixel (reason reset, age 1) in that frame only, and the next frame accepts. The exposure pass's sums over a 1080p night frame equal the host's sums over the read-back radiance plus emission within 10⁻³ relative, and its log-average exposure equals the host's `exposure_from_sums` within 10⁻³. |
| G7 | **Viewer, edits with the lights on** (release, validation off, 1920×1080, `--scene night --hour 21`, the scripted fly path, 2,000 frames): `--edit-size` 1, 8, 32 with p95 edit-to-visible ≤ 50 / 50 / 100 ms, every edit shown, 0 deferred. With validation on (400 frames, sizes 1, 8, 32, and `--run-day` from 17.5 h across the switch): exit 0, 0 errors, 0 warnings. |
| G8 | **Regressions:** `gpu --lib`; `gpu --test emitters` (4A G1–G6), `shade`, `sky`, `bounce`, `temporal`, `edit`; `denoise` R1 / R2 (`edits_relight_what_they_change*`, the changed relight rows); the viewer's M3 street run (`--edit-script`, N = 1, validation off): p95 ≤ 50 ms, every edit shown, the JSON reports no lights. |

**Measurements (fixed method, not pass/fail):**

| # | Measurement |
|---|---|
| M1 | **The equal-time curve** (validation off, 1920×1080, street camera, night): for each dressing (Lamps, Windows, Full, Dense) and k = 1, 2, 4: the shade pass's GPU time (median of 64 frames after 16 warm-up frames, emitters and the bounce on), and the one-frame relative MSE of the emitter-direct term alone (bounce, sun and sky off, 16 frames) against the reference's emitter-direct term (`max_bounces` 0, 4,096 spp, 960×540 for both). Reported with error × time, the efficiency 4C must beat at equal time. Also the whole frame's passes (shade, temporal, filter, exposure) at Full, k = 1. |
| M2 | **Night frame cost in the viewer** (release, validation off, 1920×1080, MAILBOX, the looping walk, 3,000 frames, `--scene night --hour 21`, Full and Dense): p50 / p99 frame interval and per-pass GPU time. Not a pass criterion (4G's P judges); it tells whether the control fits the 16.67 ms. |

**Work units:** 3 (M4 weight). Budget: G2–G5 references about 20–30 min of GPU once (cached in `NE_GATE_DIR`); M1 about 10 min; M2 and G7 about 10 min. Failures are diagnosed and reported, never rebaselined.

## Design notes

- **Why emission is added in the light view, not in the shade pass:** ADR-0005 Amendment 3 requires it after reconstruction. The shaded radiance is reprojected, averaged and demodulated by albedo; an emitter's own radiance (up to 10⁴ × its reflected light at night) would dominate the history's moments and the filter's luminance edge-stopping, and add nothing to estimate (it is exact per pixel).
- **Why E1 / E2 are bounds against the pixel's own history:** at night the useful scale is the pixel's own light, which spans orders of magnitude between a lamp's pool and a dark wall. A fixed radius (3E's sky rule) would relight half the street for a bulb or miss a lamp's far pool. The bounds are conservative for direct light (albedo ≤ 1, cosines ≤ 1, the whole box's solid angle) and do not cover indirect light, which G5's data measures.
- **Why k > 1 only at the primary vertex:** it is the term reservoir reuse (4C) replaces, so the curve shows what 4C competes with. The bounce hit keeps one sample, as 4D's control is the 3F bounce.

## Commands and evidence actually produced

NOT RUN yet.

## Next

1. Implement, run the cloud checks (C1–C4), write `run-local.cmd`, commit and push.
2. The user runs `run-local.cmd` on the laptop and pushes `engine/results`; G1–G8, M1, M2 are analysed from those logs.
3. The user judges the night look in the viewer (lights, colours, exposure).
