# Change: Phase 4B — many lights without reuse (the control)

Status: **run on the RTX 3050 (2026-09-26 09:20, commit `cd28474`): 27 of 29 steps pass; G4 and G5 FAIL. Local session 2026-09-26: G5 rerun with stronger edits (R1 and R2 pass in all 3 edits; R2's control still not caught, underpowered at night); G4 diagnosed (the filter's luminance edge-stopping). **G5 settled (S-026):** R2's control accepted as underpowered at night; ADR-0006 Amendment 2 accepted. **Open on G4:** the filter fix is authorized as the next task, in a cloud session, in its own record (S-026).** Criteria frozen 2026-09-25, before any code or run (commit `1da5bc8`); four corrections made before any run are recorded next to C2, G1, G4 and G5. Written in a cloud session (no GPU; [CLOUD.md](../CLOUD.md)); the laptop logs were analysed in a cloud session.
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
   - `gpu::exposure`: one small compute pass after the filter, over every 4th pixel in x and y: per frame the sum of displayed luminance (reflected + emission), and the sum of its log and the count over the pixels at least 2⁻¹⁰ of the previous frame's mean (the 4A metric exposure's rule, one frame late). Read back two frames later (frames in flight), host-visible: 64 workgroups' partial sums per frame, added on the host (2 × 64 × 16 B; first built as one workgroup, see the Observations).
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
| C2 correction (2026-09-25, at C2's first run, before any GPU run) | The oracle "the exact change of the table's total power" is a **net** change: removing the lamp's first voxel (a corner of the head) removes 3 emitting faces and exposes 3, so the net change is exactly 0 and "≤ 20 × it" cannot hold for any bound. The rule bounds the **gross** power added or removed, so the oracle is corrected to the gross change: the unit voxel faces that stop or start emitting around the edit, from the world before and after (π × luminance each). The check also asserts that this gross change is ≥ the table's net change (from-scratch tables through the 1C pipeline, as frozen). The limits (≥, ≤ 20×) are unchanged. First run: lamp bound 1.546 against gross 1.031 (net 0); bulb 0.1178 against 0.0982 (net 0.0982); neon 0.0442 against 0.0442; the quiet edit 0. |
| C3 | Exposure (`light` unit tests): `Adaptation` takes the first target at once; after 1 s of 60 fps frames toward a new target the log-exposure has closed 1 − 1/e of the gap (within 1%); it stays within [2⁻², 2¹⁶]; stops multiply it by 2^stops. The sums the GPU writes, combined on the host (`exposure_from_sums`), equal `metric_exposure` on the same pixels when the floor is the image's own mean (within 10⁻⁹ relative, f64). |
| C4 | Builds and lints: `gpu` (every test binary), `viewer`, `ref_light`; clippy on the workspace with only the 6 pre-existing `world` lints; the pure suite passes (153 + the new tests). |

**GPU (laptop, `run-local.cmd`; validation on unless stated; 0 validation errors in every test):**

| # | Criterion |
|---|---|
| G1 | **Exact, per pixel.** Frame f of the shade pass with emitters equals the reference's sample f per pixel (relative 10⁻⁵; 0 where the reference is 0), 480×270, both cameras. Arms: (a) emitters only, bounce on (sun and sky off); (b) uniform sky of 1 plus emitters, bounce on; (c) the point sun at 8 h plus emitters, bounce on (the lights forced on); (d) emitters only, bounce off, against `max_bounces` 0. Each arm: ≤ 0.1% of pixels mismatched (the 3B–3F edge set), 0 bad ids, ≥ 1000 pixels changed by the emitters. **Controls:** emitters off fails every arm with ≥ 10× the correct arms' mismatches; frame f + 1 differs; `emitter_samples` 2 differs. **M3 unchanged:** with `emitters` off, the pass bound with `bind_lit` equals the pass bound with `bind`, bit for bit (street camera, day and night). |
| G1 correction (2026-09-25, at the final review of the code, before any run) | The tolerance 10⁻⁵ is 3B–3F's, where every term is a binary visibility times quantities fixed by the stream and the exact face normal. The emitter term is continuous in the surface point p, which the real-time pass rebuilds from depth while the reference takes it from its ray hit, and its f32 solid angle Σgᵢ − 2π carries about 10⁻⁶ sr of rounding that differs with p's bits: 10⁻⁴ relative at the 10⁻² sr switch, 10⁻⁵ at 0.1 sr, the range of the neon, windows and lamp faces seen from nearby façades. So G1 is judged at **10⁻³ relative**; the pixels over 10⁻⁵ are reported as data. A different stream order, pdf or estimator moves a sample by O(1), so the controls keep their meaning. Everything else in G1 is unchanged. |
| G2 | **Convergence** (the k > 1 arms and the real transport): the shade pass with emitters and the bounce, averaged over 1024 frames at 320×180, against the reference at 16,384 spp, night, both cameras, Full and Dense, k = 1, 2, 4. Pass: the image mean per channel |z| < 4, and the pixel |z| > 4 rate ≤ max(1%, 1.5 × null + 0.2%), null the same rate of the reference against a second reference of other seeds (4A's G4 metric). **Control:** `no_solid_angle` fails (image-mean |z| > 10) in every arm it runs (k = 1, both cameras, Full). Blue hour at k = 1 is data (the table sky against the Monte Carlo sky, and the heavy tail 4A found). |
| G3 | **Emission in the light view:** drawing the light view (R8G8B8A8_UNORM) with a zero radiance buffer and the lights on gives aces(exposure · L_e[material]) on every surface pixel within 1/255 + 10⁻³, and exactly the lights-off image with the lights off; ≥ 1000 emissive pixels on the street camera at night. |
| G4 | **Night quality after the temporal pass and the filter** (M3's Q1, Q2, Q4 on reflected light): 1920×1080, the viewer's frame (bounce, temporal defaults, filter defaults, emitters, k = 1) from a fresh history, against the reference at 16,384 spp. **Q1** energy: the shown mean luminance within ±2% of the reference at ages 1, 4, 16, 64. **Q2** error: filtered relative MSE ≤ raw at gain × age (gains 8 / 4 / 4 / 1 at ages 1 / 4 / 16 / 64). **Q4** FLIP (`results/phase4b/flip.py`, the display with emission added and the reference's metric exposure): filtered ≤ raw at ages 1, 16, 64. Judged at **night**, both cameras; **blue hour is data** (4A's recommendation). Also data: the reference's own noise, the filtered error at night against 3G's at dusk (the night noise 4C must beat). |
| G4 correction (2026-09-25, at a review of the run before any GPU run; S-025's session) | **Q1 as frozen cannot pass at night, even with correct code.** One real-time frame's mean luminance over the surface has a relative 1-sigma of **7.5% (street) and 8.3% (low)** at 1080p (`light` `diagnostic_g4_noise_floor`, CPU, the G4 estimator: one bounce, emitters at both vertices, 21 h, 192×108 at 2,048 spp scaled by √pixels; the noisiest 1% of pixels carry 89–97% of the variance), so ±2% is about ¼ sigma at age 1, ½ at age 4, 1 at age 16 and 2 at age 64: a pass at all 8 judged points would be luck (probability well under 1%). M3's Q1 held at day because the dusk noise has no heavy tail (4A M2a). Q1 is split so each part measures what it names: **Q1a**, the filter keeps energy: \|bias(filtered) − bias(raw)\| ≤ 2% at each age, both from the same frames (`run_still` replays frames 1..64 with the same seeds, and the filter writes only the shown radiance, so the history is the same and the frame noise cancels); **Q1b**, the history is unbiased: \|bias(raw)\| ≤ max(2%, 4σ/√age), σ the frame's 1-sigma from the reference's per-sample variance (the test already printed it as data; now it also prints each age's z). Estimator bias at 320×180 over 1,024 frames stays G2's (z-tests). Q2 and Q4 are unchanged: they compare arms, and a Q2 failure at night is the information 4C needs. |
| G5 | **Relight with emitters** (960×540, street camera, night, the viewer's frame), three edits: remove the nearest lamp head's voxels in a 2³ box (E1), add a 2³ neon box in open air 1 m from a façade (E1), place a 4³ box of stone next to a lamp head between it and the road (E2). **R1:** after each edit the pixels the GPU marks relit equal the host's rule (3E + E1 + E2 on the same rows) with ≤ 0.05% of pixels differing and ≥ 100 relit; the arm with 3E's rule only (the boxes without power and lights) must fail R1 in at least one edit. **R2:** the pixels whose converged value (4096 frames) changed by > 25% outside the rebuilt regions: 4 frames after the edit, their filtered relative MSE ≤ 3 × that of an arm reset at the edit, in each edit with ≥ 100 such pixels; the 3E-only arm must fail R2 in at least one edit. **Data:** per edit, the changed pixels not relit (the declared approximation's misses) and the relit pixels that did not change (its cost). |
| G5 setup corrections (2026-09-25, while writing the test, before any run) | (1) **The converged value:** 4,096 real-time frames at night leave a per-pixel relative standard error of about 7% at the median and over 30% at the 90th percentile (4A's M2: one-sample σ/μ median 3.3–4.7, p90 13–21), so "> 25%" would count noise as change. The converged value is instead the reference (the same estimator per sample, G1; 16,384 spp), and a changed pixel must also differ by more than 4 standard errors. The 25% threshold and every limit are unchanged. (2) **The E2 edit's place:** from the street camera the lamps' pools on the road are out of view (the lamps stand behind and to the side of it), so a box between a lamp head and the road would shadow nothing visible. The stone box goes between the nearest lamp head and the shop façades across the road (its +z side, 1 voxel from the head), which the camera sees. |
| G5 correction (2026-09-26, **after** the laptop run; the user's "go 1 and 2") | At the run only the stone edit had ≥ 100 changed px, so R2's control was judged once (3E-only 1.7× reset against 3×). The first two edits are made stronger so R2 can judge them: **the whole nearest lamp head** (48 voxels, E1) instead of its 2³ corner, and **a 0.5 m neon cube (8³, `neon_pink`) in open air 0.5 m in front of a façade at 1–1.5 m height** instead of a 2³ box 1 m out. The stone edit is unchanged. The test prints how many street-camera px the edited box covers. Every limit (R1 0.05% / ≥ 100, R2 > 25% and > 4 SE, 3×, ≥ 100 px) is unchanged. Result under "Local session 2026-09-26". |
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

Cloud session, Linux x86_64, 4 cores, no GPU. The `gpu` crate is built with the pinned slangc 2026.13.1 (the Linux release, in the session's scratch space) and the distribution's `spirv-val` (SPIRV-Tools 2025.1~rc1, not the SDK's): a supplemental compile check, not the ADR-0001 toolchain on the laptop.

| Check | Result | Scope |
|---|---|---|
| C1 layout (`gpu --lib`: `shade::tests::host_layout_matches_the_compiled_module`, `exposure::tests::host_layout_matches_the_compiled_module`, `shade::tests::emitter_flags_follow_the_settings`, `exposure::tests::samples_are_every_stride_th_pixel`) | pass; the moved-field controls are refused; the debug-view layout is unchanged (the existing reflection checks pass) | cloud, no device |
| C2 relight rows and `changed_power` (`temporal::tests::relight_rows_follow_the_shader_layout`, `temporal::tests::changed_power_bounds_the_edit`) | pass after the oracle correction above: lamp voxel bound 1.546 against gross 1.031 (net table change 0), bulb 0.1178 / 0.0982 (net 0.0982), neon 0.0442 / 0.0442 (net 0.0442), the quiet edit 0; the listed lights equal the brute-force 8 largest | cloud, no device |
| C3 exposure (`light`: `exposure::tests::adaptation_follows_the_target`) | pass | cloud |
| C4 builds and lints | `gpu` (lib, both tools, every test binary including `lights`) and `viewer` build with no warnings; clippy `--workspace --all-targets`: exit 0, only the 6 pre-existing `world` lints (`engine/results/clippy_4b_cloud.log`) | cloud, substitute SDK |
| Pure suite (`cargo test --release -j 2`) | exit 0, **154 pass** (153 + C3), 4 ignored; `engine/results/test_pure_4b_cloud.log` | cloud |
| `gpu --lib` | exit 0, **21 pass** (15 + C1 4 + C2 2); `engine/results/test_gpu_lib_4b_cloud.log` | cloud, no device |
| G1–G8, M1, M2 | **NOT RUN** (no GPU in the cloud): `run-local.cmd` runs them on the laptop | — |

**`run-local.cmd`** (rewritten for 4B, untested: it cannot run in the cloud; re-read against [CLOUD.md](../CLOUD.md)'s rules): the pure suite, `gpu --lib`, the builds; G1–G6 one test each (`--test lights`, validation on, `NE_GATE_DIR` = `%TEMP%\ne_gate_4b` for G4's 1080p references); G4's FLIP (`results/phase4b/flip.py`, NOT RUN if Python is missing); M1 (validation off); G7 (night edits N = 1, 8, 32, validation off; then validation on with N = 1, 8, 32 and the day running across sunset); G8's M3 street edit run; M2 (the night walk, Full and Dense); the G8 regressions (`emitters`, `shade`, `sky`, `bounce`, `temporal`, `edit`, `denoise` R1 / R2). NOT RUN on purpose: `gate` (35+ min; G1 checks the M3 path bit for bit with the lights off), the rest of `denoise` (accepted failures), `reference`, `raster`, `ray`, `device`. About 1.5–2.5 hours (first written as 90 minutes, a guess; re-estimated from 4A's laptop reference times, 90–145 s per 960×540 image at 16,384 spp and 8 bounces, with G4 rendering four 1080p images at 1 bounce). Logs in `engine/results/local-run/<date>-4b/`, FLIP in `engine/results/phase4b/`.

## Observations: the user's first look in the viewer (2026-09-25, RTX 3050, not criteria)

The user ran the viewer at night (`--scene night`, 21 h, Full) before the laptop test run; screenshots in the session, with F3 (added for this, commit `aba9730`) showing every setting.

- It runs: the lights, the lights rule, the automatic exposure (about 1.2–1.7 × 10⁴ at 21 h) and the history work; with the day stopped the history is full (age view white, reason view all accepted). About 120–200 fps at 1080p.
- **The first look was noisier than it should be because the day was running** (`--run-day`): the sun-motion cap held the history at about 15 frames although the sun gives no light at night. **Fixed** (the user approved the fix, then −12° after the measurement; S-025, [ADR-0006 Amendment 3](../adr/ADR-0006-guides-and-history.md)): no cap below −12° sun elevation. See the section below.
- **With a full history, two artefacts remain**, located by toggling B and N:
  - coloured specks on the road: present with the bounce on and the filter on, absent with the bounce off. They are rare bright samples of the bounce hit's one emitter sample (a bounce landing on a façade near neon), which the filter's luminance edge-stopping keeps and spreads into dots;
  - blotchy neon light on the shop fronts: present with the bounce off too, so it is the primary vertex's one sample chosen by power, which ignores distance (a façade next to a neon tube rarely picks it). `--emitter-samples 4` visibly reduces it, at about 8.6 ms frame p50 with the filter (within 16.67 ms).
  - Both are the per-frame night noise 4A measured (p90 σ/μ 13–21) and what reservoir reuse (4C, not authorized) targets.
- **F3 numbers** (1920×991 window, validation on, 21 h, Full, bounce off, filter on, k = 1, 13,134 frames): GPU p50 gbuffer 0.57 ms, shade pass 1.60, temporal 1.16, filter 2.49, **exposure 1.24**, view 0.22; frame p50 6.4 ms, 1% low 67 fps; 2 edits shown in 24.2 / 27.5 ms; 0 validation errors, 0 warnings, no leaks. The exposure pass was one workgroup, so one multiprocessor did all 129,600 loads in series; it now runs 64 workgroups whose partial sums the host adds (the same sums, so C1 and G6 are unchanged; its new cost is measured by the laptop run).
- The lit upper windows are blown out to white at the automatic exposure: a look decision (window luminance or exposure key) for the user.

## The night history cap (S-025, 2026-09-25, cloud)

- **Measurement first** (`light::sky` `diagnostic_skylight_below_horizon`, on demand, 0.4 s; the real-time sky model computed directly, without the S-020 correction, which does not change the scale). Skylight radiance on an albedo-0.3 surface, in units:

  | Sun elevation | 0° | −3° | −6° | −9° | −12° | −15° | −18° |
  |---|---|---|---|---|---|---|---|
  | Radiance | 8.0 × 10⁻⁴ | 1.3 × 10⁻⁴ | 5.4 × 10⁻⁶ | 2.4 × 10⁻⁷ | 1.1 × 10⁻⁸ | 3.3 × 10⁻¹⁰ | 3.2 × 10⁻¹¹ |

  It falls by 1.6–7× per degree (about 2× typically). Against the night street's mean reflected emitter light (about 2 × 10⁻⁵, estimated from the 4A record), the sky is about 25% at −6°, 1% at −9° and 0.05% at −12°.
- **Change:** `TemporalSettings::sun_cap_min_elevation_deg` (default −12°); `TemporalSettings::age_cap` takes the sun's elevation and returns `max_age` below it. The light-jump reset is unchanged, and blue hour (−5.3°) keeps the cap. Daytime and every M3 test time are above −12°, so M3 behaviour is unchanged.
- **Checks (cloud, no GPU):** `sun_cap_is_off_when_the_sun_is_well_below_the_horizon` passes (day, blue hour and −12° keep caps 8 / 16 at 0.0625 / 0.03° per frame; −12.001° and 21 h give 64; a planted −90° cutoff keeps 8 at 21 h). Pure suite: exit 0, 154 pass, 6 ignored (this diagnostic and G4's noise-floor one) (`engine/results/test_pure_4b_cap_cloud.log`, rerun after the G4 correction). `gpu --lib`: exit 0, 22 pass (`engine/results/test_gpu_lib_4b_cap_cloud.log`). Clippy `--workspace --all-targets`: exit 0, only the 6 old `world` lints (`engine/results/clippy_4b_cap_cloud.log`; substitute SDK, supplemental, not ADR-0001 proof). The first clippy attempt failed only because clippy was not installed for the toolchain.
- **NOT RUN:** the viewer at night with the day running (the age view should go white once the sun is below −12°, hour 19.14). G7's validation run with the day running (400 frames from 17.5 h at 3.75° of sun per second) reaches −12° only below about 61 fps, so it may not exercise the change. `run-local.cmd` already runs `gpu --lib`, so it carries the new test.
- **G4's noise floor** (same session, before any GPU run): measured and corrected; see the "G4 correction" row in the criteria (`engine/results/diag_g4_noise_floor_cloud.log`).

## Laptop run (RTX 3050, 2026-09-26 09:20, commit `cd28474`; analysed in a cloud session, no GPU)

Logs: `engine/results/local-run/2026-09-26_0920-4b/` (`summary.txt`: 27 of 29 steps exit 0); FLIP: `engine/results/phase4b/`. Validation: 0 errors, 0 warnings in every GPU test and viewer run.

| # | Result |
|---|---|
| G1 | **pass.** 8 arms: 30–102 mismatched px (≤ 0.08%), 0 bad; emitters-off control fails all 8 arms (268,762 px); frame and sample-count controls caught; M3 path bit-identical (0 px) at 8 h and 21 h, bounce on and off. |
| G2 | **pass.** Every judged arm: image-mean \|z\| ≤ 2.35, pixel \|z\|>4 rate within its limit; `no_solid_angle` caught (\|z\| > 2,300). Blue hour (data) also within limits. |
| G3 | **pass.** 9,061 emissive px, 0 off by more than 1/255 + 10⁻³; lights off 0 nonzero. |
| G4 | **FAIL: Q1a at all 8 night points, Q2 at low_night age 1.** Q1b passes everywhere (raw history bias \|z\| ≤ 1.64): the accumulation is unbiased. **The filter removes energy:** filtered minus raw mean luminance −9.9 / −7.2 / −24.4 / −16.3% (street, ages 1 / 4 / 16 / 64) and −8.6 / −4.6 / −19.6 / −12.9% (low); blue hour (data) −3 to −17%. The old ±2% Q1 would also have failed (filtered bias −8 to −26%), so the correction did not cause this. Q2: low_night age 1 filtered rel_mse 73.1 against raw at age 8 49.1 (street passes, 9.0 against 55.8); blue hour low age 1 also above (data). Q4 FLIP **pass** (filtered < raw at ages 1, 16, 64, both night cameras; e.g. street 0.406 → 0.309 at age 1, 0.193 → 0.123 at 64). |
| G5 | **FAIL: "R2 control (3E rules only) not caught in any edit".** R1 passes in all 3 edits (0 decisions differ; 372k–395k px relit) and the 3E-only arm fails R1 in all 3 (control caught). R2 passes wherever judged, but only the stone edit had ≥ 100 changed px (10,020): full 0.107 = reset 0.107, 3E-only 0.179, which is 1.7× reset, under the 3× limit. The lamp-voxel and neon edits changed 0 px by > 25% and > 4 SE, so they judge nothing. |
| G6 | **pass.** Switch on / off resets all 395,018 surface px in that frame only, the next frame accepts all; exposure sums equal the host's exactly, exposure 14348.447 against 14348.454 (5 × 10⁻⁷). |
| G7 | **pass.** Validation off, 2,000 frames: p95 edit-to-visible 32.6 / 33.6 / 34.5 ms (N = 1, 8, 32; limits 50 / 50 / 100), every edit shown, 0 deferred. Validation on (N = 1, 8, 32 and the sunset run): exit 0, 0 errors, 0 warnings. The sunset run ended at hour 18.24 (135.6 fps average), so it did not reach −12° and did not exercise S-025, as predicted. |
| G8 | **pass.** `gpu --lib` (22), `emitters`, `shade`, `sky`, `bounce`, `temporal`, `edit`, `denoise` R1 / R2 all exit 0; M3 street viewer run p95 29.0 ms, every edit shown, no lights in its JSON. |
| M1 | Shade pass at 1080p, k = 1 / 2 / 4: Lamps 2.19 / 2.56 / 3.27 ms, Windows 2.48 / 2.57 / 3.35, Full 2.47 / 2.95 / 3.96, Dense 3.49 / 4.11 / 5.65. One-frame emitter-direct rel_mse: Lamps 28.2 / 12.8 / 6.8, Windows 51.9 / 34.9 / 19.3, Full 6,786 / 15,318 / 9,280, Dense 23,610 / 23,248 / 9,583. Full and Dense do not fall with k: a few pixels dominate (heavy tail, as `diagnostic_g4_noise_floor` found), so 16 frames do not estimate their rel_mse stably. Whole frame at Full, k = 1: temporal 1.31, filter 3.13, exposure 1.23 ms. |
| M2 | Night walk, 3,000 frames, 1080p MAILBOX: Full frame p50 7.89 / p99 11.99 ms (shade 7.23, filter 2.84, temporal 1.35, exposure 1.05 ms p50); Dense 8.83 / 11.70 ms (shade 8.16). The control fits 16.67 ms. |

**Diagnosis (hypotheses, not yet tested):**

- **G4 Q1a, the filter's energy loss.** The filter (`shaders/denoise.slang`, SVGF-style) weights each neighbour by exp(−\|Δl\| / (σ_l √variance)) and normalises per output pixel, so the weights are not symmetric. A rare bright sample has a large variance and averages itself down with its neighbours, while each dark neighbour, whose variance is small, rejects it, so its energy is not spread and is lost. With the night's heavy tail (1% of pixels hold 89–97% of the variance), this is a large loss; with M3's dusk noise it stayed within ±2%. The first check is a per-pixel map of the filtered-minus-raw energy against the raw value's rank. Changing the filter is **outside 4B** ("any change of the filter or temporal defaults" is excluded).
- **G4 Q2 at low_night age 1:** the same heavy tail. It is the information 4C needs (the criteria say so).
- **G5's R2 control has no power, and the engine did nothing wrong.** The full arm equals the reset arm. The 3E-only arm's stale history is 1.7× the reset arm's 4-frame error, since at night the reset arm's noise is large, and only one edit changed enough pixels to be judged. Fixing it means stronger edits (e.g. the whole lamp head, a neon box in view). That is a test change after the run, so it needs the user's decision.

- **Does G4 propagate? (reasoning, not measured).**
  - It does not compound. The filter writes only the shown radiance; the history stays unbiased (Q1b), so the loss does not grow over frames.
  - It is on screen in every night and blue-hour frame, and it is uneven: largest where rare bright samples carry the light (neon near façades, the bounce). The automatic exposure partly hides the overall level.
  - **It confounds later measurements.** 4C lowers the noise, so it would also shrink the filter's loss and look better for that reason. 4D's extra bounces (7–10% of the night light) are the same size as the loss.
  - Day is unaffected (M3's Q1 within ±2%).
  - Hence: understand G4 before 4C.

## Local session 2026-09-26 (RTX 3050, commit `746a0db` + the uncommitted test changes; the user's "go 1 and 2")

Logs: `engine/results/local-run/2026-09-26_local-4b-g5-g4/` (`g5_relight.log`, `g4_diagnostic.log`). `NE_GATE_DIR` = `%TEMP%\ne_gate_4b` (G4's cached 1080p references). Validation on: 0 errors, 0 warnings in both. Clippy on `gpu --test lights`: only the old `world` lints. No engine code changed.

**G5 rerun with the corrected edits (exit 101, 472 s): R1 and R2 pass in all 3 edits; R2's control is still not caught.**

| Edit | Box covers (street px) | R1 full / 3E-only relit (differ) | R2 changed px (misses) | R2 rel_mse, 4 frames after: full / 3E-only / reset | 3E-only ÷ reset |
|---|---|---|---|---|---|
| remove the lamp head (48 vox) | 0 (out of view) | 394,910 (0) / 0 (394,910) | 93,453 (0) | 0.3309 / 0.4744 / 0.3309 | 1.43 |
| neon cube 8³ (512 vox) | 926 | 375,100 (0) / 16,256 (358,844) | 11,025 (0) | 0.3206 / 0.3070 / 0.3206 | 0.96 |
| stone beside the lamp head (unchanged) | 0 | 394,588 (0) / 0 (394,588) | 10,020 (0) | 0.1068 / 0.1786 / 0.1068 | 1.67 |

- The engine: in every edit the full rule equals the reset arm to four digits, R1 has 0 differing decisions, and no changed pixel is missed. The R1 control is caught in all 3 edits.
- **The R2 control fails in all 3, and stronger edits did not help (diagnosis, not tested further).** R2 compares relative MSE 4 frames after the edit, where the reset arm's night noise is already about 0.1–0.33. A stale history's error on a pixel that brightened is bounded near 1 relative (the neon edit: 3E-only is even slightly *below* reset, the stale dark value against a noisy bright one); on a pixel that dimmed, other lamps still light it, so the change is moderate. 3× reset is therefore out of reach at night for any edit this scene allows. The limit was carried over from 3E's day scenes, where reset noise is small.
- Not rebaselined. **The user decides:** accept the R2 control as underpowered at night (a recorded test limitation; the R1 control and R2's own pass carry the evidence), or change the control's measure (e.g. the changed pixels' bias, or more frames after the edit), which is another post-run criterion change.

**G4 diagnostic (`diagnostic_g4_filter_energy`, exit 0, 39 s, no criterion):** the viewer's frame at 1080p, Full, 21 h, both cameras, raw and filtered from the same frames; ablation arms use the filter's existing settings only.

Energy change of the filtered frame against the raw frame, ages 1 / 16 / 64:

| Arm | street night | low night |
|---|---|---|
| default (σ_l 4) | −10.0 / −24.9 / −16.3% | −7.5 / −19.4 / −12.8% |
| σ_l 1 / 2 / 8 | +0.2 / −20.0 / −14.5; −9.8 / −25.0 / −17.2; −7.0 / −22.9 / −14.1 | +4.4 / −14.4 / −10.7; −6.8 / −19.2 / −13.3; −4.7 / −17.8 / −11.1 |
| σ_l 16 / 64 | −4.6 / −20.4 / −11.7; −2.4 / −15.3 / −7.4 | −2.4 / −15.9 / −9.3; −0.3 / −12.2 / −6.0 |
| **σ_l 10⁶ (no luminance edge-stopping)** | **−1.5 / −0.3 / −0.4%** | **+0.5 / +0.1 / −0.2%** |
| prefilter off (plain SVGF) | −38.7 / −41.3 / −31.3 | −36.8 / −34.1 / −24.9 |
| prefilter_age 8 (the viewer's P) | −10.0 / −41.3 / −31.3 | −7.5 / −34.1 / −24.9 |
| variance blur off | −10.6 / −43.1 / −37.6 | −7.9 / −36.5 / −30.2 |
| levels 1 | +3.5 / −6.8 / −4.8 | +4.9 / −4.8 / −3.6 |
| **levels 0 (control: demodulate, remodulate)** | **0.0000** at all ages | **0.0000** at all ages |
| **dusk 17.5 h, M3 transport, default (control)** | **−0.36 / −0.15 / −0.03%** | **−0.28 / −0.13 / −0.03%** |

Where the default arm's change sits (street; low is alike): the brightest 0.1% of raw pixels carry 62 / 45 / 31% of the energy at ages 1 / 16 / 64 and lose 57 / 41 / 28 points of it; the lower half gains. The brightest 1% lose 67.5 / 53.6 / 45.6% of the total; the non-bright pixels within the filter's reach (6 px) take back **85 / 54 / 65%** of that (low: 89 / 56 / 64%); pixels further away change by < 0.5%. By the reference's per-sample variance the loss spreads over the 50th–99.9th percentiles, not the highest-variance 0.1%.

- **Observations:** the default arm reproduces the laptop run's G4 loss (street −10.0 / −24.9 / −16.3% against −9.9 / −24.4 / −16.3%; low −7.5 / −19.4 / −12.8% against −8.6 / −19.6 / −12.9%). The instrument can fail and does not: levels 0 is exact, the dusk arm keeps energy within 0.4%.
- **Conclusion (the hypothesis is confirmed in its main part):** the loss is made by the **luminance edge-stopping** (without it the filter keeps energy within 1.5% at night); it sits at the rare bright samples, which the filter averages down while the dark pixels within its reach take up only 54–89% of what leaves them. It is a night effect: with dusk noise the same filter keeps energy.
- **Refinement (hypothesis, not tested):** the loss grows from age 1 to 16 although σ_l 64 nearly removes it at age 1. From `moments_age` (8) the variance comes from each pixel's own moments; a dark pixel that has never drawn a bright sample has variance ≈ 0, so σ ≈ σ_l · 10⁻⁶ and it rejects every brighter tap whatever σ_l is. The variance blur and the 3×3 prefilter each soften this (turning either off makes it much worse), and the second level (step 2) carries most of the loss (levels 1: −7 to +5%).
- **The filter is not changed (outside 4B).** Options for the user: (1) a filter change, in its own record or as a 4B amendment: e.g. symmetric edge-stopping (σ from the larger of the two pixels' variances, so a tap weighs a pair the same both ways) or a variance floor for young moments, added as a swept setting and judged by G4's Q1a / Q2 / Q4 at night plus M3's Q1–Q4 at day; (2) accept the loss as a 4B limitation and let 4C, which shrinks the tail, be measured with the raw history rather than through the filter.

## Next

1. Done: implemented, cloud checks C1–C4 pass, `run-local.cmd` rewritten, committed and pushed.
2. The user runs `run-local.cmd` on the laptop (about 1.5–2.5 hours; overnight) and pushes `engine/results`; G1–G8, M1, M2 are analysed from those logs in a cloud session. Failures are diagnosed, never rebaselined.
3. The user judges the night look in the viewer: `viewer.exe --scene night --hour 17.5 --run-day` walks from dusk into night (lights, colours, exposure; L flips the lights).
4. Run done (2026-09-26): G4 and G5 fail, see "Laptop run". The user decides how to handle them (4B stays open). Then 4B closes (3 units) or its failures are accepted; M1's curve and G4's night error decide what 4C must beat. 4C is not authorized.
5. Done (2026-09-26, local): G5 corrected and rerun, G4 diagnosed ("Local session 2026-09-26"). **The user decided (S-026, 2026-09-26):** (a) G5's R2 control is accepted as underpowered at night, so G5 is settled and ADR-0006 Amendment 2 is accepted; (b) G4: the filter is fixed next, in a cloud session, in its own change record. 4B closes once G4 is resolved by that fix (or its remaining failure is accepted).
