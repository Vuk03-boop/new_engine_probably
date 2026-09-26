# Change: Phase 4B — many lights without reuse (the control)

Status: **part 1 built; cloud checks pass (C8–C10); GPU checks NOT RUN until the laptop run.** Criteria were frozen 2026-09-26, before any 4B code or run, and have not changed. Both parts are frozen here. A part 2 criterion may change only before part 2's first run, with the reason written in this record.
Date and baseline: 2026-09-26, branch `claude/what-is-next-jcbt1f` from `main` at `eb17538` (4A merged). Written in a cloud session (no GPU; [CLOUD.md](../CLOUD.md)).
Authorization: S-024 (4A and 4B authorized). The user asked for the plan before any code ("Do you need to plan 4b or is it already planned?", then "You are at max greenlight", 2026-09-26), then started part 1 ("You are on medium go", 2026-09-26).

## Objective

- **The street at night in the viewer, lit by its own lights.**
  - One emitter sample per pixel at the primary and bounce hits, using the reference's estimator (ADR-0005 Amendment 3), then the existing temporal pass and filter.
  - Emission is added after reconstruction.
  - The lights come on below the horizon, L switches them by hand, and the exposure adapts at night.
  - Sources: [Phase 4 proposal](2026-09-25-phase4-proposal.md) §4B and decision 3.
- **The control for 4C:** the equal-time curve over emitter samples per pixel (1, 2, 4) and light counts (the four dressings). Every reuse arm is compared with it.
- **Inputs to the 4C and 4D decisions:**
  - the filtered error at night beside M3's;
  - the visible gap left by bounces ≥ 2;
  - how much of the night noise is indirect.
- **Part 2:** an edit that changes a light, or a light's shadows, relights the pixels it reaches (ADR-0006 Amendment 2). This is measured like 3E's R2.

## Starting point (after 4A)

- `shade.slang` has the sun, the sky and one bounce, on the reference's PCG32 streams, with up to 4 rays per pixel. It has no emitters.
- `emitters_common.slang` has the emitter row, the alias draw and spherical rectangles. The estimator itself (`emitter_sample`) is in `reference.slang`.
- `GpuScene::build_lit` publishes the table (80 B rows and per-material emission) with the meshes and TLAS. `GpuScene::emitters` refuses a stale table.
- The viewer's `--scene night` builds the table, but nothing on screen uses it.
- The night references (4A G8) are 960×540 with 8 bounces, in `engine/results/phase4a_ref/`.

## Checked before planning (cloud, 2026-09-26)

The design relies on the M3 shader modules staying byte for byte what they are. This was checked with the pinned slangc 2026.13.1 (its Linux release, in the session's scratch space). The check is supplemental: it is not the laptop's SDK.

- **The shade shader:** `shade.slang` was given an `#ifdef EMITTERS` block (a binding, an emitter sample in `main`) and a changed header comment. Compiled without the define, its SPIR-V and reflection are **byte-identical** to main's. With the define, it compiles.
- **The shared include:** after an unused helper was added to `emitters_common.slang`, `reference.spv` and its reflection stay byte-identical.

## Design, part 1

### 1. Emitter light in the shade pass

- **Two modules from one source.** `build.rs` compiles `shade.slang` twice (its `SHADERS` table):
  - **Without `EMITTERS`:** main's M3 module, byte for byte (C10). It runs whenever the lights are off, so the M3 image and cost cannot change.
  - **With `EMITTERS`:** the lit module. It adds binding 8 (the scene's emitter rows) and the emitter terms. The emitter count goes in `Params.pad0` and `s` in `pad1`, so the push constants stay 128 B.
- **Where the samples are taken:** at the primary hit, and at the bounce hit when the bounce is on.
  - The pass draws `s` emitter samples (`ShadeSettings::emitter_spp`: 1, 2 or 4) and averages them. Their numbers come after the sun's two numbers and before the continuation's.
  - Each sample is the reference's: alias selection by power, a point uniform in the quad's solid angle (by area below 10⁻² sr), and one visibility segment.
  - With s = 1 the pixel's stream is the reference's one-bounce stream with `emitters_direct` and `emitters_indirect`. So frame f equals the reference's sample f per pixel (G10), as for the bounce (3F B1).
- **The reference is not touched.**
  - The lit block is written from the shared primitives. `reference.slang` stays byte-identical (C10), so the 4A validation of the reference still holds.
  - The shade pass's copy of the estimator is checked against the reference per pixel (G10).
- **The emitter light is reflected light.** It goes into the shade radiance, the history and the filter, demodulated by the albedo like the rest. Emission does not (item 2).
- No emitter numbers are drawn when the table is empty.
- **Rays per pixel:** up to 4 + 2s (M3: 4).
- **Held, not changed:** the sun's shadow ray still runs when the sun is below the horizon, where its light is 0. Skipping it is an exact lever held for 4G.

### 2. Emission and the exposure meter (`gpu::compose`, new)

- **When it runs:** a compute pass after the temporal pass and the filter, or right after the shade pass when those are off. It runs only while the lights are on.
- **Emission:**
  - It adds the table's per-material emitted radiance to each surface pixel's shown radiance.
  - This is ADR-0005 Amendment 3's rule that emission is added after reconstruction: it is never accumulated, demodulated or filtered.
  - The w channel (age and reason, for the debug views) is kept.
- **The meter** is in the same pass. For each 16×16 workgroup it sums:
  - Y over all pixels;
  - ln Y, with a count, over the pixels where Y > 0 and Y is at least the floor.
  - The floor is 2⁻¹⁰ × the mean from the last reading, the `light::exposure::metric_exposure` rule.
  - The partial sums go to a host-visible buffer per frame slot. The viewer adds them in f64 once that frame has completed (two frames later), at the point where it reads its timestamps.
- **Memory:** 16 B per workgroup and slot, which is 131 KB × 2 at 1080p (`GpuTemporal`).

### 3. The lights

- **`light::emitters::Lights`** (pure):
  - the lights follow `lights_on` (the sun below the horizon) until L switches them;
  - a hand setting holds until the sun next crosses the horizon, and then the rule takes over again;
  - `--lights auto|on|off` sets the start state (default auto).
- **Only a scene with an emitter table has lights.** On `--scene street` they stay off and L does nothing, so the M3 street is unchanged at every hour.
- **A change of the lights is a light jump** (decision 3):
  - `History` remembers whether the lights were on, and a change resets every pixel (reason reset), like the sun's 1° jump;
  - ADR-0006 Amendment 2 adds it to the global resets.

### 4. The exposure while the lights are on

- **Target:** 0.18 over the metered log-average, which is the metric exposure's definition.
- **Adaptation:** in log₂ exposure, with a time constant of 1 s: EV ← EV + (EV_target − EV) · (1 − e^(−dt / 1 s)).
  - It is bounded to [2⁻², 2¹⁶]. For scale, 4A G8's metric exposures were about 7,400–12,100 at blue hour and night.
  - − and = still offset it by quarter stops.
  - A reading with no light keeps the current exposure.
- **When the lights turn on,** it starts from the exposure shown the frame before. At start-up it starts from the first reading, two frames in.
- **While the lights are off,** the light view keeps M3's exposure (from sun and sky irradiance). Every lights-off frame is therefore displayed exactly as in M3.
- **Pure functions:** `light::exposure::adapt`, and the target from the sums.

### 5. The viewer

- **New flags and key:** `--lights auto|on|off`, key L, `--emitter-spp N`.
- **The title** shows the lights (on or off; auto or by hand) and the exposure (automatic or M3's, with its value).
- **The JSON** adds:
  - `lights`: the state at exit and the number of switches;
  - `exposure`: the mode, the final value and the final target;
  - `emitter_spp` and `stale_table_frames`.
- **Binding:** the shade and compose sets bind the scene's current table. They are rebuilt after every edit together with the other sets.
- **Fallback:** a table the scene refuses as stale is not bound.
  - That frame has no emitter light and no emission, and the title says so.
  - `stale_table_frames` counts such frames; it must stay 0.

### 6. Cameras, times, and what the images are compared with

- **Cameras:** street and low (3A, 3G, 4A).
- **Scene:** `street_night` (Full) unless a dressing is named.
- **Times:** blue hour 18.5 h (−5.3°) and night 21 h (−30°), with the lights on. Dusk (17.75 h, +2.65°, lights off) is covered by G12.
- **References:**
  - `gpu::reference` with the real-time path's transport: one bounce, sun, sky, and emitters direct and indirect.
  - **Emission is off in them:** the shade radiance is reflected light, and G13 checks emission on its own.
  - 1920×1080, 8,192 spp, a fixed seed, cached in `NE_4B_DIR`.
- **What Q reads:** the reflected light (the radiance before compose). Display images add the table's emission on the host; G13 shows that this equals compose's.
- **Blue hour, per-pixel metrics:**
  - These compare against the real-time estimator converged, not the reference. The converged image is 8,192 frames of a still camera through the temporal pass, with `max_age` 8,192 and s = 4.
  - Reason: 4A's blue-hour references converge slowly (a mean pixel relative SE of 0.14–0.16 at 16,384 spp, consistent with a heavy-tailed Monte Carlo sky). Their noise would dominate a per-pixel error.
  - Energy still uses the reference at both times. At night the sky is black and the reference is clean.
- **Heavy tails at night:**
  - 4A M2 measured the one-sample σ/μ at the 90th percentile at 13–21.
  - So energy and error are taken as the mean over **8 independent seed sequences**, with the spread reported.
  - FLIP is bounded per pixel and uses one seed, as in 3G.

## Design, part 2: relight for lights

- **ADR-0006 Amendment 2** adds a third relight rule and reason 10, **relit by a light**, so its decisions can be told apart from 3E's (reason 9).
- **Light rows.** On the frame after an edit, the host lists up to 64 rows:
  - **Changed emitters** (at most 16):
    - the emitters of the rebuilt regions whose geometry or radiance is in only one of the old and new tables;
    - a quad split differently counts as changed, which only relights more;
    - more than 16 are merged into the 16th as one conservative row: a bounding sphere, the summed Y·A, lit from both sides.
  - **Shadow rows**, per edit box:
    - the 4 unchanged emitters with the largest unoccluded irradiance bound at the box's centre, since they can cast new shadows through the box or lose old ones;
    - any beyond these are left out, as part of the approximation.
- **The test per pixel.** It applies outside rebuilt regions, to pixels whose history is otherwise accepted, after the 3E rules. A pixel is relit by a light when, for some row, both of these hold:
  - **The light matters at the pixel:**
    - the pixel is in front of the emitter;
    - the emitter's unoccluded irradiance bound there, divided by π, is at least **2%** of the luminance of the pixel's history;
    - the bound is Y(L_e) · min(2π, A / d²), with d the distance to the emitter's bounding sphere;
    - the albedo counts as 1 on the light's side, so the test can only relight more.
  - **The edit can change that light there:**
    - the emitter changed; or
    - the segment from the pixel to the emitter's centre crosses the edit box, grown by 1 voxel plus the emitter's half-diagonal (the grown box contains every segment from the pixel to the emitter's area).
- **The declared approximation** is the 2% threshold, the 4 shadow rows per box and the row cap. R4 measures it, like 3E's R2.
- **`temporal.slang` gets the same split.**
  - Without `EMITTERS` it stays main's module (C10), so relighting with the lights off is exactly 3E's.
  - The rows reach the shader in the 3E relight buffer, written in the command stream: about 3 KB more.

## Criteria (frozen 2026-09-26, before the first run)

**Pure and build (cloud and laptop):**

| # | Criterion |
|---|---|
| C8 | **`Lights` and the exposure** (`light` unit tests). **The rule** at every 3A and M4 time: off with the sun up and at dusk; on at twilight, blue hour and night. **L** inverts the state and holds it until the next horizon crossing, in either direction; then the rule resumes. The `--lights on` and `off` start states hold the same way. **`adapt`:** after dt = τ, the log exposure has moved 1 − 1/e of the way; two steps of dt / 2 equal one step of dt; the bounds hold; no reading leaves it unchanged. **The target from the sums** equals `metric_exposure` of the same image when the floor is the image's own. All within 10⁻¹². |
| C9 | The lit shade module and the compose module match their host layouts (`gpu --lib`, device-free), and a moved field is refused. |
| C10 | Without `EMITTERS`, `shade.spv` (part 1) and `temporal.spv` (part 2) are byte-identical to main's, and `reference.spv` is unchanged. Checked in the cloud with the pinned slangc's Linux release (supplemental). |

**GPU, part 1 (laptop, `run-local.cmd`):**

| # | Criterion |
|---|---|
| G10 | **Exact per pixel (s = 1).** Frame f of the lit shade pass equals the reference's one-bounce sample f (emitters direct and indirect, emission off). **Pass:** per pixel within 10⁻³ relative on ≥ 99.9% of pixels; 0 bad ids; ≥ 1,000 pixels per arm changed by the emitter light. **Arms** (`street_night` Full, 480×270): the three B1 cameras (street, low, overhead) × five settings: emitters only (sun and sky off); emitters with the uniform sky; with the point sun at 12 h; with the disk sun at 12 h; emitters only with the bounce off (reference `max_bounces` 0). **Tolerance:** 10⁻³, not B1's 10⁻⁵. The emitter point and its weight depend smoothly on the surface point, which the shade pass rebuilds from depth (about 10⁻⁴ voxel from the ray hit); B1's terms depended on it only through visibility. **Controls**, each failing at least half of the arms with ≥ 10× the correct arms' mismatches: emitters off; `emitter_after_continuation` (the emitter numbers drawn after the continuation's); `no_bounce_emitters` (none at the bounce hit); `emission_in_shade` (emission added before reconstruction). Frame f compared with the reference's frame f − 1 must differ. |
| G11 | **Convergence (s = 1, 2, 4).** The raw accumulation (4,096 frames, summed per pixel on the host) against the reference (4,096 spp), at 240×135, 21 h, street and low cameras. **Pass (4A G4's metric):** image mean \|z\| < 4 per channel; pixel-channels with \|z\| > 4 ≤ max(1%, 1.5 × null + 0.2%), the null coming from two independent reference renders. **Control:** `emitter_sum` (the s samples added, not averaged) fails at s = 2 and 4. |
| G12 | **Lights off is M3.** These GPU files reproduce their recorded numbers: `shade`; `sky`; `bounce` (B1: 39 mismatched pixels in total, at most 7 in one arm); `temporal`; `denoise` (D1–D3, R1, R2 and S1 unchanged; D4 and C1 fail as accepted, S-018 and S-019, D4 with its recorded numbers and C1's time as data); `emitters` (4A G1–G6: G2 exact, G3 and G4 with the numbers of the 4A part 2 log); `edit`. |
| G13 | **Emission and the light jump.** Night, lights on, 480×270, both cameras. **Emission:** after compose, shown minus reflected equals the table's emission for the pixel's material on every surface pixel (10⁻⁶ relative); it is 0 on non-emitting materials; sky pixels and w are unchanged. **The history holds no emission:** after 8 frames of a still camera, its colour is bit-identical to that of the same frames without compose; the planted `compose_before_temporal` fails this. **Light jump:** a lights change between two frames, in either direction, resets every surface pixel (reason reset); unchanged lights on a still camera leave every surface pixel accepted. |
| G14 | **The meter.** On the shown image at 1920×1080, and at 1001×563 (partial workgroups), the meter's sums, added in f64, equal the host's sums over the read-back image: ΣY and Σ ln Y within 10⁻⁴ relative, the count within 0.01% of the pixels. The planted `meter_without_emission` fails. |
| Q1 | **Energy.** **Setup:** 1920×1080, validation on, lights on, s = 1, the viewer's defaults; stills (street and low × blue hour and night) from a fresh history, so frame = age; reflected light over surface pixels, with emission excluded as 4A recommended; mean over the 8 seed sequences. **Pass:** at ages 1, 4, 16 and 64, the filtered mean luminance is within ±2% of the reference, and within 1% of the raw accumulation in the same frame. |
| Q2 | **Error.** The relative MSE (the 3E metric, over surface pixels, reflected light), averaged over the 8 seeds. **Pass:** filtered ≤ raw at gain × age, with gains 8 / 4 / 4 / 1 at ages 1 / 4 / 16 / 64 (3G Q2), in all 4 still arms. Compared with the reference at night, and with the converged real-time image at blue hour. |
| Q3 | **Motion energy (night).** The 3E D4 path (street camera, 0.25 voxel sideways and 0.2° of yaw per frame), frames 8, 16 and 32, mean over the 8 seeds. **Pass:** filtered within 1% of raw in the same frame, and within 2% of the converged real-time image at that pose. |
| Q4 | **FLIP** (LDR-FLIP 1.7, 67 pixels per degree, seed 0). **Pass:** filtered ≤ raw at ages 1, 16 and 64 (the 4 still arms) and in motion frames 8, 16 and 32. **Display images:** reflected light plus emission, at the metric exposure of the image compared against (the reference or the converged image, plus emission), then the ACES fit and sRGB. |
| V | **Viewer runs with validation on.** `--scene night` at 21 h (600 frames). `--scene night` from 17.9 h with `--run-day` through sunset (900 frames): exactly 1 lights switch, then the automatic exposure. `--lights off` at 21 h and `--lights on` at 12 h (300 frames each). `--emitter-spp 2` and `4` at 21 h (300 each). `--edit-script` at 21 h (400 frames). `--scene street` at 21 h (300 frames: no lights). **Pass:** every run exits 0 with 0 errors, 0 warnings, no leaks, and `stale_table_frames` 0. |

**Part 2 (laptop, a second run):**

| # | Criterion |
|---|---|
| R3 | **Relight decisions.** Lights on, 21 h, 320×180, the 3E rig with the bounce. **Two edits:** (a) one whole lamp head removed; (b) a 3×3×3 box placed in open air between a lamp head and the road. **Pass:** after each edit, the pixels relit by a light (reason 10) equal a host re-derivation of the rule on every pixel whose history is otherwise accepted: ≤ 0.05% of pixels differ, and ≥ 100 pixels are relit by a light. **Control:** the same frame without the light rows differs from the host rule on ≥ 100 pixels. |
| R4 | **Relight lag.** **Pixels:** those whose converged value changed by > 25% because of the edit, outside the rebuilt regions (converged images of 4,096 frames before and after). **Pass:** 4 frames after the edit, their filtered relative MSE is ≤ 3 × that of an arm fully reset at the edit, for both edits. **Control:** the arm without the light rows must fail this. |
| R5 | **Lights off unchanged.** 3E R1 and R2 reproduce their recorded numbers, without the bounce (3E: 920 relit, 172 changed px, 0.0319) and with it (3G E2: 920, 171, 0.0300). `raster` (the pixel-exact views), `temporal` and `denoise` pass as in G12. |
| E4 | **Edit budget with the lights on.** **Setup:** the viewer, release, validation off, 1920×1080, 21 h, `--scene night` (Full), the scripted fly path, as 4A G7; N = 1, 8, 32, 2,000 frames each. **Pass:** p95 edit-to-visible ≤ 50 ms (N = 1, 8) and ≤ 100 ms (N = 32); every edit shown; 0 deferred. The light rows' host time per edit is reported. **With validation on** (400 frames, N = 1, 8, 32): exit 0, 0 errors, 0 warnings. |

**Measurements (a fixed method, not pass/fail):**

| # | Measurement |
|---|---|
| M1 | **The equal-time curve.** **Arms:** night; s ∈ {1, 2, 4} × the four dressings (165 / 325 / 1,143 / 7,003 emitter quads) × the street and low cameras. **Error:** the one-frame relative MSE of the raw shade output against the dressing's reference (480×270, one bounce, 16,384 spp): the mean over 64 frames with its standard error, and the share of it that comes from the worst 1% of pixels. **Time:** the shade pass at 1920×1080, validation off. For each camera, all 12 arms run in one command buffer in alternating order (the four scenes resident), 40 repetitions, median and p90 (3F's cost method). **Reported per arm:** ms, error, and error × ms. **Default s:** the smallest s whose error × ms is within 10% of the lowest on Full (mean of both cameras). The viewer's default becomes that s, and Q must then pass at that s as well (M2 has the runs). |
| M2 | Q1–Q4's numbers again at s = 2 and 4; the compose pass's GPU time at 1080p; the night filtered error beside 3G's day and twilight values. |
| M3 | **Inputs to 4D.** (a) **The visible gap of bounces ≥ 2:** FLIP between new one-bounce references (960×540, 16,384 spp, 4A's seed 74, so the samples pair with 4A's) and 4A's eight-bounce references, at blue hour and night, both cameras, emission added, at the eight-bounce image's metric exposure. (b) **Indirect noise:** at night, the filtered relative MSE at ages 16 and 64 with the bounce off (against its own converged image), beside Q2's with the bounce on. |
| M4 | **Frame cost in the viewer.** Validation off, 1920×1080, `--walk`, 3,000 frames, one run each. **Arms:** `--scene night` at 17.75 h (lights off), 18.5 h and 21 h; `--scene street` at 21 h as the control. **Reported:** frame p50 and p99, and the shade slot's GPU time. Not a gate: that is 4G. |

**Your look** closes 4B together with the numbers. Walk from dusk into night (`viewer.exe --scene night --hour 17.75 --run-day`) and say whether the lights, their colours and the exposure are right for now. If you change light values or colours (4A's table), the night references are rendered again and Q is run again under the same criteria.

## If a criterion fails

- **Method,** as in 3E and 3G:
  - one discriminating diagnostic at a time;
  - after two inconclusive hypotheses, stop and report the options;
  - no criterion is restated or rebaselined.
- **Most at risk: Q1 at ages 1–4 at night.**
  - Filter weights that depend on the samples they average lose energy on heavy-tailed input. That was 3E D2, fixed there by the prefilter.
  - The night tail is 5–8× dusk's (4A M2).
  - A change to the filter would change M3's image too, so it needs your approval.

## Choices in this plan (yours to override)

1. **The exposure follows the lights:** metered while they are on, and M3's sun-and-sky exposure while they are off, so daytime stays exactly as in M3. The alternative is metering at every hour, which changes how bright the daytime image looks.
2. **L** switches the lights by hand until the sun next crosses the horizon. `--lights` sets the start state.
3. **The default samples per pixel** come from M1's rule. The result may be 2 or 4, which spends frame time that 4C and 4D would otherwise have.
4. **Quality is measured at 1080p with 8 seeds.** This makes a part 1 laptop run about 1.5 h.
5. **Blue-hour per-pixel comparisons use the converged real-time image,** not the reference, because of the reference's sky tail.

## Scope

- **Part 1, allowed:**
  - shaders: `engine/gpu/shaders/shade.slang` (the `EMITTERS` block), `emitters_common.slang` (helpers), and a new `compose.slang`;
  - `engine/gpu/build.rs`;
  - `engine/gpu/src/`: `shade.rs`, a new `compose.rs`, `temporal.rs` (the lights in the reset decision) and `lib.rs`;
  - `engine/light/src/`: `emitters.rs` (`Lights`) and `exposure.rs`;
  - `engine/viewer/src/main.rs`;
  - tests: a new `engine/gpu/tests/night.rs`, and companion edits wherever tests build `ShadeSettings` or call `Temporal::record`;
  - a new `engine/results/phase4b/flip.py`, and `run-local.cmd`;
  - docs: this record, ADR-0005 Amendment 3's status, ADR-0006 Amendment 2, `engine/README.md` and `docs/NOW.md`.
- **Part 2, allowed:**
  - `temporal.slang` (the `EMITTERS` block) and `temporal.rs` (the rows);
  - `gpu::scene` and `gpu::emitters` (the changed emitters of an update);
  - `debug_view.slang` (a colour for reason 10);
  - the viewer's edit path;
  - `night.rs`;
  - ADR-0006 Amendment 2.
- **Excluded:**
  - reservoirs (4C) and path reuse (4D);
  - `street_block` and `reference.slang`;
  - the night light values, unless you change them at your look;
  - the filter's code and settings, and every M3 default;
  - new dependencies.
- **ADR numbering:** the proposal gave ADR-0006 Amendment 2 to 4C's reservoir history. With 4B taking it, 4C's becomes Amendment 3.

## Work units and run plan

- **Units:** 3 in all, taking M4 from 6 to 9 of 33.
  - Part 1, 2 units: one for the lit estimator, lights, emission and exposure (C8–C10, G10–G14, V); one for quality and the curve (Q1–Q4, M1–M4).
  - Part 2, 1 unit.
- **Part 1 laptop run:** one `run-local.cmd`, about 1.5 h.
  - The references and converged images take about 25 min. They are cached in `%TEMP%\ne_4b` so a rerun skips them.
  - The M3 regression files take about 20 min.
  - The rest is tests, viewer runs and FLIP.
  - Logs go to `engine/results/local-run/<date>-4b1/`, and images and FLIP results to `engine/results/phase4b/`.
  - FLIP needs the user's Python with `flip-evaluator` 1.7 (A-008). Without it, that step is NOT RUN.
- **Part 2:** a second run, about 30 min.

## Commands and evidence actually produced

| Check | Result | Scope |
|---|---|---|
| slangc byte identity (see *Checked before planning*) | holds: with the `EMITTERS` block compiled out, the shade module and its reflection are byte-identical to main's; `reference.spv` is unchanged by a new shared helper | cloud, the pinned slangc's Linux release, a stub block; supplemental |
| Everything else in the plan phase | NOT RUN: no 4B code existed yet | — |

## Part 1: what was built (2026-09-26, cloud session)

- **`gpu::shade`:** `shade.slang` is compiled twice (`shade` and `shade_lit`, `build.rs`). The lit module's `EMITTERS` block draws `emitter_spp` samples at the primary vertex and at the bounce hit, as designed. `ShadeSettings` gains `emitters` and `emitter_spp`. `ShadeFaults` gains `emitter_after_continuation`, `no_bounce_emitters`, `emission_in_shade` and `emitter_sum`. `Shade::bind_lit` binds the table, and `Shade::record` picks the module from the set.
- **`gpu::compose`** (new) and `compose.slang`:
  - emission after reconstruction, and the meter (per 16×16 workgroup, into a host-visible buffer per slot, `ComposeTargets::reading`);
  - a compute-to-host barrier after the dispatch;
  - the planted `meter_without_emission`.
- **`gpu::temporal`:**
  - `History::set_lights`: the caller says whether the lights are on for the next frame, and a change is `ResetCause::lights`, a full reset. The signature of `Temporal::record` is unchanged, so the M3 callers are untouched.
  - `History::read_colour` (for the tests).
- **`light`:**
  - `emitters::{Lights, LightsMode}`;
  - `exposure::{Meter, adapt, ADAPT_SECONDS, MIN_EXPOSURE, MAX_EXPOSURE}`.
- **Viewer:**
  - `--lights`, L, `--emitter-spp`;
  - the lit set and compose rebound at start, resize and after every edit;
  - the automatic exposure (`Exposure`), and the meter's readings when a slot's frame has completed;
  - the title and the JSON fields;
  - the start state is not counted as a switch.
- **Tests:** `gpu/tests/night.rs` (G10, G11, G13, G14, Q1–Q4 with M2, M3 (b), M1). `ShadeSettings` literals in the M3 test files gain `emitters: false, emitter_spp: 1` (no behaviour change).
- **Tools and the local run:**
  - `results/phase4b/flip.py` (Q4, and M3 (a) from `ref_light` PFMs);
  - `run-local.cmd` rewritten for part 1 (33 steps).
- **How M3 (a) runs:** `ref_light --scene night --bounces 1 --spp 16384 --times blue_hour,night` writes the one-bounce references with 4A's seed and cameras, and `flip.py` shows both at the eight-bounce image's metric exposure. Its PFM reader and exposure reproduce `ref_light`'s own exposures for two 4A images (11,549 and 7,735).
- **Implementation choices within the design** (no criterion changed):
  - M1's references and arms use the full night transport (sun, sky, emitters; the sun and sky give nothing at 21 h);
  - converged images use seed `0x4BCC`, frames 1–8,192, 16 per submission;
  - the M1 and M2 cost arms use frame index = repetition.

## Part 1: evidence

Cloud session: Linux x86_64, 4 cores, no GPU. The `gpu` crate was built with the pinned slangc 2026.13.1 (its Linux release, downloaded to the session's scratch space) and the distribution's `spirv-val`. This is a supplemental compile check, not the laptop's SDK.

| Check | Result | Scope |
|---|---|---|
| C8 (`light`: `emitters::tests::lights_switch_by_hand_until_the_horizon`, `exposure::tests::automatic_exposure_adapts_and_matches_the_metric`) | pass | cloud |
| C9 (`gpu --lib`: `shade::tests::host_layouts_match_the_compiled_modules`, `compose::tests::host_layout_matches_the_compiled_module`) | pass, including the moved-field controls and the M3 module refusing the lit layout | cloud, no device |
| C10 | pass: all 12 of main's modules (SPIR-V and reflection, 24 files) are byte-identical to this build's, including `shade.spv`, `temporal.spv` and `reference.spv`; the new modules are `shade_lit.spv` and `compose.spv` | cloud, supplemental |
| Pure suite (`cargo test --release -j 2`) | exit 0, **155 pass** (153 + C8's 2), 4 ignored; `engine/results/test_pure_4b1_cloud.log` | cloud |
| `gpu --lib` | exit 0, 17 pass (15 + C9's 2); `engine/results/test_gpu_lib_4b1_cloud.log` | cloud, no device |
| Clippy (`--workspace --all-targets`) | exit 0, only the 6 old `world` lints (4 new ones in 4B code were fixed first); `engine/results/clippy_4b1_cloud.log` | cloud |
| Builds: `viewer`, `gpu` bins and every `gpu` test binary (`night` included) | exit 0, no warnings | cloud, substitute SDK |
| G10–G14, G12's M3 files, Q1–Q4, M1–M4, V | **NOT RUN** (no GPU): `run-local.cmd` | laptop |
| `run-local.cmd` | untested (a Windows batch file cannot run here); re-read against CLOUD.md's rules | — |

## Next

1. **Part 1 on the laptop:** pull the branch, run `run-local.cmd` (about 1.5–2 h), and push `engine/results`. The G and Q criteria are judged from those logs, then your look.
2. **Part 2** comes after part 1's results and your look.
3. **Unchanged:** 4C–4G are not authorized. Before 4C, point me to the P05 and P08 reviews and PDFs, and provide the ReSTIR sources.
