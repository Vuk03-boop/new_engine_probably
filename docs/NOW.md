# NOW — Phase 4 (M4): 4A done; 4B run on the laptop, G4 and G5 fail, waiting on the user's decision

**Updated 2026-09-26 (4B laptop run analysed: 27 of 29 steps pass; G4 and G5 fail).** This file is the plan to start from. A new session reads `CLAUDE.md`, then this file, then only the files it links for the task at hand.

## State

- **Accepted milestones:**
  - M1 "walk the street in real time" (S-012);
  - M2 "edit the street and see it in real time" (S-015);
  - M3 "the street lit by sun and sky, dawn to dusk, with real shadows and one bounce, 1080p 60 fps, checked against a reference" (S-022).
- **Phases 1–3 done.** Records are in `docs/changes/`. Phase 3: the [proposal](changes/2026-09-24-phase3-proposal.md), the 3A–3F records, and the [3G gate](changes/2026-09-25-phase3g-gate.md). Accepted engine ADRs: 0001–0006 (0005: light conventions and the sky; 0006: guides and history).
- **Defaults:**
  - greedy / chunk meshes, with 2³ chunks for large sparse scenes (ADR-0003 Amendment 2);
  - MAILBOX (S-014);
  - edit budget p95 ≤ 50 ms for 1-voxel and 8³ edits, ≤ 100 ms for 32³ (S-016);
  - the light view: sun, the corrected sky (S-020), one bounce, the temporal pass and the 3E filter.
- **Accepted limitations:**
  - motion blur of hard shadow edges (S-018);
  - filter cost about 3.5 ms (S-019);
  - the Q2 shadow-edge softening on grazing surfaces (S-021).
- **M4 accepted as the target (S-024):** the [Phase 4 proposal](changes/2026-09-25-phase4-proposal.md), "the street from dusk into night, lit by its own lamps, signs, neon, windows and string lights, with multi-bounce light where it matters, 1080p 60 fps, checked against a reference". **4A and 4B are authorized; 4C–4G are not.**
- **4A done, record closed** ([record](changes/2026-09-25-phase4a-emitters.md)), written in a cloud session (no GPU; [CLOUD.md](CLOUD.md)) and checked on the RTX 3050 through `run-local.cmd`:
  - part 1 built: emitter units and sampling ([ADR-0005 Amendment 3](adr/ADR-0005-light-transport-conventions.md)), the emitter table ([ADR-0003 Amendment 3](adr/ADR-0003-device-budget-and-visibility-buffers.md)), emitters in the CPU and GPU references, `street_night`;
  - **part 1 passed:** C1–C5 (cloud), G1–G5 and the M3 GPU regressions (RTX 3050, 2026-09-25 19:12);
  - the user accepted the proposed night lights for now and started part 2 ("For now i accept it part 2 time");
  - **part 2 built** (criteria frozen first, commit `ae3ffb8`): the table in `GpuScene` (build, edits, swap, stale refusal), the viewer's `--scene` and `--dressing`, the lights rule, the metric exposure, `ref_light --scene night`, and the magnitudes;
  - **part 2 cloud checks pass:** C6, C7; the magnitudes M1–M2 are measured (below);
  - **part 2 passed on the RTX 3050:** G6–G9 (`run-local.cmd`, 2026-09-25 20:29; analysed in a cloud session from the pushed logs);
  - **L moved to 4B** with the rendering it switches (in 4A it would switch nothing on screen).
- **4B built, cloud checks pass, GPU checks NOT RUN** ([record](changes/2026-09-25-phase4b-many-lights.md)), started on the user's "go ahead then you are on high greenlight to do 4b" (2026-09-25), on branch `claude/hopeful-bell-972dbs`:
  - criteria frozen first (commit `1da5bc8`); three corrections before any run, recorded in the record (C2's oracle is the gross face change, since a lamp's corner voxel has a net change of 0; G1 is judged at 10⁻³ relative, since the emitter term is continuous in the depth-rebuilt surface point and the f32 solid angle carries about 10⁻⁶ sr of rounding; G5's converged value is the reference with a 4-SE test, and its E2 box sits between the lamp head and the visible façades);
  - built: emitters in `gpu::shade` at both vertices by the reference's own shared function (`shaders/emitter_nee.slang`), `emitter_samples`, `Shade::bind_lit`; emission added in the light view after reconstruction; the lights rule, L and `--lights`; the automatic exposure (`gpu::exposure`, `light::exposure::Adaptation`, `--exposure`); the emitter relight rules E1 / E2 and the lights switch as a light jump ([ADR-0006 Amendment 2](adr/ADR-0006-guides-and-history.md), proposed until G5 passes); `gpu/tests/lights.rs` (G1–G6, M1);
  - cloud: C1–C4 pass; pure suite 154 pass, `gpu --lib` 21 pass, clippy only the 6 old `world` lints;
  - `run-local.cmd` rewritten for 4B (untested; about 1.5–2.5 hours, re-estimated from 4A's laptop reference times; first written as 90 minutes).
- **Night history cap fixed (S-025, 2026-09-25, cloud)**, on the user's approval and their choice of −12° after the measurement: the sun-motion age cap is off while the sun is below −12° ([ADR-0006 Amendment 3](adr/ADR-0006-guides-and-history.md), `TemporalSettings::sun_cap_min_elevation_deg`). Measured first (`light::sky` `diagnostic_skylight_below_horizon`): the skylight radiance on an albedo-0.3 surface is 1.1 × 10⁻⁸ units at −12°, about 0.05% of the night street's mean reflected lamp light. Details in the [4B record](changes/2026-09-25-phase4b-many-lights.md). The work is on branch `claude/focused-faraday-ygdjx7`, which carries 4B's commits plus this fix.
- **G4's Q1 corrected before any GPU run** (same session; the user delegated the review: "check for any weird stuff and decide yourself"): one night frame's mean luminance has a 1-sigma of 7.5–8.3% at 1080p (CPU-measured, `light` `diagnostic_g4_noise_floor`), so Q1's frozen ±2% could not pass even with correct code. Q1 is now Q1a (the filter's energy change against the raw history of the same frames, ±2%) and Q1b (the raw history's bias within max(2%, 4σ/√age)); Q2 and Q4 are unchanged ("G4 correction" in the [4B record](changes/2026-09-25-phase4b-many-lights.md)). The review found nothing else to change: every test filter in `run-local.cmd` matches a test, every viewer flag exists, the viewer exits 1 on validation messages, the 1080p reference renders stay far below Windows' 2 s GPU timeout (8 samples per submission), 4A's viewer runs really were 1920×1080, and the exposure pass's 64-workgroup change is consistent with its host sums.
- **4B laptop run (RTX 3050, 2026-09-26 09:20, commit `cd28474`, `engine/results/local-run/2026-09-26_0920-4b/`; analysed in a cloud session; details in the record's "Laptop run"):** G1, G2, G3, G6, G7, G8 and G4's FLIP (Q4) pass; M1 and M2 recorded (night walk p50 7.9 / 8.8 ms, p99 12.0 / 11.7 ms for Full / Dense); 0 validation errors everywhere.
  - **G4 FAIL.** Q1a: **the filter removes 5–24% of the night image's energy** (filtered minus raw from the same frames). Q1b passes, so the history is unbiased. Q2 fails at low_night age 1 only. Hypothesis: the SVGF-style weights are asymmetric, so rare bright samples are averaged down and never spread. A filter change is outside 4B.
  - **G5 FAIL, on the negative control only.** R1 passes, and so does R2 wherever it is judged. The 3E-only arm's error is 1.7× the reset arm's (limit 3×), and only 1 of 3 edits changed enough pixels to be judged, so the test lacks power. The engine did nothing wrong.
  - **G4 propagates into measurement, not into the history:** the history stays unbiased, but 4C / 4D comparisons made through the filter would be confounded (the record's "Does G4 propagate?"). So G4 is understood before 4C.
  - S-025 was not exercised by the run (the sunset run ended at 18.24 h); the user's viewer check is still open.
- **Not authorized:**
  - 4C–4G: reservoir reuse, P05, P08, the M4 gate;
  - glass and water;
  - roadmap phases 5–8, which hold the rest of the 21 proposal techniques (none excluded, each behind its gate);
  - off-thread jobs, a no-wait acceleration update, BLAS refit or compaction;
  - exclusive fullscreen.

## What exists

- **Docs:**
  - the proposal (original, kept as written) and `CLAUDE.md`;
  - [DECISIONS.md](DECISIONS.md): ADR-0001 to ADR-0006 (ADR-0003 and ADR-0005 each gained Amendment 3 in 4A), A-006, A-008, S-001 to S-024;
  - [CLOUD.md](CLOUD.md): rules for cloud sessions only (read when `CLAUDE_CODE_REMOTE=true`), and the one-click `run-local.cmd` handoff;
  - the change records in `docs/changes/`;
  - [BUILD-ROADMAP](../BUILD-ROADMAP.md) and [TECHNIQUE-MAP](../TECHNIQUE-MAP.md).
- **`engine/`** ([README](../engine/README.md): crates, verified commands, notes):
  - pure crates `memory`, `world`, `derived`, `walk`, `light` (154 tests, 6 ignored diagnostics);
  - 4A: `light::emitters` (table, alias selection, solid-angle sampling, `lights_on`), `light::exposure::metric_exposure`, the reference's emitter terms (off by default), `world::scene::street_night(Dressing)`, `gpu::emitters` (`EmitterSet`, `SceneEmitters`, `RefEmitters`) and `Reference::bind_lit`, `GpuScene::build_lit` and `GpuScene::emitters`, `gpu/tests/emitters.rs`;
  - `gpu`, on Vulkan:
    - raster, ray query and scene updates;
    - the reference path tracer;
    - real-time shade with sun, sky and bounce, the temporal pass and the filter;
    - the tools `ref_light` (4A: `--scene night`) and `sky_bake`.
  - 4B: `shade` emitter terms (`ShadeSettings::emitters`, `emitter_samples`, `Shade::bind_lit`), `debug_view::Lighting::emission`, `gpu::exposure`, `temporal::{changed_power, Relight::with_lights, History::set_lights}`, `light::exposure::{log_sums, exposure_from_sums, Adaptation}`, `gpu/tests/lights.rs`;
  - `viewer`:
    - walking and flying; edits (left click, E); R switches raster / ray query;
    - views: 0 the light view, 1–7 debug views, 9 the M1 lit view;
    - lighting keys: [ ] and T move the sun; H accumulation, N the filter, B the bounce;
    - flags: `--no-bounce`, `--no-sky-correction`, `--edit-size N`;
    - the Q2 comparison: P, `--prefilter-age N`, `--camera low`;
    - fps in the title and on exit;
    - 4A: `--scene street|night` and `--dressing` (the night scene keeps its emitter table through edits);
    - 4B: the night scene's lights (on below the horizon, L flips them, `--lights`), `--emitter-samples K`, the automatic exposure (`--exposure auto|sky`, auto at night).
- **Emission:** has a unit (1 unit of luminance = 128,000 cd/m², ADR-0005 Amendment 3) and is in the references, off by default. The lights are on while the sun is below the horizon (`lights_on`). 4B puts it in the real-time path (built, not yet run on the GPU). `street_block` is unchanged; `street_night` carries the night lights (in the 4A record; accepted for now).
- **Repository** (since 2026-09-25):
  - Git, branch `main`, pushed to https://github.com/Vuk03-boop/new_engine_probably (public; first commit `eac532c`).
  - **4A is merged into `main`** (PR https://github.com/Vuk03-boop/new_engine_probably/pull/1, merge commit `16edb67`, 2026-09-25, at the user's request; a merge commit, so the hashes cited in the 4A record stay valid). New work starts from `main`; the 4A branches `claude/tender-keller-9u9d6t` and `claude/nice-dijkstra-xpg7tr` are finished.
  - `run-local.cmd` (repository root, CRLF by `.gitattributes`): the one-click local run, now holding the **4B** checks (G1–G8, M1, M2; untested, about 1.5–2.5 hours). Logs go to `engine/results/local-run/<date>-4b/`, FLIP to `engine/results/phase4b/`; G4's 1080p references stay in `%TEMP%\ne_gate_4b` on the laptop; the user pushes `engine/results` back.
  - 4B's work is on branch `claude/hopeful-bell-972dbs` (pushed; no PR yet); `claude/focused-faraday-ygdjx7` = that branch plus the S-025 cap fix (pushed; no PR). **Pull `claude/focused-faraday-ygdjx7` for the laptop run.**
  - Not committed, kept locally (`.gitignore`): build output (`target/`), the Phase 0 third-party assets and tools (Bistro, RenderDoc), and GPU captures (`*.rdc`, `*.ngfx-gputrace`).
  - Binary data is protected from line-ending conversion by `.gitattributes`.
  - Commits and pushes happen only on the user's request.
- **Evidence:**
  - `engine/results/`: logs per slice;
  - `engine/results/phase3g/`: FLIP, perf series and edits;
  - `engine/results/phase3a_ref/`: reference images;
  - `engine/results/phase4a_ref/`: the night references (2 cameras × dusk, blue hour, night; 16,384 spp);
  - `engine/results/local-run/`: the laptop runs of `run-local.cmd`.

## Last checks

- **Night cap fix (2026-09-25, cloud, Linux, no GPU):** the new pure test passes (with a planted −90° cutoff caught); pure suite exit 0, 154 pass, 6 ignored (`test_pure_4b_cap_cloud.log`, rerun after the G4 correction); `gpu --lib` exit 0, 22 pass (`test_gpu_lib_4b_cap_cloud.log`); clippy exit 0, only the 6 old `world` lints (`clippy_4b_cap_cloud.log`; substitute SDK). **NOT RUN:** the viewer at night with the day running. G4's noise floor: street 7.47%, low 8.28% per frame at 1080p (`diag_g4_noise_floor_cloud.log`); `lights` builds with the corrected Q1 (substitute SDK), not run (no GPU).
- **4B (2026-09-25, cloud, Linux, no GPU; [record](changes/2026-09-25-phase4b-many-lights.md)):**
  - pure suite: exit 0, 154 pass, 4 ignored (`test_pure_4b_cloud.log`); `gpu --lib` 21 pass (`test_gpu_lib_4b_cloud.log`); clippy: exit 0, only the 6 old `world` lints (`clippy_4b_cloud.log`); every `gpu` test binary, both tools and `viewer` build (substitute SDK: supplemental, not ADR-0001 proof);
  - C1 (layouts), C2 (relight rows, `changed_power` bounds the gross change within 20×), C3 (adaptation, sums) pass;
  - **NOT RUN:** G1–G8, M1, M2 (no GPU): `run-local.cmd`.

- **4A part 2 on the RTX 3050** (`run-local.cmd` at `7810f7d`, `engine/results/local-run/2026-09-25_2029-4a2/`, 15 of 15 steps exit 0; analysed in a cloud session, no GPU):
  - G6 pass: the table follows 5 edits, a refusal and a retry through `GpuScene::update`, device bytes equal every time; retirement and ledger asserts hold; the stale table is refused; 0 validation errors, no leaks;
  - G7 pass: night Full p95 27.9 / 32.5 / 34.2 ms (N = 1, 8, 32), Dense N = 1 33.3 ms; every edit shown, 0 deferred; table median 0.63 ms (Full), 4.40 ms (Dense); validation on: exit 0, 0 errors, 0 warnings;
  - G8 pass: 6 night references, 0 bad samples, 0 validation errors; the blue-hour ones keep a mean pixel relative SE of 0.14–0.16 at 16,384 spp (not a criterion; a heavy tail, see the record's Observations);
  - G9 pass: pure 153 (4 ignored), `gpu --lib` 15, `emitters` 5 (G1–G5 as before), `edit` 4, `temporal` 7; the M3 street viewer run p95 29.4 ms with no table in its JSON;
  - **NOT RUN on purpose:** `gate`, `denoise`, `reference`, `shade`, `sky`, `bounce` (unchanged code since part 1's run).
- **4A part 2 (2026-09-25, cloud, Linux, no GPU; [record](changes/2026-09-25-phase4a-emitters.md)):**
  - pure suite: exit 0, 153 pass, 4 ignored (`test_pure_4a2_cloud.log`); `gpu --lib` 15 pass (`test_gpu_lib_4a2_cloud.log`); clippy: exit 0, only the 6 old `world` lints (`clippy_4a2_cloud.log`); every `gpu` test binary, `viewer` and `ref_light` build;
  - C6 pass (the incremental table equals a from-scratch build through 5 edits; a missed region is caught); C7 pass (metric exposure, lights rule); the night street's walking start is free;
  - magnitudes (`magnitudes_4a_cloud.log`, CPU reference, 22 min): bounces ≥ 2 carry 7–10% of the reflected light at night, 3–7% at blue hour, 0.2–0.5% at dusk; one sample of the one-bounce estimator at night has σ/μ median 3.3–4.7 (dusk 1.7–1.9) and 90th percentile 13–21 (dusk 2.3–2.6); emitter light alone: median 2.4 → 2.8 and 90th percentile 3.7–4.1 → 8.5–8.6 from 165 to 7,003 emitter quads; directly seen emission is 86–90% of the night image mean;
- **4A part 1 (2026-09-25, cloud, Linux, no GPU; [record](changes/2026-09-25-phase4a-emitters.md)):**
  - pure suite: exit 0, 150 pass, 3 ignored (`test_pure_4a_cloud.log`); clippy with `gpu`: exit 0, only the 6 old `world` lints;
  - C1–C5 pass (alias χ², the rectangle's closed form, the emissive furnace, table identity, emitters-off images bit-identical); the planted faults are caught;
  - `gpu --lib` 14 pass, including the reference's reflection check (built with the pinned slangc's Linux release and a non-SDK `spirv-val`: supplemental, not ADR-0001 proof);
- **4A part 1 on the RTX 3050** (`run-local.cmd` at `07272fc`, `engine/results/local-run/2026-09-25_1912-4a/`, 8 of 8 steps exit 0):
  - pure suite 150 pass; `gpu --lib` 14; `emitters` G1–G5 pass (G2 exact, G3 0.44 SE, G4 within its limits on both cameras, G5 median 2.88 ms);
  - M3 regressions with emitters off: `reference` (all 39 result lines equal the 3A log), `shade`, `sky`, `temporal`, `bounce` pass; 0 validation errors;
  - G4's colour pattern (−3% / −4.6% in G / B on the street view, z ≈ −2) was CPU noise: `diagnostic_g4_cpu_convergence` (cloud) moves the CPU by the same amounts at 4096 samples (`results/diag_g4_cpu_convergence_cloud.log`);
  - **NOT RUN on purpose:** `gate` (35+ min, unchanged frame path) and `denoise` (accepted failures, does not bind the reference). The earlier `2026-09-25_1902-4a` start was stopped with Ctrl+C during the build (not a result).
- **3G gate** (details in the [3G record](changes/2026-09-25-phase3g-gate.md)):
  - P pass: p99 9.95–11.18 ms at dawn, midday and dusk against 16.67 ms, with the GPU thermal-limited;
  - Q1, Q3 and Q4 pass; Q2 fails in 1 of 10 arms, accepted (S-021);
  - E1–E3 pass, p95 27.7–31.1 ms.
- **Regression after 3G:**
  - pure suite: 137 pass (`test_pure_3g.log`);
  - clippy: exit 0, only the 6 old `world` lints (`clippy_3g.log`);
  - `-p gpu`: lib 12, bounce 4 and denoise 4 pass, plus C1 and D4 failing as accepted (`test_gpu_3g_regress_failfast.log`, exit 101; cargo stopped at `denoise`).
- **NOT RUN:**
  - after 3G, the `-p gpu` files `device`, `edit`, `raster`, `ray`, `reference`, `shade`, `sky`, `temporal`. None uses the filter, the only library code 3G changed; the rerun was stopped at the user's request.
  - another GPU;
  - the debug build since 2E;
  - a validation-off viewer comparison with and without the sky correction.
- **Doc review (2026-09-25):** stale status lines were corrected in `CLAUDE.md`, `README.md` and `DECISIONS.md`; the clippy count (first written as 8) was corrected in the 3G record; the proposal's §5 frame estimate was reworded to "may not fit". No engine code changed.
- **No live processes** started by me (cloud session: none left running).

## Milestone progress

- M1, M2 and M3: 100%, accepted (M2 34 units, M3 35 units).
- **M4: 6 of 33 units** (S-024): 4A done; 4B (3 units) counts once its GPU checks pass.
  - Weights: 4A 6, 4B 3, 4C 7, 4D 8, 4E 3, 4F 3, 4G 3.
  - 4E and 4F count once they are measured and recorded, admitted or not.

## Exact next action

1. **Next session: local, on the laptop with the RTX 3050** (the user, 2026-09-26: "i am starting you in a native chat where you can run it yourself"). Pull branch `claude/focused-faraday-ygdjx7`. It is a local session, so do not read CLOUD.md; build with `-j 2`, and GPU tests run directly (`cargo test --release -j 2 -p gpu --test lights <name>`; G4's 1080p references are cached in `%TEMP%\ne_gate_4b` when `NE_GATE_DIR` points there).
   - **Recommended, and the user moved the work local to run it** (the exact words "go on 1 and 2" were not given; confirm in one line before the first edit):
     - (a) **G5 test correction:** stronger edits so R2's 3E-only control has power (the whole lamp head; a neon box the street camera sees). Record it as a correction after the run in the record, with the reason ("only 1 of 3 edits judged; 3E-only 1.7× reset against 3×"). Pass limits unchanged. Then rerun `edits_relight_emitter_light` only (about 8 min).
     - (b) **G4 diagnostic, no engine change:** where the filter loses energy. Per pixel, filtered minus raw luminance at ages 1 / 16 / 64 from the same frames, binned by the raw value's rank (and by the variance the filter uses). The hypothesis to confirm or reject: the asymmetric SVGF weights average rare bright samples down and dark neighbours reject them (`shaders/denoise.slang`). Include a control that can fail (e.g. a flat-noise input where the filter should keep energy).
   - **Not authorized:** changing the filter or its defaults (outside 4B). After (b), the user chooses: a filter fix (scope change, own record or 4B amendment) or accepting the loss as a 4B limitation.
2. After those decisions, 4B closes (3 units) or its accepted failures are recorded. ADR-0006 Amendment 2 moves from proposed once G5 is settled.
3. **The user judges the night look**: `viewer.exe --scene night --hour 17.5 --run-day` (lights, colours, exposure; L flips the lights). With S-025, once the sun is below −12° (hour 19.14, about 19:08, on the default path) the age view should go white while the day runs; it was capped at 8–16 frames before.
4. **Before 4C** (not authorized):
   - the user points to the P05 / P08 reviews and PDFs;
   - the ReSTIR direct-light sources need the user's copies or a download request.
5. **Known debt** (propose only if it becomes needed):
   - the acceleration update waits on the host, edit frames cost about 4 ms more CPU, and descriptor sets are rebuilt wholesale on every edit;
   - FIFO on sky-heavy views (M1 record §3b);
   - the twilight sky reference is heavy-tailed (3A record; again in the 4A blue-hour references), and below −6° the sky correction uses its −6° slice;
   - the temporal pass is memory-bound (1.26 ms at 1080p): RGBA16F history is a lever for the M4 gate (Phase 4 proposal §4G);
   - optional: a season (declination) setting (3B);
   - the viewer's scripted walk keeps stepping while the window is minimized (3G record; only disturbed runs are affected).
