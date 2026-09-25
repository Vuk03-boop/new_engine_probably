# NOW — Phase 4 (M4): 4A part 1 built, its GPU checks waiting for the laptop

**Updated 2026-09-25.** This file is the plan to start from. A new session reads `CLAUDE.md`, then this file, then only the files it links for the task at hand.

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
- **4A in progress** ([record](changes/2026-09-25-phase4a-emitters.md)), written in a cloud session (no GPU; [CLOUD.md](CLOUD.md)):
  - part 1 built: emitter units and sampling ([ADR-0005 Amendment 3](adr/ADR-0005-light-transport-conventions.md)), the emitter table ([ADR-0003 Amendment 3](adr/ADR-0003-device-budget-and-visibility-buffers.md)), emitters in the CPU and GPU references, `street_night`;
  - CPU criteria C1–C5 pass; GPU criteria G1–G5 are **NOT RUN** (they are in `run-local.cmd`);
  - part 2 not started: viewer `--scene` and L, night exposure, night references, magnitudes, the table in `GpuScene::update`.
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
  - pure crates `memory`, `world`, `derived`, `walk`, `light` (150 tests);
  - 4A: `light::emitters` (table, alias selection, solid-angle sampling), the reference's emitter terms (off by default), `world::scene::street_night(Dressing)`, `gpu::emitters` and `Reference::bind_lit`, `gpu/tests/emitters.rs`;
  - `gpu`, on Vulkan:
    - raster, ray query and scene updates;
    - the reference path tracer;
    - real-time shade with sun, sky and bounce, the temporal pass and the filter;
    - the tools `ref_light` and `sky_bake`.
  - `viewer`:
    - walking and flying; edits (left click, E); R switches raster / ray query;
    - views: 0 the light view, 1–7 debug views, 9 the M1 lit view;
    - lighting keys: [ ] and T move the sun; H accumulation, N the filter, B the bounce;
    - flags: `--no-bounce`, `--no-sky-correction`, `--edit-size N`;
    - the Q2 comparison: P, `--prefilter-age N`, `--camera low`;
    - fps in the title and on exit.
- **Emission:** has a unit (1 unit of luminance = 128,000 cd/m², ADR-0005 Amendment 3) and is in the references, off by default. The real-time path has none yet (4B). `street_block` is unchanged; `street_night` carries the proposed night lights (colours and luminances in the 4A record, for the user to adjust).
- **Repository** (since 2026-09-25):
  - Git, branch `main`, pushed to https://github.com/Vuk03-boop/new_engine_probably (public; first commit `eac532c`).
  - 4A part 1 is on branch `claude/tender-keller-9u9d6t` (cloud session), not merged into `main`.
  - `run-local.cmd` (repository root, CRLF by `.gitattributes`): double-click to run the 4A GPU checks; logs to `engine/results/local-run/<date>-4a/`.
  - Not committed, kept locally (`.gitignore`): build output (`target/`), the Phase 0 third-party assets and tools (Bistro, RenderDoc), and GPU captures (`*.rdc`, `*.ngfx-gputrace`).
  - Binary data is protected from line-ending conversion by `.gitattributes`.
  - Commits and pushes happen only on the user's request.
- **Evidence:**
  - `engine/results/`: logs per slice;
  - `engine/results/phase3g/`: FLIP, perf series and edits;
  - `engine/results/phase3a_ref/`: reference images.

## Last checks (2026-09-25, RTX 3050)

- **4A part 1 (2026-09-25, cloud, Linux, no GPU; [record](changes/2026-09-25-phase4a-emitters.md)):**
  - pure suite: exit 0, 150 pass, 3 ignored (`test_pure_4a_cloud.log`); clippy with `gpu`: exit 0, only the 6 old `world` lints;
  - C1–C5 pass (alias χ², the rectangle's closed form, the emissive furnace, table identity, emitters-off images bit-identical); the planted faults are caught;
  - `gpu --lib` 14 pass, including the reference's reflection check (built with the pinned slangc's Linux release and a non-SDK `spirv-val`: supplemental, not ADR-0001 proof);
  - **NOT RUN:** G1–G5 and the GPU regressions (`reference`, `shade`, `sky`, `temporal`, `bounce`), all in `run-local.cmd`; the gate and `denoise` are not in it (unchanged frame path; known accepted failures).
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
- **M4: 0 of 33 units** (S-024). 4A part 1 is built but counts only when 4A passes.
  - Weights: 4A 6, 4B 3, 4C 7, 4D 8, 4E 3, 4F 3, 4G 3.
  - 4E and 4F count once they are measured and recorded, admitted or not.

## Exact next action

1. **Run `run-local.cmd` on the RTX 3050** (user: pull branch `claude/tender-keller-9u9d6t`, double-click; about 15–20 min). Then read the logs in `engine/results/local-run/<date>-4a/` against G1–G5 and the M3 regressions, and record them in the 4A record.
2. **The user adjusts the proposed night colours and luminances** (4A record table), or accepts them.
3. **4A part 2** (authorized), in order:
   - the emitter table inside `GpuScene::update`, swapped with the meshes (ADR-0003 Amendment 3), and its edit latency;
   - the viewer's `--scene` and L;
   - `ref_light` night references (the 3 times × 2 cameras, cached) and the metric exposure from them;
   - the magnitudes: bounces ≥ 2 at dusk, blue hour and night, and one-sample noise (R06);
   - its GPU parts go into `run-local.cmd` when written in the cloud.
4. **Then 4B** (authorized):
   - one emitter sample per pixel;
   - relight rules for emitters;
   - the equal-time curve;
   - automatic exposure in the viewer.
5. **Before 4C** (not authorized):
   - the user points to the P05 / P08 reviews and PDFs;
   - the ReSTIR direct-light sources need the user's copies or a download request.
6. **Known debt** (propose only if it becomes needed):
   - the acceleration update waits on the host, edit frames cost about 4 ms more CPU, and descriptor sets are rebuilt wholesale on every edit;
   - FIFO on sky-heavy views (M1 record §3b);
   - the twilight sky reference is heavy-tailed (3A record), and below −6° the sky correction uses its −6° slice;
   - the temporal pass is memory-bound (1.26 ms at 1080p): RGBA16F history is a lever for the M4 gate (Phase 4 proposal §4G);
   - optional: a season (declination) setting (3B);
   - the viewer's scripted walk keeps stepping while the window is minimized (3G record; only disturbed runs are affected).
