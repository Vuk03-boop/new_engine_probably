# Change: 4B filter fix — the filter keeps night energy (G4)

Status: **in progress**: criteria frozen, built, cloud checks run; **GPU checks NOT RUN** (`run-local.cmd`).
Date and baseline: 2026-09-26, branch `claude/sharp-faraday-o4fy96` from `df39b9a` (4B's G5 settled, G4 diagnosed).
Authorization: S-026 (the user, 2026-09-26: "accept g5 and dont fix the filter yet will boot you up in the cloud to fix the filter"); in this cloud session: "fix the filter then". The current filter default stays until the user judges the results (NOW: "the user decides the default change").

## Objective

4B's G4 fails: at night the filter removes 5–24% of the reflected light (filtered minus raw history of the same frames), and 4B's local session showed the cause is the luminance edge-stopping ([4B record](2026-09-25-phase4b-many-lights.md), "Local session 2026-09-26"). Make a filter setting that keeps the energy at night without making M3's day image worse, as a new `DenoiseSettings` field whose off state is today's filter.

## Why the filter loses energy (reasoning, then checked on the CPU mirror below)

An a-trous level computes `out_p = Σ_q w_pq x_q / Σ_q w_pq`. The total is kept only if every pixel's value is handed out in fractions that add up to 1, which a per-output normalisation does not do:

1. **Asymmetric weights.** SVGF scales the edge-stopping by the *output* pixel's variance. A pixel that never drew a bright sample has variance ≈ 0 (from its own moments after `moments_age`) and rejects a bright neighbour, while the bright pixel, whose variance is large, accepts the dark ones and averages itself down. Its energy leaves and is not received. This is 4B's refinement hypothesis; it explains why the loss grows from age 1 to 16.
2. **Normalisation.** Even with symmetric weights, a pair's exchange is divided by two different sums. A bright sample that half-accepts its neighbours keeps about 23% of itself while they take about 49% (a 5×5 B-spline estimate), so about 28% is lost. Dark pixels that reject each other have tiny weight sums, and one accepted bright tap then dominates their output, so energy is *gained*. The CPU mirror shows this: symmetric weights alone gain +30 to +55% at ages 16 and 64.

## The change: `DenoiseSettings::weights`

`gpu::denoise::Weights` (`src/denoise.rs`, flags in `shaders/denoise.slang`):

- **`Svgf`** (the default): today's filter. The flags are 0. The only code on this path that changed is that the surface test and the albedo lookup mask the key with `KEY_MASK`, and those bits are 0 unless `Conservative` is set.
- **`Symmetric`** (ablation): σ from the pixel's plus the tap's variance (each unblurred, each /9 when its guide is a 3×3 mean). A pair weighs the same both ways; still normalised per output pixel. Skips the variance pre-blur, since a blurred variance is not known for the taps.
- **`Conservative`** (the candidate): symmetric weights and *exchanges* instead of a weighted mean:
  - `out_p = x_p + Σ_{q≠p} c_pq (x_q − x_p)`, with `c_pq = w_pq / max(W_p, W_q)`.
  - W is the kernel's sum over the pixel's same-surface taps at the level's step, from the guides only, rounded **up** to 1/64.
  - `c_pq` is bit-identical at both ends of a pair, so two pixels exchange exactly opposite amounts, and each surface keeps its energy to f32 rounding.
  - `W ≥ Σ_{q≠p} w_pq` makes each output a convex mix: no negative weights, and a flat image stays flat.
  - Normalising by the geometric W smooths borders like the interior. The plain form (W = 1) under-smooths them (model below).
  - Variance: `Σ c² v_q + (1 − Σ c)² v_p`.
  - **Storage:** none added. Init computes W for the first two levels from the full guides and packs them into the compact key's free bits. The material is 16 bits, so `key.x` bits 0–18 are face and material, 19–24 hold level 0 and 25–30 level 1. The levels already read every tap's key, so W costs no extra memory traffic. From the third level on W = 1 (still exact; the default has 2 levels).
  - The init pass does 49 extra guide reads per pixel, mostly cached neighbours. F5 measures the cost.
- `DenoiseSettings::parse("conservative:4")` / `tag()` read and print `weights:sigma_l`. The tests take `NE_FILTER`. The viewer takes `--filter`, and K switches between it and the candidate (`FILTER_CANDIDATE`); the title, F3 and the JSON (`"filter"`) show which.

## Scope

- **Allowed:**
  - `engine/gpu/src/denoise.rs`, `engine/gpu/shaders/denoise.slang`;
  - the tests `engine/gpu/tests/{lights,gate,denoise}.rs`, which gain `NE_FILTER`, the image prefix, F2's arms and the F5 test;
  - the new `engine/gpu/tests/denoise_model.rs`;
  - `engine/viewer/src/main.rs` (`--filter`, K);
  - the FLIP scripts `engine/results/phase3g/flip.py` and `engine/results/phase4b/flip.py`, which gain an optional image-prefix argument (default unchanged);
  - `run-local.cmd`, `engine/README.md`, this record, `docs/DECISIONS.md`, `docs/NOW.md`.
- **Excluded:**
  - changing the filter default (the user decides after the results);
  - the temporal pass and the history (they stay unbiased, Q1b);
  - clamping or other biased firefly removal;
  - new buffers (none are added: the push constants are unchanged, `DenoiseTargets` stays 44 B/px);
  - 4C–4G.
- **Invariants:**
  - with `weights: Svgf` the filter computes what it did before (checked on the GPU by F2's reproduction of 4B's printed numbers);
  - the filter writes only the shown radiance;
  - taps stay on the same surface (exact guides).

## Before the criteria: the CPU mirror (exploration, no criterion)

`gpu/tests/denoise_model.rs` mirrors every mode and flag of the shader in f32, fed with CPU frames of the viewer's transport. G1 showed the shade pass's frame f equals the reference's sample f per pixel, so the input is the viewer's own radiance. The guides come from CPU primary rays and the history is the exact running mean with its second moment. It was run before these criteria were written, to choose what to build:

- **At 192×108** (64-spp reference; its error numbers are not reliable at night):
  - the default arm reproduces the GPU's loss on the low camera (−20.2 / −13.3% at ages 16 / 64, GPU −19.4 / −12.8%); the street is smaller at this size (−18.6 / −16.5% against −24.9 / −16.3%);
  - `levels 0` keeps energy to 10⁻⁴;
  - symmetric weights gain +30 to +56% at ages 16 and 64;
  - plain conservative keeps energy exactly, but its error at dusk age 1 is 1.7× the default's (street 2.46 against 1.48);
  - border-normalised conservative keeps energy exactly, and its dusk error is 1.74 / 1.52 at σ_l 4 / 8, close to the default's 1.48.
  - Hence the design above.
- **960×540, 256-spp reference** (`NE_MODEL_SIZE=960x540 NE_MODEL_REF_SPP=256`): results and the σ_l choice under "Cloud results".

## Criteria (frozen 2026-09-26, before any GPU run)

The candidate is **`conservative:σ*`**. σ* is chosen from the 960×540 model before the run, by this rule:

- Over the 16 (camera, time, age) points of the model's night and dusk (both cameras, ages 1 / 4 / 16 / 64), take the σ_l in {2, 4, 8, 16} with the most points where the model's filtered relative MSE ≤ the raw one at gain × age (G4's Q2).
- On a tie, take the lowest summed dusk age-1 error.
- σ* is written here, and in the viewer's `FILTER_CANDIDATE`, before `run-local.cmd` is pushed.

The model is a model, and the GPU criteria below judge.

**Cloud (device-free):**

| # | Criterion |
|---|---|
| C1 | Builds and lints: `gpu` (every test binary), `viewer`, the tools (substitute SDK: supplemental, not ADR-0001 proof); clippy on the workspace with only the 6 old `world` lints; the pure suite passes; `gpu --lib` passes, including `filter_settings_parse` and the denoise reflection check (push constants unchanged). |
| C2 | `denoise_model::conservative_weights_keep_energy` (runs by default): on a synthetic two-surface history with rare bright samples, `Conservative` changes each surface's energy by < 10⁻⁵ at σ_l 1, 4, 16, and a flat image by < 10⁻⁶. **Negative control:** `Svgf` at σ_l 4 loses > 5% there. |

**GPU (laptop, `run-local.cmd`; validation on unless stated; 0 validation errors everywhere):**

| # | Criterion |
|---|---|
| F1 | **Night quality with the candidate** (4B's G4, `night_against_the_reference` with `NE_FILTER=conservative:σ*`), both night cameras. **Q1a:** \|filtered bias − raw bias\| ≤ 2% at ages 1, 4, 16, 64. **Q1b:** as 4B. **Q2:** filtered rel_mse ≤ raw at gain × age (8 / 4 / 4 / 1). **Q4:** FLIP (`results/phase4b/flip.py` with the candidate's prefix) filtered ≤ raw at ages 1, 16, 64. The 4B G4 correction applies unchanged. Blue hour is data. **Data:** the same test at σ*/2 and 2σ* (the plateau). |
| F2 | **Energy** (`diagnostic_g4_filter_energy`, extended): the conservative arms (σ_l 2, 4, 8) change the energy by ≤ 10⁻³ at ages 1, 16, 64, both cameras, night and dusk. **The default is unchanged:** its arm reproduces 4B's printed changes (`results/local-run/2026-09-26_local-4b-g5-g4/g4_diagnostic.log`) within 10⁻⁴ at every age, camera and time; the frames are deterministic. `levels 0` stays exact. The symmetric arm is data. |
| F3 | **Day: M3 not worse** (3G's Q1–Q4 with the candidate: `gate` `stills_against_the_reference` and `motion_against_the_reference`, `NE_FILTER=conservative:σ*`, references in `%TEMP%\ne_gate`, rebuilt there if missing). Pass: every check that passed at 3G passes. Q1: all 40. Q2: 38 of 40, with the 3G-failing arm (low camera, midday, ages 16 and 64) as data against 3G's 0.0098 / 0.0090. Q3: all 6. Q4 (`results/phase3g/flip.py` with the prefix): all 36 comparisons (30 still, 6 motion). |
| F4 | **3E's own criteria with the candidate** (`denoise`, `NE_FILTER=conservative:σ*`): D1–D3 (`stills_meet_the_budget`), R1 / R2 (`edits_relight_what_they_change`, `..._with_the_bounce`) and `a_moving_sun_does_not_lag` pass. D4 (`motion_path_meets_the_budget`) and C1 (`denoise_cost_at_1080p`, validation on) are data: both fail with the default as accepted (S-018, S-019). |
| F5 | **Cost** (`filter_cost_against_the_default`, validation off, 1080p): the candidate's median of per-repetition median GPU times ≤ the default's + 0.05 ms (6 interleaved repetitions after a warm-up pair). |
| F6 | **Viewer** with `--filter conservative:σ*`: `--scene night --hour 17.5 --run-day` (400 frames, edits of size 1) and `--scene street` (400 frames), validation on: exit 0, 0 errors, 0 warnings. **Data** (validation off): the night walk (3,000 frames, 1080p, MAILBOX) with the default and with the candidate, p50 / p99 frame interval and the filter pass time. |

**Decision rule:**

- If F1–F6 pass, the record proposes `conservative:σ*` as the filter default, as an ADR-0006 amendment (the filter keeps energy). The user decides.
- A Q2 failure at night is recorded as information for 4C, not rebaselined.
- A failure in F3 or F4 is diagnosed and reported. The default stays `Svgf` until the user decides.
- 4B closes (3 units) once G4 is resolved this way, or once its remaining failure is accepted.

**Work units:** inside 4B's 3. **Budget:**
- the model: about 1.5 hours of CPU in the cloud;
- the laptop run: about 1.5–2 hours; 45 minutes more if 3G's references are no longer in `%TEMP%\ne_gate`.

## Commands and evidence actually produced

| Check | Result | Evidence | Scope |
|---|---|---|---|
| (filled in as run) | | | |

## Cloud results

- σ*: **pending** (the 960×540 model is running). `run-local.cmd` is not ready until σ* is written here.
- C1 so far: pure suite exit 0, 154 pass, 6 ignored (`results/test_pure_4b_filter_cloud.log`); `gpu --lib` exit 0, 23 pass (`results/test_gpu_lib_4b_filter_cloud.log`); clippy exit 0, the 6 old `world` lints plus one in the model, fixed afterwards and to be rerun (`results/clippy_4b_filter_cloud.log`).
- C2: `conservative_weights_keep_energy` passes (Conservative ≤ 2.4 × 10⁻⁸; Svgf −30 to −43%, Symmetric +45 to +64%).

## Laptop results

NOT RUN.
