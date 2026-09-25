# Change (planned): Phase 3 — honest sampled lighting and reconstruction

Status: **authorized 2026-09-24 (S-017): M3, slices 3A–3G.** Emissives, glass and water are not authorized.
Date: 2026-09-24. Authorization to write this proposal: S-016 (user: "Confirming it also for the proposal").
Builds on Phase 2 (2A–2E done, M1 and M2 accepted).

## Decisions (user, 2026-09-24)

> "Sure for the sunlight and dynamic ambience lets do it during the day at different angles/times dusk dawn m3 sure fhd 60 denoiser quality sure feel free to use the nvidia flip image comparison ... denoiser is our own ... Sky ... leaving it up to your discersision green lighting evertyhing else"

1. **Order:** day first (sun, sky, bounce) for M3, emissives after. The sun moves through the day: **dawn, low sun, midday and dusk** are the reference times, and the time of day can change at runtime ("dynamic ambience").
2. **M3 and gate:** as recommended; 1080p, 60 fps p99, plus the reference quality budget.
3. **Metric:** relative MSE and mean-luminance bias, **plus NVIDIA FLIP** (a Python tool, not an engine dependency; A-008).
4. **Reconstruction:** our own (SVGF-style). Vendor reconstructors are out of scope.
5. **Sky (delegated):** the recommendation changed because dawn and dusk are in scope. A gradient sky and Hosek–Wilkie both fail at a sun near or below the horizon, where the sky colour matters most. **Chosen: a physically based atmosphere in our own code** (Rayleigh + Mie + ozone; transmittance, multiple-scattering and sky-view look-up tables in the style of Hillaire 2020). The sun's colour at the ground comes from the same transmittance, so a low sun is orange by construction. The reference renderer ray-marches the same atmosphere without look-up tables, which is the reference for the tables. 3C grows from 3 to 5 units (M3: 35). *Implemented (3C):* Hillaire's tables plus Bruneton's ground irradiance; see ADR-0005 Amendment 1 and the 3C record for the measured bias.
6. **Authorization:** all of 3A–3G.

## 1. Where lighting stands today

- **Materials** (`world::material::MaterialParams`) hold only a linear base colour and an emissive colour, "in the same relative units as base colour". There are no physical units, no roughness and no glass or water semantics. `glass` is registered as a dark diffuse material (`world/src/scene.rs`).
- **The lit view** (`View::Lit`, M1) is a debug shade: base colour × (sun × N·L + hemisphere ambient) + emissive, with exposure and an ACES fit. It has no shadows, no occlusion and no bounce light. `ref_render` (CPU) adds one shadow ray, but it is also a debug view.
- **What Phase 3 can reuse:**
  - the G-buffer (depth, normal, exact material, surface id);
  - a TLAS that matches raster per pixel, so a ray query from a G-buffer pixel sees the committed scene;
  - edits that swap raster and acceleration data as one snapshot;
  - the CPU voxel DDA (`world::reference::trace`) as an exact geometric oracle.
- **What does not exist:** motion vectors, history buffers, reset tracking, light units, a light list, any reference renderer beyond debug shading.
- **Scene lights:** the street has a sun, street lamps (`lamp`) and a shop sign (`shop_sign`), both emissive voxel materials.

## 2. What Phase 3 is for

- **Roadmap** ([BUILD-ROADMAP](../../BUILD-ROADMAP.md) Phase 3): BSDFs; sun, environment and emitter sampling; PDFs and MIS; visibility; a high-sample reference; an initial low-sample real-time path; a guide schema; native reconstruction. Glass and water in a staged material suite. A vendor reconstructor only after validating its requirements.
- **Roadmap gate:** basic transport and guide/reset behaviour are understood before reuse (Phase 4). Missing light or wrong motion is not hidden under more history or clamping. The reference and the real-time target are separate acceptance results.
- **P-001:** quality first; order emissive → sun/sky → glass → water; native reconstruction before DLSS/RR; frame time measured, not an acceptance limit (M1 did set 60 fps for walking).

## 3. Proposed slices (in order)

Every slice ends with its own record, as in Phases 1–2. Each keeps a control arm and a way to switch the new term off.

### 3A — Light conventions and the reference renderer

- **ADR-0005, light-transport conventions:**
  - radiometric units: sun irradiance at normal incidence, sky radiance, emitted radiance; exposure as a separate camera setting;
  - the BSDF: Lambertian only, since materials carry only a base colour (albedo ≤ 1 enforced at registration);
  - the ray-origin offset convention for secondary rays (voxel faces are axis-aligned, so the offset is declared exactly, not tuned);
  - sampling conventions: PDFs in solid angle, MIS with the balance heuristic, a documented RNG and seed per pixel/frame.
  - Changing `MaterialParams` meaning would touch the save format (ADR-0002). Recommended: keep the stored fields and define their units in ADR-0005, with no format bump.
- **Reference renderer:** a GPU progressive path tracer (ray query, f32 accumulation over many frames) using the same TLAS and material table.
  - Selectable components and path length, so every later slice compares against its own term: direct sun, sky, emitters, one bounce, full.
  - A CPU cross-check on small images through `world::reference::trace` (the geometry oracle), so the GPU reference itself is tested.
  - Fixed reference cameras (day, low sun, night) and fixed camera paths for motion sequences, saved as HDR (f32) images.
- **Tests:**
  - white furnace: a closed box of albedo *a* under uniform emission converges to L_e / (1 − a);
  - a Lambertian plane under the sun matches the closed form;
  - GPU against CPU reference within the statistical error of the sample count;
  - negative controls: a planted wrong PDF and a planted missing cosine must fail.

### 3B — Sun and shadows

- One shadow ray per pixel from the G-buffer position toward the sun.
- **Exact control:** with a point sun (hard shadows), the shadow bit must match a CPU DDA shadow ray per pixel, with the same edge-exclusion rule as ADR-0003 Amendment 1 (excluded count reported).
- **Real-time term:** the sun as a disk (angular radius declared in ADR-0005), sampled with one ray per pixel. Its accumulation over frames must converge to the reference's direct-sun component (mean bias within 3 standard errors).
- Measure the shadow-ray cost per granularity setting. ADR-0003 Amendment 2 says to revisit the default when secondary rays make trace time weigh more.

### 3C — Sky

- An analytic sky model with documented parameters, behind a swappable function (see decision 5).
- Sky light by cosine-weighted hemisphere rays (1 per pixel), with MIS against the sun disk so the two do not double count.
- Occlusion comes from these visibility rays. There is no separate ambient-occlusion approximation; the old engine's AO record is only evidence that AO cost and value must be measured, not assumed.
- **Test:** accumulated 1-spp sky converges to the reference's sky component.

### 3D — Temporal foundation (before any filter tuning)

- **ADR-0006, guide schema and history contract:**
  - guides: depth, normal, material, surface id, motion, albedo (for demodulation);
  - the previous frame's guides are kept (the G-buffer becomes ping-pong);
  - motion from depth and the previous camera. The world is static between edits, so camera motion is exact for surviving surfaces.
- **Per-pixel history state:** age in frames; a validity flag; a rejection reason: off-screen, depth/normal mismatch, surface id or material changed, region edited (snapshot version), resize, camera cut, light change.
- **Fresh production every frame** is a first-class counter: each frame writes new samples for every visible pixel, whatever the history.
- **Debug views:** age, rejection reason, motion.
- **Tests (the instrument must be able to fail):**
  - a static camera with plain accumulation converges to the reference;
  - planted "reset every frame" makes age stay at 0 and is detected;
  - planted "never reject" leaves ghosting after a disocclusion and is detected;
  - an edit resets only the pixels of the touched regions.

### 3E — Native reconstruction

- Temporal accumulation with the 3D rejection, plus an edge-aware spatial filter guided by depth, normal and material id (SVGF-style), on demodulated illumination (lighting ÷ albedo), then re-modulated. No vendor SDK (P-001).
- **Error budget:** set before tuning (decision 3), measured against the reference on still cameras and on the motion paths:
  - the mean luminance must stay within the budget (no energy loss hidden by the filter);
  - error against the reference, compared with the unfiltered accumulation at the same frame count;
  - blur is not a win: detail at material and geometry edges is checked (material id is exact, so the filter must not cross it).
- **Ablations:** temporal only, spatial only, both, and the unfiltered noisy input.

### 3F — One bounce of indirect diffuse light

- One continuation ray per pixel; at its hit, sun and sky are evaluated with their own shadow/visibility rays.
- Without it, shadowed street areas are darker than the reference, and that missing light must not be hidden. Energy from bounces ≥ 2 stays missing and is reported per reference camera.
- **Test:** accumulated result converges to the reference with path length limited to one bounce.

### 3G — M3 gate

- Performance series of the real-time path on the looping street walk, in the M1 method (interleaved runs, p99).
- Quality against the reference on the fixed cameras and motion paths, against the 3E budget.
- Edit behaviour: history resets only where the edit lands; edits still meet the edit budget.
- The user walks the street and judges it.

**Later in Phase 3, not in M3:** emissive lights (a light list built from emissive quads as a versioned snapshot product, area sampling with MIS; the night scene), then glass, then water, each with its own reference tests. They become the next milestone.

**Excluded from Phase 3:** reservoir reuse (ReSTIR is Phase 4), extra receiver layers, LoD transport, DLSS/RR, off-thread jobs and acceleration-update changes (known debt, proposed separately only if measured necessary), new engine dependencies.

## 4. Budgets

- **Device memory** (estimate; measured through the ledger when built), 1920×1080:

  | Item | Bytes/px |
  |---|---|
  | previous-frame G-buffer (ping-pong) | 18 |
  | motion (RG16F) | 4 |
  | noisy illumination (RGBA16F) | 8 |
  | history illumination, ping-pong (RGBA16F) | 16 |
  | moments and age, ping-pong | 12 |
  | filter scratch, ping-pong (RGBA16F) | 16 |
  | **real-time total** | **about 74 (≈ 155 MB)** |
  | reference accumulation (RGBA32F, reference mode only) | 16 (≈ 33 MB) |

  That fits easily inside the 3.15 × 10⁹-byte budget; the street's scene data is small.
- **Rays per pixel in the real-time path:** 1 primary (raster) + 1 sun + 1 sky + 1 bounce + 2 at the bounce hit = 4 traced rays per pixel. The 2D primary trace cost 0.91–1.0 ms at 1080p; secondary rays are less coherent, so their cost is measured, not extrapolated.

## 5. Decisions for the user

1. **Order: day before night?**
   - P-001 lists emissive first. Recommended: **sun → sky → temporal → reconstruction → one bounce (M3), then emissives (next milestone).**
   - Why: the sun has an exact hard-shadow control and the lowest noise, so it builds and tests the shadow-ray, reference and history machinery cheaply. Emissive lamps are many small lights; they are noisy at one sample per pixel and need that machinery (and look best with Phase 4 reuse).
   - Alternative: follow P-001 literally and do emissives right after 3A.
2. **M3 definition and its performance gate.**
   - Recommended M3: *"the street lit by sun and sky, with real shadows and one bounce, in real time, checked against a reference"*, 3A–3G, 33 work units (§6).
   - Recommended performance gate: the M1 gate (p99 ≤ 16.7 ms, 1080p, MAILBOX, looping walk). If the real-time path misses it, the record reports the per-pass cost and you choose. Lowering resolution or sample count is an appearance change and needs your approval.
3. **Error metric and budget for reconstruction.**
   - Recommended: relative MSE and mean-luminance bias, computed in-engine or with the existing numpy/Pillow tooling (no new dependency). Provisional budget: mean luminance within ±2% of the reference; the numeric error limit is set from the first measured 3E input and frozen before tuning.
   - Alternative: add NVIDIA FLIP (a Python tool dependency, not an engine one) as a perceptual metric. Needs your approval.
4. **Reconstruction approach.**
   - Recommended: a native SVGF-style filter (temporal + edge-aware spatial on demodulated light). It is well understood and its reset behaviour is testable. DLSS/RR stays last, per P-001.
5. **Sky model.**
   - Recommended: start with a simple analytic sky (documented parameters, swappable), because the goal of 3C is correct sky sampling and visibility. A physically based model (e.g. Hosek–Wilkie) can replace it later as a measured upgrade.
6. **Authorization.** Recommended: authorize 3A (conventions ADR and the reference renderer). It needs no new dependency and everything else is measured against it. Authorize later slices one at a time, or together as with M2.

## 6. Proposed M3 work units (for the progress line)

| Slice | Units |
|---|---|
| 3A conventions + reference | 6 |
| 3B sun and shadows | 4 |
| 3C sky (atmosphere, since decision 5) | 5 |
| 3D temporal foundation | 6 |
| 3E native reconstruction | 7 |
| 3F one bounce | 4 |
| 3G M3 gate | 3 |
| **Total** | **35** |

## 7. Risks

- **Units change the look.** Moving the lit view to physical units changes exposure and colour; the old lit view stays as a debug view for comparison.
- **The sky and bounce terms are the first noisy inputs.** The filter's quality decides how the street looks at one sample per pixel. The budget in decision 3 keeps it honest.
- **Secondary-ray cost on this laptop is unknown.** Greedy meshes traced about 2× slower in validation-on runs (2E, undiagnosed); that is measured again in 3B with validation off.
- **Edits and history.** Region-level reset may flash lighting near edits; it is visible in the rejection view and judged on motion, not stills.
