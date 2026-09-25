# Change (planned): Phase 4 — the street at night, and reservoir reuse

Status: **accepted 2026-09-25 (S-024): M4 as proposed; 4A and 4B authorized.** 4C–4G are not authorized. Written under S-023 (user: "Write it").
Date: 2026-09-25. Builds on Phase 3 (M3 accepted, S-022). Plan sources: [BUILD-ROADMAP](../../BUILD-ROADMAP.md) Phase 4, [PROPOSITION](../../PROPOSITION.md) §5, [TECHNIQUE-MAP](../../TECHNIQUE-MAP.md) (P05, P08), decision A-003 (advanced ReSTIR: proposed, not accepted), P-001 (order: emissive → sky → glass → water).

## Decisions (user, 2026-09-25)

> "This time i agree with all your decisions"

All §6 recommendations: (1) M4 as in §3; (2) night content (b), colours proposed in 4A for you to adjust; (3) the night look as recommended; (4) sources (a): you point me to the P05 / P08 reviews and PDFs (the ReSTIR direct-light sources for 4C need your copies or a download request); (5) 4A and 4B authorized.

## 1. Where lighting stands after M3

- **Real-time path:**
  - raster G-buffer;
  - the sun by next-event estimation (a disk, one shadow ray);
  - the sky by cosine sampling against the corrected table sky;
  - one bounce (the sky ray continues, with sun and sky at its hit), for up to 4 traced rays per pixel;
  - then the temporal pass (ADR-0006) and our SVGF-style filter.
- **Reference:** a GPU path tracer with a CPU cross-check, implementing the same estimator (ADR-0005), with selectable components and `max_bounces`.
- **Cost** (3G, validation off, the looping walk, 1080p):
  - frame p50 6.6–7.4 ms, p99 10.0–11.2 ms, against 16.67 ms;
  - shade 1.66 ms, temporal 1.31 ms, filter 3.05 ms.
  - One more ray per pixel costs about 0.15–0.4 ms (3F: +0.29–0.77 ms for the bounce's two rays).
  - That leaves about 5.5 ms of p99 headroom.
- **Memory:** about 215 MB live in the ledger (`GpuTemporal` 170 MB), against the 3.15 GB budget.
- **Emission is off** (ADR-0005). The street registers two emissive materials, 3 lamp heads and 3 sign boards, whose `emissive` values have no unit yet.
- **Magnitudes that shape this phase:**
  - by day, one bounce already reaches 98–99.5% of the 8-bounce reference (3F B3), so more bounces barely change daylight;
  - 6 emitters are not a many-light workload.
- **Accepted limitations carried forward:** S-018 (motion blur of shadow edges), S-019 (filter cost), S-021 (Q2 edge softening).

## 2. What Phase 4 is for

- **Roadmap steps, in order:**
  1. a reviewed direct-light reservoir method where a many-light workload warrants it, keeping the simpler sun path;
  2. indirect path reuse, with path state, shifts, weights and canonical support;
  3. compatibility-guided candidate selection (P05);
  4. the applicable ReSTIR PT Enhanced work reductions (P08), with reciprocal pairing only where both evaluations exist.
- **Roadmap gate:**
  - reuse is observed;
  - quality matches the declared estimator;
  - there is a net time or memory benefit at matched quality;
  - biased options are labeled and judged under an explicit approximation budget.
- **Roadmap stop conditions:**
  - a cache that never becomes valid, or a reset every frame;
  - unexplained energy loss or invalid shifts;
  - overflow without a fallback;
  - a "win" that is blur hiding lost detail.
- **The workload decides.** Reservoir reuse pays where many small lights make one light sample per pixel noisy. The street has 6 lights. So Phase 4 first gives the street a real night, with its lights plus a many-light dressing. It then measures where reuse starts to pay instead of assuming it (lesson R06: check the magnitude before scoping).

## 3. Proposed milestone M4

*"The street from dusk into night, lit by its own lamps, signs, neon, windows and string lights (hundreds of small lights), with sun and sky as today and multi-bounce light where it matters, at 1080p 60 fps, checked against a reference."*

- Daytime stays exactly as accepted in M3: the lights are off while the sun is up (decision 3).
- The gate (4G) covers performance at dusk, blue hour and night plus a daytime regression, quality against the reference, edits that include lights, and your walk.

## 4. Proposed slices (in order)

Each slice ends with its own record, keeps its control arm and can switch its new term off. Criteria are frozen in each record before its first run, as in Phase 3; the numbers below are provisional. Swept parameters pick their defaults from measured plateaus.

### 4A — Emitters, units and the reference (6 units)

- **ADR-0005 Amendment 3, emitters:**
  - **Units:** `MaterialParams.emissive` becomes emitted radiance in ADR-0005 units. There is no save-format change, as with base colour.
    - For authoring: 1 unit ≈ 1.3 × 10⁵ cd/m², since E_sun = 1 is about 128,000 lux above the atmosphere.
    - Values come from real luminances, checked against lighting references in 4A and judged by you. Rough orders of magnitude: lamp heads 10⁴–10⁶ cd/m²; neon and signs 10²–10⁴; lit windows 10¹–10².
    - Today's values (lamp 1.0, sign 1.0) would make a sign as bright as a lamp.
  - **Emitter model:** one-sided Lambertian emitters; an emissive quad is one light.
  - **Sampling:** emitters are sampled only by next-event estimation, as the sun is. A bounce ray that hits an emitter does not add its emission, so nothing is counted twice. Emission is seen directly only through primary visibility.
  - **Selection:** in proportion to power, via an alias table, then a uniform point on the quad.
- **Emitter table:**
  - a derived product built from the snapshot's emissive quads (world-space corners, normal, area, radiance, power);
  - published with the meshes and the TLAS, as the proposition's publication set requires (ADR-0003 amendment);
  - each emitter has a stable identity (region key, quad index, region snapshot); an edited region's emitters are new emitters.
- **Reference:** the CPU and GPU references gain emission and emitter next-event estimation at every vertex. The components add "emitters direct" and "emitters indirect".
- **Emission at the primary hit** is deterministic. It is added after reconstruction, not demodulated or filtered; the filter's guides stay the surface's own.
- **Night dressing** (decision 2):
  - a new deterministic scene, `street_night` = the M3 street plus lights;
  - `street_block` stays unchanged, so every M3 test and reference stays reproducible;
  - a density parameter drives the light-count sweep (about 6 / 100 / 700 / 5,000 emissive quads), for measurement only;
  - the viewer gets `--scene` and L (lights on / off).
- **Night display** (decision 3):
  - the metric exposure per reference camera comes from the reference image (log-average luminance);
  - reason: M3's metric exposure (sun plus sky irradiance) goes to zero at night.
- **Tests:**
  - one emissive rectangle over a plane, against Lambert's closed-form polygon irradiance;
  - an emissive furnace (a closed box of albedo *a* with emitting walls): L_e / (1 − a);
  - GPU against CPU per pixel;
  - the alias table against the emitters' power (χ²);
  - removing a lamp voxel changes the table in the same snapshot, a planted stale table is caught, and the table build stays inside the edit budget;
  - planted faults: an area pdf without the solid-angle conversion, emission counted twice, a missing emitter cosine.
- **Magnitudes before designing reuse (R06):**
  - times: dusk (17.75 h, +2.7°), blue hour (18.5 h, −5.3°), night (21 h, −30°);
  - the share of the image in bounces ≥ 2 (reference `max_bounces` 1 against 8);
  - the noise of one emitter sample per pixel.
  - These numbers set 4D's path length, and whether 4C and 4D pay at all.

### 4B — Many lights without reuse: the control (3 units)

- One emitter sample per pixel, chosen by power, with one shadow ray, at the primary hit and at the bounce hit.
  - Then the existing temporal pass and filter.
  - ADR-0006's relight rules gain emitters: an edit that changes a light relights the pixels in its reach (a declared approximation, measured like 3E's R2).
- **Swept:**
  - samples per pixel (1, 2, 4), which draws the equal-time curve every reuse arm is compared with;
  - the light count (4A's sweep).
- **Tests:**
  - converges to the reference's emitter components (within 3 SE);
  - energy, error and FLIP after the filter, as M3's Q1, Q2 and Q4;
  - cost per pass.
- This is also the first time you can see the street at night in the viewer.

### 4C — Direct light from reservoirs (ReSTIR DI) (7 units)

- **Primary-source review first:** the roadmap requires one for the direct-light method.
  - Bitterli et al. 2020 (spatiotemporal reservoir resampling);
  - Lin et al. 2022 (generalized resampled importance sampling, GRIS);
  - the 2023 ReSTIR course notes.
  - Result: **ADR-0007, the reservoir estimator contract**, written before any code.
- **Estimator (proposed for ADR-0007):**
  - **Candidates:** from 4B's source (power × area). The target function is the luminance of the unshadowed contribution. One visibility ray checks the selected sample.
  - **Reuse:**
    - temporal, from the reprojected pixel: nearest valid tap only, since reservoirs are not interpolated;
    - spatial, from k neighbours.
    - The shift is the identity on the light point, so the Jacobian is 1.
  - **Weights:** GRIS weights with pairwise MIS against the pixel's own fresh (canonical) reservoir. The weights account for how each reused reservoir was produced, including the visibility test it passed, so the estimator is unbiased. This costs visibility rays; skipping them, or using 1/M weights, is allowed only as a labeled biased arm.
- **Reservoir history (ADR-0006 Amendment 2):**
  - reservoirs follow the colour history's rejection reasons (edited, relit, disoccluded, light jump), plus one of their own: the light's region was rebuilt;
  - an M cap bounds how long an old selection survives.
- **Arms:** RIS only; plus temporal; plus spatial. Each is compared with 4B at equal time.
- **Swept:** candidates (1–32), M cap (5–40), neighbours (0–5), radius (8–32 px), and `max_age` again, since better samples may allow shorter histories.
- **Criteria (provisional):**
  - **Unbiased:**
    - the mean over 64 independent seed sequences, at frames 1, 4 and 16, is within 3 SE of the reference's emitter-direct term, on a still camera and on the motion path;
    - planted faults must fail this test: no visibility in the weights with occluders present, a wrong pdf, no fresh candidates, a dropped M cap.
  - **Gain:**
    - at equal GPU time, the one-frame error is at most half of 4B's in arms with ≥ 100 emissive quads;
    - the light-count sweep shows where reuse starts to pay. If it doesn't pay at 6 lights, 4B stays the method for small counts.
  - **After the temporal pass and filter:** error and FLIP no worse than 4B at equal frame time; energy within ±2% at ages 1–64; motion energy within 1%.
  - **Counters (lesson R04):** valid temporal reservoirs, accepted neighbours, samples killed by visibility, and fresh candidates in every pixel every frame.
  - **Correlation:**
    - reuse correlates neighbouring pixels and frames, which our accumulation and filter assume away;
    - reported per arm: an error-clumping ratio, the error's variance after a 9×9 box blur ÷ its variance before (≈ 1/81 for independent noise);
    - plus your judgement in motion.
  - **Edits:** adding or removing a light relights what it should, a planted stale light reference is caught, and edits stay within the edit budget.

### 4D — Path reuse for indirect light (ReSTIR PT with reconnection) (8 units)

- **Entry condition:** starts only if 4A and 4B show the indirect term matters at night, either as a share of bounces ≥ 2 or as indirect noise limiting the filtered error.
- **Arms:** when on, it replaces the 3F bounce; the 3F bounce stays as the control.
- **Paths:**
  - a continuation from the primary surface, up to N bounces (swept 1–4, Russian roulette after 2);
  - sun, sky and emitter next-event estimation at every vertex, using 4B's sampler.
- **The stored sample is the first secondary vertex x₂, with its outgoing radiance:**
  - Every material is Lambertian (ADR-0005), so x₂'s outgoing radiance doesn't depend on where it's seen from. In a static scene, reconnection is exact without re-tracing the rest of the path.
  - **Shift to a neighbour:** reconnect its primary surface to x₂, with the solid-angle Jacobian (cos φ′ / cos φ · d² / d′², at x₂) and a visibility ray on the new segment.
  - **Escaped sky samples** keep their direction (Jacobian 1).
- **Estimator:** the same GRIS, pairwise-MIS and canonical-sample contract as 4C. Emission at x₂ is excluded, because the direct-light term already covers it by next-event estimation.
- **Lighting changes:**
  - the stored radiance is old after a lighting change; the sun-motion cap and the relight rules bound that, as for the colour history;
  - re-evaluating the stored radiance every frame is a measured alternative arm.
- **Criteria:**
  - unbiased against the N-bounce reference (seed ensemble); a dropped Jacobian and a missing visibility check must fail;
  - lower error than the N-bounce path without reuse, at equal time;
  - energy and motion as in 4C;
  - counters: valid reconnections, the Jacobian distribution, shift evaluations per pixel.
- **Not needed yet:** glossy materials would need the hybrid (random-replay) shift, which isn't needed while every material is Lambertian.

### 4E — Compatibility-guided neighbour selection (P05) (3 units)

- It replaces the geometric neighbour test in 4C's and 4D's spatial reuse, following the source paper, which is reviewed first.
- It only chooses neighbours; it doesn't replace visibility, weights or the canonical sample (technique map).
- **Arm:** compared with the geometric test at the same neighbour count, by accepted-shift rate, error at equal time and cost.
- Admitted only if it wins.

### 4F — ReSTIR PT Enhanced work reductions (P08) (3 units)

- The reductions from the paper that apply to our estimator, each listed with its preconditions in the review.
- **Reciprocal pairing first:** where pixel p reuses q and q reuses p, pairwise MIS evaluates each shift in both directions, and pairing shares those evaluations. It is applied only where both evaluations actually exist.
- Measured by shift evaluations and visibility rays per pixel, and by time at matched quality.
- Admitted only if it wins.

### 4G — M4 gate (3 units)

- **P:** the M1 method, 1080p, p99 ≤ 16.67 ms.
  - Arms: dusk (17.75 h), blue hour (18.5 h), night (21 h), and midday as a regression (lights off).
  - Data: the running day through sunset.
- **If P misses:**
  - first the known levers that don't change the image: RGBA16F filter buffers (S-019's list) and RGBA16F history (known debt, ADR-0006);
  - anything that changes the image (half-resolution indirect reservoirs, fewer neighbours or bounces) is your decision.
- **Q:** energy, error and FLIP against the reference on the fixed cameras and the motion path, at the declared budgets.
- **E:** edits that include lights, within the edit budget.
- **U:** your walk from dusk into night.

## 5. Budgets

- **Device memory** (estimates at 1080p, measured through the ledger under a new `GpuReservoir` category):

  | Item | Bytes/px |
  |---|---|
  | direct-light reservoirs, ping-pong (light id, point on the light, W, M) | 2 × 16 |
  | path reservoirs, ping-pong (x₂, its face, radiance RGB16F, W, M, region and snapshot) | 2 × 32 |
  | spatial-reuse outputs | 48 |
  | **total** | **about 144 (≈ 300 MB)** |

  - The emitter table is about 48 B per light (≈ 50 KB at 1,000 lights).
  - Total live memory: about 0.5 GB of the 3.15 GB budget.
- **Frame time** (a rough estimate from the costs above, not a forecast):
  - direct-light reuse: about 4 rays per pixel plus three memory-bound passes, roughly 1.5–2.5 ms;
  - path reuse at 2 bounces: about 9 rays plus passes, roughly 3–5 ms, minus the 0.3–0.8 ms of the 3F bounce it replaces.
  - Together, at full resolution, that is about 3.7–7.2 ms against 5.5 ms of headroom: it may not fit on this GPU. 4D is therefore built with a resolution / rate lever from the start, and the choice is yours once it's measured (4G).
- **GPU time for runs:**
  - night references: about 5 min each at 16,384 spp (twilight took 300–330 s), for 2 cameras × 3 times, cached;
  - 8-bounce references for the magnitudes: about twice that;
  - seed ensembles: a few minutes per arm;
  - performance series: about 40 min per gate.
  - About 3–5 h of GPU runs over the phase, each run announced.

## 6. Decisions for the user

1. **M4 definition** (§3).
   - Alternative: M4 = night with direct light only (4A–4C and 4G), with multi-bounce light later.
2. **Night content:**
   - (a) only today's 3 lamps and 3 signs. That isn't a many-light workload, so reuse would not be justified.
   - (b) **Recommended:** add neon outlines on the shop fronts, lit upper windows and string lights along the street: about 150 lights, ~700 emissive quads. You choose the colours.
   - (c) more street (Phase 6 territory).
3. **Night look:**
   - **Lights:** they switch on when the sun sets (elevation < 0°) and off when it rises, so daytime stays exactly as accepted in M3 (**recommended**). The switch is a light jump (one reset); L toggles the lights by hand. Alternative: always on.
   - **Display:**
     - automatic exposure in the viewer (log-average luminance, about 1 s of adaptation, bounded; − and = still offset it) (**recommended**), or a fixed night exposure;
     - night light is about 10⁴ times dimmer than day (a lamp-lit road gets ~10–30 lux; sunlight ~10⁵ lux), so one exposure can't serve both.
   - **Sky:** the night sky stays physically dark in M4, with no moon, stars or city glow. Any of those would be a later, labeled addition.
4. **Sources:** the owning reviews and the supplied PDFs for P05 and P08 are not in this workspace (the technique map's `../research-review/` doesn't exist here).
   - (a) **Recommended:** you point me to them.
   - (b) I download the named papers when 4C starts, asking each time with filename, source and size.
   - Working from memory is not recommended for P05 and P08, whose details I can't verify.
5. **Authorization:**
   - **Recommended:** 4A and 4B now. Every later slice needs them, and their measurements (the 4A magnitudes, the 4B curve) decide whether 4C and 4D pay here.
   - Alternative: all of 4A–4G, as with M3.

**Delegated** (lighting technique, as in Phase 3): the unbiased estimator is the contract and the default. Biased shortcuts appear only as labeled arms, admitted under a declared approximation budget.

## 7. Proposed M4 work units (for the progress line)

| Slice | Units |
|---|---|
| 4A emitters, units, reference | 6 |
| 4B many lights without reuse (control) | 3 |
| 4C reservoir direct light | 7 |
| 4D path reuse | 8 |
| 4E compatibility-guided neighbours (P05) | 3 |
| 4F Enhanced work reductions (P08) | 3 |
| 4G M4 gate | 3 |
| **Total** | **33** |

4E and 4F count as done when measured and recorded, whether or not they are admitted.

## 8. Risks

- **Correlated reuse against our reconstruction.**
  - The filter's variance comes from per-pixel moments, which assume independent samples.
  - Reuse makes errors clump in space and time ("blotches") and can leave the filter under-smoothing.
  - Measured by the clumping ratio, FLIP and your eye.
  - Mitigations, in order: the M cap, a reservoir age cap, then a decorrelation method (e.g. MCMC mutations, Sawhney et al. 2024) as its own reviewed arm.
- **Frame budget on a hot laptop GPU** (1,530 MHz under thermal slowdown). See §5: the full stack may need a lever.
- **Subtle estimator bugs** look plausible in stills. The seed-ensemble unbiasedness test with planted faults is the main protection, and a biased shortcut never becomes a default without a budget.
- **Heavy-tailed night references:** small bright emitters make them noisy, as the twilight reference already was. Next-event estimation is required in the reference, and convergence is checked by standard error, as in 3A.
- **Stale reuse after edits and light switches** (removed lights, newly unblocked lamps): handled by stable identities, the relight rules and the M cap, and measured as in 3E's R2.
- **Primary sources:** decision 4.
- **Night appearance** depends more on exposure and tone mapping than on any estimator. It is judged by you from 4B onwards.

## 9. Not in Phase 4

- **Glass and water:** transmission is the milestone after M4, per P-001.
- **Other effects:** moon, stars and skyglow; bloom and glare; fog and light shafts (participating media).
- **Later phases, none excluded; each keeps its roadmap phase and gate:**
  - extra receiver layers and LoD transport (P06, P07): Phase 5;
  - sharing, paging, streaming and empty-space bounds (P01–P03, P13): Phase 6;
  - the neural radiance cache and texture work (P04, P10, P12): Phase 7;
  - kernel, queue and upload work (P09, P14, P16, P18–P21): Phase 8.
  - P11, P15 and P17 keep their technique-map roles.
- **New engine dependencies.**
