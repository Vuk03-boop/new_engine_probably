# ADR 0005 — Light-transport conventions and the reference renderer

Status: accepted (delegated within S-017; the user delegated lighting technique and the sky model).
Date: 2026-09-24
Decision authority: S-017, "Sky ... leaving it up to your discersision green lighting evertyhing else"; the user asked for dawn, day and dusk ("different angles/times dusk dawn").
Related: [Phase 3 proposal](../changes/2026-09-24-phase3-proposal.md), [3A record](../changes/2026-09-24-phase3a-reference.md), ADR-0003 (G-buffer, surface ids), ADR-0004 (mesh winding).

## Context and actual constraint

- Materials carry a linear base colour and an emissive colour in unstated "relative units" (`world::material`). No light unit, sky or reference exists.
- Every Phase 3 term (sun, sky, bounce, reconstruction) needs one definition of light to be compared against. The roadmap requires a high-sample reference that is trustworthy, separate from the real-time result.
- Dawn and dusk are in scope, so the sky must be right for a sun near and below the horizon.

## Options considered

- **Sky:** a gradient or Preetham/Hosek–Wilkie fit (cheap, but invalid near or below the horizon); precomputed look-up tables (Bruneton 2008, Hillaire 2020: fast, but the tables are an approximation of multiple scattering). **Chosen:** the physical atmosphere is the definition; the reference estimates it by Monte Carlo with no tables; 3C builds Hillaire-style tables for real time and measures them against the reference.
- **Units:** photometric (lux, nits) or relative. Chosen relative radiometric units anchored on the sun; physical calibration adds nothing a relative scale plus exposure does not give for this renderer.

## Decision

**Units and colour.**
- Linear RGB with Rec.709/sRGB primaries, one value per channel; no spectral rendering.
- Solar irradiance at the top of the atmosphere, measured perpendicular to the sun, is **E_sun = 1.0 per channel** (a white sun). Every radiance is relative to it. The camera exposure is separate (display only).
- The sun is a uniform disk of angular radius **0.2664°** (0.004650 rad). Its radiance is L_sun = E_sun / Ω_sun, with Ω_sun = 2π(1 − cos θ_sun); no limb darkening.
- Scene geometry stays in voxel units (1/16 m); the atmosphere is in meters.

**Sun position.** `light::SunPath`: latitude (default 45° N) and solar declination (default 0°, the equinox), hour angle 15° per hour from solar noon. World axes: +X east, +Y up, −Z north. Reference times: **dawn 6.25 h** (elevation ≈ 2.7°), **morning 8 h** (≈ 20.7°), **midday 12 h** (45°), **dusk 17.75 h** (≈ 2.7°), **twilight 18.25 h** (≈ −2.7°, the sun has set).

**Atmosphere** (`light::Atmosphere`; the parameter values of Bruneton 2017 and Hillaire 2020):
- Spherical planet, ground radius 6,360 km, top of atmosphere 6,460 km.
- Rayleigh scattering (5.802, 13.558, 33.100) × 10⁻⁶ m⁻¹, scale height 8 km, no absorption; phase 3/(16π)(1 + cos²θ).
- Mie scattering 3.996 × 10⁻⁶ m⁻¹, extinction 4.440 × 10⁻⁶ m⁻¹, scale height 1.2 km; **Henyey–Greenstein phase, g = 0.8** (chosen over Cornette–Shanks because it can be sampled exactly; the real-time tables use the same phase).
- Ozone absorption (0.650, 1.881, 0.085) × 10⁻⁶ m⁻¹, density a tent of half-width 15 km centred at 25 km.
- Planet ground: Lambertian, albedo 0.3 (grey).
- The observer (the whole street) is at altitude **2 m**. The street is small against every atmospheric scale, so the sky and the sun transmittance are evaluated from that one point. **Aerial perspective between the camera and street surfaces is not modelled** (Rayleigh optical depth over 200 m is under 1% in blue).
- **Discretised medium:** along a ray segment, extinction and scattering are constant within each march step and evaluated at the step's midpoint altitude. Steps are split at the tangent point and concentrated towards the lower (denser) end. Free-flight sampling and transmittance are exact for this discretised medium. The step count is a parameter; its convergence is measured (3A record).

**Surface model.**
- Lambertian BSDF ρ/π with ρ the material's base colour. Every channel of ρ must lie in [0, 1]; the material table refuses anything else (checked when the table is built; the world registry is unchanged).
- Emission is **off** in M3 (reference and real time alike): emitter units arrive with the emissive slice.
- Secondary-ray origin: the hit point is snapped exactly onto its voxel face plane (faces lie on integer voxel coordinates), then offset along the face normal by **1/256 voxel**. Rays are opaque and cull back faces, as primary ray query does.

**Light paths (both the reference and the real-time path).**
- Primary visibility through pixel centres, with no jitter, so reference and real-time images share the G-buffer's surfaces.
- The sun is sampled only by next-event estimation: a uniform point on the disk, one shadow ray. The sun disk is **excluded** from the sky radiance function, so no MIS is needed between the sun and BSDF sampling. A primary ray that misses the scene inside the disk sees L_sun × transmittance.
- Sky light: BSDF (cosine) sampling; a ray that escapes the scene takes the sky radiance L_sky(ω), which includes the lit planet ground below the horizon. Where two strategies later sample the same light (3C sky importance sampling), MIS uses the balance heuristic.
- The sun's transmittance at the observer, T_obs(sun), is computed once per sun direction on the host (f64, fine quadrature) and applied to every direct-sun term at the street. It is 0 when the planet blocks the sun.
- `max_bounces` counts surface-to-surface bounces: 0 = direct sun and direct sky at the primary hit only.

**The reference renderer.**
- CPU (`light::reference`, f64, the voxel DDA of `world::reference`) and GPU (`gpu::reference`, f32, ray query on the TLAS) implement the same estimator. The GPU one makes the images; the CPU one checks it statistically on small images.
- **Sky in the reference:** a Monte Carlo volumetric path through the discretised atmosphere, with all scattering orders (spectral MIS over the three channels for free-flight distances, next-event estimation of the sun at every event, Russian roulette after 3 events, a hard cap of 32 events), and the Lambertian planet ground. No tables.
- Selectable components: sun, sky, bounce count; overrides for controls: point sun, uniform white sky, albedo 1.
- RNG: PCG32 RXS-M-XS (32-bit state), seeded per pixel and frame by the PCG hash; floats are the top 24 bits. The same on CPU and GPU.

## Contracts and consequences

- Every later M3 term is compared with the reference component it estimates (sun only, sky only, one bounce), not with the full image.
- Changing a unit, the atmosphere or a convention above changes every reference image: it needs an amendment and regenerated references.
- The stored `MaterialParams` fields are unchanged; this ADR gives them meaning (base colour = Lambertian albedo). No save-format change.
- Costs: the reference is slow (a sky path per escaping ray). It is a measurement tool, not a real-time mode.
- Not decided here: emitter units, glass, water, real-time sky tables (3C), history and guides (ADR-0006 in 3D).

## Validation and revisiting

- 3A tests: white furnace (albedo 1, uniform sky: every pixel 1); a lit plane against the closed form; CPU atmosphere against a single-scattering quadrature and closed-form transmittance; step-count convergence; GPU against CPU per pixel and per image mean within statistical error; planted wrong-PDF and missing-cosine faults must fail.
- Revisit if aerial perspective becomes visible (larger scenes), if a spectral effect matters (e.g. twilight colour against measured skies), or with emissives.

## Implementation status

Implemented: the reference (3A, [record](../changes/2026-09-24-phase3a-reference.md)); the real-time sun (3B, [record](../changes/2026-09-24-phase3b-sun.md)); the real-time sky (3C, [record](../changes/2026-09-24-phase3c-sky.md)).

## Amendment 1 (2026-09-24, 3C): the real-time sky

- **Decision (delegated, S-017):** the real-time sky is Hillaire 2020's table method (transmittance, isotropic multiple scattering Ψ_ms, sky view per sun) from this ADR's atmosphere, **plus Bruneton 2008's ground-irradiance table** (skylight on the planet ground). The sun-independent tables are built once on the host in f64 and uploaded; the sky-view table is rebuilt on the GPU when the sun moves. Details and sizes: `light::sky`, [3C record](../changes/2026-09-24-phase3c-sky.md).
- **It is an approximation of the Monte Carlo sky above**, measured against it: per direction within budget (3C criterion 3), but a known bias near the horizon (isotropic multiple scattering): blue −4…−13% toward the horizon at dawn and dusk. On sky light over surfaces this is −5.7% (dawn/dusk blue) and +15.5% (twilight blue), just outside the declared budget (3C criterion 4, open for the user: a baked correction table, or accepting the measured level).
- The reference renderer keeps the Monte Carlo sky; the table sky is used only in real time.

## Amendment 2 (2026-09-25, S-020): the table sky is corrected from the reference

- **Decision (user, option A of 3C criterion 4):** the sky-view table is multiplied by the ratio reference ÷ table. The reference is the Monte Carlo sky above, baked once per direction (32 azimuths × 32 zeniths, in the sky-view coordinates) and sun elevation (33 slices, −6° to 90°), and stored in a versioned, fingerprinted file (`engine/light/data/sky_reference_v1.bin`, `light::sky_ref`). The ratio is formed at load from the current table code. Below −6° the −6° slice is used.
- **Measured:** sky light on surfaces is within 0.2% of the reference from dawn to dusk and 1.2% at twilight (3C criterion 4 passes at its budget); per direction, the table sky is within the reference's noise from dawn to dusk. [Sky correction record](../changes/2026-09-24-phase3c-sky-correction.md).
- **Consequences:**
  - A change to the atmosphere or the sky estimator's settings needs a new bake (`sky_bake`, about 18 min on the RTX 3050); the old file is refused, not silently used.
  - A missing or refused file runs uncorrected with a warning. Tests that measure the old table use `SkyLuts::new`, whose ratio is exactly 1.
## Amendment 3 (2026-09-25, 4A): emitters

- **Authority:** S-024 (the user accepted the [Phase 4 proposal](../changes/2026-09-25-phase4-proposal.md) and authorized 4A; lighting technique stays delegated). Record: [4A](../changes/2026-09-25-phase4a-emitters.md).
- **Units.** `MaterialParams.emissive` is emitted radiance in this ADR's units (relative to E_sun = 1), linear Rec.709 RGB. For authoring, **1 unit of luminance Y = 128,000 cd/m²** (`world::material::CANDELA_PER_UNIT`), because E_sun = 1 is taken as about 128,000 lux above the atmosphere. Y = 0.2126 R + 0.7152 G + 0.0722 B. Every channel must be finite and ≥ 0; the emitter table refuses anything else. No save-format change: the stored field only gains a meaning.
- **Emitter model.** One-sided Lambertian emitters: every exposed face of an emissive voxel emits its material's radiance along the face's outward normal, and reflects with its base colour like any other face. **One emitter = one surface-mesh quad** (ADR-0004) whose material is emissive. The partition does not change the image: any partition of the same faces is the same light.
- **Sampling.** Emitters are sampled only by next-event estimation, as the sun is:
  - Selection: an alias table over emitter power Φ = π · A · Y(L_e). The index is uniform (Lemire's multiply-shift with rejection, on `next_u32`); the coin compares the top 24 bits of `next_u32` with a 24-bit threshold. Every threshold of an emitter with power > 0 is at least 1, so no emitter has zero probability. The estimator divides by the **realized** selection probability computed exactly from the quantized table, not by Φ / ΣΦ.
  - A point **uniform in the quad's solid angle** seen from x (Ureña, Fajardo and King 2013, spherical rectangles; two `uniform` draws). Contribution at a vertex x with normal n: β ρ/π · L_e · cos θ_x · S / P(i), with S the solid angle.
  - When S < 10⁻² sr (a far or grazing quad; in f32 the solid angle Σgᵢ − 2π carries about 10⁻⁶ sr of cancellation error, ≤ 10⁻⁴ relative above this), a uniform point on the quad instead: β ρ/π · L_e · cos θ_x · cos θ_y / d² · A / P(i). The switch depends only on (x, emitter), so each branch stays unbiased.
  - Why not area sampling everywhere: next to the edge of an emitting wall its 1/d² term has infinite variance (the 4A furnace: the standard error did not fall with more samples). Area sampling stays as a control setting (`emitter_area_sampling`).
  - Zero, with no ray traced, when x is behind or in the emitter's plane or cos θ_x ≤ 0.
  - Visibility: a segment from x to y + n_y / 256 (the ADR's secondary-ray offset, on the emitter's side), so the emitter's own face is never an occluder.
  - Random numbers: drawn at a vertex only when emitter sampling is on at that vertex, after the sun's two numbers and before the continuation's. With emitters off, every M3 stream and image is unchanged.
- **Where emission is counted.** Emitted radiance is added **only** where the primary ray hits (the camera sees the emitter) and through next-event estimation. A continuation ray that hits an emitter does not add its emission, so nothing is counted twice and no MIS is needed.
- **Components** (reference `Settings`): `emission` (L_e at the primary hit), `emitters_direct` (next-event estimation at the primary vertex), `emitters_indirect` (at vertices 1..=`max_bounces`). All three are off by default, so every M3 reference and test is unchanged.
- **Real time (4B):** emission at the primary hit is deterministic: it is added after reconstruction, not demodulated or filtered; the filter's guides stay the surface's own.
- **Validation (4A record):** an emissive rectangle over a plane against Lambert's closed-form polygon irradiance; the emissive furnace L_e Σ a^k; GPU against CPU per pixel; the alias table against its realized probabilities (χ²); the planted faults (area PDF without the solid-angle conversion, emission counted twice, a missing emitter cosine) must fail.
- **Implementation status (2026-09-25):** CPU and GPU references implemented; C1–C5 pass (cloud session) and G1–G5 pass on the RTX 3050, with the M3 GPU regressions unchanged (4A record). The real-time path has no emitters yet (4B).
