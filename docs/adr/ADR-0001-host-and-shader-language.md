# ADR 0001 — Rust host, Vulkan backend, Slang shaders

Status: accepted
Date: 2026-09-22
Decision authority: user, "Confirm rust" (host language). The user delegated the shader-language choice to the model ("Shader language i will let you choose"); Slang was chosen and stated before confirmation.
Related change/evidence: [Phase 0 prep](../changes/2026-09-22-phase0-prep.md), [phase0-probes](../../phase0-probes/README.md)

## Context and actual constraint

The proposal (A-002) named Rust + Vulkan + a candidate shader toolchain without evidence. Phase 0 probes implemented the same headless RT path in C++ and Rust, with GLSL and Slang, on the target laptop (RTX 3050 Laptop, 4 GB, driver Vulkan 1.4.351).

Measured:
- Images are bit-identical across language × shader compiler for a 26-triangle scene and for Bistro exterior (2.83 M triangles).
- Validation is clean, and a negative control proves the counter can fail.
- GPU AS build and trace times are identical.
- Host differences are ≤ ~5% and attributable to C compiler flags, not the language.

Not measured: large-codebase iteration/compile times, SDK integration effort, cold pipeline compile.

## Options and tradeoffs

- **C++ host.** Direct use of C++-only SDKs: DLSS/RR/Streamline, the NRC SDK, NRD, Nsight Aftermath. Direct use of the Slang runtime API. Most reference code is C++. No compiler-enforced ownership and lifetime checks.
- **Rust host (chosen).**
  - The compiler enforces the ownership/lifetime rules that CLAUDE.md's invariants depend on: versioned jobs, allocation generations, last-reader retirement.
  - The user has prior Rust engine experience.
  - Cargo pins dependencies.
  - C code links easily (ufbx proven).
  - Cost: C++ SDKs need FFI wrappers. All of them are optional or come last in the product contract.
- **GLSL.** Mature. Does not match the HLSL-family reference code (RTXDI, NRC). No modules or generics.
- **Slang (chosen).**
  - HLSL-like syntax.
  - Modules and generics help keep host/shader layouts in agreement.
  - Automatic differentiation for the optional neural tier.
  - Bit-identical to GLSL in both probes.
  - Risk: toolchain churn.

## Decision

- Host language: **Rust** (edition 2021, stable toolchain, MSVC target on Windows).
- Graphics API: **Vulkan** through `ash`, with no higher-level abstraction layer.
- Shader language: **Slang**, compiled **offline** by a pinned `slangc` to SPIR-V. Reflection comes from `slangc` output, not the runtime API, unless a later ADR changes that.
- GLSL remains only as a frozen control in `phase0-probes` and is not part of the engine.

This decides the language and toolchain only. It does not accept the rest of A-001/A-003 or any engine layout.

## Contracts and consequences

- Pins, until a recorded upgrade:
  - Vulkan SDK 1.4.357.0, whose `slangc` is 2026.13.1
  - `ash` 0.38.0
  - Rust stable 1.98.1
  - Crates proven in Phase 0: `ufbx` 0.11.4, `winit` 0.30.13, `ash-window` 0.13.0, `libloading` 0.8.9
  - Any other crate an authorized task adds must also be pinned exactly.
  - Capture tool: RenderDoc 1.46 (portable).
- Every shader build validates its SPIR-V (`spirv-val`) and runs Vulkan validation in debug configurations.
- C++-only SDKs, if they are ever adopted, need an FFI wrapper with its own change record. DLSS/RR remain last, as optional paths compared against native reconstruction.
- Accepted risk: Slang toolchain regressions. Mitigation: pinning, and comparison against a GLSL control when a compiler bug is suspected.

## Validation and revisiting

- Evidence: see the related change record. Remaining Phase 0 gates (windowed cold-compile responsiveness, texture/alpha path, RenderDoc capture/replay) passed on 2026-09-22 with the Rust + Slang stack; hardware-counter profiling (Nsight) is not run.
- Reopen if:
  - Slang cannot express a required feature, or produces unfixable miscompiles.
  - A required C++ SDK proves impractical to wrap.
  - Rust iteration speed becomes a measured blocker.

## Implementation status

Not implemented (no engine code). The probes exercise the chosen stack.

## Supersession or errata

None.
