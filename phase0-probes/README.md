# Phase 0 probes — throwaway, not engine code

Probes that settled the Phase 0 toolchain questions on this laptop. **Accepted stack: Rust + Vulkan (`ash`) + Slang** ([ADR-0001](../docs/adr/ADR-0001-host-and-shader-language.md)). The C++ probe and the GLSL shaders are frozen controls at probe v2; new work is Rust + Slang only.

Record of results: [../docs/changes/2026-09-22-phase0-prep.md](../docs/changes/2026-09-22-phase0-prep.md).

## What the probes do

- **`rt_probe`** (headless; Rust, plus a frozen C++ v2 twin):
  1. Selects an RT-capable GPU by extensions and enables Khronos validation.
  2. Loads a built-in 26-triangle scene or an FBX file. FBX goes through ufbx and is flattened to world-space meters, right-handed Y-up; the FBX camera and first directional light are used.
  3. Uploads geometry through staging buffers.
  4. Builds a BLAS (compaction allowed, compacted size queried) and a TLAS.
  5. Creates an RT pipeline from pre-compiled SPIR-V.
  6. Traces 1280×720 once cold and 5× warm, then prints one JSON line with timings, memory (`VK_EXT_memory_budget`) and validation counts.

  Shading is minimal: geometric normal, one sun with a shadow ray, 0.15 ambient, sky gradient.

  **v3 `--textures`** (Rust + `shaders/probe_tex.slang` only) adds:
  - all FBX textures uploaded as BC images with full mip chains, sampled bindlessly at mip 0
  - per-corner UVs
  - alpha testing through any-hit shaders on a separate non-opaque BLAS geometry (primary and shadow rays), with sRGB output
- **`window_probe`** (Rust): presents an animated clear every loop iteration (FIFO) and compiles the RT pipeline either on the main thread (control) or on a background thread. A unique specialization constant defeats the driver cache. It prints frame-interval statistics before, during and after the compile.

## Layout

| Path | Contents |
|---|---|
| `shaders/probe.slang` | v2 shading (Slang); has a `kCompileSeed` spec constant (default 0, no effect) |
| `shaders/probe_tex.slang` | v3 textured shading: bindless base colour, alpha-test any-hit, sRGB output |
| `shaders/glsl/` | Frozen v2 GLSL control |
| `shaders/build_shaders.bat` | Compiles everything to `shaders/out/{glsl,slang,slang_tex}/` and runs `spirv-val` |
| `rust/` | Rust 2021 probes. Pinned: `ash =0.38.0`, `ufbx =0.11.4`, `winit =0.30.13`, `ash-window =0.13.0`, `libloading =0.8.9` |
| `cpp/` | Frozen C++20 v2 probe (CMake + Ninja), with the C++-only `PROBE_PICK` diagnostic |
| `third_party/ufbx/` | ufbx v0.23.0 (MIT or public domain), byte-identical to the Rust crate's copy |
| `tools/RenderDoc_1.46_64/` | Portable RenderDoc 1.46 (MIT), not installed system-wide |
| `tools/rdc_replay_check.py` | Opens a capture in `qrenderdoc` for local replay and lists recorded actions |
| `compare.py`, `dds_budget.py` | Pixel diff / PNG export; texture-memory estimate from DDS headers |
| `results/v2/`, `results/v3/` | Logs, images and captures (`results/v3/rdc/bistro_tex_capture.rdc` is ~1 GB) |
| `assets/Bistro_v5_2/` | Amazon Lumberyard Bistro, CC-BY 4.0 (attribution: Amazon Lumberyard, NVIDIA ORCA) |

## Verified commands (2026-09-22, this laptop)

Run from `phase0-probes/` in `cmd`. The Vulkan SDK must set `VULKAN_SDK`.

```bat
shaders\build_shaders.bat
cd rust && cargo build --release -j 2 && cd ..
rust\target\release\rt_probe.exe shaders/out/slang results/rust_small.ppm
rust\target\release\rt_probe.exe shaders/out/slang_tex results/rust_bistro_tex.ppm --fbx assets/Bistro_v5_2/BistroExterior.fbx --textures
rust\target\release\window_probe.exe shaders/out/slang --compile background --seed unique
```

RenderDoc capture of the trace submission, then a scripted replay check. The capture layer is enabled for the probe only: no registry change, and `endlocal` before `qrenderdoc`, because loading the capture layer into RenderDoc's own replay process crashes it (`IsCaptureMode` assertion).

```bat
setlocal
set VK_ADD_LAYER_PATH=%CD%\tools\RenderDoc_1.46_64
set VK_INSTANCE_LAYERS=VK_LAYER_RENDERDOC_Capture
set PROBE_NO_VALIDATION=1
set PROBE_RENDERDOC_CAPTURE=%CD%\results\capture
rust\target\release\rt_probe.exe shaders/out/slang results/rust_small_rdc.ppm
endlocal
set RDC_CAPTURE=%CD%\results\capture_capture.rdc
set RDC_REPORT=%CD%\results\capture_replay.txt
start "" /wait tools\RenderDoc_1.46_64\qrenderdoc.exe --python "%CD%\tools\rdc_replay_check.py"
```

Notes on the replay step:
- The report must contain `open_capture_ok=True` and `chunk_count vkCmdTraceRaysKHR=6`.
- Quote the script path because the project path contains spaces.
- `start /wait` makes `cmd` wait for the GUI program.
- `qrenderdoc` does not pass extra arguments to the script, so the paths go through `RDC_*` variables.

Nsight Graphics 2026.3.1 GPU Trace, from PowerShell (verified 2026-09-23). This needs GPU counter access for all users (NVIDIA Control Panel) and no other Nsight session open:

```powershell
$n="C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1\host\windows-desktop-nomad-x64\ngfx.exe"
$p=(Get-Location).Path
Start-Process $n -Wait -ArgumentList "--activity `"GPU Trace Profiler`" --exe `"$p\rust\target\release\rt_probe.exe`" --dir `"$p`" --args `"shaders/out/slang_tex results/nsight.ppm --fbx assets/Bistro_v5_2/BistroExterior.fbx --textures`" --env `"PROBE_NO_VALIDATION=1;PROBE_HOLD_MS=12000;`" --output-dir `"$p\results\nsight`" --start-after-ms 3000 --max-duration-ms 9999 --auto-export"
```

Notes on GPU Trace:
- Do not pass `--platform`: Qt treats it as its own display-plugin option and aborts.
- `--max-duration-ms` must be below 10000.
- The start timer counts from attach, not from launch.
- The trace window must cover the probe's submits, and the probe must stay alive until the window ends (`PROBE_HOLD_MS`), or the report is lost.
- Nsight locks clocks to base, so profiled timings are slower than normal runs.

`rt_probe` arguments: `<shader_dir> <out.ppm> [--size W H] [--fbx file.fbx] [--textures-out list.txt] [--textures]`. `window_probe` arguments: `<shader_dir> [--compile main|background] [--seed unique|fixed] [--delay-ms N] [--after-ms N]`. Exit codes: 0 ok, 1 Vulkan/setup failure, 2 usage, 3 validation errors reported (Rust panics exit 101).

Environment switches:
- `PROBE_NO_VALIDATION=1`: disables the Khronos validation layer.
- `PROBE_FAULT_LEAK=1`: negative control that leaks the pipeline; must exit 3.
- `PROBE_NO_ALPHA=1`: ablation that makes everything opaque, same textures.
- `PROBE_HOLD_MS=<ms>`: sleep after the work, before teardown. It is not timed, and exists so an external profiler can finish its trace.
- `PROBE_RENDERDOC_CAPTURE=<path template>`: capture the trace submission. The probe fails if the RenderDoc layer is not loaded.
- `PROBE_PICK="x,y;..."`: C++ v2 only; reports the CPU-picked hit node, material and distance per pixel.

The frozen C++ probe still builds with `cpp\build.bat` (VS Build Tools 2022 `vcvars64.bat`). Keep build directories on short paths (MSVC fails when object/PDB paths exceed ~250 characters).

## What this does not show

- Materials beyond base colour: normal, specular and emissive maps are uploaded but unused.
- Ray-cone mip selection (mip 0 only, so aliasing).
- Emissive lighting, sky or indirect light.
- Order-independent transparency: glass is alpha-cut, not blended.
- Per-dispatch hardware-counter analysis: the Nsight GPU Trace reports exist (see below), but they were only summarized from exported whole-window averages. Per-dispatch views need the Nsight GUI.
- A cold compile of every pipeline stage: only raygen carries the unique seed.
- Timings cover one camera on one scene and are not a general performance claim.
