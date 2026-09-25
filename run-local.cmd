@echo off
rem One-click local run for checks the cloud session cannot do (docs/CLOUD.md).
rem Current task: Phase 4B (many lights without reuse: emitters in the real-time path, the lights
rem switch, the night exposure, emitter relight rules), docs/changes/2026-09-25-phase4b-many-lights.md,
rem criteria G1-G8 and measurements M1-M2.
rem Double-click it after pulling. It only builds, tests and renders: no installs, no source edits, no git.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0engine" || (echo Cannot find the engine folder next to this script. & pause & exit /b 1)

where cargo >nul 2>nul || (echo cargo is not on PATH: install Rust with rustup first. & pause & exit /b 1)
if not defined VULKAN_SDK (echo VULKAN_SDK is not set: install the pinned Vulkan SDK 1.4.357.0 ^(ADR-0001^). & pause & exit /b 1)
if not exist "%VULKAN_SDK%\Bin\slangc.exe" (echo slangc.exe is missing from %VULKAN_SDK%\Bin. & pause & exit /b 1)

set STAMP=run
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"`) do set STAMP=%%i
set OUT=results\local-run\%STAMP%-4b
if not exist "%OUT%" mkdir "%OUT%"
set SUMMARY=%OUT%\summary.txt
set EDITS=%OUT%\viewer_edits.jsonl
set WALKS=%OUT%\viewer_walks.jsonl
rem G4's references (1080p, about 100 MB each) are cached outside the repository.
set GATE=%TEMP%\ne_gate_4b
if not exist "%GATE%" mkdir "%GATE%"

> "%SUMMARY%" echo Phase 4B, local run %STAMP%
for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do >> "%SUMMARY%" echo commit %%i
for /f "delims=" %%i in ('cargo --version') do >> "%SUMMARY%" echo %%i
>> "%SUMMARY%" echo VULKAN_SDK %VULKAN_SDK%
>> "%SUMMARY%" echo reference cache %GATE%
>> "%SUMMARY%" echo.

echo Running the 4B checks. This takes about 90 minutes; keep the laptop plugged in.
echo Viewer windows will open and move by themselves: do not touch the mouse or keyboard
echo until this window says Done.
echo.
rem Quick checks first.
call :step pure_suite "cargo test --release -j 2"
call :step gpu_lib "cargo test --release -j 2 -p gpu --lib"
call :step gpu_build_tests "cargo build --release -j 2 -p gpu --tests"
call :step viewer_build "cargo build --release -j 2 -p viewer"

rem 4B GPU criteria (validation on): G1 exact per pixel, G2 convergence, G3 emission in the light
rem view, G4 night quality (references cached in %GATE%), G5 relight with emitters, G6 switch and exposure.
set NE_GATE_DIR=%GATE%
call :step g1_exact "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture emitters_equal_the_reference_sample_per_pixel"
call :step g2_convergence "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture emitters_converge_to_the_reference"
call :step g3_light_view "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture the_light_view_adds_emission"
call :step g4_night_quality "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture night_against_the_reference"
call :step g5_relight "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture edits_relight_emitter_light"
call :step g6_switch_exposure "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture lights_switch_and_exposure_sums"
set NE_GATE_DIR=

rem G4 Q4: FLIP of the display images (needs Python with flip-evaluator 1.7, A-008).
where python >nul 2>nul
if errorlevel 1 (
    >> "%SUMMARY%" echo NOT RUN  g4_flip  python is not on PATH
) else (
    call :step g4_flip "python results\phase4b\flip.py %GATE% results\phase4b"
)

rem M1: the equal-time curve, validation off.
set NE_NO_VALIDATION=1
call :step m1_equal_time_curve "cargo test --release -j 2 -p gpu --test lights -- --ignored --test-threads=1 --nocapture equal_time_curve"

rem G7: edits with the lights on in the viewer (night at 21 h), validation off as the 3G edit gate.
call :step viewer_night_edit1 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 2000 --edit-script --edit-size 1 --log %EDITS%"
call :step viewer_night_edit8 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 2000 --edit-script --edit-size 8 --log %EDITS%"
call :step viewer_night_edit32 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 2000 --edit-script --edit-size 32 --log %EDITS%"
rem G8: the M3 street's edit run (no lights).
call :step viewer_street_edit1 "target\release\viewer.exe --scene street --size 1920x1080 --frames 2000 --edit-script --edit-size 1 --log %EDITS%"
rem M2: the night frame cost on the looping walk (data).
call :step viewer_night_walk_full "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 3000 --walk --log %WALKS%"
call :step viewer_night_walk_dense "target\release\viewer.exe --scene night --dressing dense --hour 21 --size 1920x1080 --frames 3000 --walk --log %WALKS%"
set NE_NO_VALIDATION=

rem G7 with validation on: edits of every size, and the day running from 17.5 h across the switch.
call :step viewer_night_validation1 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 400 --edit-script --edit-size 1 --log %EDITS%"
call :step viewer_night_validation8 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 400 --edit-script --edit-size 8 --log %EDITS%"
call :step viewer_night_validation32 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 400 --edit-script --edit-size 32 --log %EDITS%"
call :step viewer_night_validation_sunset "target\release\viewer.exe --scene night --hour 17.5 --run-day --size 1920x1080 --frames 400 --edit-script --edit-size 1 --log %EDITS%"

rem G8: regressions of the code 4B touched (shade, temporal relight rows, the reference's shared
rem emitter function, the debug view).
call :step gpu_emitters_4a "cargo test --release -j 2 -p gpu --test emitters -- --test-threads=1 --nocapture"
call :step gpu_shade_3b "cargo test --release -j 2 -p gpu --test shade -- --test-threads=1 --nocapture"
call :step gpu_sky_3c "cargo test --release -j 2 -p gpu --test sky -- --test-threads=1 --nocapture"
call :step gpu_bounce_3f "cargo test --release -j 2 -p gpu --test bounce -- --test-threads=1 --nocapture"
call :step gpu_temporal_3d "cargo test --release -j 2 -p gpu --test temporal -- --test-threads=1 --nocapture"
call :step gpu_edit_2e "cargo test --release -j 2 -p gpu --test edit -- --test-threads=1 --nocapture"
call :step gpu_denoise_relight_3e "cargo test --release -j 2 -p gpu --test denoise -- --test-threads=1 --nocapture edits_relight_what_they_change"
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo NOT RUN on purpose: gpu gate (35+ min; M3's frame path draws no new random numbers with the lights off, G1 checks it bit for bit), the rest of gpu denoise (known accepted failures C1/D4), gpu reference and raster/ray/device (unchanged).

echo.
type "%SUMMARY%"
echo.
echo Done. Results: engine\%OUT% and engine\results\phase4b
echo Send them back: commit and push the engine\results folder (both folders are inside it).
echo The references in %GATE% are large and stay on this laptop.
pause
exit /b 0

:step
set NAME=%~1
echo === %NAME%
%~2 > "%OUT%\%NAME%.log" 2>&1
set CODE=!errorlevel!
if !CODE! equ 0 (set RESULT=PASS) else (set RESULT=FAIL)
>> "%SUMMARY%" echo !RESULT!  %NAME%  exit !CODE!  log %NAME%.log
echo     !RESULT! ^(exit !CODE!^)
exit /b 0
