@echo off
rem One-click local run for checks the cloud session cannot do (docs/CLOUD.md).
rem Current task: Phase 4B parts 1 and 2 (the lights in the real-time path; relight for lights),
rem docs/changes/2026-09-26-phase4b-many-lights.md: G10-G14, Q1-Q4, M1-M4, V; R3-R5, E4.
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
set VLOG=%OUT%\viewer_v.jsonl
set M4LOG=%OUT%\viewer_m4.jsonl
set E4LOG=%OUT%\viewer_e4.jsonl
rem The 1080p references and converged images are cached here, so a rerun skips them (not pushed).
set NE_4B_DIR=%TEMP%\ne_4b
set NIGHT=cargo test --release -j 2 -p gpu --test night -- --test-threads=1 --nocapture

> "%SUMMARY%" echo Phase 4B parts 1 and 2, local run %STAMP%
for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do >> "%SUMMARY%" echo commit %%i
for /f "delims=" %%i in ('cargo --version') do >> "%SUMMARY%" echo %%i
>> "%SUMMARY%" echo VULKAN_SDK %VULKAN_SDK%
>> "%SUMMARY%" echo NE_4B_DIR %NE_4B_DIR%
>> "%SUMMARY%" echo.

echo Running the 4B part 1 and part 2 checks. This takes about 2-2.5 hours; keep the laptop plugged in.
echo Viewer windows open after the first tests (about 20-40 minutes in) and fly or walk by themselves:
echo do not touch the mouse or keyboard until this window says Done.
echo.
call :step build "cargo build --release -j 2 -p viewer -p gpu --bins --tests"
call :step pure_suite "cargo test --release -j 2"
call :step gpu_lib "cargo test --release -j 2 -p gpu --lib"

rem G10, G11, G13, G14.
call :step g10_exact "%NIGHT% lit_shade_equals_the_reference_sample_per_pixel"
call :step g11_convergence "%NIGHT% lit_shade_converges_to_the_reference"
call :step g13_emission_light_jump "%NIGHT% emission_after_reconstruction_and_the_light_jump"
call :step g14_meter "%NIGHT% meter_matches_the_host"

rem G12: lights off is M3 (the recorded numbers). denoise ends in its two accepted failures (C1, D4):
rem that step shows FAIL by design; its numbers are what is checked.
call :step g12_shade "cargo test --release -j 2 -p gpu --test shade -- --test-threads=1 --nocapture"
call :step g12_sky "cargo test --release -j 2 -p gpu --test sky -- --test-threads=1 --nocapture"
call :step g12_bounce "cargo test --release -j 2 -p gpu --test bounce -- --test-threads=1 --nocapture"
call :step g12_temporal "cargo test --release -j 2 -p gpu --test temporal -- --test-threads=1 --nocapture"
call :step g12_denoise "cargo test --release -j 2 -p gpu --test denoise -- --test-threads=1 --nocapture"
call :step g12_emitters "cargo test --release -j 2 -p gpu --test emitters -- --test-threads=1 --nocapture"
call :step g12_edit "cargo test --release -j 2 -p gpu --test edit -- --test-threads=1 --nocapture"

rem Part 2. R3, R4: relight for lights. R5: 3E R1/R2 (inside g12_denoise, both with and without the
rem bounce), temporal and denoise as in G12, and the pixel-exact raster views.
call :step r3_r4_relight_lights "%NIGHT% edits_relight_what_they_change_of_the_lights"
call :step r5_raster "cargo test --release -j 2 -p gpu --test raster -- --test-threads=1 --nocapture"

rem V: viewer runs with validation on.
call :step v_night21 "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 600 --log %VLOG%"
call :step v_sunset "target\release\viewer.exe --scene night --hour 17.9 --run-day --size 1920x1080 --frames 900 --log %VLOG%"
call :step v_lights_off21 "target\release\viewer.exe --scene night --hour 21 --lights off --size 1920x1080 --frames 300 --log %VLOG%"
call :step v_lights_on12 "target\release\viewer.exe --scene night --hour 12 --lights on --size 1920x1080 --frames 300 --log %VLOG%"
call :step v_spp2 "target\release\viewer.exe --scene night --hour 21 --emitter-spp 2 --size 1920x1080 --frames 300 --log %VLOG%"
call :step v_spp4 "target\release\viewer.exe --scene night --hour 21 --emitter-spp 4 --size 1920x1080 --frames 300 --log %VLOG%"
rem E4 with validation on (400 frames, N = 1, 8, 32).
call :step e4_valid_n1 "target\release\viewer.exe --scene night --hour 21 --edit-script --edit-size 1 --size 1920x1080 --frames 400 --log %VLOG%"
call :step e4_valid_n8 "target\release\viewer.exe --scene night --hour 21 --edit-script --edit-size 8 --size 1920x1080 --frames 400 --log %VLOG%"
call :step e4_valid_n32 "target\release\viewer.exe --scene night --hour 21 --edit-script --edit-size 32 --size 1920x1080 --frames 400 --log %VLOG%"
call :step v_street21 "target\release\viewer.exe --scene street --hour 21 --size 1920x1080 --frames 300 --log %VLOG%"

rem M4: frame cost in the viewer, validation off (data).
set NE_NO_VALIDATION=1
call :step m4_night_1775 "target\release\viewer.exe --scene night --hour 17.75 --walk --size 1920x1080 --frames 3000 --log %M4LOG%"
call :step m4_night_185 "target\release\viewer.exe --scene night --hour 18.5 --walk --size 1920x1080 --frames 3000 --log %M4LOG%"
call :step m4_night_21 "target\release\viewer.exe --scene night --hour 21 --walk --size 1920x1080 --frames 3000 --log %M4LOG%"
call :step m4_street_21 "target\release\viewer.exe --scene street --hour 21 --walk --size 1920x1080 --frames 3000 --log %M4LOG%"
rem E4: the edit budget with the lights on (2,000 frames, N = 1, 8, 32).
call :step e4_n1 "target\release\viewer.exe --scene night --hour 21 --edit-script --edit-size 1 --size 1920x1080 --frames 2000 --log %E4LOG%"
call :step e4_n8 "target\release\viewer.exe --scene night --hour 21 --edit-script --edit-size 8 --size 1920x1080 --frames 2000 --log %E4LOG%"
call :step e4_n32 "target\release\viewer.exe --scene night --hour 21 --edit-script --edit-size 32 --size 1920x1080 --frames 2000 --log %E4LOG%"
rem M1 time and the compose cost (timing run).
call :step m1_cost "%NIGHT% equal_time_curve_cost"
set NE_NO_VALIDATION=

rem Q1-Q4 and M2 (references and converged images cached in NE_4B_DIR), M3 (b), M1 error.
call :step q_stills "%NIGHT% stills_at_blue_hour_and_night"
call :step q3_motion "%NIGHT% motion_at_night"
call :step m3b_indirect_noise "%NIGHT% indirect_noise_at_night"
call :step m1_error "%NIGHT% equal_time_curve_error"

rem M3 (a): one-bounce night references beside 4A's eight-bounce ones.
call :step m3a_ref_one_bounce "cargo run --release -j 2 -p gpu --bin ref_light -- results\phase4b\ref1 --scene night --bounces 1 --spp 16384 --times blue_hour,night"

rem Q4 and M3 (a): FLIP (needs the Python with flip-evaluator 1.7, A-008).
where python >nul 2>nul
if errorlevel 1 (
    >> "%SUMMARY%" echo NOT RUN  flip  ^(python is not on PATH^)
) else (
    call :step flip "python results\phase4b\flip.py %NE_4B_DIR% results\phase4b results\phase4a_ref results\phase4b\ref1"
)
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo Expected: g12_denoise FAIL by design ^(accepted C1 and D4^); its numbers are compared with the 3E/3G logs.
>> "%SUMMARY%" echo NOT RUN on purpose: gpu gate ^(35+ min; with the lights off the frame path is M3's byte for byte, C10^), gpu reference, ray, device ^(unchanged^).

echo.
type "%SUMMARY%"
echo.
echo Done. Results: engine\%OUT% and engine\results\phase4b
echo Send them back: commit and push the engine\results folder (both folders are inside it).
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
