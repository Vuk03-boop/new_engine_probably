@echo off
rem One-click local run for checks the cloud session cannot do (docs/CLOUD.md).
rem Current task: the 4B filter fix (the filter keeps night energy, G4),
rem docs/changes/2026-09-26-4b-filter-energy.md, criteria F1-F6.
rem Double-click it after pulling. It only builds, tests and renders: no installs, no source edits, no git.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0engine" || (echo Cannot find the engine folder next to this script. & pause & exit /b 1)

where cargo >nul 2>nul || (echo cargo is not on PATH: install Rust with rustup first. & pause & exit /b 1)
if not defined VULKAN_SDK (echo VULKAN_SDK is not set: install the pinned Vulkan SDK 1.4.357.0 ^(ADR-0001^). & pause & exit /b 1)
if not exist "%VULKAN_SDK%\Bin\slangc.exe" (echo slangc.exe is missing from %VULKAN_SDK%\Bin. & pause & exit /b 1)

rem The candidate filter (frozen in the record) and the plateau arms around it (data), with the
rem image-name prefixes the tests write (':' becomes '-').
set CAND=conservative:4
set CANDP=conservative-4_
set HALF=conservative:2
set HALFP=conservative-2_
set DOUBLE=conservative:8
set DOUBLEP=conservative-8_

set STAMP=run
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"`) do set STAMP=%%i
set OUT=results\local-run\%STAMP%-4b-filter
if not exist "%OUT%" mkdir "%OUT%"
set FLIPS=results\phase4b_filter
if not exist "%FLIPS%" mkdir "%FLIPS%"
set SUMMARY=%OUT%\summary.txt
set WALKS=%OUT%\viewer_walks.jsonl
set EDITS=%OUT%\viewer_edits.jsonl
rem References stay outside the repository: 4B's night ones and 3G's day ones (rebuilt if missing).
set GATE4B=%TEMP%\ne_gate_4b
set GATE3G=%TEMP%\ne_gate
if not exist "%GATE4B%" mkdir "%GATE4B%"
if not exist "%GATE3G%" mkdir "%GATE3G%"

> "%SUMMARY%" echo 4B filter fix, local run %STAMP%
for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do >> "%SUMMARY%" echo commit %%i
for /f "delims=" %%i in ('cargo --version') do >> "%SUMMARY%" echo %%i
>> "%SUMMARY%" echo VULKAN_SDK %VULKAN_SDK%
>> "%SUMMARY%" echo candidate %CAND% (plateau data: %HALF%, %DOUBLE%)
>> "%SUMMARY%" echo reference caches %GATE4B% and %GATE3G%
>> "%SUMMARY%" echo.

echo Running the 4B filter checks. This takes about 1.5 to 2 hours (45 minutes more if the
echo day references are no longer in %GATE3G%); keep the laptop plugged in, the lid open and
echo sleep off. Viewer windows will open and move by themselves: do not touch the mouse or
echo keyboard until this window says Done.
echo.
rem Quick checks first.
call :step pure_suite "cargo test --release -j 2"
call :step gpu_lib "cargo test --release -j 2 -p gpu --lib"
call :step gpu_build_tests "cargo build --release -j 2 -p gpu --tests"
call :step model_c2 "cargo test --release -j 2 -p gpu --test denoise_model -- --nocapture"
call :step viewer_build "cargo build --release -j 2 -p viewer"

rem F2: energy of every arm at night and dusk; the default must reproduce 4B's numbers.
set NE_GATE_DIR=%GATE4B%
call :step f2_energy "cargo test --release -j 2 -p gpu --test lights -- --ignored --test-threads=1 --nocapture diagnostic_g4_filter_energy"

rem F1: 4B's G4 at night with the candidate, then the plateau arms (data).
set NE_FILTER=%CAND%
call :step f1_g4_candidate "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture night_against_the_reference"
set NE_FILTER=%HALF%
call :step f1_g4_half_data "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture night_against_the_reference"
set NE_FILTER=%DOUBLE%
call :step f1_g4_double_data "cargo test --release -j 2 -p gpu --test lights -- --test-threads=1 --nocapture night_against_the_reference"
set NE_FILTER=

rem F3: M3's Q1-Q4 at day with the candidate (3G's references).
set NE_GATE_DIR=%GATE3G%
set NE_FILTER=%CAND%
call :step f3_gate_stills "cargo test --release -j 2 -p gpu --test gate -- --test-threads=1 --nocapture stills_against_the_reference"
call :step f3_gate_motion "cargo test --release -j 2 -p gpu --test gate -- --test-threads=1 --nocapture motion_against_the_reference"
set NE_GATE_DIR=

rem F4: 3E's criteria with the candidate; D4 and C1 are data (accepted failures with the default).
call :step f4_3e_criteria "cargo test --release -j 2 -p gpu --test denoise -- --test-threads=1 --nocapture stills_meet_the_budget edits_relight_what_they_change a_moving_sun_does_not_lag"
call :step f4_3e_d4_c1_data "cargo test --release -j 2 -p gpu --test denoise -- --test-threads=1 --nocapture motion_path_meets_the_budget denoise_cost_at_1080p"

rem F5: the filter's cost against the default, interleaved, validation off.
set NE_NO_VALIDATION=1
call :step f5_cost "cargo test --release -j 2 -p gpu --test denoise -- --ignored --test-threads=1 --nocapture filter_cost_against_the_default"
set NE_FILTER=

rem F6 data: the night walk with the default and with the candidate, validation off.
call :step f6_walk_default_data "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 3000 --walk --log %WALKS%"
call :step f6_walk_candidate_data "target\release\viewer.exe --scene night --hour 21 --size 1920x1080 --frames 3000 --walk --filter %CAND% --log %WALKS%"
set NE_NO_VALIDATION=

rem F6: the viewer with the candidate, validation on.
call :step f6_viewer_night "target\release\viewer.exe --scene night --hour 17.5 --run-day --size 1920x1080 --frames 400 --edit-script --edit-size 1 --filter %CAND% --log %EDITS%"
call :step f6_viewer_street "target\release\viewer.exe --scene street --size 1920x1080 --frames 400 --filter %CAND% --log %EDITS%"

rem F1 Q4 and F3 Q4: FLIP of the display images (needs Python with flip-evaluator 1.7, A-008).
where python >nul 2>nul
if errorlevel 1 (
    >> "%SUMMARY%" echo NOT RUN  f1_flip f3_flip  python is not on PATH
) else (
    call :step f1_flip_candidate "python results\phase4b\flip.py %GATE4B% %FLIPS%\g4_candidate %CANDP%"
    call :step f1_flip_half_data "python results\phase4b\flip.py %GATE4B% %FLIPS%\g4_half %HALFP%"
    call :step f1_flip_double_data "python results\phase4b\flip.py %GATE4B% %FLIPS%\g4_double %DOUBLEP%"
    call :step f3_flip "python results\phase3g\flip.py %GATE3G% %FLIPS%\gate %CANDP%"
)
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo Data steps (a FAIL there is not a criterion): f1_g4_half_data, f1_g4_double_data, f4_3e_d4_c1_data (D4 and C1 fail with the default too), f6_walk_*_data, f1_flip_*_data.
>> "%SUMMARY%" echo NOT RUN on purpose: the other lights/gpu tests (the filter is the only changed engine code; F2 checks the default is unchanged).

echo.
type "%SUMMARY%"
echo.
echo Done. Results: engine\%OUT% and engine\%FLIPS%
echo Send them back: commit and push the engine\results folder (both folders are inside it).
echo The references in %GATE4B% and %GATE3G% are large and stay on this laptop.
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
