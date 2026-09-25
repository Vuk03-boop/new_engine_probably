@echo off
rem One-click local run for checks the cloud session cannot do (docs/CLOUD.md).
rem Current task: Phase 4A part 2 (the emitter table in GpuScene, --scene night, night references),
rem docs/changes/2026-09-25-phase4a-emitters.md, criteria G6-G9.
rem Double-click it after pulling. It only builds, tests and renders: no installs, no source edits, no git.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0engine" || (echo Cannot find the engine folder next to this script. & pause & exit /b 1)

where cargo >nul 2>nul || (echo cargo is not on PATH: install Rust with rustup first. & pause & exit /b 1)
if not defined VULKAN_SDK (echo VULKAN_SDK is not set: install the pinned Vulkan SDK 1.4.357.0 ^(ADR-0001^). & pause & exit /b 1)
if not exist "%VULKAN_SDK%\Bin\slangc.exe" (echo slangc.exe is missing from %VULKAN_SDK%\Bin. & pause & exit /b 1)

set STAMP=run
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"`) do set STAMP=%%i
set OUT=results\local-run\%STAMP%-4a2
if not exist "%OUT%" mkdir "%OUT%"
set SUMMARY=%OUT%\summary.txt
set EDITS=%OUT%\viewer_edits.jsonl

> "%SUMMARY%" echo Phase 4A part 2, local run %STAMP%
for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do >> "%SUMMARY%" echo commit %%i
for /f "delims=" %%i in ('cargo --version') do >> "%SUMMARY%" echo %%i
>> "%SUMMARY%" echo VULKAN_SDK %VULKAN_SDK%
>> "%SUMMARY%" echo.

echo Running the 4A part 2 checks. This takes about 45-60 minutes; keep the laptop plugged in.
echo Viewer windows will open and fly down the street by themselves: do not touch the mouse or
echo keyboard until this window says Done. The last step renders 6 reference images (about 25 min).
echo.
call :step pure_suite "cargo test --release -j 2"
call :step gpu_lib "cargo test --release -j 2 -p gpu --lib"
call :step gpu_emitters_4a "cargo test --release -j 2 -p gpu --test emitters -- --test-threads=1 --nocapture"
call :step gpu_edit_2e "cargo test --release -j 2 -p gpu --test edit -- --test-threads=1 --nocapture"
call :step gpu_temporal_3d "cargo test --release -j 2 -p gpu --test temporal -- --test-threads=1 --nocapture"
call :step viewer_build "cargo build --release -j 2 -p viewer"

rem G7 and G9: edit latency in the viewer, validation off (as the 3G edit gate), then on.
set NE_NO_VALIDATION=1
call :step viewer_night_full_edit1 "target\release\viewer.exe --scene night --size 1920x1080 --frames 2000 --edit-script --edit-size 1 --log %EDITS%"
call :step viewer_night_full_edit8 "target\release\viewer.exe --scene night --size 1920x1080 --frames 2000 --edit-script --edit-size 8 --log %EDITS%"
call :step viewer_night_full_edit32 "target\release\viewer.exe --scene night --size 1920x1080 --frames 2000 --edit-script --edit-size 32 --log %EDITS%"
call :step viewer_night_dense_edit1 "target\release\viewer.exe --scene night --dressing dense --size 1920x1080 --frames 2000 --edit-script --edit-size 1 --log %EDITS%"
call :step viewer_street_edit1 "target\release\viewer.exe --scene street --size 1920x1080 --frames 2000 --edit-script --edit-size 1 --log %EDITS%"
set NE_NO_VALIDATION=
call :step viewer_night_full_validation1 "target\release\viewer.exe --scene night --size 1920x1080 --frames 400 --edit-script --edit-size 1 --log %EDITS%"
call :step viewer_night_full_validation8 "target\release\viewer.exe --scene night --size 1920x1080 --frames 400 --edit-script --edit-size 8 --log %EDITS%"
call :step viewer_night_full_validation32 "target\release\viewer.exe --scene night --size 1920x1080 --frames 400 --edit-script --edit-size 32 --log %EDITS%"

rem G8: the night references (2 cameras x dusk, blue hour, night), cached in results\phase4a_ref.
call :step ref_light_night "cargo run --release -j 2 -p gpu --bin ref_light -- results\phase4a_ref --scene night --spp 16384"
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo NOT RUN on purpose: gpu gate (35+ min; the M3 frame path and scene build did not change), gpu denoise (known accepted failures), gpu reference, shade, sky, bounce (code unchanged since part 1's run).

echo.
type "%SUMMARY%"
echo.
echo Done. Results: engine\%OUT% and engine\results\phase4a_ref
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
