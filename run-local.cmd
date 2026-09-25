@echo off
rem One-click local run for checks the cloud session cannot do (docs/CLOUD.md).
rem Current task: Phase 4A part 1 (emitters in the reference), docs/changes/2026-09-25-phase4a-emitters.md.
rem Double-click it after pulling. It only builds and tests: no installs, no source edits, no git.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0engine" || (echo Cannot find the engine folder next to this script. & pause & exit /b 1)

where cargo >nul 2>nul || (echo cargo is not on PATH: install Rust with rustup first. & pause & exit /b 1)
if not defined VULKAN_SDK (echo VULKAN_SDK is not set: install the pinned Vulkan SDK 1.4.357.0 ^(ADR-0001^). & pause & exit /b 1)
if not exist "%VULKAN_SDK%\Bin\slangc.exe" (echo slangc.exe is missing from %VULKAN_SDK%\Bin. & pause & exit /b 1)

set STAMP=run
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"`) do set STAMP=%%i
set OUT=results\local-run\%STAMP%-4a
if not exist "%OUT%" mkdir "%OUT%"
set SUMMARY=%OUT%\summary.txt

> "%SUMMARY%" echo Phase 4A part 1, local run %STAMP%
for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do >> "%SUMMARY%" echo commit %%i
for /f "delims=" %%i in ('cargo --version') do >> "%SUMMARY%" echo %%i
>> "%SUMMARY%" echo VULKAN_SDK %VULKAN_SDK%
>> "%SUMMARY%" echo.

echo Running the 4A checks. This takes about 15-20 minutes; keep the laptop plugged in.
echo.
call :step pure_suite "cargo test --release -j 2"
call :step gpu_lib "cargo test --release -j 2 -p gpu --lib"
call :step gpu_emitters_4a "cargo test --release -j 2 -p gpu --test emitters -- --test-threads=1 --nocapture"
call :step gpu_reference_3a "cargo test --release -j 2 -p gpu --test reference -- --test-threads=1 --nocapture"
call :step gpu_shade_3b "cargo test --release -j 2 -p gpu --test shade -- --test-threads=1 --nocapture"
call :step gpu_sky_3c "cargo test --release -j 2 -p gpu --test sky -- --test-threads=1 --nocapture"
call :step gpu_temporal_3d "cargo test --release -j 2 -p gpu --test temporal -- --test-threads=1 --nocapture"
call :step gpu_bounce_3f "cargo test --release -j 2 -p gpu --test bounce -- --test-threads=1 --nocapture"
>> "%SUMMARY%" echo.
>> "%SUMMARY%" echo NOT RUN on purpose: gpu gate (35+ min; unchanged frame path), gpu denoise (known accepted failures; does not use the reference).

echo.
type "%SUMMARY%"
echo.
echo Done. Results: engine\%OUT%
echo Send back summary.txt (paste it), or commit and push that folder.
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
