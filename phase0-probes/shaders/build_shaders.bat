@echo off
rem Compiles the shader variants to SPIR-V and validates them.
rem Output: shaders\out\{glsl,slang}\{rgen,miss,shadow,chit}.spv and shaders\out\slang_tex\ (+ ahit, ahit_shadow).
setlocal
set B=%VULKAN_SDK%\Bin
set D=%~dp0
set G=%D%out\glsl
set S=%D%out\slang
set T=%D%out\slang_tex
if not exist "%G%" mkdir "%G%"
if not exist "%S%" mkdir "%S%"
if not exist "%T%" mkdir "%T%"
"%B%\glslangValidator.exe" -V --target-env vulkan1.3 "%D%glsl\probe.rgen"  -o "%G%\rgen.spv"   || exit /b 1
"%B%\glslangValidator.exe" -V --target-env vulkan1.3 "%D%glsl\probe.rmiss" -o "%G%\miss.spv"   || exit /b 1
"%B%\glslangValidator.exe" -V --target-env vulkan1.3 "%D%glsl\shadow.rmiss" -o "%G%\shadow.spv" || exit /b 1
"%B%\glslangValidator.exe" -V --target-env vulkan1.3 "%D%glsl\probe.rchit" -o "%G%\chit.spv"   || exit /b 1
for %%E in (rgen:rgen miss:miss shadowMiss:shadow chit:chit) do (
  for /f "tokens=1,2 delims=:" %%a in ("%%E") do (
    "%B%\slangc.exe" "%D%probe.slang" -target spirv -profile spirv_1_5 -entry %%a -o "%S%\%%b.spv" || exit /b 2
  )
)
for %%E in (rgen:rgen miss:miss shadowMiss:shadow chit:chit ahit:ahit ahitShadow:ahit_shadow) do (
  for /f "tokens=1,2 delims=:" %%a in ("%%E") do (
    "%B%\slangc.exe" "%D%probe_tex.slang" -target spirv -profile spirv_1_5 -entry %%a -o "%T%\%%b.spv" || exit /b 2
  )
)
for %%F in ("%G%\*.spv" "%S%\*.spv" "%T%\*.spv") do (
  "%B%\spirv-val.exe" --target-env vulkan1.3 "%%~F" || exit /b 3
)
echo SHADERS_OK
