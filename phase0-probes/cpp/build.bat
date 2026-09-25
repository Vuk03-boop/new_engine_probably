@echo off
rem Configures and builds the C++ probe in cpp\build (Release, Ninja, -j 2). Restores the caller's directory.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
pushd "%~dp0"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release || (popd & exit /b 1)
cmake --build build -j 2 || (popd & exit /b 2)
popd
echo CPP_BUILD_OK
