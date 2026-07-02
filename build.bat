@echo off
REM ============================================================================
REM SpectraPBR one-shot build script (Windows)
REM
REM Usage:
REM   build.bat            configure + build (Release)
REM   build.bat run        configure + build + launch the app
REM   build.bat debug      Debug configuration instead of Release
REM   build.bat clean      wipe the build directory first (full rebuild)
REM   Args combine:        build.bat clean debug run
REM
REM Requirements: CMake 3.18+, CUDA Toolkit, OptiX SDK, Visual Studio C++.
REM If CMake cannot find OptiX automatically (it searches
REM   C:\ProgramData\NVIDIA Corporation\OptiX SDK *), set:
REM   set OptiX_INSTALL_DIR=C:\path\to\OptiX SDK x.x.x
REM
REM CUDA arch defaults to 89 (Ada / RTX 40-series) for ~3x faster device
REM compiles than the 75;86;89 multi-arch default. Building for another GPU:
REM   set SPECTRA_CUDA_ARCH=86
REM ============================================================================
setlocal enabledelayedexpansion

set BUILD_DIR=build
set CONFIG=Release
set DO_RUN=0
set DO_CLEAN=0
if "%SPECTRA_CUDA_ARCH%"=="" set SPECTRA_CUDA_ARCH=89

for %%A in (%*) do (
    if /I "%%A"=="run"   set DO_RUN=1
    if /I "%%A"=="debug" set CONFIG=Debug
    if /I "%%A"=="clean" set DO_CLEAN=1
)

if %DO_CLEAN%==1 (
    echo [build] Cleaning %BUILD_DIR% ...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
)

echo [build] Configuring (%CONFIG%, sm_%SPECTRA_CUDA_ARCH%) ...
cmake -S . -B %BUILD_DIR% -DCMAKE_CUDA_ARCHITECTURES=%SPECTRA_CUDA_ARCH%
if errorlevel 1 (
    echo.
    echo [build] CMake configure FAILED. If OptiX was not found, set
    echo         OptiX_INSTALL_DIR to your OptiX SDK path and re-run.
    exit /b 1
)

echo [build] Building %CONFIG% ...
cmake --build %BUILD_DIR% --config %CONFIG% --parallel
if errorlevel 1 (
    echo.
    echo [build] Build FAILED.
    exit /b 1
)

set EXE_DIR=%BUILD_DIR%\%CONFIG%
set EXE=%EXE_DIR%\SpectraPBR.exe
if not exist "%EXE%" (
    REM Single-config generators (Ninja) put the exe at the build root
    set EXE_DIR=%BUILD_DIR%
    set EXE=%BUILD_DIR%\SpectraPBR.exe
)

echo.
echo [build] OK: %EXE%
if %DO_RUN%==1 (
    echo [build] Launching ...
    REM Run from the exe directory: shaders/, optix_programs/ (PTX) and
    REM assets/ are copied next to the exe by the CMake post-build steps.
    pushd "%EXE_DIR%"
    SpectraPBR.exe
    popd
)
endlocal
