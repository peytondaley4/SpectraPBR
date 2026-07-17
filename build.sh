#!/usr/bin/env bash
# ============================================================================
# SpectraPBR one-shot build script (Linux)
#
# Usage:
#   ./build.sh            configure + build (Release)
#   ./build.sh run        configure + build + launch the app
#   ./build.sh debug      Debug configuration instead of Release
#   ./build.sh clean      wipe the build directory first (full rebuild)
#   Args combine:         ./build.sh clean debug run
#
# Requirements: CMake 3.20+, CUDA Toolkit, OptiX SDK, GCC 9+/Clang 10+.
# If CMake cannot find OptiX automatically, export:
#   export OptiX_INSTALL_DIR=~/NVIDIA-OptiX-SDK-x.x.x
#
# CUDA arch defaults to 89 (Ada / RTX 40-series); override with
#   SPECTRA_CUDA_ARCH=86 ./build.sh
# ============================================================================
set -euo pipefail

BUILD_DIR=build
CONFIG=Release
DO_RUN=0
ARCH="${SPECTRA_CUDA_ARCH:-89}"

for arg in "$@"; do
    case "$arg" in
        run)   DO_RUN=1 ;;
        debug) CONFIG=Debug ;;
        clean) echo "[build] Cleaning $BUILD_DIR ..."; rm -rf "$BUILD_DIR" ;;
        *)     echo "[build] Unknown arg: $arg (use: clean|debug|run)"; exit 1 ;;
    esac
done

echo "[build] Configuring ($CONFIG, sm_$ARCH) ..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$CONFIG" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH"

echo "[build] Building $CONFIG ..."
cmake --build "$BUILD_DIR" --config "$CONFIG" --parallel "$(nproc)"

EXE_DIR="$BUILD_DIR"
[ -x "$BUILD_DIR/$CONFIG/SpectraPBR" ] && EXE_DIR="$BUILD_DIR/$CONFIG"

echo
echo "[build] OK: $EXE_DIR/SpectraPBR"
if [ "$DO_RUN" -eq 1 ]; then
    echo "[build] Launching ..."
    # Run from the exe directory: shaders/, optix_programs/ (PTX) and assets/
    # are synced next to the exe by the CMake copy targets.
    (cd "$EXE_DIR" && ./SpectraPBR)
fi
