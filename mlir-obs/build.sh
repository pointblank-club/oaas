#!/bin/bash
# Build script for MLIR Obfuscation Library

set -e

echo "=========================================="
echo "  Building MLIR Obfuscation Library"
echo "=========================================="
echo ""

# Check for required tools
echo "Checking for required tools..."
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is required but not found"; exit 1; }
command -v clang >/dev/null 2>&1 || { echo "ERROR: clang is required but not found"; exit 1; }
command -v mlir-opt >/dev/null 2>&1 || { echo "ERROR: mlir-opt is required but not found"; exit 1; }

echo "✅ All required tools found"
echo ""

# Get the script directory (mlir-obs directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."

# Try to find MLIR automatically
MLIR_DIRS=(
  "/usr/lib/cmake/mlir"
  "/usr/local/lib/cmake/mlir"
  "$HOME/llvm-project/build/lib/cmake/mlir"
  "$HOME/llvm/build/lib/cmake/mlir"
)

MLIR_DIR_ARG=""
for dir in "${MLIR_DIRS[@]}"; do
  if [ -f "$dir/MLIRConfig.cmake" ]; then
    echo "Found MLIR at: $dir"
    MLIR_DIR_ARG="-DMLIR_DIR=$dir"
    break
  fi
done

if [ -z "$MLIR_DIR_ARG" ]; then
  echo "WARNING: Could not auto-detect MLIR installation."
  echo "If CMake fails, set MLIR_DIR manually:"
  echo "  export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir"
fi

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  $MLIR_DIR_ARG \
  || { echo "ERROR: CMake configuration failed"; exit 1; }

# Build
echo ""
echo "Building library..."
if [ -f "build.ninja" ]; then
    ninja || { echo "ERROR: Build failed"; exit 1; }
else
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) \
      || { echo "ERROR: Build failed"; exit 1; }
fi

echo ""
echo "=========================================="
echo "✅ Build successful!"
echo "=========================================="
echo ""
echo "Library location:"
find . -name "*MLIRObfuscation.*" -type f
echo ""
echo "To test the library, run:"
echo "  cd $SCRIPT_DIR"
echo "  ./test.sh"
