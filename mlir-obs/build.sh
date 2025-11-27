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

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  || { echo "ERROR: CMake configuration failed"; exit 1; }

# Build
echo ""
echo "Building library..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) \
  || { echo "ERROR: Build failed"; exit 1; }

echo ""
echo "=========================================="
echo "✅ Build successful!"
echo "=========================================="
echo ""
echo "Library location:"
find . -name "libMLIRObfuscation.*" -type f
echo ""
echo "To test the library, run:"
echo "  cd $SCRIPT_DIR"
echo "  ./test.sh"
