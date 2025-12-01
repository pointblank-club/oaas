#!/bin/bash
# Build script for MLIR Obfuscation Library
# Supports both standard LLVM installations and custom builds with ClangIR

set -e

echo "=========================================="
echo "  Building MLIR Obfuscation Library"
echo "=========================================="
echo ""

# Get the script directory (mlir-obs directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# ============================================================================
# Check for required tools
# ============================================================================
echo "Checking for required tools..."

MISSING_TOOLS=()

command -v cmake >/dev/null 2>&1 || MISSING_TOOLS+=("cmake")
command -v clang >/dev/null 2>&1 || MISSING_TOOLS+=("clang")
command -v mlir-opt >/dev/null 2>&1 || MISSING_TOOLS+=("mlir-opt")

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo "ERROR: Missing required tools: ${MISSING_TOOLS[*]}"
    echo ""
    echo "Please install LLVM/MLIR tools. On Ubuntu:"
    echo "  apt-get install llvm-dev mlir-tools"
    echo ""
    echo "Or build LLVM from source with MLIR enabled."
    exit 1
fi

echo "✅ All required tools found"

# Check for optional ClangIR support
if clang -fclangir --help >/dev/null 2>&1; then
    echo "✅ ClangIR support detected"
    CLANGIR_AVAILABLE=1
else
    echo "ℹ️  ClangIR not available (optional)"
    CLANGIR_AVAILABLE=0
fi

echo ""

# ============================================================================
# Detect LLVM/MLIR installation
# ============================================================================
echo "Detecting LLVM/MLIR installation..."

# Search paths for MLIR
MLIR_SEARCH_PATHS=(
    # Custom installation paths
    "/usr/local/llvm/lib/cmake/mlir"
    "/opt/llvm/lib/cmake/mlir"
    # Ubuntu/Debian package paths (various versions)
    "/usr/lib/llvm-19/lib/cmake/mlir"
    "/usr/lib/llvm-18/lib/cmake/mlir"
    "/usr/lib/llvm-17/lib/cmake/mlir"
    "/usr/lib/llvm-16/lib/cmake/mlir"
    "/usr/lib/llvm-15/lib/cmake/mlir"
    # Generic paths
    "/usr/lib/cmake/mlir"
    "/usr/local/lib/cmake/mlir"
    # Home directory builds
    "$HOME/llvm-project/build/lib/cmake/mlir"
    "$HOME/llvm/build/lib/cmake/mlir"
)

MLIR_DIR=""
for dir in "${MLIR_SEARCH_PATHS[@]}"; do
    if [ -f "$dir/MLIRConfig.cmake" ]; then
        MLIR_DIR="$dir"
        echo "  Found MLIR at: $MLIR_DIR"
        break
    fi
done

# Also find LLVM_DIR (usually sibling to MLIR)
LLVM_DIR=""
if [ -n "$MLIR_DIR" ]; then
    # Try to find LLVM in the same installation
    LLVM_DIR="${MLIR_DIR%/mlir}/llvm"
    if [ ! -f "$LLVM_DIR/LLVMConfig.cmake" ]; then
        # Alternative: LLVM might be in lib/cmake/llvm
        PARENT_DIR=$(dirname "$MLIR_DIR")
        LLVM_DIR="$PARENT_DIR/llvm"
    fi
    if [ -f "$LLVM_DIR/LLVMConfig.cmake" ]; then
        echo "  Found LLVM at: $LLVM_DIR"
    else
        LLVM_DIR=""
    fi
fi

if [ -z "$MLIR_DIR" ]; then
    echo ""
    echo "WARNING: Could not auto-detect MLIR installation."
    echo "If CMake fails, set MLIR_DIR and LLVM_DIR manually:"
    echo ""
    echo "  export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir"
    echo "  export LLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm"
    echo ""
fi

# Use environment variables if set
MLIR_DIR="${MLIR_DIR:-$MLIR_DIR}"
LLVM_DIR="${LLVM_DIR:-$LLVM_DIR}"

echo ""

# ============================================================================
# Create and configure build directory
# ============================================================================
echo "Configuring build..."

mkdir -p build
cd build

# Prepare CMake arguments
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17
)

if [ -n "$MLIR_DIR" ]; then
    CMAKE_ARGS+=(-DMLIR_DIR="$MLIR_DIR")
fi

if [ -n "$LLVM_DIR" ]; then
    CMAKE_ARGS+=(-DLLVM_DIR="$LLVM_DIR")
fi

# Try to use Ninja if available (faster)
if command -v ninja >/dev/null 2>&1; then
    CMAKE_ARGS+=(-G Ninja)
    BUILD_CMD="ninja"
else
    BUILD_CMD="make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

echo "CMake configuration:"
for arg in "${CMAKE_ARGS[@]}"; do
    echo "  $arg"
done
echo ""

cmake .. "${CMAKE_ARGS[@]}" || {
    echo ""
    echo "ERROR: CMake configuration failed"
    echo ""
    echo "Common fixes:"
    echo "  1. Set MLIR_DIR manually: export MLIR_DIR=/path/to/mlir"
    echo "  2. Install LLVM/MLIR: apt-get install llvm-dev mlir-tools"
    echo "  3. Build LLVM from source with MLIR enabled"
    exit 1
}

# ============================================================================
# Build the library
# ============================================================================
echo ""
echo "Building library..."

$BUILD_CMD || {
    echo ""
    echo "ERROR: Build failed"
    exit 1
}

# ============================================================================
# Verify build
# ============================================================================
echo ""
echo "=========================================="
echo "✅ Build successful!"
echo "=========================================="
echo ""
echo "Library location:"
find . -name "*MLIRObfuscation.*" -type f 2>/dev/null | head -5

# Get the library path for testing
LIB_PATH=$(find . -name "libMLIRObfuscation.so" -o -name "MLIRObfuscation.so" -o -name "libMLIRObfuscation.dylib" 2>/dev/null | head -1)

if [ -n "$LIB_PATH" ]; then
    echo ""
    echo "Library: $LIB_PATH"
    
    # Quick verification
    echo ""
    echo "Verifying library can be loaded..."
    if mlir-opt --load-pass-plugin="$LIB_PATH" --help 2>&1 | grep -q "string-encrypt\|symbol-obfuscate\|crypto-hash\|constant-obfuscate"; then
        echo "✅ Library passes registered successfully!"
        echo ""
        echo "Available passes:"
        mlir-opt --load-pass-plugin="$LIB_PATH" --help 2>&1 | grep -E "^\s+--string-encrypt|^\s+--symbol-obfuscate|^\s+--crypto-hash|^\s+--constant-obfuscate|^\s+--scf-obfuscate" || true
    else
        echo "⚠️  Could not verify passes (library may still work)"
    fi
fi

echo ""
echo "To test the library, run:"
echo "  cd $SCRIPT_DIR"
echo "  ./test.sh"
echo ""
echo "To use with the obfuscator:"
echo "  python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \\"
echo "      --enable-string-encrypt --output ./output"
