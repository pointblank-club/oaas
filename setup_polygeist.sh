#!/bin/bash
# Setup script for building Polygeist in the VM
# This script clones and builds Polygeist with the correct LLVM/MLIR version

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Polygeist Setup Script for OAAS VM                       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
POLYGEIST_DIR="$SCRIPT_DIR/polygeist"
LLVM_VERSION="19"
JOBS=$(nproc 2>/dev/null || echo 4)

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}[1/6] Checking Prerequisites${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check for required tools
MISSING_TOOLS=0
for tool in git cmake ninja clang mlir-opt mlir-translate; do
    if command -v $tool >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $tool: $(command -v $tool)"
    else
        echo -e "${RED}✗${NC} $tool: NOT FOUND"
        MISSING_TOOLS=1
    fi
done

if [ $MISSING_TOOLS -eq 1 ]; then
    echo ""
    echo -e "${RED}ERROR: Missing required tools${NC}"
    echo ""
    echo "Please install missing dependencies:"
    echo "  sudo apt update"
    echo "  sudo apt install -y git cmake ninja-build clang llvm-${LLVM_VERSION}-dev"
    echo "  sudo apt install -y mlir-${LLVM_VERSION}-tools libmlir-${LLVM_VERSION}-dev"
    echo ""
    exit 1
fi

echo ""

# Check MLIR version
echo "Checking MLIR version..."
MLIR_VERSION=$(mlir-opt --version 2>/dev/null | grep "MLIR version" | head -1 || echo "Unknown")
echo "  MLIR: $MLIR_VERSION"

# Find LLVM/MLIR installation
LLVM_DIR=""
for path in \
    "/usr/lib/llvm-${LLVM_VERSION}" \
    "/usr/local/lib/cmake/mlir" \
    "/usr/lib/cmake/mlir" \
    "/opt/llvm/lib/cmake/mlir"; do
    if [ -d "$path" ]; then
        if [ -f "$path/lib/cmake/mlir/MLIRConfig.cmake" ]; then
            LLVM_DIR="$path/lib/cmake/mlir"
            break
        elif [ -f "$path/MLIRConfig.cmake" ]; then
            LLVM_DIR="$path"
            break
        fi
    fi
done

if [ -z "$LLVM_DIR" ]; then
    echo -e "${YELLOW}⚠ Could not auto-detect MLIR installation${NC}"
    echo "Will try to find it during CMake configuration..."
else
    echo -e "${GREEN}✓${NC} MLIR installation: $LLVM_DIR"
fi

echo ""

# ============================================================================
# STEP 2: Clone Polygeist
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}[2/6] Cloning Polygeist Repository${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -d "$POLYGEIST_DIR" ]; then
    echo -e "${YELLOW}⚠ Polygeist directory already exists: $POLYGEIST_DIR${NC}"
    echo ""
    read -p "Do you want to remove and re-clone? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing directory..."
        rm -rf "$POLYGEIST_DIR"
    else
        echo "Keeping existing directory. Skipping clone..."
        echo ""
    fi
fi

if [ ! -d "$POLYGEIST_DIR" ]; then
    echo "Cloning Polygeist repository..."
    echo "  This may take a few minutes..."
    echo ""

    git clone --recursive https://github.com/llvm/Polygeist.git "$POLYGEIST_DIR"

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to clone Polygeist${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Polygeist cloned successfully"
else
    echo -e "${GREEN}✓${NC} Using existing Polygeist directory"
fi

echo ""

# ============================================================================
# STEP 3: Configure Build with CMake
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}[3/6] Configuring Build with CMake${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd "$POLYGEIST_DIR"
mkdir -p build
cd build

echo "Running CMake configuration..."
echo "  Build directory: $POLYGEIST_DIR/build"
echo "  Using $JOBS parallel jobs"
echo ""

CMAKE_ARGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
)

# Add MLIR_DIR if we found it
if [ -n "$LLVM_DIR" ]; then
    CMAKE_ARGS+=(-DMLIR_DIR="$LLVM_DIR")
    echo "  MLIR_DIR: $LLVM_DIR"
fi

# Try configuration
if cmake "${CMAKE_ARGS[@]}" .. 2>&1 | tee cmake_config.log; then
    echo -e "${GREEN}✓${NC} CMake configuration successful"
else
    echo -e "${RED}ERROR: CMake configuration failed${NC}"
    echo ""
    echo "Common issues:"
    echo "  1. MLIR not found - Install: sudo apt install libmlir-${LLVM_VERSION}-dev"
    echo "  2. Version mismatch - Ensure LLVM/MLIR version matches Polygeist requirements"
    echo ""
    echo "CMake log saved to: $POLYGEIST_DIR/build/cmake_config.log"
    echo ""
    echo "Manual configuration:"
    echo "  cd $POLYGEIST_DIR/build"
    echo "  cmake -G Ninja -DMLIR_DIR=/path/to/mlir/cmake .."
    echo ""
    exit 1
fi

echo ""

# ============================================================================
# STEP 4: Build Polygeist
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}[4/6] Building Polygeist${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Building Polygeist (this may take 10-30 minutes)..."
echo "  Using $JOBS parallel jobs"
echo ""

# Build with progress
if ninja -j $JOBS 2>&1 | tee build.log; then
    echo ""
    echo -e "${GREEN}✓${NC} Polygeist built successfully"
else
    echo ""
    echo -e "${RED}ERROR: Build failed${NC}"
    echo ""
    echo "Build log saved to: $POLYGEIST_DIR/build/build.log"
    echo ""
    echo "To retry build:"
    echo "  cd $POLYGEIST_DIR/build"
    echo "  ninja -j $JOBS"
    echo ""
    exit 1
fi

echo ""

# ============================================================================
# STEP 5: Verify Build
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}[5/6] Verifying Build${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check for key binaries
BINARIES_OK=1

if [ -f "$POLYGEIST_DIR/build/bin/cgeist" ]; then
    echo -e "${GREEN}✓${NC} cgeist: $POLYGEIST_DIR/build/bin/cgeist"
else
    echo -e "${RED}✗${NC} cgeist: NOT FOUND"
    BINARIES_OK=0
fi

if [ -f "$POLYGEIST_DIR/build/bin/mlir-clang" ]; then
    echo -e "${GREEN}✓${NC} mlir-clang: $POLYGEIST_DIR/build/bin/mlir-clang"
else
    echo -e "${YELLOW}⚠${NC} mlir-clang: NOT FOUND (cgeist is preferred)"
fi

if [ $BINARIES_OK -eq 0 ]; then
    echo ""
    echo -e "${RED}ERROR: Required binaries not found${NC}"
    exit 1
fi

# Test basic functionality
echo ""
echo "Testing Polygeist functionality..."

# Create simple test
TEST_C=$(mktemp --suffix=.c)
cat > "$TEST_C" << 'EOF'
int add(int a, int b) {
    return a + b;
}
EOF

if "$POLYGEIST_DIR/build/bin/cgeist" "$TEST_C" --function='*' -o /tmp/test.mlir 2>&1 >/dev/null; then
    echo -e "${GREEN}✓${NC} Basic Polygeist test passed"
    rm -f /tmp/test.mlir "$TEST_C"
else
    echo -e "${RED}✗${NC} Basic Polygeist test failed"
    rm -f "$TEST_C"
    BINARIES_OK=0
fi

echo ""

# ============================================================================
# STEP 6: Setup Environment
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}[6/6] Setting Up Environment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create environment setup script
ENV_SCRIPT="$SCRIPT_DIR/polygeist_env.sh"
cat > "$ENV_SCRIPT" << EOF
#!/bin/bash
# Polygeist environment setup
# Source this file to add Polygeist to PATH

export PATH="$POLYGEIST_DIR/build/bin:\$PATH"
export POLYGEIST_DIR="$POLYGEIST_DIR"
export POLYGEIST_BUILD_DIR="$POLYGEIST_DIR/build"

echo "Polygeist environment loaded:"
echo "  cgeist: \$(which cgeist)"
echo "  POLYGEIST_DIR: \$POLYGEIST_DIR"
EOF

chmod +x "$ENV_SCRIPT"

echo "Environment setup script created: $ENV_SCRIPT"
echo ""
echo "To use Polygeist in your current session:"
echo -e "  ${CYAN}source $ENV_SCRIPT${NC}"
echo ""
echo "Or add to your ~/.bashrc for permanent setup:"
echo -e "  ${CYAN}echo 'export PATH=\"$POLYGEIST_DIR/build/bin:\$PATH\"' >> ~/.bashrc${NC}"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  ✅ Polygeist Setup Complete!                             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Installation Summary:"
echo "  Polygeist directory: $POLYGEIST_DIR"
echo "  Build directory:     $POLYGEIST_DIR/build"
echo "  Binary directory:    $POLYGEIST_DIR/build/bin"
echo ""

echo "Key binaries:"
echo "  cgeist:      $POLYGEIST_DIR/build/bin/cgeist"
if [ -f "$POLYGEIST_DIR/build/bin/mlir-clang" ]; then
    echo "  mlir-clang:  $POLYGEIST_DIR/build/bin/mlir-clang"
fi
echo ""

echo "Next steps:"
echo ""
echo "1. Load Polygeist environment:"
echo -e "   ${CYAN}source ./polygeist_env.sh${NC}"
echo ""
echo "2. Run tests to verify integration:"
echo -e "   ${CYAN}./test_polygeist_e2e.sh${NC}"
echo ""
echo "3. Test Polygeist pipeline:"
echo -e "   ${CYAN}./mlir-obs/polygeist-pipeline.sh test.c output${NC}"
echo ""

if [ $BINARIES_OK -eq 1 ]; then
    exit 0
else
    echo -e "${YELLOW}⚠ Setup completed with warnings${NC}"
    exit 1
fi
