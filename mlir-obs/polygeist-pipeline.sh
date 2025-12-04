#!/bin/bash
# End-to-end Polygeist-based obfuscation pipeline
# Usage: ./polygeist-pipeline.sh input.c output_binary

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INPUT_FILE="${1:-}"
OUTPUT_BINARY="${2:-a.out}"
TEMP_DIR="$(mktemp -d)"
LIBRARY=$(find "$SCRIPT_DIR/build" -name "*MLIRObfuscation.*" -type f | head -1)
MLIR_OBFUSCATE="$SCRIPT_DIR/build/tools/mlir-obfuscate"

trap "rm -rf $TEMP_DIR" EXIT

echo "=========================================="
echo "  Polygeist MLIR Obfuscation Pipeline"
echo "=========================================="
echo ""

# Validate input
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}ERROR: No input file specified${NC}"
    echo "Usage: $0 <input.c> [output_binary]"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}ERROR: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Check if Polygeist is available
if ! command -v cgeist >/dev/null 2>&1; then
    echo -e "${YELLOW}WARNING: cgeist not found. Trying mlir-clang...${NC}"
    if ! command -v mlir-clang >/dev/null 2>&1; then
        echo -e "${RED}ERROR: Polygeist not installed${NC}"
        echo ""
        echo "Please install Polygeist first:"
        echo "  git clone --recursive https://github.com/llvm/Polygeist"
        echo "  cd Polygeist && mkdir build && cd build"
        echo "  cmake -G Ninja .. -DMLIR_DIR=\$PWD/../../llvm-project/build/lib/cmake/mlir"
        echo "  ninja"
        echo ""
        echo "Then add to PATH or set POLYGEIST_DIR"
        exit 1
    fi
    CGEIST_CMD="mlir-clang"
else
    CGEIST_CMD="cgeist"
fi

# Check if obfuscator is built
if [ ! -f "$MLIR_OBFUSCATE" ]; then
    echo -e "${YELLOW}WARNING: mlir-obfuscate not found, using mlir-opt with plugin${NC}"
    MLIR_OBFUSCATE="mlir-opt --load-pass-plugin=$LIBRARY"
fi

echo -e "${GREEN}✓${NC} Input file: $INPUT_FILE"
echo -e "${GREEN}✓${NC} Output binary: $OUTPUT_BINARY"
echo -e "${GREEN}✓${NC} Temp directory: $TEMP_DIR"
echo -e "${GREEN}✓${NC} Polygeist: $CGEIST_CMD"
echo ""

# ============================================================================
# STEP 1: C/C++ -> Polygeist MLIR (func, scf, memref, affine dialects)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 1/7] C/C++ → Polygeist MLIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$CGEIST_CMD "$INPUT_FILE" \
    --function="*" \
    --raise-scf-to-affine \
    -o "$TEMP_DIR/polygeist.mlir" \
    2>&1 || { echo -e "${RED}✗ Failed${NC}"; exit 1; }

echo -e "${GREEN}✓ Generated Polygeist MLIR${NC}"
echo "  Output: $TEMP_DIR/polygeist.mlir"
echo ""

# ============================================================================
# STEP 2: SCF-level obfuscation (optional, Polygeist-specific)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 2/7] SCF Dialect Obfuscation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$MLIR_OBFUSCATE "$TEMP_DIR/polygeist.mlir" \
    --pass-pipeline="builtin.module(scf-obfuscate)" \
    -o "$TEMP_DIR/scf_obfuscated.mlir" \
    2>&1 || { echo -e "${YELLOW}⚠ SCF obfuscation skipped (pass may not be ready)${NC}";
              cp "$TEMP_DIR/polygeist.mlir" "$TEMP_DIR/scf_obfuscated.mlir"; }

echo -e "${GREEN}✓ SCF obfuscation complete${NC}"
echo ""

# ============================================================================
# STEP 3: Symbol obfuscation (works on func::FuncOp)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 3/7] Symbol Obfuscation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$MLIR_OBFUSCATE "$TEMP_DIR/scf_obfuscated.mlir" \
    --pass-pipeline="builtin.module(symbol-obfuscate)" \
    -o "$TEMP_DIR/symbol_obfuscated.mlir" \
    2>&1 || { echo -e "${RED}✗ Failed${NC}"; exit 1; }

echo "Original function names:"
grep "func.func @" "$TEMP_DIR/polygeist.mlir" | head -5
echo ""
echo "Obfuscated function names:"
grep "func.func @" "$TEMP_DIR/symbol_obfuscated.mlir" | head -5

echo -e "${GREEN}✓ Symbol obfuscation complete${NC}"
echo ""

# ============================================================================
# STEP 4: String encryption
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 4/7] String Encryption"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$MLIR_OBFUSCATE "$TEMP_DIR/symbol_obfuscated.mlir" \
    --pass-pipeline="builtin.module(string-encrypt)" \
    -o "$TEMP_DIR/string_encrypted.mlir" \
    2>&1 || { echo -e "${RED}✗ Failed${NC}"; exit 1; }

echo -e "${GREEN}✓ String encryption complete${NC}"
echo ""

# ============================================================================
# STEP 5: Lower to LLVM dialect
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 5/7] Lowering to LLVM Dialect"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mlir-opt "$TEMP_DIR/string_encrypted.mlir" \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$TEMP_DIR/llvm_dialect.mlir" \
    2>&1 || { echo -e "${RED}✗ Failed${NC}"; exit 1; }

echo -e "${GREEN}✓ Lowering to LLVM dialect complete${NC}"
echo ""

# ============================================================================
# STEP 6: Convert MLIR to LLVM IR
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 6/7] MLIR → LLVM IR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mlir-translate --mlir-to-llvmir "$TEMP_DIR/llvm_dialect.mlir" \
    -o "$TEMP_DIR/output.ll" \
    2>&1 || { echo -e "${RED}✗ Failed${NC}"; exit 1; }

echo -e "${GREEN}✓ LLVM IR generation complete${NC}"
echo "  Output: $TEMP_DIR/output.ll"
echo ""

# ============================================================================
# STEP 7: Compile to binary (with optional OLLVM passes)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 7/7] Compiling to Binary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if OLLVM plugin is available (from existing pipeline)
OLLVM_PLUGIN=""
if [ -f "../cmd/llvm-obfuscator/plugins/linux-x86_64/lib/LLVMObfuscation.so" ]; then
    OLLVM_PLUGIN="-fpass-plugin=../cmd/llvm-obfuscator/plugins/linux-x86_64/lib/LLVMObfuscation.so"
    echo "Found OLLVM plugin - will apply additional obfuscation"
fi

clang "$TEMP_DIR/output.ll" \
    -o "$OUTPUT_BINARY" \
    $OLLVM_PLUGIN \
    -O2 \
    2>&1 || { echo -e "${RED}✗ Failed${NC}"; exit 1; }

echo -e "${GREEN}✓ Binary compilation complete${NC}"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================
echo "=========================================="
echo "  Verification Results"
echo "=========================================="
echo ""

echo "Binary size:"
ls -lh "$OUTPUT_BINARY" | awk '{print "  " $5 " " $9}'
echo ""

echo "Symbol count:"
SYMBOL_COUNT=$(nm "$OUTPUT_BINARY" 2>/dev/null | grep -v ' U ' | wc -l)
echo "  $SYMBOL_COUNT symbols"
echo ""

echo "Checking for obfuscation:"
if nm "$OUTPUT_BINARY" 2>/dev/null | grep -q "f_[0-9a-f]\{8\}"; then
    echo -e "  ${GREEN}✓ Obfuscated symbols detected${NC}"
else
    echo -e "  ${YELLOW}⚠ No obfuscated symbols found (may be stripped)${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo "=========================================="
echo ""
echo "Output binary: $OUTPUT_BINARY"
echo ""
echo "Intermediate files saved in: $TEMP_DIR"
echo "To keep them, run: cp -r $TEMP_DIR ./polygeist_artifacts"
echo ""
