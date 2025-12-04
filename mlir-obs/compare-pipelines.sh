#!/bin/bash
# Compare traditional LLVM pipeline vs Polygeist pipeline
# This demonstrates the benefits of Polygeist for obfuscation

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

INPUT_FILE="${1:-}"

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <input.c>"
    exit 1
fi

echo "=========================================="
echo "  Pipeline Comparison"
echo "=========================================="
echo ""
echo "Comparing:"
echo "  1. Traditional: clang → LLVM IR → mlir-translate → MLIR"
echo "  2. Polygeist:   cgeist → high-level MLIR directly"
echo ""

TEMP_DIR="$(mktemp -d)"
trap "rm -rf $TEMP_DIR" EXIT

# ============================================================================
# TRADITIONAL PIPELINE
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Traditional Pipeline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "[1/3] clang → LLVM IR..."
clang -S -emit-llvm "$INPUT_FILE" -o "$TEMP_DIR/traditional.ll" 2>&1
echo -e "${GREEN}✓${NC} Generated LLVM IR"

echo "[2/3] LLVM IR → MLIR (LLVM dialect)..."
mlir-translate --import-llvm "$TEMP_DIR/traditional.ll" -o "$TEMP_DIR/traditional.mlir" 2>&1
echo -e "${GREEN}✓${NC} Imported to MLIR (LLVM dialect only)"

echo "[3/3] Analyzing traditional MLIR..."
TRAD_LINES=$(wc -l < "$TEMP_DIR/traditional.mlir")
TRAD_FUNCS=$(grep -c "llvm.func @" "$TEMP_DIR/traditional.mlir" || echo "0")
echo "  Lines: $TRAD_LINES"
echo "  Functions (llvm.func): $TRAD_FUNCS"
echo "  Dialects: LLVM only (low-level)"
echo ""

# ============================================================================
# POLYGEIST PIPELINE
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Polygeist Pipeline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if ! command -v cgeist >/dev/null 2>&1 && ! command -v mlir-clang >/dev/null 2>&1; then
    echo "⚠ Polygeist not installed - skipping comparison"
    echo ""
    echo "Install Polygeist to see the full comparison"
    exit 0
fi

CGEIST=$(command -v cgeist || command -v mlir-clang)

echo "[1/2] cgeist → high-level MLIR..."
$CGEIST "$INPUT_FILE" --function="*" --raise-scf-to-affine \
    -o "$TEMP_DIR/polygeist.mlir" 2>&1
echo -e "${GREEN}✓${NC} Generated high-level MLIR"

echo "[2/2] Analyzing Polygeist MLIR..."
POLY_LINES=$(wc -l < "$TEMP_DIR/polygeist.mlir")
POLY_FUNCS=$(grep -c "func.func @" "$TEMP_DIR/polygeist.mlir" || echo "0")
POLY_SCF=$(grep -c "scf\." "$TEMP_DIR/polygeist.mlir" || echo "0")
POLY_AFFINE=$(grep -c "affine\." "$TEMP_DIR/polygeist.mlir" || echo "0")
POLY_MEMREF=$(grep -c "memref\." "$TEMP_DIR/polygeist.mlir" || echo "0")

echo "  Lines: $POLY_LINES"
echo "  Functions (func.func): $POLY_FUNCS"
echo "  SCF ops (structured control flow): $POLY_SCF"
echo "  Affine ops (loop analysis): $POLY_AFFINE"
echo "  MemRef ops (memory abstraction): $POLY_MEMREF"
echo "  Dialects: func, scf, memref, affine, arith (high-level)"
echo ""

# ============================================================================
# COMPARISON
# ============================================================================
echo "=========================================="
echo "  Comparison Summary"
echo "=========================================="
echo ""

echo "Traditional Pipeline:"
echo "  ✓ Simple, well-established"
echo "  ✗ Low-level LLVM dialect only"
echo "  ✗ Lost high-level semantic information"
echo "  ✗ Limited obfuscation opportunities"
echo ""

echo "Polygeist Pipeline:"
echo "  ✓ Rich semantic information preserved"
echo "  ✓ Multiple high-level dialects (SCF, affine, memref)"
echo "  ✓ Better loop/control-flow analysis"
echo "  ✓ More obfuscation opportunities (SCF-level passes)"
echo "  ✗ Requires Polygeist installation"
echo ""

echo "Key Differences:"
echo ""
printf "  %-20s %-15s %-15s\n" "Metric" "Traditional" "Polygeist"
printf "  %-20s %-15s %-15s\n" "────────────────────" "───────────────" "───────────────"
printf "  %-20s %-15s %-15s\n" "MLIR Lines" "$TRAD_LINES" "$POLY_LINES"
printf "  %-20s %-15s %-15s\n" "Dialect Level" "Low (LLVM)" "High (func/scf)"
printf "  %-20s %-15s %-15s\n" "SCF Operations" "0" "$POLY_SCF"
printf "  %-20s %-15s %-15s\n" "Affine Operations" "0" "$POLY_AFFINE"
printf "  %-20s %-15s %-15s\n" "Obfuscation Passes" "2 (basic)" "3+ (enhanced)"
echo ""

echo "Sample MLIR Output:"
echo ""
echo -e "${BLUE}Traditional (LLVM dialect):${NC}"
head -n 10 "$TEMP_DIR/traditional.mlir" | sed 's/^/  /'
echo "  ..."
echo ""

echo -e "${BLUE}Polygeist (High-level dialects):${NC}"
head -n 10 "$TEMP_DIR/polygeist.mlir" | sed 's/^/  /'
echo "  ..."
echo ""

echo "=========================================="
echo "Full outputs saved in: $TEMP_DIR"
echo "  Traditional: $TEMP_DIR/traditional.mlir"
echo "  Polygeist:   $TEMP_DIR/polygeist.mlir"
echo ""
echo "To inspect:"
echo "  cat $TEMP_DIR/traditional.mlir"
echo "  cat $TEMP_DIR/polygeist.mlir"
echo "=========================================="
