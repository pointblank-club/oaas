#!/bin/bash
# Integration test for Polygeist obfuscation pipeline
# Tests both Polygeist and traditional pipelines

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

echo "=========================================="
echo "  Polygeist Integration Tests"
echo "=========================================="
echo ""

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "[$test_name] "

    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"

    echo -e "[$test_name] ${YELLOW}⊘ SKIP${NC} ($reason)"
    ((TESTS_SKIPPED++))
}

# ============================================================================
# PREREQUISITE CHECKS
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Prerequisite Checks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check build directory
if [ ! -d "build" ]; then
    echo -e "${RED}ERROR: Build directory not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

# Find library
LIBRARY=$(find build -name "*MLIRObfuscation.*" -type f | head -1)
if [ -z "$LIBRARY" ]; then
    echo -e "${RED}ERROR: MLIRObfuscation library not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

echo -e "${GREEN}✓${NC} Library found: $LIBRARY"

# Check for mlir-opt
if ! command -v mlir-opt >/dev/null 2>&1; then
    echo -e "${RED}ERROR: mlir-opt not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} mlir-opt found"

# Check for mlir-translate
if ! command -v mlir-translate >/dev/null 2>&1; then
    echo -e "${RED}ERROR: mlir-translate not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} mlir-translate found"

# Check for clang
if ! command -v clang >/dev/null 2>&1; then
    echo -e "${RED}ERROR: clang not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} clang found"

# Check for Polygeist (optional)
HAVE_POLYGEIST=0
if command -v cgeist >/dev/null 2>&1 || command -v mlir-clang >/dev/null 2>&1; then
    HAVE_POLYGEIST=1
    CGEIST=$(command -v cgeist 2>/dev/null || command -v mlir-clang)
    echo -e "${GREEN}✓${NC} Polygeist found: $CGEIST"
else
    echo -e "${YELLOW}⚠${NC} Polygeist not found (some tests will be skipped)"
fi

echo ""

# ============================================================================
# TEST 1: Library Loading
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Suite 1: Library Loading${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

run_test "Load obfuscation plugin" \
    "mlir-opt --load-pass-plugin=$LIBRARY --help | grep -q 'symbol-obfuscate'"

run_test "Verify string-encrypt pass" \
    "mlir-opt --load-pass-plugin=$LIBRARY --help | grep -q 'string-encrypt'"

if [ $HAVE_POLYGEIST -eq 1 ]; then
    run_test "Verify scf-obfuscate pass" \
        "mlir-opt --load-pass-plugin=$LIBRARY --help | grep -q 'scf-obfuscate'"
else
    skip_test "Verify scf-obfuscate pass" "Polygeist not installed"
fi

echo ""

# ============================================================================
# TEST 2: Traditional Pipeline (LLVM dialect)
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Suite 2: Traditional Pipeline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create test C file
cat > "$TEMP_DIR/test.c" << 'EOF'
int add(int a, int b) {
    return a + b;
}

int main() {
    return add(2, 3);
}
EOF

run_test "Compile C to LLVM IR" \
    "clang -S -emit-llvm $TEMP_DIR/test.c -o $TEMP_DIR/test.ll"

run_test "Import LLVM IR to MLIR" \
    "mlir-translate --import-llvm $TEMP_DIR/test.ll -o $TEMP_DIR/test.mlir"

run_test "Apply symbol obfuscation (LLVM dialect)" \
    "mlir-opt $TEMP_DIR/test.mlir --load-pass-plugin=$LIBRARY --pass-pipeline='builtin.module(symbol-obfuscate)' -o $TEMP_DIR/test_obf.mlir"

run_test "Verify obfuscation (LLVM dialect)" \
    "grep -q 'f_[0-9a-f]\{8\}' $TEMP_DIR/test_obf.mlir"

run_test "Convert to LLVM IR" \
    "mlir-translate --mlir-to-llvmir $TEMP_DIR/test_obf.mlir -o $TEMP_DIR/test_obf.ll"

run_test "Compile to binary" \
    "clang $TEMP_DIR/test_obf.ll -o $TEMP_DIR/test_binary"

run_test "Execute obfuscated binary" \
    "$TEMP_DIR/test_binary && test \$? -eq 5"

echo ""

# ============================================================================
# TEST 3: Polygeist Pipeline (High-level dialects)
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Suite 3: Polygeist Pipeline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ $HAVE_POLYGEIST -eq 0 ]; then
    skip_test "All Polygeist tests" "Polygeist not installed"
    echo ""
else
    # Create more complex test
    cat > "$TEMP_DIR/poly_test.c" << 'EOF'
int validate(int x) {
    if (x > 0) {
        return 1;
    }
    return 0;
}

int main() {
    return validate(5);
}
EOF

    run_test "Generate Polygeist MLIR" \
        "$CGEIST $TEMP_DIR/poly_test.c --function='*' -o $TEMP_DIR/poly_test.mlir"

    run_test "Verify func dialect" \
        "grep -q 'func.func @' $TEMP_DIR/poly_test.mlir"

    run_test "Verify SCF dialect" \
        "grep -q 'scf.if' $TEMP_DIR/poly_test.mlir"

    run_test "Apply symbol obfuscation (func dialect)" \
        "mlir-opt $TEMP_DIR/poly_test.mlir --load-pass-plugin=$LIBRARY --pass-pipeline='builtin.module(symbol-obfuscate)' -o $TEMP_DIR/poly_obf.mlir"

    run_test "Verify obfuscation (func dialect)" \
        "grep -q 'func.func @f_[0-9a-f]\{8\}' $TEMP_DIR/poly_obf.mlir"

    run_test "Apply SCF obfuscation" \
        "mlir-opt $TEMP_DIR/poly_obf.mlir --load-pass-plugin=$LIBRARY --pass-pipeline='builtin.module(scf-obfuscate)' -o $TEMP_DIR/poly_scf_obf.mlir"

    run_test "Lower to LLVM dialect" \
        "mlir-opt $TEMP_DIR/poly_scf_obf.mlir --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-memref-to-llvm --reconcile-unrealized-casts -o $TEMP_DIR/poly_llvm.mlir"

    run_test "Convert to LLVM IR" \
        "mlir-translate --mlir-to-llvmir $TEMP_DIR/poly_llvm.mlir -o $TEMP_DIR/poly.ll"

    run_test "Compile Polygeist binary" \
        "clang $TEMP_DIR/poly.ll -o $TEMP_DIR/poly_binary"

    run_test "Execute Polygeist binary" \
        "$TEMP_DIR/poly_binary && test \$? -eq 1"

    echo ""
fi

# ============================================================================
# TEST 4: Example Files
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Suite 4: Example Files${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -f "examples/simple_auth.c" ]; then
    if [ $HAVE_POLYGEIST -eq 1 ]; then
        run_test "Compile simple_auth.c with Polygeist" \
            "$CGEIST examples/simple_auth.c --function='*' -o $TEMP_DIR/simple_auth.mlir"

        run_test "Obfuscate simple_auth.c" \
            "mlir-opt $TEMP_DIR/simple_auth.mlir --load-pass-plugin=$LIBRARY --pass-pipeline='builtin.module(symbol-obfuscate,string-encrypt)' -o $TEMP_DIR/simple_auth_obf.mlir"
    else
        skip_test "Polygeist example tests" "Polygeist not installed"
    fi
else
    skip_test "Example file tests" "examples/simple_auth.c not found"
fi

if [ -f "examples/loop_example.c" ]; then
    if [ $HAVE_POLYGEIST -eq 1 ]; then
        run_test "Compile loop_example.c with Polygeist" \
            "$CGEIST examples/loop_example.c --function='*' --raise-scf-to-affine -o $TEMP_DIR/loop_example.mlir"

        run_test "Verify affine dialect in loops" \
            "grep -q 'affine.for' $TEMP_DIR/loop_example.mlir || grep -q 'scf.for' $TEMP_DIR/loop_example.mlir"
    else
        skip_test "Loop example tests" "Polygeist not installed"
    fi
fi

echo ""

# ============================================================================
# TEST 5: Dual-Dialect Support
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Suite 5: Dual-Dialect Support${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create MLIR with both func and LLVM dialects
cat > "$TEMP_DIR/dual.mlir" << 'EOF'
module {
  func.func @high_level(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %arg0, %c1 : i32
    return %result : i32
  }

  llvm.func @low_level(%arg0: i32) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %result = llvm.add %arg0, %c1 : i32
    llvm.return %result : i32
  }
}
EOF

run_test "Obfuscate mixed dialect MLIR" \
    "mlir-opt $TEMP_DIR/dual.mlir --load-pass-plugin=$LIBRARY --pass-pipeline='builtin.module(symbol-obfuscate)' -o $TEMP_DIR/dual_obf.mlir"

run_test "Verify func::FuncOp obfuscation" \
    "grep -q 'func.func @f_[0-9a-f]\{8\}' $TEMP_DIR/dual_obf.mlir"

run_test "Verify LLVM::LLVMFuncOp obfuscation" \
    "grep -q 'llvm.func @f_[0-9a-f]\{8\}' $TEMP_DIR/dual_obf.mlir"

echo ""

# ============================================================================
# TEST 6: Pipeline Scripts
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Test Suite 6: Pipeline Scripts${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -f "polygeist-pipeline.sh" ]; then
    run_test "Check polygeist-pipeline.sh exists" \
        "test -x polygeist-pipeline.sh || chmod +x polygeist-pipeline.sh"

    if [ $HAVE_POLYGEIST -eq 1 ] && [ -f "examples/simple_auth.c" ]; then
        run_test "Run full Polygeist pipeline" \
            "./polygeist-pipeline.sh examples/simple_auth.c $TEMP_DIR/pipeline_test"

        run_test "Verify pipeline output binary" \
            "test -f $TEMP_DIR/pipeline_test && test -x $TEMP_DIR/pipeline_test"
    else
        skip_test "Full pipeline test" "Polygeist not installed or examples missing"
    fi
else
    skip_test "Pipeline script tests" "polygeist-pipeline.sh not found"
fi

if [ -f "compare-pipelines.sh" ]; then
    run_test "Check compare-pipelines.sh exists" \
        "test -x compare-pipelines.sh || chmod +x compare-pipelines.sh"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "  Test Results Summary"
echo "=========================================="
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

echo -e "${GREEN}✓ Passed:${NC}  $TESTS_PASSED / $TOTAL_TESTS"
echo -e "${RED}✗ Failed:${NC}  $TESTS_FAILED / $TOTAL_TESTS"
echo -e "${YELLOW}⊘ Skipped:${NC} $TESTS_SKIPPED / $TOTAL_TESTS"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "  All tests passed! ✓"
    echo "==========================================${NC}"
    echo ""

    if [ $HAVE_POLYGEIST -eq 0 ]; then
        echo -e "${YELLOW}Note: Some tests were skipped because Polygeist is not installed.${NC}"
        echo "To enable full testing, install Polygeist and re-run this script."
        echo ""
    fi

    exit 0
else
    echo -e "${RED}=========================================="
    echo "  Some tests failed! ✗"
    echo "==========================================${NC}"
    echo ""
    echo "Please check the errors above and fix the issues."
    exit 1
fi
