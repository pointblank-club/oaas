#!/bin/bash
# Comprehensive test script for ClangIR pipeline
# Tests both the standard (Clang → LLVM IR → MLIR) and ClangIR pipelines

set -e

echo "============================================================"
echo "  ClangIR Pipeline Test Suite"
echo "============================================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Create temp directory for tests
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Test directory: $TEMP_DIR"
echo ""

# ============================================================================
# Helper functions
# ============================================================================

pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

skip() {
    echo -e "${YELLOW}⏭️  SKIP${NC}: $1"
    ((TESTS_SKIPPED++))
}

# ============================================================================
# Test 0: Prerequisites
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 0: Checking prerequisites"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check clang
if command -v clang >/dev/null 2>&1; then
    echo "  clang: $(clang --version | head -1)"
    HAVE_CLANG=1
else
    echo "  clang: NOT FOUND"
    HAVE_CLANG=0
fi

# Check mlir-opt
if command -v mlir-opt >/dev/null 2>&1; then
    echo "  mlir-opt: $(mlir-opt --version 2>&1 | head -1)"
    HAVE_MLIR_OPT=1
else
    echo "  mlir-opt: NOT FOUND"
    HAVE_MLIR_OPT=0
fi

# Check mlir-translate
if command -v mlir-translate >/dev/null 2>&1; then
    echo "  mlir-translate: available"
    HAVE_MLIR_TRANSLATE=1
else
    echo "  mlir-translate: NOT FOUND"
    HAVE_MLIR_TRANSLATE=0
fi

# Check for ClangIR support
HAVE_CLANGIR=0
if [ $HAVE_CLANG -eq 1 ]; then
    if clang -fclangir --help >/dev/null 2>&1; then
        echo "  ClangIR: AVAILABLE (clang -fclangir)"
        HAVE_CLANGIR=1
    else
        echo "  ClangIR: NOT AVAILABLE"
    fi
fi

# Check for cir-opt
if command -v cir-opt >/dev/null 2>&1; then
    echo "  cir-opt: available"
    HAVE_CIR_OPT=1
else
    echo "  cir-opt: NOT AVAILABLE"
    HAVE_CIR_OPT=0
fi

# Check for MLIR obfuscation library
MLIR_PLUGIN=""
for path in "$SCRIPT_DIR/build/lib/libMLIRObfuscation.so" \
            "$SCRIPT_DIR/build/lib/MLIRObfuscation.so" \
            "$SCRIPT_DIR/build/lib/libMLIRObfuscation.dylib"; do
    if [ -f "$path" ]; then
        MLIR_PLUGIN="$path"
        break
    fi
done

if [ -n "$MLIR_PLUGIN" ]; then
    echo "  MLIR Plugin: $MLIR_PLUGIN"
    HAVE_MLIR_PLUGIN=1
else
    echo "  MLIR Plugin: NOT FOUND (run ./build.sh first)"
    HAVE_MLIR_PLUGIN=0
fi

echo ""

# ============================================================================
# Create test source files
# ============================================================================

echo "Creating test source files..."

# Simple test file
cat > "$TEMP_DIR/simple.c" << 'EOF'
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(10, 20);
    printf("Result: %d\n", result);
    return 0;
}
EOF

# Test file with strings (for string encryption)
cat > "$TEMP_DIR/strings.c" << 'EOF'
#include <stdio.h>
#include <string.h>

const char* SECRET_PASSWORD = "SuperSecretPass123!";
const char* API_KEY = "sk_live_abc123xyz789";

int validate(const char* input) {
    return strcmp(input, SECRET_PASSWORD) == 0;
}

int main() {
    if (validate("test")) {
        printf("Access granted\n");
    } else {
        printf("Access denied\n");
    }
    return 0;
}
EOF

# Test file with multiple functions (for symbol obfuscation)
cat > "$TEMP_DIR/funcs.c" << 'EOF'
int helper_function_one(int x) {
    return x * 2;
}

int helper_function_two(int x) {
    return x + 10;
}

int process_data(int input) {
    int a = helper_function_one(input);
    int b = helper_function_two(a);
    return b;
}

int main() {
    return process_data(5);
}
EOF

echo ""

# ============================================================================
# Test 1: Standard Pipeline (Clang → LLVM IR → MLIR)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Standard Pipeline (Clang → LLVM IR → MLIR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $HAVE_CLANG -eq 1 ] && [ $HAVE_MLIR_OPT -eq 1 ] && [ $HAVE_MLIR_TRANSLATE -eq 1 ]; then
    # Step 1a: Compile to LLVM IR
    if clang "$TEMP_DIR/simple.c" -S -emit-llvm -o "$TEMP_DIR/simple.ll" 2>/dev/null; then
        pass "1a: Clang → LLVM IR"
    else
        fail "1a: Clang → LLVM IR"
    fi

    # Step 1b: Import LLVM IR to MLIR
    if mlir-translate --import-llvm "$TEMP_DIR/simple.ll" -o "$TEMP_DIR/simple.mlir" 2>/dev/null; then
        pass "1b: LLVM IR → MLIR"
    else
        fail "1b: LLVM IR → MLIR"
    fi

    # Step 1c: Apply MLIR passes (if plugin available)
    if [ $HAVE_MLIR_PLUGIN -eq 1 ]; then
        if mlir-opt "$TEMP_DIR/simple.mlir" \
            --load-pass-plugin="$MLIR_PLUGIN" \
            --pass-pipeline="builtin.module(symbol-obfuscate)" \
            -o "$TEMP_DIR/simple_obf.mlir" 2>/dev/null; then
            pass "1c: MLIR obfuscation pass"
        else
            fail "1c: MLIR obfuscation pass"
        fi
    else
        skip "1c: MLIR obfuscation pass (plugin not built)"
    fi

    # Step 1d: MLIR → LLVM IR
    if mlir-translate --mlir-to-llvmir "$TEMP_DIR/simple.mlir" -o "$TEMP_DIR/simple_out.ll" 2>/dev/null; then
        pass "1d: MLIR → LLVM IR"
    else
        fail "1d: MLIR → LLVM IR"
    fi

    # Step 1e: Compile to binary
    if clang "$TEMP_DIR/simple_out.ll" -o "$TEMP_DIR/simple_bin" 2>/dev/null; then
        pass "1e: LLVM IR → Binary"
    else
        fail "1e: LLVM IR → Binary"
    fi

    # Step 1f: Run binary
    if [ -x "$TEMP_DIR/simple_bin" ]; then
        OUTPUT=$("$TEMP_DIR/simple_bin" 2>&1)
        if echo "$OUTPUT" | grep -q "Result: 30"; then
            pass "1f: Binary execution"
        else
            fail "1f: Binary execution (unexpected output: $OUTPUT)"
        fi
    else
        fail "1f: Binary execution (binary not executable)"
    fi
else
    skip "Test 1: Missing prerequisites"
fi

echo ""

# ============================================================================
# Test 2: ClangIR Pipeline
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: ClangIR Pipeline (clang -fclangir)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $HAVE_CLANGIR -eq 1 ]; then
    # Step 2a: Emit CIR MLIR
    if clang -fclangir -emit-cir "$TEMP_DIR/simple.c" -o "$TEMP_DIR/simple.cir" 2>/dev/null; then
        pass "2a: ClangIR emit (clang -fclangir -emit-cir)"
        echo "    CIR output sample:"
        head -10 "$TEMP_DIR/simple.cir" 2>/dev/null | sed 's/^/      /'
    else
        fail "2a: ClangIR emit"
    fi

    # Step 2b: Full ClangIR compilation
    if clang -fclangir "$TEMP_DIR/simple.c" -o "$TEMP_DIR/simple_cir_bin" 2>/dev/null; then
        pass "2b: Full ClangIR compilation"
    else
        fail "2b: Full ClangIR compilation"
    fi

    # Step 2c: Run ClangIR-compiled binary
    if [ -x "$TEMP_DIR/simple_cir_bin" ]; then
        OUTPUT=$("$TEMP_DIR/simple_cir_bin" 2>&1)
        if echo "$OUTPUT" | grep -q "Result: 30"; then
            pass "2c: ClangIR binary execution"
        else
            fail "2c: ClangIR binary execution (unexpected output)"
        fi
    else
        fail "2c: ClangIR binary execution (binary not created)"
    fi

    # Step 2d: CIR lowering (if cir-opt available)
    if [ $HAVE_CIR_OPT -eq 1 ] && [ -f "$TEMP_DIR/simple.cir" ]; then
        if cir-opt "$TEMP_DIR/simple.cir" --cir-to-llvm -o "$TEMP_DIR/simple_lowered.mlir" 2>/dev/null; then
            pass "2d: CIR lowering (cir-opt --cir-to-llvm)"
        else
            fail "2d: CIR lowering"
        fi
    else
        skip "2d: CIR lowering (cir-opt not available)"
    fi
else
    skip "Test 2: ClangIR not available"
fi

echo ""

# ============================================================================
# Test 3: MLIR Obfuscation Passes
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: MLIR Obfuscation Passes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $HAVE_MLIR_PLUGIN -eq 1 ] && [ $HAVE_CLANG -eq 1 ] && [ $HAVE_MLIR_OPT -eq 1 ]; then
    # Create MLIR from strings test file
    clang "$TEMP_DIR/strings.c" -S -emit-llvm -o "$TEMP_DIR/strings.ll" 2>/dev/null
    mlir-translate --import-llvm "$TEMP_DIR/strings.ll" -o "$TEMP_DIR/strings.mlir" 2>/dev/null

    # Test 3a: string-encrypt pass
    if mlir-opt "$TEMP_DIR/strings.mlir" \
        --load-pass-plugin="$MLIR_PLUGIN" \
        --pass-pipeline="builtin.module(string-encrypt)" \
        -o "$TEMP_DIR/strings_enc.mlir" 2>/dev/null; then
        pass "3a: string-encrypt pass"
    else
        fail "3a: string-encrypt pass"
    fi

    # Test 3b: symbol-obfuscate pass
    if mlir-opt "$TEMP_DIR/strings.mlir" \
        --load-pass-plugin="$MLIR_PLUGIN" \
        --pass-pipeline="builtin.module(symbol-obfuscate)" \
        -o "$TEMP_DIR/strings_sym.mlir" 2>/dev/null; then
        pass "3b: symbol-obfuscate pass"
    else
        fail "3b: symbol-obfuscate pass"
    fi

    # Test 3c: crypto-hash pass
    if mlir-opt "$TEMP_DIR/strings.mlir" \
        --load-pass-plugin="$MLIR_PLUGIN" \
        --pass-pipeline="builtin.module(crypto-hash)" \
        -o "$TEMP_DIR/strings_hash.mlir" 2>/dev/null; then
        pass "3c: crypto-hash pass"
    else
        fail "3c: crypto-hash pass"
    fi

    # Test 3d: constant-obfuscate pass
    if mlir-opt "$TEMP_DIR/strings.mlir" \
        --load-pass-plugin="$MLIR_PLUGIN" \
        --pass-pipeline="builtin.module(constant-obfuscate)" \
        -o "$TEMP_DIR/strings_const.mlir" 2>/dev/null; then
        pass "3d: constant-obfuscate pass"
    else
        fail "3d: constant-obfuscate pass"
    fi

    # Test 3e: Combined passes
    if mlir-opt "$TEMP_DIR/strings.mlir" \
        --load-pass-plugin="$MLIR_PLUGIN" \
        --pass-pipeline="builtin.module(string-encrypt,symbol-obfuscate)" \
        -o "$TEMP_DIR/strings_combined.mlir" 2>/dev/null; then
        pass "3e: Combined passes (string-encrypt + symbol-obfuscate)"
    else
        fail "3e: Combined passes"
    fi
else
    skip "Test 3: Prerequisites not met (need MLIR plugin, clang, mlir-opt)"
fi

echo ""

# ============================================================================
# Test 4: End-to-End Pipeline with Obfuscation
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: End-to-End Pipeline with Obfuscation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $HAVE_MLIR_PLUGIN -eq 1 ] && [ $HAVE_CLANG -eq 1 ] && [ $HAVE_MLIR_OPT -eq 1 ] && [ $HAVE_MLIR_TRANSLATE -eq 1 ]; then
    # Full pipeline: C → LLVM IR → MLIR → Obfuscate → LLVM IR → Binary
    
    # Step 4a: Compile to LLVM IR
    clang "$TEMP_DIR/funcs.c" -S -emit-llvm -o "$TEMP_DIR/funcs.ll" 2>/dev/null
    
    # Step 4b: Import to MLIR
    mlir-translate --import-llvm "$TEMP_DIR/funcs.ll" -o "$TEMP_DIR/funcs.mlir" 2>/dev/null
    
    # Step 4c: Apply obfuscation
    if mlir-opt "$TEMP_DIR/funcs.mlir" \
        --load-pass-plugin="$MLIR_PLUGIN" \
        --pass-pipeline="builtin.module(symbol-obfuscate,constant-obfuscate)" \
        -o "$TEMP_DIR/funcs_obf.mlir" 2>/dev/null; then
        pass "4a: Obfuscation passes applied"
    else
        fail "4a: Obfuscation passes failed"
    fi
    
    # Step 4d: Export to LLVM IR
    if mlir-translate --mlir-to-llvmir "$TEMP_DIR/funcs_obf.mlir" -o "$TEMP_DIR/funcs_obf.ll" 2>/dev/null; then
        pass "4b: MLIR → LLVM IR export"
    else
        fail "4b: MLIR → LLVM IR export"
    fi
    
    # Step 4e: Compile to binary
    if clang "$TEMP_DIR/funcs_obf.ll" -o "$TEMP_DIR/funcs_obf_bin" 2>/dev/null; then
        pass "4c: Final compilation"
    else
        # Try fixing the IR
        echo "    Attempting IR fix..."
        # Add target triple if missing
        if ! grep -q "target triple" "$TEMP_DIR/funcs_obf.ll"; then
            sed -i '1s/^/target triple = "x86_64-unknown-linux-gnu"\ntarget datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"\n\n/' "$TEMP_DIR/funcs_obf.ll"
        fi
        if clang "$TEMP_DIR/funcs_obf.ll" -o "$TEMP_DIR/funcs_obf_bin" 2>/dev/null; then
            pass "4c: Final compilation (with IR fix)"
        else
            fail "4c: Final compilation"
        fi
    fi
    
    # Step 4f: Run and verify
    if [ -x "$TEMP_DIR/funcs_obf_bin" ]; then
        EXIT_CODE=$("$TEMP_DIR/funcs_obf_bin"; echo $?)
        # process_data(5) = helper_function_two(helper_function_one(5))
        # = helper_function_two(10) = 20
        if [ "$EXIT_CODE" = "20" ]; then
            pass "4d: Obfuscated binary runs correctly (exit code: $EXIT_CODE)"
        else
            fail "4d: Obfuscated binary (unexpected exit code: $EXIT_CODE, expected: 20)"
        fi
    else
        fail "4d: Obfuscated binary not executable"
    fi
    
    # Step 4g: Verify symbols are obfuscated
    if [ -x "$TEMP_DIR/funcs_obf_bin" ]; then
        if nm "$TEMP_DIR/funcs_obf_bin" 2>/dev/null | grep -q "helper_function"; then
            fail "4e: Symbol obfuscation (original names still visible)"
        else
            pass "4e: Symbol obfuscation (original names hidden)"
        fi
    fi
else
    skip "Test 4: Prerequisites not met"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================"
echo "  TEST SUMMARY"
echo "============================================================"
echo ""
echo -e "  ${GREEN}Passed${NC}: $TESTS_PASSED"
echo -e "  ${RED}Failed${NC}: $TESTS_FAILED"
echo -e "  ${YELLOW}Skipped${NC}: $TESTS_SKIPPED"
echo ""

TOTAL=$((TESTS_PASSED + TESTS_FAILED))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((TESTS_PASSED * 100 / TOTAL))
    echo "  Pass rate: ${PASS_RATE}%"
fi

echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi

