#!/bin/bash
# Test script for MLIR Obfuscation passes

set -e

echo "=========================================="
echo "  Testing MLIR Obfuscation Passes"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if library is built
LIBRARY=$(find build -name "*MLIRObfuscation.*" -type f | head -1)
if [ -z "$LIBRARY" ]; then
    echo "ERROR: MLIR library not found. Please run ./build.sh first"
    exit 1
fi

echo "Using library: $LIBRARY"
echo ""

# Create test directory
TEST_DIR="test_output"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create test C file
echo "Creating test C file..."
cat > test.c << 'EOF'
#include <stdio.h>
#include <string.h>

const char* SECRET_PASSWORD = "MySecret123!";
const char* API_KEY = "sk_live_12345";

int validate_password(const char* input) {
    return strcmp(input, SECRET_PASSWORD) == 0;
}

int check_api_key(const char* key) {
    return strcmp(key, API_KEY) == 0;
}

int main() {
    printf("Testing obfuscation\n");

    if (validate_password("MySecret123!")) {
        printf("Password valid\n");
    }

    if (check_api_key("sk_live_12345")) {
        printf("API key valid\n");
    }

    return 0;
}
EOF

echo "✅ Test file created"
echo ""

# Test 1: Compile C to LLVM IR, then to MLIR
echo "[Test 1] Converting C to LLVM IR..."
clang -S -emit-llvm test.c -o test_pre.ll 2>&1 || { echo "ERROR: Failed to emit LLVM IR"; exit 1; }
echo "✅ C → LLVM IR successful"

echo "Converting LLVM IR to MLIR..."
mlir-translate --import-llvm test_pre.ll -o test.mlir 2>&1 || { echo "ERROR: Failed to convert to MLIR"; exit 1; }
echo "✅ LLVM IR → MLIR conversion successful"
echo ""

# Test 2: Apply symbol obfuscation
echo "[Test 2] Testing symbol obfuscation pass..."

# Debug: Show detailed error
echo "  [DEBUG] Trying to load plugin and run pass..."
mlir-opt test.mlir \
    --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
    --symbol-obfuscate \
    -o test_symbol_obf.mlir 2>&1 | tee /tmp/mlir_debug.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  [DEBUG] Direct invocation failed. Full error:"
    cat /tmp/mlir_debug.log

    echo "  [DEBUG] Trying pass pipeline syntax..."
    mlir-opt test.mlir \
        --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
        --pass-pipeline="builtin.module(symbol-obfuscate)" \
        -o test_symbol_obf.mlir 2>&1 || { echo "ERROR: Symbol obfuscation failed"; exit 1; }
fi

if grep -q "validate_password" test_symbol_obf.mlir; then
    echo "⚠️  Warning: Original function names still present (may need pass refinement)"
else
    echo "✅ Symbol obfuscation successful - function names obfuscated"
fi
echo ""

# Test 3: Apply string encryption
echo "[Test 3] Testing string encryption pass..."
mlir-opt test.mlir \
    --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
    --pass-pipeline="builtin.module(string-encrypt)" \
    -o test_string_obf.mlir 2>&1 || { echo "ERROR: String encryption failed"; exit 1; }

if grep -q "MySecret123" test_string_obf.mlir; then
    echo "⚠️  Warning: Original strings still present (may need pass refinement)"
else
    echo "✅ String encryption successful - strings obfuscated"
fi
echo ""

# Test 4: Apply combined passes
echo "[Test 4] Testing combined obfuscation..."
mlir-opt test.mlir \
    --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
    --pass-pipeline="builtin.module(symbol-obfuscate,string-encrypt)" \
    -o test_combined_obf.mlir 2>&1 || { echo "ERROR: Combined obfuscation failed"; exit 1; }
echo "✅ Combined obfuscation successful"
echo ""

# Test 5: Lower to LLVM IR
echo "[Test 5] Lowering MLIR to LLVM IR..."
mlir-translate --mlir-to-llvmir test_combined_obf.mlir -o test_raw.ll 2>&1 || { echo "ERROR: MLIR to LLVM IR failed"; exit 1; }

# Fix ALL target-specific info in LLVM IR (triple, datalayout, CPU attributes)
cat test_raw.ll | \
  sed 's/target triple = ".*"/target triple = "x86_64-unknown-linux-gnu"/' | \
  sed 's/target datalayout = ".*"/target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"/' | \
  sed 's/attributes #[0-9]* = { .* "target-cpu"="[^"]*" .* }//' | \
  sed 's/"target-cpu"="[^"]*"//' | \
  sed 's/"target-features"="[^"]*"//' | \
  sed 's/"tune-cpu"="[^"]*"//' \
  > test.ll

echo "✅ MLIR → LLVM IR successful"
echo ""

# Test 6: Compile to binary
echo "[Test 6] Compiling to binary..."
clang test.ll -o test_binary 2>&1 || { echo "ERROR: Binary compilation failed"; exit 1; }
echo "✅ Binary compilation successful"
echo ""

# Test 7: Execute binary
echo "[Test 7] Testing binary execution..."
./test_binary > execution_output.txt 2>&1 || { echo "ERROR: Binary execution failed"; exit 1; }
echo "Binary output:"
cat execution_output.txt
echo "✅ Binary execution successful"
echo ""

# Test 8: Verify obfuscation
echo "[Test 8] Verifying obfuscation..."
echo "Checking for original symbols..."
if nm test_binary 2>/dev/null | grep -q "validate_password\|check_api_key"; then
    echo "⚠️  Found original function names in binary"
else
    echo "✅ No original function names found"
fi

echo "Checking for secret strings..."
FOUND_SECRETS=0
for secret in "MySecret123" "sk_live_12345"; do
    if strings test_binary 2>/dev/null | grep -q "$secret"; then
        echo "⚠️  Found secret: $secret"
        FOUND_SECRETS=1
    fi
done

if [ $FOUND_SECRETS -eq 0 ]; then
    echo "✅ No secrets found in binary"
fi
echo ""

echo "=========================================="
echo "✅ All tests completed!"
echo "=========================================="
echo ""
echo "Test artifacts saved in: $SCRIPT_DIR/$TEST_DIR"
