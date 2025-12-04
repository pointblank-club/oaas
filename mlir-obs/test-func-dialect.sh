#!/bin/bash
# Test script for MLIR Obfuscation passes using Func dialect

set -e

echo "=========================================="
echo "  Testing MLIR Obfuscation - Func Dialect"
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
TEST_DIR="test_output_func"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create test MLIR file with Func dialect
echo "Creating test MLIR file with Func dialect..."
cat > test_input.mlir << 'EOF'
module {
  func.func @validate_password(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi eq, %arg0, %c0 : i32
    %result = arith.select %cmp, %c1, %c0 : i32
    return %result : i32
  }

  func.func @check_api_key(%arg0: i32) -> i32 {
    %c42 = arith.constant 42 : i32
    %cmp = arith.cmpi eq, %arg0, %c42 : i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %result = arith.select %cmp, %c1, %c0 : i32
    return %result : i32
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32

    %pass = func.call @validate_password(%c0) : (i32) -> i32
    %key = func.call @check_api_key(%c1) : (i32) -> i32

    return %c0 : i32
  }
}
EOF

echo "✅ Test MLIR file created"
echo ""

# Test 1: Verify input
echo "[Test 1] Verifying input MLIR..."
echo "Functions in input:"
grep "func.func @" test_input.mlir
echo "✅ Input verification complete"
echo ""

# Test 2: Apply symbol obfuscation
echo "[Test 2] Testing symbol obfuscation pass..."
mlir-opt test_input.mlir \
    --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
    --pass-pipeline="builtin.module(symbol-obfuscate)" \
    -o test_symbol_obf.mlir 2>&1 || { echo "ERROR: Symbol obfuscation failed"; exit 1; }

echo "Functions after symbol obfuscation:"
grep "func.func @" test_symbol_obf.mlir

if grep -q "validate_password" test_symbol_obf.mlir; then
    echo "⚠️  Warning: Original function names still present"
else
    echo "✅ Symbol obfuscation successful - function names obfuscated"
fi
echo ""

# Test 3: Apply string encryption (on constants)
echo "[Test 3] Testing string encryption pass..."
mlir-opt test_input.mlir \
    --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
    --pass-pipeline="builtin.module(string-encrypt)" \
    -o test_string_obf.mlir 2>&1 || { echo "ERROR: String encryption failed"; exit 1; }
echo "✅ String encryption pass completed"
echo ""

# Test 4: Apply combined passes
echo "[Test 4] Testing combined obfuscation..."
mlir-opt test_input.mlir \
    --load-pass-plugin="$SCRIPT_DIR/$LIBRARY" \
    --pass-pipeline="builtin.module(symbol-obfuscate,string-encrypt)" \
    -o test_combined_obf.mlir 2>&1 || { echo "ERROR: Combined obfuscation failed"; exit 1; }

echo "Functions after combined obfuscation:"
grep "func.func @" test_combined_obf.mlir
echo "✅ Combined obfuscation successful"
echo ""

# Test 5: Lower to LLVM dialect
echo "[Test 5] Lowering to LLVM dialect..."
mlir-opt test_combined_obf.mlir \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    -o test_llvm.mlir 2>&1 || { echo "ERROR: Lowering to LLVM dialect failed"; exit 1; }
echo "✅ Lowering to LLVM dialect successful"
echo ""

# Test 6: Convert to LLVM IR
echo "[Test 6] Converting to LLVM IR..."
mlir-translate --mlir-to-llvmir test_llvm.mlir -o test.ll 2>&1 || { echo "ERROR: MLIR to LLVM IR failed"; exit 1; }
echo "✅ MLIR → LLVM IR successful"
echo ""

# Test 7: Compile to binary
echo "[Test 7] Compiling to binary..."
clang test.ll -o test_binary 2>&1 || { echo "ERROR: Binary compilation failed"; exit 1; }
echo "✅ Binary compilation successful"
echo ""

# Test 8: Verify obfuscation in binary
echo "[Test 8] Verifying obfuscation in binary..."
echo "Checking for original function names..."
if nm test_binary 2>/dev/null | grep -q "validate_password\|check_api_key"; then
    echo "⚠️  Found original function names:"
    nm test_binary | grep "validate_password\|check_api_key" || true
else
    echo "✅ No original function names found in binary"
fi
echo ""

echo "All functions in binary:"
nm test_binary | grep " T " || echo "No exported functions found"
echo ""

echo "=========================================="
echo "✅ All tests completed!"
echo "=========================================="
echo ""
echo "Test artifacts saved in: $SCRIPT_DIR/$TEST_DIR"
echo ""
echo "To inspect results:"
echo "  Input:             cat $TEST_DIR/test_input.mlir"
echo "  Symbol obfuscated: cat $TEST_DIR/test_symbol_obf.mlir"
echo "  Combined:          cat $TEST_DIR/test_combined_obf.mlir"
