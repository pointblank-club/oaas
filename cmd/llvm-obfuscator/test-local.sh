#!/bin/bash
# Local end-to-end test script for OAAS
# Tests both MLIR and OLLVM pipelines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGINS_DIR="$SCRIPT_DIR/plugins/linux-x86_64"

# Add bundled MLIR tools to PATH (but NOT clang - use system clang for compilation)
# The bundled clang from pb server lacks proper header files
export PATH="$PLUGINS_DIR:$PATH"
export LD_LIBRARY_PATH="$PLUGINS_DIR:$LD_LIBRARY_PATH"

# Ensure system clang is used for compilation (override bundled clang)
if command -v /usr/bin/clang &> /dev/null; then
    export PATH="/usr/bin:$PATH"
fi

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "=============================================="
echo "  OAAS Local End-to-End Test"
echo "=============================================="
echo ""

# Check tools
echo "[1/6] Checking tools..."
for tool in clang mlir-opt mlir-translate opt; do
    if command -v "$tool" &> /dev/null; then
        echo "  ✅ $tool: $(which $tool)"
    else
        echo "  ❌ $tool: NOT FOUND"
        exit 1
    fi
done

# Check plugins
echo ""
echo "[2/6] Checking plugins..."
if [ -f "$PLUGINS_DIR/LLVMObfuscationPlugin.so" ]; then
    echo "  ✅ OLLVM Plugin: $PLUGINS_DIR/LLVMObfuscationPlugin.so"
else
    echo "  ❌ OLLVM Plugin: NOT FOUND"
fi

if [ -f "$PLUGINS_DIR/MLIRObfuscation.so" ]; then
    echo "  ✅ MLIR Plugin: $PLUGINS_DIR/MLIRObfuscation.so"
else
    echo "  ❌ MLIR Plugin: NOT FOUND"
fi

# Create test file
echo ""
echo "[3/6] Creating test source file..."
TEST_DIR=$(mktemp -d)
cat > "$TEST_DIR/test.c" << 'EOF'
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int secret_function(int x) {
    return x * 2 + 42;
}

int main() {
    int result = add(10, 20);
    result = secret_function(result);
    return result;
}
EOF
echo "  Created: $TEST_DIR/test.c"

# Test 1: Standard Clang Pipeline (MLIR disabled)
echo ""
echo "[4/6] Testing Standard Pipeline (MLIR disabled)..."
cd "$SCRIPT_DIR"
python3 -m cli.obfuscate compile "$TEST_DIR/test.c" \
    --output "$TEST_DIR/standard_output" \
    --enable-symbol-obfuscate \
    --enable-string-encrypt \
    --mlir-frontend clang 2>&1 | head -20

if [ -f "$TEST_DIR/standard_output/test" ]; then
    echo "  ✅ Standard pipeline: Binary created"
    "$TEST_DIR/standard_output/test" || true
    echo "  Exit code: $?"
else
    echo "  ❌ Standard pipeline: Binary NOT created"
fi

# Test 2: ClangIR Pipeline (MLIR enabled)
echo ""
echo "[5/6] Testing ClangIR Pipeline (MLIR enabled)..."

# First check if ClangIR is available
if clang -fclangir --help 2>&1 | grep -q "unknown argument"; then
    echo "  ⚠️  ClangIR not available (clang -fclangir not supported)"
    echo "  Skipping ClangIR test - will fall back to standard pipeline"
else
    python3 -m cli.obfuscate compile "$TEST_DIR/test.c" \
        --output "$TEST_DIR/clangir_output" \
        --enable-symbol-obfuscate \
        --enable-string-encrypt \
        --mlir-frontend clangir 2>&1 | head -20

    if [ -f "$TEST_DIR/clangir_output/test" ]; then
        echo "  ✅ ClangIR pipeline: Binary created"
        "$TEST_DIR/clangir_output/test" || true
        echo "  Exit code: $?"
    else
        echo "  ❌ ClangIR pipeline: Binary NOT created"
    fi
fi

# Test 3: OLLVM Passes (Layer 3)
echo ""
echo "[6/6] Testing OLLVM Passes (Layer 3)..."
python3 -m cli.obfuscate compile "$TEST_DIR/test.c" \
    --output "$TEST_DIR/ollvm_output" \
    --enable-flattening \
    --enable-substitution \
    --mlir-frontend clang 2>&1 | head -20

if [ -f "$TEST_DIR/ollvm_output/test" ]; then
    echo "  ✅ OLLVM pipeline: Binary created"
    "$TEST_DIR/ollvm_output/test" || true
    echo "  Exit code: $?"
    echo ""
    echo "  Checking symbols:"
    nm "$TEST_DIR/ollvm_output/test" 2>/dev/null | grep -E "add|secret" || echo "  ✅ Symbols obfuscated!"
else
    echo "  ❌ OLLVM pipeline: Binary NOT created"
fi

# Summary
echo ""
echo "=============================================="
echo "  Test Complete"
echo "=============================================="
echo "  Test directory: $TEST_DIR"
echo ""

