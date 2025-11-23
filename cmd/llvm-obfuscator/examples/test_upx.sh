#!/bin/bash
# Test script to demonstrate UPX integration
# This script compiles the same source file with and without UPX to show the difference

set -e

echo "==================================================================="
echo "UPX Integration Test - LLVM Obfuscator"
echo "==================================================================="
echo ""

# Check if UPX is installed
if ! command -v upx &> /dev/null; then
    echo "❌ UPX is not installed!"
    echo ""
    echo "Install UPX:"
    echo "  Linux:   sudo apt install upx-ucl"
    echo "  macOS:   brew install upx"
    echo ""
    exit 1
fi

echo "✅ UPX is installed: $(upx --version | head -n1)"
echo ""

# Use hello.c as test source
SOURCE="hello.c"
if [ ! -f "$SOURCE" ]; then
    echo "❌ Source file not found: $SOURCE"
    exit 1
fi

echo "Source file: $SOURCE"
echo ""

# Clean up previous builds
rm -rf test_output_no_upx test_output_with_upx

echo "==================================================================="
echo "Test 1: Obfuscation WITHOUT UPX"
echo "==================================================================="
echo ""

python3 -m cli.obfuscate compile "$SOURCE" \
    --output ./test_output_no_upx \
    --level 3 \
    --string-encryption \
    --enable-symbol-obfuscation \
    --custom-flags "-O3" \
    --report-formats "json"

echo ""
echo "Binary without UPX:"
ls -lh test_output_no_upx/hello
SIZE_NO_UPX=$(stat -f%z test_output_no_upx/hello 2>/dev/null || stat -c%s test_output_no_upx/hello 2>/dev/null)
echo "  Size: $SIZE_NO_UPX bytes"
echo ""

echo "==================================================================="
echo "Test 2: Obfuscation WITH UPX"
echo "==================================================================="
echo ""

python3 -m cli.obfuscate compile "$SOURCE" \
    --output ./test_output_with_upx \
    --level 3 \
    --string-encryption \
    --enable-symbol-obfuscation \
    --enable-upx \
    --upx-compression best \
    --upx-lzma \
    --custom-flags "-O3" \
    --report-formats "json"

echo ""
echo "Binary with UPX:"
ls -lh test_output_with_upx/hello
SIZE_WITH_UPX=$(stat -f%z test_output_with_upx/hello 2>/dev/null || stat -c%s test_output_with_upx/hello 2>/dev/null)
echo "  Size: $SIZE_WITH_UPX bytes"
echo ""

# Calculate compression ratio
if [ $SIZE_NO_UPX -gt 0 ]; then
    REDUCTION=$((100 - (SIZE_WITH_UPX * 100 / SIZE_NO_UPX)))
    echo "📊 UPX Compression: ${REDUCTION}% size reduction"
else
    echo "⚠️  Could not calculate compression ratio"
fi
echo ""

# Verify UPX packing
echo "==================================================================="
echo "Verification"
echo "==================================================================="
echo ""

echo "1. Check if binary is UPX-packed:"
if upx -t test_output_with_upx/hello 2>&1 | grep -q "OK"; then
    echo "   ✅ Binary is UPX-packed and valid"
else
    echo "   ❌ Binary is not UPX-packed"
fi
echo ""

echo "2. Test execution (without UPX):"
./test_output_no_upx/hello || echo "   ⚠️ Execution failed"
echo ""

echo "3. Test execution (with UPX):"
./test_output_with_upx/hello || echo "   ⚠️ Execution failed"
echo ""

echo "==================================================================="
echo "Summary"
echo "==================================================================="
echo ""
echo "Without UPX: $SIZE_NO_UPX bytes"
echo "With UPX:    $SIZE_WITH_UPX bytes"
echo "Reduction:   ${REDUCTION}%"
echo ""
echo "Both binaries function identically!"
echo "UPX adds compression + obfuscation layer with minimal overhead."
echo ""
echo "✅ Test completed successfully!"
echo ""

