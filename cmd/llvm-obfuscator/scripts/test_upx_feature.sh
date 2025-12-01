#!/bin/bash
# Comprehensive UPX Packer Feature Test
# Tests UPX integration with the LLVM obfuscator

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            UPX Packer Feature Test                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check dependencies
echo -e "${BLUE}[1/6]${NC} Checking dependencies..."

if ! command -v clang &> /dev/null; then
    echo -e "${RED}❌ clang not found${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi

if ! command -v upx &> /dev/null; then
    echo -e "${RED}❌ UPX not installed!${NC}"
    echo ""
    echo "Install UPX:"
    echo "  Linux:   sudo apt install upx-ucl"
    echo "  macOS:   brew install upx"
    exit 1
fi

UPX_VERSION=$(upx --version | head -n1)
echo -e "${GREEN}✓${NC} Found: $UPX_VERSION"
echo ""

# Create test source
echo -e "${BLUE}[2/6]${NC} Creating test source file..."

TEST_DIR=$(mktemp -d)
TEST_SOURCE="$TEST_DIR/test_upx.c"

cat > "$TEST_SOURCE" << 'EOF'
#include <stdio.h>
#include <string.h>

const char* SECRET = "MySecretPassword123!";
const char* API_KEY = "sk_live_abc123xyz";

int validate(const char* input) {
    return strcmp(input, SECRET) == 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }
    
    if (validate(argv[1])) {
        printf("Access granted!\n");
        return 0;
    } else {
        printf("Access denied!\n");
        return 1;
    }
}
EOF

echo -e "${GREEN}✓${NC} Created: $TEST_SOURCE"
echo ""

# Test 1: Direct UPX packer test
echo -e "${BLUE}[3/6]${NC} Test 1: Direct UPX Packer API..."

# Compile a simple binary first
BINARY="$TEST_DIR/test_binary"
clang "$TEST_SOURCE" -o "$BINARY" -O2

ORIGINAL_SIZE=$(stat -f%z "$BINARY" 2>/dev/null || stat -c%s "$BINARY" 2>/dev/null)
echo -e "${GREEN}✓${NC} Compiled binary: $ORIGINAL_SIZE bytes"

# Test UPX packer directly
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '$(pwd)')

from core.upx_packer import UPXPacker

packer = UPXPacker()
binary = Path('$BINARY')

print('Testing UPX packer...')
result = packer.pack(
    binary,
    compression_level='best',
    use_lzma=True,
    force=True,
    preserve_original=True
)

if result and result.get('status') == 'success':
    print(f\"✅ UPX packing successful!\")
    print(f\"   Original: {result['original_size']} bytes\")
    print(f\"   Packed:   {result['packed_size']} bytes\")
    print(f\"   Ratio:    {result['compression_ratio']:.1f}% reduction\")
    
    # Verify it's packed
    if packer._is_packed(binary):
        print(f\"   ✅ Binary is UPX-packed (verified)\")
    else:
        print(f\"   ⚠️  Binary packing not detected\")
else:
    print(f\"⚠️  UPX packing result: {result}\")
" 2>&1

PACKED_SIZE=$(stat -f%z "$BINARY" 2>/dev/null || stat -c%s "$BINARY" 2>/dev/null)
echo ""

# Test 2: Obfuscator integration without UPX
echo -e "${BLUE}[4/6]${NC} Test 2: Obfuscator WITHOUT UPX..."

OUTPUT_NO_UPX="$TEST_DIR/output_no_upx"
mkdir -p "$OUTPUT_NO_UPX"

python3 -m cli.obfuscate compile "$TEST_SOURCE" \
    --output "$OUTPUT_NO_UPX" \
    --level 2 \
    --string-encryption \
    --custom-flags "-O2" \
    2>&1 | grep -E "(INFO|WARNING|ERROR|✓|✗)" || true

BINARY_NO_UPX="$OUTPUT_NO_UPX/test_upx"
if [ -f "$BINARY_NO_UPX" ]; then
    SIZE_NO_UPX=$(stat -f%z "$BINARY_NO_UPX" 2>/dev/null || stat -c%s "$BINARY_NO_UPX" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Binary created: $SIZE_NO_UPX bytes"
    
    # Test execution
    if "$BINARY_NO_UPX" "wrong" 2>&1 | grep -q "denied"; then
        echo -e "${GREEN}✓${NC} Binary executes correctly"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Binary not found"
    SIZE_NO_UPX=0
fi
echo ""

# Test 3: Obfuscator integration WITH UPX
echo -e "${BLUE}[5/6]${NC} Test 3: Obfuscator WITH UPX..."

OUTPUT_WITH_UPX="$TEST_DIR/output_with_upx"
mkdir -p "$OUTPUT_WITH_UPX"

python3 -m cli.obfuscate compile "$TEST_SOURCE" \
    --output "$OUTPUT_WITH_UPX" \
    --level 2 \
    --string-encryption \
    --enable-upx \
    --upx-compression best \
    --upx-lzma \
    --custom-flags "-O2" \
    2>&1 | grep -E "(INFO|WARNING|ERROR|UPX|✓|✗)" || true

BINARY_WITH_UPX="$OUTPUT_WITH_UPX/test_upx"
if [ -f "$BINARY_WITH_UPX" ]; then
    SIZE_WITH_UPX=$(stat -f%z "$BINARY_WITH_UPX" 2>/dev/null || stat -c%s "$BINARY_WITH_UPX" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Binary created: $SIZE_WITH_UPX bytes"
    
    # Verify UPX packing
    if upx -t "$BINARY_WITH_UPX" 2>&1 | grep -q "OK"; then
        echo -e "${GREEN}✓${NC} Binary is UPX-packed and valid"
    else
        echo -e "${YELLOW}⚠${NC}  Binary may not be UPX-packed"
    fi
    
    # Test execution
    if "$BINARY_WITH_UPX" "wrong" 2>&1 | grep -q "denied"; then
        echo -e "${GREEN}✓${NC} Binary executes correctly (even when UPX-packed!)"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Binary not found"
    SIZE_WITH_UPX=0
fi
echo ""

# Test 4: Compression levels
echo -e "${BLUE}[6/6]${NC} Test 4: Different compression levels..."

# Create a fresh binary for testing
TEST_BINARY="$TEST_DIR/compression_test"
clang "$TEST_SOURCE" -o "$TEST_BINARY" -O2

python3 -c "
import sys
from pathlib import Path
import shutil
sys.path.insert(0, '$(pwd)')

from core.upx_packer import UPXPacker

packer = UPXPacker()
base_binary = Path('$TEST_BINARY')

levels = ['fast', 'default', 'best']
results = {}

for level in levels:
    # Make a copy for each test
    test_bin = Path('$TEST_DIR') / f'test_{level}'
    shutil.copy(base_binary, test_bin)
    
    result = packer.pack(
        test_bin,
        compression_level=level,
        use_lzma=True,
        force=True,
        preserve_original=False
    )
    
    if result and result.get('status') == 'success':
        results[level] = result['compression_ratio']
        print(f\"  {level:8s}: {result['compression_ratio']:5.1f}% reduction ({result['packed_size']} bytes)\")
    else:
        print(f\"  {level:8s}: Failed\")

if len(results) > 1:
    best = max(results, key=results.get)
    print(f\"\\n  Best compression: {best} ({results[best]:.1f}%)\")
" 2>&1

echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Test Summary                                ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"

if [ $SIZE_NO_UPX -gt 0 ] && [ $SIZE_WITH_UPX -gt 0 ]; then
    REDUCTION=$((100 - (SIZE_WITH_UPX * 100 / SIZE_NO_UPX)))
    echo -e "${BLUE}║${NC}  Without UPX:     ${SIZE_NO_UPX} bytes"
    echo -e "${BLUE}║${NC}  With UPX:        ${GREEN}${SIZE_WITH_UPX} bytes${NC}"
    echo -e "${BLUE}║${NC}  Compression:     ${GREEN}${REDUCTION}% reduction${NC}"
    
    if [ $REDUCTION -gt 0 ]; then
        echo -e "${BLUE}║${NC}  Status:          ${GREEN}✅ SUCCESS${NC}"
    else
        echo -e "${BLUE}║${NC}  Status:          ${YELLOW}⚠️  No compression${NC}"
    fi
else
    echo -e "${BLUE}║${NC}  Status:          ${YELLOW}⚠️  Could not calculate${NC}"
fi

echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Test files in: $TEST_DIR"
echo "Clean up with: rm -rf $TEST_DIR"
echo ""
echo -e "${GREEN}✅ UPX Packer feature test complete!${NC}"

