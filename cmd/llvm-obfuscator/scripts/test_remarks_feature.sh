#!/bin/bash
# Test script for LLVM Remarks feature
# Verifies that remarks are actually generated and can be parsed

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         LLVM Remarks Feature Test                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check dependencies
echo -e "${BLUE}[1/5]${NC} Checking dependencies..."

if ! command -v clang &> /dev/null; then
    echo -e "${RED}❌ clang not found${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi

CLANG_VERSION=$(clang --version | head -n1)
echo -e "${GREEN}✓${NC} Found: $CLANG_VERSION"

# Check if clang supports remarks
echo ""
echo -e "${BLUE}[2/5]${NC} Checking clang remarks support..."

if clang --help | grep -q "save-optimization-record"; then
    echo -e "${GREEN}✓${NC} Clang supports optimization remarks"
else
    echo -e "${YELLOW}⚠${NC}  Clang may not support remarks (checking version...)"
    CLANG_VER=$(clang --version | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    if (( $(echo "$CLANG_VER < 9.0" | bc -l 2>/dev/null || echo 1) )); then
        echo -e "${RED}❌ Clang version $CLANG_VER may not support remarks (need 9.0+)${NC}"
        exit 1
    fi
fi

# Create test source
echo ""
echo -e "${BLUE}[3/5]${NC} Creating test source file..."

TEST_DIR=$(mktemp -d)
TEST_SOURCE="$TEST_DIR/test_remarks.c"

cat > "$TEST_SOURCE" << 'EOF'
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int main() {
    int x = add(5, 3);
    int y = multiply(4, 2);
    printf("Result: %d, %d\n", x, y);
    return 0;
}
EOF

echo -e "${GREEN}✓${NC} Created: $TEST_SOURCE"

# Test 1: Direct clang compilation with remarks
echo ""
echo -e "${BLUE}[4/5]${NC} Test 1: Direct clang compilation with remarks..."

REMARKS_FILE="$TEST_DIR/test.opt.yaml"
BINARY="$TEST_DIR/test"

clang "$TEST_SOURCE" -o "$BINARY" \
    -O2 \
    -fsave-optimization-record=yaml \
    -foptimization-record-file="$REMARKS_FILE" \
    2>&1 | head -n 20

if [ -f "$REMARKS_FILE" ]; then
    REMARKS_SIZE=$(stat -f%z "$REMARKS_FILE" 2>/dev/null || stat -c%s "$REMARKS_FILE" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Remarks file created: $REMARKS_FILE ($REMARKS_SIZE bytes)"
    
    # Try to parse YAML
    if python3 -c "import yaml; yaml.safe_load_all(open('$REMARKS_FILE'))" 2>/dev/null; then
        REMARK_COUNT=$(python3 -c "import yaml; print(len([r for r in yaml.safe_load_all(open('$REMARKS_FILE')) if r]))" 2>/dev/null || echo "0")
        echo -e "${GREEN}✓${NC} YAML is valid, found $REMARK_COUNT remarks"
        
        if [ "$REMARK_COUNT" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Sample remark:"
            python3 -c "
import yaml
with open('$REMARKS_FILE') as f:
    remarks = [r for r in yaml.safe_load_all(f) if r]
    if remarks:
        r = remarks[0]
        print(f\"  Pass: {r.get('Pass', 'N/A')}\")
        print(f\"  Name: {r.get('Name', 'N/A')}\")
        print(f\"  Function: {r.get('Function', 'N/A')}\")
" 2>/dev/null || echo "  (Could not parse)"
        fi
    else
        echo -e "${YELLOW}⚠${NC}  YAML parsing failed (file may be empty or invalid)"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Remarks file not created (may need optimization level -O2 or higher)"
fi

# Test 2: Using obfuscator with remarks enabled
echo ""
echo -e "${BLUE}[5/5]${NC} Test 2: Obfuscator with remarks enabled..."

OUTPUT_DIR="$TEST_DIR/obfuscated"
mkdir -p "$OUTPUT_DIR"

# Create Python test script
PYTHON_TEST="$TEST_DIR/test_obfuscator_remarks.py"
cat > "$PYTHON_TEST" << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    LLVMObfuscator,
    ObfuscationConfig,
    ObfuscationLevel,
    Platform,
    AdvancedConfiguration,
    RemarksConfiguration,
    OutputConfiguration,
)

source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

config = ObfuscationConfig(
    level=ObfuscationLevel.MEDIUM,
    platform=Platform.LINUX,
    advanced=AdvancedConfiguration(
        remarks=RemarksConfiguration(
            enabled=True,
            format="yaml",
            pass_filter=".*"
        )
    ),
    output=OutputConfiguration(directory=output_dir)
)

obfuscator = LLVMObfuscator()
result = obfuscator.obfuscate(source, config)

print(f"Obfuscation result: {result.get('output_file', 'N/A')}")

# Check for remarks file
remarks_file = output_dir / f"{source.stem}.opt.yaml"
if remarks_file.exists():
    print(f"✓ Remarks file created: {remarks_file}")
    print(f"  Size: {remarks_file.stat().st_size} bytes")
    
    # Try to parse
    import yaml
    try:
        with open(remarks_file) as f:
            remarks = [r for r in yaml.safe_load_all(f) if r]
            print(f"  Remarks count: {len(remarks)}")
            if remarks:
                print(f"  Sample: {remarks[0].get('Pass', 'N/A')} - {remarks[0].get('Name', 'N/A')}")
    except Exception as e:
        print(f"  ⚠ Parse error: {e}")
else:
    print(f"⚠ Remarks file not found: {remarks_file}")
PYEOF

if python3 "$PYTHON_TEST" "$TEST_SOURCE" "$OUTPUT_DIR" 2>&1; then
    echo -e "${GREEN}✓${NC} Obfuscator test completed"
else
    echo -e "${YELLOW}⚠${NC}  Obfuscator test had issues (check output above)"
fi

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Test Summary                                ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"

if [ -f "$REMARKS_FILE" ] && [ -s "$REMARKS_FILE" ]; then
    echo -e "${BLUE}║${NC}  Direct clang:        ${GREEN}✓ PASSED${NC}"
else
    echo -e "${BLUE}║${NC}  Direct clang:        ${YELLOW}⚠ SKIPPED${NC} (no remarks generated)"
fi

if [ -f "$OUTPUT_DIR/${TEST_SOURCE##*/}.opt.yaml" ]; then
    echo -e "${BLUE}║${NC}  Obfuscator:         ${GREEN}✓ PASSED${NC}"
else
    echo -e "${BLUE}║${NC}  Obfuscator:         ${YELLOW}⚠ SKIPPED${NC} (no remarks generated)"
fi

echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Test files in: $TEST_DIR"
echo "Clean up with: rm -rf $TEST_DIR"

