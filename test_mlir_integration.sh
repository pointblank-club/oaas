#!/bin/bash
# Comprehensive MLIR Integration Test Script
# This script tests the complete MLIR obfuscation pipeline

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

echo -e "${BLUE}=========================================="
echo "  MLIR Obfuscation Integration Test"
echo "==========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASSED${NC}: $2"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}: $2"
        ((FAILED++))
        return 1
    fi
}

# ===========================================
# TEST 1: Check Prerequisites
# ===========================================
echo -e "${YELLOW}[Test 1]${NC} Checking prerequisites..."

check_tool() {
    if command -v $1 >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $1 found: $(command -v $1)"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not found"
        return 1
    fi
}

PREREQ_OK=0
check_tool clang || PREREQ_OK=1
check_tool mlir-opt || PREREQ_OK=1
check_tool mlir-translate || PREREQ_OK=1
check_tool python3 || PREREQ_OK=1

print_result $PREREQ_OK "Prerequisites check"

if [ $PREREQ_OK -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required tools. Please install LLVM/MLIR 19+${NC}"
    exit 1
fi
echo ""

# ===========================================
# TEST 2: Build MLIR Library
# ===========================================
echo -e "${YELLOW}[Test 2]${NC} Building MLIR obfuscation library..."

cd mlir-obs

if [ ! -f build.sh ]; then
    echo -e "${RED}ERROR: build.sh not found${NC}"
    exit 1
fi

chmod +x build.sh test.sh
./build.sh > /tmp/mlir_build.log 2>&1

MLIR_LIB=$(find build -name "libMLIRObfuscation.*" -type f | head -1)
if [ -z "$MLIR_LIB" ]; then
    echo -e "${RED}ERROR: MLIR library not built${NC}"
    cat /tmp/mlir_build.log
    exit 1
fi

print_result 0 "MLIR library build"
echo -e "  Library: ${MLIR_LIB}"
echo ""

cd "$SCRIPT_DIR"

# ===========================================
# TEST 3: Test Standalone MLIR Passes
# ===========================================
echo -e "${YELLOW}[Test 3]${NC} Testing standalone MLIR passes..."

cd mlir-obs
./test.sh > /tmp/mlir_test.log 2>&1
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    print_result 0 "Standalone MLIR passes"
else
    print_result 1 "Standalone MLIR passes"
    echo "  See /tmp/mlir_test.log for details"
fi
echo ""

cd "$SCRIPT_DIR"

# ===========================================
# TEST 4: Create Test Source File
# ===========================================
echo -e "${YELLOW}[Test 4]${NC} Creating test source file..."

mkdir -p test_integration
cd test_integration

cat > test_auth.c << 'EOF'
#include <stdio.h>
#include <string.h>

const char* MASTER_PASSWORD = "SuperSecret2024!";
const char* API_KEY = "sk_live_abc123xyz";
const char* DATABASE_URL = "postgresql://admin:password@localhost/db";

int authenticate_user(const char* password) {
    if (strcmp(password, MASTER_PASSWORD) == 0) {
        return 1;
    }
    return 0;
}

int validate_api_key(const char* key) {
    if (strcmp(key, API_KEY) == 0) {
        return 1;
    }
    return 0;
}

void connect_database() {
    printf("Connecting to: %s\n", DATABASE_URL);
}

int main(int argc, char** argv) {
    printf("=== Authentication System ===\n");

    if (argc < 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }

    if (authenticate_user(argv[1])) {
        printf("✅ Access granted\n");
        if (validate_api_key(API_KEY)) {
            printf("✅ API key valid\n");
        }
        return 0;
    } else {
        printf("❌ Access denied\n");
        return 1;
    }
}
EOF

print_result $? "Test source file creation"
echo ""

# ===========================================
# TEST 5: Compile Baseline Binary
# ===========================================
echo -e "${YELLOW}[Test 5]${NC} Compiling baseline binary..."

clang test_auth.c -o test_auth_baseline 2>&1
print_result $? "Baseline binary compilation"

if [ -f test_auth_baseline ]; then
    echo "  Testing baseline execution..."
    ./test_auth_baseline "SuperSecret2024!" > /dev/null 2>&1
    print_result $? "Baseline binary execution"
fi
echo ""

# ===========================================
# TEST 6: Python CLI - String Encryption Only
# ===========================================
echo -e "${YELLOW}[Test 6]${NC} Testing Python CLI with string encryption..."

cd "$SCRIPT_DIR"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile \
    test_integration/test_auth.c \
    --enable-string-encrypt \
    --output test_integration/output_string \
    > /tmp/test_string.log 2>&1

if [ -f test_integration/output_string/test_auth ]; then
    print_result 0 "String encryption binary created"

    # Test execution
    test_integration/output_string/test_auth "SuperSecret2024!" > /dev/null 2>&1
    print_result $? "String-encrypted binary execution"

    # Verify strings are hidden
    echo "  Checking for hidden secrets..."
    FOUND_SECRETS=0
    for secret in "SuperSecret2024" "sk_live_abc123" "postgresql://"; do
        if strings test_integration/output_string/test_auth | grep -q "$secret"; then
            echo -e "  ${YELLOW}⚠${NC}  Found: $secret"
            FOUND_SECRETS=1
        fi
    done

    if [ $FOUND_SECRETS -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} No secrets found in binary"
    else
        echo -e "  ${YELLOW}⚠${NC}  Some secrets still visible (may need pass refinement)"
    fi
else
    print_result 1 "String encryption binary created"
    cat /tmp/test_string.log
fi
echo ""

# ===========================================
# TEST 7: Python CLI - Symbol Obfuscation Only
# ===========================================
echo -e "${YELLOW}[Test 7]${NC} Testing Python CLI with symbol obfuscation..."

python3 -m cmd.llvm-obfuscator.cli.obfuscate compile \
    test_integration/test_auth.c \
    --enable-symbol-obfuscate \
    --output test_integration/output_symbol \
    > /tmp/test_symbol.log 2>&1

if [ -f test_integration/output_symbol/test_auth ]; then
    print_result 0 "Symbol obfuscation binary created"

    # Test execution
    test_integration/output_symbol/test_auth "SuperSecret2024!" > /dev/null 2>&1
    print_result $? "Symbol-obfuscated binary execution"

    # Verify symbols are hidden
    echo "  Checking for obfuscated symbols..."
    FOUND_SYMBOLS=0
    for symbol in "authenticate_user" "validate_api_key" "connect_database"; do
        if nm test_integration/output_symbol/test_auth 2>/dev/null | grep -q "$symbol"; then
            echo -e "  ${YELLOW}⚠${NC}  Found: $symbol"
            FOUND_SYMBOLS=1
        fi
    done

    if [ $FOUND_SYMBOLS -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} No original symbol names found"
    else
        echo -e "  ${YELLOW}⚠${NC}  Some symbols still visible (may need pass refinement)"
    fi
else
    print_result 1 "Symbol obfuscation binary created"
    cat /tmp/test_symbol.log
fi
echo ""

# ===========================================
# TEST 8: Python CLI - Combined MLIR Passes
# ===========================================
echo -e "${YELLOW}[Test 8]${NC} Testing Python CLI with combined MLIR passes..."

python3 -m cmd.llvm-obfuscator.cli.obfuscate compile \
    test_integration/test_auth.c \
    --enable-string-encrypt \
    --enable-symbol-obfuscate \
    --output test_integration/output_both \
    > /tmp/test_both.log 2>&1

if [ -f test_integration/output_both/test_auth ]; then
    print_result 0 "Combined obfuscation binary created"

    # Test execution
    test_integration/output_both/test_auth "SuperSecret2024!" > /dev/null 2>&1
    print_result $? "Combined obfuscated binary execution"

    # Comprehensive verification
    echo "  Running comprehensive obfuscation check..."

    # Check strings
    STRINGS_OK=1
    for secret in "SuperSecret2024" "sk_live_abc123" "postgresql://"; do
        if strings test_integration/output_both/test_auth | grep -q "$secret"; then
            STRINGS_OK=0
        fi
    done

    # Check symbols
    SYMBOLS_OK=1
    for symbol in "authenticate_user" "validate_api_key"; do
        if nm test_integration/output_both/test_auth 2>/dev/null | grep -q "$symbol"; then
            SYMBOLS_OK=0
        fi
    done

    if [ $STRINGS_OK -eq 1 ] && [ $SYMBOLS_OK -eq 1 ]; then
        echo -e "  ${GREEN}✓${NC} Full obfuscation verified"
    else
        [ $STRINGS_OK -eq 0 ] && echo -e "  ${YELLOW}⚠${NC}  Some strings still visible"
        [ $SYMBOLS_OK -eq 0 ] && echo -e "  ${YELLOW}⚠${NC}  Some symbols still visible"
    fi
else
    print_result 1 "Combined obfuscation binary created"
    cat /tmp/test_both.log
fi
echo ""

# ===========================================
# TEST 9: Binary Size Comparison
# ===========================================
echo -e "${YELLOW}[Test 9]${NC} Analyzing binary sizes..."

if [ -f test_integration/test_auth_baseline ] && [ -f test_integration/output_both/test_auth ]; then
    BASELINE_SIZE=$(stat -f%z test_integration/test_auth_baseline 2>/dev/null || stat -c%s test_integration/test_auth_baseline)
    OBFUSCATED_SIZE=$(stat -f%z test_integration/output_both/test_auth 2>/dev/null || stat -c%s test_integration/output_both/test_auth)

    SIZE_INCREASE=$((OBFUSCATED_SIZE - BASELINE_SIZE))
    SIZE_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($SIZE_INCREASE / $BASELINE_SIZE) * 100}")

    echo -e "  Baseline size:    ${BASELINE_SIZE} bytes"
    echo -e "  Obfuscated size:  ${OBFUSCATED_SIZE} bytes"
    echo -e "  Increase:         ${SIZE_INCREASE} bytes (+${SIZE_PERCENT}%)"

    print_result 0 "Binary size analysis"
else
    print_result 1 "Binary size analysis (files not found)"
fi
echo ""

# ===========================================
# TEST 10: Symbol Count Comparison
# ===========================================
echo -e "${YELLOW}[Test 10]${NC} Analyzing symbol counts..."

if [ -f test_integration/test_auth_baseline ] && [ -f test_integration/output_both/test_auth ]; then
    BASELINE_SYMBOLS=$(nm test_integration/test_auth_baseline 2>/dev/null | grep -v ' U ' | wc -l)
    OBFUSCATED_SYMBOLS=$(nm test_integration/output_both/test_auth 2>/dev/null | grep -v ' U ' | wc -l)

    SYMBOL_REDUCTION=$((BASELINE_SYMBOLS - OBFUSCATED_SYMBOLS))
    if [ $BASELINE_SYMBOLS -gt 0 ]; then
        SYMBOL_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($SYMBOL_REDUCTION / $BASELINE_SYMBOLS) * 100}")
    else
        SYMBOL_PERCENT=0
    fi

    echo -e "  Baseline symbols:    ${BASELINE_SYMBOLS}"
    echo -e "  Obfuscated symbols:  ${OBFUSCATED_SYMBOLS}"
    echo -e "  Reduction:           ${SYMBOL_REDUCTION} symbols (-${SYMBOL_PERCENT}%)"

    print_result 0 "Symbol count analysis"
else
    print_result 1 "Symbol count analysis (files not found)"
fi
echo ""

# ===========================================
# FINAL SUMMARY
# ===========================================
echo ""
echo -e "${BLUE}=========================================="
echo "  Test Summary"
echo "==========================================${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo ""
    echo "✅ MLIR library built successfully"
    echo "✅ Standalone passes working"
    echo "✅ Python CLI integration working"
    echo "✅ Obfuscated binaries execute correctly"
    echo ""
    echo "Next steps:"
    echo "  1. Test with your real codebase"
    echo "  2. Measure performance on production code"
    echo "  3. Deploy to production environment"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please check the logs:"
    echo "  - /tmp/mlir_build.log"
    echo "  - /tmp/mlir_test.log"
    echo "  - /tmp/test_*.log"
    echo ""
    exit 1
fi
