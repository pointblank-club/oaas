#!/bin/bash
set -e

echo "=========================================="
echo "  MLIR Integration Tests"
echo "=========================================="
echo ""

# Create test results directory
mkdir -p /app/test_results
cd /app/test_results

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASSED${NC}: $2"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC}: $2"
        ((FAILED++))
        return 1
    fi
}

# ===========================================
# TEST 1: Create Test C File
# ===========================================
echo -e "${YELLOW}[Test 1]${NC} Creating test C file..."
cat > test_simple.c << 'EOF'
#include <stdio.h>
#include <string.h>

const char* SECRET_PASSWORD = "MySecret123!";
const char* API_KEY = "sk_live_12345";
const char* DATABASE_URL = "postgres://admin:pass@localhost/db";

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
print_result $? "Test file creation"

# ===========================================
# TEST 2: C → MLIR Conversion
# ===========================================
echo ""
echo -e "${YELLOW}[Test 2]${NC} Converting C to MLIR..."
clang -emit-llvm -S -emit-mlir test_simple.c -o test_simple.mlir 2>/dev/null
print_result $? "C to MLIR conversion"

# Verify MLIR file exists and has content
if [ -s test_simple.mlir ]; then
    echo "  → MLIR file size: $(wc -l < test_simple.mlir) lines"
else
    echo -e "${RED}ERROR: MLIR file is empty or doesn't exist${NC}"
    exit 1
fi

# ===========================================
# TEST 3: MLIR Symbol Obfuscation Pass
# ===========================================
echo ""
echo -e "${YELLOW}[Test 3]${NC} Running MLIR Symbol Obfuscation..."

# Check if MLIR pass library exists
if [ ! -f /app/mlir-obs/build/lib/libMLIRObfuscation.so ]; then
    echo -e "${RED}ERROR: MLIR obfuscation library not found at /app/mlir-obs/build/lib/libMLIRObfuscation.so${NC}"
    echo "Available files in build/lib:"
    ls -la /app/mlir-obs/build/lib/ || echo "Directory doesn't exist"
    exit 1
fi

mlir-opt test_simple.mlir \
    --load-pass-plugin=/app/mlir-obs/build/lib/libMLIRObfuscation.so \
    --pass-pipeline="builtin.module(symbol-obfuscation)" \
    -o test_symbol_obf.mlir 2>&1

if [ $? -eq 0 ] && [ -s test_symbol_obf.mlir ]; then
    print_result 0 "Symbol obfuscation pass"
    
    # Check if symbols were actually obfuscated
    if grep -q "validate_password" test_symbol_obf.mlir; then
        echo -e "${YELLOW}  ⚠ Warning: Original function name still present${NC}"
    else
        echo -e "${GREEN}  → Function names successfully obfuscated${NC}"
    fi
else
    print_result 1 "Symbol obfuscation pass"
fi

# ===========================================
# TEST 4: MLIR String Obfuscation Pass
# ===========================================
echo ""
echo -e "${YELLOW}[Test 4]${NC} Running MLIR String Obfuscation..."

mlir-opt test_simple.mlir \
    --load-pass-plugin=/app/mlir-obs/build/lib/libMLIRObfuscation.so \
    --pass-pipeline="builtin.module(string-obfuscation)" \
    -o test_string_obf.mlir 2>&1

if [ $? -eq 0 ] && [ -s test_string_obf.mlir ]; then
    print_result 0 "String obfuscation pass"
    
    # Check if strings were actually obfuscated
    if grep -q "MySecret123" test_string_obf.mlir; then
        echo -e "${YELLOW}  ⚠ Warning: Original secret string still present${NC}"
    else
        echo -e "${GREEN}  → Secret strings successfully obfuscated${NC}"
    fi
else
    print_result 1 "String obfuscation pass"
fi

# ===========================================
# TEST 5: Combined MLIR Passes
# ===========================================
echo ""
echo -e "${YELLOW}[Test 5]${NC} Running Combined MLIR Obfuscation..."

mlir-opt test_simple.mlir \
    --load-pass-plugin=/app/mlir-obs/build/lib/libMLIRObfuscation.so \
    --pass-pipeline="builtin.module(symbol-obfuscation,string-obfuscation)" \
    -o test_combined_obf.mlir 2>&1

print_result $? "Combined obfuscation passes"

# ===========================================
# TEST 6: MLIR → LLVM IR Lowering
# ===========================================
echo ""
echo -e "${YELLOW}[Test 6]${NC} Lowering MLIR to LLVM IR..."

mlir-translate --mlir-to-llvmir test_combined_obf.mlir -o test.ll 2>&1
print_result $? "MLIR to LLVM IR conversion"

if [ -s test.ll ]; then
    echo "  → LLVM IR file size: $(wc -l < test.ll) lines"
fi

# ===========================================
# TEST 7: Compile to Binary
# ===========================================
echo ""
echo -e "${YELLOW}[Test 7]${NC} Compiling to binary..."

clang test.ll -o test_binary 2>&1
print_result $? "Binary compilation"

# ===========================================
# TEST 8: Binary Verification - Symbols
# ===========================================
echo ""
echo -e "${YELLOW}[Test 8]${NC} Verifying symbol obfuscation..."

SYMBOL_COUNT=$(nm test_binary 2>/dev/null | grep -v ' U ' | wc -l)
echo "  → Total symbols in binary: $SYMBOL_COUNT"

# Check for original function names
if nm test_binary 2>/dev/null | grep -q "validate_password\|check_api_key"; then
    print_result 1 "Symbol obfuscation verification"
    echo "  → Found original function names in binary"
else
    print_result 0 "Symbol obfuscation verification"
    echo "  → No original function names found"
fi

# ===========================================
# TEST 9: Binary Verification - Strings
# ===========================================
echo ""
echo -e "${YELLOW}[Test 9]${NC} Verifying string obfuscation..."

# Check for secret strings
FOUND_SECRETS=0
for secret in "MySecret123" "sk_live_12345" "postgres://"; do
    if strings test_binary 2>/dev/null | grep -q "$secret"; then
        echo -e "${RED}  → Found secret: $secret${NC}"
        FOUND_SECRETS=1
    fi
done

if [ $FOUND_SECRETS -eq 0 ]; then
    print_result 0 "String obfuscation verification"
    echo "  → No secrets found in binary"
else
    print_result 1 "String obfuscation verification"
fi

# ===========================================
# TEST 10: Binary Execution
# ===========================================
echo ""
echo -e "${YELLOW}[Test 10]${NC} Testing binary execution..."

./test_binary > /dev/null 2>&1
print_result $? "Binary execution"

# ===========================================
# TEST 11: Python CLI Integration (Symbol Only)
# ===========================================
echo ""
echo -e "${YELLOW}[Test 11]${NC} Testing Python CLI - Symbol Obfuscation Only..."

cd /app
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile \
    /app/test_results/test_simple.c \
    --enable-mlir-symbol-obfuscation \
    --output /app/test_results/cli_output_symbol \
    2>&1 | head -20

if [ -f /app/test_results/cli_output_symbol/test_simple ]; then
    print_result 0 "Python CLI symbol obfuscation"
else
    print_result 1 "Python CLI symbol obfuscation"
fi

# ===========================================
# TEST 12: Python CLI Integration (String Only)
# ===========================================
echo ""
echo -e "${YELLOW}[Test 12]${NC} Testing Python CLI - String Obfuscation Only..."

python3 -m cmd.llvm-obfuscator.cli.obfuscate compile \
    /app/test_results/test_simple.c \
    --enable-mlir-string-obfuscation \
    --output /app/test_results/cli_output_string \
    2>&1 | head -20

if [ -f /app/test_results/cli_output_string/test_simple ]; then
    print_result 0 "Python CLI string obfuscation"
else
    print_result 1 "Python CLI string obfuscation"
fi

# ===========================================
# TEST 13: Python CLI Integration (Combined)
# ===========================================
echo ""
echo -e "${YELLOW}[Test 13]${NC} Testing Python CLI - Combined Obfuscation..."

python3 -m cmd.llvm-obfuscator.cli.obfuscate compile \
    /app/test_results/test_simple.c \
    --enable-mlir-symbol-obfuscation \
    --enable-mlir-string-obfuscation \
    --output /app/test_results/cli_output_combined \
    2>&1 | head -20

if [ -f /app/test_results/cli_output_combined/test_simple ]; then
    print_result 0 "Python CLI combined obfuscation"
    
    # Verify the CLI output binary
    if ! nm /app/test_results/cli_output_combined/test_simple | grep -q "validate_password" && \
       ! strings /app/test_results/cli_output_combined/test_simple | grep -q "MySecret123"; then
        echo -e "${GREEN}  → CLI output properly obfuscated${NC}"
    fi
else
    print_result 1 "Python CLI combined obfuscation"
fi

# ===========================================
# FINAL SUMMARY
# ===========================================
echo ""
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    exit 1
fi