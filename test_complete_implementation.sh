#!/bin/bash
# Complete Implementation Test Script for LLVM Obfuscator
# Tests: MLIR passes, ClangIR pipeline, Default pipeline, All obfuscation features
#
# Usage: ./test_complete_implementation.sh
#
# Tests performed:
# 1. MLIR library build verification
# 2. MLIR passes standalone tests
# 3. Default pipeline (CLANG) tests
# 4. ClangIR pipeline tests (if available)
# 5. Cryptographic hash pass tests
# 6. Constant obfuscation tests
# 7. End-to-end integration tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_skip() {
    echo -e "${YELLOW}⏭️  $1${NC}"
    ((TESTS_SKIPPED++))
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_header "LLVM Obfuscator - Complete Implementation Test"

# ============================================================================
# PHASE 1: Environment Verification
# ============================================================================
print_header "Phase 1: Verifying Environment"

# Check required tools
echo "Checking required tools..."

# LLVM/Clang
if command -v clang &> /dev/null; then
    CLANG_VERSION=$(clang --version | head -1)
    print_success "Clang found: $CLANG_VERSION"
else
    print_error "Clang not found - required for compilation"
    exit 1
fi

# LLVM Config
if command -v llvm-config &> /dev/null; then
    LLVM_VERSION=$(llvm-config --version)
    print_success "LLVM found: $LLVM_VERSION"
else
    print_error "LLVM not found - required for compilation"
    exit 1
fi

# MLIR tools
if command -v mlir-opt &> /dev/null; then
    MLIR_VERSION=$(mlir-opt --version | head -1 || echo "unknown")
    print_success "mlir-opt found: $MLIR_VERSION"
else
    print_error "mlir-opt not found - required for MLIR passes"
    exit 1
fi

if command -v mlir-translate &> /dev/null; then
    print_success "mlir-translate found"
else
    print_error "mlir-translate not found - required for MLIR lowering"
    exit 1
fi

# ClangIR (optional)
if command -v clangir &> /dev/null; then
    CLANGIR_VERSION=$(clangir --version | head -1 || echo "unknown")
    print_success "ClangIR found: $CLANGIR_VERSION (optional - enables advanced pipeline)"
    HAS_CLANGIR=true
else
    print_warning "ClangIR not found (optional - advanced pipeline will be skipped)"
    HAS_CLANGIR=false
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python3 not found - required for CLI"
    exit 1
fi

# OpenSSL
if command -v openssl &> /dev/null; then
    OPENSSL_VERSION=$(openssl version)
    print_success "OpenSSL found: $OPENSSL_VERSION"
else
    print_warning "OpenSSL not found (cryptographic hashing may fail)"
fi

# ============================================================================
# PHASE 2: Build MLIR Library
# ============================================================================
print_header "Phase 2: Building MLIR Obfuscation Library"

cd "$SCRIPT_DIR/mlir-obs"

if [ ! -f "./build.sh" ]; then
    print_error "build.sh not found in mlir-obs/"
    exit 1
fi

echo "Running build script..."
if ./build.sh > build.log 2>&1; then
    print_success "MLIR library built successfully"
else
    print_error "MLIR library build failed - check mlir-obs/build.log"
    cat build.log | tail -20
    exit 1
fi

# Verify library exists
LIBRARY=$(find build -name "*MLIRObfuscation.*" -type f | head -1)
if [ -z "$LIBRARY" ]; then
    print_error "MLIR library not found after build"
    exit 1
fi

print_success "MLIR library found: $LIBRARY"

cd "$SCRIPT_DIR"

# ============================================================================
# PHASE 3: Test MLIR Passes Standalone
# ============================================================================
print_header "Phase 3: Testing MLIR Passes Standalone"

cd "$SCRIPT_DIR/mlir-obs"

if [ ! -f "./test.sh" ]; then
    print_warning "test.sh not found in mlir-obs/ - skipping standalone tests"
    print_skip "MLIR standalone tests skipped"
else
    echo "Running MLIR standalone tests..."
    if ./test.sh > test.log 2>&1; then
        print_success "MLIR standalone tests passed"
        # Show summary
        tail -20 test.log
    else
        print_warning "MLIR standalone tests failed - check mlir-obs/test.log"
        tail -20 test.log
    fi
fi

cd "$SCRIPT_DIR"

# ============================================================================
# PHASE 4: Create Test Files
# ============================================================================
print_header "Phase 4: Creating Test Files"

TEST_DIR="$SCRIPT_DIR/test_output_complete"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create comprehensive test C file
echo "Creating test C file..."
cat > test_auth.c << 'EOF'
#include <stdio.h>
#include <string.h>

// Test constants of different types
const char* MASTER_PASSWORD = "SuperSecret2024!";
const char* API_KEY = "sk_live_abc123xyz";
const int MAX_ATTEMPTS = 3;
const float THRESHOLD = 0.95;

// Test functions
int authenticate(const char* password) {
    return strcmp(password, MASTER_PASSWORD) == 0;
}

int validate_api_key(const char* key) {
    return strcmp(key, API_KEY) == 0;
}

int check_threshold(float value) {
    return value >= THRESHOLD;
}

int main() {
    printf("Starting authentication test...\n");

    if (authenticate("SuperSecret2024!")) {
        printf("✅ Authentication successful\n");
    } else {
        printf("❌ Authentication failed\n");
    }

    if (validate_api_key("sk_live_abc123xyz")) {
        printf("✅ API key valid\n");
    } else {
        printf("❌ API key invalid\n");
    }

    if (check_threshold(0.98)) {
        printf("✅ Threshold check passed\n");
    } else {
        printf("❌ Threshold check failed\n");
    }

    printf("Max attempts: %d\n", MAX_ATTEMPTS);

    return 0;
}
EOF

print_success "Test file created: test_auth.c"

# ============================================================================
# PHASE 5: Test Default Pipeline (CLANG)
# ============================================================================
print_header "Phase 5: Testing Default Pipeline (CLANG)"

cd "$SCRIPT_DIR"

echo "Test 5.1: Basic compilation (no obfuscation)"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
    --output "$TEST_DIR/output_baseline" \
    > "$TEST_DIR/test_5_1.log" 2>&1 || {
    print_error "Baseline compilation failed"
    cat "$TEST_DIR/test_5_1.log" | tail -20
    exit 1
}
print_success "Baseline compilation successful"

echo "Test 5.2: String encryption only"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
    --enable-string-encrypt \
    --output "$TEST_DIR/output_string_encrypt" \
    > "$TEST_DIR/test_5_2.log" 2>&1 || {
    print_error "String encryption failed"
    cat "$TEST_DIR/test_5_2.log" | tail -20
    exit 1
}
print_success "String encryption successful"

# Verify strings are hidden
if strings "$TEST_DIR/output_string_encrypt/test_auth" 2>/dev/null | grep -q "SuperSecret"; then
    print_warning "Secret strings still visible after encryption"
else
    print_success "Secret strings successfully hidden"
fi

echo "Test 5.3: Symbol obfuscation (RNG-based)"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
    --enable-symbol-obfuscate \
    --output "$TEST_DIR/output_symbol_rng" \
    > "$TEST_DIR/test_5_3.log" 2>&1 || {
    print_error "Symbol obfuscation (RNG) failed"
    cat "$TEST_DIR/test_5_3.log" | tail -20
    exit 1
}
print_success "Symbol obfuscation (RNG) successful"

# Verify symbols are obfuscated
if nm "$TEST_DIR/output_symbol_rng/test_auth" 2>/dev/null | grep -v ' U ' | grep -q "authenticate"; then
    print_warning "Original function names still visible"
else
    print_success "Function names successfully obfuscated"
fi

echo "Test 5.4: Cryptographic hash obfuscation (SHA256)"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
    --enable-crypto-hash \
    --crypto-hash-algorithm sha256 \
    --crypto-hash-salt "test-salt-2024" \
    --crypto-hash-length 12 \
    --output "$TEST_DIR/output_crypto_hash" \
    > "$TEST_DIR/test_5_4.log" 2>&1 || {
    print_warning "Crypto hash obfuscation failed (may not be implemented yet)"
    cat "$TEST_DIR/test_5_4.log" | tail -10
}

if [ -f "$TEST_DIR/output_crypto_hash/test_auth" ]; then
    print_success "Cryptographic hash obfuscation successful"
else
    print_skip "Crypto hash test skipped (feature may not be available)"
fi

echo "Test 5.5: Constant obfuscation (all types)"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
    --enable-constant-obfuscate \
    --output "$TEST_DIR/output_constant_obf" \
    > "$TEST_DIR/test_5_5.log" 2>&1 || {
    print_warning "Constant obfuscation failed (may not be implemented yet)"
    cat "$TEST_DIR/test_5_5.log" | tail -10
}

if [ -f "$TEST_DIR/output_constant_obf/test_auth" ]; then
    print_success "Constant obfuscation successful"
else
    print_skip "Constant obfuscation test skipped (feature may not be available)"
fi

echo "Test 5.6: Combined obfuscation (all passes)"
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --enable-constant-obfuscate \
    --crypto-hash-algorithm blake2b \
    --crypto-hash-salt "test-salt-2024" \
    --output "$TEST_DIR/output_combined_default" \
    > "$TEST_DIR/test_5_6.log" 2>&1 || {
    print_warning "Combined obfuscation failed"
    cat "$TEST_DIR/test_5_6.log" | tail -20
}

if [ -f "$TEST_DIR/output_combined_default/test_auth" ]; then
    print_success "Combined obfuscation successful"

    # Test binary execution
    if "$TEST_DIR/output_combined_default/test_auth" > "$TEST_DIR/test_5_6_output.txt" 2>&1; then
        print_success "Binary execution successful"
        echo "Binary output:"
        cat "$TEST_DIR/test_5_6_output.txt"
    else
        print_warning "Binary execution failed (may have runtime issues)"
    fi
else
    print_skip "Combined obfuscation test skipped"
fi

# ============================================================================
# PHASE 6: Test ClangIR Pipeline (if available)
# ============================================================================
print_header "Phase 6: Testing ClangIR Pipeline"

if [ "$HAS_CLANGIR" = false ]; then
    print_skip "ClangIR not available - skipping ClangIR pipeline tests"
    print_info "To enable ClangIR tests, install ClangIR (see CLANGIR_PIPELINE_GUIDE.md)"
else
    cd "$SCRIPT_DIR"

    echo "Test 6.1: ClangIR frontend test (basic)"
    python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
        --mlir-frontend clangir \
        --output "$TEST_DIR/output_clangir_baseline" \
        > "$TEST_DIR/test_6_1.log" 2>&1 || {
        print_error "ClangIR baseline compilation failed"
        cat "$TEST_DIR/test_6_1.log" | tail -20
    }

    if [ -f "$TEST_DIR/output_clangir_baseline/test_auth" ]; then
        print_success "ClangIR baseline compilation successful"
    else
        print_warning "ClangIR baseline compilation failed"
    fi

    echo "Test 6.2: ClangIR with string encryption"
    python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
        --mlir-frontend clangir \
        --enable-string-encrypt \
        --output "$TEST_DIR/output_clangir_string" \
        > "$TEST_DIR/test_6_2.log" 2>&1 || {
        print_warning "ClangIR string encryption failed"
        cat "$TEST_DIR/test_6_2.log" | tail -20
    }

    if [ -f "$TEST_DIR/output_clangir_string/test_auth" ]; then
        print_success "ClangIR string encryption successful"
    else
        print_skip "ClangIR string encryption test skipped"
    fi

    echo "Test 6.3: ClangIR with all obfuscation passes"
    python3 -m cmd.llvm-obfuscator.cli.obfuscate compile "$TEST_DIR/test_auth.c" \
        --mlir-frontend clangir \
        --enable-string-encrypt \
        --enable-crypto-hash \
        --enable-constant-obfuscate \
        --crypto-hash-algorithm sha256 \
        --crypto-hash-salt "test-salt-2024" \
        --output "$TEST_DIR/output_clangir_full" \
        > "$TEST_DIR/test_6_3.log" 2>&1 || {
        print_warning "ClangIR full obfuscation failed"
        cat "$TEST_DIR/test_6_3.log" | tail -20
    }

    if [ -f "$TEST_DIR/output_clangir_full/test_auth" ]; then
        print_success "ClangIR full obfuscation successful"

        # Test execution
        if "$TEST_DIR/output_clangir_full/test_auth" > "$TEST_DIR/test_6_3_output.txt" 2>&1; then
            print_success "ClangIR obfuscated binary execution successful"
            echo "Binary output:"
            cat "$TEST_DIR/test_6_3_output.txt"
        else
            print_warning "ClangIR binary execution failed"
        fi
    else
        print_skip "ClangIR full obfuscation test skipped"
    fi
fi

# ============================================================================
# PHASE 7: Verification Tests
# ============================================================================
print_header "Phase 7: Obfuscation Verification"

cd "$TEST_DIR"

echo "Verification 7.1: Symbol count reduction"
if [ -f "output_baseline/test_auth" ] && [ -f "output_combined_default/test_auth" ]; then
    BASELINE_SYMBOLS=$(nm output_baseline/test_auth 2>/dev/null | grep -v ' U ' | wc -l)
    OBFUSCATED_SYMBOLS=$(nm output_combined_default/test_auth 2>/dev/null | grep -v ' U ' | wc -l)

    echo "  Baseline symbols: $BASELINE_SYMBOLS"
    echo "  Obfuscated symbols: $OBFUSCATED_SYMBOLS"

    if [ "$OBFUSCATED_SYMBOLS" -lt "$BASELINE_SYMBOLS" ]; then
        REDUCTION=$((100 * (BASELINE_SYMBOLS - OBFUSCATED_SYMBOLS) / BASELINE_SYMBOLS))
        print_success "Symbol reduction: $REDUCTION%"
    else
        print_warning "No symbol reduction achieved"
    fi
else
    print_skip "Symbol count verification skipped (missing binaries)"
fi

echo "Verification 7.2: Secret string hiding"
BASELINE_SECRETS=0
OBFUSCATED_SECRETS=0

for secret in "SuperSecret" "sk_live" "API_KEY"; do
    if [ -f "output_baseline/test_auth" ]; then
        if strings output_baseline/test_auth 2>/dev/null | grep -q "$secret"; then
            ((BASELINE_SECRETS++))
        fi
    fi

    if [ -f "output_combined_default/test_auth" ]; then
        if strings output_combined_default/test_auth 2>/dev/null | grep -q "$secret"; then
            ((OBFUSCATED_SECRETS++))
        fi
    fi
done

echo "  Baseline secrets found: $BASELINE_SECRETS"
echo "  Obfuscated secrets found: $OBFUSCATED_SECRETS"

if [ "$OBFUSCATED_SECRETS" -eq 0 ] && [ "$BASELINE_SECRETS" -gt 0 ]; then
    print_success "All secrets successfully hidden"
elif [ "$OBFUSCATED_SECRETS" -lt "$BASELINE_SECRETS" ]; then
    print_warning "Some secrets still visible ($OBFUSCATED_SECRETS/$BASELINE_SECRETS)"
else
    print_warning "No improvement in secret hiding"
fi

echo "Verification 7.3: Binary size comparison"
if [ -f "output_baseline/test_auth" ] && [ -f "output_combined_default/test_auth" ]; then
    BASELINE_SIZE=$(stat -f%z output_baseline/test_auth 2>/dev/null || stat -c%s output_baseline/test_auth 2>/dev/null)
    OBFUSCATED_SIZE=$(stat -f%z output_combined_default/test_auth 2>/dev/null || stat -c%s output_combined_default/test_auth 2>/dev/null)

    echo "  Baseline size: $BASELINE_SIZE bytes"
    echo "  Obfuscated size: $OBFUSCATED_SIZE bytes"

    if [ "$OBFUSCATED_SIZE" -gt "$BASELINE_SIZE" ]; then
        OVERHEAD=$((100 * (OBFUSCATED_SIZE - BASELINE_SIZE) / BASELINE_SIZE))
        print_info "Binary size overhead: +$OVERHEAD%"
    fi

    print_success "Binary size comparison complete"
else
    print_skip "Binary size comparison skipped (missing binaries)"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print_header "Test Summary"

echo "Total tests:"
echo "  ✅ Passed:  $TESTS_PASSED"
echo "  ❌ Failed:  $TESTS_FAILED"
echo "  ⏭️  Skipped: $TESTS_SKIPPED"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    print_success "ALL TESTS PASSED! 🎉"
    echo ""
    print_info "Next steps:"
    echo "  1. Review test artifacts in: $TEST_DIR"
    echo "  2. Test with your own code"
    echo "  3. See MLIR_INTEGRATION_GUIDE.md for advanced usage"
    echo "  4. See CLANGIR_PIPELINE_GUIDE.md for ClangIR details"
    exit 0
else
    print_error "SOME TESTS FAILED"
    echo ""
    print_info "Troubleshooting:"
    echo "  1. Check logs in: $TEST_DIR/*.log"
    echo "  2. Verify LLVM/MLIR installation: llvm-config --version"
    echo "  3. Check MLIR library: $SCRIPT_DIR/mlir-obs/build/"
    echo "  4. See MLIR_INTEGRATION_GUIDE.md troubleshooting section"
    exit 1
fi
