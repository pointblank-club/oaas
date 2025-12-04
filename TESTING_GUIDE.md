# Complete Testing Guide for LLVM Obfuscator

This guide provides step-by-step instructions for testing the complete implementation on your VM.

## Quick Start (30 seconds)

```bash
# Clone/navigate to project
cd /path/to/oaas

# Make test script executable
chmod +x test_complete_implementation.sh

# Run all tests
./test_complete_implementation.sh
```

That's it! The script will automatically test everything and give you a pass/fail report.

---

## Manual Testing (Step-by-Step)

If you prefer to test manually or need to debug specific issues, follow these steps:

### Step 1: Verify Environment

```bash
# Check LLVM/Clang version (should be 22.x)
clang --version
llvm-config --version

# Check MLIR tools
mlir-opt --version
mlir-translate --version

# Check ClangIR (optional - for advanced pipeline)
clangir --version || echo "ClangIR not installed (optional)"

# Check Python
python3 --version

# Check OpenSSL
openssl version
```

**Expected:**
- Clang 22.x
- LLVM 22.x
- MLIR tools available
- Python 3.10+
- OpenSSL installed

---

### Step 2: Build MLIR Obfuscation Library

```bash
cd /path/to/oaas/mlir-obs

# Run build script
./build.sh

# Expected output:
# ==========================================
#   Building MLIR Obfuscation Library
# ==========================================
# ...
# ✅ Build successful!
```

**Verify build:**
```bash
ls -la build/lib/libMLIRObfuscation.*
```

**Expected:** Should see library file (`.so` on Linux, `.dylib` on macOS)

---

### Step 3: Test MLIR Passes Standalone

```bash
cd /path/to/oaas/mlir-obs

# Run standalone MLIR tests
./test.sh

# Expected output:
# ==========================================
#   Testing MLIR Obfuscation Passes
# ==========================================
# ...
# ✅ All tests completed!
```

**What this tests:**
- C → LLVM IR → MLIR conversion
- Symbol obfuscation pass
- String encryption pass
- Combined passes
- MLIR → LLVM IR lowering
- Binary compilation
- Binary execution
- Obfuscation verification

---

### Step 4: Test Default Pipeline (CLANG)

Create test file:
```bash
cd /path/to/oaas

cat > test_sample.c << 'EOF'
#include <stdio.h>
#include <string.h>

const char* SECRET = "MyPassword123!";
const char* API_KEY = "sk_test_abc123";

int validate(const char* input) {
    return strcmp(input, SECRET) == 0;
}

int main() {
    if (validate("MyPassword123!")) {
        printf("Access granted\n");
    }
    return 0;
}
EOF
```

#### Test 4.1: Baseline (No Obfuscation)

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --output ./output_baseline
```

**Verify:**
```bash
./output_baseline/test_sample
# Should output: Access granted

# Check symbols
nm ./output_baseline/test_sample | grep -v ' U '
# Should see: validate, main, SECRET, API_KEY

# Check strings
strings ./output_baseline/test_sample | grep -E "MyPassword|sk_test"
# Should see: MyPassword123!, sk_test_abc123
```

#### Test 4.2: String Encryption

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --enable-string-encrypt \
    --output ./output_string_encrypt
```

**Verify:**
```bash
./output_string_encrypt/test_sample
# Should output: Access granted

# Check strings (should be hidden)
strings ./output_string_encrypt/test_sample | grep -E "MyPassword|sk_test"
# Should return EMPTY (strings encrypted)
```

✅ **If strings are hidden**: String encryption works!
❌ **If strings still visible**: Check mlir-obs build logs

#### Test 4.3: Symbol Obfuscation (RNG)

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --enable-symbol-obfuscate \
    --output ./output_symbol_rng
```

**Verify:**
```bash
./output_symbol_rng/test_sample
# Should output: Access granted

# Check symbols (should be obfuscated)
nm ./output_symbol_rng/test_sample | grep -v ' U ' | grep "validate"
# Should return EMPTY (function names obfuscated)

# See obfuscated names
nm ./output_symbol_rng/test_sample | grep -v ' U '
# Should see hex names like: f_a3b7f8d2, v_9e2c1d4a
```

✅ **If function names obfuscated**: Symbol obfuscation works!
❌ **If names still visible**: Check MLIR pass registration

#### Test 4.4: Cryptographic Hash (SHA256)

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --enable-crypto-hash \
    --crypto-hash-algorithm sha256 \
    --crypto-hash-salt "my-secret-salt" \
    --crypto-hash-length 12 \
    --output ./output_crypto_hash
```

**Verify:**
```bash
./output_crypto_hash/test_sample
# Should output: Access granted

# Check symbols
nm ./output_crypto_hash/test_sample | grep -v ' U '
# Should see deterministic hashed names like: f_a1b2c3d4e5f6
```

✅ **If deterministic hashes generated**: Crypto hash works!
❌ **If fails**: Check OpenSSL installation

#### Test 4.5: Constant Obfuscation

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --enable-constant-obfuscate \
    --output ./output_constant_obf
```

**Verify:**
```bash
./output_constant_obf/test_sample
# Should output: Access granted

# Binary should still work but constants obfuscated internally
```

✅ **If binary executes correctly**: Constant obfuscation works!
❌ **If binary crashes**: Check MLIR pass implementation

#### Test 4.6: Combined Obfuscation (ALL PASSES)

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --enable-constant-obfuscate \
    --crypto-hash-algorithm blake2b \
    --crypto-hash-salt "production-salt-2024" \
    --crypto-hash-length 16 \
    --output ./output_full_obfuscation
```

**Verify:**
```bash
# Test execution
./output_full_obfuscation/test_sample
# Should output: Access granted

# Check strings (should be hidden)
strings ./output_full_obfuscation/test_sample | grep -E "MyPassword|sk_test|SECRET"
# Should return EMPTY

# Check symbols (should be hashed)
nm ./output_full_obfuscation/test_sample | grep -v ' U ' | grep -E "validate|main"
# Should return EMPTY (except possibly _start, main might be preserved)

# Count symbols (should be minimal)
nm ./output_full_obfuscation/test_sample | grep -v ' U ' | wc -l
# Should be < 5 symbols
```

✅ **If all checks pass**: Full obfuscation works!
❌ **If any check fails**: Review specific pass tests above

---

### Step 5: Test ClangIR Pipeline (Optional)

**Prerequisites:** ClangIR must be installed (see CLANGIR_PIPELINE_GUIDE.md)

#### Test 5.1: ClangIR Baseline

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --mlir-frontend clangir \
    --output ./output_clangir_baseline
```

**Expected errors if ClangIR not installed:**
```
ERROR: clangir command not found
```

**If ClangIR is installed:**
```bash
./output_clangir_baseline/test_sample
# Should output: Access granted
```

#### Test 5.2: ClangIR with Full Obfuscation

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test_sample.c \
    --mlir-frontend clangir \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --enable-constant-obfuscate \
    --crypto-hash-algorithm sha256 \
    --crypto-hash-salt "clangir-test" \
    --output ./output_clangir_full
```

**Verify:**
```bash
./output_clangir_full/test_sample
# Should output: Access granted

# Same verification as default pipeline
strings ./output_clangir_full/test_sample | grep "MyPassword"
# Should return EMPTY
```

✅ **If works**: ClangIR pipeline is functional!
❌ **If fails**: See CLANGIR_PIPELINE_GUIDE.md troubleshooting

---

### Step 6: Comparison Testing

```bash
# Compare symbol counts
echo "Baseline symbols:"
nm ./output_baseline/test_sample | grep -v ' U ' | wc -l

echo "Default pipeline (full obfuscation):"
nm ./output_full_obfuscation/test_sample | grep -v ' U ' | wc -l

echo "ClangIR pipeline (full obfuscation):"
nm ./output_clangir_full/test_sample | grep -v ' U ' | wc -l

# Compare binary sizes
ls -lh output_*/test_sample

# Compare secret visibility
echo "Baseline secrets:"
strings ./output_baseline/test_sample | grep -c -E "MyPassword|sk_test"

echo "Obfuscated secrets:"
strings ./output_full_obfuscation/test_sample | grep -c -E "MyPassword|sk_test"
```

**Expected Results:**
- Symbol count: Baseline >> Obfuscated (e.g., 20 → 2-5)
- Binary size: Obfuscated ~10-30% larger
- Secret visibility: Baseline > 0, Obfuscated = 0

---

## Automated Test Script

For convenience, use the provided automated test script:

```bash
cd /path/to/oaas

# Make executable
chmod +x test_complete_implementation.sh

# Run all tests
./test_complete_implementation.sh
```

**What it tests:**
1. ✅ Environment verification (LLVM, MLIR, ClangIR, Python, OpenSSL)
2. ✅ MLIR library build
3. ✅ MLIR passes standalone
4. ✅ Default pipeline (6 tests)
5. ✅ ClangIR pipeline (3 tests if ClangIR available)
6. ✅ Obfuscation verification (symbol reduction, secret hiding, size comparison)

**Output:**
```
==========================================
  Test Summary
==========================================
Total tests:
  ✅ Passed:  15
  ❌ Failed:  0
  ⏭️  Skipped: 3

✅ ALL TESTS PASSED! 🎉
```

---

## Troubleshooting

### Issue: MLIR library build fails

**Symptoms:**
```
ERROR: MLIR library not found after build
```

**Debug:**
```bash
cd mlir-obs
./build.sh 2>&1 | tee build.log
cat build.log | grep -i error
```

**Common fixes:**
- Install LLVM 22 dev packages: `apt-get install llvm-22-dev libmlir-22-dev`
- Install OpenSSL dev: `apt-get install libssl-dev`
- Check CMake version: `cmake --version` (need 3.20+)

---

### Issue: String encryption doesn't work

**Symptoms:**
```bash
strings output/binary | grep "MyPassword"
# Still shows: MyPassword123!
```

**Debug:**
```bash
# Check if MLIR pass is being applied
cd mlir-obs/test_output
grep -i "string" *.mlir
```

**Common fixes:**
- Rebuild MLIR library: `cd mlir-obs && rm -rf build && ./build.sh`
- Check pass registration: `grep -r "string-encrypt" mlir-obs/lib/`
- Verify MLIR library loaded: Check Python logs for "load-pass-plugin"

---

### Issue: ClangIR not found

**Symptoms:**
```
ERROR: clangir command not found
```

**Solutions:**

**Option 1: Use Docker (Recommended)**
```bash
# Build Docker image with ClangIR
docker-compose -f docker-compose.test.yaml build

# Run tests in Docker
docker-compose -f docker-compose.test.yaml run --rm test \
    ./test_complete_implementation.sh
```

**Option 2: Build ClangIR from source**
```bash
# See CLANGIR_PIPELINE_GUIDE.md section "Building ClangIR from Source"
# Summary:
git clone --depth=1 --branch release/22.x https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build
cmake ../llvm -G Ninja \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCLANG_ENABLE_CIR=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/clangir
ninja install
export PATH="/opt/clangir/bin:$PATH"
```

**Option 3: Skip ClangIR tests**
```bash
# Just use default pipeline (works without ClangIR)
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --enable-crypto-hash
```

---

### Issue: Obfuscated binary crashes

**Symptoms:**
```bash
./output/binary
Segmentation fault
```

**Debug:**
```bash
# Run with GDB
gdb ./output/binary
run
bt

# Check intermediate MLIR
cd mlir-obs/test_output
cat test_combined_obf.mlir | less
```

**Common causes:**
- String encryption runtime decryption failure
- Constant obfuscation broke data dependencies
- MLIR to LLVM IR lowering issues

**Fixes:**
- Test passes individually (not combined)
- Check MLIR pass implementation for bugs
- Report issue with MLIR dump

---

## Docker Testing (Recommended)

If you have issues with local setup, use Docker:

```bash
# Build test environment
docker-compose -f docker-compose.test.yaml build

# Run interactive shell
docker-compose -f docker-compose.test.yaml run --rm test bash

# Inside container, run tests
./test_complete_implementation.sh
```

**Advantages:**
- ✅ Pre-configured LLVM 22, MLIR 22
- ✅ ClangIR pre-built
- ✅ All dependencies included
- ✅ Isolated environment

---

## Expected Test Results Summary

| Test | Expected Result | Pass Criteria |
|------|----------------|---------------|
| **Environment Check** | All tools found | Clang, LLVM, MLIR, Python installed |
| **MLIR Build** | Library built | `libMLIRObfuscation.*` exists |
| **MLIR Standalone** | All 8 tests pass | Binary executes correctly |
| **String Encrypt** | Secrets hidden | `strings` returns empty for secrets |
| **Symbol Obfuscate** | Names changed | `nm` shows hex names, not original |
| **Crypto Hash** | Deterministic hashes | Same input → same hash |
| **Constant Obfuscate** | Binary works | No crashes, correct output |
| **ClangIR Pipeline** | Binary works | Same as default pipeline |
| **Symbol Reduction** | 80-95% reduction | Baseline 20 symbols → 1-5 symbols |
| **Secret Hiding** | 100% hidden | 0 secrets found in obfuscated binary |

---

## Next Steps After Testing

Once all tests pass:

1. **Test with real code**: Use your actual project source files
2. **Performance testing**: Measure compilation time and runtime overhead
3. **Integration**: Add to CI/CD pipeline
4. **Production**: Deploy obfuscated binaries

**Documentation:**
- **Usage**: See MLIR_INTEGRATION_GUIDE.md
- **ClangIR**: See CLANGIR_PIPELINE_GUIDE.md
- **API**: See README.md section "API Documentation"

---

## Support

If tests fail and troubleshooting doesn't help:

1. Check logs: `cat test_output_complete/*.log`
2. Verify LLVM version: `llvm-config --version` (must be 22.x)
3. Check MLIR library: `ldd mlir-obs/build/lib/libMLIRObfuscation.*`
4. Report issue with:
   - Test script output
   - Build logs
   - LLVM/MLIR versions
   - OS/platform info

**Common issues are documented in MLIR_INTEGRATION_GUIDE.md "Troubleshooting" section.**
