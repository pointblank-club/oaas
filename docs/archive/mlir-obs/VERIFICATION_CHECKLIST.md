# Polygeist Integration Verification Checklist

Use this checklist to verify that the Polygeist integration is working correctly.

---

## Pre-Build Verification

### ☐ 1. Source Files Present

```bash
cd mlir-obs

# Check new files exist
ls -l cmake/FindPolygeist.cmake
ls -l include/Obfuscator/Config.h.in
ls -l lib/SCFPass.cpp
ls -l polygeist-pipeline.sh
ls -l compare-pipelines.sh
ls -l test-polygeist-integration.sh
ls -l examples/simple_auth.c
ls -l examples/loop_example.c
```

**Expected:** All files should exist

### ☐ 2. Modified Files Backup

```bash
# Verify modifications (optional - for safety)
git diff CMakeLists.txt
git diff lib/SymbolPass.cpp
git diff lib/CMakeLists.txt
git diff tools/mlir-obfuscate.cpp
```

**Expected:** See Polygeist-related changes

---

## Build Verification (Without Polygeist)

### ☐ 3. Clean Build

```bash
rm -rf build
./build.sh
```

**Expected Output:**
```
✅ All required tools found
...
⚠ Polygeist support DISABLED (not found)
  Falling back to: clang + mlir-translate workflow
...
✅ Build successful!
```

### ☐ 4. Library Built

```bash
find build -name "*MLIRObfuscation*"
```

**Expected:** Should find `MLIRObfuscation.so` or `.dylib` or `.dll`

### ☐ 5. Tool Built

```bash
ls -l build/tools/mlir-obfuscate
```

**Expected:** Binary should exist and be executable

### ☐ 6. Plugin Loads

```bash
LIBRARY=$(find build -name "*MLIRObfuscation.*" -type f | head -1)
mlir-opt --load-pass-plugin="$LIBRARY" --help | grep -E "symbol-obfuscate|string-encrypt|scf-obfuscate"
```

**Expected:**
```
--symbol-obfuscate
--string-encrypt
--scf-obfuscate
```

### ☐ 7. Traditional Pipeline Still Works

```bash
./test-func-dialect.sh
```

**Expected:** All tests pass (may see some SCF warnings, that's OK)

---

## Build Verification (With Polygeist)

**Skip this section if you don't have Polygeist installed**

### ☐ 8. Polygeist Detected

```bash
# Ensure Polygeist is in PATH
which cgeist || which mlir-clang

# Rebuild
rm -rf build
./build.sh
```

**Expected Output:**
```
✅ All required tools found
...
✓ Polygeist support ENABLED
  You can now use: cgeist source.c -o source.mlir
...
✅ Build successful!
```

### ☐ 9. Config Header Generated

```bash
grep "HAVE_POLYGEIST" build/include/Obfuscator/Config.h
```

**Expected:**
```
#define HAVE_POLYGEIST 1
#define CGEIST_EXECUTABLE "/path/to/cgeist"
```

### ☐ 10. mlir-obfuscate Shows Polygeist

```bash
./build/tools/mlir-obfuscate --help | head -10
```

**Expected:**
```
MLIR Obfuscator Tool
MLIR Version: ...
Polygeist support: ENABLED
  cgeist: /path/to/cgeist
```

---

## Functionality Verification

### ☐ 11. Symbol Obfuscation (LLVM Dialect)

```bash
TEMP=$(mktemp -d)

# Create test MLIR (LLVM dialect)
cat > $TEMP/test.mlir << 'EOF'
module {
  llvm.func @my_function(%arg0: i32) -> i32 {
    llvm.return %arg0 : i32
  }
}
EOF

# Apply pass
LIBRARY=$(find build -name "*MLIRObfuscation.*" -type f | head -1)
mlir-opt $TEMP/test.mlir \
  --load-pass-plugin="$LIBRARY" \
  --pass-pipeline='builtin.module(symbol-obfuscate)' \
  -o $TEMP/obf.mlir

# Check result
grep "llvm.func @f_" $TEMP/obf.mlir
```

**Expected:** Function name changed to `@f_xxxxxxxx` (hex hash)

### ☐ 12. Symbol Obfuscation (Func Dialect)

```bash
# Create test MLIR (func dialect)
cat > $TEMP/test_func.mlir << 'EOF'
module {
  func.func @my_function(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}
EOF

# Apply pass
mlir-opt $TEMP/test_func.mlir \
  --load-pass-plugin="$LIBRARY" \
  --pass-pipeline='builtin.module(symbol-obfuscate)' \
  -o $TEMP/obf_func.mlir

# Check result
grep "func.func @f_" $TEMP/obf_func.mlir
```

**Expected:** Function name changed to `@f_xxxxxxxx`

### ☐ 13. Dual-Dialect Handling

```bash
# Create mixed dialect MLIR
cat > $TEMP/mixed.mlir << 'EOF'
module {
  func.func @high_level(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
  llvm.func @low_level(%arg0: i32) -> i32 {
    llvm.return %arg0 : i32
  }
}
EOF

# Apply pass
mlir-opt $TEMP/mixed.mlir \
  --load-pass-plugin="$LIBRARY" \
  --pass-pipeline='builtin.module(symbol-obfuscate)' \
  -o $TEMP/obf_mixed.mlir

# Check both obfuscated
grep "func.func @f_" $TEMP/obf_mixed.mlir
grep "llvm.func @f_" $TEMP/obf_mixed.mlir
```

**Expected:** Both functions obfuscated

### ☐ 14. String Encryption

```bash
cat > $TEMP/strings.mlir << 'EOF'
module {
  func.func @test() {
    %str = "my_secret_string"
    return
  }
}
EOF

mlir-opt $TEMP/strings.mlir \
  --load-pass-plugin="$LIBRARY" \
  --pass-pipeline='builtin.module(string-encrypt)' \
  -o $TEMP/encrypted.mlir

# String should be changed
grep "my_secret_string" $TEMP/encrypted.mlir && echo "FAIL: Not encrypted" || echo "PASS: Encrypted"
```

**Expected:** "PASS: Encrypted"

---

## Polygeist-Specific Verification

**Skip if Polygeist not installed**

### ☐ 15. Polygeist C to MLIR

```bash
echo 'int add(int a, int b) { return a + b; }' > $TEMP/test.c

cgeist $TEMP/test.c --function='*' -o $TEMP/poly.mlir

# Check dialects
grep "func.func @add" $TEMP/poly.mlir
grep "arith\." $TEMP/poly.mlir
```

**Expected:** func dialect and arith ops present

### ☐ 16. SCF Dialect Generated

```bash
cat > $TEMP/if_test.c << 'EOF'
int test(int x) {
    if (x > 0) return 1;
    return 0;
}
EOF

cgeist $TEMP/if_test.c --function='*' -o $TEMP/scf_test.mlir

# Check SCF
grep "scf.if" $TEMP/scf_test.mlir
```

**Expected:** SCF if operation present

### ☐ 17. SCF Obfuscation Pass

```bash
mlir-opt $TEMP/scf_test.mlir \
  --load-pass-plugin="$LIBRARY" \
  --pass-pipeline='builtin.module(scf-obfuscate)' \
  -o $TEMP/scf_obf.mlir

# Should have opaque predicates (arith.muli, arith.divsi, arith.cmpi, arith.andi)
grep "arith.andi" $TEMP/scf_obf.mlir
```

**Expected:** Opaque predicate operations inserted

### ☐ 18. End-to-End Polygeist Pipeline

```bash
./polygeist-pipeline.sh examples/simple_auth.c $TEMP/test_binary
```

**Expected:**
```
[Step 1/7] C/C++ → Polygeist MLIR
✓ Generated Polygeist MLIR
...
[Step 7/7] Compiling to Binary
✓ Binary compilation complete
...
✓ Pipeline Complete!
```

### ☐ 19. Binary Functional Test

```bash
# Binary should exist and be executable
test -x $TEMP/test_binary
echo $?  # Should be 0

# Test execution (will fail without correct password, but should run)
$TEMP/test_binary wrong_password
echo $?  # Should be 1 (access denied)
```

**Expected:** Binary runs (exit code 1 is OK - means it executed)

### ☐ 20. Obfuscation Verification

```bash
# Check symbol obfuscation in binary
nm $TEMP/test_binary | grep "validate_password" && echo "FAIL: Not obfuscated" || echo "PASS: Obfuscated"

# Check for obfuscated symbols
nm $TEMP/test_binary | grep "f_[0-9a-f]" && echo "PASS: Has obfuscated symbols" || echo "FAIL: No obfuscated symbols"
```

**Expected:**
```
PASS: Obfuscated
PASS: Has obfuscated symbols
```

---

## Pipeline Comparison Verification

### ☐ 21. Compare Script Works

```bash
./compare-pipelines.sh examples/simple_auth.c
```

**Expected:**
- Shows Traditional vs Polygeist statistics
- Displays dialect differences
- Shows sample MLIR output

---

## Integration Test Suite

### ☐ 22. Run Full Test Suite

```bash
./test-polygeist-integration.sh
```

**Expected (with Polygeist):**
```
✓ Passed: 28 / 28
✗ Failed: 0 / 28
⊘ Skipped: 0 / 28

All tests passed! ✓
```

**Expected (without Polygeist):**
```
✓ Passed: 19 / 28
✗ Failed: 0 / 28
⊘ Skipped: 9 / 28

All tests passed! ✓
Note: Some tests were skipped because Polygeist is not installed.
```

---

## Documentation Verification

### ☐ 23. Documentation Files Exist

```bash
ls -lh POLYGEIST_INTEGRATION.md
ls -lh INSTALL_POLYGEIST.md
ls -lh POLYGEIST_README.md
ls -lh ../POLYGEIST_INTEGRATION_SUMMARY.md
```

**Expected:** All files exist, reasonable sizes (10KB+)

### ☐ 24. Documentation Readable

```bash
# Open in viewer
cat POLYGEIST_README.md | head -50
```

**Expected:** Properly formatted markdown

---

## Backwards Compatibility Verification

### ☐ 25. Existing Tests Pass

```bash
./test-func-dialect.sh
```

**Expected:** All tests pass (same as before integration)

### ☐ 26. Python CLI Unchanged

```bash
cd ../cmd/llvm-obfuscator

# Check CLI still works
python3 -m cli.obfuscate --help | grep "compile"
```

**Expected:** CLI help shows (no errors)

### ☐ 27. OLLVM Integration Preserved

```bash
# Check OLLVM plugin still exists
ls -l ../cmd/llvm-obfuscator/plugins/*/lib/LLVMObfuscation.* 2>/dev/null || echo "OLLVM plugin location may differ"
```

**Expected:** Plugin files found or different location noted

---

## Performance Verification

### ☐ 28. Build Time Acceptable

```bash
time ./build.sh
```

**Expected:** Under 10 minutes on modern hardware

### ☐ 29. Pipeline Time Acceptable

```bash
time ./polygeist-pipeline.sh examples/simple_auth.c $TEMP/perf_test
```

**Expected:** Under 5 seconds for small files

---

## Cleanup

```bash
rm -rf $TEMP
```

---

## Summary Checklist

### Core Functionality
- [ ] Build succeeds without Polygeist
- [ ] Build succeeds with Polygeist
- [ ] All passes load correctly
- [ ] Symbol obfuscation works (both dialects)
- [ ] String encryption works
- [ ] SCF obfuscation works (Polygeist only)

### Backwards Compatibility
- [ ] Existing tests pass
- [ ] Python CLI unchanged
- [ ] OLLVM integration preserved
- [ ] Traditional pipeline works

### New Features
- [ ] Polygeist frontend works
- [ ] Dual-dialect support works
- [ ] End-to-end pipeline works
- [ ] Integration tests pass
- [ ] Documentation complete

### Quality
- [ ] No compilation errors
- [ ] No runtime errors
- [ ] Binaries execute correctly
- [ ] Obfuscation verified
- [ ] Performance acceptable

---

## Troubleshooting

If any check fails, consult:

1. **Build issues** → `INSTALL_POLYGEIST.md`
2. **Polygeist issues** → `POLYGEIST_INTEGRATION.md`
3. **Usage questions** → `POLYGEIST_README.md`
4. **Test failures** → Re-run with verbose: `./test-polygeist-integration.sh 2>&1 | tee test.log`

---

## Sign-off

Date: _______________

Verified by: _______________

Notes:
```
```

---

**Verification Status:**
- [ ] All critical checks passed
- [ ] All optional checks passed (Polygeist features)
- [ ] Ready for production use
- [ ] Ready for integration with main branch

**Last Updated:** 2025-11-29
