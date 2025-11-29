# Setup and Test Checklist for Polygeist Integration

Follow these steps in order to set up and test the complete Polygeist integration.

---

## ✅ **Checklist**

### **Phase 1: Polygeist Setup** (One-time, 15-45 minutes)

- [ ] **Step 1:** Run Polygeist setup script
  ```bash
  ./setup_polygeist.sh
  ```
  **Expected:** Script completes with "✅ Polygeist Setup Complete!"

- [ ] **Step 2:** Load Polygeist environment
  ```bash
  source ./polygeist_env.sh
  ```
  **Expected:** Shows "Polygeist environment loaded"

- [ ] **Step 3:** Verify Polygeist installation
  ```bash
  which cgeist
  ```
  **Expected:** Shows path like `/path/to/oaas/polygeist/build/bin/cgeist`

- [ ] **Step 4:** Test basic Polygeist functionality
  ```bash
  echo 'int add(int a, int b) { return a + b; }' > /tmp/test.c
  cgeist /tmp/test.c --function='*' -o /tmp/test.mlir
  grep "func.func" /tmp/test.mlir
  ```
  **Expected:** Shows `func.func @add(...)` in output

---

### **Phase 2: MLIR Library Build** (5-10 minutes)

- [ ] **Step 5:** Build MLIR obfuscation library
  ```bash
  cd mlir-obs
  ./build.sh
  cd ..
  ```
  **Expected:** Build completes successfully

- [ ] **Step 6:** Verify library was built
  ```bash
  find mlir-obs/build -name "*MLIRObfuscation.*"
  ```
  **Expected:** Shows library file (`.so`, `.dylib`, or `.dll`)

- [ ] **Step 7:** Test MLIR passes
  ```bash
  cd mlir-obs
  ./test.sh
  cd ..
  ```
  **Expected:** All basic tests pass

---

### **Phase 3: Integration Testing** (30-60 seconds)

- [ ] **Step 8:** Run comprehensive end-to-end tests
  ```bash
  ./test_polygeist_e2e.sh
  ```
  **Expected:** All tests pass, including Polygeist-specific tests

- [ ] **Step 9:** Verify test results
  - Check for: `✅ ALL TESTS PASSED!`
  - Verify: `✓ Polygeist integration is fully functional!`
  - Ensure: **No** tests were skipped due to missing Polygeist

---

### **Phase 4: Pipeline Testing** (Manual verification)

- [ ] **Step 10:** Test Polygeist pipeline script
  ```bash
  cat > test_example.c << 'EOF'
  #include <stdio.h>
  int main() {
      printf("Hello from Polygeist!\n");
      return 42;
  }
  EOF

  ./mlir-obs/polygeist-pipeline.sh test_example.c test_output
  ```
  **Expected:** Pipeline completes successfully

- [ ] **Step 11:** Execute obfuscated binary
  ```bash
  ./test_output
  echo $?
  ```
  **Expected:** Prints "Hello from Polygeist!" and exit code is 42

- [ ] **Step 12:** Verify obfuscation
  ```bash
  nm test_output | grep "main"
  ```
  **Expected:** `main` symbol NOT found (it should be obfuscated to `f_XXXXXXXX`)

- [ ] **Step 13:** Check for obfuscated symbols
  ```bash
  nm test_output | grep "f_"
  ```
  **Expected:** Shows obfuscated function names like `f_a3b2c1d0`

---

### **Phase 5: Real-World Testing** (Optional)

- [ ] **Step 14:** Test with example files
  ```bash
  ./mlir-obs/polygeist-pipeline.sh mlir-obs/examples/simple_auth.c auth_test
  ./auth_test
  ```
  **Expected:** Binary executes correctly

- [ ] **Step 15:** Run Polygeist-specific integration tests
  ```bash
  cd mlir-obs
  ./test-polygeist-integration.sh
  cd ..
  ```
  **Expected:** All Polygeist tests pass

- [ ] **Step 16:** Test full system integration
  ```bash
  ./test_mlir_integration.sh
  ```
  **Expected:** All Python CLI + MLIR tests pass

---

## 📊 **Success Criteria**

All checkboxes above should be checked ✅

**Key indicators of success:**

1. **Polygeist installed:**
   - `which cgeist` shows path
   - Can generate MLIR from C code

2. **MLIR library built:**
   - Library file exists
   - All passes load correctly

3. **Integration working:**
   - All tests pass (no Polygeist tests skipped)
   - Binaries execute correctly
   - Obfuscation verified

4. **Pipeline functional:**
   - Can process C files end-to-end
   - Symbols obfuscated
   - Binary produces correct output

---

## 🐛 **If Something Fails**

### Polygeist Setup Issues

**Problem:** `./setup_polygeist.sh` fails

**Solutions:**
1. Check prerequisites:
   ```bash
   sudo apt install git cmake ninja-build clang-19 llvm-19-dev mlir-19-tools
   ```

2. Check available disk space:
   ```bash
   df -h .
   ```
   Need at least 5GB free

3. Check available memory:
   ```bash
   free -h
   ```
   Need at least 4GB RAM (or use swap)

**See:** [SETUP_POLYGEIST.md](SETUP_POLYGEIST.md) for detailed troubleshooting

### MLIR Build Issues

**Problem:** MLIR library build fails

**Solutions:**
1. Check LLVM/MLIR installation:
   ```bash
   mlir-opt --version
   ```

2. Clean and rebuild:
   ```bash
   cd mlir-obs
   rm -rf build
   ./build.sh
   ```

### Test Failures

**Problem:** Tests fail or skip Polygeist tests

**Solutions:**
1. Ensure Polygeist is in PATH:
   ```bash
   source ./polygeist_env.sh
   which cgeist
   ```

2. Check test logs:
   ```bash
   cat /tmp/mlir_build.log
   ls -la /tmp/tmp.*/
   ```

3. Run tests with verbose output:
   ```bash
   ./test_polygeist_e2e.sh 2>&1 | tee test_output.log
   ```

**See:** [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed debugging

---

## ⏱️ **Time Estimates**

| Phase | Duration | Frequency |
|-------|----------|-----------|
| Phase 1: Polygeist Setup | 15-45 min | **One-time only** |
| Phase 2: MLIR Build | 5-10 min | After code changes |
| Phase 3: Integration Tests | 30-60 sec | After changes, before commits |
| Phase 4: Pipeline Testing | 2-5 min | Manual verification |
| Phase 5: Real-World Testing | 5-10 min | Before release |

**Total first-time setup:** ~20-60 minutes
**Subsequent testing:** <2 minutes

---

## 🎯 **Quick Reference**

After initial setup, your typical workflow:

```bash
# 1. Load environment (once per session)
source ./polygeist_env.sh

# 2. Build MLIR library (after code changes)
cd mlir-obs && ./build.sh && cd ..

# 3. Run tests
./test_polygeist_e2e.sh

# 4. Test pipeline
./mlir-obs/polygeist-pipeline.sh mycode.c output
./output
```

---

## 📚 **Documentation Reference**

- **Setup:** [SETUP_POLYGEIST.md](SETUP_POLYGEIST.md)
- **Testing:** [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Quick Commands:** [QUICK_TEST.md](QUICK_TEST.md)
- **Command Reference:** [TEST_COMMANDS.txt](TEST_COMMANDS.txt)
- **This Checklist:** [SETUP_AND_TEST_CHECKLIST.md](SETUP_AND_TEST_CHECKLIST.md)

---

## ✅ **Final Verification**

After completing all phases, you should be able to answer "YES" to all:

- [ ] Can I run `cgeist` from command line?
- [ ] Does `./test_polygeist_e2e.sh` pass all tests?
- [ ] Are **zero** tests skipped due to missing Polygeist?
- [ ] Can I process a C file through the full pipeline?
- [ ] Do obfuscated binaries execute correctly?
- [ ] Are symbols obfuscated in the final binary?

If all are "YES" ✅, your Polygeist integration is **fully functional**!

---

## 🎓 **What You've Achieved**

Once this checklist is complete, you have:

✅ **Working Polygeist installation** - Can generate high-level MLIR from C
✅ **MLIR obfuscation library** - Symbol, string, and SCF obfuscation
✅ **Full integration** - Both pipelines (traditional + Polygeist) working
✅ **Automated testing** - Comprehensive test suite verifying everything
✅ **Production pipeline** - End-to-end C → obfuscated binary workflow

**Next:** Use in production, integrate with CI/CD, deploy! 🚀
