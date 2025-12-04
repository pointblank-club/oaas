# Polygeist Integration Testing - Complete Guide

This directory contains everything you need to test the Polygeist layer integration end-to-end.

---

## 🚀 **TL;DR - Quick Start**

```bash
# First time only: Build Polygeist (15-45 minutes)
./setup_polygeist.sh

# Load environment
source ./polygeist_env.sh

# Run tests
./test_polygeist_e2e.sh
```

**Expected result:** All tests pass ✅

---

## 📁 **Files in This Directory**

### **Setup Scripts**
| File | Purpose |
|------|---------|
| `setup_polygeist.sh` | Automated Polygeist build and installation |
| `polygeist_env.sh` | Environment setup (created by setup script) |

### **Test Scripts**
| File | Purpose | Duration |
|------|---------|----------|
| `test_polygeist_e2e.sh` | **Main test** - Complete end-to-end testing | 30-60s |
| `mlir-obs/test-polygeist-integration.sh` | Polygeist-focused tests | 15-30s |
| `test_mlir_integration.sh` | Full system (Python CLI + MLIR) | 45-90s |
| `mlir-obs/test.sh` | Quick MLIR pass verification | 5-10s |

### **Pipeline Scripts**
| File | Purpose |
|------|---------|
| `mlir-obs/polygeist-pipeline.sh` | Full obfuscation pipeline (C → obfuscated binary) |
| `mlir-obs/compare-pipelines.sh` | Compare traditional vs Polygeist pipelines |

### **Documentation**
| File | Purpose |
|------|---------|
| `SETUP_POLYGEIST.md` | Detailed Polygeist setup guide |
| `TESTING_GUIDE.md` | Comprehensive testing documentation |
| `QUICK_TEST.md` | Quick reference for common commands |
| `TEST_COMMANDS.txt` | Plain text command cheatsheet |
| `SETUP_AND_TEST_CHECKLIST.md` | Step-by-step checklist |
| `README_TESTING.md` | This file |

---

## 🎯 **What Gets Tested**

The test suite verifies your entire Polygeist integration:

### **1. Environment & Prerequisites**
- ✅ Polygeist (cgeist) installed
- ✅ MLIR tools available
- ✅ Clang compiler working

### **2. MLIR Library**
- ✅ Builds successfully
- ✅ All passes load correctly
- ✅ symbol-obfuscate pass
- ✅ string-encrypt pass
- ✅ scf-obfuscate pass

### **3. Traditional Pipeline** (Baseline)
- ✅ C → LLVM IR → MLIR
- ✅ Symbol obfuscation on LLVM dialect
- ✅ Binary generation
- ✅ Execution correctness

### **4. Polygeist Pipeline** ⭐ (Your Integration)
- ✅ C → Polygeist MLIR (func, scf, memref, affine)
- ✅ High-level dialects present
- ✅ Symbol obfuscation on func::FuncOp
- ✅ SCF control-flow obfuscation
- ✅ String encryption
- ✅ Lowering to LLVM dialect
- ✅ Binary compilation
- ✅ Execution correctness

### **5. Obfuscation Verification**
- ✅ Secret strings hidden
- ✅ Function symbols obfuscated
- ✅ Binary size analysis
- ✅ Symbol count comparison

---

## 📊 **Two Pipelines Compared**

### **Traditional Pipeline** (Works without Polygeist)
```
C source code
    ↓ [clang -emit-llvm]
LLVM IR (.ll)
    ↓ [mlir-translate --import-llvm]
MLIR (LLVM dialect only)
    ↓ [symbol-obfuscate]
Obfuscated MLIR
    ↓ [mlir-translate --mlir-to-llvmir]
LLVM IR
    ↓ [clang]
Binary
```

**Limitations:**
- Only low-level LLVM operations
- Limited obfuscation opportunities
- No high-level control flow analysis

### **Polygeist Pipeline** ⭐ (Your New Integration)
```
C source code
    ↓ [cgeist]
MLIR (func, scf, memref, affine dialects)  ← High-level!
    ↓ [symbol-obfuscate]
Obfuscated symbols (func::FuncOp)
    ↓ [scf-obfuscate]
Obfuscated control flow (SCF dialect)
    ↓ [string-encrypt]
Encrypted strings
    ↓ [lowering passes]
MLIR (LLVM dialect)
    ↓ [mlir-translate]
LLVM IR
    ↓ [clang]
Binary
```

**Advantages:**
- High-level dialects with more semantic info
- Better obfuscation before lowering
- Control flow obfuscation on SCF ops
- More optimization opportunities

---

## 🔧 **Setup Instructions**

### **Prerequisites**

Required tools (should already be in your VM):
- git
- cmake
- ninja
- clang (19+)
- LLVM/MLIR (19+)
- Python 3

### **Step 1: Build Polygeist** (One-time, 15-45 minutes)

```bash
./setup_polygeist.sh
```

This will:
1. Check prerequisites
2. Clone Polygeist repository
3. Configure with CMake
4. Build Polygeist
5. Verify installation
6. Create environment script

**Disk space needed:** ~2-5 GB
**Time:** 15-45 minutes (depending on CPU)

### **Step 2: Load Environment**

```bash
source ./polygeist_env.sh
```

Or add permanently:
```bash
echo 'export PATH="/path/to/oaas/polygeist/build/bin:$PATH"' >> ~/.bashrc
```

### **Step 3: Verify**

```bash
which cgeist
# Should output: /path/to/oaas/polygeist/build/bin/cgeist

cgeist --version
# Should show version info
```

---

## ✅ **Running Tests**

### **Option 1: Complete End-to-End Test** (Recommended)

```bash
./test_polygeist_e2e.sh
```

**Tests:** Everything (all 7 phases)
**Duration:** 30-60 seconds
**Use when:** First-time setup, before releases, comprehensive verification

**Expected output:**
```
╔════════════════════════════════════════════════════════════╗
║  ✅ ALL TESTS PASSED!                                     ║
╚════════════════════════════════════════════════════════════╝

✓ Polygeist integration is fully functional!

What's working:
  ✓ C -> Polygeist MLIR (func, scf, memref, affine)
  ✓ Symbol obfuscation on high-level dialects
  ✓ SCF control-flow obfuscation
  ✓ String encryption
  ✓ Lowering to LLVM dialect
  ✓ Binary generation and execution
```

### **Option 2: Polygeist-Focused Tests**

```bash
cd mlir-obs
./test-polygeist-integration.sh
```

**Tests:** Polygeist pipeline only
**Duration:** 15-30 seconds
**Use when:** Testing Polygeist-specific features

### **Option 3: Full System Integration**

```bash
./test_mlir_integration.sh
```

**Tests:** Python CLI + MLIR integration
**Duration:** 45-90 seconds
**Use when:** Testing complete toolchain

### **Option 4: Quick Pass Test**

```bash
cd mlir-obs
./test.sh
```

**Tests:** MLIR passes only
**Duration:** 5-10 seconds
**Use when:** After code changes to passes

---

## 🔍 **Manual Testing**

If you want to understand each step:

```bash
# 1. Create test file
cat > test.c << 'EOF'
#include <stdio.h>

const char* SECRET = "my_secret_key";

int calculate(int x, int y) {
    if (x > y) {
        return x + y;
    }
    return x - y;
}

int main() {
    printf("Result: %d\n", calculate(10, 5));
    return 0;
}
EOF

# 2. Run Polygeist pipeline
./mlir-obs/polygeist-pipeline.sh test.c test_output

# 3. Execute
./test_output
# Should print: Result: 15

# 4. Verify obfuscation
strings test_output | grep "my_secret_key"  # Should NOT find it
nm test_output | grep "calculate"           # Should NOT find it
nm test_output | grep "f_"                  # Should find obfuscated names
```

---

## 📈 **Interpreting Results**

### **Success (All Tests Pass)**

```
✅ PASS: C to Polygeist MLIR generation
✅ PASS: func dialect present
✅ PASS: Symbol obfuscation (func dialect)
✅ PASS: Binary execution (Polygeist) - exit code 42
✅ PASS: Secret strings hidden
✅ PASS: Function symbols obfuscated

╔════════════════════════════════════════════════════════════╗
║  ✅ ALL TESTS PASSED!                                     ║
╚════════════════════════════════════════════════════════════╝
```

**Meaning:** Polygeist integration is fully functional!

### **Skipped Tests**

```
⊘ SKIP: Polygeist pipeline (Polygeist not installed)
```

**Meaning:** Polygeist not in PATH
**Fix:** Run `source ./polygeist_env.sh`

### **Failed Tests**

```
❌ FAIL: C to Polygeist MLIR generation
```

**Meaning:** Something broke
**Fix:** Check logs, see [TESTING_GUIDE.md](TESTING_GUIDE.md) debugging section

---

## 🐛 **Troubleshooting**

### **"Polygeist not found"**

```bash
# Solution 1: Load environment
source ./polygeist_env.sh

# Solution 2: Build Polygeist
./setup_polygeist.sh
```

### **"MLIR library not found"**

```bash
# Rebuild MLIR library
cd mlir-obs
./build.sh
```

### **Tests fail after passing before**

```bash
# Check environment
which cgeist
which mlir-opt

# Re-run setup
source ./polygeist_env.sh
./test_polygeist_e2e.sh
```

### **Build fails due to memory**

```bash
# Reduce parallel jobs in setup_polygeist.sh
# Edit line: JOBS=$(nproc) → JOBS=2
```

**For more:** See [SETUP_POLYGEIST.md](SETUP_POLYGEIST.md) troubleshooting section

---

## 📚 **Documentation Map**

**Where do I start?**
- First time: [SETUP_AND_TEST_CHECKLIST.md](SETUP_AND_TEST_CHECKLIST.md)
- Quick commands: [QUICK_TEST.md](QUICK_TEST.md)
- Detailed setup: [SETUP_POLYGEIST.md](SETUP_POLYGEIST.md)
- Testing details: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- Command reference: [TEST_COMMANDS.txt](TEST_COMMANDS.txt)

**Workflow:**
1. Read checklist → 2. Run setup → 3. Run tests → 4. Use quick reference

---

## ⏱️ **Time Investment**

### **First Time** (Complete Setup)
- Read documentation: 10-15 minutes
- Build Polygeist: 15-45 minutes
- Run tests: 2-5 minutes
- **Total:** ~30-60 minutes

### **Ongoing** (After Setup)
- Load environment: 5 seconds
- Run tests: 30-60 seconds
- **Total:** <2 minutes per test run

---

## 🎯 **Success Criteria**

You're done when:

1. ✅ `./setup_polygeist.sh` completes successfully
2. ✅ `which cgeist` shows path
3. ✅ `./test_polygeist_e2e.sh` passes all tests
4. ✅ **Zero** tests skipped due to missing Polygeist
5. ✅ Can process C files through pipeline
6. ✅ Obfuscated binaries execute correctly

---

## 🚀 **Next Steps**

After all tests pass:

### **1. Test with Real Code**
```bash
./mlir-obs/polygeist-pipeline.sh your_app.c obfuscated_output
```

### **2. Integrate with Build System**

**Makefile:**
```makefile
obfuscated: src/main.c
    ./mlir-obs/polygeist-pipeline.sh $< $@
```

**CMake:**
```cmake
add_custom_target(obfuscated
    COMMAND ./mlir-obs/polygeist-pipeline.sh
            ${CMAKE_SOURCE_DIR}/src/main.c
            ${CMAKE_BINARY_DIR}/obfuscated
)
```

### **3. Deploy**
- Add to CI/CD pipeline
- Include in release builds
- Document for team

---

## 📞 **Getting Help**

If you're stuck:

1. **Check documentation** (especially [TESTING_GUIDE.md](TESTING_GUIDE.md))
2. **Check logs** (`/tmp/mlir_build.log`, `/tmp/test_*.log`)
3. **Verify environment** (`which cgeist`, `mlir-opt --version`)
4. **Run minimal test** (`mlir-obs/test.sh`)
5. **Review error messages** carefully

---

## 📊 **Summary**

| Component | Status | Command |
|-----------|--------|---------|
| Polygeist | ⚙️ Needs setup | `./setup_polygeist.sh` |
| MLIR Library | ⚙️ Needs build | `cd mlir-obs && ./build.sh` |
| Tests | ▶️ Ready to run | `./test_polygeist_e2e.sh` |
| Pipeline | ▶️ Ready to use | `./mlir-obs/polygeist-pipeline.sh` |

**Goal:** All components ✅ (setup → build → test → use)

---

## 🎓 **What This Gives You**

Once setup is complete, you have:

✅ **Two obfuscation pipelines:**
- Traditional (LLVM dialect only)
- Polygeist (high-level dialects + better obfuscation)

✅ **Three obfuscation layers:**
- Symbol obfuscation (works on both LLVM and func dialects)
- String encryption
- SCF control-flow obfuscation (Polygeist-specific)

✅ **Complete testing:**
- Automated test suite
- Manual pipeline testing
- Obfuscation verification

✅ **Production-ready:**
- End-to-end pipeline script
- Integration with build systems
- Comprehensive documentation

**You're ready to obfuscate C/C++ code with MLIR + Polygeist!** 🎉
