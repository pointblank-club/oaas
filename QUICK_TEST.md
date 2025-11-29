# Quick Test Commands for Polygeist Integration

## ⚠️ **First Time Setup**

Before testing, you need to build Polygeist (one-time setup):

```bash
# 1. Build Polygeist (15-45 minutes, one time only)
./setup_polygeist.sh

# 2. Load Polygeist environment
source ./polygeist_env.sh

# 3. Verify installation
which cgeist  # Should show path
```

See [SETUP_POLYGEIST.md](SETUP_POLYGEIST.md) for detailed instructions.

---

## 🚀 **One Command - Full Test**

```bash
./test_polygeist_e2e.sh
```

This runs **everything** and shows detailed results.

---

## 📋 **Available Test Scripts**

| Script | What It Tests | Time | Use When |
|--------|---------------|------|----------|
| `./test_polygeist_e2e.sh` | **Complete E2E** - All pipelines, all passes | 30-60s | First time / comprehensive verification |
| `./mlir-obs/test-polygeist-integration.sh` | **Polygeist-focused** - High-level dialects | 15-30s | Testing Polygeist layer specifically |
| `./test_mlir_integration.sh` | **Full system** - Python CLI + MLIR | 45-90s | Testing entire toolchain |
| `./mlir-obs/test.sh` | **MLIR passes only** - Quick pass verification | 5-10s | After code changes to passes |

---

## ✅ **Expected Output**

### Success
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

### Sample Test Output
```
[1/7] Environment Prerequisites
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Polygeist (cgeist): /usr/local/bin/cgeist
✓ clang: /usr/bin/clang
✓ mlir-opt: /usr/local/bin/mlir-opt
✓ mlir-translate: /usr/local/bin/mlir-translate
✓ python3: /usr/bin/python3

[2/7] Building MLIR Obfuscation Library
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PASS: MLIR library build
  Library: mlir-obs/build/lib/libMLIRObfuscation.so

[3/7] Testing Standalone MLIR Passes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PASS: symbol-obfuscate pass available
✅ PASS: string-encrypt pass available
✅ PASS: scf-obfuscate pass available

[6/7] Polygeist Pipeline (C -> func/scf -> Obfuscation -> Binary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PASS: C to Polygeist MLIR generation
✅ PASS: func dialect present
✅ PASS: SCF/Affine dialect present
✅ PASS: Symbol obfuscation (func dialect)
✅ PASS: Symbols obfuscated (func dialect)
  Sample obfuscated symbols:
    func.func @f_a3b2c1d0(%arg0: i32, %arg1: i32) -> i32
    func.func @f_e5f6a7b8(%arg0: i32) -> i32
    func.func @f_c9d0e1f2() -> i32
✅ PASS: SCF obfuscation
✅ PASS: String encryption
✅ PASS: Lowering to LLVM dialect
✅ PASS: MLIR to LLVM IR export
✅ PASS: Binary compilation (Polygeist)
✅ PASS: Binary execution (Polygeist) - exit code 42

[7/7] Obfuscation Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Checking for hidden secrets:
  ✓ HIDDEN: sk_live_abc123
  ✓ HIDDEN: postgresql://
  ✓ HIDDEN: admin:password
✅ PASS: Secret strings hidden

2. Checking for obfuscated symbols:
  ✓ OBFUSCATED: validate_credentials
  ✓ OBFUSCATED: compute_checksum
✅ PASS: Function symbols obfuscated
```

---

## 🎯 **Quick Manual Test**

If you want to test manually with a simple example:

```bash
# 1. Build the library
cd mlir-obs && ./build.sh && cd ..

# 2. Create a test file
cat > simple_test.c << 'EOF'
int add(int a, int b) { return a + b; }
int main() { return add(2, 3); }
EOF

# 3. Run the Polygeist pipeline
./mlir-obs/polygeist-pipeline.sh simple_test.c test_output

# 4. Test execution
./test_output
echo $?  # Should output: 5

# 5. Check obfuscation
nm test_output | grep "add"  # Should NOT find it
nm test_output | grep "f_"   # Should find obfuscated names
```

---

## 🔍 **What Each Pipeline Tests**

### Traditional Pipeline (LLVM Dialect)
```
C source
   ↓ [clang -emit-llvm]
LLVM IR (.ll)
   ↓ [mlir-translate --import-llvm]
MLIR (LLVM dialect)
   ↓ [symbol-obfuscate]
Obfuscated MLIR
   ↓ [mlir-translate --mlir-to-llvmir]
Obfuscated LLVM IR
   ↓ [clang]
Binary
```

### Polygeist Pipeline (High-level Dialects) ⭐ NEW
```
C source
   ↓ [cgeist]
MLIR (func, scf, memref, affine)  ← High-level dialects
   ↓ [symbol-obfuscate]
Obfuscated symbols (func::FuncOp)
   ↓ [scf-obfuscate]
Obfuscated control flow
   ↓ [string-encrypt]
Encrypted strings
   ↓ [lowering passes]
MLIR (LLVM dialect)
   ↓ [mlir-translate]
LLVM IR
   ↓ [clang]
Binary
```

---

## 📊 **Test Coverage**

- ✅ **Environment:** Tools installation, library build
- ✅ **LLVM Dialect:** Traditional pipeline, baseline functionality
- ✅ **Polygeist Dialects:** func, scf, memref, affine
- ✅ **Symbol Obfuscation:** Both LLVM::LLVMFuncOp and func::FuncOp
- ✅ **Control Flow:** SCF obfuscation (Polygeist-specific)
- ✅ **String Encryption:** High-level and low-level
- ✅ **Lowering:** Full dialect lowering pipeline
- ✅ **Binary Execution:** Correctness verification
- ✅ **Obfuscation Verification:** Strings hidden, symbols obfuscated

---

## 🐛 **If Tests Fail**

### Check Prerequisites
```bash
which cgeist        # Polygeist
which mlir-opt      # MLIR tools
which clang         # Compiler
which python3       # CLI
```

### Check Build
```bash
cd mlir-obs
./build.sh
find build -name "*MLIRObfuscation.*"
```

### View Logs
```bash
cat /tmp/mlir_build.log
ls -la /tmp/tmp.*/  # Intermediate files
```

### Run Minimal Test
```bash
cd mlir-obs
./test.sh  # Just test MLIR passes
```

---

## 📝 **Testing Checklist**

Before considering Polygeist integration complete:

- [ ] `./test_polygeist_e2e.sh` passes all tests
- [ ] Polygeist pipeline works: `./mlir-obs/test-polygeist-integration.sh`
- [ ] Can process real C files with the pipeline script
- [ ] Obfuscated binaries execute correctly
- [ ] Symbols are obfuscated in final binary
- [ ] Strings are encrypted in final binary
- [ ] No runtime crashes or undefined behavior

---

## 🎓 **Understanding the Integration**

The Polygeist layer adds support for **high-level MLIR dialects**:

**Before (LLVM dialect only):**
- C → LLVM IR → MLIR (LLVM dialect) → Obfuscate → Binary
- Only works on low-level LLVM operations

**After (Polygeist integration):**
- C → Polygeist → MLIR (func, scf, memref) → Obfuscate → Lower → Binary
- Works on high-level constructs before lowering
- Better optimization opportunities
- More sophisticated obfuscation possible

---

## 📚 **More Information**

- Full guide: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- Pipeline details: See `mlir-obs/polygeist-pipeline.sh`
- Individual tests: See `mlir-obs/test-polygeist-integration.sh`
