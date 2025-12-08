# String Encryption Fix - Complete Solution

**Date**: 2025-12-08
**Issue**: Obfuscated binaries print garbled/encrypted strings instead of decrypted output
**Status**: FIXED ✅

---

## Problem Summary

Obfuscated binaries were executing successfully (exit code 0) but printing encrypted garbage instead of the expected readable text. The string encryption was happening at compile time, but the runtime decryption was never being called.

### Symptoms
- Binary runs without crashing
- Output is garbled: `YX[A=^X3^DE.^K^W` instead of `Hello World!`
- No `__obfs_*` symbols in the binary
- `.init_array` section contains NULL pointers

---

## Root Cause Analysis

### Issue 1: MLIR Plugin Crash
The `MLIRObfuscation.so` plugin was crashing with `std::bad_alloc` in `clonePass()` when loaded by `mlir-opt`.

**Error:**
```
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
mlir::PassWrapper<mlir::obs::StringEncryptPass>::clonePass() const + 274
```

### Issue 2: Missing MLIR Type ID Macros
MLIR 22.x requires `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID()` macro for each pass struct for proper type identification.

### Issue 3: ABI Incompatibility
The plugin was built without matching LLVM's build settings:
- LLVM was built with `LLVM_ENABLE_RTTI=OFF` and `LLVM_ENABLE_EH=OFF`
- Plugin needed `-fno-rtti -fno-exceptions` flags
- Plugin needed to link against `MLIR` and `LLVM` shared libraries

### Issue 4: Symbol Resolution
The plugin had undefined symbol errors:
```
undefined symbol: _ZTIN4mlir4PassE (fatal)
```
This is `typeinfo for mlir::Pass` which wasn't exported because RTTI was disabled.

---

## Fix Implementation

### Step 1: Update Passes.h with Type ID Macros

**File**: `/home/zrahay/oaas/mlir-obs/include/Obfuscator/Passes.h`

Add `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID()` to each pass struct:

```cpp
// ======================= STRING ENCRYPTION PASS ============================
struct StringEncryptPass
    : public PassWrapper<StringEncryptPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StringEncryptPass)  // ADD THIS

  StringEncryptPass() = default;
  StringEncryptPass(const std::string &key) : key(key) {}
  // ... rest unchanged
};

// =================== CONSTANT OBFUSCATION PASS =============================
struct ConstantObfuscationPass
    : public PassWrapper<ConstantObfuscationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantObfuscationPass)  // ADD THIS
  // ... rest unchanged
};

// ======================== SYMBOL OBFUSCATION PASS ==========================
struct SymbolObfuscatePass
    : public PassWrapper<SymbolObfuscatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SymbolObfuscatePass)  // ADD THIS
  // ... rest unchanged
};

// ===================== CRYPTOGRAPHIC HASH PASS =============================
struct CryptoHashPass
    : public PassWrapper<CryptoHashPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CryptoHashPass)  // ADD THIS
  // ... rest unchanged
};

// ======================== SCF OBFUSCATION PASS ==============================
struct SCFObfuscatePass
    : public PassWrapper<SCFObfuscatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFObfuscatePass)  // ADD THIS
  // ... rest unchanged
};

// ====================== IMPORT OBFUSCATION PASS =============================
struct ImportObfuscationPass
    : public PassWrapper<ImportObfuscationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportObfuscationPass)  // ADD THIS
  // ... rest unchanged
};
```

### Step 2: Update CMakeLists.txt for Plugin Build

**File**: `/home/zrahay/oaas/mlir-obs/lib/CMakeLists.txt`

Replace entire contents with:

```cmake
# Build MLIR Obfuscation plugin as a shared module
# This plugin is loaded by mlir-opt via --load-pass-plugin

add_library(MLIRObfuscation SHARED
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
  CryptoHashPass.cpp
  ConstantObfuscationPass.cpp
  SCFPass.cpp
  ImportObfuscationPass.cpp
)

# Set output name and properties
set_target_properties(MLIRObfuscation PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)

target_include_directories(MLIRObfuscation
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

# Link against MLIR/LLVM shared libraries and OpenSSL
target_link_libraries(MLIRObfuscation
  PRIVATE
    MLIR
    LLVM
    OpenSSL::Crypto
)

# Add LLVM/MLIR compile definitions
target_compile_definitions(MLIRObfuscation PRIVATE ${LLVM_DEFINITIONS})

# CRITICAL: Match LLVM build settings - no RTTI, no exceptions
# LLVM was built with LLVM_ENABLE_RTTI=OFF and LLVM_ENABLE_EH=OFF
target_compile_options(MLIRObfuscation PRIVATE -fno-rtti -fno-exceptions)
```

### Step 3: Rebuild the Plugin on GCP VM

```bash
# SSH to VM
ssh devalgupta4@34.93.196.34

# Navigate to mlir-obs directory
cd /home/devalgupta4/oaas/mlir-obs

# Update Passes.h (copy from local or edit directly)
# The file should have MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID for all 6 passes

# Update lib/CMakeLists.txt with the new content above

# Clean and rebuild
cd build3
rm -rf *
cmake .. \
  -DMLIR_DIR=/home/devalgupta4/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/devalgupta4/llvm-project/build/lib/cmake/llvm

make -j$(nproc)
```

### Step 4: Verify the Build

```bash
# Check the plugin was built
ls -la /home/devalgupta4/oaas/mlir-obs/build3/lib/MLIRObfuscation.so
# Should be ~1MB (linked against MLIR/LLVM)

# Check it links against MLIR
ldd /home/devalgupta4/oaas/mlir-obs/build3/lib/MLIRObfuscation.so | grep MLIR
# Should show: libMLIR.so.22.0git
```

### Step 5: Test the Plugin

```bash
# Create test input
cat > /tmp/test_input.mlir << 'EOF'
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.mlir.global internal constant @hello_str("Hello World!\00") : !llvm.array<13 x i8>

  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.addressof @hello_str : !llvm.ptr
    %2 = llvm.call @printf(%1) vararg(!llvm.func<i32 (!llvm.ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}
EOF

# Run string-encrypt pass
/home/devalgupta4/llvm-project/build/bin/mlir-opt /tmp/test_input.mlir \
  --load-pass-plugin=/home/devalgupta4/oaas/mlir-obs/build3/lib/MLIRObfuscation.so \
  --pass-pipeline="builtin.module(string-encrypt)" \
  -o /tmp/test_output.mlir

# Verify output contains encrypted string and __obfs_init
cat /tmp/test_output.mlir | grep -E "__obfs_|global_ctors"
```

**Expected output should show:**
- `@__obfs_key` - encryption key
- `@__obfs_decrypt` - decryption function
- `@__obfs_init` - initialization function
- `llvm.mlir.global_ctors` - constructor registration
- Encrypted string data (not "Hello World!")

### Step 6: Full End-to-End Test

```bash
# Translate MLIR to LLVM IR
/home/devalgupta4/llvm-project/build/bin/mlir-translate \
  --mlir-to-llvmir /tmp/test_output.mlir \
  -o /tmp/test_output.ll

# Compile to binary
clang /tmp/test_output.ll -o /tmp/test_binary

# Run the binary
/tmp/test_binary
# Should output: Hello World!
```

---

## Verification Checklist

- [x] Plugin loads without crash
- [x] `string-encrypt` pass is recognized
- [x] Strings are encrypted in MLIR output
- [x] `__obfs_key` global is created
- [x] `__obfs_decrypt` function is generated
- [x] `__obfs_init` function calls decrypt for each string
- [x] `llvm.mlir.global_ctors` registers `__obfs_init`
- [x] LLVM IR contains `@llvm.global_ctors` array
- [x] Compiled binary has non-empty `.init_array` section
- [x] Binary outputs correct decrypted strings at runtime

---

## Files Changed

| File | Change |
|------|--------|
| `mlir-obs/include/Obfuscator/Passes.h` | Added `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID()` to all 6 pass structs |
| `mlir-obs/lib/CMakeLists.txt` | Added MLIR/LLVM linking, `-fno-rtti -fno-exceptions` flags |

---

## Plugin Locations

| Location | Path |
|----------|------|
| GCP VM (build) | `/home/devalgupta4/oaas/mlir-obs/build3/lib/MLIRObfuscation.so` |
| Local copy | `/home/zrahay/oaas/MLIRObfuscation.so` |

---

## Deployment Notes

The rebuilt plugin needs to be:
1. Uploaded to GCP Cloud Storage tarball (`llvm-obfuscator-binaries.tar.gz`)
2. Deployed to production server container

**Important**: The plugin is now ~1MB (was ~200KB) because it links against MLIR/LLVM shared libraries. This is required for proper symbol resolution.

---

## Technical Details

### Why RTTI Was the Issue

LLVM was built with:
```
LLVM_ENABLE_RTTI=OFF
LLVM_ENABLE_EH=OFF
```

This means:
- No `typeinfo` symbols are generated for LLVM/MLIR classes
- The plugin must also be built with `-fno-rtti` to match
- Without this, the plugin tries to reference non-existent RTTI symbols

### Why Linking Against MLIR/LLVM Was Required

Even with `-fno-rtti`, the plugin needs symbols from MLIR for:
- Pass registration infrastructure
- MLIR dialect operations (LLVM dialect)
- Type system utilities

By linking against `MLIR` and `LLVM` shared libraries, these symbols are resolved at load time from the same libraries that `mlir-opt` uses.

### The Global Constructor Mechanism

The string encryption works via:
1. `StringEncryptPass` creates `LLVM::GlobalCtorsOp` in MLIR
2. `mlir-translate` converts this to `@llvm.global_ctors` in LLVM IR
3. The linker creates `.init_array` section from `@llvm.global_ctors`
4. At runtime, the loader calls functions in `.init_array` before `main()`
5. `__obfs_init()` XOR-decrypts all encrypted strings in-place
6. `main()` runs with decrypted strings

---

## Commands Summary

```bash
# On GCP VM (34.93.196.34)

# 1. Update Passes.h with type ID macros (see Step 1)

# 2. Update lib/CMakeLists.txt (see Step 2)

# 3. Rebuild
cd /home/devalgupta4/oaas/mlir-obs/build3
rm -rf *
cmake .. \
  -DMLIR_DIR=/home/devalgupta4/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/devalgupta4/llvm-project/build/lib/cmake/llvm
make -j$(nproc)

# 4. Test
/home/devalgupta4/llvm-project/build/bin/mlir-opt /tmp/test_input.mlir \
  --load-pass-plugin=/home/devalgupta4/oaas/mlir-obs/build3/lib/MLIRObfuscation.so \
  --pass-pipeline="builtin.module(string-encrypt)" \
  -o /tmp/test_output.mlir

# 5. Full test
/home/devalgupta4/llvm-project/build/bin/mlir-translate --mlir-to-llvmir /tmp/test_output.mlir -o /tmp/test_output.ll
clang /tmp/test_output.ll -o /tmp/test_binary
/tmp/test_binary  # Should print: Hello World!
```
