# OAAS MLIR/macOS Compilation Bugs Analysis

**Date:** 2025-12-05
**Analyst:** Claude Code
**Production URL:** https://oaas.pointblank.club

---

## Executive Summary

MLIR-based obfuscation layers (Layer 1: Symbol Obfuscation, Layer 2: String Encryption) fail when targeting macOS. Multiple distinct errors occur depending on the source language (C vs C++) and which layers are enabled. Additionally, runtime string decryption is broken across ALL platforms.

---

## Test Results Matrix

| Test Case | Platform | Layers | Result | Error Type |
|-----------|----------|--------|--------|------------|
| C code | Linux | Layer 1+2 | **COMPILES** | Runtime decryption broken |
| C code | Windows | Layer 1+2 | **COMPILES** | Runtime decryption broken |
| C code | macOS | Layer 1+2 | **FAIL** | COMDAT error |
| C++ code | Linux | Layer 1+2 | **COMPILES** | Runtime decryption broken |
| C++ code | macOS | Layer 1+2 | **FAIL** | `captures(none)` syntax error |
| C++ code | macOS | ALL layers | **FAIL** | Section specifier corruption |

---

## Bug #1: COMDAT Error on macOS (C code)

### Error Message
```
fatal error: error in backend: MachO doesn't support COMDATs, '__llvm_retpoline_r11' cannot be lowered.
```

### Reproduction Steps
1. Navigate to https://oaas.pointblank.club
2. Select PASTE > Load "Authentication System (C)"
3. Enable Layer 1 (Symbol Obfuscation) and Layer 2 (String Encryption)
4. Select Target Platform: macOS (ARM64)
5. Click OBFUSCATE

### Evidence from UI
```
Command failed with exit code 1: clang /app/reports/.../pasted_source_from_mlir.ll -o ...
-fvisibility=hidden -O3 -fno-builtin -fomit-frame-pointer -mspeculative-load-hardening -Wl,-s
--target=x86_64-apple-darwin -isysroot /app/macos-sdk/MacOSX15.4.sdk ...

fatal error: error in backend: MachO doesn't support COMDATs, '__llvm_retpoline_r11' cannot be lowered.
```

### Evidence from Server Logs
```
2025-12-05 11:56:14,230 - core.obfuscator - INFO - Running MLIR pipeline with passes: string-encrypt
2025-12-05 11:56:14,232 - core.obfuscator - INFO - Target triple: x86_64-apple-darwin (platform=macos, arch=x86_64)
fatal error: error in backend: MachO doesn't support COMDATs, '__llvm_retpoline_r11' cannot be lowered.
```

### Root Cause
The `-mspeculative-load-hardening` flag in `BASE_FLAGS` (`cmd/llvm-obfuscator/core/obfuscator.py:40`) generates retpoline code for Spectre v1 mitigation. Retpoline functions use COMDAT sections for de-duplication, but **Mach-O (macOS binary format) does not support COMDAT sections**.

### Affected Code
```python
# cmd/llvm-obfuscator/core/obfuscator.py:35-42
BASE_FLAGS = [
    "-fvisibility=hidden",
    "-O3",
    "-fno-builtin",
    "-fomit-frame-pointer",
    "-mspeculative-load-hardening",  # <-- CAUSES COMDAT ERROR ON MACOS
    "-Wl,-s",
]
```

### Fix Required
Filter out `-mspeculative-load-hardening` when targeting macOS:
```python
if config.platform == Platform.MACOS:
    compiler_flags = [f for f in compiler_flags if f != "-mspeculative-load-hardening"]
```

---

## Bug #2: LLVM Version Mismatch (C++ code on macOS)

### Error Message
```
error: expected ')' at end of argument list
4116 | declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #11
```

### Reproduction Steps
1. Navigate to https://oaas.pointblank.club
2. Select PASTE > Load "License Validator (C++)"
3. Enable Layer 1 and Layer 2 only
4. Select Target Platform: macOS (ARM64)
5. Click OBFUSCATE

### Evidence from Server Logs
```
2025-12-05 11:58:23,615 - core.obfuscator - INFO - [RESOURCE-DIR-DEBUG] Resolved 'clang++' to '/usr/bin/clang++'
2025-12-05 11:58:23,615 - core.obfuscator - INFO - [RESOURCE-DIR-DEBUG] is_custom_clang=False for path=/usr/bin/clang++
...
error: expected ')' at end of argument list
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ...)
```

### Version Evidence
```bash
$ /usr/bin/clang++ --version
Debian clang version 19.1.7 (LLVM 19)

$ /usr/local/llvm-obfuscator/bin/clang --version
clang version 22.0.0git (LLVM 22)
```

### Root Cause
1. The MLIR pipeline uses **LLVM 22** binaries (`mlir-translate`, `mlir-opt`)
2. LLVM 22's `mlir-translate` generates LLVM IR with `captures(none)` attribute (new in LLVM 22)
3. For C++ code, the compiler selection falls back to **system clang++** (`/usr/bin/clang++` = LLVM 19)
4. LLVM 19 doesn't understand the `captures(none)` syntax, causing parse error

### Affected Code
The compiler resolution logic in `obfuscator.py` doesn't correctly use the bundled LLVM 22 `clang++` for C++ files.

### Fix Required
Ensure C++ compilation uses bundled `/usr/local/llvm-obfuscator/bin/clang++` instead of system `/usr/bin/clang++`.

---

## Bug #3: Section Specifier Corruption (C++ with ALL layers on macOS)

### Error Message
```
fatal error: error in backend: Global variable '__cxx_global_var_init' has an invalid section specifier ';:2$-8X
```

### Reproduction Steps
1. Navigate to https://oaas.pointblank.club
2. Select PASTE > Load "License Validator (C++)"
3. Click "Select All" to enable ALL layers (1, 2, 2.5, 3, 4, 5)
4. Select Target Platform: macOS (ARM64)
5. Click OBFUSCATE

### Evidence from UI
```
Command failed with exit code 1: /usr/local/llvm-obfuscator/bin/clang
/app/reports/.../pasted_source_obfuscated.bc -o ...
--target=x86_64-apple-darwin ...

fatal error: error in backend: Global variable '__cxx_global_var_init' has an invalid section specifier ';:2$-8X
```

### Evidence from Server Logs
```
2025-12-05 12:06:14,533 - core.obfuscator - INFO - Running MLIR pipeline with passes: string-encrypt
2025-12-05 12:06:16,452 - core.obfuscator - INFO - Running OLLVM pipeline with passes: flattening, substitution, boguscf, split, linear-mba
fatal error: error in backend: Global variable '__cxx_global_var_init' has an invalid section specifier ';:2$-8X
```

### Root Cause
The MLIR string encryption pass (`string-encrypt`) is corrupting section specifier attributes for C++ global initializers:
1. C++ global variables (like `const std::string`) have compiler-generated initializers (`__cxx_global_var_init`)
2. These initializers have section attributes (e.g., `__TEXT,__StaticInit` on macOS)
3. The string encryption pass incorrectly treats section specifier strings as encryptable data
4. The corrupted section specifier (`;:2$-8X`) is XOR-encrypted garbage

### Affected Component
`/app/plugins/linux-x86_64/MLIRObfuscation.so` - The string encryption pass

### Fix Required
In the MLIR string encryption pass, add filters to skip:
1. Section specifier attributes (not actual string data)
2. Global initializer symbols (names starting with `__cxx_global_var_init`, `_GLOBAL__sub_I_`, etc.)
3. LLVM/compiler-generated metadata strings

---

## Bug #4: Runtime String Decryption Broken (ALL platforms)

### Symptom
Obfuscated binaries output encrypted garbage instead of readable strings at runtime.

### Reproduction Steps
1. Complete any successful obfuscation on Linux with Layer 2 enabled
2. Download and run the binary
3. Observe garbage output instead of expected strings

### Evidence - Strings Successfully Encrypted (not visible)
```bash
$ strings pasted_source | grep -iE "(admin|pass|key|secret)"
# No output - strings are encrypted
```

### Evidence - Runtime Decryption Fails
```bash
$ ./pasted_source
YX[A4?$35=1T?$35=1T
E	,Kft?$35=1Tn>546/188Y%T8?$6((L!,D$6(U&QE\ofn>320>);
```

### Expected Output
```
=== Authentication System v1.0 ===
[AUTH] Checking credentials for: admin
[AUTH] Admin access granted
[SUCCESS] Access granted
```

### Evidence - Report Shows 0 Strings Encrypted (Reporting Bug)
```json
"string_obfuscation": {
    "total_strings": 0,
    "encrypted_strings": 0,
    "encryption_method": "none",
    "encryption_percentage": 0.0
}
```

### Root Cause
The MLIR string-encrypt pass encrypts strings but fails to generate working runtime decryption. Possible causes:
1. Decryption stubs not generated
2. Decryption stubs not linked correctly
3. Key management broken
4. Decryption function not called at runtime

### Affected Component
`/app/plugins/linux-x86_64/MLIRObfuscation.so` - The string encryption pass

### Fix Required
Debug and fix the MLIR string encryption pass to ensure:
1. Decryption stubs are generated for each encrypted string
2. Static initializers call decryption before string use
3. Encryption keys are properly embedded and accessible at runtime

---

## Additional Observations

### UI/Architecture Issue: Target Triple Mismatch
When selecting "macOS (ARM64)" in the UI, the actual target triple used is `x86_64-apple-darwin` (Intel), not `arm64-apple-darwin`. This is a separate UI bug.

### C vs C++ Compiler Selection (After Fix)
- **C code**: Uses bundled `/usr/local/llvm-obfuscator/bin/clang` (LLVM 22) - ✅ WORKING
- **C++ code**: Uses bundled `/usr/local/llvm-obfuscator/bin/clang` with `-x c++` flag (LLVM 22) - ✅ Version mismatch fixed, but blocked by Bug #3

---

## Summary of Fixes

| Bug | Component | Priority | Status | Fix Applied |
|-----|-----------|----------|--------|-------------|
| #1 COMDAT | `obfuscator.py` | High | ✅ **FIXED** | Filter out `-mspeculative-load-hardening` for macOS |
| #2 Version Mismatch | `obfuscator.py` | High | ✅ **FIXED** | Use bundled LLVM 22 clang, no resource-dir override |
| #3 Section Corruption | `MLIRObfuscation.so` | Critical | ❌ Pending | Requires MLIR plugin changes |
| #4 Runtime Decryption | `MLIRObfuscation.so` | Critical | ❌ Pending | Requires MLIR plugin changes |
| UI Triple | Frontend | Low | ❌ Pending | Fix ARM64 target triple selection |

### Fix Details (2025-12-05)

**Bug #1 Fix** (`obfuscator.py:799-802`):
```python
if config.platform in [Platform.MACOS, Platform.DARWIN]:
    if "-mspeculative-load-hardening" in compiler_flags:
        compiler_flags = [f for f in compiler_flags if f != "-mspeculative-load-hardening"]
```

**Bug #2 Fix** (`obfuscator.py:524-528`, `775-797`):
- Use bundled clang at `/usr/local/llvm-obfuscator/bin/clang` for both C and C++
- For C++, add `-x c++` and `-lstdc++` flags
- Don't override resource-dir for bundled clang (it has complete headers at `/usr/local/llvm-obfuscator/lib/clang/22`)

### Current Status
- **C + macOS + MLIR layers**: ✅ WORKING
- **C++ + macOS + MLIR layers**: ❌ Blocked by Bug #3 (section specifier corruption in MLIR plugin)

---

## Appendix: Container Versions

```
System clang++:  Debian clang version 19.1.7 (LLVM 19)
Bundled clang:   clang version 22.0.0git (LLVM 22)
MLIR tools:      LLVM 22.0.0git
```
