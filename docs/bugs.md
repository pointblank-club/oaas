# OAAS MLIR/macOS Compilation Bugs - Open Issues

**Date:** 2025-12-05
**Production URL:** https://oaas.pointblank.club

---

## Current Status

| Test Case | Platform | Layers | Result |
|-----------|----------|--------|--------|
| C code | Linux/Windows | Layer 1+2 | **COMPILES** (runtime decryption broken) |
| C code | macOS | Layer 1+2 | ✅ **WORKING** |
| C++ code | Linux | Layer 1+2 | **COMPILES** (runtime decryption broken) |
| C++ code | macOS | Layer 1+2 | ❌ Blocked by Bug #3 |

---

## Bug #3: Section Specifier Corruption (C++ on macOS)

### Error Message
```
fatal error: error in backend: Global variable '__cxx_global_var_init' has an invalid section specifier ';:2$-8X
```

### Reproduction Steps
1. Navigate to https://oaas.pointblank.club
2. Select PASTE > Load "License Validator (C++)"
3. Enable Layer 1 and Layer 2
4. Select Target Platform: macOS (ARM64)
5. Click OBFUSCATE

### Root Cause
The MLIR string encryption pass (`string-encrypt`) corrupts section specifier attributes for C++ global initializers:
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

### Evidence
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

## Minor Issue: UI Target Triple Mismatch

When selecting "macOS (ARM64)" in the UI, the actual target triple used is `x86_64-apple-darwin` (Intel), not `arm64-apple-darwin`.

---

## Summary

| Bug | Component | Priority | Status |
|-----|-----------|----------|--------|
| #3 Section Corruption | `MLIRObfuscation.so` | Critical | ❌ Pending |
| #4 Runtime Decryption | `MLIRObfuscation.so` | Critical | ❌ Pending |
| UI Triple | Frontend | Low | ❌ Pending |

Both critical bugs require changes to the MLIR plugin C++ code.
