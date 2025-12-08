# MLIR String Encryption: Decryption Failure Analysis Report

**Date:** 2025-12-08
**Binaries Analyzed:**
- `obfuscated_linux (1)` - 450,248 bytes
- `obfuscated_linux (2)` - 6,080 bytes

---

## Executive Summary

**Critical Finding: Runtime string decryption is completely broken in both binaries.**

Both obfuscated binaries exhibit the same fundamental failure: the `__obfs_init` constructor function that should decrypt strings at program startup is never executed. This results in:
1. All program output appearing as encrypted gibberish
2. Programs being non-functional
3. XOR encryption key material leaking into the binary

---

## 1. Binary Analysis Evidence

### 1.1 File Properties

| Property | Binary 1 | Binary 2 |
|----------|----------|----------|
| **Size** | 450,248 bytes | 6,080 bytes |
| **Format** | ELF 64-bit LSB PIE | ELF 64-bit LSB PIE |
| **Arch** | x86-64 | x86-64 |
| **Stripped** | Yes | Yes |
| **String Count** | 937 | 68 |
| **Linker** | LLD 22.0.0 | LLD 22.0.0 |

### 1.2 Section Analysis

Both binaries have the necessary ELF sections:

```
Binary 1 Sections:
  [10] .rodata           PROGBITS    0x2840
  [15] .init             PROGBITS    0x6d838
  [19] .fini_array       FINI_ARRAY  0x6ed70
  [20] .init_array       INIT_ARRAY  0x6ed78

Binary 2 Sections:
  [10] .rodata           PROGBITS    0x5b0
  [14] .init             PROGBITS    0x1c24
  [17] .fini_array       FINI_ARRAY  0x2c90
  [18] .init_array       INIT_ARRAY  0x2c98
```

---

## 2. Root Cause: Empty .init_array

### 2.1 Critical Evidence

**Binary 1 `.init_array` contents:**
```
Hex dump of section '.init_array':
  0x0006ed78 00000000 00000000 00000000 00000000 ................
```

**Binary 2 `.init_array` contents:**
```
Hex dump of section '.init_array':
  0x00002c98 00000000 00000000                   ........
```

**Both init_array sections are filled with NULL bytes (zeros).**

This is the definitive proof that no constructor functions are registered. The `__obfs_init` function that should decrypt strings before `main()` executes is never called because its address is not present in the `.init_array`.

### 2.2 Expected vs Actual Behavior

| Expected | Actual |
|----------|--------|
| `.init_array` contains pointer to `__obfs_init` | `.init_array` contains only zeros |
| `__obfs_init` called before `main()` | No constructors called |
| Strings decrypted at startup | Strings remain encrypted |
| Program outputs readable text | Program outputs encrypted garbage |

---

## 3. Runtime Execution Evidence

### 3.1 Binary 1 Execution (without arguments)

```
$ /tmp/test_obfuscated_auth
neYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HftDEFAU#58E>A06
YX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HftDEFAU#58E>A06
DEFAU#58E>A06YX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\HQIbVXDYX[\Hf
~_+$'2*):"+<;)/"*^Dm_:)6 +( !+...
```

### 3.2 Binary 2 Execution (without arguments)

```
$ /tmp/test_obfuscated_auth2
YX[A4?$35=1T?$35=1T
E	,Kft?$35=1Tn>546/188Y%T8?$6((L!,D$6(U&QE\ofn>320>);
U?!$<U/1?!$<U/1n>320>);
UDE3>DE4	N*+sdDE'	,K)DE'	,K)
```

Both binaries output encrypted gibberish, confirming decryption failure.

---

## 4. XOR Key Leakage Analysis

### 4.1 Evidence of Key Leakage (Binary 1 Only)

The string `KEYDEFAULT` appears multiple times in Binary 1:

```
strings "obfuscated_linux (1)" | grep KEY
KEYD"
KEYDEFAULT
KEYDEFAULTUk
KEYDEFAUft? (&<"1
```

### 4.2 Hex Dump Evidence (Offset 0x2d40-0x2d80)

```
00002d40: 486c 6144 4546 4155 4c54 7f4b 4559 4422  HlaDEFAULT.KEYD"
00002d50: 070c 104c 3131 0c0c 1701 4522 0418 0354  ...L11....E"...T
00002d60: 1c04 0809 0800 1204 544c 547f 4b45 5944  ........TLT.KEYD
00002d70: 4546 4155 4c54 7f4b 4559 4445 4641 554c  EFAULT.KEYDEFAUL
00002d80: 5455 6b00 4445 4641 554c 547f 4b2a 3825  TUk.DEFAULT.K*8%
```

### 4.3 Key Leakage Mechanism

The XOR encryption key leaks when the plaintext contains null bytes:

```
encrypted = plaintext XOR key
If plaintext[i] = 0x00, then:
encrypted[i] = 0x00 XOR key[i] = key[i]
```

This means:
1. The XOR key appears to be `KEYDEFAULT` (10 bytes)
2. Wherever the original plaintext had null bytes, the key is directly visible
3. This is a **cryptographic weakness** - the key is exposed in the binary

### 4.4 Binary 2 Key Analysis

Binary 2 does NOT show `KEYDEFAULT` in strings, suggesting either:
- A different key was used
- The plaintext had fewer null bytes
- The encryption is applied differently

---

## 5. MLIR Pass Analysis

### 5.1 Code Location

The bug originates in `mlir-obs/lib/ConstantObfuscationPass.cpp`:

```cpp
// Line 231-292: GlobalCtorsOp creation
builder.create<LLVM::GlobalCtorsOp>(
    loc,
    builder.getArrayAttr(ctors),      // Contains "__obfs_init"
    builder.getArrayAttr(priorities), // Contains 101 (high priority)
    builder.getArrayAttr(data)        // Contains ZeroAttr
);
```

### 5.2 Suspected Root Cause

The `LLVM::GlobalCtorsOp` is created correctly in the MLIR pass, but the problem occurs during **MLIR-to-LLVM IR translation** or **linking**:

1. **Translation Issue**: The `GlobalCtorsOp` may not be properly lowered to `@llvm.global_ctors` during `mlir-translate`
2. **Linker Issue**: LLD may be discarding the global constructors due to:
   - Incorrect section assignment
   - Dead code elimination
   - Linkage issues with `__obfs_init`

### 5.3 Dynamic Section Evidence

```
$ readelf -d /tmp/test_obfuscated_auth | grep -i init
 0x0000000000000019 (INIT_ARRAY)         0x6ed78
 0x000000000000001b (INIT_ARRAYSZ)       16 (bytes)
```

The dynamic section correctly references `.init_array` at `0x6ed78`, but the array contains only zeros.

---

## 6. Security Implications

### 6.1 Positive: Secrets Are Hidden (Static Analysis)

No sensitive plaintext found in either binary:
```
$ strings binary | grep -iE "password|secret|api_key|postgres"
(no output)
```

### 6.2 Negative: Key Leakage (Binary 1)

The XOR encryption key `KEYDEFAULT` is partially visible, allowing:
1. Identification of the encryption scheme
2. Potential decryption of all encrypted strings
3. Complete bypass of the obfuscation

### 6.3 Critical: Non-Functional Binaries

Both binaries are **completely non-functional** due to decryption failure:
- All user-visible strings are garbled
- Program behavior is undefined
- Authentication logic cannot work properly

---

## 7. Comparison Summary

| Metric | Binary 1 (450KB) | Binary 2 (6KB) |
|--------|------------------|----------------|
| **Decryption** | BROKEN | BROKEN |
| **init_array** | Empty (zeros) | Empty (zeros) |
| **Key Visible** | Yes ("KEYDEFAULT") | No |
| **Secrets Hidden** | Yes | Yes |
| **Functional** | No | No |
| **Program Type** | C++ (complex) | C (simple) |

---

## 8. Recommendations

### 8.1 Immediate Fix Required

1. **Investigate MLIR Translation**: Check if `mlir-translate --mlir-to-llvmir` properly converts `GlobalCtorsOp`
2. **Verify LLVM IR Output**: Ensure `@llvm.global_ctors` exists in the intermediate LLVM IR
3. **Check Linker Flags**: Ensure LLD isn't stripping constructors with `-gc-sections` or similar

### 8.2 Key Management Improvements

1. Use a **random key per string** instead of a single global key
2. **Never use predictable keys** like "KEYDEFAULT" or "default_key"
3. Consider **key derivation** from binary-specific data

### 8.3 Testing Protocol

Before deployment, verify:
```bash
# Check .init_array is not empty
readelf -x .init_array binary | grep -v "00000000 00000000"

# Test program execution
./binary 2>&1 | head -5  # Should show readable text
```

---

## 9. Technical Details

### 9.1 LLVM Version

Both binaries linked with:
```
Linker: LLD 22.0.0 (https://github.com/SkySingh04/llvm-project.git 66345e7cc7edc792c3cb02466e516441aa4a65f7)
```

### 9.2 Build System

GCC/Clang compilation with Debian 14.2.0-19 toolchain.

### 9.3 Files Analyzed

```
-rw-r--r--  450248 Dec  8 10:24 obfuscated_linux (1)
-rw-r--r--    6080 Dec  8 11:09 obfuscated_linux (2)
```

---

## 10. Conclusion

**The MLIR ConstantObfuscationPass has a critical bug where the `__obfs_init` constructor is not being registered in the `.init_array` section.** This results in:

1. Complete decryption failure at runtime
2. Non-functional obfuscated binaries
3. XOR key leakage (in some cases)

The bug appears to be in the MLIR-to-LLVM-IR translation or linking stage, where the `LLVM::GlobalCtorsOp` is not being properly converted to `.init_array` entries.

**Priority: CRITICAL - This renders the entire string encryption feature non-functional.**
