# Demo Program Testing Report - OAAS LLVM Obfuscator

**Test Date**: December 5, 2025
**Production URL**: https://oaas.pointblank.club
**Testing Method**: Playwright Browser Automation
**Configuration**: All obfuscation layers enabled

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Demos Tested | 13 |
| Successful | 11 (84.6%) |
| Failed | 2 (15.4%) |
| Average Score (Successful) | 99.8/100 |

**Update (December 5, 2025)**: Bug fix applied - Demos 7 and 8 now pass after fixing LLVM 22+ attribute parsing issue.

---

## Obfuscation Layers Configuration

All tests were conducted with the following layers enabled:

### Layer 1: Symbol Obfuscation (PRE-COMPILE)
- Renames function names, variable names, and other symbols

### Layer 2: String Encryption (PRE-COMPILE)
- Encrypts string literals in the source code

### Layer 2.5: Indirect Call Obfuscation (PRE-COMPILE)
- Replaces direct function calls with indirect calls through function pointers

### Layer 3: OLLVM Passes
- **Control Flow Flattening**: Converts control flow into switch-based state machine
- **Instruction Substitution**: Replaces arithmetic operations with equivalent complex expressions
- **Bogus Control Flow**: Inserts fake conditional branches
- **Split Basic Blocks**: Breaks basic blocks into smaller pieces
- **Linear MBA (Mixed Boolean-Arithmetic)**: Applies MBA transformations

### Layer 4: Compiler Flags
- Symbol Hiding (`-fvisibility=hidden`)
- Remove Frame Pointer (`-fomit-frame-pointer`)
- Speculative Load Hardening (`-mspeculative-load-hardening`)
- Max Optimization (`-O3`)
- Strip Symbols (`-s`)
- Disable Built-in Functions (`-fno-builtin`)

### Layer 5: UPX Binary Packing (POST-COMPILE)
- Compresses and packs the final binary

---

## Test Results Summary

| # | Demo Name | Language | Status | Score | Layers Applied | Notes |
|---|-----------|----------|--------|-------|----------------|-------|
| 1 | Hello World | C | PASS | 99.1/100 | ALL | All layers successfully applied |
| 2 | Hello World | C++ | PASS | 100.0/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 3 | Authentication System | C | PASS | 100.0/100 | ALL | All layers successfully applied |
| 4 | Password Strength Checker | C | PASS | 100.0/100 | ALL | All layers successfully applied |
| 5 | Fibonacci Calculator | C | PASS | 100.0/100 | ALL | All layers successfully applied |
| 6 | QuickSort Algorithm | C++ | PASS | 100.0/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 7 | Matrix Operations | C | PASS | 100.0/100 | ALL | Fixed: LLVM 22+ attribute stripping |
| 8 | Signal Processing DSP | C | PASS | 100.0/100 | ALL | Fixed: LLVM 22+ attribute stripping |
| 9 | Exception Handler | C++ | PASS | 99.7/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 10 | License Validator | C++ | PASS | 100.0/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 11 | Configuration Manager | C++ | PASS | 100.0/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 12 | SQL Database Engine | C | FAIL | - | - | MLIR pass loses function declarations |
| 13 | Game Engine | C++ | FAIL | - | - | MLIR string encryption size mismatch |

---

## Detailed Test Results

### Demo 1: Hello World (C)
**Status**: PASS
**Score**: 99.1/100
**Layers Applied**: ALL

**Evidence**:
- All OLLVM passes successfully applied
- UPX packing completed
- All compiler flags applied
- No warnings or errors

---

### Demo 2: Hello World (C++)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: PARTIAL

**Evidence**:
- Warning: "C++ exception handling detected, flattening pass disabled for stability"
- All other OLLVM passes applied (Instruction Substitution, Bogus Control Flow, Split Basic Blocks, Linear MBA)
- UPX packing completed
- All compiler flags applied

**Partial Layer Reason**: Control Flow Flattening automatically disabled due to C++ exception handling compatibility

---

### Demo 3: Authentication System (C)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: ALL

**Evidence**:
- All OLLVM passes successfully applied
- UPX packing completed
- All compiler flags applied
- No warnings or errors

---

### Demo 4: Password Strength Checker (C)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: ALL

**Evidence**:
- All OLLVM passes successfully applied
- UPX packing completed
- All compiler flags applied
- No warnings or errors

---

### Demo 5: Fibonacci Calculator (C)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: ALL

**Evidence**:
- All OLLVM passes successfully applied
- UPX packing completed
- All compiler flags applied
- No warnings or errors

---

### Demo 6: QuickSort Algorithm (C++)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: PARTIAL

**Evidence**:
- Warning: "C++ exception handling detected, flattening pass disabled for stability"
- All other OLLVM passes applied
- UPX packing completed
- All compiler flags applied

**Partial Layer Reason**: Control Flow Flattening automatically disabled due to C++ exception handling compatibility

---

### Demo 7: Matrix Operations (C)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: ALL

**Evidence**:
- All OLLVM passes successfully applied
- UPX packing completed
- All compiler flags applied
- No warnings or errors

**Fix Applied**: LLVM 22+ attribute stripping added to obfuscator.py to handle math intrinsic attributes (`nocreateundeforpoison`, `memory(none)`, `speculatable`). Also added `-lm` flag for math library linking.

---

### Demo 8: Signal Processing DSP (C)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: ALL

**Evidence**:
- All OLLVM passes successfully applied
- UPX packing completed
- All compiler flags applied
- No warnings or errors

**Fix Applied**: Same fix as Demo 7 - LLVM 22+ attribute stripping and `-lm` flag for math library linking.

---

### Demo 9: Exception Handler (C++)
**Status**: PASS
**Score**: 99.7/100
**Layers Applied**: PARTIAL

**Evidence**:
- Warning: "C++ exception handling detected, flattening pass disabled for stability"
- All other OLLVM passes applied
- UPX packing completed
- All compiler flags applied

**Partial Layer Reason**: Control Flow Flattening automatically disabled due to C++ exception handling compatibility

---

### Demo 10: License Validator (C++)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: PARTIAL

**Evidence**:
- Warning: "C++ exception handling detected, flattening pass disabled for stability"
- All other OLLVM passes applied
- UPX packing completed
- All compiler flags applied

**Partial Layer Reason**: Control Flow Flattening automatically disabled due to C++ exception handling compatibility

---

### Demo 11: Configuration Manager (C++)
**Status**: PASS
**Score**: 100.0/100
**Layers Applied**: PARTIAL

**Evidence**:
- Warning: "C++ exception handling detected, flattening pass disabled for stability"
- All other OLLVM passes applied
- UPX packing completed
- All compiler flags applied

**Partial Layer Reason**: Control Flow Flattening automatically disabled due to C++ exception handling compatibility

---

### Demo 12: SQL Database Engine (C)
**Status**: FAIL
**Score**: N/A
**Layers Applied**: N/A

**Error**:
```
/app/reports/.../pasted_source_from_mlir.ll:5954:31: error: use of undefined value '@strcmp'
call i32 @strcmp(ptr %encrypted_str, ptr @.str.49)
```

**Root Cause**: MLIR string encryption pass loses external function declarations. When the pass encrypts string literals, it fails to preserve the declarations for external functions like `strcmp`, `strlen`, etc. that are referenced in the code. This is a separate bug from the LLVM 22+ attribute issue (which was fixed).

**Bug Classification**: MLIR Pass Bug - Function Declaration Loss

---

### Demo 13: Game Engine (C++)
**Status**: FAIL
**Score**: N/A
**Layers Applied**: N/A

**Error**:
```
/app/reports/.../pasted_source_from_mlir.ll:50886:31: error: constant expression type mismatch: got type '[22 x i8]' but expected '[24 x i8]'
```

**Root Cause**: MLIR string encryption pass corrupts string size metadata. When encrypting strings, the pass miscalculates the resulting array size, causing a type mismatch between the declared type and the actual encrypted content. This is a separate bug from Demo 12 but also in the MLIR string encryption pass.

**Note**: Original compilation error (missing `#include <climits>`) was fixed in demo code. Current failure is due to MLIR pass bug.

**Bug Classification**: MLIR Pass Bug - String Size Corruption

---

## Error Pattern Analysis

### Pattern 1: Math.h LLVM IR Parsing Errors (FIXED)
**Affected Demos**: 7, 8 (previously also 12, 13)
**Status**: RESOLVED

**Original Issue**: Programs using `<math.h>` functions generated LLVM intrinsics with LLVM 22+ attributes (`nocreateundeforpoison`, `memory(none)`, `speculatable`) that the obfuscation pipeline could not parse.

**Fix Applied**: Added regex-based attribute stripping in `obfuscator.py` before passing IR to `opt`. Also added `-lm` linker flag for math library functions.

**Result**: Demos 7 and 8 now pass successfully.

### Pattern 2: C++ Exception Handling
**Affected Demos**: 2, 6, 9, 10, 11
**Behavior**: Control Flow Flattening pass is automatically disabled when C++ exception handling is detected.

**Reason**: Control Flow Flattening can corrupt exception handling tables, causing undefined behavior at runtime.

**Status**: This is expected behavior, not a bug. The system correctly detects and adapts.

### Pattern 3: MLIR Pass Bugs (NEW - OPEN)
**Affected Demos**: 12, 13
**Status**: UNRESOLVED - Requires MLIR pass fixes

**Bug 3a: Function Declaration Loss (Demo 12)**
- MLIR string encryption pass loses external function declarations
- Error: `undefined value '@strcmp'`
- Functions like `strcmp`, `strlen` are called but not declared in output IR

**Bug 3b: String Size Corruption (Demo 13)**
- MLIR string encryption pass miscalculates encrypted string array sizes
- Error: `constant expression type mismatch: got type '[22 x i8]' but expected '[24 x i8]'`
- Type mismatch between declared array size and actual content

**Recommendation**: Both bugs require investigation and fixes in the MLIR string encryption pass (`mlir-string-obfuscation`).

---

## Layer Application Summary

### Fully Applied (All Layers)
- Demo 1: Hello World (C)
- Demo 3: Authentication System (C)
- Demo 4: Password Strength Checker (C)
- Demo 5: Fibonacci Calculator (C)
- Demo 7: Matrix Operations (C) - FIXED
- Demo 8: Signal Processing DSP (C) - FIXED

### Partially Applied (Flattening Disabled)
- Demo 2: Hello World (C++)
- Demo 6: QuickSort Algorithm (C++)
- Demo 9: Exception Handler (C++)
- Demo 10: License Validator (C++)
- Demo 11: Configuration Manager (C++)

### Not Applied (Failed - MLIR Pass Bugs)
- Demo 12: SQL Database Engine (C) - Function declaration loss
- Demo 13: Game Engine (C++) - String size corruption

---

## Recommendations

1. **COMPLETED: Fix Math.h Intrinsic Handling**: LLVM 22+ attribute stripping implemented in `obfuscator.py`. Demos 7 and 8 now pass.

2. **OPEN: Fix MLIR Function Declaration Loss**: The MLIR string encryption pass needs to preserve external function declarations (strcmp, strlen, etc.). Affects Demo 12.

3. **OPEN: Fix MLIR String Size Calculation**: The MLIR string encryption pass miscalculates array sizes for encrypted strings. Affects Demo 13.

4. **Document C++ Flattening Limitation**: Add user-facing documentation explaining that Control Flow Flattening is automatically disabled for C++ code with exception handling.

---

## Test Environment

- **Browser**: Playwright-controlled Chromium
- **Test Method**: Automated UI interaction
- **Input Mode**: DEMO (pre-loaded example programs)
- **All UI toggles**: Enabled for maximum obfuscation

---

## Re-Test Verification (December 5, 2025)

### Fix Applied: LLVM 22+ Attribute Stripping

A fix was implemented in `obfuscator.py` to strip incompatible LLVM 22+ attributes from IR before passing to `opt`:
- Strips `nocreateundeforpoison` attribute
- Strips `memory(...)` attribute syntax
- Strips `speculatable` attribute
- Strips `convergent` attribute
- Added `-lm` linker flag for math library functions

### Post-Fix Results

| Demo | Pre-Fix Result | Post-Fix Result | Status |
|------|----------------|-----------------|--------|
| 7 | FAIL (LLVM IR error) | PASS (100.0/100) | FIXED |
| 8 | FAIL (timeout/IR error) | PASS (100.0/100) | FIXED |
| 12 | FAIL (LLVM IR error) | FAIL (undefined @strcmp) | NEW BUG |
| 13 | FAIL (clang++ error) | FAIL (type mismatch) | NEW BUG |

### Remaining Issues

**Demo 12**: After LLVM 22+ attribute fix, a new error surfaced - MLIR string encryption pass loses external function declarations.

**Demo 13**: After fixing both `#include <climits>` and LLVM 22+ attributes, a new error surfaced - MLIR string encryption pass corrupts string array sizes.

Both remaining failures are MLIR pass bugs unrelated to the original LLVM 22+ attribute parsing issue.

---

*Report generated by automated Playwright testing on December 5, 2025*
*Fix applied and re-tested on December 5, 2025*
*Success rate improved from 69.2% (9/13) to 84.6% (11/13)*
