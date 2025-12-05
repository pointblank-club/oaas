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
| Successful | 9 (69.2%) |
| Failed | 4 (30.8%) |
| Average Score (Successful) | 99.8/100 |

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
| 7 | Matrix Operations | C | FAIL | - | - | LLVM IR parsing error (math.h) |
| 8 | Signal Processing DSP | C | FAIL | - | - | 502 timeout (math.h functions) |
| 9 | Exception Handler | C++ | PASS | 99.7/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 10 | License Validator | C++ | PASS | 100.0/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 11 | Configuration Manager | C++ | PASS | 100.0/100 | PARTIAL | Flattening disabled (C++ exceptions) |
| 12 | SQL Database Engine | C | FAIL | - | - | LLVM IR parsing error (math.h) |
| 13 | Game Engine | C++ | FAIL | - | - | clang++ compilation error |

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
**Status**: FAIL
**Score**: N/A
**Layers Applied**: N/A

**Error**:
```
LLVM IR parsing error: unterminated attribute group
Error at line with math intrinsics (cos, sin functions)
```

**Root Cause**: The program uses `<math.h>` functions (`cos`, `sin`) which generate LLVM intrinsics that cause parsing errors in the obfuscation pipeline.

---

### Demo 8: Signal Processing DSP (C)
**Status**: FAIL
**Score**: N/A
**Layers Applied**: N/A

**Error**:
```
502 Bad Gateway (timeout)
```

**Root Cause**: The program uses complex `<math.h>` functions which cause the obfuscation process to timeout. Same underlying issue as Demo 7.

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
/app/reports/.../pasted_source_from_mlir.ll:5954:31: error: unterminated attribute group
attributes #10 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
```

**Root Cause**: The program uses `<math.h>` functions (`sqrt`, `fmod`, `pow`, `log2`) which generate LLVM intrinsics with attributes that cause parsing errors in the obfuscation pipeline.

---

### Demo 13: Game Engine (C++)
**Status**: FAIL
**Score**: N/A
**Layers Applied**: N/A

**Error**:
```
Command failed with exit code 1: clang++ /app/reports/.../pasted_source.cpp -S -emit-llvm -o ...
```

**Root Cause**: Compilation stage failure before obfuscation could begin. Possible C++17 feature compatibility issue or complex template instantiation error.

---

## Error Pattern Analysis

### Pattern 1: Math.h LLVM IR Parsing Errors
**Affected Demos**: 7, 8, 12
**Root Cause**: Programs using `<math.h>` functions generate LLVM intrinsics with special attributes that the obfuscation pipeline cannot parse correctly.

**Functions Known to Cause Issues**:
- `cos`, `sin` (Demo 7)
- Complex DSP math functions (Demo 8)
- `sqrt`, `fmod`, `pow`, `log2` (Demo 12)

**Recommendation**: Backend fix needed to handle math intrinsic attributes properly, or pre-process LLVM IR to normalize attribute groups.

### Pattern 2: C++ Exception Handling
**Affected Demos**: 2, 6, 9, 10, 11
**Behavior**: Control Flow Flattening pass is automatically disabled when C++ exception handling is detected.

**Reason**: Control Flow Flattening can corrupt exception handling tables, causing undefined behavior at runtime.

**Status**: This is expected behavior, not a bug. The system correctly detects and adapts.

### Pattern 3: C++ Compilation Errors
**Affected Demos**: 13
**Root Cause**: Complex C++ code with advanced features may fail during initial compilation stage.

**Recommendation**: Review the Game Engine demo for C++ standard compatibility and simplify if needed.

---

## Layer Application Summary

### Fully Applied (All Layers)
- Demo 1: Hello World (C)
- Demo 3: Authentication System (C)
- Demo 4: Password Strength Checker (C)
- Demo 5: Fibonacci Calculator (C)

### Partially Applied (Flattening Disabled)
- Demo 2: Hello World (C++)
- Demo 6: QuickSort Algorithm (C++)
- Demo 9: Exception Handler (C++)
- Demo 10: License Validator (C++)
- Demo 11: Configuration Manager (C++)

### Not Applied (Failed)
- Demo 7: Matrix Operations (C)
- Demo 8: Signal Processing DSP (C)
- Demo 12: SQL Database Engine (C)
- Demo 13: Game Engine (C++)

---

## Recommendations

1. **Fix Math.h Intrinsic Handling**: Investigate the LLVM IR parsing issue with math intrinsics to enable obfuscation of programs using math functions.

2. **Review Demo 13 Compatibility**: Check Game Engine C++ demo for C++ standard compatibility issues with clang++.

3. **Document C++ Flattening Limitation**: Add user-facing documentation explaining that Control Flow Flattening is automatically disabled for C++ code with exception handling.

4. **Consider Alternative Math Handling**: Explore options like:
   - Pre-processing LLVM IR to normalize attribute groups
   - Using libm implementations instead of intrinsics
   - Selective pass application for math-heavy code

---

## Test Environment

- **Browser**: Playwright-controlled Chromium
- **Test Method**: Automated UI interaction
- **Input Mode**: DEMO (pre-loaded example programs)
- **All UI toggles**: Enabled for maximum obfuscation

---

## Re-Test Verification (December 5, 2025)

All 4 failed demos were re-tested to confirm results. **All failures confirmed consistent.**

### Demo 7: Matrix Operations (C) - CONFIRMED FAIL
**Error**: LLVM IR parsing error
```
/app/reports/.../pasted_source_from_mlir.ll: error: unterminated attribute group
attributes #N = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
```
**Root Cause**: math.h functions (cos, sin) generate LLVM intrinsics with incompatible attributes.

### Demo 8: Signal Processing DSP (C) - CONFIRMED FAIL
**Error**: LLVM IR parsing error (same pattern as Demo 7)
```
unterminated attribute group
```
**Root Cause**: Complex math.h functions cause the same LLVM IR parsing failure.

### Demo 12: SQL Database Engine (C) - CONFIRMED FAIL
**Error**: LLVM IR parsing error
```
/app/reports/.../pasted_source_from_mlir.ll:5954:31: error: unterminated attribute group
attributes #10 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
```
**Root Cause**: math.h functions (sqrt, fmod, pow, log2) generate incompatible LLVM intrinsics.

### Demo 13: Game Engine (C++) - FIXED & RE-TESTED
**Original Error**: Compilation failure (missing `#include <climits>`)
**Fix Applied**: Added `#include <climits>` to demo code in `largeDemos.ts`
**Fix Result**: Compilation now passes

**New Error After Fix**: LLVM IR parsing error (same as Demos 7, 8, 12)
```
/app/reports/.../pasted_source_from_mlir.ll:50886:31: error: unterminated attribute group
attributes #15 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
```
**Root Cause**: Demo uses `<cmath>` functions (std::sqrt, std::cos, std::sin, std::pow, std::floor, std::abs) which generate LLVM intrinsics with incompatible attributes.

**Conclusion**: Demo 13 has TWO issues:
1. Missing `#include <climits>` - **FIXED**
2. Uses cmath functions causing LLVM IR parsing error - **Same backend issue as Demos 7, 8, 12**

### Re-Test Summary
| Demo | Original Result | Re-Test Result | Consistent |
|------|-----------------|----------------|------------|
| 7 | FAIL (LLVM IR error) | FAIL (LLVM IR error) | YES |
| 8 | FAIL (timeout/IR error) | FAIL (LLVM IR error) | YES |
| 12 | FAIL (LLVM IR error) | FAIL (LLVM IR error) | YES |
| 13 | FAIL (clang++ error) | FAIL (LLVM IR error after fix) | PARTIALLY - different error after code fix |

**Note on Demo 13**: After fixing the missing `#include <climits>`, Demo 13 now fails with the same LLVM IR parsing error as Demos 7, 8, and 12. All 4 failing demos share the same root cause: math functions generating incompatible LLVM intrinsics.

---

*Report generated by automated Playwright testing on December 5, 2025*
*Re-test verification completed on December 5, 2025*
