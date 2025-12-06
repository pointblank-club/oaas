# OAAS Obfuscator Bug Report

**Date**: 2025-12-06
**Tested URL**: https://oaas.pointblank.club
**Testing Method**: Playwright E2E tests

---

## Executive Summary

Two distinct bugs were identified affecting obfuscation layer combinations:

| Bug | Affected Layers | Severity | Status |
|-----|----------------|----------|--------|
| #1: OLLVM Flag Parsing | Layer 3 (any combination) | **CRITICAL** | All Layer 3 combinations fail |
| #2: MLIR String Size Mismatch | Layer 2 (certain demos) | **HIGH** | 4 of 13 demos fail |

---

## Complete Test Results

### Layer 1+2 Test Matrix (All 13 Demos)

| Demo | Language | Layer 1+2 | Notes |
|------|----------|-----------|-------|
| Hello World | C | **PASS** | Simple strings |
| Hello World | C++ | **PASS** | Simple strings |
| Fibonacci Calculator | C | **PASS** | Simple strings |
| Password Strength Checker | C | **PASS** | Simple strings |
| QuickSort Algorithm | C++ | **PASS** | Simple strings |
| Matrix Operations | C | **PASS** | Simple strings |
| Signal Processing DSP | C | **PASS** | Simple strings |
| Exception Handler | C++ | **PASS** | Simple strings |
| SQL Database Engine | C | **PASS** | Simple strings |
| Authentication System | C | **FAIL** | Bug #2: `[22 x i8]` vs `[23 x i8]` |
| License Validator | C++ | **FAIL** | Bug #2: `[24 x i8]` vs `[25 x i8]` |
| Configuration Manager | C++ | **FAIL** | Bug #2: `[11 x i8]` vs `[12 x i8]` |
| Game Engine | C++ | **FAIL** | Bug #2: `[22 x i8]` vs `[24 x i8]` |

**Summary**: 9/13 demos PASS, 4/13 demos FAIL with Layer 1+2

---

## Bug #1: OLLVM `-bcf_loop` Flag Parsing Error

### Description
When Layer 3 (OLLVM Passes) is enabled, the obfuscation fails due to incorrect parsing of the `-bcf_loop=1` compiler flag. Clang interprets `-b` as a separate unsupported option.

### Error Message
```
Command failed with exit code 1: /usr/local/llvm-obfuscator/bin/clang ... -mllvm -split_num=3 -bcf_loop=1 --target=x86_64-unknown-linux-gnu
clang: error: unsupported option '-b' for target 'x86_64-unknown-linux-gnu'
```

### Affected Combinations
| Layer Combination | Result |
|-------------------|--------|
| Layer 3 only | FAIL |
| Layer 1 + 3 | FAIL |
| Layer 2 + 3 | FAIL |
| Layer 1 + 2 + 3 | FAIL |
| Layer 3 + 4 + 5 | FAIL |
| Layer 1 + 3 + 4 + 5 | FAIL |
| Layer 2 + 3 + 4 + 5 | FAIL |
| Layer 1 + 2 + 3 + 4 + 5 | FAIL |

### Root Cause
The `-bcf_loop=1` flag is being passed directly to clang instead of through `-mllvm`. The command shows:
```
-mllvm -split_num=3 -bcf_loop=1
```
Should be:
```
-mllvm -split_num=3 -mllvm -bcf_loop=1
```

### Reproduction Steps
1. Navigate to https://oaas.pointblank.club
2. Select DEMO -> "Hello World (C)"
3. Enable Layer 3 (OLLVM Passes)
4. Enable any OLLVM sub-passes (e.g., "Bogus Control Flow")
5. Click OBFUSCATE

### Suggested Fix
In the backend, ensure all OLLVM flags are prefixed with `-mllvm`. Check the command construction in `obfuscator.py` around lines where `-bcf_loop` and `-split_num` are added.

---

## Bug #2: MLIR String Encryption Size Mismatch

### Description
When Layer 2 (String Encryption) is enabled, certain demos with complex string patterns fail due to a mismatch between declared array size and actual content in the generated LLVM IR.

### Error Messages

**Authentication System (C)**:
```
@.str.16 = private unnamed_addr constant [23 x i8] c"n> < )\7F*\06\1A\01\16\15\41\11\09\1A\36\0E\01sd", align 1
error: constant expression type mismatch: got type '[22 x i8]' but expected '[23 x i8]'
```

**License Validator (C++)**:
```
error: constant expression type mismatch: got type '[24 x i8]' but expected '[25 x i8]'
```

**Configuration Manager (C++)**:
```
@.str.26 = private unnamed_addr constant [12 x i8] c" 0$9#$\12.+-d", align 1
error: constant expression type mismatch: got type '[11 x i8]' but expected '[12 x i8]'
```

**Game Engine (C++)**:
```
@.str.79 = private unnamed_addr constant [24 x i8] c"DEKA0\22\33\16% &(,%$;?1\00 ne", align 1
error: constant expression type mismatch: got type '[22 x i8]' but expected '[24 x i8]'
```

### Pattern Analysis

**Failing demos contain:**
- URL patterns: `postgresql://admin:dbpass123@db.internal:5432/prod`
- API key patterns: `sk_live_oaas_demo_key_12345`, `api_sk_live_oaas_game_engine_x1y2z3`
- RSA key headers: `-----BEGIN RSA PRIVATE KEY-----\nMIIE...`
- License key patterns: `OAAS_ENGINE_LIC_2024_PREMIUM_UNLIMITED`
- Special characters and escape sequences

**Passing demos contain:**
- Simple format strings: `"Hello, World!\n"`, `"%d"`, `"%s"`
- Basic text messages
- Simple variable names

### Root Cause Analysis
The MLIR string encryption pass (`string-encrypt`) appears to:
1. Calculate encrypted string length incorrectly for certain character patterns
2. Specifically fails when strings contain:
   - Embedded null bytes after encryption
   - Multi-byte escape sequences (`\22`, `\33`, etc.)
   - URL special characters (`:`, `/`, `@`)
3. The off-by-one or off-by-two errors suggest incorrect handling of:
   - Null terminator counting
   - Escape sequence byte counting during XOR encryption

### Reproduction Steps
1. Navigate to https://oaas.pointblank.club
2. Select DEMO -> "Authentication System (C)" (or License Validator, Configuration Manager, Game Engine)
3. Enable Layer 1 (Symbol Obfuscation)
4. Enable Layer 2 (String Encryption)
5. Click OBFUSCATE
6. Observe "constant expression type mismatch" error

### Suggested Fix
Review the MLIR string encryption pass in `mlir-obs/` to ensure:
1. Correct byte counting for encrypted strings (accounting for escape sequences)
2. Proper null terminator handling
3. Consistent array size declarations in generated LLVM IR
4. Special handling for strings containing URL-like patterns or RSA headers

---

## Working Combinations

| Layer Combination | Simple Demos | Complex Demos |
|-------------------|--------------|---------------|
| Layer 1 only | PASS | PASS |
| Layer 2 only | PASS | FAIL |
| Layer 1 + 2 | PASS | FAIL |
| Layer 4 + 5 | PASS | PASS |
| Layer 1 + 4 + 5 | PASS | PASS |
| Layer 2 + 4 + 5 | PASS | FAIL |
| Layer 1 + 2 + 4 + 5 | PASS | FAIL |

**Note**: Layer 3 combinations fail for ALL demos due to Bug #1.

---

## Layer Reference

| Layer | Name | Type | Description |
|-------|------|------|-------------|
| 1 | Symbol Obfuscation | MLIR (PRE-COMPILE) | Cryptographic hash renaming |
| 2 | String Encryption | MLIR (PRE-COMPILE) | XOR encryption of string literals |
| 2.5 | Indirect Calls | PRE-COMPILE | Function pointer indirection |
| 3 | OLLVM Passes | COMPILE | Control flow obfuscation |
| 4 | Compiler Flags | COMPILE | Hardening flags |
| 5 | UPX Packing | POST-COMPILE | Binary compression |

---

## Backend Logs Command

To retrieve backend logs for debugging:
```bash
ssh root@69.62.77.147 "docker logs llvm-obfuscator-backend --tail 100"
```

---

## Priority Recommendations

1. **CRITICAL**: Fix Bug #1 (OLLVM flag parsing) - All Layer 3 functionality is broken
2. **HIGH**: Fix Bug #2 (MLIR string size) - 31% of demos fail (4/13)
3. **MEDIUM**: Add comprehensive testing for all demo + layer combinations

---

## Test Environment

- **Platform**: Linux
- **Browser**: Playwright (Chromium)
- **Test Date**: 2025-12-06
- **Total Demos**: 13
- **Total Tests Run**: 13 (Layer 1+2 on all demos)
