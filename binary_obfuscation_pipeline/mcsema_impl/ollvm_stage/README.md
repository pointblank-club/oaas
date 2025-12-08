# Feature #4: OLLVM Pass Application — Obfuscate LLVM 22 IR

## Overview

Feature #4 applies OLLVM obfuscation passes to the LLVM 22 intermediate representation (IR), producing obfuscated bitcode ready for compilation to Windows PE.

**Pipeline flow:**
```
Feature #1: Source → Windows PE (-O0 -g)
    ↓
Feature #2: Windows PE → CFG (Ghidra)
    ↓
Feature #3: CFG → LLVM IR → LLVM 22
    ↓
Feature #4: LLVM 22 → Obfuscated LLVM 22 (THIS STAGE)
    ├─ OLLVM passes: flattening, substitution, boguscf, split, linear-mba
    ├─ MLIR passes: string-encrypt, symbol-obfuscate, constant-obfuscate, crypto-hash
    └─ Standard LLVM opts: -O1
    ↓
Feature #5: Obfuscated LLVM 22 → Windows PE binary
```

## What It Does

1. **Reads** a passes configuration file (JSON)
2. **Applies selected OLLVM passes** using opt binary with OLLVM plugins
3. **Applies standard LLVM optimizations** (optional)
4. **Outputs** obfuscated LLVM 22 bitcode (.bc file)

## Configuration

### passes_config.json

This JSON file controls which obfuscation passes are applied:

```json
{
  "flattening": false,
  "bogus_control_flow": false,
  "substitution": true,
  "split": false,
  "linear_mba": false,
  "string_encrypt": false,
  "symbol_obfuscate": false,
  "constant_obfuscate": false,
  "crypto_hash": false,
  "standard_llvm_opts": false
}
```

All values must be `true` or `false`.

### Available Passes

#### OLLVM Passes (applied by opt with custom plugin)

| Pass | Name | Risk Level | Effect |
|------|------|-----------|--------|
| **substitution** | Instruction Substitution | ✅ Low | Replace arithmetic ops with equivalent sequences |
| **linear_mba** | Linear Multi-Byte Arithmetic | ✅ Low | Obfuscate arithmetic using linear expressions |
| **flattening** | Control Flow Flattening | ⚠️ Medium | Flatten IF/ELSE structures to state machine |
| **split** | Block Splitting | ⚠️ Medium | Split basic blocks into smaller chunks |
| **bogus_control_flow** (boguscf) | Bogus Control Flow | 🔴 High | Insert fake control flow paths |

#### MLIR Passes (applied during Feature #5 compilation)

These run via `-mllvm` flags during final compilation:

| Pass | Name | Effect |
|------|------|--------|
| **string_encrypt** | String Encryption | Encrypt string literals |
| **symbol_obfuscate** | Symbol Obfuscation | Rename exported symbols |
| **constant_obfuscate** | Constant Obfuscation | Obfuscate numeric constants |
| **crypto_hash** | Crypto Hash | Add cryptographic hashing |

## Usage

```bash
./run_ollvm.sh <input_llvm22.bc> <output_dir> <passes_config.json>
```

### Example

```bash
./run_ollvm.sh ./program_llvm22.bc ./obfuscated_ir/ ./passes_config.json
```

### Output

```
obfuscated_ir/
└── program_obf.bc              # Obfuscated LLVM 22 bitcode
```

## Pass Selection Guide

### Recommended Safe Combinations

**Minimal obfuscation (fastest):**
```json
{
  "substitution": true,
  "linear_mba": false,
  "flattening": false,
  "bogus_control_flow": false,
  "split": false
}
```

**Moderate obfuscation:**
```json
{
  "substitution": true,
  "linear_mba": true,
  "flattening": false,
  "bogus_control_flow": false,
  "split": false
}
```

**Aggressive obfuscation (test first!):**
```json
{
  "substitution": true,
  "linear_mba": true,
  "flattening": true,
  "bogus_control_flow": false,
  "split": false
}
```

⚠️ **NEVER enable bogus_control_flow for McSema IR without extensive testing.**

## Critical Warnings

### ⚠️ McSema IR is NOT Normal Compiled Code

The IR you're obfuscating comes from lifting a Windows PE binary through Ghidra + McSema:

| Aspect | McSema IR | Normal Compiled IR |
|--------|-----------|-------------------|
| **Semantics** | x86-64 machine simulation | High-level language |
| **Memory model** | Flattened state struct | Typed LLVM pointers |
| **Control flow** | State machine dispatch | Structured basic blocks |
| **Performance** | ~10-20% slower | Native speed |

**Result**: OLLVM obfuscations may break semantics or produce non-executable code.

### Unsafe Passes for McSema IR

#### 🔴 Bogus Control Flow (boguscf)

**Status**: HIGHLY DANGEROUS

**Why it breaks:**
- McSema IR uses explicit state machine dispatch
- Bogus CFG injection adds unreachable code paths
- State machine may not transition correctly through fake blocks
- Binary often crashes or hangs

**Risk**: Non-functional binary

**Recommendation**: AVOID entirely

#### ⚠️ Flattening (flattening)

**Status**: Dangerous on untested binaries

**Why it may break:**
- State machine PC updates become harder to track
- Control flow merge points may be corrupted
- Performance degradation can be 10x slower

**Risk**: Possible crashes or unexpected behavior

**Recommendation**: Test on small programs only

#### ⚠️ Split Basic Blocks (split)

**Status**: Dangerous

**Why it may break:**
- McSema IR's BB boundaries are semantically significant
- Splitting mid-block can corrupt instruction dependencies
- Register live-range analysis may fail

**Risk**: Undefined behavior

**Recommendation**: Avoid unless tested thoroughly

### Before Using Obfuscated Code

1. **Test on simple programs first**
   - Add, multiply, fibonacci, factorial
   - Verify output matches original

2. **Run on target Windows system**
   - Execute and verify behavior
   - Check for crashes or hangs
   - Monitor performance

3. **Use debugger to trace execution**
   - Verify control flow makes sense
   - Check register values at key points
   - Validate memory access patterns

4. **Compare against original**
   - Functional equivalence test
   - Performance comparison (expect 10-20% slower)
   - Memory usage comparison

5. **Validate on production code ONLY after testing**

## Execution Flow

### Step 1: Apply OLLVM Passes (if enabled)

```bash
opt -passes="substitution,linear-mba,flattening" program_llvm22.bc -o intermediate.bc
```

**Output**: Intermediate OLLVM-obfuscated bitcode

### Step 2: Apply Standard Optimizations (if enabled)

```bash
opt -O1 intermediate.bc -o program_obf.bc
```

**Output**: Final OLLVM + optimized bitcode

### Step 3: MLIR Passes (in Feature #5)

MLIR passes run during compilation with `-mllvm` flags:

```bash
clang-22 -mllvm -string-encrypt -mllvm -symbol-obfuscate program_obf.bc -o program.exe
```

## Debugging

### Check which passes were applied

The script logs all enabled passes:

```
[INFO] OLLVM passes to apply: substitution,linear-mba
[INFO] MLIR passes to apply: -mllvm -string-encrypt
```

### Verify obfuscated bitcode

```bash
# Disassemble to inspect
llvm-dis-22 program_obf.bc -o program_obf.ll

# Check IR structure
grep "^define " program_obf.ll | wc -l
```

### Test on simple program

1. Compile simple C program:
   ```bash
   ./compile_windows_binary.py simple.c ./output/
   ```

2. Lift with Ghidra:
   ```bash
   ./run_ghidra_lifter.sh ./output/program.exe ./cfg_output/
   ```

3. Convert to LLVM IR:
   ```bash
   ./run_lift.sh ./cfg_output/program.cfg ./ir_output/
   ./convert_ir_version.sh ./ir_output/program.bc ./ir_output/
   ```

4. Apply minimal obfuscation:
   ```bash
   echo '{"substitution": true, "linear_mba": false, "flattening": false}' > minimal_config.json
   ./run_ollvm.sh ./ir_output/program_llvm22.bc ./obf_output/ ./minimal_config.json
   ```

5. Compile and test (Feature #5):
   ```bash
   # Next step will compile obf_output/program_obf.bc to Windows PE
   ```

## Performance Impact

Obfuscation has measurable performance overhead:

| Configuration | Time Overhead | Size Overhead |
|---------------|---------------|---------------|
| No obfuscation | 0% | 0% |
| substitution only | +5-10% | +10-15% |
| +linear_mba | +10-15% | +20-30% |
| +flattening | +50-100% | +30-50% |
| +split | +20-30% | +15-25% |

**Expected**: Obfuscated binaries are slower but more resistant to analysis.

## Limitations

| Aspect | Status |
|--------|--------|
| **OLLVM compatibility** | Partial (some passes unsafe) |
| **MLIR pass integration** | Via -mllvm flags in Feature #5 |
| **McSema IR safety** | Experimental |
| **Production readiness** | Research stage only |

## Next Steps: Feature #5

Feature #5 will:

1. Compile obfuscated LLVM 22 IR to Windows PE
2. Apply MLIR passes via -mllvm flags
3. Perform final optimizations
4. Output final Windows executable

```bash
clang-22 \
    -Xclang -load -Xclang libFLA.so \
    -mllvm -string-encrypt \
    -mllvm -symbol-obfuscate \
    -O2 \
    program_obf.bc \
    -target x86_64-w64-windows-gnu \
    -o program_obfuscated.exe
```

## Summary

Feature #4 applies OLLVM transformations to the lifted IR. The output is obfuscated bitcode suitable for compilation to a Windows PE binary. Be cautious with control-flow-heavy passes, test thoroughly on small programs, and validate behavior before production use.
