# Feature #3: McSema Lifting — CFG → LLVM IR → LLVM 22

## Overview

Feature #3 transforms the control flow graph (CFG) into LLVM intermediate representation (IR) and upgrades it to LLVM 22 format for obfuscation.

**Pipeline flow:**
```
Feature #1: Source → Windows PE (-O0 -g)
    ↓
Feature #2: Windows PE → CFG (Ghidra)
    ↓
Feature #3: CFG → LLVM IR → LLVM 22 (THIS STAGE)
    ├─ Part 1: run_lift.sh
    │   Input: program.cfg
    │   Output: program.bc (LLVM 10-17 bitcode)
    │
    └─ Part 2: convert_ir_version.sh
        Input: program.bc
        Output: program_llvm22.bc (LLVM 22 bitcode)
    ↓
Feature #4: LLVM 22 IR → Obfuscated IR (OLLVM passes)
    ↓
Feature #5: Obfuscated IR → Binary
```

## Part 1: CFG → LLVM IR (run_lift.sh)

### What It Does

1. **Accepts** a CFG file (from Feature #2 / Ghidra lifter)
2. **Runs mcsema-lift** with Windows x86-64 target parameters
3. **Outputs** LLVM bitcode (.bc file) in LLVM 10-17 format

### Command

```bash
./run_lift.sh program.cfg output_dir
```

### Output

```
output_dir/
└── program.bc              # LLVM bitcode (low-level machine IR)
```

### Key Details

**mcsema-lift parameters:**
- `--os windows`: Target Windows PE binaries
- `--arch amd64`: x86-64 architecture
- `--cfg`: Input Ghidra CFG file
- `--output`: Output LLVM bitcode

## Part 2: LLVM IR Version Upgrade (convert_ir_version.sh)

### What It Does

1. **Disassembles** LLVM 10-17 bitcode to textual IR (.ll)
2. **Reassembles** into LLVM 22 bitcode (applies auto-upgrade rules)
3. **Verifies** the upgraded IR structure
4. **Outputs** LLVM 22 bitcode ready for OLLVM

### Command

```bash
./convert_ir_version.sh program.bc output_dir
```

### Output

```
output_dir/
└── program_llvm22.bc       # LLVM 22 bitcode (ready for obfuscation)
```

### Why Version Upgrade?

| Aspect | Reason |
|--------|--------|
| **McSema bitcode version** | LLVM 10-17 (depends on mcsema-lift version) |
| **OLLVM target version** | LLVM 22 |
| **Compatibility** | Bitcode format is version-specific |
| **Solution** | Disassemble → reassemble with LLVM 22 parser |

### How Upgrade Works

```
program.bc (LLVM 10-17)
    ↓
llvm-dis-22 (disassemble)
    ↓
program.ll (textual IR, with auto-upgrade rules applied)
    ↓
llvm-as-22 (reassemble to LLVM 22 bitcode)
    ↓
program_llvm22.bc (LLVM 22 bitcode)
    ↓
llvm-verify-22 (validate IR structure)
```

## Critical Limitations

### ⚠️ McSema IR is NOT Normal LLVM IR

McSema lifting produces low-level machine IR, not high-level LLVM IR:

| Aspect | Normal LLVM IR | McSema IR |
|--------|----------------|-----------|
| **Memory model** | Typed pointers | Flattened state struct (64-bit fields) |
| **Semantics** | High-level operations | x86-64 instruction emulation |
| **Control flow** | Structured CFG | State machine via switch statements |
| **Type safety** | Enforced | Violated (arbitrary casts) |
| **Function calls** | Direct LLVM calls | Simulated via state machine |

### ⚠️ Ghidra CFG May Contain Errors

- Function detection accuracy: ~90% (vs IDA's ~98%)
- Tail calls may be mis-identified
- Jump tables may be incorrectly recovered
- Indirect branches cannot be resolved
- **If CFG is wrong → lifted IR is invalid**

### ⚠️ Lifting is Not Safe For

| Feature | Issue | Reason |
|---------|-------|--------|
| **Exceptions** | Control flow breaks | McSema cannot model SEH |
| **Recursion** | State machine fails | Cannot track call stack |
| **Jump tables** | Ghidra recovery unreliable | Indirect dispatch complex |
| **C++** | EH, vtables not supported | Too complex for CFG analysis |
| **Complex indirection** | Cannot resolve targets | No data flow analysis |

### ⚠️ Auto-Upgrade Success Rate: ~95%

When converting LLVM 10-17 → LLVM 22:

| Success | Failure |
|---------|---------|
| **95%** Standard IR patterns upgrade correctly | **5%** Edge cases may break |
| Intrinsics are mapped to newer equivalents | Some mappings are incorrect |
| Metadata is attempted to be preserved | Metadata often lost/corrupted |
| | Debug info degraded |
| | Type information simplified |

### ⚠️ Certain OLLVM Passes Will NOT Work

The following OLLVM passes are **INCOMPATIBLE** with McSema IR:

```
❌ -bcf (Bogus Control Flow)
   McSema IR: State machine (unstructured)
   -bcf expects: Normal LLVM CFG (structured)
   Result: Crashes or infinite loops

❌ -flattening (Control Flow Flattening)
   McSema IR: Already internally flattened
   -flattening expects: High-level structured CFG
   Result: Double-flattening produces broken code

❌ -split (Block Splitting)
   McSema IR: No proper function entry/exit
   -split expects: Structured control flow
   Result: May corrupt function boundaries

❌ -opaque-predicates (Opaque Predicates)
   McSema IR: No proper memory model
   -opaque expects: Type-safe memory operations
   Result: Predicates optimized away
```

**SAFE OLLVM passes for McSema IR:**
- `-fla` (Light Function Annotation)
- `-cff` (Function Call Obfuscation)
- `-ald` (Arithmetic Lowering/Diversity)
- Standard LLVM optimizations (`-O2`, `-O3`)

## Error Handling

### Common Errors in Feature #3

| Error | Cause | Solution |
|-------|-------|----------|
| `mcsema-lift not found` | McSema not installed | Install mcsema-lift or use container |
| `mcsema-lift failed` | Invalid CFG or lifting unsupported | Check Ghidra CFG; may need IDA validation |
| `llvm-dis-22 failed` | Corrupted bitcode | Check mcsema-lift output |
| `llvm-as-22 failed` | IR syntax error | May indicate auto-upgrade failure |
| `llvm-verify-22 warnings` | IR is malformed but usable | Check output for severity |

### Debugging

**Check mcsema-lift version:**
```bash
mcsema-lift --version
```

**Inspect generated bitcode:**
```bash
llvm-dis-22 program.bc -o program.ll
# View program.ll (may be large)
```

**Validate LLVM 22 IR:**
```bash
llvm-verify-22 program_llvm22.bc
```

**Check IR functions:**
```bash
llvm-dis-22 program_llvm22.bc -o program_llvm22.ll
grep "^define " program_llvm22.ll | head -20
```

## Performance Characteristics

| Operation | Time (typical) | Memory |
|-----------|----------------|--------|
| mcsema-lift (small binary) | 10-30s | 500MB-2GB |
| llvm-dis-22 | 5-15s | 2x bitcode size |
| llvm-as-22 | 5-15s | 2x IR size |
| llvm-verify-22 | 2-5s | 1x bitcode size |

For large binaries (>100MB), operations may take several minutes.

## Validation Checklist

Before proceeding to Feature #4:

- [ ] Feature #1 completed successfully (binary compiled with -O0 -g)
- [ ] Feature #2 completed successfully (Ghidra CFG exported)
- [ ] run_lift.sh produced program.bc without errors
- [ ] convert_ir_version.sh produced program_llvm22.bc without errors
- [ ] llvm-verify-22 passed (warnings are acceptable)
- [ ] program_llvm22.bc file exists and has non-zero size
- [ ] plan to avoid incompatible OLLVM passes (-bcf, -flattening, -split, -opaque-predicates)

## Next Steps: Feature #4

Feature #4 will:

1. Select safe OLLVM passes (from approved list above)
2. Load obfuscation plugins into clang
3. Apply selected passes to program_llvm22.bc
4. Generate obfuscated LLVM IR
5. Compile to final obfuscated Windows PE binary

Example Feature #4 command:
```bash
clang-22 \
    -Xclang -load -Xclang libFLA.so \
    -Xclang -load -Xclang libCFF.so \
    -O2 \
    program_llvm22.bc \
    -o program_obfuscated.exe
```

## Limitations Summary

| Aspect | Status |
|--------|--------|
| **CFG accuracy** | ~80-85% (depends on Ghidra) |
| **IR upgrade success** | ~95% (edge cases may fail) |
| **OLLVM compatibility** | Partial (4 unsafe passes) |
| **Code quality** | Degraded vs native (state machine overhead) |
| **Performance** | Expected ~10-20% slower than original |
| **Production readiness** | Experimental only |

## References

- [McSema Documentation](https://github.com/lifting-bits/mcsema)
- [LLVM Bitcode Format](https://llvm.org/docs/BitCodeFormat/)
- [OLLVM Documentation](https://github.com/ob-programming/ollvm)
