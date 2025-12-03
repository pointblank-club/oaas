# OLLVM Crash Analysis for Complex Projects (curl)

## Executive Summary

When obfuscating complex projects like curl with all 4 OLLVM passes (`flattening,substitution,boguscf,split`), the `opt` tool can crash with a segmentation fault during Step 2 of the wrapper pipeline. This document analyzes the root causes and provides solutions.

## Architecture Overview

### Wrapper Pipeline
```
Source.c
  → Step 1: clang -emit-llvm -c → Source.bc (LLVM bitcode)
  → Step 2: opt --load-pass-plugin=... --passes=flattening,substitution,boguscf,split → Source_obf.bc
  → Step 3: clang -c Source_obf.bc → Source.o (object file)
```

### File Locations
- **Wrapper Scripts**: `cmd/llvm-obfuscator/scripts/clang-obfuscate`
- **Plugin Source**: `/Users/akashsingh/Desktop/llvm-project/llvm/lib/Transforms/Obfuscation/`
- **Key Passes**:
  - `Flattening.cpp` - Control flow flattening
  - `BogusControlFlow.cpp` - Bogus control flow insertion
  - `SplitBasicBlocks.cpp` - Basic block splitting
  - `Substitution.cpp` - Instruction substitution

---

## Root Cause Analysis

### 1. Flattening Pass - Most Likely Crash Source

**Location**: `Flattening.cpp:flatten()`

The flattening pass transforms normal control flow into a switch-based dispatcher. Several patterns can cause crashes:

#### Issue A: Complex Switch Statements (Line 102-112)
```cpp
for (BasicBlock &BB : *f) {
  if (SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
    ProcessSwitchInst(SI, DeleteList, nullptr, nullptr);  // Can crash on complex switches
  }
}
```
**Problem**: `ProcessSwitchInst` may fail on switch statements with many cases or complex value ranges.

#### Issue B: Handling Only 1-2 Successors (Lines 260-322)
```cpp
if (i->getTerminator()->getNumSuccessors() == 1) {
  // Handle unconditional jump
}
if (i->getTerminator()->getNumSuccessors() == 2) {
  // Handle conditional jump
}
// No handling for 3+ successors (invoke, indirectbr, etc.)
```
**Problem**: Code doesn't handle `indirectbr` (goto) or exception handling (`invoke`) with 3+ successors.

#### Issue C: InvokeInst Bailout (Line 126-129)
```cpp
if (isa<InvokeInst>(bb->getTerminator())) {
  return false;  // Bails out but doesn't handle gracefully
}
```
**Problem**: Returns false for InvokeInst but other passes may have already modified the function.

### 2. BogusControlFlow Pass - Secondary Crash Source

**Location**: `BogusControlFlow.cpp:addBogusFlow()`

#### Issue A: cryptoutils Not Initialized
```cpp
if((int)llvm::cryptoutils->get_range(100) <= ObfProbRate){
  // Can crash if cryptoutils not properly initialized
}
```

#### Issue B: Clone Operation Failure
```cpp
BasicBlock *alteredBB = createAlteredBasicBlock(originalBB, *var3, &F);
// CloneBasicBlock may fail on complex blocks with unremappable values
```

### 3. Patterns in cf-https-connect.c That Trigger Crashes

Based on curl's `lib/cf-https-connect.c`:

| Pattern | Location | Impact |
|---------|----------|--------|
| State machine switch | `cf_hc_connect()` | Flattening struggles with multi-case switches |
| Function pointers | `cf->cft->query()` | Indirect calls confuse value tracking |
| goto labels | `out:` | `indirectbr` terminators not handled |
| FALLTHROUGH() macro | Multiple | Creates unusual control flow edges |
| Debug macros | `CURL_TRC_CF` | May introduce unexpected basic blocks |

---

## Solutions

### Solution 1: Graceful Fallback in Wrapper Scripts (Recommended)

Modify `clang-obfuscate` to catch `opt` crashes and fall back to compiling without OLLVM passes:

```bash
# Step 2: Apply OLLVM passes with fallback
if ! "$OPT" --load-pass-plugin="$PLUGIN" --passes="$PASSES" "$bc_file" -o "$obf_file" 2>&1; then
    log "WARNING: OLLVM passes failed on $input, compiling without obfuscation"
    # Fallback: compile without OLLVM
    cp "$bc_file" "$obf_file"  # Use original bitcode
fi
```

### Solution 2: Progressive Pass Retry

Try all passes first; if crash, retry with fewer passes:

```bash
try_obfuscation() {
    local passes="$1"
    local bc_file="$2"
    local obf_file="$3"

    if timeout 60 "$OPT" --load-pass-plugin="$PLUGIN" --passes="$passes" "$bc_file" -o "$obf_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Try in order of decreasing passes
for passes in "$PASSES" "substitution,boguscf,split" "substitution,split" "split" ""; do
    if [ -z "$passes" ]; then
        cp "$bc_file" "$obf_file"  # No obfuscation
        break
    fi
    if try_obfuscation "$passes" "$bc_file" "$obf_file"; then
        break
    fi
done
```

### Solution 3: File-Level Exclusion via Environment Variable

Allow specific files to be excluded from OLLVM passes:

```bash
# Environment variable with regex patterns to exclude
OLLVM_EXCLUDE="${OLLVM_EXCLUDE:-}"

should_obfuscate_file() {
    local input="$1"
    if [ -n "$OLLVM_EXCLUDE" ]; then
        if echo "$input" | grep -qE "$OLLVM_EXCLUDE"; then
            return 1  # Exclude this file
        fi
    fi
    return 0  # Obfuscate this file
}

# Usage: OLLVM_EXCLUDE="cf-https-connect|complex_file" make
```

### Solution 4: Per-Pass Error Handling in Plugin

Modify `PluginRegistration.cpp` to catch exceptions:

```cpp
struct FlatteningPassWrapper : public PassInfoMixin<FlatteningPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    try {
      Pass *LegacyPass = createFlattening(true);
      FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
      bool Changed = FP->runOnFunction(F);
      delete LegacyPass;
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    } catch (const std::exception &e) {
      errs() << "WARNING: Flattening failed on " << F.getName() << ": " << e.what() << "\n";
      return PreservedAnalyses::all();  // Skip this function
    } catch (...) {
      errs() << "WARNING: Flattening crashed on " << F.getName() << ", skipping\n";
      return PreservedAnalyses::all();
    }
  }
};
```

### Solution 5: Debug Build for Stack Traces

Build the plugin with debug symbols to get proper crash information:

```bash
# In llvm-project/build:
cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON ..
ninja LLVMObfuscationPlugin
```

Then run opt under gdb/lldb:
```bash
lldb -- /usr/local/llvm-obfuscator/bin/opt \
  --load-pass-plugin=/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so \
  --passes=flattening,substitution,boguscf,split \
  /tmp/problematic.bc -o /tmp/out.bc
```

---

## Recommended Implementation

### Phase 1: Immediate Fix (Wrapper Script)

Update `clang-obfuscate` with graceful fallback - see Solution 1.

### Phase 2: Short-term (Progressive Retry)

Implement Solution 2 to maximize obfuscation coverage while handling crashes.

### Phase 3: Medium-term (Plugin Hardening)

1. Build debug version to identify exact crash location
2. Implement Solution 4 in the plugin
3. Fix identified issues in the pass code

### Phase 4: Long-term (Pass Improvements)

1. Add proper handling for `indirectbr` terminators in Flattening
2. Add support for 3+ successor terminators
3. Improve `ProcessSwitchInst` robustness
4. Add pre-flight checks for complex functions

---

## Testing Plan

### 1. Isolate Problematic Pass
```bash
# Test each pass individually on cf-https-connect.c
for pass in flattening substitution boguscf split; do
    echo "Testing $pass..."
    opt --load-pass-plugin=... --passes=$pass cf-https-connect.bc -o /dev/null
    echo "Exit code: $?"
done
```

### 2. Test Wrapper Fallback
Deploy updated wrapper scripts and verify:
- curl builds successfully
- Files that crash get compiled without OLLVM
- Files that succeed get fully obfuscated

### 3. Binary Verification
After successful build:
```bash
# Verify binary works
./curl --version

# Check obfuscation applied
objdump -d curl | grep "jmp.*\*%" | wc -l  # Indirect jumps count
```

---

## Conclusion

The OLLVM crash on complex projects like curl is primarily caused by the **Flattening pass** when encountering:
1. Complex switch statements
2. Indirect branch terminators (goto)
3. Functions with 3+ successor blocks

The recommended immediate solution is **graceful fallback in wrapper scripts** (Solution 1), which allows the build to complete while maximizing obfuscation coverage. Long-term, the plugin should be hardened with per-function error handling and the pass code should be improved to handle edge cases.

---

**Document Version**: 1.0
**Date**: December 3, 2025
**Status**: Investigation Complete - Implementation Pending
