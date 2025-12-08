# OLLVM Passes for Binary Obfuscation Mode - Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Safe Passes (Low Risk)](#safe-passes-low-risk)
4. [Medium-Risk Passes](#medium-risk-passes)
5. [Building the OLLVM Plugin Binary](#building-the-ollvm-plugin-binary)
6. [Integration with Current Setup](#integration-with-current-setup)
7. [Testing & Validation](#testing--validation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Binary Obfuscation Mode in OAAS performs a 6-step pipeline:

```
Windows PE Binary → Ghidra CFG Export → McSema LLVM Lifting →
LLVM 22 IR Upgrade → OLLVM Pass Application → Final Binary Compilation
```

This guide details which OLLVM passes are safe to add to the pipeline and how to build/deploy them.

### Key Constraint: McSema IR Compatibility

McSema produces **lifted IR** - not normal compiled code. It uses a **state machine** architecture for control flow instead of traditional CFGs. This makes certain OLLVM passes dangerous because they can:

- Corrupt the PC (program counter) update logic
- Break function returns and state machine dispatch
- Invalidate lifted CFG edges
- Create infinite loops or unreachable code

**Result**: Only CFG-neutral passes work reliably with McSema IR.

---

## Architecture

### Current Pipeline Flow

```
binary_obfuscation_pipeline/
├── mcsema_impl/
│   ├── ghidra_lifter/        # Step 1: Ghidra CFG export
│   ├── lifter/               # Step 2-3: McSema IR lifting + LLVM 22 upgrade
│   └── ollvm_stage/
│       ├── run_ollvm.sh      # Step 4: OLLVM passes orchestration
│       ├── passes_config.json # Pass configuration (from UI)
│       └── README.md
├── windows_build/            # Step 5: Final compilation
└── ...
```

### Current Pass Configuration (Frontend)

File: `cmd/llvm-obfuscator/frontend/src/components/BinaryObfuscationMode.tsx:15-25`

```typescript
const [passes, setPasses] = useState({
  substitution: true,          // ✅ SAFE
  flattening: false,           // ⚠️ DANGEROUS
  bogus_control_flow: false,   // 🔴 VERY DANGEROUS
  split: false,                // 🔴 VERY DANGEROUS
  linear_mba: false,           // ⚠️ MEDIUM RISK (not yet verified)
  string_encrypt: false,       // 🟡 MLIR (different system)
  symbol_obfuscate: false,     // 🟡 MLIR (different system)
  constant_obfuscate: false,   // 🟡 MLIR (different system)
  crypto_hash: false,          // 🟡 MLIR (different system)
});
```

### Backend Pass Handling

File: `binary_obfuscation_pipeline/mcsema_impl/ollvm_stage/run_ollvm.sh:254-283`

Passes are applied via a custom `opt` binary with OLLVM integrated:

```bash
$CUSTOM_OPT -passes="substitution,linear-mba,flattening,boguscf,split" \
            input.bc -o output.bc
```

The custom `opt` is located at:
- **Production**: `/usr/local/llvm-obfuscator/bin/opt` (Docker container)
- **Local development**: `plugins/linux-x86_64/opt` (GCP binary store)

---

## Safe Passes (Low Risk)

These passes **do NOT** modify control flow and work reliably with McSema IR.

### 1️⃣ Instruction Substitution (`-substitution`)

**Status**: ✅ SAFE - Already enabled by default

#### What it does:

Replaces arithmetic and logical operations with semantically equivalent but more complex sequences:

```llvm
; Before
%0 = add i32 %a, %b

; After (one possible substitution)
%0 = xor i32 %a, %b
%1 = and i32 %a, %b
%2 = shl i32 %1, 1
%3 = add i32 %0, %2
```

Other substitutions:
- `a + b` → `(a ^ (~b)) + 1`
- `a - b` → `a + (~b + 1)`
- `a ^ b` → `(a | b) & (~a | ~b)`
- `a * b` → `(a << k) + (a * (b - 2^k))` for some k

#### Why safe:

✅ Does NOT change basic blocks
✅ Does NOT change branches
✅ Does NOT affect memory offsets
✅ Works purely on SSA values
✅ McSema IR doesn't care about macro patterns
✅ State machine dispatch completely unaffected

#### Implementation:

Already in OLLVM core. No changes needed.

**Obfuscation strength**: ⭐⭐ (moderate - 2x-3x instruction expansion)

---

### 2️⃣ Constant Obfuscation (`-constant-obfuscate`)

**Status**: 🟡 SAFE but requires custom implementation

#### What it does:

Encodes literal constants into computed forms:

```llvm
; Before
mov eax, 0x1337

; After
mov eax, 0x0
mov ebx, 55
imul eax, ebx, 42
sub eax, 83
; Result in eax: (55 * 42) - 83 = 0x1337
```

More sophisticated encoding:

```llvm
; Before: mov [rsp+8], 0xdeadbeef

; After: mov [rsp+8], decrypt(0xdeadbeef, global_key)
```

#### Why safe:

✅ Only transforms SSA values
✅ No CFG impact
✅ Works on both LLVM IR and machine-level code
✅ Transparent to control flow
✅ PE header relocation handling unaffected

#### Implementation difficulty:

Medium - requires new LLVM IR pass in OLLVM codebase

**Obfuscation strength**: ⭐⭐⭐ (strong - significant size/complexity increase)

---

### 3️⃣ NOP Insertion / Dead Code Injection (`-nop-insertion`)

**Status**: ✅ SAFE - Trivial to implement

#### What it does:

Inserts meaningless operations that have no effect:

```llvm
; Dead computation that's later unused
%unused = add i32 %x, 0
%unused2 = mul i32 %unused, 1

; OR: Store to a never-read address
store i32 %dummy, i32* %fake_slot

; OR: Call to LLVM intrinsic
call void @llvm.donothing()
```

Multiple variants:
- Dead arithmetic chains: `(x + 0) * 1 - 0`
- Dead loads/stores: `store value to dead address`
- Dead function calls: `call @fake_function()`
- Dead branches: Create unreachable blocks (then eliminate them)

#### Why safe:

✅ No effect on program semantics
✅ Does not modify CFG
✅ Lifter IR tolerates extra instructions
✅ Peephole optimization may remove them (acceptable)
✅ Binary size increases but execution unchanged

#### Implementation difficulty:

Very easy - ~100 lines of LLVM IR pass code

**Obfuscation strength**: ⭐ (weak - easily stripped by optimizations)

---

### 4️⃣ Basic Block Reordering (`-reorder-blocks`)

**Status**: ✅ SAFE - CFG-preserving

#### What it does:

Rearranges the order of basic blocks without changing branch targets:

```llvm
; Before
entry:
  ...
  br i1 %cond, label %A, label %B

A:
  ...
  br label %C

B:
  ...
  br label %C

C:
  ...
  ret i32 %result

; After (blocks reordered)
entry:
  ...
  br i1 %cond, label %A, label %B

C:                     ; Moved to position 2
  ...
  ret i32 %result

A:                     ; Moved to position 3
  ...
  br label %C

B:                     ; Moved to position 4
  ...
  br label %C
```

#### Why safe:

✅ Control-flow edges remain intact (labels still point to correct blocks)
✅ STATE struct + PC update logic unaffected
✅ Cache locality may change (neutral effect)
✅ No PHI node corruption
✅ Disassemblers get confused (good obfuscation)

#### Implementation difficulty:

Easy - built-in LLVM pass (with slight modifications for determinism)

**Obfuscation strength**: ⭐⭐ (moderate - confuses static analysis)

---

## Medium-Risk Passes

These modify control flow but **CAN** work with McSema IR if implemented very carefully.

### ⚠️ CRITICAL WARNINGS

Before using any of these:
1. Test on small binaries only (< 5KB)
2. Verify output behavior matches original
3. Use debugger to trace execution
4. Monitor for hangs/crashes

---

### 5️⃣ Indirect Call Indirection (`-indirect-call`)

**Status**: 🟡 MEDIUM RISK - Safe only for user-code functions

#### What it does:

Converts direct function calls to indirect calls via function pointers:

```llvm
; Before
call void @foo()

; After
%fptr = bitcast @foo to void (...)*
call void (...) %fptr()
```

More complex variant with function pointer arrays:

```llvm
; Create function pointer table
@fptrs = global [3 x void ()*] [@foo, @bar, @baz]

; Later: call via indexed array
%index = ...  ; computed at runtime
%entry = load i32, i32* @index
%fptr = load void ()*, void ()** getelementptr (@fptrs, %index)
call void %fptr()
```

#### Why medium-risk:

✅ Does NOT alter CFG topology
✅ Does NOT break PC updates (if applied only to user functions)
⚠️ DANGER: Lifting emulation blocks have special CALL/RET semantics
⚠️ DANGER: If applied to lifter-generated stub functions, breaks state machine

#### When it's safe:

- Apply ONLY to user-level functions (not `sub_*` functions from lifter)
- Skip lifter-generated wrappers and stubs
- Verify function signatures match
- Monitor for incorrect function dispatch

#### When it's dangerous:

- Applied to internal `sub_*` functions → state machine breaks
- Function pointers become ambiguous to symbolic execution
- Lifter IR can't verify correct target
- Result: unpredictable execution or crashes

#### Implementation difficulty:

Medium - needs function filtering logic

**Obfuscation strength**: ⭐⭐⭐ (strong - prevents direct call patterns)

---

### 6️⃣ Lightweight Flattening (Custom Mini-Version)

**Status**: 🟡 MEDIUM RISK - NOT OLLVM's standard flattening

#### What it does:

Introduces a dispatcher block that routes execution to all basic blocks:

```llvm
; BEFORE
entry:
  br label %bb1

bb1:
  %cond = icmp ...
  br i1 %cond, label %bb2, label %bb3

bb2:
  ...
  br label %exit

bb3:
  ...
  br label %exit

exit:
  ret i32 %result

; AFTER (lightweight flattening)
entry:
  %state_var = alloca i32
  store i32 0, i32* %state_var  ; state = entry
  br label %dispatcher

dispatcher:
  %state = load i32, i32* %state_var
  switch i32 %state, label %error [
    i32 0, label %entry_block
    i32 1, label %bb1_block
    i32 2, label %bb2_block
    i32 3, label %bb3_block
    i32 4, label %exit_block
  ]

entry_block:
  store i32 1, i32* %state_var  ; next = bb1
  br label %dispatcher

bb1_block:
  %cond = icmp ...
  br i1 %cond, label %set_bb2, label %set_bb3

set_bb2:
  store i32 2, i32* %state_var
  br label %dispatcher

set_bb3:
  store i32 3, i32* %state_var
  br label %dispatcher

bb2_block:
  ...
  store i32 4, i32* %state_var  ; next = exit
  br label %dispatcher

bb3_block:
  ...
  store i32 4, i32* %state_var
  br label %dispatcher

exit_block:
  ret i32 %result

error:
  unreachable
```

#### Why medium-risk:

✅ Original CFG is mathematically preserved
✅ Branch semantics remain identical
✅ PC update logic unaffected (state machine is separate)
✅ PHI nodes still correct (dispatcher doesn't introduce new PHIs)
⚠️ DANGER: Adds extra loop overhead (dispatcher loop)
⚠️ DANGER: Must NOT use OLLVM's standard `-fla` pass (that breaks everything)

#### What makes it safe:

1. **Original branches still exist**: Dispatcher just adds routing layer
2. **State variable is separate from PC**: Lifter's PC mechanism untouched
3. **No elimination of original edges**: All CFG transitions remain valid
4. **Loop is bounded**: Only iterations equal to function execution count

#### What makes it dangerous:

- **OLLVM's built-in `-fla` pass**: Creates opaque predicates that break symbolic execution
- **Improper implementation**: Can corrupt PHI dominance
- **Reachability issues**: Must ensure all blocks still reachable
- **Performance degradation**: 10-50x slowdown possible

#### Implementation difficulty:

High - requires custom CFG pass with proper dominance tree handling

**Obfuscation strength**: ⭐⭐⭐⭐ (very strong - restructures CFG completely)

---

### 7️⃣ Micro Fake Loops (Control-Flow Noise)

**Status**: 🟡 MEDIUM RISK - Acceptable if carefully tuned

#### What it does:

Inserts small bounded loops that always execute exactly N times:

```llvm
; BEFORE
entry:
  %result = call @foo()
  ret i32 %result

; AFTER
entry:
  %dummy_count = alloca i32
  store i32 0, i32* %dummy_count

fake_loop:
  %count = load i32, i32* %dummy_count
  %is_done = icmp eq i32 %count, 3
  br i1 %is_done, label %continue, label %loop_body

loop_body:
  %new_count = add i32 %count, 1
  store i32 %new_count, i32* %dummy_count
  ; ... do useless computation ...
  %dummy = mul i32 %count, 42
  store i32 %dummy, i32* %dummy_count  ; (overwrite with new_count)
  br label %fake_loop

continue:
  %result = call @foo()
  ret i32 %result
```

#### Why medium-risk:

✅ Adds CFG complexity without changing real control flow
✅ Does NOT interact with symbolic PC update
✅ Loops have bounded iterations (provably terminate)
⚠️ DANGER: Must not interact with SSA dominance
⚠️ DANGER: Lifter IR must handle loop back-edges correctly

#### When it's safe:

- Loop iteration count is constant and small (3-5 iterations max)
- Loop uses fresh variables (not state machine vars)
- Loop body has no side effects on real computation
- Loop is entirely within a single function

#### When it's dangerous:

- Applied to loop-critical code → can cause infinite loops
- Uses SSA variables that affect real computation
- Lifter can't prove loop termination → symbolic execution fails
- Breaks register liveness analysis

#### Implementation difficulty:

Medium - needs insertion logic + loop legality checks

**Obfuscation strength**: ⭐⭐ (weak - optimizers may remove it)

---

### 8️⃣ Indirect Branch Obfuscation (VERY Lightweight)

**Status**: 🟡 MEDIUM RISK - Minimal but requires backend support

#### What it does:

Converts direct branches to indirect branches:

```llvm
; BEFORE
br label %bb2

; AFTER
%target = blockaddress(@func, %bb2)
indirectbr i8* %target, [label %bb2]
```

More complex form with branch predicates:

```llvm
; BEFORE
br i1 %cond, label %true_block, label %false_block

; AFTER
%true_target = blockaddress(@func, %true_block)
%false_target = blockaddress(@func, %false_block)
%target = select i1 %cond, i8* %true_target, i8* %false_target
indirectbr i8* %target, [label %true_block, label %false_block]
```

#### Why medium-risk:

✅ IR still structurally correct
✅ Branch still targets correct block
✅ BlockAddress intrinsic is well-defined
⚠️ DANGER: Backend support varies (especially Mingw-lld)
⚠️ DANGER: Indirect branches have unique relocation requirements

#### When it's safe:

- Backend is known to support `indirectbr` (x86-64 LLVM 22 does)
- Limited number of indirectbr instructions (< 50)
- Not applied to exception handling blocks
- Not applied within lifter state machine

#### When it's dangerous:

- Backend doesn't properly handle blockaddress relocations
- Too many indirectbr instructions → linker errors
- Applied to EH blocks → unwinding broken
- Windows PE relocation tables don't support it

#### Implementation difficulty:

Easy - LLVM intrinsic, mostly straightforward

**Obfuscation strength**: ⭐⭐⭐ (strong - defeats static call graphs)

---

## Building the OLLVM Plugin Binary

### Overview

The custom `opt` binary used in Step 4 contains integrated OLLVM passes. To add new passes, you must rebuild the `opt` binary from LLVM source.

### Prerequisites

1. **LLVM 22 Source** (with OLLVM integration)
   - Location: `../llvm-project` (on this machine)
   - Branch: `ollvm-integration`
   - Already contains OLLVM passes

2. **Build Tools**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     cmake \
     ninja-build \
     clang-17 \
     llvm-17-dev \
     lld-17
   ```

3. **CMake 3.20+**
   ```bash
   cmake --version  # Should be >= 3.20
   ```

### Step 1: Build OLLVM from Source

#### Option A: Build from existing llvm-project

```bash
cd ../llvm-project

# Ensure you're on the ollvm-integration branch
git checkout ollvm-integration
git pull origin ollvm-integration

# Create build directory
mkdir -p build
cd build

# Configure CMake for optimized build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-17 \
  -DCMAKE_CXX_COMPILER=clang++-17 \
  -DLLVM_ENABLE_PROJECTS="llvm;clang" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt;libcxx" \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_OPTIMIZED=ON \
  -DLLVM_USE_LINKER=lld-17 \
  ../llvm

# Build (this takes 30-60 minutes on a typical machine)
ninja opt llc clang
```

#### Option B: Incremental rebuild (if already built)

```bash
cd ../llvm-project/build
ninja opt  # Rebuild only opt binary
```

### Step 2: Verify the Build

```bash
# Test opt binary
./bin/opt --version
# Output should show LLVM 22 with OLLVM patches

# List available passes
./bin/opt -passes=list-available 2>&1 | grep -i substitution
# Should show OLLVM passes like:
# - substitution
# - linear-mba
# - flattening
# - boguscf
# - split
```

### Step 3: Extract Only Necessary Binaries

For binary obfuscation mode, you need:
- `opt` - the optimizer with OLLVM passes
- `llvm-dis` - for disassembly (debugging)
- `clang` - for final compilation

```bash
cd ../llvm-project/build

# Verify binaries exist
ls -lh bin/opt bin/clang bin/llvm-dis

# File sizes
# opt: ~150 MB
# clang: ~200 MB
# llvm-dis: ~5 MB
```

---

## Integration with Current Setup

### Updating the `opt` Binary

The custom `opt` is used in two places:

1. **Production Container**: `/usr/local/llvm-obfuscator/bin/opt`
2. **Local Development**: `plugins/linux-x86_64/opt` (via GCP)

### Local Development Workflow

#### Step 1: Replace local `opt`

```bash
# From repo root
cp ../llvm-project/build/bin/opt plugins/linux-x86_64/opt
chmod +x plugins/linux-x86_64/opt

# Verify
./plugins/linux-x86_64/opt --version
```

#### Step 2: Test with run_ollvm.sh

```bash
# Create a test bitcode file
echo '
define i32 @test(i32 %a, i32 %b) {
entry:
  %0 = add i32 %a, %b
  ret i32 %0
}
' | llvm-as-22 - -o test.bc

# Create pass configuration
cat > passes_config.json <<EOF
{
  "substitution": true,
  "flattening": false,
  "bogus_control_flow": false,
  "split": false,
  "linear_mba": false,
  "string_encrypt": false,
  "symbol_obfuscate": false,
  "constant_obfuscate": false,
  "crypto_hash": false,
  "standard_llvm_opts": false
}
EOF

# Run the script
bash binary_obfuscation_pipeline/mcsema_impl/ollvm_stage/run_ollvm.sh \
  test.bc \
  test_output \
  passes_config.json

# Check output
ls -lh test_output/test_obf.bc
llvm-dis-22 test_output/test_obf.bc -o test_obf.ll
cat test_obf.ll
```

### Production Deployment

#### Step 1: Upload to GCP Binary Storage

```bash
# From repo root
./scripts/gcp-binary-manager.sh add \
  ../llvm-project/build/bin/opt \
  linux-x86_64/opt

# Verify upload
./scripts/gcp-binary-manager.sh list | grep opt
```

#### Step 2: Trigger CI/CD

Push changes to trigger automated deployment:

```bash
git add plugins/linux-x86_64/opt
git commit -m "Update OLLVM opt binary with new passes"
git push origin mcsema
```

The CI workflow (`.github/workflows/dockerhub-deploy.yml`) will:
1. Download updated binaries from GCP
2. Build new Docker image
3. Push to Docker Hub
4. Deploy to production server

#### Step 3: Manual Deployment (if needed)

```bash
# From repo root, copy opt to server
scp ../llvm-project/build/bin/opt \
    root@69.62.77.147:/tmp/opt

# Update container
ssh root@69.62.77.147 "
  docker cp /tmp/opt llvm-obfuscator-backend:/usr/local/llvm-obfuscator/bin/opt
  docker exec llvm-obfuscator-backend chmod +x /usr/local/llvm-obfuscator/bin/opt
  docker restart llvm-obfuscator-backend
"

# Verify
ssh root@69.62.77.147 \
  "docker exec llvm-obfuscator-backend /usr/local/llvm-obfuscator/bin/opt --version"
```

---

## Implementing New Passes

### Adding Constant Obfuscation Pass

This example shows how to add the `-constant-obfuscate` pass.

#### Step 1: Create Pass Source File

File: `../llvm-project/llvm/lib/Transforms/Obfuscation/ConstantObfuscation.cpp`

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Random.h"

using namespace llvm;

namespace {

class ConstantObfuscation : public PassInfoMixin<ConstantObfuscation> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    bool Changed = false;

    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *CI = dyn_cast<ConstantInt>(I.getOperand(0))) {
            Changed |= obfuscateConstant(CI, I);
          }
        }
      }
    }

    return Changed ? PreservedAnalyses::none()
                   : PreservedAnalyses::all();
  }

private:
  bool obfuscateConstant(ConstantInt *CI, Instruction &I) {
    int64_t value = CI->getSExtValue();

    // Simple encoding: value = (a * b) - c
    int64_t a = 55;
    int64_t b = 42;
    int64_t c = (a * b) - value;

    // Replace with computed sequence
    IRBuilder<> Builder(&I);
    Value *mulResult = Builder.CreateMul(
        ConstantInt::get(CI->getType(), a),
        ConstantInt::get(CI->getType(), b)
    );
    Value *subResult = Builder.CreateSub(
        mulResult,
        ConstantInt::get(CI->getType(), c)
    );

    CI->replaceAllUsesWith(subResult);
    return true;
  }
};

} // namespace

llvm::PassPluginLibraryInfo getConstantObfuscationPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ConstantObfuscation", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "constant-obfuscate") {
                    MPM.addPass(ConstantObfuscation());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getConstantObfuscationPluginInfo();
}
```

#### Step 2: Update CMakeLists.txt

File: `../llvm-project/llvm/lib/Transforms/Obfuscation/CMakeLists.txt`

Add:
```cmake
add_llvm_pass_plugin(ConstantObfuscation
  ConstantObfuscation.cpp
)
```

#### Step 3: Rebuild OLLVM

```bash
cd ../llvm-project/build
ninja opt

# Verify new pass is available
./bin/opt -passes=list-available 2>&1 | grep constant-obfuscate
# Output: constant-obfuscate
```

#### Step 4: Update Frontend

File: `cmd/llvm-obfuscator/frontend/src/components/BinaryObfuscationMode.tsx:15-25`

Add to pass state:
```typescript
const [passes, setPasses] = useState({
  // ... existing passes ...
  constant_obfuscate: false,  // NEW
});
```

#### Step 5: Update Shell Script

File: `binary_obfuscation_pipeline/mcsema_impl/ollvm_stage/run_ollvm.sh:210-216`

Add pass parsing:
```bash
CONSTANT_OBFUSCATE=$(jq -r '.constant_obfuscate // false' "$CONFIG_ABS")
```

Add to pass string (line 257-259):
```bash
if [ "$CONSTANT_OBFUSCATE" = "true" ]; then
    OLLVM_PASS_STRING="$OLLVM_PASS_STRING,constant-obfuscate"
fi
```

#### Step 6: Rebuild and Deploy

```bash
# Copy updated opt
cp ../llvm-project/build/bin/opt plugins/linux-x86_64/opt

# Test with new pass
cat > passes_config.json <<EOF
{
  "substitution": false,
  "constant_obfuscate": true,
  "flattening": false,
  "bogus_control_flow": false,
  "split": false,
  "linear_mba": false,
  "string_encrypt": false,
  "symbol_obfuscate": false,
  "crypto_hash": false
}
EOF

bash binary_obfuscation_pipeline/mcsema_impl/ollvm_stage/run_ollvm.sh \
  test.bc test_output passes_config.json
```

---

## Testing & Validation

### Test Case 1: Simple Arithmetic

#### Program: add.c
```c
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 7);
    printf("5 + 7 = %d\n", result);
    return 0;
}
```

#### Compilation and Obfuscation
```bash
# Compile to Windows PE (from Windows or cross-compile)
x86_64-w64-mingw32-gcc add.c -o add.exe

# Upload to OAAS and select:
# - substitution: ON
# - flattening: OFF
# - bogus_control_flow: OFF
# - All others: OFF

# Download obfuscated binary
# Test: ./add.exe → should output "5 + 7 = 12"
```

### Test Case 2: Fibonacci

#### Program: fib.c
```c
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main() {
    return fib(10);  // Should return 55
}
```

#### Validation
```bash
# Original: fib(10) = 55
# Obfuscated: Should still return 55

# Performance: Expect 5-20% slowdown
```

### Test Case 3: McSema Lifting Verification

```bash
# Step 1: Lift binary with Ghidra
# Step 2: Convert to LLVM IR with McSema
# Step 3: Apply substitution pass
# Step 4: Recompile to PE

# Verify:
# 1. Original and obfuscated produce same output
# 2. No crashes or hangs
# 3. Binary size similar (not inflated)
# 4. CFG complexity metric increases
```

### Validation Checklist

- [ ] Binary runs without crashing
- [ ] Output matches original (functional equivalence)
- [ ] Performance degradation acceptable (< 50%)
- [ ] Binary size increase reasonable (< 50%)
- [ ] No infinite loops or hangs
- [ ] Debugger can still attach
- [ ] Windows Defender doesn't flag as malware
- [ ] Works on target Windows 10/11 system

---

## Troubleshooting

### Issue 1: "Custom opt binary not found"

**Error**:
```
[ERROR] Custom opt binary not found: /usr/local/llvm-obfuscator/bin/opt
```

**Solution**:
```bash
# Check if binary exists
ls -lh /usr/local/llvm-obfuscator/bin/opt

# If missing, copy from plugins
cp plugins/linux-x86_64/opt /usr/local/llvm-obfuscator/bin/opt
chmod +x /usr/local/llvm-obfuscator/bin/opt

# Verify
/usr/local/llvm-obfuscator/bin/opt --version
```

### Issue 2: "Unknown pass in pipeline"

**Error**:
```
opt: unknown pass in pipeline 'constant-obfuscate'
```

**Solution**:
1. Verify pass is compiled into opt binary:
   ```bash
   opt -passes=list-available 2>&1 | grep constant-obfuscate
   ```
2. If missing, rebuild OLLVM with new pass
3. Ensure CMakeLists.txt was updated

### Issue 3: "Obfuscated binary crashes"

**Diagnosis**:
1. Test with smaller passes first (substitution only)
2. Check if original binary works correctly
3. Look at pipeline logs:
   ```bash
   # On server
   docker exec llvm-obfuscator-backend cat /app/reports/<jobid>/logs.txt
   ```
4. Try with flattening disabled

**Common causes**:
- Flattening corrupted state machine
- Bogus CFG created unreachable code
- Pass incompatible with McSema IR

### Issue 4: "LLVM 22 version mismatch"

**Error**:
```
LLVM version mismatch: expected 22, got 21
```

**Solution**:
```bash
# Verify opt version
opt --version  # Should show "LLVM version 22.x.x"

# Rebuild OLLVM if needed
cd ../llvm-project/build
ninja -j$(nproc) opt
```

### Issue 5: "PE relocation error in final compilation"

**Error**:
```
ld.lld-22: error: ... relocation ... requires dynamic relocation
```

**Solution**:
- Disable `indirectbr` obfuscation
- Use `-fPIC` flag in compilation
- Ensure Windows PE format support is enabled in LLVM

---

## Pass Recommendation Matrix

| Pass | Safety | Strength | Build Effort | Recommended |
|------|--------|----------|--------------|-------------|
| Substitution | ✅ Safe | ⭐⭐ | None | ✅ YES |
| Constant Obfuscation | ✅ Safe | ⭐⭐⭐ | Medium | ✅ YES |
| NOP Insertion | ✅ Safe | ⭐ | Easy | ⭐ Maybe |
| Block Reordering | ✅ Safe | ⭐⭐ | Easy | ✅ YES |
| Indirect Calls | 🟡 Medium | ⭐⭐⭐ | Medium | ⭐ Test first |
| Lightweight Flatten | 🟡 Medium | ⭐⭐⭐⭐ | High | ⚠️ Expert only |
| Fake Loops | 🟡 Medium | ⭐⭐ | Medium | ⭐ Test first |
| Indirect Branches | 🟡 Medium | ⭐⭐⭐ | Easy | ⭐ Test first |
| OLLVM Flattening | 🔴 Dangerous | ⭐⭐⭐⭐ | None | ❌ AVOID |
| Bogus CFG | 🔴 Dangerous | ⭐⭐⭐⭐ | None | ❌ AVOID |
| Split Blocks | 🔴 Dangerous | ⭐⭐ | None | ❌ AVOID |

---

## Next Steps

### Phase 1 (Current)
- [ ] Implement Instruction Substitution pass (already done)
- [ ] Document constraints for binary obfuscation

### Phase 2 (Short-term)
- [ ] Implement Constant Obfuscation pass
- [ ] Add Block Reordering pass
- [ ] Test on simple binaries

### Phase 3 (Medium-term)
- [ ] Implement lightweight custom flattening (not OLLVM's)
- [ ] Add NOP insertion pass
- [ ] Extensive testing on medium-sized binaries

### Phase 4 (Long-term)
- [ ] Implement indirect call indirection
- [ ] Add indirect branch obfuscation
- [ ] Deploy to production after validation

---

## References

- OLLVM Project: https://github.com/obfuscator-llvm/obfuscator
- McSema Lifter: https://github.com/trailofbits/mcsema
- LLVM 22 Documentation: https://llvm.org/docs/
- Obfuscation Techniques: https://en.wikipedia.org/wiki/Obfuscation_(software)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-09
**Maintainer**: OAAS Team
