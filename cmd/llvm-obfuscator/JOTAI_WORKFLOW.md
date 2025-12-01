# Jotai Integration Workflow - Clarification

## What Actually Happens

The Jotai integration does the following:

### Step-by-Step Process:

1. **Get C Source Files from Jotai**
   - Downloads Jotai benchmark collection (C source files)
   - Each file contains a function + test driver

2. **Create Baseline Binary**
   ```bash
   clang -g -O1 benchmark.c -o benchmark_baseline
   ```
   - Normal compilation, no obfuscation
   - This is our reference binary

3. **Obfuscate Source Code → Create Obfuscated Binary**
   - **Source Code Transformations:**
     - Symbol obfuscation (rename functions/variables)
     - String encryption (encrypt string literals)
   - **Compilation with Obfuscation:**
     - LLVM passes applied during compilation (control flow flattening, etc.)
     - Results in obfuscated binary
   
   The obfuscator transforms the SOURCE CODE, then compiles it to produce an obfuscated binary.

4. **Run Both Binaries**
   ```bash
   ./benchmark_baseline <input>      # Normal binary
   ./benchmark_obfuscated <input>    # Obfuscated binary
   ```

5. **Verify Functional Equivalence**
   - Compare outputs from both binaries
   - Must be identical (same return codes, same stdout)
   - Confirms obfuscation didn't break functionality

## Important Clarification

**We obfuscate SOURCE CODE, which produces OBFUSCATED BINARIES.**

The LLVM obfuscator works by:
- Taking source code as input
- Applying transformations (symbol renaming, string encryption, control flow obfuscation)
- Compiling the transformed source with LLVM passes
- Producing obfuscated binaries as output

We don't obfuscate already-compiled binaries directly. We obfuscate during the compilation process.

## CI Workflow

On every PR and merge:
1. Randomly selects 20-30 benchmarks (using run number as seed)
2. For each benchmark:
   - Compiles baseline binary
   - Obfuscates source → compiles obfuscated binary
   - Runs both with same inputs
   - Verifies outputs match
3. Reports success/failure based on functional equivalence

## Why This Approach?

- **Source-level obfuscation** allows for:
  - Symbol renaming (can't rename symbols in compiled binary)
  - String encryption (can inject decryption code)
  - Better control flow transformations
  
- **LLVM passes** work at IR level during compilation:
  - Control flow flattening
  - Instruction substitution
  - Bogus control flow
  - These require access to IR, not final binary

## Result

We end up with:
- ✅ Baseline binaries (normal, unobfuscated)
- ✅ Obfuscated binaries (transformed source compiled with obfuscation)
- ✅ Verification that both produce identical outputs

