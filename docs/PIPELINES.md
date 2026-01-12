# Obfuscation Pipelines

This guide replaces the scattered CLANGIR, Polygeist, McSema, and Windows notes.  It lists the supported pipelines and how they fit into the overall toolchain.

## 1. Native Source → Obfuscated Binary (Default)

1. **Compilation** – The CLI/API invokes `clang` (or user supplied compiler) with the hardened base flags from `core/obfuscator.py`.
2. **MLIR Passes** – If `--string-encryption`, `--symbol-obfuscation`, or `--constant-obfuscation` are enabled the shared MLIR plugin is loaded and run via `-mllvm` flags.
3. **OLLVM Passes** – Requested passes (flattening, bogus control flow, linear MBA, substitution, split) are injected via bundled LLVM plugins.  Multiple cycles are supported through the `--cycles` flag.
4. **UPX Packing (optional)** – `--enable-upx` compresses the final binary while maintaining deterministic paths for reporting.
5. **Reporting** – JSON/Markdown reports describe symbol reductions, entropy changes, string analysis, and (optionally) Phoronix metrics.

**Typical command**

```bash
python -m cmd.llvm_obfuscator.cli.obfuscate compile src/simple_auth.c \
  --output build/obf \
  --level 4 \
  --string-encryption \
  --enable-symbol-obfuscation \
  --enable-flattening --enable-bogus-cf \
  --report-formats json,md
```

## 2. Windows Binary Lifting Pipeline (Experimental)

This flow exists inside `binary_obfuscation_pipeline/` for situations where only a PE binary is available.

1. **Stage 1 – Safe Windows Build** (`windows_build/`)
   - Validates source for unsupported constructs (recursion, inline asm, C++ features).
   - Compiles with MinGW-w64 using `-O0 -g -fno-inline -fno-exceptions` to preserve CFG fidelity.

2. **Stage 2 – Ghidra CFG Export** (`mcsema_impl/ghidra_lifter/`)
   - Headless Ghidra service exports a McSema compatible CFG JSON.

3. **Stage 3 – McSema Lift & LLVM 22 Upgrade** (`mcsema_impl/lifter/`)
   - `mcsema-lift` converts the CFG to LLVM bitcode (LLVM 10–17 dialect).
   - `convert_ir_version.sh` re-assembles the bitcode using LLVM 22.

4. **Stage 4 – OLLVM Pass Application** (`mcsema_impl/ollvm_stage/`)
   - Applies a curated subset of OLLVM passes that are safe for McSema IR (`substitution`, `linear_mba`, limited flattening).

5. **Stage 5 – Recompilation & Packaging** (handled by CLI/UPX)
   - The obfuscated LLVM 22 bitcode is compiled back to a PE binary with MLIR passes and optional packing.

**Limitations** – Not suitable for binaries with recursion, switch jump tables, SEH, or complex C++ constructs.  Use on small, debug builds until additional automation is landed.

## 3. MLIR / Polygeist Flow

The MLIR pipeline allows contributors to experiment with higher-level dialects before lowering to LLVM IR.

1. **Emit MLIR** – Use Polygeist/ClangIR to emit MLIR from C/C++ (see `mlir-obs/polygeist-pipeline.sh` for the manual steps, reproduced below).
2. **Apply MLIR Passes** – Run `mlir-opt` with `--load-pass-plugin=mlir-obs/build/lib/libMLIRObfuscation.so` and choose from `symbol-obfuscate`, `string-encrypt`, `crypto-hash`, etc.
3. **Lower to LLVM IR** – `mlir-translate --mlir-to-llvmir` converts the obfuscated MLIR program to LLVM IR.
4. **Compile** – Use `clang` to compile the IR to a native binary.  You can optionally re-enter the default pipeline for additional OLLVM passes.

```bash
# Example manual sequence (abbreviated)
clang -O0 -emit-mlir -c src/simple_auth.c -o work/simple_auth.mlir
mlir-opt work/simple_auth.mlir \
  --load-pass-plugin=mlir-obs/build/lib/libMLIRObfuscation.so \
  --pass-pipeline='builtin.module(symbol-obfuscate,string-encrypt)'
mlir-translate --mlir-to-llvmir work/simple_auth.mlir -o work/simple_auth.ll
clang work/simple_auth.ll -o work/simple_auth_obf
```

## 4. Phoronix + Security Analysis Loop

For performance regressions and deep security scoring, pair the pipelines above with the analysis harnesses shipped in `cmd/llvm-obfuscator/phoronix` and `obfuscation_test_suite/`.

1. Generate baseline & obfuscated binaries.
2. Run `bash phoronix/scripts/run_obfuscation_test_suite.sh <baseline> <obfuscated> results/` to capture strings, entropy, CFG diffs, and heuristic reverse-engineering difficulty.
3. Run `bash phoronix/scripts/run_pts_tests.sh --automatic` to collect performance deltas if needed.

The CLI/API propagate these outputs into the final report so downstream reviewers can reason about the trade-offs without spelunking logs.
