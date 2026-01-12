# Building LLVM/OLLVM Binaries

Use this guide when you need to rebuild the LLVM 22 binaries and obfuscation plugins that ship with the toolkit.  The source for the passes lives in this repo under `llvm-patches/ollvm/`; it mirrors the fixes from upstream commits `a4f715900d1f` → `682b6125e289` in `../llvm-project`.

## 1. Prepare LLVM Sources

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-22.1.0    # or the release you want as a base
```

## 2. Overlay the Obfuscation Passes

From the **repo root** of this project run:

```bash
rsync -av llvm-patches/ollvm/ ../llvm-project/
```

This copies the contents of `llvm/include/llvm/Transforms/Obfuscation`, the corresponding `lib/Transforms/Obfuscation` sources (including the loadable plugin), the updated `LowerSwitch` helpers, and the necessary `opt` wiring.

## 3. Configure & Build

```bash
mkdir -p ../llvm-project/build
cd ../llvm-project/build

cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DLLVM_ENABLE_RTTI=ON

ninja opt clang mlir-opt mlir-translate LLVMObfuscationPlugin
```

Key outputs live in `../llvm-project/build/bin` (clang/opt/mlir tools) and `../llvm-project/build/lib/Transforms/Obfuscation/` (the shared libraries for flattening, bogus control flow, substitution, linear MBA, etc.).

## 4. Install into the Toolkit

Copy the binaries you need into the platform folder under `cmd/llvm-obfuscator/plugins/<platform>` (e.g., `linux-x86_64`).  Minimum set:

- `clang`, `opt`, `mlir-opt`, `mlir-translate`, `clangir`, `lld`, `upx` (optional) into the platform directory
- `LLVMObfuscationPlugin.so` (or `.dylib/.dll`) into the same directory

Update the executable bit (`chmod +x`) after copying so CI can run them.

## 5. Verifying the Build

1. Run `opt -load-pass-plugin=LLVMObfuscationPlugin.so -passes=substitution input.bc -o out.bc` to confirm the plugin loads.
2. Execute `python -m cmd.llvm_obfuscator.cli.obfuscate compile tests/test_simple.c --enable-flattening --enable-substitution` to ensure the new binaries work end-to-end.

## 6. Keeping Patches in Sync

- When upstreaming changes, modify the files under `llvm-patches/ollvm/` and re-run the `rsync` overlay.
- Note any additional upstream files you touch so contributors can re-apply them easily.
- Record new SHA256 hashes for tarballs uploaded to your artifact bucket (see `docs/DEPLOYMENT.md`).

Following this process gives every contributor the exact instructions (and source) needed to rebuild the LLVM-side components without relying on pre-uploaded tarballs.
