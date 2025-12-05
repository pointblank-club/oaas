# LTO Linker Fix Documentation

## Problem

When LTO (Link-Time Optimization) flags (`-flto`, `-flto=thin`) were enabled, compilation failed with:

### Linux Error
```
/usr/bin/ld: /usr/local/llvm-obfuscator/bin/../lib/LLVMgold.so: error loading plugin:
/usr/local/llvm-obfuscator/bin/../lib/LLVMgold.so: cannot open shared object file: No such file or directory
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

### Windows Error
```
clang: error: 'x86_64-w64-windows-gnu': unable to pass LLVM bit-code files to linker
```

## Root Cause

LTO requires a linker that can process LLVM bitcode. The default system linker (`ld`) needs `LLVMgold.so` plugin to handle LTO, which wasn't installed.

## Solution

Use LLVM's native linker `lld` which handles LTO natively without needing `LLVMgold.so`.

### Changes Made

**File:** `cmd/llvm-obfuscator/core/obfuscator.py`

Added `-fuse-ld=lld` flag for Linux and Windows targets:

```python
# Add lld linker for LTO support (required for Linux and Windows, macOS uses ld64.lld)
# lld handles LTO natively without needing LLVMgold.so
if config.platform == Platform.WINDOWS:
    final_cmd.append("-fuse-ld=lld")
elif config.platform == Platform.LINUX:
    final_cmd.append("-fuse-ld=lld")
```

### Linker Configuration by Platform

| Platform | Linker | Notes |
|----------|--------|-------|
| **Linux** | `lld` | LLVM's ELF linker, handles LTO natively |
| **Windows** | `lld` | LLVM's COFF/PE linker for MinGW targets |
| **macOS** | `ld64.lld` | LLVM's Mach-O linker (configured via `_get_macos_cross_compile_flags`) |

## Test Results

### Test Date: 2024-12-05

All tests performed with LTO flags: `-flto -flto=thin`

### Linux Build (Target: x86_64-unknown-linux-gnu)
```
$ file test_lto
test_lto: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV),
dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2,
for GNU/Linux 3.2.0, stripped

Size: 6,008 bytes
Status: SUCCESS
```

### Windows Build (Target: x86_64-w64-mingw32)
```
$ file test_lto.exe
test_lto.exe: PE32+ executable for MS Windows 6.00 (console), x86-64, 6 sections

Size: 14,848 bytes
Status: SUCCESS
```

### macOS Build (Target: x86_64-apple-darwin)
```
$ file test_macos_lto
test_macos_lto: Mach-O 64-bit x86_64 executable,
flags:<NOUNDEFS|DYLDLINK|TWOLEVEL|BINDS_TO_WEAK|PIE>

Size: 14,584 bytes
Status: SUCCESS
```

## API Test Commands

### Linux with LTO
```bash
curl -X POST http://localhost:8000/api/obfuscate/sync \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "<base64-encoded-source>",
    "filename": "test.cpp",
    "platform": "linux",
    "config": {
      "compiler_flags": ["-flto", "-flto=thin"]
    }
  }'
```

### Windows with LTO
```bash
curl -X POST http://localhost:8000/api/obfuscate/sync \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "<base64-encoded-source>",
    "filename": "test.cpp",
    "platform": "windows",
    "config": {
      "compiler_flags": ["-flto", "-flto=thin"]
    }
  }'
```

### macOS with LTO
```bash
curl -X POST http://localhost:8000/api/obfuscate/sync \
  -H "Content-Type: application/json" \
  -d '{
    "source_code": "<base64-encoded-source>",
    "filename": "test.cpp",
    "platform": "macos",
    "config": {
      "compiler_flags": ["-flto", "-flto=thin"]
    }
  }'
```

## Additional Fixes in This Commit

1. **macOS Data Layout**: Added proper Mach-O data layout (`m:o` mangling) for MLIR-generated IR
2. **LLVM Version Compatibility**: Convert `captures(none)` to `nocapture` for LTO compatibility with older tools
3. **Baseline Cross-Compilation**: Include target flags in baseline binary compilation for accurate comparisons

## Commit Details

- **Branch**: `hikari-exception-support`
- **Files Changed**:
  - `cmd/llvm-obfuscator/core/obfuscator.py`
  - `cmd/llvm-obfuscator/api/server.py`
