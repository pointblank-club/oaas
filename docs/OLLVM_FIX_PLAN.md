# OLLVM Fix Plan

## Status: IMPLEMENTED ✅

The wrapper script approach has been implemented and is working. This document now serves as reference for the implementation and ongoing improvements.

---

## Problem Statement

The original OLLVM integration had issues:

1. **Wrong flag syntax**: Using `-mllvm -fla/-sub/-bcf/-split` which the plugin doesn't recognize
2. **CMake ignores CFLAGS**: CMake bakes flags at configure time, ignores build-time CFLAGS
3. **Crashes on complex code**: Some files crash the `opt` tool during pass application

## Solution: Compiler Wrapper Scripts ✅

Wrapper scripts intercept compilation and apply OLLVM passes via `opt`:

### Pipeline for each .c/.cpp file:

```
Source.c
    → Step 1: clang -emit-llvm -c → Source.bc (LLVM bitcode)
    → Step 2: opt --load-pass-plugin=... --passes=flattening,substitution,boguscf,split → Source_obf.bc
    → Step 3: clang -c Source_obf.bc → Source.o (object file)
```

## Implementation Status

### Completed ✅

1. **Wrapper Scripts**: `scripts/clang-obfuscate` and `scripts/clang++-obfuscate`
2. **Server Integration**: `api/server.py` sets CC/CXX to wrapper scripts
3. **Docker Integration**: `Dockerfile.backend` includes bundled LLVM 22 toolchain
4. **Environment Variables**:
   - `OLLVM_PASSES` - Comma-separated list of passes
   - `OLLVM_PLUGIN` - Path to LLVMObfuscationPlugin.so
   - `OLLVM_DEBUG` - Enable verbose logging
   - `OLLVM_CFLAGS` - Additional compiler flags
   - `OLLVM_FALLBACK` - Enable graceful fallback on crash (NEW)
   - `OLLVM_EXCLUDE` - Regex pattern for files to exclude (NEW)

---

## Graceful Fallback Feature (NEW - December 2025)

### Problem: Crashes on Complex Code

When obfuscating complex projects like curl, the `opt` tool can crash with segmentation fault on certain files (e.g., `cf-https-connect.c`). This happens because:

1. **Flattening pass** struggles with:
   - Complex switch statements
   - Indirect branch terminators (goto)
   - Functions with 3+ successor blocks

2. **BogusControlFlow pass** can fail on:
   - Complex basic blocks with unremappable values
   - Functions with extensive use of function pointers

### Solution: Progressive Fallback

The wrapper scripts now implement graceful fallback:

```bash
# Default: OLLVM_FALLBACK=1 (enabled)

# Fallback order:
1. Try all requested passes (flattening,substitution,boguscf,split)
2. If crash → Try without flattening (substitution,boguscf,split)
3. If crash → Try split only
4. If crash → Compile without OLLVM (original code)
```

### Usage Examples

```bash
# Normal build with graceful fallback (default)
OLLVM_PASSES="flattening,substitution,boguscf,split" make

# Disable fallback (fail on crash)
OLLVM_FALLBACK=0 OLLVM_PASSES="flattening,substitution,boguscf,split" make

# Exclude specific files from OLLVM
OLLVM_EXCLUDE="cf-https-connect|complex_file" make

# Debug mode to see which files use fallback
OLLVM_DEBUG=1 make 2>&1 | grep WARNING
```

### Warnings Output

When fallback is used, warnings are printed:
```
[clang-obfuscate] WARNING: Partial obfuscation (no flattening): lib/cf-https-connect.c
[clang-obfuscate] WARNING: OLLVM failed, compiling without obfuscation: lib/weird_file.c
```

---

## File Locations

```
cmd/llvm-obfuscator/
├── scripts/
│   ├── clang-obfuscate      # C wrapper script (with fallback)
│   └── clang++-obfuscate    # C++ wrapper script (with fallback)
├── plugins/
│   └── linux-x86_64/
│       ├── clang            # Bundled LLVM 22 clang
│       ├── opt              # Bundled LLVM 22 opt
│       └── LLVMObfuscationPlugin.so
├── api/
│   └── server.py            # Backend API
└── Dockerfile.backend       # Docker build
```

## OLLVM Pass Source Code

Located in `/Users/akashsingh/Desktop/llvm-project/llvm/lib/Transforms/Obfuscation/`:

| Pass | File | Risk Level |
|------|------|------------|
| Flattening | `Flattening.cpp` | HIGH - Most crashes |
| BogusControlFlow | `BogusControlFlow.cpp` | MEDIUM |
| SplitBasicBlocks | `SplitBasicBlocks.cpp` | LOW |
| Substitution | `Substitution.cpp` | LOW |
| LinearMBA | `LinearMBA.cpp` | LOW |

---

## Testing

### Test Complex Projects

```bash
# curl (684 files)
cd /tmp && git clone https://github.com/curl/curl
cd curl && autoreconf -fi
CC=clang-obfuscate OLLVM_PASSES="flattening,substitution,boguscf,split" ./configure
make -j4  # Should complete with some fallback warnings
./src/curl --version  # Verify binary works
```

### Verify Obfuscation

```bash
# Check for switch dispatcher pattern (flattening)
objdump -d ./src/curl | grep -A5 "switch" | head -20

# Check for bogus conditions
strings ./src/curl | grep -E "^x$|^y$"  # Global vars from boguscf

# Count indirect jumps (obfuscation indicator)
objdump -d ./src/curl | grep "jmp.*\*" | wc -l
```

---

## Future Improvements

1. **Plugin Hardening**: Add try-catch in PluginRegistration.cpp
2. **Pass Fixes**: Improve Flattening pass to handle indirectbr
3. **Metrics**: Track which files use fallback for analysis
4. **Debug Builds**: Build plugin with debug symbols for stack traces

---

## Related Documentation

- `docs/OLLVM_CRASH_ANALYSIS.md` - Detailed crash analysis
- `docs/CURL_OBFUSCATION_TEST.md` - curl test results
- `docs/DOOM_OBFUSCATION_TEST.md` - DOOM test results
