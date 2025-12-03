# CURL Obfuscation Test - Complete Documentation

## Overview

This document details the successful obfuscation of **curl** from source using OAAS (Obfuscation-as-a-Service) with OLLVM passes via wrapper scripts.

**Latest Test Date:** December 3, 2025
**Result:** SUCCESS
**Build Time:** ~32 minutes
**Binary Size:** 5.6 MB (5,668,384 bytes)

---

## Evolution of Approaches

### Previous Approach (December 2, 2025) - DEPRECATED

The original approach used `-Xclang -load` and `-mllvm` flags in CFLAGS:

```bash
CFLAGS="-Xclang -load -Xclang /usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so \
        -mllvm -fla -mllvm -sub -mllvm -bcf -mllvm -split"
```

**Problems encountered:**
1. CMake try_compile tests fail with OLLVM flags in CFLAGS
2. Required two-phase CMake build (configure without CFLAGS, build with CFLAGS)
3. Flattening pass (`-fla`) caused segfaults on complex files like `cf-https-connect.c`
4. Plugin not recognized by bundled clang with those flag patterns

### Current Approach (December 3, 2025) - RECOMMENDED

The new approach uses **wrapper scripts** that apply OLLVM passes via the `opt` tool:

```
Source.c
    → Step 1: clang -emit-llvm -c → Source.bc (LLVM bitcode)
    → Step 2: opt --passes=substitution,boguscf,split → Source_obf.bc
    → Step 3: clang -c Source_obf.bc → Source.o (object file)
```

**Advantages:**
1. Works with any build system (CMake, Make, Autotools)
2. CMake try_compile tests automatically skipped from OLLVM
3. No CFLAGS manipulation needed
4. More reliable and maintainable

---

## Wrapper Script Implementation

### Files

- `scripts/clang-obfuscate` - C wrapper script
- `scripts/clang++-obfuscate` - C++ wrapper script

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLVM_PASSES` | Comma-separated list of passes | `substitution,boguscf,split` |
| `OLLVM_PLUGIN` | Path to LLVMObfuscationPlugin.so | `/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so` |
| `OLLVM_DEBUG` | Enable verbose logging | `1` |
| `OLLVM_CFLAGS` | Additional compiler flags | `-O3 -fno-builtin` |

### CMake Test Detection

The wrapper scripts automatically detect and skip OLLVM for CMake try_compile tests:

```bash
# Patterns that trigger passthrough mode (no OLLVM):
- *"CMakeFiles/CMakeScratch"*
- *"CMakeFiles/CMakeTmp"*
- *"CMakeFiles/CheckTypeSize"*
- *"CMakeFiles/CheckSymbol"*
- *"CMakeFiles/CheckFunction"*
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    clang-obfuscate wrapper                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: source.c with compilation flags                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Is this a CMake try_compile test?                      │   │
│  │  (Check path for CMakeFiles/CMakeScratch, etc.)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│              │                              │                   │
│            YES                             NO                   │
│              │                              │                   │
│              ▼                              ▼                   │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │  PASSTHROUGH MODE   │      │  OLLVM OBFUSCATION MODE     │  │
│  │  clang.real $@      │      │                             │  │
│  └─────────────────────┘      │  Step 1: Source → Bitcode   │  │
│                               │  clang -emit-llvm -c        │  │
│                               │                             │  │
│                               │  Step 2: Apply OLLVM        │  │
│                               │  opt --passes=$OLLVM_PASSES │  │
│                               │                             │  │
│                               │  Step 3: Bitcode → Object   │  │
│                               │  clang -c obf.bc -o .o      │  │
│                               └─────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## December 3, 2025 Test Results

### Configuration

| Setting | Value |
|---------|-------|
| Repository | curl/curl (master branch) |
| Files | 684 C/C++ source files |
| Build System | CMake |
| OLLVM Passes | substitution, boguscf, split |
| Flattening | **DISABLED** (causes segfaults) |
| Timeout | 180 minutes |

### CMake Options

```
-DBUILD_TESTING=OFF
-DBUILD_CURL_EXE=ON
-DBUILD_SHARED_LIBS=OFF
-DCURL_USE_LIBPSL=OFF
-DCURL_USE_LIBSSH2=OFF
-DUSE_NGHTTP2=OFF
```

### Timeline

| Time | Event |
|------|-------|
| 14:53:02 | Build started |
| 14:53 - 15:07 | CMake configure phase (~15 min) |
| 15:07 - 15:25 | Actual compilation with OLLVM (~18 min) |
| 15:25:21 | Build completed successfully |
| **Total** | **~32 minutes** |

### Backend Logs

```
2025-12-03 14:53:02,968 - api - INFO - Using OLLVM wrapper scripts with passes: ['substitution', 'boguscf', 'split']
2025-12-03 14:53:02,969 - api - INFO - Build environment: CC=/usr/local/llvm-obfuscator/bin/clang-obfuscate, OLLVM_PASSES=substitution,boguscf,split
2025-12-03 15:25:21,954 - api - INFO - Build completed successfully
```

### Process Verification

During the build, we observed the full 3-step OLLVM pipeline:

```bash
# Step 2 (opt applying passes):
/usr/local/llvm-obfuscator/bin/opt --load-pass-plugin=...LLVMObfuscationPlugin.so \
    --passes=substitution,boguscf,split curl_endian.bc -o curl_endian_obf.bc

# Step 3 (compiling obfuscated bitcode):
/usr/local/llvm-obfuscator/bin/clang.real -c curl_addrinfo_obf.bc -o curl_addrinfo.c.o
```

---

## Binary Analysis

### File Information

```bash
$ ls -la obfuscated_curl
-rwxr-xr-x 1 root root 5668384 Dec  3 15:25 obfuscated_curl

$ file obfuscated_curl
obfuscated_curl: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV),
dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2,
for GNU/Linux 3.2.0, not stripped
```

### Size Comparison

| Binary | Size | Notes |
|--------|------|-------|
| Obfuscated curl | 5.6 MB | With OLLVM (sub, bcf, split) |
| Previous test (with fla) | 1.19 MB | Layer 3+4 obfuscation |
| System curl | ~310 KB | Standard build |

**Note:** The larger size (5.6 MB vs 1.19 MB) is due to different obfuscation configurations and linking options.

---

## Flattening Pass Issue

### Problem

The Control Flow Flattening (`-fla` / `flattening`) pass causes segfaults on certain complex source files:

```
clang: error: clang frontend command failed with exit code 134
clang: error: unable to execute command: Aborted (core dumped)
```

### Affected Files (Examples)

- `lib/cf-https-connect.c` - Complex HTTPS connection handling
- Files with extensive switch statements
- Files with indirect branch terminators (goto)

### Root Cause

The flattening pass in `llvm/lib/Transforms/Obfuscation/Flattening.cpp` struggles with:
1. Functions with 3+ successor blocks
2. Complex switch statements
3. Indirect branch terminators

### Solution

**Do NOT enable flattening for complex projects like curl.** Use only:
- `substitution` - Instruction substitution
- `boguscf` - Bogus control flow
- `split` - Split basic blocks

These three passes provide significant obfuscation without the stability issues.

---

## Server Deployment

### Dockerfile Changes

```dockerfile
# Copy custom LLVM binaries and plugins
COPY plugins/linux-x86_64/clang /usr/local/llvm-obfuscator/bin/clang.real
COPY plugins/linux-x86_64/opt /usr/local/llvm-obfuscator/bin/opt
COPY plugins/linux-x86_64/LLVMObfuscationPlugin.so /usr/local/llvm-obfuscator/lib/

# Copy OLLVM wrapper scripts
COPY scripts/clang-obfuscate /usr/local/llvm-obfuscator/bin/clang-obfuscate
COPY scripts/clang++-obfuscate /usr/local/llvm-obfuscator/bin/clang++-obfuscate

# Create symlink for clang (wrapper scripts reference clang.real)
RUN ln -sf /usr/local/llvm-obfuscator/bin/clang.real /usr/local/llvm-obfuscator/bin/clang
```

### Server.py Changes

The server now detects wrapper scripts and uses them automatically:

```python
if ollvm_passes and plugin_path:
    wrapper_clang = custom_llvm / "clang-obfuscate"
    wrapper_clangpp = custom_llvm / "clang++-obfuscate"
    if wrapper_clang.exists() and wrapper_clangpp.exists():
        env["CC"] = str(wrapper_clang)
        env["CXX"] = str(wrapper_clangpp)
        env["OLLVM_PASSES"] = ",".join(ollvm_passes)
        env["OLLVM_PLUGIN"] = plugin_path
```

---

## Running the Obfuscated Binary

### On Linux Server

```bash
ssh root@69.62.77.147
docker exec llvm-obfuscator-backend /app/reports/73db125c06de43108f48894270bc8f7e/obfuscated_curl --version
```

### Download and Test Locally

```bash
# Download
scp root@69.62.77.147:/tmp/obfuscated_curl ~/Desktop/

# Run in Docker (requires Linux x86_64)
docker run -it --rm --platform linux/amd64 \
  -v ~/Desktop/obfuscated_curl:/curl \
  debian:bookworm bash

# Inside container
apt update && apt install -y libssl3 libbrotli1 libzstd1 libidn2-0 libldap-2.5-0 ca-certificates
chmod +x /curl
/curl --version
```

---

## Summary

### What Works

| Feature | Status |
|---------|--------|
| Wrapper script approach | Working |
| CMake try_compile detection | Working |
| Instruction Substitution | Working |
| Bogus Control Flow | Working |
| Split Basic Blocks | Working |
| Large project builds | Working (with 180min timeout) |

### What Doesn't Work

| Feature | Status | Notes |
|---------|--------|-------|
| Control Flow Flattening | Unstable | Causes segfaults on complex code |

### Recommendations

1. **Use wrapper scripts** instead of CFLAGS manipulation
2. **Disable flattening** for complex projects
3. **Set timeout to 180 minutes** for large projects
4. **Enable only**: substitution, boguscf, split

---

## Files Modified

1. `scripts/clang-obfuscate` - Wrapper script with CMake detection
2. `scripts/clang++-obfuscate` - C++ wrapper script
3. `api/server.py` - Wrapper script detection and OLLVM_PASSES env var
4. `Dockerfile.backend` - Copy wrapper scripts to container
5. `frontend/src/App.tsx` - Removed redundant `-mllvm` flags

---

## Related Documentation

- [OLLVM_FIX_PLAN.md](OLLVM_FIX_PLAN.md) - Implementation plan and details
- [OLLVM_CRASH_ANALYSIS.md](OLLVM_CRASH_ANALYSIS.md) - Analysis of flattening crashes
