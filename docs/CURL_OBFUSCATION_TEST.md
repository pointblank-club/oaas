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

## December 3, 2025 - All Layers Test

### Objective

Test applying **ALL 4 obfuscation layers** to curl:
- Layer 1: Symbol Obfuscation (SHA256, 12 chars, typed prefix)
- Layer 2: String Encryption
- Layer 2.5: Indirect Call Obfuscation
- Layer 3: OLLVM Passes (substitution, boguscf, split)
- Layer 4: Compiler Flags

### Configuration

| Setting | Value |
|---------|-------|
| Repository | curl/curl (master branch) |
| Files | 684 C/C++ source files |
| Build System | CMake |
| All Layers Enabled | Yes |
| OLLVM Passes | substitution, boguscf, split |
| Flattening | **DISABLED** (causes segfaults) |
| Timeout | 180 minutes |

### Results

| Layer | Requested | Applied | Evidence |
|-------|-----------|---------|----------|
| Layer 1: Symbol Obfuscation | Yes | **NO** | Symbols like `curl_easy_init` still visible |
| Layer 2: String Encryption | Yes | **NO** | Strings like "error initializing curl" in plaintext |
| Layer 2.5: Indirect Calls | Yes | **NO** | 0 indirect calls processed |
| Layer 3: OLLVM Passes | Yes | **YES** | Confirmed via `opt --passes=substitution,boguscf,split` |
| Layer 4: Compiler Flags | Yes | **PARTIAL** | Binary is stripped (`-Wl,-s`) |

### Bug Found: Source-Level Obfuscation Not Applied

Server logs revealed:
```
2025-12-03 15:43:12,245 - api - INFO - Source obfuscation complete: {'files_processed': 0, 'symbols_renamed': 0, 'strings_encrypted': 0, 'indirect_calls': 0}
```

**Root Cause:** The `_apply_string_encryption` and `_apply_indirect_calls` functions in `server.py` are placeholder implementations that count but don't actually modify content. The `_apply_symbol_obfuscation` function works but may not be finding symbols due to strict filtering rules.

**Impact:** For repository/CMake builds, only Layer 3 (OLLVM via wrapper scripts) and partial Layer 4 (linker flags) are currently effective.

### Binary Analysis

```bash
$ ls -la curl_obfuscated_all_layers
-rwxr-xr-x 1 root root 5508952 Dec  3 16:15 curl_obfuscated_all_layers

$ file curl_obfuscated_all_layers
ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV),
dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2,
for GNU/Linux 3.2.0, stripped

$ nm curl_obfuscated_all_layers
no symbols  # Symbol table stripped

$ strings curl_obfuscated_all_layers | grep -c curl
156  # Curl-related strings still visible (Layer 2 not working)
```

### Timeline

| Time (UTC) | Event |
|------------|-------|
| 15:42:08 | Repository download started |
| 15:43:12 | Source obfuscation attempted (0 files processed) |
| 15:43:12 | OLLVM wrapper scripts configured |
| 15:57:00 | CMake configure complete |
| 16:15:35 | Build completed successfully |
| **Total** | **~32 minutes** |

---

## December 3, 2025 - OLLVM-Only Test (Earlier)

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
| Layer 3: Instruction Substitution | Working |
| Layer 3: Bogus Control Flow | Working |
| Layer 3: Split Basic Blocks | Working |
| Layer 4: Symbol Stripping (`-Wl,-s`) | Working |
| Large project builds | Working (with 180min timeout) |

### What Doesn't Work

| Feature | Status | Notes |
|---------|--------|-------|
| Layer 1: Symbol Obfuscation | Not Working | Frontend sends `enabled: false` even when checked |
| Layer 2: String Encryption | Not Working | Backend placeholder returns unchanged content |
| Layer 2.5: Indirect Calls | Not Working | Frontend doesn't send config to backend |
| Layer 3: Control Flow Flattening | Unstable | Causes segfaults on complex code |

### Known Bug: Source-Level Obfuscation for Repository Builds

When building from a GitHub repository (CMake/Make/Autotools), the source-level obfuscation (Layers 1, 2, 2.5) processes 0 files.

**Verified via Debug Logging (Dec 3, 2025):**
```
[DEBUG] Layer 1 (Symbol Obfuscation) enabled: False   ← Bug: UI checked but not sent
[DEBUG] Layer 2 (String Encryption) enabled: True     ← Sent correctly
[DEBUG] Layer 2.5 (Indirect Calls) enabled: False     ← Not sent from frontend
[DEBUG] Found 1 source files to process               ← Files ARE being found
[DEBUG] Layer 4 (Compiler Flags): ['-O3', '-fno-builtin', '-Wl,-s', ...]  ← Working
```

**Root Causes:**
1. **Layer 1**: Frontend not sending `symbol_obfuscation.enabled: true` to backend
2. **Layer 2**: `_apply_string_encryption()` returns original content unchanged (placeholder)
3. **Layer 2.5**: Frontend not sending indirect_calls config to backend
4. **Layer 4**: Working correctly - flags are passed via `OLLVM_CFLAGS` and `LDFLAGS`

**Workaround:** For now, rely on Layer 3 (OLLVM passes) and Layer 4 (compiler flags) which work correctly.

### Recommendations

1. **Use wrapper scripts** instead of CFLAGS manipulation
2. **Disable flattening** for complex projects
3. **Set timeout to 180 minutes** for large projects
4. **Enable only**: substitution, boguscf, split
5. **Note:** Layers 1, 2, 2.5 need fixes before they work with repo builds

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
