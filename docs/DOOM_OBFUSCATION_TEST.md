# DOOM Obfuscation Test - Documentation

## Overview

This document details the obfuscation of **id Software's DOOM (linuxdoom-1.10)** from source using OAAS (Obfuscation-as-a-Service).

**Test Date:** December 3, 2025
**Repository:** https://github.com/id-Software/DOOM
**Source:** `linuxdoom-1.10/` directory (62 C source files)
**Build System:** Make

---

## Repository Analysis

### Source Structure

```
DOOM/
├── linuxdoom-1.10/          # Main source (62 .c files, 70 .h files)
│   ├── Makefile             # Build configuration
│   ├── i_main.c             # Entry point
│   ├── d_main.c             # Game initialization
│   ├── g_game.c             # Game logic
│   ├── r_*.c                # Rendering engine
│   ├── p_*.c                # Game physics/entities
│   ├── m_*.c                # Menu and misc
│   └── ...
├── sndserv/                 # Sound server (separate binary)
├── ipx/                     # IPX networking
└── sersrc/                  # Serial networking
```

### Key Statistics

| Metric | Value |
|--------|-------|
| C Source Files | 62 |
| Header Files | 70 |
| Total Lines of Code | ~35,000 |
| Build System | Makefile |
| Target Platform | Linux (X11) |
| Dependencies | X11, Xext, nsl, m |

### Original Makefile

```makefile
CC=  gcc
CFLAGS=-g -Wall -DNORMALUNIX -DLINUX
LDFLAGS=-L/usr/X11R6/lib
LIBS=-lXext -lX11 -lnsl -lm

# Output binary
$(O)/linuxxdoom: $(OBJS) $(O)/i_main.o
    $(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) $(O)/i_main.o -o $(O)/linuxxdoom $(LIBS)
```

---

## Frontend Configuration Steps

### Step 1: Load Repository

1. Navigate to http://69.62.77.147:4666/ (or your OAAS frontend URL)
2. Click **GITHUB** button
3. Enter repository URL: `https://github.com/id-Software/DOOM`
4. Select branch: `master`
5. Click **Clone Repository**
6. Wait for confirmation of source files detected

### Step 2: Select Build System

- **Build System:** Custom

**Important:** The DOOM Makefile is in the `linuxdoom-1.10/` subdirectory, not the repository root. Use "custom" build system to handle this properly.

### Step 3: Configure Output Binary Path

- **Output Binary Path:** `linuxdoom-1.10/linux/linuxxdoom`

### Step 4: Configure Custom Build Command

Since the Makefile is in a subdirectory and requires creating the output directory, use this custom build command:

```bash
cd linuxdoom-1.10 && mkdir -p linux && make CC=$CC CXX=$CXX CFLAGS="$CFLAGS -DNORMALUNIX -DLINUX" -j$(nproc)
```

Or alternatively, enter in the frontend's "Custom Build Command" field:
```
cd linuxdoom-1.10 && mkdir -p linux && make CC=$CC CXX=$CXX CFLAGS="$CFLAGS -DNORMALUNIX -DLINUX" -j$(nproc)
```

### Step 5: Enable Obfuscation Layers

**Recommended Configuration (following curl pattern):**

- [x] **Layer 1:** Symbol Obfuscation (SHA256, 12 chars, typed prefix)
- [x] **Layer 2:** String Encryption (XOR encryption)
- [x] **Layer 2.5:** Indirect Call Obfuscation (stdlib + custom functions)
- [x] **Layer 3:** OLLVM Passes
  - [x] Control Flow Flattening (-fla)
  - [x] Instruction Substitution (-sub)
  - [x] Bogus Control Flow (-bcf)
  - [x] Split Basic Blocks (-split)
- [x] **Layer 4:** Compiler Flags
  - [x] Symbol Hiding (-fvisibility=hidden)
  - [x] Remove Frame Pointer (-fomit-frame-pointer)
  - [x] Speculative Load Hardening (-mspeculative-load-hardening)
  - [x] Maximum Optimization (-O3)
  - [x] Strip Symbols (-Wl,-s)
  - [x] Disable Built-in Functions (-fno-builtin)
  - [ ] LTO - **DISABLED** (incompatible with Layer 3 OLLVM)

### Step 6: Execute

Click **"OBFUSCATE"** button and wait for completion.

---

## Expected Compiler Flags

When all layers are enabled (except LTO), the backend will use:

```bash
CC=/usr/local/llvm-obfuscator/bin/clang

CFLAGS=-Xclang -load -Xclang /usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so \
       -mllvm -fla \
       -mllvm -sub \
       -mllvm -bcf \
       -mllvm -split \
       -O3 \
       -fno-builtin \
       -Wl,-s \
       -fvisibility=hidden \
       -fomit-frame-pointer \
       -mspeculative-load-hardening \
       -DNORMALUNIX -DLINUX
```

---

## Potential Issues & Solutions

### Issue 1: X11 Dependencies

**Problem:** The DOOM source requires X11 libraries for the video/input system.

**Solution:** The Docker container needs X11 development libraries:
```bash
apt-get install libx11-dev libxext-dev
```

### Issue 2: Makefile Needs Modification for OAAS

**Problem:** The original Makefile hardcodes `gcc` and doesn't respect environment `CC`/`CFLAGS`.

**Potential Fix:** The Makefile already uses `$(CC)` and `$(CFLAGS)` variables, but the default values may need to be overridden via environment.

**Workaround:** Set environment variables before running make:
```bash
export CC=/usr/local/llvm-obfuscator/bin/clang
export CFLAGS="$CFLAGS -DNORMALUNIX -DLINUX"
make -e  # Use environment variables
```

### Issue 3: Output Directory

**Problem:** The Makefile outputs to `linux/` subdirectory which must exist.

**Solution:** Ensure the `linux/` directory is created before build:
```bash
mkdir -p linuxdoom-1.10/linux
```

---

## Verification Commands

After successful obfuscation, verify with:

```bash
# Check binary exists and is executable
file /path/to/linuxxdoom

# Check symbols are stripped
nm /path/to/linuxxdoom
# Expected: "nm: no symbols"

# Check for exposed strings (game strings may still be visible)
strings /path/to/linuxxdoom | grep -i "doom"

# Binary size comparison
ls -la /path/to/linuxxdoom
```

---

## Comparison with CURL Obfuscation

| Aspect | CURL | DOOM |
|--------|------|------|
| Build System | CMake | Make |
| Source Files | 684 | 62 |
| Language | C | C |
| Dependencies | OpenSSL, zlib, etc. | X11, Xext |
| Expected Build Time | ~19 minutes | ~5-10 minutes |
| Binary Type | CLI tool | GUI application |
| Platform | Cross-platform | Linux-only |

### Key Differences

1. **Build System:** DOOM uses Make (simpler), CURL uses CMake
   - No CMake `try_compile` issues to worry about
   - May need to ensure Make respects environment CC/CFLAGS

2. **Dependencies:** DOOM requires X11 graphics libraries
   - Container must have X11 dev packages installed

3. **Simpler Build:** DOOM has fewer files and simpler dependencies
   - Faster build time expected
   - Less likely to have compatibility issues

---

## Running the Obfuscated Binary

### Requirements

1. **WAD File:** DOOM requires a game data file (doom1.wad for shareware)
2. **X11 Display:** Must have X server running
3. **Libraries:** libX11, libXext

### Execution

```bash
# On Linux with X11
export DISPLAY=:0
./linuxxdoom -iwad /path/to/doom1.wad

# In Docker (requires X11 forwarding)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/doom1.wad:/doom1.wad \
  your-image ./linuxxdoom -iwad /doom1.wad
```

---

## Notes

- The original DOOM source is from 1997 (released by John Carmack)
- Licensed under GPL v2
- This is the "linuxdoom" port, X11-based
- For more portable testing, consider Chocolate Doom or another modern source port

---

## Alternative: Modern DOOM Source Ports

If the original linuxdoom is problematic, consider these alternatives:

1. **Chocolate Doom** (https://github.com/chocolate-doom/chocolate-doom)
   - SDL-based, more portable
   - CMake build system
   - Better maintained

2. **PrBoom+** (https://github.com/coelckers/prboom-plus)
   - Modern, cross-platform
   - CMake build system

3. **Crispy Doom** (https://github.com/fabiangreffrath/crispy-doom)
   - Fork of Chocolate Doom
   - Autotools build system

---

## Test Results (December 3, 2025)

### Original DOOM (id-Software/DOOM)

**Status:** FAILED - Source code incompatible with modern clang

**Issues Encountered:**

1. **errnos.h typo** - `i_video.c:49` includes `<errnos.h>` instead of `<errno.h>`
   - Fixed with symlink: `ln -sf /usr/include/errno.h /usr/include/errnos.h`

2. **Control Flow Flattening segfault** - The flattening pass crashes on `i_video.c`
   - Workaround: Disable flattening, use only substitution/boguscf/split

3. **m_misc.c compile error** - Modern clang rejects non-constant static initializers
   ```
   m_misc.c:257:48: error: initializer element is not a compile-time constant
       {"sndserver", (int *) &sndserver_filename, (int) "sndserver"},
   ```
   - This is a 1997 C idiom that worked in old GCC but fails in modern compilers
   - **No workaround available** - requires source code modification

### Chocolate Doom (chocolate-doom/chocolate-doom)

**Status:** FAILED - Autotools configure incompatibility

**Issues Encountered:**

1. **configure fails** - The `clang-obfuscate` wrapper doesn't pass configure's compiler test
   ```
   configure: error: C compiler cannot create executables
   ```
   - The wrapper adds obfuscation flags that cause simple test compilations to fail
   - Autotools `./configure` expects a plain working compiler

**Root Cause:** The `clang-obfuscate` wrapper applies OLLVM passes even to trivial configure test programs, which can cause:
- Segfaults during pass execution
- Longer compile times causing timeouts
- Output that differs from expected (configure checks output)

### Recommendations

For DOOM-like projects, consider:

1. **Pre-patched source** - Fork and fix the source code issues before obfuscation
2. **CMake projects** - CMake-based projects work better with OAAS
3. **Simple Make projects** - Projects with simple Makefiles (no autotools) that just run `make` work best
4. **Modern source ports** - Try doomretro or other CMake-based ports instead

### Dependencies Installed

The following were installed in the container for these tests:
- `libx11-dev`, `libxext-dev` - X11 development libraries
- `libsdl2-dev`, `libsdl2-mixer-dev`, `libsdl2-net-dev` - SDL2 libraries
- `autoconf`, `automake`, `libtool` - Autotools (already present)

---

**Document Created:** December 3, 2025
**Last Updated:** December 4, 2025
**Status:** Testing Complete - Both approaches failed due to source/build system incompatibilities
