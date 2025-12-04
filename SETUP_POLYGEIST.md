# Polygeist Setup Guide for OAAS VM

This guide explains how to build and install Polygeist in your VM to enable full end-to-end testing.

## Why You Need Polygeist

Your MLIR obfuscation passes support **two pipelines**:

1. **Traditional Pipeline** (works WITHOUT Polygeist):
   - C → LLVM IR → MLIR (LLVM dialect only)
   - Limited to low-level LLVM operations

2. **Polygeist Pipeline** (requires Polygeist) ⭐:
   - C → Polygeist → MLIR (func, scf, memref, affine dialects)
   - Works on high-level constructs before lowering
   - Better obfuscation opportunities
   - **This is what you've integrated!**

## Quick Setup

### Automated Setup (Recommended)

```bash
./setup_polygeist.sh
```

This script will:
1. ✅ Check prerequisites (git, cmake, ninja, LLVM/MLIR)
2. ✅ Clone Polygeist repository
3. ✅ Configure build with CMake
4. ✅ Build Polygeist (10-30 minutes)
5. ✅ Verify installation
6. ✅ Create environment setup script

**Expected time:** 15-45 minutes (depending on VM specs)

### After Setup

```bash
# Load Polygeist into your environment
source ./polygeist_env.sh

# Verify installation
which cgeist
# Should output: /path/to/oaas/polygeist/build/bin/cgeist

# Run full tests
./test_polygeist_e2e.sh
```

---

## Manual Setup

If the automated script fails, here's the manual process:

### 1. Install Prerequisites

```bash
sudo apt update
sudo apt install -y \
    git \
    cmake \
    ninja-build \
    clang-19 \
    llvm-19-dev \
    mlir-19-tools \
    libmlir-19-dev
```

### 2. Clone Polygeist

```bash
git clone --recursive https://github.com/llvm/Polygeist.git polygeist
cd polygeist
```

### 3. Configure with CMake

```bash
mkdir build && cd build

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
    ..
```

**Note:** Adjust `MLIR_DIR` path based on your LLVM installation:
- Ubuntu/Debian: `/usr/lib/llvm-19/lib/cmake/mlir`
- Custom install: `/path/to/llvm/lib/cmake/mlir`

### 4. Build

```bash
ninja -j $(nproc)
```

This will take 10-30 minutes depending on your CPU.

### 5. Verify

```bash
./bin/cgeist --version

# Test basic functionality
echo 'int add(int a, int b) { return a + b; }' > test.c
./bin/cgeist test.c --function='*' -o test.mlir
cat test.mlir  # Should show MLIR output
```

### 6. Add to PATH

```bash
export PATH="$PWD/bin:$PATH"

# Or add permanently to ~/.bashrc:
echo 'export PATH="/path/to/polygeist/build/bin:$PATH"' >> ~/.bashrc
```

---

## Verification

After installation, verify everything works:

### Check Polygeist

```bash
# Should show path to cgeist
which cgeist

# Should show version info
cgeist --version
```

### Run Basic Test

```bash
# Create test file
cat > simple.c << 'EOF'
#include <stdio.h>
int main() {
    printf("Hello\n");
    return 0;
}
EOF

# Generate MLIR
cgeist simple.c --function='*' -o simple.mlir

# Verify output
grep "func.func" simple.mlir  # Should show function definitions
```

### Run Full Test Suite

```bash
# This should now pass ALL tests (not skip Polygeist ones)
./test_polygeist_e2e.sh
```

Expected output:
```
[6/7] Polygeist Pipeline (C -> func/scf -> Obfuscation -> Binary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PASS: C to Polygeist MLIR generation
✅ PASS: func dialect present
✅ PASS: SCF/Affine dialect present
...
```

---

## What Gets Installed

After successful setup:

```
oaas/
├── polygeist/                    # Polygeist repository
│   ├── build/
│   │   ├── bin/
│   │   │   ├── cgeist           # Main Polygeist tool
│   │   │   ├── mlir-clang       # Alternative name
│   │   │   └── ...
│   │   └── lib/                 # Libraries
│   └── ...
├── polygeist_env.sh             # Environment setup script
└── setup_polygeist.sh           # This setup script
```

**Disk space required:** ~2-5 GB

---

## Troubleshooting

### Issue: "CMake cannot find MLIR"

**Solution:** Specify MLIR_DIR explicitly:

```bash
# Find MLIR installation
find /usr -name "MLIRConfig.cmake" 2>/dev/null

# Use the directory containing MLIRConfig.cmake
cmake -G Ninja -DMLIR_DIR=/path/to/mlir/cmake ..
```

### Issue: "LLVM version mismatch"

**Error:** `Polygeist requires LLVM 19, found LLVM 18`

**Solution:** Install LLVM 19:
```bash
sudo apt install llvm-19-dev mlir-19-tools libmlir-19-dev
```

### Issue: "Out of memory during build"

**Solution:** Reduce parallel jobs:
```bash
ninja -j 2  # Instead of -j $(nproc)
```

Or add swap space:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: "cgeist not found after build"

**Solution:** Check if binary exists:
```bash
find polygeist/build -name "cgeist" -o -name "mlir-clang"
```

If found, add to PATH:
```bash
export PATH="/path/to/polygeist/build/bin:$PATH"
```

### Issue: "Tests still skip Polygeist"

**Reason:** Polygeist not in PATH

**Solution:**
```bash
source ./polygeist_env.sh
./test_polygeist_e2e.sh
```

---

## Testing Without Polygeist

If you can't install Polygeist (e.g., resource constraints), you can still test:

### What Works Without Polygeist:
- ✅ MLIR library build
- ✅ Symbol obfuscation on LLVM dialect
- ✅ String encryption
- ✅ Traditional pipeline (C → LLVM IR → MLIR)
- ✅ Binary generation

### What Requires Polygeist:
- ❌ High-level dialect support (func, scf, memref, affine)
- ❌ SCF control-flow obfuscation
- ❌ Full Polygeist pipeline testing

### Run Tests Without Polygeist:
```bash
./test_polygeist_e2e.sh
# Will skip Polygeist tests and show:
# ⊘ SKIP: Polygeist pipeline (Polygeist not installed)
```

---

## Performance Expectations

### Build Time
- **Fast VM (8 cores, 16GB RAM):** 10-15 minutes
- **Medium VM (4 cores, 8GB RAM):** 20-30 minutes
- **Slow VM (2 cores, 4GB RAM):** 45-60 minutes

### Disk Space
- **Source:** ~500 MB
- **Build:** ~2-4 GB
- **Total:** ~2-5 GB

### Memory Usage
- **Build:** 4-8 GB RAM recommended
- **Runtime:** Minimal (<100 MB)

---

## Alternative: Use Docker

If VM resources are limited, consider using Docker:

```bash
# Use pre-built LLVM/MLIR container
docker run -it --rm \
    -v $(pwd):/workspace \
    silkeh/clang:19 \
    bash

# Inside container, install Polygeist
cd /workspace
./setup_polygeist.sh
```

---

## Next Steps After Installation

1. **Verify installation:**
   ```bash
   source ./polygeist_env.sh
   which cgeist
   ```

2. **Run full tests:**
   ```bash
   ./test_polygeist_e2e.sh
   ```

3. **Test with example:**
   ```bash
   ./mlir-obs/polygeist-pipeline.sh mlir-obs/examples/simple_auth.c test_output
   ./test_output
   ```

4. **Integrate into build system:**
   - Add Polygeist to CI/CD pipeline
   - Update build scripts to use `polygeist_env.sh`

---

## Summary

**Do you need Polygeist?**
- **For full testing:** Yes, required
- **For production use:** Depends on pipeline choice
- **For basic MLIR passes:** No, not required

**Quick install:**
```bash
./setup_polygeist.sh
source ./polygeist_env.sh
./test_polygeist_e2e.sh
```

**Expected result:**
- All tests pass (no skipped Polygeist tests)
- `cgeist` available in PATH
- Full pipeline functional
