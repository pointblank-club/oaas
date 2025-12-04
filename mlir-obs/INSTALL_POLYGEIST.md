## Polygeist Installation Guide

Complete step-by-step guide to install Polygeist and integrate it with the MLIR obfuscation system.

---

## Prerequisites

- **Ubuntu 20.04+** (or similar Linux distribution)
- **16GB RAM** minimum (32GB recommended for building LLVM)
- **50GB disk space** for LLVM + Polygeist builds
- **Build tools**: git, cmake, ninja, clang/gcc
- **Time**: ~2-3 hours for complete build

---

## Step 1: Install Build Dependencies

```bash
# Update package lists
sudo apt update

# Install build essentials
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    clang \
    lld \
    libssl-dev \
    zlib1g-dev \
    libedit-dev

# Verify installations
cmake --version    # Should be 3.13+
ninja --version
clang --version
```

---

## Step 2: Build LLVM/MLIR from Source

Polygeist requires LLVM/MLIR built with specific options.

### 2.1 Clone LLVM Project

```bash
# Create working directory
mkdir -p $HOME/llvm-workspace
cd $HOME/llvm-workspace

# Clone LLVM (this will take a while)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Checkout a stable version (optional, recommended for production)
# git checkout llvmorg-18.1.0

# For latest (bleeding edge):
git checkout main
```

### 2.2 Configure LLVM Build

```bash
mkdir build
cd build

# Configure with CMake
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
  -DLLVM_TARGETS_TO_BUILD="X86;ARM;AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install

# Notes:
# - LLVM_ENABLE_ASSERTIONS=ON helps catch bugs during development
# - LLVM_USE_LINKER=lld speeds up linking
# - Adjust CMAKE_INSTALL_PREFIX to your preferred location
```

### 2.3 Build LLVM (this takes 1-2 hours)

```bash
# Build with all CPU cores
ninja

# Optional: Run tests (takes another hour)
# ninja check-mlir

# Install to prefix location
ninja install
```

### 2.4 Add LLVM to PATH

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PATH=$HOME/llvm-install/bin:$PATH' >> ~/.bashrc
echo 'export LLVM_DIR=$HOME/llvm-workspace/llvm-project/build' >> ~/.bashrc
source ~/.bashrc

# Verify
mlir-opt --version
mlir-translate --version
```

---

## Step 3: Build Polygeist

### 3.1 Clone Polygeist

```bash
cd $HOME/llvm-workspace

# Clone with submodules
git clone --recursive https://github.com/llvm/Polygeist.git
cd Polygeist

# Or if already cloned without --recursive:
# git submodule update --init --recursive
```

### 3.2 Configure Polygeist Build

```bash
mkdir build
cd build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$HOME/llvm-workspace/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-workspace/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install

# Important: MLIR_DIR and LLVM_DIR must point to your LLVM build!
```

### 3.3 Build Polygeist

```bash
ninja

# Install
ninja install
```

### 3.4 Verify Polygeist Installation

```bash
# Add to PATH (if not using CMAKE_INSTALL_PREFIX)
export PATH=$HOME/llvm-workspace/Polygeist/build/bin:$PATH

# Test cgeist
cgeist --version

# Or try mlir-clang (alternative name)
mlir-clang --version

# Quick test
echo 'int main() { return 0; }' > test.c
cgeist test.c --function='*' -o test.mlir
cat test.mlir  # Should show MLIR output
```

---

## Step 4: Build Obfuscation System with Polygeist Support

### 4.1 Navigate to Project

```bash
cd /path/to/oaas/mlir-obs
```

### 4.2 Build with Polygeist Detection

```bash
# The build script will auto-detect Polygeist if in PATH
./build.sh

# Or specify manually:
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$HOME/llvm-workspace/llvm-project/build/lib/cmake/mlir \
  -DPOLYGEIST_DIR=$HOME/llvm-workspace/Polygeist/build

make -j$(nproc)
```

### 4.3 Verify Integration

```bash
# Check build output
./build.sh

# Should see:
# ✓ Polygeist support ENABLED
# You can now use: cgeist source.c -o source.mlir
```

---

## Step 5: Run Tests

### 5.1 Basic Functionality Test

```bash
cd /path/to/oaas/mlir-obs

# Run integration tests
./test-polygeist-integration.sh

# Expected output:
# ✓ Passed: XX / YY
# ✗ Failed: 0 / YY
```

### 5.2 Example Workflow Test

```bash
# Test with provided example
./polygeist-pipeline.sh examples/simple_auth.c test_output

# Verify binary was created
ls -lh test_output
./test_output mypassword
```

### 5.3 Compare Pipelines

```bash
# Compare traditional vs Polygeist
./compare-pipelines.sh examples/simple_auth.c

# Should show:
# - Traditional: LLVM dialect only
# - Polygeist: func, scf, memref, affine dialects
```

---

## Step 6: Configure Your Environment

### 6.1 Permanent Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# LLVM/MLIR
export LLVM_DIR=$HOME/llvm-workspace/llvm-project/build
export MLIR_DIR=$LLVM_DIR/lib/cmake/mlir
export PATH=$HOME/llvm-install/bin:$PATH

# Polygeist
export POLYGEIST_DIR=$HOME/llvm-workspace/Polygeist/build
export PATH=$POLYGEIST_DIR/bin:$PATH

# Obfuscation system
export PATH=$HOME/path/to/oaas/mlir-obs/build/tools:$PATH

# Reload
source ~/.bashrc
```

### 6.2 Verify Complete Setup

```bash
# Check all tools are available
which mlir-opt        # MLIR optimizer
which mlir-translate  # MLIR to LLVM IR converter
which cgeist          # Polygeist C/C++ frontend
which mlir-obfuscate  # Our obfuscation tool (if built)

# Check library
find $HOME/path/to/oaas/mlir-obs/build -name "*MLIRObfuscation*"
```

---

## Troubleshooting

### Issue 1: CMake can't find MLIR

**Error:**
```
Could not find a package configuration file provided by "MLIR"
```

**Solution:**
```bash
# Make sure MLIR_DIR is set correctly
export MLIR_DIR=$HOME/llvm-workspace/llvm-project/build/lib/cmake/mlir

# Verify the path exists
ls $MLIR_DIR/MLIRConfig.cmake
```

### Issue 2: Polygeist build fails with "dialect not found"

**Error:**
```
error: 'MLIRSCFDialect' was not found
```

**Solution:**
```bash
# Rebuild LLVM with MLIR enabled
cd $HOME/llvm-workspace/llvm-project/build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="mlir;clang"
ninja
```

### Issue 3: Version mismatch errors

**Error:**
```
error: MLIR version mismatch
```

**Solution:**
Rebuild both LLVM and Polygeist from scratch:

```bash
# Clean builds
rm -rf $HOME/llvm-workspace/llvm-project/build
rm -rf $HOME/llvm-workspace/Polygeist/build

# Rebuild LLVM (Step 2)
# Then rebuild Polygeist (Step 3)
# Then rebuild obfuscation system (Step 4)
```

### Issue 4: Out of memory during build

**Error:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution:**
```bash
# Reduce parallel jobs
ninja -j2  # Instead of ninja (which uses all cores)

# Or increase swap space
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue 5: cgeist produces empty MLIR

**Error:**
```
module {
}
```

**Solution:**
```bash
# Make sure to specify --function='*'
cgeist source.c --function='*' -o output.mlir

# For specific functions:
cgeist source.c --function='main' --function='helper' -o output.mlir
```

---

## Quick Start After Installation

Once everything is installed, here's the typical workflow:

```bash
# 1. Write C code
cat > mycode.c << 'EOF'
int add(int a, int b) {
    return a + b;
}
int main() {
    return add(2, 3);
}
EOF

# 2. Compile to MLIR with Polygeist
cgeist mycode.c --function='*' -o mycode.mlir

# 3. Apply obfuscation
mlir-opt mycode.mlir \
  --load-pass-plugin=./build/lib/MLIRObfuscation.so \
  --pass-pipeline='builtin.module(symbol-obfuscate,string-encrypt)' \
  -o mycode_obf.mlir

# 4. Lower to LLVM dialect
mlir-opt mycode_obf.mlir \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
  -o mycode_llvm.mlir

# 5. Generate LLVM IR
mlir-translate --mlir-to-llvmir mycode_llvm.mlir -o mycode.ll

# 6. Compile to binary
clang mycode.ll -o mycode

# 7. Run
./mycode
echo $?  # Should print 5
```

Or use the automated pipeline:

```bash
./polygeist-pipeline.sh mycode.c mycode_obfuscated
./mycode_obfuscated
```

---

## Performance Expectations

| Task                  | Time (Typical) |
|-----------------------|----------------|
| LLVM build            | 60-90 minutes  |
| Polygeist build       | 10-15 minutes  |
| Obfuscator build      | 2-5 minutes    |
| cgeist (simple file)  | 1-3 seconds    |
| Full pipeline         | 3-10 seconds   |

---

## Docker Alternative (Advanced)

If you prefer Docker:

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt install -y \
    build-essential cmake ninja-build git \
    clang lld python3

# Build LLVM
WORKDIR /build
RUN git clone https://github.com/llvm/llvm-project.git && \
    cd llvm-project && mkdir build && cd build && \
    cmake -G Ninja ../llvm \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DLLVM_TARGETS_TO_BUILD="X86" && \
    ninja

# Build Polygeist
RUN git clone --recursive https://github.com/llvm/Polygeist.git && \
    cd Polygeist && mkdir build && cd build && \
    cmake -G Ninja .. \
      -DMLIR_DIR=/build/llvm-project/build/lib/cmake/mlir && \
    ninja

ENV PATH="/build/Polygeist/build/bin:${PATH}"
```

Build and run:

```bash
docker build -t polygeist:latest .
docker run -it -v $(pwd):/work polygeist:latest
```

---

## Next Steps

After successful installation:

1. Read [POLYGEIST_INTEGRATION.md](./POLYGEIST_INTEGRATION.md) for usage guide
2. Run integration tests: `./test-polygeist-integration.sh`
3. Try examples: `./polygeist-pipeline.sh examples/simple_auth.c output`
4. Explore pass options: `mlir-opt --help | grep obfuscate`

---

## Support

For issues:
1. Check this troubleshooting section
2. Run `./test-polygeist-integration.sh` to diagnose
3. Review build logs carefully
4. Ensure LLVM and Polygeist are the same version

For Polygeist-specific issues:
- [Polygeist GitHub Issues](https://github.com/llvm/Polygeist/issues)
- [Polygeist Documentation](https://polygeist.llvm.org/)

---

**Installation tested on:**
- Ubuntu 20.04, 22.04
- LLVM 18.x, 19.x
- Polygeist main branch (2024-2025)

**Last updated:** 2025-11-29
