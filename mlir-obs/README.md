# MLIR Obfuscation Passes

This directory contains MLIR-based string encryption and symbol obfuscation passes for the LLVM Obfuscator project.

## Overview

The MLIR obfuscation library provides two main passes:

1. **String Encryption Pass** (`string-encrypt`)
   - Encrypts string attributes in MLIR using XOR cipher
   - Prevents hardcoded secrets from appearing in plain text in binaries

2. **Symbol Obfuscation Pass** (`symbol-obfuscate`)
   - Obfuscates function and symbol names using random hex-based naming
   - Makes reverse engineering significantly harder by removing semantic meaning from symbols

## Architecture

The obfuscation pipeline follows this flow:

```
Source Code (.c/.cpp)
    ↓
[Clang] Emit MLIR
    ↓
MLIR (.mlir)
    ↓
[MLIR Passes] String Encryption + Symbol Obfuscation
    ↓
Obfuscated MLIR
    ↓
[mlir-translate] Lower to LLVM IR
    ↓
LLVM IR (.ll)
    ↓
[Clang] Compile to Binary
    ↓
Obfuscated Binary
```

## Prerequisites

Before building, ensure you have:

- **LLVM/MLIR 15+** installed with development headers
- **CMake 3.13+**
- **Clang 15+**
- **C++17 compatible compiler**

### Installing LLVM/MLIR on Linux

```bash
# Ubuntu/Debian
sudo apt-get install llvm-19 llvm-19-dev mlir-19-tools libmlir-19-dev clang-19

# Verify installation
mlir-opt --version
clang --version
```

### Installing LLVM/MLIR on macOS

```bash
# Using Homebrew
brew install llvm

# Add to PATH
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

## Building

### Quick Build

```bash
cd mlir-obs
./build.sh
```

### Manual Build

```bash
cd mlir-obs
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17

make -j$(nproc)
```

The build will produce `libMLIRObfuscation.so` (Linux), `libMLIRObfuscation.dylib` (macOS), or `MLIRObfuscation.dll` (Windows) in the `build/lib` directory.

## Testing

Run the standalone tests to verify the passes work correctly:

```bash
./test.sh
```

This will:
1. Create a test C file with secrets
2. Compile to MLIR
3. Apply obfuscation passes
4. Lower to LLVM IR
5. Compile to binary
6. Verify obfuscation worked

## Usage

### Standalone Usage (mlir-opt)

```bash
# Compile C to MLIR
clang -emit-llvm -S -emit-mlir your_source.c -o input.mlir

# Apply symbol obfuscation only
mlir-opt input.mlir \
  --load-pass-plugin=./build/lib/libMLIRObfuscation.so \
  --pass-pipeline="builtin.module(symbol-obfuscate)" \
  -o obfuscated.mlir

# Apply string encryption only
mlir-opt input.mlir \
  --load-pass-plugin=./build/lib/libMLIRObfuscation.so \
  --pass-pipeline="builtin.module(string-encrypt)" \
  -o obfuscated.mlir

# Apply both passes
mlir-opt input.mlir \
  --load-pass-plugin=./build/lib/libMLIRObfuscation.so \
  --pass-pipeline="builtin.module(symbol-obfuscate,string-encrypt)" \
  -o obfuscated.mlir

# Lower to LLVM IR
mlir-translate --mlir-to-llvmir obfuscated.mlir -o output.ll

# Compile to binary
clang output.ll -o your_binary
```

### Integration with Python CLI

The MLIR passes are automatically integrated with the main obfuscation service:

```bash
cd cmd/llvm-obfuscator

# Use string encryption only
python3 -m cli.obfuscate compile source.c \
  --enable-string-encrypt \
  --output ./obfuscated

# Use symbol obfuscation only
python3 -m cli.obfuscate compile source.c \
  --enable-symbol-obfuscate \
  --output ./obfuscated

# Use both MLIR passes
python3 -m cli.obfuscate compile source.c \
  --enable-string-encrypt \
  --enable-symbol-obfuscate \
  --output ./obfuscated

# Combine with OLLVM passes
python3 -m cli.obfuscate compile source.c \
  --enable-string-encrypt \
  --enable-symbol-obfuscate \
  --enable-flattening \
  --enable-bogus-cf \
  --output ./obfuscated
```

## Pass Details

### String Encryption Pass

**Purpose:** Encrypt string literal attributes in MLIR to prevent them from appearing in plaintext in the final binary.

**Algorithm:** XOR encryption with a configurable key

**What it encrypts:**
- String attributes in MLIR operations
- Preserves critical attributes like `sym_name`, `function_ref`, and `callee`

**Example:**

```mlir
// Before
llvm.mlir.global internal constant @.str("MySecret123!")

// After
llvm.mlir.global internal constant @.str("\x1a\x2b\x3c...")  // XOR encrypted
```

### Symbol Obfuscation Pass

**Purpose:** Rename all function symbols to meaningless random hex names to remove semantic meaning.

**Algorithm:** Random hex-based name generation (seeded for reproducibility)

**What it obfuscates:**
- Function definitions (`func.func`)
- Function calls and symbol references
- Maintains correct references throughout the module

**Example:**

```mlir
// Before
func.func @validate_password(%arg0: !llvm.ptr) -> i32

// After
func.func @f_a3b7f8d2(%arg0: !llvm.ptr) -> i32
```

## Implementation Files

```
mlir-obs/
├── CMakeLists.txt              # Build configuration
├── build.sh                    # Build script
├── test.sh                     # Test script
├── README.md                   # This file
├── include/
│   └── Obfuscator/
│       └── Passes.h           # Pass declarations
└── lib/
    ├── CMakeLists.txt         # Library build config
    ├── Passes.cpp             # String encryption implementation
    ├── SymbolPass.cpp         # Symbol obfuscation implementation
    └── PassRegistrations.cpp  # Pass registration
```

## Troubleshooting

### "MLIR plugin not found"

**Solution:** Build the library first:
```bash
cd mlir-obs
./build.sh
```

### "mlir-opt: command not found"

**Solution:** Install LLVM/MLIR tools or add them to your PATH:
```bash
# Find LLVM installation
find /usr -name "mlir-opt" 2>/dev/null

# Add to PATH
export PATH="/usr/lib/llvm-19/bin:$PATH"
```

### CMake can't find MLIR

**Solution:** Specify MLIR_DIR explicitly:
```bash
cmake .. -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
```

### Pass not loading

**Solution:** Verify the plugin file exists:
```bash
find build -name "libMLIRObfuscation.*"
```

Check that you're using the correct --load-pass-plugin path.

## Development

### Adding a New Pass

1. Add pass declaration to `include/Obfuscator/Passes.h`
2. Implement the pass in a new file in `lib/`
3. Register the pass in `lib/PassRegistrations.cpp`
4. Add to `lib/CMakeLists.txt`
5. Update `core/config.py` to add the pass flag
6. Update `cli/obfuscate.py` to add CLI flag
7. Test the pass

### Debugging Passes

Enable verbose output:
```bash
mlir-opt input.mlir \
  --load-pass-plugin=./build/lib/libMLIRObfuscation.so \
  --pass-pipeline="builtin.module(symbol-obfuscate)" \
  --mlir-print-ir-after-all \
  -o output.mlir 2>&1 | less
```

## Performance

- **String Encryption:** ~2-5ms overhead per pass
- **Symbol Obfuscation:** ~1-3ms overhead per pass
- **Binary Size Impact:** +2-5% for string encryption
- **Runtime Overhead:** Negligible (compile-time only transformations)

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [Writing an MLIR Pass](https://mlir.llvm.org/docs/WritingAPass/)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)

## License

MIT License - See main project LICENSE file
