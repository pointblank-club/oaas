# LLVM Binary Obfuscator

Production-ready LLVM binary obfuscation tool with 4-layer protection.

## Features

- **Layer 1: Symbol Obfuscation** - Cryptographic symbol renaming (SHA256/Blake2b)
- **Layer 2: String Encryption** - Runtime XOR encryption with constructor-based decryption
- **Layer 3: OLLVM Passes** - 4 control-flow obfuscation passes
- **Layer 4: Compiler Flags** - 9 optimal flags (82.5/100 score)

## Installation

```bash
pip install llvm-obfuscator
```

## Quick Start

### Basic Obfuscation
```bash
llvm-obfuscate compile src/myapp.c \
  --output ./dist \
  --level 3 \
  --string-encryption
```

### Maximum Security
```bash
llvm-obfuscate compile src/myapp.c \
  --output ./dist \
  --level 4 \
  --enable-symbol-obfuscation \
  --enable-flattening \
  --enable-bogus-cf \
  --string-encryption
```

### Using Configuration File
```bash
llvm-obfuscate compile src/myapp.c --config-file config.yaml
```

## Layer Details

### Layer 1: Symbol Obfuscation
Renames all function and variable names using cryptographic hashes.

**Options:**
- `--enable-symbol-obfuscation`
- `--symbol-algorithm sha256|blake2b|siphash`
- `--symbol-hash-length 8-32`

**Result:** 90%+ symbol reduction

### Layer 1: Compiler Flags
Applies 9 optimal compiler flags for obfuscation.

**Included flags:**
- Link-time optimization (`-flto`)
- Symbol hiding (`-fvisibility=hidden`)
- Frame pointer removal (`-fomit-frame-pointer`)
- Speculative load hardening
- And more...

**Result:** 82.5/100 obfuscation score

### Layer 2: OLLVM Passes
Four powerful control-flow obfuscation techniques.

**Options:**
- `--enable-flattening` - Flatten control flow into switch statements
- `--enable-substitution` - Replace instructions with equivalent complex sequences
- `--enable-bogus-cf` - Insert fake control flow
- `--enable-split` - Split basic blocks

**Result:** 20-30x harder to reverse engineer

**Note:** Plugins auto-detected for your platform (macOS, Linux, Windows)

#### OLLVM Wrapper Scripts (Server Pipeline)

For complex projects using CMake or Autotools, the server uses wrapper scripts that apply OLLVM passes via the `opt` tool:

```
Source.c
    → Step 1: clang -emit-llvm -c → Source.bc (LLVM bitcode)
    → Step 2: opt --passes=substitution,boguscf,split → Source_obf.bc
    → Step 3: clang -c Source_obf.bc → Source.o (object file)
```

**Pipeline Diagram:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    clang-obfuscate wrapper                      │
├─────────────────────────────────────────────────────────────────┤
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
│                               │  Step 2: Apply OLLVM passes │  │
│                               │  Step 3: Bitcode → Object   │  │
│                               └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLVM_PASSES` | Comma-separated passes | `substitution,boguscf,split` |
| `OLLVM_PLUGIN` | Path to LLVMObfuscationPlugin.so | `/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so` |
| `OLLVM_DEBUG` | Enable verbose logging | `1` |
| `OLLVM_CFLAGS` | Additional compiler flags | `-O3 -fno-builtin` |

**Recommended Passes for Complex Projects:**
- ✅ `substitution` - Instruction substitution (stable)
- ✅ `boguscf` - Bogus control flow (stable)
- ✅ `split` - Split basic blocks (stable)
- ⚠️ `flattening` - **AVOID** for complex projects (causes segfaults on code with complex switch statements)

### Layer 3: String Encryption
Encrypts all string literals at compile time.

**Options:**
- `--string-encryption`

**Result:** 100% string hiding (secrets not visible in `strings` output)

## Testing with Jotai Benchmarks

Test obfuscation effectiveness on real-world C code using the [Jotai benchmark collection](https://github.com/lac-dcc/jotai-benchmarks):

```bash
# Run Jotai benchmarks through obfuscator
python -m cli.obfuscate jotai --limit 10 --level 3

# Full obfuscation test
python -m cli.obfuscate jotai \
    --limit 20 \
    --level 4 \
    --enable-symbol-obfuscation \
    --string-encryption \
    --enable-flattening
```

**Quick test:**
```bash
cd cmd/llvm-obfuscator
python3 test_jotai_simple.py
```

See [JOTAI_BENCHMARKS.md](JOTAI_BENCHMARKS.md) for detailed documentation.

## Supported Platforms

✅ **Bundled plugins for:**
- macOS arm64 (Apple Silicon)
- macOS x86_64 (Intel)
- Linux x86_64
- Windows x86_64

## Configuration File Example

```yaml
obfuscation:
  level: 4
  platform: linux

  passes:
    flattening: true
    substitution: true
    bogus_control_flow: true
    split: true

  advanced:
    string_encryption: true
    symbol_obfuscation:
      enabled: true
      algorithm: sha256

  output:
    directory: "./obfuscated"
    report_formats:
      - json
      - html
```

## Commands

### Compile
Obfuscate a source file:
```bash
llvm-obfuscate compile <source> [OPTIONS]
```

### Analyze
Analyze obfuscation metrics:
```bash
llvm-obfuscate analyze <binary> --output report.json
```

### Compare
Compare original vs obfuscated:
```bash
llvm-obfuscate compare <original> <obfuscated>
```

### Batch
Process multiple files:
```bash
llvm-obfuscate batch config.yaml
```

## Options Reference

```
--level <1-5>                 Obfuscation level
--platform linux|darwin       Target platform
--string-encryption           Enable string encryption
--enable-symbol-obfuscation   Enable symbol obfuscation
--enable-flattening           Enable control flow flattening
--enable-substitution         Enable instruction substitution
--enable-bogus-cf             Enable bogus control flow
--enable-split                Enable basic block splitting
--cycles <1-5>                Number of obfuscation cycles
--fake-loops <0-50>           Insert fake loops
--report-formats json,html    Output formats
--config-file <path>          Load from config file
```

## Performance Impact

| Configuration | Overhead | Security Gain |
|---------------|----------|---------------|
| Layer 1+2+3   | ~10%     | 10-15x        |
| Layer 1+2     | ~15-20%  | 20-30x        |
| All layers    | ~25-30%  | 50x+          |

## Testing

Run comprehensive tests:
```bash
cd /path/to/llvm
./test_all_layers.sh
```

## Documentation

- Full docs: [OBFUSCATION_COMPLETE.md](../../OBFUSCATION_COMPLETE.md)
- OLLVM fix: [OLLVM_INTEGRATION_FIX.md](../../OLLVM_INTEGRATION_FIX.md)
- CLI guide: [CLAUDE.md](../../CLAUDE.md)
- Curl test: [CURL_OBFUSCATION_TEST.md](../../docs/CURL_OBFUSCATION_TEST.md) - Real-world curl build with OLLVM

## License

Apache License 2.0

Includes LLVM components under Apache License 2.0 with LLVM Exceptions.

## Support

- Issues: https://github.com/yourorg/llvm-obfuscator/issues
- Docs: https://github.com/yourorg/llvm-obfuscator/docs

## Citation

If you use this tool in research, please cite:

```bibtex
@software{llvm-obfuscator,
  title = {LLVM Binary Obfuscator},
  author = {LLVM Obfuscation Team},
  year = {2025},
  url = {https://github.com/yourorg/llvm-obfuscator}
}
```

---

**Maintained by:** LLVM Obfuscation Team
**Version:** 1.0.0
