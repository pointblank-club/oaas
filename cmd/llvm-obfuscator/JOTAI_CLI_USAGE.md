# Jotai CLI Usage Guide

## Quick Start

### Prerequisites
```bash
cd cmd/llvm-obfuscator
pip install -r requirements.txt
# OR
pip install typer click rich pyyaml
```

### Basic Usage

```bash
# From cmd/llvm-obfuscator directory
python -m cli.obfuscate jotai [OPTIONS]
```

## Command Options

### Basic Options

```bash
# Run 10 benchmarks with default settings (level 3)
python -m cli.obfuscate jotai --limit 10

# Specify output directory
python -m cli.obfuscate jotai --limit 10 --output ./my_results

# Filter by category
python -m cli.obfuscate jotai --category anghaLeaves --limit 20
python -m cli.obfuscate jotai --category anghaMath --limit 20
```

### Obfuscation Level

```bash
# Level 1 (minimal)
python -m cli.obfuscate jotai --limit 10 --level 1

# Level 3 (default, recommended)
python -m cli.obfuscate jotai --limit 10 --level 3

# Level 5 (maximum)
python -m cli.obfuscate jotai --limit 10 --level 5
```

### Enable Obfuscation Features

```bash
# Symbol obfuscation
python -m cli.obfuscate jotai --limit 10 --enable-symbol-obfuscation

# String encryption
python -m cli.obfuscate jotai --limit 10 --string-encryption

# Control flow flattening
python -m cli.obfuscate jotai --limit 10 --enable-flattening

# Instruction substitution
python -m cli.obfuscate jotai --limit 10 --enable-substitution

# Bogus control flow
python -m cli.obfuscate jotai --limit 10 --enable-bogus-cf

# Basic block splitting
python -m cli.obfuscate jotai --limit 10 --enable-split
```

### Combined Examples

```bash
# Full obfuscation test
python -m cli.obfuscate jotai \
    --limit 20 \
    --level 4 \
    --enable-symbol-obfuscation \
    --string-encryption \
    --enable-flattening \
    --enable-bogus-cf \
    --output ./full_test_results

# Production-ready test (level 3 + string encryption)
python -m cli.obfuscate jotai \
    --limit 30 \
    --level 3 \
    --string-encryption \
    --enable-symbol-obfuscation \
    --output ./production_test

# Maximum security test
python -m cli.obfuscate jotai \
    --limit 15 \
    --level 5 \
    --enable-symbol-obfuscation \
    --string-encryption \
    --enable-flattening \
    --enable-substitution \
    --enable-bogus-cf \
    --enable-split \
    --output ./max_security_test
```

### Advanced Options

```bash
# Custom compiler flags
python -m cli.obfuscate jotai \
    --limit 10 \
    --custom-flags "-O2 -fno-inline"

# Custom LLVM pass plugin
python -m cli.obfuscate jotai \
    --limit 10 \
    --custom-pass-plugin /path/to/plugin.so

# Stop after N failures (default: 5)
python -m cli.obfuscate jotai \
    --limit 50 \
    --max-failures 10

# Custom cache directory
python -m cli.obfuscate jotai \
    --limit 10 \
    --cache-dir ~/my_jotai_cache
```

## Complete Flag Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output` | Path | `./jotai_results` | Output directory for results |
| `--category` | str | None | Filter: `anghaLeaves` or `anghaMath` |
| `--limit` | int | None | Maximum benchmarks to test |
| `--level` | int | 3 | Obfuscation level (1-5) |
| `--enable-flattening` | flag | False | Enable control flow flattening |
| `--enable-substitution` | flag | False | Enable instruction substitution |
| `--enable-bogus-cf` | flag | False | Enable bogus control flow |
| `--enable-split` | flag | False | Enable basic block splitting |
| `--string-encryption` | flag | False | Enable string encryption |
| `--enable-symbol-obfuscation` | flag | False | Enable symbol renaming |
| `--custom-flags` | str | None | Additional compiler flags |
| `--custom-pass-plugin` | Path | None | Custom LLVM pass plugin |
| `--max-failures` | int | 5 | Stop after N failures |
| `--cache-dir` | Path | None | Custom cache directory |

## Output

The command generates:

1. **Directory structure:**
   ```
   jotai_results/
   ├── jotai_report.json          # Summary report
   ├── benchmark1/
   │   ├── baseline/
   │   │   └── benchmark1_baseline
   │   └── obfuscated/
   │       └── benchmark1_obfuscated
   └── benchmark2/
       └── ...
   ```

2. **Console output:**
   ```
   Running Jotai benchmarks with obfuscation level 3...
   Output directory: ./jotai_results
   
   [1/10] Testing benchmark1...
   ✅ benchmark1: PASSED
   [2/10] Testing benchmark2...
   ⏭️  benchmark2: SKIPPED (compilation error)
   ...
   
   ============================================================
   Jotai Benchmark Results Summary
   ============================================================
   Total benchmarks: 10
   Compilation success: 8
   Obfuscation success: 8
   Functional tests passed: 7
   
   Full report: ./jotai_results/jotai_report.json
   ```

## Examples by Use Case

### Quick Validation (5-10 benchmarks)
```bash
python -m cli.obfuscate jotai --limit 5 --level 2
```

### Regression Testing (20-30 benchmarks)
```bash
python -m cli.obfuscate jotai --limit 25 --level 3 --string-encryption
```

### Comprehensive Testing (50+ benchmarks)
```bash
python -m cli.obfuscate jotai \
    --limit 50 \
    --level 4 \
    --enable-symbol-obfuscation \
    --string-encryption \
    --max-failures 10
```

### Research/Evaluation (all benchmarks)
```bash
# Test all anghaLeaves benchmarks
python -m cli.obfuscate jotai \
    --category anghaLeaves \
    --level 4 \
    --enable-symbol-obfuscation \
    --string-encryption \
    --enable-flattening \
    --output ./research_results
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'typer'"
```bash
cd cmd/llvm-obfuscator
pip install -r requirements.txt
```

### "No benchmarks found"
- First run downloads benchmarks automatically
- Check internet connection
- Manually download: `git clone https://github.com/lac-dcc/jotai-benchmarks.git`

### Many benchmarks failing
- This is normal - some Jotai benchmarks have compatibility issues
- Use `--max-failures` to control when to stop
- The integration automatically skips compilation errors

### Slow execution
- Reduce `--limit` for faster testing
- Use `--category` to filter benchmarks
- First run is slower (downloads benchmarks)

## Tips

1. **Start small**: Test with `--limit 5` first
2. **Use categories**: `anghaLeaves` benchmarks are simpler
3. **Check reports**: Look at `jotai_report.json` for details
4. **Combine flags**: Use multiple obfuscation features together
5. **Save output**: Use `--output` to organize results

