# Jotai Benchmarks Integration

This document describes how to use the Jotai benchmark collection with the LLVM Obfuscator to test obfuscation effectiveness on real-world C programs.

## What is Jotai?

[Jotai](https://github.com/lac-dcc/jotai-benchmarks) is a large collection of executable C benchmarks mined from open source repositories. Each benchmark consists of a single function written in C, plus a driver to run that function.

The Jotai collection includes:
- **anghaLeaves**: Benchmark functions that do not call any other function
- **anghaMath**: Benchmark functions that call functions from `math.h`

## Quick Start

### Prerequisites

First, install dependencies:

```bash
cd cmd/llvm-obfuscator
pip install -r requirements.txt
```

Or install in development mode:

```bash
cd cmd/llvm-obfuscator
pip install -e .
```

### Running Jotai Benchmarks

**Option 1: Run from the `cmd/llvm-obfuscator` directory:**

```bash
cd cmd/llvm-obfuscator

# Run 10 benchmarks with default obfuscation (level 3)
python -m cli.obfuscate jotai --limit 10

# Run with specific obfuscation settings
python -m cli.obfuscate jotai \
    --limit 20 \
    --level 4 \
    --enable-symbol-obfuscation \
    --string-encryption \
    --enable-flattening

# Run only anghaLeaves benchmarks
python -m cli.obfuscate jotai \
    --category anghaLeaves \
    --limit 50

# Run with custom output directory
python -m cli.obfuscate jotai \
    --output ./my_benchmark_results \
    --limit 30
```

**Option 2: If installed via pip, use the entry point:**

```bash
llvm-obfuscate jotai --limit 10 --level 3
```

**Option 3: Run with PYTHONPATH from project root:**

```bash
# From project root
cd /home/dhruv/Documents/Code/oaas
PYTHONPATH=cmd/llvm-obfuscator:$PYTHONPATH python -m cli.obfuscate jotai --limit 10
```

### Command Options

```
--output <path>              Output directory for benchmark results (default: ./jotai_results)
--category <name>            Benchmark category: anghaLeaves or anghaMath
--limit <n>                  Maximum number of benchmarks to test
--level <1-5>                Obfuscation level (default: 3)
--enable-flattening          Enable control flow flattening
--enable-substitution        Enable instruction substitution
--enable-bogus-cf            Enable bogus control flow
--enable-split               Enable basic block splitting
--string-encryption           Enable string encryption
--enable-symbol-obfuscation  Enable cryptographic symbol renaming
--custom-flags <flags>       Additional compiler flags
--custom-pass-plugin <path>  Path to custom LLVM pass plugin
--max-failures <n>           Stop after N consecutive failures (default: 5)
--cache-dir <path>           Directory to cache Jotai benchmarks
```

## How It Works

1. **Automatic Download**: The first time you run the command, it automatically downloads the Jotai benchmarks repository to `~/.cache/llvm-obfuscator/jotai-benchmarks`

2. **Benchmark Selection**: Benchmarks are selected from the specified category (or all if not specified)

3. **Baseline Compilation**: Each benchmark is compiled normally to create a baseline binary

4. **Obfuscation**: The benchmark is run through the obfuscator with your specified settings

5. **Functional Testing**: Both binaries are executed with various inputs to verify functional correctness

6. **Report Generation**: A JSON report is generated with detailed results

## Output Structure

```
jotai_results/
├── jotai_report.json              # Summary report
├── benchmark1/
│   ├── baseline/
│   │   └── benchmark1_baseline   # Baseline binary
│   └── obfuscated/
│       └── benchmark1_obfuscated # Obfuscated binary
├── benchmark2/
│   └── ...
└── ...
```

## Report Format

The `jotai_report.json` file contains:

```json
{
  "summary": {
    "total": 10,
    "compilation_success": 10,
    "obfuscation_success": 9,
    "functional_pass": 8
  },
  "results": [
    {
      "benchmark_name": "extr_example_Final",
      "category": "anghaLeaves",
      "source_file": "/path/to/benchmark.c",
      "baseline_binary": "/path/to/baseline",
      "obfuscated_binary": "/path/to/obfuscated",
      "compilation_success": true,
      "obfuscation_success": true,
      "functional_test_passed": true,
      "size_baseline": 16384,
      "size_obfuscated": 24576,
      "inputs_tested": 3,
      "inputs_passed": 3
    }
  ]
}
```

## Integration with Test Suite

You can integrate Jotai benchmarks with the existing obfuscation test suite:

```python
from core.jotai_benchmark import JotaiBenchmarkManager, BenchmarkCategory
from core import LLVMObfuscator, ObfuscationConfig, ObfuscationLevel

# Initialize manager
manager = JotaiBenchmarkManager()

# Get list of benchmarks
benchmarks = manager.list_benchmarks(
    category=BenchmarkCategory.ANGHALEAVES,
    limit=10
)

# Run benchmarks
obfuscator = LLVMObfuscator()
config = ObfuscationConfig(level=ObfuscationLevel(3))

results = manager.run_benchmark_suite(
    obfuscator=obfuscator,
    config=config,
    output_dir=Path("./results"),
    limit=10
)

# Generate report
manager.generate_report(results, Path("./report.json"))
```

## Use Cases

1. **Obfuscation Validation**: Test that obfuscation preserves functionality
2. **Performance Testing**: Measure overhead on real-world code
3. **Regression Testing**: Ensure new obfuscation features don't break existing code
4. **Research**: Evaluate obfuscation effectiveness across diverse code patterns

## Requirements

- Git (for downloading benchmarks)
- Clang (for compiling benchmarks)
- Python 3.10+

## Notes

- Benchmarks are cached locally after first download
- Use `--max-failures` to stop early if many benchmarks fail
- Functional tests verify that obfuscated binaries produce the same output as baseline
- Some benchmarks may fail due to unsupported C features or compilation issues

## References

- [Jotai Repository](https://github.com/lac-dcc/jotai-benchmarks)
- [Jotai Technical Report](https://raw.githubusercontent.com/lac-dcc/jotai-benchmarks/main/assets/doc/LaC_TechReport022022.pdf)
- [CompilerGym Integration](https://compilergym.com/llvm/api.html#compiler_gym.envs.llvm.datasets.JotaiBenchDataset)

