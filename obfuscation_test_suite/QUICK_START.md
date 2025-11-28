# Quick Start Guide - OLLVM Obfuscation Test Suite

## Usage

Once you have baseline and obfuscated binaries, run the test suite:

```bash
cd /home/incharaj/oaas/obfuscation_test_suite
./run_tests.sh <path_to_baseline_binary> <path_to_obfuscated_binary> [program_name]
```

### Example:
```bash
./run_tests.sh ./my_baseline ./my_obfuscated my_program
```

## Output

Results are automatically organized in:
```
results/
├── baseline/my_program/           # Baseline binary copy
├── obfuscated/my_program/         # Obfuscated binary copy
├── reports/my_program/            # Generated reports
│   ├── my_program_results.json    # Machine-readable results
│   ├── my_program_report.txt      # Comprehensive analysis
│   └── my_program_summary.txt     # Quick overview
└── metrics/my_program/            # Raw metrics data
```

## What Gets Tested

The suite evaluates across 11 dimensions:

1. **Functional Correctness** - Verifies behavior is preserved
2. **Control Flow Metrics** - CFG complexity analysis
3. **Binary Complexity** - Size and structural changes
4. **String Obfuscation** - Percentage of strings obfuscated
5. **Binary Properties** - Entropy and randomization
6. **Symbol Analysis** - Symbol count reduction
7. **Performance Overhead** - Execution time impact
8. **Debuggability** - Debug info impact
9. **Code Coverage** - Reachable code estimation
10. **Complexity Analysis** - Cyclomatic complexity
11. **Reverse Engineering Difficulty** - Composite difficulty score (0-100)

## Requirements

- Python 3.6+
- Standard UNIX tools: `objdump`, `nm`, `file`, `strings`, `readelf`
- Binaries must be executable ELF format (x86-64 or ARM)

## Notes

- Baseline and obfuscated binaries should have same architecture
- Functional tests assume binaries produce deterministic output
- Performance is measured with 5-second timeout
- All metrics are non-intrusive (no binary modification)

