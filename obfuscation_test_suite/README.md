# OLLVM Obfuscation Test Suite

Comprehensive evaluation framework for OLLVM/LLVM-based code obfuscation systems. This test suite evaluates obfuscation effectiveness across 11 dimensions using industry-standard metrics.

## Directory Structure

```
obfuscation_test_suite/
├── obfuscation_test_suite.py    # Main test orchestrator
├── lib/                          # Core test modules
│   ├── test_utils.py            # Helper utilities
│   ├── test_metrics.py          # Metric computation
│   ├── test_report.py           # Report generation
│   ├── test_functional.py       # Functional testing
│   └── __init__.py
├── results/                      # Test results storage
│   ├── baseline/                # Baseline binaries (organized by program)
│   ├── obfuscated/              # Obfuscated binaries (organized by program)
│   ├── reports/                 # Generated reports (JSON/TXT)
│   └── metrics/                 # Raw metrics data
├── programs/                     # Source files for testing
└── scripts/                      # Helper scripts
```

## Usage

### Basic Test

```bash
python3 obfuscation_test_suite.py <baseline_binary> <obfuscated_binary>
```

### With Custom Options

```bash
python3 obfuscation_test_suite.py <baseline> <obfuscated> \
    -r ./results \
    -n program_name
```

### Arguments

- `baseline`: Path to baseline (unobfuscated) binary
- `obfuscated`: Path to obfuscated binary
- `-r, --results`: Results directory (default: ./results)
- `-n, --name`: Program name for reporting (default: "program")

## Test Dimensions

The suite evaluates obfuscation across 11 key dimensions:

### 1. **Functional Correctness**
- Verifies obfuscated binary maintains correct behavior
- Tests I/O equivalence, exit codes, and basic functionality

### 2. **Control Flow Metrics**
- Analyzes CFG complexity using objdump
- Measures indirect jumps, branch instructions, basic block estimates

### 3. **Binary Complexity**
- Size increase analysis
- Symbol reduction metrics
- Section count changes

### 4. **String Obfuscation**
- Measures string reduction percentage
- Tracks plaintext strings in binaries
- Samples obfuscated vs baseline strings

### 5. **Binary Properties**
- Shannon entropy calculation
- Size analysis (bytes and percentage)
- Entropy increase measurement

### 6. **Symbol Analysis**
- Symbol count comparison
- Symbol reduction verification
- Symbol table analysis

### 7. **Performance Overhead**
- Execution time comparison
- Overhead percentage calculation
- Acceptability threshold (< 100%)

### 8. **Debuggability Impact**
- Debug information preservation
- Debug complexity estimation
- Debugger resistance analysis

### 9. **Code Coverage**
- Reachable code estimation
- Path coverage analysis
- Estimated coverage metrics

### 10. **Complexity Analysis**
- Cyclomatic complexity estimation
- CFG density analysis
- Instruction complexity

### 11. **Reverse Engineering Difficulty**
- Composite difficulty score (0-100)
- Multi-factor RE resistance assessment
- Difficulty level classification

## Output Reports

The suite generates three types of reports:

### 1. JSON Report (`{program}_results.json`)
Complete test results in machine-readable JSON format with all metrics and test data.

### 2. Text Report (`{program}_report.txt`)
Comprehensive human-readable report with detailed analysis of all test results.

### 3. Summary Report (`{program}_summary.txt`)
Quick overview with key findings and pass/fail status.

## Example Results Structure

```
results/
├── baseline/program_name/
│   └── app                       # Baseline binary copy
├── obfuscated/program_name/
│   └── app                       # Obfuscated binary copy
├── reports/program_name/
│   ├── program_name_results.json   # Full JSON report
│   ├── program_name_report.txt     # Full text report
│   └── program_name_summary.txt    # Summary report
└── metrics/program_name/
    └── metrics.json              # Raw metrics data
```

## Key Metrics Explained

### Cyclomatic Complexity
Measure of control flow complexity. Higher values indicate more complex control flow after obfuscation.

### String Reduction
Percentage of plaintext strings removed or obfuscated. Target: > 20%

### Entropy
Shannon entropy of binary. Higher values indicate more randomization. Target: increase > 0.1

### Performance Overhead
Percentage increase in execution time. Target: < 100% (2x slowdown acceptable)

### RE Difficulty Score
Composite score (0-100) based on multiple factors:
- 0-25: LOW
- 25-50: MEDIUM
- 50-75: HIGH
- 75-100: VERY HIGH

## Requirements

- Python 3.6+
- Standard UNIX tools: `objdump`, `nm`, `file`, `strings`, `readelf`
- x86-64 or ARM binary architecture support

## Testing Multiple Programs

To test multiple programs, create a batch script:

```bash
#!/bin/bash
for program in program1 program2 program3; do
    baseline="./binaries/${program}_baseline"
    obfuscated="./binaries/${program}_obfuscated"
    python3 obfuscation_test_suite.py "$baseline" "$obfuscated" -n "$program"
done
```

## Troubleshooting

### Binary not executable
The suite automatically sets execute permission if needed.

### No debug information
Expected behavior for production builds. Debug info can be preserved with `-g` flag.

### Performance timeout
Binaries that timeout during execution are logged as 5000ms (5 seconds).

### Missing tools
Ensure standard UNIX tools are installed:
```bash
sudo apt-get install binutils coreutils
```

## Advanced Usage

### Custom Functional Tests

Extend `test_functional.py` with custom test cases:

```python
def test_custom_behavior(self, test_inputs: list) -> bool:
    """Custom functional test"""
    return self.test_with_input(test_inputs)
```

### Adding New Metrics

Extend `test_metrics.py` to add custom metric computations:

```python
def compute_custom_metrics(baseline: str, obfuscated: str) -> Dict[str, Any]:
    """Compute custom metrics"""
    # Implementation
    pass
```

## Output Examples

### Summary Report Example
```
OBFUSCATION TEST SUMMARY
==================================================

Program: my_program
Date:    2024-11-28T12:34:56

✓ Functional correctness maintained
✓ RE Difficulty: HIGH (78/100)
✓ String reduction: 42.3%
✓ Performance overhead: 35.7% (acceptable)

For detailed results, see the full report.
```

### Key Findings (Text Report)
```
REVERSE ENGINEERING DIFFICULTY
-------------------------------
Score:               78/100
Difficulty Level:    HIGH

STRING OBFUSCATION
-----------
Baseline Strings:    245
Obfuscated Strings:  141
Reduction:           42.4%

PERFORMANCE ANALYSIS
--------------------
Baseline Time:       45.23 ms
Obfuscated Time:     61.37 ms
Overhead:            +35.7%
Acceptable:          True
```

## Methodology

This test suite follows industry-standard obfuscation evaluation methodology:

1. **Deterministic Testing**: All functional tests are deterministic and reproducible
2. **Multi-Dimensional**: Evaluates across 11 independent dimensions
3. **Automated**: Minimal manual intervention required
4. **Comprehensive**: Produces detailed reports suitable for academic papers
5. **Comparative**: Direct baseline vs obfuscated comparison

## References

- OLLVM: https://github.com/obfuscator-llvm/obfuscator
- Binary Analysis: Ghidra, Binary Ninja, Angr
- RE Resistance Metrics: Academic research on code obfuscation effectiveness

## License

Same as parent OaaS project
