# Jotai CI Testing

## Overview

The `test_jotai_ci.py` script is designed for CI/CD integration. It:

1. **Gets all C source files** from Jotai benchmark collection
2. **Creates baseline binaries** (normal compilation)
3. **Runs obfuscation** on source files
4. **Creates obfuscated binaries**
5. **Runs both binaries** with same inputs
6. **Confirms identical output** (functional equivalence)

## Usage

### Command Line

```bash
cd cmd/llvm-obfuscator

# Basic usage
python3 tests/test_jotai_ci.py

# With options
python3 tests/test_jotai_ci.py \
    --limit 50 \
    --level 3 \
    --min-success-rate 0.7 \
    --output ./results \
    --json-report ./summary.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output <path>` | `./jotai_ci_results` | Output directory |
| `--level <1-5>` | `3` | Obfuscation level |
| `--limit <n>` | `50` | Max benchmarks to test |
| `--min-success-rate <0.0-1.0>` | `0.7` | Minimum success rate for CI pass |
| `--json-report <path>` | None | Path to save JSON summary |

### Exit Codes

- `0` - All tests passed (success rate meets minimum)
- `1` - Tests failed (success rate below minimum or no tests passed)

## CI Integration

### GitHub Actions

The workflow `.github/workflows/jotai-tests.yml` automatically runs on:
- Push to `main` or `develop` branches
- Pull requests
- Manual trigger via `workflow_dispatch`

### Example CI Output

```
============================================================
Jotai CI Integration Test
============================================================
Output directory: ./jotai_ci_results
Obfuscation level: 3
Max benchmarks: 50
Min success rate: 70%

[1/5] Initializing benchmark manager...
✓ Benchmark cache: /home/runner/.cache/llvm-obfuscator/jotai-benchmarks

[2/5] Getting benchmarks (limit: 50)...
✓ Found 50 benchmarks

[3/5] Setting up obfuscator...
✓ Obfuscator ready

[4/5] Running 50 benchmarks...
  This will:
    1. Compile baseline binaries (normal compilation)
    2. Obfuscate source files
    3. Compile obfuscated binaries
    4. Run both with same inputs
    5. Verify identical output

[5/5] Generating report...
✓ Report saved: ./jotai_ci_results/jotai_ci_report.json

============================================================
Test Results Summary
============================================================
Total benchmarks:        50
Tested:                  42
Skipped (compilation):   8

Compilation success:     42/42
Obfuscation success:     40/42
Functional tests passed: 38/42

Success rate:            90.5%
Input tests:             38/38 passed
Input success rate:      100.0%

✅ CI PASS: Success rate 90.5% meets minimum 70%
```

## What Gets Tested

### For Each Benchmark:

1. **Baseline Compilation**
   ```bash
   clang -g -O1 benchmark.c -o benchmark_baseline
   ```

2. **Obfuscation**
   - Applies obfuscation to source code
   - Symbol renaming (if enabled)
   - String encryption (if enabled)
   - Control flow obfuscation (if enabled)

3. **Obfuscated Compilation**
   ```bash
   clang [obfuscated_source.c] -o benchmark_obfuscated
   ```

4. **Functional Testing**
   ```bash
   ./benchmark_baseline <input>    # Get baseline output
   ./benchmark_obfuscated <input>  # Get obfuscated output
   # Compare: outputs must be identical
   ```

## Success Criteria

CI passes if:
- ✅ At least one benchmark is successfully tested
- ✅ Success rate ≥ minimum (default 70%)
- ✅ At least one functional test passes

CI fails if:
- ❌ No benchmarks were tested
- ❌ Success rate < minimum
- ❌ No functional tests passed

## Handling Failures

### Compilation Errors
- **Expected**: Some Jotai benchmarks have compatibility issues
- **Handling**: Automatically skipped, not counted as failures
- **Impact**: Reduces total tested count, but doesn't fail CI

### Obfuscation Failures
- **Cause**: Obfuscation process errors
- **Handling**: Counted as failure, but CI continues
- **Impact**: Reduces success rate

### Functional Test Failures
- **Cause**: Obfuscated binary produces different output
- **Handling**: Counted as failure
- **Impact**: Critical - indicates obfuscation broke functionality

## Reports

### JSON Report (`jotai_ci_report.json`)
Contains detailed results for each benchmark:
```json
{
  "summary": {
    "total_benchmarks": 50,
    "tested": 42,
    "success_rate": 0.905,
    ...
  },
  "results": [
    {
      "benchmark_name": "...",
      "compilation_success": true,
      "obfuscation_success": true,
      "functional_test_passed": true,
      ...
    }
  ]
}
```

### Console Output
- Real-time progress
- Summary statistics
- Detailed per-benchmark results
- CI pass/fail status

## Best Practices

1. **Start with small limit**: Use `--limit 10` for quick tests
2. **Adjust success rate**: Lower for experimental features
3. **Check reports**: Review JSON for detailed failure reasons
4. **Monitor trends**: Track success rate over time
5. **Fix regressions**: Investigate functional test failures immediately

## Troubleshooting

### "No benchmarks found"
- Check internet connection (downloads on first run)
- Verify Git is installed
- Check cache directory permissions

### "All benchmarks skipped"
- Many benchmarks may have compilation issues
- Try different category: `--category anghaMath`
- Increase limit to test more benchmarks

### "Low success rate"
- Check obfuscation configuration
- Review error messages in report
- Test with lower obfuscation level first

## Integration with Other Tests

This test can be combined with:
- Unit tests (`pytest tests/`)
- Integration tests (`test_remarks_integration.py`)
- Manual testing scripts

Run all tests:
```bash
# Unit tests
pytest tests/

# Jotai CI tests
python3 tests/test_jotai_ci.py --limit 20

# Integration tests
python3 tests/test_remarks_integration.py
```

