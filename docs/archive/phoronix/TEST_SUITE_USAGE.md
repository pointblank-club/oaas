# Unified Obfuscation Test Suite - Complete Guide

## Overview

The **Unified Obfuscation Test Suite** (`run_obfuscation_test_suite.sh`) is a single command to analyze any two binaries (baseline and obfuscated) and generate comprehensive reports.

### What It Does

1. ✅ Collects obfuscation metrics (file size, functions, instructions, entropy)
2. ✅ Performs security analysis (decompilation difficulty, CFG reconstruction)
3. ✅ Generates aggregated reports (JSON, Markdown, HTML, CSV)
4. ✅ Organizes all results in timestamped directories
5. ✅ Creates execution logs for troubleshooting

---

## Quick Start

### Basic Usage

```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh <baseline_binary> <obfuscated_binary> [output_dir]
```

### Examples

#### Example 1: Test with Default Output Directory
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    ./phoronix/option1_binaries/hello_world_full_symbols \
    /home/user/Downloads/obfuscated_linux \
    ./results/
```

**Output Directory Created:**
```
./results/obfuscation_analysis_hello_world_full_symbols_vs_obfuscated_linux_20251206_023620/
```

#### Example 2: Test Any Two Binaries
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    ./baseline_program \
    ./obfuscated_program
```

**Output Directory Created:**
```
./obfuscation_analysis_baseline_program_vs_obfuscated_program_20251206_025000/
```

#### Example 3: Custom Output Location
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    /path/to/baseline \
    /path/to/obfuscated \
    /home/results/tests/
```

---

## Output Directory Structure

Each test run creates a **timestamped directory** with organized subdirectories:

```
obfuscation_analysis_{baseline_name}_vs_{obfuscated_name}_{YYYYMMDD_HHMMSS}/
│
├── metrics/
│   ├── metrics.json          # JSON: Function count, instruction count, entropy
│   ├── metrics.csv           # Spreadsheet-compatible format
│   └── metrics.md            # Markdown table format
│
├── security/
│   └── security_analysis.json # Decompilation difficulty, CFG analysis
│
├── reports/
│   ├── final_report.json     # Complete aggregated analysis (programmatic)
│   ├── final_report.md       # Markdown report (human-readable)
│   ├── final_report.html     # Interactive HTML visualization
│   └── final_report.csv      # Spreadsheet import format
│
├── logs/
│   ├── execution.log         # Full execution timeline
│   ├── metrics.log           # Metrics collection details
│   ├── security.log          # Security analysis details
│   └── report.log            # Report generation details
│
└── INDEX.md                  # Navigation guide to all reports
```

---

## Reports Explained

### 1. Metrics Report
**Files:** `metrics/metrics.json`, `metrics/metrics.csv`, `metrics/metrics.md`

Contains binary analysis metrics:
- **File size** comparison (bytes and percentage)
- **Function count** (extracted from symbols)
- **Basic blocks** (CFG nodes)
- **Instruction count** (estimated)
- **Text entropy** (code randomness)
- **Cyclomatic complexity** (code branches)

**Best for:** Understanding what changed between binaries

### 2. Security Analysis Report
**File:** `security/security_analysis.json`

Contains obfuscation difficulty metrics:
- **Irreducible CFG detection** (% of complex control flow)
- **Opaque predicates count** (fake conditions)
- **Basic block recovery** (% of successful CFG reconstruction)
- **String obfuscation ratio** (hidden strings)
- **Symbol obfuscation ratio** (renamed symbols)
- **Decompilation readability score** (0-10 scale)

**Best for:** Understanding reverse engineering difficulty

### 3. Final Aggregated Report
**Files:** `reports/final_report.json`, `final_report.md`, `final_report.html`, `final_report.csv`

Contains everything combined:
- Summary of all metrics
- Performance analysis
- Binary complexity assessment
- Final obfuscation score (0-10 with letter rating)
- Detailed interpretation

**Best for:** Executive summary and decision-making

### 4. Execution Logs
**Directory:** `logs/`

Detailed logs for each phase:
- What was executed
- Any warnings or errors
- Processing times
- Data validation results

**Best for:** Troubleshooting issues

---

## Step-by-Step Example

### Scenario: Test Two Real Binaries

```bash
# Create output directory
mkdir -p ~/obfuscation_tests

# Run test suite
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    /home/user/baseline_program \
    /home/user/obfuscated_program \
    ~/obfuscation_tests/

# Output shows:
# ✅ Metrics collection completed
# ✅ Security analysis completed
# ✅ Aggregated reports generated
# ✅ Summary index created
```

### Navigate Results

```bash
# Go to results directory
cd ~/obfuscation_tests/obfuscation_analysis_*/

# Quick review
cat INDEX.md

# Detailed analysis
cat reports/final_report.md

# Import to spreadsheet
cp metrics/metrics.csv ~/Desktop/

# View interactive HTML (in browser)
# reports/final_report.html
```

---

## Real-World Use Cases

### Use Case 1: Compare Optimization Levels

**Test if different optimization levels affect obfuscation:**

```bash
# Compile baseline with -O2
gcc -O2 source.c -o baseline_O2

# Compile baseline with -O3
gcc -O3 source.c -o baseline_O3

# Test both
bash run_obfuscation_test_suite.sh ./baseline_O2 ./baseline_O3 ./results/
```

### Use Case 2: Validate Obfuscation Effectiveness

**Verify your obfuscation transformation is working:**

```bash
# Original binary (before obfuscation)
./my_original_binary

# After obfuscation pass
./my_obfuscated_binary

# Test effectiveness
bash run_obfuscation_test_suite.sh ./my_original_binary ./my_obfuscated_binary ./results/
```

### Use Case 3: Regression Testing

**Track obfuscation metrics across multiple versions:**

```bash
# Version 1.0 vs Version 2.0 (after obfuscation)
bash run_obfuscation_test_suite.sh ./v1.0 ./v2.0 ./results/

# Version 2.0 vs Version 3.0
bash run_obfuscation_test_suite.sh ./v2.0 ./v3.0 ./results/

# Compare trends in results/
```

### Use Case 4: Performance Benchmark

**Measure if obfuscation affects performance:**

```bash
# Run PTS benchmark on baseline
./run_benchmark.sh ./baseline_binary > baseline_perf.json

# Run PTS benchmark on obfuscated
./run_benchmark.sh ./obfuscated_binary > obfuscated_perf.json

# Analyze with test suite (combines perf + code metrics)
bash run_obfuscation_test_suite.sh ./baseline ./obfuscated ./results/
```

---

## Tips & Tricks

### Organize Multiple Test Runs

```bash
# Create organized test directory
mkdir -p ~/obfuscation_analysis/{baseline,test1,test2}

# Run multiple tests
bash run_obfuscation_test_suite.sh baseline obfuscated1 ~/obfuscation_analysis/test1/
bash run_obfuscation_test_suite.sh baseline obfuscated2 ~/obfuscation_analysis/test2/

# Compare results easily
diff ~/obfuscation_analysis/test1/*/reports/final_report.json \
     ~/obfuscation_analysis/test2/*/reports/final_report.json
```

### Batch Process Multiple Binaries

```bash
#!/bin/bash
# batch_test.sh - Test multiple obfuscated binaries against one baseline

BASELINE="./baseline_program"
OBFUSCATED_DIR="/home/binaries/obfuscated/"
OUTPUT_DIR="./batch_results/"

mkdir -p "$OUTPUT_DIR"

for binary in "$OBFUSCATED_DIR"/*; do
    echo "Testing: $(basename $binary)"
    bash phoronix/scripts/run_obfuscation_test_suite.sh \
        "$BASELINE" \
        "$binary" \
        "$OUTPUT_DIR/"
done

echo "All tests complete. Results in: $OUTPUT_DIR"
```

### Extract Specific Metrics

```bash
# Get only the obfuscation score
cd obfuscation_analysis_*/
cat reports/final_report.json | grep -A5 '"obfuscation_score"'

# Get binary sizes
cat metrics/metrics.json | grep -E 'file_size|instruction_count'

# Get decompilation difficulty
cat security/security_analysis.json | grep decompilation_readability_score
```

---

## Requirements

### Binaries
- **Baseline binary**: Any ELF x86-64 executable
- **Obfuscated binary**: Any ELF x86-64 executable to compare
- Both must be on the system (readable)

### Dependencies
- `python3` - For metrics and report generation
- `bash` - For script execution
- Standard Linux tools: `file`, `nm`, `objdump`, `readelf`, `strings`

### Optional
- **Ghidra** (at `/opt/ghidra`) - For 85-90% accuracy analysis
- Without Ghidra - Uses heuristics (40% accuracy)

### Hardware
- Minimal: Works on systems with 512MB RAM
- Typical test: Takes 2-5 seconds per binary

---

## Troubleshooting

### Error: "Binary not found"
```bash
# Verify binary exists and is readable
ls -lh ./your_binary
file ./your_binary
```

### Error: "Python modules not found"
```bash
# Check Python version
python3 --version

# Verify modules in place
python3 phoronix/scripts/collect_obfuscation_metrics.py --help
```

### Empty Reports
```bash
# Check execution log
cat obfuscation_analysis_*/logs/execution.log

# Verify binaries are valid ELF
file ./baseline_binary
file ./obfuscated_binary
```

### Want Ghidra Analysis?
```bash
# Install Ghidra first
bash phoronix/scripts/option2_ghidra_integration.sh

# Then re-run test suite
bash run_obfuscation_test_suite.sh baseline obfuscated ./results/
```

---

## Output Files Quick Reference

| File | Purpose | Format | Who Uses It |
|------|---------|--------|-----------|
| `metrics.json` | Raw metrics | JSON | Programmers |
| `metrics.csv` | Metrics spreadsheet | CSV | Analysts |
| `metrics.md` | Metrics table | Markdown | Documentation |
| `security_analysis.json` | Decompilation data | JSON | Analysts |
| `final_report.json` | Complete analysis | JSON | Automation |
| `final_report.md` | Human report | Markdown | Reviewers |
| `final_report.html` | Interactive view | HTML | Presentations |
| `final_report.csv` | Excel import | CSV | Spreadsheets |
| `execution.log` | Debug info | Text | Troubleshooting |
| `INDEX.md` | Navigation | Markdown | Getting started |

---

## Automation Examples

### GitHub Actions Integration
```yaml
- name: Run Obfuscation Test Suite
  run: |
    bash phoronix/scripts/run_obfuscation_test_suite.sh \
      ./build/baseline \
      ./build/obfuscated \
      ./test-results/
```

### CI/CD Pipeline
```bash
# In your build script
./build.sh baseline
./build.sh obfuscated
bash run_obfuscation_test_suite.sh ./baseline ./obfuscated ./ci-results/
```

### Scheduled Regression Testing
```bash
# In crontab
0 2 * * * bash /path/to/run_obfuscation_test_suite.sh baseline obfuscated ~/weekly-results/
```

---

## Summary

The **Unified Test Suite** provides:

✅ **Single command** to test any two binaries
✅ **Organized reports** with timestamps
✅ **Multiple formats** (JSON, CSV, MD, HTML)
✅ **Detailed logs** for troubleshooting
✅ **Reusable** for repeated testing

**Get started:**
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline obfuscated ./results/
```

**View results:**
```bash
cat ./results/obfuscation_analysis_*/INDEX.md
```

Done! 🎯
