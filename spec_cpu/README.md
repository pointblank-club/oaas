# SPEC CPU 2017 Benchmarking Module

Local-only benchmarking module for evaluating LLVM obfuscation performance impact using SPEC CPU 2017.

**Important**: This module is designed for local development machines only and is **never** executed in CI/CD pipelines.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Module Overview](#module-overview)
4. [Script Reference](#script-reference)
5. [Toolchain Configuration](#toolchain-configuration)
6. [Result Analysis](#result-analysis)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Workflow

```bash
# 1. Configure SPEC CPU environment
./scripts/configure_spec_cpu.sh baseline

# 2. Build baseline benchmarks
./scripts/build_spec_targets.sh baseline

# 3. Run speed benchmarks (single-threaded)
./scripts/run_spec_speed.sh baseline

# 4. Run rate benchmarks (multi-threaded)
./scripts/run_spec_rate.sh baseline

# 5. Repeat for obfuscated builds with custom configuration
./scripts/configure_spec_cpu.sh obfuscated layer1-2
./scripts/build_spec_targets.sh obfuscated layer1-2
./scripts/run_spec_speed.sh obfuscated layer1-2
./scripts/run_spec_rate.sh obfuscated layer1-2 4  # 4 copies

# 6. Generate comparison report
./scripts/compare_spec_results.py results/baseline/<timestamp> results/obfuscated/layer1-2/<timestamp> --config layer1-2
```

### First Run Checklist

- [ ] SPEC CPU 2017 installed at `/opt/spec2017` (or set `SPEC_CPU_HOME`)
- [ ] Custom Clang toolchain available in `cmd/llvm-obfuscator/plugins/clang` (optional for baseline)
- [ ] System GCC/G++ available as fallback
- [ ] Enough disk space for benchmark binaries and results (~10-20 GB)
- [ ] No other heavy processes running during benchmarks

---

## Prerequisites

### System Requirements

- Linux x86_64 architecture
- 16+ GB RAM (32+ GB recommended for rate benchmarks)
- 50+ GB free disk space
- GCC/G++ 7.0 or later
- Python 3.7+

### SPEC CPU 2017

Download and install SPEC CPU 2017 from https://www.spec.org/cpu2017/:

```bash
# Extract to /opt/spec2017
tar -xzf cpu2017-1.1.0.tar.xz
cd cpu2017-1.1.0
./install.sh
```

Verify installation:

```bash
ls /opt/spec2017/bin/runcpu
ls /opt/spec2017/benchspec/CPU
```

Expected: ~54 benchmark directories in `benchspec/CPU/`

### LLVM Obfuscator Plugin (Optional)

For obfuscated builds, ensure custom Clang is available:

```bash
ls -lh cmd/llvm-obfuscator/plugins/clang
ls -lh cmd/llvm-obfuscator/plugins/clang++
```

If not present, only baseline builds will be possible.

---

## Module Overview

### Directory Structure

```
spec_cpu/
├── configs/
│   └── linux-x86_64.cfg           SPEC CPU compiler configuration
├── scripts/
│   ├── configure_spec_cpu.sh       Environment setup and validation
│   ├── build_spec_targets.sh       Build baseline and obfuscated binaries
│   ├── run_spec_speed.sh           Execute speed benchmarks
│   ├── run_spec_rate.sh            Execute rate benchmarks
│   ├── collect_spec_metrics.py     Extract and aggregate results
│   └── compare_spec_results.py     Generate comparison reports
├── results/
│   ├── baseline/
│   │   └── <timestamp>/            Baseline run results
│   │       ├── speed/              SPECspeed results
│   │       └── rate/               SPECrate results
│   ├── obfuscated/
│   │   └── <config>/
│   │       └── <timestamp>/        Obfuscated run results
│   │           ├── speed/
│   │           └── rate/
│   └── comparisons/
│       └── <config>/
│           └── <timestamp>/        Comparison reports
├── SPECIFICATION.md                Technical specification
└── README.md                       This file
```

### Benchmark Suites

#### SPECspeed 2017 (Single-threaded)

23 benchmarks measuring CPU performance on single-threaded workloads:

**Integer**: perlbench, gcc, mcf, omnetpp, xalancbmk, x264, deepsjeng, leela, exchange2, xz

**Floating-Point**: bwaves, cactuBSSN, namd, parest, povray, lbm, wrf, blender, cam4, imagick, nab, fotonik3d, roms

#### SPECrate 2017 (Multi-threaded)

Same 23 benchmarks executed with multiple concurrent copies to measure throughput.

---

## Script Reference

### 1. `configure_spec_cpu.sh`

Initialize and validate SPEC CPU environment.

**Usage**:
```bash
./scripts/configure_spec_cpu.sh <build_type> [obfuscation_config]
```

**Arguments**:
- `build_type`: `baseline` or `obfuscated`
- `obfuscation_config`: Required for obfuscated builds (e.g., `layer1-2`, `full-obf`)

**Examples**:
```bash
# Configure for baseline
./scripts/configure_spec_cpu.sh baseline

# Configure for obfuscated
./scripts/configure_spec_cpu.sh obfuscated layer1-2
```

**Exit Codes**:
- `0`: Configuration successful
- `1`: SPEC CPU not found
- `2`: Required tools missing
- `3`: No valid compiler chain available
- `4`: Plugin directory invalid

**Output**:
- Validates SPEC CPU installation
- Detects available compilers (custom Clang or system GCC)
- Creates result directories
- Generates config file: `$SPEC_CPU_HOME/config/llvm-obfuscation.cfg`
- Logs to: `spec_cpu/results/configure_spec_cpu.log`

---

### 2. `build_spec_targets.sh`

Compile benchmark binaries with specified toolchain.

**Usage**:
```bash
./scripts/build_spec_targets.sh <build_type> [obfuscation_config] [target_list]
```

**Arguments**:
- `build_type`: `baseline` or `obfuscated`
- `obfuscation_config`: Required for obfuscated
- `target_list`: Comma-separated benchmarks or `all` (default)

**Examples**:
```bash
# Build all baseline benchmarks
./scripts/build_spec_targets.sh baseline

# Build obfuscated configuration
./scripts/build_spec_targets.sh obfuscated layer1-2

# Build specific benchmarks
./scripts/build_spec_targets.sh baseline "" "perlbench,gcc,mcf"
```

**Exit Codes**:
- `0`: Build successful
- `1`: Config not found
- `2`: Baseline build failed
- `3`: Obfuscated build failed or plugin unavailable
- `4`: Invalid configuration
- `5`: Directory creation failed

**Output**:
- Binaries stored in SPEC CPU build directory
- Metadata: `build/<baseline|obfuscated_spec>/<config>/BUILD_METADATA.txt`
- Build logs: `spec_cpu/results/<baseline|obfuscated>/build_*.log`

---

### 3. `run_spec_speed.sh`

Execute SPECspeed (single-threaded) benchmarks.

**Usage**:
```bash
./scripts/run_spec_speed.sh <run_type> [obfuscation_config]
```

**Arguments**:
- `run_type`: `baseline`, `obfuscated`, or `all`
- `obfuscation_config`: Configuration name (required if run_type is `obfuscated`)

**Examples**:
```bash
# Run baseline speed benchmarks
./scripts/run_spec_speed.sh baseline

# Run specific obfuscated configuration
./scripts/run_spec_speed.sh obfuscated layer1-2

# Run all available obfuscated configurations
./scripts/run_spec_speed.sh all
```

**Exit Codes**:
- `0`: Benchmarks completed successfully
- `1`: Configuration error
- `2`: Baseline benchmarks failed
- `3`: Obfuscated benchmarks failed
- `4`: No configurations found (for `all` mode)
- `5`: Directory creation failed

**Output**:
- Results: `results/baseline/<timestamp>/speed/` or `results/obfuscated/<config>/<timestamp>/speed/`
- Execution details: `EXECUTION_DETAILS.txt` (timestamp, duration, status)
- Summary: `speed_run_summary_<timestamp>.txt` (pass/fail counts)
- Raw SPEC output: `runcpu_*.log` files

---

### 4. `run_spec_rate.sh`

Execute SPECrate (multi-threaded) benchmarks.

**Usage**:
```bash
./scripts/run_spec_rate.sh <run_type> [obfuscation_config] [num_copies]
```

**Arguments**:
- `run_type`: `baseline`, `obfuscated`, or `all`
- `obfuscation_config`: Configuration name (required if run_type is `obfuscated`)
- `num_copies`: Number of concurrent copies (default: CPU count)

**Examples**:
```bash
# Run baseline rate benchmarks (uses CPU count)
./scripts/run_spec_rate.sh baseline

# Run with specific copy count
./scripts/run_spec_rate.sh baseline "" 8

# Run obfuscated configuration with 4 copies
./scripts/run_spec_rate.sh obfuscated layer1-2 4

# Run all configurations with 4 copies each
./scripts/run_spec_rate.sh all "" 4
```

**Exit Codes**: Same as `run_spec_speed.sh`

**Output**: Similar to speed runner, but in `rate/` subdirectory

**Notes**:
- Script validates `num_copies` is a positive integer
- Warns if copies > 4 × CPU count (potential system overload)
- Recommended: num_copies = CPU count for realistic throughput measurement

---

### 5. `collect_spec_metrics.py`

Extract and aggregate SPEC CPU results into structured metrics.

**Usage**:
```bash
./scripts/collect_spec_metrics.py <result_directory> [--test-type speed|rate] [--format json|csv|both] [--output-dir output]
```

**Arguments**:
- `result_directory`: Path to speed/ or rate/ results
- `--test-type`: Benchmark type (default: speed)
- `--format`: Output format (default: json)
- `--output-dir`: Output directory (default: same as input)

**Examples**:
```bash
# Extract metrics from speed results
./scripts/collect_spec_metrics.py results/baseline/2025-12-06T12:00:00Z/speed

# Export both JSON and CSV
./scripts/collect_spec_metrics.py results/baseline/2025-12-06T12:00:00Z/speed --format both

# Save to custom directory
./scripts/collect_spec_metrics.py results/baseline/2025-12-06T12:00:00Z/speed --output-dir ./metrics
```

**Exit Codes**:
- `0`: Metrics collected successfully
- `1`: Invalid result directory
- `2`: No results found
- `3`: Parsing error
- `4`: File I/O error

**Output**:
- `metrics.json`: Machine-readable metrics (base score, peak score, runtime, etc.)
- `metrics.csv`: CSV export with per-benchmark rows
- Summary statistics printed to stdout

**Metrics Included**:
- Benchmark name and type
- Compiler and version
- Base and peak scores
- Runtime and memory usage
- Geometric mean and standard deviation
- Component scores (if available)

---

### 6. `compare_spec_results.py`

Generate detailed comparison reports between baseline and obfuscated results.

**Usage**:
```bash
./scripts/compare_spec_results.py <baseline_path> <obfuscated_path> [--config name] [--format html|json|csv|both] [--output-dir output]
```

**Arguments**:
- `baseline_path`: Path to baseline results directory
- `obfuscated_path`: Path to obfuscated results directory
- `--config`: Obfuscation configuration name (for report title)
- `--format`: Output format (default: html)
- `--output-dir`: Output directory (default: obfuscated results directory)

**Examples**:
```bash
# Generate HTML comparison report
./scripts/compare_spec_results.py results/baseline/2025-12-06T12:00:00Z/speed results/obfuscated/layer1-2/2025-12-06T12:30:00Z/speed --config layer1-2

# Generate all output formats
./scripts/compare_spec_results.py ... --format both

# Save to custom directory
./scripts/compare_spec_results.py ... --output-dir ./comparisons
```

**Exit Codes**:
- `0`: Report generated successfully
- `1`: Baseline directory not found
- `2`: Obfuscated directory not found
- `3`: No matching benchmarks found
- `4`: Report generation error
- `5`: File I/O error

**Output**:
- `comparison_report.html`: Interactive HTML report with styling and tables
- `comparison_metrics.json`: Machine-readable comparison data
- `regression_analysis.csv`: Per-benchmark detailed analysis

**Report Includes**:
- Performance impact (% change in scores)
- Runtime overhead analysis
- Regression identification (>5% slowdown)
- Improvement tracking (>5% speedup)
- Summary statistics (mean, median, worst case)
- Benchmark-by-benchmark breakdown

---

## Toolchain Configuration

### Compiler Priority

The module uses a hierarchical compiler selection strategy:

```
Baseline builds:
  1. Try custom Clang from plugins/
     └─ If available: use custom Clang (preferred)
  2. Fall back to system GCC/G++
     └─ If available: use with enforced -O3 flag
  3. Fail if neither available

Obfuscated builds:
  1. Require custom Clang from plugins/
  2. Fail with clear error if unavailable (no fallback)
```

### Detecting Custom Clang

The scripts automatically look for:

```
cmd/llvm-obfuscator/plugins/clang
cmd/llvm-obfuscator/plugins/clang++
cmd/llvm-obfuscator/plugins/clang--
```

To verify availability:

```bash
./scripts/configure_spec_cpu.sh baseline  # Will report which compiler is detected
```

### Verifying Compiler Chain

```bash
# Check custom Clang
ls -lh cmd/llvm-obfuscator/plugins/clang
file cmd/llvm-obfuscator/plugins/clang

# Check system GCC
gcc --version
g++ --version
gfortran --version  # (Optional, for Fortran benchmarks)
```

---

## Result Analysis

### Understanding Results

#### Performance Scores

- **Base Score**: Baseline performance with standard optimizations
- **Peak Score**: Performance with additional tuning and optimizations
- **Geometric Mean**: Statistical center of all benchmark scores

#### Performance Impact (%)

```
Percentage Change = (Obfuscated Score - Baseline Score) / Baseline Score × 100
```

- **Negative**: Performance degradation from obfuscation overhead
- **Positive**: Performance improvement (rare, usually due to cache effects)
- **Threshold**: ±5% considered neutral

#### Regressions vs Improvements

- **Regression**: >5% performance decrease (red in reports)
- **Improvement**: >5% performance increase (green in reports)
- **Neutral**: -5% to +5% change (yellow in reports)

### Example Report Interpretation

```html
Performance Summary:
├─ Total Benchmarks: 23
├─ Valid Comparisons: 23
├─ Regressions: 5
├─ Improvements: 1
├─ Neutral: 17
└─ Average Impact: -3.2%
```

This indicates:
- 5 benchmarks experienced significant slowdown from obfuscation
- 17 benchmarks were minimally affected
- 1 benchmark actually improved (edge case)
- On average, 3.2% performance overhead from obfuscation layer

### Analyzing Specific Benchmarks

Use the detailed CSV export to identify problem benchmarks:

```bash
./scripts/compare_spec_results.py ... --format csv
cat regression_analysis.csv | grep -E "regression|improvement"
```

---

## Advanced Usage

### Custom Obfuscation Configurations

Create multiple obfuscation configs to compare strategies:

```bash
# Configuration 1: Light obfuscation
./scripts/configure_spec_cpu.sh obfuscated light
./scripts/build_spec_targets.sh obfuscated light
./scripts/run_spec_speed.sh obfuscated light
./scripts/run_spec_rate.sh obfuscated light

# Configuration 2: Heavy obfuscation
./scripts/configure_spec_cpu.sh obfuscated heavy
./scripts/build_spec_targets.sh obfuscated heavy
./scripts/run_spec_speed.sh obfuscated heavy
./scripts/run_spec_rate.sh obfuscated heavy

# Compare both
./scripts/compare_spec_results.py results/baseline/<ts>/speed results/obfuscated/light/<ts>/speed --config light
./scripts/compare_spec_results.py results/baseline/<ts>/speed results/obfuscated/heavy/<ts>/speed --config heavy
```

### Partial Benchmark Runs

Test with fewer benchmarks during development:

```bash
./scripts/build_spec_targets.sh baseline "" "perlbench,gcc,mcf"
./scripts/run_spec_speed.sh baseline
```

### Environment Customization

Override SPEC CPU location:

```bash
export SPEC_CPU_HOME=/custom/spec2017/path
./scripts/configure_spec_cpu.sh baseline
```

### Manual Result Processing

Process results programmatically:

```python
import json
from pathlib import Path

# Load metrics
with open("results/baseline/2025-12-06T12:00:00Z/speed/metrics.json") as f:
    metrics = json.load(f)

# Access summary stats
summary = metrics['summary']
print(f"Geometric mean: {summary['base_score_geomean']}")

# Iterate benchmarks
for metric in metrics['metrics']:
    print(f"{metric['benchmark_name']}: {metric['base_score']}")
```

---

## Troubleshooting

### SPEC CPU Installation Issues

**Error**: `SPEC CPU home directory not found: /opt/spec2017`

**Solution**:
```bash
# Verify installation
ls /opt/spec2017/bin/runcpu

# If not found, set custom path
export SPEC_CPU_HOME=/path/to/spec2017
./scripts/configure_spec_cpu.sh baseline
```

### Compiler Not Found

**Error**: `Custom Clang toolchain NOT found at: cmd/llvm-obfuscator/plugins`

**Solution** (for baseline - optional):
```bash
# System GCC will be used as fallback
./scripts/configure_spec_cpu.sh baseline  # Proceeds with GCC
```

**Solution** (for obfuscated - required):
```bash
# Build custom Clang first
cd cmd/llvm-obfuscator
./build.sh  # or appropriate build command
# Then retry
./scripts/configure_spec_cpu.sh obfuscated layer1-2
```

### Build Failures

**Error**: `Baseline build failed for benchmark: perlbench_r`

**Investigation**:
```bash
# Check build logs
tail -100 spec_cpu/results/baseline/build_*.log

# Retry with verbose output
./scripts/build_spec_targets.sh baseline "" "perlbench_r" 2>&1 | tee debug.log
```

### Memory/Disk Space Issues

**Error**: `No space left on device`

**Solution**:
- Rate benchmarks with multiple copies require significant disk space (~20 GB)
- Reduce `num_copies` for rate runs
- Delete old result directories: `rm -rf spec_cpu/results/baseline/<old_timestamp>`

### Results Parsing Errors

**Error**: `Failed to load metrics.json` when comparing results

**Solution**:
```bash
# Verify results directory contains output files
ls spec_cpu/results/baseline/*/speed/
ls spec_cpu/results/baseline/*/speed/metrics.json

# Manually collect metrics if missing
./scripts/collect_spec_metrics.py spec_cpu/results/baseline/*/speed --format both
```

### Permission Issues

**Error**: `Permission denied` when running scripts

**Solution**:
```bash
# Ensure scripts are executable
chmod +x spec_cpu/scripts/*.sh
chmod +x spec_cpu/scripts/*.py

# Verify
ls -l spec_cpu/scripts/
```

---

## Performance Tips

1. **Disable CPU frequency scaling** for consistent results:
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **Close unnecessary applications** to reduce noise in measurements

3. **Use consistent system state** between baseline and obfuscated runs

4. **For rate benchmarks**: Run copies = CPU count for realistic throughput measurement

5. **Multiple iterations**: Run benchmarks 2-3 times and average results for better statistical confidence

---

## Support and Reporting Issues

For issues or questions:

1. Check the troubleshooting section above
2. Review log files: `spec_cpu/results/*/*.log`
3. Verify prerequisites are met (SPEC installation, compilers, disk space)
4. Run `configure_spec_cpu.sh` in verbose mode for diagnostics

---

## Notes

- **Local-Only**: This module is never committed to CI/CD. Results are not versioned.
- **Long-Running**: Full benchmark suite can take 2-6 hours depending on hardware
- **Resource Intensive**: Rate benchmarks with high copy counts can stress system resources
- **Determinism**: Results vary based on system load. Multiple runs recommended.
- **Configuration**: Modify `configs/linux-x86_64.cfg` only for advanced tuning

---

Last Updated: December 6, 2025
