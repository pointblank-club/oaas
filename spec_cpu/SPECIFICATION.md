# SPEC CPU 2017 Benchmarking Module - Specification

## Overview

The SPEC CPU module provides local-only benchmarking capabilities for evaluating LLVM obfuscation performance impact using SPEC CPU 2017 benchmarks. This module is **never executed in CI/CD** and is designed for detailed performance analysis on development machines.

---

## Module Structure

```
spec_cpu/
├── configs/
│   └── linux-x86_64.cfg          (SPEC CPU configuration file)
├── scripts/
│   ├── configure_spec_cpu.sh      (SPEC CPU environment setup)
│   ├── build_spec_targets.sh      (Build baseline and obfuscated binaries)
│   ├── run_spec_speed.sh          (Execute SPEED benchmarks)
│   ├── run_spec_rate.sh           (Execute RATE benchmarks)
│   ├── collect_spec_metrics.py    (Extract and aggregate results)
│   └── compare_spec_results.py    (Generate comparison reports)
├── results/
│   ├── baseline/
│   │   └── <timestamp>/
│   │       ├── speed/              (Speed benchmark results)
│   │       └── rate/               (Rate benchmark results)
│   ├── obfuscated/
│   │   └── <config_name>/
│   │       └── <timestamp>/
│   │           ├── speed/          (Speed benchmark results)
│   │           └── rate/           (Rate benchmark results)
│   └── comparisons/
│       └── <config_name>/
│           └── <timestamp>/        (Comparison reports)
├── SPECIFICATION.md               (This file)
└── README.md                       (User guide and usage instructions)
```

---

## Key Design Principles

### 1. Local-Only Execution
- Module is **never integrated into CI/CD pipelines**
- Designed for manual execution on development/testing machines
- All results are stored locally in `spec_cpu/results/`

### 2. Toolchain Rules

#### Priority-Based Fallback Chain:
```
1. Custom Clang (plugins/clang and plugins/clang++)
   ├─ For obfuscated builds: REQUIRED
   └─ For baseline builds: OPTIONAL (but preferred)

2. System GCC/G++ (fallback)
   ├─ For baseline builds: ALLOWED with -O3 flag
   └─ For obfuscated builds: NOT ALLOWED (fail with clear error)

3. Clang-like Flags on GCC:
   ├─ When GCC is used for baseline: Apply Clang-compatible flags
   └─ Ensure semantic compatibility between baseline and obfuscated
```

#### Build Requirements:
- **Baseline builds**: Must include `-O3` optimization flag (GCC or custom Clang)
- **Obfuscated builds**: Must use custom Clang from `plugins/` directory
- If custom Clang is unavailable for obfuscated builds → explicit failure with detailed error message

### 3. Result Organization

#### Baseline Results (`results/baseline/<timestamp>/`)
- Single timestamp per baseline run
- Subdirectories: `speed/` and `rate/`
- Contains raw SPEC CPU results

#### Obfuscated Results (`results/obfuscated/<config_name>/<timestamp>/`)
- Organized by obfuscation configuration name
- Multiple timestamps for different obfuscation runs
- Subdirectories: `speed/` and `rate/`
- Tracks different obfuscation settings separately

#### Comparison Results (`results/comparisons/<config_name>/<timestamp>/`)
- Organized by obfuscation configuration name
- Contains side-by-side comparison reports
- Metrics: performance impact, binary size ratio, obfuscation overhead

---

## Benchmark Targets

### SPEC CPU 2017 Suites

#### SPECspeed 2017 (Single-threaded)
Benchmarks:
- int: 500.perlbench_r, 502.gcc_r, 505.mcf_r, 520.omnetpp_r, 523.xalancbmk_r, 525.x264_r, 531.deepsjeng_r, 541.leela_r, 548.exchange2_r, 557.xz_r
- fp: 503.bwaves_r, 507.cactuBSSN_r, 508.namd_r, 510.parest_r, 511.povray_r, 519.lbm_r, 521.wrf_r, 526.blender_r, 527.cam4_r, 538.imagick_r, 544.nab_r, 549.fotonik3d_r, 554.roms_r

#### SPECrate 2017 (Multi-threaded)
Same benchmarks as speed but with multiple concurrent copies

---

## Script Specifications

### 1. `configure_spec_cpu.sh`
**Purpose**: Initialize and validate SPEC CPU environment

**Responsibilities**:
- Locate SPEC CPU 2017 installation directory
- Validate SPEC CPU tools presence (specinvoke, runcpu, etc.)
- Check for custom Clang toolchain in `plugins/`
- Validate fallback GCC/G++ availability
- Set up environment variables (SPEC, PATH, LD_LIBRARY_PATH)
- Create/validate result directories

**Exit Codes**:
- 0: Configuration successful
- 1: SPEC CPU not found
- 2: Required tools missing
- 3: No valid compiler chain available

---

### 2. `build_spec_targets.sh`
**Purpose**: Build baseline and obfuscated SPEC CPU binaries

**Responsibilities**:
- Accept build type parameter (baseline or obfuscated)
- Accept obfuscation configuration name (if obfuscated)
- Apply correct compiler and flags based on toolchain rules
- Build selected benchmark targets
- Validate build success
- Store build artifacts with metadata

**Parameters**:
```
./build_spec_targets.sh <build_type> [config_name] [target_list]
  build_type:   baseline | obfuscated
  config_name:  (required for obfuscated) e.g., "layer1-2", "full-obf"
  target_list:  (optional) comma-separated benchmarks or 'all'
```

**Build Output**:
- Binaries stored in SPEC CPU build directory
- Build logs recorded with timestamps
- Metadata file: compiler version, flags, build duration

---

### 3. `run_spec_speed.sh`
**Purpose**: Execute SPEC CPU 2017 SPECspeed benchmarks

**Responsibilities**:
- Accept baseline or obfuscated result identifier
- Run SPEC CPU speed benchmarks via runcpu
- Collect raw performance metrics
- Store results in timestamped directory under `speed/`
- Generate execution logs

**Parameters**:
```
./run_spec_speed.sh <result_type> [config_name] [iterations] [benchmark_list]
  result_type:    baseline | obfuscated
  config_name:    (required for obfuscated)
  iterations:     number of runs per benchmark (default: 3)
  benchmark_list: comma-separated or 'all'
```

**Result Files**:
- Raw results XML from SPEC CPU
- Execution logs with timing information
- Performance metrics (SPECspeed score, component scores)

---

### 4. `run_spec_rate.sh`
**Purpose**: Execute SPEC CPU 2017 SPECrate benchmarks

**Responsibilities**:
- Accept baseline or obfuscated result identifier
- Run SPEC CPU rate benchmarks via runcpu
- Parallelize across available CPU cores
- Collect throughput metrics
- Store results in timestamped directory under `rate/`

**Parameters**:
```
./run_spec_rate.sh <result_type> [config_name] [num_threads] [benchmark_list]
  result_type:   baseline | obfuscated
  config_name:   (required for obfuscated)
  num_threads:   threads per benchmark (default: system CPU count)
  benchmark_list: comma-separated or 'all'
```

**Result Files**:
- Raw results XML from SPEC CPU
- Execution logs with parallel execution details
- Throughput metrics (SPECrate score, component scores)

---

### 5. `collect_spec_metrics.py`
**Purpose**: Extract and aggregate SPEC CPU results into structured metrics

**Responsibilities**:
- Parse SPEC CPU raw result files (XML)
- Extract performance metrics: base score, peak score, component scores
- Extract resource metrics: runtime, memory usage, power (if available)
- Calculate derived metrics: geometric mean, standard deviation
- Generate machine-readable output (JSON, CSV)
- Handle missing/incomplete results gracefully

**Input**:
```python
collect_spec_metrics.py <result_directory> [--format json|csv|both]
  result_directory: path to speed/ or rate/ results
  --format: output format (default: json)
```

**Output Format**:
```json
{
  "benchmark_name": "...",
  "test_type": "speed|rate",
  "compiler": "clang|gcc",
  "compiler_version": "...",
  "optimization_flags": "...",
  "base_score": 123.45,
  "peak_score": 128.56,
  "runtime_seconds": 3600.5,
  "component_scores": {...},
  "error_rate": 0.0,
  "timestamp": "2025-12-06T12:30:45Z"
}
```

---

### 6. `compare_spec_results.py`
**Purpose**: Generate detailed comparison reports between baseline and obfuscated results

**Responsibilities**:
- Load baseline results from latest baseline run
- Load obfuscated results from specified configuration
- Calculate performance deltas (absolute and percentage)
- Generate performance impact summary
- Create comparison tables and charts (if graphing enabled)
- Identify regressions and improvements
- Generate HTML report for easy viewing

**Input**:
```python
compare_spec_results.py <baseline_path> <obfuscated_path> [--config config_name] [--format html|json|both]
  baseline_path: path to baseline results directory
  obfuscated_path: path to obfuscated results directory
  --config: obfuscation configuration name (for report title)
  --format: output format (default: html)
```

**Output Files**:
- `comparison_report.html` - Interactive comparison report
- `comparison_metrics.json` - Machine-readable metrics
- `regression_analysis.csv` - Detailed per-benchmark analysis

**Metrics Included**:
- Performance impact: % slowdown/speedup per benchmark
- Binary size ratio: obfuscated vs baseline
- Obfuscation overhead: estimated cost of transformations
- Variability: standard deviation across runs

---

## Configuration Files

### `linux-x86_64.cfg`
**Purpose**: SPEC CPU 2017 compiler configuration for Linux x86_64

**Contents**:
- Compiler identification (Clang or GCC)
- Optimization flags baseline set
- Obfuscation flags additions
- Baseline build rule definition
- Peak build rule definition (optional)
- Portability settings
- Output format specifications

**Key Sections**:
```
[system]
label = LLVM Obfuscator SPEC CPU 2017

[compiler_setup]
cc = <path_to_clang_or_gcc>
cxx = <path_to_clang++_or_g++>
fc = <path_to_gfortran>

[default_pcp]
SPEC = <SPEC_CPU_2017_ROOT>
EXTRA_FLAGS = -m64

[baseline]
CC = ${cc}
CXXFLAGS = -O3 ...
CFLAGS = -O3 ...

[obfuscation]
CC_OBF = ${clang_custom}
CXXFLAGS_OBF = -O3 <obfuscation_flags>
CFLAGS_OBF = -O3 <obfuscation_flags>
```

---

## README Documentation

### Contents Overview:
1. **Quick Start Guide**
   - Prerequisites and dependencies
   - Installation instructions
   - First run example

2. **Detailed Usage**
   - Script descriptions with examples
   - Parameter reference
   - Output interpretation

3. **Toolchain Configuration**
   - Custom Clang plugin detection
   - GCC fallback behavior
   - Compiler verification

4. **Result Analysis**
   - Interpreting benchmark scores
   - Understanding performance metrics
   - Comparing configurations

5. **Troubleshooting**
   - Common issues and solutions
   - Log file locations
   - Compiler detection problems

6. **Advanced Usage**
   - Custom benchmark selection
   - Partial runs
   - Manual result processing

---

## Important Notes

### Never CI/CD Integration
- ✗ Do NOT add spec_cpu tasks to GitHub Actions
- ✗ Do NOT add spec_cpu to automated pipelines
- ✗ Results directory is intentionally excluded from versioning
- ✓ Module is purely for local development and analysis

### Compiler Detection Logic
```
IF custom_clang exists in plugins/:
    BASELINE: use custom clang (preferred) OR system gcc with -O3
    OBFUSCATED: MUST use custom clang (fail if unavailable)
ELSE:
    BASELINE: use system gcc with -O3
    OBFUSCATED: FAIL with error message (no custom clang available)
```

### Result Retention
- Results are not versioned (add `spec_cpu/results/` to .gitignore)
- Old results should be manually cleaned up
- Keep comparison reports for documentation

---

## Implementation Checklist

The following files will be generated in future prompts:

### Shell Scripts (4 files)
- [ ] `scripts/configure_spec_cpu.sh` - Environment setup and validation
- [ ] `scripts/build_spec_targets.sh` - Binary compilation (baseline + obfuscated)
- [ ] `scripts/run_spec_speed.sh` - SPECspeed execution
- [ ] `scripts/run_spec_rate.sh` - SPECrate execution

### Python Scripts (2 files)
- [ ] `scripts/collect_spec_metrics.py` - Results aggregation and extraction
- [ ] `scripts/compare_spec_results.py` - Comparison report generation

### Configuration Files (1 file)
- [ ] `configs/linux-x86_64.cfg` - SPEC CPU compiler configuration

### Documentation (1 file)
- [ ] `README.md` - User guide and usage instructions

---

## Total Implementation: 8 Files

**Status**: Structure and specification complete. Ready for code generation in future prompts.

---

**Last Updated**: 2025-12-06
**Module Version**: 1.0 (Specification)
**Status**: Specification Complete - Awaiting Implementation
