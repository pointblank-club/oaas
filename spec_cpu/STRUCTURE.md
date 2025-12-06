# SPEC CPU Module - Directory Structure & File Manifest

## Complete Directory Tree

```
spec_cpu/
│
├── 📋 SPECIFICATION.md           (Detailed technical specification)
├── 📋 STRUCTURE.md               (This file - visual reference)
├── 📚 README.md                  (User guide - TO BE CREATED)
│
├── 📁 configs/
│   └── 🔧 linux-x86_64.cfg       (SPEC CPU compiler configuration - TO BE CREATED)
│
├── 📁 scripts/
│   ├── 🔨 configure_spec_cpu.sh  (Environment setup - TO BE CREATED)
│   ├── 🔨 build_spec_targets.sh  (Build binaries - TO BE CREATED)
│   ├── 🔨 run_spec_speed.sh      (Execute SPECspeed - TO BE CREATED)
│   ├── 🔨 run_spec_rate.sh       (Execute SPECrate - TO BE CREATED)
│   ├── 🐍 collect_spec_metrics.py (Results aggregation - TO BE CREATED)
│   └── 🐍 compare_spec_results.py (Comparison reports - TO BE CREATED)
│
└── 📁 results/                   (Local results storage - NOT VERSIONED)
    ├── 📁 baseline/
    │   ├── 📁 <timestamp_1>/
    │   │   ├── 📁 speed/         (SPECspeed results)
    │   │   └── 📁 rate/          (SPECrate results)
    │   ├── 📁 <timestamp_2>/
    │   │   ├── 📁 speed/
    │   │   └── 📁 rate/
    │   └── ...
    │
    ├── 📁 obfuscated/
    │   ├── 📁 layer1-2/          (Configuration name)
    │   │   ├── 📁 <timestamp_1>/
    │   │   │   ├── 📁 speed/
    │   │   │   └── 📁 rate/
    │   │   ├── 📁 <timestamp_2>/
    │   │   │   ├── 📁 speed/
    │   │   │   └── 📁 rate/
    │   │   └── ...
    │   ├── 📁 full-obf/          (Another configuration)
    │   │   ├── 📁 <timestamp_1>/
    │   │   │   ├── 📁 speed/
    │   │   │   └── 📁 rate/
    │   │   └── ...
    │   └── ...
    │
    └── 📁 comparisons/
        ├── 📁 layer1-2/          (Configuration name)
        │   ├── 📁 <timestamp_1>/
        │   │   ├── 📄 comparison_report.html
        │   │   ├── 📄 comparison_metrics.json
        │   │   └── 📄 regression_analysis.csv
        │   ├── 📁 <timestamp_2>/
        │   │   ├── 📄 comparison_report.html
        │   │   ├── 📄 comparison_metrics.json
        │   │   └── 📄 regression_analysis.csv
        │   └── ...
        ├── 📁 full-obf/
        │   ├── 📁 <timestamp_1>/
        │   │   ├── 📄 comparison_report.html
        │   │   ├── 📄 comparison_metrics.json
        │   │   └── 📄 regression_analysis.csv
        │   └── ...
        └── ...
```

---

## File Manifest & Generation Order

### Phase 1: Configuration & Setup
**Generated in Prompt 1:**

| # | File | Type | Purpose | Status |
|---|------|------|---------|--------|
| 1 | `configs/linux-x86_64.cfg` | Config | SPEC CPU compiler configuration | ⏳ TO CREATE |
| 2 | `scripts/configure_spec_cpu.sh` | Shell | Environment validation & setup | ⏳ TO CREATE |

### Phase 2: Build & Execution
**Generated in Prompt 2:**

| # | File | Type | Purpose | Status |
|---|------|------|---------|--------|
| 3 | `scripts/build_spec_targets.sh` | Shell | Build baseline and obfuscated binaries | ⏳ TO CREATE |
| 4 | `scripts/run_spec_speed.sh` | Shell | Execute SPECspeed benchmarks | ⏳ TO CREATE |
| 5 | `scripts/run_spec_rate.sh` | Shell | Execute SPECrate benchmarks | ⏳ TO CREATE |

### Phase 3: Analysis & Reporting
**Generated in Prompt 3:**

| # | File | Type | Purpose | Status |
|---|------|------|---------|--------|
| 6 | `scripts/collect_spec_metrics.py` | Python | Extract and aggregate results | ⏳ TO CREATE |
| 7 | `scripts/compare_spec_results.py` | Python | Generate comparison reports | ⏳ TO CREATE |

### Phase 4: Documentation
**Generated in Prompt 4:**

| # | File | Type | Purpose | Status |
|---|------|------|---------|--------|
| 8 | `README.md` | Markdown | User guide and quick reference | ⏳ TO CREATE |

---

## Key Design Elements

### Toolchain Intelligence
```
Compiler Detection Flow:
┌─────────────────────────────────────────────────────┐
│ Check for plugins/clang and plugins/clang++         │
├─────────────────────────────────────────────────────┤
│ YES: Custom Clang Found                             │
│   ├─ BASELINE: Use custom clang OR gcc with -O3     │
│   └─ OBFUSCATED: Use custom clang (REQUIRED)        │
├─────────────────────────────────────────────────────┤
│ NO: No Custom Clang                                 │
│   ├─ BASELINE: Use gcc with -O3                     │
│   └─ OBFUSCATED: FAIL with error                    │
└─────────────────────────────────────────────────────┘
```

### Results Organization Strategy
```
By Build Type:
├─ baseline/         (Single canonical baseline)
│  └─ <timestamp>/   (When baseline was built)
│
└─ obfuscated/       (Multiple configurations)
   ├─ config1/       (e.g., "layer1-2")
   │  ├─ <ts1>/      (Run 1)
   │  ├─ <ts2>/      (Run 2)
   │  └─ <ts3>/      (Run 3)
   │
   └─ config2/       (e.g., "full-obf")
      ├─ <ts1>/      (Run 1)
      └─ <ts2>/      (Run 2)
```

### Benchmark Structure
```
Each benchmark run creates:
├─ speed/           SPECspeed (single-threaded)
│  ├─ 500.perlbench_r/
│  ├─ 502.gcc_r/
│  ├─ 505.mcf_r/
│  └─ ... (26 INT + 28 FP = 54 total)
│
└─ rate/            SPECrate (multi-threaded)
   ├─ 500.perlbench_r/
   ├─ 502.gcc_r/
   ├─ 505.mcf_r/
   └─ ... (26 INT + 28 FP = 54 total)
```

---

## Execution Workflows

### Workflow 1: Baseline Benchmark
```
1. configure_spec_cpu.sh
   └─> Validate SPEC CPU installation
       Detect compiler toolchain
       Setup environment

2. build_spec_targets.sh baseline
   └─> Compile benchmarks with baseline flags (-O3)
       Use custom clang OR gcc

3. run_spec_speed.sh baseline
   └─> Execute SPECspeed tests
       Store in: results/baseline/<timestamp>/speed/

4. run_spec_rate.sh baseline
   └─> Execute SPECrate tests
       Store in: results/baseline/<timestamp>/rate/

5. collect_spec_metrics.py results/baseline/<timestamp>/
   └─> Extract metrics to JSON/CSV
       Generate summary statistics
```

### Workflow 2: Obfuscated Benchmark + Comparison
```
1. configure_spec_cpu.sh
   └─> Validate SPEC CPU installation (reuse from baseline)

2. build_spec_targets.sh obfuscated layer1-2
   └─> Compile with obfuscation flags
       MUST use custom clang from plugins/
       Fail if unavailable

3. run_spec_speed.sh obfuscated layer1-2
   └─> Execute SPECspeed tests
       Store in: results/obfuscated/layer1-2/<timestamp>/speed/

4. run_spec_rate.sh obfuscated layer1-2
   └─> Execute SPECrate tests
       Store in: results/obfuscated/layer1-2/<timestamp>/rate/

5. collect_spec_metrics.py results/obfuscated/layer1-2/<timestamp>/
   └─> Extract metrics to JSON/CSV

6. compare_spec_results.py results/baseline/<latest>/ results/obfuscated/layer1-2/<latest>/
   └─> Generate comparison report
       Store in: results/comparisons/layer1-2/<timestamp>/
       Create HTML, JSON, and CSV outputs
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ Input: SPEC CPU 2017 Installation + LLVM Obfuscator Plugins │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    [Baseline]                      [Obfuscated]
    configure.sh                    configure.sh
         │                               │
         ├─> build (gcc -O3)            ├─> build (clang obf flags)
         │                               │
         ├─> run_spec_speed.sh          ├─> run_spec_speed.sh
         │       │                       │       │
         │       └─> collect_metrics.py  │       └─> collect_metrics.py
         │               │               │               │
         │               ▼               │               ▼
         │          results/baseline/    │          results/obfuscated/
         │                               │
         └───────────────┬───────────────┘
                         │
                    [Comparison]
                    compare_spec_results.py
                         │
                         ▼
        results/comparisons/<config>/<timestamp>/
        ├── comparison_report.html
        ├── comparison_metrics.json
        └── regression_analysis.csv
```

---

## Implementation Checklist

### ✅ Complete
- [x] Directory structure created
- [x] SPECIFICATION.md written
- [x] STRUCTURE.md written (this file)
- [x] Architecture designed
- [x] Toolchain rules documented

### ⏳ Pending (Future Prompts)

**Prompt 1 - Setup & Configuration:**
- [ ] Create `configs/linux-x86_64.cfg`
- [ ] Create `scripts/configure_spec_cpu.sh`

**Prompt 2 - Build & Execution:**
- [ ] Create `scripts/build_spec_targets.sh`
- [ ] Create `scripts/run_spec_speed.sh`
- [ ] Create `scripts/run_spec_rate.sh`

**Prompt 3 - Analysis & Reporting:**
- [ ] Create `scripts/collect_spec_metrics.py`
- [ ] Create `scripts/compare_spec_results.py`

**Prompt 4 - Documentation:**
- [ ] Create `README.md`

---

## Integration with Existing Modules

### Compatible With:
- ✅ Phoronix Test Suite (`phoronix/`)
  - Independent execution model
  - Separate results directories

- ✅ LLVM Obfuscator (`cmd/llvm-obfuscator/`)
  - Uses plugins from `cmd/llvm-obfuscator/plugins/`
  - Works with backend API for obfuscation configs

- ✅ Existing Metrics (`obfuscation metrics`, `decompilation metrics`)
  - Can use same SPEC CPU binaries for analysis

### NOT Integrated With:
- ✗ CI/CD Pipelines (intentionally excluded)
- ✗ GitHub Actions workflows
- ✗ Automated testing systems

---

## Notes for Implementation

1. **Timestamp Format**: Use `YYYY-MM-DDTHH:MM:SSZ` (ISO 8601) for consistency
2. **Error Handling**: All scripts should have clear error messages for compiler detection failures
3. **Logging**: Each script should create a `.log` file in the same directory as results
4. **Idempotency**: Scripts should be safe to re-run without data loss
5. **Documentation**: Every script needs inline help (`script.sh --help`)

---

**Status**: ✅ Specification & Structure Complete
**Ready for**: Code Implementation Phase
**Estimated Implementation Time**: 3-4 prompts

---

**Created**: 2025-12-06
**Version**: 1.0
