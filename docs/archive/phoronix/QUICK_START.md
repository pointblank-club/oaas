# Quick Start - Obfuscation Test Suite

## TL;DR

```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline obfuscated ./results/
```

That's it. Everything else is automatic.

---

## What You Get

A new directory with this structure:

```
./results/obfuscation_analysis_baseline_vs_obfuscated_20251206_023620/
├── INDEX.md                    # ← Start here
├── metrics/
│   ├── metrics.json
│   ├── metrics.csv
│   └── metrics.md
├── security/
│   └── security_analysis.json
├── reports/
│   ├── final_report.json
│   ├── final_report.md
│   ├── final_report.html
│   └── final_report.csv
└── logs/
    ├── execution.log
    ├── metrics.log
    ├── security.log
    └── report.log
```

---

## How to Use

### Step 1: Run Test
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh ./baseline ./obfuscated ./results/
```

### Step 2: Check Results
Each run creates a **new timestamped directory**. No mixing of results.

```bash
ls ./results/
# obfuscation_analysis_baseline_vs_obfuscated_20251206_023620/
# obfuscation_analysis_baseline_vs_obfuscated_20251206_025000/
```

### Step 3: View Reports
```bash
# Quick summary
cat ./results/obfuscation_analysis_*/INDEX.md

# Detailed analysis
cat ./results/obfuscation_analysis_*/reports/final_report.md

# JSON for scripts
cat ./results/obfuscation_analysis_*/reports/final_report.json

# HTML in browser (open in browser)
./results/obfuscation_analysis_*/reports/final_report.html
```

---

## Examples

### Test Any Two Binaries
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    /path/to/baseline \
    /path/to/obfuscated \
    ./my_results/
```

### Test Your Recent Builds
```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    ./build/app \
    ./build/app.obfuscated \
    ./analysis/
```

### Track Multiple Tests
```bash
# Run test 1
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline v1 ./results/

# Run test 2
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline v2 ./results/

# Each gets its own timestamped directory!
ls ./results/
```

---

## What Gets Analyzed

| Metric | What It Shows | File |
|--------|--------------|------|
| **File Size** | Code bloat from obfuscation | metrics.json |
| **Function Count** | Symbol extraction capability | metrics.json |
| **Instructions** | Instruction bloat | metrics.json |
| **CFG Complexity** | Control flow changes | security.json |
| **Entropy** | Code randomness | metrics.json |
| **Readability Score** | Reverse engineering difficulty | security.json |
| **Final Score** | Overall obfuscation effectiveness | final_report.json |

---

## Reports Explained

### 📋 INDEX.md
Navigation guide. Start here.

### 📊 metrics.json / metrics.csv / metrics.md
Binary metrics:
- File sizes
- Function count
- Instruction counts
- Comparisons between baseline and obfuscated

### 🔒 security_analysis.json
Obfuscation difficulty:
- CFG complexity
- String/symbol obfuscation
- Decompilation difficulty score

### 📄 final_report.json / final_report.md / final_report.html
Complete analysis:
- All metrics combined
- Final obfuscation score (0-10)
- Ratings (A+ to D)
- Interpretation

### 📝 logs/
For troubleshooting:
- execution.log - Full timeline
- metrics.log - Metrics details
- security.log - Analysis details
- report.log - Report generation

---

## Directory Organization

Every run creates a **new directory** with timestamp:

```
obfuscation_analysis_{baseline}_{obfuscated}_{YYYYMMDD_HHMMSS}/
```

This means:
- ✅ No overwriting old results
- ✅ Track multiple tests over time
- ✅ Easy to compare results
- ✅ Organized by date/time automatically

---

## Requirements

### Must Have
- Linux system
- Python 3
- Bash
- `file`, `nm`, `objdump`, `readelf` (usually pre-installed)

### Optional (Better Accuracy)
- Ghidra (at `/opt/ghidra`)
  - Gives 85-90% accuracy
  - Without it: 40% accuracy

### To Install Ghidra
```bash
bash phoronix/scripts/option2_ghidra_integration.sh
```

---

## Troubleshooting

### "Binary not found"
```bash
# Make sure path is correct
ls -lh ./your_binary
```

### "Permission denied"
```bash
# Make script executable
chmod +x phoronix/scripts/run_obfuscation_test_suite.sh
```

### Empty reports
```bash
# Check logs
cat obfuscation_analysis_*/logs/execution.log

# Verify binary is valid
file ./baseline
file ./obfuscated
```

---

## Next Steps

1. **Run a test** - `bash run_obfuscation_test_suite.sh baseline obfuscated ./results/`
2. **View results** - `cat ./results/obfuscation_analysis_*/INDEX.md`
3. **Read full guide** - `cat phoronix/TEST_SUITE_USAGE.md`
4. **Automate tests** - Use in CI/CD, cron jobs, etc.

---

## Command Reference

```bash
# Single test
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline obfuscated

# Custom output directory
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline obfuscated ./results/

# Full path example
bash phoronix/scripts/run_obfuscation_test_suite.sh \
    /home/user/build/app \
    /home/user/build/app.obfuscated \
    /home/user/test-results/
```

---

Done! Run your first test now:

```bash
bash phoronix/scripts/run_obfuscation_test_suite.sh baseline obfuscated results/
```
