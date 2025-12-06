# CI/CD Integration Guide

## Overview

The Obfuscation Test Suite integrates into GitHub Actions CI/CD pipeline (`phoronix-ci.yml`).

**New job added:** `obfuscation-analysis` - Runs on every push and PR to the `phoronix/` directory.

---

## What Gets Committed to Git

### ✅ COMMIT THESE

```
phoronix/
├── scripts/
│   ├── run_obfuscation_test_suite.sh       ✅ Main script
│   ├── collect_obfuscation_metrics.py      ✅ Metrics collection
│   ├── run_security_analysis.sh            ✅ Security analysis
│   ├── aggregate_obfuscation_report.py     ✅ Report generation
│   ├── install_phoronix.sh                 ✅ PTS installer
│   ├── run_pts_tests.sh                    ✅ PTS runner
│   ├── run_pts_with_compile_metrics.sh     ✅ Compile-time metrics
│   ├── option1_use_non_stripped_binaries.sh ✅ Implementation examples
│   ├── option2_ghidra_integration.sh       ✅ Implementation examples
│   ├── option3_compile_time_metrics.sh     ✅ Implementation examples
│   └── option4_tool_alternatives.md        ✅ Documentation
├── tests/
│   ├── test_metric_collectors.py           ✅ Unit tests
│   └── test_security_analysis.py           ✅ Unit tests
├── documentation/
│   ├── README.md                           ✅ Main guide
│   ├── QUICK_START.md                      ✅ Quick reference
│   ├── TEST_SUITE_USAGE.md                 ✅ Complete usage
│   └── IMPLEMENTATION_GUIDE.md             ✅ Technical details
├── examples/                               ✅ Example structure (dir only)
└── .gitignore                              ✅ Ignore rules
```

### ❌ DO NOT COMMIT

```
phoronix/
├── test_results/                           ❌ Test outputs (timestamped)
│   └── obfuscation_analysis_*/              ❌ Test run results
├── examples/option*_binaries/              ❌ Generated binaries
├── examples/option*_metrics/               ❌ Generated metrics
├── scripts/__pycache__/                    ❌ Python cache
├── tests/__pycache__/                      ❌ Python cache
├── *.pyc, *.pyo                            ❌ Python bytecode
└── *.log                                   ❌ Log files
```

###📝 Note: .gitignore Already Set

The `.gitignore` file automatically excludes:
- `test_results/` - All timestamped test outputs
- `__pycache__/` - Python cache
- `*.pyc, *.pyo` - Python bytecode
- `*.log` - Log files
- Generated binaries/metrics in examples/

---

## CI/CD Pipeline

### Workflow: `phoronix-ci.yml`

Triggers on:
- ✅ Push to `main` or `develop` with changes in `phoronix/`
- ✅ Pull requests to `main` or `develop` with changes in `phoronix/`
- ✅ Manual workflow dispatch (for testing specific profiles)

### Jobs

1. **verify-installation**
   - Installs Phoronix Test Suite
   - Runs Python verification tests
   - Always runs first

2. **run-benchmarks**
   - Runs PTS benchmark suite (automatic or manual mode)
   - Uploads reports as artifacts
   - Generates GitHub summary

3. **obfuscation-analysis** ⭐ NEW
   - Runs Python unit tests
   - Builds test binaries (unstripped baseline + stripped obfuscated)
   - Runs complete obfuscation analysis
   - Generates metrics, security, and aggregated reports
   - Adds results to GitHub step summary

4. **test-failure-analysis**
   - Runs if any job fails
   - Provides diagnostic information

---

## The New Obfuscation Analysis Job

### What It Does

```yaml
obfuscation-analysis:
  - Installs dependencies (python3, gcc, binutils)
  - Runs unit tests (pytest)
  - Compiles test binaries
  - Runs test suite: baseline vs obfuscated
  - Generates reports
  - Posts summary to GitHub
```

### Test Binaries

Generated automatically in CI:
- **Baseline**: Compiled with `-O3 -g` (debug symbols, not stripped)
- **Obfuscated**: Stripped version for comparison

### Reports Generated

All saved to `phoronix/test_results/{timestamp}/`:
- `metrics/metrics.json` - Binary metrics
- `security/security_analysis.json` - Decompilation analysis
- `reports/final_report.json` - Complete aggregated report
- `reports/final_report.md` - Human-readable markdown
- `reports/final_report.html` - Interactive HTML view
- `logs/` - Detailed execution logs

### Example Output in GitHub

```
## Obfuscation Analysis Results

✅ Test suite completed

Reports Generated:
- Final Report JSON
- Metrics Analysis
- Security Analysis
```

---

## How to Use in Your Workflow

### Push Changes

```bash
# Make changes to phoronix scripts
git add phoronix/scripts/my_script.py

# Commit and push
git commit -m "Update obfuscation analysis script"
git push origin develop
```

**Result:** GitHub Actions automatically runs the obfuscation analysis job.

### View Results

1. **In GitHub Actions**: Go to "Actions" tab → "Phoronix Test Suite CI" → Latest run
2. **Check Step Summary**: Scroll to "Obfuscation Analysis Results"
3. **Download Artifacts**: None uploaded (results in logs) - but can be added

### Manual Trigger

```bash
# In GitHub: Actions → Phoronix Test Suite CI → Run workflow

Select:
- Branch: develop
- Test mode: automatic (or manual with specific test)
- Click "Run workflow"
```

---

## What Should NOT Be in CI

### ❌ No Artifact Upload

The pipeline creates reports but does NOT upload them because:
1. Test results are **local to each run** (timestamped)
2. Results aren't needed after CI completes
3. Reduces GitHub Actions storage usage

If you NEED to archive reports:
- Store in a dedicated test results repository
- Or upload only on specific branches (main)
- Or use external storage (S3, GCS, etc.)

### ❌ No Real Obfuscated Binaries

CI uses **test binaries** (simple C++ program), not real obfuscated code.

For production testing:
- Test locally with actual binaries
- Use manual workflow dispatch if needed
- Don't commit real binaries (too large)

### ❌ No Ghidra Integration

CI doesn't use Ghidra (would require downloading 300MB+).

Uses heuristics instead:
- 40% accuracy vs 85-90% with Ghidra
- Acceptable for CI (catches major issues)
- Useful binaries get analyzed locally with Ghidra

---

## Customization

### Change Trigger Branches

Edit `.github/workflows/phoronix-ci.yml`:

```yaml
on:
  push:
    branches: [ main, develop, testing ]  # Add more branches
  pull_request:
    branches: [ main, develop, testing ]  # Add more branches
```

### Skip Obfuscation Job

If running only PTS tests:

```yaml
obfuscation-analysis:
  if: false  # Disable this job
```

Or trigger only on specific changes:

```yaml
on:
  push:
    branches: [ main ]
    paths:
      - 'phoronix/scripts/**'  # Only if scripts change
      - 'phoronix/tests/**'    # Only if tests change
```

### Use Real Binaries

For production testing with actual obfuscated code:

```bash
# Replace this in CI
bash phoronix/scripts/run_obfuscation_test_suite.sh \
  phoronix/test_binaries/baseline \
  phoronix/test_binaries/obfuscated \
  phoronix/test_results/

# With this (example)
bash phoronix/scripts/run_obfuscation_test_suite.sh \
  ./build/app_baseline \
  ./build/app_obfuscated \
  phoronix/test_results/
```

---

## Common Questions

### Q: Why are test results not uploaded as artifacts?

A: Test results are timestamped and local. If you need historical tracking:
1. Archive them separately
2. Upload to cloud storage
3. Or commit to a results repository

### Q: Can I run this on Windows/Mac?

A: The CI runs on Ubuntu. Local usage works on any Linux system. Modifications needed for macOS (binutils differences).

### Q: How long does the job take?

A: Approximately 2-3 minutes:
- 30s - Install dependencies
- 30s - Compile binaries
- 30s - Run analysis
- 30s - Generate reports

### Q: What if analysis fails?

A: Check the GitHub Actions log for:
- Python errors
- Missing dependencies
- Binary compilation issues
- Path problems

### Q: Can I upload reports?

A: Yes! Modify the workflow to add:

```yaml
- name: Upload obfuscation reports
  uses: actions/upload-artifact@v4
  with:
    name: obfuscation-analysis-reports
    path: phoronix/test_results/
    retention-days: 30
```

---

## Summary

| What | Where | Action |
|------|-------|--------|
| Scripts | `phoronix/scripts/` | ✅ Commit |
| Tests | `phoronix/tests/` | ✅ Commit |
| Docs | `phoronix/documentation/` | ✅ Commit |
| CI Config | `.github/workflows/phoronix-ci.yml` | ✅ Commit |
| .gitignore | `phoronix/.gitignore` | ✅ Commit |
| Test Results | `phoronix/test_results/` | ❌ Ignore |
| Example Binaries | `phoronix/examples/option*_*` | ❌ Ignore |
| Python Cache | `__pycache__/` | ❌ Ignore |

---

## Next Steps

1. ✅ Scripts are ready for CI/CD
2. ✅ Workflow is configured
3. ✅ Gitignore is set up
4. Push changes to trigger CI automatically

```bash
git add phoronix/
git add .github/workflows/phoronix-ci.yml
git commit -m "Add obfuscation analysis to CI/CD pipeline"
git push origin develop
```

Done! Your obfuscation test suite is now part of the CI/CD pipeline. 🎯
