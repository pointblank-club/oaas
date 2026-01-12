# Testing & Verification

This document provides a single source of truth for validating changes.  The removed `README_TESTING.md`, `TESTING_GUIDE.md`, `QUICK_TEST*.md`, and `TEST_SUITE_AUDIT_REPORT.md` have been condensed here.

## 1. Quick Local Checks

```bash
# Install dependencies
cd cmd/llvm-obfuscator
pip install -r requirements.txt

# Run unit tests
pytest

# Run CLI smoke test
python -m cmd.llvm_obfuscator.cli.obfuscate compile tests/test_simple.c \
  --output build/test \
  --level 2 --string-encryption --report-formats json
```

Recommended pylint/mypy invocations are defined in `pyproject.toml`; run them before posting a PR when you touch Python files.

## 2. MLIR Library Tests

```bash
cd mlir-obs
./build.sh              # configure + build libMLIRObfuscation
./test.sh               # exercises string/symbol passes on sample MLIR
```

For Polygeist/ClangIR experiments, follow `docs/PIPELINES.md` and capture results in the PR summary (the old `SETUP_POLYGEIST.md` has been retired).

## 3. Integration Workflows

### CLI → Reports

Use the CLI to compile the demo programs in `cmd/llvm-obfuscator/tests/` and inspect the `reports/` output to ensure metrics remain consistent.

```bash
python -m cmd.llvm_obfuscator.cli.obfuscate compile tests/test_obfuscator.c \
  --output build/obf_demo --level 3 \
  --enable-symbol-obfuscation --string-encryption --report-formats json,md
```

### Phoronix Harness

1. Build two binaries (baseline vs obfuscated).
2. Run `bash phoronix/scripts/run_obfuscation_test_suite.sh baseline obf results/` to generate the full security analysis.
3. Optional: run `bash phoronix/scripts/run_pts_tests.sh --automatic` to capture performance deltas.

Attach the resulting Markdown summary when validating large pass changes.

### Jotai Benchmarks

`core/jotai_benchmark.py` orchestrates the public Jotai suite.  The benchmark repo is cached under `~/.cache/llvm-obfuscator/jotai-benchmarks`.

```bash
python -m cmd.llvm_obfuscator.cli.obfuscate benchmark jotai \
  --output build/jotai_ci --level 2 --string-encryption --skip-compilation-errors
```

The command reports pass/fail counts plus functional mismatches.  Use this when modifying control-flow transforms.

## 4. Windows Lifting Regression Tests

When touching scripts in `binary_obfuscation_pipeline/`:

1. Use `binary_obfuscation_pipeline/windows_build/compile_windows_binary.py` to build a simple PE binary.
2. Run the headless Ghidra lifter service (Docker compose) and export a CFG.
3. Execute `mcsema_impl/lifter/run_lift.sh` and `.../convert_ir_version.sh` to regenerate LLVM 22 bitcode.
4. Invoke `mcsema_impl/ollvm_stage/run_ollvm.sh` with the minimal `passes_config.json` to ensure the obfuscated bitcode is emitted.

Document any constraints in the PR so downstream users know how to reproduce the environment.

## 5. Release Checklist

1. `pytest` green.
2. `mlir-obs/test.sh` green.
3. CLI smoke test for Linux/macOS.
4. At least one Phoronix security analysis (`run_obfuscation_test_suite.sh`).
5. Optional performance validation via Phoronix Test Suite (for pass-heavy releases).
6. Updated documentation if new flags/passes were added.

Keep this list in sync when adding new tooling or automation.
