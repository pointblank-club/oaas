# Contributing Guide

Welcome!  This document streamlines the onboarding process for new contributors and prospective Google Summer of Code (GSoC) applicants.  It describes how to set up a development environment, coding expectations, and the review workflow.

## 1. Prerequisites

- Python 3.10+
- `clang`/`llvm` 15+ (LLVM 22 recommended to match bundled plugins)
- CMake 3.20+
- `pip` + virtualenv or `uv`
- Optional: Docker (for the Windows lifting services) and the Phoronix Test Suite

## 2. Pre-built Binaries

Pre-built LLVM obfuscator binaries are available in the [GitHub Releases](https://github.com/pointblank-club/oaas/releases/tag/v1.0.0-binaries). Download and extract to `cmd/llvm-obfuscator/plugins/` before building Docker images:

```bash
# Download pre-built binaries
wget https://github.com/pointblank-club/oaas/releases/download/v1.0.0-binaries/LLVM-bin.zip
unzip LLVM-bin.zip -d cmd/llvm-obfuscator/plugins/
```

## 3. Repository Setup

```bash
git clone https://github.com/<org>/llvm-obfuscator.git
cd llvm-obfuscator

# Use a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Python dependencies
pip install -r cmd/llvm-obfuscator/requirements.txt

# Build MLIR passes
cd mlir-obs
./build.sh
cd ..
```

`mlir-obs/build/lib/` now contains `libMLIRObfuscation.*`, which the CLI/API load automatically.

## 4. Running the CLI & Tests

```bash
# Smoke test
python -m cmd.llvm_obfuscator.cli.obfuscate compile cmd/llvm-obfuscator/tests/test_simple.c \
  --output build/demo --string-encryption --enable-symbol-obfuscation

# Run unit tests
cd cmd/llvm-obfuscator
pytest

# Run MLIR tests
cd ../../mlir-obs
./test.sh
```

For deeper validation (benchmarks, security analysis, Windows pipeline) see `docs/TESTING.md` and `docs/PIPELINES.md`.

## 5. Development Workflow

1. **Create a branch** – `git checkout -b feature/my-change`.
2. **Make focused changes** – Reference `docs/ARCHITECTURE.md` to understand where your change belongs.
3. **Update docs/tests** – Every new flag, pass, or pipeline tweak should update the relevant docs under `docs/` and include regression tests when possible.
4. **Run the checklist** – `pytest`, `mlir-obs/test.sh`, and at least one CLI smoke test before opening a PR.
5. **Submit PR** – Fill out the template, summarize the impact, attach relevant reports (JSON/MD) if you touched passes.

## 6. Coding Standards

- Python: keep functions small, prefer dataclasses / TypedDicts where helpful, and match existing logging style.
- C++ (MLIR): follow LLVM coding conventions.  Register new passes in `include/Obfuscator/Passes.h` and document the CLI flags they map to.
- Documentation: update or extend the consolidated docs in `docs/` rather than creating new scattered files.

## 7. Getting Involved (GSoC friendly)

- Browse `good-first-issue` and `gsoc-idea` labels.
- Join discussions in GitHub Discussions/Issues to understand roadmap priorities (e.g., new MLIR passes, Windows pipeline stability, benchmarking automation).
- Pair with maintainers via drafted PRs—ask for feedback early if you’re touching critical passes.

Thanks for contributing!  Keeping this guide updated makes it easier for the next wave of contributors to succeed.

## 8. CI/CD

All pull requests are automatically validated by our CI/CD pipeline:

- **Frontend Checks**: Linting and build verification
- **Backend Checks**: Python tests with coverage reporting
- **Auto-labeling**: PRs are automatically labeled based on changed files
- **Size Labels**: PRs receive size labels (XS/S/M/L/XL) based on lines changed

You can also use slash commands on issues and PRs. Type `/help` in a comment to see available commands.
