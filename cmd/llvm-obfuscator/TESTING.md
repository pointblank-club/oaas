# 🧪 LLVM Obfuscator - Complete Testing Guide

Comprehensive testing framework for validating obfuscation effectiveness, correctness, and performance.

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Jotai Benchmark Testing](#jotai-benchmark-testing)
5. [CI/CD Integration](#cicd-integration)
6. [Writing New Tests](#writing-new-tests)
7. [Performance Testing](#performance-testing)
8. [Docker Testing](#docker-testing)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd cmd/llvm-obfuscator
make setup
```

This will:
- Verify system requirements (clang, llvm, upx)
- Install Python dependencies
- Install test dependencies

### 2. Run Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific category
make test-unit
make test-integration
```

### 3. View Results

```bash
# Open coverage report
open htmlcov/index.html

# Open test report
open test-report.html
```

---

## 📁 Test Organization

```
cmd/llvm-obfuscator/
├── tests/                           # Test suite
│   ├── conftest.py                  # Pytest configuration
│   ├── test_config.py               # Configuration tests
│   ├── test_upx_packer.py           # UPX packer tests
│   ├── test_obfuscator_integration.py  # Integration tests
│   ├── requirements-test.txt        # Test dependencies
│   └── README.md                    # Test documentation
│
├── scripts/
│   └── test_on_jotai.sh             # Jotai benchmark testing
│
├── pytest.ini                       # Pytest configuration
├── Makefile                         # Test automation
└── TESTING.md                       # This file
```

### Test Categories

#### 1. **Unit Tests** (`test_config.py`, `test_upx_packer.py`)
- Test individual modules in isolation
- Fast execution (< 1 second per test)
- No external dependencies

#### 2. **Integration Tests** (`test_obfuscator_integration.py`)
- Test complete obfuscation pipeline
- Multiple layers (symbol, string, UPX)
- Correctness validation

#### 3. **Benchmark Tests** (`test_on_jotai.sh`)
- Real-world C code from Jotai collection
- 18,761+ executable benchmarks
- Effectiveness metrics

---

## 🏃 Running Tests

### Using Make (Recommended)

```bash
# Show all available commands
make help

# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage
make test-cov

# Run fast tests (skip slow ones)
make test-fast

# Run in parallel
make test-parallel

# Clean up
make clean
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/

# Run specific file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestPlatform::test_platform_values

# Run with markers
pytest -m "not slow"            # Skip slow tests
pytest -m "requires_upx"        # Only UPX tests
pytest -m "integration"         # Only integration tests

# Verbose output
pytest tests/ -vv

# Show print statements
pytest tests/ -s

# Drop into debugger on failure
pytest tests/ --pdb
```

### Test Markers

```python
@pytest.mark.slow               # Slow tests (> 1 min)
@pytest.mark.requires_upx       # Requires UPX
@pytest.mark.requires_ollvm     # Requires OLLVM plugin
@pytest.mark.integration        # Integration test
@pytest.mark.unit               # Unit test
```

---

## 📊 Jotai Benchmark Testing

[Jotai](https://github.com/lac-dcc/jotai-benchmarks) is a collection of 18,761+ real-world C benchmarks perfect for validating obfuscation.

### Quick Test (10 benchmarks)

```bash
make test-jotai
```

### Full Test (1000+ benchmarks)

```bash
make test-jotai-full
```

### Custom Test

```bash
cd scripts
./test_on_jotai.sh --max 100 --parallel 8
```

### Options

```bash
--max N          # Test only first N benchmarks
--parallel N     # Run N parallel jobs
--help           # Show help
```

### Output

The script generates:
1. **CSV Report** - `jotai_obfuscation_results/results.csv`
2. **HTML Report** - `jotai_obfuscation_results/report.html`
3. **Logs** - `jotai_obfuscation_results/logs/`
4. **Binaries** - `jotai_obfuscation_results/obfuscated/`

### Example Report

```
╔════════════════════════════════════════════════════════════════╗
║                     Test Summary                               ║
╠════════════════════════════════════════════════════════════════╣
║  Total Benchmarks:     100                                     ║
║  Successful:           95                                      ║
║  Failed:               5                                       ║
║  Success Rate:         95.0%                                   ║
║  Avg Original Size:    24 KB                                   ║
║  Avg Obfuscated:       56 KB                                   ║
║  Avg with UPX:         28 KB                                   ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🔄 CI/CD Integration

### GitHub Actions

Tests run automatically on:
- **Push to main/develop**
- **Pull requests**
- **Daily schedule** (2 AM UTC)
- **Manual trigger**

### Test Matrix

```
OS:       [ubuntu-latest, macos-latest]
Python:   ['3.10', '3.11', '3.12']
Jobs:     6 (2 OS × 3 Python)
```

### Workflow

```yaml
1. Setup (Install clang, llvm, upx)
2. Unit Tests
3. Integration Tests
4. Coverage (upload to Codecov)
5. Example Tests
6. Jotai Tests (limited)
7. Docker Build & Test
8. Security Scan (Trivy)
```

### Local CI Simulation

```bash
make ci-test
```

This runs the same sequence as CI locally.

### View CI Results

- **Actions**: https://github.com/SkySingh04/oaas/actions
- **Coverage**: https://codecov.io/gh/SkySingh04/oaas

---

## ✍️ Writing New Tests

### Unit Test Template

```python
"""Test my_module.py"""

import pytest
from core.my_module import MyClass

class TestMyClass:
    """Test suite for MyClass."""
    
    @pytest.fixture
    def my_instance(self):
        """Create instance for testing."""
        return MyClass()
    
    def test_basic_functionality(self, my_instance):
        """Test basic functionality."""
        # Arrange
        input_data = "test"
        
        # Act
        result = my_instance.process(input_data)
        
        # Assert
        assert result is not None
        assert result == expected_output
```

### Integration Test Template

```python
def test_full_pipeline(self, tmp_path):
    """Test complete obfuscation pipeline."""
    # Create test source
    source = tmp_path / "test.c"
    source.write_text("int main() { return 0; }")
    
    # Configure obfuscation
    config = ObfuscationConfig(
        level=ObfuscationLevel.MEDIUM,
        advanced=AdvancedConfiguration(
            string_encryption=True,
            upx_packing=UPXConfiguration(enabled=True)
        ),
        output=OutputConfiguration(directory=tmp_path / "output")
    )
    
    # Run obfuscation
    obfuscator = LLVMObfuscator()
    result = obfuscator.obfuscate(source, config)
    
    # Verify results
    assert result is not None
    assert Path(result["output_file"]).exists()
    
    # Test execution
    subprocess.run([result["output_file"]], check=True)
```

### Best Practices

1. **Use descriptive names**: `test_symbol_obfuscation_reduces_symbols`
2. **One assertion per test** (when possible)
3. **Use fixtures** for reusable setup
4. **Test edge cases** (empty input, invalid config, etc.)
5. **Clean up** temporary files (use `tmp_path` fixture)

---

## ⚡ Performance Testing

### Benchmark Tests

```bash
# Run benchmarks
make benchmark

# Compare with baseline
pytest tests/ --benchmark-save=baseline
# Make changes...
pytest tests/ --benchmark-compare=baseline
```

### Manual Performance Testing

```python
import time

def test_performance(sample_code):
    """Measure obfuscation time."""
    config = ObfuscationConfig(level=3)
    obfuscator = LLVMObfuscator()
    
    start = time.time()
    result = obfuscator.obfuscate(sample_code, config)
    duration = time.time() - start
    
    # Should complete in < 5 seconds
    assert duration < 5.0
```

### Profile Tests

```bash
# Profile with py-spy
py-spy record -o profile.svg -- pytest tests/

# Profile with cProfile
python -m cProfile -o profile.stats -m pytest tests/
python -m pstats profile.stats
```

---

## 🐳 Docker Testing

### Build Image

```bash
make docker-build
```

### Run Tests in Container

```bash
make docker-test
```

### Open Shell in Container

```bash
make docker-shell
```

### Test Obfuscation in Container

```bash
docker run --rm -v $(pwd)/examples:/examples llvm-obfuscator:test \
  python3 -m cli.obfuscate compile /examples/hello.c --output /tmp/test --level 3
```

---

## 📊 Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Core modules | >80% | TBD |
| CLI modules | >70% | TBD |
| Overall | >75% | TBD |

### Generate Coverage

```bash
make test-cov
```

### View Coverage

```bash
# HTML report
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=core --cov-report=term
```

---

## 🚨 Troubleshooting

### Tests Failing

```bash
# Clean and retry
make clean
make install
make test
```

### UPX Tests Skipped

```bash
# Install UPX
sudo apt install upx-ucl  # Linux
brew install upx          # macOS
```

### Import Errors

```bash
# Install in development mode
pip install -e .
```

### Slow Tests

```bash
# Skip slow tests
make test-fast
```

---

## 📈 Test Statistics

```bash
# Show test stats
make stats
```

Output:
```
Test Statistics:
Total tests: 45
Test files: 3
Lines of test code: 1200
```

---

## 🔍 Debugging Tests

### Verbose Output

```bash
pytest tests/ -vv --tb=long
```

### Show Logs

```bash
pytest tests/ --log-cli-level=DEBUG
```

### Drop into Debugger

```bash
pytest tests/ --pdb
```

### Run Single Test

```bash
pytest tests/test_config.py::TestPlatform::test_platform_values -vv
```

---

## 📚 Resources

- **Pytest Docs**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Jotai Benchmarks**: https://github.com/lac-dcc/jotai-benchmarks
- **GitHub Actions**: https://docs.github.com/en/actions

---

## 🤝 Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure >80% coverage
3. Update documentation
4. Run `make check` before committing
5. Ensure CI passes

---

## 📞 Support

- **Issues**: https://github.com/SkySingh04/oaas/issues
- **CI Logs**: https://github.com/SkySingh04/oaas/actions
- **Documentation**: See `tests/README.md`

---

**Last Updated:** November 2025  
**Framework Version:** 1.0.0

