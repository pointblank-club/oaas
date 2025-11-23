# LLVM Obfuscator Testing Framework

Comprehensive testing suite for the LLVM Binary Obfuscator project.

## 📋 Overview

This testing framework provides:
- **Unit tests** for individual modules
- **Integration tests** for the complete obfuscation pipeline
- **Benchmark tests** on real-world code (Jotai collection)
- **Performance tests** for measuring overhead
- **CI/CD integration** via GitHub Actions

## 🚀 Quick Start

### Install Test Dependencies

```bash
cd cmd/llvm-obfuscator
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_config.py tests/test_upx_packer.py

# Integration tests only
pytest tests/test_obfuscator_integration.py

# Skip slow tests
pytest -m "not slow"

# Run only UPX tests
pytest -m "requires_upx"
```

## 📁 Test Structure

```
tests/
├── conftest.py                          # Pytest configuration and fixtures
├── test_config.py                       # Configuration module tests
├── test_upx_packer.py                   # UPX packer tests
├── test_obfuscator_integration.py       # Full pipeline integration tests
├── requirements-test.txt                # Test dependencies
└── README.md                            # This file

scripts/
└── test_on_jotai.sh                     # Jotai benchmark testing script
```

## 🧪 Test Categories

### 1. Unit Tests

Test individual modules in isolation.

**test_config.py:**
- Platform enum
- Obfuscation levels
- Configuration classes
- YAML parsing

**test_upx_packer.py:**
- UPX availability detection
- Binary packing/unpacking
- Compression levels
- Error handling

```bash
pytest tests/test_config.py -v
pytest tests/test_upx_packer.py -v
```

### 2. Integration Tests

Test the complete obfuscation pipeline.

**test_obfuscator_integration.py:**
- Basic obfuscation (Level 1-3)
- Symbol obfuscation layer
- String encryption layer
- UPX packing layer
- Full pipeline (all layers)
- Correctness preservation
- Report generation

```bash
pytest tests/test_obfuscator_integration.py -v
```

### 3. Benchmark Tests

Test on real-world code from the Jotai collection.

```bash
cd scripts
./test_on_jotai.sh --max 100
```

This will:
- Clone the Jotai benchmark repository
- Test obfuscation on 100 benchmarks
- Generate HTML report with statistics
- Measure size reduction, symbol reduction, etc.

## 📊 Coverage

### Generate Coverage Report

```bash
pytest tests/ --cov=core --cov-report=html --cov-report=term
```

Open `htmlcov/index.html` to view the coverage report.

### Current Coverage Goals

- **Core modules:** >80%
- **CLI modules:** >70%
- **Overall:** >75%

## 🎯 Test Markers

Tests are organized with pytest markers:

```python
@pytest.mark.slow              # Slow tests (> 1 minute)
@pytest.mark.requires_upx      # Requires UPX installed
@pytest.mark.requires_ollvm    # Requires OLLVM plugin
@pytest.mark.integration       # Integration test
@pytest.mark.unit              # Unit test
```

### Running Specific Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m "integration"

# Run unit tests only
pytest -m "unit"
```

## 🔧 Writing New Tests

### Basic Test Template

```python
import pytest
from core import LLVMObfuscator, ObfuscationConfig

class TestMyFeature:
    """Test suite for my feature."""
    
    @pytest.fixture
    def sample_code(self, tmp_path):
        """Create sample code for testing."""
        source = tmp_path / "test.c"
        source.write_text("int main() { return 0; }")
        return source
    
    def test_basic_functionality(self, sample_code):
        """Test basic functionality."""
        # Arrange
        config = ObfuscationConfig(level=3)
        obfuscator = LLVMObfuscator()
        
        # Act
        result = obfuscator.obfuscate(sample_code, config)
        
        # Assert
        assert result is not None
        assert "output_file" in result
```

### Integration Test Template

```python
def test_full_pipeline(self, complex_c_source, tmp_path):
    """Test complete obfuscation pipeline."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    config = ObfuscationConfig(
        level=ObfuscationLevel.HIGH,
        advanced=AdvancedConfiguration(
            symbol_obfuscation=SymbolObfuscationConfiguration(enabled=True),
            string_encryption=True,
            upx_packing=UPXConfiguration(enabled=True)
        ),
        output=OutputConfiguration(directory=output_dir)
    )
    
    obfuscator = LLVMObfuscator()
    result = obfuscator.obfuscate(complex_c_source, config)
    
    assert result is not None
    # Add more assertions...
```

## 🏃 Performance Testing

### Benchmark a Single File

```bash
pytest tests/test_obfuscator_integration.py::TestObfuscationPipeline::test_full_pipeline \
  --benchmark-only
```

### Compare Performance

```bash
# Generate baseline
pytest tests/test_obfuscator_integration.py --benchmark-save=baseline

# After changes
pytest tests/test_obfuscator_integration.py --benchmark-compare=baseline
```

## 🐛 Debugging Tests

### Run with Verbose Output

```bash
pytest tests/ -vv --tb=long
```

### Run Single Test

```bash
pytest tests/test_config.py::TestPlatform::test_platform_values -v
```

### Drop into Debugger on Failure

```bash
pytest tests/ --pdb
```

### Show Print Statements

```bash
pytest tests/ -s
```

## 📈 Continuous Integration

Tests run automatically on:
- **Push to main/develop**
- **Pull requests**
- **Daily at 2 AM UTC** (scheduled)
- **Manual trigger** (workflow_dispatch)

See `.github/workflows/test-obfuscator.yml` for configuration.

### CI Test Matrix

```yaml
OS:              [ubuntu-latest, macos-latest]
Python:          ['3.10', '3.11', '3.12']
Total Jobs:      6 (2 OS × 3 Python versions)
```

### CI Steps

1. **Setup** - Install system dependencies (clang, llvm, upx)
2. **Unit Tests** - Test individual modules
3. **Integration Tests** - Test complete pipeline
4. **Coverage** - Generate coverage report
5. **Example Tests** - Test on example files
6. **Jotai Tests** - Test on real-world benchmarks (limited)
7. **Docker Tests** - Test Docker builds
8. **Security Scan** - Run Trivy vulnerability scanner

## 📝 Test Requirements

### System Requirements

- **Compiler:** clang 15+
- **LLVM:** llvm 15+
- **Python:** 3.10+
- **UPX:** Any version (optional, for UPX tests)

### Python Dependencies

See `requirements-test.txt`:
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- pytest-timeout >= 2.1.0
- pytest-xdist >= 3.3.1 (for parallel execution)

## 🎓 Best Practices

### 1. Test Naming

```python
# Good
def test_symbol_obfuscation_reduces_symbols()
def test_upx_packing_reduces_size()

# Bad
def test1()
def test_stuff()
```

### 2. Test Organization

```python
class TestSymbolObfuscation:
    """Group related tests in classes."""
    
    def test_sha256_algorithm(self): ...
    def test_blake2b_algorithm(self): ...
    def test_hash_length(self): ...
```

### 3. Use Fixtures

```python
@pytest.fixture
def obfuscator():
    """Reusable obfuscator instance."""
    return LLVMObfuscator()

def test_something(obfuscator):
    # Use fixture
    result = obfuscator.obfuscate(...)
```

### 4. Test Independence

Each test should be independent and not rely on other tests.

```python
# Good - each test is independent
def test_feature_a(tmp_path): ...
def test_feature_b(tmp_path): ...

# Bad - test_b depends on test_a
def test_a(self):
    self.data = "test"

def test_b(self):
    assert self.data == "test"  # Depends on test_a
```

### 5. Clear Assertions

```python
# Good - clear what's being tested
assert result["symbols_count"] < baseline["symbols_count"]

# Bad - unclear assertion
assert result
```

## 🚨 Troubleshooting

### Tests Failing Locally

```bash
# Clean up previous test artifacts
rm -rf .pytest_cache __pycache__

# Reinstall dependencies
pip install -r tests/requirements-test.txt --force-reinstall

# Run with verbose output
pytest tests/ -vv
```

### UPX Tests Skipped

```bash
# Install UPX
sudo apt install upx-ucl  # Linux
brew install upx          # macOS

# Verify
upx --version
```

### Import Errors

```bash
# Ensure you're in the right directory
cd cmd/llvm-obfuscator

# Install in development mode
pip install -e .
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Jotai Benchmarks](https://github.com/lac-dcc/jotai-benchmarks)

## 🤝 Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Add unit tests** for new modules
3. **Add integration tests** for new features
4. **Update documentation** (this file)
5. **Ensure CI passes** before merging

### Test Coverage Requirements

- New modules: **>80% coverage**
- Modified modules: **Maintain or improve** coverage
- Critical paths: **100% coverage**

## 📞 Support

Issues with tests? Check:
1. [GitHub Issues](https://github.com/SkySingh04/oaas/issues)
2. [CI Logs](https://github.com/SkySingh04/oaas/actions)
3. Test documentation (this file)

---

**Last Updated:** November 2025  
**Maintainers:** @SkySingh04

