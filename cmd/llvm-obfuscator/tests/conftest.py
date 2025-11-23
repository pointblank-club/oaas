"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def examples_dir(project_root):
    """Get examples directory."""
    return project_root / "examples"


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get test data directory."""
    test_dir = project_root / "tests" / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def sample_c_file(tmp_path):
    """Create a simple C file for testing."""
    source = tmp_path / "sample.c"
    source.write_text("""
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    printf("Result: %d\\n", result);
    return 0;
}
""")
    return source


@pytest.fixture
def sample_cpp_file(tmp_path):
    """Create a simple C++ file for testing."""
    source = tmp_path / "sample.cpp"
    source.write_text("""
#include <iostream>

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Calculator calc;
    int result = calc.add(5, 3);
    std::cout << "Result: " << result << std::endl;
    return 0;
}
""")
    return source


def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_upx: marks tests that require UPX to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_ollvm: marks tests that require OLLVM plugin"
    )
