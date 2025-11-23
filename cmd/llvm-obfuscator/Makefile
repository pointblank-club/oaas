.PHONY: help test test-unit test-integration test-all test-cov test-jotai clean install lint format check docker-test

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)LLVM Obfuscator - Testing Framework$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	pip install -r tests/requirements-test.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Development environment ready$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/test_config.py tests/test_upx_packer.py -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/test_obfuscator_integration.py -v

test-all: test-unit test-integration ## Run all test suites
	@echo "$(GREEN)✓ All tests complete$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=core --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

test-cov-xml: ## Generate XML coverage report (for CI)
	pytest tests/ --cov=core --cov-report=xml

test-fast: ## Run tests excluding slow ones
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -m "not slow" -v

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	pytest tests/ -n auto

test-jotai: ## Run Jotai benchmark tests (limited)
	@echo "$(BLUE)Running Jotai benchmark tests...$(NC)"
	bash scripts/test_on_jotai.sh --max 10

test-jotai-full: ## Run full Jotai benchmark tests
	@echo "$(BLUE)Running full Jotai benchmark tests (this will take a while)...$(NC)"
	bash scripts/test_on_jotai.sh --max 1000

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 core/ cli/ --max-line-length=120 --ignore=E203,W503
	mypy core/ cli/ --ignore-missing-imports || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black core/ cli/ tests/ --line-length=120
	@echo "$(GREEN)✓ Code formatted$(NC)"

check: lint test ## Run lint and tests
	@echo "$(GREEN)✓ All checks passed$(NC)"

clean: ## Clean up test artifacts
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test-report.html
	rm -rf jotai_obfuscation_results
	rm -rf jotai-benchmarks
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -f Dockerfile.backend -t llvm-obfuscator:test .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-test: docker-build ## Test Docker container
	@echo "$(BLUE)Testing Docker container...$(NC)"
	docker run --rm llvm-obfuscator:test upx --version
	docker run --rm llvm-obfuscator:test clang --version
	docker run --rm llvm-obfuscator:test python3 --version
	@echo "$(GREEN)✓ Docker tests passed$(NC)"

docker-shell: ## Open shell in Docker container
	docker run --rm -it -v $(PWD):/workspace llvm-obfuscator:test /bin/bash

example-hello: ## Test on hello.c example
	@echo "$(BLUE)Testing on hello.c...$(NC)"
	python3 -m cli.obfuscate compile examples/hello.c \
		--output test_output \
		--level 3 \
		--string-encryption \
		--enable-symbol-obfuscation
	@test -f test_output/hello && echo "$(GREEN)✓ Binary created successfully$(NC)"
	@./test_output/hello 0 && echo "$(GREEN)✓ Binary executes successfully$(NC)" || true

example-matmul: ## Test on matmul.c example
	@echo "$(BLUE)Testing on matmul.c...$(NC)"
	python3 -m cli.obfuscate compile examples/matmul.c \
		--output test_output \
		--level 3 \
		--string-encryption \
		--enable-symbol-obfuscation \
		--enable-upx
	@test -f test_output/matmul && echo "$(GREEN)✓ Binary created successfully$(NC)"

watch: ## Watch for changes and run tests
	@echo "$(BLUE)Watching for changes...$(NC)"
	@while true; do \
		inotifywait -qq -r -e modify core/ cli/ tests/; \
		clear; \
		make test-fast; \
	done

ci-test: ## Run CI test suite locally
	@echo "$(BLUE)Running CI test suite...$(NC)"
	make clean
	make install
	make lint
	make test-cov-xml
	make example-hello
	@echo "$(GREEN)✓ CI tests complete$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	pytest tests/test_obfuscator_integration.py --benchmark-only || echo "$(YELLOW)Install pytest-benchmark for benchmarks$(NC)"

report: test-cov ## Generate comprehensive test report
	@echo "$(BLUE)Generating test report...$(NC)"
	pytest tests/ --html=test-report.html --self-contained-html
	@echo "$(GREEN)✓ Test report generated: test-report.html$(NC)"

stats: ## Show test statistics
	@echo "$(BLUE)Test Statistics:$(NC)"
	@echo "Total tests: $$(pytest tests/ --collect-only -q | tail -n 1 | cut -d' ' -f1)"
	@echo "Test files: $$(find tests/ -name 'test_*.py' | wc -l)"
	@echo "Lines of test code: $$(find tests/ -name '*.py' -exec cat {} \; | wc -l)"

verify-install: ## Verify installation
	@echo "$(BLUE)Verifying installation...$(NC)"
	@command -v python3 >/dev/null 2>&1 && echo "$(GREEN)✓ Python3 installed$(NC)" || echo "$(YELLOW)✗ Python3 not found$(NC)"
	@command -v clang >/dev/null 2>&1 && echo "$(GREEN)✓ Clang installed$(NC)" || echo "$(YELLOW)✗ Clang not found$(NC)"
	@command -v upx >/dev/null 2>&1 && echo "$(GREEN)✓ UPX installed$(NC)" || echo "$(YELLOW)✗ UPX not found (optional)$(NC)"
	@python3 -c "import pytest" 2>/dev/null && echo "$(GREEN)✓ Pytest installed$(NC)" || echo "$(YELLOW)✗ Pytest not found$(NC)"

setup: verify-install install ## Initial setup (install + verify)
	@echo "$(GREEN)✓ Setup complete! Run 'make test' to run tests.$(NC)"

# Quick shortcuts
t: test ## Shortcut for 'test'
tu: test-unit ## Shortcut for 'test-unit'
ti: test-integration ## Shortcut for 'test-integration'
tc: test-cov ## Shortcut for 'test-cov'
c: clean ## Shortcut for 'clean'
l: lint ## Shortcut for 'lint'
f: format ## Shortcut for 'format'

