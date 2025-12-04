#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

case "$1" in
  build)
    echo -e "${YELLOW}Building test container...${NC}"
    docker build -f Dockerfile.test -t mlir-obfuscator-test .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Build successful${NC}"
    else
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    fi
    ;;
  
  test)
    echo -e "${YELLOW}Running all integration tests...${NC}"
    docker run --rm \
      -v "$(pwd)/test_results:/app/test_results" \
      mlir-obfuscator-test \
      bash -c "cd /app/tests && ./run_integration_tests.sh"
    ;;
  
  test-passes)
    echo -e "${YELLOW}Running MLIR passes tests only...${NC}"
    docker run --rm \
      -v "$(pwd)/test_results:/app/test_results" \
      mlir-obfuscator-test \
      bash -c "cd /app/tests && ./test_mlir_passes_only.sh"
    ;;
  
  shell)
    echo -e "${YELLOW}Starting interactive shell...${NC}"
    docker run --rm -it \
      -v "$(pwd):/app" \
      -v "$(pwd)/test_results:/app/test_results" \
      mlir-obfuscator-test \
      bash
    ;;
  
  compose-test)
    echo -e "${YELLOW}Running tests via docker-compose...${NC}"
    docker-compose -f docker-compose.test.yml up test-mlir
    ;;
  
  compose-dev)
    echo -e "${YELLOW}Starting dev container...${NC}"
    docker-compose -f docker-compose.test.yml run --rm dev
    ;;
  
  clean)
    echo -e "${YELLOW}Cleaning up...${NC}"
    docker rmi mlir-obfuscator-test 2>/dev/null || true
    docker-compose -f docker-compose.test.yml down 2>/dev/null || true
    rm -rf test_results
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    ;;
  
  logs)
    echo -e "${YELLOW}Showing test results...${NC}"
    if [ -d "test_results" ]; then
        ls -la test_results/
        echo ""
        echo "Binary files:"
        file test_results/test_binary 2>/dev/null || echo "No binary found"
    else
        echo -e "${RED}No test results found${NC}"
    fi
    ;;
  
  *)
    echo "Usage: $0 {build|test|test-passes|shell|compose-test|compose-dev|clean|logs}"
    echo ""
    echo "Commands:"
    echo "  build          - Build the test Docker container"
    echo "  test           - Run all integration tests"
    echo "  test-passes    - Run only MLIR pass tests (quick)"
    echo "  shell          - Start interactive shell in container"
    echo "  compose-test   - Run tests using docker-compose"
    echo "  compose-dev    - Start dev container using docker-compose"
    echo "  clean          - Remove test artifacts and containers"
    echo "  logs           - Show test results"
    exit 1
    ;;
esac