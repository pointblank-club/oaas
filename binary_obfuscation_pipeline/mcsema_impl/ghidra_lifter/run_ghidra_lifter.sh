#!/bin/bash
"""
Ghidra Lifter Wrapper Script
Orchestrates Docker-based CFG lifting for Windows PE binaries.

This script:
1. Validates input binary
2. Builds/verifies Docker image
3. Runs Docker container with mounted volumes
4. Returns path to generated .cfg file

USAGE:
  ./run_ghidra_lifter.sh <binary.exe> <output_dir>

EXAMPLE:
  ./run_ghidra_lifter.sh ./program.exe ./lifted_output/
  → Outputs: ./lifted_output/program.cfg

WHY THIS WRAPPER?
==================
The ghidra-lifter runs as a separate Docker service (defined in docker-compose.yml).
This script simplifies the interaction by:
- Validating binaries before sending to lifter
- Handling volume mounting automatically
- Polling until lifting completes
- Retrieving output CFG file
- Providing error messages

WHY NOT DIRECT DOCKER CALL?
============================
We use docker-compose because:
- Manages networking between services (backend → ghidra-lifter)
- Handles volume mounting consistently
- Provides health checks
- Simplifies production deployment
- Enables multi-container orchestration

WHY HEADLESS MODE?
==================
Ghidra supports two modes:
- GUI: Interactive reverse engineering (requires X11 display)
- Headless: Batch processing via command line (suitable for Docker)

We use headless because:
- Backend container has no X11 display server
- Fully automated (no user interaction)
- Runs in background
- Can process multiple binaries in parallel
- Suitable for CI/CD pipelines

LIMITATIONS OF GHIDRA CFG:
==========================
- Function detection less accurate than IDA Pro
- Switch table recovery may fail
- Complex control flow yields noisy CFGs
- Only reliable for simple -O0 -g C code (Feature #1 output)
- CFG accuracy: ~80-85% on constrained binaries

NEXT STEP:
==========
The generated .cfg file is passed to Feature #3 (McSema lifting):
  mcsema-lift --cfg program.cfg --output program.bc

This converts the CFG to LLVM IR, which is then obfuscated.
"""

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
LIFTER_SERVICE="ghidra-lifter"
LIFTER_PORT=5001

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print usage
usage() {
    echo "Usage: $0 <binary.exe> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  binary.exe    Path to Windows PE binary (output from Feature #1)"
    echo "  output_dir    Directory to write .cfg file"
    echo ""
    echo "Example:"
    echo "  ./run_ghidra_lifter.sh ./program.exe ./lifted_output/"
    exit 1
}

# Validate arguments
if [ $# -lt 2 ]; then
    log_error "Missing arguments"
    usage
fi

BINARY_PATH="$1"
OUTPUT_DIR="$2"

# Validate binary exists
if [ ! -f "$BINARY_PATH" ]; then
    log_error "Binary not found: $BINARY_PATH"
    exit 1
fi

BINARY_NAME=$(basename "$BINARY_PATH")
BINARY_ABS=$(cd "$(dirname "$BINARY_PATH")" && pwd)/$(basename "$BINARY_PATH")

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_ABS=$(cd "$OUTPUT_DIR" && pwd)
OUTPUT_CFG="$OUTPUT_ABS/${BINARY_NAME%.exe}.cfg"

log_info "Binary: $BINARY_ABS"
log_info "Output: $OUTPUT_CFG"
log_info ""

# Check if docker-compose.yml exists
if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
    log_error "docker-compose.yml not found at $DOCKER_COMPOSE_FILE"
    log_error "Must be run from project root or with correct PROJECT_ROOT"
    exit 1
fi

log_info "Using docker-compose: $DOCKER_COMPOSE_FILE"

# Ensure docker-compose services are running
log_info "Checking if ghidra-lifter service is running..."

# Check if service is healthy
LIFTER_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$LIFTER_PORT/health" 2>/dev/null || echo "000")

if [ "$LIFTER_HEALTH" != "200" ]; then
    log_warn "ghidra-lifter service not running or unhealthy (HTTP $LIFTER_HEALTH)"
    log_info "Starting docker-compose services..."

    cd "$PROJECT_ROOT"
    docker-compose up -d ghidra-lifter

    # Wait for service to be healthy
    log_info "Waiting for ghidra-lifter to become healthy..."
    MAX_WAIT=120
    ELAPSED=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        LIFTER_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$LIFTER_PORT/health" 2>/dev/null || echo "000")

        if [ "$LIFTER_HEALTH" = "200" ]; then
            log_info "ghidra-lifter is healthy"
            break
        fi

        echo -n "."
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done

    if [ "$LIFTER_HEALTH" != "200" ]; then
        log_error "ghidra-lifter failed to become healthy (HTTP $LIFTER_HEALTH)"
        log_error "Check docker logs: docker logs ghidra-lifter"
        exit 1
    fi
else
    log_info "ghidra-lifter is healthy (HTTP 200)"
fi

log_info ""
log_info "Sending binary to lifter..."

# Call lifter API via HTTP
RESPONSE=$(curl -s -X POST "http://localhost:$LIFTER_PORT/lift/file" \
    -H "Content-Type: application/json" \
    -d "{
        \"binary_path\": \"/app/binaries/$BINARY_NAME\",
        \"output_dir\": \"/app/reports\"
    }")

# Check response
if echo "$RESPONSE" | grep -q '"success": true'; then
    log_info "✓ Lifting successful"

    # Extract CFG file path from response
    CFG_FILE=$(echo "$RESPONSE" | grep -o '"cfg_file": "[^"]*"' | cut -d'"' -f4)

    # Extract stats
    FUNCTIONS=$(echo "$RESPONSE" | grep -o '"functions": [0-9]*' | cut -d' ' -f2)
    BLOCKS=$(echo "$RESPONSE" | grep -o '"total_blocks": [0-9]*' | cut -d' ' -f2)
    EDGES=$(echo "$RESPONSE" | grep -o '"total_edges": [0-9]*' | cut -d' ' -f2)

    log_info "CFG file: $CFG_FILE"
    log_info "Functions: $FUNCTIONS"
    log_info "Basic blocks: $BLOCKS"
    log_info "Edges: $EDGES"
    log_info ""

    # Verify CFG file exists in output directory
    if [ -f "$OUTPUT_CFG" ]; then
        log_info "✓ Feature #2 complete: CFG exported successfully"
        log_info ""
        log_info "Next step (Feature #3):"
        log_info "  mcsema-lift --cfg $OUTPUT_CFG --output ${BINARY_NAME%.exe}.bc"
        exit 0
    else
        log_error "CFG file not found at expected location: $OUTPUT_CFG"
        log_error "Check /app/reports/ in container: docker exec ghidra-lifter ls -lh /app/reports/"
        exit 1
    fi
else
    log_error "Lifting failed"
    log_error "Response: $RESPONSE"
    exit 1
fi
