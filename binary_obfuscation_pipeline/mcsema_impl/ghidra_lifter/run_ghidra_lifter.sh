#!/usr/bin/env bash
# Ghidra Lifter Wrapper Script
# Orchestrates Docker-based CFG lifting for Windows PE binaries.
#
# This script:
# 1. Validates input binary
# 2. Assumes the ghidra-lifter service is running
# 3. Copies the binary to a shared volume
# 4. Calls the lifter service
# 5. Returns path to generated .cfg file
#
# USAGE:
#   ./run_ghidra_lifter.sh <binary.exe> <output_dir>

set -e

# Configuration
LIFTER_SERVICE="ghidra-lifter"
LIFTER_PORT=5000

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

# Copy binary to the mounted volume
cp "$BINARY_ABS" "/app/binaries/"

log_info "Sending binary to lifter..."

# Call lifter API via HTTP
RESPONSE=$(curl -s -X POST "http://$LIFTER_SERVICE:$LIFTER_PORT/lift/file" \
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
