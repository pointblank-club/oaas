#!/usr/bin/env bash
# Feature #3 Part 1: McSema CFG → LLVM IR Lifting
#
# This script converts a Ghidra-exported .cfg file into LLVM IR using mcsema-lift.
#
# USAGE:
#   ./run_lift.sh program.cfg output_dir
#
# EXAMPLE:
#   ./run_lift.sh ./program.cfg ./lifted_ir/
#   → Outputs: ./lifted_ir/program.bc
#
# CRITICAL WARNINGS ABOUT McSEMA IR:
# ==================================
#
# 1. McSema IR is LOW-LEVEL MACHINE IR, NOT high-level source IR
#    ► It represents x86-64 semantics directly (registers, memory, flags)
#    ► It is NOT a normal LLVM IR that compilers produce
#    ► Subsequent OLLVM obfuscation passes assume LLVM IR semantics
#    ► McSema IR may violate LLVM IR assumptions → passes may fail
#
# 2. Memory Model is FLATTENED
#    ► All memory is represented as a giant state struct (64-bit fields)
#    ► No proper LLVM pointer types or type safety
#    ► Load/store operations are emulated via extractvalue/insertvalue
#    ► Pointer arithmetic is done on raw integers
#    ► This breaks many OLLVM passes designed for structured IR
#
# 3. Control Flow is a STATE MACHINE, NOT structured CFG
#    ► Functions do not use normal LLVM basic block structure
#    ► Control flow jumps are implemented via switch statements on state variables
#    ► No proper branch instructions—indirect dispatch via state machine
#    ► CFG analysis tools break on McSema IR
#    ► OLLVM passes expecting structured control flow will fail
#
# 4. Ghidra CFG may contain ERRORS or NOISE
#    ► Function detection accuracy: ~90% (vs IDA's ~98%)
#    ► Tail calls may be mis-identified as function calls
#    ► Jump tables may be incorrectly recovered
#    ► Indirect branches cannot be resolved (no data flow)
#    ► If Ghidra CFG is wrong → mcsema-lift produces invalid IR
#
# 5. Lifter is NOT SAFE for:
#    ► Exceptions (SEH tables, C++ EH) → breaks control flow
#    ► Recursion → state machine cannot model recursive calls
#    ► Jump tables → Ghidra recovery is unreliable
#    ► C++ code → vtables, name mangling not supported
#    ► Complex indirect calls → cannot resolve targets
#
# 6. This pipeline is INTENDED ONLY for simple -O0 C code
#    ► As enforced by Feature #1 source validation
#    ► Optimization breaks CFG recovery
#    ► Complex language features break lifting
#
# WHAT THIS SCRIPT DOES:
# ======================
# 1. Validates .cfg file exists
# 2. Checks mcsema-lift is installed
# 3. Runs mcsema-lift with target parameters (Windows x86-64)
# 4. Generates LLVM bitcode (.bc file)
# 5. Outputs status message for next feature
#
# OUTPUT:
# =======
# - program.bc: LLVM bitcode (low-level machine IR in LLVM 10-17 format)
# - Next: Feature #3 Part 2 (convert_ir_version.sh) upgrades to LLVM 22

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
}

usage() {
    echo "Usage: $0 <program.cfg> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  program.cfg   Path to Ghidra-exported CFG (output from Feature #2)"
    echo "  output_dir    Directory to write LLVM bitcode"
    echo ""
    echo "Example:"
    echo "  ./run_lift.sh ./program.cfg ./lifted_ir/"
    exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
    log_error "Missing arguments"
    usage
fi

CFG_FILE="$1"
OUTPUT_DIR="$2"

# Validate CFG file
if [ ! -f "$CFG_FILE" ]; then
    log_error "CFG file not found: $CFG_FILE"
    exit 1
fi

CFG_ABS=$(cd "$(dirname "$CFG_FILE")" && pwd)/$(basename "$CFG_FILE")
CFG_NAME=$(basename "$CFG_FILE")
BINARY_NAME="${CFG_NAME%.cfg}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_ABS=$(cd "$OUTPUT_DIR" && pwd)
OUTPUT_BC="$OUTPUT_ABS/${BINARY_NAME}.bc"

log_section "Feature #3 Part 1: McSema CFG → LLVM IR Lifting"
log_info "CFG file: $CFG_ABS"
log_info "Output: $OUTPUT_BC"
log_info ""

# Check if mcsema-lift is available
log_info "Checking for mcsema-lift..."

if ! command -v mcsema-lift &> /dev/null; then
    log_error "mcsema-lift not found in PATH"
    log_error "Install McSema: https://github.com/lifting-bits/mcsema"
    log_error "or run inside container: docker exec llvm-obfuscator-backend mcsema-lift ..."
    exit 1
fi

MCSEMA_VERSION=$(mcsema-lift --version 2>&1 | head -1 || echo "unknown")
log_info "Found mcsema-lift: $MCSEMA_VERSION"
log_info ""

# Run mcsema-lift
log_info "Running mcsema-lift..."
log_info "Command: mcsema-lift --os windows --arch amd64 --cfg $CFG_ABS --output $OUTPUT_BC"
log_info ""

if mcsema-lift \
    --os windows \
    --arch amd64 \
    --cfg "$CFG_ABS" \
    --output "$OUTPUT_BC" 2>&1; then

    # Verify output
    if [ ! -f "$OUTPUT_BC" ]; then
        log_error "mcsema-lift succeeded but output file not created: $OUTPUT_BC"
        exit 1
    fi

    BC_SIZE=$(du -h "$OUTPUT_BC" | cut -f1)
    log_info "✓ Lifting successful"
    log_info "Output bitcode: $OUTPUT_BC ($BC_SIZE)"
    log_info ""

    # Print next step
    log_section "Next: Feature #3 Part 2 — IR Version Upgrade"
    log_info "READY_FOR_IR_UPGRADE"
    log_info "Path: $OUTPUT_BC"
    log_info ""
    log_info "Next command:"
    log_info "  ./convert_ir_version.sh $OUTPUT_BC $OUTPUT_DIR"
    log_info ""

    exit 0
else
    log_error "mcsema-lift failed"
    log_error "Check error messages above for details"
    log_error ""
    log_error "Common issues:"
    log_error "1. Ghidra CFG contains errors (indirect calls, jump tables, etc.)"
    log_error "2. Binary contains exceptions or recursion (disabled in Feature #1)"
    log_error "3. Ghidra function detection failed (use IDA for validation)"
    log_error ""
    exit 1
fi
