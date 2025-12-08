#!/bin/bash
"""
Feature #4: OLLVM Pass Application on LLVM 22 IR

This script applies user-selected OLLVM obfuscation passes to the lifted IR.

USAGE:
  ./run_ollvm.sh input_llvm22.bc output_dir passes_config.json

EXAMPLE:
  ./run_ollvm.sh ./program_llvm22.bc ./obfuscated_ir/ ./passes_config.json
  → Outputs: ./obfuscated_ir/program_obf.bc

WHY USE CUSTOM OPT FROM PLUGINS?
=================================
The opt binary in ./plugins/linux-x86_64/opt is built with OLLVM passes compiled in.
This is the production opt used by the backend, so we use the same binary for consistency:
- Same LLVM 22 version
- Same OLLVM pass implementations
- Same configuration as production

We do NOT use system opt because:
- System opt lacks OLLVM pass plugins
- Would require dynamic plugin loading (error-prone)
- Custom opt is already in the pipeline

CRITICAL WARNINGS ABOUT McSEMA IR + OLLVM PASSES:
==================================================

1. FLATTENING (-fla):
   McSema IR uses a state machine for control flow (not a normal CFG).
   Flattening operations may:
   - Corrupt the PC (program counter) update logic
   - Break function returns (state machine dispatch fails)
   - Create infinite loops in state machine
   - Produce non-executable binaries
   ⚠️ Use only on simple, thoroughly tested binaries

2. BOGUS CONTROL FLOW (-bcf):
   Bogus CFG injection is EXTREMELY DANGEROUS for McSema IR:
   - Often introduces unreachable code blocks
   - Can invalidate lifted CFG edges
   - May prevent execution from reaching function exit
   - Results in crashes or hangs
   ⚠️ AVOID for McSema IR entirely

3. SPLIT BASIC BLOCKS (-split):
   Splitting blocks breaks McSema's strict BB boundaries:
   - McSema IR derives BB boundaries from x86-64 instruction semantics
   - Splitting mid-block corrupts semantic lifting
   - May break register live-range analysis
   - Causes undefined behavior
   ⚠️ AVOID unless absolutely necessary

4. OPAQUE PREDICATES (-opaque):
   Can break symbolic PC control:
   - McSema IR relies on explicit PC updates
   - Opaque predicates confuse register analysis
   - May cause early crashes before main code executes
   - Symbolic execution tools cannot handle mangled PC
   ⚠️ Use with extreme caution

SAFER PASSES FOR McSEMA IR:
===========================
✅ INSTRUCTION SUBSTITUTION (-sub):
   Safe because it operates at IR level without changing control flow
   Does not modify CFG structure
   Only replaces arithmetic/logical operations with equivalent sequences

✅ LIGHTWEIGHT FLATTENING (if enabled):
   Use only on simple programs with validated CFGs
   Test thoroughly before production use
   Monitor binary behavior (may cause slowdowns)

✅ STANDARD LLVM OPTIMIZATIONS (-O1, -O2):
   Safe for post-obfuscation optimization
   OLLVM passes should run FIRST, then standard opts
   Standard opts can improve performance and code quality

EXPERIMENTAL PIPELINE NOTICE:
=============================
This OLLVM + McSema combination is EXPERIMENTAL:
- OLLVM was designed for normal compiled code, not lifter IR
- McSema IR is low-level machine semantics (not high-level IR)
- Combination may produce incorrect or unpredictable code
- Only small test programs should be used until equivalence is validated
- Larger binaries have higher risk of obfuscation-induced bugs

VALIDATION REQUIRED:
====================
Before using obfuscated binaries:
1. Test on simple programs (add, multiply, fibonacci)
2. Verify output matches original (functional equivalence)
3. Check execution on target Windows system
4. Use debugger to trace control flow
5. Compare performance (expect 10-20% slowdown)

DO NOT use obfuscated output in production without thorough testing.
"""

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Path to custom opt with OLLVM passes (from docker build/backend)
CUSTOM_OPT="${CUSTOM_OPT:-/usr/local/llvm-obfuscator/bin/opt}"

# Fallback to plugins directory if running locally
if [ ! -x "$CUSTOM_OPT" ] && [ -x "$PROJECT_ROOT/plugins/linux-x86_64/opt" ]; then
    CUSTOM_OPT="$PROJECT_ROOT/plugins/linux-x86_64/opt"
fi

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_critical() {
    echo -e "${CYAN}⚠️  CRITICAL:${NC} $1"
}

usage() {
    echo "Usage: $0 <input_llvm22.bc> <output_dir> <passes_config.json>"
    echo ""
    echo "Arguments:"
    echo "  input_llvm22.bc      LLVM 22 bitcode (from Feature #3)"
    echo "  output_dir           Directory to write obfuscated bitcode"
    echo "  passes_config.json   Configuration file for enabled passes"
    echo ""
    echo "Example:"
    echo "  ./run_ollvm.sh ./program_llvm22.bc ./obfuscated_ir/ ./passes_config.json"
    exit 1
}

# Parse arguments
if [ $# -lt 3 ]; then
    log_error "Missing arguments"
    usage
fi

INPUT_BC="$1"
OUTPUT_DIR="$2"
CONFIG_FILE="$3"

# Validate inputs
if [ ! -f "$INPUT_BC" ]; then
    log_error "Input bitcode not found: $INPUT_BC"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Validate opt binary
if [ ! -x "$CUSTOM_OPT" ]; then
    log_error "Custom opt binary not found: $CUSTOM_OPT"
    log_error "Expected at: /usr/local/llvm-obfuscator/bin/opt"
    log_error "Or: $PROJECT_ROOT/plugins/linux-x86_64/opt"
    exit 1
fi

INPUT_ABS=$(cd "$(dirname "$INPUT_BC")" && pwd)/$(basename "$INPUT_BC")
CONFIG_ABS=$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")
BINARY_NAME=$(basename "$INPUT_BC" .bc)

mkdir -p "$OUTPUT_DIR"
OUTPUT_ABS=$(cd "$OUTPUT_DIR" && pwd)
OUTPUT_BC="$OUTPUT_ABS/${BINARY_NAME}_obf.bc"

log_section "Feature #4: OLLVM Pass Application"
log_info "Input bitcode: $INPUT_ABS"
log_info "Configuration: $CONFIG_ABS"
log_info "Output: $OUTPUT_BC"
log_info "Custom opt: $CUSTOM_OPT"
log_info ""

# Parse configuration file
log_info "Reading pass configuration..."

# Use jq if available, fallback to grep
if command -v jq &> /dev/null; then
    FLATTENING=$(jq -r '.flattening // false' "$CONFIG_ABS")
    BCF=$(jq -r '.bogus_control_flow // false' "$CONFIG_ABS")
    SUBSTITUTION=$(jq -r '.substitution // false' "$CONFIG_ABS")
    SPLIT=$(jq -r '.split // false' "$CONFIG_ABS")
    LINEAR_MBA=$(jq -r '.linear_mba // false' "$CONFIG_ABS")
    STRING_ENCRYPT=$(jq -r '.string_encrypt // false' "$CONFIG_ABS")
    SYMBOL_OBFUSCATE=$(jq -r '.symbol_obfuscate // false' "$CONFIG_ABS")
    CONSTANT_OBFUSCATE=$(jq -r '.constant_obfuscate // false' "$CONFIG_ABS")
    CRYPTO_HASH=$(jq -r '.crypto_hash // false' "$CONFIG_ABS")
    STANDARD_OPTS=$(jq -r '.standard_llvm_opts // false' "$CONFIG_ABS")
else
    # Fallback: basic grep parsing
    FLATTENING=$(grep -o '"flattening": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    BCF=$(grep -o '"bogus_control_flow": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    SUBSTITUTION=$(grep -o '"substitution": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    SPLIT=$(grep -o '"split": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    LINEAR_MBA=$(grep -o '"linear_mba": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    STRING_ENCRYPT=$(grep -o '"string_encrypt": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    SYMBOL_OBFUSCATE=$(grep -o '"symbol_obfuscate": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    CONSTANT_OBFUSCATE=$(grep -o '"constant_obfuscate": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    CRYPTO_HASH=$(grep -o '"crypto_hash": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
    STANDARD_OPTS=$(grep -o '"standard_llvm_opts": *[^,}]*' "$CONFIG_ABS" | grep -o '[^:]*$' | tr -d ' ' | tr -d '\n')
fi

log_info "Available OLLVM passes:"
log_info "  Flattening: $FLATTENING"
log_info "  Bogus Control Flow: $BCF"
log_info "  Substitution: $SUBSTITUTION"
log_info "  Split: $SPLIT"
log_info "  Linear MBA: $LINEAR_MBA"
log_info "Available MLIR passes:"
log_info "  String Encrypt: $STRING_ENCRYPT"
log_info "  Symbol Obfuscate: $SYMBOL_OBFUSCATE"
log_info "  Constant Obfuscate: $CONSTANT_OBFUSCATE"
log_info "  Crypto Hash: $CRYPTO_HASH"
log_info "Standard Opts: $STANDARD_OPTS"
log_info ""

# Validate configuration values
for val in "$FLATTENING" "$BCF" "$SUBSTITUTION" "$SPLIT" "$LINEAR_MBA" "$STRING_ENCRYPT" "$SYMBOL_OBFUSCATE" "$CONSTANT_OBFUSCATE" "$CRYPTO_HASH" "$STANDARD_OPTS"; do
    if [ "$val" != "true" ] && [ "$val" != "false" ]; then
        log_error "Invalid pass configuration value: $val"
        log_error "All values must be 'true' or 'false'"
        exit 1
    fi
done

# Build OLLVM pass string dynamically
OLLVM_PASS_STRING=""

if [ "$SUBSTITUTION" = "true" ]; then
    OLLVM_PASS_STRING="$OLLVM_PASS_STRING,substitution"
fi

if [ "$LINEAR_MBA" = "true" ]; then
    OLLVM_PASS_STRING="$OLLVM_PASS_STRING,linear-mba"
fi

if [ "$FLATTENING" = "true" ]; then
    log_warn "Flattening enabled - McSema IR may be affected (state machine corruption possible)"
    OLLVM_PASS_STRING="$OLLVM_PASS_STRING,flattening"
fi

if [ "$BCF" = "true" ]; then
    log_critical "Bogus Control Flow enabled - HIGH RISK for McSema IR"
    log_critical "This can invalidate CFG edges and break binary execution"
    OLLVM_PASS_STRING="$OLLVM_PASS_STRING,boguscf"
fi

if [ "$SPLIT" = "true" ]; then
    log_critical "Split Basic Blocks enabled - DANGEROUS for McSema IR"
    log_critical "May corrupt BB boundaries and break semantic lifting"
    OLLVM_PASS_STRING="$OLLVM_PASS_STRING,split"
fi

# Remove leading comma if present
OLLVM_PASS_STRING="${OLLVM_PASS_STRING#,}"

# Build MLIR pass string (runs in separate Clang invocation with -mllvm flags)
MLIR_PASSES=""

if [ "$STRING_ENCRYPT" = "true" ]; then
    MLIR_PASSES="$MLIR_PASSES -mllvm -string-encrypt"
fi

if [ "$SYMBOL_OBFUSCATE" = "true" ]; then
    MLIR_PASSES="$MLIR_PASSES -mllvm -symbol-obfuscate"
fi

if [ "$CONSTANT_OBFUSCATE" = "true" ]; then
    MLIR_PASSES="$MLIR_PASSES -mllvm -constant-obfuscate"
fi

if [ "$CRYPTO_HASH" = "true" ]; then
    MLIR_PASSES="$MLIR_PASSES -mllvm -crypto-hash"
fi

# Remove leading comma if present
OLLVM_PASS_STRING="${OLLVM_PASS_STRING#,}"

log_info "OLLVM passes to apply: ${OLLVM_PASS_STRING:-none}"
log_info "MLIR passes to apply: ${MLIR_PASSES:-none}"
log_info ""

# Create intermediate files for opt chain
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

STEP=1
TOTAL_STEPS=3

# Run opt with OLLVM passes
if [ -n "$OLLVM_PASS_STRING" ]; then
    OLLVM_OUTPUT="$TEMP_DIR/ollvm_result.bc"

    log_info "Step $STEP/$TOTAL_STEPS: Applying OLLVM passes..."
    log_info "Passes: $OLLVM_PASS_STRING"
    log_info "Command: $CUSTOM_OPT -passes=\"$OLLVM_PASS_STRING\" $INPUT_ABS -o $OLLVM_OUTPUT"
    log_info ""

    if $CUSTOM_OPT -passes="$OLLVM_PASS_STRING" "$INPUT_ABS" -o "$OLLVM_OUTPUT" 2>&1; then
        log_info "✓ OLLVM passes applied successfully"
        CURRENT_BC="$OLLVM_OUTPUT"
    else
        log_error "OLLVM pass application failed"
        log_error "Check configuration and ensure passes are compatible with opt binary"
        exit 1
    fi
    STEP=$((STEP + 1))
else
    log_info "Step $STEP/$TOTAL_STEPS: Skipping OLLVM passes (none enabled)"
    CURRENT_BC="$INPUT_ABS"
    STEP=$((STEP + 1))
fi

log_info ""

# Apply standard LLVM optimizations if enabled
if [ "$STANDARD_OPTS" = "true" ]; then
    log_info "Step $STEP/$TOTAL_STEPS: Applying standard LLVM optimizations (-O1)..."
    log_info "Command: $CUSTOM_OPT -O1 $CURRENT_BC -o $OUTPUT_BC"
    log_info ""

    if $CUSTOM_OPT -O1 "$CURRENT_BC" -o "$OUTPUT_BC" 2>&1; then
        log_info "✓ Standard optimizations applied successfully"
        CURRENT_BC="$OUTPUT_BC"
    else
        log_error "Standard optimization failed"
        exit 1
    fi
    STEP=$((STEP + 1))
else
    log_info "Step $STEP/$TOTAL_STEPS: Skipping standard optimizations"
    if [ "$CURRENT_BC" != "$OUTPUT_BC" ]; then
        cp "$CURRENT_BC" "$OUTPUT_BC"
    fi
    STEP=$((STEP + 1))
fi

log_info ""

# Note about MLIR passes
if [ -n "$MLIR_PASSES" ]; then
    log_info "Step $STEP/$TOTAL_STEPS: MLIR passes will be applied during final compilation"
    log_info "These passes run in Feature #5 (PE codegen) via clang with -mllvm flags"
    log_info "MLIR passes: $MLIR_PASSES"
else
    log_info "Step $STEP/$TOTAL_STEPS: No MLIR passes enabled"
fi

log_info ""

# Verify output
if [ ! -f "$OUTPUT_BC" ]; then
    log_error "Output file not created: $OUTPUT_BC"
    exit 1
fi

BC_SIZE=$(du -h "$OUTPUT_BC" | cut -f1)
log_info "✓ Obfuscation complete"
log_info "Output bitcode: $OUTPUT_BC ($BC_SIZE)"
log_info ""

# Print warnings about McSema compatibility
log_section "⚠️  IMPORTANT VALIDATION NOTES"

log_critical "This obfuscated IR comes from McSema lifting - NOT normal compiled code"
echo ""
echo "Before using this binary:"
echo "  1. Test on simple programs first (fibonacci, factorial, etc.)"
echo "  2. Verify output matches original binary's behavior"
echo "  3. Run on target Windows system to confirm execution"
echo "  4. Monitor for crashes, hangs, or unexpected behavior"
echo "  5. Compare performance (expect 10-20% slowdown)"
echo ""
echo "Known risks:"
echo "  - Control flow obfuscations may break state machine dispatch"
echo "  - Indirect calls may not resolve correctly"
echo "  - Performance may be significantly degraded"
echo "  - Binary may be non-functional for large programs"
echo ""

# Final status
log_section "Feature #4 Complete: IR Obfuscated"
log_info "READY_FOR_FINAL_EMIT"
log_info "Path: $OUTPUT_BC"
log_info ""
log_info "Next step (Feature #5):"
log_info "  Compile obfuscated IR back to Windows PE binary"
log_info "  clang-22 $OUTPUT_BC -o program_obfuscated.exe"
log_info ""

exit 0
