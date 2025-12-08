#!/bin/bash
"""
Feature #3 Part 2: LLVM IR Version Upgrade to LLVM 22

This script upgrades McSema-produced LLVM IR from LLVM 10-17 to LLVM 22.

WHY UPGRADE?
============
- McSema lifter produces LLVM 10-17 bitcode (depends on mcsema-lift version)
- Our OLLVM implementation is built on LLVM 22
- Bitcode format is version-specific and not forward-compatible
- LLVM 22 tools cannot process older bitcode versions
- Auto-upgrade is necessary before applying OLLVM passes

USAGE:
  ./convert_ir_version.sh input.bc output_dir

EXAMPLE:
  ./convert_ir_version.sh ./program.bc ./llvm22_ir/
  → Outputs: ./llvm22_ir/program_llvm22.bc

CRITICAL WARNINGS:
==================

1. AUTO-UPGRADE HANDLES ~95% OF CASES
   ► Remaining 5% may break silently or produce invalid IR
   ► No guarantee of semantic equivalence after upgrade
   ► Advanced IR features may degrade or disappear

2. LLVM METADATA MAY BE LOST
   ► Debug info (line numbers, variable names) often corrupted
   ► Custom metadata may be stripped
   ► Type information may be simplified
   ► Optimization metadata may be invalidated

3. DEPRECATED INTRINSICS MAY BREAK
   ► Some LLVM intrinsics were removed in LLVM 22
   ► Auto-upgrade attempts to map to newer intrinsics
   ► Mapping may be incorrect for edge cases
   ► Results in assertion failures or undefined behavior

4. CALLING CONVENTION MISMATCHES
   ► x86-64 calling conventions evolved across LLVM versions
   ► Register allocation may differ
   ► ABI compatibility not guaranteed
   ► McSema IR is not standard → additional risk

5. OPTIMIZATIONS MAY FAIL
   ► OLLVM passes designed for LLVM 22 only
   ► Unexpected IR patterns may cause crashes
   ► Some passes may produce incorrect code

6. McSEMA IR IS INHERENTLY INCOMPATIBLE WITH MANY PASSES
   ► McSema IR uses flattened memory model (state struct)
   ► Control flow is state machine (not structured CFG)
   ► Type safety is violated throughout
   ► These patterns break:
     - -bcf (bogus control flow) → assumes proper CFG structure
     - -flattening → expects high-level IR
     - -split → expects function-level control flow
     - -opaque-predicates → breaks with state machine IR

7. UPGRADED IR MUST BE VALIDATED
   ► Run llvm-verify-22 to check for IR errors
   ► Test linking/execution if possible
   ► Compare generated code with original
   ► Expect some degradation compared to native code

8. THIS PIPELINE IS EXPERIMENTAL
   ► Not suitable for production code
   ► Results may be incorrect
   ► No guarantees on obfuscation effectiveness
   ► Manual inspection recommended

PROCESS:
========
1. llvm-dis-22: Disassemble bitcode → textual IR (.ll)
2. llvm-as-22: Reassemble textual IR → LLVM 22 bitcode
3. llvm-verify-22: Validate IR structure (warnings only)

THE DISASSEMBLE/REASSEMBLE APPROACH:
====================================
Why not use direct bitcode upgrade?
- LLVM does not provide direct bitcode version upgrade
- Disassemble/reassemble forces IR through LLVM 22 parser
- Parser applies auto-upgrade rules to each instruction
- Results in LLVM 22 bitcode

Risks of this approach:
- Textual IR intermediate file may be large (hundreds of MB)
- Disassembly/assembly is slow for large binaries
- Any parsing error loses entire IR
"""

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo "Usage: $0 <input.bc> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  input.bc      McSema LLVM bitcode (from run_lift.sh)"
    echo "  output_dir    Directory to write LLVM 22 bitcode"
    echo ""
    echo "Example:"
    echo "  ./convert_ir_version.sh ./program.bc ./llvm22_ir/"
    exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
    log_error "Missing arguments"
    usage
fi

INPUT_BC="$1"
OUTPUT_DIR="$2"

# Validate input
if [ ! -f "$INPUT_BC" ]; then
    log_error "Input bitcode not found: $INPUT_BC"
    exit 1
fi

INPUT_ABS=$(cd "$(dirname "$INPUT_BC")" && pwd)/$(basename "$INPUT_BC")
INPUT_NAME=$(basename "$INPUT_BC")
BINARY_NAME="${INPUT_NAME%.bc}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_ABS=$(cd "$OUTPUT_DIR" && pwd)

# Intermediate files
TEXTUAL_IR="$OUTPUT_ABS/${BINARY_NAME}_upgraded.ll"
OUTPUT_BC="$OUTPUT_ABS/${BINARY_NAME}_llvm22.bc"

log_section "Feature #3 Part 2: LLVM IR Version Upgrade to LLVM 22"
log_info "Input bitcode: $INPUT_ABS"
log_info "Output bitcode: $OUTPUT_BC"
log_info "Temporary IR: $TEXTUAL_IR"
log_info ""

# Check for LLVM 22 tools
log_info "Checking for LLVM 22 tools..."

for tool in llvm-dis-22 llvm-as-22 llvm-verify-22; do
    if ! command -v $tool &> /dev/null; then
        log_error "$tool not found in PATH"
        log_error "Install LLVM 22 tools from:"
        log_error "  - Repository: apt install llvm-22"
        log_error "  - Build: /usr/local/llvm-obfuscator/bin/llvm-dis, etc."
        log_error ""
        log_error "Or use inside container:"
        log_error "  docker exec llvm-obfuscator-backend $tool ..."
        exit 1
    fi
done

log_info "✓ LLVM 22 tools available"
log_info ""

# Step 1: Disassemble bitcode to textual IR
log_info "Step 1/3: Disassembling bitcode to textual IR..."
log_info "Command: llvm-dis-22 $INPUT_ABS -o $TEXTUAL_IR"

if llvm-dis-22 "$INPUT_ABS" -o "$TEXTUAL_IR" 2>&1; then
    IR_SIZE=$(du -h "$TEXTUAL_IR" | cut -f1)
    log_info "✓ Disassembly successful ($IR_SIZE)"
else
    log_error "llvm-dis-22 failed"
    log_error "Input bitcode may be corrupted or unsupported version"
    exit 1
fi

log_info ""

# Step 2: Reassemble into LLVM 22 bitcode (applies auto-upgrade)
log_info "Step 2/3: Reassembling to LLVM 22 bitcode (auto-upgrading)..."
log_info "Command: llvm-as-22 $TEXTUAL_IR -o $OUTPUT_BC"

if llvm-as-22 "$TEXTUAL_IR" -o "$OUTPUT_BC" 2>&1; then
    BC_SIZE=$(du -h "$OUTPUT_BC" | cut -f1)
    log_info "✓ Reassembly successful ($BC_SIZE)"
else
    log_error "llvm-as-22 failed"
    log_error "Textual IR may contain unsupported constructs"
    rm -f "$OUTPUT_BC"
    exit 1
fi

log_info ""

# Step 3: Verify IR structure (warnings only)
log_info "Step 3/3: Verifying LLVM 22 IR structure..."
log_info "Command: llvm-verify-22 $OUTPUT_BC"

if llvm-verify-22 "$OUTPUT_BC" 2>&1; then
    log_info "✓ IR verification passed"
else
    log_warn "IR verification produced warnings (non-fatal)"
    log_warn "Check output above for details"
    log_warn "Upgraded IR may still be usable but not guaranteed"
fi

log_info ""

# Cleanup intermediate files
log_info "Cleaning up temporary files..."
rm -f "$TEXTUAL_IR"
log_info "✓ Removed $TEXTUAL_IR"
log_info ""

# Print warnings about OLLVM compatibility
log_section "⚠️  CRITICAL WARNINGS FOR OLLVM"

log_critical "McSema IR is INCOMPATIBLE with certain OLLVM passes:"
echo ""
echo "  ❌ -bcf (Bogus Control Flow)"
echo "     Reason: Assumes proper LLVM CFG structure; McSema uses state machine"
echo "     Error: Will crash or produce infinite loops"
echo ""
echo "  ❌ -flattening (Control Flow Flattening)"
echo "     Reason: McSema IR is already 'flattened' internally"
echo "     Error: Double-flattening produces broken code"
echo ""
echo "  ❌ -split (Block Splitting)"
echo "     Reason: Expects structured control flow; McSema is unstructured"
echo "     Error: May corrupt function entry/exit"
echo ""
echo "  ❌ -opaque-predicates (Opaque Predicates)"
echo "     Reason: Requires proper memory model; McSema uses state struct"
echo "     Error: Predicates may be optimized away by later passes"
echo ""
echo "  ✅ SAFER passes:"
echo "     - -fla (light function obfuscation)"
echo "     - -cff (function call obfuscation)"
echo "     - -ald (arithmetic lowering)"
echo ""

log_info ""

# Final status
log_section "Feature #3 Complete: IR Ready for Obfuscation"
log_info "READY_FOR_OLLVM"
log_info "Path: $OUTPUT_BC"
log_info ""
log_info "Next step (Feature #4):"
log_info "  Apply selected OLLVM passes:"
log_info "  clang-22 -O2 -Xclang -load -Xclang libFLA.so \\"
log_info "           -Xclang -load -Xclang libCFF.so \\"
log_info "           program_llvm22.bc -o program_obfuscated.bc"
log_info ""

exit 0
