#!/bin/bash

##############################################################################
# Obfuscation Security Analysis Script
#
# Performs automated decompilation & difficulty scoring on binaries.
# Uses Ghidra (headless) or fallbacks to objdump + heuristics.
##############################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GHIDRA_INSTALL_PATH="${GHIDRA_INSTALL_PATH:-/opt/ghidra}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

# Logging
LOG_LEVEL="${LOG_LEVEL:-INFO}"

log_info() {
    echo "[INFO] $*" >&2
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

##############################################################################
# Check for required tools
##############################################################################
check_dependencies() {
    local missing=0

    for tool in objdump readelf nm strings; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Missing required tool: $tool"
            missing=$((missing + 1))
        fi
    done

    if [ "$missing" -gt 0 ]; then
        log_error "Missing $missing required tools. Install binutils."
        return 1
    fi

    return 0
}

##############################################################################
# Check if Ghidra is available
##############################################################################
check_ghidra() {
    if [ -f "$GHIDRA_INSTALL_PATH/support/analyzeHeadless" ]; then
        log_info "Ghidra found at $GHIDRA_INSTALL_PATH"
        return 0
    else
        log_warn "Ghidra not found. Will fallback to heuristics."
        return 1
    fi
}

##############################################################################
# Extract decompilation metrics using Ghidra
##############################################################################
analyze_with_ghidra() {
    local binary_path="$1"
    local output_json="$2"

    log_info "Analyzing $binary_path with Ghidra..."

    # Create temporary project
    local temp_project="/tmp/ghidra_analysis_$$"
    mkdir -p "$temp_project"

    # Run Ghidra headless analysis
    if timeout "$TIMEOUT_SECONDS" "$GHIDRA_INSTALL_PATH/support/analyzeHeadless" \
        "$temp_project" \
        "project_$$" \
        -import "$binary_path" \
        -scriptPath "$SCRIPT_DIR" \
        -postScript extract_metrics.py "$output_json" \
        -deleteProject 2>/dev/null; then
        log_info "Ghidra analysis completed"
        return 0
    else
        log_warn "Ghidra analysis failed or timed out"
        return 1
    fi
}

##############################################################################
# Fallback: Heuristic-based decompilation analysis
##############################################################################
analyze_with_heuristics() {
    local binary_path="$1"
    local output_json="$2"

    log_info "Analyzing $binary_path with heuristics..."

    # Create Python script to perform analysis
    cat > /tmp/heuristic_analysis_$$.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import json
import subprocess
import re
import sys
from collections import Counter
from pathlib import Path

def analyze_binary(binary_path):
    """Perform heuristic-based decompilation analysis."""
    results = {
        "irreducible_cfg_detected": False,
        "irreducible_cfg_percentage": 0.0,
        "opaque_predicates_count": 0,
        "opaque_predicates_percentage": 0.0,
        "basic_blocks_recovered": 0,
        "recovery_percentage": 0.0,
        "string_obfuscation_ratio": 0.0,
        "symbol_obfuscation_ratio": 0.0,
        "decompilation_readability_score": 5.0,
        "analysis_method": "heuristics",
    }

    try:
        # Extract control flow
        irreducible_count = estimate_irreducible_cfg(binary_path)
        results["irreducible_cfg_detected"] = irreducible_count > 0
        results["irreducible_cfg_percentage"] = irreducible_count

        # Count opaque predicates (heuristic: repeated patterns)
        opaque_count = count_opaque_predicates(binary_path)
        results["opaque_predicates_count"] = opaque_count

        # Estimate basic block recovery
        bb_recovered = estimate_basic_block_recovery(binary_path)
        results["basic_blocks_recovered"] = bb_recovered["count"]
        results["recovery_percentage"] = bb_recovered["percentage"]

        # String obfuscation analysis
        string_ratio = analyze_string_obfuscation(binary_path)
        results["string_obfuscation_ratio"] = round(string_ratio, 3)

        # Symbol obfuscation analysis
        symbol_ratio = analyze_symbol_obfuscation(binary_path)
        results["symbol_obfuscation_ratio"] = round(symbol_ratio, 3)

        # Compute readability score
        results["decompilation_readability_score"] = compute_readability_score(results)

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        return None

    return results

def estimate_irreducible_cfg(binary_path):
    """Estimate percentage of irreducible control flow structures."""
    try:
        result = subprocess.run(
            ["objdump", "-d", binary_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Count loops with complex control flow (heuristic)
        complex_patterns = 0
        total_jumps = 0

        for line in result.stdout.split('\n'):
            if re.match(r'^\s+[0-9a-f]+:\s+.*\s(j[a-z]+|b[a-z]*)', line):
                total_jumps += 1
                # Detect backward jumps (loops)
                if any(x in line for x in ['je', 'jne', 'jle', 'jge', 'jl', 'jg']):
                    complex_patterns += 1

        percentage = (complex_patterns / total_jumps * 100) if total_jumps > 0 else 0
        return round(percentage, 2)
    except Exception as e:
        print(f"[WARN] Failed to estimate irreducible CFG: {e}", file=sys.stderr)
        return 0.0

def count_opaque_predicates(binary_path):
    """Heuristically count opaque predicates."""
    try:
        result = subprocess.run(
            ["objdump", "-d", binary_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Look for patterns: consecutive identical comparisons
        opcodes = []
        for line in result.stdout.split('\n'):
            match = re.search(r'\s([a-z]+)\s+', line)
            if match:
                opcodes.append(match.group(1))

        # Count repeated instruction sequences (opaque predicate heuristic)
        counter = Counter(opcodes)
        high_frequency = sum(1 for count in counter.values() if count > 10)

        return high_frequency
    except Exception as e:
        print(f"[WARN] Failed to count opaque predicates: {e}", file=sys.stderr)
        return 0

def estimate_basic_block_recovery(binary_path):
    """Estimate percentage of basic blocks that could be recovered."""
    try:
        result = subprocess.run(
            ["objdump", "-d", binary_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Count function headers (clear basic block starts)
        func_pattern = r'^[0-9a-f]+ <[^>]+>:'
        functions = sum(1 for line in result.stdout.split('\n') if re.match(func_pattern, line))

        # Count jumps (basic block boundaries)
        jump_pattern = r'^\s+[0-9a-f]+:\s+.*\s(j[a-z]+)'
        jumps = sum(1 for line in result.stdout.split('\n') if re.search(jump_pattern, line))

        # Estimate: functions + jumps = recoverable basic blocks
        recoverable = functions + jumps

        return {
            "count": recoverable,
            "percentage": min(100.0, (recoverable / max(functions, 1)) * 100),
        }
    except Exception as e:
        print(f"[WARN] Failed to estimate BB recovery: {e}", file=sys.stderr)
        return {"count": 0, "percentage": 0.0}

def analyze_string_obfuscation(binary_path):
    """Analyze string obfuscation ratio."""
    try:
        result = subprocess.run(
            ["strings", binary_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        total_strings = len(result.stdout.strip().split('\n'))
        # Count readable strings (length > 3, printable)
        readable = sum(
            1 for s in result.stdout.split('\n')
            if len(s.strip()) > 3 and s.isprintable()
        )

        ratio = 1.0 - (readable / total_strings) if total_strings > 0 else 0.0
        return max(0.0, min(1.0, ratio))
    except Exception as e:
        print(f"[WARN] String analysis failed: {e}", file=sys.stderr)
        return 0.0

def analyze_symbol_obfuscation(binary_path):
    """Analyze symbol obfuscation ratio."""
    try:
        result = subprocess.run(
            ["nm", "-D", binary_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        total_symbols = len(result.stdout.strip().split('\n'))
        # Count obfuscated symbols (short, random names)
        obfuscated = sum(
            1 for line in result.stdout.split('\n')
            if len(line.split()[-1]) <= 2
        )

        ratio = obfuscated / total_symbols if total_symbols > 0 else 0.0
        return max(0.0, min(1.0, ratio))
    except Exception as e:
        print(f"[WARN] Symbol analysis failed: {e}", file=sys.stderr)
        return 0.0

def compute_readability_score(results):
    """
    Compute a decompilation readability score (0-10).

    Higher scores = more readable (less obfuscated).
    """
    score = 10.0

    # Irreducible CFG makes decompilation harder
    score -= results["irreducible_cfg_percentage"] / 10.0

    # Opaque predicates reduce readability
    score -= min(3.0, results["opaque_predicates_count"] / 10.0)

    # Symbol obfuscation impacts readability
    score -= results["symbol_obfuscation_ratio"] * 2.0

    # String obfuscation impacts readability
    score -= results["string_obfuscation_ratio"] * 1.0

    # Low BB recovery means poor decompilation
    score -= (100 - results["recovery_percentage"]) / 20.0

    return max(0.0, min(10.0, score))

if __name__ == '__main__':
    binary_path = sys.argv[1]
    output_json = sys.argv[2]

    results = analyze_binary(binary_path)
    if results:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
    else:
        sys.exit(1)
PYTHON_SCRIPT

    python3 /tmp/heuristic_analysis_$$.py "$binary_path" "$output_json"
    local exit_code=$?
    rm -f /tmp/heuristic_analysis_$$.py
    return $exit_code
}

##############################################################################
# Main analysis function
##############################################################################
analyze_binary() {
    local binary_path="$1"
    local output_dir="$2"

    if [ ! -f "$binary_path" ]; then
        log_error "Binary not found: $binary_path"
        return 1
    fi

    local binary_name=$(basename "$binary_path")
    local output_json="$output_dir/${binary_name}_security_analysis.json"

    mkdir -p "$output_dir"

    # Try Ghidra first, fallback to heuristics
    if check_ghidra; then
        if analyze_with_ghidra "$binary_path" "$output_json"; then
            log_info "Successfully analyzed with Ghidra"
            return 0
        fi
    fi

    # Fallback to heuristics
    if analyze_with_heuristics "$binary_path" "$output_json"; then
        log_info "Successfully analyzed with heuristics"
        return 0
    fi

    log_error "All analysis methods failed for $binary_path"
    return 1
}

##############################################################################
# Main
##############################################################################
main() {
    local binary_path=""
    local output_dir="."

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--binary)
                binary_path="$2"
                shift 2
                ;;
            -o|--output)
                output_dir="$2"
                shift 2
                ;;
            *)
                if [ -z "$binary_path" ]; then
                    binary_path="$1"
                fi
                shift
                ;;
        esac
    done

    if [ -z "$binary_path" ]; then
        log_error "Usage: $0 <binary_path> [-o output_dir]"
        return 1
    fi

    check_dependencies || return 1
    analyze_binary "$binary_path" "$output_dir"
}

main "$@"
