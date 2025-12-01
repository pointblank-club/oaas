#!/bin/bash
# Quick test runner script

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SUITE_DIR}/results"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <baseline_binary> <obfuscated_binary> [program_name]"
    echo "Example: $0 ./app_baseline ./app_obfuscated my_app"
    exit 1
fi

BASELINE="$1"
OBFUSCATED="$2"
PROGRAM_NAME="${3:-program}"

if [ ! -f "$BASELINE" ]; then
    echo "Error: Baseline binary not found: $BASELINE"
    exit 1
fi

if [ ! -f "$OBFUSCATED" ]; then
    echo "Error: Obfuscated binary not found: $OBFUSCATED"
    exit 1
fi

echo "=========================================="
echo "OLLVM Obfuscation Test Suite"
echo "=========================================="
echo "Baseline:    $BASELINE"
echo "Obfuscated:  $OBFUSCATED"
echo "Program:     $PROGRAM_NAME"
echo "Results:     $RESULTS_DIR"
echo "=========================================="
echo ""

python3 "$SUITE_DIR/obfuscation_test_suite.py" "$BASELINE" "$OBFUSCATED" \
    -r "$RESULTS_DIR" \
    -n "$PROGRAM_NAME"

REPORT_DIR="$RESULTS_DIR/reports/$PROGRAM_NAME"

if [ -d "$REPORT_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "Reports Generated"
    echo "=========================================="
    echo "Summary:  $REPORT_DIR/${PROGRAM_NAME}_summary.txt"
    echo "Full:     $REPORT_DIR/${PROGRAM_NAME}_report.txt"
    echo "JSON:     $REPORT_DIR/${PROGRAM_NAME}_results.json"
    echo ""
    echo "Quick Summary:"
    echo "==========================================="
    cat "$REPORT_DIR/${PROGRAM_NAME}_summary.txt" 2>/dev/null || echo "Summary not available"
fi
