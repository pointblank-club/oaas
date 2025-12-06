#!/bin/bash

##############################################################################
# Unified Obfuscation Analysis Test Suite
#
# Runs complete obfuscation analysis on any pair of binaries:
# - Metrics collection
# - Security analysis (with Ghidra if available)
# - Aggregated reporting
#
# Usage:
#   bash run_obfuscation_test_suite.sh <baseline_binary> <obfuscated_binary> [output_dir]
#
# Example:
#   bash run_obfuscation_test_suite.sh ./baseline /home/user/Downloads/obfuscated results/
##############################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASELINE_BINARY="${1:?ERROR: Baseline binary path required}"
OBFUSCATED_BINARY="${2:?ERROR: Obfuscated binary path required}"
OUTPUT_BASE_DIR="${3:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate binaries exist
if [ ! -f "$BASELINE_BINARY" ]; then
    echo -e "${RED}ERROR: Baseline binary not found: $BASELINE_BINARY${NC}"
    exit 1
fi

if [ ! -f "$OBFUSCATED_BINARY" ]; then
    echo -e "${RED}ERROR: Obfuscated binary not found: $OBFUSCATED_BINARY${NC}"
    exit 1
fi

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_NAME=$(basename "$BASELINE_BINARY")_vs_$(basename "$OBFUSCATED_BINARY")
RESULTS_DIR="${OUTPUT_BASE_DIR}/obfuscation_analysis_${TEST_NAME}_${TIMESTAMP}"

# Create subdirectories for organized reports
mkdir -p "$RESULTS_DIR"/{metrics,security,reports,logs}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$RESULTS_DIR/logs/execution.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$RESULTS_DIR/logs/execution.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$RESULTS_DIR/logs/execution.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$RESULTS_DIR/logs/execution.log"
}

##############################################################################
# Phase 1: Collect Obfuscation Metrics
##############################################################################
run_metrics_collection() {
    log_info "=== Phase 1: Collecting Obfuscation Metrics ==="

    local metrics_script="${SCRIPT_DIR}/collect_obfuscation_metrics.py"

    if [ ! -f "$metrics_script" ]; then
        log_error "Metrics collection script not found: $metrics_script"
        return 1
    fi

    log_info "Baseline:    $BASELINE_BINARY"
    log_info "Obfuscated:  $OBFUSCATED_BINARY"
    log_info "Output:      $RESULTS_DIR/metrics/"

    if python3 "$metrics_script" \
        "$BASELINE_BINARY" \
        "$OBFUSCATED_BINARY" \
        --config "test_${TEST_NAME}" \
        --output "$RESULTS_DIR/metrics/" 2>&1 | tee -a "$RESULTS_DIR/logs/metrics.log"; then
        log_success "Metrics collection completed"
        return 0
    else
        log_error "Metrics collection failed"
        return 1
    fi
}

##############################################################################
# Phase 2: Security Analysis
##############################################################################
run_security_analysis() {
    log_info "=== Phase 2: Running Security Analysis ==="

    local security_script="${SCRIPT_DIR}/run_security_analysis.sh"
    local security_output="${RESULTS_DIR}/security/security_analysis.json"

    if [ ! -f "$security_script" ]; then
        log_error "Security analysis script not found: $security_script"
        return 1
    fi

    log_info "Analyzing: $BASELINE_BINARY"

    # Check if Ghidra is available
    if [ -d "/opt/ghidra" ]; then
        log_info "Ghidra found - will use real decompilation (85-90% accuracy)"
        export GHIDRA_INSTALL_PATH="/opt/ghidra"
    else
        log_warn "Ghidra not found - will use heuristics (40% accuracy)"
    fi

    if bash "$security_script" "$BASELINE_BINARY" -o "$security_output" 2>&1 | tee -a "$RESULTS_DIR/logs/security.log"; then
        log_success "Security analysis completed"
        return 0
    else
        log_error "Security analysis failed"
        return 1
    fi
}

##############################################################################
# Phase 3: Generate Aggregated Report
##############################################################################
run_aggregated_report() {
    log_info "=== Phase 3: Generating Aggregated Reports ==="

    local report_script="${SCRIPT_DIR}/aggregate_obfuscation_report.py"

    if [ ! -f "$report_script" ]; then
        log_error "Report aggregation script not found: $report_script"
        return 1
    fi

    log_info "Input metrics:  $RESULTS_DIR/metrics/metrics.json"
    log_info "Input security: $RESULTS_DIR/security/security_analysis.json"
    log_info "Output reports: $RESULTS_DIR/reports/"

    if python3 "$report_script" "test_${TEST_NAME}" \
        --metrics "$RESULTS_DIR/metrics/metrics.json" \
        --security "$RESULTS_DIR/security/security_analysis.json" \
        --output "$RESULTS_DIR/reports/" 2>&1 | tee -a "$RESULTS_DIR/logs/report.log"; then
        log_success "Aggregated reports generated"
        return 0
    else
        log_error "Report generation failed"
        return 1
    fi
}

##############################################################################
# Phase 4: Create Summary Index
##############################################################################
create_summary_index() {
    log_info "=== Phase 4: Creating Summary Index ==="

    local index_file="$RESULTS_DIR/INDEX.md"

    cat > "$index_file" << EOF
# Obfuscation Analysis Test Results

**Test Date:** $(date)

**Timestamp:** $TIMESTAMP

**Test Name:** $TEST_NAME

---

## Test Binaries

- **Baseline:** $BASELINE_BINARY
  - Size: $(stat -f%z "$BASELINE_BINARY" 2>/dev/null || stat -c%s "$BASELINE_BINARY" 2>/dev/null) bytes
  - Type: $(file -b "$BASELINE_BINARY")

- **Obfuscated:** $OBFUSCATED_BINARY
  - Size: $(stat -f%z "$OBFUSCATED_BINARY" 2>/dev/null || stat -c%s "$OBFUSCATED_BINARY" 2>/dev/null) bytes
  - Type: $(file -b "$OBFUSCATED_BINARY")

---

## Report Structure

\`\`\`
$RESULTS_DIR/
├── metrics/
│   ├── metrics.json          # Raw metrics data
│   ├── metrics.csv           # Spreadsheet format
│   └── metrics.md            # Markdown format
├── security/
│   └── security_analysis.json # Security & decompilation analysis
├── reports/
│   ├── final_report.json     # Complete aggregated report
│   ├── final_report.md       # Markdown report
│   ├── final_report.html     # Interactive HTML report
│   └── final_report.csv      # CSV format
├── logs/
│   ├── execution.log         # Full execution log
│   ├── metrics.log           # Metrics collection log
│   ├── security.log          # Security analysis log
│   └── report.log            # Report generation log
└── INDEX.md                  # This file
\`\`\`

---

## Quick Results

### Metrics
- See \`metrics/metrics.json\` for detailed binary metrics
- See \`metrics/metrics.csv\` for spreadsheet view

### Security Analysis
- See \`security/security_analysis.json\` for decompilation analysis

### Final Report
- **JSON:** \`reports/final_report.json\` - Programmatic access
- **Markdown:** \`reports/final_report.md\` - Human-readable
- **HTML:** \`reports/final_report.html\` - Interactive view
- **CSV:** \`reports/final_report.csv\` - Spreadsheet import

---

## Execution Log

See \`logs/execution.log\` for complete execution details.

---

## How to Use These Results

1. **For quick review:** Open \`reports/final_report.md\`
2. **For detailed analysis:** Use \`reports/final_report.json\`
3. **For visualization:** Open \`reports/final_report.html\` in browser
4. **For spreadsheets:** Import \`reports/final_report.csv\` to Excel

---

Generated with Obfuscation Test Suite
EOF

    log_success "Summary index created: $index_file"
}

##############################################################################
# Main Execution Flow
##############################################################################
main() {
    log_info "╔════════════════════════════════════════════════════════════╗"
    log_info "║     Unified Obfuscation Analysis Test Suite                ║"
    log_info "║                                                            ║"
    log_info "║  Testing complete obfuscation metrics and security         ║"
    log_info "╚════════════════════════════════════════════════════════════╝"
    log_info ""
    log_info "Output Directory: $RESULTS_DIR"
    log_info ""

    # Run all phases
    if ! run_metrics_collection; then
        log_error "Metrics collection failed"
        return 1
    fi
    log_info ""

    if ! run_security_analysis; then
        log_error "Security analysis failed"
        return 1
    fi
    log_info ""

    if ! run_aggregated_report; then
        log_error "Report generation failed"
        return 1
    fi
    log_info ""

    create_summary_index
    log_info ""

    # Print summary
    log_success "╔════════════════════════════════════════════════════════════╗"
    log_success "║                 Test Suite Completed!                      ║"
    log_success "╚════════════════════════════════════════════════════════════╝"
    log_info ""
    log_info "All reports generated in: $RESULTS_DIR"
    log_info ""
    log_info "Quick access:"
    log_info "  📋 Summary:  $RESULTS_DIR/INDEX.md"
    log_info "  📊 JSON:     $RESULTS_DIR/reports/final_report.json"
    log_info "  📝 Markdown: $RESULTS_DIR/reports/final_report.md"
    log_info "  🌐 HTML:     $RESULTS_DIR/reports/final_report.html"
    log_info "  📈 CSV:      $RESULTS_DIR/reports/final_report.csv"
    log_info "  📋 Metrics:  $RESULTS_DIR/metrics/metrics.json"
    log_info "  🔒 Security: $RESULTS_DIR/security/security_analysis.json"
    log_info ""
    log_info "View logs: tail -f $RESULTS_DIR/logs/execution.log"
    log_info ""
}

main "$@"
