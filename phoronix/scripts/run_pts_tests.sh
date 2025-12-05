#!/usr/bin/env bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PTS_INSTALL_DIR="${PTS_INSTALL_DIR:-/opt/phoronix-test-suite}"
PTS_CMD="$PTS_INSTALL_DIR/phoronix-test-suite"
REPORTS_BASE_DIR="${REPORTS_BASE_DIR:-$(pwd)/reports}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/tmp/pts_run_${TIMESTAMP}.log"

# Test suite definitions for automatic mode
AUTOMATIC_TESTS=(
    "pts/compress-7zip"
    "pts/fio"
    "pts/stream"
    "pts/sysbench"
)

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${CYAN}=========================================="
    echo "$1"
    echo "==========================================${NC}" | tee -a "$LOG_FILE"
}

check_pts_installed() {
    if [ ! -f "$PTS_CMD" ]; then
        log_error "Phoronix Test Suite not found at $PTS_CMD"
        log_info "Please run: scripts/install_phoronix.sh"
        return 1
    fi

    if ! $PTS_CMD version 2>/dev/null | grep -q "Phoronix"; then
        log_error "Phoronix Test Suite verification failed"
        return 1
    fi

    return 0
}

setup_directories() {
    mkdir -p "$REPORTS_BASE_DIR/raw/$TIMESTAMP"
    mkdir -p "$REPORTS_BASE_DIR/combined"
    mkdir -p "$REPORTS_BASE_DIR/manual"

    log_success "Report directories created"
}

install_test_dependencies() {
    local test_profile=$1

    log_info "Installing dependencies for $test_profile..."

    if ! $PTS_CMD batch-install "$test_profile" 2>&1 | tee -a "$LOG_FILE"; then
        log_warning "Some dependencies may not have installed (non-fatal)"
    fi

    log_success "Dependency installation completed"
}

run_single_test() {
    local test_profile=$1
    local output_dir=$2

    log_info "Running test: $test_profile"

    # Create a unique identifier for this test run
    local test_identifier="${test_profile##*/}_${TIMESTAMP}"

    # Run the test with batch mode (non-interactive)
    if $PTS_CMD batch-run "$test_profile" -s "$test_identifier" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Test completed: $test_profile"
        return 0
    else
        log_error "Test failed: $test_profile"
        return 1
    fi
}

export_results_json() {
    local result_file=$1
    local output_json=$2

    log_info "Exporting results to JSON: $output_json"

    if [ ! -f "$result_file" ]; then
        log_error "Result file not found: $result_file"
        return 1
    fi

    if $PTS_CMD result-file-to-json "$result_file" > "$output_json" 2>&1; then
        log_success "JSON export completed: $output_json"
        return 0
    else
        log_warning "JSON export had issues (non-fatal)"
        return 0
    fi
}

export_results_html() {
    local result_file=$1
    local output_html=$2

    log_info "Exporting results to HTML: $output_html"

    if [ ! -f "$result_file" ]; then
        log_error "Result file not found: $result_file"
        return 1
    fi

    if $PTS_CMD result-file-to-html "$result_file" > "$output_html" 2>&1; then
        log_success "HTML export completed: $output_html"
        return 0
    else
        log_warning "HTML export had issues (non-fatal)"
        return 0
    fi
}

run_automatic_mode() {
    print_header "Running in AUTOMATIC MODE - All Tests"

    setup_directories

    local raw_dir="$REPORTS_BASE_DIR/raw/$TIMESTAMP"
    local test_count=0
    local passed_count=0
    local failed_tests=()

    log_info "Total tests to run: ${#AUTOMATIC_TESTS[@]}"

    for test_profile in "${AUTOMATIC_TESTS[@]}"; do
        ((test_count++))
        log_info "[$test_count/${#AUTOMATIC_TESTS[@]}] Processing: $test_profile"

        if install_test_dependencies "$test_profile"; then
            if run_single_test "$test_profile" "$raw_dir"; then
                ((passed_count++))
            else
                failed_tests+=("$test_profile")
            fi
        else
            log_warning "Skipping $test_profile due to dependency installation failure"
            failed_tests+=("$test_profile")
        fi

        log_info "Progress: $passed_count/$test_count tests passed"
    done

    # Generate combined report
    log_info "Generating combined report..."

    local combined_report="$REPORTS_BASE_DIR/combined/pts_full_report_${TIMESTAMP}.html"
    local combined_json="$REPORTS_BASE_DIR/combined/pts_full_report_${TIMESTAMP}.json"

    # Create a summary report
    cat > "$combined_report" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Phoronix Test Suite - Full Report $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background-color: #333; color: white; padding: 20px; border-radius: 5px; }
        .summary { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .test-result { border-left: 4px solid #ddd; padding: 15px; margin: 10px 0; }
        .passed { border-left-color: #28a745; background-color: #f0f8f0; }
        .failed { border-left-color: #dc3545; background-color: #f8f0f0; }
        .footer { text-align: center; color: #666; margin-top: 30px; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Phoronix Test Suite - Full Benchmark Report</h1>
        <p>Generated: $TIMESTAMP</p>
    </div>

    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Tests</td>
                <td>$test_count</td>
            </tr>
            <tr>
                <td>Passed</td>
                <td><span style="color: green;">$passed_count</span></td>
            </tr>
            <tr>
                <td>Failed</td>
                <td><span style="color: red;">${#failed_tests[@]}</span></td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td>$(echo "scale=2; $passed_count * 100 / $test_count" | bc)%</td>
            </tr>
            <tr>
                <td>Timestamp</td>
                <td>$TIMESTAMP</td>
            </tr>
        </table>
    </div>

    <div class="summary">
        <h2>Test Results</h2>
        $(for test in "${AUTOMATIC_TESTS[@]}"; do
            if [[ " ${failed_tests[@]} " =~ " ${test} " ]]; then
                echo "<div class=\"test-result failed\">❌ $test - FAILED</div>"
            else
                echo "<div class=\"test-result passed\">✅ $test - PASSED</div>"
            fi
        done)
    </div>

    <div class="summary">
        <h2>Output Details</h2>
        <ul>
            <li><strong>Raw Results:</strong> $raw_dir</li>
            <li><strong>JSON Report:</strong> $combined_json</li>
            <li><strong>Log File:</strong> $LOG_FILE</li>
        </ul>
    </div>

    <div class="footer">
        <p>Phoronix Test Suite - Automated Benchmarking Report</p>
        <p>Report generated on $(date)</p>
    </div>
</body>
</html>
EOF

    # Create summary JSON
    cat > "$combined_json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "mode": "automatic",
    "summary": {
        "total_tests": $test_count,
        "passed": $passed_count,
        "failed": ${#failed_tests[@]},
        "success_rate": $(echo "scale=2; $passed_count * 100 / $test_count" | bc)
    },
    "tests": [
        $(for test in "${AUTOMATIC_TESTS[@]}"; do
            if [[ " ${failed_tests[@]} " =~ " ${test} " ]]; then
                echo "        { \"name\": \"$test\", \"status\": \"failed\" },"
            else
                echo "        { \"name\": \"$test\", \"status\": \"passed\" },"
            fi
        done | sed '$ s/,$//')
    ],
    "failed_tests": [
        $(for test in "${failed_tests[@]}"; do
            echo "        \"$test\","
        done | sed '$ s/,$//')
    ],
    "log_file": "$LOG_FILE",
    "report_dir": "$raw_dir"
}
EOF

    log_success "Combined reports generated:"
    log_info "  HTML: $combined_report"
    log_info "  JSON: $combined_json"

    print_header "AUTOMATIC MODE COMPLETED"

    if [ ${#failed_tests[@]} -eq 0 ]; then
        log_success "All tests passed!"
        return 0
    else
        log_warning "Some tests failed: ${failed_tests[*]}"
        return 1
    fi
}

run_manual_mode() {
    local test_profile=$1

    print_header "Running in MANUAL MODE - Single Test"

    log_info "Test Profile: $test_profile"

    setup_directories

    local manual_dir="$REPORTS_BASE_DIR/manual/${test_profile##*/}_${TIMESTAMP}"
    mkdir -p "$manual_dir"

    # Install dependencies
    if ! install_test_dependencies "$test_profile"; then
        log_error "Failed to install dependencies"
        return 1
    fi

    # Run test
    if ! run_single_test "$test_profile" "$manual_dir"; then
        log_error "Test execution failed"
        return 1
    fi

    # Generate reports
    # Note: PTS stores results in its own directory, we'll try to find and export them
    local pts_results_dir="$HOME/.phoronix-test-suite/results"

    if [ -d "$pts_results_dir" ]; then
        log_info "Searching for test results in $pts_results_dir..."

        # Find the most recent result
        local latest_result=$(find "$pts_results_dir" -name "*.xml" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

        if [ -n "$latest_result" ] && [ -f "$latest_result" ]; then
            log_info "Found result: $latest_result"

            local result_base="${latest_result%.xml}"
            local result_name=$(basename "$result_base")

            # Export to JSON
            export_results_json "$latest_result" "$manual_dir/${result_name}.json"

            # Export to HTML
            export_results_html "$latest_result" "$manual_dir/${result_name}.html"

            log_success "Results exported to $manual_dir"
        else
            log_warning "Could not find test result files"
        fi
    else
        log_warning "PTS results directory not found at $pts_results_dir"
    fi

    # Create summary report for manual run
    cat > "$manual_dir/summary.txt" << EOF
Manual Test Execution Summary
============================
Test Profile: $test_profile
Timestamp: $TIMESTAMP
Output Directory: $manual_dir
Log File: $LOG_FILE

Results:
- Test execution completed
- Reports generated in: $manual_dir
- JSON Report: $manual_dir/result.json (if available)
- HTML Report: $manual_dir/result.html (if available)

For more information, check:
- $LOG_FILE
EOF

    log_success "Manual test completed"
    log_info "Results stored in: $manual_dir"

    print_header "MANUAL MODE COMPLETED"

    return 0
}

print_usage() {
    cat << EOF
${CYAN}Phoronix Test Suite Runner${NC}

Usage: $0 [MODE] [OPTIONS]

Modes:
  --automatic          Run all default tests (automatic mode)
  --manual [TEST]      Run a single test profile (manual mode)
                       Example: $0 --manual pts/compress-7zip
  --help              Show this help message

Examples:
  $0 --automatic                    # Run all tests
  $0 --manual pts/stream            # Run only stream test
  $0 --manual pts/sysbench          # Run only sysbench test

Default test suites (automatic mode):
$(for test in "${AUTOMATIC_TESTS[@]}"; do echo "  - $test"; done)

Environment Variables:
  PTS_INSTALL_DIR     Location of Phoronix Test Suite (default: $PTS_INSTALL_DIR)
  REPORTS_BASE_DIR    Base directory for reports (default: $REPORTS_BASE_DIR)

Report Locations:
  Automatic: \$REPORTS_BASE_DIR/combined/
  Manual:    \$REPORTS_BASE_DIR/manual/
  Raw:       \$REPORTS_BASE_DIR/raw/

EOF
}

main() {
    # Initialize log file
    > "$LOG_FILE"

    log_info "=========================================="
    log_info "Phoronix Test Suite Runner"
    log_info "=========================================="

    # Check PTS installation
    if ! check_pts_installed; then
        log_error "Phoronix Test Suite is not installed"
        exit 1
    fi

    log_success "Phoronix Test Suite is installed"

    # Parse arguments
    if [ $# -eq 0 ]; then
        log_error "No mode specified"
        print_usage
        exit 1
    fi

    case "${1:-}" in
        --automatic)
            run_automatic_mode
            ;;
        --manual)
            if [ $# -lt 2 ]; then
                log_error "No test profile specified for manual mode"
                echo ""
                print_usage
                exit 1
            fi

            # Build the test profile from remaining arguments
            local test_profile="${2}"
            shift 2

            run_manual_mode "$test_profile" "$@"
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown mode: $1"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
