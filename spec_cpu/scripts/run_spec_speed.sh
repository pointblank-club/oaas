#!/usr/bin/env bash

################################################################################
# SPEC CPU 2017 SPECspeed Runner for LLVM Obfuscator
################################################################################
#
# Purpose: Execute SPECspeed (single-threaded) benchmarks for baseline and obfuscated builds
#
# Functionality:
#   1. Execute SPECspeed benchmarks for baseline or obfuscated configurations
#   2. Organize results by timestamp and build type
#   3. Support running single config or all existing obfuscation configs
#   4. Validate build completion before running benchmarks
#   5. Log execution details, timing, and performance metrics
#   6. Handle failures gracefully and report summary statistics
#
# Exit Codes:
#   0 = All benchmarks completed successfully
#   1 = Configuration or setup error
#   2 = Baseline benchmark run failed
#   3 = Obfuscated benchmark run failed
#   4 = No valid benchmark runs found
#   5 = Results directory creation failed
#
################################################################################

set -euo pipefail

# =============================================================================
# Configuration and Paths
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$MODULE_ROOT")")"
CONFIGS_DIR="$MODULE_ROOT/configs"
RESULTS_DIR="$MODULE_ROOT/results"

# Build output directories
BUILD_BASE_DIR="$PROJECT_ROOT/build"
BUILD_BASELINE_DIR="$BUILD_BASE_DIR/baseline_spec"
BUILD_OBFUSCATED_BASE_DIR="$BUILD_BASE_DIR/obfuscated_spec"

# SPEC CPU location
SPEC_CPU_HOME="${SPEC_CPU_HOME:-/opt/spec2017}"
SPEC_CONFIG="$SPEC_CPU_HOME/config/llvm-obfuscation.cfg"

# Logging
LOG_DIR="$RESULTS_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_spec_speed.log"

# Run parameters
RUN_TYPE="${1:-baseline}"
CONFIG_NAME="${2:-}"
ITERATIONS="${3:-3}"

# Runtime tracking
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TIMESTAMP_DIR=$(date -u +"%Y%m%d_%H%M%S")
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# =============================================================================
# Color Output
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Logging Functions
# =============================================================================

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

log_header() {
    echo -e "${CYAN}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}========================================${NC}" | tee -a "$LOG_FILE"
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_spec_installation() {
    log_info "Validating SPEC CPU installation..."

    if [ ! -d "$SPEC_CPU_HOME" ]; then
        log_error "SPEC CPU not found at: $SPEC_CPU_HOME"
        return 1
    fi

    if [ ! -f "$SPEC_CONFIG" ]; then
        log_error "SPEC CPU config not found: $SPEC_CONFIG"
        log_info "Run: scripts/configure_spec_cpu.sh $RUN_TYPE $CONFIG_NAME"
        return 1
    fi

    if [ ! -x "$SPEC_CPU_HOME/bin/runcpu" ]; then
        log_error "SPEC CPU runcpu tool not found or not executable"
        return 1
    fi

    log_success "SPEC CPU installation validated"
    return 0
}

check_baseline_build_exists() {
    log_info "Checking for baseline build artifacts..."

    if [ ! -d "$BUILD_BASELINE_DIR" ]; then
        log_error "Baseline build directory not found: $BUILD_BASELINE_DIR"
        log_info "Run: scripts/build_spec_targets.sh baseline"
        return 1
    fi

    # Check for any compiled binaries or build artifacts
    if [ ! -f "$BUILD_BASELINE_DIR/BUILD_METADATA.txt" ]; then
        log_warning "Baseline build metadata not found (may still be valid)"
    fi

    log_success "Baseline build artifacts found"
    return 0
}

check_obfuscated_build_exists() {
    local config=$1

    log_info "Checking for obfuscated build artifacts ($config)..."

    local obf_build_dir="$BUILD_OBFUSCATED_BASE_DIR/$config"
    if [ ! -d "$obf_build_dir" ]; then
        log_error "Obfuscated build directory not found: $obf_build_dir"
        log_info "Run: scripts/build_spec_targets.sh obfuscated $config"
        return 1
    fi

    # Check for metadata
    if [ ! -f "$obf_build_dir/BUILD_METADATA.txt" ]; then
        log_warning "Obfuscated build metadata not found for $config"
    fi

    log_success "Obfuscated build artifacts found for: $config"
    return 0
}

# =============================================================================
# Directory Creation and Organization
# =============================================================================

create_results_directories() {
    log_info "Creating results directories..."

    case "$RUN_TYPE" in
        baseline)
            local result_dir="$RESULTS_DIR/baseline/$TIMESTAMP_DIR/speed"
            if ! mkdir -p "$result_dir"; then
                log_error "Failed to create baseline results directory"
                return 1
            fi
            log_success "Baseline results directory: $result_dir"
            echo "$result_dir"
            ;;

        obfuscated)
            if [ -z "$CONFIG_NAME" ]; then
                log_error "CONFIG_NAME required for obfuscated runs"
                return 1
            fi
            local result_dir="$RESULTS_DIR/obfuscated/$CONFIG_NAME/$TIMESTAMP_DIR/speed"
            if ! mkdir -p "$result_dir"; then
                log_error "Failed to create obfuscated results directory"
                return 1
            fi
            log_success "Obfuscated results directory: $result_dir"
            echo "$result_dir"
            ;;

        *)
            log_error "Unknown run type: $RUN_TYPE"
            return 1
            ;;
    esac

    return 0
}

get_available_obfuscation_configs() {
    log_info "Scanning for available obfuscation configurations..."

    if [ ! -d "$BUILD_OBFUSCATED_BASE_DIR" ]; then
        log_warning "No obfuscated build directory found"
        return 1
    fi

    local configs=()
    while IFS= read -r -d '' config_dir; do
        local config_name=$(basename "$config_dir")
        if [ -f "$config_dir/BUILD_METADATA.txt" ]; then
            configs+=("$config_name")
            log_info "  Found: $config_name"
        fi
    done < <(find "$BUILD_OBFUSCATED_BASE_DIR" -maxdepth 1 -mindepth 1 -type d -print0)

    if [ ${#configs[@]} -eq 0 ]; then
        log_warning "No valid obfuscation configurations found"
        return 1
    fi

    # Return array as space-separated string
    echo "${configs[@]}"
    return 0
}

# =============================================================================
# SPECspeed Execution
# =============================================================================

run_baseline_speed() {
    log_header "Running Baseline SPECspeed Benchmarks"

    local result_dir=$(create_results_directories)
    if [ $? -ne 0 ]; then
        log_error "Failed to create results directories"
        return 1
    fi

    # Verify build exists
    if ! check_baseline_build_exists; then
        log_error "Baseline build validation failed"
        return 1
    fi

    log_info "Results directory: $result_dir"
    log_info "Iterations: $ITERATIONS"
    log_info "Starting baseline SPECspeed benchmarks..."

    # Create benchmark run script
    local run_script="/tmp/spec_baseline_speed_$$.sh"
    cat > "$run_script" << 'SPEC_SCRIPT'
#!/bin/bash
cd "SPEC_HOME"
export SPEC_RUN="TIMESTAMP_DIR"
bin/runcpu \
    --config llvm-obfuscation.cfg \
    --size ref \
    --iterations ITERATIONS \
    --tune base \
    --output_format text,html,json \
    --loose \
    2>&1 | tee "RESULT_DIR/spec_baseline_speed.log"
SPEC_SCRIPT

    sed -i "s|SPEC_HOME|$SPEC_CPU_HOME|g" "$run_script"
    sed -i "s|TIMESTAMP_DIR|baseline_$TIMESTAMP_DIR|g" "$run_script"
    sed -i "s|ITERATIONS|$ITERATIONS|g" "$run_script"
    sed -i "s|RESULT_DIR|$result_dir|g" "$run_script"

    # Log execution details
    {
        echo "Baseline SPECspeed Run Details"
        echo "==============================="
        echo "Timestamp: $TIMESTAMP"
        echo "Run Directory: $TIMESTAMP_DIR"
        echo "Results Directory: $result_dir"
        echo "Iterations: $ITERATIONS"
        echo "SPEC CPU Home: $SPEC_CPU_HOME"
        echo "Config: $SPEC_CONFIG"
        echo "Execution Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
    } | tee -a "$LOG_FILE" > "$result_dir/EXECUTION_DETAILS.txt"

    # Execute benchmarks
    chmod +x "$run_script"
    if bash "$run_script"; then
        log_success "Baseline SPECspeed benchmarks completed"
        {
            echo "Execution End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Status: COMPLETED"
        } >> "$result_dir/EXECUTION_DETAILS.txt"
        rm -f "$run_script"
        return 0
    else
        log_error "Baseline SPECspeed benchmarks failed"
        {
            echo "Execution End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Status: FAILED"
        } >> "$result_dir/EXECUTION_DETAILS.txt"
        rm -f "$run_script"
        return 1
    fi
}

run_obfuscated_speed() {
    local config=$1

    log_header "Running Obfuscated SPECspeed Benchmarks ($config)"

    # Temporarily set CONFIG_NAME for directory creation
    CONFIG_NAME="$config"
    local result_dir=$(create_results_directories)
    if [ $? -ne 0 ]; then
        log_error "Failed to create results directories"
        return 1
    fi

    # Verify build exists
    if ! check_obfuscated_build_exists "$config"; then
        log_error "Obfuscated build validation failed for $config"
        return 1
    fi

    log_info "Results directory: $result_dir"
    log_info "Iterations: $ITERATIONS"
    log_info "Starting obfuscated SPECspeed benchmarks ($config)..."

    # Create benchmark run script
    local run_script="/tmp/spec_obfuscated_speed_$$.sh"
    cat > "$run_script" << 'SPEC_SCRIPT'
#!/bin/bash
cd "SPEC_HOME"
export SPEC_RUN="TIMESTAMP_DIR"
bin/runcpu \
    --config llvm-obfuscation.cfg \
    --size ref \
    --iterations ITERATIONS \
    --tune peak \
    --output_format text,html,json \
    --loose \
    2>&1 | tee "RESULT_DIR/spec_obfuscated_speed_CONFIG.log"
SPEC_SCRIPT

    sed -i "s|SPEC_HOME|$SPEC_CPU_HOME|g" "$run_script"
    sed -i "s|TIMESTAMP_DIR|obfuscated_${config}_$TIMESTAMP_DIR|g" "$run_script"
    sed -i "s|ITERATIONS|$ITERATIONS|g" "$run_script"
    sed -i "s|RESULT_DIR|$result_dir|g" "$run_script"
    sed -i "s|CONFIG|$config|g" "$run_script"

    # Log execution details
    {
        echo "Obfuscated SPECspeed Run Details"
        echo "================================="
        echo "Configuration: $config"
        echo "Timestamp: $TIMESTAMP"
        echo "Run Directory: $TIMESTAMP_DIR"
        echo "Results Directory: $result_dir"
        echo "Iterations: $ITERATIONS"
        echo "SPEC CPU Home: $SPEC_CPU_HOME"
        echo "Config: $SPEC_CONFIG"
        echo "Execution Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
    } | tee -a "$LOG_FILE" > "$result_dir/EXECUTION_DETAILS.txt"

    # Execute benchmarks
    chmod +x "$run_script"
    if bash "$run_script"; then
        log_success "Obfuscated SPECspeed benchmarks completed ($config)"
        {
            echo "Execution End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Status: COMPLETED"
        } >> "$result_dir/EXECUTION_DETAILS.txt"
        ((PASSED_TESTS++))
        rm -f "$run_script"
        return 0
    else
        log_error "Obfuscated SPECspeed benchmarks failed ($config)"
        {
            echo "Execution End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "Status: FAILED"
        } >> "$result_dir/EXECUTION_DETAILS.txt"
        ((FAILED_TESTS++))
        rm -f "$run_script"
        return 1
    fi
}

# =============================================================================
# Summary and Reporting
# =============================================================================

generate_summary() {
    log_header "SPECspeed Benchmark Summary"

    local summary_file="$RESULTS_DIR/speed_run_summary_$(date -u +%Y%m%d_%H%M%S).txt"

    {
        echo "SPEC CPU 2017 SPECspeed Run Summary"
        echo "===================================="
        echo ""
        echo "Execution Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""

        case "$RUN_TYPE" in
            baseline)
                echo "Run Type: Baseline"
                echo "Results: $RESULTS_DIR/baseline/$TIMESTAMP_DIR/speed"
                ;;
            obfuscated)
                echo "Run Type: Obfuscated"
                echo "Configurations Tested: $TOTAL_TESTS"
                echo "Passed: $PASSED_TESTS"
                echo "Failed: $FAILED_TESTS"
                echo "Results Directory: $RESULTS_DIR/obfuscated"
                ;;
        esac

        echo ""
        echo "Log File: $LOG_FILE"
        echo ""
    } | tee "$summary_file"

    log_success "Summary saved to: $summary_file"
}

cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/spec_*_speed_$$.sh
}

print_help() {
    cat << EOF
${CYAN}SPEC CPU 2017 SPECspeed Runner${NC}

Usage: $(basename "$0") [RUN_TYPE] [CONFIG_NAME] [ITERATIONS]

Run Types:
  baseline      Run SPECspeed for baseline build
  obfuscated    Run SPECspeed for obfuscated build(s)
  all           Run SPECspeed for all available obfuscation configs

Arguments:
  RUN_TYPE       : baseline | obfuscated | all (default: baseline)
  CONFIG_NAME    : Obfuscation config name (required for 'obfuscated' type)
                   Examples: layer1-2, full-obf
  ITERATIONS     : Number of benchmark iterations (default: 3)

Environment Variables:
  SPEC_CPU_HOME  : Path to SPEC CPU 2017 installation
                   (default: /opt/spec2017)

Examples:
  # Run baseline SPECspeed
  $(basename "$0") baseline

  # Run obfuscated SPECspeed with specific config
  $(basename "$0") obfuscated layer1-2

  # Run SPECspeed for all obfuscation configs
  $(basename "$0") all

  # Run with custom iteration count
  $(basename "$0") baseline 5

Output Directories:
  Baseline:    spec_cpu/results/baseline/<timestamp>/speed/
  Obfuscated:  spec_cpu/results/obfuscated/<config>/<timestamp>/speed/

Exit Codes:
  0 = All benchmark runs completed successfully
  1 = Configuration or setup error
  2 = Baseline benchmark run failed
  3 = Obfuscated benchmark run failed
  4 = No valid benchmark runs found
  5 = Results directory creation failed

Log Output:
  $LOG_FILE

EOF
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_header "SPEC CPU 2017 SPECspeed Benchmark Runner"

    log_info "Run Type: $RUN_TYPE"
    [ -n "$CONFIG_NAME" ] && log_info "Configuration: $CONFIG_NAME"
    log_info "Iterations: $ITERATIONS"
    log_info "SPEC Home: $SPEC_CPU_HOME"
    log_info "Results Base: $RESULTS_DIR"
    log_info ""

    # Validate SPEC installation
    if ! validate_spec_installation; then
        log_error "SPEC CPU validation failed"
        exit 1
    fi

    # Execute benchmarks based on run type
    case "$RUN_TYPE" in
        baseline)
            if ! run_baseline_speed; then
                log_error "Baseline SPECspeed execution failed"
                exit 2
            fi
            ;;

        obfuscated)
            if [ -z "$CONFIG_NAME" ]; then
                log_error "CONFIG_NAME required for obfuscated runs"
                print_help
                exit 1
            fi
            if ! run_obfuscated_speed "$CONFIG_NAME"; then
                log_error "Obfuscated SPECspeed execution failed for $CONFIG_NAME"
                exit 3
            fi
            ((TOTAL_TESTS++))
            ;;

        all)
            log_info "Running SPECspeed for all available obfuscation configs..."
            local available_configs=$(get_available_obfuscation_configs)
            if [ $? -ne 0 ]; then
                log_error "No obfuscation configurations found"
                exit 4
            fi

            local failed_configs=()
            for config in $available_configs; do
                ((TOTAL_TESTS++))
                if ! run_obfuscated_speed "$config"; then
                    failed_configs+=("$config")
                fi
            done

            if [ ${#failed_configs[@]} -gt 0 ]; then
                log_warning "Some configurations failed: ${failed_configs[*]}"
            fi
            ;;

        *)
            log_error "Unknown run type: $RUN_TYPE"
            print_help
            exit 1
            ;;
    esac

    # Generate summary
    generate_summary

    # Cleanup
    cleanup_temp_files

    # Final summary
    log_header "Benchmark Execution Complete"
    log_success "SPECspeed benchmarks finished"
    log_info "Results saved to: $RESULTS_DIR"
    log_info "Log file: $LOG_FILE"
    log_info ""

    return 0
}

# =============================================================================
# Entry Point
# =============================================================================

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    print_help
    exit 0
fi

main "$@"
exit $?
