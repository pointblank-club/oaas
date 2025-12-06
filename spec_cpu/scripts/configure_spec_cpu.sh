#!/usr/bin/env bash

################################################################################
# SPEC CPU 2017 Configuration Script for LLVM Obfuscator
################################################################################
#
# Purpose: Initialize and validate SPEC CPU environment for benchmarking
#
# Functionality:
#   1. Locate and validate SPEC CPU 2017 installation
#   2. Verify license and benchmark availability
#   3. Detect custom Clang toolchain in plugins/
#   4. Setup compiler configuration with fallback to GCC
#   5. Copy and customize SPEC config file
#   6. Validate toolchain for both baseline and obfuscated builds
#
# Exit Codes:
#   0 = Configuration successful
#   1 = SPEC CPU installation not found or invalid
#   2 = Required tools/licenses missing
#   3 = No valid compiler chain available
#   4 = Plugin directory structure invalid
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

# SPEC CPU installation location (user can override via SPEC_CPU_HOME env var)
SPEC_CPU_HOME="${SPEC_CPU_HOME:-/opt/spec2017}"

# Toolchain detection
PLUGINS_DIR="$PROJECT_ROOT/cmd/llvm-obfuscator/plugins"
CUSTOM_CLANG="$PLUGINS_DIR/clang"
CUSTOM_CLANGXX="$PLUGINS_DIR/clang++"

# System compilers
SYSTEM_GCC=$(command -v gcc 2>/dev/null || echo "")
SYSTEM_GXX=$(command -v g++ 2>/dev/null || echo "")
SYSTEM_GFORTRAN=$(command -v gfortran 2>/dev/null || echo "")

# Build type (will be set by calling script or defaults to baseline)
BUILD_TYPE="${1:-baseline}"
OBFUSCATION_CONFIG="${2:-}"

# Logging
LOG_DIR="$RESULTS_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/configure_spec_cpu.log"

# =============================================================================
# Color Output
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# =============================================================================
# Validation Functions
# =============================================================================

validate_spec_installation() {
    log_info "Validating SPEC CPU 2017 installation at: $SPEC_CPU_HOME"

    # Check if directory exists
    if [ ! -d "$SPEC_CPU_HOME" ]; then
        log_error "SPEC CPU home directory not found: $SPEC_CPU_HOME"
        log_info "Set SPEC_CPU_HOME environment variable to override default location"
        return 1
    fi

    # Check for required SPEC CPU tools
    local required_tools=("bin/runcpu" "bin/specinvoke" "config")
    for tool in "${required_tools[@]}"; do
        if [ ! -e "$SPEC_CPU_HOME/$tool" ]; then
            log_error "Required SPEC CPU component missing: $tool"
            return 1
        fi
    done

    # Check for license file
    if [ ! -f "$SPEC_CPU_HOME/SPEC-license.txt" ]; then
        log_warning "SPEC CPU license file not found (may be required for execution)"
    fi

    # Check for benchmark sources
    if [ ! -d "$SPEC_CPU_HOME/benchspec/CPU" ]; then
        log_error "SPEC CPU benchmarks not found: $SPEC_CPU_HOME/benchspec/CPU"
        return 1
    fi

    # Count available benchmarks
    local benchmark_count=$(find "$SPEC_CPU_HOME/benchspec/CPU" -maxdepth 1 -type d -name "[0-9]*.*" | wc -l)
    if [ "$benchmark_count" -lt 40 ]; then
        log_warning "Expected ~54 benchmarks, found only $benchmark_count (possible incomplete installation)"
    fi

    log_success "SPEC CPU 2017 installation validated at: $SPEC_CPU_HOME"
    log_info "  - Found $benchmark_count benchmark(s)"
    return 0
}

detect_custom_clang() {
    log_info "Detecting custom Clang toolchain..."

    if [ -x "$CUSTOM_CLANG" ] && [ -x "$CUSTOM_CLANGXX" ]; then
        log_success "Custom Clang toolchain found:"
        log_info "  - clang:   $CUSTOM_CLANG"
        log_info "  - clang++: $CUSTOM_CLANGXX"

        # Verify it's actually executable
        if "$CUSTOM_CLANG" --version > /dev/null 2>&1; then
            CLANG_VERSION=$("$CUSTOM_CLANG" --version | head -1)
            log_info "  - Version: $CLANG_VERSION"
            return 0
        else
            log_error "Custom Clang found but not executable or not working"
            return 1
        fi
    else
        log_warning "Custom Clang toolchain NOT found at: $PLUGINS_DIR"
        log_info "  - Expected: $CUSTOM_CLANG"
        log_info "  - Expected: $CUSTOM_CLANGXX"
        return 1
    fi
}

detect_system_compiler() {
    log_info "Detecting system C/C++ compiler..."

    if [ -z "$SYSTEM_GCC" ] || [ -z "$SYSTEM_GXX" ]; then
        log_error "System GCC/G++ not found"
        return 1
    fi

    log_success "System compiler found:"
    log_info "  - gcc:  $SYSTEM_GCC"
    log_info "  - g++:  $SYSTEM_GXX"

    if [ -n "$SYSTEM_GFORTRAN" ]; then
        log_info "  - gfortran: $SYSTEM_GFORTRAN"
    else
        log_warning "gfortran not found (some benchmarks may not compile)"
    fi

    # Get compiler version
    GCC_VERSION=$("$SYSTEM_GCC" --version | head -1)
    log_info "  - GCC Version: $GCC_VERSION"

    return 0
}

validate_toolchain_for_build() {
    local build_type=$1

    log_info "Validating toolchain for $build_type build..."

    case "$build_type" in
        baseline)
            log_info "Baseline build toolchain selection:"

            # Try custom Clang first
            if detect_custom_clang > /dev/null 2>&1; then
                log_success "Using custom Clang for baseline build (preferred)"
                BASELINE_CC="$CUSTOM_CLANG"
                BASELINE_CXX="$CUSTOM_CLANGXX"
                BASELINE_COMPILER="clang"
                return 0
            fi

            # Fallback to system GCC
            log_info "Custom Clang not available, falling back to system GCC"
            if detect_system_compiler > /dev/null 2>&1; then
                log_success "Using system GCC for baseline build (with -O3 enforced)"
                BASELINE_CC="$SYSTEM_GCC"
                BASELINE_CXX="$SYSTEM_GXX"
                BASELINE_COMPILER="gcc"
                # GCC is fallback, so -O3 will be enforced in flags
                return 0
            fi

            log_error "No valid baseline compiler found"
            return 1
            ;;

        obfuscated)
            log_info "Obfuscated build requires custom Clang (no fallback allowed)"

            if detect_custom_clang > /dev/null 2>&1; then
                log_success "Custom Clang available for obfuscated build"
                OBFUSCC="$CUSTOM_CLANG"
                OBFUSCXX="$CUSTOM_CLANGXX"
                OBFUSCATED_COMPILER="clang"
                return 0
            else
                log_error "CRITICAL: Custom Clang from plugins/ is REQUIRED for obfuscated builds"
                log_error "No fallback compiler available for obfuscated builds"
                log_error "Please ensure plugins/clang and plugins/clang++ exist and are executable"
                return 1
            fi
            ;;

        *)
            log_error "Unknown build type: $build_type"
            return 1
            ;;
    esac
}

copy_and_customize_config() {
    local build_type=$1

    log_info "Setting up SPEC CPU config file..."

    # Source config file
    if [ ! -f "$CONFIGS_DIR/linux-x86_64.cfg" ]; then
        log_error "Template config not found: $CONFIGS_DIR/linux-x86_64.cfg"
        return 1
    fi

    # Destination in SPEC CPU directory
    local spec_config_dir="$SPEC_CPU_HOME/config"
    if [ ! -d "$spec_config_dir" ]; then
        log_error "SPEC CPU config directory not found: $spec_config_dir"
        return 1
    fi

    local dest_config="$spec_config_dir/llvm-obfuscation.cfg"

    log_info "Copying config template to: $dest_config"
    cp "$CONFIGS_DIR/linux-x86_64.cfg" "$dest_config"

    # Perform variable substitution based on detected compilers
    log_info "Customizing config for $build_type build..."

    case "$build_type" in
        baseline)
            # Substitute baseline compilers
            sed -i "s|\${BASELINE_CC}|$BASELINE_CC|g" "$dest_config"
            sed -i "s|\${BASELINE_CXX}|$BASELINE_CXX|g" "$dest_config"

            # For GCC fallback, ensure -O3 is forced
            if [ "$BASELINE_COMPILER" = "gcc" ]; then
                log_info "Enforcing -O3 optimization for GCC baseline build"
                sed -i 's/CFLAGS = -O3/CFLAGS = -O3 -fno-strict-aliasing/g' "$dest_config"
                sed -i 's/CXXFLAGS = -O3/CXXFLAGS = -O3 -fno-strict-aliasing/g' "$dest_config"
            fi

            # Set baseline-specific variables
            sed -i "s|\${BASELINE_FLAGS}|-O3 -march=native -fno-strict-aliasing|g" "$dest_config"
            log_success "Baseline configuration applied"
            ;;

        obfuscated)
            # Substitute obfuscated compilers
            sed -i "s|\${OBFUSCC}|$OBFUSCC|g" "$dest_config"
            sed -i "s|\${OBFUSCXX}|$OBFUSCXX|g" "$dest_config"
            sed -i "s|\${LLVM_PLUGIN_PATH}|$PLUGINS_DIR|g" "$dest_config"
            sed -i "s|\${OBFUS_CONFIG}|$OBFUSCATION_CONFIG|g" "$dest_config"

            # Set obfuscation-specific variables
            # Note: ${OBFUS_FLAGS} will be set by build_spec_targets.sh with actual obfuscation flags
            sed -i "s|\${OBFUS_FLAGS}|-O3 -march=native|g" "$dest_config"
            log_success "Obfuscated configuration applied"
            ;;
    esac

    # Set common variables for all builds
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    sed -i "s|\${TIMESTAMP}|$timestamp|g" "$dest_config"
    sed -i "s|\${NUM_THREADS}|$(nproc)|g" "$dest_config"

    log_success "Config file customized and saved: $dest_config"
    return 0
}

setup_environment_variables() {
    local build_type=$1

    log_info "Setting up environment variables..."

    # Export SPEC CPU home
    export SPEC="$SPEC_CPU_HOME"
    log_info "  SPEC=$SPEC"

    # Export compiler variables
    case "$build_type" in
        baseline)
            export CC="$BASELINE_CC"
            export CXX="$BASELINE_CXX"
            log_info "  CC=$CC"
            log_info "  CXX=$CXX"
            ;;
        obfuscated)
            export CC="$OBFUSCC"
            export CXX="$OBFUSCXX"
            export LD_LIBRARY_PATH="$PLUGINS_DIR/../lib:${LD_LIBRARY_PATH:-}"
            log_info "  CC=$CC"
            log_info "  CXX=$CXX"
            log_info "  LD_LIBRARY_PATH includes: $PLUGINS_DIR/../lib"
            ;;
    esac

    # Add SPEC bin to PATH if not already present
    if [[ ":$PATH:" != *":$SPEC_CPU_HOME/bin:"* ]]; then
        export PATH="$SPEC_CPU_HOME/bin:$PATH"
        log_info "  Added SPEC bin to PATH"
    fi

    log_success "Environment variables configured"
    return 0
}

create_result_directories() {
    log_info "Creating result directories..."

    local dirs=(
        "$RESULTS_DIR/baseline"
        "$RESULTS_DIR/obfuscated"
        "$RESULTS_DIR/comparisons"
    )

    for dir in "${dirs[@]}"; do
        if mkdir -p "$dir"; then
            log_info "  Created: $dir"
        else
            log_error "Failed to create directory: $dir"
            return 1
        fi
    done

    log_success "Result directories ready"
    return 0
}

print_help() {
    cat << EOF
${BLUE}SPEC CPU 2017 Configuration Script${NC}

Usage: $(basename "$0") [BUILD_TYPE] [OBFUSCATION_CONFIG]

Build Types:
  baseline      Configure for baseline (unobfuscated) benchmarks
  obfuscated    Configure for obfuscated benchmarks (requires plugins/clang)

Arguments:
  BUILD_TYPE           : baseline | obfuscated (default: baseline)
  OBFUSCATION_CONFIG   : Configuration name for obfuscated builds
                         Example: layer1-2, full-obf
                         (required for obfuscated builds)

Environment Variables:
  SPEC_CPU_HOME        : Path to SPEC CPU 2017 installation
                         (default: /opt/spec2017)

Examples:
  # Configure for baseline
  $(basename "$0") baseline

  # Configure for obfuscated with specific config
  $(basename "$0") obfuscated layer1-2

Exit Codes:
  0 = Configuration successful
  1 = SPEC CPU installation not found or invalid
  2 = Required tools/licenses missing
  3 = No valid compiler chain available
  4 = Plugin directory structure invalid

Log Output:
  $LOG_FILE

EOF
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_info "=========================================="
    log_info "SPEC CPU 2017 Configuration"
    log_info "=========================================="
    log_info "Build Type: $BUILD_TYPE"
    [ -n "$OBFUSCATION_CONFIG" ] && log_info "Obfuscation Config: $OBFUSCATION_CONFIG"
    log_info "SPEC Home: $SPEC_CPU_HOME"
    log_info "Plugins Dir: $PLUGINS_DIR"
    log_info ""

    # Validate SPEC CPU installation
    if ! validate_spec_installation; then
        log_error "SPEC CPU validation failed"
        exit 1
    fi

    # Create result directories
    if ! create_result_directories; then
        log_error "Failed to create result directories"
        exit 1
    fi

    # Validate toolchain for requested build type
    if ! validate_toolchain_for_build "$BUILD_TYPE"; then
        log_error "Toolchain validation failed for $BUILD_TYPE build"
        exit 3
    fi

    # Check obfuscation config parameter if needed
    if [ "$BUILD_TYPE" = "obfuscated" ] && [ -z "$OBFUSCATION_CONFIG" ]; then
        log_error "OBFUSCATION_CONFIG is required for obfuscated builds"
        print_help
        exit 1
    fi

    # Setup environment variables
    if ! setup_environment_variables "$BUILD_TYPE"; then
        log_error "Environment setup failed"
        exit 1
    fi

    # Copy and customize config
    if ! copy_and_customize_config "$BUILD_TYPE"; then
        log_error "Config customization failed"
        exit 1
    fi

    # Final summary
    log_info "=========================================="
    log_success "Configuration completed successfully!"
    log_info "=========================================="
    log_info ""
    log_info "Next steps:"
    if [ "$BUILD_TYPE" = "baseline" ]; then
        log_info "  1. Run: scripts/build_spec_targets.sh baseline"
        log_info "  2. Run: scripts/run_spec_speed.sh baseline"
        log_info "  3. Run: scripts/run_spec_rate.sh baseline"
    else
        log_info "  1. Run: scripts/build_spec_targets.sh obfuscated $OBFUSCATION_CONFIG"
        log_info "  2. Run: scripts/run_spec_speed.sh obfuscated $OBFUSCATION_CONFIG"
        log_info "  3. Run: scripts/run_spec_rate.sh obfuscated $OBFUSCATION_CONFIG"
        log_info "  4. Run: scripts/compare_spec_results.py results/baseline/<timestamp> results/obfuscated/$OBFUSCATION_CONFIG/<timestamp>"
    fi
    log_info ""
    log_info "Config file: $SPEC_CPU_HOME/config/llvm-obfuscation.cfg"
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
