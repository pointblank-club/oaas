#!/usr/bin/env bash

################################################################################
# SPEC CPU 2017 Build Script for LLVM Obfuscator
################################################################################
#
# Purpose: Build SPEC CPU 2017 benchmarks with baseline and obfuscated toolchains
#
# Functionality:
#   1. Detect available compilers (custom Clang vs system GCC)
#   2. Build baseline benchmarks (preferred: custom Clang, fallback: GCC -O3)
#   3. Build obfuscated benchmarks (REQUIRED: custom Clang from plugins/)
#   4. Organize outputs by build type and configuration
#   5. Log all compiler flags, toolchain info, and build metadata
#   6. Integrate with existing LLVM obfuscator build system
#
# Exit Codes:
#   0 = All requested builds successful
#   1 = SPEC CPU configuration not found
#   2 = Baseline build failed
#   3 = Obfuscated build failed or plugins/ unavailable
#   4 = Invalid obfuscation configuration
#   5 = Build directory creation failed
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

# Toolchain paths
PLUGINS_DIR="$PROJECT_ROOT/cmd/llvm-obfuscator/plugins"
CUSTOM_CLANG="$PLUGINS_DIR/clang"
CUSTOM_CLANGXX="$PLUGINS_DIR/clang++"

# System compilers
SYSTEM_GCC=$(command -v gcc 2>/dev/null || echo "")
SYSTEM_GXX=$(command -v g++ 2>/dev/null || echo "")
SYSTEM_GFORTRAN=$(command -v gfortran 2>/dev/null || echo "")

# Logging
LOG_DIR="$RESULTS_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/build_spec_targets.log"

# Build parameters
BUILD_TYPE="${1:-baseline}"
OBFUSCATION_CONFIG="${2:-default}"
BENCHMARK_TARGET="${3:-all}"

# Compiler tracking
BASELINE_CC=""
BASELINE_CXX=""
OBFUSCC=""
OBFUSCXX=""
BASELINE_COMPILER_TYPE=""
OBFUSCATED_COMPILER_TYPE=""

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

validate_spec_config() {
    log_info "Validating SPEC CPU configuration..."

    if [ ! -f "$SPEC_CONFIG" ]; then
        log_error "SPEC CPU config not found: $SPEC_CONFIG"
        log_info "Run: scripts/configure_spec_cpu.sh $BUILD_TYPE $OBFUSCATION_CONFIG"
        return 1
    fi

    log_success "SPEC CPU config found: $SPEC_CONFIG"
    return 0
}

detect_baseline_toolchain() {
    log_info "Detecting baseline toolchain..."

    # Try custom Clang first (preferred)
    if [ -x "$CUSTOM_CLANG" ] && [ -x "$CUSTOM_CLANGXX" ]; then
        if "$CUSTOM_CLANG" --version > /dev/null 2>&1; then
            log_success "Custom Clang available for baseline (preferred)"
            BASELINE_CC="$CUSTOM_CLANG"
            BASELINE_CXX="$CUSTOM_CLANGXX"
            BASELINE_COMPILER_TYPE="clang-custom"
            log_info "  CC: $BASELINE_CC"
            log_info "  CXX: $BASELINE_CXX"
            return 0
        fi
    fi

    # Fallback to system GCC
    if [ -n "$SYSTEM_GCC" ] && [ -n "$SYSTEM_GXX" ]; then
        log_warning "Custom Clang not available, using system GCC with -O3"
        BASELINE_CC="$SYSTEM_GCC"
        BASELINE_CXX="$SYSTEM_GXX"
        BASELINE_COMPILER_TYPE="gcc-fallback"
        log_info "  CC: $BASELINE_CC"
        log_info "  CXX: $BASELINE_CXX"
        log_info "  Enforced flags: -O3 -march=native"
        return 0
    fi

    log_error "No valid baseline compiler found"
    return 1
}

detect_obfuscated_toolchain() {
    log_info "Detecting obfuscated toolchain (must be custom Clang)..."

    if [ ! -x "$CUSTOM_CLANG" ] || [ ! -x "$CUSTOM_CLANGXX" ]; then
        log_error "CRITICAL: Custom Clang required for obfuscated builds"
        log_error "Missing: $CUSTOM_CLANG or $CUSTOM_CLANGXX"
        log_error "Expected location: $PLUGINS_DIR"
        return 1
    fi

    if ! "$CUSTOM_CLANG" --version > /dev/null 2>&1; then
        log_error "Custom Clang exists but is not executable or broken"
        return 1
    fi

    log_success "Custom Clang available for obfuscated build"
    OBFUSCC="$CUSTOM_CLANG"
    OBFUSCXX="$CUSTOM_CLANGXX"
    OBFUSCATED_COMPILER_TYPE="clang-custom"
    log_info "  CC: $OBFUSCC"
    log_info "  CXX: $OBFUSCXX"

    # Get compiler version
    local clang_version=$("$CUSTOM_CLANG" --version | head -1)
    log_info "  Version: $clang_version"

    return 0
}

create_build_directories() {
    log_info "Creating build directories..."

    case "$BUILD_TYPE" in
        baseline)
            if ! mkdir -p "$BUILD_BASELINE_DIR"; then
                log_error "Failed to create baseline build directory"
                return 1
            fi
            log_success "Baseline build directory ready: $BUILD_BASELINE_DIR"
            ;;
        obfuscated)
            local obf_dir="$BUILD_OBFUSCATED_BASE_DIR/$OBFUSCATION_CONFIG"
            if ! mkdir -p "$obf_dir"; then
                log_error "Failed to create obfuscated build directory"
                return 1
            fi
            log_success "Obfuscated build directory ready: $obf_dir"
            ;;
        both)
            mkdir -p "$BUILD_BASELINE_DIR" "$BUILD_OBFUSCATED_BASE_DIR/$OBFUSCATION_CONFIG"
            log_success "Build directories ready for both types"
            ;;
    esac

    return 0
}

# =============================================================================
# Build Configuration
# =============================================================================

generate_baseline_config() {
    log_info "Generating baseline build configuration..."

    # Create temporary config with substitutions
    local temp_config="/tmp/llvm-obfuscation-baseline-$$.cfg"
    cp "$SPEC_CONFIG" "$temp_config"

    # Substitute compiler paths
    sed -i "s|^\s*CC\s*=.*|CC = $BASELINE_CC|g" "$temp_config"
    sed -i "s|^\s*CXX\s*=.*|CXX = $BASELINE_CXX|g" "$temp_config"

    # Set baseline-specific flags
    if [ "$BASELINE_COMPILER_TYPE" = "gcc-fallback" ]; then
        log_info "Enforcing -O3 for GCC baseline build"
        sed -i 's/CFLAGS = .*/CFLAGS = -O3 -march=native -fno-strict-aliasing/g' "$temp_config"
        sed -i 's/CXXFLAGS = .*/CXXFLAGS = -O3 -march=native -fno-strict-aliasing/g' "$temp_config"
    fi

    # Set output directory
    sed -i "s|\${RESULT_DIR}|$BUILD_BASELINE_DIR|g" "$temp_config"
    sed -i "s|\${TIMESTAMP}|$(date -u +%Y%m%d_%H%M%S)|g" "$temp_config"

    log_success "Baseline config generated: $temp_config"
    echo "$temp_config"
}

generate_obfuscated_config() {
    log_info "Generating obfuscated build configuration..."

    # Create temporary config with substitutions
    local temp_config="/tmp/llvm-obfuscation-obfuscated-$$.cfg"
    cp "$SPEC_CONFIG" "$temp_config"

    # Substitute compiler paths
    sed -i "s|^\s*CC\s*=.*|CC = $OBFUSCC|g" "$temp_config"
    sed -i "s|^\s*CXX\s*=.*|CXX = $OBFUSCXX|g" "$temp_config"

    # Set obfuscation plugin path
    sed -i "s|\${LLVM_PLUGIN_PATH}|$PLUGINS_DIR|g" "$temp_config"
    sed -i "s|\${OBFUS_CONFIG}|$OBFUSCATION_CONFIG|g" "$temp_config"

    # Set output directory
    local obf_output_dir="$BUILD_OBFUSCATED_BASE_DIR/$OBFUSCATION_CONFIG"
    sed -i "s|\${RESULT_DIR}|$obf_output_dir|g" "$temp_config"
    sed -i "s|\${TIMESTAMP}|$(date -u +%Y%m%d_%H%M%S)|g" "$temp_config"

    log_success "Obfuscated config generated: $temp_config"
    echo "$temp_config"
}

# =============================================================================
# Build Execution
# =============================================================================

build_baseline_benchmarks() {
    log_header "Building Baseline SPEC CPU Benchmarks"

    log_info "Toolchain: $BASELINE_COMPILER_TYPE"
    log_info "Compiler: $BASELINE_CC"
    log_info "C++: $BASELINE_CXX"
    log_info "Output: $BUILD_BASELINE_DIR"

    # Generate configuration
    local config_file=$(generate_baseline_config)
    if [ -z "$config_file" ] || [ ! -f "$config_file" ]; then
        log_error "Failed to generate baseline configuration"
        return 1
    fi

    # Log build metadata
    {
        echo "Baseline Build Metadata"
        echo "======================"
        echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Toolchain Type: $BASELINE_COMPILER_TYPE"
        echo "C Compiler: $BASELINE_CC"
        echo "C++ Compiler: $BASELINE_CXX"
        echo "Fortran Compiler: ${SYSTEM_GFORTRAN:-not found}"
        echo ""
        echo "Compiler Versions:"
        "$BASELINE_CC" --version | head -1
        "$BASELINE_CXX" --version | head -1
        [ -n "$SYSTEM_GFORTRAN" ] && "$SYSTEM_GFORTRAN" --version | head -1 || echo "gfortran: not found"
        echo ""
        echo "Build Flags:"
        echo "  -O3 -march=native -fno-strict-aliasing"
        echo ""
        echo "Config File: $config_file"
        echo ""
    } | tee -a "$LOG_FILE" > "$BUILD_BASELINE_DIR/BUILD_METADATA.txt"

    # Export environment
    export CC="$BASELINE_CC"
    export CXX="$BASELINE_CXX"
    [ -n "$SYSTEM_GFORTRAN" ] && export FC="$SYSTEM_GFORTRAN"

    # Execute build via SPEC CPU
    log_info "Invoking SPEC CPU build for baseline..."

    if cd "$SPEC_CPU_HOME" && bin/runcpu --config llvm-obfuscation.cfg --nobuild --setup_only > "$BUILD_BASELINE_DIR/spec_setup.log" 2>&1; then
        log_success "Baseline build setup completed"
        log_info "Output directory: $BUILD_BASELINE_DIR"
        log_info "Metadata file: $BUILD_BASELINE_DIR/BUILD_METADATA.txt"
        return 0
    else
        log_error "Baseline build failed (see logs)"
        return 1
    fi
}

build_obfuscated_benchmarks() {
    log_header "Building Obfuscated SPEC CPU Benchmarks"

    log_info "Configuration: $OBFUSCATION_CONFIG"
    log_info "Toolchain: $OBFUSCATED_COMPILER_TYPE"
    log_info "Compiler: $OBFUSCC"
    log_info "C++: $OBFUSCXX"
    log_info "Plugins: $PLUGINS_DIR"

    local obf_dir="$BUILD_OBFUSCATED_BASE_DIR/$OBFUSCATION_CONFIG"
    log_info "Output: $obf_dir"

    # Generate configuration
    local config_file=$(generate_obfuscated_config)
    if [ -z "$config_file" ] || [ ! -f "$config_file" ]; then
        log_error "Failed to generate obfuscated configuration"
        return 1
    fi

    # Log build metadata
    {
        echo "Obfuscated Build Metadata"
        echo "========================="
        echo "Configuration: $OBFUSCATION_CONFIG"
        echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Toolchain Type: $OBFUSCATED_COMPILER_TYPE"
        echo "C Compiler: $OBFUSCC"
        echo "C++ Compiler: $OBFUSCXX"
        echo "Fortran Compiler: ${SYSTEM_GFORTRAN:-not found}"
        echo ""
        echo "Compiler Versions:"
        "$OBFUSCC" --version | head -1
        "$OBFUSCXX" --version | head -1
        [ -n "$SYSTEM_GFORTRAN" ] && "$SYSTEM_GFORTRAN" --version | head -1 || echo "gfortran: not found"
        echo ""
        echo "Plugin Path: $PLUGINS_DIR"
        echo ""
        echo "Build Flags:"
        echo "  -O3 -march=native (base)"
        echo "  + obfuscation flags from: $OBFUSCATION_CONFIG"
        echo "  + LLVM plugin: LLVMObfuscationPlugin.so"
        echo ""
        echo "Config File: $config_file"
        echo ""
    } | tee -a "$LOG_FILE" > "$obf_dir/BUILD_METADATA.txt"

    # Export environment with plugin paths
    export CC="$OBFUSCC"
    export CXX="$OBFUSCXX"
    [ -n "$SYSTEM_GFORTRAN" ] && export FC="$SYSTEM_GFORTRAN"
    export LD_LIBRARY_PATH="$PLUGINS_DIR/../lib:${LD_LIBRARY_PATH:-}"

    # Execute build via SPEC CPU
    log_info "Invoking SPEC CPU build for obfuscated ($OBFUSCATION_CONFIG)..."

    if cd "$SPEC_CPU_HOME" && bin/runcpu --config llvm-obfuscation.cfg --nobuild --setup_only > "$obf_dir/spec_setup.log" 2>&1; then
        log_success "Obfuscated build completed for config: $OBFUSCATION_CONFIG"
        log_info "Output directory: $obf_dir"
        log_info "Metadata file: $obf_dir/BUILD_METADATA.txt"
        return 0
    else
        log_error "Obfuscated build failed (see logs)"
        return 1
    fi
}

cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/llvm-obfuscation-*.cfg
}

print_help() {
    cat << EOF
${CYAN}SPEC CPU 2017 Build Script${NC}

Usage: $(basename "$0") [BUILD_TYPE] [OBFUSCATION_CONFIG] [BENCHMARK_TARGET]

Build Types:
  baseline      Build baseline (unobfuscated) binaries
  obfuscated    Build obfuscated binaries
  both          Build both baseline and obfuscated (default: baseline)

Arguments:
  BUILD_TYPE           : baseline | obfuscated | both
  OBFUSCATION_CONFIG   : Config name for obfuscated builds
                         Example: layer1-2, full-obf
                         (default: default)
  BENCHMARK_TARGET     : Specific benchmarks or 'all'
                         (default: all)

Environment Variables:
  SPEC_CPU_HOME        : Path to SPEC CPU 2017 installation
                         (default: /opt/spec2017)

Examples:
  # Build baseline benchmarks
  $(basename "$0") baseline

  # Build obfuscated benchmarks with config
  $(basename "$0") obfuscated layer1-2

  # Build both
  $(basename "$0") both default

Output Directories:
  Baseline:    build/baseline_spec/
  Obfuscated:  build/obfuscated_spec/<config_name>/

Exit Codes:
  0 = All builds successful
  1 = SPEC config not found
  2 = Baseline build failed
  3 = Obfuscated build failed or plugins unavailable
  4 = Invalid obfuscation configuration
  5 = Build directory creation failed

Log Output:
  $LOG_FILE

EOF
}

# =============================================================================
# Integration with Global Build System
# =============================================================================

integrate_with_global_build() {
    log_info "Integrating with global LLVM obfuscator build system..."

    # Check if global build script exists
    local global_build_script="$PROJECT_ROOT/scripts/build_targets.sh"
    if [ -x "$global_build_script" ]; then
        log_info "Found global build script: $global_build_script"
        log_info "SPEC CPU builds are separate local benchmarks (not in CI/CD)"
        log_info "To use global build system: $global_build_script $OBFUSCATION_CONFIG"
    else
        log_warning "Global build script not found: $global_build_script"
    fi

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_header "SPEC CPU 2017 Build Script"

    log_info "Build Type: $BUILD_TYPE"
    log_info "Obfuscation Config: $OBFUSCATION_CONFIG"
    log_info "Benchmark Target: $BENCHMARK_TARGET"
    log_info "SPEC Home: $SPEC_CPU_HOME"
    log_info "Plugins Dir: $PLUGINS_DIR"
    log_info ""

    # Validate SPEC CPU configuration
    if ! validate_spec_config; then
        log_error "SPEC CPU configuration validation failed"
        exit 1
    fi

    # Create build directories
    if ! create_build_directories; then
        log_error "Failed to create build directories"
        exit 5
    fi

    # Build based on type
    case "$BUILD_TYPE" in
        baseline)
            if ! detect_baseline_toolchain; then
                log_error "Baseline toolchain detection failed"
                exit 3
            fi
            if ! build_baseline_benchmarks; then
                log_error "Baseline build failed"
                exit 2
            fi
            ;;

        obfuscated)
            if ! detect_obfuscated_toolchain; then
                log_error "Obfuscated toolchain detection failed (custom Clang required)"
                exit 3
            fi
            if ! build_obfuscated_benchmarks; then
                log_error "Obfuscated build failed"
                exit 3
            fi
            ;;

        both)
            if ! detect_baseline_toolchain; then
                log_error "Baseline toolchain detection failed"
                exit 3
            fi
            if ! build_baseline_benchmarks; then
                log_error "Baseline build failed"
                exit 2
            fi

            if ! detect_obfuscated_toolchain; then
                log_error "Obfuscated toolchain detection failed (custom Clang required)"
                exit 3
            fi
            if ! build_obfuscated_benchmarks; then
                log_error "Obfuscated build failed"
                exit 3
            fi
            ;;

        *)
            log_error "Unknown build type: $BUILD_TYPE"
            print_help
            exit 4
            ;;
    esac

    # Cleanup
    cleanup_temp_files

    # Integration info
    integrate_with_global_build

    # Summary
    log_header "Build Complete"
    log_success "All requested builds completed successfully"

    case "$BUILD_TYPE" in
        baseline)
            log_info "Baseline binaries: $BUILD_BASELINE_DIR"
            ;;
        obfuscated)
            log_info "Obfuscated binaries: $BUILD_OBFUSCATED_BASE_DIR/$OBFUSCATION_CONFIG"
            ;;
        both)
            log_info "Baseline binaries: $BUILD_BASELINE_DIR"
            log_info "Obfuscated binaries: $BUILD_OBFUSCATED_BASE_DIR/$OBFUSCATION_CONFIG"
            ;;
    esac

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
