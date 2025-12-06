#!/usr/bin/env bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PTS_VERSION="10.8.4"
PTS_DOWNLOAD_URL="https://www.phoronix-test-suite.com/releases/phoronix-test-suite-${PTS_VERSION}.tar.gz"
PTS_INSTALL_DIR="/opt/phoronix-test-suite"
TEMP_DIR="${TEMP_DIR:-/tmp}"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

check_dependencies() {
    log_info "Checking system dependencies..."

    local missing_deps=()

    # Check for required commands
    command -v php > /dev/null || missing_deps+=("php")
    command -v tar > /dev/null || missing_deps+=("tar")
    command -v curl > /dev/null || missing_deps+=("curl")
    command -v wget > /dev/null || missing_deps+=("wget")

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: sudo apt-get install -y php php-xml php-cli curl wget tar build-essential"
        return 1
    fi

    log_success "All dependencies present"
    return 0
}

download_phoronix() {
    log_info "Downloading Phoronix Test Suite v${PTS_VERSION}..."

    local pts_tarball="${TEMP_DIR}/phoronix-test-suite-${PTS_VERSION}.tar.gz"

    if [ -f "$pts_tarball" ]; then
        log_warning "Tarball already exists at $pts_tarball, skipping download"
        echo "$pts_tarball"
        return 0
    fi

    if ! curl -fsSL -o "$pts_tarball" "$PTS_DOWNLOAD_URL"; then
        log_error "Failed to download Phoronix Test Suite from $PTS_DOWNLOAD_URL"
        return 1
    fi

    # Verify tarball integrity
    if ! tar -tzf "$pts_tarball" > /dev/null 2>&1; then
        log_error "Downloaded tarball is corrupted"
        rm -f "$pts_tarball"
        return 1
    fi

    log_success "Downloaded Phoronix Test Suite tarball"
    echo "$pts_tarball"
}

extract_and_install() {
    local pts_tarball=$1

    log_info "Extracting Phoronix Test Suite..."

    # Create installation directory
    if [ ! -d "$PTS_INSTALL_DIR" ]; then
        if ! mkdir -p "$PTS_INSTALL_DIR"; then
            log_error "Failed to create installation directory $PTS_INSTALL_DIR"
            log_info "Trying with sudo..."
            sudo mkdir -p "$PTS_INSTALL_DIR" || {
                log_error "Failed to create installation directory even with sudo"
                return 1
            }
        fi
    fi

    # Extract to temp location first
    local extract_dir="${TEMP_DIR}/phoronix-extract-$$"
    mkdir -p "$extract_dir"

    if ! tar -xzf "$pts_tarball" -C "$extract_dir"; then
        log_error "Failed to extract tarball"
        rm -rf "$extract_dir"
        return 1
    fi

    # Find the extracted directory (usually phoronix-test-suite-X.X.X)
    local source_dir=$(find "$extract_dir" -maxdepth 1 -type d -name "phoronix*" | head -1)

    if [ -z "$source_dir" ]; then
        log_error "Could not find extracted phoronix directory"
        rm -rf "$extract_dir"
        return 1
    fi

    # Copy to installation directory
    log_info "Installing to $PTS_INSTALL_DIR..."
    if ! sudo cp -r "$source_dir"/* "$PTS_INSTALL_DIR/"; then
        log_error "Failed to copy files to installation directory"
        rm -rf "$extract_dir"
        return 1
    fi

    # Ensure proper permissions
    sudo chmod +x "$PTS_INSTALL_DIR/phoronix-test-suite" || true

    # Create symlink for easy access
    if ! sudo ln -sf "$PTS_INSTALL_DIR/phoronix-test-suite" /usr/local/bin/phoronix-test-suite 2>/dev/null; then
        log_warning "Could not create symlink in /usr/local/bin (may need sudo)"
    fi

    # Cleanup
    rm -rf "$extract_dir"

    log_success "Phoronix Test Suite installed to $PTS_INSTALL_DIR"
    return 0
}

configure_pts() {
    log_info "Configuring Phoronix Test Suite..."

    # Set up PTS configuration directory
    local pts_config_dir="$HOME/.phoronix-test-suite"
    mkdir -p "$pts_config_dir"

    # Create configuration to auto-accept license and run non-interactively
    cat > "$pts_config_dir/user-config.xml" << 'EOF'
<?xml version="1.0"?>
<PhoronixTestSuite>
	<Options>
		<General>
			<UseColoredOutput>true</UseColoredOutput>
			<CheckForNewTestProfiles>false</CheckForNewTestProfiles>
			<CheckForUpdates>false</CheckForUpdates>
			<ShowPostRunStatistics>true</ShowPostRunStatistics>
		</General>
		<Networking>
			<UploadResults>false</UploadResults>
			<ProxyAddress></ProxyAddress>
		</Networking>
		<Testing>
			<AlwaysSaveResults>true</AlwaysSaveResults>
			<RunAllTestCombinations>true</RunAllTestCombinations>
			<UploadSystemLogs>false</UploadSystemLogs>
			<PromptForPhoneHome>false</PromptForPhoneHome>
			<PromptForTestIdentifier>false</PromptForTestIdentifier>
			<SaveIntermediate>true</SaveIntermediate>
			<RunTestAsRoot>false</RunTestAsRoot>
		</Testing>
		<Server>
			<PrintComments>false</PrintComments>
			<PrintTestFailureOutput>true</PrintTestFailureOutput>
		</Server>
	</Options>
</PhoronixTestSuite>
EOF

    log_success "Phoronix Test Suite configured"
    return 0
}

verify_installation() {
    log_info "Verifying Phoronix Test Suite installation..."

    local pts_cmd="$PTS_INSTALL_DIR/phoronix-test-suite"

    if [ ! -f "$pts_cmd" ]; then
        log_error "Phoronix Test Suite executable not found at $pts_cmd"
        return 1
    fi

    if [ ! -x "$pts_cmd" ]; then
        log_warning "Making executable..."
        sudo chmod +x "$pts_cmd"
    fi

    # Try to run version command
    log_info "Running version check..."
    if $pts_cmd version 2>/dev/null | grep -q "Phoronix Test Suite"; then
        log_success "Phoronix Test Suite installed and verified"
        return 0
    else
        log_error "Failed to verify Phoronix Test Suite"
        return 1
    fi
}

run_diagnostics() {
    log_info "Running Phoronix Test Suite diagnostics..."

    local pts_cmd="$PTS_INSTALL_DIR/phoronix-test-suite"

    if $pts_cmd diagnose 2>/dev/null; then
        log_success "Diagnostics completed successfully"
        return 0
    else
        log_warning "Diagnostics encountered some issues (non-fatal)"
        return 0
    fi
}

main() {
    log_info "=========================================="
    log_info "Phoronix Test Suite Installation Script"
    log_info "=========================================="

    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi

    # Download
    local pts_tarball
    if ! pts_tarball=$(download_phoronix); then
        log_error "Download failed"
        exit 1
    fi

    # Extract and install
    if ! extract_and_install "$pts_tarball"; then
        log_error "Installation failed"
        exit 1
    fi

    # Configure
    if ! configure_pts; then
        log_warning "Configuration had issues but continuing..."
    fi

    # Verify
    if ! verify_installation; then
        log_error "Verification failed"
        exit 1
    fi

    # Run diagnostics
    run_diagnostics

    log_info "=========================================="
    log_success "Installation completed successfully!"
    log_info "=========================================="
    log_info "Usage: phoronix-test-suite <command> [options]"
    log_info "Example: phoronix-test-suite list-tests"
    log_info "Example: phoronix-test-suite run pts/compress-7zip"

    return 0
}

main "$@"
