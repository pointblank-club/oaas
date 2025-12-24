#!/bin/bash
# =============================================================================
# GCP Binary Manager - Single Source of Truth for LLVM Binaries
# =============================================================================
#
# This script manages a SINGLE TARBALL in GCP that contains ALL binaries.
# It supports adding, updating, listing, and downloading binaries.
#
# USAGE:
#   ./scripts/gcp-binary-manager.sh <command> [options]
#
# COMMANDS:
#   add <local_path> [target_path]  - Add/update a binary in the tarball
#   remove <target_path>            - Remove a binary from the tarball
#   list                            - List all binaries in the tarball
#   download                        - Download and extract the tarball
#   sync                            - Download tarball for local editing
#   push                            - Push local changes back to GCP
#
# EXAMPLES:
#   # Add a new clang binary (will be placed at linux-x86_64/clang)
#   ./scripts/gcp-binary-manager.sh add /path/to/clang linux-x86_64/clang
#
#   # Add a new plugin (preserves existing structure)
#   ./scripts/gcp-binary-manager.sh add ./MyPlugin.so linux-x86_64/MyPlugin.so
#
#   # Add clang headers directory
#   ./scripts/gcp-binary-manager.sh add ./include linux-x86_64/lib/clang/22/include
#
#   # List all binaries in the tarball
#   ./scripts/gcp-binary-manager.sh list
#
#   # Download for CI/CD
#   ./scripts/gcp-binary-manager.sh download
#
# =============================================================================

set -e

# Configuration
GCP_BUCKET="llvmbins"
TARBALL_NAME="llvm-obfuscator-binaries.tar.gz"
TARBALL_PATH="gs://${GCP_BUCKET}/${TARBALL_NAME}"
LOCAL_WORK_DIR="/tmp/gcp-binary-manager"
LOCAL_TARBALL="/tmp/${TARBALL_NAME}"
PLUGINS_DIR="cmd/llvm-obfuscator/plugins"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_gsutil() {
    if command -v gsutil &> /dev/null; then
        GSUTIL="gsutil"
    elif [ -f "$HOME/google-cloud-sdk/bin/gsutil" ]; then
        GSUTIL="$HOME/google-cloud-sdk/bin/gsutil"
        export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    else
        log_error "gsutil not found. Install Google Cloud SDK:"
        echo "  https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

check_auth() {
    GCLOUD="gcloud"
    if ! command -v gcloud &> /dev/null && [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
        GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
    fi

    if ! $GCLOUD auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
        log_error "Not authenticated with GCP"
        echo ""
        echo "Authenticate using:"
        echo "  1. gcloud auth login"
        echo "  2. Or: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
        exit 1
    fi

    ACTIVE_ACCOUNT=$($GCLOUD auth list --filter=status:ACTIVE --format="value(account)")
    log_info "Authenticated as: ${ACTIVE_ACCOUNT}"
}

download_tarball() {
    log_info "Downloading tarball from GCP..."

    # Check if tarball exists in GCP
    if ! $GSUTIL ls "$TARBALL_PATH" &>/dev/null; then
        log_warn "Tarball not found in GCP. Creating new one."
        mkdir -p "$LOCAL_WORK_DIR"
        return 1
    fi

    $GSUTIL cp "$TARBALL_PATH" "$LOCAL_TARBALL"

    # Extract to work directory
    rm -rf "$LOCAL_WORK_DIR"
    mkdir -p "$LOCAL_WORK_DIR"
    tar -xzf "$LOCAL_TARBALL" -C "$LOCAL_WORK_DIR"

    log_success "Downloaded and extracted tarball"
    return 0
}

upload_tarball() {
    log_info "Creating tarball..."

    # Create tarball from work directory
    cd "$LOCAL_WORK_DIR"
    tar -czf "$LOCAL_TARBALL" .
    cd - > /dev/null

    log_info "Uploading tarball to GCP..."
    $GSUTIL cp "$LOCAL_TARBALL" "$TARBALL_PATH"

    # Get file size
    SIZE=$(ls -lh "$LOCAL_TARBALL" | awk '{print $5}')
    log_success "Uploaded tarball ($SIZE) to $TARBALL_PATH"
}

show_structure() {
    echo ""
    echo "Current tarball structure:"
    echo "=========================="
    if [ -d "$LOCAL_WORK_DIR" ]; then
        find "$LOCAL_WORK_DIR" -type f | sed "s|$LOCAL_WORK_DIR/||" | sort | head -50
        TOTAL=$(find "$LOCAL_WORK_DIR" -type f | wc -l)
        if [ "$TOTAL" -gt 50 ]; then
            echo "... and $((TOTAL - 50)) more files"
        fi
    fi
    echo ""
}

# =============================================================================
# Commands
# =============================================================================

cmd_add() {
    local SOURCE_PATH="$1"
    local TARGET_PATH="$2"

    if [ -z "$SOURCE_PATH" ]; then
        log_error "Usage: $0 add <local_path> [target_path]"
        echo ""
        echo "Examples:"
        echo "  $0 add ./clang linux-x86_64/clang"
        echo "  $0 add ./LLVMObfuscationPlugin.so linux-x86_64/LLVMObfuscationPlugin.so"
        echo "  $0 add ./include linux-x86_64/lib/clang/22/include"
        exit 1
    fi

    # Validate source exists
    if [ ! -e "$SOURCE_PATH" ]; then
        log_error "Source path not found: $SOURCE_PATH"
        exit 1
    fi

    # If target not specified, use source filename under linux-x86_64/
    if [ -z "$TARGET_PATH" ]; then
        TARGET_PATH="linux-x86_64/$(basename "$SOURCE_PATH")"
        log_warn "No target path specified. Using: $TARGET_PATH"
    fi

    check_gsutil
    check_auth

    echo ""
    echo "============================================"
    echo "  ADD BINARY TO TARBALL"
    echo "============================================"
    echo ""
    echo "  Source: $SOURCE_PATH"
    echo "  Target: $TARGET_PATH"
    echo ""

    # Download existing tarball
    download_tarball || true

    # Create target directory
    TARGET_DIR="$LOCAL_WORK_DIR/$(dirname "$TARGET_PATH")"
    mkdir -p "$TARGET_DIR"

    # Copy source to target
    if [ -d "$SOURCE_PATH" ]; then
        # It's a directory, copy recursively
        log_info "Copying directory: $SOURCE_PATH -> $TARGET_PATH"
        cp -r "$SOURCE_PATH" "$LOCAL_WORK_DIR/$TARGET_PATH"
    else
        # It's a file
        log_info "Copying file: $SOURCE_PATH -> $TARGET_PATH"
        cp "$SOURCE_PATH" "$LOCAL_WORK_DIR/$TARGET_PATH"

        # Make executable if it's a binary
        if file "$SOURCE_PATH" | grep -qE "ELF|shared object"; then
            chmod +x "$LOCAL_WORK_DIR/$TARGET_PATH"
        fi
    fi

    # Upload updated tarball
    upload_tarball

    show_structure

    log_success "Binary added successfully!"
    echo ""
    echo "The CI will now use this updated tarball."
}

cmd_remove() {
    local TARGET_PATH="$1"

    if [ -z "$TARGET_PATH" ]; then
        log_error "Usage: $0 remove <target_path>"
        echo ""
        echo "Example: $0 remove linux-x86_64/old-binary"
        exit 1
    fi

    check_gsutil
    check_auth

    echo ""
    echo "============================================"
    echo "  REMOVE BINARY FROM TARBALL"
    echo "============================================"
    echo ""
    echo "  Target: $TARGET_PATH"
    echo ""

    # Download existing tarball
    if ! download_tarball; then
        log_error "No tarball found in GCP. Nothing to remove."
        exit 1
    fi

    # Check if target exists
    if [ ! -e "$LOCAL_WORK_DIR/$TARGET_PATH" ]; then
        log_error "Target not found in tarball: $TARGET_PATH"
        show_structure
        exit 1
    fi

    # Remove target
    rm -rf "$LOCAL_WORK_DIR/$TARGET_PATH"
    log_info "Removed: $TARGET_PATH"

    # Upload updated tarball
    upload_tarball

    show_structure

    log_success "Binary removed successfully!"
}

cmd_list() {
    check_gsutil
    check_auth

    echo ""
    echo "============================================"
    echo "  LIST TARBALL CONTENTS"
    echo "============================================"
    echo ""

    # Download tarball
    if ! download_tarball; then
        log_error "No tarball found in GCP."
        exit 1
    fi

    show_structure

    # Show sizes of main binaries
    echo "Binary sizes:"
    echo "============="
    for f in clang opt mlir-opt clangir LLVMObfuscationPlugin.so MLIRObfuscation.so libLLVM.so.22.0git; do
        if [ -f "$LOCAL_WORK_DIR/linux-x86_64/$f" ]; then
            SIZE=$(ls -lh "$LOCAL_WORK_DIR/linux-x86_64/$f" | awk '{print $5}')
            printf "  %-30s %s\n" "$f" "$SIZE"
        fi
    done
    echo ""
}

cmd_download() {
    check_gsutil
    check_auth

    echo ""
    echo "============================================"
    echo "  DOWNLOAD TARBALL FOR CI/CD"
    echo "============================================"
    echo ""

    # Download tarball
    if ! download_tarball; then
        log_error "No tarball found in GCP. Run 'init' first."
        exit 1
    fi

    # Copy to plugins directory
    mkdir -p "$PLUGINS_DIR"
    cp -r "$LOCAL_WORK_DIR"/* "$PLUGINS_DIR/"

    # Make binaries executable
    chmod +x "$PLUGINS_DIR/linux-x86_64/clang" 2>/dev/null || true
    chmod +x "$PLUGINS_DIR/linux-x86_64/opt" 2>/dev/null || true
    chmod +x "$PLUGINS_DIR/linux-x86_64/mlir-opt" 2>/dev/null || true
    chmod +x "$PLUGINS_DIR/linux-x86_64/mlir-translate" 2>/dev/null || true
    chmod +x "$PLUGINS_DIR/linux-x86_64/clangir" 2>/dev/null || true
    chmod +x "$PLUGINS_DIR/linux-x86_64/lld" 2>/dev/null || true

    log_success "Downloaded to $PLUGINS_DIR/"
    echo ""
    ls -lh "$PLUGINS_DIR/linux-x86_64/" | head -15
}

cmd_init() {
    check_gsutil
    check_auth

    echo ""
    echo "============================================"
    echo "  INITIALIZE TARBALL FROM EXISTING GCP FILES"
    echo "============================================"
    echo ""

    log_info "Creating tarball from existing GCP files..."

    # Create work directory
    rm -rf "$LOCAL_WORK_DIR"
    mkdir -p "$LOCAL_WORK_DIR/linux-x86_64"

    # Download all existing individual files
    log_info "Downloading linux-x86_64 binaries..."
    $GSUTIL -m cp -r "gs://${GCP_BUCKET}/linux-x86_64/*" "$LOCAL_WORK_DIR/linux-x86_64/" || true

    # Download macOS SDK if exists
    if $GSUTIL ls "gs://${GCP_BUCKET}/macos-sdk-15.4-minimal.tar.gz" &>/dev/null; then
        log_info "Downloading macOS SDK..."
        mkdir -p "$LOCAL_WORK_DIR/macos-sdk"
        $GSUTIL cp "gs://${GCP_BUCKET}/macos-sdk-15.4-minimal.tar.gz" /tmp/macos-sdk.tar.gz
        tar -xzf /tmp/macos-sdk.tar.gz -C /tmp/
        mv /tmp/macos-sdk/MacOSX15.4.sdk "$LOCAL_WORK_DIR/macos-sdk/" 2>/dev/null || mv /tmp/MacOSX15.4.sdk "$LOCAL_WORK_DIR/macos-sdk/" 2>/dev/null || true
        rm -rf /tmp/macos-sdk /tmp/macos-sdk.tar.gz /tmp/MacOSX15.4.sdk 2>/dev/null || true
    fi

    # Download llvm-mingw if exists
    if $GSUTIL ls "gs://${GCP_BUCKET}/llvm-mingw-20251202-ucrt-ubuntu-22.04-x86_64.tar.xz" &>/dev/null; then
        log_info "Downloading llvm-mingw..."
        mkdir -p "$LOCAL_WORK_DIR/llvm-mingw"
        $GSUTIL cp "gs://${GCP_BUCKET}/llvm-mingw-20251202-ucrt-ubuntu-22.04-x86_64.tar.xz" /tmp/llvm-mingw.tar.xz
        tar -xf /tmp/llvm-mingw.tar.xz -C /tmp/
        mv /tmp/llvm-mingw-20251202-ucrt-ubuntu-22.04-x86_64/* "$LOCAL_WORK_DIR/llvm-mingw/" 2>/dev/null || true
        rm -rf /tmp/llvm-mingw-20251202-ucrt-ubuntu-22.04-x86_64 /tmp/llvm-mingw.tar.xz 2>/dev/null || true
    fi

    # Create and upload tarball
    upload_tarball

    show_structure

    log_success "Tarball initialized from existing GCP files!"
    echo ""
    echo "You can now use 'add', 'remove', 'list' commands."
}

cmd_help() {
    echo ""
    echo "============================================"
    echo "  GCP Binary Manager"
    echo "============================================"
    echo ""
    echo "USAGE:"
    echo "  $0 <command> [options]"
    echo ""
    echo "COMMANDS:"
    echo "  init                          - Create tarball from existing GCP files (run once)"
    echo "  add <source> [target]         - Add/update a binary in the tarball"
    echo "  remove <target>               - Remove a binary from the tarball"
    echo "  list                          - List all binaries in the tarball"
    echo "  download                      - Download tarball to plugins directory"
    echo "  help                          - Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo ""
    echo "  # First time setup (converts individual files to tarball):"
    echo "  $0 init"
    echo ""
    echo "  # Add a new/updated clang binary:"
    echo "  $0 add /path/to/clang linux-x86_64/clang"
    echo ""
    echo "  # Add a new plugin:"
    echo "  $0 add ./NewPlugin.so linux-x86_64/NewPlugin.so"
    echo ""
    echo "  # Add clang headers directory:"
    echo "  $0 add ./include linux-x86_64/lib/clang/22/include"
    echo ""
    echo "  # List tarball contents:"
    echo "  $0 list"
    echo ""
    echo "  # Download for local development:"
    echo "  $0 download"
    echo ""
    echo "PATH STRUCTURE:"
    echo "  linux-x86_64/              - Linux x86_64 binaries"
    echo "    clang                    - Clang compiler"
    echo "    opt                      - LLVM optimizer"
    echo "    mlir-opt                 - MLIR optimizer"
    echo "    clangir                  - ClangIR compiler"
    echo "    LLVMObfuscationPlugin.so - Obfuscation plugin"
    echo "    MLIRObfuscation.so       - MLIR obfuscation plugin"
    echo "    lib/clang/22/include/    - Clang headers"
    echo "  macos-sdk/                 - macOS SDK for cross-compilation"
    echo "  llvm-mingw/                - MinGW for Windows cross-compilation"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    add)
        cmd_add "$@"
        ;;
    remove)
        cmd_remove "$@"
        ;;
    list)
        cmd_list "$@"
        ;;
    download)
        cmd_download "$@"
        ;;
    init)
        cmd_init "$@"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        cmd_help
        exit 1
        ;;
esac
