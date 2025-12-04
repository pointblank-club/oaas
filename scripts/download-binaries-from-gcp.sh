#!/bin/bash
# Download LLVM binaries from GCP Cloud Storage
# This script downloads individual binaries from GCP for local development/testing
#
# Usage: ./scripts/download-binaries-from-gcp.sh

set -e

# Configuration
GCP_BUCKET="llvmbins"
PLUGINS_DIR="cmd/llvm-obfuscator/plugins"
REMOTE_PATH="linux-x86_64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Download LLVM Binaries from GCP"
echo "=========================================="
echo ""
echo "Bucket: gs://${GCP_BUCKET}/${REMOTE_PATH}/"
echo ""

# Check if gcloud/gsutil is installed
GSUTIL=""
if command -v gsutil &> /dev/null; then
    GSUTIL="gsutil"
elif [ -f "$HOME/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL="$HOME/google-cloud-sdk/bin/gsutil"
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
else
    echo -e "${RED}ERROR: gsutil is not installed${NC}"
    echo ""
    echo "Install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Or if already installed, add to PATH:"
    echo "  export PATH=\"\$HOME/google-cloud-sdk/bin:\$PATH\""
    exit 1
fi

# Check if authenticated
GCLOUD="gcloud"
if ! command -v gcloud &> /dev/null && [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
fi

if ! $GCLOUD auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}Not authenticated with GCP${NC}"
    echo ""
    echo "Please authenticate using one of these methods:"
    echo ""
    echo "1. Service account (recommended for CI/CD):"
    echo "   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
    echo "   gcloud auth activate-service-account --key-file=\$GOOGLE_APPLICATION_CREDENTIALS"
    echo ""
    echo "2. User account:"
    echo "   gcloud auth login"
    echo ""
    exit 1
fi

ACTIVE_ACCOUNT=$($GCLOUD auth list --filter=status:ACTIVE --format="value(account)")
echo -e "${GREEN}Authenticated as: ${ACTIVE_ACCOUNT}${NC}"
echo ""

# Create plugins directory
mkdir -p "$PLUGINS_DIR/linux-x86_64"

# Download all individual files from GCP
echo "Downloading individual binaries from GCP..."
$GSUTIL -m cp -r "gs://${GCP_BUCKET}/${REMOTE_PATH}/*" "$PLUGINS_DIR/linux-x86_64/" || {
    echo -e "${RED}ERROR: Failed to download binaries from GCP${NC}"
    echo ""
    echo "Make sure the binaries exist in the bucket:"
    echo "  gsutil ls gs://${GCP_BUCKET}/${REMOTE_PATH}/"
    echo ""
    echo "To upload binaries to GCP:"
    echo "  ./scripts/upload-binaries-to-gcp.sh"
    echo ""
    exit 1
}

echo -e "${GREEN}Downloaded successfully${NC}"
echo ""

# Make binaries executable
echo "Making binaries executable..."
chmod +x "$PLUGINS_DIR/linux-x86_64/clang" 2>/dev/null || true
chmod +x "$PLUGINS_DIR/linux-x86_64/opt" 2>/dev/null || true
chmod +x "$PLUGINS_DIR/linux-x86_64/mlir-opt" 2>/dev/null || true
chmod +x "$PLUGINS_DIR/linux-x86_64/mlir-translate" 2>/dev/null || true
chmod +x "$PLUGINS_DIR/linux-x86_64/clangir" 2>/dev/null || true

# Verify binaries
echo ""
echo "Verifying binaries..."
VERIFICATION_FAILED=0

if [ -f "$PLUGINS_DIR/linux-x86_64/clang" ] && file "$PLUGINS_DIR/linux-x86_64/clang" | grep -q "ELF"; then
    echo -e "${GREEN}  clang binary verified${NC}"
else
    echo -e "${RED}  clang binary invalid or missing${NC}"
    VERIFICATION_FAILED=1
fi

if [ -f "$PLUGINS_DIR/linux-x86_64/opt" ] && file "$PLUGINS_DIR/linux-x86_64/opt" | grep -q "ELF"; then
    echo -e "${GREEN}  opt binary verified${NC}"
else
    echo -e "${RED}  opt binary invalid or missing${NC}"
    VERIFICATION_FAILED=1
fi

if [ -f "$PLUGINS_DIR/linux-x86_64/LLVMObfuscationPlugin.so" ] && file "$PLUGINS_DIR/linux-x86_64/LLVMObfuscationPlugin.so" | grep -q "shared object"; then
    echo -e "${GREEN}  plugin binary verified${NC}"
else
    echo -e "${RED}  plugin binary invalid or missing${NC}"
    VERIFICATION_FAILED=1
fi

if [ $VERIFICATION_FAILED -eq 1 ]; then
    echo ""
    echo -e "${RED}ERROR: Binary verification failed!${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Download Completed Successfully!${NC}"
echo "=========================================="
echo ""
echo "Binaries location: ${PLUGINS_DIR}/linux-x86_64/"
echo ""
echo "Downloaded files:"
ls -lh "$PLUGINS_DIR/linux-x86_64/" | grep -v "^d" | head -10
echo ""
