#!/bin/bash
# Download LLVM binaries from GCP Cloud Storage
# This script downloads binaries to the local plugins directory
# Usage: ./download-binaries-from-gcp.sh [version]
#   version: optional, defaults to "latest"

set -e

# Configuration
GCP_BUCKET="llvmbins"
VERSION="${1:-latest}"
PLUGINS_DIR="cmd/llvm-obfuscator/plugins"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "📥 Download LLVM Binaries from GCP"
echo "=========================================="
echo ""
echo "Version: ${VERSION}"
echo "Bucket: gs://${GCP_BUCKET}/"
echo ""

# Check if gcloud/gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}❌ ERROR: gsutil is not installed${NC}"
    echo ""
    echo "Install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Or if already installed, add to PATH:"
    echo "  export PATH=\"\$HOME/google-cloud-sdk/bin:\$PATH\""
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated with GCP${NC}"
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

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
echo -e "${GREEN}✓ Authenticated as: ${ACTIVE_ACCOUNT}${NC}"
echo ""

# Determine archive name
if [ "$VERSION" = "latest" ]; then
    ARCHIVE_NAME="llvm-binaries-linux-x86_64-latest.tar.gz"
else
    ARCHIVE_NAME="llvm-binaries-linux-x86_64-${VERSION}.tar.gz"
fi

# Download archive
TEMP_ARCHIVE="/tmp/llvm-binaries-${VERSION}.tar.gz"
echo "Downloading: ${ARCHIVE_NAME}"
gsutil cp "gs://${GCP_BUCKET}/${ARCHIVE_NAME}" "$TEMP_ARCHIVE" || {
    echo -e "${RED}❌ ERROR: Failed to download binaries from GCP${NC}"
    echo ""
    echo "Make sure the binaries exist in the bucket:"
    echo "  gsutil ls gs://${GCP_BUCKET}/"
    echo ""
    echo "To upload binaries to GCP:"
    echo "  ./scripts/upload-binaries-to-gcp.sh [version]"
    echo ""
    exit 1
}

echo -e "${GREEN}✓ Downloaded successfully${NC}"
echo ""

# Create plugins directory
mkdir -p "$PLUGINS_DIR"

# Extract archive
echo "Extracting binaries to ${PLUGINS_DIR}..."
cd "$PLUGINS_DIR"
tar -xzf "$TEMP_ARCHIVE"
rm "$TEMP_ARCHIVE"

echo -e "${GREEN}✓ Binaries extracted${NC}"
echo ""

# Verify binaries
echo "Verifying binaries..."
VERIFICATION_FAILED=0

if [ -f "linux-x86_64/clang" ] && file "linux-x86_64/clang" | grep -q "ELF"; then
    echo -e "${GREEN}  ✓ clang binary verified${NC}"
else
    echo -e "${RED}  ✗ clang binary invalid or missing${NC}"
    VERIFICATION_FAILED=1
fi

if [ -f "linux-x86_64/opt" ] && file "linux-x86_64/opt" | grep -q "ELF"; then
    echo -e "${GREEN}  ✓ opt binary verified${NC}"
else
    echo -e "${RED}  ✗ opt binary invalid or missing${NC}"
    VERIFICATION_FAILED=1
fi

if [ -f "linux-x86_64/LLVMObfuscationPlugin.so" ] && file "linux-x86_64/LLVMObfuscationPlugin.so" | grep -q "shared object"; then
    echo -e "${GREEN}  ✓ plugin binary verified${NC}"
else
    echo -e "${RED}  ✗ plugin binary invalid or missing${NC}"
    VERIFICATION_FAILED=1
fi

if [ $VERIFICATION_FAILED -eq 1 ]; then
    echo ""
    echo -e "${RED}❌ ERROR: Binary verification failed!${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Download Completed Successfully!${NC}"
echo "=========================================="
echo ""
echo "Binaries location: ${PLUGINS_DIR}/linux-x86_64/"
echo "Version: ${VERSION}"
echo ""
echo "Binary sizes:"
ls -lh linux-x86_64/clang linux-x86_64/opt linux-x86_64/LLVMObfuscationPlugin.so
echo ""
