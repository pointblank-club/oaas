#!/bin/bash
# Download LLVM binaries from GCP Cloud Storage on the server
# This is useful if you want to update binaries without rebuilding Docker images

set -e

# Configuration
GCP_BUCKET="llvmbins"
VERSION="${1:-latest}"
BINARIES_DIR="/app/plugins/linux-x86_64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Download LLVM Binaries from GCP (Server)"
echo "=========================================="
echo ""

# Check if gcloud/gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo -e "${YELLOW}⚠️  gsutil not found. Installing Google Cloud SDK...${NC}"
    
    # Install gcloud CLI
    curl https://sdk.cloud.google.com | bash
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    
    if ! command -v gsutil &> /dev/null; then
        echo -e "${RED}❌ ERROR: Failed to install gsutil${NC}"
        exit 1
    fi
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated with GCP${NC}"
    echo ""
    echo "You need to authenticate. Options:"
    echo "  1. Use service account key:"
    echo "     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
    echo "     gcloud auth activate-service-account --key-file=\$GOOGLE_APPLICATION_CREDENTIALS"
    echo ""
    echo "  2. Use interactive login:"
    echo "     gcloud auth login"
    exit 1
fi

# Determine archive name
if [ "$VERSION" = "latest" ]; then
    ARCHIVE_NAME="llvm-binaries-linux-x86_64-latest.tar.gz"
else
    ARCHIVE_NAME="llvm-binaries-linux-x86_64-${VERSION}.tar.gz"
fi

echo "Downloading: ${ARCHIVE_NAME}"
echo "From: gs://${GCP_BUCKET}/${ARCHIVE_NAME}"
echo ""

# Create binaries directory
mkdir -p "$BINARIES_DIR"

# Download archive
TEMP_ARCHIVE="/tmp/llvm-binaries-${VERSION}.tar.gz"
gsutil cp "gs://${GCP_BUCKET}/${ARCHIVE_NAME}" "$TEMP_ARCHIVE" || {
    echo -e "${RED}❌ ERROR: Failed to download binaries${NC}"
    exit 1
}

# Extract archive
echo "Extracting binaries..."
cd "$(dirname "$BINARIES_DIR")"
tar -xzf "$TEMP_ARCHIVE"
rm "$TEMP_ARCHIVE"

# Verify binaries
echo ""
echo "Verifying binaries..."
if [ -f "$BINARIES_DIR/clang" ] && file "$BINARIES_DIR/clang" | grep -q "ELF"; then
    echo -e "${GREEN}✅ clang binary verified${NC}"
else
    echo -e "${RED}❌ clang binary invalid or missing${NC}"
    exit 1
fi

if [ -f "$BINARIES_DIR/opt" ] && file "$BINARIES_DIR/opt" | grep -q "ELF"; then
    echo -e "${GREEN}✅ opt binary verified${NC}"
else
    echo -e "${RED}❌ opt binary invalid or missing${NC}"
    exit 1
fi

if [ -f "$BINARIES_DIR/LLVMObfuscationPlugin.so" ] && file "$BINARIES_DIR/LLVMObfuscationPlugin.so" | grep -q "shared object"; then
    echo -e "${GREEN}✅ plugin binary verified${NC}"
else
    echo -e "${RED}❌ plugin binary invalid or missing${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Binaries downloaded successfully!${NC}"
echo "=========================================="
echo ""
echo "Location: $BINARIES_DIR"
echo "Version: $VERSION"
echo ""
echo "Note: If running in Docker, you may need to restart the container"
echo "      for the new binaries to take effect."

