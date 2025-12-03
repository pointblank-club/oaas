#!/bin/bash
# Upload LLVM binaries directly to GCP Cloud Storage
# This bypasses Git LFS quota limits by uploading from your local machine

set -e

# Configuration
GCP_BUCKET="llvmbins"
VERSION="${1:-v1.0.0}"
BINARIES_DIR="cmd/llvm-obfuscator/plugins/linux-x86_64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Upload LLVM Binaries to GCP Cloud Storage"
echo "=========================================="
echo ""

# Check if gcloud/gsutil is installed
# Try to find gsutil in common locations
GSUTIL=""
if command -v gsutil &> /dev/null; then
    GSUTIL="gsutil"
elif [ -f "$HOME/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL="$HOME/google-cloud-sdk/bin/gsutil"
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
elif [ -f "/usr/bin/gsutil" ]; then
    GSUTIL="/usr/bin/gsutil"
else
    echo -e "${RED}❌ ERROR: gsutil is not installed${NC}"
    echo "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Or if already installed, add to PATH:"
    echo "  export PATH=\"\$HOME/google-cloud-sdk/bin:\$PATH\""
    exit 1
fi

# Also check for gcloud
if ! command -v gcloud &> /dev/null && [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Check if authenticated (either via service account or user account)
# Use gcloud from PATH or full path
GCLOUD="gcloud"
if ! command -v gcloud &> /dev/null && [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

if ! $GCLOUD auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated with GCP${NC}"
    echo ""
    
    # Try common locations for service account key
    SERVICE_ACCOUNT_KEY=""
    if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        SERVICE_ACCOUNT_KEY="$GOOGLE_APPLICATION_CREDENTIALS"
    elif [ -f "$HOME/Downloads/unified-coyote-478817-r3-79066ecf407e.json" ]; then
        SERVICE_ACCOUNT_KEY="$HOME/Downloads/unified-coyote-478817-r3-79066ecf407e.json"
        echo "Found service account key in Downloads folder"
    fi
    
    if [ -n "$SERVICE_ACCOUNT_KEY" ]; then
        echo "Authenticating with service account: $SERVICE_ACCOUNT_KEY"
        $GCLOUD auth activate-service-account --key-file="$SERVICE_ACCOUNT_KEY"
        $GCLOUD config set project unified-coyote-478817-r3
    else
        echo "Please authenticate:"
        echo "  1. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
        echo "  2. Or run: gcloud auth login"
        exit 1
    fi
fi

# Check if binaries directory exists
if [ ! -d "$BINARIES_DIR" ]; then
    echo -e "${RED}❌ ERROR: Binaries directory not found: $BINARIES_DIR${NC}"
    echo "Please run this script from the repository root"
    exit 1
fi

# Verify required binaries exist
echo "Checking for required binaries..."
MISSING=0

if [ ! -f "$BINARIES_DIR/clang" ]; then
    echo -e "${RED}❌ clang binary not found${NC}"
    MISSING=1
elif ! file "$BINARIES_DIR/clang" | grep -q "ELF"; then
    echo -e "${RED}❌ clang exists but is not a valid ELF binary${NC}"
    MISSING=1
else
    echo -e "${GREEN}✅ clang binary found${NC}"
fi

if [ ! -f "$BINARIES_DIR/opt" ]; then
    echo -e "${RED}❌ opt binary not found${NC}"
    MISSING=1
elif ! file "$BINARIES_DIR/opt" | grep -q "ELF"; then
    echo -e "${RED}❌ opt exists but is not a valid ELF binary${NC}"
    MISSING=1
else
    echo -e "${GREEN}✅ opt binary found${NC}"
fi

if [ ! -f "$BINARIES_DIR/LLVMObfuscationPlugin.so" ]; then
    echo -e "${RED}❌ LLVMObfuscationPlugin.so not found${NC}"
    MISSING=1
elif ! file "$BINARIES_DIR/LLVMObfuscationPlugin.so" | grep -q "shared object"; then
    echo -e "${RED}❌ plugin exists but is not a valid shared object${NC}"
    MISSING=1
else
    echo -e "${GREEN}✅ plugin binary found${NC}"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${RED}❌ ERROR: Required binaries are missing!${NC}"
    echo ""
    echo "Make sure you have the actual binary files (not LFS pointers) in:"
    echo "  $BINARIES_DIR"
    exit 1
fi

# Create temporary directory for archive
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo ""
echo "Creating archive..."
mkdir -p "$TEMP_DIR/linux-x86_64/lib"

# Copy binaries
cp "$BINARIES_DIR/clang" "$TEMP_DIR/linux-x86_64/"
cp "$BINARIES_DIR/opt" "$TEMP_DIR/linux-x86_64/"
cp "$BINARIES_DIR/LLVMObfuscationPlugin.so" "$TEMP_DIR/linux-x86_64/"

# Copy library files if they exist
if [ -d "$BINARIES_DIR/lib" ]; then
    cp -r "$BINARIES_DIR/lib"/* "$TEMP_DIR/linux-x86_64/lib/" 2>/dev/null || true
    echo "✅ Copied library files"
fi

# Create archive
cd "$TEMP_DIR"
ARCHIVE_NAME="llvm-binaries-linux-x86_64-${VERSION}.tar.gz"
tar -czf "$ARCHIVE_NAME" linux-x86_64/
ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)

echo -e "${GREEN}✅ Archive created: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})${NC}"
echo ""

# Upload to GCP
echo "Uploading to GCP Cloud Storage..."
echo "  Bucket: gs://${GCP_BUCKET}/"
echo "  Version: ${VERSION}"
echo ""

# Upload versioned file
$GSUTIL cp "$ARCHIVE_NAME" "gs://${GCP_BUCKET}/${ARCHIVE_NAME}"
echo -e "${GREEN}✅ Uploaded: gs://${GCP_BUCKET}/${ARCHIVE_NAME}${NC}"

# Upload as latest
$GSUTIL cp "$ARCHIVE_NAME" "gs://${GCP_BUCKET}/llvm-binaries-linux-x86_64-latest.tar.gz"
echo -e "${GREEN}✅ Uploaded: gs://${GCP_BUCKET}/llvm-binaries-linux-x86_64-latest.tar.gz${NC}"

# Set metadata
$GSUTIL -m setmeta -h "Content-Type:application/gzip" \
  -h "Cache-Control:public, max-age=3600" \
  "gs://${GCP_BUCKET}/${ARCHIVE_NAME}" \
  "gs://${GCP_BUCKET}/llvm-binaries-linux-x86_64-latest.tar.gz"

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Upload completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Binaries are now available at:"
echo "  gs://${GCP_BUCKET}/${ARCHIVE_NAME}"
echo "  gs://${GCP_BUCKET}/llvm-binaries-linux-x86_64-latest.tar.gz"
echo ""
echo "Your CI workflows can now download from GCP instead of Git LFS!"

