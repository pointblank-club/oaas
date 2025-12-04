#!/bin/bash
# Upload LLVM binaries to GCP Cloud Storage
# This script uploads INDIVIDUAL binaries - only what you have locally gets uploaded
# Other binaries in GCP remain untouched (no overwriting!)
#
# Usage: ./scripts/upload-binaries-to-gcp.sh [binary_name]
#   - No args: uploads all binaries found locally
#   - With arg: uploads only the specified binary (e.g., ./upload-binaries-to-gcp.sh LLVMObfuscationPlugin.so)

set -e

# Configuration
GCP_BUCKET="llvmbins"
BINARIES_DIR="cmd/llvm-obfuscator/plugins/linux-x86_64"
REMOTE_PATH="linux-x86_64"

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
GSUTIL=""
if command -v gsutil &> /dev/null; then
    GSUTIL="gsutil"
elif [ -f "$HOME/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL="$HOME/google-cloud-sdk/bin/gsutil"
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
elif [ -f "/usr/bin/gsutil" ]; then
    GSUTIL="/usr/bin/gsutil"
else
    echo -e "${RED}ERROR: gsutil is not installed${NC}"
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

# Check if authenticated
GCLOUD="gcloud"
if ! command -v gcloud &> /dev/null && [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

if ! $GCLOUD auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}Not authenticated with GCP${NC}"
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

ACTIVE_ACCOUNT=$($GCLOUD auth list --filter=status:ACTIVE --format="value(account)")
echo -e "${GREEN}Authenticated as: ${ACTIVE_ACCOUNT}${NC}"
echo ""

# Check if binaries directory exists
if [ ! -d "$BINARIES_DIR" ]; then
    echo -e "${RED}ERROR: Binaries directory not found: $BINARIES_DIR${NC}"
    echo "Please run this script from the repository root"
    exit 1
fi

# List of known binaries
BINARIES=(
    "clang"
    "opt"
    "mlir-opt"
    "mlir-translate"
    "clangir"
    "LLVMObfuscationPlugin.so"
    "MLIRObfuscation.so"
    "libLLVM.so.22.0git"
)

# Function to upload a single binary
upload_binary() {
    local binary="$1"
    local local_path="${BINARIES_DIR}/${binary}"
    local remote_path="gs://${GCP_BUCKET}/${REMOTE_PATH}/${binary}"

    if [ -f "$local_path" ]; then
        # Verify it's a valid binary
        if file "$local_path" | grep -qE "ELF|shared object"; then
            local size=$(ls -lh "$local_path" | awk '{print $5}')
            echo -e "Uploading: ${GREEN}$binary${NC} ($size)"
            $GSUTIL cp "$local_path" "$remote_path"
            echo -e "  ${GREEN}-> $remote_path${NC}"
            return 0
        else
            echo -e "${YELLOW}SKIP: $binary (not a valid ELF binary)${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}SKIP: $binary (not found locally)${NC}"
        return 1
    fi
}

# Upload lib/clang/22/include directory if it exists
upload_clang_headers() {
    local local_headers="${BINARIES_DIR}/lib/clang/22/include"
    local remote_headers="gs://${GCP_BUCKET}/${REMOTE_PATH}/lib/clang/22/include/"

    if [ -d "$local_headers" ]; then
        echo ""
        echo "Uploading clang headers..."
        $GSUTIL -m cp -r "$local_headers/*" "$remote_headers"
        echo -e "${GREEN}  -> $remote_headers${NC}"
    fi
}

echo "Bucket: gs://${GCP_BUCKET}/${REMOTE_PATH}/"
echo "Local:  ${BINARIES_DIR}/"
echo ""
echo -e "${YELLOW}NOTE: Only binaries that exist locally will be uploaded.${NC}"
echo -e "${YELLOW}      Other binaries in GCP will NOT be overwritten.${NC}"
echo ""

UPLOADED=0

# If specific binary specified, upload only that
if [ -n "$1" ]; then
    echo "Uploading specific binary: $1"
    echo ""
    if upload_binary "$1"; then
        UPLOADED=1
    fi
else
    # Upload all binaries that exist locally
    echo "Checking for binaries to upload..."
    echo ""

    for binary in "${BINARIES[@]}"; do
        if upload_binary "$binary"; then
            UPLOADED=$((UPLOADED + 1))
        fi
    done

    # Also upload clang headers if they exist
    upload_clang_headers
fi

echo ""
echo "=========================================="
if [ $UPLOADED -gt 0 ]; then
    echo -e "${GREEN}Upload completed! ($UPLOADED binaries uploaded)${NC}"
else
    echo -e "${YELLOW}No binaries were uploaded${NC}"
fi
echo "=========================================="
echo ""
echo "Current contents of gs://${GCP_BUCKET}/${REMOTE_PATH}/:"
$GSUTIL ls -lh "gs://${GCP_BUCKET}/${REMOTE_PATH}/" 2>/dev/null || echo "(unable to list)"
