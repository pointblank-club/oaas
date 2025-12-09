#!/bin/bash
# Quick script to upload UPX binary to GCP

set -e

UPX_SOURCE="/home/dhruv/Documents/upx/build/upx"
PLUGINS_DIR="cmd/llvm-obfuscator/plugins/linux-x86_64"

echo "=========================================="
echo "Uploading Custom UPX Binary to GCP"
echo "=========================================="
echo ""

# Check if source exists
if [ ! -f "$UPX_SOURCE" ]; then
    echo "❌ UPX binary not found at: $UPX_SOURCE"
    exit 1
fi

# Create plugins directory if needed
mkdir -p "$PLUGINS_DIR"

# Copy to plugins directory
echo "📋 Copying UPX binary..."
cp "$UPX_SOURCE" "$PLUGINS_DIR/upx"
chmod +x "$PLUGINS_DIR/upx"

echo "✅ UPX binary copied to: $PLUGINS_DIR/upx"
ls -lh "$PLUGINS_DIR/upx"
echo ""

# Upload to GCP
echo "☁️  Uploading to GCP..."
./scripts/gcp-binary-manager.sh add \
    "$PLUGINS_DIR/upx" \
    linux-x86_64/upx

echo ""
echo "✅ Done! UPX binary is now in GCP and will be included in Docker builds."
