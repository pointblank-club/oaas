#!/bin/bash
# Build a simple Windows PE binary for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if mingw is installed
if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
    echo "Installing MinGW-w64 cross-compiler..."
    sudo apt-get update
    sudo apt-get install -y mingw-w64
fi

echo "Building simple_add.exe..."

# Compile the test binary
x86_64-w64-mingw32-gcc \
    -O2 \
    -static \
    "$SCRIPT_DIR/simple_add.c" \
    -o "$SCRIPT_DIR/simple_add.exe"

echo "✓ Build complete!"
echo ""
echo "Binary created: $SCRIPT_DIR/simple_add.exe"
ls -lh "$SCRIPT_DIR/simple_add.exe"

echo ""
echo "File type:"
file "$SCRIPT_DIR/simple_add.exe"

echo ""
echo "You can now upload this to the OAAS website:"
echo "  https://oaas.pointblank.club"
