#!/bin/bash
# PTS Test Suite with Compile-Time Metrics Integration

set -e

SOURCE_FILE="$1"
TEST_NAME="${2:-compile-metrics-test}"
OUTPUT_DIR="${3:-results/compile-metrics}"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Running PTS benchmarks with compile-time metrics collection..."

# Get baseline metrics (unstripped)
BASELINE_BINARY="${OUTPUT_DIR}/baseline_unstripped"
gcc -O3 -g "$SOURCE_FILE" -o "$BASELINE_BINARY"

# Collect baseline metrics
echo "[INFO] Collecting baseline metrics..."
python3 phoronix/scripts/collect_obfuscation_metrics.py \
    "$BASELINE_BINARY" "$BASELINE_BINARY" \
    --config "$TEST_NAME" \
    --output "$OUTPUT_DIR/metrics/"

# Now strip for testing (if needed)
STRIPPED_BINARY="${OUTPUT_DIR}/baseline_stripped"
cp "$BASELINE_BINARY" "$STRIPPED_BINARY"
strip "$STRIPPED_BINARY"

echo "[SUCCESS] Compile-time metrics collected and saved"
echo "  Unstripped: $BASELINE_BINARY"
echo "  Metrics: $OUTPUT_DIR/metrics/"
