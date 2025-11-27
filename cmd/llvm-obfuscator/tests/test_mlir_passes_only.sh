#!/bin/bash
# Quick test for MLIR passes only (no Python CLI)

set -e

echo "Testing MLIR Passes Only..."
cd /app/test_results

# Generate MLIR
clang -emit-llvm -S -emit-mlir /app/tests/test_simple.c -o test.mlir

# Test symbol pass
echo "Testing symbol obfuscation pass..."
mlir-opt test.mlir \
    --load-pass-plugin=/app/mlir-obs/build/lib/libMLIRObfuscation.so \
    --pass-pipeline="builtin.module(symbol-obfuscation)" \
    -o test_symbol.mlir

# Test string pass
echo "Testing string obfuscation pass..."
mlir-opt test.mlir \
    --load-pass-plugin=/app/mlir-obs/build/lib/libMLIRObfuscation.so \
    --pass-pipeline="builtin.module(string-obfuscation)" \
    -o test_string.mlir

# Test combined
echo "Testing combined passes..."
mlir-opt test.mlir \
    --load-pass-plugin=/app/mlir-obs/build/lib/libMLIRObfuscation.so \
    --pass-pipeline="builtin.module(symbol-obfuscation,string-obfuscation)" \
    -o test_combined.mlir

echo "✅ All MLIR passes executed successfully"
echo "Check output files: test_symbol.mlir, test_string.mlir, test_combined.mlir"