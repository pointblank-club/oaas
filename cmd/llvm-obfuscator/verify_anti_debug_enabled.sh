#!/bin/bash
# Quick script to verify if a binary was compiled with anti-debugging enabled

if [ $# -eq 0 ]; then
    echo "Usage: $0 <binary_path>"
    echo ""
    echo "This script checks if anti-debugging code was injected into the binary."
    exit 1
fi

BINARY="$1"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    exit 1
fi

echo "Checking binary: $BINARY"
echo "=========================================="
echo ""

# Check 1: Look for ptrace strings
echo "[1] Checking for ptrace references..."
if strings "$BINARY" 2>/dev/null | grep -qi "ptrace\|PTRACE"; then
    echo "  ✓ Found 'ptrace' string (likely has anti-debug code)"
    FOUND_PTRACE=1
else
    echo "  ✗ No 'ptrace' string found"
    FOUND_PTRACE=0
fi

# Check 2: Look for /proc/self/status
echo "[2] Checking for /proc/self/status references..."
if strings "$BINARY" 2>/dev/null | grep -qi "/proc/self/status\|TracerPid"; then
    echo "  ✓ Found '/proc/self/status' or 'TracerPid' string"
    FOUND_PROC=1
else
    echo "  ✗ No '/proc/self/status' string found"
    FOUND_PROC=0
fi

# Check 3: Look for ptrace syscall in disassembly
echo "[3] Checking disassembly for ptrace syscall..."
if command -v objdump >/dev/null 2>&1; then
    if objdump -d "$BINARY" 2>/dev/null | grep -qi "ptrace\|syscall.*0x65"; then
        echo "  ✓ Found ptrace syscall in disassembly"
        FOUND_SYSCALL=1
    else
        echo "  ✗ No ptrace syscall found in disassembly"
        FOUND_SYSCALL=0
    fi
else
    echo "  ⚠ objdump not available"
    FOUND_SYSCALL=0
fi

# Check 4: Look for _exit calls (anti-debug exits with _exit(1))
echo "[4] Checking for _exit calls..."
if strings "$BINARY" 2>/dev/null | grep -qi "_exit"; then
    echo "  ✓ Found '_exit' string (anti-debug uses _exit(1))"
    FOUND_EXIT=1
else
    echo "  ⚠ No '_exit' string found (may be inlined)"
    FOUND_EXIT=0
fi

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="

SCORE=$((FOUND_PTRACE + FOUND_PROC + FOUND_SYSCALL))

if [ $SCORE -ge 2 ]; then
    echo "✓ LIKELY: Binary appears to have anti-debugging code"
    echo ""
    echo "To verify it works:"
    echo "  gdb $BINARY"
    echo "  (gdb) run"
    echo "  # Should exit immediately with code 1"
else
    echo "✗ UNLIKELY: Binary does NOT appear to have anti-debugging code"
    echo ""
    echo "This binary was probably compiled WITHOUT anti-debugging enabled."
    echo ""
    echo "To enable anti-debugging:"
    echo "  1. Go to the frontend"
    echo "  2. Check the 'Anti-Debugging Protection' checkbox"
    echo "  3. Re-compile your code"
    echo "  4. Download the new binary"
fi

echo ""

