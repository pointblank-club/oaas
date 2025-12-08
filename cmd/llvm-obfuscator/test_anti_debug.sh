#!/bin/bash
# Test script to demonstrate anti-debugging protection on an existing binary
# Usage: ./test_anti_debug.sh <path_to_binary>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_obfuscated_binary>"
    echo ""
    echo "Example:"
    echo "  $0 ./obfuscated/my_program"
    echo "  $0 /path/to/downloaded/binary"
    exit 1
fi

BINARY="$1"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "Making binary executable..."
    chmod +x "$BINARY"
fi

echo "=========================================="
echo "Anti-Debugging Protection Test Suite"
echo "=========================================="
echo "Testing binary: $BINARY"
echo ""

# Test 1: Normal execution (should work)
echo "[Test 1/7] Normal Execution"
echo "---------------------------"
echo "Running binary normally (should work)..."
if timeout 5 "$BINARY" > /tmp/normal_run.log 2>&1; then
    echo "✓ PASS: Binary runs normally"
    echo "  Output:"
    head -5 /tmp/normal_run.log | sed 's/^/  /'
else
    EXIT_CODE=$?
    echo "✗ FAIL: Binary exited with code $EXIT_CODE"
    echo "  Output:"
    head -5 /tmp/normal_run.log | sed 's/^/  /'
fi
echo ""

# Test 2: GDB debugging (should be detected and exit)
echo "[Test 2/7] GDB Debugging Detection"
echo "-----------------------------------"
if command -v gdb >/dev/null 2>&1; then
    echo "Attempting to debug with GDB..."
    echo "run" | timeout 3 gdb -batch -ex "set confirm off" -ex "set pagination off" -ex "run" "$BINARY" > /tmp/gdb_test.log 2>&1
    if grep -q "exited with code 1" /tmp/gdb_test.log || grep -q "exited normally" /tmp/gdb_test.log; then
        if grep -q "exited with code 1" /tmp/gdb_test.log; then
            echo "✓ PASS: Binary detected GDB and exited (anti-debug working!)"
            echo "  Evidence: Binary exited with code 1 when run under GDB"
        else
            echo "⚠ WARNING: Binary ran under GDB without detection"
            echo "  This suggests anti-debugging may not be working"
        fi
    else
        echo "⚠ Could not determine exit status from GDB output"
    fi
    echo "  GDB output snippet:"
    head -10 /tmp/gdb_test.log | sed 's/^/  /'
else
    echo "⚠ SKIP: GDB not installed"
fi
echo ""

# Test 3: strace tracing (should be detected)
echo "[Test 3/7] strace Tracing Detection"
echo "-----------------------------------"
if command -v strace >/dev/null 2>&1; then
    echo "Attempting to trace with strace..."
    if timeout 3 strace -e trace=none "$BINARY" > /tmp/strace_test.log 2>&1; then
        echo "⚠ WARNING: Binary ran under strace without detection"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 1 ]; then
            echo "✓ PASS: Binary detected strace and exited (anti-debug working!)"
        else
            echo "⚠ Binary exited with code $EXIT_CODE under strace"
        fi
    fi
    echo "  strace output snippet:"
    head -5 /tmp/strace_test.log | sed 's/^/  /'
else
    echo "⚠ SKIP: strace not installed"
fi
echo ""

# Test 4: ltrace library tracing (should be detected)
echo "[Test 4/7] ltrace Library Tracing Detection"
echo "--------------------------------------------"
if command -v ltrace >/dev/null 2>&1; then
    echo "Attempting to trace with ltrace..."
    if timeout 3 ltrace "$BINARY" > /tmp/ltrace_test.log 2>&1; then
        echo "⚠ WARNING: Binary ran under ltrace without detection"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 1 ]; then
            echo "✓ PASS: Binary detected ltrace and exited (anti-debug working!)"
        else
            echo "⚠ Binary exited with code $EXIT_CODE under ltrace"
        fi
    fi
    echo "  ltrace output snippet:"
    head -5 /tmp/ltrace_test.log | sed 's/^/  /'
else
    echo "⚠ SKIP: ltrace not installed"
fi
echo ""

# Test 5: Check for anti-debug patterns in binary
echo "[Test 5/7] Binary Analysis - String Patterns"
echo "---------------------------------------------"
if command -v strings >/dev/null 2>&1; then
    echo "Searching for anti-debug patterns..."
    PATTERNS_FOUND=0
    if strings "$BINARY" | grep -qi "ptrace"; then
        echo "  ✓ Found 'ptrace' string"
        PATTERNS_FOUND=$((PATTERNS_FOUND + 1))
    fi
    if strings "$BINARY" | grep -qi "tracerpid\|TracerPid"; then
        echo "  ✓ Found 'TracerPid' string"
        PATTERNS_FOUND=$((PATTERNS_FOUND + 1))
    fi
    if strings "$BINARY" | grep -qi "/proc/self/status"; then
        echo "  ✓ Found '/proc/self/status' string"
        PATTERNS_FOUND=$((PATTERNS_FOUND + 1))
    fi
    if [ $PATTERNS_FOUND -eq 0 ]; then
        echo "  ⚠ No obvious anti-debug strings found (may be obfuscated)"
    else
        echo "  ✓ Found $PATTERNS_FOUND anti-debug pattern(s)"
    fi
else
    echo "⚠ SKIP: strings command not available"
fi
echo ""

# Test 6: Disassembly analysis
echo "[Test 6/7] Binary Analysis - Disassembly"
echo "-----------------------------------------"
if command -v objdump >/dev/null 2>&1; then
    echo "Searching for ptrace/syscall in disassembly..."
    if objdump -d "$BINARY" 2>/dev/null | grep -qi "ptrace\|syscall.*0x65"; then
        echo "  ✓ Found ptrace/syscall calls in disassembly"
        echo "  Sample:"
        objdump -d "$BINARY" 2>/dev/null | grep -i "ptrace\|syscall.*0x65" | head -3 | sed 's/^/    /'
    else
        echo "  ⚠ ptrace calls not visible (may be inlined or obfuscated)"
    fi
elif command -v readelf >/dev/null 2>&1; then
    echo "  Using readelf for analysis..."
    readelf -h "$BINARY" > /dev/null 2>&1 && echo "  ✓ Binary is valid ELF"
else
    echo "  ⚠ SKIP: objdump/readelf not available"
fi
echo ""

# Test 7: Rizin/Radare2 analysis (if available)
echo "[Test 7/7] Advanced Analysis - Rizin/Radare2"
echo "---------------------------------------------"
if command -v rizin >/dev/null 2>&1; then
    echo "Analyzing with Rizin..."
    echo "  Running: rizin -c 'aaa; pdf @main' $BINARY"
    echo "  (This will show the main function disassembly)"
    echo ""
    echo "  To manually analyze:"
    echo "    rizin $BINARY"
    echo "    [0x00000000]> aaa          # Analyze all"
    echo "    [0x00000000]> pdf @main    # Disassemble main"
    echo "    [0x00000000]> / ptrace     # Search for ptrace"
elif command -v r2 >/dev/null 2>&1; then
    echo "Analyzing with Radare2..."
    echo "  Running: r2 -c 'aaa; pdf @main' $BINARY"
else
    echo "  ⚠ SKIP: Rizin/Radare2 not installed"
    echo "  Install: sudo apt install rizin  # or: sudo pacman -S rizin"
fi
echo ""

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Binary: $BINARY"
echo ""
echo "Expected Results:"
echo "  ✓ Normal execution: Should work"
echo "  ✓ GDB debugging:    Should exit immediately (code 1)"
echo "  ✓ strace tracing:    Should exit immediately (code 1)"
echo "  ✓ ltrace tracing:    Should exit immediately (code 1)"
echo ""
echo "Manual Testing Commands:"
echo "  # Normal run"
echo "  $BINARY"
echo ""
echo "  # Try GDB (should fail)"
echo "  gdb $BINARY"
echo "  (gdb) run"
echo ""
echo "  # Try strace (should fail)"
echo "  strace $BINARY"
echo ""
echo "  # Try ltrace (should fail)"
echo "  ltrace $BINARY"
echo ""
echo "  # Analyze with Rizin"
echo "  rizin $BINARY"
echo "  [0x00000000]> aaa"
echo "  [0x00000000]> pdf @main"
echo "  [0x00000000]> / ptrace"
echo ""
