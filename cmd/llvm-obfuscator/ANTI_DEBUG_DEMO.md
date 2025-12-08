# Anti-Debugging Protection Demonstration Guide

This guide shows how to demonstrate to judges that your obfuscated binary has anti-debugging protection.

## Quick Test Script

```bash
# Make script executable
chmod +x test_anti_debug.sh

# Test your downloaded binary
./test_anti_debug.sh /path/to/downloaded/binary
```

## Manual Testing Methods

### Method 1: GDB (GNU Debugger)

**Expected Result**: Binary should exit immediately when run under GDB.

```bash
# Start GDB
gdb ./your_obfuscated_binary

# In GDB:
(gdb) run
# Binary should exit with code 1 immediately
# This proves anti-debugging detected GDB

# Check exit code
(gdb) info registers
# Look for exit code 1
```

**What to show the judge:**
- Screenshot/terminal showing binary exits immediately under GDB
- Exit code 1 indicates anti-debugging triggered

### Method 2: strace (System Call Tracer)

**Expected Result**: Binary should exit immediately when traced.

```bash
# Try to trace the binary
strace ./your_obfuscated_binary

# Expected: Binary exits immediately (code 1)
# This proves anti-debugging detected strace
```

**What to show the judge:**
- Terminal output showing binary exits immediately
- Compare with normal execution (which works fine)

### Method 3: ltrace (Library Call Tracer)

**Expected Result**: Binary should exit immediately when traced.

```bash
# Try to trace library calls
ltrace ./your_obfuscated_binary

# Expected: Binary exits immediately (code 1)
```

### Method 4: Rizin/Radare2 (Reverse Engineering Tools)

**Expected Result**: You can analyze the binary, but running it under debugger fails.

```bash
# Install Rizin (if not installed)
sudo apt install rizin  # Debian/Ubuntu
# or
sudo pacman -S rizin    # Arch Linux

# Analyze the binary
rizin ./your_obfuscated_binary

# In Rizin:
[0x00000000]> aaa              # Analyze all
[0x00000000]> pdf @main        # Disassemble main function
[0x00000000]> / ptrace         # Search for ptrace calls
[0x00000000]> / TracerPid      # Search for TracerPid checks

# Try to debug (should fail)
[0x00000000]> doo               # Run binary
# Binary should exit immediately
```

**What to show the judge:**
- Disassembly showing ptrace calls
- Evidence that running under debugger causes immediate exit

### Method 5: Binary Analysis

**Check for anti-debugging patterns:**

```bash
# Search for anti-debug strings
strings ./your_obfuscated_binary | grep -i "ptrace\|tracerpid\|/proc/self/status"

# Disassemble and search for syscalls
objdump -d ./your_obfuscated_binary | grep -i "ptrace\|syscall"

# Check for ptrace syscall (0x65 on x86_64)
objdump -d ./your_obfuscated_binary | grep "syscall" | grep "0x65"
```

## Demonstration Script for Judges

Create a simple demo script:

```bash
#!/bin/bash
# demo_anti_debug.sh

BINARY="./your_obfuscated_binary"

echo "=== Anti-Debugging Demonstration ==="
echo ""

echo "1. Normal Execution (should work):"
echo "-----------------------------------"
$BINARY
echo "Exit code: $?"
echo ""

echo "2. Under GDB (should exit immediately):"
echo "----------------------------------------"
echo "run" | gdb -batch -ex "set confirm off" $BINARY 2>&1 | tail -5
echo ""

echo "3. Under strace (should exit immediately):"
echo "-------------------------------------------"
strace $BINARY 2>&1 | head -10
echo ""

echo "4. Binary Analysis:"
echo "-------------------"
echo "Searching for anti-debug patterns..."
strings $BINARY | grep -i "ptrace\|tracerpid" | head -3
echo ""

echo "Disassembly showing ptrace calls:"
objdump -d $BINARY | grep -i "ptrace" | head -3
```

## What Judges Should See

### ✅ Evidence of Anti-Debugging:

1. **Normal execution works**: Binary runs fine when executed normally
2. **GDB fails**: Binary exits immediately (code 1) when run under GDB
3. **strace fails**: Binary exits immediately when traced with strace
4. **Binary contains anti-debug code**: 
   - Strings show "ptrace", "TracerPid", "/proc/self/status"
   - Disassembly shows ptrace syscalls
   - Code checks for debuggers at startup

### 📊 Comparison Table

| Test Method | Normal Binary | Obfuscated Binary (No Anti-Debug) | Obfuscated Binary (With Anti-Debug) |
|------------|---------------|-----------------------------------|--------------------------------------|
| Normal run | ✅ Works | ✅ Works | ✅ Works |
| GDB | ✅ Can debug | ✅ Can debug | ❌ Exits immediately |
| strace | ✅ Can trace | ✅ Can trace | ❌ Exits immediately |
| ltrace | ✅ Can trace | ✅ Can trace | ❌ Exits immediately |

## Technical Details

The anti-debugging implementation uses:

1. **ptrace(PTRACE_TRACEME)**: Detects if process is being traced
   - Returns -1 if already being traced (debugger attached)
   - Binary exits if ptrace returns -1

2. **/proc/self/status TracerPid**: Checks Linux proc filesystem
   - Reads TracerPid field
   - If non-zero, a debugger is attached
   - Binary exits if TracerPid != 0

3. **Parent process check**: Scans parent process name
   - Checks if parent is gdb, lldb, strace, ltrace
   - Binary exits if debugger detected in parent

4. **Timing checks** (optional): Detects slow execution
   - Measures execution time
   - Exits if execution is suspiciously slow (being single-stepped)

## Verification Commands

```bash
# 1. Verify binary exists and is executable
file ./your_obfuscated_binary
ls -lh ./your_obfuscated_binary

# 2. Test normal execution
./your_obfuscated_binary
echo "Exit code: $?"  # Should be 0

# 3. Test under GDB
gdb -batch -ex "run" ./your_obfuscated_binary
echo "Exit code: $?"  # Should be 1 (anti-debug triggered)

# 4. Test under strace
strace ./your_obfuscated_binary 2>&1 | tail -1
echo "Exit code: $?"  # Should be 1 (anti-debug triggered)

# 5. Verify anti-debug code exists
strings ./your_obfuscated_binary | grep -E "ptrace|TracerPid|/proc/self/status"
objdump -d ./your_obfuscated_binary | grep -i "ptrace\|syscall.*0x65"
```

## Troubleshooting

### Binary doesn't exit under GDB
- Check if anti-debugging was enabled during compilation
- Verify binary was compiled with `--enable-anti-debug` flag
- Check compilation logs for "Injecting anti-debugging protection"

### Binary exits even when run normally
- This might indicate a false positive
- Check if running in a container/VM (may trigger false positives)
- Verify the binary works in a clean environment

### Can't find ptrace in disassembly
- Code may be inlined or obfuscated
- Try searching for syscall 0x65 (ptrace on x86_64)
- Use Rizin/Radare2 for better analysis

## For Judges: How to Verify

1. **Download the obfuscated binary** from the website
2. **Run the test script**: `./test_anti_debug.sh <binary>`
3. **Manually verify**:
   ```bash
   # Should work
   ./binary
   
   # Should exit immediately
   gdb ./binary
   (gdb) run
   ```
4. **Check binary analysis**:
   ```bash
   strings ./binary | grep ptrace
   objdump -d ./binary | grep ptrace
   ```

The key evidence is: **Binary runs normally, but exits immediately when debugged.**

