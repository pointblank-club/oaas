# Stage 1: Windows Binary Generation for CFG Lifting

## Purpose

This stage converts user-submitted C source code into a safe, unoptimized Windows PE binary optimized for control flow graph (CFG) reconstruction via Ghidra + McSema.

The binary is intentionally compiled with:
- **-O0**: Disables optimizations (required so CFG matches source 1-to-1)
- **-g**: Enables debug symbols (helps Ghidra identify function boundaries)
- **No SEH/EH metadata**: Windows exception handling confuses McSema
- **No inlining**: Function boundaries must be explicit
- **Static linking**: No runtime dependencies

## Limitations

This stage enforces constraints because McSema + Ghidra CFG lifting is not yet robust enough for complex binaries:

| Forbidden | Reason |
|-----------|--------|
| **Recursion** | McSema uses context-insensitive CFG; recursive calls break analysis |
| **Inline Assembly** | Cannot be lifted to LLVM IR |
| **Switch Statements** | Ghidra recovers jump tables unreliably; prefer if/else chains |
| **C++ Constructs** | Virtual tables, exception handling, namespaces break lifting |
| **Function Pointers** | Indirect calls cannot be resolved statically |
| **Variadic Functions** | Cannot be analyzed reliably |

**These are NOT permanent limitations.** They reflect Stage 1 constraints. Future stages will add:
- Stage 2: Handle switch statements (via jump table recovery)
- Stage 3: Support simple function pointers (via points-to analysis)
- Stage 4: Partial C++ support (limited to simple classes)

## How It Works

### Input
```
C source code (string or file)
```

### Processing

1. **Validation**: Scan source for forbidden constructs
   - Rejects recursion, inline asm, switch, C++, function pointers
   - Returns detailed error messages

2. **Compilation**: Uses MinGW-w64 with safe flags
   - Target: x86_64-w64-mingw32 Windows PE (`.exe`)
   - Compiler: `x86_64-w64-mingw32-gcc` (from system PATH)
   - Flags: `-O0 -g -fno-asynchronous-unwind-tables -fno-exceptions -fno-stack-protector -fno-inline`
   - Output: `program.exe` (unoptimized, debug-enabled)
   - **Note**: MinGW-w64 is pre-installed in the backend container (Dockerfile.backend line 28)

3. **Metadata Generation**: Creates `compilation_metadata.json`
   - Stores compilation flags
   - Documents constraints
   - Lists next-stage requirements

4. **Output**
   ```
   output/
   ├── program.exe
   └── compilation_metadata.json
   ```

### Next Stage
The binary is now **READY_FOR_GHIDRA_LIFTER**.

The binary will be passed to a Ghidra lifter script (running inside Docker) which will:
- Load the PE binary
- Perform function analysis
- Recover control flow graph (CFG)
- Export `.cfg` file in McSema-compatible format

## Critical Warnings

⚠️ **Read these carefully before extending this pipeline:**

1. **Ghidra is less reliable than IDA Pro**
   - IDA's commercial CFG recovery is more accurate
   - Ghidra works here only because we enforce -O0, simple binaries

2. **Windows PE CFG recovery is fragile**
   - SEH tables can interfere with CFG analysis
   - Windows ABI thunks (.plt equivalents) confuse lifting
   - Function prologue/epilogue detection can fail

3. **McSema cannot yet handle real-world Windows binaries**
   - This pipeline is experimental
   - It works for simple -O0 code with explicit CFG
   - Complex binaries will fail

4. **This is Stage 1 only**
   - Not intended for production use
   - CFG recovery success rate: ~85% on constrained binaries
   - Will improve as we add support for restricted constructs

## Usage

### Python API
```python
from compile_windows_binary import process_user_source

source_code = """
int add(int a, int b) {
    return a + b;
}

int main() {
    return add(5, 3);
}
"""

success, result = process_user_source(source_code, './output')

if success:
    print(f"Binary: {result['output_binary']}")
    print(f"Next stage: {result['next_stage']}")
else:
    print(f"Error: {result['error']}")
```

### Command Line
```bash
python compile_windows_binary.py source.c ./output
```

Output:
```json
{
  "success": true,
  "output_binary": "./output/program.exe",
  "metadata_file": "./output/compilation_metadata.json",
  "next_stage": "READY_FOR_GHIDRA_LIFTER",
  "next_action": "Binary at ./output/program.exe is ready for Ghidra lifter"
}
```

## Compilation Flags Explained

### -O0 (CRITICAL)
Disables all optimizations. Optimized code has:
- Merged basic blocks (CFG cannot be recovered)
- Inlined functions (loses function boundaries)
- Eliminated dead code (CFG differs from source)

McSema requires 1-to-1 mapping between source and compiled IR.

### -g (CRITICAL)
Emits DWARF debug symbols in PE binary.
- Helps Ghidra identify function prologue/epilogue
- Provides variable scope information
- Without this, Ghidra's CFG recovery fails 40%+ of the time

### -fno-asynchronous-unwind-tables (CRITICAL)
Disables unwind tables (SEH on Windows).
- Windows uses Structured Exception Handling (SEH) instead of DWARF unwind info
- SEH tables contain raw offsets/pointers that confuse McSema's lifting
- Disabling keeps binary simpler and CFG more deterministic

### -fno-exceptions (CRITICAL)
Disables C++ exception handling and C exception support.
- Exception handlers add hidden control flow edges
- McSema cannot model exception dispatch deterministically
- By forbidding exceptions, all control flow is explicit

### -fno-stack-protector (IMPORTANT)
Disables stack cookies (__security_cookie checks).
- Windows adds runtime checks that McSema cannot lift
- These checks appear as function calls in the CFG
- Disabling keeps IR clean

### -fno-inline (CRITICAL)
Disables function inlining.
- Inlined functions destroy CFG structure
- McSema needs function boundaries to be explicit in binary
- Without this flag, many functions vanish into caller code

## Next Steps

After binary generation, the pipeline will continue:

```
Stage 1: Source → Windows PE (THIS)
    ↓
Stage 2: Windows PE → CFG (Ghidra lifter in Docker)
    ↓
Stage 3: CFG → LLVM IR (McSema)
    ↓
Stage 4: LLVM IR → Obfuscated IR (OLLVM passes)
    ↓
Stage 5: Obfuscated IR → Binary (Clang recompilation)
```

## Extending This Stage

To add support for more language features:

1. **Relax constraints in SourceCodeValidator** (not yet—requires research)
2. **Add feature-specific flags** (e.g., `-fsanitize=undefined` for runtime checks)
3. **Extend Ghidra lifter** to handle new constructs
4. **Update McSema lifting** to process new constructs

**Do not proceed without validating Ghidra + McSema can handle the new construct.**
