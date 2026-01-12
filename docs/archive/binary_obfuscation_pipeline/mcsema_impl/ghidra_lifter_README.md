# Feature #2: Ghidra Lifter — Windows PE → McSema CFG

## Purpose

This stage takes a Windows PE binary (output from Feature #1) and uses Ghidra to export a control flow graph (CFG) in McSema-compatible format.

**Pipeline flow:**
```
Feature #1: Source → Windows PE (-O0 -g, MinGW-w64)
    ↓
Feature #2: Windows PE → CFG (Ghidra lifter, THIS STAGE)
    ↓
Feature #3: CFG → LLVM IR (McSema lifting)
    ↓
Feature #4: LLVM IR → Obfuscated IR (OLLVM passes)
    ↓
Feature #5: Obfuscated IR → Binary (Clang recompilation)
```

## How It Works

### Input
- Windows PE binary (`.exe`) from Feature #1
- Must have been compiled with `-O0 -g` (debug symbols required)
- Must NOT contain exceptions, recursion, switch statements, C++ constructs

### Processing

1. **Ghidra loads the binary**
   - Parses PE header
   - Identifies entry point, sections, imports
   - Runs auto-analysis

2. **Ghidra auto-analyzer runs**
   - Disassembles code
   - Identifies functions (via prologue detection)
   - Builds control flow graphs
   - Recovers data types (where possible)

3. **export_cfg.py script executes**
   - Iterates all recovered functions
   - Extracts basic blocks and edges
   - Outputs JSON CFG format compatible with McSema

### Output
```json
{
  "format": "mcsema-cfg",
  "version": "1.0",
  "binary_info": {
    "path": "/app/binaries/program.exe",
    "architecture": "x86:LE:64:default",
    "base_address": "0x400000"
  },
  "functions": [
    {
      "name": "main",
      "address": "0x401000",
      "size": 256,
      "parameters": 0,
      "basic_blocks": [
        {"start_addr": "0x401000", "end_addr": "0x401010", "size": 16},
        {"start_addr": "0x401010", "end_addr": "0x401020", "size": 16}
      ],
      "edges": [
        {"from": "0x401000", "to": "0x401010", "type": "fallthrough"},
        {"from": "0x401010", "to": "0x401000", "type": "branch"}
      ]
    }
  ]
}
```

## Architecture

### Docker Service

The ghidra-lifter runs as a separate Docker service in docker-compose.yml:

```yaml
ghidra-lifter:
  build:
    context: ./binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter
    dockerfile: Dockerfile
  image: pb-ghidra-lifter:latest
  ports:
    - "5001:5000"
  volumes:
    - ./cmd/llvm-obfuscator/reports:/app/reports
    - ./cmd/llvm-obfuscator/binaries:/app/binaries
  depends_on:
    - backend
```

### Service API

The lifter exposes a REST API on port 5000:

**Endpoint: `/lift` (POST)**
- Accept: multipart/form-data with binary file
- Return: `{"success": true, "cfg_file": "/path/to/output.cfg", "stats": {...}}`

**Endpoint: `/lift/file` (POST)**
- Accept: JSON `{"binary_path": "/app/binaries/program.exe", "output_dir": "/app/reports"}`
- Return: CFG file and stats

**Endpoint: `/health` (GET)**
- Return: `{"status": "healthy", "service": "ghidra-lifter"}`

### Backend Integration

The backend (`llvm-obfuscator-backend`) calls the lifter after Feature #1:

```python
# In backend API
response = requests.post(
    'http://ghidra-lifter:5000/lift/file',
    json={
        'binary_path': '/app/binaries/program.exe',
        'output_dir': '/app/reports'
    }
)

cfg_file = response.json()['cfg_file']
print(f"Next stage: mcsema-lift --cfg {cfg_file} --output program.bc")
```

## Limitations

### ⚠️ CRITICAL: Ghidra's CFG recovery is less reliable than IDA Pro

- **Function detection accuracy**: ~90% (vs IDA's ~98%)
- **Function prologue/epilogue**: Can fail on non-standard prologues
- **Tail call optimization**: May mis-identify tail calls as separate functions
- **Thunk functions**: Windows import thunks may be merged with callers

### Switch Statement Recovery

- Ghidra detects jump tables but recovery is unreliable
- Switch statements with >10 cases often misparsed
- Workaround: Enforce if/else chains in Feature #1 (via source validation)

### Control Flow Complexity

- Nested loops: Correctly recovered
- Indirect branches: CANNOT be resolved (no data flow analysis)
- Exception handlers: NOT PRESENT (disabled in Feature #1)
- Signal handlers: NOT PRESENT (C-only code)

### Windows-specific issues

- SEH tables: Disabled via Feature #1 flags (-fno-asynchronous-unwind-tables)
- ASLR relocations: Basic support (assumes static base address)
- Import address tables: Correctly identified
- .rsrc section: Ignored (no resource analysis)

### Noisy CFG Output

The CFG may contain:
- False function boundaries (split at unreachable code)
- Phantom edges (control flow analysis artifacts)
- Missing edges (complex conditions)
- Unreachable blocks (dead code analysis limitations)

**CFG accuracy on constrained binaries: ~80-85%**

The next stage (Feature #3: McSema lifting) will flag errors but cannot automatically correct malformed CFGs.

## Why Headless Mode?

```bash
analyzeHeadless <project> <name> -scriptPath <path> -postScript export_cfg.py <binary> <output>
```

Ghidra supports two modes:

| Mode | Use Case | This Pipeline |
|------|----------|---------------|
| **GUI** | Interactive reverse engineering | ❌ Not available in container |
| **Headless** | Batch processing, automation | ✅ Used here |

The headless mode:
- Requires no X11 display
- Fully automated (no user interaction)
- Runs in background
- Suitable for Docker containers
- Can process multiple binaries in parallel

## Why Ghidra (not IDA Pro)?

| Aspect | Ghidra | IDA Pro |
|--------|--------|---------|
| **Cost** | Free, open-source | $$$$ (expensive) |
| **Licensing** | NSA public release | Commercial |
| **Licensing for CI/CD** | ✅ Allowed | ❌ Not feasible |
| **Accuracy** | Good for -O0 binaries | Excellent (gold standard) |
| **Extensibility** | Python/Java scripts | Python plugin API |
| **Maintenance** | Active development | Corporate support |

**Trade-off**: We use Ghidra because it's free and works well for our constrained binaries. IDA would be more accurate but impossible to license for production.

## Next Steps

After CFG export, Feature #3 will process the .cfg file:

```bash
mcsema-lift --cfg program.cfg --output program.bc
```

This will:
1. Parse the CFG from Ghidra
2. Convert to LLVM IR
3. Output LLVM bitcode (.bc)
4. Forward to Feature #4 (OLLVM obfuscation)

## Debugging

### Checking lifter logs:

```bash
docker logs ghidra-lifter --tail 50
```

### Manual CFG inspection:

The .cfg file is JSON, so you can inspect it:

```bash
python3 -m json.tool /app/reports/program.cfg | less
```

### Common errors:

| Error | Cause | Fix |
|-------|-------|-----|
| "analyzeHeadless not found" | GHIDRA_HOME incorrect | Rebuild Docker image |
| "Timeout (5 minutes)" | Binary too large or Ghidra hung | Split binary or increase timeout |
| "CFG file not created" | export_cfg.py failed silently | Check Ghidra console output |
| "Invalid JSON" | CFG corrupted during write | Check disk space |

## Testing

### Test with a simple binary:

```bash
# Feature #1: Compile simple C code
python3 compile_windows_binary.py simple.c ./output/

# Feature #2: Lift with Ghidra
curl -X POST -F "file=@./output/program.exe" http://localhost:5001/lift

# Check output
cat /app/reports/program.cfg | python3 -m json.tool
```

## Future Improvements

- [ ] Add function signature recovery (parameter/return types)
- [ ] Improve switch table detection
- [ ] Add data flow analysis for indirect branches
- [ ] Support for C++ vtables (limited)
- [ ] Parallel batch lifting
- [ ] CFG validation and noise reduction
- [ ] Integration with IDA for comparison/validation
