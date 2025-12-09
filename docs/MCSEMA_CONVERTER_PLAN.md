# Ghidra → McSema CFG Converter Plan

## Goal
Convert Ghidra's JSON CFG output to McSema's protobuf CFG format to enable binary-to-LLVM IR lifting for Windows PE binaries compiled with `-O0 -g`.

---

## Current State (What Exists)

### 1. Ghidra Lifter Service (WORKING)
- **Location**: `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/`
- **Files**:
  - `lifter_script/lifter_service.py` - Flask API on port 5000
  - `lifter_script/export_cfg.py` - Ghidra headless script
- **Output Format**: JSON
- **Data Exported**:
  ```json
  {
    "format": "mcsema-cfg",
    "version": "1.0",
    "binary_info": { "path", "architecture", "base_address" },
    "functions": [
      {
        "name": "main",
        "address": "0x00401000",
        "size": 256,
        "basic_blocks": [{ "start_addr", "end_addr", "size" }],
        "edges": [{ "from", "to", "type" }]
      }
    ]
  }
  ```

### 2. McSema Pre-built Binary (AVAILABLE)
- **Location**: `mcsema-install/mcsema/bin/mcsema-lift-11.0`
- **Requires**: x86_64 Linux (use `--platform linux/amd64` in Docker)
- **Input**: McSema protobuf CFG (`.cfg` binary file)
- **Output**: LLVM 11 bitcode (`.bc`)

### 3. McSema Protobuf Schema (REFERENCE)
- **Location**: `mcsema-install/mcsema/lib/python3/site-packages/mcsema_disass-3.1.3.8-py3.8.egg/mcsema_disass/ida7/CFG_pb2.py`
- **Key Messages**:
  - `Module` - top level (name, funcs, segments, external_funcs, external_vars)
  - `Function` - (ea, blocks, is_entrypoint, name)
  - `Block` - (ea, instructions, successor_eas, is_referenced_by_data)
  - `Instruction` - (ea, xrefs, lp_ea)
  - `Segment` - (ea, data, read_only, is_external, name)
  - `ExternalFunction` - (name, ea, cc, has_return, no_return, argument_count)

---

## Target Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Windows PE     │     │  Ghidra CFG      │     │  McSema CFG     │
│  Binary (.exe)  │────▶│  (JSON)          │────▶│  (Protobuf)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │                       │                        ▼
        │                       │               ┌─────────────────┐
        │                       │               │  mcsema-lift    │
        │                       │               │  (LLVM 11 BC)   │
        └───────────────────────┼──────────────▶└─────────────────┘
              (raw bytes)       │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                │               │  OLLVM Pass     │
                                │               │  (Obfuscate)    │
                                │               └─────────────────┘
```

---

## Implementation Tasks

### TASK 1: Enhance Ghidra Export Script
**File**: `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/lifter_script/export_cfg.py`

**ADD the following data to JSON output**:

```python
# 1. Memory segments with raw bytes (base64 encoded)
"segments": [
  {
    "name": ".text",
    "start_addr": "0x00401000",
    "end_addr": "0x00402000",
    "permissions": "rx",
    "data_base64": "<base64 encoded bytes>"
  }
]

# 2. Instructions with full detail
"instructions": [
  {
    "address": "0x00401000",
    "bytes_base64": "<base64>",
    "mnemonic": "push",
    "operands": "rbp",
    "length": 1,
    "xrefs": [{ "type": "code"|"data", "target": "0x..." }]
  }
]

# 3. External functions (imports)
"external_functions": [
  {
    "name": "printf",
    "address": "0x00405000",
    "library": "msvcrt.dll",
    "calling_convention": "cdecl",
    "has_return": true,
    "no_return": false
  }
]

# 4. External variables (imported data)
"external_variables": [
  {
    "name": "__imp_stdin",
    "address": "0x00406000",
    "size": 8
  }
]
```

**Ghidra API calls needed**:
```python
# Segments
memory = currentProgram.getMemory()
for block in memory.getBlocks():
    name = block.getName()
    start = block.getStart()
    end = block.getEnd()
    data = getBytes(start, block.getSize())  # raw bytes

# External functions
symbol_table = currentProgram.getSymbolTable()
ext_manager = currentProgram.getExternalManager()
for ext_loc in ext_manager.getExternalLocations():
    name = ext_loc.getLabel()
    address = ext_loc.getAddress()
    library = ext_loc.getLibraryName()

# Instructions with bytes
for instr in listing.getInstructions(func.getBody(), True):
    addr = instr.getAddress()
    bytes = instr.getBytes()
    refs = instr.getReferencesFrom()
```

---

### TASK 2: Create Protobuf Converter
**File**: `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/lifter_script/json_to_mcsema.py`

**Purpose**: Convert enhanced JSON CFG to McSema protobuf binary

**Steps**:
1. Parse JSON CFG
2. Create McSema Module protobuf
3. Populate segments with raw bytes
4. Populate functions with blocks and instructions
5. Populate external functions with calling conventions
6. Serialize to binary protobuf file

**Key Code Structure**:
```python
import json
import sys
sys.path.insert(0, '/mcsema/mcsema/lib/python3/site-packages/mcsema_disass-3.1.3.8-py3.8.egg')
from mcsema_disass.ida7 import CFG_pb2

def convert(json_path, output_path, binary_path):
    with open(json_path) as f:
        cfg_json = json.load(f)

    module = CFG_pb2.Module()
    module.name = cfg_json["binary_info"]["path"]

    # Add segments
    for seg in cfg_json["segments"]:
        segment = module.segments.add()
        segment.ea = int(seg["start_addr"], 16)
        segment.data = base64.b64decode(seg["data_base64"])
        segment.read_only = "w" not in seg["permissions"]
        segment.is_external = False
        segment.name = seg["name"]
        segment.is_exported = False
        segment.is_thread_local = False

    # Add functions
    for func in cfg_json["functions"]:
        function = module.funcs.add()
        function.ea = int(func["address"], 16)
        function.name = func["name"]
        function.is_entrypoint = func["name"] == "main"

        # Add blocks
        for block in func["basic_blocks"]:
            blk = function.blocks.add()
            blk.ea = int(block["start_addr"], 16)
            blk.is_referenced_by_data = False

            # Add instructions (from instruction list)
            # Add successor_eas (from edges)

    # Add external functions
    for ext in cfg_json["external_functions"]:
        ext_func = module.external_funcs.add()
        ext_func.name = ext["name"]
        ext_func.ea = int(ext["address"], 16)
        ext_func.cc = get_calling_convention(ext["calling_convention"])
        ext_func.has_return = ext["has_return"]
        ext_func.no_return = ext["no_return"]
        ext_func.argument_count = ext.get("arg_count", 0)
        ext_func.is_weak = False

    # Write protobuf
    with open(output_path, "wb") as f:
        f.write(module.SerializeToString())
```

**Calling Convention Mapping**:
```python
CC_MAP = {
    "cdecl": CFG_pb2.C,
    "stdcall": CFG_pb2.X86_StdCall,
    "fastcall": CFG_pb2.X86_FastCall,
    "thiscall": CFG_pb2.X86_ThisCall,
    "win64": CFG_pb2.Win64,
    "sysv64": CFG_pb2.X86_64_SysV,
}
```

---

### TASK 3: Update Lifter Service
**File**: `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/lifter_script/lifter_service.py`

**ADD new endpoint or modify existing**:
```python
@app.route('/lift/mcsema', methods=['POST'])
def lift_to_mcsema():
    # 1. Run Ghidra export (enhanced JSON)
    # 2. Run json_to_mcsema.py converter
    # 3. Return path to .cfg protobuf file
```

---

### TASK 4: Create McSema Lift Container/Service
**New File**: `binary_obfuscation_pipeline/mcsema_impl/mcsema_lift/Dockerfile`

```dockerfile
FROM ubuntu:20.04
# Must be x86_64 for mcsema-lift binary
COPY mcsema-install/mcsema /mcsema
COPY mcsema-install/remill /remill
ENV PATH="/mcsema/bin:$PATH"
# ... service to call mcsema-lift
```

**Service Flow**:
```python
# Input: .cfg protobuf + original binary
# Output: .bc LLVM bitcode

cmd = [
    "mcsema-lift-11.0",
    "--cfg", cfg_path,
    "--output", output_bc,
    "--arch", "amd64",      # or x86 for 32-bit
    "--os", "windows",      # for PE binaries
]
```

---

### TASK 5: Integrate into Pipeline Worker
**File**: `cmd/llvm-obfuscator/core/binary_pipeline_worker.py`

**Update stage flow**:
```python
def execute(self):
    # Stage 1: Ghidra CFG Export (JSON) - EXISTS
    # Stage 2: JSON→McSema Protobuf - NEW
    # Stage 3: McSema Lift (CFG→LLVM 11 BC) - NEW
    # Stage 4: LLVM Version Upgrade (11→22) - NEEDED?
    # Stage 5: OLLVM Obfuscation - EXISTS
    # Stage 6: Final Binary Generation - EXISTS
```

---

## Tradeoffs & Limitations

### What Will Work (for -O0 -g binaries)
- ✅ Function boundaries (debug symbols help)
- ✅ Basic block structure (straightforward CF)
- ✅ Simple external calls (printf, malloc, etc.)
- ✅ Global data access (segments included)
- ✅ Stack-based locals (no register allocation)

### What May Fail
- ❌ C++ exceptions (no exception frame support initially)
- ❌ Virtual function calls (vtable resolution complex)
- ❌ Thread-local storage (TLS not handled)
- ❌ Inline assembly (passed through as-is)
- ❌ Computed jumps/switch tables (limited support)

### LLVM Version Gap
- McSema outputs LLVM 11 bitcode
- Our OLLVM is LLVM 22
- **Solution**: Use `llvm-upgrade` or `llvm-dis`/`llvm-as` pipeline
- **Alternative**: May work directly if IR is compatible

---

## File Locations Summary

```
binary_obfuscation_pipeline/mcsema_impl/
├── ghidra_lifter/
│   ├── Dockerfile                    # EXISTS - Ghidra container
│   └── lifter_script/
│       ├── lifter_service.py         # EXISTS - Flask API
│       ├── export_cfg.py             # MODIFY - Add segments, imports, instructions
│       └── json_to_mcsema.py         # NEW - JSON→Protobuf converter
├── mcsema_lift/                      # NEW DIRECTORY
│   ├── Dockerfile                    # NEW - McSema container (x86_64)
│   └── lift_service.py               # NEW - mcsema-lift wrapper
└── HOTFIXES.txt                      # DELETED - Now in docs/

mcsema-install/                       # EXISTS - Downloaded pre-built
├── mcsema/
│   └── bin/mcsema-lift-11.0          # EXISTS - Main lifter binary
└── remill/                           # EXISTS - Semantics library
```

---

## Execution Order for LLM

1. **First**: Modify `export_cfg.py` to add segments, external functions, and instruction bytes
2. **Second**: Create `json_to_mcsema.py` converter script
3. **Third**: Test converter locally with a sample CFG
4. **Fourth**: Create McSema lift Dockerfile and service
5. **Fifth**: Update `binary_pipeline_worker.py` to call new stages
6. **Sixth**: Test end-to-end with simple Windows PE binary

---

## Test Binary Requirements

For initial testing, use a simple Windows PE:
```c
// test.c - compile with: x86_64-w64-mingw32-gcc -O0 -g test.c -o test.exe
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int x = 5;
    int y = 10;
    int z = add(x, y);
    printf("Result: %d\n", z);
    return 0;
}
```

This has:
- Simple functions (main, add)
- One external call (printf)
- Local variables
- No exceptions, no C++, no complex features
