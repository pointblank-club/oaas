# Binary-to-Binary Obfuscation: Alternative Approaches

## Current Status

McSema lifting produces LLVM IR with hardcoded absolute addresses that cannot be recompiled to working executables. This document outlines alternative approaches.

## Chosen Approach: Direct Binary Patching (Option 3)

**Status: SELECTED FOR IMPLEMENTATION**
**Estimated Effort: 3-5 days**

### Overview

Skip lifting entirely - patch obfuscation directly into the original binary by:
1. Parse PE sections and identify code
2. Disassemble target functions
3. Apply obfuscation transforms at assembly level
4. Insert obfuscated code in new PE section
5. Redirect original function entries to obfuscated versions
6. Preserve original code layout (addresses remain valid)

### Implementation Plan

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Original PE    │────▶│  Disassemble &   │────▶│  Apply ASM-level│
│  (hello.exe)    │     │  Parse Sections  │     │  Obfuscation    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Obfuscated PE  │◀────│  Patch Entry     │◀────│  Add New Code   │
│  (hello_obf.exe)│     │  Points          │     │  Section        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### ASM-Level Obfuscation Techniques

1. **Instruction Substitution**
   ```asm
   ; Before
   add eax, ebx

   ; After (equivalent)
   push ecx
   mov ecx, ebx
   not ecx
   not eax
   and eax, ecx
   not eax
   sub eax, ecx
   add eax, ebx
   add eax, ebx
   pop ecx
   ```

2. **Dead Code Insertion**
   ```asm
   ; Insert between real instructions
   push eax
   xor eax, eax
   add eax, 0x12345678
   pop eax
   ; Continue with real code
   ```

3. **Opaque Predicates**
   ```asm
   ; Always-true condition
   mov eax, 7
   imul eax, eax    ; eax = 49
   and eax, 1       ; eax = 1 (49 is odd)
   jnz real_code    ; Always taken
   jmp fake_code    ; Never taken
   ```

4. **Control Flow Flattening (simplified)**
   ```asm
   ; Dispatcher-based execution
   mov [state], 0
   dispatcher:
     cmp [state], 0
     je block_0
     cmp [state], 1
     je block_1
     jmp exit
   block_0:
     ; original code
     mov [state], 1
     jmp dispatcher
   ```

### Tools to Use

- **Capstone** - Disassembly engine
- **Keystone** - Assembly engine
- **LIEF** - PE parsing and modification
- **pefile** - Python PE manipulation

### Advantages

- ✅ Works immediately - no lifting required
- ✅ Preserves original memory layout
- ✅ All addresses remain valid
- ✅ Can selectively obfuscate functions
- ✅ Lower complexity than full lifting

### Limitations

- ⚠️ Limited to assembly-level transforms
- ⚠️ Cannot apply LLVM IR passes
- ⚠️ Increases binary size
- ⚠️ Architecture-specific (x86/x64)

---

## Alternative Approaches (Not Selected)

### Option 1: Fix McSema Inline Assembly
**Effort: 2-4 weeks**

Post-process lifted IR to replace hardcoded addresses:
- Parse inline assembly blocks
- Replace `movq $$0x1400014a0` with symbol references
- Regenerate trampolines with PC-relative addressing

**Why not selected:** High complexity, fragile solution

---

### Option 2: Use RetDec Lifter
**Effort: 1-2 weeks**

RetDec (by Avast) produces cleaner LLVM IR:
```bash
retdec-decompiler hello.exe --backend-emit-llvm -o hello.ll
```

**Why not selected:** Still requires full lifting pipeline, medium accuracy

**Could revisit if:** Direct patching proves insufficient

---

### Option 4: VM-Based Obfuscation
**Effort: 1-2 weeks**

Convert functions to custom bytecode + embedded interpreter:
1. Design simple VM instruction set
2. Compile target functions to VM bytecode
3. Embed interpreter in binary
4. Replace functions with `call vm_interpret`

**Why not selected:** Significant performance overhead (10-100x)

**Could revisit if:** Maximum protection needed regardless of performance

---

### Option 5: Use rev.ng Lifter
**Effort: 1-2 weeks**

rev.ng is designed specifically for recompilation:
```bash
revng lift hello.exe -o hello.ll
revng opt hello.ll -o hello_opt.ll
revng translate hello_opt.ll -o hello_new.exe
```

**Why not selected:** Less mature, primarily Linux-focused

**Could revisit if:** Need full LLVM IR transformation capability

---

### Option 6: Commercial Tools
**Effort: 1-3 days**

Existing solutions:
- **Themida/WinLicense** - Industry standard PE protection
- **VMProtect** - VM-based code virtualization
- **Enigma Protector** - Comprehensive protection suite

**Why not selected:** Expensive licensing, doesn't fit our open-source model

---

## Implementation Priority

| Priority | Option | Status |
|----------|--------|--------|
| 1️⃣ | Direct Binary Patching | **SELECTED** |
| 2️⃣ | RetDec (backup) | On hold |
| 3️⃣ | VM Obfuscation | Future consideration |
| 4️⃣ | rev.ng | Future consideration |
| 5️⃣ | Fix McSema | Not recommended |
| 6️⃣ | Commercial | Not applicable |

---

## Next Steps

1. **Day 1-2:** Set up binary patching infrastructure
   - Integrate LIEF/pefile for PE manipulation
   - Integrate Capstone for disassembly
   - Integrate Keystone for reassembly

2. **Day 2-3:** Implement core obfuscation transforms
   - Instruction substitution at ASM level
   - Dead code insertion
   - Basic opaque predicates

3. **Day 4-5:** Integration and testing
   - Add new PE section for obfuscated code
   - Patch function entries
   - Test with hello.exe and larger binaries

---

**Document Version:** 1.0
**Created:** 2025-12-09
**Decision:** Direct Binary Patching (Option 3)
