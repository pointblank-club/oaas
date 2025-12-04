# ClangIR/Polygeist Pipeline Implementation Plan

## CRITICAL REQUIREMENTS ⚠️

1. **NO BREAKING CHANGES** - Existing LLVM pipeline MUST continue to work
2. **BACKWARD COMPATIBLE** - All existing passes (OLLVM, MLIR) must work unchanged
3. **OPTIONAL FEATURE** - ClangIR/Polygeist is an OPT-IN enhancement
4. **DEFAULT BEHAVIOR** - System defaults to current working pipeline

## Current Pipeline (MUST REMAIN WORKING)

```
C/C++ Source
    ↓
Clang → LLVM IR
    ↓
mlir-translate --import-llvm → MLIR
    ↓
MLIR Passes (string-encrypt, symbol-obfuscate, crypto-hash, constant-obfuscate)
    ↓
mlir-translate --mlir-to-llvmir → LLVM IR
    ↓
OLLVM Passes (optional)
    ↓
Clang → Binary
```

## New Pipeline (OPTIONAL, OPT-IN)

```
C/C++ Source
    ↓
┌─────────────────────────────────┐
│   FRONTEND CHOICE (NEW!)        │
│   ├─ Option 1: Clang (default) │  ← EXISTING (NO CHANGE)
│   ├─ Option 2: ClangIR (new)   │  ← NEW FEATURE
│   └─ Option 3: Polygeist (new) │  ← NEW FEATURE
└─────────────────────────────────┘
    ↓
High-Level MLIR (ClangIR or Polygeist dialect)
    ↓
MLIR Passes (string-encrypt, symbol-obfuscate, crypto-hash, constant-obfuscate)
    ↓
Lower to LLVM Dialect MLIR
    ↓
mlir-translate --mlir-to-llvmir → LLVM IR
    ↓
OLLVM Passes (optional)
    ↓
Clang → Binary
```

## Implementation Strategy

### Phase 1: Add Dependencies (NO CODE CHANGES)

**File**: `Dockerfile.test`

```dockerfile
# Add ClangIR and Polygeist repositories (LLVM 22.0.0)
# This is ADDITIVE ONLY - doesn't change existing tools
```

**Risk**: ⚠️ LOW - Only adds new tools, doesn't modify existing ones

### Phase 2: Configuration (ADDITIVE ONLY)

**File**: `cmd/llvm-obfuscator/core/config.py`

```python
# Add new enum (doesn't break existing code)
class MLIRFrontend(str, Enum):
    CLANG = "clang"          # DEFAULT - existing behavior
    CLANGIR = "clangir"      # NEW
    POLYGEIST = "polygeist"  # NEW

# Add new field to ObfuscationConfig (with default)
@dataclass
class ObfuscationConfig:
    # ... existing fields ...
    mlir_frontend: MLIRFrontend = MLIRFrontend.CLANG  # DEFAULT to existing
```

**Risk**: ⚠️ VERY LOW - New field with safe default (existing behavior)

### Phase 3: Pipeline Selector (CONDITIONAL LOGIC)

**File**: `cmd/llvm-obfuscator/core/obfuscator.py`

```python
def _compile(self, source, destination, config, ...):
    # EXISTING CODE PATH (default)
    if config.mlir_frontend == MLIRFrontend.CLANG:
        # Current implementation - NO CHANGES
        self._compile_with_clang_llvm(...)  # Existing code

    # NEW CODE PATHS (opt-in)
    elif config.mlir_frontend == MLIRFrontend.CLANGIR:
        self._compile_with_clangir(...)     # New function

    elif config.mlir_frontend == MLIRFrontend.POLYGEIST:
        self._compile_with_polygeist(...)   # New function
```

**Risk**: ⚠️ LOW - Existing code path unchanged, new paths are isolated

### Phase 4: New Helper Functions (ISOLATED)

**File**: `cmd/llvm-obfuscator/core/obfuscator.py`

```python
def _compile_with_clangir(self, ...):
    """NEW FUNCTION - doesn't touch existing code"""
    # 1. clangir → MLIR
    # 2. Apply MLIR passes (same as existing)
    # 3. Lower to LLVM IR
    # 4. Continue with existing pipeline
    pass

def _compile_with_polygeist(self, ...):
    """NEW FUNCTION - doesn't touch existing code"""
    # 1. polygeist → MLIR
    # 2. Apply MLIR passes (same as existing)
    # 3. Lower to LLVM IR
    # 4. Continue with existing pipeline
    pass
```

**Risk**: ⚠️ VERY LOW - New isolated functions, no impact on existing code

## Testing Strategy

### 1. Regression Tests (EXISTING PIPELINE)

```bash
# Test 1: Existing MLIR pipeline (string-encrypt)
python3 -m cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --output ./test1
# MUST WORK - no changes to this path

# Test 2: Existing OLLVM pipeline
python3 -m cli.obfuscate compile test.c \
    --enable-flattening \
    --output ./test2
# MUST WORK - no changes to this path

# Test 3: Combined MLIR + OLLVM
python3 -m cli.obfuscate compile test.c \
    --enable-constant-obfuscate \
    --enable-crypto-hash \
    --enable-flattening \
    --output ./test3
# MUST WORK - no changes to this path
```

### 2. New Feature Tests (ClangIR/Polygeist)

```bash
# Test 4: ClangIR pipeline (new)
python3 -m cli.obfuscate compile test.c \
    --mlir-frontend clangir \
    --enable-constant-obfuscate \
    --output ./test4
# NEW FEATURE - should work with ClangIR

# Test 5: Polygeist pipeline (new)
python3 -m cli.obfuscate compile test.c \
    --mlir-frontend polygeist \
    --enable-crypto-hash \
    --output ./test5
# NEW FEATURE - should work with Polygeist
```

## Safety Guarantees

### ✅ Guaranteed Safe:

1. **Default behavior unchanged** - No `--mlir-frontend` flag = existing behavior
2. **Existing passes work** - All MLIR/OLLVM passes unchanged
3. **No API changes** - Existing CLI flags work identically
4. **Isolated new code** - ClangIR/Polygeist in separate functions
5. **Conditional execution** - New code only runs when explicitly requested

### ⚠️ Potential Risks (Mitigated):

1. **Dockerfile changes** - MITIGATION: Only ADD tools, don't REMOVE or MODIFY
2. **Config schema** - MITIGATION: New fields have safe defaults
3. **Pipeline logic** - MITIGATION: Existing path in separate function, untouched

## Implementation Order (SAFE)

### Step 1: Update Dockerfile (ADDITIVE)
- Add ClangIR build from source (LLVM 22.0.0)
- Add Polygeist build from source (LLVM 22.0.0)
- Keep all existing tools (clang, mlir-opt, etc.)

### Step 2: Update config.py (BACKWARD COMPATIBLE)
- Add `MLIRFrontend` enum
- Add `mlir_frontend` field with default = CLANG
- Update `from_dict()` to parse new field (optional)

### Step 3: Refactor obfuscator.py (EXTRACT, NO MODIFY)
- Extract current MLIR pipeline to `_compile_with_clang_llvm()`
- No logic changes, just code movement
- Test that existing behavior still works

### Step 4: Add ClangIR function (NEW CODE)
- Implement `_compile_with_clangir()`
- Only executes when frontend = CLANGIR
- No impact on existing code

### Step 5: Add Polygeist function (NEW CODE)
- Implement `_compile_with_polygeist()`
- Only executes when frontend = POLYGEIST
- No impact on existing code

### Step 6: Update documentation (SAFE)
- Document new `--mlir-frontend` flag
- Emphasize default behavior (clang)
- Provide migration guide

## Rollback Plan

If anything breaks:

1. **Git revert** - All changes in isolated commits
2. **Feature flag** - Can disable ClangIR/Polygeist via config
3. **Default safe** - System defaults to working state

## File Modification Summary

| File | Change Type | Risk | Rollback |
|------|-------------|------|----------|
| `Dockerfile.test` | ADDITIVE | LOW | Remove new RUN commands |
| `config.py` | ADDITIVE | VERY LOW | Remove new enum/field |
| `obfuscator.py` | REFACTOR + ADD | LOW | Git revert |
| `MLIR_INTEGRATION_GUIDE.md` | UPDATE | NONE | N/A |

## Success Criteria

✅ **MUST PASS:**
1. All existing tests pass without changes
2. Existing CLI commands work identically
3. No changes to MLIR pass implementations
4. No changes to OLLVM pass implementations
5. Default behavior (no flags) produces same output

✅ **NICE TO HAVE:**
1. ClangIR frontend works for simple C files
2. Polygeist frontend works for simple C files
3. Combined with MLIR passes (constant-obfuscate, crypto-hash)

## Next Steps

1. ✅ Review this plan with user
2. ⏭️ Implement Step 1 (Dockerfile updates)
3. ⏭️ Implement Step 2 (config.py updates)
4. ⏭️ Implement Step 3 (obfuscator.py refactor)
5. ⏭️ Test existing pipeline (regression tests)
6. ⏭️ Implement Step 4 (ClangIR support)
7. ⏭️ Implement Step 5 (Polygeist support)
8. ⏭️ Final integration testing

---

**CRITICAL**: At EVERY step, we run regression tests to ensure existing functionality is NOT broken.

**Version**: 1.0.0
**Target LLVM**: 22.0.0
**Backward Compatible**: YES ✅
