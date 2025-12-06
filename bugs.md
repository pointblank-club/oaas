# OAAS Bug Tracker

## Fixed Bugs

### Bug #1: "All Platforms" Button Not Working [FIXED]
- **Status**: Fixed
- **Error**: 422 HTTP error + `T.split is not a function` TypeError
- **Cause**: Backend didn't accept "all" as valid platform enum value
- **Fix Applied**:
  - Added `ALL = "all"` to Platform enum in `core/config.py`
  - Updated `server.py` to handle "all" platform - builds for Linux and Windows
  - Fixed frontend error handling for FastAPI validation errors (array vs string detail)

### Bug #2: Windows Selection Creates Linux + Windows Binaries [FIXED]
- **Status**: Fixed
- **Error**: Console showed `download_urls: {linux: ..., windows: ...}` when only Windows selected
- **Cause**: `target_platforms` was hardcoded to always include Linux and Windows
- **Fix Applied**:
  - Modified `server.py` to only build for selected platform(s)
  - When "all" is selected, builds for all platforms
  - When specific platform selected, only builds for that platform

### Bug #4: LLVM Remarks Output Not Found [FIXED]
- **Status**: Fixed
- **Error**: 404 error, "Remarks file not found"
- **Cause**: Remarks were disabled by default
- **Fix Applied**:
  - Changed `RemarksConfiguration.enabled` default to `True` in `core/config.py`
  - Updated `from_dict()` to default remarks.enabled to True

## Pending Bugs

### Bug #3: PDF Report Quality Issues [LOW - Fix Last]
- **Status**: Partially Confirmed (download works but quality is poor)
- **Issue**: PDF report quality is not good
- **Notes**: PDF downloads successfully but content/formatting needs improvement
- **Fix**: Improve PDF generation template and styling

---

## Additional Issues Found During Testing

### Windows Cross-Compilation with OLLVM Passes
- **Error**: `clang: error: unsupported option '-b' for target 'x86_64-w64-windows-gnu'`
- **Cause**: OLLVM pass flags incompatible with Windows cross-compilation
- **Workaround**: Disable Layer 3 (OLLVM Passes) for Windows builds

### Baseline Compilation Fails for Windows
- **Note**: Baseline comparison unavailable for Windows platform

### Layer 3 (OLLVM Passes) Crashes with String Constants [CRITICAL]
- **Status**: Confirmed Bug
- **Error**: `constant expression type mismatch: got type '[15 x i8]' but expected '[16 x i8]'`
- **Reproduction**: Enable Layer 3 with any OLLVM pass on Fibonacci Calculator demo
- **Cause**: OLLVM plugin has a bug with string constant type handling in LLVM IR
- **Impact**: Layer 3 is currently broken ("cooked") for programs with string constants
- **Workaround**: Use Layer 2 (String Encryption) instead, which works correctly

### LLVM Remarks Show "Missed" for Everything [NOT A BUG]
- **Status**: Expected Behavior
- **Question**: "Is it a bug that remarks show 'Missed' for everything?"
- **Answer**: NO - This is normal LLVM behavior
- **Explanation**:
  - `!Missed` remarks show optimizations that were *considered* but *not applied*
  - Common reasons in our case:
    - `NeverInline` with `optnone attribute` - functions marked to not inline
    - `NoDefinition` for `printf` - external function, definition unavailable
    - `FastISelFailure` - Fast instruction selection couldn't handle certain calls
  - The remarks also include `!Analysis` entries showing instruction counts, stack sizes, etc.
  - This is valuable information for understanding compilation behavior
- **Note**: When Layer 3 works, you would see `!Passed` remarks for applied obfuscation passes
