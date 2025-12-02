# Custom Build System Support Plan

## Problem Statement

Current OAAS uses a "naive compilation" approach that:
1. Finds all C/C++ files in a repository
2. Attempts to compile them ALL together with a single `clang` command
3. Ignores the project's native build system (cmake, make, configure, etc.)

This fails for complex projects like CURL because:
- Multiple `main()` functions exist (in tests, examples)
- Missing cmake-generated config headers (`curl_config.h`)
- Missing include paths and compile flags
- Missing library dependencies

## Solution: Build-System Agnostic Obfuscation

### User-Facing Changes

Add a **Build System Type** selector in the UI with options:

| Option | Description | Example Projects |
|--------|-------------|------------------|
| **Simple (Single/Few Files)** | Direct clang compilation, no build system needed | Small utilities, CTF challenges |
| **CMake** | Uses `cmake -B build && cmake --build build` | CURL, OpenSSL, many modern C/C++ projects |
| **Make** | Uses `make` or `make all` | Linux kernel, older projects |
| **Autotools** | Uses `./configure && make` | GNU projects (wget, grep, etc.) |
| **Custom Command** | User provides exact build command | Anything else |

### Technical Implementation

#### Flow for "Simple" Mode (Current Behavior)
```
1. Find main file
2. Compile all C/C++ files together with clang
3. Apply obfuscation during compilation
```

#### Flow for "Custom Build" Modes (cmake/make/autotools/custom)
```
1. Clone repository to working directory
2. Apply SOURCE-LEVEL transformations to all C/C++ files IN-PLACE:
   - Layer 1: Symbol obfuscation (rename functions/variables)
   - Layer 2: String encryption (XOR encrypt literals)
   - Layer 2.5: Indirect call obfuscation
3. Set up compiler hijacking:
   - Export CC=/app/bin/clang (our clang with OLLVM)
   - Export CXX=/app/bin/clang++
   - Export CFLAGS with OLLVM plugin flags for Layer 3
   - Export CXXFLAGS with OLLVM plugin flags for Layer 3
4. Run user's build command in project root
5. Find output binaries (search for ELF/PE files in build/, bin/, etc.)
6. Package and return results
```

### Layer Application Matrix

| Layer | Type | Simple Mode | Custom Build Mode |
|-------|------|-------------|-------------------|
| Layer 1 (Symbol) | Source | During compile | Before build (in-place) |
| Layer 2 (String) | Source | During compile | Before build (in-place) |
| Layer 2.5 (Indirect) | Source | During compile | Before build (in-place) |
| Layer 3 (OLLVM) | Compiler | Via clang flags | Via CC/CXX hijack |
| Layer 4 (Flags) | Compiler | Via clang flags | Via CFLAGS/CXXFLAGS |

### API Changes

#### New Request Fields
```python
class ObfuscateRequest(BaseModel):
    # ... existing fields ...

    # NEW: Build system configuration
    build_system: str = "simple"  # "simple", "cmake", "make", "autotools", "custom"
    build_command: Optional[str] = None  # Custom build command (for "custom" mode)
    output_binary_path: Optional[str] = None  # Hint for where to find output binary
```

#### Build Commands by Type
```python
BUILD_COMMANDS = {
    "simple": None,  # Use direct clang compilation
    "cmake": "cmake -B build -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX && cmake --build build",
    "make": "make CC=$CC CXX=$CXX",
    "autotools": "./configure CC=$CC CXX=$CXX && make",
    "custom": None,  # Use user-provided build_command
}
```

### Environment Setup for Compiler Hijacking

```python
def setup_build_environment(config: ObfuscationConfig) -> Dict[str, str]:
    """Set up environment variables to hijack the build."""
    env = os.environ.copy()

    # Point to our clang
    env["CC"] = "/app/bin/clang"
    env["CXX"] = "/app/bin/clang++"

    # Build OLLVM flags based on enabled passes
    ollvm_flags = []
    plugin_path = "/app/plugins/linux-x86_64/LLVMObfuscationPlugin.so"

    if config.advanced.passes.flattening:
        ollvm_flags.append("-mllvm -fla")
    if config.advanced.passes.substitution:
        ollvm_flags.append("-mllvm -sub")
    if config.advanced.passes.bogus_control_flow:
        ollvm_flags.append("-mllvm -bcf")
    if config.advanced.passes.split_basic_blocks:
        ollvm_flags.append("-mllvm -split")

    # Add plugin loading if any OLLVM passes enabled
    if ollvm_flags:
        plugin_flags = f"-Xclang -load -Xclang {plugin_path} " + " ".join(ollvm_flags)
    else:
        plugin_flags = ""

    # Add Layer 4 compiler flags
    layer4_flags = " ".join(config.compiler_flags)

    env["CFLAGS"] = f"{plugin_flags} {layer4_flags}".strip()
    env["CXXFLAGS"] = env["CFLAGS"]

    return env
```

### Finding Output Binaries

```python
def find_output_binaries(project_root: Path, hint: Optional[str] = None) -> List[Path]:
    """Find compiled binaries in project after build."""
    binaries = []

    # If user provided a hint, check there first
    if hint:
        hint_path = project_root / hint
        if hint_path.exists() and is_executable(hint_path):
            return [hint_path]

    # Search common locations
    search_dirs = ["build", "bin", "out", ".", "src", "Release", "Debug"]

    for search_dir in search_dirs:
        dir_path = project_root / search_dir
        if not dir_path.exists():
            continue

        for file in dir_path.rglob("*"):
            if file.is_file() and is_elf_or_pe(file):
                binaries.append(file)

    return binaries
```

### Source-Level Obfuscation (In-Place)

```python
def obfuscate_project_sources(
    project_root: Path,
    config: ObfuscationConfig,
    logger: logging.Logger
) -> Dict:
    """Apply source-level obfuscation to all C/C++ files in project."""

    source_extensions = ['*.c', '*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp']
    results = {
        "files_processed": 0,
        "symbols_renamed": 0,
        "strings_encrypted": 0,
        "indirect_calls": 0,
    }

    # Find all source files
    all_sources = []
    for ext in source_extensions:
        all_sources.extend(project_root.rglob(ext))

    # Build shared symbol map for consistency across files
    symbol_map = {}

    for source_file in all_sources:
        try:
            content = source_file.read_text(encoding='utf-8', errors='ignore')
            modified = content

            # Layer 1: Symbol obfuscation
            if config.advanced.symbol_obfuscation.enabled:
                modified, file_symbols = apply_symbol_obfuscation(
                    modified,
                    symbol_map,
                    config.advanced.symbol_obfuscation
                )
                results["symbols_renamed"] += file_symbols

            # Layer 2: String encryption
            if config.advanced.string_encryption:
                modified, strings_count = apply_string_encryption(modified)
                results["strings_encrypted"] += strings_count

            # Layer 2.5: Indirect call obfuscation
            if config.advanced.indirect_calls.enabled:
                modified, calls_count = apply_indirect_calls(modified)
                results["indirect_calls"] += calls_count

            # Write back in-place
            if modified != content:
                source_file.write_text(modified, encoding='utf-8')
                results["files_processed"] += 1

        except Exception as e:
            logger.warning(f"Failed to obfuscate {source_file}: {e}")

    return results
```

## Frontend Changes

### New UI Section in Configuration

```tsx
// Build System Configuration
<div className="build-system-config">
  <label>Build System Type:</label>
  <select
    value={buildSystem}
    onChange={(e) => setBuildSystem(e.target.value)}
  >
    <option value="simple">Simple (Direct Compilation)</option>
    <option value="cmake">CMake</option>
    <option value="make">Make</option>
    <option value="autotools">Autotools (configure + make)</option>
    <option value="custom">Custom Command</option>
  </select>

  {buildSystem === 'custom' && (
    <div className="custom-command">
      <label>Build Command:</label>
      <input
        type="text"
        placeholder="e.g., meson build && ninja -C build"
        value={customBuildCommand}
        onChange={(e) => setCustomBuildCommand(e.target.value)}
      />
    </div>
  )}

  <div className="output-hint">
    <label>Output Binary Path (optional):</label>
    <input
      type="text"
      placeholder="e.g., build/bin/curl"
      value={outputBinaryPath}
      onChange={(e) => setOutputBinaryPath(e.target.value)}
    />
    <small>Helps find the compiled binary after build</small>
  </div>
</div>
```

## Implementation Checklist

- [ ] Add `build_system`, `build_command`, `output_binary_path` fields to API request
- [ ] Create `obfuscate_project_sources()` function for in-place source obfuscation
- [ ] Create `setup_build_environment()` function for CC/CXX hijacking
- [ ] Create `find_output_binaries()` function for locating compiled binaries
- [ ] Modify `api_obfuscate_sync()` to handle custom build modes
- [ ] Add build system selector to frontend
- [ ] Add custom command input field to frontend
- [ ] Add output path hint field to frontend
- [ ] Test with CURL repository
- [ ] Test with a simple Makefile project
- [ ] Test with autotools project

## Testing Plan

1. **Simple Mode**: Verify existing behavior still works
2. **CMake Mode**: Test with CURL repository
3. **Make Mode**: Test with a simple Makefile project
4. **Autotools Mode**: Test with a GNU project (e.g., wget)
5. **Custom Mode**: Test with user-provided build command

## Expected Results

After implementation, users should be able to:
1. Select their project's build system type
2. Optionally provide a custom build command
3. Have all obfuscation layers applied correctly
4. Receive the obfuscated binary(ies) as output
