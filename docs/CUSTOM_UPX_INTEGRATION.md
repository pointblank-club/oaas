# Custom UPX Packer Integration

## Overview

The obfuscation pipeline now supports using a custom UPX binary instead of the system-installed UPX. This allows you to use a custom-built or modified version of UPX for packing binaries.

## Custom UPX Location

Your custom UPX is located at:
```
/home/dhruv/Documents/upx/build/upx
```

## Usage

### CLI Usage

Use the `--upx-custom-path` option to specify your custom UPX binary:

```bash
python -m cli.obfuscate compile examples/hello.c \
    --output ./output \
    --enable-upx \
    --upx-custom-path /home/dhruv/Documents/upx/build/upx
```

### API Usage

Include `custom_upx_path` in the UPX configuration:

```json
{
  "source_code": "...",
  "config": {
    "upx": {
      "enabled": true,
      "compression_level": "best",
      "use_lzma": true,
      "preserve_original": false,
      "custom_upx_path": "/home/dhruv/Documents/upx/build/upx"
    }
  }
}
```

### Configuration File (YAML/JSON)

```yaml
obfuscation:
  advanced:
    upx_packing:
      enabled: true
      compression_level: best
      use_lzma: true
      preserve_original: false
      custom_upx_path: /home/dhruv/Documents/upx/build/upx
```

## Implementation Details

### Changes Made

1. **UPXConfiguration** (`core/config.py`)
   - Added `custom_upx_path: Optional[Path]` field

2. **UPXPacker** (`core/upx_packer.py`)
   - Updated `__init__()` to accept `custom_upx_path` parameter
   - Added `_resolve_upx_binary()` method to resolve custom or system UPX
   - Updated `_check_upx_available()` to verify custom UPX binary
   - All UPX commands now use the resolved binary path

3. **LLVMObfuscator** (`core/obfuscator.py`)
   - UPX packer is now initialized lazily with custom path from config
   - Logs custom UPX path when used

4. **CLI** (`cli/obfuscate.py`)
   - Added `--upx-custom-path` option
   - Passes custom path to configuration

5. **API Server** (`api/server.py`)
   - Added `custom_upx_path` field to `UPXModel`
   - Includes custom path in configuration building

6. **Binary Obfuscator** (`core/binary_obfuscator.py`)
   - Updated to use custom UPX path when packing binaries

## Behavior

- **Priority**: Custom UPX path takes precedence over system UPX
- **Validation**: The custom UPX binary is checked for existence and executability
- **Fallback**: If custom UPX is invalid, falls back to system UPX (if available)
- **Error Handling**: Graceful warnings if custom UPX is not available

## Testing

The integration has been tested and verified:
- Configuration creation with custom path ✓
- UPXPacker initialization with custom path ✓
- UPX binary detection and availability check ✓

## Example Workflow

```bash
# 1. Build your custom UPX (if needed)
cd /home/dhruv/Documents/upx
make

# 2. Use custom UPX in obfuscation
python -m cli.obfuscate compile my_program.c \
    --output ./obfuscated \
    --enable-upx \
    --upx-custom-path /home/dhruv/Documents/upx/build/upx \
    --upx-compression best
```

## Notes

- The custom UPX path must point to an executable binary
- The binary will be tested with `--version` to verify it's a valid UPX
- If the custom path is invalid, the system will log a warning and attempt to use system UPX
- All existing UPX functionality (compression levels, LZMA, etc.) works with custom UPX


