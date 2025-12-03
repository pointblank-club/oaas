# UPX Binary Packing - Layer 5

## Overview

**UPX (Ultimate Packer for eXecutables)** is an open-source executable packer that provides binary compression and an additional obfuscation layer. In the LLVM Obfuscator, UPX serves as an **optional Layer 5** applied after all other obfuscation layers.

## How It Works

### Pipeline Position

UPX is applied as the **final step** after all other obfuscation:

```
Source Code
    ↓
[Layer 1] Symbol Obfuscation (PRE-COMPILE)
    ↓
[Layer 2] String Encryption (PRE-COMPILE)
    ↓
[Layer 3] OLLVM Passes (COMPILE - optional)
    ↓
[Layer 4] Compiler Flags (COMPILE)
    ↓
[Layer 5] UPX Packing (POST-COMPILE - optional)  ← HERE
    ↓
Final Obfuscated Binary
```

### Compression Process

1. UPX compresses the binary using LZMA or other algorithms
2. Adds a small decompression stub to the binary header
3. At runtime, the stub decompresses the binary into memory
4. Execution proceeds normally after decompression

## Benefits

| Benefit | Description |
|---------|-------------|
| **Size Reduction** | 50-70% smaller binaries |
| **Static Analysis Harder** | Binary content is compressed/encrypted |
| **Counteracts Bloat** | OLLVM passes increase size; UPX compensates |
| **Additional Protection** | Adds friction for reverse engineers |

## Trade-offs

| Concern | Impact |
|---------|--------|
| **Startup Overhead** | 10-50ms decompression time at launch |
| **Antivirus Flags** | Some AV software may flag UPX-packed binaries |
| **Reversible** | Can be unpacked with `upx -d` (but adds friction) |
| **Not All Binaries** | Some statically-linked or special binaries may fail |

## Configuration Options

### Compression Levels

| Level | Flag | Speed | Compression | Use Case |
|-------|------|-------|-------------|----------|
| `fast` | `--fast` | Fastest | ~40-50% | Quick builds, testing |
| `default` | (none) | Balanced | ~50-60% | General use |
| `best` | `--best` | Slow | ~60-70% | Production builds |
| `brute` | `--brute` | Very slow | ~65-75% | Maximum compression |

### LZMA Compression

- **Enabled by default** (`--lzma` flag)
- Better compression ratio than default algorithm
- Slightly slower decompression at runtime
- Recommended for production use

### Preserve Original

- When enabled, keeps a `.pre-upx` backup of the original binary
- Useful for debugging or comparison
- Disabled by default to save disk space

## Usage

### CLI

```bash
# Enable UPX with best compression and LZMA
python3 -m cli.obfuscate compile source.c \
  --enable-upx \
  --upx-compression best \
  --upx-lzma \
  --upx-preserve-original

# Available options:
#   --enable-upx                  Enable UPX binary packing
#   --upx-compression <level>     fast | default | best | brute
#   --upx-lzma / --no-upx-lzma    Use LZMA compression (default: true)
#   --upx-preserve-original       Keep backup of pre-UPX binary
```

### API

```json
{
  "config": {
    "upx": {
      "enabled": true,
      "compression_level": "best",
      "use_lzma": true,
      "preserve_original": false
    }
  }
}
```

### Frontend

Enable "Layer 5: UPX Binary Packing" in the obfuscation layers section and configure:
- Compression level dropdown
- LZMA compression toggle
- Preserve original toggle

## Performance Impact

### On User Binaries (Layer 5)

| Metric | Without UPX | With UPX |
|--------|-------------|----------|
| Binary Size | 100% (baseline) | 30-50% (50-70% reduction) |
| Startup Time | 0ms | +10-50ms |
| Runtime Speed | 100% | 100% (no impact) |
| RE Difficulty | 8-12 weeks | 10-14 weeks |

### On LLVM Tools (Docker Image)

The Docker image also uses UPX to compress bundled LLVM binaries (`clang.real`, `opt`) for smaller image size:

```dockerfile
RUN upx --best --lzma /usr/local/llvm-obfuscator/bin/clang.real
RUN upx --best --lzma /usr/local/llvm-obfuscator/bin/opt
```

**Impact on compilation speed:**

| Aspect | Impact |
|--------|--------|
| Startup per invocation | +50-200ms (decompression) |
| Multi-file projects | Noticeable slowdown (tools invoked many times) |
| Single file | Minimal impact |

**Trade-off decision:** Smaller Docker image (~50-70% reduction on those binaries) vs slower compilation. Current choice prioritizes smaller image size.

## Implementation Details

### File Locations

| File | Purpose |
|------|---------|
| `cmd/llvm-obfuscator/core/upx_packer.py` | Main UPXPacker class |
| `cmd/llvm-obfuscator/core/config.py` | UPXConfiguration dataclass |
| `cmd/llvm-obfuscator/core/obfuscator.py` | Integration in main pipeline |
| `cmd/llvm-obfuscator/cli/obfuscate.py` | CLI options |
| `cmd/llvm-obfuscator/api/server.py` | API integration |
| `cmd/llvm-obfuscator/frontend/src/App.tsx` | Frontend UI |

### UPXPacker Class

```python
class UPXPacker:
    COMPRESSION_LEVELS = {
        "fast": ["--fast"],
        "default": [],
        "best": ["--best"],
        "brute": ["--brute"],
    }

    def pack(self, binary_path, compression_level="best", use_lzma=True,
             force=True, preserve_original=True) -> dict:
        # Returns: {status, original_size, packed_size, compression_ratio, ...}

    def unpack(self, binary_path) -> bool:
        # Unpacks a UPX-compressed binary

    def _is_packed(self, binary_path) -> bool:
        # Checks for "UPX!" signature in binary
```

### Detection

UPX-packed binaries contain a `UPX!` signature in the first 1KB:

```python
def _is_packed(self, binary_path: Path) -> bool:
    with open(binary_path, "rb") as f:
        content = f.read(1024)
        return b"UPX!" in content
```

## Installation

UPX must be installed on the system:

```bash
# Linux (Debian/Ubuntu)
sudo apt install upx-ucl

# macOS
brew install upx

# Verify installation
upx --version
```

The obfuscator gracefully handles missing UPX - it logs a warning and skips packing rather than failing.

## Security Considerations

1. **Not encryption**: UPX is compression, not encryption. Determined attackers can unpack.
2. **AV detection**: Some antivirus software flags UPX-packed binaries as suspicious.
3. **Defense in depth**: UPX should be used alongside other layers, not as sole protection.
4. **Adds friction**: Even if reversible, it increases time/effort for reverse engineering.

## Troubleshooting

### UPX Packing Failed

Common causes:
- Binary is statically linked (some static binaries incompatible)
- Binary already packed
- Unsupported binary format
- UPX not installed

Check logs for specific error messages.

### Binary Won't Run After Packing

- Try unpacking: `upx -d binary`
- Test with `upx -t binary` to verify integrity
- Some systems block UPX-packed binaries (security software)

### Large Startup Delay

- Expected: 10-50ms for small binaries
- Large binaries (>10MB) may have 100-500ms delay
- Consider using `fast` compression for development builds
