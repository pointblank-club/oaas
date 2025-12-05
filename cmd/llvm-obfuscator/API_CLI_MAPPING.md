# API Endpoint to CLI Flag Mapping

This document maps all API endpoints in the backend to their corresponding CLI flags in the obfuscator CLI.

## Main Obfuscation Endpoints

### POST `/api/obfuscate/sync` and POST `/api/obfuscate`

These endpoints accept an `ObfuscateRequest` with the following structure:

#### Request Body Structure

```json
{
  "source_code": "string (base64)",           // CLI: input_file (positional argument)
  "filename": "string",                        // CLI: derived from input_file
  "platform": "LINUX|WINDOWS|MACOS|DARWIN",   // CLI: --platform
  "architecture": "X86_64|ARM64|X86",        // CLI: --architecture (not directly in CLI, inferred from platform)
  "entrypoint_command": "string",              // CLI: N/A (internal use)
  "config": {
    "level": 1-5,                             // CLI: --level
    "passes": {
      "flattening": bool,                     // CLI: --enable-flattening
      "substitution": bool,                    // CLI: --enable-substitution
      "bogus_control_flow": bool,            // CLI: --enable-bogus-cf
      "split": bool,                          // CLI: --enable-split
      "linear_mba": bool,                     // CLI: --enable-linear-mba
      "string_encrypt": bool,                 // CLI: --enable-string-encrypt
      "symbol_obfuscate": bool,               // CLI: --enable-symbol-obfuscate
      "constant_obfuscate": bool              // CLI: N/A (not in CLI yet)
    },
    "cycles": 1-5,                            // CLI: --cycles
    "string_encryption": bool,                // CLI: --enable-string-encrypt (legacy, same as passes.string_encrypt)
    "fake_loops": 0-50,                       // CLI: --fake-loops
    "upx": {
      "enabled": bool,                        // CLI: --enable-upx
      "compression_level": "fast|default|best|brute",  // CLI: --upx-compression
      "use_lzma": bool,                       // CLI: --upx-lzma / --no-upx-lzma
      "preserve_original": bool                // CLI: --upx-preserve-original
    },
    "indirect_calls": {
      "enabled": bool,                        // CLI: --enable-indirect-calls
      "obfuscate_stdlib": bool,               // CLI: --indirect-stdlib / --no-indirect-stdlib
      "obfuscate_custom": bool                // CLI: --indirect-custom / --no-indirect-custom
    }
  },
  "report_formats": ["json", "markdown", "pdf"],  // CLI: --report-formats
  "custom_flags": ["string"],                 // CLI: --custom-flags
  "custom_pass_plugin": "string",             // CLI: --custom-pass-plugin
  "source_files": [...],                      // CLI: N/A (multi-file projects)
  "repo_session_id": "string",                 // CLI: N/A (GitHub integration)
  "build_system": "simple|cmake|make|autotools|custom",  // CLI: N/A (multi-file projects)
  "build_command": "string",                  // CLI: N/A (multi-file projects)
  "output_binary_path": "string",              // CLI: N/A (multi-file projects)
  "cmake_options": "string"                   // CLI: N/A (multi-file projects)
}
```

#### Detailed Mapping

| API Field | CLI Flag/Option | CLI Command | Notes |
|-----------|----------------|-------------|-------|
| `source_code` | `input_file` (positional) | `compile <input_file>` | Base64 encoded in API, file path in CLI |
| `filename` | Derived from `input_file` | N/A | Auto-derived from input file name |
| `platform` | `--platform` | `--platform LINUX\|WINDOWS\|MACOS\|DARWIN` | Case-insensitive in CLI |
| `architecture` | `--architecture` | `--architecture X86_64\|ARM64\|X86` | Case-insensitive in CLI, default: X86_64 |
| `config.level` | `--level` | `--level 1\|2\|3\|4\|5` | Default: 3 |
| `config.passes.flattening` | `--enable-flattening` | `--enable-flattening` | Boolean flag |
| `config.passes.substitution` | `--enable-substitution` | `--enable-substitution` | Boolean flag |
| `config.passes.bogus_control_flow` | `--enable-bogus-cf` | `--enable-bogus-cf` | Boolean flag |
| `config.passes.split` | `--enable-split` | `--enable-split` | Boolean flag |
| `config.passes.linear_mba` | `--enable-linear-mba` | `--enable-linear-mba` | Boolean flag |
| `config.passes.string_encrypt` | `--enable-string-encrypt` | `--enable-string-encrypt` | Boolean flag |
| `config.passes.symbol_obfuscate` | `--enable-symbol-obfuscate` | `--enable-symbol-obfuscate` | Boolean flag |
| `config.passes.constant_obfuscate` | `--enable-constant-obfuscate` | `--enable-constant-obfuscate` | Boolean flag |
| `config.cycles` | `--cycles` | `--cycles <1-5>` | Default: 1 |
| `config.string_encryption` | `--enable-string-encrypt` | `--enable-string-encrypt` | Legacy field, same as `passes.string_encrypt` |
| `config.fake_loops` | `--fake-loops` | `--fake-loops <0-50>` | Default: 0 |
| `config.upx.enabled` | `--enable-upx` | `--enable-upx` | Boolean flag |
| `config.upx.compression_level` | `--upx-compression` | `--upx-compression fast\|default\|best\|brute` | Default: "best" |
| `config.upx.use_lzma` | `--upx-lzma` / `--no-upx-lzma` | `--upx-lzma` or `--no-upx-lzma` | Default: `--upx-lzma` (True) |
| `config.upx.preserve_original` | `--upx-preserve-original` | `--upx-preserve-original` | Boolean flag |
| `config.indirect_calls.enabled` | `--enable-indirect-calls` | `--enable-indirect-calls` | Boolean flag |
| `config.indirect_calls.obfuscate_stdlib` | `--indirect-stdlib` / `--no-indirect-stdlib` | `--indirect-stdlib` or `--no-indirect-stdlib` | Default: `--indirect-stdlib` (True) |
| `config.indirect_calls.obfuscate_custom` | `--indirect-custom` / `--no-indirect-custom` | `--indirect-custom` or `--no-indirect-custom` | Default: `--indirect-custom` (True) |
| `report_formats` | `--report-formats` | `--report-formats json,markdown,pdf` | Comma-separated, default: "json" |
| `custom_flags` | `--custom-flags` | `--custom-flags "-O2 -Wall"` | Space-separated compiler flags |
| `custom_pass_plugin` | `--custom-pass-plugin` | `--custom-pass-plugin /path/to/plugin.so` | Path to custom LLVM pass plugin |
| `output` | `--output` | `--output <directory>` | Default: "./obfuscated" |
| `config_file` | `--config-file` | `--config-file config.yaml` | YAML/JSON config file (alternative to all above flags) |

#### CLI Command Example

```bash
# Equivalent CLI command for a typical API request:
obfuscator compile input.c \
  --output ./obfuscated \
  --platform LINUX \
  --level 3 \
  --enable-flattening \
  --enable-substitution \
  --enable-bogus-cf \
  --enable-split \
  --enable-linear-mba \
  --enable-string-encrypt \
  --enable-symbol-obfuscate \
  --cycles 2 \
  --fake-loops 5 \
  --enable-indirect-calls \
  --indirect-stdlib \
  --indirect-custom \
  --enable-upx \
  --upx-compression best \
  --upx-lzma \
  --report-formats json,markdown,pdf \
  --custom-flags "-O2 -Wall"
```

---

### GET `/api/analyze/{job_id}`

Analyzes an obfuscated binary from a completed job.

| API Parameter | CLI Command | Notes |
|---------------|-------------|-------|
| `job_id` (path) | `analyze <binary>` | CLI takes binary path directly |

#### CLI Command Example

```bash
# API: GET /api/analyze/{job_id}
# CLI equivalent:
obfuscator analyze ./obfuscated/output_binary
```

---

### POST `/api/compare`

Compares original and obfuscated binaries.

| API Field | CLI Command | Notes |
|-----------|-------------|-------|
| `original_b64` | `original` (positional) | Base64 encoded in API, file path in CLI |
| `obfuscated_b64` | `obfuscated` (positional) | Base64 encoded in API, file path in CLI |
| `filename` | N/A | Used for report naming only |

#### CLI Command Example

```bash
# API: POST /api/compare with base64 binaries
# CLI equivalent:
obfuscator compare original.bin obfuscated.bin
```

---

### GET `/api/jobs`

Lists all obfuscation jobs.

| API Parameter | CLI Command | Notes |
|---------------|-------------|-------|
| N/A | `jobs` | CLI command exists but job management is API-only (no persistent storage in CLI) |

#### CLI Command Example

```bash
# API: GET /api/jobs
# CLI equivalent:
obfuscator jobs
```

---

### GET `/api/report/{job_id}`

Gets a report for a completed job.

| API Parameter | CLI Behavior | Notes |
|---------------|--------------|-------|
| `job_id` (path) | Reports auto-generated | CLI generates reports automatically |
| `fmt` (query: json\|markdown\|pdf) | `--report-formats` | CLI supports multiple formats via flag |

#### CLI Command Example

```bash
# API: GET /api/report/{job_id}?fmt=markdown
# CLI: Reports are generated automatically with --report-formats
obfuscator compile input.c --report-formats json,markdown,pdf
```

---

### GET `/api/download/{job_id}/{platform}` and GET `/api/download/{job_id}`

Downloads obfuscated binaries. No direct CLI equivalent (CLI outputs to local directory).

---

### GET `/api/health`

Health check endpoint.

| API Parameter | CLI Command | Notes |
|---------------|-------------|-------|
| N/A | `health` | Returns status: "ok" |

#### CLI Command Example

```bash
# API: GET /api/health
# CLI equivalent:
obfuscator health
```

---

### GET `/api/flags`

Returns available compiler flags.

| API Parameter | CLI Command | Notes |
|---------------|-------------|-------|
| N/A | `flags` | Returns list of available compiler flags |

#### CLI Command Example

```bash
# API: GET /api/flags
# CLI equivalent:
obfuscator flags
```

---

### GET `/api/capabilities`

Returns API capabilities.

| API Parameter | CLI Command | Notes |
|---------------|-------------|-------|
| N/A | `capabilities` | Returns pass plugin availability and GitHub OAuth status |

#### CLI Command Example

```bash
# API: GET /api/capabilities
# CLI equivalent:
obfuscator capabilities
```

---

## GitHub Integration Endpoints

These endpoints are API-only and have no CLI equivalents:

- `GET /api/github/login` - OAuth login
- `GET /api/github/callback` - OAuth callback
- `GET /api/github/status` - OAuth status
- `POST /api/github/disconnect` - Disconnect GitHub
- `GET /api/github/repos` - List repositories
- `GET /api/github/repo/branches` - List branches
- `POST /api/github/repo/files` - List repository files
- `POST /api/github/repo/clone` - Clone repository
- `GET /api/github/repo/session/{session_id}` - Get session
- `DELETE /api/github/repo/session/{session_id}` - Delete session
- `POST /api/github/repo/session/{session_id}` - Update session

---

## Local File Upload Endpoints

- `POST /api/local/folder/upload` - Upload local folder for multi-file projects

No CLI equivalent (CLI works with file paths directly).

---

## CLI Commands Not in API

### `batch` Command

```bash
obfuscator batch config.yaml
```

Processes multiple obfuscation jobs from a YAML configuration file.

| CLI Command | API Endpoint | Notes |
|-------------|--------------|-------|
| `batch <config_path>` | `POST /api/batch` | Both accept YAML config file path |

#### API Request Example

```bash
# CLI: obfuscator batch config.yaml
# API equivalent:
POST /api/batch
{
  "config_path": "/path/to/config.yaml"
}
```

### `jotai` Command

```bash
obfuscator jotai \
  --output ./jotai_results \
  --category anghaLeaves \
  --limit 10 \
  --level 3 \
  --enable-flattening \
  --enable-substitution \
  --enable-bogus-cf \
  --enable-split \
  --enable-string-encrypt \
  --custom-flags "-O2" \
  --max-failures 5 \
  --cache-dir ./cache
```

Runs Jotai benchmark suite.

| CLI Flag | API Field | Notes |
|----------|-----------|-------|
| `--output` | `output` | Output directory for results |
| `--category` | `category` | Benchmark category (anghaLeaves, anghaMath) |
| `--limit` | `limit` | Maximum number of benchmarks |
| `--level` | `level` | Obfuscation level 1-5 |
| `--enable-flattening` | `enable_flattening` | Enable control flow flattening |
| `--enable-substitution` | `enable_substitution` | Enable instruction substitution |
| `--enable-bogus-cf` | `enable_bogus_cf` | Enable bogus control flow |
| `--enable-split` | `enable_split` | Enable basic block splitting |
| `--enable-string-encrypt` | `enable_string_encrypt` | Enable string encryption |
| `--custom-flags` | `custom_flags` | Additional compiler flags (list in API) |
| `--custom-pass-plugin` | `custom_pass_plugin` | Path to custom LLVM pass plugin |
| `--max-failures` | `max_failures` | Stop after this many consecutive failures |
| `--cache-dir` | `cache_dir` | Directory to cache Jotai benchmarks |

#### API Request Example

```bash
# CLI: obfuscator jotai --output ./results --category anghaLeaves --level 3
# API equivalent:
POST /api/jotai
{
  "output": "./results",
  "category": "anghaLeaves",
  "level": 3,
  "enable_flattening": false,
  "enable_substitution": false,
  "enable_bogus_cf": false,
  "enable_split": false,
  "enable_string_encrypt": false,
  "max_failures": 5
}
```

**Note**: Fixed inconsistency - `jotai` command now uses `--enable-string-encrypt` (matching `compile` command).

---

## Summary

### Fully Mapped Endpoints (1:1 Mapping)
- ✅ `POST /api/obfuscate/sync` → `compile` command
- ✅ `POST /api/obfuscate` → `compile` command (async)
- ✅ `GET /api/analyze/{job_id}` → `analyze` command
- ✅ `POST /api/compare` → `compare` command
- ✅ `GET /api/jobs` → `jobs` command
- ✅ `GET /api/health` → `health` command
- ✅ `GET /api/flags` → `flags` command
- ✅ `GET /api/capabilities` → `capabilities` command
- ✅ `POST /api/batch` → `batch` command
- ✅ `POST /api/jotai` → `jotai` command

### Partially Mapped Endpoints
- ⚠️ `GET /api/report/{job_id}` → Reports auto-generated in CLI (no direct command, but reports are generated automatically)

### API-Only Endpoints (No CLI Equivalent - by design)
- ❌ Download endpoints (`GET /api/download/{job_id}/*`) - CLI outputs to local directory
- ❌ GitHub integration (all `/api/github/*` endpoints) - Web UI feature
- ❌ Local folder upload (`POST /api/local/folder/upload`) - CLI works with file paths directly

### CLI-Only Features (No API Equivalent - by design)
- ❌ Direct file path access (CLI works with local files, API uses base64 encoding)

---

## Notes

1. **Config File Alternative**: Both API and CLI support loading configuration from a YAML/JSON file via `config_file` (API) or `--config-file` (CLI), which can replace all individual flags.

2. **Multi-file Projects**: The API supports multi-file projects via `source_files` or `repo_session_id`, but the CLI `compile` command only handles single files. Use the `batch` command for multiple files.

3. **Architecture**: Both API and CLI now support explicit `architecture` field (`--architecture` flag in CLI).

4. **Constant Obfuscation**: Both API and CLI now support `--enable-constant-obfuscate` flag.

5. **String Encryption Flag**: Fixed inconsistency - `jotai` command now uses `--enable-string-encrypt` (matching `compile` command).

4. **Output Directory**: The API uses `output_binary_path` for multi-file projects, while CLI uses `--output` for the output directory.

5. **Report Formats**: API accepts a list `["json", "markdown", "pdf"]`, CLI accepts comma-separated string `"json,markdown,pdf"`.

