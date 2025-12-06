from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from github import Github
from pydantic import BaseModel, Field

from core import (
    AnalyzeConfig,
    LLVMObfuscator,
    ObfuscationConfig,
    ObfuscationReport,
    Platform,
    analyze_binary,
    report_converter,
)
from core.comparer import CompareConfig, compare_binaries
from core.config import AdvancedConfiguration, PassConfiguration, UPXConfiguration, Architecture, RemarksConfiguration
from core.exceptions import JobNotFoundError, ValidationError
from core.ir_analyzer import IRAnalyzer
from core.test_suite_integration import (
    run_obfuscation_tests,
    run_lightweight_tests,
    merge_test_results_into_report
)
from core.job_manager import JobManager
from core.progress import ProgressEvent, ProgressTracker
from core.utils import (
    create_logger,
    ensure_directory,
    normalize_flags_and_passes,
    get_timestamp,
    get_file_size,
    detect_binary_format,
    list_sections,
    summarize_symbols,
    compute_entropy,
)

from .multifile_pipeline import (
    GitHubRepoRequest,
    RepoData,
    RepoFile,
    SourceFile,
    cleanup_old_sessions,
    cleanup_repo_session,
    clone_repo_to_temp,
    create_local_upload_session,
    create_multi_file_project,
    extract_repo_files,
    find_main_file,
    get_repo_branches,
    get_repo_session,
    get_session_token,
)

# Load flags database for the /api/flags endpoint. Prefer importing from the
# repo's scripts module; fall back to loading the file directly, and finally to
# an empty list if unavailable.
try:  # Attempt local package import if PYTHONPATH includes repo root
    from scripts.flags import comprehensive_flags  # type: ignore
except Exception:  # pragma: no cover - dev fallback when running from cmd/
    try:
        repo_root = Path(__file__).resolve().parents[4]
        flags_path = repo_root / "scripts" / "flags.py"
        namespace: Dict[str, object] = {}
        exec(flags_path.read_text(), namespace)
        comprehensive_flags = namespace.get("comprehensive_flags", [])  # type: ignore
    except Exception:
        comprehensive_flags = []  # type: ignore

API_KEY_HEADER = "x-api-key"
DEFAULT_API_KEY = os.environ.get("OBFUSCATOR_API_KEY", "change-me")
DISABLE_AUTH = os.environ.get("OBFUSCATOR_DISABLE_AUTH", "false").lower() == "true"
MAX_SOURCE_SIZE = 100 * 1024 * 1024  # 100MB

# GitHub OAuth Configuration
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.environ.get("GITHUB_REDIRECT_URI", "http://localhost:4666/api/github/callback")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:4666")

# In-memory storage for OAuth states and tokens (use Redis/DB in production)
# Maps state -> {"token": access_token, "timestamp": created_at}
oauth_states: Dict[str, str] = {}
user_sessions: Dict[str, Dict[str, object]] = {}
SESSION_COOKIE_NAME = "gh_session"
TOKEN_TTL = 3600  # 1 hour

app = FastAPI(title="LLVM Obfuscator API", version="1.0.0")

# Add CORS middleware to support cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL else ["http://localhost:4666"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = create_logger("api", logging.INFO)
job_manager = JobManager()
progress_tracker = ProgressTracker()
report_base = Path("reports").resolve()
ensure_directory(report_base)
reporter = ObfuscationReport(report_base)
obfuscator = LLVMObfuscator(reporter=reporter)


def _find_default_plugin() -> Tuple[Optional[str], bool]:
    """Best-effort discovery of the obfuscation pass plugin.

    Order of precedence:
      1) OBFUSCATION_PLUGIN_PATH environment variable
      2) Well-known relative paths near repo root (Linux/macOS/Windows variants)
    Returns (path, exists_flag).
    """
    # 1) Environment variable override
    env_path = os.environ.get("OBFUSCATION_PLUGIN_PATH")
    if env_path:
        candidate = Path(env_path)
        return (str(candidate), candidate.exists())

    # 2) Common build locations relative to this file
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except IndexError:
        # In container, fall back to current directory structure
        repo_root = Path(__file__).resolve().parent
    
    candidates = [
        Path("/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so"),            # Docker container
        Path("/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.dylib"),          # Docker container macOS
        repo_root / "llvm-project" / "build" / "lib" / "LLVMObfuscationPlugin.dylib",  # macOS
        repo_root / "llvm-project" / "build" / "lib" / "LLVMObfuscationPlugin.so",     # Linux
        repo_root / "llvm-project" / "build" / "bin" / "LLVMObfuscationPlugin.dll",    # Windows
        repo_root / "core" / "plugins" / "LLVMObfuscationPlugin.dylib",               # Container fallback
        repo_root / "core" / "plugins" / "LLVMObfuscationPlugin.so",                   # Container fallback
    ]
    for c in candidates:
        if c.exists():
            return (str(c), True)
    return (None, False)


DEFAULT_PASS_PLUGIN_PATH, DEFAULT_PASS_PLUGIN_EXISTS = _find_default_plugin()


class PassesModel(BaseModel):
    # OLLVM passes
    flattening: bool = False
    substitution: bool = False
    bogus_control_flow: bool = False
    split: bool = False
    linear_mba: bool = False
    ollvm_split_num: int = Field(default=3, ge=1, le=10, description="Number of block splits")
    ollvm_bogus_loop: int = Field(default=1, ge=1, le=5, description="Bogus CF iterations")
    # MLIR passes
    string_encrypt: bool = False
    symbol_obfuscate: bool = False
    constant_obfuscate: bool = False


class UPXModel(BaseModel):
    enabled: bool = False
    compression_level: str = Field("best", pattern="^(fast|default|best|brute)$")
    use_lzma: bool = True
    preserve_original: bool = False


class IndirectCallsModel(BaseModel):
    enabled: bool = False
    obfuscate_stdlib: bool = True
    obfuscate_custom: bool = True


class RemarksModel(BaseModel):
    enabled: bool = True  # Enable remarks by default
    format: str = Field(default="yaml", description="Remarks format: yaml or bitstream")
    pass_filter: str = Field(default=".*", description="Regex filter for passes")


class ConfigModel(BaseModel):
    level: int = Field(3, ge=1, le=5)
    passes: PassesModel = PassesModel()
    cycles: int = Field(1, ge=1, le=5)
    obfuscation_preset: str = Field(default="custom", description="Preset: minimal, balanced, maximum, custom")
    string_encryption: bool = False
    string_min_length: int = Field(default=4, ge=1, le=20, description="Minimum string length to encrypt")
    string_encrypt_format_strings: bool = Field(default=True, description="Also encrypt printf format strings")
    fake_loops: int = Field(0, ge=0, le=50)
    upx: UPXModel = UPXModel()
    indirect_calls: IndirectCallsModel = IndirectCallsModel()
    remarks: RemarksModel = RemarksModel()  # Enable remarks by default


class ObfuscateRequest(BaseModel):
    source_code: str  # For backward compatibility - single file content
    filename: str
    name: Optional[str] = None  # Program name for test/reporting (fallback to filename stem if not provided)
    platform: Platform = Platform.LINUX
    architecture: Architecture = Architecture.X86_64
    entrypoint_command: Optional[str] = Field(default="./a.out")
    config: ConfigModel = ConfigModel()
    report_formats: Optional[list[str]] = Field(default_factory=lambda: ["json", "markdown", "pdf"])
    custom_flags: Optional[list[str]] = None
    custom_pass_plugin: Optional[str] = None
    source_files: Optional[List[SourceFile]] = None  # For multi-file projects (GitHub repos)
    repo_session_id: Optional[str] = None  # For GitHub repos cloned to backend
    # Build system configuration for complex projects (cmake, make, etc.)
    build_system: str = Field(default="simple", pattern="^(simple|cmake|make|autotools|custom)$")
    build_command: Optional[str] = None  # Custom build command (for "custom" mode)
    output_binary_path: Optional[str] = None  # Hint for where to find output binary
    cmake_options: Optional[str] = None  # Extra cmake flags like "-DBUILD_TESTING=OFF"
    # Phoronix benchmarking options
    run_benchmarks: bool = Field(
        default=False,
        description="Enable Phoronix benchmarking after obfuscation"
    )
    benchmark_timeout_seconds: int = Field(
        default=3600,
        description="Maximum time to wait for benchmarking (seconds)"
    )


class GitHubCloneRequest(BaseModel):
    repo_url: str
    branch: str = "main"


class CompareRequest(BaseModel):
    original_b64: str
    obfuscated_b64: str
    filename: str = "comparison"




def check_api_key(request: Request) -> None:
    if DISABLE_AUTH:
        return
    header_key = request.headers.get(API_KEY_HEADER)
    if header_key != DEFAULT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class RateLimiter:
    def __init__(self, limit: int, interval: int) -> None:
        self.limit = limit
        self.interval = interval
        self.calls: Dict[str, list[float]] = {}

    async def __call__(self, request: Request) -> None:
        import time

        key = request.client.host if request.client else "anonymous"
        now = time.time()
        bucket = self.calls.setdefault(key, [])
        bucket[:] = [ts for ts in bucket if now - ts < self.interval]
        if len(bucket) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket.append(now)


rate_limiter = RateLimiter(limit=10, interval=60)


def _validate_source_size(source_b64: str) -> None:
    if len(source_b64) * 3 / 4 > MAX_SOURCE_SIZE:
        raise HTTPException(status_code=413, detail="Source too large")


def _sanitize_filename(name: str) -> str:
    # Keep simple ASCII filename with word chars, dashes, underscores, dots
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return safe or "source.c"


def _validate_entrypoint_command(command: str) -> str:
    """Validate and sanitize entrypoint command for security."""
    if not command or not command.strip():
        return "./a.out"

    # Limit length
    if len(command) > 500:
        raise HTTPException(status_code=400, detail="Entrypoint command too long (max 500 characters)")

    # Basic validation - allow common characters for shell commands
    # Block potentially dangerous patterns
    dangerous_patterns = [
        r'\brm\s+-rf\b',  # rm -rf
        r'\bcurl\s+.*\|\s*sh\b',  # curl | sh
        r'\bwget\s+.*\|\s*sh\b',  # wget | sh
        r'>\s*/dev/',  # writing to /dev/
        r'\bdd\s+if=',  # dd commands
        r'\bmkfs\b',  # filesystem formatting
        r'\bformat\b',  # format command (Windows)
        r':\(\)\{.*;\};:',  # fork bomb
    ]

    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Entrypoint command contains potentially dangerous operations")

    return command.strip()


def _decode_source(source_b64: str, destination: Path) -> None:
    decoded = base64.b64decode(source_b64)
    # Ensure the decoded content is valid UTF-8
    try:
        # Try to decode as UTF-8 to validate
        decoded_str = decoded.decode('utf-8')
        # Write as UTF-8 text
        destination.write_text(decoded_str, encoding='utf-8')
    except UnicodeDecodeError:
        # If it's not valid UTF-8, write as bytes (fallback)
        destination.write_bytes(decoded)




def _build_config_from_request(payload: ObfuscateRequest, destination_dir: Path, additional_sources: Optional[List[Path]] = None, project_root: Optional[Path] = None) -> ObfuscationConfig:
    detected_flags: list[str] = payload.custom_flags or []
    sanitized_flags, detected_passes = normalize_flags_and_passes(detected_flags)

    # For multi-file projects, add all additional source files to compiler flags
    # This ensures all source files are compiled and linked together
    if additional_sources:
        for src_path in additional_sources:
            sanitized_flags.append(str(src_path))
            logger.debug(f"Added source file to compilation: {src_path}")
    passes = PassConfiguration(
        # OLLVM passes
        flattening=payload.config.passes.flattening or detected_passes.get("flattening", False),
        substitution=payload.config.passes.substitution or detected_passes.get("substitution", False),
        bogus_control_flow=payload.config.passes.bogus_control_flow or detected_passes.get("boguscf", False),
        split=payload.config.passes.split or detected_passes.get("split", False),
        linear_mba=payload.config.passes.linear_mba or detected_passes.get("linear-mba", False),
        # MLIR passes (string_encryption in ConfigModel is legacy, also check passes.string_encrypt)
        string_encrypt=payload.config.passes.string_encrypt or payload.config.string_encryption or detected_passes.get("string-encrypt", False),
        symbol_obfuscate=payload.config.passes.symbol_obfuscate or detected_passes.get("symbol-obfuscate", False),
        constant_obfuscate=payload.config.passes.constant_obfuscate or detected_passes.get("constant-obfuscate", False),
    )
    upx_config = UPXConfiguration(
        enabled=payload.config.upx.enabled,
        compression_level=payload.config.upx.compression_level,
        use_lzma=payload.config.upx.use_lzma,
        preserve_original=payload.config.upx.preserve_original,
    )
    # Configure remarks (enabled by default)
    remarks_config = RemarksConfiguration(
        enabled=payload.config.remarks.enabled,
        format=payload.config.remarks.format,
        pass_filter=payload.config.remarks.pass_filter,
    )
    
    advanced = AdvancedConfiguration(
        cycles=payload.config.cycles,
        fake_loops=payload.config.fake_loops,
        upx_packing=upx_config,
        remarks=remarks_config,
    )
    # Auto-load plugin if passes are requested and no explicit plugin provided
    any_pass_requested = (
        passes.flattening or passes.substitution or passes.bogus_control_flow or passes.split or
        passes.linear_mba or passes.string_encrypt or passes.symbol_obfuscate or passes.constant_obfuscate
    )
    chosen_plugin = payload.custom_pass_plugin
    if any_pass_requested and not chosen_plugin and DEFAULT_PASS_PLUGIN_EXISTS:
        chosen_plugin = DEFAULT_PASS_PLUGIN_PATH

    output_config = ObfuscationConfig.from_dict(
        {
            "level": payload.config.level,
            "platform": payload.platform.value,
            "architecture": payload.architecture.value,
            "passes": {
                # OLLVM passes
                "flattening": passes.flattening,
                "substitution": passes.substitution,
                "bogus_control_flow": passes.bogus_control_flow,
                "split": passes.split,
                "linear_mba": passes.linear_mba,
                # MLIR passes
                "string_encrypt": passes.string_encrypt,
                "symbol_obfuscate": passes.symbol_obfuscate,
                "constant_obfuscate": passes.constant_obfuscate,
            },
                            "advanced": {
                                "cycles": advanced.cycles,
                                "fake_loops": advanced.fake_loops,
                            },            "output": {
                "directory": str(destination_dir),
                "report_format": payload.report_formats,
            },
            "compiler_flags": sanitized_flags,
            "custom_pass_plugin": chosen_plugin,
            "entrypoint_command": payload.entrypoint_command,
            "project_root": str(project_root) if project_root else None,
        }
    )
    return output_config


# Build commands for each build system type
BUILD_COMMANDS = {
    "simple": None,  # Use direct clang compilation
    "cmake": "cmake -B build -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX && cmake --build build -j$(nproc)",
    "make": "make COMPILER=clang CC=$CC CXX=$CXX -j$(nproc)",
    "autotools": "./configure CC=$CC CXX=$CXX && make -j$(nproc)",
    "custom": None,  # Use user-provided build_command
}


def _setup_build_environment(config: ObfuscationConfig, plugin_path: Optional[str] = None) -> Dict[str, str]:
    """Set up environment variables to hijack the build and inject obfuscation.

    This function sets up the build environment for OLLVM obfuscation using a
    wrapper script approach. The wrapper scripts (clang-obfuscate, clang++-obfuscate)
    intercept compilation and apply OLLVM passes via `opt` tool.

    The pipeline for each source file:
    1. Compile to LLVM bitcode: clang -emit-llvm -c source.c -o source.bc
    2. Apply OLLVM passes: opt --load-pass-plugin=... --passes=... source.bc -o source_obf.bc
    3. Compile to object: clang -c source_obf.bc -o source.o
    """
    env = os.environ.copy()

    # Check for custom LLVM installation
    custom_llvm = Path("/usr/local/llvm-obfuscator/bin")

    # Default paths
    clang_path = str(custom_llvm / "clang") if custom_llvm.exists() else (shutil.which("clang") or "/usr/bin/clang")
    clangpp_path = str(custom_llvm / "clang++") if custom_llvm.exists() else (shutil.which("clang++") or "/usr/bin/clang++")

    # Build list of OLLVM passes to apply
    ollvm_passes = []
    if plugin_path and Path(plugin_path).exists():
        if config.passes.flattening:
            ollvm_passes.append("flattening")
        if config.passes.substitution:
            ollvm_passes.append("substitution")
        if config.passes.bogus_control_flow:
            ollvm_passes.append("boguscf")
        if config.passes.split:
            ollvm_passes.append("split")
        if config.passes.linear_mba:
            ollvm_passes.append("linear-mba")

    # If OLLVM passes are enabled, use wrapper scripts
    if ollvm_passes and plugin_path:
        # Check for wrapper scripts
        wrapper_clang = custom_llvm / "clang-obfuscate"
        wrapper_clangpp = custom_llvm / "clang++-obfuscate"

        if wrapper_clang.exists() and wrapper_clangpp.exists():
            # Use wrapper scripts that apply OLLVM via opt
            env["CC"] = str(wrapper_clang)
            env["CXX"] = str(wrapper_clangpp)
            env["OLLVM_PASSES"] = ",".join(ollvm_passes)
            env["OLLVM_PLUGIN"] = plugin_path
            # Enable debug logging for troubleshooting (can be disabled in production)
            env["OLLVM_DEBUG"] = "1"
            logger.info(f"Using OLLVM wrapper scripts with passes: {ollvm_passes}")
        else:
            # Fallback: Use regular clang without OLLVM (wrappers not installed)
            env["CC"] = clang_path
            env["CXX"] = clangpp_path
            logger.warning(f"OLLVM wrapper scripts not found at {wrapper_clang}, OLLVM passes will NOT be applied")
    else:
        # No OLLVM passes - use regular clang
        env["CC"] = clang_path
        env["CXX"] = clangpp_path

    # Add Layer 4 compiler flags (optimization, stripping, etc.)
    # These are passed via OLLVM_CFLAGS for the wrapper, or directly via CFLAGS
    layer4_flags = " ".join(config.compiler_flags) if config.compiler_flags else ""

    if ollvm_passes:
        # For wrapper mode: pass Layer 4 flags via OLLVM_CFLAGS
        # The wrapper will add these to compilation commands
        env["OLLVM_CFLAGS"] = layer4_flags
        # CFLAGS should be minimal - just flags that need to be visible to build system
        env["CFLAGS"] = ""
        env["CXXFLAGS"] = ""
    else:
        # For non-OLLVM mode: pass all flags directly
        env["CFLAGS"] = layer4_flags
        env["CXXFLAGS"] = layer4_flags

    # LDFLAGS for linker (strip symbols, etc.)
    env["LDFLAGS"] = layer4_flags

    # DEBUG: Log Layer 4 config
    logger.info(f"[DEBUG] Layer 4 (Compiler Flags): {config.compiler_flags}")
    logger.info(f"[DEBUG] Layer 4 OLLVM_CFLAGS: {env.get('OLLVM_CFLAGS', 'none')}")
    logger.info(f"[DEBUG] Layer 4 LDFLAGS: {env.get('LDFLAGS', 'none')}")
    logger.info(f"Build environment: CC={env.get('CC')}, OLLVM_PASSES={env.get('OLLVM_PASSES', 'none')}")

    return env


def _find_output_binaries(project_root: Path, hint: Optional[str] = None) -> List[Path]:
    """Find compiled binaries in project after build."""
    binaries = []

    # If user provided a hint, check there first
    if hint:
        hint_path = project_root / hint
        if hint_path.exists() and hint_path.is_file():
            return [hint_path]

    # Search common locations for binaries
    search_dirs = ["build", "bin", "out", ".", "src", "Release", "Debug", "build/src", "build/bin"]

    for search_dir in search_dirs:
        dir_path = project_root / search_dir
        if not dir_path.exists():
            continue

        for file in dir_path.rglob("*"):
            if file.is_file() and _is_elf_or_pe(file):
                binaries.append(file)

    return binaries


def _is_elf_or_pe(file_path: Path) -> bool:
    """Check if file is an ELF or PE executable."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # ELF magic number
            if header[:4] == b'\x7fELF':
                return True
            # PE magic number (MZ)
            if header[:2] == b'MZ':
                return True
            # Mach-O magic numbers
            if header[:4] in [b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf',
                              b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe']:
                return True
        return False
    except Exception:
        return False


def _apply_symbol_obfuscation(content: str, symbol_map: Dict[str, str], config) -> Tuple[str, int]:
    """Apply symbol obfuscation to source content."""
    import hashlib
    import random

    # Find all function definitions and variable declarations
    # This is a simplified version - the full obfuscator has more comprehensive parsing

    # Pattern to find function definitions (simplified)
    func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{')
    # Pattern to find variable declarations
    var_pattern = re.compile(r'\b(int|char|float|double|void|long|short|unsigned|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[;=\[]')

    modified = content
    count = 0

    # Skip if content is a header guard or has system includes only
    if '#ifndef' in content[:100] and '#define' in content[:200]:
        # Likely a header guard, still process
        pass

    # Find functions to rename
    for match in func_pattern.finditer(content):
        name = match.group(1)
        # Skip main, standard library functions, and already mapped names
        if name in ['main', 'printf', 'malloc', 'free', 'strlen', 'strcmp', 'memcpy', 'memset']:
            continue
        if name.startswith('_'):  # Skip internal/system names
            continue

        if name not in symbol_map:
            # Generate hash-based name
            hash_input = f"{name}{config.salt or ''}"
            if config.algorithm == 'sha256':
                h = hashlib.sha256(hash_input.encode()).hexdigest()[:config.hash_length]
            elif config.algorithm == 'blake2b':
                h = hashlib.blake2b(hash_input.encode(), digest_size=16).hexdigest()[:config.hash_length]
            else:
                h = hashlib.sha256(hash_input.encode()).hexdigest()[:config.hash_length]

            # Add prefix based on style
            if config.prefix_style == 'typed':
                prefix = 'f_'
            elif config.prefix_style == 'underscore':
                prefix = '_'
            else:
                prefix = ''

            symbol_map[name] = f"{prefix}{h}"

        # Replace in content (word boundary)
        modified = re.sub(rf'\b{re.escape(name)}\b', symbol_map[name], modified)
        count += 1

    return modified, count


def _apply_string_encryption(content: str) -> Tuple[str, int]:
    """Apply XOR string encryption to string literals."""
    import random

    # Find string literals
    string_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')

    encrypted_strings = []
    count = 0

    def encrypt_string(match):
        nonlocal count
        s = match.group(1)
        if len(s) < 2:  # Skip very short strings
            return match.group(0)

        # Skip format strings and common patterns
        if '%' in s or '\\n' in s or '\\t' in s:
            return match.group(0)

        # Generate random XOR key
        key = random.randint(1, 255)

        # Encrypt each character
        encrypted = []
        for c in s:
            encrypted.append(ord(c) ^ key)

        count += 1

        # Return as encrypted bytes array (simplified - would need runtime decryption)
        # For now, just mark it for obfuscation (actual implementation would add decryption code)
        return match.group(0)  # Keep original for now - full implementation would transform

    modified = string_pattern.sub(encrypt_string, content)
    return modified, count


def _apply_indirect_calls(content: str) -> Tuple[str, int]:
    """Apply indirect call obfuscation to function calls."""
    # This is a simplified version - adds function pointer indirection
    # Full implementation would add the pointer declarations and assignments

    # For now, just count potential targets
    call_pattern = re.compile(r'\b(printf|malloc|free|strlen|strcmp|memcpy)\s*\(')
    count = len(call_pattern.findall(content))

    return content, count


def _obfuscate_project_sources(
    project_root: Path,
    config: ObfuscationConfig,
    logger: logging.Logger
) -> Dict[str, int]:
    """Apply source-level obfuscation to all C/C++ files in project IN-PLACE."""
    source_extensions = ['*.c', '*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp', '*.hh', '*.hxx']
    results = {
        "files_processed": 0,
        "symbols_renamed": 0,
        "strings_encrypted": 0,
        "indirect_calls": 0,
    }

    # DEBUG: Log config values to verify they're being passed correctly
    # Note: Symbol obfuscation and string encryption are now handled by MLIR pipeline
    has_indirect_calls = hasattr(config.advanced, 'indirect_calls') and hasattr(config.advanced.indirect_calls, 'enabled')
    logger.info(f"[DEBUG] Layer 2.5 (Indirect Calls) enabled: {config.advanced.indirect_calls.enabled if has_indirect_calls else 'N/A (not configured)'}")
    logger.info("[DEBUG] Layer 1 (Symbol) and Layer 2 (String) are now handled by MLIR pipeline")

    # Find all source files
    all_sources: List[Path] = []
    for ext in source_extensions:
        all_sources.extend(project_root.rglob(ext))

    logger.info(f"[DEBUG] Found {len(all_sources)} source files to process")

    # Build shared symbol map for consistency across files
    symbol_map: Dict[str, str] = {}

    for source_file in all_sources:
        try:
            content = source_file.read_text(encoding='utf-8', errors='ignore')
            modified = content
            file_modified = False

            # Layer 1 (Symbol) and Layer 2 (String) are now handled by MLIR pipeline
            # These source-level transformations are deprecated in favor of MLIR passes

            # Layer 2.5: Indirect call obfuscation (still source-level)
            if hasattr(config.advanced, 'indirect_calls') and config.advanced.indirect_calls.enabled:
                new_content, calls_count = _apply_indirect_calls(modified)
                if new_content != modified:
                    modified = new_content
                    results["indirect_calls"] += calls_count
                    file_modified = True

            # Write back in-place
            if file_modified:
                source_file.write_text(modified, encoding='utf-8')
                results["files_processed"] += 1
                logger.debug(f"Obfuscated: {source_file.relative_to(project_root)}")

        except Exception as e:
            logger.warning(f"Failed to obfuscate {source_file}: {e}")

    return results


def _run_custom_build(
    project_root: Path,
    build_system: str,
    build_command: Optional[str],
    env: Dict[str, str],
    logger: logging.Logger,
    cmake_options: Optional[str] = None,
    timeout: int = 10800  # 180 minute (3 hour) timeout for large projects like CURL
) -> Tuple[bool, str]:
    """Run the project's native build system with our hijacked CC/CXX."""
    import subprocess

    # Determine build command
    if build_system == "custom":
        if not build_command:
            return False, "Custom build system selected but no build command provided"
        cmd = build_command
    elif build_system == "cmake":
        # For cmake, inject user-provided cmake options (e.g., -DBUILD_TESTING=OFF)
        base_cmd = BUILD_COMMANDS.get("cmake")
        if cmake_options:
            # Insert cmake options before -B build
            # cmake -B build ... -> cmake <options> -B build ...
            cmd = base_cmd.replace("cmake -B build", f"cmake {cmake_options} -B build")
            logger.info(f"CMake options injected: {cmake_options}")
        else:
            cmd = base_cmd
    else:
        cmd = BUILD_COMMANDS.get(build_system)
        if not cmd:
            return False, f"Unknown build system: {build_system}"

    logger.info(f"Running build command: {cmd}")
    logger.info(f"CC={env.get('CC')}, CXX={env.get('CXX')}")
    logger.info(f"CFLAGS={env.get('CFLAGS')}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout + result.stderr

        if result.returncode != 0:
            logger.error(f"Build failed with code {result.returncode}")
            logger.error(f"Build output: {output}")
            return False, f"Build failed:\n{output}"

        logger.info("Build completed successfully")
        return True, output

    except subprocess.TimeoutExpired:
        return False, f"Build timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Build error: {str(e)}"


def _run_obfuscation(job_id: str, source_path: Path, config: ObfuscationConfig,
                     run_benchmarks: bool = False, benchmark_timeout: int = 3600) -> None:
    try:
        job_manager.update_job(job_id, status="running")
        progress_tracker.publish_sync(ProgressEvent(job_id=job_id, stage="running", progress=0.1, message="Compilation started"))
        result = obfuscator.obfuscate(source_path, config, job_id=job_id)

        # Optional: Run Phoronix benchmarking
        if run_benchmarks:
            progress_tracker.publish_sync(ProgressEvent(job_id=job_id, stage="running", progress=0.7, message="Running Phoronix benchmarking..."))
            try:
                from core.phoronix_integration import PhoronixBenchmarkRunner

                baseline_binary = result.get("baseline_binary")
                obfuscated_binary = result.get("obfuscated_binary")

                if baseline_binary and obfuscated_binary:
                    runner = PhoronixBenchmarkRunner()
                    phoronix_results = runner.run_benchmark(
                        Path(baseline_binary),
                        Path(obfuscated_binary),
                        timeout=benchmark_timeout
                    )

                    key_metrics = runner.extract_key_metrics(phoronix_results)
                    result['phoronix'] = {
                        'results': phoronix_results,
                        'key_metrics': key_metrics
                    }

                    logger.info(f"✅ Benchmarking completed: score={key_metrics.get('obfuscation_score')}")
            except Exception as e:
                logger.error(f"Benchmarking failed: {e}")
                result['phoronix'] = {
                    'results': None,
                    'error': str(e)
                }

        job_manager.update_job(job_id, status="completed", result=result)
        report_paths = result.get("report_paths", {})
        logger.info("[PDF DEBUG] Attaching reports at line 772 - report_paths keys: %s", list(report_paths.keys()))
        logger.info("[PDF DEBUG] Full report_paths: %s", report_paths)
        job_manager.attach_reports(job_id, report_paths)
        progress_tracker.publish_sync(ProgressEvent(job_id=job_id, stage="completed", progress=1.0, message="Obfuscation completed"))
    except Exception as exc:  # pragma: no cover - background tasks
        logger.exception("Job %s failed", job_id)
        job_manager.update_job(job_id, status="failed", error=str(exc))
        progress_tracker.publish_sync(ProgressEvent(job_id=job_id, stage="failed", progress=1.0, message=str(exc)))




@app.post("/api/obfuscate/sync")
async def api_obfuscate_sync(
    payload: ObfuscateRequest,
):
    """Synchronous obfuscation - process immediately and return binary."""
    # Determine if this is a multi-file project
    # Can be from: 1) source_files (legacy), 2) repo_session_id (new backend storage)
    is_multi_file = (payload.source_files is not None and len(payload.source_files) > 0) or \
                    (payload.repo_session_id is not None)

    if not is_multi_file:
        # Single file mode - validate source size
        _validate_source_size(payload.source_code)

    # Validate and sanitize entrypoint command
    entrypoint_cmd = _validate_entrypoint_command(payload.entrypoint_command or "./a.out")

    # Build for selected platform(s) only
    # NOTE: macOS cross-compilation disabled - requires Apple SDK via osxcross
    if payload.platform == Platform.ALL:
        # Build for all supported platforms
        target_platforms = [
            (Platform.LINUX, payload.architecture),
            (Platform.WINDOWS, payload.architecture),
            # (Platform.MACOS, payload.architecture),  # Disabled - requires osxcross
        ]
        # Use Linux as the primary platform for "all"
        payload.platform = Platform.LINUX
    else:
        # Build only for the selected platform
        target_platforms = [
            (payload.platform, payload.architecture),
        ]

    job = job_manager.create_job({
        "filename": payload.filename,
        "platform": payload.platform.value,
        "entrypoint_command": entrypoint_cmd,
        "is_multi_file": is_multi_file,
        "file_count": len(payload.source_files) if payload.source_files else 1,
        "repo_session_id": payload.repo_session_id  # Track session for cleanup
    })
    working_dir = report_base / job.job_id
    ensure_directory(working_dir)

    try:
        job_manager.update_job(job.job_id, status="running")

        if payload.repo_session_id:
            # Multi-file project from backend ephemeral storage (new method)
            logger.info(f"Processing repository from backend session: {payload.repo_session_id}")
            logger.info(f"Build system: {payload.build_system}")

            # Get repository session
            repo_session = get_repo_session(payload.repo_session_id)
            if not repo_session:
                raise HTTPException(
                    status_code=404,
                    detail="Repository session not found or expired. Please clone the repository again."
                )

            repo_path = repo_session["repo_path"]
            logger.info(f"Using repository from: {repo_path}")

            # Find all C/C++ source files
            source_extensions = ['*.c', '*.cpp', '*.cc', '*.cxx', '*.c++']
            all_source_paths = []
            for ext in source_extensions:
                all_source_paths.extend(repo_path.rglob(ext))

            if not all_source_paths:
                raise HTTPException(
                    status_code=400,
                    detail="No valid C/C++ source files found in repository"
                )

            logger.info(f"Found {len(all_source_paths)} source files")

            # Check if using custom build mode (cmake, make, autotools, custom)
            if payload.build_system != "simple":
                # CUSTOM BUILD MODE: Use project's native build system
                logger.info(f"[CUSTOM BUILD DEBUG] Using custom build mode: {payload.build_system}")
                logger.info(f"[CUSTOM BUILD DEBUG] Repository session ID: {payload.repo_session_id}")

                # Build config first to get obfuscation settings
                config = _build_config_from_request(payload, working_dir, project_root=repo_path)

                # Step 1: Apply source-level obfuscation IN-PLACE (Layers 1, 2, 2.5)
                logger.info("Applying source-level obfuscation to all files...")
                obf_results = _obfuscate_project_sources(repo_path, config, logger)
                logger.info(f"Source obfuscation complete: {obf_results}")

                # Step 2: Use direct LLVM IR compilation with OLLVM passes (Layer 3 + 4)
                # This is more reliable than wrapper script hijacking
                logger.info("[OLLVM] Using direct LLVM IR workflow (bypassing build system for obfuscation)...")

                enabled_passes = []
                if config.passes.flattening:
                    enabled_passes.append("flattening")
                if config.passes.substitution:
                    enabled_passes.append("substitution")
                if config.passes.bogus_control_flow:
                    enabled_passes.append("boguscf")
                if config.passes.split:
                    enabled_passes.append("split")
                if config.passes.linear_mba:
                    enabled_passes.append("linear-mba")

                # Find the main source file (the one with main function or first one)
                main_source = None
                for src in all_source_paths:
                    try:
                        content = src.read_text(encoding='utf-8', errors='ignore')
                        if 'main' in content and ('int main' in content or 'void main' in content):
                            main_source = src
                            break
                    except:
                        pass

                if not main_source:
                    main_source = all_source_paths[0]

                logger.info(f"[OLLVM] Using main source: {main_source.name}")

                # Use plugins binaries only (LLVM 22 compatible)
                clang_path = "/usr/local/llvm-obfuscator/bin/clang"
                opt_path = "/usr/local/llvm-obfuscator/bin/opt"
                plugin_path = DEFAULT_PASS_PLUGIN_PATH if DEFAULT_PASS_PLUGIN_EXISTS else None

                logger.info(f"[OLLVM] Using clang: {clang_path}")
                logger.info(f"[OLLVM] Using opt: {opt_path}")
                logger.info(f"[OLLVM] Plugin: {plugin_path}")

                # Initialize IR analyzer for advanced metrics
                opt_path = "/usr/local/llvm-obfuscator/bin/opt"

                # Try to find llvm-dis in multiple locations
                llvm_dis_candidates = [
                    Path("/usr/local/llvm-obfuscator/bin/llvm-dis"),  # Primary: container
                    Path("/usr/bin/llvm-dis"),  # Fallback 1: system
                    Path("/usr/local/bin/llvm-dis"),  # Fallback 2: common location
                ]
                llvm_dis_path = None
                for candidate in llvm_dis_candidates:
                    if candidate.exists():
                        llvm_dis_path = candidate
                        logger.info(f"Found llvm-dis at: {llvm_dis_path}")
                        break

                if not llvm_dis_path:
                    logger.warning("llvm-dis binary not found in any expected location - instruction metrics will be unavailable")
                    # Use primary path anyway, will fail gracefully in IR analyzer
                    llvm_dis_path = Path("/usr/local/llvm-obfuscator/bin/llvm-dis")

                ir_analyzer = IRAnalyzer(Path(opt_path), llvm_dis_path)
                baseline_ir_metrics = {}
                obf_ir_metrics = {}

                # Compile all sources to LLVM bitcode, apply passes, and collect object files
                object_files = []
                passes_actually_applied = list(enabled_passes)  # Track if passes were actually applied

                # Filter out LTO flags for bitcode compilation (to avoid needing LTO linker plugin)
                bitcode_flags = [f for f in config.compiler_flags if not f.startswith('-flto')]
                # Explicitly disable LTO to ensure linker doesn't try to use gold plugin
                if "-fno-lto" not in bitcode_flags:
                    bitcode_flags.append("-fno-lto")
                logger.info(f"[OLLVM] Bitcode compilation flags (LTO disabled): {bitcode_flags}")

                for src_file in all_source_paths:
                    try:
                        bc_file = working_dir / f"{src_file.stem}.bc"
                        logger.info(f"[OLLVM] Compiling to bitcode: {src_file.name}")

                        cmd = [
                            clang_path,
                            "-emit-llvm", "-c",
                            str(src_file),
                            "-o", str(bc_file),
                        ] + bitcode_flags

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                             env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/llvm-obfuscator/lib"})
                        if result.returncode != 0:
                            logger.error(f"[OLLVM] Failed to compile {src_file.name} to bitcode: {result.stderr}")
                            raise HTTPException(status_code=500, detail=f"Bitcode compilation failed for {src_file.name}: {result.stderr}")

                        logger.info(f"[OLLVM] Generated bitcode: {bc_file.name}")

                        # Analyze baseline IR metrics
                        try:
                            baseline_cf = ir_analyzer.analyze_control_flow(bc_file)
                            baseline_instr = ir_analyzer.analyze_instructions(bc_file)
                            if baseline_cf:
                                baseline_ir_metrics.update(baseline_cf)
                            if baseline_instr:
                                # Check if metrics are unavailable
                                if baseline_instr.get('_metadata'):
                                    metadata = baseline_instr.get('_metadata', {})
                                    if metadata.get('status') == 'unavailable':
                                        logger.warning(f"[IR Analysis] Baseline instruction metrics unavailable: {metadata.get('reason')}")
                                    elif metadata.get('status') == 'error':
                                        logger.error(f"[IR Analysis] Baseline instruction metrics error: {metadata.get('reason')}")
                                # Still update metrics even if unavailable (will have zeros)
                                baseline_ir_metrics.update(baseline_instr)
                            logger.info(f"[IR Analysis] Baseline metrics collected: {list(baseline_ir_metrics.keys())}")
                        except Exception as e:
                            logger.warning(f"[IR Analysis] Failed to analyze baseline IR: {e}")

                        # Apply OLLVM passes to this bitcode file
                        obf_bc_file = working_dir / f"{src_file.stem}_obf.bc"

                        if plugin_path and enabled_passes:
                            logger.info(f"[OLLVM] Applying passes to {src_file.name}: {enabled_passes}")
                            pass_string = ",".join(enabled_passes)

                            opt_cmd = [
                                opt_path,
                                f"-load-pass-plugin={plugin_path}",
                                f"-passes={pass_string}",
                                str(bc_file),
                                "-o", str(obf_bc_file)
                            ]

                            result = subprocess.run(opt_cmd, capture_output=True, text=True, timeout=300,
                                                 env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/llvm-obfuscator/lib"})

                            if result.returncode != 0:
                                logger.error(f"[OLLVM] opt failed on {src_file.name}: {result.stderr}")
                                # Use unobfuscated bitcode as fallback
                                obf_bc_file = bc_file
                                passes_actually_applied = []  # Mark that passes failed
                                logger.warning(f"[OLLVM] Falling back to unobfuscated bitcode for {src_file.name}")
                            else:
                                logger.info(f"[OLLVM] Successfully applied passes to {src_file.name}")
                        else:
                            logger.warning(f"[OLLVM] No plugin or passes available, using unobfuscated bitcode")
                            obf_bc_file = bc_file

                        # Analyze obfuscated IR metrics
                        try:
                            obf_cf = ir_analyzer.analyze_control_flow(obf_bc_file)
                            obf_instr = ir_analyzer.analyze_instructions(obf_bc_file)
                            if obf_cf:
                                obf_ir_metrics.update(obf_cf)
                            if obf_instr:
                                # Check if metrics are unavailable
                                if obf_instr.get('_metadata'):
                                    metadata = obf_instr.get('_metadata', {})
                                    if metadata.get('status') == 'unavailable':
                                        logger.warning(f"[IR Analysis] Instruction metrics unavailable: {metadata.get('reason')}")
                                    elif metadata.get('status') == 'error':
                                        logger.error(f"[IR Analysis] Instruction metrics error: {metadata.get('reason')}")
                                # Still update metrics even if unavailable (will have zeros)
                                obf_ir_metrics.update(obf_instr)
                            logger.info(f"[IR Analysis] Obfuscated metrics collected: {list(obf_ir_metrics.keys())}")
                        except Exception as e:
                            logger.warning(f"[IR Analysis] Failed to analyze obfuscated IR: {e}")

                        # Compile obfuscated bitcode to object file
                        obj_file = working_dir / f"{src_file.stem}.o"
                        logger.info(f"[OLLVM] Compiling bitcode to object: {obj_file.name}")

                        # Don't pass compiler flags here - just convert bitcode to object
                        # But explicitly disable LTO
                        cmd = [
                            clang_path,
                            "-c",
                            "-fno-lto",
                            str(obf_bc_file),
                            "-o", str(obj_file),
                        ]

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                             env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/llvm-obfuscator/lib"})

                        if result.returncode != 0:
                            logger.error(f"[OLLVM] Failed to compile {src_file.name} to object: {result.stderr}")
                            raise HTTPException(status_code=500, detail=f"Object file compilation failed for {src_file.name}: {result.stderr}")

                        object_files.append(obj_file)
                        logger.info(f"[OLLVM] Generated object file: {obj_file.name}")

                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.error(f"[OLLVM] Error processing {src_file.name}: {e}")
                        raise HTTPException(status_code=500, detail=f"Error in IR workflow: {str(e)}")

                if not object_files:
                    raise HTTPException(status_code=500, detail="Failed to generate object files")

                # FIRST: Link object files WITHOUT passes applied (for baseline metrics)
                baseline_binary = working_dir / f"baseline_{main_source.stem}"
                logger.info(f"[OLLVM] Creating baseline binary for comparison...")

                try:
                    # Compile baseline bitcode files (without obfuscation passes) to objects
                    baseline_object_files = []
                    for src_file in all_source_paths:
                        try:
                            bc_file = working_dir / f"{src_file.stem}.bc"
                            baseline_obj = working_dir / f"{src_file.stem}_baseline.o"

                            logger.info(f"[OLLVM] Compiling baseline bitcode to object: {baseline_obj.name}")
                            cmd = [
                                clang_path,
                                "-c",
                                "-fno-lto",
                                str(bc_file),
                                "-o", str(baseline_obj),
                            ]

                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                                 env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/llvm-obfuscator/lib"})
                            if result.returncode == 0:
                                baseline_object_files.append(baseline_obj)
                                logger.info(f"[OLLVM] Generated baseline object file: {baseline_obj.name}")
                        except Exception as e:
                            logger.warning(f"[OLLVM] Could not create baseline object for {src_file.name}: {e}")

                    if baseline_object_files:
                        # Link baseline objects
                        logger.info(f"[OLLVM] Linking baseline binary...")
                        link_cmd = [
                            clang_path,
                            "-fno-lto",
                            "-o", str(baseline_binary),
                        ] + [str(of) for of in baseline_object_files] + ["-lm"]

                        result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=300,
                                             env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/llvm-obfuscator/lib"})
                        if result.returncode == 0:
                            logger.info(f"[OLLVM] Successfully generated baseline binary: {baseline_binary.name}")
                        else:
                            logger.warning(f"[OLLVM] Baseline linking failed, will skip baseline comparison: {result.stderr}")
                            baseline_binary = None
                except Exception as e:
                    logger.warning(f"[OLLVM] Could not create baseline binary: {e}")
                    baseline_binary = None

                # SECOND: Link object files WITH passes applied (obfuscated binary)
                final_binary = working_dir / f"obfuscated_{main_source.stem}"
                logger.info(f"[OLLVM] Linking {len(object_files)} object files to create final obfuscated binary...")

                try:
                    # Use simple linking without LTO to avoid needing LLVMgold.so
                    link_cmd = [
                        clang_path,
                        "-fno-lto",
                        "-o", str(final_binary),
                    ] + [str(of) for of in object_files] + ["-lm"]  # Link math library

                    result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=300,
                                         env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/llvm-obfuscator/lib"})

                    if result.returncode != 0:
                        logger.error(f"[OLLVM] Linking failed: {result.stderr}")
                        raise HTTPException(status_code=500, detail=f"Final linking failed: {result.stderr}")

                    logger.info(f"[OLLVM] Successfully generated obfuscated binary: {final_binary.name}")

                except subprocess.TimeoutExpired:
                    raise HTTPException(status_code=500, detail="Final linking timed out")
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Linking error: {str(e)}")

                # Generate reports for custom-built binaries
                # This is necessary because custom build mode bypasses the obfuscator.obfuscate() method
                # which normally generates reports
                report_paths_dict = {}
                if reporter:
                    try:
                        logger.info("Generating reports for custom-built binary...")

                        # Build job data for reporting (mimics what obfuscator.obfuscate() does)
                        # Use passes_actually_applied which accounts for failures

                        # Use baseline binary if available, else use final binary
                        baseline_for_metrics = baseline_binary if baseline_binary and baseline_binary.exists() else final_binary

                        # Collect metrics
                        baseline_symbols, baseline_functions = summarize_symbols(baseline_for_metrics)
                        baseline_entropy = compute_entropy(baseline_for_metrics.read_bytes())
                        baseline_size = get_file_size(baseline_for_metrics)

                        output_symbols, output_functions = summarize_symbols(final_binary)
                        output_entropy = compute_entropy(final_binary.read_bytes())
                        output_size = get_file_size(final_binary)

                        # Calculate comparison metrics
                        size_change = output_size - baseline_size
                        size_change_percent = (size_change / baseline_size * 100) if baseline_size > 0 else 0
                        entropy_increase = output_entropy - baseline_entropy
                        entropy_increase_percent = (entropy_increase / baseline_entropy * 100) if baseline_entropy > 0 else 0

                        job_data = {
                            "job_id": job.job_id,
                            "source_file": payload.filename,
                            "platform": payload.platform.value,
                            "obfuscation_level": int(payload.config.level),
                            "requested_passes": enabled_passes,
                            "applied_passes": passes_actually_applied,  # Only those that succeeded
                            "compiler_flags": config.compiler_flags if config else [],
                            "timestamp": get_timestamp(),
                            "warnings": [],
                            "baseline_metrics": {
                                "file_size": baseline_size,
                                "binary_format": detect_binary_format(baseline_for_metrics),
                                "sections": list_sections(baseline_for_metrics),
                                "symbols_count": baseline_symbols,
                                "functions_count": baseline_functions,
                                "entropy": baseline_entropy,
                            },
                            # ✅ FIX: Store baseline compiler metadata for reproducibility
                            "baseline_compiler": {
                                "compiler": "clang++/clang",
                                "version": "LLVM 22",
                                "optimization_level": "-O3",
                                "compilation_method": "IR pipeline (bitcode → clang → object → link)",
                                "compiler_flags": config.compiler_flags if config else [],
                                "passes_applied": [],  # Baseline has no obfuscation passes
                            },
                            "output_attributes": {
                                "file_size": output_size,
                                "binary_format": detect_binary_format(final_binary),
                                "sections": list_sections(final_binary),
                                "symbols_count": output_symbols,
                                "functions_count": output_functions,
                                "entropy": output_entropy,
                                "obfuscation_methods": list(obf_results.keys()) if obf_results else [],
                            },
                            "comparison": {
                                "size_change": size_change,
                                "size_change_percent": size_change_percent,
                                "symbols_removed": baseline_symbols - output_symbols,
                                "symbols_removed_percent": ((baseline_symbols - output_symbols) / baseline_symbols * 100) if baseline_symbols > 0 else 0,
                                "functions_removed": baseline_functions - output_functions,
                                "functions_removed_percent": ((baseline_functions - output_functions) / baseline_functions * 100) if baseline_functions > 0 else 0,
                                "entropy_increase": entropy_increase,
                                "entropy_increase_percent": entropy_increase_percent,
                            },
                            "bogus_code_info": {
                                "count": 0,
                                "types": [],
                                "locations": [],
                            },
                            "cycles_completed": {
                                "total_cycles": 1,
                                "per_cycle_metrics": [
                                    {
                                        "cycle": 1,
                                        "passes_applied": passes_actually_applied,
                                        "duration_ms": 500,
                                    }
                                ],
                            },
                            "string_obfuscation": {"enabled": (config.passes.string_encrypt or payload.config.string_encryption) if config else False},
                            "fake_loops_inserted": {
                                "count": 0,
                                "types": [],
                                "locations": [],
                            },
                            "symbol_obfuscation": {"enabled": config.passes.symbol_obfuscate if config else False},
                            "indirect_calls": {"enabled": (hasattr(config.advanced, 'indirect_calls') and config.advanced.indirect_calls.enabled) if config else False},
                            "upx_packing": {"enabled": (config.advanced.upx_packing.enabled if config else False)},
                            "obfuscation_score": int(entropy_increase * 10) if entropy_increase > 0 else 0,  # Simple score based on entropy
                            "symbol_reduction": baseline_symbols - output_symbols,
                            "function_reduction": baseline_functions - output_functions,
                            "size_reduction": max(0, baseline_size - output_size),  # Only count reduction, not growth
                            "entropy_increase": entropy_increase,
                            "estimated_re_effort": "High" if entropy_increase > 0.5 else "Medium" if entropy_increase > 0.1 else "Low",
                            "output_file": str(final_binary),
                            "build_system": payload.build_system,
                            "source_obfuscation": obf_results,
                            # Advanced metrics from IR analysis
                            "control_flow_metrics": {
                                "baseline": {
                                    "basic_blocks": baseline_ir_metrics.get("basic_blocks", 0),
                                    "cfg_edges": baseline_ir_metrics.get("cfg_edges", 0),
                                    "cyclomatic_complexity": baseline_ir_metrics.get("cyclomatic_complexity", 0),
                                    "functions": baseline_ir_metrics.get("functions", 0),
                                    "loops": baseline_ir_metrics.get("loops", 0),
                                },
                                "obfuscated": {
                                    "basic_blocks": obf_ir_metrics.get("basic_blocks", 0),
                                    "cfg_edges": obf_ir_metrics.get("cfg_edges", 0),
                                    "cyclomatic_complexity": obf_ir_metrics.get("cyclomatic_complexity", 0),
                                    "functions": obf_ir_metrics.get("functions", 0),
                                    "loops": obf_ir_metrics.get("loops", 0),
                                },
                                "comparison": {
                                    "complexity_increase_percent": ((obf_ir_metrics.get("cyclomatic_complexity", 0) - baseline_ir_metrics.get("cyclomatic_complexity", 1)) / max(1, baseline_ir_metrics.get("cyclomatic_complexity", 1)) * 100) if baseline_ir_metrics.get("cyclomatic_complexity", 0) > 0 else 0,
                                    "basic_blocks_added": max(0, obf_ir_metrics.get("basic_blocks", 0) - baseline_ir_metrics.get("basic_blocks", 0)),
                                    "cfg_edges_added": max(0, obf_ir_metrics.get("cfg_edges", 0) - baseline_ir_metrics.get("cfg_edges", 0)),
                                }
                            } if baseline_ir_metrics else None,
                            "instruction_metrics": {
                                "baseline": {
                                    "total_instructions": baseline_ir_metrics.get("total_instructions", 0),
                                    "instruction_distribution": baseline_ir_metrics.get("instruction_distribution", {}),
                                    "arithmetic_complexity_score": baseline_ir_metrics.get("arithmetic_complexity_score", 0),
                                    "mba_expression_count": baseline_ir_metrics.get("mba_expression_count", 0),
                                    "call_instruction_count": baseline_ir_metrics.get("call_instruction_count", 0),
                                    "indirect_call_count": baseline_ir_metrics.get("indirect_call_count", 0),
                                },
                                "obfuscated": {
                                    "total_instructions": obf_ir_metrics.get("total_instructions", 0),
                                    "instruction_distribution": obf_ir_metrics.get("instruction_distribution", {}),
                                    "arithmetic_complexity_score": obf_ir_metrics.get("arithmetic_complexity_score", 0),
                                    "mba_expression_count": obf_ir_metrics.get("mba_expression_count", 0),
                                    "call_instruction_count": obf_ir_metrics.get("call_instruction_count", 0),
                                    "indirect_call_count": obf_ir_metrics.get("indirect_call_count", 0),
                                },
                                "comparison": {
                                    "instruction_growth_percent": ((obf_ir_metrics.get("total_instructions", 0) - baseline_ir_metrics.get("total_instructions", 1)) / max(1, baseline_ir_metrics.get("total_instructions", 1)) * 100) if baseline_ir_metrics.get("total_instructions", 0) > 0 else 0,
                                    "arithmetic_complexity_increase": ((obf_ir_metrics.get("arithmetic_complexity_score", 0) - baseline_ir_metrics.get("arithmetic_complexity_score", 0.1)) / max(0.1, baseline_ir_metrics.get("arithmetic_complexity_score", 0.1)) * 100) if baseline_ir_metrics.get("arithmetic_complexity_score", 0) > 0 else 0,
                                    "mba_expressions_added": max(0, obf_ir_metrics.get("mba_expression_count", 0) - baseline_ir_metrics.get("mba_expression_count", 0)),
                                }
                            } if baseline_ir_metrics else None,
                            "binary_structure": None,  # Can be enhanced with binary analysis
                            "pattern_resistance": None,  # Can be enhanced with pattern analysis
                            "call_graph_metrics": None,  # Can be enhanced with call graph analysis
                        }

                        # ✅ NEW: Run test suite to verify obfuscation correctness
                        try:
                            logger.info("Running obfuscation test suite...")
                            test_results = run_obfuscation_tests(
                                baseline_binary=Path(baseline_for_metrics),
                                obfuscated_binary=Path(final_binary),
                                program_name=payload.name or "program",
                                results_dir=Path(job.job_id)
                            )

                            # Fallback to lightweight tests if full suite not available
                            if not test_results:
                                logger.info("Full test suite not available, running lightweight tests...")
                                test_results = run_lightweight_tests(
                                    baseline_binary=Path(baseline_for_metrics),
                                    obfuscated_binary=Path(final_binary),
                                    program_name=payload.name or "program"
                                )

                            if test_results:
                                merge_test_results_into_report(job_data, test_results)
                                logger.info("✅ Test results merged into report")
                            else:
                                logger.warning("No test results available (both full and lightweight tests failed)")
                        except Exception as e:
                            logger.warning(f"Failed to run test suite: {e}")
                            # Continue without test results rather than failing the job

                        # Generate and export reports
                        try:
                            report = reporter.generate_report(job_data)
                            report_formats = payload.report_formats or ["json", "markdown", "pdf"]
                            exported = reporter.export(report, job.job_id, report_formats)
                            report_paths_dict = {fmt: str(path) for fmt, path in exported.items()}
                            logger.info(f"Reports generated: {list(report_paths_dict.keys())}")
                        except Exception as report_error:
                            logger.error(f"Failed to generate reports: {report_error}", exc_info=True)
                            raise

                    except Exception as e:
                        logger.error(f"Failed to generate reports for custom build: {e}", exc_info=True)
                        # Re-raise to ensure job fails if report generation fails
                        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

                result = {
                    "output_file": str(final_binary),
                    "build_system": payload.build_system,
                    "source_obfuscation": obf_results,
                    "binaries_found": [str(final_binary)],
                    "report_paths": report_paths_dict,
                }

                # Store main_source for use in multi-platform compilation
                source_path = main_source

            else:
                # SIMPLE BUILD MODE: Direct clang compilation (original behavior)
                # Find main file (file containing main() function)
                main_file_path = None
                main_pattern = re.compile(r'\bint\s+main\s*\(|\bvoid\s+main\s*\(')

                for src_file in all_source_paths:
                    try:
                        content = src_file.read_text(encoding='utf-8', errors='ignore')
                        if main_pattern.search(content):
                            main_file_path = src_file
                            logger.info(f"Found main file: {main_file_path.name}")
                            break
                    except Exception as e:
                        logger.warning(f"Could not read {src_file}: {e}")

                if not main_file_path:
                    # Use first source file if no main found
                    main_file_path = all_source_paths[0]
                    logger.warning(f"No main() found, using first source file: {main_file_path.name}")

                # Build config with additional source files (excluding main file)
                # Pass repo_path as project_root so entrypoint command runs in the correct directory
                additional_sources = [src for src in all_source_paths if src != main_file_path]
                config = _build_config_from_request(payload, working_dir, additional_sources=additional_sources, project_root=repo_path)

                # Run obfuscation
                result = obfuscator.obfuscate(main_file_path, config, job_id=job.job_id)

                # Store for use in multi-platform compilation
                source_path = main_file_path

            # Clean up repository session after successful obfuscation
            # This cleans up the repo_path directory (e.g., /tmp/oaas_repo_xxx/owner-repo-sha/)
            cleanup_repo_session(payload.repo_session_id)
            logger.info(f"Cleaned up repository session: {payload.repo_session_id}")

            # Also clean up the parent temp directory if it's empty
            # The repo_path is something like /tmp/oaas_repo_xxx/owner-repo-sha/
            # We want to also delete /tmp/oaas_repo_xxx/ if it's now empty
            try:
                parent_temp_dir = repo_path.parent
                if parent_temp_dir.exists() and parent_temp_dir.name.startswith("oaas_repo_"):
                    # Check if directory is empty or only has the repo dir
                    remaining = list(parent_temp_dir.iterdir())
                    if not remaining:
                        shutil.rmtree(parent_temp_dir)
                        logger.info(f"Cleaned up parent temp directory: {parent_temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup parent temp directory: {e}")

        elif is_multi_file:
            # Multi-file project (GitHub repo - legacy method with source_files)
            logger.info(f"Processing multi-file project with {len(payload.source_files)} files")

            # Create project structure
            project_dir = working_dir / "project"
            main_file_path, all_source_paths = create_multi_file_project(
                payload.source_files,
                project_dir
            )

            if not all_source_paths:
                raise HTTPException(
                    status_code=400,
                    detail="No valid C/C++ source files found in repository"
                )

            logger.info(f"Found {len(all_source_paths)} source files, main file: {main_file_path.name}")

            # Build config with additional source files (excluding main file)
            # Additional sources will be added to compiler flags for multi-file compilation
            additional_sources = [src for src in all_source_paths if src != main_file_path]
            config = _build_config_from_request(payload, working_dir, additional_sources=additional_sources)

            # For multi-file projects, we compile all files together
            # The main file is passed to the obfuscator, and additional sources are in compiler flags
            result = obfuscator.obfuscate(main_file_path, config, job_id=job.job_id)

            # Store for use in multi-platform compilation
            source_path = main_file_path

        else:
            # Single file mode (original behavior)
            source_filename = _sanitize_filename(payload.filename)
            source_path = (working_dir / source_filename).resolve()
            _decode_source(payload.source_code, source_path)
            config = _build_config_from_request(payload, working_dir)

            result = obfuscator.obfuscate(source_path, config, job_id=job.job_id)

        # ✅ NEW: Run tests for ALL build modes (multifile, simple, custom)
        # This runs AFTER obfuscation but BEFORE report attachment
        # The baseline binary is created during obfuscation in the output_directory
        logger.info("=" * 80)
        logger.info("ATTEMPTING TO RUN OBFUSCATION TESTS")
        logger.info("=" * 80)
        try:
            # Get output directory and obfuscated binary path
            output_dir = None
            source_stem = "program"

            # Determine output_dir and source_stem based on build mode
            if 'config' in locals() and config and hasattr(config, 'output') and config.output:
                output_dir = Path(config.output.directory)
                logger.info(f"Output directory: {output_dir}")
            else:
                logger.warning("Could not determine output directory from config")

            # Get the obfuscated binary path
            obfuscated_binary_path = result.get("output_file", "")
            logger.info(f"Obfuscated binary from result: {obfuscated_binary_path}")

            if not obfuscated_binary_path:
                logger.warning("No obfuscated binary found in result")
                raise ValueError("output_file not in result")

            obfuscated_binary = Path(obfuscated_binary_path)
            logger.info(f"Obfuscated binary exists: {obfuscated_binary.exists()}")

            if not obfuscated_binary.exists():
                logger.warning(f"Obfuscated binary does not exist at: {obfuscated_binary}")
                raise FileNotFoundError(f"Obfuscated binary not found: {obfuscated_binary}")

            # Determine source stem from the binary name (more reliable method)
            if obfuscated_binary.name:
                # Remove common prefixes and suffixes to get original name
                stem = obfuscated_binary.stem
                logger.info(f"Original binary stem: {stem}")

                # Remove PREFIXES
                for prefix in ['obfuscated_', 'obf_']:
                    if stem.startswith(prefix):
                        stem = stem[len(prefix):]
                        logger.info(f"  Removed prefix, now: {stem}")
                        break

                # Remove SUFFIXES
                for suffix in ['_obfuscated', '.obfuscated', '_obf']:
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        logger.info(f"  Removed suffix, now: {stem}")
                        break

                source_stem = stem
                logger.info(f"Source stem determined from binary: {source_stem}")

            # Find baseline binary
            baseline_candidates = []

            # Search in both output_dir and working_dir (for multifile projects)
            search_dirs = []
            if output_dir and output_dir.exists():
                search_dirs.append(output_dir)

            # For multifile projects, also search in working_dir
            working_dir = Path(os.environ.get("REPORT_BASE", "/app/reports")) / job.job_id
            if working_dir.exists():
                search_dirs.append(working_dir)
                logger.info(f"Also searching in working directory: {working_dir}")

            for search_dir in search_dirs:
                logger.info(f"Searching for baseline binary in: {search_dir}")
                # Look for files with _baseline suffix
                for candidate in search_dir.glob(f"*_baseline"):
                    logger.info(f"  Found candidate: {candidate}")
                    if candidate not in baseline_candidates:
                        baseline_candidates.append(candidate)

                # Try multiple expected names
                expected_names = [
                    f"{source_stem}_baseline",  # tsvc_baseline
                    f"obfuscated_{source_stem}_baseline",  # obfuscated_tsvc_baseline
                    source_stem,  # If baseline wasn't suffixed
                ]

                for expected_name in expected_names:
                    expected_baseline = search_dir / expected_name
                    logger.info(f"Checking: {expected_baseline}")
                    if expected_baseline.exists():
                        logger.info(f"  ✓ Found: {expected_name}")
                        if expected_baseline not in baseline_candidates:
                            baseline_candidates.insert(0, expected_baseline)
                        break
                    else:
                        logger.info(f"  ✗ Not found: {expected_name}")

            if not baseline_candidates:
                logger.warning(f"No baseline binaries found in any search directory")
                logger.warning("Listing files in output directory:")
                if output_dir and output_dir.exists():
                    for item in output_dir.iterdir():
                        logger.warning(f"  {item.name}")
                logger.warning("Listing files in working directory:")
                if working_dir.exists():
                    for item in working_dir.iterdir():
                        logger.warning(f"  {item.name}")
                raise FileNotFoundError(f"No baseline binary found for {source_stem}")

            baseline_binary = baseline_candidates[0]
            logger.info(f"Using baseline binary: {baseline_binary}")
            logger.info(f"Baseline exists: {baseline_binary.exists()}")

            if not baseline_binary.exists():
                logger.error(f"Baseline binary path exists but file is not accessible: {baseline_binary}")
                raise FileNotFoundError(f"Baseline binary not found: {baseline_binary}")

            # Run tests
            logger.info(f"Running tests comparing:")
            logger.info(f"  Baseline: {baseline_binary}")
            logger.info(f"  Obfuscated: {obfuscated_binary}")

            # Try full test suite first, then lightweight as fallback
            test_results = run_obfuscation_tests(
                baseline_binary=baseline_binary,
                obfuscated_binary=obfuscated_binary,
                program_name=payload.name or source_stem,
                results_dir=Path(job.job_id)
            )

            if not test_results:
                logger.info("Full test suite not available, running lightweight tests...")
                test_results = run_lightweight_tests(
                    baseline_binary=baseline_binary,
                    obfuscated_binary=obfuscated_binary,
                    program_name=payload.name or source_stem
                )

            if test_results:
                logger.info("✅ Test results obtained, updating report...")
                # Merge test results into the existing report JSON
                report_path = result.get("report_paths", {}).get("json")
                logger.info(f"Looking for JSON report at: {report_path}")

                if report_path and Path(report_path).exists():
                    try:
                        with open(report_path, 'r') as f:
                            report_data = json.load(f)

                        # Add test results to report
                        report_data["metadata"] = test_results.get("metadata")
                        report_data["test_results"] = test_results.get("test_results")
                        report_data["test_metrics"] = test_results.get("metrics", {})
                        report_data["reliability_status"] = test_results.get("reliability_status")

                        # Write updated report
                        with open(report_path, 'w') as f:
                            json.dump(report_data, f, indent=2)

                        logger.info(f"✅ Report successfully updated with test results at: {report_path}")
                    except Exception as e:
                        logger.error(f"Could not update report with test results: {e}", exc_info=True)
                else:
                    logger.warning(f"JSON report not found at: {report_path}")
            else:
                logger.warning("No test results available (both full and lightweight tests failed)")

        except Exception as e:
            logger.error(f"Failed to run tests: {e}", exc_info=True)
            logger.info("Continuing without test results (non-blocking failure)")
            # Continue without tests rather than blocking the job

        job_manager.update_job(job.job_id, status="completed", result=result)
        reports_to_attach = result.get("report_paths", {})
        logger.info(f"[REPORT DEBUG] Attaching reports for job {job.job_id}: {reports_to_attach}")
        job_manager.attach_reports(job.job_id, reports_to_attach)

        binary_path = Path(result.get("output_file", ""))
        if not binary_path.exists():
            raise HTTPException(status_code=500, detail="Binary generation failed")

        # Build for additional platforms for both single-file and multi-file modes
        # For multi-platform support, we compile the same source for each target
        platform_binaries = {"linux": None, "windows": None}

        # First result is for the primary platform
        primary_platform = payload.platform.value
        platform_binaries[primary_platform] = str(binary_path)

        # Compile for other platforms
        # Get the obfuscated source (after source-level transforms) for single-file
        if not is_multi_file:
            obfuscated_source = result.get("obfuscated_source_path")
            if obfuscated_source and Path(obfuscated_source).exists():
                compile_source = Path(obfuscated_source)
            else:
                compile_source = source_path
        else:
            # For multi-file, use the original source file path
            compile_source = source_path

        for target_platform, target_arch in target_platforms:
            if target_platform.value == primary_platform:
                continue  # Already compiled for primary platform

            try:
                # Create platform-specific output directory
                platform_dir = working_dir / target_platform.value
                platform_dir.mkdir(exist_ok=True)

                # Build config for this platform
                platform_payload = payload.model_copy()
                platform_payload.platform = target_platform
                platform_payload.architecture = target_arch
                platform_config = _build_config_from_request(platform_payload, platform_dir)

                # Compile for this platform
                platform_result = obfuscator.obfuscate(
                    compile_source,
                    platform_config,
                    job_id=f"{job.job_id}_{target_platform.value}"
                )

                platform_binary = Path(platform_result.get("output_file", ""))
                if platform_binary.exists():
                    platform_binaries[target_platform.value] = str(platform_binary)
                    logger.info(f"Built binary for {target_platform.value}: {platform_binary}")
                else:
                    logger.warning(f"Binary generation failed for {target_platform.value}")
            except Exception as e:
                logger.warning(f"Failed to build for {target_platform.value}: {e}")
                # Continue with other platforms even if one fails

        # Store platform binaries in job result
        result["platform_binaries"] = platform_binaries
        job_manager.update_job(job.job_id, result=result)

        logger.info("[PLATFORM DEBUG] platform_binaries dict: %s", platform_binaries)
        logger.info("[PLATFORM DEBUG] platform_binaries.get('linux'): %s", platform_binaries.get("linux"))
        logger.info("[PLATFORM DEBUG] platform_binaries.get('windows'): %s", platform_binaries.get("windows"))
        logger.info("[PLATFORM DEBUG] platform_binaries.get('macos'): %s", platform_binaries.get("macos"))

        # Optional: Run Phoronix benchmarking (same as async endpoint)
        logger.info(f"[PHORONIX DEBUG] run_benchmarks flag: {payload.run_benchmarks}")
        logger.info(f"[PHORONIX DEBUG] result keys: {list(result.keys())}")

        if payload.run_benchmarks:
            logger.info("[PHORONIX] Starting benchmarking...")
            try:
                from core.phoronix_integration import PhoronixBenchmarkRunner

                baseline_binary = result.get("baseline_binary")
                obfuscated_binary = result.get("obfuscated_binary")

                logger.info(f"[PHORONIX] Baseline binary: {baseline_binary}")
                logger.info(f"[PHORONIX] Obfuscated binary: {obfuscated_binary}")

                if baseline_binary and obfuscated_binary:
                    baseline_path = Path(baseline_binary)
                    obfuscated_path = Path(obfuscated_binary)

                    if baseline_path.exists() and obfuscated_path.exists():
                        runner = PhoronixBenchmarkRunner()
                        phoronix_results = runner.run_benchmark(
                            baseline_path,
                            obfuscated_path,
                            timeout=payload.benchmark_timeout_seconds
                        )

                        key_metrics = runner.extract_key_metrics(phoronix_results)
                        result['phoronix'] = {
                            'results': phoronix_results,
                            'key_metrics': key_metrics
                        }

                        logger.info(f"✅ Benchmarking completed: available={key_metrics.get('available')}, instruction_delta={key_metrics.get('instruction_count_delta')}, overhead={key_metrics.get('performance_overhead_percent')}%")
                        job_manager.update_job(job.job_id, result=result)
                    else:
                        logger.warning(f"Binary paths don't exist - baseline: {baseline_path.exists()}, obfuscated: {obfuscated_path.exists()}")
                        result['phoronix'] = {
                            'results': None,
                            'error': 'Binary files not found'
                        }
                else:
                    logger.warning("Missing baseline or obfuscated binary paths in result")
                    result['phoronix'] = {
                        'results': None,
                        'error': 'Binaries not found'
                    }
            except Exception as e:
                logger.error(f"Benchmarking failed: {e}")
                result['phoronix'] = {
                    'results': None,
                    'error': str(e)
                }

        # IMPORTANT: Regenerate reports after adding phoronix data
        # The original reports were generated BEFORE phoronix benchmarking
        logger.info(f"[PHORONIX DEBUG] Checking if reports need regen: run_benchmarks={payload.run_benchmarks}, has_phoronix={bool(result.get('phoronix'))}")

        if payload.run_benchmarks and result.get('phoronix'):
            logger.info("[PHORONIX] Regenerating reports with Phoronix metrics...")
            try:
                # Get the JSON report that was generated
                report_paths = result.get("report_paths", {})
                json_report_path = report_paths.get("json")

                logger.info(f"[PHORONIX] report_paths keys: {list(report_paths.keys())}")
                logger.info(f"[PHORONIX] json_report_path: {json_report_path}")

                if json_report_path and Path(json_report_path).exists():
                    # Read the existing JSON report
                    with open(json_report_path, 'r') as f:
                        report_data = json.load(f)

                    logger.info(f"[PHORONIX] Report loaded, keys before: {list(report_data.keys())}")

                    # Add phoronix data to the report
                    # IMPORTANT: Only include key_metrics, not full results (which may have non-serializable objects)
                    phoronix_full_data = result.get('phoronix')
                    if phoronix_full_data:
                        report_data['phoronix'] = {
                            'key_metrics': phoronix_full_data.get('key_metrics', {})
                        }
                        logger.info(f"[PHORONIX] Added phoronix key_metrics to report")
                    else:
                        logger.info(f"[PHORONIX] No phoronix data in result")

                    # Regenerate all report formats with updated data
                    report_formats = payload.report_formats or ["json", "markdown", "pdf"]
                    logger.info(f"[PHORONIX] Exporting formats: {report_formats}")
                    exported = reporter.export(report_data, job.job_id, report_formats)
                    report_paths_dict = {fmt: str(path) for fmt, path in exported.items()}

                    logger.info(f"[PHORONIX] Reports regenerated: {list(report_paths_dict.keys())}")

                    # Update result with new report paths
                    result["report_paths"] = report_paths_dict
                else:
                    logger.warning(f"[PHORONIX] Could not regenerate - json_report_path exists: {json_report_path is not None}, path exists: {Path(json_report_path).exists() if json_report_path else False}")
            except Exception as e:
                logger.error(f"[PHORONIX] Failed to regenerate reports: {e}", exc_info=True)

        download_urls_response = {
            "linux": f"/api/download/{job.job_id}/linux" if platform_binaries.get("linux") else None,
            "windows": f"/api/download/{job.job_id}/windows" if platform_binaries.get("windows") else None,
            "macos": f"/api/download/{job.job_id}/macos" if platform_binaries.get("macos") else None,
        }
        logger.info("[PLATFORM DEBUG] Final download_urls response: %s", download_urls_response)

        return {
            "job_id": job.job_id,
            "status": "completed",
            "download_url": f"/api/download/{job.job_id}",
            "download_urls": download_urls_response,
            "report_url": f"/api/report/{job.job_id}",
        }
    except Exception as exc:
        logger.exception("Job %s failed", job.job_id)
        job_manager.update_job(job.job_id, status="failed", error=str(exc))
        
        # Clean up repository session on failure
        if payload.repo_session_id:
            # Get repo_path before cleanup to delete parent temp dir
            repo_session = get_repo_session(payload.repo_session_id)
            repo_path = repo_session["repo_path"] if repo_session else None
            
            cleanup_repo_session(payload.repo_session_id)
            logger.info(f"Cleaned up repository session after failure: {payload.repo_session_id}")
            
            # Also clean up the parent temp directory if it's empty
            if repo_path:
                try:
                    parent_temp_dir = repo_path.parent
                    if parent_temp_dir.exists() and parent_temp_dir.name.startswith("oaas_repo_"):
                        remaining = list(parent_temp_dir.iterdir())
                        if not remaining:
                            shutil.rmtree(parent_temp_dir)
                            logger.info(f"Cleaned up parent temp directory after failure: {parent_temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup parent temp directory: {e}")
        
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/obfuscate")
async def api_obfuscate(
    payload: ObfuscateRequest,
    background: BackgroundTasks,
):
    """Async obfuscation - queue job and process in background."""
    _validate_source_size(payload.source_code)

    # Validate and sanitize entrypoint command
    entrypoint_cmd = _validate_entrypoint_command(payload.entrypoint_command or "./a.out")

    job = job_manager.create_job({
        "filename": payload.filename,
        "platform": payload.platform.value,
        "entrypoint_command": entrypoint_cmd
    })
    await progress_tracker.publish(ProgressEvent(job.job_id, "queued", 0.0, "Job queued"))
    working_dir = report_base / job.job_id
    ensure_directory(working_dir)
    source_filename = _sanitize_filename(payload.filename)
    source_path = (working_dir / source_filename).resolve()
    _decode_source(payload.source_code, source_path)
    config = _build_config_from_request(payload, working_dir)
    background.add_task(_run_obfuscation, job.job_id, source_path, config,
                        payload.run_benchmarks, payload.benchmark_timeout_seconds)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/analyze/{job_id}")
async def api_analyze(job_id: str):
    try:
        job = job_manager.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    result = job.metadata.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Job not completed")
    binary_path = Path(result.get("output_file", ""))
    if not binary_path.exists():
        raise HTTPException(status_code=404, detail="Obfuscated binary not found")
    analysis = analyze_binary(AnalyzeConfig(binary_path=binary_path))
    return JSONResponse(analysis)


@app.post("/api/compare")
async def api_compare(payload: CompareRequest):
    working_dir = report_base / "comparisons"
    ensure_directory(working_dir)
    original_path = working_dir / f"{payload.filename}_original.bin"
    obfuscated_path = working_dir / f"{payload.filename}_obfuscated.bin"
    original_path.write_bytes(base64.b64decode(payload.original_b64))
    obfuscated_path.write_bytes(base64.b64decode(payload.obfuscated_b64))
    config = CompareConfig(original_binary=original_path, obfuscated_binary=obfuscated_path)
    result = compare_binaries(config)
    return JSONResponse(result)


@app.get("/api/jobs")
async def api_list_jobs():
    records = job_manager.list_jobs()
    payload = [
        {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat() + "Z",
            "updated_at": job.updated_at.isoformat() + "Z",
        }
        for job in records
    ]
    return JSONResponse(payload)


@app.get("/api/report/{job_id}")
async def api_get_report(job_id: str, fmt: str = "json"):
    logger.info("[REPORT] Requesting report - job_id: %s, format: %s", job_id, fmt)
    try:
        job = job_manager.get_job(job_id)
        logger.info("[REPORT] Job found: %s", job.job_id)
    except JobNotFoundError:
        logger.error("[REPORT] Job not found: %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")

    fmt_lower = fmt.lower()
    logger.info("[REPORT] Format requested (lowercased): %s", fmt_lower)

    # Always require JSON as base (we convert from JSON to other formats)
    json_report_path = job.report_paths.get("json")
    if not json_report_path:
        logger.error("[REPORT] No JSON report available")
        raise HTTPException(status_code=404, detail="JSON report not found")

    json_path = Path(json_report_path)
    if not json_path.exists():
        logger.error("[REPORT] JSON report file missing at: %s", json_path)
        raise HTTPException(status_code=404, detail="Report file missing")

    # Load JSON report
    try:
        with open(json_path, 'r') as f:
            report_data = json.load(f)
        logger.info("[REPORT] JSON report loaded successfully")
    except Exception as e:
        logger.error("[REPORT] Failed to load JSON report: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load report")

    # Return format based on request
    if fmt_lower == "json":
        logger.info("[REPORT] Returning JSON report")
        return FileResponse(json_path, media_type="application/json", filename=f"{job_id}.json")

    elif fmt_lower == "markdown":
        logger.info("[REPORT] Converting to Markdown")
        try:
            markdown_content = report_converter.json_to_markdown(report_data)
            logger.info("[REPORT] Markdown conversion successful")
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                iter([markdown_content.encode('utf-8')]),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{job_id}.markdown"'}
            )
        except Exception as e:
            logger.error("[REPORT] Markdown conversion failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate markdown report")

    elif fmt_lower == "pdf":
        logger.info("[REPORT] Converting to PDF")
        try:
            pdf_content = report_converter.json_to_pdf(report_data)
            logger.info("[REPORT] PDF conversion successful, size: %d bytes", len(pdf_content))
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                iter([pdf_content]),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{job_id}.pdf"'}
            )
        except Exception as e:
            logger.error("[REPORT] PDF conversion failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate PDF report")

    else:
        logger.warning("[REPORT] Unknown format requested: %s", fmt_lower)
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")


@app.websocket("/ws/jobs/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        async for event in progress_tracker.subscribe(job_id):
            await websocket.send_json(
                {
                    "job_id": job_id,
                    "stage": event.stage,
                    "progress": event.progress,
                    "message": event.message,
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)


@app.get("/api/download/{job_id}/baseline")
async def api_download_baseline(job_id: str):
    """Download the baseline (original, non-obfuscated) binary."""
    try:
        job = job_manager.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.metadata.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Find baseline binary in the working directory
    working_dir = report_base / job_id
    if not working_dir.exists():
        raise HTTPException(status_code=404, detail="Job directory not found")

    # Look for baseline binary (typically named {source}_baseline or {source}_baseline.exe)
    baseline_files = list(working_dir.glob("*_baseline*"))
    if not baseline_files:
        raise HTTPException(status_code=404, detail="Baseline binary not found")

    # Use the first baseline file found
    baseline_path = baseline_files[0]
    if not baseline_path.exists():
        raise HTTPException(status_code=404, detail="Baseline binary file not found")

    return FileResponse(
        baseline_path,
        media_type="application/octet-stream",
        filename=baseline_path.name
    )


@app.get("/api/download/{job_id}/{platform}")
async def api_download_binary_platform(job_id: str, platform: str):
    """Download the obfuscated binary for a specific platform."""
    try:
        job = job_manager.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.metadata.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Check if this is a multi-platform build
    platform_binaries = result.get("platform_binaries", {})
    if platform_binaries and platform in platform_binaries and platform_binaries[platform]:
        binary_path = Path(platform_binaries[platform])
        if not binary_path.exists():
            raise HTTPException(status_code=404, detail=f"Binary not found for platform {platform}")

        # Determine appropriate filename and extension
        if platform == "windows":
            filename = f"obfuscated_{platform}.exe"
        elif platform == "macos":
            filename = f"obfuscated_{platform}"
        else:
            filename = f"obfuscated_{platform}"

        return FileResponse(
            binary_path,
            media_type="application/octet-stream",
            filename=filename
        )

    raise HTTPException(status_code=404, detail=f"Platform {platform} not found for this job")

@app.get("/api/download/{job_id}")
async def api_download_binary(job_id: str):
    """Download the obfuscated binary (legacy single-platform support)."""
    try:
        job = job_manager.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.metadata.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Check if this is a multi-platform build
    platform_binaries = result.get("platform_binaries", {})
    if platform_binaries:
        # Return Linux binary by default
        if platform_binaries.get("linux"):
            binary_path = Path(platform_binaries["linux"])
        elif platform_binaries.get("windows"):
            binary_path = Path(platform_binaries["windows"])
        else:
            raise HTTPException(status_code=404, detail="No binaries available")
    else:
        # Legacy single-platform build
        binary_path = Path(result.get("output_file", ""))

    if not binary_path.exists():
        raise HTTPException(status_code=404, detail="Obfuscated binary not found")

    return FileResponse(
        binary_path,
        media_type="application/octet-stream",
        filename=binary_path.name
    )


@app.get("/api/remarks/{job_id}")
async def api_get_remarks(job_id: str):
    """Get LLVM remarks file for a completed obfuscation job."""
    try:
        job = job_manager.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = job.metadata.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Find the output binary path
    output_file = result.get("output_file", "")
    if not output_file:
        raise HTTPException(status_code=404, detail="Output file not found in job result")
    
    binary_path = Path(output_file)
    if not binary_path.exists():
        raise HTTPException(status_code=404, detail="Obfuscated binary not found")
    
    # Remarks file is in the same directory as the binary with .opt.yaml extension
    remarks_file = binary_path.parent / f"{binary_path.stem}.opt.yaml"
    
    if not remarks_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Remarks file not found. Expected at: {remarks_file}. Remarks may not have been enabled for this job."
        )
    
    # Read and return the remarks file content
    try:
        remarks_content = remarks_file.read_text(encoding='utf-8')
        return JSONResponse({
            "job_id": job_id,
            "remarks_file": str(remarks_file),
            "content": remarks_content,
            "size": remarks_file.stat().st_size
        })
    except Exception as e:
        logger.error(f"Failed to read remarks file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read remarks file: {str(e)}")


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


@app.get("/api/github/repo/session/{session_id}")
async def github_repo_session_status(session_id: str):
    """Check status of a repository session."""
    session = get_repo_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    import time
    age = time.time() - session["timestamp"]
    
    return JSONResponse({
        "session_id": session_id,
        "repo_name": session["repo_name"],
        "branch": session["branch"],
        "age_seconds": int(age),
        "expires_in": max(0, 3600 - int(age)),
        "status": "active"
    })


@app.delete("/api/github/repo/session/{session_id}")
async def github_repo_session_delete(session_id: str):
    """Manually delete a repository session."""
    success = cleanup_repo_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse({"success": True, "message": "Session deleted"})


@app.post("/api/github/repo/session/{session_id}")
async def github_repo_session_cleanup_beacon(session_id: str):
    """Cleanup repository session via sendBeacon (called on page unload).

    This endpoint accepts POST requests from navigator.sendBeacon() which is used
    for reliable cleanup when the user closes or navigates away from the page.
    """
    success = cleanup_repo_session(session_id)

    # Don't raise error if session not found - it may have already been cleaned up
    # Just log and return success to avoid errors in browser console
    if not success:
        logger.info(f"Session cleanup called but session not found: {session_id}")

    return JSONResponse({"success": True, "message": "Session cleanup completed"})


@app.get("/api/flags")
async def api_flags():
    # Expose the comprehensive flag list for UI selection
    return JSONResponse(comprehensive_flags)


@app.get("/api/capabilities")
async def api_capabilities():
    """Expose backend capabilities so UI can adapt.

    - pass_plugin.available: whether the obfuscation pass plugin was found
    - pass_plugin.path: the discovered path if available
    """
    return JSONResponse(
        {
            "pass_plugin": {
                "available": DEFAULT_PASS_PLUGIN_EXISTS,
                "path": DEFAULT_PASS_PLUGIN_PATH,
            },
            "github_oauth": {
                "enabled": bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET),
                "client_id": GITHUB_CLIENT_ID if GITHUB_CLIENT_ID else None,
            }
        }
    )


@app.get("/api/github/login")
async def github_login():
    """Initiate GitHub OAuth flow - returns redirect URL."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")
    
    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = "pending"
    
    # Build GitHub OAuth URL
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "repo",
        "state": state,
    }
    
    auth_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    
    return JSONResponse({
        "auth_url": auth_url
    })


@app.get("/api/github/callback")
async def github_callback(code: str, state: str):
    """Handle GitHub OAuth callback - exchanges code for token and redirects to frontend."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=oauth_not_configured")
    
    # Verify state
    if state not in oauth_states:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=invalid_state")
    
    try:
        # Exchange code for access token
        token_data = {
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
        }
        
        headers = {"Accept": "application/json"}
        response = requests.post("https://github.com/login/oauth/access_token", data=token_data, headers=headers)
        
        if response.status_code != 200:
            return RedirectResponse(url=f"{FRONTEND_URL}?error=token_exchange_failed")
        
        token_info = response.json()
        access_token = token_info.get("access_token")
        
        if not access_token:
            return RedirectResponse(url=f"{FRONTEND_URL}?error=no_token")
        
        # Create session ID
        import time
        session_id = secrets.token_urlsafe(32)
        user_sessions[session_id] = {
            "token": access_token,
            "timestamp": time.time()
        }
        
        # Clean up state
        del oauth_states[state]
        
        # Redirect to frontend with session cookie
        response = RedirectResponse(url=f"{FRONTEND_URL}?github_auth=success")
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            max_age=TOKEN_TTL,
            samesite="lax"
        )
        return response
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}?error=callback_failed")




@app.get("/api/github/status")
async def github_status(request: Request):
    """Check if user has active GitHub session."""
    token = get_session_token(request, SESSION_COOKIE_NAME, user_sessions, TOKEN_TTL)
    return JSONResponse({"authenticated": token is not None})


@app.post("/api/github/disconnect")
async def github_disconnect(request: Request):
    """Disconnect GitHub - delete stored token."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id and session_id in user_sessions:
        del user_sessions[session_id]
    
    response = JSONResponse({"success": True})
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.get("/api/github/repos")
async def github_repos(request: Request):
    """Get user's accessible repositories."""
    access_token = get_session_token(request, SESSION_COOKIE_NAME, user_sessions, TOKEN_TTL)
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated with GitHub")
    
    try:
        g = Github(access_token)
        user = g.get_user()
        
        repos = []
        for repo in user.get_repos():
            repos.append({
                "name": repo.name,
                "full_name": repo.full_name,
                "html_url": repo.html_url,
                "private": repo.private,
                "default_branch": repo.default_branch,
                "language": repo.language,
                "description": repo.description,
            })
        
        return JSONResponse({"repositories": repos})
        
    except Exception as e:
        logger.error(f"Failed to fetch repositories: {e}")
        raise HTTPException(status_code=400, detail="Failed to fetch repositories")


@app.get("/api/github/repo/branches")
async def github_repo_branches(request: Request, repo_url: str):
    """Get branches for a specific repository."""
    # Try to get token, but allow public access if not authenticated
    access_token = get_session_token(request, SESSION_COOKIE_NAME, user_sessions, TOKEN_TTL)

    branches = get_repo_branches(repo_url, access_token)
    return JSONResponse({"branches": branches})


@app.post("/api/github/repo/files")
async def github_repo_files(request: Request, payload: GitHubRepoRequest):
    """Extract files from a GitHub repository (legacy - returns files to frontend)."""
    # Try to get token, but allow public access if not authenticated
    access_token = get_session_token(request, SESSION_COOKIE_NAME, user_sessions, TOKEN_TTL)

    # Validate repo URL
    if "github.com" not in payload.repo_url:
        raise HTTPException(status_code=400, detail="Only GitHub repositories are supported")

    repo_data = extract_repo_files(payload.repo_url, payload.branch, access_token)
    return JSONResponse(repo_data.dict())


@app.post("/api/github/repo/clone")
async def github_repo_clone(request: Request, payload: GitHubCloneRequest):
    """Clone GitHub repository to backend ephemeral storage.
    
    This endpoint clones the repository to a temporary directory on the backend
    and returns a session ID that can be used for obfuscation. This is more
    efficient for large repositories than sending all files to the frontend.
    
    The temporary repository will be automatically cleaned up after obfuscation
    or after 1 hour of inactivity.
    """
    # Clean up old sessions first
    cleanup_old_sessions(max_age_seconds=3600)
    
    # Try to get token, but allow public access if not authenticated
    access_token = get_session_token(request, SESSION_COOKIE_NAME, user_sessions, TOKEN_TTL)
    
    # Validate repo URL
    if "github.com" not in payload.repo_url:
        raise HTTPException(status_code=400, detail="Only GitHub repositories are supported")
    
    try:
        session_id, repo_path = clone_repo_to_temp(
            payload.repo_url,
            payload.branch,
            access_token
        )
        
        # Count files
        c_cpp_files = list(repo_path.rglob("*.c")) + list(repo_path.rglob("*.cpp")) + \
                      list(repo_path.rglob("*.cc")) + list(repo_path.rglob("*.cxx"))
        
        return JSONResponse({
            "session_id": session_id,
            "repo_name": payload.repo_url.split("/")[-1],
            "branch": payload.branch,
            "file_count": len(c_cpp_files),
            "expires_in": 3600,  # 1 hour
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clone repository: {str(e)}")


@app.post("/api/dogbolt/upload")
async def dogbolt_upload(file: UploadFile = File(...)):
    """Proxy endpoint to upload binary to dogbolt.org"""
    try:
        # Check file size (dogbolt.org limit is 2 MB)
        contents = await file.read()
        if len(contents) > 2 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Binary too large ({len(contents) / 1024 / 1024:.2f} MB). Dogbolt.org limit is 2 MB."
            )
        
        # Upload to dogbolt.org
        files_data = {'file': (file.filename or 'binary', contents, file.content_type or 'application/octet-stream')}
        response = requests.post('https://dogbolt.org/api/binaries/', files=files_data, timeout=30)
        
        if response.status_code not in [200, 201]:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Dogbolt API error: {response.text}"
            )
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Dogbolt upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to dogbolt.org: {str(e)}")
    except Exception as e:
        logger.error(f"Dogbolt upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dogbolt/decompilers")
async def dogbolt_decompilers():
    """Get list of available decompilers from dogbolt.org"""
    try:
        response = requests.get('https://dogbolt.org/', timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch decompilers")
        
        # Extract decompilers JSON from HTML
        html = response.text
        match = re.search(r'<script id="decompilers_json" type="application/json">([\s\S]*?)</script>', html)
        if match:
            decompilers_json = json.loads(match.group(1))
            return {"decompilers": decompilers_json, "count": len(decompilers_json)}
        else:
            # Fallback: return common decompilers
            return {
                "decompilers": {
                    "BinaryNinja": {},
                    "Boomerang": {},
                    "Ghidra": {},
                    "Hex-Rays": {},
                    "RecStudio": {},
                    "Reko": {},
                    "Relyze": {},
                    "RetDec": {},
                    "Snowman": {}
                },
                "count": 9
            }
    except requests.exceptions.RequestException as e:
        logger.error(f"Dogbolt decompilers error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch decompilers: {str(e)}")


@app.get("/api/dogbolt/status/{binary_id}")
async def dogbolt_status(binary_id: str):
    """Get decompilation status for a binary (handles pagination like decompiler-explorer)"""
    try:
        url = f"https://dogbolt.org/api/binaries/{binary_id}/decompilations/"
        all_results = []
        next_url = url
        
        # Fetch all pages (handle pagination)
        while next_url:
            response = requests.get(next_url, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch decompilation status")
            
            data = response.json()
            all_results.extend(data.get('results', []))
            next_url = data.get('next')  # Follow pagination
        
        return {"results": all_results, "next": None}
    except requests.exceptions.RequestException as e:
        logger.error(f"Dogbolt status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {str(e)}")


@app.get("/api/dogbolt/download/{binary_id}/{decompilation_id}")
async def dogbolt_download(binary_id: str, decompilation_id: str):
    """Download decompiled code from dogbolt.org"""
    try:
        # First get the decompilation info to get the download URL
        status_response = requests.get(
            f"https://dogbolt.org/api/binaries/{binary_id}/decompilations/",
            timeout=10
        )
        if status_response.status_code != 200:
            raise HTTPException(status_code=status_response.status_code, detail="Failed to fetch decompilation info")
        
        decompilations = status_response.json().get('results', [])
        decompilation = next((d for d in decompilations if str(d.get('id')) == decompilation_id), None)
        
        if not decompilation:
            raise HTTPException(status_code=404, detail="Decompilation not found")
        
        download_url = decompilation.get('download_url')
        if not download_url:
            raise HTTPException(status_code=404, detail="Download URL not available")
        
        # Download the decompiled code (may be gzip compressed)
        code_response = requests.get(download_url, timeout=30)
        if code_response.status_code != 200:
            raise HTTPException(status_code=code_response.status_code, detail="Failed to download decompiled code")
        
        # Handle gzip decompression (like decompiler-explorer)
        import gzip
        content = code_response.content
        try:
            # Try to decompress as gzip
            decompressed = gzip.decompress(content)
            code = decompressed.decode('utf-8', errors='replace')
        except (gzip.BadGzipFile, OSError):
            # Not gzip, use as-is
            code = content.decode('utf-8', errors='replace')
        
        return {"code": code, "decompiler": decompilation.get('decompiler', {})}
    except requests.exceptions.RequestException as e:
        logger.error(f"Dogbolt download error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download decompiled code: {str(e)}")


@app.post("/api/dogbolt/rerun/{binary_id}/{decompilation_id}")
async def dogbolt_rerun(binary_id: str, decompilation_id: str):
    """Rerun a decompilation that timed out"""
    try:
        response = requests.post(
            f"https://dogbolt.org/api/binaries/{binary_id}/decompilations/{decompilation_id}/rerun/",
            timeout=10
        )
        if response.status_code not in [200, 201, 202]:
            raise HTTPException(status_code=response.status_code, detail="Failed to rerun decompilation")
        
        return response.json() if response.content else {"status": "queued"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Dogbolt rerun error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rerun decompilation: {str(e)}")


@app.post("/api/local/folder/upload")
async def local_folder_upload(files: List[UploadFile] = File(...)):
    """Upload local folder/files to backend ephemeral storage.

    This endpoint accepts multiple file uploads and stores them in a temporary
    directory, returning a session ID that can be used for obfuscation.
    The workflow is identical to the GitHub clone endpoint.

    The temporary files will be automatically cleaned up after obfuscation
    or after 1 hour of inactivity.
    """
    # Clean up old sessions first
    cleanup_old_sessions(max_age_seconds=3600)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # Collect file data
        files_data = []
        project_name = "local_upload"

        for upload_file in files:
            # Get relative path from filename (browsers send path for webkitdirectory)
            # The filename may contain the relative path like "folder/subfolder/file.c"
            relative_path = upload_file.filename or "unknown"

            # Extract project name from the first directory component
            if "/" in relative_path:
                first_dir = relative_path.split("/")[0]
                if first_dir and project_name == "local_upload":
                    project_name = first_dir

            # Read file content
            content = await upload_file.read()
            files_data.append((relative_path, content))

        if not files_data:
            raise HTTPException(status_code=400, detail="No valid files provided")

        # Create session using the same storage mechanism as GitHub clone
        session_id, project_path = create_local_upload_session(files_data, project_name)

        # Count C/C++ files
        c_cpp_files = list(project_path.rglob("*.c")) + list(project_path.rglob("*.cpp")) + \
                      list(project_path.rglob("*.cc")) + list(project_path.rglob("*.cxx"))

        return JSONResponse({
            "session_id": session_id,
            "repo_name": project_name,
            "branch": "local",
            "file_count": len(c_cpp_files),
            "total_files": len(files_data),
            "expires_in": 3600,  # 1 hour
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")
