from __future__ import annotations

import base64
import logging
import os
import re
import secrets
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
)
from core.comparer import CompareConfig, compare_binaries
from core.config import AdvancedConfiguration, PassConfiguration, SymbolObfuscationConfiguration
from core.exceptions import JobNotFoundError, ValidationError
from core.job_manager import JobManager
from core.progress import ProgressEvent, ProgressTracker
from core.utils import create_logger, ensure_directory, normalize_flags_and_passes

from .multifile_pipeline import (
    GitHubRepoRequest,
    RepoData,
    RepoFile,
    SourceFile,
    cleanup_old_sessions,
    cleanup_repo_session,
    clone_repo_to_temp,
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
    flattening: bool = False
    substitution: bool = False
    bogus_control_flow: bool = False
    split: bool = False


class SymbolObfuscationModel(BaseModel):
    enabled: bool = False
    algorithm: str = Field("sha256", pattern="^(sha256|blake2b|siphash)$")
    hash_length: int = Field(12, ge=8, le=32)
    prefix_style: str = Field("typed", pattern="^(none|typed|underscore)$")
    salt: Optional[str] = None


class ConfigModel(BaseModel):
    level: int = Field(3, ge=1, le=5)
    passes: PassesModel = PassesModel()
    cycles: int = Field(1, ge=1, le=5)
    string_encryption: bool = False
    fake_loops: int = Field(0, ge=0, le=50)
    symbol_obfuscation: SymbolObfuscationModel = SymbolObfuscationModel()




class IndirectCallsModel(BaseModel):
    enabled: bool = False
    obfuscate_stdlib: bool = True
    obfuscate_custom: bool = True


class ObfuscateRequest(BaseModel):
    source_code: str  # For backward compatibility - single file content
    filename: str
    platform: Platform = Platform.LINUX
    entrypoint_command: Optional[str] = Field(default="./a.out")
    config: ConfigModel = ConfigModel()
    report_formats: Optional[list[str]] = Field(default_factory=lambda: ["json", "markdown"])
    custom_flags: Optional[list[str]] = None
    custom_pass_plugin: Optional[str] = None
    source_files: Optional[List[SourceFile]] = None  # For multi-file projects (GitHub repos)
    repo_session_id: Optional[str] = None  # For GitHub repos cloned to backend
    # Build system configuration for complex projects (cmake, make, etc.)
    build_system: str = Field(default="simple", pattern="^(simple|cmake|make|autotools|custom)$")
    build_command: Optional[str] = None  # Custom build command (for "custom" mode)
    output_binary_path: Optional[str] = None  # Hint for where to find output binary
    cmake_options: Optional[str] = None  # Extra cmake flags like "-DBUILD_TESTING=OFF"


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
        flattening=payload.config.passes.flattening or detected_passes.get("flattening", False),
        substitution=payload.config.passes.substitution or detected_passes.get("substitution", False),
        bogus_control_flow=payload.config.passes.bogus_control_flow or detected_passes.get("boguscf", False),
        split=payload.config.passes.split or detected_passes.get("split", False),
    )
    symbol_obf = SymbolObfuscationConfiguration(
        enabled=payload.config.symbol_obfuscation.enabled,
        algorithm=payload.config.symbol_obfuscation.algorithm,
        hash_length=payload.config.symbol_obfuscation.hash_length,
        prefix_style=payload.config.symbol_obfuscation.prefix_style,
        salt=payload.config.symbol_obfuscation.salt,
    )
    advanced = AdvancedConfiguration(
        cycles=payload.config.cycles,
        string_encryption=payload.config.string_encryption,
        fake_loops=payload.config.fake_loops,
        symbol_obfuscation=symbol_obf,
    )
    # Auto-load plugin if passes are requested and no explicit plugin provided
    any_pass_requested = (
        passes.flattening or passes.substitution or passes.bogus_control_flow or passes.split
    )
    chosen_plugin = payload.custom_pass_plugin
    if any_pass_requested and not chosen_plugin and DEFAULT_PASS_PLUGIN_EXISTS:
        chosen_plugin = DEFAULT_PASS_PLUGIN_PATH

    output_config = ObfuscationConfig.from_dict(
        {
            "level": payload.config.level,
            "platform": payload.platform.value,
            "passes": {
                "flattening": passes.flattening,
                "substitution": passes.substitution,
                "bogus_control_flow": passes.bogus_control_flow,
                "split": passes.split,
            },
            "advanced": {
                "cycles": advanced.cycles,
                "string_encryption": advanced.string_encryption,
                "fake_loops": advanced.fake_loops,
                "symbol_obfuscation": {
                    "enabled": symbol_obf.enabled,
                    "algorithm": symbol_obf.algorithm,
                    "hash_length": symbol_obf.hash_length,
                    "prefix_style": symbol_obf.prefix_style,
                    "salt": symbol_obf.salt,
                },
            },
            "output": {
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
    "make": "make CC=$CC CXX=$CXX -j$(nproc)",
    "autotools": "./configure CC=$CC CXX=$CXX && make -j$(nproc)",
    "custom": None,  # Use user-provided build_command
}


def _setup_build_environment(config: ObfuscationConfig, plugin_path: Optional[str] = None) -> Dict[str, str]:
    """Set up environment variables to hijack the build and inject obfuscation."""
    env = os.environ.copy()

    # Point to our clang
    clang_path = shutil.which("clang") or "/usr/bin/clang"
    clangpp_path = shutil.which("clang++") or "/usr/bin/clang++"

    # Check for custom LLVM installation
    custom_llvm = Path("/usr/local/llvm-obfuscator/bin")
    if custom_llvm.exists():
        clang_path = str(custom_llvm / "clang")
        clangpp_path = str(custom_llvm / "clang++")

    env["CC"] = clang_path
    env["CXX"] = clangpp_path

    # Build OLLVM flags based on enabled passes
    ollvm_flags = []

    if plugin_path and Path(plugin_path).exists():
        if config.passes.flattening:
            ollvm_flags.extend(["-mllvm", "-fla"])
        if config.passes.substitution:
            ollvm_flags.extend(["-mllvm", "-sub"])
        if config.passes.bogus_control_flow:
            ollvm_flags.extend(["-mllvm", "-bcf"])
        if config.passes.split:
            ollvm_flags.extend(["-mllvm", "-split"])

        # Add plugin loading if any OLLVM passes enabled
        if ollvm_flags:
            plugin_flags = f"-Xclang -load -Xclang {plugin_path} " + " ".join(ollvm_flags)
        else:
            plugin_flags = ""
    else:
        plugin_flags = ""

    # Add compiler flags from config
    layer4_flags = " ".join(config.compiler_flags) if config.compiler_flags else ""

    env["CFLAGS"] = f"{plugin_flags} {layer4_flags}".strip()
    env["CXXFLAGS"] = env["CFLAGS"]

    # Also set LDFLAGS for linking
    env["LDFLAGS"] = layer4_flags

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

    # Find all source files
    all_sources: List[Path] = []
    for ext in source_extensions:
        all_sources.extend(project_root.rglob(ext))

    # Build shared symbol map for consistency across files
    symbol_map: Dict[str, str] = {}

    for source_file in all_sources:
        try:
            content = source_file.read_text(encoding='utf-8', errors='ignore')
            modified = content
            file_modified = False

            # Layer 1: Symbol obfuscation
            if config.advanced.symbol_obfuscation.enabled:
                new_content, file_symbols = _apply_symbol_obfuscation(
                    modified,
                    symbol_map,
                    config.advanced.symbol_obfuscation
                )
                if new_content != modified:
                    modified = new_content
                    results["symbols_renamed"] += file_symbols
                    file_modified = True

            # Layer 2: String encryption
            if config.advanced.string_encryption:
                new_content, strings_count = _apply_string_encryption(modified)
                if new_content != modified:
                    modified = new_content
                    results["strings_encrypted"] += strings_count
                    file_modified = True

            # Layer 2.5: Indirect call obfuscation
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
    timeout: int = 1800  # 30 minute timeout for large projects like CURL
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


def _run_obfuscation(job_id: str, source_path: Path, config: ObfuscationConfig) -> None:
    try:
        job_manager.update_job(job_id, status="running")
        progress_tracker.publish_sync(ProgressEvent(job_id=job_id, stage="running", progress=0.1, message="Compilation started"))
        result = obfuscator.obfuscate(source_path, config, job_id=job_id)
        job_manager.update_job(job_id, status="completed", result=result)
        job_manager.attach_reports(job_id, result.get("report_paths", {}))
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

    # Always compile for Linux, regardless of what user selected
    payload.platform = Platform.LINUX

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
                logger.info(f"Using custom build mode: {payload.build_system}")

                # Build config first to get obfuscation settings
                config = _build_config_from_request(payload, working_dir, project_root=repo_path)

                # Step 1: Apply source-level obfuscation IN-PLACE (Layers 1, 2, 2.5)
                logger.info("Applying source-level obfuscation to all files...")
                obf_results = _obfuscate_project_sources(repo_path, config, logger)
                logger.info(f"Source obfuscation complete: {obf_results}")

                # Step 2: Set up compiler hijacking for OLLVM passes (Layer 3) and flags (Layer 4)
                build_env = _setup_build_environment(
                    config,
                    plugin_path=DEFAULT_PASS_PLUGIN_PATH if DEFAULT_PASS_PLUGIN_EXISTS else None
                )

                # Step 3: Run the project's native build system
                logger.info("Running native build system...")
                build_success, build_output = _run_custom_build(
                    repo_path,
                    payload.build_system,
                    payload.build_command,
                    build_env,
                    logger,
                    cmake_options=payload.cmake_options
                )

                if not build_success:
                    raise HTTPException(status_code=500, detail=build_output)

                # Step 4: Find output binaries
                logger.info("Searching for output binaries...")
                binaries = _find_output_binaries(repo_path, payload.output_binary_path)

                if not binaries:
                    raise HTTPException(
                        status_code=500,
                        detail="Build succeeded but no output binaries found. "
                               "Try specifying the output path in 'Output Binary Path' field."
                    )

                # Copy the first binary to working directory
                output_binary = binaries[0]
                logger.info(f"Found binary: {output_binary}")

                final_binary = working_dir / f"obfuscated_{output_binary.name}"
                shutil.copy2(output_binary, final_binary)

                result = {
                    "output_file": str(final_binary),
                    "build_system": payload.build_system,
                    "source_obfuscation": obf_results,
                    "binaries_found": [str(b) for b in binaries],
                }

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

        else:
            # Single file mode (original behavior)
            source_filename = _sanitize_filename(payload.filename)
            source_path = (working_dir / source_filename).resolve()
            _decode_source(payload.source_code, source_path)
            config = _build_config_from_request(payload, working_dir)

            result = obfuscator.obfuscate(source_path, config, job_id=job.job_id)

        job_manager.update_job(job.job_id, status="completed", result=result)
        job_manager.attach_reports(job.job_id, result.get("report_paths", {}))

        binary_path = Path(result.get("output_file", ""))
        if not binary_path.exists():
            raise HTTPException(status_code=500, detail="Binary generation failed")

        return {
            "job_id": job.job_id,
            "status": "completed",
            "download_url": f"/api/download/{job.job_id}",
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
    background.add_task(_run_obfuscation, job.job_id, source_path, config)
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
    try:
        job = job_manager.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    report_path = job.report_paths.get(fmt)
    if not report_path:
        raise HTTPException(status_code=404, detail="Report not available")
    path = Path(report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")
    return FileResponse(path)


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
    download_urls = result.get("download_urls", {})
    if download_urls and platform in download_urls:
        binary_path = Path(download_urls[platform]["path"])
        if not binary_path.exists():
            raise HTTPException(status_code=404, detail=f"Binary not found for platform {platform}")

        return FileResponse(
            binary_path,
            media_type="application/octet-stream",
            filename=binary_path.name
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
    download_urls = result.get("download_urls", {})
    if download_urls:
        # Return Linux binary by default
        if "linux" in download_urls:
            binary_path = Path(download_urls["linux"]["path"])
        elif "windows" in download_urls:
            binary_path = Path(download_urls["windows"]["path"])
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
