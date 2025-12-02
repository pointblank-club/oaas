from __future__ import annotations

import base64
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .exceptions import ObfuscationError, ToolchainNotFoundError

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(command: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    logger.debug("Executing command: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    logger.debug("Command stdout: %s", stdout)
    if stderr:
        logger.debug("Command stderr: %s", stderr)
    if process.returncode != 0:
        raise ObfuscationError(f"Command failed with exit code {process.returncode}: {' '.join(command)}\n{stderr}")
    return process.returncode, stdout, stderr


def tool_exists(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def require_tool(tool_name: str) -> None:
    if not tool_exists(tool_name):
        raise ToolchainNotFoundError(f"Required tool '{tool_name}' not found in PATH")


def get_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def detect_binary_format(binary_path: Path) -> str:
    if not binary_path.exists():
        return "unknown"
    with binary_path.open("rb") as f:
        magic = f.read(4)
    if magic.startswith(b"\x7fELF"):
        return "ELF"
    if magic[:2] in (b"MZ", b"ZM"):
        return "PE"
    return "unknown"


def compute_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    from math import log2

    entropy = 0.0
    length = len(data)
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    for count in counts:
        if count == 0:
            continue
        p = count / length
        entropy -= p * log2(p)
    return round(entropy, 3)


def get_file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def base64_to_file(content_b64: str, destination: Path) -> None:
    decoded = base64.b64decode(content_b64)
    destination.write_bytes(decoded)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def create_temp_directory(prefix: str = "obf-") -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix=prefix)


def get_platform_triple(target_platform: str) -> str:
    system = target_platform.lower()
    if system == "linux":
        return "x86_64-pc-linux-gnu"
    if system == "windows":
        return "x86_64-pc-windows-msvc"
    return platform.machine()


def merge_flags(base: Iterable[str], extra: Optional[Iterable[str]] = None) -> List[str]:
    merged = list(base)
    if extra:
        for flag in extra:
            if flag not in merged:
                merged.append(flag)
    return merged


def normalize_flags_and_passes(flags: Iterable[str]) -> Tuple[List[str], Dict[str, bool]]:
    """Split pass-like flags out of compiler flags.

    Recognizes common obfuscation pass indicators (flattening, substitution,
    boguscf, split) in various forms and returns a tuple of:
      - sanitized compiler flags (with pass flags removed)
      - a mapping of pass name -> enabled bool

    Examples of recognized inputs (case-sensitive as used typically):
      - "-fla", "-flattening", "flattening"
      - "-sub", "-substitution", "substitution"
      - "-bcf", "-boguscf", "boguscf"
      - "-split", "split"
      - sequences like "-mllvm", "-fla" (will be stripped and converted)
    """
    known_aliases = {
        "flattening": {"-fla", "-flattening", "flattening"},
        "substitution": {"-sub", "-substitution", "substitution"},
        "boguscf": {"-bcf", "-boguscf", "boguscf"},
        "split": {"-split", "split"},
    }

    pass_enabled: Dict[str, bool] = {k: False for k in known_aliases.keys()}
    cleaned: List[str] = []

    flags_list = list(flags)
    i = 0
    while i < len(flags_list):
        token = flags_list[i]
        # Handle "-mllvm <pass-flag>" pattern by consuming the next token
        if token == "-mllvm" and i + 1 < len(flags_list):
            next_tok = flags_list[i + 1]
            matched = False
            for pname, aliases in known_aliases.items():
                if next_tok in aliases:
                    pass_enabled[pname] = True
                    matched = True
                    break
            # Skip both tokens if matched; else keep both
            if matched:
                i += 2
                continue
            cleaned.append(token)
            # keep the next token as well since it's not a known alias
            cleaned.append(next_tok)
            i += 2
            continue

        # Standalone alias tokens
        matched = False
        for pname, aliases in known_aliases.items():
            if token in aliases:
                pass_enabled[pname] = True
                matched = True
                break
        if matched:
            i += 1
            continue

        # Everything else is a normal compiler/linker flag
        cleaned.append(token)
        i += 1

    return cleaned, pass_enabled


def summarize_symbols(binary_path: Path) -> Tuple[int, int]:
    if not binary_path.exists():
        return 0, 0
    nm_tool = "llvm-nm" if tool_exists("llvm-nm") else "nm"
    try:
        _, stdout, _ = run_command([nm_tool, str(binary_path)])
    except ObfuscationError:
        return 0, 0
    symbols = stdout.splitlines()
    functions = [line for line in symbols if " T " in line or " t " in line]
    return len(symbols), len(functions)


def write_html(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def write_pdf_placeholder(path: Path) -> None:
    ensure_directory(path.parent)
    placeholder = "PDF generation not implemented in sandbox."
    path.write_text(placeholder, encoding="utf-8")


def list_sections(binary_path: Path) -> Dict[str, int]:
    if not binary_path.exists():
        return {}
    objdump = "llvm-objdump" if tool_exists("llvm-objdump") else "objdump"
    try:
        _, stdout, _ = run_command([objdump, "-h", str(binary_path)])
    except ObfuscationError:
        return {}
    sections: Dict[str, int] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            name = parts[1]
            size = int(parts[2], 16)
            sections[name] = size
    return sections


def current_platform() -> str:
    return platform.system().lower()


def is_windows_platform() -> bool:
    return current_platform().startswith("win")


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def load_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: Dict) -> None:
    import yaml

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def detect_project_compile_flags(project_root: Path, entrypoint_command: Optional[str] = None) -> List[str]:
    """Automatically detect compile flags for a multi-file project.
    
    This function runs the FULL build sequence to extract compile flags.
    For large projects like curl, this means:
    1. Running ./buildconf && ./configure && make (for Autotools)
    2. Or cmake -B build && cmake --build build (for CMake)
    3. Capturing verbose compiler invocations
    4. Extracting -I, -D, -isystem, --sysroot, -std= flags
    
    Detection strategy (in order):
    1. If entrypoint command provided: Run FULL build to generate compile_commands.json
    2. Read existing compile_commands.json if present
    3. Use pkg-config for known libraries (curl, openssl, zlib, etc.)
    4. Scan project for header directories and generate -I flags
    5. Fallback: detect generated headers (config.h, curl_config.h) and add -DHAVE_CONFIG_H
    
    Args:
        project_root: Root directory of the project
        entrypoint_command: Optional build command to run (e.g., "make", "cmake --build build")
        
    Returns:
        List of compiler flags (includes -I, -D, --sysroot, etc.)
        
    Raises:
        ObfuscationError: If build fails and we cannot extract sufficient flags
    """
    logger.info(f"Auto-detecting compile flags for project: {project_root}")
    flags: Set[str] = set()
    build_succeeded = False
    
    # Strategy 0: Run FULL build sequence to extract flags
    # This is the primary strategy for large projects like curl
    try:
        from .entrypoint_handler import process_entrypoint, BuildError
        
        # Always try to run the build, even without explicit entrypoint command
        # The entrypoint handler will auto-detect the build system
        effective_command = entrypoint_command or "auto"
        logger.info(f"Running full build sequence (command: {effective_command})")
        
        success, entrypoint_flags, compile_commands_path = process_entrypoint(
            effective_command, project_root
        )
        
        if success and entrypoint_flags:
            logger.info(f"Extracted {len(entrypoint_flags)} flags from build")
            flags.update(entrypoint_flags)
            build_succeeded = True
            
            # If compile_commands.json was generated and we have good flags, return early
            if compile_commands_path and compile_commands_path.exists() and len(entrypoint_flags) >= 3:
                logger.info("compile_commands.json generated successfully by build")
                return sorted(list(flags))
        else:
            logger.warning("Build did not produce compile flags, trying fallback strategies")
            
    except ImportError:
        logger.warning("entrypoint_handler not available, skipping build")
    except Exception as e:
        # Log the error but continue with fallback strategies
        logger.warning(f"Build failed: {e}")
        logger.warning("Continuing with fallback strategies...")
    
    # Strategy 1: Try existing compile_commands.json
    compile_commands_flags = _read_compile_commands_json(project_root)
    if compile_commands_flags:
        logger.info(f"Found {len(compile_commands_flags)} flags from compile_commands.json")
        flags.update(compile_commands_flags)
    
    # Strategy 2: Use pkg-config for known libraries
    pkg_config_flags = _detect_pkg_config_flags(project_root)
    if pkg_config_flags:
        logger.info(f"Found {len(pkg_config_flags)} flags from pkg-config")
        flags.update(pkg_config_flags)
    
    # Strategy 3: Scan for header directories
    include_flags = _scan_for_include_paths(project_root)
    if include_flags:
        logger.info(f"Found {len(include_flags)} include paths in project")
        flags.update(include_flags)
    
    # Strategy 4: Fallback - detect generated headers
    if not flags or len(flags) < 3:
        logger.info("Trying fallback: detecting generated config headers")
        fallback_flags = _detect_generated_headers(project_root)
        if fallback_flags:
            logger.info(f"Found {len(fallback_flags)} flags from generated headers")
            flags.update(fallback_flags)
    
    # Deduplicate and sort flags
    final_flags = sorted(list(flags))
    logger.info(f"Total auto-detected flags: {len(final_flags)}")
    
    # If we still have no flags and build didn't succeed, this is a problem
    if not final_flags and not build_succeeded:
        logger.error("Failed to extract any compile flags from project")
        logger.error("The project may not have been built correctly")
    
    return final_flags


def _detect_generated_headers(project_root: Path) -> List[str]:
    """Detect generated config headers and create flags for them.
    
    This is a fallback strategy when the build fails or doesn't produce
    compile_commands.json. It looks for common generated headers like:
    - config.h
    - curl_config.h
    - *_config.h
    
    Args:
        project_root: Project root directory
        
    Returns:
        List of flags for generated headers
    """
    flags: List[str] = []
    
    # Common generated config header patterns
    config_patterns = [
        "config.h",
        "*_config.h",
        "include/*/config.h",
        "lib/*/config.h",
        "src/config.h",
    ]
    
    found_config = False
    
    for pattern in config_patterns:
        for header in project_root.rglob(pattern):
            if header.is_file():
                # Add the directory containing the header
                include_dir = header.parent
                flags.append(f"-I{include_dir}")
                
                # Add -DHAVE_CONFIG_H if this looks like a config.h
                if "config.h" in header.name.lower():
                    found_config = True
                
                logger.debug(f"Found generated header: {header}")
    
    if found_config:
        flags.append("-DHAVE_CONFIG_H")
    
    # Add common include directories that might be needed
    common_dirs = ["include", "src", "lib", "source", "inc"]
    for dir_name in common_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            flags.append(f"-I{dir_path}")
    
    # Always add project root
    flags.append(f"-I{project_root}")
    
    return list(set(flags))


def _read_compile_commands_json(project_root: Path) -> List[str]:
    """Read compile flags from compile_commands.json if present.
    
    Args:
        project_root: Project root directory
        
    Returns:
        List of extracted compiler flags
    """
    compile_commands_path = project_root / "compile_commands.json"
    
    if not compile_commands_path.exists():
        logger.debug("compile_commands.json not found")
        return []
    
    try:
        with compile_commands_path.open('r', encoding='utf-8') as f:
            compile_commands = json.load(f)
        
        flags: Set[str] = set()
        
        for entry in compile_commands:
            if 'command' in entry:
                # Parse command string to extract flags
                command = entry['command']
                extracted = _extract_relevant_flags_from_command(command)
                flags.update(extracted)
            elif 'arguments' in entry:
                # Arguments already split
                args = entry['arguments']
                extracted = _extract_relevant_flags_from_args(args)
                flags.update(extracted)
        
        logger.info(f"Extracted {len(flags)} unique flags from compile_commands.json")
        return list(flags)
        
    except Exception as e:
        logger.warning(f"Failed to parse compile_commands.json: {e}")
        return []


def _extract_relevant_flags_from_command(command: str) -> List[str]:
    """Extract relevant compiler flags from a command string.
    
    Only extracts: -I, -D, -isystem, --sysroot, -std=
    
    Args:
        command: Full compiler command string
        
    Returns:
        List of extracted flags
    """
    flags: List[str] = []
    
    # Split command into tokens (simple split by space, doesn't handle all quoting)
    tokens = command.split()
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # -I with space separator (e.g., "-I /path")
        if token == '-I' and i + 1 < len(tokens):
            flags.append(token)
            flags.append(tokens[i + 1])
            i += 2
        # -I with no space (e.g., "-I/path")
        elif token.startswith('-I'):
            flags.append(token)
            i += 1
        # -D with space separator
        elif token == '-D' and i + 1 < len(tokens):
            flags.append(token)
            flags.append(tokens[i + 1])
            i += 2
        # -D with no space (e.g., "-DDEBUG")
        elif token.startswith('-D'):
            flags.append(token)
            i += 1
        # -isystem
        elif token == '-isystem' and i + 1 < len(tokens):
            flags.append(token)
            flags.append(tokens[i + 1])
            i += 2
        elif token.startswith('-isystem'):
            flags.append(token)
            i += 1
        # --sysroot
        elif token == '--sysroot' and i + 1 < len(tokens):
            flags.append(token)
            flags.append(tokens[i + 1])
            i += 2
        elif token.startswith('--sysroot='):
            flags.append(token)
            i += 1
        # -std= (C/C++ standard)
        elif token.startswith('-std='):
            flags.append(token)
            i += 1
        else:
            i += 1
    
    return flags


def _extract_relevant_flags_from_args(args: List[str]) -> List[str]:
    """Extract relevant compiler flags from argument list.
    
    Args:
        args: List of compiler arguments
        
    Returns:
        List of extracted relevant flags
    """
    flags: List[str] = []
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        # Check for flags we want to preserve
        if arg in ['-I', '-D', '-isystem', '--sysroot']:
            flags.append(arg)
            if i + 1 < len(args):
                flags.append(args[i + 1])
                i += 2
            else:
                i += 1
        elif arg.startswith(('-I', '-D', '-isystem', '--sysroot=', '-std=')):
            flags.append(arg)
            i += 1
        else:
            i += 1
    
    return flags


def _detect_pkg_config_flags(project_root: Path) -> List[str]:
    """Detect compiler flags using pkg-config for known libraries.
    
    Checks for commonly used libraries: curl, openssl, zlib, libxml2, etc.
    
    Args:
        project_root: Project root directory
        
    Returns:
        List of compiler flags from pkg-config
    """
    # Check if pkg-config is available
    if not shutil.which('pkg-config'):
        logger.debug("pkg-config not found, skipping library detection")
        return []
    
    # List of common libraries to check
    known_libs = [
        'libcurl',
        'openssl',
        'zlib',
        'libxml-2.0',
        'sqlite3',
        'libpng',
        'libjpeg',
        'ncurses',
        'readline',
        'glib-2.0',
        'gtk+-3.0',
        'libssl',
        'libcrypto',
    ]
    
    flags: Set[str] = set()
    
    for lib in known_libs:
        try:
            # Check if library exists using pkg-config
            result = subprocess.run(
                ['pkg-config', '--exists', lib],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Library found, get its compile flags
                result = subprocess.run(
                    ['pkg-config', '--cflags', lib],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    lib_flags = result.stdout.strip().split()
                    flags.update(lib_flags)
                    logger.info(f"Found library '{lib}' via pkg-config: {len(lib_flags)} flags")
                    
        except Exception as e:
            logger.debug(f"Error checking pkg-config for {lib}: {e}")
            continue
    
    return list(flags)


def _scan_for_include_paths(project_root: Path) -> List[str]:
    """Scan project directory for header files and generate -I flags.
    
    Looks for directories containing .h or .hpp files, with priority for
    common directory names like 'include', 'inc', 'headers', etc.
    
    Special handling for library directories like 'include/curl', 'include/openssl':
    - Adds the parent 'include' directory instead of the subdirectory
    - This allows #include <curl/header.h> to work correctly
    
    Args:
        project_root: Project root directory
        
    Returns:
        List of -I flags for discovered include directories
    """
    include_dirs: Set[Path] = set()
    
    # Priority directories to check first
    priority_dirs = ['include', 'inc', 'includes', 'headers', 'src', 'lib']
    
    # Check priority directories
    for dir_name in priority_dirs:
        potential_dir = project_root / dir_name
        if potential_dir.exists() and potential_dir.is_dir():
            # Check if it contains header files (directly or in subdirectories)
            if _contains_headers(potential_dir) or _has_header_subdirs(potential_dir):
                include_dirs.add(potential_dir)
                logger.debug(f"Found include directory (priority): {potential_dir.name}")
    
    # Scan entire project for directories with headers
    # But be smart about library-style includes (e.g., curl/, openssl/)
    header_dirs: Set[Path] = set()
    
    for path in project_root.rglob('*.h'):
        parent = path.parent
        # Don't go more than 3 levels deep
        try:
            rel_path = parent.relative_to(project_root)
            if len(rel_path.parts) <= 3:
                header_dirs.add(parent)
        except ValueError:
            pass
    
    for path in project_root.rglob('*.hpp'):
        parent = path.parent
        try:
            rel_path = parent.relative_to(project_root)
            if len(rel_path.parts) <= 3:
                header_dirs.add(parent)
        except ValueError:
            pass
    
    # Process header directories with smart parent detection
    for header_dir in header_dirs:
        # Check if this looks like a library subdirectory (e.g., include/curl)
        # Common patterns: include/libname, include/libname/*, lib/libname, etc.
        parent = header_dir.parent
        dir_name = header_dir.name.lower()
        parent_name = parent.name.lower() if parent != project_root else ""
        
        # If parent is 'include', 'inc', 'includes', use parent instead
        if parent_name in ['include', 'inc', 'includes', 'headers']:
            include_dirs.add(parent)
            logger.debug(f"Using parent directory for {header_dir.name}: {parent.name}")
        # If this is a well-known library directory, use parent
        elif dir_name in ['curl', 'openssl', 'ssl', 'zlib', 'sqlite', 'gtk', 'glib', 'libxml2', 'json']:
            include_dirs.add(parent)
            logger.debug(f"Detected library directory {dir_name}, using parent: {parent}")
        else:
            # Add the directory itself
            include_dirs.add(header_dir)
    
    # Convert to -I flags
    flags = [f"-I{str(d)}" for d in sorted(include_dirs)]
    
    return flags


def _contains_headers(directory: Path) -> bool:
    """Check if a directory contains header files.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if directory contains .h or .hpp files
    """
    for ext in ['*.h', '*.hpp', '*.hxx', '*.h++']:
        if any(directory.glob(ext)):
            return True
    return False


def _has_header_subdirs(directory: Path) -> bool:
    """Check if a directory has subdirectories containing header files.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if directory has subdirs with .h or .hpp files
    """
    if not directory.is_dir():
        return False
    
    for subdir in directory.iterdir():
        if subdir.is_dir() and _contains_headers(subdir):
            return True
    
    return False


def extract_build_flags_from_entrypoint(entrypoint_command: str, project_root: Path) -> List[str]:
    """Extract compiler flags by analyzing build output from entrypoint command.
    
    This function attempts to run the build command (if safe) and extract
    compiler flags from the build output.
    
    Args:
        entrypoint_command: User-provided build/run command
        project_root: Project root directory
        
    Returns:
        List of extracted compiler flags
    """
    logger.info(f"Attempting to extract build flags from entrypoint: {entrypoint_command}")
    
    # Common build commands that might reveal compiler flags
    build_indicators = ['make', 'cmake', 'configure', 'build', 'gcc', 'clang', 'g++']
    
    # Check if entrypoint looks like it might trigger a build
    is_build_command = any(indicator in entrypoint_command.lower() for indicator in build_indicators)
    
    if not is_build_command:
        logger.debug("Entrypoint doesn't appear to be a build command")
        return []
    
    # For safety, only attempt to parse existing build logs
    # Don't actually run user commands
    logger.debug("Checking for existing build logs")
    
    build_log_paths = [
        project_root / 'build.log',
        project_root / 'make.log',
        project_root / 'cmake_build.log',
        project_root / 'config.log',
    ]
    
    for log_path in build_log_paths:
        if log_path.exists():
            logger.info(f"Found build log: {log_path}")
            try:
                with log_path.open('r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    flags = _extract_flags_from_build_output(log_content)
                    if flags:
                        logger.info(f"Extracted {len(flags)} flags from {log_path.name}")
                        return flags
            except Exception as e:
                logger.warning(f"Failed to read build log {log_path}: {e}")
    
    return []


def _extract_flags_from_build_output(build_output: str) -> List[str]:
    """Extract compiler flags from build output/logs.
    
    Looks for compiler invocations and extracts relevant flags.
    
    Args:
        build_output: Build log content
        
    Returns:
        List of extracted flags
    """
    flags: Set[str] = set()
    
    # Find lines that look like compiler invocations
    compiler_patterns = [
        r'gcc\s+.*',
        r'g\+\+\s+.*',
        r'clang\s+.*',
        r'clang\+\+\s+.*',
        r'cc\s+.*',
        r'c\+\+\s+.*',
    ]
    
    for line in build_output.split('\n'):
        for pattern in compiler_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Extract flags from this compiler invocation
                extracted = _extract_relevant_flags_from_command(line)
                flags.update(extracted)
    
    return list(flags)
