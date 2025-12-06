"""Build system integration and entrypoint handler.

This module handles:
1. Build system detection (CMake, Autotools, Make, etc.)
2. Build execution with optional compiler wrapper (CC/CXX override)
3. Compile flags extraction from build output and compile_commands.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .exceptions import ObfuscationError
from .utils import create_logger

logger = create_logger(__name__)


class BuildError(ObfuscationError):
    """Exception raised when build fails."""
    pass


def process_entrypoint(
    command: str,
    project_root: Path,
    compiler_wrapper: Optional[str] = None,
    plugin_path: Optional[Path] = None,
    obf_opt_args: Optional[List[str]] = None,
) -> Tuple[bool, List[str], Optional[Path]]:
    """Execute build entrypoint and extract compile flags.

    Args:
        command: Build command to execute (or "auto" for auto-detection)
        project_root: Project root directory
        compiler_wrapper: Optional path to compiler wrapper (obf-clang)
        plugin_path: Optional path to OLLVM plugin (for wrapper env vars)
        obf_opt_args: Optional list of opt arguments for obfuscation passes

    Returns:
        Tuple of (success, flags_list, compile_commands_path)

    Raises:
        BuildError: If build fails critically
    """
    logger.info("=" * 80)
    logger.info("ENTRYPOINT HANDLER: Build System Integration")
    logger.info("=" * 80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Command: {command}")
    logger.info(f"Compiler wrapper: {compiler_wrapper or 'None (direct build)'}")
    logger.info("")

    # Auto-detect build system if command is "auto"
    if command == "auto":
        command = _detect_build_system(project_root)
        logger.info(f"Auto-detected build system: {command}")

    # Prepare environment variables
    env = os.environ.copy()

    # If compiler wrapper is provided, export CC/CXX to use it
    if compiler_wrapper:
        wrapper_path = Path(compiler_wrapper).resolve()
        if not wrapper_path.exists():
            logger.error(f"Compiler wrapper not found: {wrapper_path}")
            raise BuildError(f"Compiler wrapper not found: {wrapper_path}")

        # Make wrapper executable
        wrapper_path.chmod(wrapper_path.stat().st_mode | 0o111)

        # Export CC/CXX to wrapper
        env['CC'] = str(wrapper_path)
        env['CXX'] = str(wrapper_path)

        logger.info("━" * 80)
        logger.info("CUSTOM COMPILER CONFIGURED")
        logger.info("━" * 80)
        logger.info(f"  CC  = {env['CC']}")
        logger.info(f"  CXX = {env['CXX']}")
        logger.info("")
        logger.info("Build system will use specified compiler.")
        logger.info("━" * 80)
        logger.info("")

    # Execute build command
    logger.info("Executing build command...")
    logger.info(f"  Command: {command}")
    logger.info(f"  Working directory: {project_root}")
    logger.info("")

    build_log_path = project_root / "build.log"

    try:
        # Run build command with verbose output
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Write build log
        with build_log_path.open('w', encoding='utf-8') as f:
            f.write(f"Command: {command}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        logger.info(f"Build log written to: {build_log_path}")

        if result.returncode != 0:
            logger.error(f"Build command failed with exit code {result.returncode}")
            logger.error("Build output (last 50 lines):")
            for line in (result.stdout + result.stderr).split('\n')[-50:]:
                logger.error(f"  {line}")

            # Don't raise immediately, try to extract flags anyway
            logger.warning("Build failed, but attempting to extract flags from partial build...")
        else:
            logger.info("✓ Build command completed successfully")

    except subprocess.TimeoutExpired:
        logger.error("Build command timed out after 10 minutes")
        raise BuildError("Build command timed out")
    except Exception as e:
        logger.error(f"Build command execution failed: {e}")
        raise BuildError(f"Build execution failed: {e}")

    # Extract compile flags from various sources
    flags: Set[str] = set()
    compile_commands_path = None

    # 1. Try to read compile_commands.json
    cc_path = project_root / "compile_commands.json"
    if cc_path.exists():
        logger.info("")
        logger.info("✓ Found compile_commands.json")
        compile_commands_path = cc_path

        try:
            with cc_path.open('r', encoding='utf-8') as f:
                cc_data = json.load(f)

            logger.info(f"  Entries: {len(cc_data)}")

            # Extract flags from compile commands
            for entry in cc_data:
                if 'command' in entry:
                    extracted = _extract_relevant_flags_from_command(entry['command'])
                    flags.update(extracted)
                elif 'arguments' in entry:
                    extracted = _extract_relevant_flags_from_args(entry['arguments'])
                    flags.update(extracted)

            logger.info(f"  Extracted {len(flags)} unique flags")
        except Exception as e:
            logger.warning(f"Failed to parse compile_commands.json: {e}")
    else:
        logger.warning("✗ compile_commands.json not found")

    # 2. Try to extract flags from build output
    if build_log_path.exists():
        try:
            with build_log_path.open('r', encoding='utf-8') as f:
                build_output = f.read()

            output_flags = _extract_flags_from_build_output(build_output)
            if output_flags:
                logger.info(f"Extracted {len(output_flags)} flags from build output")
                flags.update(output_flags)
        except Exception as e:
            logger.warning(f"Failed to extract flags from build output: {e}")

    # Convert to sorted list
    flags_list = sorted(list(flags))

    logger.info("")
    logger.info("=" * 80)
    logger.info("ENTRYPOINT HANDLER: Summary")
    logger.info("=" * 80)
    logger.info(f"Total flags extracted: {len(flags_list)}")
    logger.info(f"compile_commands.json: {'✓ Found' if compile_commands_path else '✗ Not found'}")
    logger.info(f"Build status: {'✓ Success' if result.returncode == 0 else '✗ Failed'}")
    logger.info("=" * 80)
    logger.info("")

    success = result.returncode == 0 or len(flags_list) > 0
    return success, flags_list, compile_commands_path


def _detect_build_system(project_root: Path) -> str:
    """Auto-detect build system and return appropriate build command.

    Args:
        project_root: Project root directory

    Returns:
        Build command string
    """
    logger.info("Auto-detecting build system...")

    # Check for CMakeLists.txt
    if (project_root / "CMakeLists.txt").exists():
        logger.info("  Detected: CMake project")
        # Use cmake with compile_commands.json export
        return "cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build && cmake --build build"

    # Check for configure script (Autotools)
    if (project_root / "configure").exists():
        logger.info("  Detected: Autotools project (configure script)")
        return "./configure && make -j$(nproc)"

    # Check for configure.ac (need to run autogen/buildconf first)
    if (project_root / "configure.ac").exists() or (project_root / "configure.in").exists():
        logger.info("  Detected: Autotools project (needs autogen)")
        # Check for buildconf script (curl uses this)
        if (project_root / "buildconf").exists():
            return "./buildconf && ./configure && make -j$(nproc)"
        elif (project_root / "autogen.sh").exists():
            return "./autogen.sh && ./configure && make -j$(nproc)"
        else:
            return "autoreconf -fi && ./configure && make -j$(nproc)"

    # Check for Makefile
    if (project_root / "Makefile").exists():
        logger.info("  Detected: Make project")
        return "make -j$(nproc)"

    # Check for build.sh or similar scripts
    for script in ["build.sh", "build", "compile.sh"]:
        script_path = project_root / script
        if script_path.exists() and script_path.is_file():
            logger.info(f"  Detected: Build script ({script})")
            return f"./{script}"

    # Default fallback
    logger.warning("  Could not detect build system, using default 'make'")
    return "make"


def _extract_relevant_flags_from_command(command: str) -> List[str]:
    """Extract relevant compiler flags from a command string.

    Only extracts: -I, -D, -isystem, --sysroot, -std=

    Args:
        command: Full compiler command string

    Returns:
        List of extracted flags
    """
    flags: List[str] = []

    # Split command into tokens
    tokens = command.split()

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # -I with space separator
        if token == '-I' and i + 1 < len(tokens):
            flags.append(token)
            flags.append(tokens[i + 1])
            i += 2
        # -I with no space
        elif token.startswith('-I'):
            flags.append(token)
            i += 1
        # -D with space separator
        elif token == '-D' and i + 1 < len(tokens):
            flags.append(token)
            flags.append(tokens[i + 1])
            i += 2
        # -D with no space
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
        # -std=
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


def _extract_flags_from_build_output(build_output: str) -> List[str]:
    """Extract compiler flags from build output/logs.

    Args:
        build_output: Build log content

    Returns:
        List of extracted flags
    """
    import re

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
