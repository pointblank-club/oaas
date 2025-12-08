#!/usr/bin/env python3
"""
FEATURE #1: Safe Windows Binary Compilation for CFG Lifting

This module handles the first stage of the binary obfuscation pipeline:
converting user-submitted C source code into a safe, unoptimized Windows PE binary
suitable for Ghidra-based CFG reconstruction via McSema.

WHY THIS APPROACH?
==================
McSema + Ghidra CFG lifting requires deterministic, simple binaries. Optimizations,
complex language features, and Windows-specific metadata can break CFG recovery.

KEY COMPILATION FLAGS EXPLAINED:
- -O0: CRITICAL. Disables ALL optimizations. Optimized code has merged basic blocks,
  inlined functions, and eliminated dead code—making CFG reconstruction impossible.
  McSema needs 1-to-1 mapping between source and IR.

- -g: CRITICAL. Emits DWARF debug symbols (even on Windows PE). Helps Ghidra identify
  function boundaries and variable scope. Without this, Ghidra struggles to recover CFG.

- -fno-asynchronous-unwind-tables: CRITICAL. Windows uses SEH (Structured Exception Handling)
  tables instead of DWARF unwind info. These tables contain pointers/offsets that confuse
  McSema's IR lifting. Disabling them keeps the binary simpler.

- -fno-exceptions: CRITICAL. C++ EH and Windows EH metadata create hidden control flow
  (exception handlers). McSema cannot model exception control flow deterministically.
  By forbidding exceptions, we ensure all control flow is explicit.

- -fno-stack-protector: IMPORTANT. Windows stack cookies (__security_cookie) add runtime
  checks that McSema cannot lift correctly. Disabling keeps IR clean.

- -fno-inline: CRITICAL. Function inlining destroys CFG structure. McSema needs
  function boundaries to be explicit in the binary.

NO LTO, NO SIMD, NO link-time optimizations.

LIMITATIONS & CONSTRAINTS:
==========================
This pipeline stage enforces restrictions:
- NO recursion (breaks McSema's context-insensitive CFG)
- NO inline assembly (cannot be lifted)
- NO switch statements (Ghidra recovers them poorly as jump tables)
- NO C++ constructs (vtables, EH, namespaces break lifting)
- NO function pointers (indirect calls cannot be resolved statically)

These are NOT permanent limitations—they reflect Stage 1 constraints.
Stage 2 will add incremental support for these features.

PIPELINE FLOW:
==============
1. User submits C code
2. THIS SCRIPT: Validate + compile to safe Windows PE
3. NEXT STAGE: Ghidra lifter (Docker) exports .cfg file
4. LATER: McSema IR generation + obfuscation + recompilation
"""

import os
import sys
import json
import tempfile
import subprocess
import re
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SourceCodeValidator:
    """
    Validates C source code before compilation.
    Enforces restrictions required for safe CFG lifting.
    """

    # Patterns that indicate forbidden language constructs
    FORBIDDEN_PATTERNS = {
        'recursion': r'\b(\w+)\s*\([^)]*\)\s*\{[^}]*\1\s*\(',
        'inline_asm': r'(__asm__|asm\s*\{|__inline__|inline\s+)',
        'switch_statement': r'\bswitch\s*\(',
        'cpp_construct': r'(class\s+|namespace\s+|::|virtual\s+|public:|private:|protected:)',
        'function_pointer': r'(\(\s*\*\s*\w+\s*\)|\bfnptr\b)',
        'variadic_function': r'\.\.\.',
    }

    FORBIDDEN_KEYWORDS = {
        'goto',           # breaks CFG recovery
        'setjmp',         # exception handling
        'longjmp',        # non-local jumps
        'sigsetjmp',      # signal handling
        'signal',         # signal handlers
        'exception_ptr',  # C++ EH
        'catch',          # C++ EH
        'throw',          # C++ EH
    }

    @staticmethod
    def validate(source_code: str) -> Tuple[bool, List[str]]:
        """
        Validates source code against forbidden patterns.

        Args:
            source_code: Raw C source code string

        Returns:
            (is_valid: bool, errors: List[str])
        """
        errors = []

        # Check for forbidden keywords
        for keyword in SourceCodeValidator.FORBIDDEN_KEYWORDS:
            if re.search(rf'\b{keyword}\b', source_code, re.IGNORECASE):
                errors.append(
                    f"Forbidden keyword '{keyword}' detected. "
                    f"Reason: Cannot be lifted by McSema."
                )

        # Check for forbidden patterns
        for pattern_name, pattern in SourceCodeValidator.FORBIDDEN_PATTERNS.items():
            if re.search(pattern, source_code, re.MULTILINE):
                error_msg = {
                    'recursion': "Recursion detected. McSema cannot resolve recursive calls in CFG.",
                    'inline_asm': "Inline assembly detected. McSema cannot lift assembly code.",
                    'switch_statement': "Switch statement detected. Ghidra recovers jump tables unreliably.",
                    'cpp_construct': "C++ construct detected. Use C only; C++ EH breaks CFG lifting.",
                    'function_pointer': "Function pointer detected. Indirect calls cannot be resolved statically.",
                    'variadic_function': "Variadic function detected. Cannot be lifted reliably.",
                }.get(pattern_name, f"Pattern '{pattern_name}' detected.")
                errors.append(error_msg)

        return len(errors) == 0, errors


class WindowsBinaryCompiler:
    """
    Compiles validated C source into a safe Windows PE binary.
    Uses system MinGW-w64 for cross-compilation.

    The backend container has mingw-w64 installed (x86_64-w64-mingw32-gcc),
    along with all necessary cross-compilation tooling.
    """

    # Compilation flags explained in module docstring
    SAFE_COMPILATION_FLAGS = [
        '-O0',                              # No optimizations (CFG must match source)
        '-g',                               # Debug symbols (helps Ghidra identify functions)
        '-fno-asynchronous-unwind-tables',  # Disable SEH tables (confuse McSema)
        '-fno-exceptions',                  # Disable exception handling
        '-fno-stack-protector',             # No stack cookies (McSema cannot lift)
        '-fno-inline',                      # Keep functions explicit (no inlining)
        '-fno-optimize-sibling-calls',      # No tail call optimization
        '-march=x86-64',                    # Target x86-64 architecture
        '-static',                          # Static linking (no runtime dependencies)
    ]

    def __init__(self):
        """
        Initialize compiler (MinGW-w64 from system PATH).
        Verifies x86_64-w64-mingw32-gcc is available.
        """
        self.mingw_gcc = 'x86_64-w64-mingw32-gcc'
        self.mingw_gxx = 'x86_64-w64-mingw32-g++'
        self._verify_compiler_available()

    def _verify_compiler_available(self) -> None:
        """Check if MinGW-w64 toolchain is available in PATH."""
        try:
            result = subprocess.run(
                [self.mingw_gcc, '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"MinGW-w64 compiler available: {result.stdout.split()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_msg = (
                f"MinGW-w64 not found in PATH. "
                f"Please install via: apt-get install mingw-w64"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def compile(self, source_file: str, output_file: str) -> Tuple[bool, Optional[str]]:
        """
        Compile C source to Windows PE binary.

        Args:
            source_file: Path to .c source file
            output_file: Path to output .exe

        Returns:
            (success: bool, error_message: Optional[str])
        """
        cmd = self._build_compile_command(source_file, output_file)

        try:
            logger.info(f"Compiling: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                error_msg = f"Compilation failed:\n{result.stderr}"
                logger.error(error_msg)
                return False, error_msg

            if not os.path.exists(output_file):
                error_msg = f"Compilation succeeded but output file not found: {output_file}"
                logger.error(error_msg)
                return False, error_msg

            logger.info(f"Compilation successful: {output_file}")
            return True, None

        except subprocess.TimeoutExpired:
            error_msg = "Compilation timeout (60s exceeded)"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected compilation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def _build_compile_command(self, source_file: str, output_file: str) -> List[str]:
        """Build the compilation command using MinGW-w64."""
        return [
            self.mingw_gcc,
            *self.SAFE_COMPILATION_FLAGS,
            '-o', output_file,
            source_file,
        ]


class CompilationMetadata:
    """Generates metadata describing compilation parameters and constraints."""

    @staticmethod
    def generate(
        source_file: str,
        output_file: str,
        source_hash: str,
    ) -> Dict:
        """
        Generate compilation metadata.

        Returns:
            Dictionary with metadata
        """
        return {
            'stage': 'windows_binary_generation',
            'purpose': 'Safe unoptimized PE for CFG lifting via Ghidra+McSema',
            'source_file': source_file,
            'output_file': output_file,
            'source_hash': source_hash,
            'compiler': 'x86_64-w64-mingw32-gcc (MinGW-w64)',
            'target': 'x86_64-w64-mingw32',
            'compilation_flags': WindowsBinaryCompiler.SAFE_COMPILATION_FLAGS,
            'constraints': {
                'no_recursion': True,
                'no_inline_assembly': True,
                'no_switch_statements': True,
                'no_cpp_constructs': True,
                'no_function_pointers': True,
                'no_variadic_functions': True,
                'optimization_level': 'O0',
                'debug_info': 'enabled',
            },
            'next_stage': 'READY_FOR_GHIDRA_LIFTER',
            'next_action': 'Binary must be passed to Ghidra lifter (Docker) to export .cfg file',
            'warnings': [
                'Ghidra lifter is less reliable than IDA Pro.',
                'SEH tables, C++ EH, Windows ABI thunks can break lifting.',
                'McSema cannot yet handle complex real-world Windows binaries.',
                'This is Stage 1 only—restrictions will be relaxed in future stages.',
                'C++, switch statements, recursion, and function pointers unsupported.',
            ]
        }


def process_user_source(source_code: str, output_dir: str) -> Tuple[bool, Dict]:
    """
    Main pipeline function: validate and compile user source to Windows PE.

    Args:
        source_code: Raw C source code
        output_dir: Directory to write output files

    Returns:
        (success: bool, result_dict: Dict with output info)
    """
    result = {
        'success': False,
        'error': None,
        'output_binary': None,
        'metadata_file': None,
        'next_stage': None,
    }

    # Step 1: Validate source code
    logger.info("Step 1: Validating source code...")
    is_valid, validation_errors = SourceCodeValidator.validate(source_code)

    if not is_valid:
        result['error'] = 'Validation failed: ' + '; '.join(validation_errors)
        logger.error(result['error'])
        return False, result

    logger.info("Validation passed")

    # Step 2: Create temp directory with randomized name
    logger.info("Step 2: Creating temporary build directory...")
    source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:8]
    temp_dir = tempfile.mkdtemp(prefix=f'oaas_windows_build_{source_hash}_')
    logger.info(f"Build directory: {temp_dir}")

    try:
        source_file = os.path.join(temp_dir, 'program.c')
        output_file = os.path.join(temp_dir, 'program.exe')

        # Write source to temp file
        with open(source_file, 'w') as f:
            f.write(source_code)
        logger.info(f"Source written to: {source_file}")

        # Step 3: Compile to Windows PE
        logger.info("Step 3: Compiling to Windows PE binary...")
        compiler = WindowsBinaryCompiler()
        success, error_msg = compiler.compile(source_file, output_file)

        if not success:
            result['error'] = error_msg
            return False, result

        # Step 4: Copy binary to output directory
        logger.info("Step 4: Finalizing output...")
        os.makedirs(output_dir, exist_ok=True)
        final_binary = os.path.join(output_dir, 'program.exe')
        with open(output_file, 'rb') as src:
            with open(final_binary, 'wb') as dst:
                dst.write(src.read())
        logger.info(f"Binary written to: {final_binary}")

        # Step 5: Generate metadata
        logger.info("Step 5: Generating compilation metadata...")
        metadata = CompilationMetadata.generate(
            source_file=source_file,
            output_file=final_binary,
            source_hash=source_hash,
        )

        metadata_file = os.path.join(output_dir, 'compilation_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata written to: {metadata_file}")

        # Success
        result['success'] = True
        result['output_binary'] = final_binary
        result['metadata_file'] = metadata_file
        result['next_stage'] = 'READY_FOR_GHIDRA_LIFTER'
        result['next_action'] = f'Binary at {final_binary} is ready for Ghidra lifter'

        logger.info("✓ Windows binary compilation complete")
        return True, result

    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        logger.error(result['error'], exc_info=True)
        return False, result
    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not cleanup temp directory: {e}")


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python compile_windows_binary.py <source_file> [output_dir]")
        sys.exit(1)

    source_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './windows_binaries'

    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        sys.exit(1)

    with open(source_file, 'r') as f:
        source_code = f.read()

    success, result = process_user_source(source_code, output_dir)

    if success:
        print(json.dumps(result, indent=2))
        sys.exit(0)
    else:
        print(json.dumps(result, indent=2))
        sys.exit(1)
