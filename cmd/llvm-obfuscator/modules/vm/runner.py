"""VM Obfuscation Runner - Subprocess wrapper with timeout protection."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VMResult:
    """Result of VM obfuscation execution.

    Attributes:
        success: Whether virtualization succeeded.
        output_path: Path to the virtualized IR file, or None on failure.
        functions_virtualized: List of function names that were virtualized.
        error: Error message if failed, None on success.
        metrics: Optional dict with bytecode_size, opcodes_used, etc.
    """
    success: bool
    output_path: Optional[Path] = None
    functions_virtualized: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metrics: Optional[Dict] = None


def run_vm_isolated(
    input_ll: Path,
    output_ll: Path,
    functions: Optional[List[str]] = None,
    timeout: int = 60,
) -> VMResult:
    """Run VM obfuscation in an isolated subprocess with timeout.

    This function NEVER raises exceptions to the caller. All errors are
    captured and returned as a failed VMResult with graceful fallback.

    Args:
        input_ll: Path to input LLVM IR file (e.g., obfuscated.ll from OLLVM).
        output_ll: Path where virtualized IR should be written.
        functions: List of function names to virtualize. Empty/None = auto-detect.
        timeout: Maximum seconds before killing the subprocess.

    Returns:
        VMResult with success/failure status and relevant data.
    """
    if functions is None:
        functions = []

    # Validate input file exists
    if not input_ll.exists():
        logger.warning(f"VM input file not found: {input_ll}")
        return VMResult(
            success=False,
            error=f"Input file not found: {input_ll}",
        )

    # Get the base directory containing the modules package
    # We need to run from a directory where "modules.vm.virtualizer.main" is importable
    base_dir = Path(__file__).parent.parent.parent  # cmd/llvm-obfuscator directory

    # Verify virtualizer module exists
    virtualizer_module = base_dir / "modules" / "vm" / "virtualizer" / "main.py"
    if not virtualizer_module.exists():
        logger.warning(f"VM virtualizer module not found: {virtualizer_module}")
        return VMResult(
            success=False,
            error=f"Virtualizer module not found: {virtualizer_module}",
        )

    # Build subprocess command - run as module to handle relative imports
    cmd = [
        sys.executable,
        "-m", "modules.vm.virtualizer.main",
        "--input", str(input_ll),
        "--output", str(output_ll),
    ]

    # Add functions if specified
    if functions:
        cmd.extend(["--functions", ",".join(functions)])

    logger.info(f"Running VM virtualizer: {' '.join(cmd)}")

    try:
        # Run virtualizer as subprocess with timeout
        # cwd must be set to base_dir so the module import works
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(base_dir),
        )

        # Check for process failure
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else f"Exit code {result.returncode}"
            logger.warning(f"VM virtualizer failed: {error_msg}")
            return VMResult(
                success=False,
                error=error_msg,
            )

        # Parse JSON output from virtualizer
        try:
            output_data = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"VM virtualizer returned invalid JSON: {e}")
            # Check if output file was created anyway (graceful recovery)
            if output_ll.exists():
                return VMResult(
                    success=True,
                    output_path=output_ll,
                    functions_virtualized=[],
                    error=None,
                    metrics={"note": "JSON parse failed but output exists"},
                )
            return VMResult(
                success=False,
                error=f"Invalid JSON output: {e}",
            )

        # Verify output file exists
        if not output_ll.exists():
            logger.warning(f"VM output file not created: {output_ll}")
            return VMResult(
                success=False,
                error="Output file not created",
            )

        return VMResult(
            success=output_data.get("success", False),
            output_path=output_ll if output_data.get("success") else None,
            functions_virtualized=output_data.get("functions", []),
            error=output_data.get("error"),
            metrics=output_data.get("metrics"),
        )

    except subprocess.TimeoutExpired:
        logger.warning(f"VM virtualizer timed out after {timeout}s")
        return VMResult(
            success=False,
            error=f"Timeout after {timeout} seconds",
        )

    except FileNotFoundError:
        logger.warning(f"Python interpreter not found: {sys.executable}")
        return VMResult(
            success=False,
            error="Python interpreter not found",
        )

    except PermissionError as e:
        logger.warning(f"Permission denied running virtualizer: {e}")
        return VMResult(
            success=False,
            error=f"Permission denied: {e}",
        )

    except Exception as e:
        # Catch-all for any unexpected errors - NEVER raise to caller
        logger.warning(f"VM virtualizer unexpected error: {type(e).__name__}: {e}")
        return VMResult(
            success=False,
            error=f"Unexpected error: {type(e).__name__}: {e}",
        )
