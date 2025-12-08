#!/usr/bin/env python3
"""VM Virtualizer Entry Point.

Parses LLVM IR, generates bytecode for supported functions, and outputs
modified IR with stub functions that call the VM interpreter.

Usage:
    python -m modules.vm.virtualizer.main \\
        --input obfuscated.ll \\
        --output virtualized.ll \\
        [--functions func1,func2] \\
        [--bytecode-header bytecode.h]

Output:
    Prints JSON status to stdout.
    Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .ir_parser import IRModule, IRFunction, parse_ll_file, get_supported_functions
from .bytecode_gen import generate_bytecode, disassemble_bytecode
from .ir_writer import write_virtualized_module, write_passthrough
from .utils import get_logger

logger = get_logger(__name__)


def virtualize_functions(
    module: IRModule,
    target_functions: Optional[List[str]] = None,
) -> Dict[str, bytes]:
    """Generate bytecode for target functions.

    Args:
        module: Parsed IR module
        target_functions: Optional list of function names to virtualize

    Returns:
        Dictionary mapping function name to bytecode
    """
    bytecode_map: Dict[str, bytes] = {}

    # Get supported functions
    supported = get_supported_functions(module, target_functions)

    for func in supported:
        try:
            bytecode = generate_bytecode(func)
            if bytecode:
                bytecode_map[func.name] = bytecode
                logger.debug(f"Generated {len(bytecode)} bytes for {func.name}")

                # Debug: show disassembly
                if logger.isEnabledFor(10):  # DEBUG level
                    for line in disassemble_bytecode(bytecode):
                        logger.debug(f"  {line}")

        except Exception as e:
            logger.warning(f"Failed to generate bytecode for {func.name}: {e}")

    return bytecode_map


def run_virtualizer(
    input_path: Path,
    output_path: Path,
    functions: Optional[List[str]] = None,
    bytecode_header: Optional[Path] = None,
) -> Dict:
    """Run the VM virtualizer pipeline.

    Args:
        input_path: Path to input .ll file
        output_path: Path to output .ll file
        functions: Optional list of function names to virtualize
        bytecode_header: Optional path to output bytecode C header

    Returns:
        Result dictionary with status and metrics
    """
    # Parse input file
    logger.info(f"Parsing {input_path}")
    module = parse_ll_file(input_path)

    total_functions = len(module.functions)
    supported_count = total_functions - len(module.unsupported_functions)

    # Generate bytecode for target functions
    bytecode_map = virtualize_functions(module, functions)

    virtualized_names = list(bytecode_map.keys())
    total_bytecode_size = sum(len(bc) for bc in bytecode_map.values())

    # Build skip reasons
    skip_reasons: Dict[str, str] = {}
    for func_name in module.unsupported_functions:
        func = module.functions.get(func_name)
        if func and func.skip_reason:
            skip_reasons[func_name] = func.skip_reason

    # Add functions that were filtered out
    if functions:
        for func_name, func in module.functions.items():
            if func_name not in virtualized_names and func_name not in skip_reasons:
                if func.is_supported:
                    skip_reasons[func_name] = "not in target function list"

    # Determine skipped functions
    skipped_names = [
        name for name in module.functions.keys()
        if name not in virtualized_names
    ]

    # Write output
    if bytecode_map:
        write_virtualized_module(module, bytecode_map, output_path, bytecode_header)
    else:
        write_passthrough(module, output_path)

    # Build result
    return {
        "success": True,
        "functions_virtualized": virtualized_names,
        "functions_skipped": skipped_names,
        "skip_reasons": skip_reasons,
        "bytecode_size": total_bytecode_size,
        "error": None,
        "metrics": {
            "total_functions": total_functions,
            "supported": supported_count,
            "virtualized": len(virtualized_names),
            "skipped": len(skipped_names),
            "bytecode_bytes": total_bytecode_size,
        },
    }


def main() -> int:
    """Main entry point for VM virtualizer."""
    parser = argparse.ArgumentParser(
        description="VM-based code virtualizer for LLVM IR"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input LLVM IR file path"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output LLVM IR file path"
    )
    parser.add_argument(
        "--functions", default="",
        help="Comma-separated function names to virtualize (empty = all supported)"
    )
    parser.add_argument(
        "--bytecode-header", default="",
        help="Output path for C bytecode header file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    import logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    input_path = Path(args.input)
    output_path = Path(args.output)
    functions = [f.strip() for f in args.functions.split(",") if f.strip()] or None
    bytecode_header = Path(args.bytecode_header) if args.bytecode_header else None

    # Validate input exists
    if not input_path.exists():
        result = {
            "success": False,
            "error": f"Input file not found: {input_path}",
            "functions_virtualized": [],
            "functions_skipped": [],
            "skip_reasons": {},
            "bytecode_size": 0,
            "metrics": None,
        }
        print(json.dumps(result))
        return 1

    try:
        result = run_virtualizer(
            input_path,
            output_path,
            functions,
            bytecode_header,
        )
        print(json.dumps(result))
        return 0

    except Exception as e:
        import traceback
        logger.error(f"Virtualizer failed: {e}")
        if args.debug:
            traceback.print_exc()

        result = {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "functions_virtualized": [],
            "functions_skipped": [],
            "skip_reasons": {},
            "bytecode_size": 0,
            "metrics": None,
        }
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    sys.exit(main())
