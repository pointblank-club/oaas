#!/usr/bin/env python3
"""VM Virtualizer Entry Point - Stub Implementation.

This is a stub that passes through the input IR unchanged.
Future implementation will convert selected functions to custom bytecode.

Usage:
    python main.py --input input.ll --output output.ll [--functions func1,func2]

Output:
    Prints JSON to stdout: {"success": true, "functions": [], "metrics": {...}}
    Exit 0 on success, exit 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for VM virtualizer."""
    parser = argparse.ArgumentParser(description="VM-based code virtualizer")
    parser.add_argument("--input", required=True, help="Input LLVM IR file path")
    parser.add_argument("--output", required=True, help="Output LLVM IR file path")
    parser.add_argument("--functions", default="", help="Comma-separated function names to virtualize")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    functions = [f.strip() for f in args.functions.split(",") if f.strip()]

    # Validate input exists
    if not input_path.exists():
        result = {
            "success": False,
            "error": f"Input file not found: {input_path}",
            "functions": [],
            "metrics": None,
        }
        print(json.dumps(result))
        return 1

    try:
        # STUB IMPLEMENTATION: Just copy input to output unchanged
        # This is a passthrough - future implementation will perform actual virtualization
        shutil.copy2(input_path, output_path)

        # Return success with stub metrics
        result = {
            "success": True,
            "functions": functions if functions else [],  # Empty for now (no actual virtualization)
            "metrics": {
                "bytecode_size": 0,
                "opcodes_used": 0,
                "vm_handlers": 0,
                "stub": True,  # Indicates this is a stub implementation
            },
            "error": None,
        }
        print(json.dumps(result))
        return 0

    except PermissionError as e:
        result = {
            "success": False,
            "error": f"Permission denied: {e}",
            "functions": [],
            "metrics": None,
        }
        print(json.dumps(result))
        return 1

    except Exception as e:
        result = {
            "success": False,
            "error": f"Unexpected error: {type(e).__name__}: {e}",
            "functions": [],
            "metrics": None,
        }
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    sys.exit(main())
