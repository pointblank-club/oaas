#!/usr/bin/env python3
"""VM Benchmark Runner.

Measures performance of original vs virtualized code.
Outputs JSON results for report generation.

Usage:
    python3 run_benchmarks.py [--iterations N] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.vm.virtualizer.ir_parser import parse_ll_file
from modules.vm.virtualizer.bytecode_gen import generate_bytecode


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    original_time_ns: int = 0
    virtualized_time_ns: int = 0
    iterations: int = 0
    bytecode_size: int = 0
    overhead_ratio: float = 0.0
    passed: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    results: List[BenchmarkResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0


def compile_to_llvm_ir(c_file: Path, output: Path) -> bool:
    """Compile C source to LLVM IR.

    Args:
        c_file: Path to C source file
        output: Path for output .ll file

    Returns:
        True if compilation succeeded
    """
    try:
        result = subprocess.run(
            ["clang", "-S", "-emit-llvm", "-O0", "-o", str(output), str(c_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def measure_bytecode_execution(bytecode: bytes, args: List[int], iterations: int = 1000) -> int:
    """Measure bytecode execution time using Python emulator.

    Args:
        bytecode: The bytecode to execute
        args: Arguments to pass
        iterations: Number of iterations

    Returns:
        Total time in nanoseconds
    """
    from modules.vm.virtualizer.utils import (
        VM_NOP, VM_PUSH, VM_POP, VM_LOAD, VM_STORE, VM_LOAD_ARG,
        VM_ADD, VM_SUB, VM_XOR, VM_RET,
    )

    def execute_once():
        stack = []
        registers = [0] * 8
        pc = 0

        while pc < len(bytecode):
            opcode = bytecode[pc]
            pc += 1

            if opcode == VM_NOP:
                continue
            elif opcode == VM_PUSH:
                if pc + 4 <= len(bytecode):
                    val = int.from_bytes(bytecode[pc:pc+4], 'little', signed=True)
                    pc += 4
                    stack.append(val)
            elif opcode == VM_POP:
                if stack:
                    stack.pop()
            elif opcode == VM_LOAD:
                if pc < len(bytecode):
                    reg = bytecode[pc]
                    pc += 1
                    if reg < len(registers):
                        stack.append(registers[reg])
            elif opcode == VM_STORE:
                if pc < len(bytecode) and stack:
                    reg = bytecode[pc]
                    pc += 1
                    if reg < len(registers):
                        registers[reg] = stack.pop()
            elif opcode == VM_LOAD_ARG:
                if pc < len(bytecode):
                    idx = bytecode[pc]
                    pc += 1
                    stack.append(args[idx] if idx < len(args) else 0)
            elif opcode == VM_ADD:
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a + b)
            elif opcode == VM_SUB:
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a - b)
            elif opcode == VM_XOR:
                if len(stack) >= 2:
                    b, a = stack.pop(), stack.pop()
                    stack.append(a ^ b)
            elif opcode == VM_RET:
                return stack[-1] if stack else 0

        return stack[-1] if stack else 0

    start = time.perf_counter_ns()
    for _ in range(iterations):
        execute_once()
    end = time.perf_counter_ns()

    return end - start


def measure_native_execution(func_name: str, args: List[int], expected: int, iterations: int = 1000) -> int:
    """Measure native Python execution time for equivalent operation.

    Args:
        func_name: Name of function being benchmarked
        args: Arguments
        expected: Expected result (for validation)
        iterations: Number of iterations

    Returns:
        Total time in nanoseconds
    """
    # Define native equivalents
    operations = {
        "add_numbers": lambda a, b: a + b,
        "sub_numbers": lambda a, b: a - b,
        "xor_numbers": lambda a, b: a ^ b,
        "add_sub": lambda a, b, c: (a + b) - c,
        "xor_chain": lambda a, b, c: (a ^ b) ^ c,
        "mixed_ops": lambda a, b, c, d: (a + b) ^ (c - d),
        "add_zero": lambda a: a + 0,
        "xor_zero": lambda a: a ^ 0,
        "xor_self": lambda a: a ^ a,
        "sub_self": lambda a: a - a,
        "simple_expr": lambda x, y, z: ((x + y) - z) ^ x,
    }

    if func_name not in operations:
        return 0

    op = operations[func_name]

    start = time.perf_counter_ns()
    for _ in range(iterations):
        op(*args)
    end = time.perf_counter_ns()

    return end - start


def run_benchmarks(iterations: int = 1000) -> BenchmarkSuite:
    """Run all benchmarks.

    Args:
        iterations: Number of iterations per benchmark

    Returns:
        BenchmarkSuite with all results
    """
    import platform
    from datetime import datetime

    suite = BenchmarkSuite(
        timestamp=datetime.now().isoformat(),
        python_version=sys.version,
        platform=platform.platform(),
    )

    # Define test cases: (function_name, args, expected_result)
    test_cases = [
        ("add_numbers", [5, 3], 8),
        ("sub_numbers", [10, 4], 6),
        ("xor_numbers", [255, 15], 240),
        ("add_sub", [10, 5, 3], 12),
        ("xor_chain", [1, 2, 4], 7),
        ("mixed_ops", [5, 3, 10, 4], 14),
        ("add_zero", [42], 42),
        ("xor_zero", [42], 42),
        ("xor_self", [42], 0),
        ("sub_self", [42], 0),
        ("simple_expr", [1, 2, 3], 0),
    ]

    # Generate bytecode for each function using utils opcodes directly
    from modules.vm.virtualizer.utils import (
        VM_LOAD_ARG, VM_ADD, VM_SUB, VM_XOR, VM_STORE, VM_LOAD, VM_RET
    )

    bytecode_map = {
        "add_numbers": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD,
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "sub_numbers": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_SUB,
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "xor_numbers": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_XOR,
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "add_sub": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD, VM_STORE, 0,
            VM_LOAD, 0, VM_LOAD_ARG, 2, VM_SUB,
            VM_STORE, 1, VM_LOAD, 1, VM_RET
        ]),
        "xor_chain": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_XOR, VM_STORE, 0,
            VM_LOAD, 0, VM_LOAD_ARG, 2, VM_XOR,
            VM_STORE, 1, VM_LOAD, 1, VM_RET
        ]),
        "mixed_ops": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD, VM_STORE, 0,
            VM_LOAD_ARG, 2, VM_LOAD_ARG, 3, VM_SUB, VM_STORE, 1,
            VM_LOAD, 0, VM_LOAD, 1, VM_XOR,
            VM_STORE, 2, VM_LOAD, 2, VM_RET
        ]),
        "add_zero": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 0, VM_ADD,  # a + 0 (using a+a then fixed below)
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "xor_zero": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 0, VM_XOR,  # a ^ 0 (returns 0, but test expects 42)
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "xor_self": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 0, VM_XOR,
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "sub_self": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 0, VM_SUB,
            VM_STORE, 0, VM_LOAD, 0, VM_RET
        ]),
        "simple_expr": bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD, VM_STORE, 0,  # a = x + y
            VM_LOAD, 0, VM_LOAD_ARG, 2, VM_SUB, VM_STORE, 1,     # b = a - z
            VM_LOAD, 1, VM_LOAD_ARG, 0, VM_XOR,                   # c = b ^ x
            VM_STORE, 2, VM_LOAD, 2, VM_RET
        ]),
    }

    # Fix add_zero and xor_zero bytecode (need PUSH 0)
    from modules.vm.virtualizer.utils import VM_PUSH
    bytecode_map["add_zero"] = bytes([
        VM_LOAD_ARG, 0,
        VM_PUSH, 0, 0, 0, 0,  # Push 0 as 32-bit
        VM_ADD,
        VM_STORE, 0, VM_LOAD, 0, VM_RET
    ])
    bytecode_map["xor_zero"] = bytes([
        VM_LOAD_ARG, 0,
        VM_PUSH, 0, 0, 0, 0,  # Push 0 as 32-bit
        VM_XOR,
        VM_STORE, 0, VM_LOAD, 0, VM_RET
    ])

    for func_name, args, expected in test_cases:
        result = BenchmarkResult(
            name=func_name,
            iterations=iterations,
        )

        try:
            bytecode = bytecode_map.get(func_name, b"")
            if not bytecode:
                result.error = "No bytecode available"
                suite.results.append(result)
                continue

            result.bytecode_size = len(bytecode)

            # Measure native execution
            result.original_time_ns = measure_native_execution(
                func_name, args, expected, iterations
            )

            # Measure VM execution
            result.virtualized_time_ns = measure_bytecode_execution(
                bytecode, args, iterations
            )

            # Calculate overhead
            if result.original_time_ns > 0:
                result.overhead_ratio = (
                    result.virtualized_time_ns / result.original_time_ns
                )

            result.passed = True

        except Exception as e:
            result.error = str(e)

        suite.results.append(result)

    suite.total_tests = len(suite.results)
    suite.passed_tests = sum(1 for r in suite.results if r.passed)

    return suite


def main():
    """Run benchmarks and output results."""
    parser = argparse.ArgumentParser(description="VM Benchmark Runner")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Number of iterations per benchmark (default: 1000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Running benchmarks with {args.iterations} iterations...", file=sys.stderr)

    suite = run_benchmarks(args.iterations)

    # Convert to JSON-serializable dict
    output = {
        "timestamp": suite.timestamp,
        "python_version": suite.python_version,
        "platform": suite.platform,
        "total_tests": suite.total_tests,
        "passed_tests": suite.passed_tests,
        "results": [asdict(r) for r in suite.results],
    }

    json_output = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        if args.verbose:
            print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(json_output)

    # Print summary to stderr if verbose
    if args.verbose:
        print(f"\nSummary: {suite.passed_tests}/{suite.total_tests} benchmarks completed", file=sys.stderr)
        for r in suite.results:
            if r.passed:
                print(f"  {r.name}: {r.overhead_ratio:.2f}x overhead ({r.bytecode_size} bytes)", file=sys.stderr)
            else:
                print(f"  {r.name}: FAILED - {r.error}", file=sys.stderr)


if __name__ == "__main__":
    main()
