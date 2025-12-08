#!/usr/bin/env python3
"""Benchmark Report Generator.

Generates markdown reports from benchmark JSON results.

Usage:
    python3 report_gen.py benchmark_results.json > report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "VM Virtualization Benchmark Report"
    show_overhead: bool = True
    show_bytecode_size: bool = True
    show_raw_times: bool = False


def format_time_ns(ns: int) -> str:
    """Format nanoseconds to human-readable string.

    Args:
        ns: Time in nanoseconds

    Returns:
        Formatted string
    """
    if ns < 1000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.2f} us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def generate_markdown_report(data: dict, config: ReportConfig) -> str:
    """Generate a markdown report from benchmark data.

    Args:
        data: Benchmark results JSON data
        config: Report configuration

    Returns:
        Markdown formatted report
    """
    lines = []

    # Header
    lines.append(f"# {config.title}")
    lines.append("")
    lines.append(f"**Generated:** {data.get('timestamp', 'Unknown')}")
    lines.append(f"**Platform:** {data.get('platform', 'Unknown')}")
    lines.append(f"**Python:** {data.get('python_version', 'Unknown').split()[0]}")
    lines.append("")

    # Summary
    total = data.get('total_tests', 0)
    passed = data.get('passed_tests', 0)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Benchmarks:** {total}")
    lines.append(f"- **Successful:** {passed}")
    lines.append(f"- **Failed:** {total - passed}")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")

    # Build table header
    headers = ["Function", "Status"]
    if config.show_bytecode_size:
        headers.append("Bytecode Size")
    if config.show_overhead:
        headers.append("Overhead")
    if config.show_raw_times:
        headers.append("Native Time")
        headers.append("VM Time")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Build table rows
    results = data.get('results', [])
    for r in results:
        row = [r.get('name', 'Unknown')]

        if r.get('passed', False):
            row.append("PASS")
        else:
            row.append(f"FAIL: {r.get('error', 'Unknown error')}")

        if config.show_bytecode_size:
            row.append(f"{r.get('bytecode_size', 0)} bytes")

        if config.show_overhead:
            overhead = r.get('overhead_ratio', 0)
            if overhead > 0:
                row.append(f"{overhead:.2f}x")
            else:
                row.append("N/A")

        if config.show_raw_times:
            iterations = r.get('iterations', 1)
            native = r.get('original_time_ns', 0)
            vm = r.get('virtualized_time_ns', 0)
            row.append(format_time_ns(native // iterations) + "/op")
            row.append(format_time_ns(vm // iterations) + "/op")

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    # Calculate averages
    overheads = [r.get('overhead_ratio', 0) for r in results if r.get('passed', False) and r.get('overhead_ratio', 0) > 0]
    bytecode_sizes = [r.get('bytecode_size', 0) for r in results if r.get('passed', False)]

    if overheads:
        avg_overhead = sum(overheads) / len(overheads)
        min_overhead = min(overheads)
        max_overhead = max(overheads)
        lines.append(f"- **Average Overhead:** {avg_overhead:.2f}x")
        lines.append(f"- **Min Overhead:** {min_overhead:.2f}x")
        lines.append(f"- **Max Overhead:** {max_overhead:.2f}x")
    else:
        lines.append("- **Overhead:** No valid measurements")

    if bytecode_sizes:
        avg_size = sum(bytecode_sizes) / len(bytecode_sizes)
        total_size = sum(bytecode_sizes)
        lines.append(f"- **Average Bytecode Size:** {avg_size:.1f} bytes")
        lines.append(f"- **Total Bytecode Size:** {total_size} bytes")

    lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    if passed == total and total > 0:
        lines.append("All benchmarks completed successfully. The VM virtualization ")
        lines.append("produces semantically correct results with measurable but ")
        lines.append("acceptable overhead for obfuscation purposes.")
    elif passed > 0:
        lines.append(f"{passed}/{total} benchmarks passed. Some failures may indicate ")
        lines.append("unsupported operations or configuration issues.")
    else:
        lines.append("No benchmarks completed successfully. Please check the configuration.")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by OAAS VM Benchmark Suite*")

    return "\n".join(lines)


def generate_json_summary(data: dict) -> str:
    """Generate a JSON summary of key metrics.

    Args:
        data: Benchmark results JSON data

    Returns:
        JSON formatted summary
    """
    results = data.get('results', [])

    overheads = [r.get('overhead_ratio', 0) for r in results if r.get('passed', False) and r.get('overhead_ratio', 0) > 0]
    bytecode_sizes = [r.get('bytecode_size', 0) for r in results if r.get('passed', False)]

    summary = {
        "total_tests": data.get('total_tests', 0),
        "passed_tests": data.get('passed_tests', 0),
        "timestamp": data.get('timestamp', ''),
        "metrics": {
            "avg_overhead": sum(overheads) / len(overheads) if overheads else 0,
            "min_overhead": min(overheads) if overheads else 0,
            "max_overhead": max(overheads) if overheads else 0,
            "avg_bytecode_size": sum(bytecode_sizes) / len(bytecode_sizes) if bytecode_sizes else 0,
            "total_bytecode_size": sum(bytecode_sizes),
        },
    }

    return json.dumps(summary, indent=2)


def main():
    """Generate benchmark report."""
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="-",
        help="Input JSON file (default: stdin)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="VM Virtualization Benchmark Report",
        help="Report title"
    )
    parser.add_argument(
        "--raw-times",
        action="store_true",
        help="Include raw timing data"
    )

    args = parser.parse_args()

    # Read input
    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.input) as f:
            data = json.load(f)

    # Configure report
    config = ReportConfig(
        title=args.title,
        show_raw_times=args.raw_times,
    )

    # Generate report
    if args.format == "markdown":
        output = generate_markdown_report(data, config)
    else:
        output = generate_json_summary(data)

    print(output)


if __name__ == "__main__":
    main()
