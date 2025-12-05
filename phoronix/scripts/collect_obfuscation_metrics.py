#!/usr/bin/env python3
"""
Obfuscation Metrics Collector

Analyzes binary-level properties of baseline and obfuscated binaries,
computing static metrics, entropy, CFG analysis, and cyclomatic complexity.
"""

import json
import logging
import re
import struct
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BinaryMetrics:
    """Container for binary metrics."""
    file_size_bytes: int
    file_size_percent_increase: float
    text_section_size: int
    num_functions: int
    num_basic_blocks: int
    instruction_count: int
    text_entropy: float
    cyclomatic_complexity: float
    stripped: bool
    pie_enabled: bool


class MetricsCollector:
    """Collects and analyzes obfuscation metrics."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.logger = logging.getLogger(__name__)

    def collect_metrics(
        self,
        baseline_binary: Path,
        obfuscated_binaries: List[Path],
        config_name: str,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Collect metrics for baseline and obfuscated binaries.

        Args:
            baseline_binary: Path to the baseline binary
            obfuscated_binaries: List of paths to obfuscated binaries
            config_name: Name of the obfuscation configuration
            output_dir: Optional output directory for results

        Returns:
            Dictionary with comprehensive metrics
        """
        if not baseline_binary.exists():
            raise FileNotFoundError(f"Baseline binary not found: {baseline_binary}")

        timestamp = datetime.now().isoformat()
        baseline_metrics = self._analyze_binary(baseline_binary)
        obfuscated_metrics = {
            binary.stem: self._analyze_binary(binary)
            for binary in obfuscated_binaries
            if binary.exists()
        }

        results = {
            "timestamp": timestamp,
            "config_name": config_name,
            "baseline": asdict(baseline_metrics) if baseline_metrics else {},
            "obfuscated": {
                name: asdict(m) if m else {} for name, m in obfuscated_metrics.items()
            },
            "comparison": self._compute_comparison(baseline_metrics, obfuscated_metrics),
        }

        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(results, output_dir)

        return results

    def _analyze_binary(self, binary_path: Path) -> Optional[BinaryMetrics]:
        """
        Analyze a single binary and extract metrics.

        Args:
            binary_path: Path to the binary

        Returns:
            BinaryMetrics object or None on failure
        """
        try:
            file_size = binary_path.stat().st_size
            stripped = self._is_stripped(binary_path)

            # Log stripping status with warning if stripped
            if stripped:
                self.logger.warning(f"Binary is STRIPPED: {binary_path}")
                self.logger.warning("  ⚠️ Symbol-based metrics will be unavailable (function count, etc.)")
                self.logger.warning("  📝 Recommendation: Use non-stripped binary for accurate analysis")
            else:
                self.logger.info(f"Binary has symbols: {binary_path}")

            text_size = self._get_text_section_size(binary_path)
            num_functions = self._count_functions(binary_path)

            # Warn if function count is 0 on non-stripped binary
            if not stripped and num_functions == 0:
                self.logger.warning(f"  ⚠️ No functions extracted from non-stripped binary: {binary_path}")
                self.logger.warning("  📝 Check if binary format is supported")

            num_basic_blocks = self._count_basic_blocks(binary_path)
            instruction_count = self._count_instructions(binary_path)
            text_entropy = self._compute_text_entropy(binary_path)

            # Warn if entropy is 0
            if text_entropy == 0.0:
                self.logger.warning(f"  ⚠️ Could not compute .text entropy for: {binary_path}")
                self.logger.warning("  📝 This may indicate issues with ELF section reading")

            cyclomatic_complexity = self._estimate_cyclomatic_complexity(binary_path)
            pie_enabled = self._is_pie_enabled(binary_path)

            return BinaryMetrics(
                file_size_bytes=file_size,
                file_size_percent_increase=0.0,  # Set later in comparison
                text_section_size=text_size,
                num_functions=num_functions,
                num_basic_blocks=num_basic_blocks,
                instruction_count=instruction_count,
                text_entropy=round(text_entropy, 3),
                cyclomatic_complexity=round(cyclomatic_complexity, 2),
                stripped=stripped,
                pie_enabled=pie_enabled,
            )
        except Exception as e:
            self.logger.error(f"Failed to analyze {binary_path}: {e}")
            return None

    def _get_text_section_size(self, binary_path: Path) -> int:
        """Extract .text section size using readelf."""
        try:
            result = subprocess.run(
                ["readelf", "-S", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split('\n'):
                if '.text' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            return int(parts[5], 16)
                        except ValueError:
                            pass
            return 0
        except Exception as e:
            self.logger.warning(f"Failed to get text section size: {e}")
            return 0

    def _count_functions(self, binary_path: Path) -> int:
        """Count number of functions using nm."""
        try:
            result = subprocess.run(
                ["nm", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Count symbols that are functions (type 'T' or 't' for text)
            return sum(1 for line in result.stdout.split('\n') if ' T ' in line or ' t ' in line)
        except Exception as e:
            self.logger.warning(f"Failed to count functions: {e}")
            return 0

    def _count_basic_blocks(self, binary_path: Path) -> int:
        """
        Estimate basic block count using objdump.

        This is a heuristic counting jumps and branches.
        """
        try:
            result = subprocess.run(
                ["objdump", "-d", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Count branch/jump instructions as proxy for basic blocks
            jump_pattern = r'^\s+[0-9a-f]+:\s+(j[a-z]+|b[a-z]*|call)\s'
            jumps = sum(1 for line in result.stdout.split('\n') if re.match(jump_pattern, line))

            return max(jumps, 1)
        except Exception as e:
            self.logger.warning(f"Failed to count basic blocks: {e}")
            return 0

    def _count_instructions(self, binary_path: Path) -> int:
        """Count total instructions using objdump."""
        try:
            result = subprocess.run(
                ["objdump", "-d", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Count lines that look like instructions
            instr_pattern = r'^\s+[0-9a-f]+:\s+[0-9a-f]+'
            return sum(1 for line in result.stdout.split('\n') if re.match(instr_pattern, line))
        except Exception as e:
            self.logger.warning(f"Failed to count instructions: {e}")
            return 0

    def _compute_text_entropy(self, binary_path: Path) -> float:
        """Compute Shannon entropy of .text section."""
        try:
            # Read entire binary
            with open(binary_path, 'rb') as f:
                binary_data = f.read()

            # Find .text section offset and size
            text_offset, text_size = self._find_text_section_range(binary_path)
            if text_offset < 0 or text_size <= 0:
                return 0.0

            # Extract text section
            text_data = binary_data[text_offset : text_offset + text_size]

            # Calculate entropy
            byte_counts = {}
            for byte in text_data:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1

            # Use standard information theory formula
            import math
            entropy = 0.0
            for count in byte_counts.values():
                probability = count / len(text_data)
                if probability > 0:
                    entropy -= probability * math.log2(probability)

            return entropy
        except Exception as e:
            self.logger.warning(f"Failed to compute entropy: {e}")
            return 0.0

    def _find_text_section_range(self, binary_path: Path) -> Tuple[int, int]:
        """Find offset and size of .text section."""
        try:
            result = subprocess.run(
                ["readelf", "-S", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            for line in result.stdout.split('\n'):
                if '.text' in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            offset = int(parts[4], 16)
                            size = int(parts[5], 16)
                            return offset, size
                        except ValueError:
                            pass
            return -1, 0
        except Exception as e:
            self.logger.warning(f"Failed to find text section: {e}")
            return -1, 0

    def _estimate_cyclomatic_complexity(self, binary_path: Path) -> float:
        """
        Estimate cyclomatic complexity from control flow.

        Heuristic: CC = E - N + 2P where:
        - E = edges (branches)
        - N = nodes (basic blocks)
        - P = connected components (functions)
        """
        try:
            result = subprocess.run(
                ["objdump", "-d", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Count functions (P)
            func_pattern = r'^[0-9a-f]+ <[^>]+>:'
            functions = sum(1 for line in result.stdout.split('\n') if re.match(func_pattern, line))

            # Count jumps/branches (E)
            branch_pattern = r'^\s+[0-9a-f]+:\s+(j[a-z]+|b[a-z]*)'
            branches = sum(1 for line in result.stdout.split('\n') if re.match(branch_pattern, line))

            # Estimate basic blocks as functions + branches
            basic_blocks = functions + branches

            # CC = E - N + 2P
            if functions > 0:
                cc = (branches - basic_blocks + 2 * functions) / functions
                return max(1.0, cc)
            return 1.0
        except Exception as e:
            self.logger.warning(f"Failed to estimate complexity: {e}")
            return 1.0

    def _is_stripped(self, binary_path: Path) -> bool:
        """Check if binary is stripped."""
        try:
            result = subprocess.run(
                ["file", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "stripped" in result.stdout.lower()
        except Exception:
            return False

    def _is_pie_enabled(self, binary_path: Path) -> bool:
        """Check if binary is Position Independent Executable."""
        try:
            result = subprocess.run(
                ["readelf", "-h", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Look for ET_DYN which indicates PIE
            for line in result.stdout.split('\n'):
                if 'Type:' in line and 'DYN' in line:
                    return True
            return False
        except Exception:
            return False

    def _compute_comparison(
        self,
        baseline: Optional[BinaryMetrics],
        obfuscated: Dict[str, Optional[BinaryMetrics]],
    ) -> Dict:
        """Compute comparison metrics between baseline and obfuscated."""
        if not baseline or not obfuscated:
            return {}

        comparison = {}
        for name, metrics in obfuscated.items():
            if not metrics:
                continue

            comparison[name] = {
                "file_size_increase_bytes": metrics.file_size_bytes - baseline.file_size_bytes,
                "file_size_increase_percent": round(
                    ((metrics.file_size_bytes - baseline.file_size_bytes) / baseline.file_size_bytes * 100),
                    2,
                ) if baseline.file_size_bytes > 0 else 0.0,
                "text_section_increase_bytes": metrics.text_section_size - baseline.text_section_size,
                "function_count_delta": metrics.num_functions - baseline.num_functions,
                "basic_block_count_delta": metrics.num_basic_blocks - baseline.num_basic_blocks,
                "instruction_count_delta": metrics.instruction_count - baseline.instruction_count,
                "entropy_increase": round(metrics.text_entropy - baseline.text_entropy, 3),
                "complexity_increase": round(metrics.cyclomatic_complexity - baseline.cyclomatic_complexity, 2),
            }

        return comparison

    def _save_results(self, results: Dict, output_dir: Path) -> None:
        """Save results in JSON, CSV, and Markdown formats."""
        # JSON output
        json_file = output_dir / "metrics.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved JSON metrics to {json_file}")

        # Markdown table output
        md_file = output_dir / "metrics.md"
        self._write_markdown_report(results, md_file)
        self.logger.info(f"Saved Markdown report to {md_file}")

        # CSV output
        csv_file = output_dir / "metrics.csv"
        self._write_csv_report(results, csv_file)
        self.logger.info(f"Saved CSV report to {csv_file}")

    def _write_markdown_report(self, results: Dict, output_file: Path) -> None:
        """Write Markdown table report."""
        with open(output_file, 'w') as f:
            f.write("# Obfuscation Metrics Report\n\n")
            f.write(f"**Config:** {results.get('config_name')}\n\n")
            f.write(f"**Timestamp:** {results.get('timestamp')}\n\n")

            baseline = results.get('baseline', {})
            obfuscated = results.get('obfuscated', {})

            f.write("## Baseline Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in baseline.items():
                if key != 'file_size_percent_increase':
                    f.write(f"| {key} | {value} |\n")

            f.write("\n## Obfuscated Binaries Metrics\n\n")
            for binary_name, metrics in obfuscated.items():
                f.write(f"### {binary_name}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in metrics.items():
                    if key != 'file_size_percent_increase':
                        f.write(f"| {key} | {value} |\n")

            f.write("\n## Comparison\n\n")
            comparison = results.get('comparison', {})
            for binary_name, deltas in comparison.items():
                f.write(f"### {binary_name}\n\n")
                f.write("| Metric | Δ Value |\n")
                f.write("|--------|----------|\n")
                for key, value in deltas.items():
                    f.write(f"| {key} | {value} |\n")

    def _write_csv_report(self, results: Dict, output_file: Path) -> None:
        """Write CSV report."""
        import csv

        baseline = results.get('baseline', {})
        obfuscated = results.get('obfuscated', {})

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            metrics_keys = list(baseline.keys())
            writer.writerow(['Binary'] + metrics_keys)

            # Baseline
            writer.writerow(['baseline'] + [baseline.get(k, '') for k in metrics_keys])

            # Obfuscated
            for binary_name, metrics in obfuscated.items():
                writer.writerow([binary_name] + [metrics.get(k, '') for k in metrics_keys])


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect obfuscation metrics')
    parser.add_argument('baseline', type=Path, help='Path to baseline binary')
    parser.add_argument('obfuscated', type=Path, nargs='+', help='Paths to obfuscated binaries')
    parser.add_argument('--config', default='default', help='Configuration name')
    parser.add_argument('--output', type=Path, help='Output directory')

    args = parser.parse_args()

    collector = MetricsCollector()
    results = collector.collect_metrics(
        baseline_binary=args.baseline,
        obfuscated_binaries=args.obfuscated,
        config_name=args.config,
        output_dir=args.output,
    )

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
