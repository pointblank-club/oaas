#!/usr/bin/env python3

################################################################################
# SPEC CPU 2017 Metrics Collection Script
################################################################################
#
# Purpose: Extract and aggregate SPEC CPU results into structured metrics
#
# Functionality:
#   1. Parse SPEC CPU raw result files (XML format)
#   2. Extract performance metrics (base score, peak score, component scores)
#   3. Extract resource metrics (runtime, memory usage, power)
#   4. Calculate derived metrics (geometric mean, standard deviation)
#   5. Generate machine-readable output (JSON, CSV)
#   6. Handle missing/incomplete results gracefully
#
# Exit Codes:
#   0 = Metrics collected successfully
#   1 = Invalid result directory
#   2 = No results found
#   3 = Parsing error
#   4 = File I/O error
#
################################################################################

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from statistics import mean, stdev, geometric_mean

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Container for extracted benchmark metrics."""
    benchmark_name: str
    test_type: str  # 'speed' or 'rate'
    compiler: str  # 'clang' or 'gcc'
    compiler_version: str
    optimization_flags: str
    base_score: Optional[float] = None
    peak_score: Optional[float] = None
    copies: int = 1
    runtime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_rate: float = 0.0
    component_scores: Dict[str, float] = None
    timestamp: str = ""

    def __post_init__(self):
        if self.component_scores is None:
            self.component_scores = {}
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


# =============================================================================
# SPEC Result Parser
# =============================================================================

class SPECResultParser:
    """Parse SPEC CPU result XML files."""

    def __init__(self, result_dir: str, test_type: str = "speed"):
        self.result_dir = Path(result_dir)
        self.test_type = test_type
        self.metrics: List[BenchmarkMetrics] = []

    def parse_result_files(self) -> bool:
        """Parse all result files in directory."""
        if not self.result_dir.exists():
            print(f"[ERROR] Result directory not found: {self.result_dir}", file=sys.stderr)
            return False

        result_files = list(self.result_dir.glob("*.xml"))
        if not result_files:
            result_files = list(self.result_dir.glob("*.txt"))

        if not result_files:
            print(f"[ERROR] No result files found in: {self.result_dir}", file=sys.stderr)
            return False

        print(f"[INFO] Found {len(result_files)} result file(s)", file=sys.stderr)

        for result_file in result_files:
            try:
                if result_file.suffix == ".xml":
                    self._parse_xml_result(result_file)
                elif result_file.suffix == ".txt":
                    self._parse_txt_result(result_file)
            except Exception as e:
                print(f"[WARNING] Failed to parse {result_file.name}: {e}", file=sys.stderr)
                continue

        return len(self.metrics) > 0

    def _parse_xml_result(self, xml_file: Path) -> None:
        """Parse SPEC CPU XML result file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Extract benchmark name from filename or XML content
            benchmark_name = xml_file.stem

            # Look for run section
            for run in root.findall(".//run"):
                metrics = self._extract_run_metrics(run, benchmark_name)
                if metrics:
                    self.metrics.append(metrics)

        except ET.ParseError as e:
            raise ValueError(f"XML parsing failed: {e}")

    def _parse_txt_result(self, txt_file: Path) -> None:
        """Parse SPEC CPU text result file (fallback)."""
        with open(txt_file, 'r') as f:
            content = f.read()

        # Extract benchmark name
        benchmark_name = txt_file.stem

        # Try to extract base and peak scores
        base_score = self._extract_score_from_text(content, r"BaseResult\s*:\s*([\d.]+)")
        peak_score = self._extract_score_from_text(content, r"PeakResult\s*:\s*([\d.]+)")

        # Extract runtime
        runtime = self._extract_runtime_from_text(content)

        metrics = BenchmarkMetrics(
            benchmark_name=benchmark_name,
            test_type=self.test_type,
            compiler="clang",
            compiler_version="unknown",
            optimization_flags="unknown",
            base_score=base_score,
            peak_score=peak_score,
            runtime_seconds=runtime
        )
        self.metrics.append(metrics)

    def _extract_run_metrics(self, run_elem, benchmark_name: str) -> Optional[BenchmarkMetrics]:
        """Extract metrics from XML run element."""
        try:
            # Extract basic information
            result_elem = run_elem.find("result")
            if result_elem is None:
                return None

            base_score = self._safe_float(result_elem.findtext("BaseResult"))
            peak_score = self._safe_float(result_elem.findtext("PeakResult"))

            # Extract compiler info
            compiler_info = self._extract_compiler_info(run_elem)

            # Extract runtime
            runtime = self._safe_float(result_elem.findtext("TotalRuntime"))

            # Extract component scores
            component_scores = self._extract_component_scores(result_elem)

            metrics = BenchmarkMetrics(
                benchmark_name=benchmark_name,
                test_type=self.test_type,
                compiler=compiler_info['name'],
                compiler_version=compiler_info['version'],
                optimization_flags=compiler_info['flags'],
                base_score=base_score,
                peak_score=peak_score,
                runtime_seconds=runtime,
                component_scores=component_scores
            )

            return metrics

        except Exception as e:
            print(f"[WARNING] Failed to extract metrics from run element: {e}", file=sys.stderr)
            return None

    def _extract_compiler_info(self, run_elem) -> Dict[str, str]:
        """Extract compiler information from run element."""
        compiler_name = "clang"
        compiler_version = "unknown"
        compiler_flags = "unknown"

        # Try various XML paths where compiler info might be stored
        compiler_elem = run_elem.find(".//compiler")
        if compiler_elem is not None:
            compiler_name = compiler_elem.findtext("name", "clang")
            compiler_version = compiler_elem.findtext("version", "unknown")

        # Extract flags from CFLAGS or similar
        cflags_elem = run_elem.find(".//CFLAGS")
        if cflags_elem is not None:
            compiler_flags = cflags_elem.text or "unknown"

        return {
            'name': compiler_name,
            'version': compiler_version,
            'flags': compiler_flags
        }

    def _extract_component_scores(self, result_elem) -> Dict[str, float]:
        """Extract component/benchmark scores."""
        component_scores = {}

        # Look for component score elements
        for component in result_elem.findall("component"):
            name = component.findtext("name")
            score = self._safe_float(component.findtext("score"))
            if name and score is not None:
                component_scores[name] = score

        return component_scores

    def _extract_score_from_text(self, text: str, pattern: str) -> Optional[float]:
        """Extract numeric score from text using regex."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return self._safe_float(match.group(1))
        return None

    def _extract_runtime_from_text(self, text: str) -> Optional[float]:
        """Extract runtime from text."""
        # Look for patterns like "Runtime: 3600.5 seconds"
        match = re.search(r"runtime\s*:\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            return self._safe_float(match.group(1))
        return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# =============================================================================
# Metrics Aggregation
# =============================================================================

class MetricsAggregator:
    """Aggregate and analyze collected metrics."""

    def __init__(self, metrics: List[BenchmarkMetrics]):
        self.metrics = metrics

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics across all metrics."""
        if not self.metrics:
            return {}

        base_scores = [m.base_score for m in self.metrics if m.base_score is not None]
        peak_scores = [m.peak_score for m in self.metrics if m.peak_score is not None]
        runtimes = [m.runtime_seconds for m in self.metrics if m.runtime_seconds is not None]

        stats = {
            'total_benchmarks': len(self.metrics),
            'benchmarks_completed': sum(1 for m in self.metrics if m.base_score is not None),
            'benchmarks_failed': sum(1 for m in self.metrics if m.base_score is None),
        }

        if base_scores:
            try:
                stats['base_score_mean'] = mean(base_scores)
                stats['base_score_geomean'] = geometric_mean(base_scores)
                if len(base_scores) > 1:
                    stats['base_score_stdev'] = stdev(base_scores)
            except Exception as e:
                print(f"[WARNING] Failed to calculate base score stats: {e}", file=sys.stderr)

        if peak_scores:
            try:
                stats['peak_score_mean'] = mean(peak_scores)
                stats['peak_score_geomean'] = geometric_mean(peak_scores)
                if len(peak_scores) > 1:
                    stats['peak_score_stdev'] = stdev(peak_scores)
            except Exception as e:
                print(f"[WARNING] Failed to calculate peak score stats: {e}", file=sys.stderr)

        if runtimes:
            try:
                stats['total_runtime_seconds'] = sum(runtimes)
                stats['avg_runtime_seconds'] = mean(runtimes)
            except Exception as e:
                print(f"[WARNING] Failed to calculate runtime stats: {e}", file=sys.stderr)

        return stats


# =============================================================================
# Output Generators
# =============================================================================

class MetricsExporter:
    """Export metrics in various formats."""

    @staticmethod
    def to_json(metrics: List[BenchmarkMetrics], summary: Dict, output_file: str) -> bool:
        """Export metrics to JSON format."""
        try:
            output_data = {
                'summary': summary,
                'metrics': [asdict(m) for m in metrics],
                'export_timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            }

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"[SUCCESS] JSON metrics exported: {output_file}", file=sys.stderr)
            return True

        except IOError as e:
            print(f"[ERROR] Failed to write JSON file: {e}", file=sys.stderr)
            return False

    @staticmethod
    def to_csv(metrics: List[BenchmarkMetrics], output_file: str) -> bool:
        """Export metrics to CSV format."""
        try:
            if not metrics:
                print("[WARNING] No metrics to export", file=sys.stderr)
                return False

            # Use first metric as field reference
            fieldnames = [
                'benchmark_name',
                'test_type',
                'compiler',
                'compiler_version',
                'optimization_flags',
                'base_score',
                'peak_score',
                'copies',
                'runtime_seconds',
                'memory_usage_mb',
                'error_rate',
                'timestamp'
            ]

            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for metric in metrics:
                    row = {
                        'benchmark_name': metric.benchmark_name,
                        'test_type': metric.test_type,
                        'compiler': metric.compiler,
                        'compiler_version': metric.compiler_version,
                        'optimization_flags': metric.optimization_flags,
                        'base_score': metric.base_score,
                        'peak_score': metric.peak_score,
                        'copies': metric.copies,
                        'runtime_seconds': metric.runtime_seconds,
                        'memory_usage_mb': metric.memory_usage_mb,
                        'error_rate': metric.error_rate,
                        'timestamp': metric.timestamp
                    }
                    writer.writerow(row)

            print(f"[SUCCESS] CSV metrics exported: {output_file}", file=sys.stderr)
            return True

        except IOError as e:
            print(f"[ERROR] Failed to write CSV file: {e}", file=sys.stderr)
            return False


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract and aggregate SPEC CPU 2017 results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract metrics from speed benchmark results
  %(prog)s /path/to/speed/results --format json

  # Export both JSON and CSV
  %(prog)s /path/to/rate/results --format both

  # Specify test type
  %(prog)s /path/to/results --test-type rate --format csv
        """
    )

    parser.add_argument(
        'result_directory',
        help='Path to SPEC CPU result directory (speed/ or rate/)'
    )

    parser.add_argument(
        '--test-type',
        choices=['speed', 'rate'],
        default='speed',
        help='Benchmark test type (default: speed)'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'both'],
        default='json',
        help='Output format (default: json)'
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory for metrics (default: same as input)'
    )

    args = parser.parse_args()

    # Validate input directory
    result_dir = Path(args.result_directory)
    if not result_dir.exists():
        print(f"[ERROR] Result directory not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse results
    print(f"[INFO] Parsing SPEC CPU results from: {result_dir}", file=sys.stderr)
    parser_obj = SPECResultParser(str(result_dir), args.test_type)

    if not parser_obj.parse_result_files():
        print("[ERROR] No metrics collected", file=sys.stderr)
        sys.exit(2)

    print(f"[SUCCESS] Collected metrics from {len(parser_obj.metrics)} benchmark(s)", file=sys.stderr)

    # Aggregate metrics
    aggregator = MetricsAggregator(parser_obj.metrics)
    summary = aggregator.get_summary_stats()

    # Export metrics
    exporter = MetricsExporter()
    success = True

    if args.format in ['json', 'both']:
        json_file = output_dir / 'metrics.json'
        if not exporter.to_json(parser_obj.metrics, summary, str(json_file)):
            success = False

    if args.format in ['csv', 'both']:
        csv_file = output_dir / 'metrics.csv'
        if not exporter.to_csv(parser_obj.metrics, str(csv_file)):
            success = False

    # Print summary to stdout
    print("\n=== Metrics Summary ===")
    print(json.dumps(summary, indent=2))

    sys.exit(0 if success else 4)


if __name__ == '__main__':
    main()
