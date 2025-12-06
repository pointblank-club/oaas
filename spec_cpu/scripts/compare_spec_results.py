#!/usr/bin/env python3

################################################################################
# SPEC CPU 2017 Comparison Report Generator
################################################################################
#
# Purpose: Generate detailed comparison reports between baseline and obfuscated
#          SPEC CPU results
#
# Functionality:
#   1. Load baseline results from specified directory
#   2. Load obfuscated results from specified configuration
#   3. Calculate performance deltas (absolute and percentage)
#   4. Generate performance impact summary
#   5. Identify regressions and improvements
#   6. Generate HTML report for easy viewing
#   7. Export machine-readable comparison metrics (JSON)
#   8. Create detailed per-benchmark analysis (CSV)
#
# Exit Codes:
#   0 = Comparison generated successfully
#   1 = Invalid baseline directory
#   2 = Invalid obfuscated directory
#   3 = No matching benchmarks found
#   4 = Report generation error
#   5 = File I/O error
#
################################################################################

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from statistics import mean, median

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkComparison:
    """Container for baseline vs obfuscated comparison."""
    benchmark_name: str
    baseline_score: Optional[float]
    obfuscated_score: Optional[float]
    baseline_runtime: Optional[float]
    obfuscated_runtime: Optional[float]
    performance_delta_pct: Optional[float] = None
    runtime_delta_pct: Optional[float] = None
    regression: bool = False
    improvement: bool = False
    status: str = "unknown"

    def __post_init__(self):
        """Calculate deltas automatically."""
        if self.baseline_score and self.obfuscated_score:
            # Higher score is better - calculate percentage change
            self.performance_delta_pct = (
                (self.obfuscated_score - self.baseline_score) / self.baseline_score * 100
            )
            self.regression = self.performance_delta_pct < -5  # Threshold: 5% regression
            self.improvement = self.performance_delta_pct > 5   # Threshold: 5% improvement

        if self.baseline_runtime and self.obfuscated_runtime:
            # Lower runtime is better - calculate percentage change
            self.runtime_delta_pct = (
                (self.obfuscated_runtime - self.baseline_runtime) / self.baseline_runtime * 100
            )

        # Determine overall status
        if self.baseline_score is None or self.obfuscated_score is None:
            self.status = "missing_data"
        elif self.regression:
            self.status = "regression"
        elif self.improvement:
            self.status = "improvement"
        else:
            self.status = "neutral"


# =============================================================================
# Results Loader
# =============================================================================

class ResultsLoader:
    """Load and parse SPEC CPU results."""

    @staticmethod
    def load_metrics(result_dir: str) -> Dict[str, float]:
        """Load metrics from result directory."""
        metrics = {}
        result_path = Path(result_dir)

        # Try to find metrics.json file
        metrics_file = result_path / 'metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    for metric in data.get('metrics', []):
                        benchmark_name = metric.get('benchmark_name')
                        base_score = metric.get('base_score')
                        runtime = metric.get('runtime_seconds')

                        if benchmark_name and base_score:
                            metrics[benchmark_name] = {
                                'score': base_score,
                                'runtime': runtime,
                                'raw_data': metric
                            }
            except Exception as e:
                print(f"[WARNING] Failed to load metrics.json: {e}", file=sys.stderr)

        # Fallback: Try to find metrics.csv
        if not metrics:
            metrics_csv = result_path / 'metrics.csv'
            if metrics_csv.exists():
                try:
                    with open(metrics_csv, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            benchmark_name = row.get('benchmark_name')
                            try:
                                base_score = float(row.get('base_score', 0)) or None
                                runtime = float(row.get('runtime_seconds', 0)) or None
                            except ValueError:
                                continue

                            if benchmark_name and base_score:
                                metrics[benchmark_name] = {
                                    'score': base_score,
                                    'runtime': runtime,
                                    'raw_data': row
                                }
                except Exception as e:
                    print(f"[WARNING] Failed to load metrics.csv: {e}", file=sys.stderr)

        return metrics


# =============================================================================
# Comparison Generator
# =============================================================================

class ComparisonAnalyzer:
    """Analyze and compare baseline vs obfuscated results."""

    def __init__(self, baseline_metrics: Dict, obfuscated_metrics: Dict):
        self.baseline = baseline_metrics
        self.obfuscated = obfuscated_metrics
        self.comparisons: List[BenchmarkComparison] = []

    def generate_comparisons(self) -> List[BenchmarkComparison]:
        """Generate comparisons for all benchmarks."""
        # Get union of all benchmark names
        all_benchmarks = set(self.baseline.keys()) | set(self.obfuscated.keys())

        for benchmark in sorted(all_benchmarks):
            baseline_data = self.baseline.get(benchmark)
            obfuscated_data = self.obfuscated.get(benchmark)

            comparison = BenchmarkComparison(
                benchmark_name=benchmark,
                baseline_score=baseline_data['score'] if baseline_data else None,
                obfuscated_score=obfuscated_data['score'] if obfuscated_data else None,
                baseline_runtime=baseline_data['runtime'] if baseline_data else None,
                obfuscated_runtime=obfuscated_data['runtime'] if obfuscated_data else None,
            )
            self.comparisons.append(comparison)

        return self.comparisons

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics."""
        if not self.comparisons:
            return {}

        valid_comparisons = [
            c for c in self.comparisons
            if c.baseline_score is not None and c.obfuscated_score is not None
        ]

        perf_deltas = [c.performance_delta_pct for c in valid_comparisons
                       if c.performance_delta_pct is not None]
        runtime_deltas = [c.runtime_delta_pct for c in valid_comparisons
                          if c.runtime_delta_pct is not None]

        stats = {
            'total_benchmarks': len(self.comparisons),
            'valid_comparisons': len(valid_comparisons),
            'regressions': sum(1 for c in valid_comparisons if c.regression),
            'improvements': sum(1 for c in valid_comparisons if c.improvement),
            'neutral': sum(1 for c in valid_comparisons if c.status == 'neutral'),
            'missing_data': sum(1 for c in self.comparisons if c.status == 'missing_data'),
        }

        if perf_deltas:
            stats['avg_performance_delta_pct'] = mean(perf_deltas)
            stats['median_performance_delta_pct'] = median(perf_deltas)
            stats['worst_regression_pct'] = min(perf_deltas)
            stats['best_improvement_pct'] = max(perf_deltas)

        if runtime_deltas:
            stats['avg_runtime_delta_pct'] = mean(runtime_deltas)
            stats['median_runtime_delta_pct'] = median(runtime_deltas)

        return stats


# =============================================================================
# Report Generators
# =============================================================================

class HTMLReportGenerator:
    """Generate HTML comparison report."""

    @staticmethod
    def generate(comparisons: List[BenchmarkComparison], summary: Dict,
                 config_name: str, output_file: str) -> bool:
        """Generate HTML report."""
        try:
            html = HTMLReportGenerator._build_html(comparisons, summary, config_name)

            with open(output_file, 'w') as f:
                f.write(html)

            print(f"[SUCCESS] HTML report generated: {output_file}", file=sys.stderr)
            return True

        except IOError as e:
            print(f"[ERROR] Failed to write HTML report: {e}", file=sys.stderr)
            return False

    @staticmethod
    def _build_html(comparisons: List[BenchmarkComparison], summary: Dict,
                    config_name: str) -> str:
        """Build HTML content."""
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        # Build table rows
        table_rows = []
        for comp in comparisons:
            status_class = f"status-{comp.status}"
            delta_str = f"{comp.performance_delta_pct:+.2f}%" if comp.performance_delta_pct else "N/A"
            runtime_str = f"{comp.runtime_delta_pct:+.2f}%" if comp.runtime_delta_pct else "N/A"

            row = f"""
            <tr class="{status_class}">
                <td>{comp.benchmark_name}</td>
                <td>{comp.baseline_score or 'N/A':.2f if comp.baseline_score else 'N/A'}</td>
                <td>{comp.obfuscated_score or 'N/A':.2f if comp.obfuscated_score else 'N/A'}</td>
                <td class="delta-cell">{delta_str}</td>
                <td>{runtime_str}</td>
                <td>{comp.status.upper()}</td>
            </tr>
            """
            table_rows.append(row)

        table_html = "\n".join(table_rows)

        # Build summary section
        summary_items = []
        for key, value in summary.items():
            if isinstance(value, float):
                summary_items.append(f"<li><strong>{key}:</strong> {value:.2f}</li>")
            else:
                summary_items.append(f"<li><strong>{key}:</strong> {value}</li>")
        summary_html = "\n".join(summary_items)

        # Build complete HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPEC CPU 2017 Comparison Report - {config_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 30px;
            font-size: 0.9em;
            color: #555;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }}
        .summary-card.regression {{ border-left-color: #e74c3c; }}
        .summary-card.improvement {{ border-left-color: #27ae60; }}
        .summary-card h3 {{
            color: #2c3e50;
            font-size: 0.9em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .summary-card .value.negative {{ color: #e74c3c; }}
        .summary-card .value.positive {{ color: #27ae60; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.95em;
        }}
        thead {{
            background: #34495e;
            color: white;
            font-weight: 600;
        }}
        th {{
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #2c3e50;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tbody tr:hover {{
            background: #f9f9f9;
        }}
        .status-regression {{
            background: #fadbd8;
        }}
        .status-improvement {{
            background: #d5f4e6;
        }}
        .status-neutral {{
            background: #fef9e7;
        }}
        .delta-cell {{
            font-weight: 600;
        }}
        .delta-cell.negative {{
            color: #e74c3c;
        }}
        .delta-cell.positive {{
            color: #27ae60;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            font-size: 0.85em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SPEC CPU 2017 Comparison Report</h1>
        <div class="metadata">
            <strong>Configuration:</strong> {config_name}<br>
            <strong>Generated:</strong> {timestamp}<br>
            <strong>Comparison Type:</strong> Baseline vs Obfuscated
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Total Benchmarks</h3>
                <div class="value">{summary.get('total_benchmarks', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Valid Comparisons</h3>
                <div class="value">{summary.get('valid_comparisons', 0)}</div>
            </div>
            <div class="summary-card regression">
                <h3>Regressions</h3>
                <div class="value negative">{summary.get('regressions', 0)}</div>
            </div>
            <div class="summary-card improvement">
                <h3>Improvements</h3>
                <div class="value positive">+{summary.get('improvements', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Performance Impact</h3>
                <div class="value {'negative' if summary.get('avg_performance_delta_pct', 0) < 0 else 'positive'}">
                    {summary.get('avg_performance_delta_pct', 0):+.2f}%
                </div>
            </div>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Baseline Score</th>
                    <th>Obfuscated Score</th>
                    <th>Performance Δ</th>
                    <th>Runtime Δ</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_html}
            </tbody>
        </table>

        <div class="footer">
            <p>Report generated by SPEC CPU 2017 Comparison Tool</p>
            <p>Lower performance scores may indicate overhead from obfuscation transformations.</p>
        </div>
    </div>
</body>
</html>
"""
        return html


class CSVExporter:
    """Export comparison results to CSV."""

    @staticmethod
    def export(comparisons: List[BenchmarkComparison], output_file: str) -> bool:
        """Export comparisons to CSV."""
        try:
            fieldnames = [
                'benchmark_name',
                'baseline_score',
                'obfuscated_score',
                'performance_delta_pct',
                'baseline_runtime',
                'obfuscated_runtime',
                'runtime_delta_pct',
                'status'
            ]

            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for comp in comparisons:
                    writer.writerow({
                        'benchmark_name': comp.benchmark_name,
                        'baseline_score': comp.baseline_score,
                        'obfuscated_score': comp.obfuscated_score,
                        'performance_delta_pct': comp.performance_delta_pct,
                        'baseline_runtime': comp.baseline_runtime,
                        'obfuscated_runtime': comp.obfuscated_runtime,
                        'runtime_delta_pct': comp.runtime_delta_pct,
                        'status': comp.status
                    })

            print(f"[SUCCESS] CSV export generated: {output_file}", file=sys.stderr)
            return True

        except IOError as e:
            print(f"[ERROR] Failed to write CSV file: {e}", file=sys.stderr)
            return False


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate SPEC CPU 2017 comparison reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate HTML report comparing baseline and obfuscated results
  %(prog)s /path/to/baseline/results /path/to/obfuscated/results

  # Specify obfuscation configuration name
  %(prog)s /path/to/baseline /path/to/obfuscated --config layer1-2

  # Generate all output formats
  %(prog)s /path/to/baseline /path/to/obfuscated --format both

  # Save to custom output directory
  %(prog)s /path/to/baseline /path/to/obfuscated --output-dir ./reports
        """
    )

    parser.add_argument(
        'baseline_path',
        help='Path to baseline SPEC CPU results directory'
    )

    parser.add_argument(
        'obfuscated_path',
        help='Path to obfuscated SPEC CPU results directory'
    )

    parser.add_argument(
        '--config',
        default='unknown-config',
        help='Obfuscation configuration name for report title'
    )

    parser.add_argument(
        '--format',
        choices=['html', 'json', 'csv', 'both'],
        default='html',
        help='Output format (default: html)'
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory for reports (default: obfuscated results directory)'
    )

    args = parser.parse_args()

    # Validate input directories
    baseline_path = Path(args.baseline_path)
    obfuscated_path = Path(args.obfuscated_path)

    if not baseline_path.exists():
        print(f"[ERROR] Baseline directory not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    if not obfuscated_path.exists():
        print(f"[ERROR] Obfuscated directory not found: {obfuscated_path}", file=sys.stderr)
        sys.exit(2)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else obfuscated_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"[INFO] Loading baseline results from: {baseline_path}", file=sys.stderr)
    baseline_metrics = ResultsLoader.load_metrics(str(baseline_path))

    print(f"[INFO] Loading obfuscated results from: {obfuscated_path}", file=sys.stderr)
    obfuscated_metrics = ResultsLoader.load_metrics(str(obfuscated_path))

    if not baseline_metrics or not obfuscated_metrics:
        print("[ERROR] Failed to load results", file=sys.stderr)
        sys.exit(3)

    # Generate comparisons
    print("[INFO] Generating comparisons...", file=sys.stderr)
    analyzer = ComparisonAnalyzer(baseline_metrics, obfuscated_metrics)
    comparisons = analyzer.generate_comparisons()
    summary = analyzer.get_summary_stats()

    if not comparisons:
        print("[ERROR] No comparisons generated", file=sys.stderr)
        sys.exit(3)

    print(f"[SUCCESS] Generated {len(comparisons)} comparison(s)", file=sys.stderr)

    # Export reports
    success = True

    if args.format in ['html', 'both']:
        html_file = output_dir / 'comparison_report.html'
        if not HTMLReportGenerator.generate(comparisons, summary, args.config, str(html_file)):
            success = False

    if args.format in ['json', 'both']:
        json_file = output_dir / 'comparison_metrics.json'
        try:
            with open(json_file, 'w') as f:
                json.dump({
                    'config': args.config,
                    'summary': summary,
                    'comparisons': [asdict(c) for c in comparisons]
                }, f, indent=2)
            print(f"[SUCCESS] JSON metrics exported: {json_file}", file=sys.stderr)
        except IOError as e:
            print(f"[ERROR] Failed to write JSON: {e}", file=sys.stderr)
            success = False

    if args.format in ['csv', 'both']:
        csv_file = output_dir / 'regression_analysis.csv'
        if not CSVExporter.export(comparisons, str(csv_file)):
            success = False

    # Print summary
    print("\n=== Comparison Summary ===")
    print(json.dumps(summary, indent=2))

    sys.exit(0 if success else 5)


if __name__ == '__main__':
    main()
