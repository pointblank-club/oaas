#!/usr/bin/env python3
"""
Aggregated Obfuscation Report Generator

Merges performance reports, static metrics, and decompilation metrics
into a single unified report with weighted obfuscation scoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObfuscationReportAggregator:
    """Aggregates and generates unified obfuscation reports."""

    # Weights for final obfuscation score calculation
    SCORE_WEIGHTS = {
        "performance_cost": 0.20,      # 20% - Performance impact
        "binary_complexity": 0.25,     # 25% - Complexity metrics
        "cfg_distortion": 0.25,        # 25% - Control flow distortion
        "decompilation_difficulty": 0.30,  # 30% - Decompilation impact
    }

    def __init__(self):
        """Initialize the report aggregator."""
        self.logger = logging.getLogger(__name__)

    def aggregate_reports(
        self,
        config_name: str,
        performance_report_path: Optional[Path] = None,
        metrics_report_path: Optional[Path] = None,
        security_report_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate multiple reports into a unified report.

        Args:
            config_name: Name of the obfuscation configuration
            performance_report_path: Path to PTS performance report (JSON)
            metrics_report_path: Path to static metrics report (JSON)
            security_report_path: Path to security analysis report (JSON)
            output_dir: Output directory for generated reports

        Returns:
            Dictionary with aggregated report
        """
        timestamp = datetime.now().isoformat()

        # Load individual reports
        performance_data = self._load_json_report(performance_report_path)
        metrics_data = self._load_json_report(metrics_report_path)
        security_data = self._load_json_report(security_report_path)

        # Build aggregated report
        report = {
            "metadata": {
                "config_name": config_name,
                "timestamp": timestamp,
                "report_version": "1.0",
            },
            "performance_overhead_summary": self._build_performance_summary(performance_data),
            "binary_growth_complexity_summary": self._build_complexity_summary(metrics_data),
            "cfg_instruction_metrics": self._build_cfg_metrics(metrics_data),
            "decompilation_difficulty_report": self._build_decompilation_report(security_data),
            "obfuscation_score": self._compute_obfuscation_score(
                performance_data, metrics_data, security_data
            ),
        }

        # Save reports if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_reports(report, output_dir)

        return report

    def _load_json_report(self, report_path: Optional[Path]) -> Optional[Dict]:
        """Load JSON report from file."""
        if not report_path or not Path(report_path).exists():
            self.logger.warning(f"Report not found: {report_path}")
            return None

        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load report {report_path}: {e}")
            return None

    def _build_performance_summary(self, performance_data: Optional[Dict]) -> Dict:
        """Build performance overhead summary section."""
        if not performance_data:
            return self._empty_performance_summary()

        return {
            "benchmark_suite": performance_data.get("benchmark_suite", "unknown"),
            "performance_metrics": {
                "baseline_throughput": performance_data.get("baseline_throughput"),
                "obfuscated_throughput": performance_data.get("obfuscated_throughput"),
                "overhead_percent": performance_data.get("overhead_percent", 0.0),
                "latency_increase_ms": performance_data.get("latency_increase_ms", 0.0),
            },
            "affected_benchmarks": performance_data.get("affected_benchmarks", []),
            "acceptable": performance_data.get("acceptable", True),
        }

    def _build_complexity_summary(self, metrics_data: Optional[Dict]) -> Dict:
        """Build binary growth and complexity summary."""
        if not metrics_data:
            return self._empty_complexity_summary()

        baseline = metrics_data.get("baseline", {})
        obfuscated = metrics_data.get("obfuscated", {})
        comparison = metrics_data.get("comparison", {})

        # Calculate average growth across all obfuscated binaries
        size_increases = [
            c.get("file_size_increase_percent", 0)
            for c in comparison.values()
        ]
        avg_size_increase = sum(size_increases) / len(size_increases) if size_increases else 0.0

        entropy_increases = [
            c.get("entropy_increase", 0)
            for c in comparison.values()
        ]
        avg_entropy_increase = sum(entropy_increases) / len(entropy_increases) if entropy_increases else 0.0

        return {
            "baseline_size_bytes": baseline.get("file_size_bytes", 0),
            "obfuscated_size_bytes": obfuscated.get("file_size_bytes", 0) if obfuscated else 0,
            "average_size_increase_percent": round(avg_size_increase, 2),
            "text_section_growth_bytes": baseline.get("text_section_size", 0),
            "function_count_baseline": baseline.get("num_functions", 0),
            "function_count_obfuscated": obfuscated.get("num_functions", 0) if obfuscated else 0,
            "avg_entropy_increase": round(avg_entropy_increase, 3),
            "complexity_increased": avg_size_increase > 5.0 or avg_entropy_increase > 0.5,
        }

    def _build_cfg_metrics(self, metrics_data: Optional[Dict]) -> Dict:
        """Build CFG and instruction metrics."""
        if not metrics_data:
            return self._empty_cfg_metrics()

        baseline = metrics_data.get("baseline", {})
        obfuscated = metrics_data.get("obfuscated", {})
        comparison = metrics_data.get("comparison", {})

        # Calculate distortion metrics
        bb_deltas = [
            abs(c.get("basic_block_count_delta", 0))
            for c in comparison.values()
        ]
        avg_bb_increase = sum(bb_deltas) / len(bb_deltas) if bb_deltas else 0

        instr_deltas = [
            abs(c.get("instruction_count_delta", 0))
            for c in comparison.values()
        ]
        avg_instr_increase = sum(instr_deltas) / len(instr_deltas) if instr_deltas else 0

        return {
            "baseline_basic_blocks": baseline.get("num_basic_blocks", 0),
            "obfuscated_basic_blocks": obfuscated.get("num_basic_blocks", 0) if obfuscated else 0,
            "avg_basic_block_increase": round(avg_bb_increase, 2),
            "baseline_instructions": baseline.get("instruction_count", 0),
            "obfuscated_instructions": obfuscated.get("instruction_count", 0) if obfuscated else 0,
            "avg_instruction_increase": round(avg_instr_increase, 2),
            "baseline_cyclomatic_complexity": round(
                baseline.get("cyclomatic_complexity", 1.0), 2
            ),
            "obfuscated_cyclomatic_complexity": round(
                obfuscated.get("cyclomatic_complexity", 1.0), 2
            ) if obfuscated else 1.0,
            "cfg_distorted": avg_bb_increase > 10,
        }

    def _build_decompilation_report(self, security_data: Optional[Dict]) -> Dict:
        """Build decompilation difficulty report."""
        if not security_data:
            return self._empty_decompilation_report()

        return {
            "irreducible_cfg_detected": security_data.get("irreducible_cfg_detected", False),
            "irreducible_cfg_percentage": security_data.get("irreducible_cfg_percentage", 0.0),
            "opaque_predicates_count": security_data.get("opaque_predicates_count", 0),
            "opaque_predicates_percentage": security_data.get("opaque_predicates_percentage", 0.0),
            "basic_blocks_recovered": security_data.get("basic_blocks_recovered", 0),
            "basic_block_recovery_percentage": security_data.get("recovery_percentage", 0.0),
            "string_obfuscation_ratio": round(
                security_data.get("string_obfuscation_ratio", 0.0), 3
            ),
            "symbol_obfuscation_ratio": round(
                security_data.get("symbol_obfuscation_ratio", 0.0), 3
            ),
            "decompilation_readability_score": round(
                security_data.get("decompilation_readability_score", 5.0), 2
            ),
            "analysis_method": security_data.get("analysis_method", "unknown"),
            "recommendation": self._decompilation_difficulty_recommendation(security_data),
        }

    def _decompilation_difficulty_recommendation(self, security_data: Dict) -> str:
        """Generate recommendation based on decompilation metrics."""
        readability = security_data.get("decompilation_readability_score", 5.0)
        opaque_count = security_data.get("opaque_predicates_count", 0)
        recovery_pct = security_data.get("recovery_percentage", 100.0)

        if readability < 2.0 and opaque_count > 5 and recovery_pct < 50:
            return "Excellent - Very difficult to decompile, strong anti-RE protection"
        elif readability < 4.0 and opaque_count > 3:
            return "Good - Moderately difficult to decompile, effective protection"
        elif readability < 6.0:
            return "Moderate - Some protection, but decompilable with effort"
        else:
            return "Weak - Limited decompilation protection"

    def _compute_obfuscation_score(
        self,
        performance_data: Optional[Dict],
        metrics_data: Optional[Dict],
        security_data: Optional[Dict],
    ) -> Dict:
        """
        Compute final weighted obfuscation score (0-10).

        Score components:
        - Performance cost (0-10): lower overhead = higher score
        - Binary complexity (0-10): higher complexity = higher score
        - CFG distortion (0-10): more distorted = higher score
        - Decompilation difficulty (0-10): harder to decompile = higher score
        """
        scores = {}

        # Performance cost (lower overhead = higher score)
        overhead = performance_data.get("overhead_percent", 0.0) if performance_data else 0.0
        performance_score = max(0.0, 10.0 - (overhead / 10.0))
        scores["performance_cost"] = round(performance_score, 2)

        # Binary complexity (size and entropy increase)
        complexity_data = self._build_complexity_summary(metrics_data)
        size_increase = complexity_data.get("average_size_increase_percent", 0.0)
        entropy_increase = complexity_data.get("avg_entropy_increase", 0.0)
        complexity_score = min(10.0, (size_increase / 5.0) + (entropy_increase * 10.0))
        scores["binary_complexity"] = round(complexity_score, 2)

        # CFG distortion
        cfg_data = self._build_cfg_metrics(metrics_data)
        bb_increase = cfg_data.get("avg_basic_block_increase", 0)
        cc_obf = cfg_data.get("obfuscated_cyclomatic_complexity", 1.0)
        cc_base = cfg_data.get("baseline_cyclomatic_complexity", 1.0)
        cfg_score = min(
            10.0,
            (bb_increase / 20.0) + ((cc_obf / max(cc_base, 1.0) - 1.0) * 5.0)
        )
        scores["cfg_distortion"] = round(cfg_score, 2)

        # Decompilation difficulty
        decompilation_data = self._build_decompilation_report(security_data)
        readability = decompilation_data.get("decompilation_readability_score", 5.0)
        decompilation_score = 10.0 - readability  # Inverse mapping
        scores["decompilation_difficulty"] = round(decompilation_score, 2)

        # Compute weighted final score
        final_score = (
            scores["performance_cost"] * self.SCORE_WEIGHTS["performance_cost"] +
            scores["binary_complexity"] * self.SCORE_WEIGHTS["binary_complexity"] +
            scores["cfg_distortion"] * self.SCORE_WEIGHTS["cfg_distortion"] +
            scores["decompilation_difficulty"] * self.SCORE_WEIGHTS["decompilation_difficulty"]
        )

        return {
            "component_scores": scores,
            "weights": self.SCORE_WEIGHTS,
            "final_score": round(min(10.0, max(0.0, final_score)), 2),
            "rating": self._score_to_rating(final_score),
            "interpretation": self._score_interpretation(final_score),
        }

    def _score_to_rating(self, score: float) -> str:
        """Convert score to text rating."""
        if score >= 8.5:
            return "A+"
        elif score >= 8.0:
            return "A"
        elif score >= 7.0:
            return "B+"
        elif score >= 6.0:
            return "B"
        elif score >= 5.0:
            return "C"
        else:
            return "D"

    def _score_interpretation(self, score: float) -> str:
        """Provide interpretation of obfuscation score."""
        if score >= 8.5:
            return "Excellent obfuscation with strong protection against reverse engineering"
        elif score >= 8.0:
            return "Very good obfuscation with high effectiveness"
        elif score >= 7.0:
            return "Good obfuscation with solid protection"
        elif score >= 6.0:
            return "Acceptable obfuscation with moderate protection"
        elif score >= 5.0:
            return "Adequate protection but with room for improvement"
        else:
            return "Limited obfuscation effectiveness"

    def _save_reports(self, report: Dict, output_dir: Path) -> None:
        """Save reports in JSON, Markdown, and HTML formats."""
        # JSON report
        json_file = output_dir / "final_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"Saved JSON report to {json_file}")

        # Markdown report
        md_file = output_dir / "final_report.md"
        self._write_markdown_report(report, md_file)
        self.logger.info(f"Saved Markdown report to {md_file}")

        # HTML report
        html_file = output_dir / "final_report.html"
        self._write_html_report(report, html_file)
        self.logger.info(f"Saved HTML report to {html_file}")

    def _write_markdown_report(self, report: Dict, output_file: Path) -> None:
        """Write comprehensive Markdown report."""
        with open(output_file, 'w') as f:
            # Header
            f.write("# Obfuscation Aggregate Report\n\n")
            metadata = report.get("metadata", {})
            f.write(f"**Configuration:** {metadata.get('config_name')}\n\n")
            f.write(f"**Timestamp:** {metadata.get('timestamp')}\n\n")
            f.write(f"**Version:** {metadata.get('report_version')}\n\n")

            # Obfuscation Score (prominent)
            score_data = report.get("obfuscation_score", {})
            final_score = score_data.get("final_score", 0.0)
            rating = score_data.get("rating", "N/A")
            f.write("## 🎯 Final Obfuscation Score\n\n")
            f.write(f"**Score:** {final_score}/10.0 ({rating})\n\n")
            f.write(f"**Interpretation:** {score_data.get('interpretation', 'N/A')}\n\n")

            # Component Scores
            component_scores = score_data.get("component_scores", {})
            f.write("### Component Breakdown\n\n")
            f.write("| Component | Score | Weight |\n")
            f.write("|-----------|-------|--------|\n")
            weights = score_data.get("weights", {})
            for component, score in component_scores.items():
                weight = weights.get(component, 0.0)
                f.write(f"| {component} | {score}/10.0 | {weight * 100:.0f}% |\n")

            # Performance Summary
            f.write("\n## 📊 Performance Overhead Summary\n\n")
            perf = report.get("performance_overhead_summary", {})
            f.write(f"- **Benchmark Suite:** {perf.get('benchmark_suite', 'N/A')}\n")
            f.write(f"- **Overhead:** {perf.get('performance_metrics', {}).get('overhead_percent', 0.0)}%\n")
            f.write(f"- **Acceptable:** {'✓' if perf.get('acceptable', False) else '✗'}\n")

            # Binary Complexity Summary
            f.write("\n## 📦 Binary Growth & Complexity Summary\n\n")
            complexity = report.get("binary_growth_complexity_summary", {})
            f.write(f"- **Baseline Size:** {complexity.get('baseline_size_bytes', 0)} bytes\n")
            f.write(f"- **Obfuscated Size:** {complexity.get('obfuscated_size_bytes', 0)} bytes\n")
            f.write(f"- **Average Growth:** {complexity.get('average_size_increase_percent', 0.0)}%\n")
            f.write(f"- **Avg Entropy Increase:** {complexity.get('avg_entropy_increase', 0.0)}\n")

            # CFG Metrics
            f.write("\n## 🔀 CFG & Instruction Metrics\n\n")
            cfg = report.get("cfg_instruction_metrics", {})
            f.write(f"- **Baseline Basic Blocks:** {cfg.get('baseline_basic_blocks', 0)}\n")
            f.write(f"- **Obfuscated Basic Blocks:** {cfg.get('obfuscated_basic_blocks', 0)}\n")
            f.write(f"- **BB Increase:** {cfg.get('avg_basic_block_increase', 0.0)}\n")
            f.write(f"- **Baseline CC:** {cfg.get('baseline_cyclomatic_complexity', 1.0)}\n")
            f.write(f"- **Obfuscated CC:** {cfg.get('obfuscated_cyclomatic_complexity', 1.0)}\n")

            # Decompilation Difficulty
            f.write("\n## 🔐 Decompilation Difficulty Report\n\n")
            decomp = report.get("decompilation_difficulty_report", {})
            f.write(f"- **Readability Score:** {decomp.get('decompilation_readability_score', 5.0)}/10.0\n")
            f.write(f"- **Opaque Predicates:** {decomp.get('opaque_predicates_count', 0)}\n")
            f.write(f"- **Irreducible CFG:** {decomp.get('irreducible_cfg_percentage', 0.0)}%\n")
            f.write(f"- **BB Recovery:** {decomp.get('basic_block_recovery_percentage', 100.0)}%\n")
            f.write(f"- **String Obfuscation:** {decomp.get('string_obfuscation_ratio', 0.0):.1%}\n")
            f.write(f"- **Symbol Obfuscation:** {decomp.get('symbol_obfuscation_ratio', 0.0):.1%}\n")
            f.write(f"- **Recommendation:** {decomp.get('recommendation', 'N/A')}\n")

    def _write_html_report(self, report: Dict, output_file: Path) -> None:
        """Write comprehensive HTML report."""
        html_content = self._generate_html_template(report)
        with open(output_file, 'w') as f:
            f.write(html_content)

    def _generate_html_template(self, report: Dict) -> str:
        """Generate HTML report template."""
        metadata = report.get("metadata", {})
        score_data = report.get("obfuscation_score", {})
        final_score = score_data.get("final_score", 0.0)
        rating = score_data.get("rating", "N/A")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Obfuscation Report - {metadata.get('config_name', 'Unknown')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .score-box {{
            background: white;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
        }}
        .rating {{
            font-size: 24px;
            color: #764ba2;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f9f9f9;
            font-weight: 600;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Obfuscation Analysis Report</h1>
        <p><strong>Configuration:</strong> {metadata.get('config_name', 'N/A')}</p>
        <p><strong>Generated:</strong> {metadata.get('timestamp', 'N/A')}</p>
    </div>

    <div class="score-box">
        <h2>Final Obfuscation Score</h2>
        <div class="score-value">{final_score}/10.0</div>
        <div class="rating">Rating: {rating}</div>
        <p>{score_data.get('interpretation', 'N/A')}</p>
    </div>

    <h2>Component Scores</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Score</th>
            <th>Weight</th>
        </tr>
"""
        component_scores = score_data.get("component_scores", {})
        weights = score_data.get("weights", {})
        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            html += f"<tr><td>{component}</td><td>{score}/10.0</td><td>{weight * 100:.0f}%</td></tr>\n"

        html += """
    </table>

    <h2>Detailed Metrics</h2>
"""

        # Add performance metrics
        perf = report.get("performance_overhead_summary", {})
        html += f"""
    <h3>Performance Overhead</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Overhead</div>
            <div class="metric-value">{perf.get('performance_metrics', {}).get('overhead_percent', 0.0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Benchmark Suite</div>
            <div class="metric-value">{perf.get('benchmark_suite', 'N/A')}</div>
        </div>
    </div>
"""

        # Add complexity metrics
        complexity = report.get("binary_growth_complexity_summary", {})
        html += f"""
    <h3>Binary Complexity</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Size Growth</div>
            <div class="metric-value">{complexity.get('average_size_increase_percent', 0.0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Entropy Increase</div>
            <div class="metric-value">{complexity.get('avg_entropy_increase', 0.0):.2f}</div>
        </div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    # Empty/default responses
    @staticmethod
    def _empty_performance_summary() -> Dict:
        return {
            "benchmark_suite": "unknown",
            "performance_metrics": {
                "baseline_throughput": None,
                "obfuscated_throughput": None,
                "overhead_percent": 0.0,
            },
            "affected_benchmarks": [],
            "acceptable": True,
        }

    @staticmethod
    def _empty_complexity_summary() -> Dict:
        return {
            "baseline_size_bytes": 0,
            "obfuscated_size_bytes": 0,
            "average_size_increase_percent": 0.0,
            "text_section_growth_bytes": 0,
            "function_count_baseline": 0,
            "function_count_obfuscated": 0,
            "avg_entropy_increase": 0.0,
            "complexity_increased": False,
        }

    @staticmethod
    def _empty_cfg_metrics() -> Dict:
        return {
            "baseline_basic_blocks": 0,
            "obfuscated_basic_blocks": 0,
            "avg_basic_block_increase": 0.0,
            "baseline_instructions": 0,
            "obfuscated_instructions": 0,
            "avg_instruction_increase": 0.0,
            "baseline_cyclomatic_complexity": 1.0,
            "obfuscated_cyclomatic_complexity": 1.0,
            "cfg_distorted": False,
        }

    @staticmethod
    def _empty_decompilation_report() -> Dict:
        return {
            "irreducible_cfg_detected": False,
            "irreducible_cfg_percentage": 0.0,
            "opaque_predicates_count": 0,
            "opaque_predicates_percentage": 0.0,
            "basic_blocks_recovered": 0,
            "basic_block_recovery_percentage": 100.0,
            "string_obfuscation_ratio": 0.0,
            "symbol_obfuscation_ratio": 0.0,
            "decompilation_readability_score": 5.0,
            "analysis_method": "unknown",
            "recommendation": "No analysis available",
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate aggregated obfuscation report')
    parser.add_argument('config_name', help='Configuration name')
    parser.add_argument('--performance', type=Path, help='Path to performance report')
    parser.add_argument('--metrics', type=Path, help='Path to metrics report')
    parser.add_argument('--security', type=Path, help='Path to security analysis report')
    parser.add_argument('--output', type=Path, help='Output directory')

    args = parser.parse_args()

    aggregator = ObfuscationReportAggregator()
    report = aggregator.aggregate_reports(
        config_name=args.config_name,
        performance_report_path=args.performance,
        metrics_report_path=args.metrics,
        security_report_path=args.security,
        output_dir=args.output,
    )

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
