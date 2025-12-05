from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .exceptions import ReportGenerationError
from .utils import ensure_directory, get_timestamp, write_html, write_json


class ObfuscationReport:
    """Generate comprehensive obfuscation report per SIH requirements."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def generate_report(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive obfuscation report with proper null-safety and validation."""
        try:
            # ✅ FIX: Add baseline failure handling
            baseline_metrics = job_data.get("baseline_metrics", {})
            baseline_status = "success"
            if baseline_metrics and baseline_metrics.get("file_size", 0) <= 0:
                baseline_status = "failed"

            # ✅ FIX: Validate numeric fields with safe defaults
            def safe_float(val, default=0.0):
                try:
                    f = float(val) if val is not None else default
                    return f if not (f != f) else default  # NaN check
                except (TypeError, ValueError):
                    return default

            def safe_int(val, default=0):
                try:
                    return int(val) if val is not None else default
                except (TypeError, ValueError):
                    return default

            # ✅ FIX: Handle null sections with safe defaults
            string_obf = job_data.get("string_obfuscation") or {}
            symbol_obf = job_data.get("symbol_obfuscation") or {}
            fake_loops = job_data.get("fake_loops_inserted") or {}
            bogus_code = job_data.get("bogus_code_info") or {}
            cycles = job_data.get("cycles_completed") or {}
            comparison = job_data.get("comparison") or {}
            # ✅ FIX: Get baseline compiler metadata (for reproducibility)
            baseline_compiler = job_data.get("baseline_compiler") or {}

            report = {
                "input_parameters": {
                    "source_file": job_data.get("source_file"),
                    "platform": job_data.get("platform"),
                    "obfuscation_level": job_data.get("obfuscation_level"),
                    "requested_passes": job_data.get("requested_passes", []),  # What user requested
                    "applied_passes": job_data.get("applied_passes", []),  # What was actually applied
                    "compiler_flags": job_data.get("compiler_flags", []),
                    "timestamp": job_data.get("timestamp", get_timestamp()),
                },
                "warnings": job_data.get("warnings", []),  # Warnings from obfuscation process
                "baseline_status": baseline_status,  # ✅ NEW: Indicate if baseline failed
                "baseline_metrics": baseline_metrics,  # Before obfuscation metrics
                # ✅ FIX: Add baseline compiler metadata for reproducibility and verification
                "baseline_compiler": baseline_compiler,
                "output_attributes": job_data.get("output_attributes", {}),
                "comparison": comparison,  # Before/after comparison
                "comparison_valid": baseline_status == "success",  # ✅ NEW: Validity flag
                "bogus_code_info": bogus_code if bogus_code else self._default_bogus_code(),
                "cycles_completed": cycles if cycles else {"total_cycles": 0, "per_cycle_metrics": []},
                "string_obfuscation": string_obf if string_obf else self._default_string_obfuscation(),
                "fake_loops_inserted": fake_loops if fake_loops else self._default_fake_loops(),
                "symbol_obfuscation": symbol_obf if symbol_obf else self._default_symbol_obfuscation(),
                "obfuscation_score": safe_float(job_data.get("obfuscation_score"), 0.0),
                "symbol_reduction": safe_float(job_data.get("symbol_reduction"), 0.0),
                "function_reduction": safe_float(job_data.get("function_reduction"), 0.0),
                "size_reduction": safe_float(job_data.get("size_reduction"), 0.0),
                "entropy_increase": safe_float(job_data.get("entropy_increase"), 0.0),
                "estimated_re_effort": job_data.get("estimated_re_effort", "4-6 weeks"),
                # ✅ NEW: Test suite results (optional, if tests were run)
                "metadata": job_data.get("metadata"),  # Optional test suite metadata
                "test_results": job_data.get("test_results"),  # Optional test suite results
                "test_metrics": job_data.get("test_metrics"),  # Optional test metrics
                "metrics_reliability": job_data.get("metrics_reliability"),  # Optional reliability status
                "functional_correctness_passed": job_data.get("functional_correctness_passed"),  # Optional functional test result
                "reliability_status": {
                    "level": job_data.get("metrics_reliability", "UNKNOWN"),
                    "warning": job_data.get("reliability_warning", "")
                } if job_data.get("test_results") else None,  # Only include if test results exist
                # ✅ NEW: Advanced IR and binary analysis metrics
                "control_flow_metrics": job_data.get("control_flow_metrics"),
                "instruction_metrics": job_data.get("instruction_metrics"),
                "binary_structure": job_data.get("binary_structure"),
                "pattern_resistance": job_data.get("pattern_resistance"),
                "call_graph_metrics": job_data.get("call_graph_metrics"),
            }
        except Exception as exc:  # pragma: no cover - defensive
            raise ReportGenerationError("Failed to assemble report") from exc
        return report

    def _default_bogus_code(self) -> Dict[str, Any]:
        """✅ FIX: Default safe values for bogus code section."""
        return {
            "dead_code_blocks": 0,
            "opaque_predicates": 0,
            "junk_instructions": 0,
            "code_bloat_percentage": 0,
        }

    def _default_string_obfuscation(self) -> Dict[str, Any]:
        """✅ FIX: Default safe values for string obfuscation section."""
        return {
            "total_strings": 0,
            "encrypted_strings": 0,
            "encryption_method": "none",
            "encryption_percentage": 0.0,
        }

    def _default_fake_loops(self) -> Dict[str, Any]:
        """✅ FIX: Default safe values for fake loops section."""
        return {
            "count": 0,
            "types": [],
            "locations": [],
        }

    def _default_symbol_obfuscation(self) -> Dict[str, Any]:
        """✅ FIX: Default safe values for symbol obfuscation section."""
        return {
            "enabled": False,
            "symbols_obfuscated": 0,
            "algorithm": "N/A",
        }

    def export(self, report: Dict[str, Any], job_id: str, formats: List[str]) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        for fmt in formats:
            fmt_lower = fmt.lower()
            if fmt_lower == "json":
                path = self.output_dir / f"{job_id}.json"
                write_json(path, report)
                outputs["json"] = path
            elif fmt_lower == "html":
                path = self.output_dir / f"{job_id}.html"
                html = self._render_html(report, job_id)
                write_html(path, html)
                outputs["html"] = path
            elif fmt_lower in ["pdf", "markdown"]:
                path = self.output_dir / f"{job_id}.{fmt_lower}"
                if fmt_lower == "pdf":
                    self._write_pdf(path, report, job_id)
                else:
                    markdown = self._render_markdown(report, job_id)
                    path.write_text(markdown, encoding="utf-8")
                outputs[fmt_lower] = path
        return outputs

    def _render_html(self, report: Dict[str, Any], job_id: str) -> str:
        """Render a clean, simple HTML report matching existing website style."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Extract data for easier access
        input_params = report.get("input_parameters", {})
        warnings = report.get("warnings", [])
        baseline_metrics = report.get("baseline_metrics", {})
        output_attrs = report.get("output_attributes", {})
        comparison = report.get("comparison", {})
        bogus_code = report.get("bogus_code_info", {})
        cycles = report.get("cycles_completed", {})
        string_obf = report.get("string_obfuscation", {})
        fake_loops = report.get("fake_loops_inserted", {})
        symbol_obf = report.get("symbol_obfuscation", {})

        # Format warnings list
        warnings_html = ""
        if warnings:
            for i, warning in enumerate(warnings, 1):
                warnings_html += f'<tr><td>{i}</td><td>{warning}</td></tr>'
        else:
            warnings_html = '<tr><td colspan="2">No warnings - all obfuscation techniques applied successfully</td></tr>'

        # Format file size
        file_size = output_attrs.get("file_size", 0)
        file_size_kb = file_size / 1024 if file_size > 0 else 0

        # Format obfuscation methods
        methods = output_attrs.get("obfuscation_methods", [])
        methods_list = ", ".join(methods) if methods else "None"

        # Format compiler flags
        flags = input_params.get("compiler_flags", [])
        flags_list = " ".join(flags) if flags else "None"

        # Format sections
        sections = output_attrs.get("sections", {})
        sections_html = ""
        if sections:
            for name, size in sections.items():
                sections_html += f'<tr><td>{name}</td><td>{size} bytes</td></tr>'
        else:
            sections_html = '<tr><td colspan="2">No section information available</td></tr>'

        # Cycles information
        total_cycles = cycles.get("total_cycles", 1)
        per_cycle_metrics = cycles.get("per_cycle_metrics", [])
        cycles_html = ""
        for cycle_info in per_cycle_metrics:
            cycle_num = cycle_info.get("cycle", 0)
            passes = cycle_info.get("passes_applied", [])
            duration = cycle_info.get("duration_ms", 0)
            passes_str = ", ".join(passes) if passes else "None"
            cycles_html += f'<tr><td>Cycle {cycle_num}</td><td>{passes_str}</td><td>{duration} ms</td></tr>'

        # String obfuscation details
        total_strings = string_obf.get("total_strings", 0)
        encrypted_strings = string_obf.get("encrypted_strings", 0)
        encryption_pct = string_obf.get("encryption_percentage", 0.0)
        encryption_method = string_obf.get("encryption_method", "none")

        # Fake loops details
        fake_loop_count = fake_loops.get("count", 0)
        fake_loop_types = fake_loops.get("types", [])
        fake_loops_html = ""
        if fake_loop_count > 0:
            for i, loop_type in enumerate(fake_loop_types, 1):
                location = fake_loops.get("locations", [])[i-1] if i <= len(fake_loops.get("locations", [])) else "Unknown"
                fake_loops_html += f'<tr><td>Loop {i}</td><td>{loop_type}</td><td>{location}</td></tr>'
        else:
            fake_loops_html = '<tr><td colspan="3">No fake loops inserted</td></tr>'

        # Symbol obfuscation details
        symbol_enabled = symbol_obf.get("enabled", False)
        symbols_obfuscated = symbol_obf.get("symbols_obfuscated", 0)
        symbol_algorithm = symbol_obf.get("algorithm", "N/A")

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset='utf-8'>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLVM Obfuscation Report - {job_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 24px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #000;
        }}

        h2 {{
            font-size: 18px;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid #000;
        }}

        h3 {{
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .header-info {{
            margin-bottom: 30px;
        }}

        .field {{
            margin: 10px 0;
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 15px;
        }}

        .field-label {{
            font-weight: bold;
        }}

        .field-value {{
            word-wrap: break-word;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        th, td {{
            border: 1px solid #000;
            padding: 10px;
            text-align: left;
        }}

        th {{
            font-weight: bold;
            background: #f0f0f0;
        }}

        code {{
            background: #f5f5f5;
            padding: 2px 5px;
            font-family: monospace;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .summary-card {{
            border: 1px solid #000;
            padding: 15px;
        }}

        .summary-label {{
            font-size: 14px;
            margin-bottom: 5px;
        }}

        .summary-value {{
            font-size: 24px;
            font-weight: bold;
        }}

        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #000;
            text-align: center;
            font-size: 14px;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .comparison-item {{
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 4px;
        }}

        .comparison-item h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }}

        .metric-label {{
            font-weight: 500;
        }}

        .metric-value {{
            font-family: monospace;
        }}

        .change-indicator {{
            margin-top: 8px;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
        }}

        .change-positive {{
            background-color: #d4edda;
            color: #155724;
        }}

        .change-negative {{
            background-color: #f8d7da;
            color: #721c24;
        }}

        .change-neutral {{
            background-color: #e2e3e5;
            color: #383d41;
        }}

        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}

        @media print {{
            body {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header-info">
        <h1>LLVM Obfuscation Report</h1>
        <div class="field">
            <div class="field-label">Job ID:</div>
            <div class="field-value"><code>{job_id}</code></div>
        </div>
        <div class="field">
            <div class="field-label">Generated:</div>
            <div class="field-value">{timestamp}</div>
        </div>
    </div>

    <!-- Key Metrics Summary -->
    <h2>Key Metrics</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-label">Obfuscation Score</div>
            <div class="summary-value">{report.get('obfuscation_score', 0)}/100</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Symbol Reduction</div>
            <div class="summary-value">{report.get('symbol_reduction', 0)}%</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">Function Reduction</div>
            <div class="summary-value">{report.get('function_reduction', 0)}%</div>
        </div>
        <div class="summary-card">
            <div class="summary-label">RE Effort Estimate</div>
            <div class="summary-value">{report.get('estimated_re_effort', 'N/A')}</div>
        </div>
    </div>

    <!-- Input Parameters -->
    <h2>Input Parameters</h2>
    <div class="field">
        <div class="field-label">Source File:</div>
        <div class="field-value"><code>{input_params.get('source_file', 'N/A')}</code></div>
    </div>
    <div class="field">
        <div class="field-label">Platform:</div>
        <div class="field-value">{input_params.get('platform', 'unknown')}</div>
    </div>
    <div class="field">
        <div class="field-label">Obfuscation Level:</div>
        <div class="field-value">Level {input_params.get('obfuscation_level', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Requested OLLVM Passes:</div>
        <div class="field-value">{", ".join(input_params.get('requested_passes', [])) or "None"}</div>
    </div>
    <div class="field">
        <div class="field-label">Actually Applied OLLVM Passes:</div>
        <div class="field-value"><strong>{", ".join(input_params.get('applied_passes', [])) or "None"}</strong></div>
    </div>
    <div class="field">
        <div class="field-label">Timestamp:</div>
        <div class="field-value">{input_params.get('timestamp', 'N/A')}</div>
    </div>
    <div class="field">
        <div class="field-label">Compiler Flags:</div>
        <div class="field-value"><code>{flags_list}</code></div>
    </div>

    <!-- Warnings and Logs -->
    <h2>Warnings & Processing Logs</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 60px;">#</th>
                <th>Message</th>
            </tr>
        </thead>
        <tbody>
            {warnings_html}
        </tbody>
    </table>

    <!-- Before/After Comparison -->
    {f'''<h2>Before/After Comparison</h2>
    <div class="comparison-grid">
        <div class="comparison-item">
            <h3>File Size</h3>
            <div class="metric-row">
                <span class="metric-label">Before:</span>
                <span class="metric-value">{baseline_metrics.get("file_size", 0) / 1024:.2f} KB</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">After:</span>
                <span class="metric-value">{output_attrs.get("file_size", 0) / 1024:.2f} KB</span>
            </div>
            <div class="change-indicator {("change-negative" if comparison.get("size_change_percent", 0) > 0 else "change-positive" if comparison.get("size_change_percent", 0) < 0 else "change-neutral")}">
                {("+" if comparison.get("size_change_percent", 0) > 0 else "")}{comparison.get("size_change_percent", 0):.2f}% ({comparison.get("size_change", 0):+} bytes)
            </div>
        </div>

        <div class="comparison-item">
            <h3>Symbol Count</h3>
            <div class="metric-row">
                <span class="metric-label">Before:</span>
                <span class="metric-value">{baseline_metrics.get("symbols_count", 0)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">After:</span>
                <span class="metric-value">{output_attrs.get("symbols_count", 0)}</span>
            </div>
            <div class="change-indicator {("change-positive" if comparison.get("symbols_removed", 0) > 0 else "change-neutral")}">
                {comparison.get("symbols_removed", 0)} symbols removed ({comparison.get("symbols_removed_percent", 0):.1f}%)
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(100, abs(comparison.get("symbols_removed_percent", 0)))}%">
                    {comparison.get("symbols_removed_percent", 0):.1f}%
                </div>
            </div>
        </div>

        <div class="comparison-item">
            <h3>Function Count</h3>
            <div class="metric-row">
                <span class="metric-label">Before:</span>
                <span class="metric-value">{baseline_metrics.get("functions_count", 0)}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">After:</span>
                <span class="metric-value">{output_attrs.get("functions_count", 0)}</span>
            </div>
            <div class="change-indicator {("change-positive" if comparison.get("functions_removed", 0) > 0 else "change-neutral")}">
                {comparison.get("functions_removed", 0)} functions hidden ({comparison.get("functions_removed_percent", 0):.1f}%)
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(100, abs(comparison.get("functions_removed_percent", 0)))}%">
                    {comparison.get("functions_removed_percent", 0):.1f}%
                </div>
            </div>
        </div>

        <div class="comparison-item">
            <h3>Binary Entropy</h3>
            <div class="metric-row">
                <span class="metric-label">Before:</span>
                <span class="metric-value">{baseline_metrics.get("entropy", 0):.3f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">After:</span>
                <span class="metric-value">{output_attrs.get("entropy", 0):.3f}</span>
            </div>
            <div class="change-indicator {("change-positive" if comparison.get("entropy_increase", 0) > 0 else "change-neutral")}">
                +{comparison.get("entropy_increase", 0):.3f} entropy increase ({comparison.get("entropy_increase_percent", 0):+.1f}%)
            </div>
        </div>
    </div>''' if baseline_metrics and comparison else ''}

    <!-- Output File Attributes -->
    <h2>Output File Attributes</h2>
    <div class="field">
        <div class="field-label">File Size:</div>
        <div class="field-value">{file_size_kb:.2f} KB ({file_size} bytes)</div>
    </div>
    <div class="field">
        <div class="field-label">Binary Format:</div>
        <div class="field-value">{output_attrs.get('binary_format', 'unknown')}</div>
    </div>
    <div class="field">
        <div class="field-label">Symbol Count:</div>
        <div class="field-value">{output_attrs.get('symbols_count', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Function Count:</div>
        <div class="field-value">{output_attrs.get('functions_count', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Entropy:</div>
        <div class="field-value">{output_attrs.get('entropy', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Obfuscation Methods:</div>
        <div class="field-value">{methods_list}</div>
    </div>

    <h3>Binary Sections</h3>
    <table>
        <thead>
            <tr>
                <th>Section Name</th>
                <th>Size</th>
            </tr>
        </thead>
        <tbody>
            {sections_html}
        </tbody>
    </table>

    <!-- Bogus Code Generation -->
    <h2>Bogus Code Generation</h2>
    <div class="field">
        <div class="field-label">Dead Code Blocks:</div>
        <div class="field-value">{bogus_code.get('dead_code_blocks', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Opaque Predicates:</div>
        <div class="field-value">{bogus_code.get('opaque_predicates', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Junk Instructions:</div>
        <div class="field-value">{bogus_code.get('junk_instructions', 0)}</div>
    </div>
    <div class="field">
        <div class="field-label">Code Bloat Percentage:</div>
        <div class="field-value">{bogus_code.get('code_bloat_percentage', 0)}%</div>
    </div>

    <!-- Obfuscation Cycles -->
    <h2>Obfuscation Cycles</h2>
    <div class="field">
        <div class="field-label">Total Cycles:</div>
        <div class="field-value">{total_cycles}</div>
    </div>

    {f'''<h3>Per-Cycle Breakdown</h3>
    <table>
        <thead>
            <tr>
                <th>Cycle</th>
                <th>Passes Applied</th>
                <th>Duration</th>
            </tr>
        </thead>
        <tbody>
            {cycles_html}
        </tbody>
    </table>''' if cycles_html else ''}

    <!-- String Obfuscation -->
    <h2>String Encryption</h2>
    <div class="field">
        <div class="field-label">Enabled:</div>
        <div class="field-value">{"Yes" if string_obf.get('enabled', False) else "No"}</div>
    </div>
    <div class="field">
        <div class="field-label">Encryption Method:</div>
        <div class="field-value">{encryption_method.upper()}</div>
    </div>
    <div class="field">
        <div class="field-label">Encryption Rate:</div>
        <div class="field-value">{encryption_pct:.1f}%</div>
    </div>

    <!-- Fake Loops -->
    <h2>Fake Loops Inserted</h2>
    <div class="field">
        <div class="field-label">Total Fake Loops:</div>
        <div class="field-value">{fake_loop_count}</div>
    </div>

    {f'''<h3>Loop Details</h3>
    <table>
        <thead>
            <tr>
                <th>Loop ID</th>
                <th>Type</th>
                <th>Location</th>
            </tr>
        </thead>
        <tbody>
            {fake_loops_html}
        </tbody>
    </table>''' if fake_loop_count > 0 else ''}

    <!-- Symbol Obfuscation -->
    <h2>Symbol Obfuscation</h2>
    <div class="field">
        <div class="field-label">Enabled:</div>
        <div class="field-value">{"Yes" if symbol_enabled else "No"}</div>
    </div>
    {f'''<div class="field">
        <div class="field-label">Symbols Renamed:</div>
        <div class="field-value">{symbols_obfuscated}</div>
    </div>
    <div class="field">
        <div class="field-label">Algorithm:</div>
        <div class="field-value">{symbol_algorithm}</div>
    </div>''' if symbol_enabled else ''}

    <!-- Footer -->
    <div class="footer">
        <p>Generated with LLVM Obfuscator API</p>
    </div>
</body>
</html>
        """
        return html

    def _render_markdown(self, report: Dict[str, Any], job_id: str) -> str:
        """Render a markdown version of the report."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        input_params = report.get("input_parameters", {})
        baseline_metrics = report.get("baseline_metrics", {})
        output_attrs = report.get("output_attributes", {})
        comparison = report.get("comparison", {})
        bogus_code = report.get("bogus_code_info", {})
        cycles = report.get("cycles_completed", {})
        string_obf = report.get("string_obfuscation", {})
        fake_loops = report.get("fake_loops_inserted", {})
        symbol_obf = report.get("symbol_obfuscation", {})

        md = f"""# 🛡️ LLVM Obfuscation Report

**Job ID:** `{job_id}`
**Generated:** {timestamp}

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Obfuscation Score | {report.get('obfuscation_score', 0)}/100 |
| Symbol Reduction | {report.get('symbol_reduction', 0)}% |
| Function Reduction | {report.get('function_reduction', 0)}% |
| Estimated RE Effort | {report.get('estimated_re_effort', 'N/A')} |

---

## 📥 Input Parameters

- **Source File:** `{input_params.get('source_file', 'N/A')}`
- **Platform:** {input_params.get('platform', 'unknown')}
- **Obfuscation Level:** Level {input_params.get('obfuscation_level', 0)}
- **Requested OLLVM Passes:** {", ".join(input_params.get('requested_passes', [])) or "None"}
- **Actually Applied OLLVM Passes:** **{", ".join(input_params.get('applied_passes', [])) or "None"}**
- **Timestamp:** {input_params.get('timestamp', 'N/A')}

### Compiler Flags
```
{' '.join(input_params.get('compiler_flags', []))}
```

---

## ⚠️ Warnings & Processing Logs

{self._format_warnings_markdown(report.get('warnings', []))}

---

{self._format_comparison_markdown(baseline_metrics, output_attrs, comparison)}

## 📦 Output File Attributes

- **File Size:** {output_attrs.get('file_size', 0) / 1024:.2f} KB ({output_attrs.get('file_size', 0)} bytes)
- **Binary Format:** {output_attrs.get('binary_format', 'unknown')}
- **Symbol Count:** {output_attrs.get('symbols_count', 0)}
- **Function Count:** {output_attrs.get('functions_count', 0)}
- **Entropy:** {output_attrs.get('entropy', 0)}

### Obfuscation Methods Applied
{", ".join(output_attrs.get('obfuscation_methods', [])) or "None"}

---

## 🔀 Bogus Code Generation

| Type | Count |
|------|-------|
| Dead Code Blocks | {bogus_code.get('dead_code_blocks', 0)} |
| Opaque Predicates | {bogus_code.get('opaque_predicates', 0)} |
| Junk Instructions | {bogus_code.get('junk_instructions', 0)} |
| Code Bloat | {bogus_code.get('code_bloat_percentage', 0)}% |

---

## 🔄 Obfuscation Cycles

**Total Cycles:** {cycles.get('total_cycles', 1)}

---

## 🔐 String Obfuscation

- **Total Strings:** {string_obf.get('total_strings', 0)}
- **Encrypted Strings:** {string_obf.get('encrypted_strings', 0)}
- **Encryption Method:** {string_obf.get('encryption_method', 'none').upper()}
- **Encryption Rate:** {string_obf.get('encryption_percentage', 0.0):.1f}%

---

## ➰ Fake Loops Inserted

**Total Fake Loops:** {fake_loops.get('count', 0)}

---

## 🏷️ Symbol Obfuscation

- **Enabled:** {"Yes" if symbol_obf.get('enabled', False) else "No"}
- **Symbols Renamed:** {symbol_obf.get('symbols_obfuscated', 0)}
- **Algorithm:** {symbol_obf.get('algorithm', 'N/A')}

---

## 📊 Advanced Metrics Dashboard

### Control Flow Analysis
{self._format_control_flow_markdown(report.get('control_flow_metrics'))}

### Instruction-Level Metrics
{self._format_instruction_metrics_markdown(report.get('instruction_metrics'))}

### Binary Structure Analysis
{self._format_binary_structure_markdown(report.get('binary_structure'))}

### Pattern Resistance
{self._format_pattern_resistance_markdown(report.get('pattern_resistance'))}

---

*🤖 Generated with LLVM Obfuscator API*
"""
        return md

    def _format_warnings_markdown(self, warnings: List[str]) -> str:
        """Format warnings list for markdown output."""
        if not warnings:
            return "✅ **No warnings** - All obfuscation techniques applied successfully"

        md = ""
        for i, warning in enumerate(warnings, 1):
            md += f"{i}. {warning}\n"
        return md.strip()

    def _format_comparison_markdown(self, baseline_metrics: Dict[str, Any],
                                   output_attrs: Dict[str, Any],
                                   comparison: Dict[str, Any]) -> str:
        """Format before/after comparison for markdown output."""
        if not baseline_metrics or not comparison:
            return ""

        md = """## 🔄 Before/After Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
"""

        # File Size
        before_size = baseline_metrics.get("file_size", 0) / 1024
        after_size = output_attrs.get("file_size", 0) / 1024
        size_change = comparison.get("size_change_percent", 0)
        size_indicator = "📈" if size_change > 0 else "📉" if size_change < 0 else "➡️"
        md += f"| **File Size** | {before_size:.2f} KB | {after_size:.2f} KB | {size_indicator} {size_change:+.2f}% |\n"

        # Symbol Count
        before_symbols = baseline_metrics.get("symbols_count", 0)
        after_symbols = output_attrs.get("symbols_count", 0)
        symbols_removed = comparison.get("symbols_removed", 0)
        symbols_percent = comparison.get("symbols_removed_percent", 0)
        md += f"| **Symbols** | {before_symbols} | {after_symbols} | ✅ {symbols_removed} removed ({symbols_percent:.1f}%) |\n"

        # Function Count
        before_functions = baseline_metrics.get("functions_count", 0)
        after_functions = output_attrs.get("functions_count", 0)
        functions_removed = comparison.get("functions_removed", 0)
        functions_percent = comparison.get("functions_removed_percent", 0)
        md += f"| **Functions** | {before_functions} | {after_functions} | ✅ {functions_removed} hidden ({functions_percent:.1f}%) |\n"

        # Binary Entropy
        before_entropy = baseline_metrics.get("entropy", 0)
        after_entropy = output_attrs.get("entropy", 0)
        entropy_increase = comparison.get("entropy_increase", 0)
        entropy_percent = comparison.get("entropy_increase_percent", 0)
        md += f"| **Entropy** | {before_entropy:.3f} | {after_entropy:.3f} | 🔒 +{entropy_increase:.3f} ({entropy_percent:+.1f}%) |\n"

        md += "\n---\n\n"
        return md

    def _write_pdf(self, path: Path, report: Dict[str, Any], job_id: str) -> None:
        """Generate a PDF report using ReportLab."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph,
                Spacer, PageBreak, Image
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            # Fallback if reportlab is not available
            ensure_directory(path.parent)
            path.write_text(
                f"PDF Report for Job {job_id}\n\n"
                f"Note: ReportLab library not installed. Install with: pip install reportlab\n\n"
                f"Please use HTML or Markdown format for detailed reports.",
                encoding="utf-8"
            )
            return

        ensure_directory(path.parent)

        # Create PDF
        doc = SimpleDocTemplate(str(path), pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=1*inch, bottomMargin=0.75*inch)

        # Container for the 'Flowable' objects
        elements = []

        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3498db'),
            spaceAfter=12,
            spaceBefore=12,
        )

        # Title
        elements.append(Paragraph("🛡️ LLVM Obfuscation Report", title_style))
        elements.append(Paragraph(f"<b>Job ID:</b> {job_id}", styles['Normal']))
        elements.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))

        # Key Metrics
        elements.append(Paragraph("📊 Key Metrics", heading_style))
        metrics_data = [
            ['Metric', 'Value'],
            ['Obfuscation Score', f"{report.get('obfuscation_score', 0)}/100"],
            ['Symbol Reduction', f"{report.get('symbol_reduction', 0)}%"],
            ['Function Reduction', f"{report.get('function_reduction', 0)}%"],
            ['Estimated RE Effort', report.get('estimated_re_effort', 'N/A')],
        ]
        metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.2*inch))

        # Input Parameters
        elements.append(Paragraph("📥 Input Parameters", heading_style))
        input_params = report.get("input_parameters", {})
        input_data = [
            ['Parameter', 'Value'],
            ['Source File', str(input_params.get('source_file', 'N/A'))],
            ['Platform', str(input_params.get('platform', 'unknown'))],
            ['Obfuscation Level', f"Level {input_params.get('obfuscation_level', 0)}"],
            ['Applied Passes', ", ".join(input_params.get('applied_passes', [])) or "None"],
        ]
        input_table = Table(input_data, colWidths=[2.5*inch, 3.5*inch])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(input_table)
        elements.append(Spacer(1, 0.2*inch))

        # Output Attributes
        elements.append(Paragraph("📦 Output File Attributes", heading_style))
        output_attrs = report.get("output_attributes", {})
        file_size = output_attrs.get('file_size', 0)
        output_data = [
            ['Attribute', 'Value'],
            ['File Size', f"{file_size / 1024:.2f} KB ({file_size} bytes)"],
            ['Binary Format', str(output_attrs.get('binary_format', 'unknown'))],
            ['Symbol Count', str(output_attrs.get('symbols_count', 0))],
            ['Function Count', str(output_attrs.get('functions_count', 0))],
            ['Entropy', str(output_attrs.get('entropy', 0))],
        ]
        output_table = Table(output_data, colWidths=[2.5*inch, 3.5*inch])
        output_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(output_table)
        elements.append(Spacer(1, 0.2*inch))

        # Bogus Code
        elements.append(Paragraph("🔀 Bogus Code Generation", heading_style))
        bogus_code = report.get("bogus_code_info", {})
        bogus_data = [
            ['Type', 'Count'],
            ['Dead Code Blocks', str(bogus_code.get('dead_code_blocks', 0))],
            ['Opaque Predicates', str(bogus_code.get('opaque_predicates', 0))],
            ['Junk Instructions', str(bogus_code.get('junk_instructions', 0))],
            ['Code Bloat', f"{bogus_code.get('code_bloat_percentage', 0)}%"],
        ]
        bogus_table = Table(bogus_data, colWidths=[3*inch, 3*inch])
        bogus_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(bogus_table)
        elements.append(Spacer(1, 0.2*inch))

        # String Obfuscation
        elements.append(Paragraph("🔐 String Obfuscation", heading_style))
        string_obf = report.get("string_obfuscation", {})
        string_data = [
            ['Metric', 'Value'],
            ['Enabled', "Yes" if string_obf.get('enabled', False) else "No"],
            ['Encryption Method', string_obf.get('method', string_obf.get('encryption_method', 'none')).upper()],
            ['Encryption Rate', f"{string_obf.get('encryption_percentage', 0.0):.1f}%"],
        ]
        string_table = Table(string_data, colWidths=[3*inch, 3*inch])
        string_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(string_table)
        elements.append(Spacer(1, 0.2*inch))

        # Cycles
        cycles = report.get("cycles_completed", {})
        total_cycles = cycles.get("total_cycles", 1)
        elements.append(Paragraph(f"🔄 Obfuscation Cycles: {total_cycles}", heading_style))
        elements.append(Spacer(1, 0.1*inch))

        # Fake Loops
        fake_loops = report.get("fake_loops_inserted", {})
        fake_loop_count = fake_loops.get("count", 0)
        elements.append(Paragraph(f"➰ Fake Loops Inserted: {fake_loop_count}", heading_style))
        elements.append(Spacer(1, 0.1*inch))

        # Symbol Obfuscation
        symbol_obf = report.get("symbol_obfuscation", {})
        if symbol_obf.get("enabled", False):
            elements.append(Paragraph("🏷️ Symbol Obfuscation", heading_style))
            symbol_data = [
                ['Attribute', 'Value'],
                ['Symbols Renamed', str(symbol_obf.get('symbols_obfuscated', 0))],
                ['Algorithm', str(symbol_obf.get('algorithm', 'N/A'))],
            ]
            symbol_table = Table(symbol_data, colWidths=[3*inch, 3*inch])
            symbol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(symbol_table)
            elements.append(Spacer(1, 0.2*inch))

        # PAGE BREAK for advanced metrics
        elements.append(PageBreak())

        # ========== ADVANCED METRICS SECTION ==========
        elements.append(Paragraph("📊 Advanced Analysis Dashboard", heading_style))
        elements.append(Spacer(1, 0.1*inch))

        # Control Flow Metrics
        control_flow = report.get("control_flow_metrics")
        if control_flow:
            elements.append(Paragraph("🔀 Control Flow Analysis", heading_style))
            baseline_cf = control_flow.get("baseline", {})
            obf_cf = control_flow.get("obfuscated", {})
            comparison_cf = control_flow.get("comparison", {})

            cf_data = [
                ['Metric', 'Baseline', 'Obfuscated', 'Change'],
                ['Basic Blocks', str(baseline_cf.get('basic_blocks', 0)), str(obf_cf.get('basic_blocks', 0)),
                 f"+{comparison_cf.get('basic_blocks_added', 0)}"],
                ['CFG Edges', str(baseline_cf.get('cfg_edges', 0)), str(obf_cf.get('cfg_edges', 0)),
                 f"+{comparison_cf.get('cfg_edges_added', 0)}"],
                ['Cyclomatic Complexity', str(baseline_cf.get('cyclomatic_complexity', 0)),
                 str(obf_cf.get('cyclomatic_complexity', 0)),
                 f"+{comparison_cf.get('complexity_increase_percent', 0):.1f}%"],
                ['Functions', str(baseline_cf.get('functions', 0)), str(obf_cf.get('functions', 0)),
                 f"{obf_cf.get('functions', 0) - baseline_cf.get('functions', 0)}"],
                ['Loops', str(baseline_cf.get('loops', 0)), str(obf_cf.get('loops', 0)),
                 f"{obf_cf.get('loops', 0) - baseline_cf.get('loops', 0)}"],
            ]

            cf_table = Table(cf_data, colWidths=[1.8*inch, 1.5*inch, 1.5*inch, 1.2*inch])
            cf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e8f8f5')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f8f5')]),
            ]))
            elements.append(cf_table)
            elements.append(Spacer(1, 0.2*inch))

        # Instruction-Level Metrics
        instr_metrics = report.get("instruction_metrics")
        if instr_metrics:
            elements.append(Paragraph("💾 Instruction-Level Metrics", heading_style))
            baseline_instr = instr_metrics.get("baseline", {})
            obf_instr = instr_metrics.get("obfuscated", {})
            comparison_instr = instr_metrics.get("comparison", {})

            instr_data = [
                ['Metric', 'Baseline', 'Obfuscated', 'Change'],
                ['Total Instructions', str(baseline_instr.get('total_instructions', 0)),
                 str(obf_instr.get('total_instructions', 0)),
                 f"+{comparison_instr.get('instruction_growth_percent', 0):.1f}%"],
                ['Arithmetic Complexity', f"{baseline_instr.get('arithmetic_complexity_score', 0):.2f}",
                 f"{obf_instr.get('arithmetic_complexity_score', 0):.2f}",
                 f"+{comparison_instr.get('arithmetic_complexity_increase', 0):.1f}%"],
                ['Call Instructions', str(baseline_instr.get('call_instruction_count', 0)),
                 str(obf_instr.get('call_instruction_count', 0)),
                 f"{obf_instr.get('call_instruction_count', 0) - baseline_instr.get('call_instruction_count', 0)}"],
                ['Indirect Calls', str(baseline_instr.get('indirect_call_count', 0)),
                 str(obf_instr.get('indirect_call_count', 0)),
                 f"+{obf_instr.get('indirect_call_count', 0) - baseline_instr.get('indirect_call_count', 0)}"],
                ['MBA Expressions', str(baseline_instr.get('mba_expression_count', 0)),
                 str(obf_instr.get('mba_expression_count', 0)),
                 f"+{comparison_instr.get('mba_expressions_added', 0)}"],
            ]

            instr_table = Table(instr_data, colWidths=[1.8*inch, 1.5*inch, 1.5*inch, 1.2*inch])
            instr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ebf5fb')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ebf5fb')]),
            ]))
            elements.append(instr_table)
            elements.append(Spacer(1, 0.2*inch))

        # Binary Structure Metrics
        binary_struct = report.get("binary_structure")
        if binary_struct:
            elements.append(Paragraph("📦 Binary Structure Analysis", heading_style))
            binary_data = [
                ['Attribute', 'Value'],
                ['Section Count', str(binary_struct.get('section_count', 0))],
                ['Import Libraries', str(binary_struct.get('import_table', {}).get('library_count', 0))],
                ['Imported Symbols', str(binary_struct.get('import_table', {}).get('imported_symbols', 0))],
                ['Exported Symbols', str(binary_struct.get('export_table', {}).get('exported_symbols', 0))],
                ['Relocations', str(binary_struct.get('relocations', {}).get('relocation_count', 0))],
                ['Code-to-Data Ratio', f"{binary_struct.get('code_to_data_ratio', 0):.2f}"],
            ]

            binary_table = Table(binary_data, colWidths=[3*inch, 3*inch])
            binary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fadbd8')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fadbd8')]),
            ]))
            elements.append(binary_table)
            elements.append(Spacer(1, 0.2*inch))

        # Pattern Resistance Metrics
        pattern_resist = report.get("pattern_resistance")
        if pattern_resist:
            elements.append(Paragraph("🛡️ Pattern Resistance & RE Difficulty", heading_style))
            string_analysis = pattern_resist.get('string_analysis', {})
            code_analysis = pattern_resist.get('code_analysis', {})
            re_difficulty = pattern_resist.get('reverse_engineering_difficulty', {})

            pattern_data = [
                ['Metric', 'Value'],
                ['Visible Strings', str(string_analysis.get('visible_string_count', 0))],
                ['String Entropy', f"{string_analysis.get('string_entropy', 0):.2f}"],
                ['Opcode Distribution Entropy', f"{code_analysis.get('opcode_distribution_entropy', 0):.2f}"],
                ['Known Patterns Detected', str(code_analysis.get('known_pattern_count', 0))],
                ['Decompiler Confusion Score', f"{re_difficulty.get('decompiler_confusion_score', 0):.1f}/100"],
                ['CFG Obfuscation Rating', str(re_difficulty.get('cfg_obfuscation_rating', 'N/A')).upper()],
            ]

            pattern_table = Table(pattern_data, colWidths=[3*inch, 3*inch])
            pattern_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f4ecf7')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4ecf7')]),
            ]))
            elements.append(pattern_table)
            elements.append(Spacer(1, 0.2*inch))

        # Footer
        elements.append(Spacer(1, 0.5*inch))
        footer_text = Paragraph(
            "<para align=center><i>🤖 Generated with LLVM Obfuscator API</i></para>",
            styles['Normal']
        )
        elements.append(footer_text)

        # Build PDF
        doc.build(elements)

    def _format_control_flow_markdown(self, metrics: Dict) -> str:
        """Format control flow metrics for markdown."""
        if not metrics:
            return "⚠️ Control flow metrics not available"

        baseline = metrics.get("baseline", {})
        obfuscated = metrics.get("obfuscated", {})
        comparison = metrics.get("comparison", {})

        return f"""| Metric | Baseline | Obfuscated | Change |
|--------|----------|-----------|--------|
| Basic Blocks | {baseline.get('basic_blocks', 0)} | {obfuscated.get('basic_blocks', 0)} | +{comparison.get('basic_blocks_added', 0)} |
| CFG Edges | {baseline.get('cfg_edges', 0)} | {obfuscated.get('cfg_edges', 0)} | +{comparison.get('cfg_edges_added', 0)} |
| Cyclomatic Complexity | {baseline.get('cyclomatic_complexity', 0)} | {obfuscated.get('cyclomatic_complexity', 0)} | +{comparison.get('complexity_increase_percent', 0):.1f}% |
| Functions | {baseline.get('functions', 0)} | {obfuscated.get('functions', 0)} | {obfuscated.get('functions', 0) - baseline.get('functions', 0)} |
| Loops | {baseline.get('loops', 0)} | {obfuscated.get('loops', 0)} | {obfuscated.get('loops', 0) - baseline.get('loops', 0)} |"""

    def _format_instruction_metrics_markdown(self, metrics: Dict) -> str:
        """Format instruction-level metrics for markdown."""
        if not metrics:
            return "⚠️ Instruction metrics not available"

        baseline = metrics.get("baseline", {})
        obfuscated = metrics.get("obfuscated", {})
        comparison = metrics.get("comparison", {})

        return f"""| Metric | Baseline | Obfuscated | Change |
|--------|----------|-----------|--------|
| Total Instructions | {baseline.get('total_instructions', 0)} | {obfuscated.get('total_instructions', 0)} | +{comparison.get('instruction_growth_percent', 0):.1f}% |
| Arithmetic Complexity | - | - | +{comparison.get('arithmetic_complexity_increase', 0):.1f}% |
| MBA Expressions | - | - | +{comparison.get('mba_expressions_added', 0)} |"""

    def _format_binary_structure_markdown(self, metrics: Dict) -> str:
        """Format binary structure metrics for markdown."""
        if not metrics:
            return "⚠️ Binary structure metrics not available"

        return f"""- **Section Count:** {metrics.get('section_count', 0)}
- **Import Count:** {metrics.get('import_table', {}).get('library_count', 0)}
- **Export Count:** {metrics.get('export_table', {}).get('exported_symbols', 0)}
- **Relocations:** {metrics.get('relocations', {}).get('relocation_count', 0)}
- **Code-to-Data Ratio:** {metrics.get('code_to_data_ratio', 0):.2f}"""

    def _format_pattern_resistance_markdown(self, metrics: Dict) -> str:
        """Format pattern resistance metrics for markdown."""
        if not metrics:
            return "⚠️ Pattern resistance metrics not available"

        string_analysis = metrics.get('string_analysis', {})
        code_analysis = metrics.get('code_analysis', {})
        re_difficulty = metrics.get('reverse_engineering_difficulty', {})

        return f"""- **String Entropy:** {string_analysis.get('string_entropy', 0):.2f}
- **Opcode Distribution Entropy:** {code_analysis.get('opcode_distribution_entropy', 0):.2f}
- **Known Patterns Detected:** {code_analysis.get('known_pattern_count', 0)}
- **Decompiler Confusion Score:** {re_difficulty.get('decompiler_confusion_score', 0):.1f}/100
- **CFG Obfuscation Rating:** {re_difficulty.get('cfg_obfuscation_rating', 'UNKNOWN')}"""
