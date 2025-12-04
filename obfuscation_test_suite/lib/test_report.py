#!/usr/bin/env python3
"""Report generation for obfuscation test results"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate test reports in multiple formats"""

    def __init__(self, output_dir: Path, program_name: str):
        self.output_dir = Path(output_dir)
        self.program_name = program_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(self, results: Dict[str, Any]) -> Path:
        """Generate JSON format report"""
        json_file = self.output_dir / f"{self.program_name}_results.json"

        try:
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✓ JSON report: {json_file}")
            return json_file
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return None

    def generate_text_report(self, results: Dict[str, Any]) -> Path:
        """Generate human-readable text report"""
        txt_file = self.output_dir / f"{self.program_name}_report.txt"

        try:
            with open(txt_file, 'w') as f:
                f.write(self._format_text_report(results))
            logger.info(f"✓ Text report: {txt_file}")
            return txt_file
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
            return None

    def generate_summary(self, results: Dict[str, Any]) -> Path:
        """Generate quick summary report"""
        summary_file = self.output_dir / f"{self.program_name}_summary.txt"

        try:
            with open(summary_file, 'w') as f:
                f.write(self._format_summary(results))
            logger.info(f"✓ Summary report: {summary_file}")
            return summary_file
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

    def _format_text_report(self, results: Dict[str, Any]) -> str:
        """Format comprehensive text report"""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("OLLVM OBFUSCATION TEST SUITE - COMPREHENSIVE REPORT")
        lines.append("=" * 70)
        lines.append("")

        # ✅ FIX #5: Add metrics reliability warning at the top
        reliability = results.get('reliability_status', {})
        if reliability.get('level') == 'FAILED':
            lines.append("⚠️  ⚠️  ⚠️  CRITICAL WARNING ⚠️  ⚠️  ⚠️")
            lines.append("-" * 70)
            lines.append(reliability.get('warning', 'Metrics reliability unknown'))
            lines.append("This report should NOT be used for obfuscation comparison purposes.")
            lines.append("-" * 70)
            lines.append("")
        elif reliability.get('level') == 'UNCERTAIN':
            lines.append("⚠️  WARNING ⚠️")
            lines.append("-" * 70)
            lines.append(reliability.get('warning', 'Metrics reliability unknown'))
            lines.append("Metrics should be treated with caution.")
            lines.append("-" * 70)
            lines.append("")
        elif reliability.get('level') == 'RELIABLE':
            lines.append("✅ METRICS RELIABILITY")
            lines.append("-" * 70)
            lines.append(reliability.get('warning', 'Metrics are reliable'))
            lines.append("-" * 70)
            lines.append("")

        # Metadata
        meta = results.get('metadata', {})
        lines.append("METADATA")
        lines.append("-" * 70)
        lines.append(f"Timestamp:     {meta.get('timestamp', 'N/A')}")
        lines.append(f"Program:       {meta.get('program', 'N/A')}")
        lines.append(f"Baseline:      {meta.get('baseline', 'N/A')}")
        lines.append(f"Obfuscated:    {meta.get('obfuscated', 'N/A')}")
        lines.append(f"Metrics Reliability: {meta.get('metrics_reliability', 'UNKNOWN')}")
        lines.append(f"Functional Test: {meta.get('functional_correctness_passed', 'UNKNOWN')}")
        lines.append("")

        # Test Results
        results_data = results.get('test_results', {})

        if 'functional' in results_data:
            lines.append("FUNCTIONAL CORRECTNESS")
            lines.append("-" * 70)
            func = results_data['functional']
            lines.append(f"Same Behavior:  {func.get('same_behavior', 'N/A')}")
            lines.append(f"Tests Passed:   {func.get('passed', 0)}/{func.get('test_count', 0)}")
            lines.append("")

        if 'strings' in results_data:
            lines.append("STRING OBFUSCATION")
            lines.append("-" * 70)
            strings = results_data['strings']
            lines.append(f"Baseline Strings:    {strings.get('baseline_strings', 0)}")
            lines.append(f"Obfuscated Strings:  {strings.get('obf_strings', 0)}")
            lines.append(f"Reduction:           {strings.get('reduction_percent', 0):.1f}%")
            lines.append("")

        if 'binary_properties' in results_data:
            lines.append("BINARY PROPERTIES")
            lines.append("-" * 70)
            props = results_data['binary_properties']
            baseline_size = props.get('baseline_size_bytes', 0)
            obf_size = props.get('obf_size_bytes', 0)
            lines.append(f"Baseline Size:       {baseline_size:,} bytes")
            lines.append(f"Obfuscated Size:     {obf_size:,} bytes")
            lines.append(f"Size Increase:       {props.get('size_increase_percent', 0):+.1f}%")
            lines.append(f"Baseline Entropy:    {props.get('baseline_entropy', 0):.4f}")
            lines.append(f"Obfuscated Entropy:  {props.get('obf_entropy', 0):.4f}")
            lines.append(f"Entropy Increase:    {props.get('entropy_increase', 0):+.4f}")
            lines.append("")

        if 'symbols' in results_data:
            lines.append("SYMBOL ANALYSIS")
            lines.append("-" * 70)
            syms = results_data['symbols']
            lines.append(f"Baseline Symbols:    {syms.get('baseline_symbol_count', 0)}")
            lines.append(f"Obfuscated Symbols:  {syms.get('obf_symbol_count', 0)}")
            lines.append(f"Symbols Reduced:     {syms.get('symbols_reduced', False)}")
            lines.append("")

        if 'performance' in results_data:
            lines.append("PERFORMANCE ANALYSIS")
            lines.append("-" * 70)
            perf = results_data['performance']
            # ✅ FIX #4: Handle SKIPPED and None values in performance
            status = perf.get('status', 'UNKNOWN')
            if status == 'SKIPPED':
                lines.append(f"Status:              SKIPPED")
                lines.append(f"Reason:              {perf.get('reason', 'N/A')}")
            elif status in ['FAILED', 'TIMEOUT']:
                lines.append(f"Status:              {status}")
                lines.append(f"Reason:              {perf.get('reason', 'N/A')}")
                baseline_ms = perf.get('baseline_ms')
                obf_ms = perf.get('obf_ms')
                if baseline_ms is not None:
                    lines.append(f"Baseline Time:       {baseline_ms:.2f} ms ({status.lower()})")
                if obf_ms is not None:
                    lines.append(f"Obfuscated Time:     {obf_ms:.2f} ms ({status.lower()})")
            else:
                baseline_ms = perf.get('baseline_ms', 0)
                obf_ms = perf.get('obf_ms', 0)
                overhead = perf.get('overhead_percent', 0)
                if baseline_ms is not None and obf_ms is not None:
                    lines.append(f"Baseline Time:       {baseline_ms:.2f} ms")
                    lines.append(f"Obfuscated Time:     {obf_ms:.2f} ms")
                    lines.append(f"Overhead:            {overhead:+.1f}%")
                    lines.append(f"Acceptable:          {perf.get('acceptable', False)}")
            lines.append("")

        if 'debuggability' in results_data:
            lines.append("DEBUGGABILITY")
            lines.append("-" * 70)
            dbg = results_data['debuggability']
            lines.append(f"Baseline Debug Info: {dbg.get('baseline_has_debug_info', False)}")
            lines.append(f"Obfuscated Debug:    {dbg.get('obf_has_debug_info', False)}")
            lines.append(f"Difficulty:          {dbg.get('difficulty_estimate', 'N/A')}")
            lines.append("")

        if 're_difficulty' in results_data:
            lines.append("REVERSE ENGINEERING DIFFICULTY")
            lines.append("-" * 70)
            re = results_data['re_difficulty']
            lines.append(f"Score:               {re.get('re_difficulty_score', 0)}/100")
            lines.append(f"Difficulty Level:    {re.get('re_difficulty_level', 'N/A')}")
            lines.append("")

        # Advanced Analysis
        if 'advanced_analysis' in results_data:
            lines.append("ADVANCED ANALYSIS")
            lines.append("-" * 70)
            adv = results_data['advanced_analysis']

            if 'ghidra' in adv and adv['ghidra'].get('status') != 'not_installed':
                lines.append("Ghidra Decompilation:")
                ghidra = adv['ghidra']
                if 'decompilation' in ghidra:
                    decomp = ghidra['decompilation']
                    lines.append(f"  Status: {decomp.get('status', 'N/A')}")
                    lines.append(f"  Functions: {decomp.get('function_count', 0)}")

            if 'binja' in adv and adv['binja'].get('status') != 'not_installed':
                lines.append("Binary Ninja Analysis:")
                binja = adv['binja']
                if 'hlil' in binja:
                    hlil = binja['hlil']
                    lines.append(f"  Status: {hlil.get('status', 'N/A')}")
                    lines.append(f"  Functions: {hlil.get('function_count', 0)}")
                    lines.append(f"  Avg HLIL Lines: {hlil.get('avg_hlil_lines', 0):.1f}")
                    lines.append(f"  CFG Complexity: {hlil.get('total_cfg_complexity', 0)}")

            if 'ida' in adv and adv['ida'].get('status') != 'not_installed':
                lines.append("IDA Pro Analysis:")
                ida = adv['ida']
                if 'analysis' in ida:
                    analysis = ida['analysis']
                    lines.append(f"  Status: {analysis.get('status', 'N/A')}")
                    lines.append(f"  Functions: {analysis.get('function_count', 0)}")
                    lines.append(f"  Strings: {analysis.get('string_count', 0)}")
                    lines.append(f"  Imports: {analysis.get('import_count', 0)}")
                    lines.append(f"  CFG Complexity: {analysis.get('cfg_complexity', 0)}")
                    lines.append(f"  Hex-Rays Available: {analysis.get('decompilation_available', False)}")
                if 'decompilation_comparison' in ida:
                    decomp = ida['decompilation_comparison']
                    if decomp.get('status') == 'success':
                        lines.append(f"  Baseline Code Lines: {decomp.get('baseline_lines', 0)}")
                        lines.append(f"  Obfuscated Code Lines: {decomp.get('obfuscated_lines', 0)}")
                        lines.append(f"  Code Expansion: {decomp.get('code_expansion_ratio', 0):.2f}x")

            if 'angr' in adv and adv['angr'].get('status') != 'not_installed':
                lines.append("Angr Symbolic Execution:")
                angr = adv['angr']
                if 'symbolic_execution' in angr:
                    symexec = angr['symbolic_execution']
                    lines.append(f"  Entry Point: {symexec.get('entry_point', 'N/A')}")
                    lines.append(f"  Architecture: {symexec.get('arch', 'N/A')}")
                    lines.append(f"  Reachable Paths: {symexec.get('reachable_paths', 0)}")
                    lines.append(f"  Functions Discovered: {symexec.get('discovered_functions', 0)}")

                if 'vulnerabilities' in angr:
                    vulns = angr['vulnerabilities']
                    lines.append("Vulnerability Analysis:")
                    vuln_patterns = vulns.get('vulnerable_patterns', {})
                    lines.append(f"  Risk Level: {vulns.get('overall_risk', 'N/A')}")
                    lines.append(f"  Format Strings: {vuln_patterns.get('format_string_vulns', 0)}")
                    lines.append(f"  Buffer Overflows: {vuln_patterns.get('buffer_overflow_patterns', 0)}")
            lines.append("")

        # String Obfuscation Analysis
        if 'string_obfuscation_advanced' in results_data:
            lines.append("STRING OBFUSCATION ANALYSIS")
            lines.append("-" * 70)
            str_adv = results_data['string_obfuscation_advanced']
            lines.append(f"Detection Confidence: {str_adv.get('detection_confidence', 0):.1f}%")

            techniques = str_adv.get('obfuscation_techniques', {})
            lines.append("Detected Techniques:")
            for tech, detected in techniques.items():
                status = "✓" if detected else "✗"
                lines.append(f"  {status} {tech.replace('_', ' ').title()}")
            lines.append("")

        # Debuggability Analysis
        if 'debuggability_advanced' in results_data:
            lines.append("DEBUGGER RESISTANCE ANALYSIS")
            lines.append("-" * 70)
            dbg_adv = results_data['debuggability_advanced']
            if 'baseline' in dbg_adv:
                baseline_dbg = dbg_adv['baseline']
                lines.append(f"Baseline Debuggability: {baseline_dbg.get('debuggability_score', 0):.1f}/100")
            if 'obfuscated' in dbg_adv:
                obf_dbg = dbg_adv['obfuscated']
                lines.append(f"Obfuscated Debuggability: {obf_dbg.get('debuggability_score', 0):.1f}/100")
            lines.append("")

        # Code Coverage Analysis
        if 'code_coverage' in results_data:
            lines.append("CODE COVERAGE ANALYSIS")
            lines.append("-" * 70)
            cov = results_data['code_coverage']
            if 'baseline' in cov:
                baseline_cov = cov['baseline']
                lines.append(f"Baseline Coverage: {baseline_cov.get('estimated_coverage', 0):.1f}%")
            if 'obfuscated' in cov:
                obf_cov = cov['obfuscated']
                lines.append(f"Obfuscated Coverage: {obf_cov.get('estimated_coverage', 0):.1f}%")
            lines.append("")

        # Patchability Analysis
        if 'patchability' in results_data:
            lines.append("PATCHABILITY ASSESSMENT")
            lines.append("-" * 70)
            patch = results_data['patchability']
            if 'baseline' in patch:
                baseline_patch = patch['baseline']
                lines.append(f"Baseline Patch Difficulty: {baseline_patch.get('patch_difficulty', 'N/A')}")
                lines.append(f"Position Independent: {baseline_patch.get('position_independent', False)}")
            if 'obfuscated' in patch:
                obf_patch = patch['obfuscated']
                lines.append(f"Obfuscated Patch Difficulty: {obf_patch.get('patch_difficulty', 'N/A')}")
                lines.append(f"Relocation Entries: {obf_patch.get('relocation_entries', 0)}")
            lines.append("")

        # Metrics
        metrics = results.get('metrics', {})

        if 'complexity' in metrics:
            lines.append("COMPLEXITY METRICS")
            lines.append("-" * 70)
            comp = metrics['complexity']
            base = comp.get('baseline', {})
            obf = comp.get('obfuscated', {})
            lines.append(f"Baseline Size:      {base.get('size_bytes', 0):,} bytes")
            lines.append(f"Obfuscated Size:    {obf.get('size_bytes', 0):,} bytes")
            lines.append(f"Baseline Symbols:   {base.get('symbol_count', 0)}")
            lines.append(f"Obfuscated Symbols: {obf.get('symbol_count', 0)}")
            changes = comp.get('changes', {})
            lines.append(f"Symbol Reduction:   {changes.get('symbol_reduction_percent', 0):.1f}%")
            lines.append("")

        # Risk Assessment and Analysis Summary
        lines.append("RISK ASSESSMENT & SECURITY ANALYSIS")
        lines.append("-" * 70)

        risk_score = 0
        findings = []

        # Evaluate various risk factors
        if 'functional' in results_data:
            func = results_data['functional']
            if not func.get('same_behavior'):
                risk_score += 30
                findings.append("✗ Critical: Functional correctness broken - obfuscation altered behavior")

        if 'debuggability' in results_data:
            dbg = results_data['debuggability']
            if not dbg.get('baseline_has_debug_info') and dbg.get('obf_has_debug_info'):
                findings.append("○ Debug info stripped - reverse engineering difficulty increased")

        if 'strings' in results_data:
            strings = results_data['strings']
            reduction = strings.get('reduction_percent', 0)
            if reduction > 80:
                findings.append(f"✓ Strings obfuscated ({reduction:.0f}% reduction) - reduces information leakage")
            else:
                findings.append(f"△ Limited string obfuscation ({reduction:.0f}% reduction)")

        if 'symbols' in results_data:
            syms = results_data['symbols']
            if syms.get('symbols_reduced'):
                findings.append("✓ Symbols stripped - binary hardening applied")

        if 'performance' in results_data:
            perf = results_data['performance']
            overhead = perf.get('overhead_percent', 0)
            if overhead > 50:
                findings.append(f"△ Performance overhead {overhead:.1f}% - significant slowdown detected")
                risk_score += 10
            elif overhead > 0:
                findings.append(f"○ Performance overhead {overhead:.1f}% - acceptable")

        if 're_difficulty' in results_data:
            re = results_data['re_difficulty']
            score = re.get('re_difficulty_score', 0)
            if score < 30:
                findings.append(f"✗ Low RE difficulty ({score}/100) - weak obfuscation")
                risk_score += 20
            elif score < 70:
                findings.append(f"○ Medium RE difficulty ({score}/100) - moderate protection")
            else:
                findings.append(f"✓ High RE difficulty ({score}/100) - strong protection")

        if 'binary_properties' in results_data:
            props = results_data['binary_properties']
            entropy_increase = props.get('entropy_increase', 0)
            if entropy_increase > 0:
                findings.append(f"✓ Entropy increased by {entropy_increase:.4f} - randomization detected")

        # Print findings
        for finding in findings:
            lines.append(f"  {finding}")

        lines.append(f"\nOverall Risk Score: {min(100, max(0, 100 - risk_score))}/100")
        risk_level = "LOW" if min(100, max(0, 100 - risk_score)) > 70 else "MEDIUM" if min(100, max(0, 100 - risk_score)) > 40 else "HIGH"
        lines.append(f"Risk Level: {risk_level}")
        lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        recommendations = []

        if 'strings' in results_data and results_data['strings'].get('reduction_percent', 0) < 50:
            recommendations.append("• Increase string obfuscation to protect sensitive data")

        if 'symbols' in results_data and not results_data['symbols'].get('symbols_reduced'):
            recommendations.append("• Strip debug symbols and exported functions")

        if 'performance' in results_data and results_data['performance'].get('overhead_percent', 0) > 50:
            recommendations.append("• Reduce obfuscation intensity - performance impact too high")

        if 're_difficulty' in results_data and results_data['re_difficulty'].get('re_difficulty_score', 0) < 50:
            recommendations.append("• Apply stronger obfuscation techniques (control flow flattening, virtualization)")

        if not recommendations:
            recommendations.append("• Binary obfuscation parameters are well-balanced")

        for rec in recommendations:
            lines.append(f"  {rec}")
        lines.append("")

        # Detailed Metrics Comparison
        lines.append("DETAILED METRICS COMPARISON")
        lines.append("-" * 70)

        if 'cfg_metrics' in metrics:
            cfg = metrics['cfg_metrics']
            baseline_cfg = cfg.get('baseline', {})
            obf_cfg = cfg.get('obfuscated', {})
            comparison = cfg.get('comparison', {})

            lines.append("Control Flow Graph Metrics:")
            lines.append(f"  Baseline Indirect Jumps:    {baseline_cfg.get('indirect_jumps', 0)}")
            lines.append(f"  Obfuscated Indirect Jumps:  {obf_cfg.get('indirect_jumps', 0)}")
            lines.append(f"  Baseline Branch Instructions: {baseline_cfg.get('branch_instructions', 0)}")
            lines.append(f"  Obfuscated Branch Instructions: {obf_cfg.get('branch_instructions', 0)}")
            lines.append(f"  Control Flow Complexity Change: {comparison.get('control_flow_complexity_increase', 0):.2f}x")
            lines.append("")

        if 'coverage' in metrics:
            cov = metrics['coverage']
            lines.append("Code Coverage Metrics:")
            lines.append(f"  Baseline Coverage Analysis: {cov.get('baseline', {}).get('estimated_coverage', 'N/A')}")
            lines.append(f"  Obfuscated Coverage Analysis: {cov.get('obfuscated', {}).get('estimated_coverage', 'N/A')}")
            lines.append(f"  Note: {cov.get('analysis_note', '')}")
            lines.append("")

        # Obfuscation Effectiveness Analysis
        if 'string_obfuscation_advanced' in results_data:
            lines.append("OBFUSCATION EFFECTIVENESS")
            lines.append("-" * 70)
            str_adv = results_data['string_obfuscation_advanced']

            detected_techs = []
            techniques = str_adv.get('obfuscation_techniques', {})
            for tech, detected in techniques.items():
                if detected:
                    detected_techs.append(tech.replace('_', ' ').title())

            if detected_techs:
                lines.append(f"Detected Techniques ({len(detected_techs)}):")
                for tech in detected_techs:
                    lines.append(f"  • {tech}")
            else:
                lines.append("No string obfuscation techniques detected")

            lines.append(f"Detection Confidence: {str_adv.get('detection_confidence', 0):.1f}%")
            lines.append(f"String Reduction Ratio: {str_adv.get('reduction_percentage', 0):.1f}%")
            lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append("REPORT GENERATED BY OLLVM OBFUSCATION TEST SUITE")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Format quick summary"""
        lines = []

        lines.append("OBFUSCATION TEST SUMMARY")
        lines.append("=" * 50)
        lines.append("")

        meta = results.get('metadata', {})
        lines.append(f"Program: {meta.get('program', 'N/A')}")
        lines.append(f"Date:    {meta.get('timestamp', 'N/A')}")
        lines.append("")

        # Key metrics
        test_res = results.get('test_results', {})
        metrics = results.get('metrics', {})

        if 'functional' in test_res and test_res['functional'].get('same_behavior'):
            lines.append("✓ Functional correctness maintained")
        else:
            lines.append("✗ Functional correctness issue")

        if 're_difficulty' in test_res:
            re_info = test_res['re_difficulty']
            lines.append(f"✓ RE Difficulty: {re_info.get('re_difficulty_level', 'N/A')} ({re_info.get('re_difficulty_score', 0)}/100)")

        if 'strings' in test_res:
            str_reduction = test_res['strings'].get('reduction_percent', 0)
            if str_reduction > 0:
                lines.append(f"✓ String reduction: {str_reduction:.1f}%")

        if 'performance' in test_res:
            perf = test_res['performance'].get('overhead_percent', 0)
            if perf < 100:
                lines.append(f"✓ Performance overhead: {perf:.1f}% (acceptable)")
            else:
                lines.append(f"✗ Performance overhead: {perf:.1f}% (high)")

        lines.append("")
        lines.append("For detailed results, see the full report.")

        return "\n".join(lines)
