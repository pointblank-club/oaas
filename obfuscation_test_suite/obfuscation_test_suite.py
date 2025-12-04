#!/usr/bin/env python3
"""
OLLVM/LLVM Obfuscation Test Suite - Industry Standard Evaluation
Comprehensive evaluation of obfuscation effectiveness across multiple dimensions
"""

import os
import sys
import json
import subprocess
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
sys.path.insert(0, str(Path(__file__).parent / 'lib'))

from test_utils import run_command, safe_run, file_hash, extract_strings
from test_metrics import compute_cfg_metrics, analyze_complexity, compute_coverage
from test_report import ReportGenerator
from test_functional import FunctionalTester
from advanced_analysis import (
    GhidraAnalyzer, BinaryNinjaAnalyzer, AngrAnalyzer,
    StringObfuscationAnalyzer, DebuggabilityAnalyzer,
    CodeCoverageAnalyzer, PatchabilityAnalyzer, IDAProAnalyzer
)


class ObfuscationTestSuite:
    """Main orchestrator for complete obfuscation testing"""

    def __init__(self, baseline_path: str, obf_path: str, results_dir: str, program_name: str = "program"):
        self.baseline = Path(baseline_path)
        self.obfuscated = Path(obf_path)
        self.results_dir = Path(results_dir)
        self.program_name = program_name
        self.test_results = {}
        self.metrics = {}

        # Create results subdirectories
        self.baseline_dir = self.results_dir / "baseline" / program_name
        self.obf_dir = self.results_dir / "obfuscated" / program_name
        self.reports_dir = self.results_dir / "reports" / program_name
        self.metrics_dir = self.results_dir / "metrics" / program_name

        for d in [self.baseline_dir, self.obf_dir, self.reports_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.report_gen = ReportGenerator(self.reports_dir, program_name)
        self.functional_tester = FunctionalTester(self.baseline, self.obfuscated)

        logger.info(f"Initialized test suite for '{program_name}'")
        logger.info(f"Baseline: {self.baseline}")
        logger.info(f"Obfuscated: {self.obfuscated}")
        logger.info(f"Results: {self.results_dir}")

    def run_all_tests(self) -> bool:
        """Run all tests and generate comprehensive report"""
        logger.info("=" * 60)
        logger.info("Starting OLLVM Obfuscation Test Suite")
        logger.info("=" * 60)

        try:
            # 1. Verify binaries exist
            if not self._verify_binaries():
                return False

            # 2. Copy binaries to results directories
            self._copy_binaries()

            # 3. Functional correctness tests
            logger.info("\n[1/11] Running functional correctness tests...")
            self.test_results['functional'] = self._test_functional_correctness()

            # ✅ FIX #1: Check functional correctness and flag if failed
            functional_passed = self.test_results['functional'].get('same_behavior', False)
            if not functional_passed:
                logger.warning("⚠️  FUNCTIONAL CORRECTNESS FAILED - Subsequent metrics may be unreliable")
                logger.warning("    Binary behavior differs between baseline and obfuscated versions")
                logger.warning("    Performance and execution-dependent metrics should be treated with caution")
                # Mark all subsequent metrics as potentially unreliable
                self._metrics_reliability = "COMPROMISED"
            else:
                self._metrics_reliability = "RELIABLE"

            # 4. Control flow metrics
            logger.info("\n[2/11] Computing control flow metrics...")
            self.metrics['cfg_metrics'] = compute_cfg_metrics(
                str(self.baseline), str(self.obfuscated)
            )

            # 5. Complexity analysis
            logger.info("\n[3/11] Analyzing binary complexity...")
            self.metrics['complexity'] = analyze_complexity(
                str(self.baseline), str(self.obfuscated)
            )

            # 6. String analysis
            logger.info("\n[4/11] Analyzing string obfuscation...")
            self.test_results['strings'] = self._test_string_analysis()

            # 7. Size and entropy analysis
            logger.info("\n[5/11] Analyzing binary properties...")
            self.test_results['binary_properties'] = self._test_binary_properties()

            # 8. Symbol analysis
            logger.info("\n[6/11] Analyzing symbols...")
            self.test_results['symbols'] = self._test_symbol_analysis()

            # 9. Coverage analysis
            logger.info("\n[7/11] Computing code coverage...")
            self.metrics['coverage'] = compute_coverage(
                str(self.baseline), str(self.obfuscated)
            )

            # 10. Performance overhead
            logger.info("\n[8/11] Measuring performance overhead...")
            self.test_results['performance'] = self._test_performance()

            # 11. Debuggability analysis
            logger.info("\n[9/11] Analyzing debuggability...")
            self.test_results['debuggability'] = self._test_debuggability()

            # 12. Reverse engineering difficulty
            logger.info("\n[10/11] Estimating RE difficulty...")
            self.test_results['re_difficulty'] = self._estimate_re_difficulty()

            # 13. Advanced analysis with Ghidra, Binary Ninja, Angr
            logger.info("\n[11/15] Running advanced analysis...")
            self.test_results['advanced_analysis'] = self._run_advanced_analysis()

            # 14. String obfuscation analysis
            logger.info("\n[12/15] Analyzing string obfuscation techniques...")
            self.test_results['string_obfuscation_advanced'] = self._advanced_string_analysis()

            # 15. Debuggability and anti-debug testing
            logger.info("\n[13/15] Testing debugger resistance...")
            self.test_results['debuggability_advanced'] = self._advanced_debuggability_analysis()

            # 16. Code coverage analysis
            logger.info("\n[14/15] Analyzing code coverage...")
            self.test_results['code_coverage'] = self._analyze_code_coverage()

            # 17. Patchability assessment
            logger.info("\n[15/15] Assessing patchability...")
            self.test_results['patchability'] = self._assess_patchability()

            # 18. Generate reports
            logger.info("\nGenerating comprehensive reports...")
            self._generate_all_reports()

            logger.info("\n" + "=" * 60)
            logger.info("Test Suite Completed Successfully!")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _verify_binaries(self) -> bool:
        """Verify that both binaries exist and are executable"""
        logger.info("Verifying binaries...")

        if not self.baseline.exists():
            logger.error(f"Baseline binary not found: {self.baseline}")
            return False

        if not self.obfuscated.exists():
            logger.error(f"Obfuscated binary not found: {self.obfuscated}")
            return False

        if not os.access(self.baseline, os.X_OK):
            logger.warning(f"Baseline is not executable, attempting to fix...")
            os.chmod(self.baseline, 0o755)

        if not os.access(self.obfuscated, os.X_OK):
            logger.warning(f"Obfuscated is not executable, attempting to fix...")
            os.chmod(self.obfuscated, 0o755)

        logger.info("✓ Binaries verified")
        return True

    def _copy_binaries(self):
        """Copy binaries to results directories"""
        logger.info("Copying binaries to results directories...")
        import shutil

        baseline_copy = self.baseline_dir / self.baseline.name
        obf_copy = self.obf_dir / self.obfuscated.name

        shutil.copy2(self.baseline, baseline_copy)
        shutil.copy2(self.obfuscated, obf_copy)

        logger.info(f"✓ Copied to {baseline_copy}")
        logger.info(f"✓ Copied to {obf_copy}")

    def _test_functional_correctness(self) -> Dict[str, Any]:
        """Test that obfuscated binary maintains functional correctness"""
        logger.info("  Testing functional correctness...")

        results = {
            'test_count': 0,
            'passed': 0,
            'failed': 0,
            'same_behavior': True,
            'details': []
        }

        try:
            # Run basic functional tests
            results['same_behavior'] = self.functional_tester.test_basic_io()
            if results['same_behavior']:
                logger.info("  ✓ Basic I/O behavior matches")
                results['passed'] += 1
            else:
                logger.warning("  ✗ I/O behavior differs")
                results['failed'] += 1

            results['test_count'] = 1

        except Exception as e:
            logger.warning(f"  Functional testing error: {e}")
            results['same_behavior'] = None

        return results

    def _test_string_analysis(self) -> Dict[str, Any]:
        """Analyze string obfuscation effectiveness"""
        logger.info("  Analyzing strings...")

        baseline_strings = extract_strings(str(self.baseline))
        obf_strings = extract_strings(str(self.obfuscated))

        reduction = 0
        if len(baseline_strings) > 0:
            reduction = 100 * (len(baseline_strings) - len(obf_strings)) / len(baseline_strings)

        results = {
            'baseline_strings': len(baseline_strings),
            'obf_strings': len(obf_strings),
            'reduction_percent': round(reduction, 2),
            'sample_baseline': baseline_strings[:5],
            'sample_obf': obf_strings[:5]
        }

        logger.info(f"  ✓ Baseline strings: {len(baseline_strings)}")
        logger.info(f"  ✓ Obfuscated strings: {len(obf_strings)} ({reduction:.1f}% reduction)")

        return results

    def _test_binary_properties(self) -> Dict[str, Any]:
        """Analyze binary properties (size, entropy, etc.)"""
        logger.info("  Analyzing binary properties...")

        baseline_size = self.baseline.stat().st_size
        obf_size = self.obfuscated.stat().st_size

        # Calculate entropy
        baseline_entropy = self._calculate_entropy(self.baseline)
        obf_entropy = self._calculate_entropy(self.obfuscated)

        size_increase = 100 * (obf_size - baseline_size) / baseline_size if baseline_size > 0 else 0

        results = {
            'baseline_size_bytes': baseline_size,
            'obf_size_bytes': obf_size,
            'size_increase_percent': round(size_increase, 2),
            'baseline_entropy': round(baseline_entropy, 4),
            'obf_entropy': round(obf_entropy, 4),
            'entropy_increase': round(obf_entropy - baseline_entropy, 4)
        }

        logger.info(f"  ✓ Baseline size: {baseline_size} bytes")
        logger.info(f"  ✓ Obfuscated size: {obf_size} bytes ({size_increase:+.1f}%)")
        logger.info(f"  ✓ Entropy: {baseline_entropy:.4f} → {obf_entropy:.4f}")

        return results

    def _test_symbol_analysis(self) -> Dict[str, Any]:
        """Analyze symbol obfuscation"""
        logger.info("  Analyzing symbols...")

        baseline_symbols = self._extract_symbols(str(self.baseline))
        obf_symbols = self._extract_symbols(str(self.obfuscated))

        results = {
            'baseline_symbol_count': len(baseline_symbols),
            'obf_symbol_count': len(obf_symbols),
            'symbols_reduced': len(baseline_symbols) > len(obf_symbols),
            'sample_baseline_symbols': baseline_symbols[:5],
            'sample_obf_symbols': obf_symbols[:5]
        }

        logger.info(f"  ✓ Baseline symbols: {len(baseline_symbols)}")
        logger.info(f"  ✓ Obfuscated symbols: {len(obf_symbols)}")

        return results

    def _test_performance(self) -> Dict[str, Any]:
        """Measure performance overhead of obfuscation

        ✅ FIX #3: Validate measurements before returning
        Returns error status if binaries didn't execute properly
        """
        logger.info("  Measuring performance...")

        baseline_time = self._measure_execution_time(self.baseline)
        obf_time = self._measure_execution_time(self.obfuscated)

        # ✅ FIX #3d: Check for error codes from measurement
        if baseline_time < 0:
            logger.warning("  ✗ Baseline binary execution failed/timed out - skipping overhead calculation")
            return {
                'baseline_ms': baseline_time,
                'obf_ms': obf_time,
                'overhead_percent': None,
                'acceptable': None,
                'status': 'FAILED' if baseline_time == -1.0 else 'TIMEOUT',
                'reason': 'Baseline binary could not execute properly'
            }

        if obf_time < 0:
            logger.warning("  ✗ Obfuscated binary execution failed/timed out - skipping overhead calculation")
            return {
                'baseline_ms': baseline_time,
                'obf_ms': obf_time,
                'overhead_percent': None,
                'acceptable': None,
                'status': 'FAILED' if obf_time == -1.0 else 'TIMEOUT',
                'reason': 'Obfuscated binary could not execute properly'
            }

        # Both binaries executed successfully - calculate overhead
        overhead = 100 * (obf_time - baseline_time) / baseline_time if baseline_time > 0 else 0

        results = {
            'baseline_ms': round(baseline_time, 2),
            'obf_ms': round(obf_time, 2),
            'overhead_percent': round(overhead, 2),
            'acceptable': overhead < 100,
            'status': 'SUCCESS'
        }

        logger.info(f"  ✓ Baseline: {baseline_time:.2f}ms")
        logger.info(f"  ✓ Obfuscated: {obf_time:.2f}ms ({overhead:+.1f}%)")

        return results

    def _test_debuggability(self) -> Dict[str, Any]:
        """Analyze debuggability impact"""
        logger.info("  Analyzing debuggability...")

        # Simple heuristics for debuggability
        baseline_has_debug = self._has_debug_info(str(self.baseline))
        obf_has_debug = self._has_debug_info(str(self.obfuscated))

        results = {
            'baseline_has_debug_info': baseline_has_debug,
            'obf_has_debug_info': obf_has_debug,
            'debug_info_preserved': baseline_has_debug == obf_has_debug,
            'difficulty_estimate': 'HIGH' if not obf_has_debug else 'MEDIUM'
        }

        logger.info(f"  ✓ Baseline debug info: {baseline_has_debug}")
        logger.info(f"  ✓ Obfuscated debug info: {obf_has_debug}")

        return results

    def _estimate_re_difficulty(self) -> Dict[str, Any]:
        """Estimate reverse engineering difficulty"""
        logger.info("  Estimating RE difficulty...")

        # Composite score based on all metrics
        score = 0

        if self.metrics.get('complexity', {}).get('cyclomatic_complexity_obf', 0) > \
           self.metrics.get('complexity', {}).get('cyclomatic_complexity_base', 1) * 1.5:
            score += 20

        if self.test_results.get('strings', {}).get('reduction_percent', 0) > 20:
            score += 15

        if self.test_results.get('binary_properties', {}).get('entropy_increase', 0) > 0:
            score += 15

        if self.test_results.get('symbols', {}).get('symbols_reduced', False):
            score += 15

        if self.test_results.get('performance', {}).get('overhead_percent', 0) < 100:
            score += 20

        # Normalize to 0-100
        score = min(100, max(0, score))

        difficulty_levels = {
            (0, 25): 'LOW',
            (25, 50): 'MEDIUM',
            (50, 75): 'HIGH',
            (75, 100): 'VERY HIGH'
        }

        difficulty = 'MEDIUM'
        for range_tuple, level in difficulty_levels.items():
            if range_tuple[0] <= score <= range_tuple[1]:
                difficulty = level
                break

        results = {
            're_difficulty_score': score,
            're_difficulty_level': difficulty,
            'factors': {
                'complexity_increase': 20,
                'string_obfuscation': 15,
                'entropy_increase': 15,
                'symbol_obfuscation': 15,
                'performance_acceptable': 20
            }
        }

        logger.info(f"  ✓ RE Difficulty Score: {score}/100 ({difficulty})")

        return results

    def _generate_all_reports(self):
        """Generate all report formats"""
        # Compile all results
        all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'program': self.program_name,
                'baseline': str(self.baseline),
                'obfuscated': str(self.obfuscated)
            },
            'test_results': self.test_results,
            'metrics': self.metrics
        }

        # Generate JSON report
        self.report_gen.generate_json_report(all_results)

        # Generate text report
        self.report_gen.generate_text_report(all_results)

        # Generate summary
        self.report_gen.generate_summary(all_results)

        logger.info(f"✓ Reports generated in {self.reports_dir}")

    def _run_advanced_analysis(self) -> Dict[str, Any]:
        """Run Ghidra, Binary Ninja, IDA Pro, and Angr analysis"""
        logger.info("  Running advanced analysis with Ghidra/BinaryNinja/IDA Pro/Angr...")

        results = {
            "ghidra": {},
            "binja": {},
            "ida": {},
            "angr": {}
        }

        try:
            # Ghidra analysis
            ghidra = GhidraAnalyzer(str(self.obfuscated))
            if ghidra.has_ghidra():
                logger.info("    [Ghidra] Decompiling functions...")
                results["ghidra"]["decompilation"] = ghidra.decompile_functions()
                results["ghidra"]["cfg_analysis"] = ghidra.analyze_control_flow()
            else:
                logger.warning("    [Ghidra] Not installed, skipping")
                results["ghidra"]["status"] = "not_installed"

            # Binary Ninja analysis
            binja = BinaryNinjaAnalyzer()
            if binja.has_binja():
                logger.info("    [Binary Ninja] Extracting HLIL...")
                results["binja"]["hlil"] = binja.extract_hlil(str(self.obfuscated))
                results["binja"]["baseline_hlil"] = binja.extract_hlil(str(self.baseline))
            else:
                logger.warning("    [Binary Ninja] Not installed, skipping")
                results["binja"]["status"] = "not_installed"

            # IDA Pro analysis
            ida = IDAProAnalyzer()
            if ida.has_ida():
                logger.info("    [IDA Pro] Analyzing binary...")
                results["ida"]["analysis"] = ida.analyze_binary(str(self.obfuscated))
                logger.info("    [IDA Pro] Comparing decompilation...")
                results["ida"]["decompilation_comparison"] = ida.compare_decompilation(
                    str(self.baseline), str(self.obfuscated)
                )
            else:
                logger.warning("    [IDA Pro] Not installed, skipping")
                results["ida"]["status"] = "not_installed"

            # Angr analysis
            angr_analyzer = AngrAnalyzer()
            if angr_analyzer.has_angr():
                logger.info("    [Angr] Running symbolic execution...")
                results["angr"]["symbolic_execution"] = angr_analyzer.symbolic_execution_analysis(str(self.obfuscated))
                results["angr"]["vulnerabilities"] = angr_analyzer.identify_vulnerable_patterns(str(self.obfuscated))
            else:
                logger.warning("    [Angr] Not installed, skipping")
                results["angr"]["status"] = "not_installed"

            return results

        except Exception as e:
            logger.warning(f"  Advanced analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _advanced_string_analysis(self) -> Dict[str, Any]:
        """Advanced string obfuscation detection"""
        logger.info("  Analyzing string obfuscation techniques...")

        try:
            analyzer = StringObfuscationAnalyzer()
            results = analyzer.analyze_string_patterns(str(self.baseline), str(self.obfuscated))
            logger.info(f"    String obfuscation confidence: {results.get('detection_confidence', 0):.1f}%")
            return results
        except Exception as e:
            logger.warning(f"  String analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _advanced_debuggability_analysis(self) -> Dict[str, Any]:
        """Advanced debugger resistance testing"""
        logger.info("  Testing debugger resistance...")

        try:
            analyzer = DebuggabilityAnalyzer()
            obf_results = analyzer.analyze_debuggability(str(self.obfuscated))
            baseline_results = analyzer.analyze_debuggability(str(self.baseline))

            logger.info(f"    Baseline debuggability: {baseline_results.get('debuggability_score', 0):.1f}")
            logger.info(f"    Obfuscated debuggability: {obf_results.get('debuggability_score', 0):.1f}")

            return {
                "baseline": baseline_results,
                "obfuscated": obf_results
            }
        except Exception as e:
            logger.warning(f"  Debuggability analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _analyze_code_coverage(self) -> Dict[str, Any]:
        """Analyze code coverage metrics"""
        logger.info("  Analyzing code coverage...")

        try:
            analyzer = CodeCoverageAnalyzer()
            baseline_coverage = analyzer.analyze_coverage(str(self.baseline))
            obf_coverage = analyzer.analyze_coverage(str(self.obfuscated))

            logger.info(f"    Baseline coverage: {baseline_coverage.get('estimated_coverage', 0):.1f}%")
            logger.info(f"    Obfuscated coverage: {obf_coverage.get('estimated_coverage', 0):.1f}%")

            return {
                "baseline": baseline_coverage,
                "obfuscated": obf_coverage
            }
        except Exception as e:
            logger.warning(f"  Coverage analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _assess_patchability(self) -> Dict[str, Any]:
        """Assess binary patchability"""
        logger.info("  Assessing binary patchability...")

        try:
            analyzer = PatchabilityAnalyzer()
            baseline_patch = analyzer.analyze_patchability(str(self.baseline))
            obf_patch = analyzer.analyze_patchability(str(self.obfuscated))

            logger.info(f"    Baseline patch difficulty: {baseline_patch.get('patch_difficulty', 'UNKNOWN')}")
            logger.info(f"    Obfuscated patch difficulty: {obf_patch.get('patch_difficulty', 'UNKNOWN')}")

            return {
                "baseline": baseline_patch,
                "obfuscated": obf_patch
            }
        except Exception as e:
            logger.warning(f"  Patchability analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    # Helper methods
    def _calculate_entropy(self, filepath: Path) -> float:
        """Calculate Shannon entropy of file"""
        with open(filepath, 'rb') as f:
            data = f.read()

        if not data:
            return 0.0

        entropy = 0.0
        for byte_val in range(256):
            freq = data.count(bytes([byte_val]))
            if freq > 0:
                p = freq / len(data)
                entropy -= p * (p and __import__('math').log2(p) or 0)

        return entropy

    def _extract_symbols(self, filepath: str) -> List[str]:
        """Extract symbols from binary"""
        try:
            result = run_command(f"nm {filepath} 2>/dev/null | awk '{{print $3}}' | grep -v '^$'")
            return result.strip().split('\n') if result else []
        except:
            return []

    def _measure_execution_time(self, filepath: Path, iterations: int = 3) -> float:
        """Measure average execution time

        Returns:
            float: Average execution time in ms, or negative value if binary failed
                  -1.0: Binary execution failed
                  -2.0: Binary timed out
        """
        times = []
        timeout_count = 0
        failure_count = 0

        for _ in range(iterations):
            try:
                start = time.time()
                # ✅ FIX #3a: Increased timeout from 5s to 30s to accommodate obfuscated binaries
                result = subprocess.run([str(filepath)], timeout=30, capture_output=True)
                elapsed = (time.time() - start) * 1000  # Convert to ms
                times.append(elapsed)
            except subprocess.TimeoutExpired:
                # ✅ FIX #3b: Track timeouts instead of fabricating data
                logger.warning(f"Execution timeout for {filepath.name} (exceeded 30s)")
                timeout_count += 1
            except Exception as e:
                logger.warning(f"Could not measure execution time for {filepath.name}: {e}")
                failure_count += 1

        # ✅ FIX #3c: Return error codes if binary didn't execute properly
        if failure_count > 0:
            logger.warning(f"  ⚠️  Binary execution failed for {filepath.name}")
            return -1.0  # Error code for failure

        if timeout_count >= iterations:
            logger.warning(f"  ⚠️  All execution attempts timed out for {filepath.name}")
            return -2.0  # Error code for timeout

        if not times:
            return -1.0  # Error: no successful measurements

        avg_time = sum(times) / len(times)
        logger.info(f"  ℹ️  {filepath.name}: {avg_time:.2f}ms (successful runs: {len(times)}/{iterations})")
        return avg_time

    def _has_debug_info(self, filepath: str) -> bool:
        """Check if binary has debug info"""
        try:
            result = run_command(f"file {filepath}")
            return 'not stripped' in result
        except:
            return False


def main():
    parser = argparse.ArgumentParser(description='OLLVM Obfuscation Test Suite')
    parser.add_argument('baseline', help='Path to baseline binary')
    parser.add_argument('obfuscated', help='Path to obfuscated binary')
    parser.add_argument('-r', '--results', default='/home/incharaj/oaas/obfuscation_test_suite/results',
                       help='Results directory')
    parser.add_argument('-n', '--name', default='program', help='Program name')

    args = parser.parse_args()

    suite = ObfuscationTestSuite(
        args.baseline,
        args.obfuscated,
        args.results,
        args.name
    )

    success = suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
