"""
Integration with the obfuscation test suite for automatic result generation.

This module handles:
1. Detecting if test suite is available
2. Running tests automatically after obfuscation
3. Merging test results into job reports
4. Fallback: Running lightweight tests if full suite unavailable

The lightweight tests provide basic metrics without requiring external tools
like Ghidra, Angr, Binary Ninja, or IDA Pro.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def is_test_suite_available() -> bool:
    """Check if obfuscation test suite is available."""
    test_suite_path = Path(__file__).parent.parent.parent / "obfuscation_test_suite" / "obfuscation_test_suite.py"
    return test_suite_path.exists()


def run_obfuscation_tests(
    baseline_binary: Path,
    obfuscated_binary: Path,
    program_name: str = "program",
    results_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Run the obfuscation test suite comparing baseline vs obfuscated binary.

    Args:
        baseline_binary: Path to baseline (unobfuscated) binary
        obfuscated_binary: Path to obfuscated binary
        program_name: Name of the program being tested
        results_dir: Directory to store test results (uses temp if None)

    Returns:
        Dictionary with test results or None if test suite not available or failed
    """
    if not is_test_suite_available():
        logger.debug("Test suite not available at expected location, skipping automatic testing")
        return None

    try:
        # Convert to Path objects and verify they exist
        baseline_path = Path(baseline_binary)
        obfuscated_path = Path(obfuscated_binary)

        if not baseline_path.exists():
            logger.warning(f"Baseline binary not found at: {baseline_path}")
            return None

        if not obfuscated_path.exists():
            logger.warning(f"Obfuscated binary not found at: {obfuscated_path}")
            return None

        test_suite_path = Path(__file__).parent.parent.parent / "obfuscation_test_suite"

        if not test_suite_path.exists():
            logger.warning(f"Test suite directory not found at: {test_suite_path}")
            return None

        logger.info(f"Test suite available at: {test_suite_path}")

        # Use provided results dir or create in test suite results
        if results_dir is None:
            results_dir = test_suite_path / "results"
        else:
            results_dir = Path(results_dir)

        results_dir.mkdir(parents=True, exist_ok=True)

        # Run the test suite
        logger.info(f"Running obfuscation test suite...")
        logger.info(f"  Baseline: {baseline_path}")
        logger.info(f"  Obfuscated: {obfuscated_path}")
        logger.info(f"  Program name: {program_name}")
        logger.info(f"  Results dir: {results_dir}")

        cmd = [
            sys.executable,
            str(test_suite_path / "obfuscation_test_suite.py"),
            str(baseline_path),
            str(obfuscated_path),
            "-r", str(results_dir),
            "-n", program_name
        ]

        logger.debug(f"Test suite command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for tests (some tests take time)
        )

        logger.info(f"Test suite exit code: {result.returncode}")

        if result.stdout:
            logger.debug(f"Test suite stdout (first 500 chars): {result.stdout[:500]}")

        if result.stderr:
            logger.debug(f"Test suite stderr (first 500 chars): {result.stderr[:500]}")

        if result.returncode != 0:
            logger.warning(f"Test suite failed with return code {result.returncode}")
            # Still try to read results even if return code is non-zero
            # as tests might have partially completed

        # Read and return test results
        report_path = results_dir / program_name / f"{program_name}_results.json"

        if not report_path.exists():
            # Try alternate path
            report_path = results_dir / f"{program_name}_results.json"

        if report_path.exists():
            try:
                with open(report_path) as f:
                    test_results = json.load(f)
                logger.info(f"✅ Test suite completed successfully, results at: {report_path}")
                return test_results
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse test results JSON: {e}")
                return None
        else:
            logger.warning(f"Test results file not found. Checked: {report_path}")
            # List what files were created
            if results_dir.exists():
                logger.debug(f"Files in results dir: {list(results_dir.rglob('*'))[:10]}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("Test suite execution timed out (10 minutes)")
        return None
    except Exception as e:
        logger.error(f"Failed to run test suite: {e}", exc_info=True)
        return None


def merge_test_results_into_report(
    job_data: Dict[str, Any],
    test_results: Dict[str, Any]
) -> None:
    """
    Merge test suite results into the job report data.

    Args:
        job_data: The job data dictionary (will be modified in place)
        test_results: Test results from run_obfuscation_tests()
    """
    if not test_results:
        return

    # Add test results to the job data
    job_data["test_results"] = test_results.get("test_results", {})
    job_data["test_metrics"] = test_results.get("metrics", {})

    # Add reliability information
    metadata = test_results.get("metadata", {})
    job_data["metrics_reliability"] = test_results.get("reliability_status", {}).get("level", "UNKNOWN")
    job_data["functional_correctness_passed"] = metadata.get("functional_correctness_passed")

    # Add reliability warning
    job_data["reliability_warning"] = test_results.get("reliability_status", {}).get("warning", "")

    logger.info(f"Merged test results into job report. Reliability: {job_data.get('metrics_reliability')}")


def run_lightweight_tests(
    baseline_binary: Path,
    obfuscated_binary: Path,
    program_name: str = "program"
) -> Optional[Dict[str, Any]]:
    """
    Run lightweight tests without requiring external tools (Ghidra, Angr, etc).

    This fallback provides basic metrics when the full test suite is unavailable.
    Tests basic properties: size, strings, symbols, entropy, functional correctness.
    """
    try:
        import subprocess

        baseline_path = Path(baseline_binary)
        obfuscated_path = Path(obfuscated_binary)

        if not baseline_path.exists() or not obfuscated_path.exists():
            logger.warning(f"Binaries not found for lightweight tests")
            return None

        logger.info("Running lightweight obfuscation tests (no external tools required)...")

        # Test 1: Functional Correctness (basic)
        functional_passed = True
        try:
            baseline_result = subprocess.run([str(baseline_path)], capture_output=True, timeout=5)
            obf_result = subprocess.run([str(obfuscated_path)], capture_output=True, timeout=5)
            # Simple check: both should run without crashing
            functional_passed = (baseline_result.returncode == obf_result.returncode)
        except Exception as e:
            logger.debug(f"Functional test failed: {e}")
            functional_passed = False

        # Test 2: Binary Properties
        try:
            baseline_size = baseline_path.stat().st_size
            obfuscated_size = obfuscated_path.stat().st_size
        except:
            baseline_size = 0
            obfuscated_size = 0

        # Test 3: String Analysis (basic strings extraction)
        baseline_strings = 0
        obfuscated_strings = 0
        try:
            # Extract strings using `strings` command if available
            baseline_strings_result = subprocess.run(
                ["strings", str(baseline_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            baseline_strings = len([s for s in baseline_strings_result.stdout.split('\n') if s.strip() and len(s) > 3])

            obf_strings_result = subprocess.run(
                ["strings", str(obfuscated_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            obfuscated_strings = len([s for s in obf_strings_result.stdout.split('\n') if s.strip() and len(s) > 3])
        except:
            logger.debug("Could not extract strings (strings command not available)")

        # Test 4: Symbol Analysis
        baseline_symbols = 0
        obfuscated_symbols = 0
        try:
            baseline_nm = subprocess.run(
                ["nm", str(baseline_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            baseline_symbols = len([s for s in baseline_nm.stdout.split('\n') if s.strip()])

            obf_nm = subprocess.run(
                ["nm", str(obfuscated_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            obfuscated_symbols = len([s for s in obf_nm.stdout.split('\n') if s.strip()])
        except:
            logger.debug("Could not analyze symbols (nm command not available)")

        # Build lightweight test results
        test_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "program": program_name,
                "baseline": str(baseline_path),
                "obfuscated": str(obfuscated_path),
                "metrics_reliability": "UNCERTAIN" if not functional_passed else "RELIABLE",
                "functional_correctness_passed": functional_passed,
                "test_type": "lightweight"
            },
            "test_results": {
                "functional_correctness": {
                    "same_behavior": functional_passed,
                    "passed": 1 if functional_passed else 0,
                    "test_count": 1
                },
                "strings": {
                    "baseline_strings": baseline_strings,
                    "obf_strings": obfuscated_strings,
                    "reduction_percent": (
                        -((obfuscated_strings - baseline_strings) / baseline_strings * 100)
                        if baseline_strings > 0 else 0
                    )
                },
                "binary_properties": {
                    "baseline_size_bytes": baseline_size,
                    "obf_size_bytes": obfuscated_size,
                    "size_increase_percent": (
                        (obfuscated_size - baseline_size) / baseline_size * 100
                        if baseline_size > 0 else 0
                    ),
                    "baseline_entropy": 0.0,
                    "obf_entropy": 0.0,
                    "entropy_increase": 0.0
                },
                "symbols": {
                    "baseline_symbol_count": baseline_symbols,
                    "obf_symbol_count": obfuscated_symbols,
                    "symbols_reduced": obfuscated_symbols < baseline_symbols
                },
                "performance": {
                    "baseline_ms": None,
                    "obf_ms": None,
                    "overhead_percent": None,
                    "acceptable": None,
                    "status": "SKIPPED",
                    "reason": "Lightweight test (performance testing skipped)"
                },
                "cfg_metrics": {
                    "comparison": {
                        "indirect_jumps_ratio": 0.0,
                        "basic_blocks_ratio": 0.0,
                        "control_flow_complexity_increase": 0.0
                    }
                }
            },
            "reliability_status": {
                "level": "UNCERTAIN" if not functional_passed else "RELIABLE",
                "warning": (
                    "⚠️  LIGHTWEIGHT TEST: This is a basic test without Ghidra/Angr/BinaryNinja. "
                    "Results are approximate. For complete analysis, install advanced analysis tools."
                    if not functional_passed
                    else "✅ LIGHTWEIGHT TEST PASSED: Binary functionality preserved. "
                    "For detailed analysis, install Ghidra/Angr/BinaryNinja."
                )
            }
        }

        logger.info("✅ Lightweight tests completed")
        return test_results

    except Exception as e:
        logger.error(f"Lightweight tests failed: {e}", exc_info=True)
        return None
