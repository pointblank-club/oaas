"""
Integration with the obfuscation test suite for automatic result generation.

This module handles:
1. Detecting if test suite is available
2. Running tests automatically after obfuscation
3. Merging test results into job reports
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

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
        logger.debug("Test suite not available, skipping automatic testing")
        return None

    try:
        test_suite_path = Path(__file__).parent.parent.parent / "obfuscation_test_suite"

        # Use provided results dir or create in test suite results
        if results_dir is None:
            results_dir = test_suite_path / "results" / program_name
        else:
            results_dir = Path(results_dir) / program_name

        results_dir.mkdir(parents=True, exist_ok=True)

        # Run the test suite
        logger.info(f"Running obfuscation test suite: {baseline_binary} vs {obfuscated_binary}")

        cmd = [
            sys.executable,
            str(test_suite_path / "obfuscation_test_suite.py"),
            str(baseline_binary),
            str(obfuscated_binary),
            "-r", str(results_dir.parent),
            "-n", program_name
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for tests
        )

        if result.returncode != 0:
            logger.warning(f"Test suite failed with return code {result.returncode}")
            logger.debug(f"Test suite stderr: {result.stderr}")
            return None

        # Read and return test results
        report_path = results_dir / f"{program_name}_results.json"
        if report_path.exists():
            with open(report_path) as f:
                test_results = json.load(f)
            logger.info(f"Test suite completed, results at: {report_path}")
            return test_results
        else:
            logger.warning(f"Test results file not found at {report_path}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("Test suite execution timed out (5 minutes)")
        return None
    except Exception as e:
        logger.warning(f"Failed to run test suite: {e}")
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
