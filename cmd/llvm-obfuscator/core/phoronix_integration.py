"""
Phoronix Benchmarking Integration

Runs the obfuscation test suite comparing baseline vs obfuscated binaries,
and aggregates results into the obfuscation report.
"""

import json
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class PhoronixBenchmarkRunner:
    """Runs Phoronix benchmarking suite and collects results."""

    def __init__(self, script_dir: Path = None):
        """
        Initialize Phoronix runner.

        Args:
            script_dir: Path to phoronix scripts directory
                       Defaults to <project_root>/phoronix/scripts/
        """
        if script_dir is None:
            # Auto-detect phoronix scripts directory
            # __file__ = /app/core/phoronix_integration.py (in container) or <project_root>/cmd/llvm-obfuscator/core/phoronix_integration.py
            # parents[1] gets us to /app or <project_root>/cmd/llvm-obfuscator
            app_root = Path(__file__).resolve().parents[1]
            # Try app structure first (Docker container: /app)
            script_dir = app_root / "phoronix" / "scripts"
            if not script_dir.exists():
                # Fallback: try going up to project root (local development)
                project_root = Path(__file__).resolve().parents[3]
                script_dir = project_root / "phoronix" / "scripts"

        self.script_dir = Path(script_dir)
        self.main_script = self.script_dir / "run_obfuscation_test_suite.sh"
        self.logger = logging.getLogger(__name__)

        if not self.main_script.exists():
            raise FileNotFoundError(f"Phoronix script not found: {self.main_script}")

    def run_benchmark(
        self,
        baseline_binary: Path,
        obfuscated_binary: Path,
        output_dir: Optional[Path] = None,
        timeout: int = 3600  # 1 hour default
    ) -> Dict[str, Any]:
        """
        Run obfuscation benchmarking suite.

        Args:
            baseline_binary: Path to baseline (unobfuscated) binary
            obfuscated_binary: Path to obfuscated binary
            output_dir: Output directory for results (auto-generated with timestamp)
            timeout: Maximum time to wait for benchmarking (seconds)

        Returns:
            Dictionary with benchmark results
        """
        import time

        start_time = time.time()
        results = {
            'success': False,
            'output_dir': None,
            'results': {},
            'errors': None,
            'execution_time_seconds': 0.0
        }

        try:
            # Validate binaries exist
            if not baseline_binary.exists():
                raise FileNotFoundError(f"Baseline binary not found: {baseline_binary}")
            if not obfuscated_binary.exists():
                raise FileNotFoundError(f"Obfuscated binary not found: {obfuscated_binary}")

            # Create output directory if not specified
            if output_dir is None:
                output_dir = Path(tempfile.gettempdir()) / f"phoronix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)
            results['output_dir'] = str(output_dir)

            self.logger.info(f"Running Phoronix benchmarking...")
            self.logger.info(f"  Baseline:   {baseline_binary}")
            self.logger.info(f"  Obfuscated: {obfuscated_binary}")
            self.logger.info(f"  Output:     {output_dir}")

            # Run the benchmark script
            cmd = [
                "bash",
                str(self.main_script),
                str(baseline_binary),
                str(obfuscated_binary),
                str(output_dir)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                error_msg = f"Benchmark script failed: {result.stderr}"
                self.logger.error(error_msg)
                results['errors'] = error_msg
                results['execution_time_seconds'] = time.time() - start_time
                return results

            # Parse results from output directory
            analysis_dir = self._find_analysis_dir(output_dir)
            if not analysis_dir:
                error_msg = "No analysis directory found in output"
                self.logger.error(error_msg)
                results['errors'] = error_msg
                results['execution_time_seconds'] = time.time() - start_time
                return results

            # Load all result files
            parsed_results = self._parse_results(analysis_dir)
            results['results'] = parsed_results
            results['success'] = True

            self.logger.info(f"✅ Benchmarking completed successfully")
            self.logger.info(f"Results saved to: {analysis_dir}")

        except subprocess.TimeoutExpired:
            results['errors'] = f"Benchmark timeout after {timeout} seconds"
            self.logger.error(results['errors'])
        except Exception as e:
            results['errors'] = str(e)
            self.logger.error(f"Benchmark failed: {e}", exc_info=True)
        finally:
            results['execution_time_seconds'] = time.time() - start_time

        return results

    def _find_analysis_dir(self, output_base: Path) -> Optional[Path]:
        """
        Find the analysis directory created by the script.

        The script creates: obfuscation_analysis_{name1}_vs_{name2}_{timestamp}/
        """
        import glob

        pattern = str(output_base / "obfuscation_analysis_*")
        matches = glob.glob(pattern)

        if matches:
            # Return the most recent (latest timestamp)
            return Path(max(matches))

        return None

    def _parse_results(self, analysis_dir: Path) -> Dict[str, Any]:
        """
        Parse all result files from the analysis directory.

        Returns:
            {
                'metrics': {...},          # From metrics/metrics.json
                'security': {...},         # From security/security_analysis.json
                'final_report': {...},     # From reports/final_report.json
                'logs': {...}              # Log file locations
            }
        """
        results = {
            'metrics': None,
            'security': None,
            'final_report': None,
            'logs': {}
        }

        # Parse metrics
        metrics_file = analysis_dir / "metrics" / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    results['metrics'] = json.load(f)
                self.logger.debug(f"Loaded metrics from {metrics_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load metrics: {e}")

        # Parse security analysis
        security_file = analysis_dir / "security" / "security_analysis.json"
        if security_file.exists():
            try:
                with open(security_file) as f:
                    results['security'] = json.load(f)
                self.logger.debug(f"Loaded security analysis from {security_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load security analysis: {e}")

        # Parse final report
        final_file = analysis_dir / "reports" / "final_report.json"
        if final_file.exists():
            try:
                with open(final_file) as f:
                    results['final_report'] = json.load(f)
                self.logger.debug(f"Loaded final report from {final_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load final report: {e}")

        # Log file locations
        logs_dir = analysis_dir / "logs"
        if logs_dir.exists():
            results['logs'] = {
                'execution_log': str(logs_dir / "execution.log"),
                'metrics_log': str(logs_dir / "metrics.log"),
                'security_log': str(logs_dir / "security.log"),
                'report_log': str(logs_dir / "report.log"),
            }

        return results

    def extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only RELIABLE metrics for PDF report integration.

        Based on extensive analysis, only these metrics are scientifically sound:
        1. Instruction count increase (70% accurate - directly measurable)
        2. Performance overhead percentage (if available)

        All other metrics (entropy, CFG distortion, decompilation score) are
        based on heuristics with <30% accuracy and are NOT included.
        """
        extracted = {
            'available': False,
            'instruction_count_delta': None,
            'instruction_count_increase_percent': None,
            'performance_overhead_percent': None,
        }

        if not results.get('results'):
            return extracted

        # Extract instruction count delta (RELIABLE METRIC)
        metrics = results['results'].get('metrics', {})
        if metrics:
            extracted['available'] = True

            # Get instruction count increase from comparison
            comparison = metrics.get('comparison', {})
            if comparison:
                first_comparison = next(iter(comparison.values()), {})
                instr_delta = first_comparison.get('instruction_count_delta', 0)
                baseline_instr = metrics.get('baseline', {}).get('instruction_count', 1)

                extracted['instruction_count_delta'] = instr_delta
                if baseline_instr > 0:
                    extracted['instruction_count_increase_percent'] = round(
                        (instr_delta / baseline_instr) * 100, 2
                    )

        # Extract performance overhead from final report (if available)
        final_report = results['results'].get('final_report', {})
        if final_report:
            perf = final_report.get('performance_overhead_summary', {})
            perf_metrics = perf.get('performance_metrics', {})
            overhead = perf_metrics.get('overhead_percent')
            # Only include if it's a non-zero actual measurement (not default 0.0)
            if overhead is not None and overhead != 0.0:
                extracted['performance_overhead_percent'] = overhead

        return extracted


def run_phoronix_benchmark(
    baseline_binary: Path,
    obfuscated_binary: Path,
    output_dir: Optional[Path] = None,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Convenience function to run Phoronix benchmarking.

    Returns results dict suitable for adding to job_data.
    """
    runner = PhoronixBenchmarkRunner()
    return runner.run_benchmark(baseline_binary, obfuscated_binary, output_dir, timeout)
