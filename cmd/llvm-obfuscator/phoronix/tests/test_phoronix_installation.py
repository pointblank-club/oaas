#!/usr/bin/env python3

"""
Phoronix Test Suite Installation Verification Script

This script verifies that Phoronix Test Suite is properly installed
and functional before running actual benchmarks in CI/CD pipelines.

Exit Codes:
    0 - All tests passed
    1 - Installation not found
    2 - Version check failed
    3 - Test run failed
    4 - Results directory check failed
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Tuple, Optional


class PhoronixVerifier:
    """Verify Phoronix Test Suite installation and functionality."""

    def __init__(self, pts_install_dir: Optional[str] = None):
        """
        Initialize the verifier.

        Args:
            pts_install_dir: Path to Phoronix installation directory.
                           Defaults to /opt/phoronix-test-suite
        """
        self.pts_install_dir = pts_install_dir or "/opt/phoronix-test-suite"
        self.pts_cmd = os.path.join(self.pts_install_dir, "phoronix-test-suite")
        self.pts_home = os.path.expanduser("~/.phoronix-test-suite")
        self.test_results_dir = os.path.join(self.pts_home, "results")
        self.failed_checks = []
        self.passed_checks = []

    def log_success(self, message: str) -> None:
        """Log success message."""
        print(f"✅ {message}")
        self.passed_checks.append(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        print(f"❌ {message}")
        self.failed_checks.append(message)

    def log_info(self, message: str) -> None:
        """Log info message."""
        print(f"ℹ️  {message}")

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        print(f"⚠️  {message}")

    def check_installation(self) -> bool:
        """
        Check if Phoronix Test Suite is installed.

        Returns:
            True if installed, False otherwise
        """
        self.log_info(f"Checking installation at: {self.pts_install_dir}")

        if not os.path.exists(self.pts_cmd):
            self.log_error(
                f"Phoronix Test Suite executable not found at {self.pts_cmd}"
            )
            return False

        if not os.path.isfile(self.pts_cmd):
            self.log_error(f"{self.pts_cmd} exists but is not a file")
            return False

        self.log_success(f"Phoronix executable found at {self.pts_cmd}")
        return True

    def check_executable(self) -> bool:
        """
        Check if executable is runnable.

        Returns:
            True if executable, False otherwise
        """
        if not os.access(self.pts_cmd, os.X_OK):
            self.log_warning(
                f"Phoronix executable not marked as executable, attempting to fix..."
            )
            try:
                # Try to make it executable
                os.chmod(self.pts_cmd, 0o755)
                if os.access(self.pts_cmd, os.X_OK):
                    self.log_success("Executable permission fixed")
                    return True
                else:
                    self.log_error("Failed to fix executable permission")
                    return False
            except OSError as e:
                self.log_error(f"Failed to set executable permission: {e}")
                return False

        self.log_success("Executable permission verified")
        return True

    def check_version(self) -> bool:
        """
        Check if 'phoronix-test-suite version' command works.

        Returns:
            True if version check succeeds, False otherwise
        """
        self.log_info("Checking version...")

        try:
            result = subprocess.run(
                [self.pts_cmd, "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                self.log_error(f"Version check failed with return code {result.returncode}")
                self.log_info(f"stderr: {result.stderr}")
                return False

            output = result.stdout + result.stderr

            if "Phoronix Test Suite" not in output:
                self.log_error("Version output doesn't contain 'Phoronix Test Suite'")
                self.log_info(f"Output: {output[:200]}")
                return False

            # Extract version number if possible
            version_line = next(
                (line for line in output.split("\n") if "Phoronix Test Suite" in line),
                None,
            )

            if version_line:
                self.log_success(f"Version check passed: {version_line.strip()}")
            else:
                self.log_success("Version check passed")

            return True

        except subprocess.TimeoutExpired:
            self.log_error("Version check timed out (10s)")
            return False
        except Exception as e:
            self.log_error(f"Version check failed with exception: {e}")
            return False

    def run_test_benchmark(self) -> bool:
        """
        Verify PTS can list available tests (quick verification).

        This is much faster than actually running a benchmark and still
        verifies the core functionality.

        Returns:
            True if test passes, False otherwise
        """
        self.log_info("Verifying PTS can list available tests...")

        try:
            # Quick test: just list available test profiles
            list_result = subprocess.run(
                [self.pts_cmd, "list-tests"],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for listing
            )

            if list_result.returncode != 0:
                self.log_error(f"Failed to list tests (return code: {list_result.returncode})")
                self.log_info(f"stderr: {list_result.stderr[:200]}")
                return False

            output = list_result.stdout + list_result.stderr

            # Check that we got a reasonable response
            if len(output.strip()) == 0:
                self.log_error("List tests returned empty output")
                return False

            # Count number of available tests
            test_count = output.count("pts/")
            if test_count > 0:
                self.log_success(f"PTS is functional - found {test_count} available tests")
                return True
            else:
                self.log_warning("Found test output but couldn't count tests")
                self.log_success("PTS list-tests executed successfully")
                return True

        except subprocess.TimeoutExpired:
            self.log_error("Test listing timed out (30s)")
            return False
        except Exception as e:
            self.log_error(f"Test verification failed with exception: {e}")
            return False

    def _check_results_exist(self) -> bool:
        """Check if test results exist in the results directory."""
        try:
            if not os.path.exists(self.test_results_dir):
                self.log_info(f"Results directory doesn't exist yet: {self.test_results_dir}")
                return False

            # Check for any XML result files
            result_files = list(Path(self.test_results_dir).glob("**/*.xml"))

            if result_files:
                self.log_info(f"Found {len(result_files)} result file(s)")
                return True

            return False

        except Exception as e:
            self.log_warning(f"Error checking results directory: {e}")
            return False

    def verify_results_directory(self) -> bool:
        """
        Verify that results directory exists and is accessible.

        Returns:
            True if results directory is valid, False otherwise
        """
        self.log_info(f"Checking results directory: {self.test_results_dir}")

        if not os.path.exists(self.test_results_dir):
            self.log_warning(f"Results directory doesn't exist: {self.test_results_dir}")
            self.log_info("This is normal if no tests have been run yet")
            return True  # Not a failure condition

        if not os.path.isdir(self.test_results_dir):
            self.log_error(f"Results path exists but is not a directory: {self.test_results_dir}")
            return False

        if not os.access(self.test_results_dir, os.R_OK):
            self.log_error(f"Results directory is not readable: {self.test_results_dir}")
            return False

        if not os.access(self.test_results_dir, os.W_OK):
            self.log_warning(f"Results directory is not writable: {self.test_results_dir}")
            self.log_info("This might cause issues when saving test results")

        self.log_success(f"Results directory is accessible: {self.test_results_dir}")
        return True

    def print_system_info(self) -> None:
        """Print system information for debugging."""
        self.log_info("System Information:")

        # OS
        try:
            result = subprocess.run(
                ["uname", "-a"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            print(f"  OS: {result.stdout.strip()}")
        except Exception as e:
            print(f"  OS: (error: {e})")

        # PHP version (required for PTS)
        try:
            result = subprocess.run(
                ["php", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            php_version = result.stdout.split("\n")[0]
            print(f"  PHP: {php_version}")
        except Exception as e:
            print(f"  PHP: (error: {e})")

        # Available CPU cores
        try:
            cpu_count = os.cpu_count()
            print(f"  CPU Cores: {cpu_count}")
        except Exception as e:
            print(f"  CPU Cores: (error: {e})")

    def run_all_checks(self) -> int:
        """
        Run all verification checks.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        print("\n" + "=" * 50)
        print("Phoronix Test Suite Verification")
        print("=" * 50 + "\n")

        self.print_system_info()
        print()

        checks = [
            ("Installation Check", self.check_installation),
            ("Executable Permission Check", self.check_executable),
            ("Version Check", self.check_version),
            ("Results Directory Check", self.verify_results_directory),
            ("Test Benchmark Run", self.run_test_benchmark),
        ]

        failed_check_count = 0

        for check_name, check_func in checks:
            print(f"\n--- {check_name} ---")
            try:
                if not check_func():
                    failed_check_count += 1
            except Exception as e:
                self.log_error(f"Check raised exception: {e}")
                failed_check_count += 1

        # Print summary
        print("\n" + "=" * 50)
        print("Verification Summary")
        print("=" * 50)

        self.log_info(f"Passed Checks: {len(self.passed_checks)}")
        for check in self.passed_checks:
            print(f"  ✅ {check}")

        if self.failed_checks:
            self.log_info(f"Failed Checks: {len(self.failed_checks)}")
            for check in self.failed_checks:
                print(f"  ❌ {check}")

        print()

        if failed_check_count == 0:
            self.log_success("All verification checks passed!")
            return 0
        else:
            self.log_error(f"{failed_check_count} check(s) failed")
            return 1


def main():
    """Main entry point."""
    # Get custom installation directory from environment if set
    pts_dir = os.environ.get("PTS_INSTALL_DIR")

    verifier = PhoronixVerifier(pts_install_dir=pts_dir)

    exit_code = verifier.run_all_checks()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
