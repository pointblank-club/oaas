#!/usr/bin/env python3
"""Functional testing for obfuscation verification"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FunctionalTester:
    """Test functional equivalence between baseline and obfuscated binaries"""

    def __init__(self, baseline: Path, obfuscated: Path):
        self.baseline = Path(baseline)
        self.obfuscated = Path(obfuscated)

    def test_basic_io(self) -> bool:
        """Test basic I/O equivalence"""
        logger.debug("Testing basic I/O equivalence...")

        try:
            # Run both binaries with no input
            baseline_out = self._run_binary(self.baseline)
            obf_out = self._run_binary(self.obfuscated)

            # Compare outputs
            if baseline_out is None or obf_out is None:
                logger.warning("Could not run binaries for I/O test")
                return False

            return baseline_out == obf_out

        except Exception as e:
            logger.warning(f"I/O test failed: {e}")
            return False

    def test_with_input(self, test_inputs: list) -> bool:
        """Test with provided inputs"""
        logger.debug(f"Testing with {len(test_inputs)} inputs...")

        try:
            for test_input in test_inputs:
                baseline_out = self._run_binary(self.baseline, test_input)
                obf_out = self._run_binary(self.obfuscated, test_input)

                if baseline_out != obf_out:
                    logger.warning(f"Output mismatch for input: {test_input}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Input test failed: {e}")
            return False

    def test_exit_codes(self) -> bool:
        """Test exit code equivalence"""
        logger.debug("Testing exit codes...")

        try:
            baseline_code = self._get_exit_code(self.baseline)
            obf_code = self._get_exit_code(self.obfuscated)

            return baseline_code == obf_code

        except Exception as e:
            logger.warning(f"Exit code test failed: {e}")
            return False

    # Helper methods
    def _run_binary(self, binary: Path, input_data: Optional[str] = None) -> Optional[str]:
        """Run binary and return output"""
        try:
            # Increased timeout from 5s to 30s to accommodate obfuscated binaries
            # Obfuscated code is slower due to control flow complexity
            result = subprocess.run(
                [str(binary)],
                input=input_data.encode() if input_data else None,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning(f"Binary execution timed out (>30s): {binary}")
            return None
        except Exception as e:
            logger.warning(f"Could not run binary {binary}: {e}")
            return None

    def _get_exit_code(self, binary: Path) -> int:
        """Get exit code from binary"""
        try:
            result = subprocess.run(
                [str(binary)],
                capture_output=True,
                timeout=30  # Increased from 5s to 30s
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"Binary exit code test timed out (>30s): {binary}")
            return -2  # Different code to distinguish timeout from other failures
        except Exception as e:
            logger.warning(f"Could not get exit code for {binary}: {e}")
            return -1
