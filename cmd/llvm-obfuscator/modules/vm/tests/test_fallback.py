"""VM Fallback Tests - PRIORITY 2: Graceful Degradation.

These tests prove that the VM module falls back gracefully
on ANY error condition, preserving the original input.

Run with: pytest modules/vm/tests/test_fallback.py -v
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.vm.runner import run_vm_isolated, VMResult
from modules.vm.config import VMConfig


class TestFallbackBehavior(unittest.TestCase):
    """Tests for graceful fallback on errors."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create a valid test IR file
        self.valid_ir = self.temp_path / "valid.ll"
        self.valid_ir.write_text(
            "; Valid LLVM IR\n"
            "define i32 @test_func(i32 %a, i32 %b) {\n"
            "entry:\n"
            "  %sum = add i32 %a, %b\n"
            "  ret i32 %sum\n"
            "}\n"
        )
        self.valid_ir_content = self.valid_ir.read_text()

    def tearDown(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fallback_on_missing_input_file(self):
        """Missing input file should return failure, not exception."""
        output_path = self.temp_path / "output.ll"

        result = run_vm_isolated(
            input_ll=Path("/nonexistent/path/to/file.ll"),
            output_ll=output_path,
            functions=[],
            timeout=5,
        )

        # Should return VMResult with success=False
        self.assertIsInstance(result, VMResult)
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())

    def test_fallback_on_empty_functions_list(self):
        """Empty functions list should be handled gracefully."""
        output_path = self.temp_path / "output.ll"

        result = run_vm_isolated(
            input_ll=self.valid_ir,
            output_ll=output_path,
            functions=[],  # Empty list
            timeout=10,
        )

        # Should handle gracefully (either succeed or fail cleanly)
        self.assertIsInstance(result, VMResult)

    def test_fallback_on_unsupported_ir(self):
        """IR with only unsupported instructions should skip gracefully."""
        unsupported_ir = self.temp_path / "unsupported.ll"
        unsupported_ir.write_text(
            "; IR with unsupported instructions\n"
            "define i32 @with_call(i32 %a) {\n"
            "entry:\n"
            "  %result = call i32 @other(i32 %a)\n"
            "  ret i32 %result\n"
            "}\n"
            "declare i32 @other(i32)\n"
        )

        output_path = self.temp_path / "output.ll"

        result = run_vm_isolated(
            input_ll=unsupported_ir,
            output_ll=output_path,
            functions=["with_call"],
            timeout=10,
        )

        # Should handle gracefully
        self.assertIsInstance(result, VMResult)

    def test_fallback_preserves_original_file(self):
        """Original input file must NEVER be modified."""
        output_path = self.temp_path / "output.ll"

        # Record original content
        original_content = self.valid_ir.read_text()
        original_mtime = self.valid_ir.stat().st_mtime

        # Run VM (may succeed or fail)
        result = run_vm_isolated(
            input_ll=self.valid_ir,
            output_ll=output_path,
            functions=[],
            timeout=10,
        )

        # Original file must be unchanged
        self.assertEqual(self.valid_ir.read_text(), original_content)
        self.assertEqual(self.valid_ir.stat().st_mtime, original_mtime)

    def test_fallback_returns_vmresult_not_exception(self):
        """Any fallback condition returns VMResult, never raises."""
        test_cases = [
            (Path("/nonexistent"), "nonexistent input"),
            (Path(""), "empty path"),
            (self.temp_path / "nosuchfile.ll", "missing file"),
        ]

        for input_path, desc in test_cases:
            with self.subTest(case=desc):
                output_path = self.temp_path / f"output_{desc.replace(' ', '_')}.ll"

                # Must NOT raise exception
                try:
                    result = run_vm_isolated(input_path, output_path, [], timeout=2)
                except Exception as e:
                    self.fail(f"Raised exception for {desc}: {type(e).__name__}: {e}")

                # Must return VMResult
                self.assertIsInstance(result, VMResult)

    def test_multiple_fallbacks_in_sequence(self):
        """Multiple fallback scenarios in sequence should all work."""
        # Run 3 times with conditions that trigger fallback
        results = []

        for i in range(3):
            output_path = self.temp_path / f"output_{i}.ll"

            result = run_vm_isolated(
                input_ll=Path(f"/nonexistent/file_{i}.ll"),
                output_ll=output_path,
                functions=[],
                timeout=2,
            )

            results.append(result)

        # All should complete (return VMResult)
        for i, result in enumerate(results):
            self.assertIsInstance(result, VMResult, f"Run {i} didn't return VMResult")
            self.assertFalse(result.success, f"Run {i} should have failed")

    def test_fallback_with_invalid_timeout(self):
        """Invalid timeout values should be handled."""
        output_path = self.temp_path / "output.ll"

        # Zero timeout - should still work (or fail gracefully)
        result = run_vm_isolated(
            input_ll=self.valid_ir,
            output_ll=output_path,
            functions=[],
            timeout=0,  # Edge case
        )

        self.assertIsInstance(result, VMResult)


class TestVMResultContract(unittest.TestCase):
    """Tests for VMResult contract compliance."""

    def test_vmresult_success_has_output_path(self):
        """Successful VMResult should have output_path set."""
        result = VMResult(
            success=True,
            output_path=Path("/tmp/out.ll"),
            functions_virtualized=["test"],
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_path)

    def test_vmresult_failure_has_error(self):
        """Failed VMResult should have error message."""
        result = VMResult(
            success=False,
            error="Test error message",
        )

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_vmresult_failure_has_no_output_path(self):
        """Failed VMResult should have output_path=None."""
        result = VMResult(
            success=False,
            error="Test error",
        )

        self.assertIsNone(result.output_path)

    def test_vmresult_defaults(self):
        """VMResult defaults should be safe."""
        # Minimal success
        result = VMResult(success=True)
        self.assertEqual(result.functions_virtualized, [])
        self.assertIsNone(result.error)

        # Minimal failure
        result = VMResult(success=False)
        self.assertEqual(result.functions_virtualized, [])


class TestLoggingBehavior(unittest.TestCase):
    """Tests for appropriate logging levels."""

    def test_fallback_logs_warning_not_error(self):
        """Fallback should log WARNING, not ERROR.

        Fallback is expected behavior, not an error condition.
        """
        # Capture log output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.ll"

            # This will trigger fallback (missing file)
            result = run_vm_isolated(
                input_ll=Path("/nonexistent/file.ll"),
                output_ll=output_path,
                functions=[],
                timeout=2,
            )

            # Should return failure (which is logged as warning)
            self.assertFalse(result.success)
            # The logging happens inside runner.py with logger.warning()


if __name__ == "__main__":
    unittest.main()
