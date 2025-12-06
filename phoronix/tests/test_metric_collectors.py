#!/usr/bin/env python3
"""
Unit tests for metric collectors.

Verifies that binary metric extraction, JSON output, entropy computation,
and CFG parsing work correctly without crashing.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Assuming the metric collector is in scripts/collect_obfuscation_metrics.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from collect_obfuscation_metrics import MetricsCollector, BinaryMetrics
except ImportError:
    # Fallback for testing without actual module import
    BinaryMetrics = None
    MetricsCollector = None


class TestBinaryMetrics(unittest.TestCase):
    """Test BinaryMetrics dataclass."""

    def test_binary_metrics_creation(self):
        """Test BinaryMetrics dataclass instantiation."""
        if BinaryMetrics is None:
            self.skipTest("BinaryMetrics not available")

        metrics = BinaryMetrics(
            file_size_bytes=1024,
            file_size_percent_increase=10.0,
            text_section_size=512,
            num_functions=50,
            num_basic_blocks=200,
            instruction_count=1000,
            text_entropy=7.5,
            cyclomatic_complexity=15.0,
            stripped=False,
            pie_enabled=True,
        )

        self.assertEqual(metrics.file_size_bytes, 1024)
        self.assertEqual(metrics.num_functions, 50)
        self.assertEqual(metrics.text_entropy, 7.5)
        self.assertEqual(metrics.pie_enabled, True)

    def test_binary_metrics_to_dict(self):
        """Test BinaryMetrics can be converted to dict."""
        if BinaryMetrics is None:
            self.skipTest("BinaryMetrics not available")

        from dataclasses import asdict

        metrics = BinaryMetrics(
            file_size_bytes=1024,
            file_size_percent_increase=0.0,
            text_section_size=512,
            num_functions=50,
            num_basic_blocks=200,
            instruction_count=1000,
            text_entropy=7.5,
            cyclomatic_complexity=15.0,
            stripped=False,
            pie_enabled=True,
        )

        metrics_dict = asdict(metrics)
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('file_size_bytes', metrics_dict)
        self.assertIn('num_functions', metrics_dict)


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if MetricsCollector is None:
            self.skipTest("MetricsCollector not available")
        self.collector = MetricsCollector()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch('subprocess.run')
    def test_get_text_section_size(self, mock_run):
        """Test .text section extraction."""
        mock_run.return_value = MagicMock(
            stdout='  [1] .text    PROGBITS    0000000000001000  00001000  000000ab',
            returncode=0
        )

        with tempfile.NamedTemporaryFile() as tmp:
            size = self.collector._get_text_section_size(Path(tmp.name))
            self.assertGreaterEqual(size, 0)
            mock_run.assert_called()

    @patch('subprocess.run')
    def test_count_functions(self, mock_run):
        """Test function count extraction."""
        mock_run.return_value = MagicMock(
            stdout='0000000000001000 T main\n0000000000002000 T helper\n',
            returncode=0
        )

        with tempfile.NamedTemporaryFile() as tmp:
            count = self.collector._count_functions(Path(tmp.name))
            self.assertEqual(count, 2)
            mock_run.assert_called()

    @patch('subprocess.run')
    def test_count_instructions(self, mock_run):
        """Test instruction count extraction."""
        mock_run.return_value = MagicMock(
            stdout="""00001000 <main>:
    1000:    55                      push   %rbp
    1001:    48 89 e5                mov    %rsp,%rbp
    1004:    c3                      retq
""",
            returncode=0
        )

        with tempfile.NamedTemporaryFile() as tmp:
            count = self.collector._count_instructions(Path(tmp.name))
            self.assertGreater(count, 0)
            mock_run.assert_called()

    @patch('subprocess.run')
    def test_is_stripped(self, mock_run):
        """Test stripped binary detection."""
        mock_run.return_value = MagicMock(
            stdout='binary.o: ELF 64-bit LSB executable, x86-64, not stripped',
            returncode=0
        )

        with tempfile.NamedTemporaryFile() as tmp:
            is_stripped = self.collector._is_stripped(Path(tmp.name))
            self.assertFalse(is_stripped)

    @patch('subprocess.run')
    def test_is_pie_enabled(self, mock_run):
        """Test PIE detection."""
        mock_run.return_value = MagicMock(
            stdout='Type: DYN (Position-independent executable file)',
            returncode=0
        )

        with tempfile.NamedTemporaryFile() as tmp:
            pie = self.collector._is_pie_enabled(Path(tmp.name))
            self.assertTrue(pie)

    def test_entropy_computation(self):
        """Test Shannon entropy computation."""
        # Create a test binary with known entropy
        with tempfile.NamedTemporaryFile() as tmp:
            # Write some test data
            test_data = b'\x00\x00\x00\x00\x01\x01\x01\x01'
            tmp.write(test_data)
            tmp.flush()

            # Mock readelf to return a valid offset
            with patch.object(self.collector, '_find_text_section_range',
                            return_value=(0, len(test_data))):
                entropy = self.collector._compute_text_entropy(Path(tmp.name))
                self.assertGreaterEqual(entropy, 0.0)
                self.assertLessEqual(entropy, 8.0)  # Max entropy for 8-bit data

    @patch('subprocess.run')
    def test_cyclomatic_complexity_estimation(self, mock_run):
        """Test cyclomatic complexity estimation."""
        mock_run.return_value = MagicMock(
            stdout="""00001000 <main>:
    1000:    je 2000 <exit>
    1002:    jne 3000 <loop>
    1004:    retq
00002000 <exit>:
    2000:    retq
00003000 <loop>:
    3000:    jmp 1000 <main>
""",
            returncode=0
        )

        with tempfile.NamedTemporaryFile() as tmp:
            cc = self.collector._estimate_cyclomatic_complexity(Path(tmp.name))
            self.assertGreaterEqual(cc, 1.0)

    def test_missing_binary_handling(self):
        """Test handling of missing binary file."""
        missing_path = Path(self.temp_dir.name) / "nonexistent.bin"

        result = self.collector._analyze_binary(missing_path)
        self.assertIsNone(result)

    def test_collect_metrics_json_output(self):
        """Test that collect_metrics produces valid JSON structure."""
        if MetricsCollector is None:
            self.skipTest("MetricsCollector not available")

        output_dir = Path(self.temp_dir.name)

        # Create mock binaries
        baseline = output_dir / "baseline"
        obfuscated1 = output_dir / "obfuscated1"
        baseline.touch()
        obfuscated1.touch()

        with patch.object(self.collector, '_analyze_binary') as mock_analyze:
            # Return valid metrics
            if BinaryMetrics is not None:
                mock_metrics = BinaryMetrics(
                    file_size_bytes=1000,
                    file_size_percent_increase=0.0,
                    text_section_size=500,
                    num_functions=10,
                    num_basic_blocks=50,
                    instruction_count=200,
                    text_entropy=7.0,
                    cyclomatic_complexity=5.0,
                    stripped=False,
                    pie_enabled=True,
                )
                mock_analyze.return_value = mock_metrics

                results = self.collector.collect_metrics(
                    baseline,
                    [obfuscated1],
                    "test_config",
                    output_dir=None,
                )

                # Verify JSON structure
                self.assertIn("timestamp", results)
                self.assertIn("config_name", results)
                self.assertIn("baseline", results)
                self.assertIn("obfuscated", results)
                self.assertIn("comparison", results)

    def test_comparison_calculation(self):
        """Test comparison metric calculations."""
        if BinaryMetrics is None:
            self.skipTest("BinaryMetrics not available")

        baseline = BinaryMetrics(
            file_size_bytes=1000,
            file_size_percent_increase=0.0,
            text_section_size=500,
            num_functions=10,
            num_basic_blocks=50,
            instruction_count=200,
            text_entropy=7.0,
            cyclomatic_complexity=5.0,
            stripped=False,
            pie_enabled=True,
        )

        obfuscated = BinaryMetrics(
            file_size_bytes=1100,
            file_size_percent_increase=10.0,
            text_section_size=550,
            num_functions=15,
            num_basic_blocks=80,
            instruction_count=300,
            text_entropy=7.5,
            cyclomatic_complexity=7.0,
            stripped=False,
            pie_enabled=True,
        )

        comparison = self.collector._compute_comparison(
            baseline,
            {"obfuscated": obfuscated}
        )

        self.assertIn("obfuscated", comparison)
        self.assertEqual(comparison["obfuscated"]["file_size_increase_bytes"], 100)
        self.assertEqual(comparison["obfuscated"]["file_size_increase_percent"], 10.0)
        self.assertEqual(comparison["obfuscated"]["function_count_delta"], 5)

    def test_save_results_creates_files(self):
        """Test that save_results creates output files."""
        output_dir = Path(self.temp_dir.name)

        results = {
            "timestamp": "2024-01-01T00:00:00",
            "config_name": "test",
            "baseline": {"file_size_bytes": 1000},
            "obfuscated": {},
            "comparison": {},
        }

        self.collector._save_results(results, output_dir)

        # Verify output files exist
        json_file = output_dir / "metrics.json"
        md_file = output_dir / "metrics.md"
        csv_file = output_dir / "metrics.csv"

        self.assertTrue(json_file.exists())
        self.assertTrue(md_file.exists())
        self.assertTrue(csv_file.exists())

        # Verify JSON is valid
        with open(json_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["config_name"], "test")


class TestMetricsOutputFormats(unittest.TestCase):
    """Test output format generation."""

    def setUp(self):
        """Set up test fixtures."""
        if MetricsCollector is None:
            self.skipTest("MetricsCollector not available")
        self.collector = MetricsCollector()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        output_dir = Path(self.temp_dir.name)

        results = {
            "timestamp": "2024-01-01T00:00:00",
            "config_name": "test_config",
            "baseline": {"file_size_bytes": 1000, "num_functions": 10},
            "obfuscated": {"obf1": {"file_size_bytes": 1100, "num_functions": 15}},
            "comparison": {"obf1": {"file_size_increase_percent": 10.0}},
        }

        self.collector._write_markdown_report(results, output_dir / "test.md")

        md_file = output_dir / "test.md"
        self.assertTrue(md_file.exists())

        with open(md_file, 'r') as f:
            content = f.read()
            self.assertIn("Obfuscation Metrics Report", content)
            self.assertIn("test_config", content)
            self.assertIn("Baseline Metrics", content)

    def test_csv_report_generation(self):
        """Test CSV report generation."""
        output_dir = Path(self.temp_dir.name)

        results = {
            "baseline": {"file_size_bytes": 1000, "num_functions": 10},
            "obfuscated": {"obf1": {"file_size_bytes": 1100, "num_functions": 15}},
            "comparison": {},
        }

        self.collector._write_csv_report(results, output_dir / "test.csv")

        csv_file = output_dir / "test.csv"
        self.assertTrue(csv_file.exists())

        with open(csv_file, 'r') as f:
            content = f.read()
            self.assertIn("Binary", content)
            self.assertIn("baseline", content)
            self.assertIn("obf1", content)


if __name__ == '__main__':
    unittest.main()
