"""OLLVM Obfuscation Test Suite Library"""

from .test_utils import (
    run_command,
    safe_run,
    file_hash,
    extract_strings,
    get_file_size,
    is_executable,
    get_arch,
    get_sections,
    compare_outputs
)

from .test_metrics import (
    compute_cfg_metrics,
    analyze_complexity,
    compute_coverage
)

from .test_report import ReportGenerator

from .test_functional import FunctionalTester

__all__ = [
    'run_command',
    'safe_run',
    'file_hash',
    'extract_strings',
    'get_file_size',
    'is_executable',
    'get_arch',
    'get_sections',
    'compare_outputs',
    'compute_cfg_metrics',
    'analyze_complexity',
    'compute_coverage',
    'ReportGenerator',
    'FunctionalTester'
]
