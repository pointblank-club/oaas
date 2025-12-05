"""Core package for LLVM obfuscator."""

from . import compat as _compat  # noqa: F401  # ensure compatibility patches load
from . import report_converter
from .analyzer import analyze_binary
from .comparer import compare_binaries
from .config import (
    AnalyzeConfig,
    AdvancedConfiguration,
    Architecture,
    CompareConfig,
    ObfuscationConfig,
    ObfuscationLevel,
    OutputConfiguration,
    PassConfiguration,
    Platform,
    RemarksConfiguration,
    UPXConfiguration,
)
from .obfuscator import LLVMObfuscator
from .reporter import ObfuscationReport
from .upx_packer import UPXPacker
from .jotai_benchmark import JotaiBenchmarkManager, BenchmarkCategory, BenchmarkResult

__all__ = [
    "LLVMObfuscator",
    "ObfuscationConfig",
    "PassConfiguration",
    "AdvancedConfiguration",
    "Architecture",
    "OutputConfiguration",
    "Platform",
    "ObfuscationLevel",
    "AnalyzeConfig",
    "CompareConfig",
    "ObfuscationReport",
    "RemarksConfiguration",
    "UPXConfiguration",
    "UPXPacker",
    "JotaiBenchmarkManager",
    "BenchmarkCategory",
    "BenchmarkResult",
    "analyze_binary",
    "compare_binaries",
]
