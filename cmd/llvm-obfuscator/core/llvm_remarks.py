"""LLVM Remarks integration for obfuscation validation and metrics.

This module uses LLVM's diagnostic framework to extract detailed information
about compilation and optimization passes applied during obfuscation.

References:
- https://llvm.org/docs/Remarks.html
- https://llvm.org/docs/RemarksPerfGuide.html
"""

from __future__ import annotations

import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .utils import create_logger

logger = logging.getLogger(__name__)


@dataclass
class Remark:
    """Single LLVM remark entry."""
    
    pass_name: str
    remark_name: str
    function: str
    args: Dict[str, Any] = field(default_factory=dict)
    hotness: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Remark":
        """Create Remark from YAML dict."""
        return cls(
            pass_name=data.get("Pass", "unknown"),
            remark_name=data.get("Name", "unknown"),
            function=data.get("Function", "unknown"),
            args={arg.get("Key", ""): arg.get("Value", "") for arg in data.get("Args", [])},
            hotness=data.get("Hotness")
        )


@dataclass
class RemarksAnalysis:
    """Analysis results from LLVM remarks."""
    
    total_remarks: int = 0
    optimizations_applied: int = 0
    inlining_decisions: int = 0
    loop_transformations: int = 0
    vectorization_attempts: int = 0
    remarks_by_pass: Dict[str, int] = field(default_factory=dict)
    remarks_by_function: Dict[str, int] = field(default_factory=dict)
    hottest_functions: List[tuple[str, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            "total_remarks": self.total_remarks,
            "optimizations_applied": self.optimizations_applied,
            "inlining_decisions": self.inlining_decisions,
            "loop_transformations": self.loop_transformations,
            "vectorization_attempts": self.vectorization_attempts,
            "remarks_by_pass": self.remarks_by_pass,
            "remarks_by_function": self.remarks_by_function,
            "hottest_functions": self.hottest_functions[:10],  # Top 10
        }


class LLVMRemarksParser:
    """Parser for LLVM remarks output."""
    
    def __init__(self):
        self.logger = create_logger(__name__)
    
    def parse_remarks_file(self, remarks_file: Path) -> List[Remark]:
        """
        Parse LLVM remarks YAML file.
        
        LLVM remarks use custom YAML tags (!Passed, !Missed, !Analysis, etc.)
        We need a custom loader to handle these.
        """
        if not remarks_file.exists():
            self.logger.warning(f"Remarks file not found: {remarks_file}")
            return []
        
        try:
            # Create custom YAML loader that handles LLVM remark tags
            class LLVMRemarkLoader(yaml.SafeLoader):
                """Custom YAML loader for LLVM remarks with custom tags."""
                pass
            
            # Register constructors for LLVM remark tags
            # These tags are just metadata, the actual data is in the mapping
            for tag in ['!Passed', '!Missed', '!Analysis', 
                       '!AnalysisFPCommute', '!AnalysisAliasing', '!Failure']:
                LLVMRemarkLoader.add_constructor(
                    tag,
                    lambda loader, node: loader.construct_mapping(node)
                )
            
            with open(remarks_file, 'r') as f:
                # LLVM remarks YAML is a stream of documents
                remarks = []
                for doc in yaml.load_all(f, Loader=LLVMRemarkLoader):
                    if doc and isinstance(doc, dict):
                        remarks.append(Remark.from_dict(doc))
                return remarks
        except Exception as e:
            self.logger.error(f"Failed to parse remarks file: {e}")
            return []
    
    def analyze_remarks(self, remarks: List[Remark]) -> RemarksAnalysis:
        """Analyze remarks to extract metrics."""
        analysis = RemarksAnalysis()
        analysis.total_remarks = len(remarks)
        
        function_hotness = {}
        
        for remark in remarks:
            # Count by pass
            analysis.remarks_by_pass[remark.pass_name] = \
                analysis.remarks_by_pass.get(remark.pass_name, 0) + 1
            
            # Count by function
            analysis.remarks_by_function[remark.function] = \
                analysis.remarks_by_function.get(remark.function, 0) + 1
            
            # Track hotness
            if remark.hotness:
                function_hotness[remark.function] = \
                    max(function_hotness.get(remark.function, 0), remark.hotness)
            
            # Categorize remarks
            if remark.pass_name == "inline":
                analysis.inlining_decisions += 1
            elif "loop" in remark.pass_name.lower():
                analysis.loop_transformations += 1
            elif "vector" in remark.pass_name.lower():
                analysis.vectorization_attempts += 1
            
            # Count optimizations (anything marked as 'applied', 'transformed', etc.)
            remark_lower = remark.remark_name.lower()
            if any(word in remark_lower for word in ['applied', 'transformed', 'optimized']):
                analysis.optimizations_applied += 1
        
        # Sort functions by hotness
        analysis.hottest_functions = sorted(
            function_hotness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return analysis


class RemarksCollector:
    """
    Collect LLVM remarks during compilation.
    
    Supports both serialized remarks (YAML/bitstream files) and
    remark diagnostics (printed to stderr).
    
    Reference: https://llvm.org/docs/Remarks.html
    """
    
    def __init__(self):
        self.logger = create_logger(__name__)
        self.parser = LLVMRemarksParser()
    
    def get_diagnostic_remarks_flags(self, pass_filter: str = ".*") -> List[str]:
        """
        Get flags for remark diagnostics (printed to stderr).
        
        Based on https://llvm.org/docs/Remarks.html#remark-diagnostics
        
        These flags enable optimization remarks as diagnostics:
        - -Rpass=<regex>              : Passed optimizations
        - -Rpass-missed=<regex>       : Missed optimizations
        - -Rpass-analysis=<regex>     : Analysis remarks
        
        Or for llc/opt:
        - -pass-remarks=<regex>
        - -pass-remarks-missed=<regex>
        - -pass-remarks-analysis=<regex>
        
        Args:
            pass_filter: Regex to filter which passes emit remarks
        
        Returns:
            List of diagnostic flags for clang
        """
        return [
            f"-Rpass={pass_filter}",  # Successful optimizations
            f"-Rpass-missed={pass_filter}",  # Missed optimizations
            f"-Rpass-analysis={pass_filter}",  # Analysis remarks
        ]
    
    def get_remarks_flags(
        self,
        output_file: Path,
        remark_filter: str = ".*",
        format: str = "yaml",
        with_hotness: bool = False
    ) -> List[str]:
        """
        Get compiler flags to enable remarks collection.
        
        Based on https://llvm.org/docs/Remarks.html
        
        For clang, use:
        - -fsave-optimization-record[=<format>]
        - -foptimization-record-file=<file>
        - -foptimization-record-passes=<regex>
        
        For llc/opt, use:
        - -pass-remarks-output=<file>
        - -pass-remarks-format=<format>
        - -pass-remarks-filter=<regex>
        
        Args:
            output_file: Path where remarks will be saved
            remark_filter: Regex filter for passes (default: all)
            format: Output format (yaml or bitstream)
            with_hotness: Include profile hotness data
        
        Returns:
            List of compiler flags for clang
        """
        flags = [
            f"-fsave-optimization-record={format}",  # LLVM docs: -fsave-optimization-record[=<format>]
            f"-foptimization-record-file={output_file}",  # LLVM docs: output file path
        ]
        
        # Add filter if not default (.*= all)
        if remark_filter != ".*":
            flags.append(f"-foptimization-record-passes={remark_filter}")
        
        # Add hotness if requested (requires PGO)
        # Note: This requires profile data from PGO
        if with_hotness:
            flags.append("-fdiagnostics-show-hotness")
        
        return flags
    
    def get_advanced_remarks_flags(
        self,
        output_file: Path,
        format: str = "yaml",
        pass_filter: str = ".*",
        with_hotness: bool = False,
        hotness_threshold: Optional[int] = None,
        emit_section: bool = False
    ) -> List[str]:
        """
        Get advanced remarks flags with all options.
        
        Based on https://llvm.org/docs/Remarks.html#serialized-remarks
        
        Content configuration options:
        - pass-remarks-filter: Only serialize specific passes
        - pass-remarks-with-hotness: Include profile count (requires PGO)
        - pass-remarks-hotness-threshold: Minimum profile count for emission
        - remarks-section: Emit remarks in object file section
        
        Args:
            output_file: Output file path
            format: yaml or bitstream
            pass_filter: Regex filter for passes
            with_hotness: Include profile hotness
            hotness_threshold: Minimum hotness to emit remark
            emit_section: Emit remarks metadata in object file section
        
        Returns:
            Complete list of flags
        """
        flags = [
            f"-fsave-optimization-record={format}",
            f"-foptimization-record-file={output_file}",
        ]
        
        if pass_filter != ".*":
            flags.append(f"-foptimization-record-passes={pass_filter}")
        
        if with_hotness:
            flags.append("-fdiagnostics-show-hotness")
        
        if hotness_threshold is not None:
            # Note: Clang may use different flag name
            # Check clang version for exact flag
            flags.append(f"-fdiagnostics-hotness-threshold={hotness_threshold}")
        
        if emit_section:
            # Emit metadata section in object file
            # Section name: __LLVM,__remarks (MachO)
            flags.append("-remarks-section")
        
        return flags
    
    def get_remarks_flags_for_llc_opt(
        self,
        output_file: Path,
        remark_filter: str = ".*",
        format: str = "yaml",
        with_hotness: bool = False
    ) -> List[str]:
        """
        Get flags for llc/opt (different from clang).
        
        Based on https://llvm.org/docs/Remarks.html#serialized-remarks
        
        Args:
            output_file: Path where remarks will be saved
            remark_filter: Regex filter for passes
            format: Output format (yaml or bitstream)
            with_hotness: Include profile hotness
        
        Returns:
            List of flags for llc/opt
        """
        flags = [
            f"-pass-remarks-output={output_file}",  # LLVM docs: output file
            f"-pass-remarks-format={format}",  # LLVM docs: yaml or bitstream
        ]
        
        if remark_filter != ".*":
            flags.append(f"-pass-remarks-filter={remark_filter}")
        
        if with_hotness:
            flags.append("-pass-remarks-with-hotness")
        
        return flags
    
    def collect_and_analyze(
        self,
        source_file: Path,
        output_binary: Path,
        compiler: str = "clang",
        extra_flags: Optional[List[str]] = None
    ) -> Optional[RemarksAnalysis]:
        """
        Compile with remarks collection and analyze results.
        
        Args:
            source_file: Source code to compile
            output_binary: Where to write binary
            compiler: Compiler to use
            extra_flags: Additional compiler flags
        
        Returns:
            Analysis of remarks, or None if compilation failed
        """
        remarks_file = output_binary.with_suffix(".opt.yaml")
        
        # Build compilation command
        cmd = [compiler, str(source_file), "-o", str(output_binary)]
        
        # Add remarks flags
        cmd.extend(self.get_remarks_flags(remarks_file))
        
        # Add extra flags
        if extra_flags:
            cmd.extend(extra_flags)
        
        self.logger.info(f"Compiling with remarks: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                self.logger.error(f"Compilation failed: {result.stderr}")
                return None
            
            # Parse and analyze remarks
            remarks = self.parser.parse_remarks_file(remarks_file)
            analysis = self.parser.analyze_remarks(remarks)
            
            self.logger.info(
                f"Collected {analysis.total_remarks} remarks, "
                f"{analysis.optimizations_applied} optimizations applied"
            )
            
            return analysis
            
        except subprocess.TimeoutExpired:
            self.logger.error("Compilation timed out")
            return None
        except Exception as e:
            self.logger.error(f"Failed to collect remarks: {e}")
            return None
    
    def compare_remarks(
        self,
        baseline_analysis: RemarksAnalysis,
        obfuscated_analysis: RemarksAnalysis
    ) -> Dict[str, Any]:
        """
        Compare baseline vs obfuscated compilation remarks.
        
        This helps validate that obfuscation is working:
        - More optimizations = more transformation opportunities
        - Different pass usage = obfuscation affecting optimization
        - Function hotness changes = control flow changes
        """
        comparison = {
            "baseline": baseline_analysis.to_dict(),
            "obfuscated": obfuscated_analysis.to_dict(),
            "delta": {
                "total_remarks": obfuscated_analysis.total_remarks - baseline_analysis.total_remarks,
                "optimizations_applied": obfuscated_analysis.optimizations_applied - baseline_analysis.optimizations_applied,
                "inlining_decisions": obfuscated_analysis.inlining_decisions - baseline_analysis.inlining_decisions,
            },
            "pass_differences": {}
        }
        
        # Compare pass usage
        all_passes = set(baseline_analysis.remarks_by_pass.keys()) | \
                    set(obfuscated_analysis.remarks_by_pass.keys())
        
        for pass_name in all_passes:
            baseline_count = baseline_analysis.remarks_by_pass.get(pass_name, 0)
            obf_count = obfuscated_analysis.remarks_by_pass.get(pass_name, 0)
            
            if baseline_count != obf_count:
                comparison["pass_differences"][pass_name] = {
                    "baseline": baseline_count,
                    "obfuscated": obf_count,
                    "delta": obf_count - baseline_count
                }
        
        return comparison


class ObfuscationRemarksValidator:
    """Validate obfuscation effectiveness using LLVM remarks."""
    
    def __init__(self):
        self.logger = create_logger(__name__)
        self.collector = RemarksCollector()
    
    def validate_obfuscation(
        self,
        source_file: Path,
        baseline_binary: Path,
        obfuscated_binary: Path,
        compiler_flags: List[str]
    ) -> Dict[str, Any]:
        """
        Validate obfuscation by comparing LLVM remarks.
        
        This provides insights into:
        1. What optimizations were applied
        2. How obfuscation affected compilation
        3. Function-level transformation metrics
        """
        self.logger.info("Validating obfuscation with LLVM remarks...")
        
        # Collect baseline remarks
        baseline_analysis = self.collector.collect_and_analyze(
            source_file,
            baseline_binary,
            extra_flags=["-O2"]  # Baseline with standard optimization
        )
        
        if not baseline_analysis:
            self.logger.warning("Failed to collect baseline remarks")
            return {"status": "failed", "error": "baseline collection failed"}
        
        # Collect obfuscated remarks
        obfuscated_analysis = self.collector.collect_and_analyze(
            source_file,
            obfuscated_binary,
            extra_flags=compiler_flags  # With obfuscation flags
        )
        
        if not obfuscated_analysis:
            self.logger.warning("Failed to collect obfuscated remarks")
            return {"status": "failed", "error": "obfuscated collection failed"}
        
        # Compare
        comparison = self.collector.compare_remarks(baseline_analysis, obfuscated_analysis)
        
        # Generate insights
        insights = self._generate_insights(comparison)
        
        return {
            "status": "success",
            "comparison": comparison,
            "insights": insights
        }
    
    def _generate_insights(self, comparison: Dict) -> List[str]:
        """Generate human-readable insights from comparison."""
        insights = []
        
        delta = comparison["delta"]
        
        if delta["total_remarks"] > 0:
            insights.append(
                f"Obfuscation generated {delta['total_remarks']} additional remarks, "
                "indicating more compilation work"
            )
        
        if delta["optimizations_applied"] != 0:
            insights.append(
                f"Optimization count changed by {delta['optimizations_applied']}, "
                "suggesting different optimization opportunities"
            )
        
        if delta["inlining_decisions"] > 0:
            insights.append(
                f"Inlining increased by {delta['inlining_decisions']}, "
                "which can reduce visible function symbols"
            )
        elif delta["inlining_decisions"] < 0:
            insights.append(
                f"Inlining decreased by {abs(delta['inlining_decisions'])}, "
                "which may preserve obfuscated function boundaries"
            )
        
        # Analyze pass differences
        pass_diffs = comparison.get("pass_differences", {})
        if pass_diffs:
            significant_changes = [
                f"{name}: {diff['delta']:+d}"
                for name, diff in pass_diffs.items()
                if abs(diff['delta']) > 5
            ]
            if significant_changes:
                insights.append(
                    f"Significant pass usage changes: {', '.join(significant_changes)}"
                )
        
        return insights


# Convenience function for integration
def analyze_obfuscation_with_remarks(
    source_file: Path,
    baseline_binary: Path,
    obfuscated_binary: Path,
    compiler_flags: List[str]
) -> Dict[str, Any]:
    """
    One-shot function to validate obfuscation using LLVM remarks.
    
    Usage:
        result = analyze_obfuscation_with_remarks(
            source_file=Path("test.c"),
            baseline_binary=Path("test_baseline"),
            obfuscated_binary=Path("test_obfuscated"),
            compiler_flags=["-O3", "-flto"]
        )
        
        print(result["insights"])
    """
    validator = ObfuscationRemarksValidator()
    return validator.validate_obfuscation(
        source_file,
        baseline_binary,
        obfuscated_binary,
        compiler_flags
    )

