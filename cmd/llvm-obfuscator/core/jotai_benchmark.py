"""
Jotai Benchmark Integration Module

Integrates the Jotai benchmark collection (https://github.com/lac-dcc/jotai-benchmarks)
for testing obfuscation effectiveness on real-world C programs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum

from .utils import create_logger, run_command, ensure_directory, tool_exists
from .exceptions import ObfuscationError

# Forward reference for type hints to avoid circular imports
if TYPE_CHECKING:
    from .obfuscator import LLVMObfuscator
    from .config import ObfuscationConfig


class BenchmarkCategory(Enum):
    """Jotai benchmark categories."""
    ANGHALEAVES = "anghaLeaves"  # Functions without external calls
    ANGHAMATH = "anghaMath"  # Functions using math.h


@dataclass
class BenchmarkResult:
    """Results from running a benchmark through obfuscation."""
    benchmark_name: str
    category: str
    source_file: Path
    baseline_binary: Optional[Path] = None
    obfuscated_binary: Optional[Path] = None
    compilation_success: bool = False
    obfuscation_success: bool = False
    functional_test_passed: bool = False
    execution_time_baseline: Optional[float] = None
    execution_time_obfuscated: Optional[float] = None
    size_baseline: Optional[int] = None
    size_obfuscated: Optional[int] = None
    error_message: Optional[str] = None
    inputs_tested: int = 0
    inputs_passed: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert Path objects to strings
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result


class JotaiBenchmarkManager:
    """Manages Jotai benchmark collection for obfuscation testing."""

    JOTAI_REPO_URL = "https://github.com/lac-dcc/jotai-benchmarks.git"
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llvm-obfuscator" / "jotai-benchmarks"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        auto_download: bool = True
    ):
        """
        Initialize Jotai benchmark manager.

        Args:
            cache_dir: Directory to cache benchmarks (default: ~/.cache/llvm-obfuscator/jotai-benchmarks)
            auto_download: Automatically download benchmarks if not present
        """
        self.logger = create_logger(__name__)
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.benchmarks_dir = self.cache_dir / "benchmarks"
        self.repo_dir = self.cache_dir / "jotai-benchmarks"

        if auto_download and not self.repo_dir.exists():
            self.download_benchmarks()
    
    def _find_clang_binary(self) -> Path:
        """
        Find the best clang binary to use.
        
        Priority:
        1. Custom clang from plugins/linux-x86_64/ (LLVM 22) - relative to current working directory
        2. Custom clang from plugins/linux-x86_64/ (LLVM 22) - relative to this file
        3. Custom clang from /usr/local/llvm-obfuscator/bin/ (Docker)
        4. System clang (fallback)
        
        Returns:
            Path to clang binary
        """
        import os
        
        # Try plugins directory relative to current working directory (CI runs from cmd/llvm-obfuscator)
        cwd_plugins = Path(os.getcwd()) / "plugins" / "linux-x86_64"
        if (cwd_plugins / "clang").exists():
            clang_path = cwd_plugins / "clang"
            self.logger.debug(f"Using custom clang from plugins (cwd): {clang_path}")
            return clang_path
        
        # Try plugins directory relative to this file
        file_plugins_dir = Path(__file__).parent.parent.parent / "plugins" / "linux-x86_64"
        if (file_plugins_dir / "clang").exists():
            clang_path = file_plugins_dir / "clang"
            self.logger.debug(f"Using custom clang from plugins (file-relative): {clang_path}")
            return clang_path
        
        # Try Docker installation path
        docker_clang = Path("/usr/local/llvm-obfuscator/bin/clang")
        if docker_clang.exists():
            self.logger.debug(f"Using custom clang from Docker: {docker_clang}")
            return docker_clang
        
        # Fallback to system clang
        system_clang = shutil.which("clang")
        if system_clang:
            self.logger.debug(f"Using system clang: {system_clang}")
            return Path(system_clang)
        
        # Last resort - hardcoded common path
        self.logger.warning("Could not find clang, using /usr/bin/clang as fallback")
        return Path("/usr/bin/clang")
    
    def _find_clang_resource_dir(self, clang_binary: Path) -> Optional[str]:
        """
        Find the clang resource directory (for standard headers).
        
        Args:
            clang_binary: Path to clang binary
            
        Returns:
            Path to clang resource directory, or None if not found
        """
        try:
            # Try to get resource directory from clang
            result = subprocess.run(
                [str(clang_binary), "-print-resource-dir"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                resource_dir = result.stdout.strip()
                if resource_dir and os.path.exists(resource_dir):
                    return resource_dir
        except Exception:
            pass
        
        # Fallback: try common locations
        import os
        common_paths = [
            "/usr/lib/llvm-22/lib/clang/22/include",
            "/usr/lib/llvm-19/lib/clang/19/include",
            "/usr/lib/llvm-18/lib/clang/18/include",
            "/usr/lib/llvm-17/lib/clang/17/include",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return os.path.dirname(path)  # Return parent (the clang version dir)
        
        return None

    def download_benchmarks(self, force: bool = False) -> bool:
        """
        Download/clone Jotai benchmarks repository.

        Args:
            force: Force re-download even if already present

        Returns:
            True if successful, False otherwise
        """
        if self.repo_dir.exists() and not force:
            self.logger.info(f"Jotai benchmarks already cached at {self.repo_dir}")
            return True

        try:
            self.logger.info(f"Downloading Jotai benchmarks to {self.cache_dir}...")
            ensure_directory(self.cache_dir)

            # Clone repository
            cmd = [
                "git", "clone",
                "--depth", "1",
                self.JOTAI_REPO_URL,
                str(self.repo_dir)
            ]

            try:
                returncode, stdout, stderr = run_command(cmd)
                self.logger.info(f"Successfully downloaded benchmarks to {self.repo_dir}")
                return True
            except ObfuscationError as e:
                self.logger.error(f"Failed to clone Jotai repository: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error downloading benchmarks: {e}")
            return False

    def list_benchmarks(
        self,
        category: Optional[BenchmarkCategory] = None,
        limit: Optional[int] = None
    ) -> List[Path]:
        """
        List available benchmark files.

        Args:
            category: Filter by category (None = all)
            limit: Maximum number of benchmarks to return

        Returns:
            List of benchmark source file paths
        """
        if not self.repo_dir.exists():
            self.logger.warning("Jotai benchmarks not downloaded. Run download_benchmarks() first.")
            return []

        benchmarks = []
        search_dirs = []

        if category:
            search_dirs.append(self.repo_dir / "benchmarks" / category.value)
        else:
            search_dirs.extend([
                self.repo_dir / "benchmarks" / BenchmarkCategory.ANGHALEAVES.value,
                self.repo_dir / "benchmarks" / BenchmarkCategory.ANGHAMATH.value,
            ])

        for search_dir in search_dirs:
            if search_dir.exists():
                benchmarks.extend(search_dir.glob("*.c"))

        if limit:
            benchmarks = benchmarks[:limit]

        return sorted(benchmarks)

    def test_benchmark_compiles(self, benchmark_path: Path) -> bool:
        """
        Quick test to see if a benchmark compiles.
        
        Args:
            benchmark_path: Path to benchmark C file
            
        Returns:
            True if benchmark compiles, False otherwise
        """
        import tempfile
        
        clang_binary = self._find_clang_binary()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_binary = tmpdir_path / "test_compile"
            
            # Use same compilation flags as baseline compilation
            compile_flags = [
                "-g", "-O1",
                "-std=c11",
                "-Wno-everything",
                "-Wno-error",
                "-fno-strict-aliasing",
                "-fno-common",
                "-Wno-typedef-redefinition",
            ]
            
            # Add resource directory if available
            clang_resource_dir = self._find_clang_resource_dir(clang_binary)
            if clang_resource_dir:
                compile_flags.append(f"-isystem{clang_resource_dir}")
            
            # Add system includes
            for inc_path in ["/usr/include", "/usr/local/include"]:
                if os.path.exists(inc_path):
                    compile_flags.append(f"-isystem{inc_path}")
            
            compile_cmd = [
                str(clang_binary)
            ] + compile_flags + [
                str(benchmark_path),
                "-o", str(test_binary),
                "-lm",
            ]
            
            try:
                returncode, stdout, stderr = run_command(compile_cmd)
                return returncode == 0 and test_binary.exists()
            except ObfuscationError:
                return False
    
    def filter_compilable_benchmarks(
        self,
        benchmarks: List[Path],
        max_to_test: int = 100,
        min_compilable: int = 10
    ) -> List[Path]:
        """
        Filter benchmarks to find ones that compile.
        
        Args:
            benchmarks: List of benchmark paths to test
            max_to_test: Maximum number of benchmarks to test for compilability
            min_compilable: Minimum number of compilable benchmarks to find
            
        Returns:
            List of compilable benchmark paths
        """
        compilable = []
        tested = 0
        
        self.logger.info(f"Filtering benchmarks to find compilable ones (testing up to {max_to_test})...")
        
        for benchmark in benchmarks:
            if tested >= max_to_test:
                break
            if len(compilable) >= min_compilable:
                break
                
            tested += 1
            if tested % 10 == 0:
                self.logger.info(f"  Tested {tested}/{max_to_test}, found {len(compilable)} compilable...")
            
            if self.test_benchmark_compiles(benchmark):
                compilable.append(benchmark)
                self.logger.debug(f"  ✓ {benchmark.name} compiles")
        
        self.logger.info(f"Found {len(compilable)} compilable benchmarks out of {tested} tested")
        return compilable

    def get_benchmark_info(self, benchmark_path: Path) -> Dict:
        """
        Extract information about a benchmark.

        Args:
            benchmark_path: Path to benchmark C file

        Returns:
            Dictionary with benchmark metadata
        """
        info = {
            "name": benchmark_path.stem,
            "path": str(benchmark_path),
            "category": self._detect_category(benchmark_path),
            "size": benchmark_path.stat().st_size if benchmark_path.exists() else 0,
        }

        # Try to detect available inputs by compiling and running
        try:
            inputs = self._detect_inputs(benchmark_path)
            info["inputs"] = inputs
        except Exception as e:
            self.logger.debug(f"Could not detect inputs for {benchmark_path}: {e}")
            info["inputs"] = [0]  # Default to input 0

        return info

    def _detect_category(self, benchmark_path: Path) -> str:
        """Detect benchmark category from path."""
        path_str = str(benchmark_path)
        if "anghaLeaves" in path_str:
            return BenchmarkCategory.ANGHALEAVES.value
        elif "anghaMath" in path_str:
            return BenchmarkCategory.ANGHAMATH.value
        return "unknown"

    def _detect_inputs(self, benchmark_path: Path) -> List[int]:
        """
        Detect available inputs for a benchmark by running it.

        Returns:
            List of available input indices
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            binary = tmpdir_path / "test_binary"

            # Try to compile
            clang_binary = self._find_clang_binary()
            
            # Build compilation flags with standard includes - same as baseline
            compile_flags = [
                "-g", "-O1",
                "-std=c11",
                "-Wno-everything",
                "-Wno-error",
                "-fno-strict-aliasing",
                "-fno-common",
                "-Wno-typedef-redefinition",
            ]
            
            # Add resource directory if available
            clang_resource_dir = self._find_clang_resource_dir(clang_binary)
            if clang_resource_dir:
                compile_flags.append(f"-isystem{clang_resource_dir}")
            
            # Add system includes
            for inc_path in ["/usr/include", "/usr/local/include"]:
                if os.path.exists(inc_path):
                    compile_flags.append(f"-isystem{inc_path}")
            
            compile_cmd = [
                str(clang_binary)
            ] + compile_flags + [
                "-fsanitize=address,undefined,signed-integer-overflow",
                "-fno-sanitize-recover=all",
                str(benchmark_path),
                "-o", str(binary)
            ]

            try:
                returncode, stdout, stderr = run_command(compile_cmd)
            except ObfuscationError:
                return [0]  # Default fallback

            # Try running without args to see usage
            try:
                returncode, stdout, stderr = run_command([str(binary)])
                if returncode == 0 and "Usage" in stdout:
                    # Parse usage to find available inputs
                    inputs = [0]  # Always has input 0
                    # Try to detect more inputs (simplified - could be improved)
                    for i in range(1, 10):
                        try:
                            test_returncode, test_stdout, test_stderr = run_command([str(binary), str(i)])
                            if test_returncode == 0:
                                inputs.append(i)
                            else:
                                break
                        except ObfuscationError:
                            break
                    return inputs
            except ObfuscationError:
                pass

            return [0]  # Default to input 0

    def run_benchmark_test(
        self,
        benchmark_path: Path,
        obfuscator: "LLVMObfuscator",
        config: "ObfuscationConfig",
        output_dir: Path,
        inputs: Optional[List[int]] = None
    ) -> BenchmarkResult:
        """
        Run a single benchmark through obfuscation and test.

        Args:
            benchmark_path: Path to benchmark C source
            obfuscator: LLVMObfuscator instance
            config: Obfuscation configuration
            output_dir: Directory for output binaries
            inputs: List of input indices to test (None = auto-detect)

        Returns:
            BenchmarkResult with test results
        """
        result = BenchmarkResult(
            benchmark_name=benchmark_path.stem,
            category=self._detect_category(benchmark_path),
            source_file=benchmark_path
        )

        ensure_directory(output_dir)
        baseline_dir = output_dir / "baseline"
        obfuscated_dir = output_dir / "obfuscated"
        ensure_directory(baseline_dir)
        ensure_directory(obfuscated_dir)

        baseline_binary = baseline_dir / f"{benchmark_path.stem}_baseline"
        obfuscated_binary = obfuscated_dir / f"{benchmark_path.stem}_obfuscated"

        try:
            # 1. Compile baseline binary (normal compilation, no obfuscation)
            self.logger.info(f"Step 1: Compiling baseline binary for {benchmark_path.name}...")
            
            # Find the best clang to use - prefer custom LLVM 22 clang from plugins
            clang_binary = self._find_clang_binary()
            
            # Build compilation flags - Jotai benchmarks are extracted functions
            # Use very permissive flags to handle incomplete extracted code
            compile_flags = [
                "-g", "-O1",
                "-std=c11",
                "-Wno-everything",  # Ignore all warnings
                "-Wno-error",  # Don't treat warnings as errors
                "-fno-strict-aliasing",  # Allow type punning
                "-fno-common",  # Better for extracted code
                "-fno-builtin",  # Don't assume builtin functions
                "-Wno-typedef-redefinition",  # Allow typedef redefinition (treat as warning)
            ]
            
            # Add standard include paths if they exist
            clang_resource_dir = self._find_clang_resource_dir(clang_binary)
            if clang_resource_dir:
                compile_flags.append(f"-isystem{clang_resource_dir}")
            
            # Add system include paths
            system_includes = [
                "/usr/include",
                "/usr/local/include",
                "/usr/include/x86_64-linux-gnu",  # Linux-specific
            ]
            for inc_path in system_includes:
                if os.path.exists(inc_path):
                    compile_flags.append(f"-isystem{inc_path}")
            
            # Handle typedef redefinition conflicts by undefining system types
            problematic_types = ["ssize_t", "size_t", "off_t", "pid_t", "uid_t", "gid_t"]
            for ptype in problematic_types:
                compile_flags.append(f"-U{ptype}")
            
            compile_cmd = [
                str(clang_binary)
            ] + compile_flags + [
                str(benchmark_path),
                "-o", str(baseline_binary),
                "-lm",  # Link math library (some benchmarks use math functions)
            ]

            # Try compilation with multiple strategies if first attempt fails
            compilation_strategies = [
                ("standard", compile_flags),
                ("no-system-includes", [f for f in compile_flags if not f.startswith("-isystem")]),
                ("minimal", ["-g", "-O1", "-std=c11", "-Wno-everything", "-fno-builtin"]),
            ]
            
            last_stderr = ""
            for strategy_name, strategy_flags in compilation_strategies:
                strategy_cmd = [str(clang_binary)] + strategy_flags + [
                    str(benchmark_path),
                    "-o", str(baseline_binary),
                    "-lm",
                ]
                
                try:
                    process = subprocess.Popen(
                        strategy_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    stdout, stderr = process.communicate()
                    returncode = process.returncode
                    
                    if returncode == 0:
                        # Success!
                        if strategy_name != "standard":
                            self.logger.info(f"✓ Compiled with {strategy_name} strategy")
                        break
                    else:
                        # Failed, try next strategy
                        last_stderr = stderr
                        if strategy_name == compilation_strategies[-1][0]:
                            # Last strategy failed, log errors
                            self.logger.info(f"❌ Compilation failed for {benchmark_path.name} (tried all strategies)")
                            if stderr:
                                error_lines = [line for line in stderr.split('\n') if 'error:' in line]
                                if error_lines:
                                    error_summary = '\n'.join(error_lines[:3])
                                    self.logger.info(f"Final compilation errors:\n{error_summary}")
                                    result.error_message = f"Baseline compilation failed: {error_lines[0].strip()[:250]}"
                                else:
                                    stderr_lines = [l for l in stderr.strip().split('\n') if l.strip()][:2]
                                    if stderr_lines:
                                        result.error_message = f"Baseline compilation failed: {stderr_lines[0].strip()[:250]}"
                                        self.logger.info(f"Stderr: {stderr_lines[0][:200]}")
                                    else:
                                        result.error_message = f"Baseline compilation failed (exit {returncode})"
                            else:
                                result.error_message = f"Baseline compilation failed (exit {returncode}, no stderr)"
                            return result
                except Exception as e:
                    last_stderr = str(e)
                    if strategy_name == compilation_strategies[-1][0]:
                        result.error_message = f"Baseline compilation failed: {str(e)}"
                        self.logger.error(f"Compilation exception: {e}")
                        return result
                    continue
            
            # If we get here, compilation succeeded with one of the strategies

            result.baseline_binary = baseline_binary
            result.compilation_success = True
            result.size_baseline = baseline_binary.stat().st_size if baseline_binary.exists() else 0
            self.logger.info(f"✓ Baseline binary created: {baseline_binary.name} ({result.size_baseline} bytes)")

            # 2. Obfuscate the binary using LLVM obfuscator
            # The obfuscator takes source and produces obfuscated binary
            self.logger.info(f"Step 2: Obfuscating binary using LLVM obfuscator...")
            try:
                # Update config output directory
                config.output.directory = obfuscated_dir
                
                # Run obfuscation on source file (produces obfuscated binary)
                obf_result = obfuscator.obfuscate(
                    source_file=benchmark_path,
                    config=config
                )

                # Find the obfuscated binary (obfuscator creates it with the source file name)
                # The binary is typically named after the source file without extension
                expected_binary = obfuscated_dir / benchmark_path.stem
                if not expected_binary.exists():
                    # Try to find any binary file in the output directory
                    obfuscated_files = list(obfuscated_dir.glob(f"{benchmark_path.stem}*"))
                    # Filter for executable files
                    obfuscated_files = [f for f in obfuscated_files if f.is_file() and not f.suffix in ['.c', '.cpp', '.ll', '.json', '.md']]
                    if obfuscated_files:
                        result.obfuscated_binary = obfuscated_files[0]
                    else:
                        result.error_message = "Obfuscated binary not found"
                        return result
                else:
                    result.obfuscated_binary = expected_binary
                
                result.obfuscation_success = True
                result.size_obfuscated = result.obfuscated_binary.stat().st_size

            except Exception as e:
                result.error_message = f"Obfuscation failed: {str(e)}"
                return result

            # 3. Functional testing
            if inputs is None:
                inputs = self._detect_inputs(benchmark_path)

            result.inputs_tested = len(inputs)
            passed = 0

            for input_idx in inputs:
                try:
                    # Run baseline
                    try:
                        baseline_returncode, baseline_stdout, baseline_stderr = run_command(
                            [str(baseline_binary), str(input_idx)]
                        )
                    except ObfuscationError:
                        baseline_returncode = 1
                        baseline_stdout = ""
                        baseline_stderr = ""

                    # Run obfuscated
                    try:
                        obfuscated_returncode, obfuscated_stdout, obfuscated_stderr = run_command(
                            [str(result.obfuscated_binary), str(input_idx)]
                        )
                    except ObfuscationError:
                        obfuscated_returncode = 1
                        obfuscated_stdout = ""
                        obfuscated_stderr = ""

                    # Compare outputs
                    if (baseline_returncode == obfuscated_returncode and
                        baseline_stdout == obfuscated_stdout):
                        passed += 1
                    else:
                        self.logger.warning(
                            f"Input {input_idx} failed: "
                            f"baseline={baseline_returncode}, "
                            f"obfuscated={obfuscated_returncode}"
                        )

                except Exception as e:
                    self.logger.warning(f"Error testing input {input_idx}: {e}")

            result.inputs_passed = passed
            result.functional_test_passed = (passed == len(inputs))

        except Exception as e:
            result.error_message = f"Unexpected error: {str(e)}"
            self.logger.error(f"Error running benchmark {benchmark_path.name}: {e}")

        return result

    def run_benchmark_suite(
        self,
        obfuscator: "LLVMObfuscator",
        config: "ObfuscationConfig",
        output_dir: Path,
        category: Optional[BenchmarkCategory] = None,
        limit: Optional[int] = None,
        max_failures: int = 5,
        skip_compilation_errors: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run multiple benchmarks through obfuscation.

        Args:
            obfuscator: LLVMObfuscator instance
            config: Obfuscation configuration
            output_dir: Base directory for outputs
            category: Filter by category
            limit: Maximum number of benchmarks to test
            max_failures: Stop after this many consecutive failures
            skip_compilation_errors: Skip benchmarks that fail to compile (common with Jotai)

        Returns:
            List of BenchmarkResult objects
        """
        benchmarks = self.list_benchmarks(category=category, limit=limit)
        if not benchmarks:
            self.logger.warning("No benchmarks found")
            return []

        self.logger.info(f"Running {len(benchmarks)} benchmarks...")
        results = []
        consecutive_failures = 0
        skipped = 0

        for i, benchmark in enumerate(benchmarks, 1):
            self.logger.info(f"[{i}/{len(benchmarks)}] Testing {benchmark.name}...")

            benchmark_output = output_dir / benchmark.stem
            result = self.run_benchmark_test(
                benchmark_path=benchmark,
                obfuscator=obfuscator,
                config=config,
                output_dir=benchmark_output
            )

            results.append(result)

            # Skip compilation errors if requested (many Jotai benchmarks have compatibility issues)
            if skip_compilation_errors and not result.compilation_success:
                skipped += 1
                self.logger.info(f"⏭️  {benchmark.name}: SKIPPED (compilation error - common with Jotai)")
                consecutive_failures = 0  # Don't count compilation errors as failures
                continue

            if result.functional_test_passed:
                consecutive_failures = 0
                self.logger.info(f"✅ {benchmark.name}: PASSED")
            else:
                consecutive_failures += 1
                if result.error_message:
                    self.logger.warning(f"❌ {benchmark.name}: FAILED - {result.error_message}")
                else:
                    self.logger.warning(f"❌ {benchmark.name}: FAILED")

            if consecutive_failures >= max_failures:
                self.logger.warning(f"Stopping after {max_failures} consecutive failures")
                break

        if skipped > 0:
            self.logger.info(f"Skipped {skipped} benchmarks due to compilation errors (this is normal)")

        return results

    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_file: Path
    ) -> None:
        """
        Generate JSON report from benchmark results.

        Args:
            results: List of benchmark results
            output_file: Path to output JSON file
        """
        report = {
            "summary": {
                "total": len(results),
                "compilation_success": sum(1 for r in results if r.compilation_success),
                "obfuscation_success": sum(1 for r in results if r.obfuscation_success),
                "functional_pass": sum(1 for r in results if r.functional_test_passed),
            },
            "results": [r.to_dict() for r in results]
        }

        ensure_directory(output_file.parent)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to {output_file}")

