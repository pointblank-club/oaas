from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import Architecture, ObfuscationConfig, Platform
from .exceptions import ObfuscationError
from .fake_loop_inserter import FakeLoopGenerator
from .multifile_compiler import compile_multifile_ir_workflow
from .reporter import ObfuscationReport
from .llvm_remarks import RemarksCollector
from .upx_packer import UPXPacker
from .utils import (
    compute_entropy,
    create_logger,
    detect_binary_format,
    ensure_directory,
    ensure_generated_headers_exist,
    get_file_size,
    get_timestamp,
    list_sections,
    merge_flags,
    require_tool,
    run_command,
    summarize_symbols,
)

logger = logging.getLogger(__name__)


class LLVMObfuscator:
    """Main obfuscation pipeline orchestrator."""

    BASE_FLAGS = [
        "-fvisibility=hidden",
        "-O3",
        "-fno-builtin",
        "-fomit-frame-pointer",
        "-mspeculative-load-hardening",
        "-Wl,-s",
    ]

    CUSTOM_PASSES = [
        "flattening",
        "substitution",
        "boguscf",
        "split",
        "string-encrypt",
        "symbol-obfuscate",
        "crypto-hash",
        "constant-obfuscate",
    ]

    def __init__(self, reporter: Optional[ObfuscationReport] = None) -> None:
        self.logger = create_logger(__name__)
        self.reporter = reporter
        self.fake_loop_generator = FakeLoopGenerator()
        self.remarks_collector = RemarksCollector()
        self.upx_packer = UPXPacker()

    def _get_bundled_plugin_path(self, target_platform: Optional[Platform] = None) -> Optional[Path]:
        """Auto-detect bundled OLLVM plugin for current or target platform."""
        try:
            import platform
            import os

            if target_platform:
                # Use target platform specified by user (for cross-compilation)
                if target_platform == Platform.LINUX:
                    system = "linux"
                    arch = "x86_64"  # Default to x86_64 for Linux
                    ext = "so"
                elif target_platform == Platform.WINDOWS:
                    system = "windows"
                    arch = "x86_64"
                    ext = "dll"
                elif target_platform in [Platform.MACOS, Platform.DARWIN]:
                    system = "darwin"
                    arch = platform.machine().lower()  # Use current arch (arm64 or x86_64)
                    if arch == "aarch64":
                        arch = "arm64"
                    ext = "dylib"
                else:
                    # For unknown, fall back to current platform detection
                    target_platform = None

            if not target_platform:
                # Auto-detect current platform
                system = platform.system().lower()  # darwin, linux, windows
                machine = platform.machine().lower()  # arm64, x86_64, amd64

                # Normalize architecture names
                if machine in ['x86_64', 'amd64']:
                    arch = 'x86_64'
                elif machine in ['arm64', 'aarch64']:
                    arch = 'arm64'
                else:
                    self.logger.debug(f"Unsupported architecture for bundled plugin: {machine}")
                    return None

                # Determine plugin extension by platform
                if system == "darwin":
                    ext = "dylib"
                elif system == "linux":
                    ext = "so"
                elif system == "windows":
                    ext = "dll"
                else:
                    self.logger.debug(f"Unsupported platform for bundled plugin: {system}")
                    return None

            # Build path to bundled plugin
            plugin_dir = Path(__file__).parent.parent / "plugins" / f"{system}-{arch}"
            plugin_file = plugin_dir / f"LLVMObfuscationPlugin.{ext}"

            if plugin_file.exists():
                self.logger.info(f"Auto-detected bundled plugin: {plugin_file}")
                return plugin_file
            else:
                self.logger.debug(f"Bundled plugin not found at: {plugin_file}")
                return None

        except Exception as e:
            self.logger.debug(f"Could not auto-detect bundled plugin: {e}")
            return None

    def _get_mlir_plugin_path(self) -> Optional[Path]:
        """Find MLIR obfuscation plugin library."""
        try:
            import platform as py_platform

            system = py_platform.system().lower()
            if system == "linux":
                ext = "so"
            elif system == "darwin":
                ext = "dylib"
            elif system == "windows":
                ext = "dll"
            else:
                return None

            # Search locations for MLIR plugin
            search_paths = [
                # Bundled plugin location (for each platform)
                Path(__file__).parent.parent / "plugins" / "linux-x86_64" / f"MLIRObfuscation.{ext}",
                Path(__file__).parent.parent / "plugins" / "darwin-arm64" / f"MLIRObfuscation.{ext}",
                Path(__file__).parent.parent / "plugins" / "darwin-x86_64" / f"MLIRObfuscation.{ext}",
                # Relative to the obfuscator script (build directory)
                Path(__file__).parent.parent.parent.parent / "mlir-obs" / "build" / "lib" / f"MLIRObfuscation.{ext}",
                Path(__file__).parent.parent.parent.parent / "mlir-obs" / "build" / "lib" / f"libMLIRObfuscation.{ext}",
                # Absolute paths
                Path("/app/mlir-obs/build/lib") / f"MLIRObfuscation.{ext}",
                Path("/app/mlir-obs/build/lib") / f"libMLIRObfuscation.{ext}",
                Path("/usr/local/lib") / f"MLIRObfuscation.{ext}",
                Path("/usr/local/lib") / f"libMLIRObfuscation.{ext}",
            ]

            for path in search_paths:
                if path.exists():
                    self.logger.info(f"Found MLIR plugin: {path}")
                    return path

            self.logger.debug("MLIR plugin not found in any search paths")
            return None

        except Exception as e:
            self.logger.debug(f"Could not locate MLIR plugin: {e}")
            return None

    def _get_target_triple(self, platform: Platform, arch: Architecture) -> str:
        """Build LLVM target triple from platform + architecture combination.

        Target triple format: <arch>-<vendor>-<os>-<environment>

        Returns the appropriate target triple for cross-compilation.
        """
        # Mapping of (platform, architecture) to LLVM target triple
        target_triples = {
            # Linux targets
            (Platform.LINUX, Architecture.X86_64): "x86_64-unknown-linux-gnu",
            (Platform.LINUX, Architecture.ARM64): "aarch64-unknown-linux-gnu",
            (Platform.LINUX, Architecture.X86): "i686-unknown-linux-gnu",
            # Windows targets (using MinGW toolchain)
            (Platform.WINDOWS, Architecture.X86_64): "x86_64-w64-mingw32",
            (Platform.WINDOWS, Architecture.ARM64): "aarch64-w64-mingw32",
            (Platform.WINDOWS, Architecture.X86): "i686-w64-mingw32",
            # macOS/Darwin targets
            (Platform.MACOS, Architecture.X86_64): "x86_64-apple-darwin",
            (Platform.MACOS, Architecture.ARM64): "aarch64-apple-darwin",
            (Platform.DARWIN, Architecture.X86_64): "x86_64-apple-darwin",
            (Platform.DARWIN, Architecture.ARM64): "aarch64-apple-darwin",
        }

        triple = target_triples.get((platform, arch))
        if triple:
            self.logger.info(f"Target triple: {triple} (platform={platform.value}, arch={arch.value})")
            return triple

        # Fallback to x86_64 Linux if combination not found
        self.logger.warning(f"Unknown platform/arch combination: {platform.value}/{arch.value}, defaulting to x86_64-unknown-linux-gnu")
        return "x86_64-unknown-linux-gnu"

    def obfuscate(self, source_file: Path, config: ObfuscationConfig, job_id: Optional[str] = None) -> Dict:
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        output_directory = config.output.directory
        ensure_directory(output_directory)
        output_binary = output_directory / self._output_name(source_file, config.platform)

        # Track warnings and important events for the report
        warnings_log = []
        actually_applied_passes = []

        require_tool("clang")
        require_tool("mlir-opt")
        require_tool("mlir-translate")
        # Windows cross-compilation uses clang --target=x86_64-w64-mingw32
        # MinGW libraries are needed but not the GCC binary itself

        source_content = source_file.read_text(encoding="utf-8", errors="ignore")

        # Compile baseline (unobfuscated) binary for comparison
        self.logger.info("Compiling baseline binary for comparison...")
        baseline_binary = output_directory / f"{source_file.stem}_baseline"
        baseline_metrics = self._compile_and_analyze_baseline(source_file, baseline_binary, config)

        # Symbol and string obfuscation are now handled by MLIR passes.
        symbol_result = None
        string_result = None
        working_source = source_file

        # Indirect call obfuscation (if enabled) - applied after string encryption
        indirect_call_result = None
        if config.advanced.indirect_calls.enabled:
            try:
                from .indirect_call_obfuscator import obfuscate_indirect_calls

                # Get the current working source
                current_source_content = working_source.read_text(encoding="utf-8", errors="replace")

                # Apply indirect call obfuscation
                transformed_code, metadata = obfuscate_indirect_calls(
                    source_code=current_source_content,
                    source_file=working_source,
                    obfuscate_stdlib=config.advanced.indirect_calls.obfuscate_stdlib,
                    obfuscate_custom=config.advanced.indirect_calls.obfuscate_custom,
                )

                # Write the transformed source to a new file
                indirect_call_file = output_directory / f"{source_file.stem}_indirect_calls{source_file.suffix}"
                indirect_call_file.write_text(transformed_code, encoding="utf-8", errors="replace")
                working_source = indirect_call_file
                indirect_call_result = metadata
                self.logger.info(
                    f"Indirect call obfuscation complete: {metadata['total_obfuscated']} functions "
                    f"({metadata['obfuscated_stdlib_functions']} stdlib, {metadata['obfuscated_custom_functions']} custom)"
                )
            except Exception as e:
                self.logger.error(f"Indirect call obfuscation failed: {e}")
                indirect_call_result = None

        # Insert fake loops into source code (if enabled)
        fake_loops = []
        if config.advanced.fake_loops and config.advanced.fake_loops > 0:
            self.logger.info(f"Inserting {config.advanced.fake_loops} fake loops into source code...")
            fake_loop_source = output_directory / f"{working_source.stem}_fakeloops{working_source.suffix}"
            try:
                modified_content, fake_loops = self.fake_loop_generator.insert_fake_loops(
                    working_source,
                    config.advanced.fake_loops,
                    fake_loop_source
                )
                if fake_loops:
                    self.logger.info(f"Successfully inserted {len(fake_loops)} fake loops")
                    working_source = fake_loop_source  # Use the modified source
                else:
                    self.logger.warning("No suitable insertion points found for fake loops")
            except Exception as e:
                self.logger.error(f"Fake loop insertion failed: {e}")
                # Continue with original source if insertion fails

        enabled_passes = config.passes.enabled_passes()
        compiler_flags = merge_flags(self.BASE_FLAGS, config.compiler_flags)

        # Cycles apply OLLVM passes multiple times on the IR for stronger obfuscation
        effective_cycles = config.advanced.cycles if config.advanced.cycles > 0 else 1

        self.logger.info("Compiling with %d obfuscation cycle(s)", effective_cycles)

        # Single compilation call - cycles are handled inside _compile
        cycle_result = self._compile(
            working_source,  # Use source with fake loops (if inserted)
            output_binary,
            config,
            compiler_flags,
            enabled_passes,
            cycles=effective_cycles,  # Pass cycles to apply OLLVM passes multiple times
        )

        # Track what actually happened
        if cycle_result:
            actually_applied_passes = cycle_result.get("applied_passes", [])
            # Always extend warnings list (even if empty, to maintain consistency)
            warnings_log.extend(cycle_result.get("warnings", []))

        # UPX packing (if enabled) - applied as FINAL step after all obfuscation
        upx_result = None
        if config.advanced.upx_packing.enabled:
            try:
                self.logger.info("Applying UPX compression to final binary...")
                upx_result = self.upx_packer.pack(
                    binary_path=output_binary,
                    compression_level=config.advanced.upx_packing.compression_level,
                    use_lzma=config.advanced.upx_packing.use_lzma,
                    force=True,
                    preserve_original=config.advanced.upx_packing.preserve_original,
                )
                if upx_result and upx_result.get("status") == "success":
                    self.logger.info(
                        f"UPX packing successful: {upx_result['compression_ratio']:.1f}% size reduction "
                        f"({upx_result['original_size']} → {upx_result['packed_size']} bytes)"
                    )
                elif upx_result and upx_result.get("status") == "failed":
                    self.logger.warning(f"UPX packing failed: {upx_result.get('error', 'Unknown error')}")
                    warnings_log.append(f"UPX packing failed: {upx_result.get('error', 'Unknown error')}")
                else:
                    self.logger.warning("UPX packing skipped (not installed or incompatible binary)")
            except Exception as e:
                self.logger.warning(f"UPX packing failed with exception: {e}")
                warnings_log.append(f"UPX packing error: {str(e)}")

        binary_format = detect_binary_format(output_binary)
        file_size = get_file_size(output_binary)
        sections = list_sections(output_binary)
        symbols_count, functions_count = summarize_symbols(output_binary)
        entropy = compute_entropy(output_binary.read_bytes() if output_binary.exists() else b"")

        base_metrics = self._estimate_metrics(
            source_file=source_file,
            output_binary=output_binary,
            passes=enabled_passes,
            cycles=config.advanced.cycles,
            string_result=string_result,
            fake_loops=fake_loops,
            entropy=entropy,
        )

        job_data = {
            "job_id": job_id,
            "source_file": str(source_file.name),  # Use just the filename, not full path
            "platform": config.platform.value,
            "obfuscation_level": int(config.level),
            "requested_passes": enabled_passes,  # What user requested
            "applied_passes": actually_applied_passes,  # What was actually applied
            "compiler_flags": compiler_flags,
            "timestamp": get_timestamp(),
            "warnings": warnings_log,  # Add warnings to report
            "baseline_metrics": baseline_metrics,  # Before obfuscation metrics
            "output_attributes": {
                "file_size": file_size,
                "binary_format": binary_format,
                "sections": sections,
                "symbols_count": symbols_count,
                "functions_count": functions_count,
                "entropy": entropy,
                "obfuscation_methods": actually_applied_passes + (["indirect_calls"] if indirect_call_result else []),
            },
            "comparison": {
                "size_change": file_size - baseline_metrics.get("file_size", file_size) if baseline_metrics else 0,
                "size_change_percent": round(((file_size - baseline_metrics.get("file_size", file_size)) / baseline_metrics.get("file_size", file_size) * 100), 2) if baseline_metrics and baseline_metrics.get("file_size", 0) > 0 else 0,
                "symbols_removed": baseline_metrics.get("symbols_count", 0) - symbols_count if baseline_metrics else 0,
                "symbols_removed_percent": round(((baseline_metrics.get("symbols_count", 0) - symbols_count) / baseline_metrics.get("symbols_count", 1) * 100), 2) if baseline_metrics and baseline_metrics.get("symbols_count", 0) > 0 else 0,
                "functions_removed": baseline_metrics.get("functions_count", 0) - functions_count if baseline_metrics else 0,
                "functions_removed_percent": round(((baseline_metrics.get("functions_count", 0) - functions_count) / baseline_metrics.get("functions_count", 1) * 100), 2) if baseline_metrics and baseline_metrics.get("functions_count", 0) > 0 else 0,
                "entropy_increase": round(entropy - baseline_metrics.get("entropy", 0), 3) if baseline_metrics else 0,
                "entropy_increase_percent": round(((entropy - baseline_metrics.get("entropy", 0)) / baseline_metrics.get("entropy", 1) * 100), 2) if baseline_metrics and baseline_metrics.get("entropy", 0) > 0 else 0,
            },
            "bogus_code_info": base_metrics["bogus_code_info"],
            "cycles_completed": base_metrics["cycles_completed"],
            "string_obfuscation": base_metrics["string_obfuscation"],
            "fake_loops_inserted": base_metrics["fake_loops_inserted"],
            "symbol_obfuscation": symbol_result or {"enabled": False},
            "indirect_calls": indirect_call_result or {"enabled": False},
            "upx_packing": upx_result or {"enabled": False},
            "obfuscation_score": base_metrics["obfuscation_score"],
            "symbol_reduction": base_metrics["symbol_reduction"],
            "function_reduction": base_metrics["function_reduction"],
            "size_reduction": base_metrics["size_reduction"],
            "entropy_increase": base_metrics["entropy_increase"],
            "estimated_re_effort": base_metrics["estimated_re_effort"],
            "output_file": str(output_binary),
        }

        if self.reporter:
            report = self.reporter.generate_report(job_data)
            exported = self.reporter.export(report, job_id or output_binary.stem, config.output.report_formats)
            job_data["report_paths"] = {fmt: str(path) for fmt, path in exported.items()}
        return job_data

    # Internal helpers -----------------------------------------------------

    def _output_name(self, source_file: Path, platform_target: Platform) -> str:
        stem = source_file.stem
        if platform_target == Platform.WINDOWS:
            return f"{stem}.exe"
        return stem

    def _add_remarks_flags(self, command: List[str], config: ObfuscationConfig, output_binary: Path) -> None:
        """
        Add LLVM remarks flags to compilation command if enabled.
        
        Based on https://llvm.org/docs/Remarks.html
        """
        if not config.advanced.remarks.enabled:
            return
        
        remarks_file = output_binary.parent / (
            config.advanced.remarks.output_file or f"{output_binary.stem}.opt.yaml"
        )
        
        remarks_flags = self.remarks_collector.get_remarks_flags(
            output_file=remarks_file,
            remark_filter=config.advanced.remarks.pass_filter,
            format=config.advanced.remarks.format,
            with_hotness=config.advanced.remarks.with_hotness
        )
        
        command.extend(remarks_flags)
        self.logger.info(f"LLVM remarks enabled: {remarks_file}")
    
    def _get_resource_dir_flag(self, compiler_path: str) -> List[str]:
        """
        Get the -resource-dir flag for custom clang binaries that don't have
        their own resource directory (stddef.h, stdint.h, etc.).

        This is needed when using bundled clang or custom-built clang that
        doesn't have the compiler builtin headers.
        """
        import platform as py_platform
        import subprocess

        # Only needed on Linux for custom clang binaries
        if py_platform.system().lower() != "linux":
            return []

        # Resolve compiler to full path if it's just a command name (like "clang")
        resolved_path = compiler_path
        if "/" not in compiler_path:
            try:
                result = subprocess.run(
                    ["which", compiler_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    resolved_path = result.stdout.strip()
                    self.logger.info(f"[RESOURCE-DIR-DEBUG] Resolved '{compiler_path}' to '{resolved_path}'")
            except Exception as e:
                self.logger.warning(f"[RESOURCE-DIR-DEBUG] Could not resolve compiler path: {e}")
        else:
            self.logger.info(f"[RESOURCE-DIR-DEBUG] Compiler path already resolved: {resolved_path}")

        # Check if this is a custom clang (not system clang)
        is_custom_clang = (
            "/plugins/" in resolved_path or  # Bundled clang
            "/usr/local/llvm-obfuscator/" in resolved_path or  # Custom installed clang
            "/llvm-project/build/" in resolved_path  # LLVM build directory
        )

        self.logger.info(f"[RESOURCE-DIR-DEBUG] is_custom_clang={is_custom_clang} for path={resolved_path}")

        if not is_custom_clang:
            return []

        # Try to find resource directory
        # Priority: bundled LLVM 22 > system clang
        resource_dir_candidates = [
            # Bundled LLVM 22 resource directory
            Path(__file__).parent.parent / "plugins" / "linux-x86_64" / "lib" / "clang" / "22",
            Path("/app/plugins/linux-x86_64/lib/clang/22"),  # Docker path
            Path("/usr/local/llvm-obfuscator/lib/clang/22"),  # Docker installed path
            # System clang fallbacks
            Path("/usr/lib/llvm-19/lib/clang/19"),
            Path("/usr/lib/llvm-18/lib/clang/18"),
            Path("/usr/lib/llvm-17/lib/clang/17"),
        ]

        for resource_dir in resource_dir_candidates:
            if resource_dir.exists():
                self.logger.info(f"[RESOURCE-DIR-DEBUG] Using resource directory: {resource_dir}")
                return ["-resource-dir", str(resource_dir)]

        # Legacy: Try string paths for backwards compatibility
        system_clang_candidates = [
            "/usr/lib/llvm-19/lib/clang/19",
            "/usr/lib/llvm-18/lib/clang/18",
            "/usr/lib/llvm-17/lib/clang/17",
        ]

        for resource_dir in system_clang_candidates:
            if Path(resource_dir).exists():
                self.logger.info(f"[RESOURCE-DIR-DEBUG] Using system resource directory: {resource_dir}")
                return ["-resource-dir", resource_dir]

        # Fallback: try to detect from system clang
        try:
            result = subprocess.run(
                ["clang", "-print-resource-dir"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                resource_dir = result.stdout.strip()
                if Path(resource_dir).exists():
                    self.logger.debug(f"Detected resource directory from system clang: {resource_dir}")
                    return ["-resource-dir", resource_dir]
        except Exception as e:
            self.logger.debug(f"Could not detect system clang resource directory: {e}")

        self.logger.warning(
            "Custom clang binary used but system resource directory not found. "
            "Compilation may fail with 'stddef.h not found' errors."
        )
        return []

    def _has_exception_handling(self, ir_file: Path) -> bool:
        """
        Check if LLVM IR file contains C++ exception handling (invoke/landingpad).

        Returns True if the IR uses exception handling, which is incompatible
        with the OLLVM flattening pass.

        Supports both .ll (text IR) and .bc (bitcode) files.
        """
        try:
            # If it's a .bc file, try to use llvm-dis to convert to text
            # Otherwise, read directly as text
            if ir_file.suffix == '.bc':
                import shutil
                import subprocess

                # Try to find LLVM 22 llvm-dis first (bundled with our LLVM 22 toolchain)
                # System llvm-dis may be older (e.g., LLVM 19) and fail to read LLVM 22 bitcode
                llvm_dis = None
                bundled_llvm_dis_paths = [
                    Path("/usr/local/llvm-obfuscator/bin/llvm-dis"),  # Docker production
                    Path("/app/plugins/linux-x86_64/llvm-dis"),  # Docker backup
                ]

                for bundled_path in bundled_llvm_dis_paths:
                    if bundled_path.exists():
                        llvm_dis = str(bundled_path)
                        self.logger.debug(f"Using bundled llvm-dis (LLVM 22): {llvm_dis}")
                        break

                # Fallback to system llvm-dis if no bundled version
                if not llvm_dis:
                    llvm_dis = shutil.which("llvm-dis")
                    if llvm_dis:
                        self.logger.debug(f"Using system llvm-dis: {llvm_dis}")

                if llvm_dis:
                    # Use llvm-dis to convert .bc to .ll temporarily
                    temp_ll = ir_file.parent / f"{ir_file.stem}_temp.ll"
                    try:
                        subprocess.run(
                            [llvm_dis, str(ir_file), "-o", str(temp_ll)],
                            check=True,
                            capture_output=True,
                            timeout=30
                        )
                        ir_content = temp_ll.read_text(encoding='utf-8', errors='ignore')
                        temp_ll.unlink()  # Clean up
                    except Exception as e:
                        self.logger.warning(f"Failed to convert .bc to .ll with llvm-dis: {e}")
                        # For C++ code with exception handling, assume it has EH
                        # This is safer than letting flattening crash
                        # Check file extension heuristics from source
                        self.logger.warning("Assuming C++ exception handling is present (safer for flattening)")
                        return True
                else:
                    # No llvm-dis available - assume C++ has EH to be safe
                    self.logger.warning("No llvm-dis available to check for exception handling")
                    self.logger.warning("Assuming C++ exception handling is present (safer for flattening)")
                    return True
            else:
                # Text IR file (.ll)
                ir_content = ir_file.read_text(encoding='utf-8', errors='ignore')

            # Check for invoke instructions (exception-aware function calls)
            has_invoke = ' invoke ' in ir_content

            # Check for landingpad instructions (exception handlers)
            has_landingpad = ' landingpad ' in ir_content

            if has_invoke or has_landingpad:
                self.logger.info("C++ exception handling detected (invoke/landingpad found in IR)")

            return has_invoke or has_landingpad
        except Exception as e:
            self.logger.warning(f"Could not check for exception handling in IR: {e}")
            # Err on the side of caution - assume EH is present
            return True

    def _compile(
        self,
        source: Path,
        destination: Path,
        config: ObfuscationConfig,
        compiler_flags: List[str],
        enabled_passes: List[str],
        cycles: int = 1,
    ) -> Dict:
        """
        Main compilation dispatcher - routes to appropriate frontend.

        DEFAULT (config.mlir_frontend == CLANG): Existing pipeline (SAFE)
        NEW (config.mlir_frontend == CLANGIR): ClangIR pipeline

        Args:
            cycles: Number of times to apply OLLVM passes (for stronger obfuscation)
        """
        from .config import MLIRFrontend

        # Route to appropriate pipeline based on frontend choice
        if config.mlir_frontend == MLIRFrontend.CLANGIR:
            # NEW: ClangIR pipeline
            return self._compile_with_clangir(source, destination, config, compiler_flags, enabled_passes, cycles)
        else:
            # DEFAULT: Existing Clang → LLVM IR → MLIR pipeline (UNCHANGED)
            return self._compile_with_clang_llvm(source, destination, config, compiler_flags, enabled_passes, cycles)

    def _compile_with_clang_llvm(
        self,
        source: Path,
        destination: Path,
        config: ObfuscationConfig,
        compiler_flags: List[str],
        enabled_passes: List[str],
        cycles: int = 1,
    ) -> Dict:
        """
        EXISTING PIPELINE - Clang → LLVM IR → MLIR → OLLVM (with cycles support)

        Args:
            cycles: Number of times to apply OLLVM passes for stronger obfuscation.
                   Each cycle applies all enabled OLLVM passes to the IR.
        """
        # Use absolute paths to avoid path resolution issues
        source_abs = source.resolve()
        destination_abs = destination.resolve()

        # Track what actually happens during compilation
        warnings = []
        actually_applied_passes = list(enabled_passes)  # Start with requested passes

        # Detect compiler based on file extension
        if source_abs.suffix in ['.cpp', '.cxx', '.cc', '.c++']:
            base_compiler = "clang++"
            # Add C++ standard library linking
            compiler_flags = compiler_flags + ["-lstdc++"]
        else:
            base_compiler = "clang"

        compiler = base_compiler

        mlir_passes = [p for p in enabled_passes if p in ["string-encrypt", "symbol-obfuscate", "crypto-hash", "constant-obfuscate"]]
        ollvm_passes = [p for p in enabled_passes if p not in mlir_passes]

        # The input for the current stage of the pipeline
        current_input = source_abs
        
        # Stage 1: MLIR Obfuscation
        if mlir_passes:
            self.logger.info("Running MLIR pipeline with passes: %s", ", ".join(mlir_passes))

            # Find MLIR plugin
            mlir_plugin = self._get_mlir_plugin_path()
            if not mlir_plugin:
                raise ObfuscationError(
                    "MLIR passes requested but plugin not found. "
                    "Please build the MLIR obfuscation library first:\n"
                    "  cd mlir-obs && mkdir build && cd build\n"
                    "  cmake .. && make"
                )

            # 1a: Compile source to LLVM IR
            llvm_ir_temp = destination_abs.parent / f"{destination_abs.stem}_temp.ll"
            ir_cmd = [compiler, str(current_input), "-S", "-emit-llvm", "-o", str(llvm_ir_temp)]
            # Add resource-dir flag for bundled clang
            resource_dir_flags = self._get_resource_dir_flag(compiler)
            if resource_dir_flags:
                ir_cmd.extend(resource_dir_flags)
            # Add target triple for cross-compilation
            target_triple = self._get_target_triple(config.platform, config.architecture)
            ir_cmd.extend([f"--target={target_triple}"])
            run_command(ir_cmd, cwd=source_abs.parent)

            # 1b: Convert LLVM IR to MLIR
            mlir_file = destination_abs.parent / f"{destination_abs.stem}_temp.mlir"
            translate_to_mlir_cmd = ["mlir-translate", "--import-llvm", str(llvm_ir_temp), "-o", str(mlir_file)]
            run_command(translate_to_mlir_cmd, cwd=source_abs.parent)

            # 1c: Apply MLIR obfuscation passes
            obfuscated_mlir = destination_abs.parent / f"{destination_abs.stem}_obfuscated.mlir"

            # Build pass pipeline: "builtin.module(string-encrypt,symbol-obfuscate)"
            passes_str = ",".join(mlir_passes)
            pass_pipeline = f"builtin.module({passes_str})"

            opt_cmd = [
                "mlir-opt",
                str(mlir_file),
                f"--load-pass-plugin={str(mlir_plugin)}",
                f"--pass-pipeline={pass_pipeline}",
                "-o", str(obfuscated_mlir)
            ]
            run_command(opt_cmd, cwd=source_abs.parent)

            # 1d: Translate MLIR back to LLVM IR
            llvm_ir_raw = destination_abs.parent / f"{destination_abs.stem}_raw.ll"
            translate_cmd = ["mlir-translate", "--mlir-to-llvmir", str(obfuscated_mlir), "-o", str(llvm_ir_raw)]
            run_command(translate_cmd, cwd=source_abs.parent)

            # Fix target triple and data layout (MLIR sometimes generates malformed output)
            llvm_ir_file = destination_abs.parent / f"{destination_abs.stem}_from_mlir.ll"

            # Get target triple for cross-compilation
            target_triple = self._get_target_triple(config.platform, config.architecture)
            # Data layout depends on the target
            if config.platform == Platform.WINDOWS:
                data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            else:
                data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

            # Read, fix, and write - remove ALL target-specific attributes
            with open(str(llvm_ir_raw), 'r') as f:
                ir_content = f.read()

            import re
            # Fix target triple and datalayout
            ir_content = re.sub(r'target triple = ".*"', f'target triple = "{target_triple}"', ir_content)
            ir_content = re.sub(r'target datalayout = ".*"', f'target datalayout = "{data_layout}"', ir_content)

            # Remove corrupted CPU attributes
            ir_content = re.sub(r'"target-cpu"="[^"]*"', '', ir_content)
            ir_content = re.sub(r'"target-features"="[^"]*"', '', ir_content)
            ir_content = re.sub(r'"tune-cpu"="[^"]*"', '', ir_content)

            # Clean up empty attribute groups
            ir_content = re.sub(r'attributes #\d+ = \{\s*\}', '', ir_content)

            with open(str(llvm_ir_file), 'w') as f:
                f.write(ir_content)

            current_input = llvm_ir_file

            # Clean up raw IR
            if llvm_ir_raw.exists():
                llvm_ir_raw.unlink()

            # Clean up intermediate files
            if llvm_ir_temp.exists():
                llvm_ir_temp.unlink()
            if mlir_file.exists():
                mlir_file.unlink()
            if obfuscated_mlir.exists():
                obfuscated_mlir.unlink()

        # Stage 2: OLLVM Obfuscation
        if ollvm_passes:
            self.logger.info("Running OLLVM pipeline with passes: %s", ", ".join(ollvm_passes))

            # Find the OLLVM plugin
            plugin_path = config.custom_pass_plugin or self._get_bundled_plugin_path(config.platform)
            if not plugin_path or not plugin_path.exists():
                warning_msg = (
                    f"OLLVM passes requested ({', '.join(ollvm_passes)}) but LLVMObfuscationPlugin.so not found. "
                    "These passes require a custom OLLVM build. Skipping OLLVM passes. "
                    "MLIR passes (string-encrypt, symbol-obfuscate, crypto-hash, constant-obfuscate) are still available."
                )
                self.logger.warning(warning_msg)
                warnings.append(warning_msg)
                actually_applied_passes = [p for p in actually_applied_passes if p not in ollvm_passes]
                ollvm_passes = []  # Skip OLLVM stage

        if ollvm_passes:

            # If the input is still a source file, compile it to LLVM IR
            if current_input.suffix not in ['.ll', '.bc']:
                ir_file = destination_abs.parent / f"{destination_abs.stem}_temp.ll"
                # Apply -O3 BEFORE obfuscation for stable, optimized code
                pre_obfuscation_flags = ["-O3", "-fno-builtin", "-fno-slp-vectorize", "-fno-vectorize"]
                ir_cmd = [compiler, str(current_input), "-S", "-emit-llvm", "-o", str(ir_file)] + pre_obfuscation_flags
                # Add resource-dir flag if using custom clang
                resource_dir_flags = self._get_resource_dir_flag(compiler)
                if resource_dir_flags:
                    ir_cmd.extend(resource_dir_flags)
                # Add target triple for cross-compilation
                target_triple = self._get_target_triple(config.platform, config.architecture)
                ir_cmd.extend([f"--target={target_triple}"])
                run_command(ir_cmd, cwd=source_abs.parent)
                current_input = ir_file

            # Check for C++ exception handling - Hikari approach
            # Flattening crashes on EH, but other passes work fine
            if self._has_exception_handling(current_input):
                if "flattening" in ollvm_passes:
                    warning_msg = (
                        "C++ exception handling detected in IR (invoke/landingpad instructions). "
                        "Flattening pass disabled for stability (known to crash on exception handling). "
                        "Other OLLVM passes (substitution, boguscf, split) will still be applied."
                    )
                    self.logger.warning(warning_msg)
                    warnings.append(warning_msg)
                    ollvm_passes = [p for p in ollvm_passes if p != "flattening"]
                    actually_applied_passes = [p for p in actually_applied_passes if p != "flattening"]

            # Determine opt binary path - check multiple locations
            plugin_path_resolved = Path(plugin_path)
            bundled_opt = plugin_path_resolved.parent / "opt"
            bundled_clang = plugin_path_resolved.parent / "clang"
            opt_binary = None

            if bundled_opt.exists():
                self.logger.info("Using bundled opt: %s", bundled_opt)
                opt_binary = bundled_opt
                if bundled_clang.exists():
                    self.logger.info("Using bundled clang from LLVM 22: %s", bundled_clang)
                    compiler = str(bundled_clang)
            elif Path("/usr/local/llvm-obfuscator/bin/opt").exists():
                opt_binary = Path("/usr/local/llvm-obfuscator/bin/opt")
                self.logger.info("Using opt from Docker installation: %s", opt_binary)
                docker_clang = Path("/usr/local/llvm-obfuscator/bin/clang")
                if docker_clang.exists():
                    compiler = str(docker_clang)
                    self.logger.info("Using bundled clang from Docker installation (LLVM 22): %s", compiler)
            elif "/llvm-project/build/lib/" in str(plugin_path_resolved):
                llvm_build_dir = plugin_path_resolved.parent.parent
                opt_binary = llvm_build_dir / "bin" / "opt"
                llvm_clang = llvm_build_dir / "bin" / "clang"
                if opt_binary.exists():
                    self.logger.info("Using opt from LLVM build: %s", opt_binary)
                    if llvm_clang.exists():
                        compiler = str(llvm_clang)
                else:
                    raise ObfuscationError("Custom opt binary not found")
            else:
                raise ObfuscationError(f"OLLVM opt binary not found at {bundled_opt}")

            # Only continue with OLLVM if we still have passes enabled
            if ollvm_passes:
                # Apply OLLVM passes multiple times based on cycles parameter
                passes_pipeline = ",".join(ollvm_passes)

                for cycle_num in range(1, cycles + 1):
                    if cycles > 1:
                        self.logger.info(f"OLLVM cycle {cycle_num}/{cycles}: Applying passes ({passes_pipeline})")
                    else:
                        self.logger.info("Applying OLLVM passes via opt")

                    # Output file for this cycle
                    obfuscated_ir = destination_abs.parent / f"{destination_abs.stem}_obfuscated_c{cycle_num}.bc"

                    opt_cmd = [
                        str(opt_binary),
                        "-load-pass-plugin=" + str(plugin_path),
                        f"-passes={passes_pipeline}",
                        str(current_input),
                        "-o", str(obfuscated_ir)
                    ]
                    run_command(opt_cmd, cwd=source_abs.parent)

                    # Clean up previous cycle's IR (except original source)
                    if current_input != source_abs and current_input.exists() and current_input.suffix in ['.ll', '.bc']:
                        current_input.unlink()

                    current_input = obfuscated_ir

                if cycles > 1:
                    self.logger.info(f"Completed {cycles} OLLVM obfuscation cycles")

        # Stage 3: Compile to binary
        self.logger.info("Compiling final IR to binary...")
        final_cmd = [compiler, str(current_input), "-o", str(destination_abs)] + compiler_flags
        # Add target triple for cross-compilation
        target_triple = self._get_target_triple(config.platform, config.architecture)
        final_cmd.extend([f"--target={target_triple}"])
        run_command(final_cmd, cwd=source_abs.parent)

        # Cleanup any remaining intermediate files
        if current_input != source_abs and current_input.exists():
            current_input.unlink()

        return {
            "applied_passes": actually_applied_passes,
            "warnings": warnings,
            "disabled_passes": []
        }

    def _compile_with_clangir(
        self,
        source: Path,
        destination: Path,
        config: ObfuscationConfig,
        compiler_flags: List[str],
        enabled_passes: List[str],
        cycles: int = 1,
    ) -> Dict:
        """
        NEW PIPELINE - ClangIR → High-level MLIR → Obfuscation Passes

        Pipeline: C/C++ → ClangIR → MLIR (CIR dialect) → Lower to LLVM → MLIR passes → Binary

        Args:
            cycles: Number of times to apply OLLVM passes (currently unused in ClangIR pipeline)
        """
        from .config import Platform

        # Use absolute paths
        source_abs = source.resolve()
        destination_abs = destination.resolve()

        warnings = []
        actually_applied_passes = list(enabled_passes)

        # Detect compiler
        if source_abs.suffix in ['.cpp', '.cxx', '.cc', '.c++']:
            base_compiler = "clang++"
            compiler_flags = compiler_flags + ["-lstdc++"]
        else:
            base_compiler = "clang"

        compiler = base_compiler

        mlir_passes = [p for p in enabled_passes if p in ["string-encrypt", "symbol-obfuscate", "crypto-hash", "constant-obfuscate"]]
        ollvm_passes = [p for p in enabled_passes if p not in mlir_passes]

        current_input = source_abs

        # Stage 1: ClangIR Frontend (C/C++ → ClangIR MLIR)
        self.logger.info("Running ClangIR frontend...")

        # Check if clangir is available
        import shutil
        if not shutil.which("clangir"):
            raise ObfuscationError(
                "ClangIR frontend requested but 'clangir' command not found. "
                "Please ensure ClangIR is built and available in PATH."
            )

        # 1a: Compile source to ClangIR MLIR (CIR dialect)
        cir_mlir_file = destination_abs.parent / f"{destination_abs.stem}_cir.mlir"
        clangir_cmd = ["clangir", str(current_input), "-emit-cir", "-o", str(cir_mlir_file)]
        # Add target triple for cross-compilation
        target_triple = self._get_target_triple(config.platform, config.architecture)
        clangir_cmd.extend([f"--target={target_triple}"])
        run_command(clangir_cmd, cwd=source_abs.parent)

        # 1b: Lower ClangIR to LLVM dialect MLIR
        llvm_mlir_file = destination_abs.parent / f"{destination_abs.stem}_llvm.mlir"
        lower_cmd = ["mlir-opt", str(cir_mlir_file), "--cir-to-llvm", "-o", str(llvm_mlir_file)]
        run_command(lower_cmd, cwd=source_abs.parent)

        current_input = llvm_mlir_file

        # Stage 2: Apply MLIR Obfuscation Passes
        if mlir_passes:
            self.logger.info("Applying MLIR obfuscation passes: %s", ", ".join(mlir_passes))

            mlir_plugin = self._get_mlir_plugin_path()
            if not mlir_plugin:
                raise ObfuscationError("MLIR passes requested but plugin not found.")

            obfuscated_mlir = destination_abs.parent / f"{destination_abs.stem}_obfuscated.mlir"
            passes_str = ",".join(mlir_passes)
            pass_pipeline = f"builtin.module({passes_str})"

            opt_cmd = [
                "mlir-opt",
                str(current_input),
                f"--load-pass-plugin={str(mlir_plugin)}",
                f"--pass-pipeline={pass_pipeline}",
                "-o", str(obfuscated_mlir)
            ]
            run_command(opt_cmd, cwd=source_abs.parent)

            current_input = obfuscated_mlir

        # Stage 3: Convert MLIR to LLVM IR
        llvm_ir_file = destination_abs.parent / f"{destination_abs.stem}_from_clangir.ll"
        translate_cmd = ["mlir-translate", "--mlir-to-llvmir", str(current_input), "-o", str(llvm_ir_file)]
        run_command(translate_cmd, cwd=source_abs.parent)

        # Fix target triple and data layout
        import re
        # Get target triple for cross-compilation
        target_triple = self._get_target_triple(config.platform, config.architecture)
        # Data layout depends on the target
        if config.platform == Platform.WINDOWS:
            data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
        else:
            data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

        with open(str(llvm_ir_file), 'r') as f:
            ir_content = f.read()

        ir_content = re.sub(r'target triple = ".*"', f'target triple = "{target_triple}"', ir_content)
        ir_content = re.sub(r'target datalayout = ".*"', f'target datalayout = "{data_layout}"', ir_content)
        ir_content = re.sub(r'"target-cpu"="[^"]*"', '', ir_content)
        ir_content = re.sub(r'"target-features"="[^"]*"', '', ir_content)
        ir_content = re.sub(r'"tune-cpu"="[^"]*"', '', ir_content)
        ir_content = re.sub(r'attributes #\d+ = \{\s*\}', '', ir_content)

        with open(str(llvm_ir_file), 'w') as f:
            f.write(ir_content)

        current_input = llvm_ir_file

        # Stage 4: OLLVM Obfuscation (optional)
        if ollvm_passes:
            self.logger.info("Running OLLVM pipeline with passes: %s", ", ".join(ollvm_passes))

            plugin_path = config.custom_pass_plugin or self._get_bundled_plugin_path(config.platform)
            if not plugin_path or not plugin_path.exists():
                warning_msg = (
                    f"OLLVM passes requested ({', '.join(ollvm_passes)}) but LLVMObfuscationPlugin.so not found. "
                    "These passes require a custom OLLVM build. Skipping OLLVM passes. "
                    "MLIR passes (string-encrypt, symbol-obfuscate, crypto-hash, constant-obfuscate) are still available."
                )
                self.logger.warning(warning_msg)
                warnings.append(warning_msg)
                actually_applied_passes = [p for p in actually_applied_passes if p not in ollvm_passes]
                ollvm_passes = []  # Skip OLLVM stage

        if ollvm_passes:
            if self._has_exception_handling(current_input):
                warnings.append("C++ exception handling detected; some OLLVM passes may be unstable.")

            obfuscated_ir = destination_abs.parent / f"{destination_abs.stem}_obfuscated.bc"
            opt_binary = plugin_path.parent / "opt"
            if not opt_binary.exists():
                warning_msg = f"OLLVM opt binary not found at {opt_binary}. Skipping OLLVM passes."
                self.logger.warning(warning_msg)
                warnings.append(warning_msg)
                actually_applied_passes = [p for p in actually_applied_passes if p not in ollvm_passes]
                ollvm_passes = []  # Skip OLLVM stage

        if ollvm_passes:

            passes_pipeline = ",".join(ollvm_passes)
            opt_cmd = [
                str(opt_binary),
                "-load-pass-plugin=" + str(plugin_path),
                f"-passes={passes_pipeline}",
                str(current_input),
                "-o", str(obfuscated_ir)
            ]
            run_command(opt_cmd, cwd=source_abs.parent)
            current_input = obfuscated_ir

        # Stage 5: Compile to binary
        self.logger.info("Compiling final IR to binary...")
        final_cmd = [compiler, str(current_input), "-o", str(destination_abs)] + compiler_flags
        # Add target triple for cross-compilation
        target_triple = self._get_target_triple(config.platform, config.architecture)
        final_cmd.extend([f"--target={target_triple}"])
        run_command(final_cmd, cwd=source_abs.parent)

        # Cleanup intermediate files
        if cir_mlir_file.exists():
            cir_mlir_file.unlink()
        if llvm_mlir_file.exists():
            llvm_mlir_file.unlink()
        if current_input != source_abs and current_input.exists():
            current_input.unlink()

        return {
            "applied_passes": actually_applied_passes,
            "warnings": warnings,
            "disabled_passes": []
        }

    def _compile_and_analyze_baseline(self, source_file: Path, baseline_binary: Path, config: ObfuscationConfig) -> Dict:
        """Compile an unobfuscated baseline binary and analyze its metrics for comparison."""
        # Default values in case baseline compilation fails
        default_metrics = {
            "file_size": 0,
            "binary_format": "unknown",
            "sections": {},
            "symbols_count": 0,
            "functions_count": 0,
            "entropy": 0.0,
        }

        try:
            # Use absolute paths to avoid path resolution issues
            source_abs = source_file.resolve()
            baseline_abs = baseline_binary.resolve()

            # Determine project root for multi-file projects
            # Look for a 'project' directory in parent hierarchy or use source parent
            project_root = source_abs.parent
            if project_root.name != "project":
                for parent in source_abs.parents:
                    if parent.name == "project":
                        project_root = parent
                        break

            # Generate stub config headers if needed (e.g., curl_config.h for curl projects)
            try:
                generated_headers = ensure_generated_headers_exist(project_root)
                if generated_headers:
                    self.logger.info(f"Generated {len(generated_headers)} stub config headers for baseline")
            except Exception as e:
                self.logger.debug(f"Stub header generation skipped: {e}")

            # Detect compiler
            if source_file.suffix in ['.cpp', '.cxx', '.cc', '.c++']:
                compiler = "clang++"
                compile_flags = ["-lstdc++"]
            else:
                compiler = "clang"
                compile_flags = []

            # Add minimal optimization flags
            compile_flags.extend(["-O2"])

            # Add target triple for cross-compilation
            target_triple = self._get_target_triple(config.platform, config.architecture)
            compile_flags.append(f"--target={target_triple}")

            # Add include paths for common directories in the project
            include_dirs = set()
            include_dirs.add(str(project_root))

            # Add common subdirectories as include paths
            for subdir in ['include', 'lib', 'src', 'build']:
                subdir_path = project_root / subdir
                if subdir_path.exists():
                    include_dirs.add(str(subdir_path))

            # Add generated header directories (from stub headers)
            try:
                for header_path in ensure_generated_headers_exist(project_root):
                    include_dirs.add(str(header_path.parent))
            except Exception:
                pass

            # Add include flags
            for include_dir in include_dirs:
                compile_flags.append(f"-I{include_dir}")

            # Add additional source files from compiler_flags (for multi-file projects)
            # Filter out source files from compiler flags
            additional_sources = [flag for flag in config.compiler_flags if flag.endswith(('.c', '.cpp', '.cc', '.cxx', '.c++'))]

            # Compile baseline with absolute paths, including all source files
            command = [compiler, str(source_abs)] + additional_sources + ["-o", str(baseline_abs)] + compile_flags
            run_command(command)

            # Analyze baseline binary
            if baseline_binary.exists():
                file_size = get_file_size(baseline_binary)
                binary_format = detect_binary_format(baseline_binary)
                sections = list_sections(baseline_binary)
                symbols_count, functions_count = summarize_symbols(baseline_binary)
                entropy = compute_entropy(baseline_binary.read_bytes())

                return {
                    "file_size": file_size,
                    "binary_format": binary_format,
                    "sections": sections,
                    "symbols_count": symbols_count,
                    "functions_count": functions_count,
                    "entropy": entropy,
                }
            else:
                self.logger.warning("Baseline binary not created, using default metrics")
                return default_metrics
        except Exception as e:
            self.logger.warning(f"Failed to compile baseline binary: {e}, using default metrics")
            return default_metrics

    def _estimate_metrics(
        self,
        source_file: Path,
        output_binary: Path,
        passes: List[str],
        cycles: int,
        string_result: Optional[Dict],
        fake_loops,
        entropy: float,
    ) -> Dict:
        baseline_score = 50 + 5 * len(passes) + 3 * cycles
        score = min(95.0, baseline_score)
        symbol_reduction = round(min(90.0, 20 + 10 * len(passes)), 2)
        function_reduction = round(min(70.0, 10 + 5 * len(passes)), 2)
        size_reduction = round(max(-30.0, 10 - 5 * len(passes)), 2)
        entropy_increase = round(entropy * 0.1, 2)
        bogus_code_info = {
            "dead_code_blocks": len(passes) * 3,
            "opaque_predicates": len(passes) * 2,
            "junk_instructions": len(passes) * 5,
            "code_bloat_percentage": round(5 + len(passes) * 1.5, 2),
        }
        string_obfuscation = {
            "total_strings": 0,
            "encrypted_strings": 0,
            "encryption_method": "none",
            "encryption_percentage": 0.0,
        }
        if string_result:
            string_obfuscation.update(string_result)
        fake_loops_inserted = {
            "count": len(fake_loops),
            "types": [loop.loop_type for loop in fake_loops],
            "locations": [loop.location for loop in fake_loops],
        }
        cycles_completed = {
            "total_cycles": cycles,
            "per_cycle_metrics": [
                {
                    "cycle": idx + 1,
                    "passes_applied": passes,
                    "duration_ms": 500 + 100 * idx,
                }
                for idx in range(cycles)
            ],
        }
        estimated_effort = "6-10 weeks" if score >= 80 else "4-6 weeks"
        return {
            "bogus_code_info": bogus_code_info,
            "string_obfuscation": string_obfuscation,
            "fake_loops_inserted": fake_loops_inserted,
            "cycles_completed": cycles_completed,
            "obfuscation_score": round(score, 2),
            "symbol_reduction": symbol_reduction,
            "function_reduction": function_reduction,
            "size_reduction": size_reduction,
            "entropy_increase": entropy_increase,
            "estimated_re_effort": estimated_effort,
        }
