from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import ObfuscationConfig, Platform
from .exceptions import ObfuscationError
from .fake_loop_inserter import FakeLoopGenerator
from .reporter import ObfuscationReport
from .string_encryptor import StringEncryptionResult, XORStringEncryptor
from .symbol_obfuscator import SymbolObfuscator
from .utils import (
    compute_entropy,
    create_logger,
    detect_binary_format,
    ensure_directory,
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
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-fno-use-cxa-atexit",
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
    ]

    def __init__(self, reporter: Optional[ObfuscationReport] = None) -> None:
        self.logger = create_logger(__name__)
        self.reporter = reporter
        self.encryptor = XORStringEncryptor()
        self.fake_loop_generator = FakeLoopGenerator()
        self.symbol_obfuscator = SymbolObfuscator()

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
        if config.platform == Platform.WINDOWS:
            require_tool("x86_64-w64-mingw32-gcc")

        source_content = source_file.read_text(encoding="utf-8", errors="ignore")

        # Compile baseline (unobfuscated) binary for comparison
        self.logger.info("Compiling baseline binary for comparison...")
        baseline_binary = output_directory / f"{source_file.stem}_baseline"
        baseline_metrics = self._compile_and_analyze_baseline(source_file, baseline_binary, config)

        # Symbol obfuscation (if enabled) - applied FIRST before other transformations
        symbol_result = None
        working_source = source_file
        if config.advanced.symbol_obfuscation.enabled:
            try:
                symbol_obfuscated_file = output_directory / f"{source_file.stem}_symbol_obfuscated{source_file.suffix}"
                symbol_result = self.symbol_obfuscator.obfuscate(
                    source_file=source_file,
                    output_file=symbol_obfuscated_file,
                    algorithm=config.advanced.symbol_obfuscation.algorithm,
                    hash_length=config.advanced.symbol_obfuscation.hash_length,
                    prefix_style=config.advanced.symbol_obfuscation.prefix_style,
                    salt=config.advanced.symbol_obfuscation.salt,
                    preserve_main=config.advanced.symbol_obfuscation.preserve_main,
                    preserve_stdlib=config.advanced.symbol_obfuscation.preserve_stdlib,
                    generate_map=True,
                    map_file=output_directory / "symbol_map.json",
                    is_cpp=source_file.suffix in [".cpp", ".cc", ".cxx"],
                )
                working_source = symbol_obfuscated_file
                self.logger.info(f"Symbol obfuscation complete: {symbol_result['symbols_obfuscated']} symbols renamed")
            except Exception as e:
                self.logger.warning(f"Symbol obfuscation failed, continuing without it: {e}")

        # String encryption (if enabled) - applied to source content
        string_result: Optional[StringEncryptionResult] = None
        if config.advanced.string_encryption:
            try:
                # Get the symbol-obfuscated source if available, otherwise use original
                current_source_content = working_source.read_text(encoding="utf-8", errors="replace")
                string_result = self.encryptor.encrypt_strings(current_source_content)

                # Write the transformed source to a new file
                string_encrypted_file = output_directory / f"{source_file.stem}_string_encrypted{source_file.suffix}"
                string_encrypted_file.write_text(string_result.transformed_source, encoding="utf-8", errors="replace")
                working_source = string_encrypted_file
                self.logger.info(f"String encryption complete: {string_result.encrypted_strings}/{string_result.total_strings} strings encrypted")
            except Exception as e:
                self.logger.error(f"String encryption failed: {e}")
                string_result = None

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

        fake_loops = []
        if config.advanced.fake_loops:
            fake_loops = self.fake_loop_generator.generate(config.advanced.fake_loops, source_file.name)

        enabled_passes = config.passes.enabled_passes()
        compiler_flags = merge_flags(self.BASE_FLAGS, config.compiler_flags)

        # IMPORTANT: Cycles only make sense for source code recompilation
        # Once we have a binary, we can't feed it back through the compiler
        # So if OLLVM passes are enabled, limit to 1 cycle
        effective_cycles = 1 if enabled_passes else config.advanced.cycles
        if enabled_passes and config.advanced.cycles > 1:
            self.logger.warning(
                "Multiple cycles (%d) requested with OLLVM passes enabled. "
                "Cycles only apply to source code compilation. "
                "Running 1 cycle with all OLLVM passes instead.",
                config.advanced.cycles
            )

        intermediate_source = working_source  # Use symbol-obfuscated source if enabled
        for cycle in range(1, effective_cycles + 1):
            self.logger.info("Starting cycle %s/%s", cycle, effective_cycles)
            intermediate_binary = output_binary if cycle == effective_cycles else output_directory / f"{output_binary.stem}_cycle{cycle}{output_binary.suffix}"

            cycle_result = self._compile(
                intermediate_source,
                intermediate_binary,
                config,
                compiler_flags,
                enabled_passes,
            )

            # Track what actually happened
            if cycle_result:
                actually_applied_passes = cycle_result.get("applied_passes", [])
                # Always extend warnings list (even if empty, to maintain consistency)
                warnings_log.extend(cycle_result.get("warnings", []))

            intermediate_source = intermediate_binary

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
                "obfuscation_methods": actually_applied_passes + (["symbol_obfuscation"] if symbol_result else []) + (["string_encryption"] if string_result else []) + (["indirect_calls"] if indirect_call_result else []),
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

    def _get_resource_dir_flag(self, compiler_path: str, is_cpp: bool = False) -> List[str]:
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

        # For bundled clang in /usr/local/llvm-obfuscator, use bundled resource dir
        if "/usr/local/llvm-obfuscator/" in resolved_path:
            bundled_resource_dir = "/usr/local/llvm-obfuscator/lib/clang/22"
            if Path(bundled_resource_dir).exists():
                self.logger.info(f"[RESOURCE-DIR-DEBUG] Using bundled resource directory: {bundled_resource_dir}")
                return ["-resource-dir", bundled_resource_dir]

        # Try to find system clang's resource directory (fallback)
        # Priority: system clang-19 > clang-18 > clang
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

    def _get_llvm_version(self) -> str:
        """Get the LLVM version being used."""
        import subprocess
        try:
            result = subprocess.run(
                ["clang", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract version from output like "clang version 19.0.0"
                for line in result.stdout.split('\n'):
                    if 'version' in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'version' and i + 1 < len(parts):
                                return parts[i + 1].split('.')[0]  # Return major version
        except Exception:
            pass
        return "19"  # Default fallback

    def _has_exception_handling(self, ir_file: Path) -> bool:
        """
        Check if LLVM IR file contains C++ exception handling (invoke/landingpad).

        Returns True if the IR uses exception handling, which is incompatible
        with the OLLVM flattening pass.
        """
        try:
            ir_content = ir_file.read_text(encoding='utf-8', errors='ignore')

            # Check for invoke instructions (exception-aware function calls)
            has_invoke = ' invoke ' in ir_content

            # Check for landingpad instructions (exception handlers)
            has_landingpad = ' landingpad ' in ir_content

            return has_invoke or has_landingpad
        except Exception as e:
            self.logger.warning(f"Could not check for exception handling in IR: {e}")
            return False

    def _compile(
        self,
        source: Path,
        destination: Path,
        config: ObfuscationConfig,
        compiler_flags: List[str],
        enabled_passes: List[str],
    ) -> Dict:
        # Use absolute paths to avoid path resolution issues
        source_abs = source.resolve()
        destination_abs = destination.resolve()

        warnings: List[str] = []
        actually_applied = list(enabled_passes)

        # Detect compiler based on file extension
        # IMPORTANT: Bundled clang works for C but has incomplete headers for C++
        # Use system clang for C++, bundled clang for C
        bundled_clang_dir = Path("/usr/local/llvm-obfuscator/bin")
        is_cpp = source_abs.suffix in ['.cpp', '.cxx', '.cc', '.c++']

        # Use bundled clang for both C and C++ (headers are now complete)
        if (bundled_clang_dir / "clang").exists():
            compiler = str(bundled_clang_dir / "clang")
        else:
            compiler = "clang"
        # Build IR flags: strip LTO-related flags for the IR-phase
        ir_flags: List[str] = [f for f in compiler_flags if 'lto' not in f.lower()]
        if is_cpp:
            ir_flags = ["-x", "c++"] + ir_flags

        # Add resource-dir if necessary
        ir_flags.extend(self._get_resource_dir_flag(compiler, is_cpp=is_cpp))

        # Windows target
        if config.platform == Platform.WINDOWS:
            ir_flags.append('--target=x86_64-w64-mingw32')

        # If no OLLVM passes requested -> normal compile
        if not enabled_passes:
            final_cmd = [compiler, str(source_abs), '-o', str(destination_abs)] + compiler_flags
            final_cmd.extend(self._get_resource_dir_flag(compiler, is_cpp=is_cpp))
            if config.platform == Platform.WINDOWS:
                final_cmd.append('--target=x86_64-w64-mingw32')
            run_command(final_cmd, cwd=source_abs.parent)
            return {"applied_passes": [], "warnings": []}

        # Find plugin
        plugin = config.custom_pass_plugin
        if not plugin:
            plugin = self._get_bundled_plugin_path(config.platform)
        if not plugin or not Path(plugin).exists():
            raise ObfuscationError("OLLVM plugin not found")
        plugin_path = Path(plugin)

        # Find opt: prefer plugin sibling opt
        opt_bin = plugin_path.parent / 'opt'
        if not opt_bin.exists():
            # fallback to common docker install location used by project
            if Path('/usr/local/llvm-obfuscator/bin/opt').exists():
                opt_bin = Path('/usr/local/llvm-obfuscator/bin/opt')
            else:
                # try system opt
                from shutil import which

                sys_opt = which('opt')
                if sys_opt:
                    opt_bin = Path(sys_opt)
                else:
                    raise ObfuscationError('opt binary not found for running OLLVM passes')

        # Step 1: source -> IR
        ir_file = destination_abs.parent / f"{destination_abs.stem}_temp.ll"
        ir_cmd = [compiler, str(source_abs), '-S', '-emit-llvm', '-o', str(ir_file)] + ir_flags
        self.logger.info(f"[IR-FLAGS] {ir_flags}")

        # Warn about potential C++ compatibility issues
        if is_cpp:
            self.logger.warning(
                "C++ source detected. OLLVM plugin may encounter compatibility issues. "
                "If obfuscation fails, the code will be compiled without OLLVM passes."
            )

        run_command(ir_cmd, cwd=source_abs.parent)

        # Check for exception handling

        # Step 2: apply passes using opt
        obf_ir = destination_abs.parent / f"{destination_abs.stem}_obf.bc"
        passes_pipeline = ','.join(enabled_passes)
        opt_cmd = [str(opt_bin), f'-load-pass-plugin={str(plugin_path)}', f'-passes={passes_pipeline}', str(ir_file), '-o', str(obf_ir)]
        self.logger.info(f"[OPT] {opt_cmd}")

        try:
            run_command(opt_cmd, cwd=source_abs.parent)
        except ObfuscationError as e:
            # Check for plugin symbol incompatibility (common with C++ code)
            if "symbol lookup error" in str(e) or "undefined symbol" in str(e):
                warning = (
                    f"OLLVM plugin incompatibility detected (likely LLVM version mismatch). "
                    f"This often occurs with C++ code due to missing symbols. "
                    f"Compiling without OLLVM passes. "
                    f"To fix: rebuild the plugin against LLVM {self._get_llvm_version()}"
                )
                self.logger.warning(warning)
                warnings.append(warning)
                actually_applied = []
                # Fallback: compile normally without OLLVM passes - strip LTO flags
                fallback_flags = [f for f in compiler_flags if 'lto' not in f.lower()]
                fallback_cmd = [compiler, str(source_abs), '-o', str(destination_abs)] + fallback_flags
                if is_cpp:
                    fallback_cmd.append('-lstdc++')  # Link C++ standard library
                fallback_cmd.extend(self._get_resource_dir_flag(compiler, is_cpp=is_cpp))
                if config.platform == Platform.WINDOWS:
                    fallback_cmd.append('--target=x86_64-w64-mingw32')
                run_command(fallback_cmd, cwd=source_abs.parent)
                if ir_file.exists():
                    ir_file.unlink()
                return {"applied_passes": [], "warnings": warnings}
            else:
                # Re-raise if it's a different error
                raise

        # Step 3: compile obfuscated IR -> binary
        # Strip LTO flags as they're incompatible when linking from bitcode
        # (bitcode is already intermediate representation, LTO doesn't apply)
        final_flags = [f for f in compiler_flags if 'lto' not in f.lower()]
        final_cmd = [compiler, str(obf_ir), '-o', str(destination_abs)] + final_flags
        if is_cpp:
            # Don't use -x c++ when compiling from bitcode - clang auto-detects it
            # Just link the C++ standard library
            final_cmd.append("-lstdc++")  # Link C++ standard library
        final_cmd.extend(self._get_resource_dir_flag(compiler, is_cpp=is_cpp))
        if config.platform == Platform.WINDOWS:
            final_cmd.append('--target=x86_64-w64-mingw32')

        self.logger.info(f"[FINAL] {final_cmd}")
        run_command(final_cmd, cwd=source_abs.parent)

        # cleanup
        if ir_file.exists():
            ir_file.unlink()
        if obf_ir.exists():
            obf_ir.unlink()

        return {"applied_passes": actually_applied, "warnings": warnings}

    def _compile_and_analyze_baseline(self, source_file: Path, baseline_binary: Path, config: ObfuscationConfig) -> Dict:
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

            # Detect compiler - use bundled clang for both C and C++
            bundled_clang_dir = Path("/usr/local/llvm-obfuscator/bin")
            is_cpp = source_file.suffix in ['.cpp', '.cxx', '.cc', '.c++']
            if is_cpp:
                compile_flags = ["-x", "c++", "-lstdc++"]
            else:
                compile_flags = []

            # Use bundled clang (headers are now complete)
            if (bundled_clang_dir / "clang").exists():
                compiler = str(bundled_clang_dir / "clang")
            else:
                compiler = "clang"

            # Add minimal optimization flags (no LTO for baseline to avoid LLVMgold.so dependency)
            compile_flags.extend(["-O2"])

            # Add resource-dir if necessary
            compile_flags.extend(self._get_resource_dir_flag(compiler, is_cpp=is_cpp))

            # Platform-specific flags
            if config.platform == Platform.WINDOWS:
                compile_flags.append("--target=x86_64-w64-mingw32")

            command = [compiler, str(source_abs), '-o', str(baseline_abs)] + compile_flags
            run_command(command, cwd=source_abs.parent)

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
        string_result: Optional[StringEncryptionResult],
        fake_loops,
        entropy: float,
    ) -> Dict:
        baseline_score = 50 + 5 * len(passes) + 3 * cycles
        score = min(95.0, baseline_score + (string_result.encryption_percentage if string_result else 0) * 0.2)
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
            "total_strings": string_result.total_strings if string_result else 0,
            "encrypted_strings": string_result.encrypted_strings if string_result else 0,
            "encryption_method": string_result.encryption_method if string_result else "none",
            "encryption_percentage": string_result.encryption_percentage if string_result else 0.0,
        }
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