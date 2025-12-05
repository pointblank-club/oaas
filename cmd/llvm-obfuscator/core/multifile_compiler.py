"""Multi-file IR compilation workflow for LLVM obfuscation.

This module handles the compilation of multi-file C/C++ projects with OLLVM passes:
  1) Apply source-level obfuscation (Layer 1 & 2) - handled externally
  2) Compile each TU to LLVM bitcode (.bc)
  3) Link all per-TU .bc files into a single unified module (unified.bc) using llvm-link
  4) Run OLLVM passes (Layer 3 & 4) with opt on unified.bc
  5) Produce final binary from obfuscated.bc
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import ObfuscationConfig, Platform, Architecture
from .exceptions import ObfuscationError
from .utils import create_logger, run_command, detect_project_compile_flags, ensure_generated_headers_exist

logger = create_logger(__name__)


def _get_cross_compile_flags(platform: Platform, arch: Architecture) -> list:
    """Get cross-compilation flags including target triple and sysroot.

    Returns list of flags like ["--target=x86_64-apple-darwin", "--sysroot=/opt/MacOSX.sdk"]
    """
    # Target triple mapping
    target_triples = {
        (Platform.LINUX, Architecture.X86_64): "x86_64-unknown-linux-gnu",
        (Platform.LINUX, Architecture.ARM64): "aarch64-unknown-linux-gnu",
        (Platform.LINUX, Architecture.X86): "i686-unknown-linux-gnu",
        (Platform.WINDOWS, Architecture.X86_64): "x86_64-w64-mingw32",
        (Platform.WINDOWS, Architecture.ARM64): "aarch64-w64-mingw32",
        (Platform.WINDOWS, Architecture.X86): "i686-w64-mingw32",
        (Platform.MACOS, Architecture.X86_64): "x86_64-apple-darwin",
        (Platform.MACOS, Architecture.ARM64): "aarch64-apple-darwin",
        (Platform.DARWIN, Architecture.X86_64): "x86_64-apple-darwin",
        (Platform.DARWIN, Architecture.ARM64): "aarch64-apple-darwin",
    }

    triple = target_triples.get((platform, arch), "x86_64-unknown-linux-gnu")
    flags = [f"--target={triple}"]

    # macOS requires sysroot for cross-compilation
    if platform in [Platform.MACOS, Platform.DARWIN]:
        flags.append("--sysroot=/opt/MacOSX.sdk")
        logger.info(f"Using macOS SDK sysroot: /opt/MacOSX.sdk")

    return flags


def compile_multifile_ir_workflow(
    source_abs: Path,
    destination_abs: Path,
    config: ObfuscationConfig,
    compiler_flags: List[str],
    enabled_passes: List[str],
    plugin_path: Path,
    compiler: str,
    symbol_obfuscator: Any,
    encryptor: Any,
    get_resource_dir_flag_fn: Callable[[str], List[str]],
    has_exception_handling_fn: Callable[[Path], bool],
    entrypoint_command: Optional[str] = None,
    project_root_override: Optional[Path] = None,
) -> Dict:
    """Compile multi-file project with OLLVM passes using unified IR approach.
    
    Args:
        source_abs: Main source file (absolute path)
        destination_abs: Output binary path (absolute path)
        config: Obfuscation configuration
        compiler_flags: Compiler flags (includes additional source files)
        enabled_passes: OLLVM passes to apply
        plugin_path: Path to OLLVM plugin
        compiler: Compiler command (clang or clang++)
        symbol_obfuscator: Symbol obfuscator instance (unused, for API compatibility)
        encryptor: String encryptor instance (unused, for API compatibility)
        get_resource_dir_flag_fn: Function to get resource directory flags
        has_exception_handling_fn: Function to check for exception handling in IR
        entrypoint_command: Optional build command to extract compile flags
        
    Returns:
        Dict with applied_passes, warnings, and disabled_passes
    """
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " MULTI-FILE IR WORKFLOW STARTED ".center(78) + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")
    logger.info("EXACT WORKFLOW TO BE EXECUTED:")
    logger.info("  1. Run the repo's ACTUAL build")
    logger.info("  2. Capture compile_commands.json")
    logger.info("  3. For each .c file: clang <EXACT FLAGS> -emit-llvm -c file.c -o file.bc")
    logger.info("  4. llvm-link all .bc files")
    logger.info("  5. Obfuscate with OLLVM passes")
    logger.info("  6. Recompile final binary")
    logger.info("")
    
    warnings: List[str] = []
    actually_applied_passes = list(enabled_passes)
    
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 0: Setup and Detection")
    logger.info("━" * 80)
    
    # Determine project root early (needed for filtering)
    if project_root_override:
        project_root = project_root_override
        logger.info(f"Using provided project root: {project_root}")
    else:
        project_root = source_abs.parent
        logger.info(f"Initial project root (from source): {project_root}")
    
    # Extract additional source files from compiler flags
    additional_sources = [
        flag for flag in compiler_flags 
        if flag.endswith(('.c', '.cpp', '.cc', '.cxx', '.c++'))
    ]
    
    # All source files (main + additional)
    all_sources = [source_abs] + [Path(src).resolve() for src in additional_sources]
    
    # Filter out example/test files and platform-specific packages
    # These files are often not part of the main build and require:
    # - External dependencies (examples)
    # - Platform-specific headers (OS400, VMS, etc.)
    excluded_patterns = [
        '/examples/',
        '/docs/examples/',
        '/tests/',               # ALL test files (libtest, server, unit)
        '/samples/',
        '/packages/',            # ALL platform packages (OS400, VMS, etc.)
    ]
    
    filtered_sources = []
    excluded_sources = []
    
    logger.info("")
    logger.info("Filtering source files...")
    
    for src in all_sources:
        src_str = str(src)
        is_excluded = any(pattern in src_str for pattern in excluded_patterns)
        
        if is_excluded:
            excluded_sources.append(src)
            # Determine exclusion reason
            if '/packages/' in src_str:
                reason = "Platform-specific code (OS400/VMS/etc.) - requires platform headers"
            elif '/tests/' in src_str:
                reason = "Test file - not part of main library"
            elif '/examples/' in src_str:
                reason = "Example file - requires optional external dependencies"
            else:
                reason = "Non-core file"
            logger.debug(f"  Excluding: {src.name} - {reason}")
        else:
            filtered_sources.append(src)
    
    if excluded_sources:
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning(f"⚠ EXCLUDED {len(excluded_sources)} files (examples/tests/platform-specific)")
        logger.warning("=" * 80)
        
        # Group by category
        examples = [s for s in excluded_sources if '/examples/' in str(s)]
        tests = [s for s in excluded_sources if '/tests/' in str(s)]
        packages = [s for s in excluded_sources if '/packages/' in str(s)]
        
        if examples:
            logger.warning(f"Example files ({len(examples)}): Require optional external dependencies")
            for src in examples[:3]:
                logger.warning(f"   - {src.name}")
            if len(examples) > 3:
                logger.warning(f"   ... and {len(examples) - 3} more examples")
        
        if tests:
            logger.warning(f"Test files ({len(tests)}): Not part of main library")
            for src in tests[:3]:
                logger.warning(f"   - {src.name}")
            if len(tests) > 3:
                logger.warning(f"   ... and {len(tests) - 3} more tests")
        
        if packages:
            logger.warning(f"Platform-specific ({len(packages)}): OS400/VMS/etc. require platform headers")
            for src in packages[:3]:
                logger.warning(f"   - {src.name}")
            if len(packages) > 3:
                logger.warning(f"   ... and {len(packages) - 3} more")
        
        logger.warning("")
        logger.warning("These files are NOT part of the main library build.")
        logger.warning("We're compiling only the core library files (lib/*.c)")
        logger.warning("=" * 80)
        logger.warning("")
    
    all_sources = filtered_sources
    
    if not all_sources:
        raise ObfuscationError(
            "No source files to compile after filtering. "
            "All files were examples/tests requiring external dependencies."
        )
    
    logger.info("")
    logger.info(f"Source files to compile: {len(all_sources)}")
    for idx, src in enumerate(all_sources, 1):
        logger.info(f"  [{idx}] {src.name}")
        logger.info(f"       Path: {src}")
        logger.info(f"       Exists: {src.exists()}")
    
    # Refine project root by finding common parent of all sources
    # (could be a parent directory if sources are in subdirectories)
    if not project_root_override and all_sources and len(all_sources) > 1:
        logger.info("")
        logger.info("Refining project root from source files...")
        common_parent = source_abs.parent
        for src in all_sources[1:]:
            try:
                # Find common parent
                src_parent = src.parent
                while common_parent not in src_parent.parents and common_parent != src_parent:
                    if common_parent.parent == common_parent:
                        break
                    common_parent = common_parent.parent
            except:
                pass
        
        if common_parent != project_root:
            project_root = common_parent
            logger.info(f"  → Refined to common parent: {project_root}")
    
    logger.info("")
    logger.info(f"Final project root: {project_root}")
    logger.info(f"  → Exists: {project_root.exists()}")
    logger.info(f"  → Is directory: {project_root.is_dir()}")
    
    # Auto-detect compile flags (including running FULL build sequence if entrypoint provided)
    # This is critical for large projects like curl that need ./buildconf && ./configure && make
    logger.info("")
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 1: Run Repo's ACTUAL Build & Capture compile_commands.json")
    logger.info("━" * 80)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Entrypoint command: {entrypoint_command or 'auto-detect'}")
    logger.info(f"Number of source files: {len(all_sources)}")
    logger.info("")
    logger.info("This step will:")
    logger.info("  → Detect build system (CMake/Autotools/Make/etc.)")
    logger.info("  → Run FULL build sequence")
    logger.info("  → Generate/capture compile_commands.json")
    logger.info("  → Extract exact -I/-D flags for each source file")
    logger.info("")
    logger.info("Starting build execution...")
    logger.info("━" * 80)
    
    try:
        logger.info("▶ Calling detect_project_compile_flags()...")
        auto_detected_flags = detect_project_compile_flags(project_root, entrypoint_command)
        logger.info(f"▶ detect_project_compile_flags() returned {len(auto_detected_flags)} flags")
    except Exception as e:
        logger.error(f"▶ detect_project_compile_flags() RAISED EXCEPTION: {type(e).__name__}")
        logger.error(f"▶ Exception message: {e}")
        import traceback
        logger.error("▶ Full traceback:")
        for line in traceback.format_exc().split('\n'):
            logger.error(f"  {line}")
        logger.error("=" * 80)
        logger.error("ERROR: COMPILE FLAG DETECTION FAILED")
        logger.error("=" * 80)
        logger.error(f"Failed to auto-detect compile flags: {e}")
        logger.error("")
        logger.error("The project build may have failed. This is a CRITICAL error.")
        logger.error("Without proper compile flags, per-TU IR generation will fail.")
        logger.error("")
        
        # Check if build.log exists and show excerpt
        build_log = project_root / "build.log"
        if build_log.exists():
            logger.error(f"Build log found at: {build_log}")
            try:
                log_content = build_log.read_text(encoding='utf-8', errors='ignore')
                # Find error lines
                error_lines = [l for l in log_content.split('\n') if 'error' in l.lower()][:20]
                if error_lines:
                    logger.error("Build log errors (first 20):")
                    for line in error_lines:
                        logger.error(f"  {line[:200]}")
                else:
                    # Show last 50 lines if no specific errors found
                    logger.error("Build log (last 50 lines):")
                    for line in log_content.split('\n')[-50:]:
                        logger.error(f"  {line[:200]}")
            except Exception as log_error:
                logger.error(f"Failed to read build log: {log_error}")
        else:
            logger.error("No build log found. The build may not have been executed.")
        
        logger.error("=" * 80)
        logger.warning("Build failed. Attempting to generate stub config headers...")
        logger.error("=" * 80)

        # Try to generate stub headers for missing config files
        try:
            generated_headers = ensure_generated_headers_exist(project_root)
            if generated_headers:
                logger.info(f"Generated {len(generated_headers)} stub config headers")
                for header in generated_headers:
                    logger.info(f"  - {header}")
                logger.info("Retrying with stub headers...")
                # Set empty flags and continue - we'll use stub headers
                auto_detected_flags = []
            else:
                logger.error("Could not generate stub headers")
                raise ObfuscationError(f"Failed to extract compile flags from project: {e}")
        except Exception as stub_error:
            logger.error(f"Failed to generate stub headers: {stub_error}")
            raise ObfuscationError(f"Failed to extract compile flags from project: {e}")
    
    logger.info("")
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 1: RESULTS")
    logger.info("━" * 80)
    
    if auto_detected_flags:
        logger.info("✓ SUCCESS: Build executed and flags extracted")
        logger.info("━" * 80)
        logger.info(f"Total auto-detected flags: {len(auto_detected_flags)}")
        logger.info("")
        logger.info("All auto-detected flags:")
        for idx, flag in enumerate(auto_detected_flags, 1):
            logger.info(f"  [{idx:3d}] {flag}")
        
        # Merge auto-detected flags with user-provided flags
        # Remove duplicates while preserving order
        existing_flags_set = set(compiler_flags)
        new_flags = [flag for flag in auto_detected_flags if flag not in existing_flags_set]
        
        if new_flags:
            compiler_flags = compiler_flags + new_flags
            logger.info("")
            logger.info(f"✓ Merged {len(new_flags)} new flags with existing compiler_flags")
            logger.info(f"  Total compiler_flags now: {len(compiler_flags)}")
        
        # Verify compile_commands.json exists
        cc_path = project_root / "compile_commands.json"
        if cc_path.exists():
            logger.info("")
            logger.info(f"✓ compile_commands.json found at: {cc_path}")
            try:
                import json
                with cc_path.open('r') as f:
                    cc_data = json.load(f)
                logger.info(f"  Entries in compile_commands.json: {len(cc_data)}")
            except:
                pass
        else:
            logger.warning("")
            logger.warning(f"⚠ compile_commands.json NOT found at: {cc_path}")
            logger.warning("  Flags extracted from build output only")
        
        logger.info("=" * 80)
    else:
        logger.warning("=" * 80)
        logger.warning("⚠ No compile flags auto-detected!")
        logger.warning("=" * 80)
        logger.warning("This may cause header file errors during per-TU compilation.")
        logger.warning("The build system was likely not executed properly.")
        logger.warning("")
        logger.warning("Attempting to generate stub config headers...")

        # Try to generate stub headers for missing config files
        generated_headers = ensure_generated_headers_exist(project_root)
        if generated_headers:
            logger.info(f"Generated {len(generated_headers)} stub config headers:")
            for header in generated_headers:
                logger.info(f"  - {header}")
            # Add include paths for generated headers
            for header in generated_headers:
                include_flag = f"-I{header.parent}"
                if include_flag not in compiler_flags:
                    compiler_flags.append(include_flag)
            compiler_flags.append("-DHAVE_CONFIG_H")
            logger.info("Added include paths and -DHAVE_CONFIG_H for stub headers")
        else:
            logger.warning("No stub headers could be generated")
            logger.warning("Compilation may fail due to missing headers")

        logger.warning("")
        logger.warning("For Autotools projects (like curl):")
        logger.warning("  Entrypoint: ./buildconf && ./configure && make")
        logger.warning("For CMake projects:")
        logger.warning("  Entrypoint: cmake -B build && cmake --build build")
        logger.warning("=" * 80)
    
    # Note: Build system diagnostics removed (optional feature)
    logger.info("")
    logger.info("Build system state will be verified during compilation...")
    
    # Determine compiler type for C++ support
    base_compiler = "clang++" if source_abs.suffix in ['.cpp', '.cxx', '.cc', '.c++'] else "clang"
    
    # Get resource directory flags
    resource_dir_flags = get_resource_dir_flag_fn(compiler)
    
    # Determine opt and clang binary paths
    plugin_path_resolved = Path(plugin_path)
    bundled_opt = plugin_path_resolved.parent / "opt"
    bundled_clang = plugin_path_resolved.parent.parent / "bin" / "clang.real"

    # Find opt binary
    opt_binary = None
    if bundled_opt.exists():
        logger.info(f"Using bundled opt: {bundled_opt}")
        opt_binary = bundled_opt

        # Also use bundled clang if available
        if bundled_clang.exists():
            logger.info(f"Using bundled clang from LLVM 22: {bundled_clang}")
            compiler = str(bundled_clang)
        else:
            logger.warning("Bundled clang not found, using system clang (may have version mismatch)")

    elif Path("/usr/local/llvm-obfuscator/lib/opt").exists():
        opt_binary = Path("/usr/local/llvm-obfuscator/lib/opt")
        logger.info(f"Using opt from Docker installation: {opt_binary}")

        # Use bundled clang from Docker installation (LLVM 22) for version compatibility
        docker_clang = Path("/usr/local/llvm-obfuscator/bin/clang.real")
        if docker_clang.exists():
            compiler = str(docker_clang)
            logger.info(f"Using bundled clang from Docker installation (LLVM 22): {compiler}")
        else:
            compiler = "/usr/bin/clang++" if base_compiler == "clang++" else "/usr/bin/clang"
            logger.warning(f"Bundled clang not found, falling back to system clang ({compiler}) - may have version mismatch")
        
    elif "/llvm-project/build/lib/" in str(plugin_path_resolved):
        # Plugin is from LLVM build directory
        llvm_build_dir = plugin_path_resolved.parent.parent
        opt_binary = llvm_build_dir / "bin" / "opt"
        llvm_clang = llvm_build_dir / "bin" / "clang"
        
        if opt_binary.exists():
            logger.info(f"Using opt from LLVM build: {opt_binary}")
            
            if llvm_clang.exists():
                logger.info(f"Using clang from LLVM build: {llvm_clang}")
                compiler = str(llvm_clang)
        else:
            logger.error(
                "OLLVM passes require custom opt binary.\n"
                f"Expected at: {opt_binary}"
            )
            raise ObfuscationError("Custom opt binary not found")
    else:
        logger.warning(
            "Using bundled plugin without bundled opt.\n"
            "Stock LLVM 'opt' does NOT include OLLVM passes."
        )
        # Try known locations
        opt_paths = [
            Path("/Users/akashsingh/Desktop/llvm-project/build/bin/opt"),
            Path("/usr/local/bin/opt"),
            Path("/opt/homebrew/bin/opt"),
        ]
        
        for opt_path in opt_paths:
            if opt_path.exists():
                opt_binary = opt_path
                logger.warning(f"Trying opt at: {opt_binary} (may not have OLLVM passes)")
                break
        
        if not opt_binary:
            logger.error(
                "No opt binary found and plugin needs compatible opt.\n"
                "Stock system LLVM does NOT include OLLVM passes."
            )
            raise ObfuscationError("Compatible opt binary not found")
    
    # Check for llvm-link
    llvm_link_binary = None
    if bundled_opt:
        # Check in same directory as bundled opt
        bundled_llvm_link = bundled_opt.parent / "llvm-link"
        if bundled_llvm_link.exists():
            llvm_link_binary = bundled_llvm_link
            logger.info(f"Using bundled llvm-link: {llvm_link_binary}")
    
    if not llvm_link_binary:
        # Try Docker installation
        if Path("/usr/local/llvm-obfuscator/bin/llvm-link").exists():
            llvm_link_binary = Path("/usr/local/llvm-obfuscator/bin/llvm-link")
            logger.info(f"Using llvm-link from Docker installation: {llvm_link_binary}")
        # Try LLVM build directory
        elif "/llvm-project/build/lib/" in str(plugin_path_resolved):
            llvm_build_dir = plugin_path_resolved.parent.parent
            llvm_link_candidate = llvm_build_dir / "bin" / "llvm-link"
            if llvm_link_candidate.exists():
                llvm_link_binary = llvm_link_candidate
                logger.info(f"Using llvm-link from LLVM build: {llvm_link_binary}")
    
    if not llvm_link_binary:
        # Try to find llvm-link in PATH
        import shutil
        import subprocess
        llvm_link_path = shutil.which("llvm-link")
        if llvm_link_path:
            llvm_link_binary = Path(llvm_link_path)
            logger.info(f"Using llvm-link from PATH: {llvm_link_binary}")

            # Check for version mismatch between clang and llvm-link
            # LLVM 22 bitcode is incompatible with LLVM 19 llvm-link
            try:
                result = subprocess.run([str(llvm_link_binary), '--version'], capture_output=True, text=True, timeout=5)
                llvm_link_version = result.stdout if result.stdout else result.stderr
                if 'LLVM version 19' in llvm_link_version or 'version 19' in llvm_link_version:
                    if '/usr/local/llvm-obfuscator' in str(compiler) or '/app/plugins' in str(compiler):
                        logger.warning("=" * 80)
                        logger.warning("⚠ LLVM VERSION MISMATCH DETECTED!")
                        logger.warning("=" * 80)
                        logger.warning(f"  Compiler (clang): LLVM 22 at {compiler}")
                        logger.warning(f"  Linker (llvm-link): LLVM 19 at {llvm_link_binary}")
                        logger.warning("")
                        logger.warning("LLVM 22 bitcode is incompatible with LLVM 19 llvm-link.")
                        logger.warning("OLLVM passes will be DISABLED for this multi-file build.")
                        logger.warning("Falling back to direct compilation without IR workflow.")
                        logger.warning("=" * 80)

                        # Fall back to direct compilation without OLLVM
                        warnings.append(
                            "LLVM version mismatch: llvm-link (v19) cannot read LLVM 22 bitcode. "
                            "OLLVM passes disabled. Add bundled llvm-link (LLVM 22) to fix."
                        )
                        enabled_passes = []
                        actually_applied_passes = []

                        # Compile all sources directly without IR workflow
                        all_source_paths = [str(src) for src in all_sources]
                        direct_compile_cmd = [compiler] + all_source_paths + ["-o", str(destination_abs)]
                        direct_compile_cmd.extend([f for f in compiler_flags if not f.endswith(('.c', '.cpp', '.cc', '.cxx'))])
                        if resource_dir_flags:
                            direct_compile_cmd.extend(resource_dir_flags)

                        logger.info("Direct compilation command:")
                        logger.info(f"  {' '.join(direct_compile_cmd[:10])} ...")
                        run_command(direct_compile_cmd, cwd=project_root)

                        logger.info("✓ Direct compilation completed (without OLLVM)")

                        return {
                            "applied_passes": [],
                            "warnings": warnings,
                            "disabled_passes": list(enabled_passes) if enabled_passes else [],
                        }
            except Exception as e:
                logger.warning(f"Could not check llvm-link version: {e}")
        else:
            logger.error("llvm-link not found. Required for multi-file obfuscation.")
            raise ObfuscationError("llvm-link binary not found")
    
    # STEP 3: Compile each source file to LLVM bitcode (.bc)
    logger.info("")
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 3: Per-TU Compilation with EXACT FLAGS")
    logger.info("━" * 80)
    logger.info("For each .c file: clang <EXACT FLAGS> -emit-llvm -c file.c -o file.bc")
    logger.info("")
    logger.info("Details:")
    logger.info("  → Each translation unit (TU) compiled separately to LLVM IR (.bc)")
    logger.info("  → Using EXACT compile commands from compile_commands.json")
    logger.info("  → All -I/-D flags extracted from build system in Step 1")
    logger.info("  → Relative include paths resolved from project root")
    logger.info("━" * 80)
    
    # Try to load compile_commands.json for per-TU specific flags
    logger.info("")
    logger.info("Loading compile_commands.json for exact per-file flags...")
    compile_commands_data = {}
    cc_path = project_root / "compile_commands.json"
    logger.info(f"  Looking for: {cc_path}")
    logger.info(f"  Exists: {cc_path.exists()}")
    
    if cc_path.exists():
        try:
            import json
            with cc_path.open('r') as f:
                cc_list = json.load(f)
            # Create a map of file -> entry for quick lookup
            for entry in cc_list:
                file_path = entry.get('file', '')
                compile_commands_data[file_path] = entry
            logger.info(f"  ✓ Loaded compile_commands.json with {len(cc_list)} entries")
            logger.info(f"  ✓ Created lookup table for {len(compile_commands_data)} files")
        except Exception as e:
            logger.error(f"  ✗ Failed to load compile_commands.json: {e}")
            logger.error("  This means we cannot use exact per-file flags!")
    else:
        logger.warning("  ⚠ compile_commands.json NOT FOUND")
        logger.warning("  Will use generic flags (may cause header errors)")
    
    logger.info("")
    logger.info("Beginning per-TU compilation...")
    logger.info("─" * 80)
    
    bc_files: List[Path] = []
    for idx, src_file in enumerate(all_sources):
        bc_file = destination_abs.parent / f"{src_file.stem}_tu{idx}.bc"
        
        logger.info("")
        logger.info(f"[TU {idx+1}/{len(all_sources)}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"  Source: {src_file.name}")
        logger.info(f"  Full path: {src_file}")
        logger.info(f"  Output BC: {bc_file.name}")
        
        # Check if we have specific flags for this file from compile_commands.json
        src_file_str = str(src_file)
        per_file_flags = None
        
        if src_file_str in compile_commands_data:
            logger.info(f"  ✓ Found in compile_commands.json - extracting exact flags")
            entry = compile_commands_data[src_file_str]
            
            # Extract flags from compile command
            if 'command' in entry:
                # Parse command string
                import shlex
                cmd_parts = shlex.split(entry['command'])
                per_file_flags = []
                i = 0
                while i < len(cmd_parts):
                    part = cmd_parts[i]
                    # Skip compiler name and source file
                    if part.endswith(('clang', 'gcc', 'cc', 'c++', 'g++', 'clang++')) or part.endswith(('.c', '.cpp', '.cc', '.cxx')):
                        i += 1
                        continue
                    # Skip -o and output file
                    if part == '-o':
                        i += 2  # Skip -o and next arg
                        continue
                    # Keep all other flags
                    per_file_flags.append(part)
                    i += 1
            elif 'arguments' in entry:
                # Use arguments list
                per_file_flags = [arg for arg in entry['arguments'] 
                                 if not arg.endswith(('.c', '.cpp', '.cc', '.cxx'))
                                 and arg != '-o']
        else:
            logger.warning(f"  ⚠ NOT in compile_commands.json - using auto-detected flags")
            logger.warning(f"     This may cause missing header/macro errors!")
        
        # Build compilation command
        logger.info(f"  Building command: clang <flags> -emit-llvm -c {src_file.name} -o {bc_file.name}")
        ir_cmd = [compiler, str(src_file), "-c", "-emit-llvm", "-o", str(bc_file)]
        
        # Add resource-dir flag if needed
        if resource_dir_flags:
            ir_cmd.extend(resource_dir_flags)
        
        # Add cross-compilation flags (target triple + sysroot for macOS)
        cross_compile_flags = _get_cross_compile_flags(config.platform, config.architecture)
        ir_cmd.extend(cross_compile_flags)
        
        # Use per-file flags if available, otherwise use generic flags
        if per_file_flags:
            # Use exact flags from compile_commands.json
            # Filter out problematic flags
            filtered_flags = [
                flag for flag in per_file_flags
                if not flag.startswith('-Wl,')  # Remove linker flags
                and not flag.startswith('-flto')  # Remove LTO flags
                and flag not in ['-lstdc++', '-c', '-o']  # Remove linking/compilation mode flags
            ]
            ir_cmd.extend(filtered_flags)
            logger.info(f"  Using {len(filtered_flags)} per-file flags from compile_commands.json")
        else:
            # Fall back to generic flags (may not work for complex projects)
            non_linker_flags = [
                flag for flag in compiler_flags 
                if not flag.endswith(('.c', '.cpp', '.cc', '.cxx', '.c++'))
                and not flag.startswith('-Wl,')
                and not flag.startswith('-flto')
                and flag not in ['-lstdc++']
            ]
            ir_cmd.extend(non_linker_flags)
            logger.warning(f"  Using {len(non_linker_flags)} generic flags (may fail for projects needing ./configure)")
        
        logger.info(f"  Working dir: {project_root}")
        logger.info(f"  Flags count: {len(non_linker_flags)}")
        
        # Show first few -I flags for verification
        include_flags = [f for f in non_linker_flags if f.startswith('-I')]
        if include_flags:
            logger.info(f"  Include paths ({len(include_flags)} total):")
            for flag in include_flags[:5]:
                logger.info(f"    {flag}")
            if len(include_flags) > 5:
                logger.info(f"    ... and {len(include_flags) - 5} more")
        else:
            logger.warning(f"  ⚠ NO include paths (-I flags)!")
        
        logger.info(f"  Full command:")
        cmd_str = ' '.join(ir_cmd)
        if len(cmd_str) > 200:
            logger.info(f"    {cmd_str[:200]}...")
            logger.info(f"    (command truncated, {len(cmd_str)} chars total)")
        else:
            logger.info(f"    {cmd_str}")
        
        # Run compilation in project root so relative include paths work correctly
        logger.info(f"  Executing compilation...")
        try:
            run_command(ir_cmd, cwd=project_root)
            logger.info(f"  ✓ SUCCESS - {bc_file.name} created")
            logger.info(f"  ✓ File exists: {bc_file.exists()}")
            if bc_file.exists():
                logger.info(f"  ✓ File size: {bc_file.stat().st_size} bytes")
        except ObfuscationError as e:
            # Provide more helpful error message for header file errors
            error_str = str(e)
            if "fatal error:" in error_str and ".h" in error_str:
                logger.error("=" * 80)
                logger.error("✗ HEADER FILE ERROR DETECTED")
                logger.error("=" * 80)
                logger.error(f"Failed to compile: {src_file.name}")
                logger.error("")
                logger.error("Root cause: The project's build system was not run correctly,")
                logger.error("or compile_commands.json is missing/incomplete.")
                logger.error("")
                logger.error("Expected workflow:")
                logger.error("  1. Fetch files from GitHub")
                logger.error("  2. Apply Layer 1 & 2 obfuscation")
                logger.error("  3. Run build system (./configure && make OR cmake + build)")
                logger.error("  4. Generate compile_commands.json with all -I/-D flags")
                logger.error("  5. Compile each TU to IR using those flags")
                logger.error("")
                logger.error("For Autotools projects (like curl):")
                logger.error("  Entrypoint: ./buildconf && ./configure && make")
                logger.error("")
                logger.error("For CMake projects:")
                logger.error("  Entrypoint: cmake -B build && cmake --build build")
                logger.error("")
                
                # Show which flags were used
                logger.error("Flags that were used for this compilation:")
                for flag in non_linker_flags[:20]:
                    logger.error(f"  {flag}")
                if len(non_linker_flags) > 20:
                    logger.error(f"  ... and {len(non_linker_flags) - 20} more")
                
                logger.error("")
                logger.error(f"Check build.log at: {project_root / 'build.log'}")
                logger.error("=" * 80)
            raise
        
        bc_files.append(bc_file)
    
    logger.info("")
    logger.info("━" * 80)
    logger.info(f"WORKFLOW STEP 3: COMPLETED")
    logger.info(f"  ✓ All {len(bc_files)} translation units compiled to LLVM IR")
    logger.info(f"  ✓ Generated .bc files:")
    for bc in bc_files:
        logger.info(f"     - {bc.name} ({bc.stat().st_size if bc.exists() else 0} bytes)")
    logger.info("━" * 80)
    
    # STEP 4: Link all .bc files into unified.bc using llvm-link
    logger.info("")
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 4: Link all .bc files with llvm-link")
    logger.info("━" * 80)
    
    unified_bc = destination_abs.parent / f"{destination_abs.stem}_unified.bc"
    
    link_cmd = [str(llvm_link_binary)] + [str(bc) for bc in bc_files] + ["-o", str(unified_bc)]
    
    logger.info(f"  Input: {len(bc_files)} .bc files")
    logger.info(f"  Output: {unified_bc.name}")
    logger.info(f"  Command: {' '.join([str(x) for x in link_cmd[:5]])} ...")
    logger.info(f"  Executing llvm-link...")
    
    run_command(link_cmd, cwd=destination_abs.parent)
    
    logger.info(f"  ✓ SUCCESS - unified.bc created")
    logger.info(f"  ✓ File exists: {unified_bc.exists()}")
    if unified_bc.exists():
        logger.info(f"  ✓ File size: {unified_bc.stat().st_size} bytes")
    logger.info("━" * 80)
    
    # Check for C++ exception handling in unified IR - HIKARI APPROACH
    # Only disable flattening, allow other passes (substitution, boguscf, split)
    if has_exception_handling_fn(unified_bc):
        if "flattening" in enabled_passes:
            warning_msg = (
                "C++ exception handling detected in unified IR (invoke/landingpad instructions). "
                "Flattening pass disabled for stability (known to crash on exception handling). "
                "Other OLLVM passes (substitution, boguscf, split) will still be applied. "
                "This is the Hikari-style exception-aware obfuscation approach."
            )
            logger.warning(warning_msg)
            warnings.append(warning_msg)

            # Remove only flattening pass (Hikari approach)
            original_passes = list(enabled_passes)
            enabled_passes = [p for p in enabled_passes if p != "flattening"]
            actually_applied_passes = list(enabled_passes)
            
            logger.info(f"  Original passes: {original_passes}")
            logger.info(f"  After EH check: {enabled_passes}")
            logger.info(f"  Disabled: flattening")
        else:
            # No flattening requested, proceed with all requested passes
            logger.info("C++ exception handling detected, but flattening is not enabled. Proceeding with requested passes.")
    
    # If all passes were disabled (e.g., only flattening was requested), skip OLLVM
    if not enabled_passes:
        logger.warning("No OLLVM passes remaining after exception handling check. Compiling without OLLVM.")
        warnings.append("No OLLVM passes applied (only flattening was requested, but it's incompatible with C++ exception handling)")
        
        # Fall back to standard compilation without OLLVM passes
        command = [compiler, str(unified_bc), "-o", str(destination_abs)]
        
        # Add linker flags (exclude LTO flags to avoid LLVMgold.so dependency)
        linker_flags = [
            flag for flag in compiler_flags
            if (flag.startswith('-Wl,') or flag in ['-lstdc++']) 
            and flag not in ['-flto', '-flto=thin', '-flto=full']
        ]
        command.extend(linker_flags)
        
        if resource_dir_flags:
            command.extend(resource_dir_flags)

        # Add cross-compilation flags (target triple + sysroot for macOS)
        cross_compile_flags = _get_cross_compile_flags(config.platform, config.architecture)
        command.extend(cross_compile_flags)

        logger.info("Compiling unified IR to binary (without OLLVM passes)")
        run_command(command, cwd=destination_abs.parent)
        
        # Cleanup temporary files
        for bc_file in bc_files:
            if bc_file.exists():
                bc_file.unlink()
        if unified_bc.exists():
            unified_bc.unlink()
        
        return {
            "applied_passes": actually_applied_passes,
            "warnings": warnings,
            "disabled_passes": ["flattening"]
        }
    
    # STEP 5: Apply OLLVM passes using opt on unified.bc
    logger.info("")
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 5: Obfuscate with OLLVM passes")
    logger.info("━" * 80)
    
    obfuscated_bc = destination_abs.parent / f"{destination_abs.stem}_obfuscated.bc"
    
    passes_pipeline = ",".join(enabled_passes)
    opt_cmd = [
        str(opt_binary),
        "-load-pass-plugin=" + str(plugin_path),
        f"-passes={passes_pipeline}",
        str(unified_bc),
        "-o", str(obfuscated_bc)
    ]
    
    logger.info(f"  Input: {unified_bc.name}")
    logger.info(f"  Output: {obfuscated_bc.name}")
    logger.info(f"  Passes: {passes_pipeline}")
    logger.info(f"  Plugin: {plugin_path.name}")
    logger.info(f"  Command: {' '.join([str(x) for x in opt_cmd])}")
    logger.info(f"  Executing opt...")
    
    run_command(opt_cmd, cwd=destination_abs.parent)
    
    logger.info(f"  ✓ SUCCESS - obfuscated.bc created")
    logger.info(f"  ✓ File exists: {obfuscated_bc.exists()}")
    if obfuscated_bc.exists():
        logger.info(f"  ✓ File size: {obfuscated_bc.stat().st_size} bytes")
    logger.info("━" * 80)
    
    # STEP 6: Compile obfuscated.bc to final binary
    logger.info("")
    logger.info("━" * 80)
    logger.info("WORKFLOW STEP 6: Recompile final binary")
    logger.info("━" * 80)
    
    # Strip LTO flags if using bundled clang
    final_flags = compiler_flags
    if str(compiler) == str(bundled_clang):
        final_flags = [f for f in compiler_flags if 'lto' not in f.lower()]
        if len(final_flags) != len(compiler_flags):
            logger.info("  → Removed LTO flags (incompatible with bundled clang)")
    
    # Remove source files from flags (we're compiling IR now)
    final_flags = [
        flag for flag in final_flags 
        if not flag.endswith(('.c', '.cpp', '.cc', '.cxx', '.c++'))
    ]
    
    # Remove LTO flags (we're linking an already-obfuscated IR, LTO plugin not needed)
    # LTO requires LLVMgold.so which may not be installed with custom LLVM builds
    final_flags = [
        flag for flag in final_flags
        if flag not in ['-flto', '-flto=thin', '-flto=full']
    ]
    
    final_cmd = [compiler, str(obfuscated_bc), "-o", str(destination_abs)] + final_flags

    # Add cross-compilation flags (target triple + sysroot for macOS)
    cross_compile_flags = _get_cross_compile_flags(config.platform, config.architecture)
    final_cmd.extend(cross_compile_flags)
    
    logger.info(f"  Input: {obfuscated_bc.name}")
    logger.info(f"  Output: {destination_abs.name}")
    logger.info(f"  Compiler: {compiler}")
    logger.info(f"  Final flags: {len(final_flags)}")
    cmd_str = ' '.join([str(x) for x in final_cmd])
    if len(cmd_str) > 200:
        logger.info(f"  Command: {cmd_str[:200]}...")
    else:
        logger.info(f"  Command: {cmd_str}")
    logger.info(f"  Executing final compilation...")
    
    run_command(final_cmd, cwd=destination_abs.parent)
    
    logger.info(f"  ✓ SUCCESS - final binary created")
    logger.info(f"  ✓ Binary exists: {destination_abs.exists()}")
    if destination_abs.exists():
        logger.info(f"  ✓ Binary size: {destination_abs.stat().st_size} bytes")
    logger.info("━" * 80)
    
    # Cleanup temporary files
    logger.info("")
    logger.info("Cleaning up temporary IR files...")
    cleanup_count = 0
    for bc_file in bc_files:
        if bc_file.exists():
            bc_file.unlink()
            cleanup_count += 1
    if unified_bc.exists():
        unified_bc.unlink()
        cleanup_count += 1
    if obfuscated_bc.exists():
        obfuscated_bc.unlink()
        cleanup_count += 1
    logger.info(f"  ✓ Cleaned up {cleanup_count} temporary files")
    
    logger.info("")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " MULTI-FILE IR WORKFLOW COMPLETED SUCCESSFULLY ".center(78) + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  ✓ Built project and captured compile_commands.json")
    logger.info(f"  ✓ Compiled {len(bc_files)} TUs to LLVM IR with exact flags")
    logger.info(f"  ✓ Linked all .bc files into unified module")
    logger.info(f"  ✓ Applied {len(actually_applied_passes)} OLLVM passes")
    logger.info(f"  ✓ Generated final binary: {destination_abs.name}")
    logger.info("")
    
    return {
        "applied_passes": actually_applied_passes,
        "warnings": warnings,
        "disabled_passes": []
    }

