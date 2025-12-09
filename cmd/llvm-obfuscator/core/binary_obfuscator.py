"""
Binary Obfuscation Module

Obfuscates already-compiled binaries by:
1. Converting binary to LLVM IR
2. Applying obfuscation passes to IR
3. Recompiling IR back to binary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from .config import ObfuscationConfig, Platform
from .exceptions import ObfuscationError
from .utils import create_logger, run_command, ensure_directory, require_tool


logger = create_logger(__name__)


class BinaryObfuscator:
    """Obfuscates already-compiled binaries."""
    
    def __init__(self):
        self.logger = create_logger(__name__)
    
    def obfuscate_binary(
        self,
        binary_path: Path,
        output_binary: Path,
        config: ObfuscationConfig,
    ) -> Dict:
        """
        Obfuscate an already-compiled binary.
        
        Process:
        1. Convert binary to LLVM IR (using llvm-dis or similar)
        2. Apply obfuscation passes to IR
        3. Recompile IR back to binary
        
        Args:
            binary_path: Path to input binary
            output_binary: Path to output obfuscated binary
            config: Obfuscation configuration
            
        Returns:
            Dict with obfuscation results
        """
        if not binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")
        
        ensure_directory(output_binary.parent)
        
        # For now, we'll use a workaround:
        # Since LLVM obfuscation works best on source/IR, we'll need to
        # either disassemble the binary or use a different approach
        
        # Check if we have llvm-dis (LLVM disassembler)
        # If binary was compiled with debug info, we might be able to extract IR
        # Otherwise, we need to use objdump/llvm-objdump to get assembly
        
        self.logger.info(f"Obfuscating binary: {binary_path.name}")
        
        # Method 1: Try to extract IR if binary has LLVM bitcode embedded
        # (This requires special compilation flags)
        
        # Method 2: Use llvm-objdump to disassemble, then reassemble with obfuscation
        # This is more complex and may not preserve all semantics
        
        # For now, we'll use a simpler approach:
        # Note: True binary obfuscation (without source) is limited.
        # The current obfuscator is designed for source->binary obfuscation.
        
        # We can use UPX packing as a form of binary obfuscation
        from .upx_packer import UPXPacker
        
        if config.advanced.upx_packing.enabled:
            upx = UPXPacker(custom_upx_path=config.advanced.upx_packing.custom_upx_path)
            try:
                result = upx.pack(
                    binary_path=binary_path,
                    compression_level=config.advanced.upx_packing.compression_level,
                    use_lzma=config.advanced.upx_packing.use_lzma,
                    force=True,
                    preserve_original=config.advanced.upx_packing.preserve_original,
                )
                if result and result.get("status") == "success":
                    if binary_path != output_binary:
                        import shutil
                        shutil.copy2(binary_path, output_binary)
                    self.logger.info("Binary packed with UPX")
                    return {"success": True, "method": "upx_packing"}
            except Exception as e:
                self.logger.warning(f"UPX packing failed: {e}")
        
        # If UPX not enabled or failed, we need source code to properly obfuscate
        # For true binary obfuscation without source, we'd need specialized tools
        raise ObfuscationError(
            "Binary obfuscation without source code is limited. "
            "The LLVM obfuscator is designed to work on source code or LLVM IR. "
            "To obfuscate binaries, either:\n"
            "1. Use source code obfuscation (recommended)\n"
            "2. Enable UPX packing for binary-level obfuscation\n"
            "3. Use specialized binary obfuscation tools"
        )

