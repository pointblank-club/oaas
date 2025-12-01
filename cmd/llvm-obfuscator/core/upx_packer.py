"""UPX Packer - Additional binary compression and obfuscation layer."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .utils import create_logger, get_file_size

logger = logging.getLogger(__name__)


class UPXPacker:
    """
    UPX (Ultimate Packer for eXecutables) integration.
    
    UPX provides:
    - 50-70% binary size reduction
    - Additional obfuscation layer (packed executable)
    - Runtime decompression (transparent to user)
    - Cross-platform support (Linux, Windows, macOS)
    
    Benefits for obfuscation:
    - Makes static analysis harder (binary is compressed)
    - Reduces binary size (important since obfuscation increases size)
    - Adds another layer of protection
    
    Trade-offs:
    - Small runtime overhead (decompression at startup)
    - Some antivirus may flag UPX-packed binaries
    - Can be unpacked with `upx -d`, but still adds friction
    """
    
    COMPRESSION_LEVELS = {
        "fast": ["--fast"],
        "default": [],
        "best": ["--best"],
        "brute": ["--brute"],  # Very slow but maximum compression
    }
    
    def __init__(self):
        self.logger = create_logger(__name__)
        self.upx_available = self._check_upx_available()
    
    def _check_upx_available(self) -> bool:
        """Check if UPX is installed and available."""
        return shutil.which("upx") is not None
    
    def pack(
        self,
        binary_path: Path,
        compression_level: str = "best",
        use_lzma: bool = True,
        force: bool = True,
        preserve_original: bool = True,
    ) -> Optional[Dict]:
        """
        Pack a binary using UPX.
        
        Args:
            binary_path: Path to binary to pack
            compression_level: Compression level (fast, default, best, brute)
            use_lzma: Use LZMA compression (better ratio, slightly slower decompression)
            force: Force packing even if already packed
            preserve_original: Keep backup of original binary
        
        Returns:
            Dict with packing results or None if packing failed/skipped
        """
        if not self.upx_available:
            self.logger.warning("UPX not installed. Install with: apt install upx-ucl (Linux) or brew install upx (macOS)")
            return None
        
        if not binary_path.exists():
            self.logger.error(f"Binary not found: {binary_path}")
            return None
        
        # Check if binary is already packed
        if self._is_packed(binary_path) and not force:
            self.logger.info(f"Binary already UPX-packed: {binary_path}")
            return {"status": "already_packed", "skipped": True}
        
        # Get original size
        original_size = get_file_size(binary_path)
        
        # Create backup if requested
        backup_path = None
        if preserve_original:
            backup_path = binary_path.with_suffix(binary_path.suffix + ".pre-upx")
            shutil.copy2(binary_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
        
        # Build UPX command
        cmd = ["upx"]
        
        # Add compression level flags
        if compression_level in self.COMPRESSION_LEVELS:
            cmd.extend(self.COMPRESSION_LEVELS[compression_level])
        else:
            self.logger.warning(f"Unknown compression level '{compression_level}', using 'best'")
            cmd.extend(self.COMPRESSION_LEVELS["best"])
        
        # Add LZMA compression (better compression ratio)
        if use_lzma:
            cmd.append("--lzma")
        
        # Force packing (overwrite if already packed)
        if force:
            cmd.append("--force")
        
        # Add binary path
        cmd.append(str(binary_path))
        
        try:
            # Run UPX
            self.logger.info(f"Packing binary with UPX: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            # Check if packing succeeded
            if result.returncode != 0:
                # Restore backup if packing failed
                if backup_path and backup_path.exists():
                    shutil.move(str(backup_path), str(binary_path))
                    self.logger.debug("Restored original binary after UPX failure")
                
                # UPX may fail on certain binaries (statically linked, certain formats, etc.)
                self.logger.warning(f"UPX packing failed (exit code {result.returncode})")
                self.logger.debug(f"UPX stdout: {result.stdout}")
                self.logger.debug(f"UPX stderr: {result.stderr}")
                
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "original_size": original_size,
                }
            
            # Get packed size
            packed_size = get_file_size(binary_path)
            compression_ratio = (1 - packed_size / original_size) * 100 if original_size > 0 else 0
            
            self.logger.info(
                f"UPX packing successful: {original_size} → {packed_size} bytes "
                f"({compression_ratio:.1f}% reduction)"
            )
            
            # Clean up backup if packing succeeded (unless preserve_original=True)
            if backup_path and backup_path.exists() and not preserve_original:
                backup_path.unlink()
            
            return {
                "status": "success",
                "original_size": original_size,
                "packed_size": packed_size,
                "compression_ratio": round(compression_ratio, 2),
                "compression_level": compression_level,
                "lzma": use_lzma,
                "backup_path": str(backup_path) if preserve_original and backup_path else None,
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error("UPX packing timed out (> 5 minutes)")
            if backup_path and backup_path.exists():
                shutil.move(str(backup_path), str(binary_path))
            return {"status": "timeout"}
        
        except Exception as e:
            self.logger.error(f"UPX packing failed with exception: {e}")
            if backup_path and backup_path.exists():
                shutil.move(str(backup_path), str(binary_path))
            return {"status": "error", "error": str(e)}
    
    def _is_packed(self, binary_path: Path) -> bool:
        """Check if a binary is already UPX-packed."""
        try:
            # UPX adds a "UPX!" signature to packed binaries
            with open(binary_path, "rb") as f:
                content = f.read(1024)  # Check first 1KB
                return b"UPX!" in content
        except Exception as e:
            self.logger.debug(f"Could not check if binary is packed: {e}")
            return False
    
    def unpack(self, binary_path: Path) -> bool:
        """
        Unpack a UPX-packed binary.
        
        Args:
            binary_path: Path to packed binary
        
        Returns:
            True if unpacking succeeded, False otherwise
        """
        if not self.upx_available:
            self.logger.warning("UPX not installed")
            return False
        
        if not self._is_packed(binary_path):
            self.logger.warning(f"Binary is not UPX-packed: {binary_path}")
            return False
        
        try:
            cmd = ["upx", "-d", str(binary_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully unpacked: {binary_path}")
                return True
            else:
                self.logger.error(f"Unpacking failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Unpacking failed with exception: {e}")
            return False
    
    def test_packed(self, binary_path: Path) -> bool:
        """
        Test if a packed binary is valid (can be unpacked).
        
        Args:
            binary_path: Path to packed binary
        
        Returns:
            True if binary is valid, False otherwise
        """
        if not self.upx_available:
            return False
        
        try:
            cmd = ["upx", "-t", str(binary_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            self.logger.debug(f"UPX test failed: {e}")
            return False

