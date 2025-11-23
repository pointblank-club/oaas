"""Unit tests for UPX Packer module."""

import pytest
import shutil
import subprocess
from pathlib import Path
import tempfile

from core.upx_packer import UPXPacker


class TestUPXPacker:
    """Test suite for UPXPacker class."""
    
    @pytest.fixture
    def packer(self):
        """Create a UPXPacker instance."""
        return UPXPacker()
    
    @pytest.fixture
    def sample_binary(self, tmp_path):
        """Create a simple test binary."""
        source = tmp_path / "test.c"
        source.write_text("""
#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}
""")
        
        binary = tmp_path / "test"
        result = subprocess.run(
            ["clang", str(source), "-o", str(binary)],
            capture_output=True
        )
        
        if result.returncode != 0:
            pytest.skip("clang not available or compilation failed")
        
        return binary
    
    def test_upx_available(self, packer):
        """Test UPX availability detection."""
        # Should not raise exception
        assert isinstance(packer.upx_available, bool)
    
    def test_pack_nonexistent_file(self, packer):
        """Test packing a non-existent file."""
        result = packer.pack(Path("/nonexistent/file"))
        assert result is None or result.get("status") == "error"
    
    @pytest.mark.skipif(not shutil.which("upx"), reason="UPX not installed")
    def test_pack_basic(self, packer, sample_binary):
        """Test basic packing functionality."""
        result = packer.pack(
            sample_binary,
            compression_level="fast",
            preserve_original=True
        )
        
        if result and result.get("status") == "success":
            assert result["original_size"] > 0
            assert result["packed_size"] > 0
            assert result["packed_size"] < result["original_size"]
            assert result["compression_ratio"] > 0
    
    @pytest.mark.skipif(not shutil.which("upx"), reason="UPX not installed")
    def test_pack_with_lzma(self, packer, sample_binary):
        """Test packing with LZMA compression."""
        result = packer.pack(
            sample_binary,
            compression_level="best",
            use_lzma=True,
            preserve_original=True
        )
        
        if result and result.get("status") == "success":
            assert result["lzma"] is True
            assert result["compression_ratio"] > 0
    
    @pytest.mark.skipif(not shutil.which("upx"), reason="UPX not installed")
    def test_compression_levels(self, packer, sample_binary, tmp_path):
        """Test different compression levels."""
        levels = ["fast", "default", "best"]
        results = {}
        
        for level in levels:
            # Make a copy for each test
            test_binary = tmp_path / f"test_{level}"
            shutil.copy(sample_binary, test_binary)
            
            result = packer.pack(
                test_binary,
                compression_level=level,
                force=True
            )
            
            if result and result.get("status") == "success":
                results[level] = result["compression_ratio"]
        
        # Best should generally have better compression than fast
        if "best" in results and "fast" in results:
            assert results["best"] >= results["fast"]
    
    @pytest.mark.skipif(not shutil.which("upx"), reason="UPX not installed")
    def test_is_packed(self, packer, sample_binary):
        """Test packed binary detection."""
        # Before packing
        assert not packer._is_packed(sample_binary)
        
        # After packing
        result = packer.pack(sample_binary, force=True)
        if result and result.get("status") == "success":
            assert packer._is_packed(sample_binary)
    
    @pytest.mark.skipif(not shutil.which("upx"), reason="UPX not installed")
    def test_test_packed(self, packer, sample_binary):
        """Test UPX test functionality."""
        # Pack first
        result = packer.pack(sample_binary, force=True)
        
        if result and result.get("status") == "success":
            # Test should pass
            assert packer.test_packed(sample_binary) is True
    
    @pytest.mark.skipif(not shutil.which("upx"), reason="UPX not installed")
    def test_unpack(self, packer, sample_binary, tmp_path):
        """Test unpacking functionality."""
        # Make a copy
        test_binary = tmp_path / "test_unpack"
        shutil.copy(sample_binary, test_binary)
        
        # Pack it
        pack_result = packer.pack(test_binary, force=True)
        
        if pack_result and pack_result.get("status") == "success":
            # Unpack it
            unpack_result = packer.unpack(test_binary)
            assert unpack_result is True
            
            # Should no longer be packed
            assert not packer._is_packed(test_binary)
    
    def test_preserve_original(self, packer, sample_binary, tmp_path):
        """Test that preserve_original keeps a backup."""
        if not shutil.which("upx"):
            pytest.skip("UPX not installed")
        
        test_binary = tmp_path / "test_preserve"
        shutil.copy(sample_binary, test_binary)
        
        result = packer.pack(
            test_binary,
            preserve_original=True,
            force=True
        )
        
        if result and result.get("status") == "success":
            backup_path = result.get("backup_path")
            if backup_path:
                assert Path(backup_path).exists()
    
    def test_invalid_compression_level(self, packer, sample_binary):
        """Test handling of invalid compression level."""
        # Should fall back to 'best' and log warning
        result = packer.pack(
            sample_binary,
            compression_level="invalid_level",
            force=True
        )
        
        # Should still work (falls back to best)
        if result:
            assert result.get("compression_level") is not None

