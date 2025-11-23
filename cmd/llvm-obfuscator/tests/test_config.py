"""Unit tests for configuration module."""

import pytest
from pathlib import Path

from core.config import (
    Platform,
    ObfuscationLevel,
    PassConfiguration,
    SymbolObfuscationConfiguration,
    UPXConfiguration,
    AdvancedConfiguration,
    OutputConfiguration,
    ObfuscationConfig,
)


class TestPlatform:
    """Test Platform enum."""
    
    def test_platform_values(self):
        """Test platform enum values."""
        assert Platform.LINUX.value == "linux"
        assert Platform.WINDOWS.value == "windows"
        assert Platform.MACOS.value == "macos"
        assert Platform.DARWIN.value == "darwin"
    
    def test_platform_from_string(self):
        """Test platform creation from string."""
        assert Platform.from_string("linux") == Platform.LINUX
        assert Platform.from_string("WINDOWS") == Platform.WINDOWS
        assert Platform.from_string("darwin") == Platform.MACOS
    
    def test_platform_invalid_string(self):
        """Test invalid platform string."""
        with pytest.raises(ValueError):
            Platform.from_string("invalid")


class TestObfuscationLevel:
    """Test ObfuscationLevel enum."""
    
    def test_levels(self):
        """Test obfuscation levels."""
        assert ObfuscationLevel.MINIMAL == 1
        assert ObfuscationLevel.LOW == 2
        assert ObfuscationLevel.MEDIUM == 3
        assert ObfuscationLevel.HIGH == 4
        assert ObfuscationLevel.MAXIMUM == 5


class TestPassConfiguration:
    """Test PassConfiguration."""
    
    def test_default_passes(self):
        """Test default pass configuration."""
        passes = PassConfiguration()
        assert passes.flattening is False
        assert passes.substitution is False
        assert passes.bogus_control_flow is False
        assert passes.split is False
        assert passes.linear_mba is False
    
    def test_enabled_passes_none(self):
        """Test no enabled passes."""
        passes = PassConfiguration()
        assert passes.enabled_passes() == []
    
    def test_enabled_passes_all(self):
        """Test all enabled passes."""
        passes = PassConfiguration(
            flattening=True,
            substitution=True,
            bogus_control_flow=True,
            split=True,
            linear_mba=True
        )
        enabled = passes.enabled_passes()
        assert "flattening" in enabled
        assert "substitution" in enabled
        assert "boguscf" in enabled
        assert "split" in enabled
        assert "linear-mba" in enabled
    
    def test_enabled_passes_partial(self):
        """Test partially enabled passes."""
        passes = PassConfiguration(
            flattening=True,
            bogus_control_flow=True
        )
        enabled = passes.enabled_passes()
        assert len(enabled) == 2
        assert "flattening" in enabled
        assert "boguscf" in enabled


class TestSymbolObfuscationConfiguration:
    """Test SymbolObfuscationConfiguration."""
    
    def test_defaults(self):
        """Test default symbol obfuscation config."""
        config = SymbolObfuscationConfiguration()
        assert config.enabled is False
        assert config.algorithm == "sha256"
        assert config.hash_length == 12
        assert config.prefix_style == "typed"
        assert config.salt is None
        assert config.preserve_main is True
        assert config.preserve_stdlib is True
    
    def test_custom_values(self):
        """Test custom symbol obfuscation config."""
        config = SymbolObfuscationConfiguration(
            enabled=True,
            algorithm="blake2b",
            hash_length=16,
            prefix_style="underscore",
            salt="custom_salt",
            preserve_main=False
        )
        assert config.enabled is True
        assert config.algorithm == "blake2b"
        assert config.hash_length == 16
        assert config.prefix_style == "underscore"
        assert config.salt == "custom_salt"
        assert config.preserve_main is False


class TestUPXConfiguration:
    """Test UPXConfiguration."""
    
    def test_defaults(self):
        """Test default UPX config."""
        config = UPXConfiguration()
        assert config.enabled is False
        assert config.compression_level == "best"
        assert config.use_lzma is True
        assert config.preserve_original is False
    
    def test_custom_values(self):
        """Test custom UPX config."""
        config = UPXConfiguration(
            enabled=True,
            compression_level="brute",
            use_lzma=False,
            preserve_original=True
        )
        assert config.enabled is True
        assert config.compression_level == "brute"
        assert config.use_lzma is False
        assert config.preserve_original is True


class TestAdvancedConfiguration:
    """Test AdvancedConfiguration."""
    
    def test_defaults(self):
        """Test default advanced config."""
        config = AdvancedConfiguration()
        assert config.cycles == 1
        assert config.string_encryption is False
        assert config.fake_loops == 0
        assert isinstance(config.symbol_obfuscation, SymbolObfuscationConfiguration)
        assert isinstance(config.upx_packing, UPXConfiguration)
    
    def test_custom_values(self):
        """Test custom advanced config."""
        symbol_config = SymbolObfuscationConfiguration(enabled=True)
        upx_config = UPXConfiguration(enabled=True)
        
        config = AdvancedConfiguration(
            cycles=3,
            string_encryption=True,
            fake_loops=10,
            symbol_obfuscation=symbol_config,
            upx_packing=upx_config
        )
        
        assert config.cycles == 3
        assert config.string_encryption is True
        assert config.fake_loops == 10
        assert config.symbol_obfuscation.enabled is True
        assert config.upx_packing.enabled is True


class TestOutputConfiguration:
    """Test OutputConfiguration."""
    
    def test_defaults(self):
        """Test default output config."""
        config = OutputConfiguration(directory=Path("/tmp/test"))
        assert config.directory == Path("/tmp/test")
        assert config.report_formats == ["json"]
    
    def test_custom_formats(self):
        """Test custom report formats."""
        config = OutputConfiguration(
            directory=Path("/tmp/test"),
            report_formats=["json", "html", "markdown"]
        )
        assert len(config.report_formats) == 3
        assert "json" in config.report_formats
        assert "html" in config.report_formats


class TestObfuscationConfig:
    """Test ObfuscationConfig."""
    
    def test_defaults(self):
        """Test default obfuscation config."""
        config = ObfuscationConfig()
        assert config.level == ObfuscationLevel.MEDIUM
        assert config.platform == Platform.LINUX
        assert config.compiler_flags == []
        assert isinstance(config.passes, PassConfiguration)
        assert isinstance(config.advanced, AdvancedConfiguration)
        assert isinstance(config.output, OutputConfiguration)
        assert config.custom_pass_plugin is None
    
    def test_from_dict_basic(self):
        """Test creating config from dict."""
        data = {
            "level": 3,
            "platform": "linux",
            "compiler_flags": ["-O3", "-flto"]
        }
        
        config = ObfuscationConfig.from_dict(data)
        assert config.level == ObfuscationLevel.MEDIUM
        assert config.platform == Platform.LINUX
        assert config.compiler_flags == ["-O3", "-flto"]
    
    def test_from_dict_complete(self):
        """Test creating complete config from dict."""
        data = {
            "level": 4,
            "platform": "linux",
            "compiler_flags": ["-O2"],
            "passes": {
                "flattening": True,
                "bogus_control_flow": True
            },
            "advanced": {
                "cycles": 2,
                "string_encryption": True,
                "fake_loops": 5,
                "upx_packing": {
                    "enabled": True,
                    "compression_level": "brute"
                }
            },
            "output": {
                "directory": "/tmp/output",
                "report_format": ["json", "html"]
            },
            "custom_pass_plugin": "/path/to/plugin.so"
        }
        
        config = ObfuscationConfig.from_dict(data)
        assert config.level == ObfuscationLevel.HIGH
        assert config.platform == Platform.LINUX
        assert config.passes.flattening is True
        assert config.passes.bogus_control_flow is True
        assert config.advanced.cycles == 2
        assert config.advanced.string_encryption is True
        assert config.custom_pass_plugin == Path("/path/to/plugin.so")

