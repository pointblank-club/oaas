from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class Platform(str, Enum):
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    DARWIN = "darwin"  # Alias for macOS

    @classmethod
    def from_string(cls, value: str) -> "Platform":
        normalized = value.lower()
        # Handle darwin as alias for macos
        if normalized == "darwin":
            normalized = "macos"
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported platform: {value}") from exc


class ObfuscationLevel(int, Enum):
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    MAXIMUM = 5


@dataclass
class PassConfiguration:
    flattening: bool = False
    substitution: bool = False
    bogus_control_flow: bool = False
    split: bool = False
    linear_mba: bool = False
    string_encrypt: bool = False
    symbol_obfuscate: bool = False

    def enabled_passes(self) -> List[str]:
        mapping = {
            "flattening": self.flattening,
            "substitution": self.substitution,
            "boguscf": self.bogus_control_flow,
            "split": self.split,
            "linear-mba": self.linear_mba,
            "string-encrypt": self.string_encrypt,
            "symbol-obfuscate": self.symbol_obfuscate,
        }
        return [name for name, enabled in mapping.items() if enabled]

@dataclass
class IndirectCallConfiguration:
    enabled: bool = False
    obfuscate_stdlib: bool = True
    obfuscate_custom: bool = True


@dataclass
class UPXConfiguration:
    enabled: bool = False
    compression_level: str = "best"  # fast, default, best, brute
    use_lzma: bool = True
    preserve_original: bool = False


@dataclass
class RemarksConfiguration:
    enabled: bool = False
    format: str = "yaml"  # yaml or bitstream
    output_file: Optional[str] = None  # If None, auto-generate
    pass_filter: str = ".*"  # Regex filter for passes
    with_hotness: bool = False  # Include profile hotness (requires PGO)


@dataclass
class AdvancedConfiguration:
    cycles: int = 1
    fake_loops: int = 0
    indirect_calls: IndirectCallConfiguration = field(default_factory=IndirectCallConfiguration)
    remarks: RemarksConfiguration = field(default_factory=RemarksConfiguration)
    upx_packing: UPXConfiguration = field(default_factory=UPXConfiguration)

@dataclass
class OutputConfiguration:
    directory: Path
    report_formats: List[str] = field(default_factory=lambda: ["json"])  # json, html, pdf

@dataclass
class ObfuscationConfig:
    level: ObfuscationLevel = ObfuscationLevel.MEDIUM
    platform: Platform = Platform.LINUX
    compiler_flags: List[str] = field(default_factory=list)
    passes: PassConfiguration = field(default_factory=PassConfiguration)
    advanced: AdvancedConfiguration = field(default_factory=AdvancedConfiguration)
    output: OutputConfiguration = field(default_factory=lambda: OutputConfiguration(Path("./obfuscated")))
    custom_pass_plugin: Optional[Path] = None
    entrypoint_command: Optional[str] = None  # Build command for compile flag extraction
    project_root: Optional[Path] = None  # Root directory for multi-file projects (where entrypoint runs)
    custom_compiler_wrapper: Optional[str] = None  # Path to compiler wrapper (obf-clang) for transparent build interception

    @classmethod
    def from_dict(cls, data: Dict) -> "ObfuscationConfig":
        level = ObfuscationLevel(data.get("level", ObfuscationLevel.MEDIUM))
        platform = Platform.from_string(data.get("platform", Platform.LINUX.value))
        compiler_flags = data.get("compiler_flags", [])
        passes_data = data.get("passes", {})
        passes = PassConfiguration(
            flattening=passes_data.get("flattening", False),
            substitution=passes_data.get("substitution", False),
            bogus_control_flow=passes_data.get("bogus_control_flow", False),
            split=passes_data.get("split", False),
            linear_mba=passes_data.get("linear_mba", False),
            string_encrypt=passes_data.get("string_encrypt", False),
            symbol_obfuscate=passes_data.get("symbol_obfuscate", False),
        )
        adv_data = data.get("advanced", {})

        # Parse indirect calls configuration
        indirect_data = adv_data.get("indirect_calls", {})
        indirect_calls = IndirectCallConfiguration(
            enabled=indirect_data.get("enabled", False),
            obfuscate_stdlib=indirect_data.get("obfuscate_stdlib", True),
            obfuscate_custom=indirect_data.get("obfuscate_custom", True),
        )

        remarks_data = adv_data.get("remarks", {})
        remarks_config = RemarksConfiguration(
            enabled=remarks_data.get("enabled", False),
            format=remarks_data.get("format", "yaml"),
            output_file=remarks_data.get("output_file"),
            pass_filter=remarks_data.get("pass_filter", ".*"),
            with_hotness=remarks_data.get("with_hotness", False),
        )

        upx_data = adv_data.get("upx_packing", {})
        upx_config = UPXConfiguration(
            enabled=upx_data.get("enabled", False),
            compression_level=upx_data.get("compression_level", "best"),
            use_lzma=upx_data.get("use_lzma", True),
            preserve_original=upx_data.get("preserve_original", False),
        )
        advanced = AdvancedConfiguration(
            cycles=adv_data.get("cycles", 1),
            fake_loops=adv_data.get("fake_loops", 0),
            indirect_calls=indirect_calls,
            remarks=remarks_config,
            upx_packing=upx_config,
        )
        output_data = data.get("output", {})
        output = OutputConfiguration(
            directory=Path(output_data.get("directory", "./obfuscated")),
            report_formats=output_data.get("report_format", ["json"]),
        )
        custom_pass_plugin = data.get("custom_pass_plugin")
        if custom_pass_plugin:
            custom_pass_plugin = Path(custom_pass_plugin)
        entrypoint_command = data.get("entrypoint_command")
        project_root = data.get("project_root")
        if project_root:
            project_root = Path(project_root)
        custom_compiler_wrapper = data.get("custom_compiler_wrapper")
        return cls(
            level=level,
            platform=platform,
            compiler_flags=compiler_flags,
            passes=passes,
            advanced=advanced,
            output=output,
            custom_pass_plugin=custom_pass_plugin,
            entrypoint_command=entrypoint_command,
            project_root=project_root,
            custom_compiler_wrapper=custom_compiler_wrapper,
        )


@dataclass
class AnalyzeConfig:
    binary_path: Path
    output: Optional[Path] = None


@dataclass
class CompareConfig:
    original_binary: Path
    obfuscated_binary: Path
    output: Optional[Path] = None