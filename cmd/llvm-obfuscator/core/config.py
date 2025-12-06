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
    ALL = "all"  # Build for all supported platforms

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


class Architecture(str, Enum):
    X86_64 = "x86_64"    # 64-bit Intel/AMD (default)
    ARM64 = "arm64"      # 64-bit ARM (Apple M1/M2, ARM servers)
    X86 = "i686"         # 32-bit Intel/AMD

    @classmethod
    def from_string(cls, value: str) -> "Architecture":
        normalized = value.lower()
        # Handle common aliases
        if normalized in ["amd64", "x64"]:
            normalized = "x86_64"
        elif normalized in ["aarch64", "arm64"]:
            normalized = "arm64"
        elif normalized in ["i386", "i686", "x86", "ia32"]:
            normalized = "i686"
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported architecture: {value}") from exc


class MLIRFrontend(str, Enum):
    """
    MLIR Frontend Selection (ClangIR Pipeline)

    CLANG: Default - Current working pipeline (C/C++ → Clang → LLVM IR → MLIR)
    CLANGIR: New - ClangIR frontend (C/C++ → ClangIR → High-level MLIR) [LLVM 22 native]
    """
    CLANG = "clang"          # DEFAULT - existing pipeline (SAFE)
    CLANGIR = "clangir"      # NEW - ClangIR frontend (LLVM 22 compatible)

    @classmethod
    def from_string(cls, value: str) -> "MLIRFrontend":
        normalized = value.lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported MLIR frontend: {value}. Use 'clang' or 'clangir'.") from exc


class ObfuscationLevel(int, Enum):
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    MAXIMUM = 5


class CryptoHashAlgorithm(str, Enum):
    SHA256 = "sha256"
    BLAKE2B = "blake2b"
    SIPHASH = "siphash"


@dataclass
class CryptoHashConfiguration:
    enabled: bool = False
    algorithm: CryptoHashAlgorithm = CryptoHashAlgorithm.SHA256
    salt: str = ""
    hash_length: int = 12


@dataclass
class PassConfiguration:
    flattening: bool = False
    substitution: bool = False
    bogus_control_flow: bool = False
    split: bool = False
    linear_mba: bool = False
    string_encrypt: bool = False
    symbol_obfuscate: bool = False
    constant_obfuscate: bool = False
    crypto_hash: Optional[CryptoHashConfiguration] = None

    def enabled_passes(self) -> List[str]:
        mapping = {
            "flattening": self.flattening,
            "substitution": self.substitution,
            "boguscf": self.bogus_control_flow,
            "split": self.split,
            "linear-mba": self.linear_mba,
            "string-encrypt": self.string_encrypt,
            "symbol-obfuscate": self.symbol_obfuscate,
            "constant-obfuscate": self.constant_obfuscate,
        }
        passes = [name for name, enabled in mapping.items() if enabled]

        # Add crypto-hash if enabled (replaces symbol-obfuscate)
        if self.crypto_hash and self.crypto_hash.enabled:
            passes.append("crypto-hash")

        return passes

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
    enabled: bool = True  # Enabled by default to show optimization info
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
    # ✅ NEW: IR and advanced metrics analysis options
    preserve_ir: bool = True  # Keep IR files after compilation for analysis
    ir_metrics_enabled: bool = True  # Extract CFG and instruction metrics
    per_pass_metrics: bool = False  # Analyze IR after each pass (expensive)
    binary_analysis_extended: bool = True  # Extended binary structure analysis

@dataclass
class OutputConfiguration:
    directory: Path
    report_formats: List[str] = field(default_factory=lambda: ["json", "markdown", "pdf"])  # json, markdown, pdf

@dataclass
class ObfuscationConfig:
    level: ObfuscationLevel = ObfuscationLevel.MEDIUM
    platform: Platform = Platform.LINUX
    architecture: Architecture = Architecture.X86_64
    compiler_flags: List[str] = field(default_factory=list)
    passes: PassConfiguration = field(default_factory=PassConfiguration)
    advanced: AdvancedConfiguration = field(default_factory=AdvancedConfiguration)
    output: OutputConfiguration = field(default_factory=lambda: OutputConfiguration(Path("./obfuscated")))
    custom_pass_plugin: Optional[Path] = None
    entrypoint_command: Optional[str] = None  # Build command for compile flag extraction
    project_root: Optional[Path] = None  # Root directory for multi-file projects (where entrypoint runs)
    custom_compiler_wrapper: Optional[str] = None  # Path to compiler wrapper (obf-clang) for transparent build interception
    mlir_frontend: MLIRFrontend = MLIRFrontend.CLANG  # DEFAULT to existing pipeline (SAFE)

    @classmethod
    def from_dict(cls, data: Dict) -> "ObfuscationConfig":
        level = ObfuscationLevel(data.get("level", ObfuscationLevel.MEDIUM))
        platform = Platform.from_string(data.get("platform", Platform.LINUX.value))
        architecture = Architecture.from_string(data.get("architecture", Architecture.X86_64.value))
        compiler_flags = data.get("compiler_flags", [])
        passes_data = data.get("passes", {})

        # Parse crypto_hash configuration
        crypto_hash_data = passes_data.get("crypto_hash", {})
        crypto_hash = None
        if crypto_hash_data:
            crypto_hash = CryptoHashConfiguration(
                enabled=crypto_hash_data.get("enabled", False),
                algorithm=CryptoHashAlgorithm(crypto_hash_data.get("algorithm", "sha256")),
                salt=crypto_hash_data.get("salt", ""),
                hash_length=crypto_hash_data.get("hash_length", 12),
            )

        # ✅ FIX: Check both legacy passes.symbol_obfuscate and new symbol_obfuscation.enabled from frontend
        symbol_obfuscate_enabled = passes_data.get("symbol_obfuscate", False)
        symbol_obf_config = data.get("symbol_obfuscation", {})
        print(f"[CONFIG DEBUG] symbol_obf_config from payload: {symbol_obf_config}")
        print(f"[CONFIG DEBUG] symbol_obf_config.get('enabled'): {symbol_obf_config.get('enabled') if isinstance(symbol_obf_config, dict) else 'NOT A DICT'}")
        if isinstance(symbol_obf_config, dict) and symbol_obf_config.get("enabled"):
            symbol_obfuscate_enabled = True
        print(f"[CONFIG DEBUG] Final symbol_obfuscate_enabled: {symbol_obfuscate_enabled}")

        passes = PassConfiguration(
            flattening=passes_data.get("flattening", False),
            substitution=passes_data.get("substitution", False),
            bogus_control_flow=passes_data.get("bogus_control_flow", False),
            split=passes_data.get("split", False),
            linear_mba=passes_data.get("linear_mba", False),
            string_encrypt=passes_data.get("string_encrypt", False),
            symbol_obfuscate=symbol_obfuscate_enabled,
            constant_obfuscate=passes_data.get("constant_obfuscate", False),
            crypto_hash=crypto_hash,
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
            enabled=remarks_data.get("enabled", True),  # Enabled by default
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

        # Parse mlir_frontend (optional, defaults to CLANG for backward compatibility)
        mlir_frontend_str = data.get("mlir_frontend", "clang")
        mlir_frontend = MLIRFrontend.from_string(mlir_frontend_str)

        return cls(
            level=level,
            platform=platform,
            architecture=architecture,
            compiler_flags=compiler_flags,
            passes=passes,
            advanced=advanced,
            output=output,
            custom_pass_plugin=custom_pass_plugin,
            entrypoint_command=entrypoint_command,
            project_root=project_root,
            custom_compiler_wrapper=custom_compiler_wrapper,
            mlir_frontend=mlir_frontend,
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