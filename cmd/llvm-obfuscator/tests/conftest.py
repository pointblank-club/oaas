import base64
from pathlib import Path
from typing import Callable

import pytest

from core import LLVMObfuscator, ObfuscationConfig, ObfuscationLevel, OutputConfiguration, PassConfiguration, Platform
from core.config import AdvancedConfiguration
from core.reporter import ObfuscationReport


@pytest.fixture
def sample_source(tmp_path: Path) -> Path:
    source = tmp_path / "sample.c"
    source.write_text("""
    #include <stdio.h>
    int secret() { return 42; }
    int main() { printf("%s", "secret"); return 0; }
    """, encoding="utf-8")
    return source


@pytest.fixture
def obfuscation_config(tmp_path: Path) -> ObfuscationConfig:
    return ObfuscationConfig(
        level=ObfuscationLevel.MEDIUM,
        platform=Platform.LINUX,
        compiler_flags=["-g"],
        passes=PassConfiguration(
            flattening=True,
            substitution=True,
            bogus_control_flow=True,
            split=True,
            string_encrypt=True, 
            symbol_obfuscate=True
        ),
        advanced=AdvancedConfiguration(cycles=1, fake_loops=2),
        output=OutputConfiguration(directory=tmp_path / "out", report_formats=["json"]),
    )


@pytest.fixture
def obfuscator(tmp_path: Path) -> LLVMObfuscator:
    reporter = ObfuscationReport(tmp_path / "reports")
    return LLVMObfuscator(reporter=reporter)


@pytest.fixture(autouse=True)
def patch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.obfuscator import LLVMObfuscator, ObfuscationConfig
    from core import utils

    def fake_compile(self: LLVMObfuscator, source: Path, destination: Path, config: ObfuscationConfig, *_, **__):
        
        mlir_passes = [p for p in config.passes.enabled_passes() if p in ["string-encrypt", "symbol-obfuscate"]]
        ollvm_passes = [p for p in config.passes.enabled_passes() if p not in mlir_passes]

        if mlir_passes:
            mlir_file = destination.parent / f"{destination.stem}_temp.mlir"
            mlir_file.touch()
            obfuscated_mlir = destination.parent / f"{destination.stem}_obfuscated.mlir"
            obfuscated_mlir.touch()
            llvm_ir_file = destination.parent / f"{destination.stem}_from_mlir.ll"
            llvm_ir_file.touch()

        if ollvm_passes:
            obfuscated_ir = destination.parent / f"{destination.stem}_obfuscated.bc"
            obfuscated_ir.touch()

        destination.write_bytes(b"\x7fELFfakebinary")
        
        return {"applied_passes": config.passes.enabled_passes(), "warnings": [], "disabled_passes": []}

    def fake_run_command(*args, **kwargs):
        return 0, "", ""

    monkeypatch.setattr(LLVMObfuscator, "_compile", fake_compile)
    monkeypatch.setattr(utils, "run_command", fake_run_command)

    from core.obfuscator import require_tool

    monkeypatch.setattr("core.obfuscator.require_tool", lambda *args, **kwargs: None)


@pytest.fixture
def base64_source(sample_source: Path) -> str:
    return base64.b64encode(sample_source.read_bytes()).decode("ascii")


@pytest.fixture(autouse=True)
def set_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBFUSCATOR_API_KEY", "test-key")
