"""
Binary Obfuscation Pipeline Worker

Orchestrates the complete pipeline for binary-to-binary obfuscation:
1. Ghidra CFG Export
2. McSema LLVM Lifting
3. LLVM 22 IR Upgrade
4. OLLVM Pass Application
5. Final Binary Compilation
6. Metrics Generation
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PipelineLogger:
    """Aggregates logs from all pipeline stages."""

    def __init__(self, log_file_path: str):
        self.log_file = Path(log_file_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Write timestamped log message."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] [{level}] {message}\n"

        with open(self.log_file, "a") as f:
            f.write(log_line)

        if level == "ERROR":
            logger.error(message)
        elif level == "WARN":
            logger.warning(message)
        else:
            logger.info(message)

    def log_command(self, cmd: str, output: str, returncode: int):
        """Log command execution with output."""
        self.log(f"$ {cmd}")
        if output.strip():
            for line in output.strip().split("\n"):
                self.log(f"  {line}")
        if returncode != 0:
            self.log(f"Command failed with exit code {returncode}", level="ERROR")

    def read_logs(self) -> str:
        """Read all accumulated logs."""
        if self.log_file.exists():
            return self.log_file.read_text()
        return ""


class BinaryPipelineWorker:
    """Executes the complete binary obfuscation pipeline."""

    def __init__(self, job_dir: str, passes_config: Dict[str, bool]):
        self.job_dir = Path(job_dir)
        self.passes_config = passes_config
        self.logger = PipelineLogger(str(self.job_dir / "logs.txt"))

        # Create subdirectories
        self.cfg_dir = self.job_dir / "cfg"
        self.ir_dir = self.job_dir / "ir"
        self.obf_dir = self.job_dir / "obf"
        self.final_dir = self.job_dir / "final"

        for d in [self.cfg_dir, self.ir_dir, self.obf_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: list, stage_name: str) -> Tuple[bool, str]:
        """Execute shell command and capture output."""
        self.logger.log(f"Starting {stage_name}...")
        self.logger.log(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined = stdout + stderr

            self.logger.log_command(" ".join(cmd), combined, result.returncode)

            if result.returncode != 0:
                self.logger.log(f"{stage_name} failed with exit code {result.returncode}", level="ERROR")
                return False, combined

            self.logger.log(f"{stage_name} completed successfully")
            return True, combined

        except subprocess.TimeoutExpired:
            self.logger.log(f"{stage_name} timeout (10 minutes exceeded)", level="ERROR")
            return False, "Timeout"
        except Exception as e:
            self.logger.log(f"{stage_name} error: {str(e)}", level="ERROR")
            return False, str(e)

    def step1_ghidra_lifting(self) -> bool:
        """Step 1: Export CFG using Ghidra."""
        input_exe = self.job_dir / "input.exe"

        if not input_exe.exists():
            self.logger.log("input.exe not found", level="ERROR")
            return False

        # Find run_ghidra_lifter.sh (script is at root/binary_obfuscation_pipeline/...)
        script_path = Path(__file__).parent.parent.parent.parent / "binary_obfuscation_pipeline" / "mcsema_impl" / "ghidra_lifter" / "run_ghidra_lifter.sh"

        if not script_path.exists():
            self.logger.log(f"run_ghidra_lifter.sh not found at {script_path}", level="ERROR")
            return False

        cmd = [
            str(script_path),
            str(input_exe),
            str(self.cfg_dir)
        ]

        success, output = self.run_command(cmd, "Ghidra CFG Export")
        return success and (self.cfg_dir / "program.cfg").exists()

    def step2_mcsema_lifting(self) -> bool:
        """Step 2: Lift CFG to LLVM IR using McSema."""
        cfg_file = self.cfg_dir / "program.cfg"

        if not cfg_file.exists():
            self.logger.log("CFG file not found", level="ERROR")
            return False

        script_path = Path(__file__).parent.parent.parent.parent / "binary_obfuscation_pipeline" / "mcsema_impl" / "lifter" / "run_lift.sh"

        if not script_path.exists():
            self.logger.log(f"run_lift.sh not found at {script_path}", level="ERROR")
            return False

        cmd = [
            str(script_path),
            str(cfg_file),
            str(self.ir_dir)
        ]

        success, output = self.run_command(cmd, "McSema LLVM Lifting")
        return success and (self.ir_dir / "program.bc").exists()

    def step3_ir_version_upgrade(self) -> bool:
        """Step 3: Upgrade LLVM IR to LLVM 22."""
        input_bc = self.ir_dir / "program.bc"

        if not input_bc.exists():
            self.logger.log("program.bc not found", level="ERROR")
            return False

        script_path = Path(__file__).parent.parent.parent.parent / "binary_obfuscation_pipeline" / "mcsema_impl" / "lifter" / "convert_ir_version.sh"

        if not script_path.exists():
            self.logger.log(f"convert_ir_version.sh not found at {script_path}", level="ERROR")
            return False

        cmd = [
            str(script_path),
            str(input_bc),
            str(self.ir_dir)
        ]

        success, output = self.run_command(cmd, "LLVM 22 IR Upgrade")
        return success and (self.ir_dir / "program_llvm22.bc").exists()

    def step4_ollvm_passes(self) -> bool:
        """Step 4: Apply OLLVM obfuscation passes."""
        input_bc = self.ir_dir / "program_llvm22.bc"

        if not input_bc.exists():
            self.logger.log("program_llvm22.bc not found", level="ERROR")
            return False

        # Only substitution is enabled for now
        if not self.passes_config.get("substitution", False):
            self.logger.log("No OLLVM passes enabled, skipping")
            # Copy bitcode to obf directory as fallback
            import shutil
            shutil.copy(input_bc, self.obf_dir / "program_obf.bc")
            return True

        # Create passes config file
        passes_config_file = self.job_dir / "passes_config.json"
        passes_config_file.write_text(json.dumps(self.passes_config, indent=2))

        script_path = Path(__file__).parent.parent.parent.parent / "binary_obfuscation_pipeline" / "mcsema_impl" / "ollvm_stage" / "run_ollvm.sh"

        if not script_path.exists():
            self.logger.log(f"run_ollvm.sh not found at {script_path}", level="ERROR")
            return False

        cmd = [
            str(script_path),
            str(input_bc),
            str(self.obf_dir),
            str(passes_config_file)
        ]

        success, output = self.run_command(cmd, "OLLVM Pass Application")
        return success and (self.obf_dir / "program_obf.bc").exists()

    def step5_final_compilation(self) -> bool:
        """Step 5: Compile obfuscated IR back to Windows PE."""
        input_bc = self.obf_dir / "program_obf.bc"

        if not input_bc.exists():
            self.logger.log("program_obf.bc not found", level="ERROR")
            return False

        output_exe = self.final_dir / "final.exe"

        cmd = [
            "clang",
            "--target=x86_64-w64-mingw32",
            "-O2",
            str(input_bc),
            "-o", str(output_exe)
        ]

        success, output = self.run_command(cmd, "Final Binary Compilation")
        return success and output_exe.exists()

    def step6_metrics_generation(self) -> bool:
        """Step 6: Generate metrics comparing input and output."""
        input_exe = self.job_dir / "input.exe"
        final_exe = self.final_dir / "final.exe"

        if not input_exe.exists():
            self.logger.log("input.exe not found for metrics", level="ERROR")
            return False

        metrics = {
            "input_size": input_exe.stat().st_size if input_exe.exists() else 0,
            "output_size": final_exe.stat().st_size if final_exe.exists() else 0,
            "size_diff_percent": 0,
            "llvm_instruction_count_before": 0,
            "llvm_instruction_count_after": 0,
            "cfg_complexity": "medium",
            "pipeline_duration": "unknown",
            "timestamp": datetime.now().isoformat()
        }

        # Calculate size difference
        if metrics["input_size"] > 0:
            metrics["size_diff_percent"] = (
                (metrics["output_size"] - metrics["input_size"]) / metrics["input_size"] * 100
            )

        # Try to extract instruction counts from IR files
        try:
            # Count instructions in program_llvm22.bc (before OLLVM)
            if (self.ir_dir / "program_llvm22.bc").exists():
                result = subprocess.run(
                    ["llvm-dis-22", str(self.ir_dir / "program_llvm22.bc"), "-o", "-"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    metrics["llvm_instruction_count_before"] = result.stdout.count(" = ")

            # Count instructions in program_obf.bc (after OLLVM)
            if (self.obf_dir / "program_obf.bc").exists():
                result = subprocess.run(
                    ["llvm-dis-22", str(self.obf_dir / "program_obf.bc"), "-o", "-"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    metrics["llvm_instruction_count_after"] = result.stdout.count(" = ")
        except Exception as e:
            self.logger.log(f"Could not extract instruction counts: {e}", level="WARN")

        metrics_file = self.job_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))

        self.logger.log(f"Metrics generated: {json.dumps(metrics, indent=2)}")
        return True

    def execute(self) -> Tuple[bool, str]:
        """Execute the complete pipeline."""
        try:
            self.logger.log("=== Binary Obfuscation Pipeline Started ===")
            self.logger.log(f"Job directory: {self.job_dir}")
            self.logger.log(f"Passes config: {json.dumps(self.passes_config)}")

            # Step 1: Ghidra
            if not self.step1_ghidra_lifting():
                return False, "Ghidra CFG Export failed"

            # Step 2: McSema
            if not self.step2_mcsema_lifting():
                return False, "McSema LLVM Lifting failed"

            # Step 3: IR Upgrade
            if not self.step3_ir_version_upgrade():
                return False, "LLVM 22 IR Upgrade failed"

            # Step 4: OLLVM
            if not self.step4_ollvm_passes():
                return False, "OLLVM Pass Application failed"

            # Step 5: Final Compilation
            if not self.step5_final_compilation():
                return False, "Final Binary Compilation failed"

            # Step 6: Metrics
            if not self.step6_metrics_generation():
                self.logger.log("Metrics generation failed (non-critical)", level="WARN")

            self.logger.log("=== Binary Obfuscation Pipeline Completed Successfully ===")
            return True, "Pipeline completed successfully"

        except Exception as e:
            error_msg = f"Pipeline execution error: {str(e)}"
            self.logger.log(error_msg, level="ERROR")
            return False, error_msg

    def get_logs(self) -> str:
        """Get accumulated logs."""
        return self.logger.read_logs()

    def get_metrics(self) -> Optional[Dict]:
        """Get metrics from completed pipeline."""
        metrics_file = self.job_dir / "metrics.json"
        if metrics_file.exists():
            try:
                return json.loads(metrics_file.read_text())
            except Exception as e:
                logger.error(f"Could not parse metrics: {e}")
                return None
        return None

    def get_artifacts(self) -> Dict[str, Optional[str]]:
        """Get artifact paths."""
        return {
            "final_exe": str(self.final_dir / "final.exe") if (self.final_dir / "final.exe").exists() else None,
            "program_obf_bc": str(self.obf_dir / "program_obf.bc") if (self.obf_dir / "program_obf.bc").exists() else None,
            "program_llvm22_bc": str(self.ir_dir / "program_llvm22.bc") if (self.ir_dir / "program_llvm22.bc").exists() else None,
            "logs": str(self.job_dir / "logs.txt") if (self.job_dir / "logs.txt").exists() else None,
            "metrics": str(self.job_dir / "metrics.json") if (self.job_dir / "metrics.json").exists() else None,
        }
