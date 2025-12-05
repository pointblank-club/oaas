"""
LLVM IR Analysis Module
Extracts control flow and instruction-level metrics from LLVM IR files.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .utils import run_command

logger = logging.getLogger(__name__)


class IRAnalyzer:
    """Analyzes LLVM IR files to extract CFG and instruction metrics."""

    def __init__(self, opt_binary: Path, llvm_dis_binary: Path):
        """Initialize with LLVM tool paths.

        Args:
            opt_binary: Path to LLVM opt binary
            llvm_dis_binary: Path to llvm-dis binary
        """
        self.opt_binary = opt_binary
        self.llvm_dis_binary = llvm_dis_binary
        self.logger = logging.getLogger(__name__)

    def analyze_control_flow(self, ir_file: Path) -> Dict:
        """Extract CFG metrics from LLVM IR.

        Analyzes:
        - Basic block count
        - CFG edge count (estimated from branches)
        - Cyclomatic complexity (V = E - N + 2)
        - Loop count
        - Function count
        - Average basic blocks per function

        Args:
            ir_file: Path to .ll or .bc IR file

        Returns:
            Dict with CFG metrics
        """
        if not ir_file.exists():
            self.logger.warning(f"IR file not found: {ir_file}")
            return self._empty_control_flow_metrics()

        # Convert bitcode to text if needed
        ir_text = self._get_ir_text(ir_file)
        if not ir_text:
            return self._empty_control_flow_metrics()

        basic_blocks = self._count_basic_blocks(ir_text)
        functions = self._count_functions(ir_text)
        loops = self._count_loops(ir_text)
        cfg_edges = self._count_cfg_edges(ir_text)

        # Cyclomatic complexity: V = E - N + 2
        # where E = edges, N = nodes (basic blocks)
        cyclomatic_complexity = max(1, cfg_edges - basic_blocks + 2) if basic_blocks > 0 else 1

        avg_bb_per_func = basic_blocks / functions if functions > 0 else 0

        return {
            "basic_blocks": basic_blocks,
            "cfg_edges": cfg_edges,
            "cyclomatic_complexity": cyclomatic_complexity,
            "functions": functions,
            "loops": loops,
            "avg_bb_per_function": round(avg_bb_per_func, 2),
        }

    def analyze_instructions(self, ir_file: Path) -> Dict:
        """Extract instruction-level metrics from LLVM IR.

        Analyzes:
        - Total instruction count
        - Instruction distribution (load, store, call, branch, phi, arithmetic, other)
        - Arithmetic complexity score
        - MBA expression estimate
        - Call instruction count
        - Indirect call count

        Args:
            ir_file: Path to .ll or .bc IR file

        Returns:
            Dict with instruction metrics
        """
        if not ir_file.exists():
            self.logger.warning(f"IR file not found: {ir_file}")
            return self._empty_instruction_metrics()

        ir_text = self._get_ir_text(ir_file)
        if not ir_text:
            return self._empty_instruction_metrics()

        distribution = self._count_instruction_distribution(ir_text)
        total_instructions = sum(distribution.values())
        arithmetic_complexity = self._calculate_arithmetic_complexity(ir_text)
        mba_expressions = self._estimate_mba_expressions(ir_text)
        call_count = distribution.get("call", 0)
        indirect_call_count = self._count_indirect_calls(ir_text)

        return {
            "total_instructions": total_instructions,
            "instruction_distribution": distribution,
            "arithmetic_complexity_score": round(arithmetic_complexity, 2),
            "mba_expression_count": mba_expressions,
            "call_instruction_count": call_count,
            "indirect_call_count": indirect_call_count,
        }

    def compare_ir_metrics(self, baseline: Dict, obfuscated: Dict) -> Dict:
        """Calculate improvement metrics between baseline and obfuscated IR.

        Args:
            baseline: Baseline IR metrics dict
            obfuscated: Obfuscated IR metrics dict

        Returns:
            Dict with comparison metrics
        """
        # Control flow comparison
        baseline_bb = baseline.get("basic_blocks", 0) or 1
        baseline_edges = baseline.get("cfg_edges", 0) or 1
        baseline_complexity = baseline.get("cyclomatic_complexity", 0) or 1

        obf_bb = obfuscated.get("basic_blocks", 0)
        obf_edges = obfuscated.get("cfg_edges", 0)
        obf_complexity = obfuscated.get("cyclomatic_complexity", 0)

        complexity_increase = ((obf_complexity - baseline_complexity) / baseline_complexity * 100) if baseline_complexity > 0 else 0
        bb_added = max(0, obf_bb - baseline_bb)
        edges_added = max(0, obf_edges - baseline_edges)

        # Instruction comparison
        baseline_instr = baseline.get("total_instructions", 0) or 1
        obf_instr = obfuscated.get("total_instructions", 0)

        instruction_growth = ((obf_instr - baseline_instr) / baseline_instr * 100) if baseline_instr > 0 else 0

        # Arithmetic complexity
        baseline_arith = baseline.get("arithmetic_complexity_score", 0) or 0.1
        obf_arith = obfuscated.get("arithmetic_complexity_score", 0)

        arith_increase = ((obf_arith - baseline_arith) / baseline_arith * 100) if baseline_arith > 0 else 0

        return {
            "complexity_increase_percent": round(complexity_increase, 2),
            "basic_blocks_added": bb_added,
            "cfg_edges_added": edges_added,
            "instruction_growth_percent": round(instruction_growth, 2),
            "mba_expressions_added": obfuscated.get("mba_expression_count", 0) - baseline.get("mba_expression_count", 0),
            "arithmetic_complexity_increase": round(arith_increase, 2),
        }

    # ========== PRIVATE HELPER METHODS ==========

    def _get_ir_text(self, ir_file: Path) -> Optional[str]:
        """Get LLVM IR as text, converting from bitcode if needed."""
        try:
            if ir_file.suffix == ".bc":
                # Convert bitcode to text using llvm-dis
                return self._bitcode_to_text(ir_file)
            elif ir_file.suffix == ".ll":
                # Already text format
                return ir_file.read_text(errors='ignore')
            else:
                self.logger.warning(f"Unknown IR format: {ir_file.suffix}")
                return None
        except Exception as e:
            self.logger.error(f"Error reading IR file {ir_file}: {e}")
            return None

    def _bitcode_to_text(self, bc_file: Path) -> Optional[str]:
        """Convert LLVM bitcode to text IR using llvm-dis."""
        try:
            # Check if llvm-dis exists
            if not self.llvm_dis_binary.exists():
                self.logger.warning(f"llvm-dis not found at {self.llvm_dis_binary} - cannot convert bitcode to text")
                return None

            # Use llvm-dis to convert
            returncode, stdout, stderr = run_command(
                [str(self.llvm_dis_binary), str(bc_file), "-o", "-"]
            )
            if returncode == 0:
                return stdout
            else:
                self.logger.warning(f"llvm-dis failed for {bc_file}: {stderr}")
                return None
        except Exception as e:
            self.logger.error(f"Error converting bitcode to text: {e}")
            return None

    def _count_basic_blocks(self, ir_text: str) -> int:
        """Count basic blocks in IR (labels at start of line)."""
        # Basic block labels: ^[name]:
        # Pattern: start of line, alphanumeric/underscore, colon
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*:'
        matches = re.findall(pattern, ir_text, re.MULTILINE)
        return len(set(matches))  # Remove duplicates

    def _count_functions(self, ir_text: str) -> int:
        """Count functions in IR."""
        # Function definitions: define ... @function_name(
        pattern = r'define\s+\w+\s+@\w+'
        matches = re.findall(pattern, ir_text)
        return len(matches)

    def _count_loops(self, ir_text: str) -> int:
        """Estimate loop count from IR (phi nodes and back edges)."""
        # Loops typically have phi nodes and branches back to earlier blocks
        phi_pattern = r'=\s+phi\s+'
        phi_count = len(re.findall(phi_pattern, ir_text))

        # Back-edges detected by finding labels referenced before definition
        # This is an approximation
        return max(0, phi_count // 2)  # Rough estimate: 2 phis per loop

    def _count_cfg_edges(self, ir_text: str) -> int:
        """Count CFG edges (branches and direct jumps)."""
        # Branch instructions: br i1, br label, switch, etc.
        br_pattern = r'\bbr\s'
        branches = len(re.findall(br_pattern, ir_text))

        # Switch statements (one per case)
        switch_pattern = r'\bswitch\s'
        switches = len(re.findall(switch_pattern, ir_text))

        # Estimate: each branch creates 2 edges (true/false), switch varies
        edges = branches * 2 + switches * 3  # Rough estimate

        return max(1, edges)

    def _count_instruction_distribution(self, ir_text: str) -> Dict[str, int]:
        """Count different instruction types."""
        distribution = {
            "load": 0,
            "store": 0,
            "call": 0,
            "br": 0,
            "phi": 0,
            "arithmetic": 0,
            "other": 0,
        }

        for line in ir_text.split('\n'):
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            # Load instructions
            if ' load ' in line:
                distribution["load"] += 1
            # Store instructions
            elif ' store ' in line:
                distribution["store"] += 1
            # Call instructions
            elif 'call ' in line or '@' in line and '(' in line:
                distribution["call"] += 1
            # Branch instructions
            elif line.startswith('br '):
                distribution["br"] += 1
            # PHI nodes
            elif ' phi ' in line:
                distribution["phi"] += 1
            # Arithmetic (add, sub, mul, div, xor, and, or, shl, lshr, etc.)
            elif any(op in line for op in [' add ', ' sub ', ' mul ', ' udiv ', ' sdiv ',
                                          ' urem ', ' srem ', ' xor ', ' and ', ' or ',
                                          ' shl ', ' lshr ', ' ashr ']):
                distribution["arithmetic"] += 1
            else:
                distribution["other"] += 1

        return distribution

    def _calculate_arithmetic_complexity(self, ir_text: str) -> float:
        """Calculate arithmetic complexity score based on operations."""
        complexity = 0.0

        for line in ir_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Weight different operation types
            if ' mul ' in line or ' udiv ' in line or ' sdiv ' in line:
                complexity += 3  # Multiplication/division are expensive
            elif ' xor ' in line or ' and ' in line or ' or ' in line:
                complexity += 2  # Bitwise operations
            elif ' add ' in line or ' sub ' in line:
                complexity += 1  # Simple arithmetic
            elif ' shl ' in line or ' lshr ' in line or ' ashr ' in line:
                complexity += 1.5  # Shift operations
            elif ' call ' in line:
                complexity += 2  # Function calls add complexity

        # Normalize by instruction count
        total_instructions = len([l for l in ir_text.split('\n') if l.strip() and not l.strip().startswith(';')])

        if total_instructions > 0:
            complexity /= total_instructions
            complexity *= 100  # Scale to 0-100 range

        return min(100.0, complexity)

    def _estimate_mba_expressions(self, ir_text: str) -> int:
        """Estimate MBA (Mixed Boolean-Arithmetic) expressions.

        MBA expressions mix arithmetic and boolean operations.
        Heuristic: Count complex arithmetic chains.
        """
        count = 0

        # Look for lines with multiple operations
        for line in ir_text.split('\n'):
            line = line.strip()

            # Count operations on same line (would indicate chaining/MBA)
            ops = 0
            if ' add ' in line:
                ops += 1
            if ' mul ' in line or ' div ' in line:
                ops += 1
            if ' xor ' in line or ' and ' in line or ' or ' in line:
                ops += 1
            if ' shl ' in line or ' lshr ' in line:
                ops += 1

            # If multiple operations in expression, likely MBA-like
            if ops >= 2:
                count += 1

        return count

    def _count_indirect_calls(self, ir_text: str) -> int:
        """Count indirect function calls (calls through pointers)."""
        # Indirect calls in IR look like: call type %variable(...)
        # Direct calls look like: call type @function_name(...)

        indirect_pattern = r'call\s+\w+\s+%\w+'
        matches = re.findall(indirect_pattern, ir_text)
        return len(matches)

    def _empty_control_flow_metrics(self) -> Dict:
        """Return empty CFG metrics dict."""
        return {
            "basic_blocks": 0,
            "cfg_edges": 0,
            "cyclomatic_complexity": 0,
            "functions": 0,
            "loops": 0,
            "avg_bb_per_function": 0.0,
        }

    def _empty_instruction_metrics(self) -> Dict:
        """Return empty instruction metrics dict."""
        return {
            "total_instructions": 0,
            "instruction_distribution": {
                "load": 0,
                "store": 0,
                "call": 0,
                "br": 0,
                "phi": 0,
                "arithmetic": 0,
                "other": 0,
            },
            "arithmetic_complexity_score": 0.0,
            "mba_expression_count": 0,
            "call_instruction_count": 0,
            "indirect_call_count": 0,
        }
