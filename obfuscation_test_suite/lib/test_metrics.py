#!/usr/bin/env python3
"""Metrics computation for obfuscation analysis"""

import logging
from typing import Dict, Any
try:
    from .test_utils import run_command, get_file_size
except ImportError:
    from test_utils import run_command, get_file_size

logger = logging.getLogger(__name__)


def compute_cfg_metrics(baseline: str, obfuscated: str) -> Dict[str, Any]:
    """Compute control flow graph metrics"""
    logger.debug("Computing CFG metrics...")

    # Use objdump to analyze instructions
    baseline_metrics = _analyze_objdump(baseline)
    obf_metrics = _analyze_objdump(obfuscated)

    return {
        'baseline': baseline_metrics,
        'obfuscated': obf_metrics,
        'comparison': {
            'indirect_jumps_ratio': _compute_ratio(
                obf_metrics.get('indirect_jumps', 0),
                baseline_metrics.get('indirect_jumps', 1)
            ),
            'basic_blocks_ratio': _compute_ratio(
                obf_metrics.get('basic_blocks_estimate', 0),
                baseline_metrics.get('basic_blocks_estimate', 1)
            ),
            'control_flow_complexity_increase': _compute_ratio(
                obf_metrics.get('branch_instructions', 0),
                baseline_metrics.get('branch_instructions', 1)
            )
        }
    }


def analyze_complexity(baseline: str, obfuscated: str) -> Dict[str, Any]:
    """Analyze complexity metrics"""
    logger.debug("Analyzing complexity...")

    baseline_size = get_file_size(baseline)
    obf_size = get_file_size(obfuscated)

    baseline_sections = _count_sections(baseline)
    obf_sections = _count_sections(obfuscated)

    baseline_syms = _count_symbols(baseline)
    obf_syms = _count_symbols(obfuscated)

    return {
        'baseline': {
            'size_bytes': baseline_size,
            'section_count': baseline_sections,
            'symbol_count': baseline_syms,
        },
        'obfuscated': {
            'size_bytes': obf_size,
            'section_count': obf_sections,
            'symbol_count': obf_syms,
        },
        'changes': {
            'size_increase_percent': _compute_percent_change(baseline_size, obf_size),
            'section_increase': obf_sections - baseline_sections,
            'symbol_reduction': baseline_syms - obf_syms,
            # ✅ FIX #6f: Symbol reduction percent now correctly shows:
            # - Negative % = reduction (good)
            # - Positive % = increase (bad, obfuscation added symbols)
            'symbol_reduction_percent': _compute_percent_change(baseline_syms, obf_syms)
        }
    }


def compute_coverage(baseline: str, obfuscated: str) -> Dict[str, Any]:
    """Estimate code coverage metrics"""
    logger.debug("Computing coverage metrics...")

    return {
        'baseline': {
            'estimated_coverage': 'N/A',
            'reachable_code': _estimate_reachable_code(baseline)
        },
        'obfuscated': {
            'estimated_coverage': 'N/A',
            'reachable_code': _estimate_reachable_code(obfuscated)
        },
        'analysis_note': 'Coverage analysis requires runtime instrumentation or binary analysis tools'
    }


# Helper functions
def _analyze_objdump(filepath: str) -> Dict[str, Any]:
    """Analyze binary using objdump"""
    metrics = {
        'indirect_jumps': 0,
        'branch_instructions': 0,
        'basic_blocks_estimate': 0,
        'call_instructions': 0,
        'nop_instructions': 0
    }

    try:
        result = run_command(f"objdump -d {filepath} 2>/dev/null")

        for line in result.split('\n'):
            # Count different instruction types
            # ✅ FIX #6a: Detect indirect jumps more accurately (look for register dereference pattern)
            if 'jmp' in line and ('*' in line or '%' in line):  # x86: *(reg) or %reg
                metrics['indirect_jumps'] += 1
            # ✅ FIX #6b: Count branch instructions (excluding indirect jumps to avoid double-counting)
            elif any(x in line for x in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jl', 'jle', 'jge', 'jo', 'jno']):
                metrics['branch_instructions'] += 1
            # Also count conditional jump instructions that aren't caught above
            if 'call' in line:
                metrics['call_instructions'] += 1
            if 'nop' in line:
                metrics['nop_instructions'] += 1

        # ✅ FIX #6c: Improve basic block estimation
        # Basic blocks are typically separated by branches, but need better heuristic
        # Each branch instruction marks an end of a basic block
        # Add 1 to account for the entry block
        metrics['basic_blocks_estimate'] = max(1, metrics['branch_instructions'] + metrics['indirect_jumps'] + 1)

    except Exception as e:
        logger.warning(f"Error analyzing {filepath}: {e}")

    return metrics


def _count_sections(filepath: str) -> int:
    """Count sections in binary"""
    try:
        result = run_command(f"objdump -h {filepath} 2>/dev/null")
        # Skip header lines and count actual sections
        return len([line for line in result.split('\n')[4:] if line.strip()])
    except:
        return 0


def _count_symbols(filepath: str) -> int:
    """Count symbols in binary"""
    try:
        result = run_command(f"nm {filepath} 2>/dev/null")
        # ✅ FIX #6d: Count only actual symbol entries, excluding empty lines
        # nm output format: "address type name" or "         type name" for undefined symbols
        symbol_lines = [line for line in result.split('\n') if line.strip() and not line.startswith('nm:')]
        return len(symbol_lines)
    except:
        return 0


def _estimate_reachable_code(filepath: str) -> str:
    """Estimate reachable code percentage"""
    try:
        # Simple heuristic: count non-zero sections
        result = run_command(f"objdump -h {filepath} 2>/dev/null")
        lines = result.split('\n')[4:]
        total_size = 0
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) > 2:
                    try:
                        size = int(parts[2], 16)
                        total_size += size
                    except:
                        pass
        return f"~{min(100, max(0, (total_size // 1024)))//10 * 10}%"
    except:
        return "Unknown"


def _compute_ratio(obf_val: float, base_val: float) -> float:
    """Compute obfuscated to baseline ratio"""
    if base_val <= 0:
        return 1.0
    return round(obf_val / base_val, 2)


def _compute_percent_change(old_val: float, new_val: float, inverse: bool = False) -> float:
    """Compute percentage change

    ✅ FIX #6e: Correct percent change calculation
    Always computes: (new - old) / old * 100
    Result interpretation depends on metric:
    - Size increase: positive % = growth
    - Symbol reduction: negative % = reduction (good), positive % = increase (bad)
    """
    if old_val <= 0:
        return 0.0

    # Standard percent change formula: (new - old) / old * 100
    change = 100 * (new_val - old_val) / old_val

    return round(change, 2)
