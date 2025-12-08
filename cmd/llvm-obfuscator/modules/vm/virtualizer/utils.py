"""VM Virtualizer Utilities.

Constants, logging, and helper functions for the IR parser and bytecode generator.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

# =============================================================================
# VM Opcode Constants (must match runtime/opcodes.h)
# =============================================================================

VM_NOP      = 0x00
VM_PUSH     = 0x01
VM_POP      = 0x02
VM_LOAD     = 0x03
VM_STORE    = 0x04
VM_LOAD_ARG = 0x05
VM_ADD      = 0x10
VM_SUB      = 0x11
VM_XOR      = 0x12
VM_RET      = 0xFF

# Map IR opcodes to VM opcodes
IR_TO_VM_OPCODE = {
    "add": VM_ADD,
    "sub": VM_SUB,
    "xor": VM_XOR,
}

# =============================================================================
# Supported Operations and Types
# =============================================================================

# Supported binary operations (Level 1)
SUPPORTED_BINOPS = {"add", "sub", "xor"}

# Supported integer types (Level 1)
SUPPORTED_TYPES = {"i32", "i64", "i8", "i16"}

# Instructions that make a function unsupported (Level 1)
UNSUPPORTED_INSTRUCTIONS = {
    "call",       # Function calls
    "invoke",     # Exception handling calls
    "phi",        # SSA phi nodes (require control flow)
    "br",         # Branches (loops, conditionals)
    "switch",     # Switch statements
    "indirectbr", # Indirect branches
    "resume",     # Exception handling
    "catchswitch", "catchret", "cleanupret",  # Exception handling
    "unreachable",
    "fneg",       # Floating point
    "fadd", "fsub", "fmul", "fdiv", "frem",  # Floating point arithmetic
    "fcmp",       # Floating point comparison
    "fptrunc", "fpext", "fptoui", "fptosi", "uitofp", "sitofp",  # FP conversions
    "load",       # Memory operations (Level 1)
    "store",      # Memory operations (Level 1)
    "alloca",     # Stack allocation
    "getelementptr",  # Pointer arithmetic
    "atomicrmw", "cmpxchg", "fence",  # Atomic operations
}

# =============================================================================
# Logging
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(levelname)s [%(name)s] %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # Default to WARNING for subprocess use
    return logger

# =============================================================================
# IR Parsing Helpers
# =============================================================================

# Regex patterns for LLVM IR parsing
FUNC_DEF_PATTERN = re.compile(
    r'define\s+(?:(?:dso_local|internal|private|linkonce_odr|weak_odr|'
    r'external|available_externally|linkonce|weak|appending|common|'
    r'extern_weak|thread_local|local_unnamed_addr|unnamed_addr)\s+)*'
    r'(\w+)\s+'  # Return type
    r'@(["\w.]+)\s*'  # Function name (may be quoted or contain dots)
    r'\(([^)]*)\)'  # Arguments
)

INSTRUCTION_PATTERN = re.compile(
    r'^\s*(%[\w.]+)\s*=\s*(\w+)\s+(.+)$'
)

RET_PATTERN = re.compile(
    r'^\s*ret\s+(\w+)\s+(%?[\w.]+|[-\d]+)$'
)

RET_VOID_PATTERN = re.compile(
    r'^\s*ret\s+void\s*$'
)


def is_supported_instruction(line: str) -> tuple[bool, Optional[str]]:
    """Check if an instruction line is supported.

    Returns:
        Tuple of (is_supported, skip_reason_or_none)
    """
    line = line.strip()

    # Skip empty lines, comments, labels, metadata
    if not line or line.startswith(";") or line.endswith(":"):
        return True, None
    if line.startswith("!") or line.startswith("source_filename"):
        return True, None
    if line.startswith("target "):
        return True, None
    if line.startswith("attributes "):
        return True, None
    if line == "}":
        return True, None

    # Check for unsupported instructions
    for unsup in UNSUPPORTED_INSTRUCTIONS:
        # Check if instruction appears as the opcode
        if re.search(rf'\b{unsup}\b', line):
            # Special case: "store" in IR is unsupported, but VM_STORE is different
            # Check if it's actually the IR opcode
            match = INSTRUCTION_PATTERN.match(line)
            if match and match.group(2) == unsup:
                return False, f"contains '{unsup}' instruction"
            # Also check for standalone instructions like "call ..."
            if line.strip().startswith(unsup + " "):
                return False, f"contains '{unsup}' instruction"
            if f"= {unsup} " in line:
                return False, f"contains '{unsup}' instruction"

    return True, None


def extract_function_name(line: str) -> Optional[str]:
    """Extract function name from a define line.

    Args:
        line: A line that may contain a function definition

    Returns:
        Function name or None if not a function definition
    """
    match = FUNC_DEF_PATTERN.search(line)
    if match:
        name = match.group(2)
        # Remove quotes if present
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        return name
    return None


def parse_type(type_str: str) -> str:
    """Normalize an LLVM type string.

    Args:
        type_str: Raw type string from IR

    Returns:
        Normalized type string
    """
    # Remove pointer markers for now (Level 1 doesn't support pointers)
    type_str = type_str.strip()
    type_str = type_str.rstrip("*")

    # Handle common aliases
    type_str = type_str.replace("i1", "i8")  # Treat bool as i8

    return type_str


def parse_value(value_str: str) -> tuple[str, bool]:
    """Parse an IR value string.

    Args:
        value_str: Value string like "%a", "42", or "true"

    Returns:
        Tuple of (name, is_constant)
    """
    value_str = value_str.strip()

    # Check if it's a constant
    if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
        return value_str, True
    if value_str in ("true", "false"):
        return value_str, True
    if value_str == "null" or value_str == "undef":
        return value_str, True

    # It's a variable reference
    return value_str, False


def is_label(line: str) -> bool:
    """Check if a line is a basic block label."""
    line = line.strip()
    return bool(line and line.endswith(":") and not line.startswith(";"))


def strip_metadata(line: str) -> str:
    """Remove metadata annotations from an instruction line."""
    # Remove trailing metadata like ", !dbg !123"
    if ", !" in line:
        line = line.split(", !")[0]
    # Remove inline attributes
    if " #" in line:
        parts = line.split(" #")
        # Keep only the part before attributes
        line = parts[0]
    return line.strip()
