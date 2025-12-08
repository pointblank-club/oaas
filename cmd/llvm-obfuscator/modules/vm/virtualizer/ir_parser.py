"""LLVM IR Parser for VM Virtualization.

Parses .ll files to extract function definitions and instructions.
Only supports simple functions (no loops, no calls) for Level 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import (
    FUNC_DEF_PATTERN,
    INSTRUCTION_PATTERN,
    RET_PATTERN,
    RET_VOID_PATTERN,
    SUPPORTED_BINOPS,
    SUPPORTED_TYPES,
    extract_function_name,
    get_logger,
    is_label,
    is_supported_instruction,
    parse_type,
    parse_value,
    strip_metadata,
)

logger = get_logger(__name__)


# =============================================================================
# IR Data Classes
# =============================================================================

@dataclass
class IRValue:
    """Represents an IR value (variable or constant).

    Attributes:
        name: The value name (e.g., "%a", "42")
        value_type: The LLVM type (e.g., "i32")
        is_arg: Whether this is a function argument
        arg_index: Index if this is an argument
        is_constant: Whether this is a constant value
    """
    name: str
    value_type: str = "i32"
    is_arg: bool = False
    arg_index: Optional[int] = None
    is_constant: bool = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, IRValue):
            return self.name == other.name
        return False


@dataclass
class IRInstruction:
    """Represents a single IR instruction.

    Attributes:
        opcode: The instruction opcode (e.g., "add", "sub", "ret")
        result: The result value (for instructions that produce a value)
        operands: List of operand values
        result_type: Type of the result
        raw_line: Original IR line
    """
    opcode: str
    result: Optional[IRValue] = None
    operands: List[IRValue] = field(default_factory=list)
    result_type: str = "i32"
    raw_line: str = ""


@dataclass
class IRFunction:
    """Represents a parsed IR function.

    Attributes:
        name: Function name
        return_type: Return type
        arguments: List of argument values
        instructions: List of parsed instructions
        is_supported: Whether this function can be virtualized
        skip_reason: Reason if not supported
        raw_lines: Original IR lines
    """
    name: str
    return_type: str = "i32"
    arguments: List[IRValue] = field(default_factory=list)
    instructions: List[IRInstruction] = field(default_factory=list)
    is_supported: bool = True
    skip_reason: Optional[str] = None
    raw_lines: List[str] = field(default_factory=list)

    @property
    def arg_map(self) -> Dict[str, IRValue]:
        """Map argument names to IRValue objects."""
        return {arg.name: arg for arg in self.arguments}


@dataclass
class IRModule:
    """Represents a parsed IR module (.ll file).

    Attributes:
        functions: Dictionary of function name -> IRFunction
        global_lines: Lines before first function (globals, declarations, etc.)
        between_lines: Lines between functions
        unsupported_functions: List of function names that couldn't be virtualized
        source_file: Original source file path
    """
    functions: Dict[str, IRFunction] = field(default_factory=dict)
    global_lines: List[str] = field(default_factory=list)
    between_lines: Dict[str, List[str]] = field(default_factory=dict)
    unsupported_functions: List[str] = field(default_factory=list)
    source_file: Optional[Path] = None


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_function_signature(line: str) -> Tuple[str, str, List[IRValue]]:
    """Parse a function definition line.

    Args:
        line: The 'define ...' line

    Returns:
        Tuple of (function_name, return_type, arguments)

    Raises:
        ValueError: If the line cannot be parsed
    """
    match = FUNC_DEF_PATTERN.search(line)
    if not match:
        raise ValueError(f"Cannot parse function signature: {line}")

    return_type = match.group(1)
    func_name = match.group(2)
    args_str = match.group(3)

    # Remove quotes from function name if present
    if func_name.startswith('"') and func_name.endswith('"'):
        func_name = func_name[1:-1]

    # Parse arguments
    arguments = []
    if args_str.strip():
        # Split by comma, but be careful with types like "i32*"
        arg_parts = []
        depth = 0
        current = ""
        for char in args_str:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                arg_parts.append(current.strip())
                current = ""
                continue
            current += char
        if current.strip():
            arg_parts.append(current.strip())

        for idx, arg in enumerate(arg_parts):
            arg = arg.strip()
            if not arg:
                continue

            # Handle "i32 %name" or "i32 noundef %name" or just "i32"
            # Remove attributes like noundef, nonnull, etc.
            parts = arg.split()
            arg_type = None
            arg_name = None

            for i, part in enumerate(parts):
                # Skip attributes
                if part in ("noundef", "nonnull", "signext", "zeroext",
                           "inreg", "byval", "sret", "align", "nocapture",
                           "readonly", "writeonly"):
                    continue
                # Skip alignment values
                if part.isdigit():
                    continue
                # First non-attribute is the type
                if arg_type is None:
                    arg_type = part
                # Next is the name
                elif arg_name is None and part.startswith("%"):
                    arg_name = part
                    break

            if arg_type is None:
                continue

            if arg_name is None:
                arg_name = f"%arg{idx}"

            arguments.append(IRValue(
                name=arg_name,
                value_type=parse_type(arg_type),
                is_arg=True,
                arg_index=idx,
            ))

    return func_name, return_type, arguments


def parse_instruction(line: str, arg_map: Dict[str, IRValue]) -> Optional[IRInstruction]:
    """Parse a single IR instruction.

    Args:
        line: The instruction line
        arg_map: Map of argument names to IRValue

    Returns:
        IRInstruction or None if not an instruction we care about
    """
    line = strip_metadata(line)
    orig_line = line

    # Skip empty lines, comments, labels
    if not line or line.startswith(";") or is_label(line):
        return None
    if line == "}":
        return None

    # Handle return instruction
    ret_match = RET_PATTERN.match(line)
    if ret_match:
        ret_type = ret_match.group(1)
        ret_val = ret_match.group(2)

        val_name, is_const = parse_value(ret_val)

        operand = IRValue(
            name=val_name,
            value_type=ret_type,
            is_constant=is_const,
        )

        # Check if it's an argument
        if val_name in arg_map:
            operand = arg_map[val_name]

        return IRInstruction(
            opcode="ret",
            operands=[operand],
            result_type=ret_type,
            raw_line=orig_line,
        )

    # Handle void return
    if RET_VOID_PATTERN.match(line):
        return IRInstruction(
            opcode="ret",
            operands=[],
            result_type="void",
            raw_line=orig_line,
        )

    # Handle binary operations: %result = opcode type %op1, %op2
    inst_match = INSTRUCTION_PATTERN.match(line)
    if inst_match:
        result_name = inst_match.group(1)
        opcode = inst_match.group(2)
        rest = inst_match.group(3)

        # Only handle supported binary operations
        if opcode in SUPPORTED_BINOPS:
            # Parse: type %op1, %op2
            # Handle optional nsw/nuw flags
            rest = rest.replace("nsw ", "").replace("nuw ", "")

            # Split by comma
            parts = rest.split(",")
            if len(parts) >= 2:
                # First part: type operand1
                first = parts[0].strip().split()
                if len(first) >= 2:
                    result_type = first[0]
                    op1_str = first[-1]
                else:
                    result_type = "i32"
                    op1_str = first[0] if first else ""

                # Second part: operand2
                op2_str = parts[1].strip()

                # Parse operands
                op1_name, op1_const = parse_value(op1_str)
                op2_name, op2_const = parse_value(op2_str)

                op1 = IRValue(name=op1_name, value_type=result_type, is_constant=op1_const)
                op2 = IRValue(name=op2_name, value_type=result_type, is_constant=op2_const)

                # Check if operands are arguments
                if op1_name in arg_map:
                    op1 = arg_map[op1_name]
                if op2_name in arg_map:
                    op2 = arg_map[op2_name]

                result = IRValue(name=result_name, value_type=result_type)

                return IRInstruction(
                    opcode=opcode,
                    result=result,
                    operands=[op1, op2],
                    result_type=result_type,
                    raw_line=orig_line,
                )

    return None


def parse_function_body(
    lines: List[str],
    arguments: List[IRValue]
) -> Tuple[List[IRInstruction], bool, Optional[str]]:
    """Parse the body of a function.

    Args:
        lines: Lines of the function body (between { and })
        arguments: Function arguments

    Returns:
        Tuple of (instructions, is_supported, skip_reason)
    """
    arg_map = {arg.name: arg for arg in arguments}
    instructions = []
    is_supported = True
    skip_reason = None

    # Track defined values for later reference
    defined_values: Dict[str, IRValue] = dict(arg_map)

    for line in lines:
        # Check if instruction is supported
        supported, reason = is_supported_instruction(line)
        if not supported:
            is_supported = False
            skip_reason = reason
            logger.debug(f"Unsupported instruction: {line.strip()} - {reason}")
            continue

        # Try to parse the instruction
        inst = parse_instruction(line, defined_values)
        if inst:
            instructions.append(inst)
            # Track the result value
            if inst.result:
                defined_values[inst.result.name] = inst.result

    return instructions, is_supported, skip_reason


def parse_ll_file(filepath: Path) -> IRModule:
    """Parse an LLVM IR file.

    Args:
        filepath: Path to the .ll file

    Returns:
        IRModule containing parsed functions and globals
    """
    module = IRModule(source_file=filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = content.splitlines()

    # State machine for parsing
    in_function = False
    current_func: Optional[IRFunction] = None
    current_body_lines: List[str] = []
    last_func_name: Optional[str] = None
    brace_depth = 0

    for line_num, line in enumerate(lines):
        # Track brace depth for multi-line function bodies
        brace_depth += line.count("{") - line.count("}")

        # Check for function start
        func_name = extract_function_name(line)
        if func_name and not in_function:
            try:
                name, ret_type, args = parse_function_signature(line)

                current_func = IRFunction(
                    name=name,
                    return_type=ret_type,
                    arguments=args,
                )
                current_func.raw_lines.append(line)
                in_function = True
                current_body_lines = []

                # Check if function body starts on same line
                if "{" in line:
                    # Body starts after the {
                    body_start = line.index("{") + 1
                    if body_start < len(line):
                        current_body_lines.append(line[body_start:])

                logger.debug(f"Found function: {name} with {len(args)} args")

            except ValueError as e:
                logger.warning(f"Failed to parse function at line {line_num}: {e}")
                module.global_lines.append(line)
                continue

        elif in_function:
            current_func.raw_lines.append(line)

            # Check for function end
            if "}" in line and brace_depth <= 0:
                # Add body content before }
                body_end = line.index("}")
                if body_end > 0:
                    current_body_lines.append(line[:body_end])

                # Parse the function body
                instructions, is_supported, skip_reason = parse_function_body(
                    current_body_lines,
                    current_func.arguments
                )

                current_func.instructions = instructions
                current_func.is_supported = is_supported
                current_func.skip_reason = skip_reason

                # Add to module
                module.functions[current_func.name] = current_func

                if not is_supported:
                    module.unsupported_functions.append(current_func.name)
                    logger.debug(f"Function {current_func.name} marked unsupported: {skip_reason}")

                # Reset state
                last_func_name = current_func.name
                current_func = None
                in_function = False
                brace_depth = 0
            else:
                current_body_lines.append(line)

        else:
            # Not in a function - store as global/between lines
            if last_func_name:
                if last_func_name not in module.between_lines:
                    module.between_lines[last_func_name] = []
                module.between_lines[last_func_name].append(line)
            else:
                module.global_lines.append(line)

    logger.info(f"Parsed {len(module.functions)} functions from {filepath}")
    logger.info(f"Supported: {len(module.functions) - len(module.unsupported_functions)}, "
                f"Unsupported: {len(module.unsupported_functions)}")

    return module


def get_supported_functions(
    module: IRModule,
    filter_names: Optional[List[str]] = None
) -> List[IRFunction]:
    """Get list of functions that can be virtualized.

    Args:
        module: Parsed IR module
        filter_names: Optional list of function names to filter

    Returns:
        List of supported IRFunction objects
    """
    supported = []

    for name, func in module.functions.items():
        # Skip if not supported
        if not func.is_supported:
            continue

        # Skip if not in filter list (when filter is provided)
        if filter_names and name not in filter_names:
            continue

        # Skip void return functions for now (Level 1)
        if func.return_type == "void":
            continue

        # Skip if no instructions (empty function)
        if not func.instructions:
            continue

        # Must have at least a ret instruction
        has_ret = any(inst.opcode == "ret" for inst in func.instructions)
        if not has_ret:
            continue

        supported.append(func)

    return supported
