"""Bytecode Generator for VM Virtualization.

Translates IR instructions to VM bytecode.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ir_parser import IRFunction, IRInstruction, IRValue
from .utils import (
    IR_TO_VM_OPCODE,
    VM_LOAD,
    VM_LOAD_ARG,
    VM_PUSH,
    VM_RET,
    VM_STORE,
    get_logger,
)

logger = get_logger(__name__)


# =============================================================================
# Bytecode Emitter
# =============================================================================

@dataclass
class BytecodeEmitter:
    """Emits VM bytecode from IR instructions.

    Attributes:
        bytecode: The bytecode buffer
        value_to_reg: Maps IR value names to virtual register indices
        next_reg: Next available virtual register index
        arg_indices: Maps argument names to their arg index
    """
    bytecode: bytearray = field(default_factory=bytearray)
    value_to_reg: Dict[str, int] = field(default_factory=dict)
    next_reg: int = 0
    arg_indices: Dict[str, int] = field(default_factory=dict)

    def emit_byte(self, value: int) -> None:
        """Emit a single byte."""
        self.bytecode.append(value & 0xFF)

    def emit_i32(self, value: int) -> None:
        """Emit a 32-bit little-endian integer."""
        # Handle signed values correctly
        if value < 0:
            value = value & 0xFFFFFFFF
        self.bytecode.extend(struct.pack("<I", value))

    def emit_load_arg(self, arg_index: int) -> None:
        """Emit VM_LOAD_ARG instruction."""
        self.emit_byte(VM_LOAD_ARG)
        self.emit_byte(arg_index)

    def emit_load(self, reg_index: int) -> None:
        """Emit VM_LOAD instruction."""
        self.emit_byte(VM_LOAD)
        self.emit_byte(reg_index)

    def emit_store(self, reg_index: int) -> None:
        """Emit VM_STORE instruction."""
        self.emit_byte(VM_STORE)
        self.emit_byte(reg_index)

    def emit_push(self, value: int) -> None:
        """Emit VM_PUSH instruction with immediate value."""
        self.emit_byte(VM_PUSH)
        self.emit_i32(value)

    def emit_binop(self, vm_opcode: int) -> None:
        """Emit a binary operation opcode."""
        self.emit_byte(vm_opcode)

    def emit_ret(self) -> None:
        """Emit VM_RET instruction."""
        self.emit_byte(VM_RET)

    def allocate_reg(self, value_name: str) -> int:
        """Allocate a virtual register for a value."""
        if value_name in self.value_to_reg:
            return self.value_to_reg[value_name]

        reg = self.next_reg
        self.value_to_reg[value_name] = reg
        self.next_reg += 1
        return reg

    def get_reg(self, value_name: str) -> Optional[int]:
        """Get the register for a value, or None if not allocated."""
        return self.value_to_reg.get(value_name)

    def load_value(self, value: IRValue) -> None:
        """Load a value onto the stack.

        Handles arguments, registers, and constants.
        """
        if value.is_constant:
            # Push constant value
            try:
                const_val = int(value.name)
            except ValueError:
                # Handle boolean constants
                if value.name == "true":
                    const_val = 1
                elif value.name == "false":
                    const_val = 0
                else:
                    const_val = 0
                    logger.warning(f"Unknown constant value: {value.name}")
            self.emit_push(const_val)

        elif value.is_arg:
            # Load from argument
            self.emit_load_arg(value.arg_index)

        elif value.name in self.value_to_reg:
            # Load from register
            self.emit_load(self.value_to_reg[value.name])

        elif value.name in self.arg_indices:
            # It's an argument referenced by name
            self.emit_load_arg(self.arg_indices[value.name])

        else:
            logger.warning(f"Unknown value: {value.name}, treating as 0")
            self.emit_push(0)


# =============================================================================
# Bytecode Generation
# =============================================================================

def generate_bytecode(func: IRFunction) -> bytes:
    """Generate VM bytecode for a function.

    The bytecode strategy:
    1. For arguments: use VM_LOAD_ARG directly when needed
    2. For intermediate results: allocate vregs and use VM_STORE
    3. For ret: load return value and emit VM_RET

    Args:
        func: The IR function to generate bytecode for

    Returns:
        Bytecode as bytes
    """
    emitter = BytecodeEmitter()

    # Map argument names to indices for quick lookup
    for arg in func.arguments:
        emitter.arg_indices[arg.name] = arg.arg_index

    logger.debug(f"Generating bytecode for {func.name}")
    logger.debug(f"  Arguments: {[a.name for a in func.arguments]}")

    # Process each instruction
    for inst in func.instructions:
        emit_instruction(emitter, inst)

    logger.debug(f"  Generated {len(emitter.bytecode)} bytes")
    logger.debug(f"  Used {emitter.next_reg} registers")

    return bytes(emitter.bytecode)


def emit_instruction(emitter: BytecodeEmitter, inst: IRInstruction) -> None:
    """Emit bytecode for a single instruction.

    Args:
        emitter: The bytecode emitter
        inst: The instruction to emit
    """
    if inst.opcode == "ret":
        emit_ret(emitter, inst)
    elif inst.opcode in IR_TO_VM_OPCODE:
        emit_binop(emitter, inst)
    else:
        logger.warning(f"Unsupported opcode: {inst.opcode}")


def emit_binop(emitter: BytecodeEmitter, inst: IRInstruction) -> None:
    """Emit bytecode for a binary operation.

    Pattern:
        %result = op type %a, %b
    Becomes:
        LOAD %a (or LOAD_ARG if argument)
        LOAD %b (or LOAD_ARG if argument)
        OP
        STORE %result

    Args:
        emitter: The bytecode emitter
        inst: The binary operation instruction
    """
    if len(inst.operands) < 2:
        logger.warning(f"Binary op with < 2 operands: {inst.raw_line}")
        return

    op1, op2 = inst.operands[0], inst.operands[1]

    # Load first operand
    emitter.load_value(op1)

    # Load second operand
    emitter.load_value(op2)

    # Emit the operation
    vm_opcode = IR_TO_VM_OPCODE[inst.opcode]
    emitter.emit_binop(vm_opcode)

    # Store result to a new register
    if inst.result:
        reg = emitter.allocate_reg(inst.result.name)
        emitter.emit_store(reg)


def emit_ret(emitter: BytecodeEmitter, inst: IRInstruction) -> None:
    """Emit bytecode for a return instruction.

    Pattern:
        ret type %value
    Becomes:
        LOAD %value
        RET

    Args:
        emitter: The bytecode emitter
        inst: The return instruction
    """
    if inst.result_type == "void" or not inst.operands:
        # Void return - push 0 and return
        emitter.emit_push(0)
        emitter.emit_ret()
        return

    # Load the return value
    ret_value = inst.operands[0]
    emitter.load_value(ret_value)

    # Return
    emitter.emit_ret()


# =============================================================================
# Bytecode Formatting Utilities
# =============================================================================

def bytecode_to_c_array(bytecode: bytes, name: str) -> str:
    """Format bytecode as a C array declaration.

    Args:
        bytecode: The bytecode bytes
        name: Name for the array variable

    Returns:
        C code string
    """
    hex_bytes = ", ".join(f"0x{b:02X}" for b in bytecode)
    return f"static const uint8_t {name}[] = {{{hex_bytes}}};"


def bytecode_to_llvm_constant(bytecode: bytes, name: str) -> str:
    """Format bytecode as an LLVM IR constant.

    Args:
        bytecode: The bytecode bytes
        name: Name for the constant

    Returns:
        LLVM IR constant declaration
    """
    # Create the array type
    length = len(bytecode)

    # Create the initializer string
    # Format: c"\x05\x00\x05\x01\x10\xFF"
    hex_str = "".join(f"\\{b:02X}" for b in bytecode)

    return f'@{name} = private constant [{length} x i8] c"{hex_str}"'


def disassemble_bytecode(bytecode: bytes) -> List[str]:
    """Disassemble bytecode to human-readable format.

    Args:
        bytecode: The bytecode bytes

    Returns:
        List of disassembly lines
    """
    lines = []
    i = 0

    opcode_names = {
        0x00: "NOP",
        0x01: "PUSH",
        0x02: "POP",
        0x03: "LOAD",
        0x04: "STORE",
        0x05: "LOAD_ARG",
        0x10: "ADD",
        0x11: "SUB",
        0x12: "XOR",
        0xFF: "RET",
    }

    while i < len(bytecode):
        offset = i
        opcode = bytecode[i]
        i += 1

        name = opcode_names.get(opcode, f"UNKNOWN(0x{opcode:02X})")

        if opcode == 0x01:  # PUSH - 4 byte operand
            if i + 4 <= len(bytecode):
                value = struct.unpack("<i", bytecode[i:i+4])[0]
                lines.append(f"{offset:04X}: {name} {value}")
                i += 4
            else:
                lines.append(f"{offset:04X}: {name} <incomplete>")
        elif opcode in (0x03, 0x04, 0x05):  # LOAD, STORE, LOAD_ARG - 1 byte operand
            if i < len(bytecode):
                reg = bytecode[i]
                lines.append(f"{offset:04X}: {name} {reg}")
                i += 1
            else:
                lines.append(f"{offset:04X}: {name} <incomplete>")
        else:
            lines.append(f"{offset:04X}: {name}")

    return lines
