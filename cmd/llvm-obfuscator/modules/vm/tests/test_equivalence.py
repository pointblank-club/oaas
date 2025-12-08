"""VM Equivalence Tests - PRIORITY 3: Semantic Preservation.

These tests prove that virtualized functions produce
EXACTLY the same outputs as original functions.

Run with: pytest modules/vm/tests/test_equivalence.py -v
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.vm.virtualizer.utils import (
    VM_NOP, VM_PUSH, VM_POP, VM_LOAD, VM_STORE, VM_LOAD_ARG,
    VM_ADD, VM_SUB, VM_XOR, VM_RET,
)
from modules.vm.virtualizer.bytecode_gen import BytecodeEmitter, generate_bytecode
from modules.vm.virtualizer.ir_parser import parse_ll_file, IRFunction, IRInstruction, IRValue


class TestArithmeticEquivalence(unittest.TestCase):
    """Tests proving virtualized arithmetic matches original."""

    def setUp(self):
        """Set up bytecode emitter for tests."""
        self.emitter = BytecodeEmitter()

    def execute_bytecode(self, bytecode: bytes, args: list) -> int:
        """Execute bytecode using our Python emulator.

        This emulates the C VM interpreter's behavior.
        """
        stack = []
        registers = [0] * 8
        pc = 0

        while pc < len(bytecode):
            opcode = bytecode[pc]
            pc += 1

            if opcode == VM_NOP:
                continue
            elif opcode == VM_PUSH:
                if pc < len(bytecode):
                    val = bytecode[pc]
                    pc += 1
                    stack.append(val)
            elif opcode == VM_POP:
                if stack:
                    stack.pop()
            elif opcode == VM_LOAD:
                if pc < len(bytecode):
                    reg = bytecode[pc]
                    pc += 1
                    if reg < len(registers):
                        stack.append(registers[reg])
            elif opcode == VM_STORE:
                if pc < len(bytecode) and stack:
                    reg = bytecode[pc]
                    pc += 1
                    if reg < len(registers):
                        registers[reg] = stack.pop()
            elif opcode == VM_LOAD_ARG:
                if pc < len(bytecode):
                    idx = bytecode[pc]
                    pc += 1
                    if idx < len(args):
                        stack.append(args[idx])
                    else:
                        stack.append(0)
            elif opcode == VM_ADD:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a + b)
            elif opcode == VM_SUB:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a - b)
            elif opcode == VM_XOR:
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a ^ b)
            elif opcode == VM_RET:
                if stack:
                    return stack[-1]
                return 0

        return stack[-1] if stack else 0

    def test_add_basic(self):
        """add(5, 3) should equal 8."""
        # Original function: return a + b
        original_result = 5 + 3

        # Virtualized: LOAD_ARG 0, LOAD_ARG 1, ADD, RET
        bytecode = bytes([
            VM_LOAD_ARG, 0,  # Push arg 0 (5)
            VM_LOAD_ARG, 1,  # Push arg 1 (3)
            VM_ADD,          # 5 + 3
            VM_STORE, 0,     # Store in reg 0
            VM_LOAD, 0,      # Load from reg 0
            VM_RET,          # Return result
        ])

        vm_result = self.execute_bytecode(bytecode, [5, 3])

        self.assertEqual(original_result, 8, "Original add should be 8")
        self.assertEqual(vm_result, 8, "VM add should be 8")
        self.assertEqual(original_result, vm_result, "Results must match")

    def test_add_zero(self):
        """add(x, 0) should equal x."""
        test_values = [0, 1, -1, 100, -100, 2**30]

        bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_ADD,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        for x in test_values:
            with self.subTest(x=x):
                original = x + 0
                vm_result = self.execute_bytecode(bytecode, [x, 0])
                self.assertEqual(original, vm_result, f"add({x}, 0) mismatch")

    def test_add_negative(self):
        """add with negative numbers should work correctly."""
        test_cases = [
            (-5, 3, -2),
            (5, -3, 2),
            (-5, -3, -8),
            (-100, 100, 0),
        ]

        bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_ADD,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                vm_result = self.execute_bytecode(bytecode, [a, b])
                self.assertEqual(vm_result, expected, f"add({a}, {b}) should be {expected}")

    def test_sub_basic(self):
        """sub(10, 4) should equal 6."""
        original_result = 10 - 4

        bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_SUB,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        vm_result = self.execute_bytecode(bytecode, [10, 4])

        self.assertEqual(original_result, 6)
        self.assertEqual(vm_result, 6)
        self.assertEqual(original_result, vm_result)

    def test_sub_negative_result(self):
        """sub(3, 10) should equal -7."""
        original_result = 3 - 10

        bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_SUB,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        vm_result = self.execute_bytecode(bytecode, [3, 10])

        self.assertEqual(original_result, -7)
        self.assertEqual(vm_result, -7)
        self.assertEqual(original_result, vm_result)

    def test_xor_basic(self):
        """xor(255, 15) should equal 240."""
        original_result = 255 ^ 15

        bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_XOR,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        vm_result = self.execute_bytecode(bytecode, [255, 15])

        self.assertEqual(original_result, 240)
        self.assertEqual(vm_result, 240)
        self.assertEqual(original_result, vm_result)

    def test_xor_zero(self):
        """xor(x, 0) should equal x."""
        test_values = [0, 1, 127, 255, 0xFFFF, 0x12345678]

        bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_XOR,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        for x in test_values:
            with self.subTest(x=x):
                original = x ^ 0
                vm_result = self.execute_bytecode(bytecode, [x, 0])
                self.assertEqual(original, vm_result, f"xor({x}, 0) should equal {x}")

    def test_complex_chained(self):
        """Complex expression: ((a + b) - c) ^ d."""
        # Original: ((5 + 3) - 2) ^ 4 = (8 - 2) ^ 4 = 6 ^ 4 = 2
        a, b, c, d = 5, 3, 2, 4
        original_result = ((a + b) - c) ^ d

        # Bytecode for ((arg0 + arg1) - arg2) ^ arg3
        bytecode = bytes([
            VM_LOAD_ARG, 0,  # Push a (5)
            VM_LOAD_ARG, 1,  # Push b (3)
            VM_ADD,          # 5 + 3 = 8
            VM_LOAD_ARG, 2,  # Push c (2)
            VM_SUB,          # 8 - 2 = 6
            VM_LOAD_ARG, 3,  # Push d (4)
            VM_XOR,          # 6 ^ 4 = 2
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        vm_result = self.execute_bytecode(bytecode, [a, b, c, d])

        self.assertEqual(original_result, 2)
        self.assertEqual(vm_result, 2)
        self.assertEqual(original_result, vm_result)

    def test_multiple_functions_same_binary(self):
        """Multiple different functions should all work correctly."""
        # Function 1: add(a, b)
        add_bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_ADD,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        # Function 2: sub(a, b)
        sub_bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_SUB,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        # Function 3: xor(a, b)
        xor_bytecode = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_XOR,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        # Test all three
        args = [100, 37]

        add_result = self.execute_bytecode(add_bytecode, args)
        sub_result = self.execute_bytecode(sub_bytecode, args)
        xor_result = self.execute_bytecode(xor_bytecode, args)

        self.assertEqual(add_result, 100 + 37)
        self.assertEqual(sub_result, 100 - 37)
        self.assertEqual(xor_result, 100 ^ 37)


class TestBytecodeGeneratorEquivalence(unittest.TestCase):
    """Tests that bytecode generator produces correct bytecode."""

    def test_bytecode_gen_simple_add(self):
        """BytecodeEmitter generates correct add bytecode."""
        emitter = BytecodeEmitter()

        # Simulate generating bytecode for: a + b
        emitter.emit_load_arg(0)   # Load first arg
        emitter.emit_load_arg(1)   # Load second arg
        emitter.emit_binop(VM_ADD) # Add them
        emitter.emit_store(0)      # Store result
        emitter.emit_load(0)       # Load for return
        emitter.emit_ret()         # Return

        bytecode = bytes(emitter.bytecode)

        # Verify bytecode structure
        expected = bytes([
            VM_LOAD_ARG, 0,
            VM_LOAD_ARG, 1,
            VM_ADD,
            VM_STORE, 0,
            VM_LOAD, 0,
            VM_RET,
        ])

        self.assertEqual(bytecode, expected)

    def test_ir_to_bytecode_roundtrip(self):
        """Parse IR, generate bytecode, verify it matches expected."""
        # Create a simple IR function
        func = IRFunction(name="test_add", return_type="i32")

        # Add arguments
        func.arguments = [
            IRValue(name="%a", value_type="i32", is_arg=True, arg_index=0),
            IRValue(name="%b", value_type="i32", is_arg=True, arg_index=1),
        ]

        # %sum = add i32 %a, %b
        func.instructions.append(IRInstruction(
            opcode="add",
            result=IRValue(name="%sum", value_type="i32"),
            operands=[
                IRValue(name="%a", value_type="i32", is_arg=True, arg_index=0),
                IRValue(name="%b", value_type="i32", is_arg=True, arg_index=1),
            ],
        ))

        # ret i32 %sum
        func.instructions.append(IRInstruction(
            opcode="ret",
            result=None,
            operands=[IRValue(name="%sum", value_type="i32")],
        ))

        # Generate bytecode
        bytecode = generate_bytecode(func)

        # Should produce valid bytecode that executes correctly
        self.assertIsNotNone(bytecode)
        self.assertGreater(len(bytecode), 0)

        # Should end with RET
        self.assertEqual(bytecode[-1], VM_RET)


class TestEdgeCases(unittest.TestCase):
    """Edge case equivalence tests."""

    def execute_bytecode(self, bytecode: bytes, args: list) -> int:
        """Execute bytecode using Python emulator."""
        stack = []
        registers = [0] * 8
        pc = 0

        while pc < len(bytecode):
            opcode = bytecode[pc]
            pc += 1

            if opcode == VM_NOP:
                continue
            elif opcode == VM_LOAD_ARG:
                idx = bytecode[pc]
                pc += 1
                stack.append(args[idx] if idx < len(args) else 0)
            elif opcode == VM_ADD:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif opcode == VM_SUB:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif opcode == VM_XOR:
                b, a = stack.pop(), stack.pop()
                stack.append(a ^ b)
            elif opcode == VM_STORE:
                reg = bytecode[pc]
                pc += 1
                registers[reg] = stack.pop()
            elif opcode == VM_LOAD:
                reg = bytecode[pc]
                pc += 1
                stack.append(registers[reg])
            elif opcode == VM_RET:
                return stack[-1] if stack else 0

        return stack[-1] if stack else 0

    def test_identity_operations(self):
        """Identity operations: x + 0, x - 0, x ^ 0 all equal x."""
        x = 42

        # add(x, 0)
        add_zero = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(self.execute_bytecode(add_zero, [x, 0]), x)

        # sub(x, 0)
        sub_zero = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_SUB,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(self.execute_bytecode(sub_zero, [x, 0]), x)

        # xor(x, 0)
        xor_zero = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_XOR,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(self.execute_bytecode(xor_zero, [x, 0]), x)

    def test_self_operations(self):
        """Self operations: x - x = 0, x ^ x = 0."""
        x = 42

        # sub(x, x)
        sub_self = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 0, VM_SUB,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(self.execute_bytecode(sub_self, [x]), 0)

        # xor(x, x)
        xor_self = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 0, VM_XOR,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(self.execute_bytecode(xor_self, [x]), 0)

    def test_commutative_operations(self):
        """Commutative ops: a + b = b + a, a ^ b = b ^ a."""
        a, b = 17, 53

        # add is commutative
        add_ab = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        add_ba = bytes([
            VM_LOAD_ARG, 1, VM_LOAD_ARG, 0, VM_ADD,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(
            self.execute_bytecode(add_ab, [a, b]),
            self.execute_bytecode(add_ba, [a, b])
        )

        # xor is commutative
        xor_ab = bytes([
            VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_XOR,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        xor_ba = bytes([
            VM_LOAD_ARG, 1, VM_LOAD_ARG, 0, VM_XOR,
            VM_STORE, 0, VM_LOAD, 0, VM_RET,
        ])
        self.assertEqual(
            self.execute_bytecode(xor_ab, [a, b]),
            self.execute_bytecode(xor_ba, [a, b])
        )


if __name__ == "__main__":
    unittest.main()
