"""Tests for the VM Virtualizer modules."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from modules.vm.virtualizer.utils import (
    VM_ADD, VM_SUB, VM_XOR, VM_LOAD, VM_LOAD_ARG, VM_STORE, VM_RET,
    is_supported_instruction, extract_function_name,
)
from modules.vm.virtualizer.ir_parser import (
    parse_ll_file, parse_function_signature, IRValue, IRFunction,
    get_supported_functions,
)
from modules.vm.virtualizer.bytecode_gen import (
    generate_bytecode, disassemble_bytecode, BytecodeEmitter,
)
from modules.vm.virtualizer.ir_writer import (
    generate_stub_function, sanitize_name,
)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_is_supported_add(self):
        """ADD instruction should be supported."""
        supported, reason = is_supported_instruction("  %sum = add i32 %a, %b")
        self.assertTrue(supported)
        self.assertIsNone(reason)

    def test_is_supported_call(self):
        """CALL instruction should be unsupported."""
        supported, reason = is_supported_instruction("  %x = call i32 @foo()")
        self.assertFalse(supported)
        self.assertIn("call", reason)

    def test_is_supported_br(self):
        """BR instruction should be unsupported."""
        supported, reason = is_supported_instruction("  br i1 %cond, label %a, label %b")
        self.assertFalse(supported)
        self.assertIn("br", reason)

    def test_is_supported_load(self):
        """LOAD instruction should be unsupported (Level 1)."""
        supported, reason = is_supported_instruction("  %x = load i32, i32* %ptr")
        self.assertFalse(supported)
        self.assertIn("load", reason)

    def test_extract_function_name(self):
        """Test function name extraction."""
        line = "define i32 @add_numbers(i32 %a, i32 %b) {"
        name = extract_function_name(line)
        self.assertEqual(name, "add_numbers")

    def test_extract_function_name_with_attrs(self):
        """Test function name extraction with attributes."""
        line = "define dso_local i32 @main() #0 {"
        name = extract_function_name(line)
        self.assertEqual(name, "main")


class TestIRParser(unittest.TestCase):
    """Test IR parser functions."""

    def test_parse_function_signature_simple(self):
        """Test parsing simple function signature."""
        line = "define i32 @add(i32 %a, i32 %b) {"
        name, ret_type, args = parse_function_signature(line)

        self.assertEqual(name, "add")
        self.assertEqual(ret_type, "i32")
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0].name, "%a")
        self.assertEqual(args[1].name, "%b")

    def test_parse_function_signature_no_args(self):
        """Test parsing function with no arguments."""
        line = "define i32 @get_value() {"
        name, ret_type, args = parse_function_signature(line)

        self.assertEqual(name, "get_value")
        self.assertEqual(ret_type, "i32")
        self.assertEqual(len(args), 0)

    def test_parse_ll_file_simple(self):
        """Test parsing simple .ll file."""
        test_file = Path(__file__).parent / "test_simple.ll"
        if not test_file.exists():
            self.skipTest("test_simple.ll not found")

        module = parse_ll_file(test_file)

        self.assertIn("add_numbers", module.functions)
        self.assertIn("sub_numbers", module.functions)
        self.assertIn("xor_numbers", module.functions)

        add_func = module.functions["add_numbers"]
        self.assertTrue(add_func.is_supported)
        self.assertEqual(len(add_func.arguments), 2)

    def test_parse_ll_file_unsupported(self):
        """Test parsing file with unsupported functions."""
        test_file = Path(__file__).parent / "test_unsupported.ll"
        if not test_file.exists():
            self.skipTest("test_unsupported.ll not found")

        module = parse_ll_file(test_file)

        # These should be marked unsupported
        self.assertIn("with_call", module.unsupported_functions)
        self.assertIn("with_branch", module.unsupported_functions)
        self.assertIn("with_memory", module.unsupported_functions)

        # simple_add should still be supported
        self.assertIn("simple_add", module.functions)
        self.assertTrue(module.functions["simple_add"].is_supported)


class TestBytecodeGen(unittest.TestCase):
    """Test bytecode generation."""

    def test_bytecode_add(self):
        """Test bytecode generation for add function."""
        # Create a simple add function
        func = IRFunction(
            name="add",
            return_type="i32",
            arguments=[
                IRValue(name="%a", value_type="i32", is_arg=True, arg_index=0),
                IRValue(name="%b", value_type="i32", is_arg=True, arg_index=1),
            ],
        )

        # Add instructions manually
        from modules.vm.virtualizer.ir_parser import IRInstruction

        # %sum = add i32 %a, %b
        func.instructions.append(IRInstruction(
            opcode="add",
            result=IRValue(name="%sum", value_type="i32"),
            operands=[
                IRValue(name="%a", value_type="i32", is_arg=True, arg_index=0),
                IRValue(name="%b", value_type="i32", is_arg=True, arg_index=1),
            ],
            result_type="i32",
        ))

        # ret i32 %sum
        func.instructions.append(IRInstruction(
            opcode="ret",
            operands=[IRValue(name="%sum", value_type="i32")],
            result_type="i32",
        ))

        bytecode = generate_bytecode(func)

        # Expected bytecode:
        # LOAD_ARG 0 (05 00)
        # LOAD_ARG 1 (05 01)
        # ADD (10)
        # STORE 0 (04 00)
        # LOAD 0 (03 00)
        # RET (FF)
        expected = bytes([
            VM_LOAD_ARG, 0x00,
            VM_LOAD_ARG, 0x01,
            VM_ADD,
            VM_STORE, 0x00,
            VM_LOAD, 0x00,
            VM_RET,
        ])

        self.assertEqual(bytecode, expected)

    def test_disassemble(self):
        """Test bytecode disassembly."""
        bytecode = bytes([
            VM_LOAD_ARG, 0x00,
            VM_LOAD_ARG, 0x01,
            VM_ADD,
            VM_RET,
        ])

        lines = disassemble_bytecode(bytecode)
        self.assertEqual(len(lines), 4)
        self.assertIn("LOAD_ARG", lines[0])
        self.assertIn("ADD", lines[2])
        self.assertIn("RET", lines[3])


class TestIRWriter(unittest.TestCase):
    """Test IR writing functions."""

    def test_sanitize_name(self):
        """Test name sanitization."""
        self.assertEqual(sanitize_name("add_numbers"), "add_numbers")
        self.assertEqual(sanitize_name("my.func"), "my_func")
        self.assertEqual(sanitize_name("123start"), "_123start")

    def test_generate_stub(self):
        """Test stub function generation."""
        func = IRFunction(
            name="add",
            return_type="i32",
            arguments=[
                IRValue(name="%a", value_type="i32", is_arg=True, arg_index=0),
                IRValue(name="%b", value_type="i32", is_arg=True, arg_index=1),
            ],
        )

        bytecode = bytes([VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD, VM_RET])
        stub = generate_stub_function(func, bytecode, "bytecode_add")

        # Check that stub contains key elements
        self.assertIn("define i32 @add", stub)
        self.assertIn("vm_execute", stub)
        self.assertIn("bytecode_add", stub)
        self.assertIn("ret i32", stub)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests."""

    def test_virtualize_simple_file(self):
        """Test virtualizing a simple file."""
        test_file = Path(__file__).parent / "test_simple.ll"
        if not test_file.exists():
            self.skipTest("test_simple.ll not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.ll"
            bytecode_file = Path(tmpdir) / "bytecode.h"

            from modules.vm.virtualizer.main import run_virtualizer

            result = run_virtualizer(
                test_file,
                output_file,
                functions=None,
                bytecode_header=bytecode_file,
            )

            self.assertTrue(result["success"])
            self.assertGreater(len(result["functions_virtualized"]), 0)
            self.assertGreater(result["bytecode_size"], 0)

            # Check output files exist
            self.assertTrue(output_file.exists())
            self.assertTrue(bytecode_file.exists())

            # Check output contains vm_execute
            content = output_file.read_text()
            self.assertIn("vm_execute", content)

            # Check bytecode header
            bc_content = bytecode_file.read_text()
            self.assertIn("bytecode_add_numbers", bc_content)


if __name__ == "__main__":
    unittest.main()
