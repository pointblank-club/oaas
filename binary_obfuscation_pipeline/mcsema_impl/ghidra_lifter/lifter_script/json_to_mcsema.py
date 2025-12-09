#!/usr/bin/env python3
"""
JSON to McSema Protobuf Converter

Converts Ghidra's enhanced JSON CFG output to McSema's protobuf binary format.

Usage:
    python json_to_mcsema.py input.json output.cfg [--arch amd64|x86]

The output .cfg file can be used with mcsema-lift:
    mcsema-lift-11.0 --cfg output.cfg --output program.bc --arch amd64 --os windows
"""

import json
import sys
import os
import base64
import argparse
import logging

# IMPORTANT: Set this before importing protobuf to handle old CFG_pb2.py
# The McSema CFG_pb2.py was generated with older protobuf, this workaround allows it to work
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add McSema protobuf path
# The CFG_pb2.py is located in the McSema installation
MCSEMA_PROTO_PATHS = [
    '/mcsema/mcsema/lib/python3/site-packages/mcsema_disass-3.1.3.8-py3.8.egg',
    '/app/mcsema_proto',  # Fallback path in container
    os.path.join(os.path.dirname(__file__), 'mcsema_proto'),  # Local fallback
]

# Try to import CFG_pb2
CFG_pb2 = None
for proto_path in MCSEMA_PROTO_PATHS:
    if os.path.exists(proto_path):
        sys.path.insert(0, proto_path)
        try:
            from mcsema_disass.ida7 import CFG_pb2
            logger.info(f"Loaded CFG_pb2 from {proto_path}")
            break
        except ImportError:
            continue

if CFG_pb2 is None:
    # If we can't find the installed version, use our bundled copy
    try:
        # Try relative import first (when used as a module)
        from .mcsema_proto import CFG_pb2
        logger.info("Loaded CFG_pb2 from bundled mcsema_proto (relative import)")
    except ImportError:
        try:
            # Try absolute import (when run as script)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            from mcsema_proto import CFG_pb2
            logger.info("Loaded CFG_pb2 from bundled mcsema_proto (absolute import)")
        except ImportError:
            logger.error("Could not import CFG_pb2. Ensure McSema is installed or CFG_pb2.py is available.")
            sys.exit(1)


# Calling Convention Mapping (Global enum in CFG_pb2)
# From CFG_pb2.py:
#   C = 0
#   X86_StdCall = 64
#   X86_FastCall = 65
#   X86_ThisCall = 70
#   X86_64_SysV = 78
#   Win64 = 79
CALLING_CONVENTION_MAP = {
    'cdecl': CFG_pb2.C,
    'c': CFG_pb2.C,
    'stdcall': CFG_pb2.X86_StdCall,
    'fastcall': CFG_pb2.X86_FastCall,
    'thiscall': CFG_pb2.X86_ThisCall,
    'sysv64': CFG_pb2.X86_64_SysV,
    'x86_64_sysv': CFG_pb2.X86_64_SysV,
    'win64': CFG_pb2.Win64,
    'ms_abi': CFG_pb2.Win64,
}

# ExternalFunction has its own CallingConvention enum:
#   CallerCleanup = 0 (cdecl)
#   CalleeCleanup = 1 (stdcall)
#   FastCall = 2
EXT_CALLING_CONVENTION_MAP = {
    'cdecl': CFG_pb2.ExternalFunction.CallerCleanup,
    'c': CFG_pb2.ExternalFunction.CallerCleanup,
    'stdcall': CFG_pb2.ExternalFunction.CalleeCleanup,
    'fastcall': CFG_pb2.ExternalFunction.FastCall,
    'thiscall': CFG_pb2.ExternalFunction.CalleeCleanup,  # thiscall is callee-cleanup
    'win64': CFG_pb2.ExternalFunction.CallerCleanup,  # Win64 is caller cleanup
    'sysv64': CFG_pb2.ExternalFunction.CallerCleanup,
}

# CodeReference OperandType mapping
OPERAND_TYPE_MAP = {
    'immediate': CFG_pb2.CodeReference.ImmediateOperand,
    'memory': CFG_pb2.CodeReference.MemoryOperand,
    'displacement': CFG_pb2.CodeReference.MemoryDisplacementOperand,
    'controlflow': CFG_pb2.CodeReference.ControlFlowOperand,
    'code': CFG_pb2.CodeReference.ControlFlowOperand,  # Ghidra "code" xref
    'data': CFG_pb2.CodeReference.MemoryOperand,  # Ghidra "data" xref
    'offset': CFG_pb2.CodeReference.OffsetTable,
}


def parse_address(addr_str):
    """
    Parse address string (hex or decimal) to integer.

    Handles various Ghidra address formats:
    - Normal hex: "140001000"
    - Prefixed hex: "0x140001000"
    - External: "EXTERNAL:00000007" -> returns the offset
    - Stack: "Stack[-0x4c]" -> returns None (skip these)
    - Register: various register refs -> returns None (skip these)
    """
    if addr_str is None:
        return None
    if isinstance(addr_str, int):
        return addr_str
    addr_str = str(addr_str).strip()

    # Handle EXTERNAL:XXXXXXXX format from Ghidra
    # These are external function references - extract the hex offset
    if addr_str.startswith('EXTERNAL:'):
        hex_part = addr_str.split(':')[1]
        return int(hex_part, 16)

    # Handle Stack references - these are symbolic, skip them
    if addr_str.startswith('Stack['):
        return None

    # Handle register references (e.g., RAX, RBX) - skip them
    if addr_str in ['RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RBP', 'RSP',
                    'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
                    'EAX', 'EBX', 'ECX', 'EDX', 'ESI', 'EDI', 'EBP', 'ESP']:
        return None

    # Handle any other symbolic reference (contains letters not matching hex)
    # Check if it's a pure hex string first
    if addr_str.startswith('0x') or addr_str.startswith('0X'):
        return int(addr_str, 16)

    # Try hex first (common for addresses)
    try:
        return int(addr_str, 16)
    except ValueError:
        # If it contains only digits, try decimal
        if addr_str.isdigit():
            return int(addr_str)
        # Otherwise it's a symbolic reference, skip it
        return None


def get_calling_convention(cc_str, for_external=False, arch='amd64', os_type='windows'):
    """
    Map calling convention string to protobuf enum.

    For 64-bit Windows (amd64+windows), default to Win64 calling convention.
    For 64-bit Linux (amd64+linux), default to SysV calling convention.
    For 32-bit, default to cdecl.
    """
    if cc_str is None:
        cc_str = ''
    cc_str = cc_str.lower().strip()

    # Handle unknown/empty calling conventions based on architecture and OS
    if cc_str in ['', 'unknown', '__cdecl']:
        if arch in ['amd64', 'x86_64']:
            if os_type.lower() in ['windows', 'win']:
                cc_str = 'win64'
            else:
                cc_str = 'sysv64'
        else:
            cc_str = 'cdecl'

    if for_external:
        return EXT_CALLING_CONVENTION_MAP.get(cc_str, CFG_pb2.ExternalFunction.CallerCleanup)
    return CALLING_CONVENTION_MAP.get(cc_str, CFG_pb2.C)


def convert_json_to_mcsema(json_path, output_path, arch='amd64', os_type='windows'):
    """
    Convert Ghidra JSON CFG to McSema protobuf format.

    Args:
        json_path: Path to Ghidra JSON CFG
        output_path: Path to write McSema .cfg protobuf
        arch: Architecture (amd64 or x86)
        os_type: Operating system (windows, linux, macos)

    Returns:
        dict with conversion stats
    """
    logger.info(f"Loading JSON CFG from: {json_path}")

    with open(json_path, 'r') as f:
        cfg_json = json.load(f)

    # Validate format
    fmt = cfg_json.get('format', '')
    if 'mcsema' not in fmt.lower():
        logger.warning(f"Unexpected format: {fmt}. Expected 'mcsema-cfg-enhanced'")

    # Build executable address ranges from segments
    # This helps filter out functions/blocks in non-code sections like .idata
    executable_ranges = []
    non_executable_sections = set()  # Names of non-executable sections
    for seg_json in cfg_json.get('segments', []):
        perms = seg_json.get('permissions', '')
        seg_name = seg_json.get('name', '')
        start = parse_address(seg_json.get('start_address', '0')) or 0
        end = parse_address(seg_json.get('end_address', '0')) or start
        size = seg_json.get('size', 0)
        if end < start:
            end = start + size

        if 'x' in perms:
            executable_ranges.append((start, end))
            logger.debug(f"  Executable range: {seg_name} 0x{start:x}-0x{end:x}")
        else:
            non_executable_sections.add(seg_name.lower())
            logger.debug(f"  Non-executable: {seg_name} 0x{start:x}-0x{end:x}")

    def is_executable_address(addr):
        """Check if an address falls within an executable segment."""
        if addr is None:
            return False
        for start, end in executable_ranges:
            if start <= addr <= end:
                return True
        return False

    # Create Module
    module = CFG_pb2.Module()

    # Set module name (binary path)
    binary_info = cfg_json.get('binary_info', {})
    module.name = binary_info.get('path', binary_info.get('name', 'unknown'))

    logger.info(f"Converting module: {module.name}")
    logger.info(f"Architecture: {arch}")
    logger.info(f"Executable ranges: {len(executable_ranges)}, Non-executable sections: {non_executable_sections}")

    stats = {
        'segments': 0,
        'functions': 0,
        'blocks': 0,
        'instructions': 0,
        'external_functions': 0,
        'external_variables': 0,
        'global_variables': 0,
        'skipped_thunks': 0,
        'skipped_non_executable': 0,
    }

    # =========================================================================
    # SEGMENTS
    # =========================================================================
    logger.info("Converting segments...")

    for seg_json in cfg_json.get('segments', []):
        perms = seg_json.get('permissions', 'r')
        seg_name = seg_json.get('name', '')
        is_executable = 'x' in perms

        segment = module.segments.add()

        # ea = effective address (start address)
        segment.ea = parse_address(seg_json.get('start_address', '0')) or 0

        # Get segment size
        seg_size = seg_json.get('size', 0)

        # Include data for ALL segments - mcsema needs data segments for memory references
        # The key is to NOT create functions/basic blocks in non-executable segments
        data_b64 = seg_json.get('data_base64', '')
        if data_b64:
            try:
                segment.data = base64.b64decode(data_b64)
            except Exception as e:
                logger.warning(f"Failed to decode segment data: {e}")
                segment.data = bytes(seg_size) if seg_size > 0 else b'\x00'
        else:
            segment.data = bytes(seg_size) if seg_size > 0 else b'\x00'

        # Permissions - data segments should be read-only or writable, NOT executable
        segment.read_only = 'w' not in perms

        # Segment properties
        segment.is_external = False
        segment.name = seg_name
        segment.is_exported = False
        segment.is_thread_local = False

        stats['segments'] += 1
        exec_mark = " [EXEC]" if is_executable else " [DATA]"
        logger.debug(f"  Segment: {seg_name} @ 0x{segment.ea:x} ({len(segment.data)} bytes){exec_mark}")

    logger.info(f"  Converted {stats['segments']} segments")

    # =========================================================================
    # FUNCTIONS
    # =========================================================================
    logger.info("Converting functions...")

    for func_json in cfg_json.get('functions', []):
        func_name = func_json.get('name', '')
        func_addr = parse_address(func_json.get('address', '0')) or 0

        # Check if this is a thunk function (jumps to import table)
        is_thunk = func_json.get('is_thunk', False)

        # Skip functions outside executable sections
        if not is_executable_address(func_addr):
            stats['skipped_non_executable'] += 1
            logger.debug(f"  Skipping non-executable function: {func_name} @ 0x{func_addr:x}")
            continue

        function = module.funcs.add()

        # Function address and name
        function.ea = func_addr

        if func_name:
            function.name = func_name

        # Entry point detection
        function.is_entrypoint = func_json.get('is_entrypoint', False)

        # If name is main/entry variants, mark as entrypoint
        if func_name.lower() in ['main', '_main', 'wmain', '_wmain', 'winmain', 'wwinmain', 'entry', '_start']:
            function.is_entrypoint = True

        # Basic blocks
        # For thunk functions, only include blocks that stay in executable sections
        for block_json in func_json.get('basic_blocks', []):
            block_addr = parse_address(block_json.get('start_address', '0')) or 0

            # Skip blocks outside executable sections
            if not is_executable_address(block_addr):
                logger.debug(f"    Skipping non-executable block @ 0x{block_addr:x}")
                continue

            block = function.blocks.add()

            block.ea = block_addr
            block.is_referenced_by_data = block_json.get('is_referenced_by_data', False)

            # Instructions
            for instr_json in block_json.get('instructions', []):
                instr_addr = parse_address(instr_json.get('address', '0')) or 0

                # Skip instructions that are outside executable sections
                if not is_executable_address(instr_addr):
                    continue

                instruction = block.instructions.add()
                instruction.ea = instr_addr

                # Cross-references
                for xref_json in instr_json.get('xrefs', []):
                    target_addr = parse_address(xref_json.get('target', '0'))

                    # Skip symbolic references (Stack, registers, etc.)
                    if target_addr is None:
                        continue

                    xref_type = xref_json.get('type', 'code').lower()

                    # For code xrefs to non-executable sections (like .idata),
                    # these are likely indirect jumps to import entries
                    # Skip them to prevent mcsema from trying to decode data as code
                    if xref_type == 'code' and not is_executable_address(target_addr):
                        logger.debug(f"      Skipping code xref to non-executable @ 0x{target_addr:x}")
                        continue

                    xref = instruction.xrefs.add()
                    xref.ea = target_addr

                    # Operand type based on xref type
                    xref.operand_type = OPERAND_TYPE_MAP.get(
                        xref_type,
                        CFG_pb2.CodeReference.ControlFlowOperand
                    )

                    # Mask (not used by Ghidra, default to 0)
                    xref.mask = 0

                stats['instructions'] += 1

            # Successor addresses (control flow targets) - only to executable sections
            for succ_addr in block_json.get('successor_addresses', []):
                parsed = parse_address(succ_addr)
                if parsed is not None and is_executable_address(parsed):
                    block.successor_eas.append(parsed)

            stats['blocks'] += 1

        # Track thunks
        if is_thunk:
            stats['skipped_thunks'] += 1  # Tracking as "processed thunks" now

        stats['functions'] += 1

        if stats['functions'] <= 10 or stats['functions'] % 50 == 0:
            logger.info(f"  Function {stats['functions']}: {func_name} @ 0x{function.ea:x}")

    logger.info(f"  Converted {stats['functions']} functions, {stats['blocks']} blocks, {stats['instructions']} instructions")
    if stats['skipped_thunks'] > 0 or stats['skipped_non_executable'] > 0:
        logger.info(f"  Thunk functions: {stats['skipped_thunks']}, Skipped non-executable: {stats['skipped_non_executable']}")

    # =========================================================================
    # EXTERNAL FUNCTIONS (Imports)
    # =========================================================================
    logger.info("Converting external functions...")

    for ext_json in cfg_json.get('external_functions', []):
        ext_func = module.external_funcs.add()

        ext_func.name = ext_json.get('name', '')
        ext_func.ea = parse_address(ext_json.get('address', '0')) or 0

        # Calling convention
        cc_str = ext_json.get('calling_convention', 'cdecl')
        ext_func.cc = get_calling_convention(cc_str, for_external=True, arch=arch, os_type=os_type)

        # Return behavior
        ext_func.has_return = ext_json.get('has_return', True)
        ext_func.no_return = ext_json.get('no_return', False)

        # Argument count (-1 means unknown/variadic)
        arg_count = ext_json.get('argument_count', -1)
        ext_func.argument_count = arg_count if arg_count >= 0 else 0

        # Weak symbol
        ext_func.is_weak = ext_json.get('is_weak', False)

        stats['external_functions'] += 1

    logger.info(f"  Converted {stats['external_functions']} external functions")

    # =========================================================================
    # EXTERNAL VARIABLES
    # =========================================================================
    logger.info("Converting external variables...")

    for ext_var_json in cfg_json.get('external_variables', []):
        ext_var = module.external_vars.add()

        ext_var.name = ext_var_json.get('name', '')
        ext_var.ea = parse_address(ext_var_json.get('address', '0')) or 0
        ext_var.size = ext_var_json.get('size', 8)  # Default pointer size
        ext_var.is_weak = ext_var_json.get('is_weak', False)
        ext_var.is_thread_local = ext_var_json.get('is_thread_local', False)

        stats['external_variables'] += 1

    logger.info(f"  Converted {stats['external_variables']} external variables")

    # =========================================================================
    # GLOBAL VARIABLES
    # =========================================================================
    logger.info("Converting global variables...")

    for gvar_json in cfg_json.get('global_variables', []):
        gvar = module.global_vars.add()

        gvar.ea = parse_address(gvar_json.get('address', '0')) or 0
        gvar.name = gvar_json.get('name', '')
        gvar.size = gvar_json.get('size', 0)

        stats['global_variables'] += 1

    logger.info(f"  Converted {stats['global_variables']} global variables")

    # =========================================================================
    # SERIALIZE TO PROTOBUF
    # =========================================================================
    logger.info(f"Writing protobuf CFG to: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'wb') as f:
        f.write(module.SerializeToString())

    file_size = os.path.getsize(output_path)
    logger.info(f"Output size: {file_size} bytes ({file_size / 1024:.2f} KB)")

    # Print summary
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Module: {module.name}")
    logger.info(f"  Segments: {stats['segments']}")
    logger.info(f"  Functions: {stats['functions']}")
    logger.info(f"  Basic Blocks: {stats['blocks']}")
    logger.info(f"  Instructions: {stats['instructions']}")
    logger.info(f"  External Functions: {stats['external_functions']}")
    logger.info(f"  External Variables: {stats['external_variables']}")
    logger.info(f"  Global Variables: {stats['global_variables']}")
    if stats['skipped_thunks'] > 0 or stats['skipped_non_executable'] > 0:
        logger.info(f"  Thunk Functions: {stats['skipped_thunks']}")
        logger.info(f"  Skipped Non-Executable: {stats['skipped_non_executable']}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert Ghidra JSON CFG to McSema protobuf format'
    )
    parser.add_argument('input_json', help='Input JSON CFG file from Ghidra')
    parser.add_argument('output_cfg', help='Output McSema .cfg protobuf file')
    parser.add_argument('--arch', choices=['amd64', 'x86', 'aarch64', 'arm'],
                        default='amd64', help='Target architecture (default: amd64)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.input_json):
        logger.error(f"Input file not found: {args.input_json}")
        sys.exit(1)

    try:
        stats = convert_json_to_mcsema(args.input_json, args.output_cfg, args.arch)

        # Print next steps
        print("\nNext step:")
        print(f"  mcsema-lift-11.0 --cfg {args.output_cfg} --output program.bc --arch {args.arch} --os windows")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
