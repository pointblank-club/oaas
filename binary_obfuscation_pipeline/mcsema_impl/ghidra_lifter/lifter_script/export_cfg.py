# Export CFG for McSema - Enhanced Version
# encoding: utf-8
#
# This script exports a comprehensive CFG from Ghidra that can be
# converted to McSema's protobuf format for binary lifting.
#
# Output includes:
# - Functions with basic blocks and edges
# - Memory segments with raw bytes
# - External function imports
# - Instruction details with cross-references

import json
import sys
import os
import base64

print("=" * 60)
print("SCRIPT STARTED: export_cfg.py (Enhanced for McSema)")
print("=" * 60)

try:
    args = getScriptArgs()
    print("Script arguments: %s" % str(args))

    if len(args) == 0:
        print("ERROR: No output CFG path provided")
        sys.exit(1)

    output_file = args[0]
    output_file_abs = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_file_abs)

    print("Output file: %s" % output_file_abs)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: %s" % output_dir)

    program = currentProgram
    if program is None:
        print("ERROR: No program loaded")
        sys.exit(1)

    print("Program: %s" % program.getName())
    print("Architecture: %s" % program.getLanguage().getProcessor())
    print("Base address: %s" % program.getImageBase())

    listing = program.getListing()
    memory = program.getMemory()
    symbol_table = program.getSymbolTable()

    # Determine architecture
    lang = program.getLanguage()
    proc = str(lang.getProcessor())
    addr_size = lang.getLanguageDescription().getSize()

    if "x86" in proc.lower() or "386" in proc:
        if addr_size == 64:
            arch = "amd64"
        else:
            arch = "x86"
    elif "arm" in proc.lower():
        if addr_size == 64:
            arch = "aarch64"
        else:
            arch = "arm"
    else:
        arch = "amd64"  # default

    print("Detected architecture: %s (%d-bit)" % (arch, addr_size))

    # Initialize CFG structure
    cfg = {
        "format": "mcsema-cfg-enhanced",
        "version": "2.0",
        "binary_info": {
            "name": program.getName(),
            "path": str(program.getExecutablePath()),
            "architecture": arch,
            "address_size": addr_size,
            "base_address": str(program.getImageBase()),
            "entry_point": str(program.getSymbolTable().getPrimarySymbol(
                program.getSymbolTable().getExternalEntryPointIterator().next()
            ).getAddress()) if program.getSymbolTable().getExternalEntryPointIterator().hasNext() else "0x0"
        },
        "segments": [],
        "functions": [],
        "external_functions": [],
        "external_variables": [],
        "global_variables": []
    }

    # =========================================================================
    # SEGMENT EXTRACTION
    # =========================================================================
    print("\n--- Extracting Memory Segments ---")

    for block in memory.getBlocks():
        block_name = block.getName()
        block_start = block.getStart()
        block_end = block.getEnd()
        block_size = block.getSize()

        # Determine permissions
        perms = ""
        if block.isRead():
            perms += "r"
        if block.isWrite():
            perms += "w"
        if block.isExecute():
            perms += "x"

        # Get raw bytes (only for initialized blocks)
        data_b64 = ""
        if block.isInitialized() and block_size > 0 and block_size < 50 * 1024 * 1024:  # Max 50MB
            try:
                # Use jarray for Java byte array (required by Ghidra's getBytes API)
                import jarray
                from java.lang import Byte
                java_bytes = jarray.zeros(block_size, 'b')
                block.getBytes(block_start, java_bytes)
                # Convert signed Java bytes to unsigned Python bytes
                # Java bytes are signed (-128 to 127), need to convert to 0-255
                # Use bytearray for Jython 2.7 compatibility
                python_bytes = bytearray(block_size)
                for i in range(block_size):
                    # Convert signed byte to unsigned (0-255)
                    python_bytes[i] = java_bytes[i] & 0xFF
                data_b64 = base64.b64encode(str(python_bytes)).decode('ascii')
            except Exception as e:
                print("  Warning: Could not read bytes from %s: %s" % (block_name, str(e)))
                import traceback
                traceback.print_exc()
                data_b64 = ""

        segment = {
            "name": block_name,
            "start_address": str(block_start),
            "end_address": str(block_end),
            "size": int(block_size),
            "permissions": perms,
            "is_initialized": block.isInitialized(),
            "is_executable": block.isExecute(),
            "is_read_only": block.isRead() and not block.isWrite(),
            "data_base64": data_b64
        }

        cfg["segments"].append(segment)
        print("  Segment: %s [%s - %s] %s (%d bytes)" % (
            block_name, block_start, block_end, perms, block_size))

    # =========================================================================
    # EXTERNAL FUNCTION EXTRACTION (Imports)
    # =========================================================================
    print("\n--- Extracting External Functions (Imports) ---")

    ext_manager = program.getExternalManager()
    ext_count = 0

    # Iterate through all external library names, then get locations for each
    for library_name in ext_manager.getExternalLibraryNames():
        for ext_loc in ext_manager.getExternalLocations(library_name):
            try:
                name = ext_loc.getLabel()
                address = ext_loc.getAddress()
                library = library_name

                if address is None:
                    # Try to get from symbol
                    symbol = ext_loc.getSymbol()
                    if symbol:
                        address = symbol.getAddress()

                # Determine calling convention based on library/name heuristics
                # For Windows PE, default to Win64 for 64-bit, cdecl for 32-bit
                if addr_size == 64:
                    calling_conv = "win64"
                else:
                    # Check for known stdcall functions
                    if library and library.lower() in ["kernel32.dll", "user32.dll", "gdi32.dll"]:
                        calling_conv = "stdcall"
                    else:
                        calling_conv = "cdecl"

                # Check for noreturn functions
                no_return = name.lower() in ["exit", "_exit", "abort", "exitprocess", "terminateprocess"]

                ext_func = {
                    "name": name,
                    "address": str(address) if address else "0x0",
                    "library": library if library else "",
                    "calling_convention": calling_conv,
                    "has_return": not no_return,
                    "no_return": no_return,
                    "argument_count": -1,  # Unknown
                    "is_weak": False
                }

                cfg["external_functions"].append(ext_func)
                ext_count += 1

                if ext_count <= 10:
                    print("  Import: %s from %s @ %s" % (name, library, address))
            except Exception as e:
                print("  Warning: Error processing external: %s" % str(e))

    print("  Total external functions: %d" % ext_count)

    # =========================================================================
    # FUNCTION EXTRACTION
    # =========================================================================
    print("\n--- Extracting Functions ---")

    functions = listing.getFunctions(True)
    func_count = 0

    for func in functions:
        func_count += 1
        func_name = func.getName()
        func_entry = func.getEntryPoint()

        if func_count <= 5 or func_count % 20 == 0:
            print("  Processing function %d: %s @ %s" % (func_count, func_name, func_entry))

        # Check if this is the entry point
        is_entry = False
        entry_iter = program.getSymbolTable().getExternalEntryPointIterator()
        while entry_iter.hasNext():
            ep = entry_iter.next()
            if ep.equals(func_entry):
                is_entry = True
                break

        # Also check if it's main
        if func_name.lower() in ["main", "_main", "wmain", "_wmain", "wwinmain", "winmain"]:
            is_entry = True

        # Get calling convention
        cc = func.getCallingConventionName()
        if cc:
            cc_name = str(cc).lower()
        else:
            cc_name = "win64" if addr_size == 64 else "cdecl"

        # =====================================================================
        # BASIC BLOCKS
        # =====================================================================
        blocks = []
        block_map = {}  # address -> block index

        # Use Ghidra's basic block model
        body = func.getBody()
        bm = ghidra.program.model.block.BasicBlockModel(program)
        code_blocks = bm.getCodeBlocksContaining(body, monitor)

        while code_blocks.hasNext():
            cb = code_blocks.next()
            cb_start = cb.getFirstStartAddress()
            cb_end = cb.getMaxAddress()

            # Get instructions in this block
            instructions = []
            instr_iter = listing.getInstructions(cb, True)

            while instr_iter.hasNext():
                instr = instr_iter.next()
                instr_addr = instr.getAddress()
                instr_bytes = instr.getBytes()
                instr_len = instr.getLength()

                # Get cross-references from this instruction
                xrefs = []
                refs = instr.getReferencesFrom()
                for ref in refs:
                    ref_type = str(ref.getReferenceType())
                    if "CALL" in ref_type or "JUMP" in ref_type:
                        xref_type = "code"
                    else:
                        xref_type = "data"

                    xrefs.append({
                        "target": str(ref.getToAddress()),
                        "type": xref_type,
                        "ref_type": ref_type
                    })

                instructions.append({
                    "address": str(instr_addr),
                    "bytes_base64": base64.b64encode(bytes(instr_bytes)).decode('ascii'),
                    "length": int(instr_len),
                    "mnemonic": instr.getMnemonicString(),
                    "operands": str(instr.getDefaultOperandRepresentation(0)) if instr.getNumOperands() > 0 else "",
                    "xrefs": xrefs
                })

            # Get successor addresses
            successors = []
            dest_iter = cb.getDestinations(monitor)
            while dest_iter.hasNext():
                dest = dest_iter.next()
                dest_addr = dest.getDestinationAddress()
                if dest_addr:
                    successors.append(str(dest_addr))

            block_data = {
                "start_address": str(cb_start),
                "end_address": str(cb_end),
                "instructions": instructions,
                "successor_addresses": successors,
                "is_referenced_by_data": False  # TODO: detect data refs
            }

            blocks.append(block_data)
            block_map[str(cb_start)] = len(blocks) - 1

        # =====================================================================
        # EDGES (Control Flow)
        # =====================================================================
        edges = []
        for block in blocks:
            src_addr = block["start_address"]
            for succ_addr in block["successor_addresses"]:
                edge_type = "flow"
                # Check if it's a branch by looking at last instruction
                if block["instructions"]:
                    last_instr = block["instructions"][-1]
                    mnem = last_instr["mnemonic"].lower()
                    if mnem.startswith("j") or mnem.startswith("b") or mnem == "ret":
                        edge_type = "branch"
                    if mnem.startswith("call"):
                        edge_type = "call"

                edges.append({
                    "from": src_addr,
                    "to": succ_addr,
                    "type": edge_type
                })

        # Build function data
        func_data = {
            "name": func_name,
            "address": str(func_entry),
            "size": int(func.getBody().getNumAddresses()),
            "calling_convention": cc_name,
            "is_entrypoint": is_entry,
            "is_thunk": func.isThunk(),
            "basic_blocks": blocks,
            "edges": edges
        }

        cfg["functions"].append(func_data)

    print("\n  Total functions processed: %d" % func_count)

    # =========================================================================
    # GLOBAL VARIABLES
    # =========================================================================
    print("\n--- Extracting Global Variables ---")

    data_iter = listing.getDefinedData(True)
    global_count = 0

    while data_iter.hasNext() and global_count < 1000:  # Limit to prevent huge outputs
        data = data_iter.next()
        try:
            addr = data.getAddress()
            label = data.getLabel()
            size = data.getLength()

            if label and not label.startswith("DAT_"):
                cfg["global_variables"].append({
                    "name": label,
                    "address": str(addr),
                    "size": int(size)
                })
                global_count += 1
        except:
            pass

    print("  Global variables extracted: %d" % global_count)

    # =========================================================================
    # STATISTICS
    # =========================================================================
    total_blocks = sum(len(f["basic_blocks"]) for f in cfg["functions"])
    total_edges = sum(len(f["edges"]) for f in cfg["functions"])
    total_instructions = sum(
        sum(len(b["instructions"]) for b in f["basic_blocks"])
        for f in cfg["functions"]
    )

    cfg["statistics"] = {
        "total_functions": len(cfg["functions"]),
        "total_blocks": total_blocks,
        "total_edges": total_edges,
        "total_instructions": total_instructions,
        "total_segments": len(cfg["segments"]),
        "total_external_functions": len(cfg["external_functions"]),
        "total_global_variables": len(cfg["global_variables"])
    }

    # =========================================================================
    # WRITE OUTPUT
    # =========================================================================
    print("\n" + "=" * 60)
    print("WRITING OUTPUT")
    print("=" * 60)

    with open(output_file_abs, "w") as f:
        json.dump(cfg, f, indent=2)

    file_size = os.path.getsize(output_file_abs)

    print("Output file: %s" % output_file_abs)
    print("File size: %d bytes (%.2f KB)" % (file_size, file_size / 1024.0))
    print("\nStatistics:")
    print("  Functions: %d" % cfg["statistics"]["total_functions"])
    print("  Basic Blocks: %d" % cfg["statistics"]["total_blocks"])
    print("  Edges: %d" % cfg["statistics"]["total_edges"])
    print("  Instructions: %d" % cfg["statistics"]["total_instructions"])
    print("  Segments: %d" % cfg["statistics"]["total_segments"])
    print("  External Functions: %d" % cfg["statistics"]["total_external_functions"])
    print("  Global Variables: %d" % cfg["statistics"]["total_global_variables"])

    print("\n" + "=" * 60)
    print("SUCCESS: Enhanced CFG exported for McSema")
    print("=" * 60)

except Exception as e:
    print("=" * 60)
    print("SCRIPT FAILED")
    print("=" * 60)
    print("Error: %s" % str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
