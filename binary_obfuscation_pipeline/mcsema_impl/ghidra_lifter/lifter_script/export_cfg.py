# Export CFG for McSema
# encoding: utf-8

import json
import sys
import os

print("=" * 60)
print("SCRIPT STARTED: export_cfg.py")
print("=" * 60)
print("Current working directory: %s" % os.getcwd())

try:
    args = getScriptArgs()
    print("Script arguments received: %s" % str(args))
    
    if len(args) == 0:
        print("ERROR: No output CFG path provided")
        sys.exit(1)
    
    output_file = args[0]
    print("Target output file: %s" % output_file)
    
    # Convert to absolute path if needed
    output_file_abs = os.path.abspath(output_file)
    print("Absolute output path: %s" % output_file_abs)
    
    # Check if output directory exists
    output_dir = os.path.dirname(output_file_abs)
    print("Output directory: %s" % output_dir)
    
    if not os.path.exists(output_dir):
        print("WARNING: Output directory does not exist, creating: %s" % output_dir)
        try:
            os.makedirs(output_dir)
            print("SUCCESS: Created output directory")
        except Exception as e:
            print("ERROR: Failed to create output directory: %s" % str(e))
            sys.exit(1)
    else:
        print("Output directory exists: YES")
    
    # Check write permissions
    if os.access(output_dir, os.W_OK):
        print("Write permission check: PASS")
    else:
        print("ERROR: No write permission to directory: %s" % output_dir)
        sys.exit(1)
    
    program = currentProgram
    if program is None:
        print("ERROR: No program loaded (currentProgram is None)")
        sys.exit(1)
    
    print("Program loaded: %s" % program.getName())
    listing = program.getListing()
    
    cfg = {
        "format": "mcsema-cfg",
        "version": "1.0",
        "binary_info": {
            "path": str(program.getExecutablePath()),
            "architecture": str(program.getLanguage().getProcessor()),
            "base_address": str(program.getImageBase()),
        },
        "functions": []
    }
    
    print("Starting function enumeration...")
    functions = listing.getFunctions(True)
    func_count = 0
    
    for func in functions:
        func_count += 1
        func_name = func.getName()
        
        if func_count <= 5 or func_count % 10 == 0:
            print("Processing function %d: %s" % (func_count, func_name))
        
        func_entry = str(func.getEntryPoint())
        
        # Basic Blocks
        blocks = []
        body = func.getBody()
        ranges = body.getAddressRanges()
        
        for r in ranges:
            blocks.append({
                "start_addr": str(r.getMinAddress()),
                "end_addr": str(r.getMaxAddress()),
                "size": int(r.getLength())
            })
        
        # Edges
        edges = []
        instructions = listing.getInstructions(func.getBody(), True)
        
        for instr in instructions:
            src = instr.getAddress()
            flows = instr.getFlows()
            flow_type = instr.getFlowType()
            is_branch = flow_type.isJump() or flow_type.isConditional()

            for f in flows:
                edges.append({
                    "from": str(src),
                    "to": str(f),
                    "type": "branch" if is_branch else "flow"
                })
        
        cfg["functions"].append({
            "name": func_name,
            "address": func_entry,
            "size": int(func.getBody().getNumAddresses()),
            "basic_blocks": blocks,
            "edges": edges,
        })
    
    print("Finished processing %d functions" % func_count)
    print("=" * 60)
    print("WRITING OUTPUT FILE")
    print("=" * 60)
    print("Path: %s" % output_file_abs)
    
    # Write output file
    try:
        with open(output_file_abs, "w") as f:
            json.dump(cfg, f, indent=2)
        
        print("File written successfully")
        
        # Verify file exists
        if os.path.exists(output_file_abs):
            file_size = os.path.getsize(output_file_abs)
            print("File exists: YES")
            print("File size: %d bytes" % file_size)
        else:
            print("ERROR: File does not exist after writing!")
            sys.exit(1)
        
        print("=" * 60)
        print("SUCCESS: CFG EXPORTED")
        print("=" * 60)
        print("Output: %s" % output_file_abs)
        print("Functions: %d" % len(cfg["functions"]))
        print("=" * 60)
        
    except Exception as e:
        print("ERROR: Failed to write output file")
        print("Exception: %s" % str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

except Exception as e:
    print("=" * 60)
    print("SCRIPT FAILED WITH EXCEPTION")
    print("=" * 60)
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)