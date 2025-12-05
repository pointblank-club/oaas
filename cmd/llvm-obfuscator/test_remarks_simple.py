#!/usr/bin/env python3
"""
Simple test script to verify LLVM remarks integration in obfuscator.

This tests that:
1. Remarks can be enabled in obfuscation config
2. Remarks files (.opt.yaml) are generated during obfuscation
3. Remarks files are valid YAML and contain optimization information
"""

import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    LLVMObfuscator,
    ObfuscationConfig,
    ObfuscationLevel,
    Platform,
    AdvancedConfiguration,
    RemarksConfiguration,
    OutputConfiguration,
)


def test_remarks_integration():
    """Test that remarks are generated during obfuscation."""
    print("=" * 70)
    print("Testing LLVM Remarks Integration")
    print("=" * 70)
    print()
    
    # Create a simple test source file
    test_source = Path(__file__).parent / "examples" / "hello.c"
    if not test_source.exists():
        # Create a minimal test file if examples/hello.c doesn't exist
        test_source = Path(__file__).parent / "test_remarks_temp.c"
        test_source.write_text("""
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int x = add(5, 3);
    printf("Result: %d\\n", x);
    return 0;
}
""")
        print(f"✓ Created test source: {test_source}")
    else:
        print(f"✓ Using test source: {test_source}")
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_remarks_output"
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    print()
    
    # Configure obfuscator with remarks enabled
    print("[1/3] Configuring obfuscator with remarks enabled...")
    config = ObfuscationConfig(
        level=ObfuscationLevel.MEDIUM,
        platform=Platform.LINUX,
        advanced=AdvancedConfiguration(
            remarks=RemarksConfiguration(
                enabled=True,
                format="yaml",
                pass_filter=".*"  # Collect remarks from all passes
            )
        ),
        output=OutputConfiguration(directory=output_dir)
    )
    print("✓ Configuration created with remarks enabled")
    print()
    
    # Run obfuscation
    print("[2/3] Running obfuscation...")
    obfuscator = LLVMObfuscator()
    try:
        result = obfuscator.obfuscate(test_source, config)
        print(f"✓ Obfuscation completed")
        print(f"  Output binary: {result.get('output_file', 'N/A')}")
    except Exception as e:
        print(f"❌ Obfuscation failed: {e}")
        return False
    print()
    
    # Check for remarks file
    print("[3/3] Verifying remarks file...")
    expected_remarks_file = output_dir / f"{test_source.stem}.opt.yaml"
    
    if not expected_remarks_file.exists():
        print(f"❌ Remarks file not found: {expected_remarks_file}")
        print("   This could mean:")
        print("   - Clang doesn't support remarks (need LLVM 9.0+)")
        print("   - No optimizations were applied (need -O2 or higher)")
        print("   - Remarks flags weren't added to compilation command")
        return False
    
    print(f"✓ Remarks file found: {expected_remarks_file}")
    
    # Check file size
    file_size = expected_remarks_file.stat().st_size
    print(f"  File size: {file_size} bytes")
    
    if file_size == 0:
        print("⚠️  Warning: Remarks file is empty")
        return False
    
    # Try to parse YAML
    try:
        with open(expected_remarks_file) as f:
            remarks = [r for r in yaml.safe_load_all(f) if r]
        
        print(f"✓ YAML is valid")
        print(f"  Total remarks: {len(remarks)}")
        
        if len(remarks) == 0:
            print("⚠️  Warning: No remarks found in file")
            return False
        
        # Show sample remark
        if remarks:
            sample = remarks[0]
            print(f"  Sample remark:")
            print(f"    Pass: {sample.get('Pass', 'N/A')}")
            print(f"    Name: {sample.get('Name', 'N/A')}")
            print(f"    Function: {sample.get('Function', 'N/A')}")
        
        # Count remarks by type
        passed = sum(1 for r in remarks if r.get('Type') == 'Passed' or '!Passed' in str(r))
        missed = sum(1 for r in remarks if r.get('Type') == 'Missed' or '!Missed' in str(r))
        analysis = sum(1 for r in remarks if r.get('Type') == 'Analysis' or '!Analysis' in str(r))
        
        print(f"  Remarks breakdown:")
        print(f"    Passed optimizations: {passed}")
        print(f"    Missed optimizations: {missed}")
        print(f"    Analysis remarks: {analysis}")
        
    except yaml.YAMLError as e:
        print(f"❌ Failed to parse YAML: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading remarks: {e}")
        return False
    
    print()
    print("=" * 70)
    print("✅ Test PASSED: Remarks integration is working!")
    print("=" * 70)
    print()
    print(f"Remarks file location: {expected_remarks_file}")
    print("You can view it with:")
    print(f"  cat {expected_remarks_file}")
    print()
    
    # Cleanup temp file if we created it
    if test_source.name == "test_remarks_temp.c":
        test_source.unlink()
    
    return True


if __name__ == "__main__":
    success = test_remarks_integration()
    sys.exit(0 if success else 1)

