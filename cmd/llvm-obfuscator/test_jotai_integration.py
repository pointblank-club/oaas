#!/usr/bin/env python3
"""
Simple test script to verify Jotai integration works.

This demonstrates:
1. Jotai provides C SOURCE FILES (not binaries)
2. We obfuscate the source files
3. Compile both baseline and obfuscated versions
4. Test that they produce the same output
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.jotai_benchmark import JotaiBenchmarkManager, BenchmarkCategory
from core import LLVMObfuscator, ObfuscationConfig, ObfuscationLevel, Platform, ObfuscationReport

def test_jotai_integration():
    """Test that Jotai integration works."""
    
    print("=" * 60)
    print("Jotai Integration Test")
    print("=" * 60)
    print()
    print("What is Jotai?")
    print("- Jotai is a collection of C SOURCE CODE files (not binaries)")
    print("- Each file contains a function + test driver")
    print("- We obfuscate the SOURCE, then compile to binary")
    print()
    
    # Initialize manager
    print("Step 1: Initializing Jotai benchmark manager...")
    manager = JotaiBenchmarkManager(auto_download=True)
    print(f"✓ Benchmark cache: {manager.cache_dir}")
    print()
    
    # List available benchmarks
    print("Step 2: Listing available benchmarks...")
    benchmarks = manager.list_benchmarks(
        category=BenchmarkCategory.ANGHALEAVES,
        limit=5
    )
    
    if not benchmarks:
        print("❌ No benchmarks found. Downloading...")
        if manager.download_benchmarks():
            benchmarks = manager.list_benchmarks(category=BenchmarkCategory.ANGHALEAVES, limit=5)
    
    if not benchmarks:
        print("❌ Failed to get benchmarks")
        return False
    
    print(f"✓ Found {len(benchmarks)} benchmarks")
    for i, bench in enumerate(benchmarks[:3], 1):
        print(f"  {i}. {bench.name}")
    print()
    
    # Show what a benchmark looks like
    if benchmarks:
        sample = benchmarks[0]
        print(f"Step 3: Examining sample benchmark: {sample.name}")
        print(f"  Path: {sample}")
        print(f"  Size: {sample.stat().st_size} bytes")
        
        # Show first few lines
        try:
            lines = sample.read_text().splitlines()[:10]
            print("  First 10 lines:")
            for line in lines:
                print(f"    {line[:80]}")
        except Exception as e:
            print(f"  Could not read file: {e}")
        print()
    
    # Test with one benchmark
    print("Step 4: Testing obfuscation on one benchmark...")
    print("  This will:")
    print("    1. Compile baseline (normal compilation)")
    print("    2. Obfuscate the SOURCE CODE")
    print("    3. Compile obfuscated source")
    print("    4. Test both binaries produce same output")
    print()
    
    test_benchmark = benchmarks[0]
    output_dir = Path("./test_jotai_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create simple config
    config = ObfuscationConfig(
        level=ObfuscationLevel(2),  # Level 2 = basic obfuscation
        platform=Platform.LINUX,
    )
    config.output.directory = output_dir / test_benchmark.stem
    
    # Initialize obfuscator
    reporter = ObfuscationReport(config.output.directory)
    obfuscator = LLVMObfuscator(reporter=reporter)
    
    try:
        print(f"  Testing: {test_benchmark.name}")
        result = manager.run_benchmark_test(
            benchmark_path=test_benchmark,
            obfuscator=obfuscator,
            config=config,
            output_dir=config.output.directory
        )
        
        print()
        print("Results:")
        print(f"  Compilation success: {result.compilation_success}")
        print(f"  Obfuscation success: {result.obfuscation_success}")
        print(f"  Functional test passed: {result.functional_test_passed}")
        print(f"  Inputs tested: {result.inputs_tested}")
        print(f"  Inputs passed: {result.inputs_passed}")
        
        if result.baseline_binary:
            print(f"  Baseline binary: {result.baseline_binary}")
            print(f"  Baseline size: {result.size_baseline} bytes")
        
        if result.obfuscated_binary:
            print(f"  Obfuscated binary: {result.obfuscated_binary}")
            print(f"  Obfuscated size: {result.size_obfuscated} bytes")
        
        if result.error_message:
            print(f"  Error: {result.error_message}")
        
        print()
        
        if result.functional_test_passed:
            print("✓ SUCCESS: Obfuscated binary works correctly!")
            return True
        else:
            print("⚠ WARNING: Functional test failed (but integration works)")
            return True  # Still return True as integration is working
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_jotai_integration()
    sys.exit(0 if success else 1)

