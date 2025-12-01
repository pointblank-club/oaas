#!/usr/bin/env python3
"""
Simple test to find a Jotai benchmark that compiles and works.
Some Jotai benchmarks have compilation issues - this finds working ones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.jotai_benchmark import JotaiBenchmarkManager, BenchmarkCategory
from core import LLVMObfuscator, ObfuscationConfig, ObfuscationLevel, Platform, ObfuscationReport

def find_working_benchmark():
    """Find a benchmark that compiles successfully."""
    
    print("Finding a working Jotai benchmark...")
    print("(Some benchmarks have compilation issues - this is normal)\n")
    
    manager = JotaiBenchmarkManager()
    benchmarks = manager.list_benchmarks(category=BenchmarkCategory.ANGHALEAVES, limit=20)
    
    if not benchmarks:
        print("No benchmarks found")
        return None
    
    print(f"Testing {len(benchmarks)} benchmarks to find one that compiles...\n")
    
    # Simple test: just try to compile baseline
    from core.utils import run_command
    from core.exceptions import ObfuscationError
    
    for i, bench in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] Testing: {bench.name[:60]}...")
        
        # Quick compile test
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_binary = tmpdir_path / "test_bin"
            
            compile_cmd = [
                "clang", "-g", "-O1",
                str(bench),
                "-o", str(test_binary)
            ]
            
            try:
                returncode, stdout, stderr = run_command(compile_cmd)
                print(f"✅ SUCCESS! This benchmark compiles: {bench.name}\n")
                print(f"   Path: {bench}")
                print(f"   Size: {bench.stat().st_size} bytes\n")
                return bench
            except ObfuscationError as e:
                # Check if it's a simple error we can see
                error_msg = str(e)
                if "error:" in error_msg:
                    # Extract first error line
                    error_lines = error_msg.split('\n')
                    for line in error_lines:
                        if 'error:' in line:
                            print(f"   ❌ Compilation error: {line[:80]}")
                            break
                else:
                    print(f"   ❌ Failed: {error_msg[:80]}")
                continue
    
    print("\n⚠️  No working benchmarks found in this batch. Try more benchmarks or a different category.")
    return None

def test_obfuscation_on_working_benchmark(benchmark_path):
    """Test obfuscation on a benchmark that we know compiles."""
    
    print("\n" + "="*60)
    print("Testing Obfuscation on Working Benchmark")
    print("="*60 + "\n")
    
    manager = JotaiBenchmarkManager()
    output_dir = Path("./test_jotai_output")
    output_dir.mkdir(exist_ok=True)
    
    # Simple config - level 2 (basic obfuscation)
    config = ObfuscationConfig(
        level=ObfuscationLevel(2),
        platform=Platform.LINUX,
    )
    config.output.directory = output_dir / benchmark_path.stem
    
    reporter = ObfuscationReport(config.output.directory)
    obfuscator = LLVMObfuscator(reporter=reporter)
    
    print(f"Benchmark: {benchmark_path.name}")
    print(f"Obfuscation level: 2 (basic)")
    print()
    
    result = manager.run_benchmark_test(
        benchmark_path=benchmark_path,
        obfuscator=obfuscator,
        config=config,
        output_dir=config.output.directory
    )
    
    print("\nResults:")
    print(f"  ✓ Compilation: {'PASS' if result.compilation_success else 'FAIL'}")
    print(f"  ✓ Obfuscation: {'PASS' if result.obfuscation_success else 'FAIL'}")
    print(f"  ✓ Functional:  {'PASS' if result.functional_test_passed else 'FAIL'}")
    
    if result.baseline_binary and result.obfuscated_binary:
        print(f"\n  Baseline size:   {result.size_baseline:,} bytes")
        print(f"  Obfuscated size: {result.size_obfuscated:,} bytes")
        if result.size_baseline > 0:
            overhead = ((result.size_obfuscated - result.size_baseline) / result.size_baseline) * 100
            print(f"  Size overhead:   {overhead:+.1f}%")
    
    if result.inputs_tested > 0:
        print(f"\n  Inputs tested:   {result.inputs_tested}")
        print(f"  Inputs passed:   {result.inputs_passed}")
        print(f"  Success rate:    {(result.inputs_passed/result.inputs_tested)*100:.1f}%")
    
    if result.error_message:
        print(f"\n  Error: {result.error_message}")
    
    return result.functional_test_passed

if __name__ == "__main__":
    working_bench = find_working_benchmark()
    
    if working_bench:
        success = test_obfuscation_on_working_benchmark(working_bench)
        sys.exit(0 if success else 1)
    else:
        print("\n💡 Tip: Try running with more benchmarks:")
        print("   python3 -c \"from core.jotai_benchmark import *; m = JotaiBenchmarkManager(); print(len(m.list_benchmarks(limit=100)))\"")
        sys.exit(1)

