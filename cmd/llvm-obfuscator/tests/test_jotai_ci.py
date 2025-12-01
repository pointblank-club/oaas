#!/usr/bin/env python3
"""
CI Integration Test for Jotai Benchmarks

This script:
1. Gets all C source files from Jotai
2. Creates baseline binaries (normal compilation)
3. Runs obfuscation on source files
4. Creates obfuscated binaries
5. Runs both binaries with same inputs
6. Confirms they produce identical output

Designed for CI/CD integration with proper exit codes and reporting.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.jotai_benchmark import (
    JotaiBenchmarkManager,
    BenchmarkCategory,
    BenchmarkResult
)
from core import (
    LLVMObfuscator,
    ObfuscationConfig,
    ObfuscationLevel,
    Platform,
    ObfuscationReport
)


class JotaiCITester:
    """CI-friendly Jotai benchmark tester."""
    
    def __init__(
        self,
        output_dir: Path = Path("./jotai_ci_results"),
        obfuscation_level: int = 3,
        max_benchmarks: int = 50,
        min_success_rate: float = 0.7,  # 70% success rate required
        random_seed: Optional[int] = None,  # For reproducible random selection
    ):
        self.output_dir = output_dir
        self.obfuscation_level = obfuscation_level
        self.max_benchmarks = max_benchmarks
        self.min_success_rate = min_success_rate
        self.random_seed = random_seed
        self.results: List[BenchmarkResult] = []
        self.summary: Dict = {}
        
    def run_tests(self) -> bool:
        """Run all tests and return True if CI should pass."""
        print("=" * 70)
        print("Jotai CI Integration Test")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Obfuscation level: {self.obfuscation_level}")
        print(f"Max benchmarks: {self.max_benchmarks}")
        print(f"Min success rate: {self.min_success_rate * 100:.0f}%")
        print()
        
        # Initialize
        print("[1/5] Initializing benchmark manager...")
        manager = JotaiBenchmarkManager(auto_download=True)
        print(f"✓ Benchmark cache: {manager.cache_dir}")
        print()
        
        # Get benchmarks
        print(f"[2/5] Getting benchmarks...")
        all_benchmarks = manager.list_benchmarks(
            category=BenchmarkCategory.ANGHALEAVES,
            limit=None  # Get all available
        )
        
        if not all_benchmarks:
            print("❌ ERROR: No benchmarks found!")
            return False
        
        print(f"✓ Found {len(all_benchmarks)} total benchmarks")
        
        # Randomly select subset if needed
        if len(all_benchmarks) > self.max_benchmarks:
            if self.random_seed is not None:
                random.seed(self.random_seed)
                print(f"  Using random seed: {self.random_seed}")
            benchmarks = random.sample(all_benchmarks, self.max_benchmarks)
            print(f"  Randomly selected {len(benchmarks)} benchmarks")
        else:
            benchmarks = all_benchmarks
            print(f"  Using all {len(benchmarks)} benchmarks")
        print()
        
        # Setup obfuscator
        print("[3/5] Setting up obfuscator...")
        config = ObfuscationConfig(
            level=ObfuscationLevel(self.obfuscation_level),
            platform=Platform.LINUX,
        )
        config.output.directory = self.output_dir
        
        reporter = ObfuscationReport(self.output_dir)
        obfuscator = LLVMObfuscator(reporter=reporter)
        print("✓ Obfuscator ready")
        print()
        
        # Run benchmarks
        print(f"[4/5] Running {len(benchmarks)} benchmarks...")
        print("  This will:")
        print("    1. Get C source files from Jotai")
        print("    2. Compile sources → baseline binaries (normal compilation)")
        print("    3. Use LLVM obfuscator on sources → obfuscated binaries")
        print("    4. Run both binaries with same inputs")
        print("    5. Verify functional equivalence (outputs must match)")
        print()
        
        self.results = manager.run_benchmark_suite(
            obfuscator=obfuscator,
            config=config,
            output_dir=self.output_dir,
            category=BenchmarkCategory.ANGHALEAVES,
            limit=self.max_benchmarks,
            max_failures=10,  # Stop after 10 consecutive failures
            skip_compilation_errors=True  # Skip benchmarks that don't compile
        )
        
        print()
        
        # Generate report
        print("[5/5] Generating report...")
        report_file = self.output_dir / "jotai_ci_report.json"
        manager.generate_report(self.results, report_file)
        print(f"✓ Report saved: {report_file}")
        print()
        
        # Calculate summary
        self._calculate_summary()
        
        # Print results
        self._print_summary()
        
        # Check if CI should pass
        return self._check_ci_pass()
    
    def _calculate_summary(self):
        """Calculate test summary statistics."""
        total = len(self.results)
        
        # Filter out skipped benchmarks (compilation errors)
        tested = [r for r in self.results if r.compilation_success]
        skipped = total - len(tested)
        
        # Success metrics
        obfuscation_success = sum(1 for r in tested if r.obfuscation_success)
        functional_success = sum(1 for r in tested if r.functional_test_passed)
        
        # Input testing
        total_inputs = sum(r.inputs_tested for r in tested)
        passed_inputs = sum(r.inputs_passed for r in tested)
        
        # Size metrics
        size_changes = []
        for r in tested:
            if r.size_baseline and r.size_obfuscated and r.size_baseline > 0:
                overhead = ((r.size_obfuscated - r.size_baseline) / r.size_baseline) * 100
                size_changes.append(overhead)
        
        avg_overhead = sum(size_changes) / len(size_changes) if size_changes else 0
        
        self.summary = {
            "total_benchmarks": total,
            "tested": len(tested),
            "skipped": skipped,
            "compilation_success": len(tested),
            "obfuscation_success": obfuscation_success,
            "functional_success": functional_success,
            "total_inputs_tested": total_inputs,
            "inputs_passed": passed_inputs,
            "success_rate": functional_success / len(tested) if tested else 0,
            "input_success_rate": passed_inputs / total_inputs if total_inputs else 0,
            "avg_size_overhead": avg_overhead,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    
    def _print_summary(self):
        """Print test summary."""
        s = self.summary
        
        print("=" * 70)
        print("Test Results Summary")
        print("=" * 70)
        print(f"Total benchmarks:        {s['total_benchmarks']}")
        print(f"Tested:                  {s['tested']}")
        print(f"Skipped (compilation):   {s['skipped']}")
        print()
        print(f"Compilation success:     {s['compilation_success']}/{s['tested']}")
        print(f"Obfuscation success:     {s['obfuscation_success']}/{s['tested']}")
        print(f"Functional tests passed: {s['functional_success']}/{s['tested']}")
        print()
        print(f"Success rate:            {s['success_rate']*100:.1f}%")
        print(f"Input tests:             {s['inputs_passed']}/{s['total_inputs_tested']} passed")
        print(f"Input success rate:      {s['input_success_rate']*100:.1f}%")
        print()
        if s['avg_size_overhead']:
            print(f"Average size overhead:   {s['avg_size_overhead']:+.1f}%")
        print()
        
        # Detailed results
        print("Detailed Results:")
        print("-" * 70)
        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result.functional_test_passed else "❌ FAIL"
            if not result.compilation_success:
                status = "⏭️  SKIP"
            
            name = result.benchmark_name[:50]
            print(f"{i:3d}. {status} {name}")
            
            if result.error_message and not result.compilation_success:
                error = result.error_message.split('\n')[0][:60]
                print(f"     Error: {error}")
        
        print()
    
    def _check_ci_pass(self) -> bool:
        """Check if CI should pass based on success rate."""
        s = self.summary
        
        if s['tested'] == 0:
            print("❌ CI FAIL: No benchmarks were successfully tested")
            return False
        
        if s['success_rate'] < self.min_success_rate:
            print(f"❌ CI FAIL: Success rate {s['success_rate']*100:.1f}% is below minimum {self.min_success_rate*100:.0f}%")
            return False
        
        if s['functional_success'] == 0:
            print("❌ CI FAIL: No functional tests passed")
            return False
        
        print(f"✅ CI PASS: Success rate {s['success_rate']*100:.1f}% meets minimum {self.min_success_rate*100:.0f}%")
        return True
    
    def save_summary(self, file: Path):
        """Save summary to JSON file."""
        summary_data = {
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(file, 'w') as f:
            json.dump(summary_data, f, indent=2)


def main():
    """Main entry point for CI testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CI integration test for Jotai benchmarks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./jotai_ci_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Obfuscation level (1-5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of benchmarks to test"
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.7,
        help="Minimum success rate for CI to pass (0.0-1.0)"
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Path to save JSON summary report"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for benchmark selection (for reproducibility)"
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = JotaiCITester(
        output_dir=args.output,
        obfuscation_level=args.level,
        max_benchmarks=args.limit,
        min_success_rate=args.min_success_rate,
        random_seed=args.random_seed,
    )
    
    success = tester.run_tests()
    
    # Save summary if requested
    if args.json_report:
        tester.save_summary(args.json_report)
        print(f"\nSummary saved to: {args.json_report}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

