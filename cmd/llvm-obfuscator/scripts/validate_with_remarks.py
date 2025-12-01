#!/usr/bin/env python3
"""
Validate obfuscation effectiveness using LLVM Remarks.

This script compiles source code with and without obfuscation,
collects LLVM remarks, and validates that obfuscation is working.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llvm_remarks import analyze_obfuscation_with_remarks


def main():
    parser = argparse.ArgumentParser(
        description="Validate obfuscation using LLVM Remarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  %(prog)s --source src/auth.c --output-dir results/

  # With threshold requirements
  %(prog)s --source src/auth.c --output-dir results/ \\
    --min-remarks 100 --min-optimizations 50

  # Custom compiler flags
  %(prog)s --source src/auth.c --output-dir results/ \\
    --flags "-O3 -flto -fvisibility=hidden"

  # Output JSON for CI/CD
  %(prog)s --source src/auth.c --output-dir results/ \\
    --json results/metrics.json
"""
    )
    
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source file to compile and validate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Output directory for binaries and remarks"
    )
    
    parser.add_argument(
        "--flags",
        type=str,
        default="-O2 -flto",
        help="Compiler flags for obfuscated build (default: -O2 -flto)"
    )
    
    parser.add_argument(
        "--min-remarks",
        type=int,
        default=0,
        help="Minimum number of remarks required (0 = no check)"
    )
    
    parser.add_argument(
        "--min-optimizations",
        type=int,
        default=0,
        help="Minimum number of optimizations required (0 = no check)"
    )
    
    parser.add_argument(
        "--min-inlining",
        type=int,
        default=0,
        help="Minimum number of inlining decisions required (0 = no check)"
    )
    
    parser.add_argument(
        "--json",
        type=Path,
        help="Output results as JSON to specified file"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.source.exists():
        print(f"❌ Error: Source file not found: {args.source}", file=sys.stderr)
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    baseline_binary = args.output_dir / f"{args.source.stem}_baseline"
    obfuscated_binary = args.output_dir / f"{args.source.stem}_obfuscated"
    
    # Parse compiler flags
    compiler_flags = args.flags.split()
    
    print("🔍 LLVM Remarks Validation")
    print("=" * 60)
    print(f"Source:           {args.source}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Compiler Flags:   {args.flags}")
    print("=" * 60)
    print()
    
    # Run analysis
    print("📊 Analyzing obfuscation with LLVM remarks...")
    print()
    
    result = analyze_obfuscation_with_remarks(
        source_file=args.source,
        baseline_binary=baseline_binary,
        obfuscated_binary=obfuscated_binary,
        compiler_flags=compiler_flags
    )
    
    if result["status"] != "success":
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return 1
    
    # Display results
    comparison = result["comparison"]
    baseline = comparison["baseline"]
    obfuscated = comparison["obfuscated"]
    delta = comparison["delta"]
    
    print("📈 Results")
    print("-" * 60)
    print()
    
    print("Baseline (no obfuscation):")
    print(f"  Total remarks:         {baseline['total_remarks']}")
    print(f"  Optimizations applied: {baseline['optimizations_applied']}")
    print(f"  Inlining decisions:    {baseline['inlining_decisions']}")
    print(f"  Loop transformations:  {baseline['loop_transformations']}")
    print()
    
    print("Obfuscated:")
    print(f"  Total remarks:         {obfuscated['total_remarks']}")
    print(f"  Optimizations applied: {obfuscated['optimizations_applied']}")
    print(f"  Inlining decisions:    {obfuscated['inlining_decisions']}")
    print(f"  Loop transformations:  {obfuscated['loop_transformations']}")
    print()
    
    print("Delta (obfuscated - baseline):")
    print(f"  Total remarks:         {delta['total_remarks']:+d}")
    print(f"  Optimizations applied: {delta['optimizations_applied']:+d}")
    print(f"  Inlining decisions:    {delta['inlining_decisions']:+d}")
    print()
    
    # Display insights
    insights = result["insights"]
    if insights:
        print("💡 Insights:")
        for insight in insights:
            print(f"  ✓ {insight}")
        print()
    
    # Check thresholds
    validation_passed = True
    
    if args.min_remarks > 0:
        if obfuscated['total_remarks'] < args.min_remarks:
            print(f"❌ Failed: Total remarks ({obfuscated['total_remarks']}) < threshold ({args.min_remarks})")
            validation_passed = False
    
    if args.min_optimizations > 0:
        if obfuscated['optimizations_applied'] < args.min_optimizations:
            print(f"❌ Failed: Optimizations ({obfuscated['optimizations_applied']}) < threshold ({args.min_optimizations})")
            validation_passed = False
    
    if args.min_inlining > 0:
        if obfuscated['inlining_decisions'] < args.min_inlining:
            print(f"❌ Failed: Inlining decisions ({obfuscated['inlining_decisions']}) < threshold ({args.min_inlining})")
            validation_passed = False
    
    # Output JSON if requested
    if args.json:
        json_data = {
            "source": str(args.source),
            "validation": {
                "passed": validation_passed,
                "thresholds": {
                    "min_remarks": args.min_remarks,
                    "min_optimizations": args.min_optimizations,
                    "min_inlining": args.min_inlining
                }
            },
            "results": result
        }
        
        args.json.write_text(json.dumps(json_data, indent=2))
        print(f"📄 JSON report written to: {args.json}")
    
    # Final status
    print()
    print("=" * 60)
    if validation_passed:
        print("✅ Validation PASSED")
        return 0
    else:
        print("❌ Validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

