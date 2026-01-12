#!/usr/bin/env python3
"""
Standalone Header Obfuscation Tool

Obfuscates function calls from headers (both stdlib and custom) by converting
direct calls to indirect calls via function pointers.

Usage:
    python3 obfuscate_headers.py input.c -o output.c
    python3 obfuscate_headers.py input.c -o output.c --stdlib-only
    python3 obfuscate_headers.py input.c -o output.c --custom-only
"""

import argparse
import sys
from pathlib import Path

# Ensure the cmd/llvm-obfuscator package is importable when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.indirect_call_obfuscator import obfuscate_indirect_calls


def main():
    parser = argparse.ArgumentParser(
        description='Obfuscate function calls from headers using indirect calling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Obfuscate both stdlib and custom functions
  python3 obfuscate_headers.py program.c -o program_obf.c

  # Obfuscate only standard library functions
  python3 obfuscate_headers.py program.c -o program_obf.c --stdlib-only

  # Obfuscate only custom functions
  python3 obfuscate_headers.py program.c -o program_obf.c --custom-only

  # Show what would be obfuscated without writing
  python3 obfuscate_headers.py program.c --dry-run
        """
    )

    parser.add_argument('input', type=Path, help='Input C/C++ source file')
    parser.add_argument('-o', '--output', type=Path, help='Output file (default: input_obfuscated.c)')
    parser.add_argument('--stdlib-only', action='store_true',
                        help='Obfuscate only standard library functions')
    parser.add_argument('--custom-only', action='store_true',
                        help='Obfuscate only custom functions')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be obfuscated without writing output')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.input.parent / f"{args.input.stem}_obfuscated{args.input.suffix}"

    # Determine what to obfuscate
    if args.stdlib_only:
        obfuscate_stdlib = True
        obfuscate_custom = False
    elif args.custom_only:
        obfuscate_stdlib = False
        obfuscate_custom = True
    else:
        obfuscate_stdlib = True
        obfuscate_custom = True

    print(f"📄 Reading: {args.input}")
    source_code = args.input.read_text(encoding='utf-8', errors='replace')

    print(f"🔒 Obfuscating function calls...")
    print(f"   - Standard library: {'Yes' if obfuscate_stdlib else 'No'}")
    print(f"   - Custom functions: {'Yes' if obfuscate_custom else 'No'}")

    try:
        transformed_code, metadata = obfuscate_indirect_calls(
            source_code,
            args.input,
            obfuscate_stdlib=obfuscate_stdlib,
            obfuscate_custom=obfuscate_custom
        )

        # Print results
        print(f"\n✅ Obfuscation complete:")
        print(f"   - Standard library functions: {metadata['obfuscated_stdlib_functions']}")
        print(f"   - Custom functions: {metadata['obfuscated_custom_functions']}")
        print(f"   - Total obfuscated: {metadata['total_obfuscated']}")

        if args.verbose:
            print(f"\n📋 Function pointers created:")
            for func, ptr in sorted(metadata['function_pointers'].items()):
                print(f"   {func} → {ptr}")

        if args.dry_run:
            print(f"\n⚠️  Dry run mode - no file written")
            print(f"\nPreview (first 50 lines):")
            print("=" * 60)
            for i, line in enumerate(transformed_code.split('\n')[:50], 1):
                print(f"{i:3}: {line}")
            print("=" * 60)
        else:
            output_file.write_text(transformed_code, encoding='utf-8')
            print(f"\n💾 Output written to: {output_file}")
            print(f"\n🎯 Next steps:")
            print(f"   1. Compile: clang {output_file} -o program")
            print(f"   2. Test: ./program")
            print(f"   3. Verify RE resistance: objdump -d program | grep call")

        return 0

    except Exception as e:
        print(f"\n❌ Error during obfuscation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
