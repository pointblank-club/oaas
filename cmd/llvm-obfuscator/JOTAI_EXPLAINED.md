# Understanding Jotai Integration

## What is Jotai?

**Jotai is a collection of C SOURCE CODE files**, not binaries. Each file contains:
- A function extracted from real-world open source code
- A test driver that can run the function with different inputs
- Everything needed to compile and execute it

## How Does the Integration Work?

The integration follows this flow:

```
┌─────────────────────────────────────────────────────────┐
│  Jotai C Source File (e.g., extr_example_Final.c)       │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
        ↓                               ↓
┌──────────────────┐          ┌──────────────────────┐
│ Baseline Path    │          │ Obfuscation Path     │
│                  │          │                      │
│ 1. Compile       │          │ 1. Obfuscate SOURCE   │
│    normally      │          │    (symbol rename,    │
│                  │          │     string encrypt)  │
│ 2. Get binary    │          │                      │
│                  │          │ 2. Compile obfuscated │
│                  │          │    source            │
│                  │          │                      │
│ 3. Run with      │          │ 3. Get obfuscated    │
│    inputs        │          │    binary            │
│                  │          │                      │
│ 4. Get output    │          │ 4. Run with same      │
│                  │          │    inputs            │
└──────────────────┘          │                      │
        │                      │ 5. Get output        │
        │                      └──────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
              ┌──────────────────┐
              │ Compare Outputs  │
              │ (should match!)  │
              └──────────────────┘
```

## Key Points

1. **We obfuscate SOURCE CODE, not binaries**
   - Symbol obfuscation renames functions/variables in the C source
   - String encryption encrypts string literals in the C source
   - Then we compile the obfuscated source to a binary

2. **We test functional correctness**
   - Both baseline and obfuscated binaries are run with the same inputs
   - We verify they produce identical outputs
   - This ensures obfuscation didn't break functionality

3. **Why Jotai?**
   - Provides real-world C code patterns
   - Thousands of test cases
   - Each benchmark is self-contained and executable
   - Great for testing obfuscation on diverse code

## Example: What Happens to a Jotai Benchmark

### Original Jotai Source (simplified):
```c
int my_function(int x) {
    const char* secret = "password123";
    if (x > 0) {
        return x * 2;
    }
    return 0;
}

int main(int argc, char** argv) {
    int input = atoi(argv[1]);
    int result = my_function(input);
    printf("%d\n", result);
    return 0;
}
```

### After Symbol Obfuscation:
```c
int f_a1b2c3d4e5f6(int v_x1y2z3) {
    const char* v_s1t2u3v4 = "password123";  // Still visible!
    if (v_x1y2z3 > 0) {
        return v_x1y2z3 * 2;
    }
    return 0;
}

int main(int argc, char** argv) {
    int input = atoi(argv[1]);
    int result = f_a1b2c3d4e5f6(input);
    printf("%d\n", result);
    return 0;
}
```

### After String Encryption:
```c
int f_a1b2c3d4e5f6(int v_x1y2z3) {
    char* v_s1t2u3v4 = NULL;  // Will be decrypted at runtime
    // ... decryption code injected ...
    if (v_x1y2z3 > 0) {
        return v_x1y2z3 * 2;
    }
    return 0;
}
```

### Then Compile:
- Baseline: `clang original.c -o baseline_binary`
- Obfuscated: `clang obfuscated.c -o obfuscated_binary`

### Test:
```bash
./baseline_binary 5      # Output: 10
./obfuscated_binary 5   # Output: 10 (should match!)
```

## How to Test the Integration

### Quick Test Script:
```bash
cd cmd/llvm-obfuscator
python3 test_jotai_integration.py
```

This will:
1. Download Jotai benchmarks (if needed)
2. Show you what a benchmark looks like
3. Run one benchmark through obfuscation
4. Show you the results

### Full Test with CLI:
```bash
cd cmd/llvm-obfuscator

# Install dependencies first
pip install -r requirements.txt

# Run 5 benchmarks
python -m cli.obfuscate jotai --limit 5 --level 2
```

## What Gets Generated

When you run a Jotai benchmark, you get:

```
output_dir/
├── benchmark_name/
│   ├── baseline/
│   │   └── benchmark_name_baseline    # Normal binary
│   └── obfuscated/
│       ├── benchmark_name              # Obfuscated binary
│       ├── benchmark_name_encrypted.c  # Obfuscated source
│       └── report.json                 # Obfuscation report
└── jotai_report.json                    # Summary of all benchmarks
```

## Why This Is Useful

1. **Validation**: Ensures obfuscation doesn't break code
2. **Testing**: Tests obfuscation on real-world code patterns
3. **Research**: Evaluate obfuscation effectiveness
4. **Regression**: Catch bugs when changing obfuscation code

## Common Questions

**Q: Do I need to have Jotai binaries?**  
A: No! Jotai provides C source files. We compile them.

**Q: What if a benchmark fails?**  
A: Some benchmarks may fail due to:
- Unsupported C features
- Compilation errors
- Missing dependencies
- This is normal - not all code obfuscates perfectly

**Q: Can I use my own C files?**  
A: Yes! The obfuscator works on any C/C++ source. Jotai just provides a large test suite.

**Q: How long does it take?**  
A: Depends on number of benchmarks. Each benchmark:
- Downloads: ~30 seconds (first time only)
- Compilation: 1-5 seconds
- Obfuscation: 2-10 seconds
- Testing: 1-3 seconds

So 10 benchmarks ≈ 1-2 minutes total.

