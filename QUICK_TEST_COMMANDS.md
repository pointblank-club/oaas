# Quick Test Commands - Copy & Paste Ready

This file contains ready-to-use commands for testing on your VM. Just copy and paste into your terminal.

---

## ⚡ FASTEST WAY - Run Automated Test Script

```bash
cd /path/to/oaas
chmod +x test_complete_implementation.sh
./test_complete_implementation.sh
```

**Done!** This tests everything automatically.

---

## 🔧 Quick Manual Tests (5 minutes)

### 1️⃣ Build MLIR Library

```bash
cd /path/to/oaas/mlir-obs
./build.sh
```

### 2️⃣ Test MLIR Passes

```bash
cd /path/to/oaas/mlir-obs
./test.sh
```

### 3️⃣ Create Test File

```bash
cd /path/to/oaas
cat > test.c << 'EOF'
#include <stdio.h>
#include <string.h>
const char* SECRET = "MyPassword123!";
int validate(const char* input) {
    return strcmp(input, SECRET) == 0;
}
int main() {
    if (validate("MyPassword123!")) {
        printf("Access granted\n");
    }
    return 0;
}
EOF
```

### 4️⃣ Test Default Pipeline (String Encryption)

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --output ./output_test
```

**Verify:**
```bash
./output_test/test
strings ./output_test/test | grep "MyPassword"  # Should be empty
```

### 5️⃣ Test Full Obfuscation

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --enable-constant-obfuscate \
    --crypto-hash-algorithm sha256 \
    --crypto-hash-salt "test-salt" \
    --output ./output_full
```

**Verify:**
```bash
./output_full/test
strings ./output_full/test | grep "MyPassword"  # Should be empty
nm ./output_full/test | grep -v ' U ' | grep "validate"  # Should be empty
```

### 6️⃣ Test ClangIR Pipeline (if available)

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --mlir-frontend clangir \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --output ./output_clangir
```

**Verify:**
```bash
./output_clangir/test
```

---

## 🐳 Docker Commands

### Build Docker Image

```bash
cd /path/to/oaas
docker-compose -f docker-compose.test.yaml build
```

### Run Tests in Docker

```bash
docker-compose -f docker-compose.test.yaml run --rm test ./test_complete_implementation.sh
```

### Interactive Docker Shell

```bash
docker-compose -f docker-compose.test.yaml run --rm test bash
```

---

## 🔍 Verification Commands

### Check Tools

```bash
# Check versions
clang --version
llvm-config --version
mlir-opt --version
python3 --version
openssl version

# Check ClangIR (optional)
clangir --version || echo "Not installed (optional)"
```

### Check MLIR Library

```bash
ls -la mlir-obs/build/lib/libMLIRObfuscation.*
```

### Compare Baseline vs Obfuscated

```bash
# Compile baseline
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c --output ./baseline

# Compile obfuscated
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --output ./obfuscated

# Compare symbols
echo "Baseline symbols:"
nm ./baseline/test | grep -v ' U ' | wc -l

echo "Obfuscated symbols:"
nm ./obfuscated/test | grep -v ' U ' | wc -l

# Compare strings
echo "Baseline secrets:"
strings ./baseline/test | grep -i "password\|secret" | wc -l

echo "Obfuscated secrets:"
strings ./obfuscated/test | grep -i "password\|secret" | wc -l

# Compare sizes
ls -lh baseline/test obfuscated/test
```

---

## 🧪 Individual Pass Tests

### Test 1: String Encryption Only

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --output ./test_string

strings ./test_string/test | grep "MyPassword"
```

### Test 2: Symbol Obfuscation (RNG) Only

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-symbol-obfuscate \
    --output ./test_symbol

nm ./test_symbol/test | grep -v ' U ' | grep "validate"
```

### Test 3: Crypto Hash (SHA256) Only

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-crypto-hash \
    --crypto-hash-algorithm sha256 \
    --crypto-hash-salt "my-salt" \
    --crypto-hash-length 12 \
    --output ./test_crypto

nm ./test_crypto/test | grep -v ' U '
```

### Test 4: Crypto Hash (BLAKE2B) Only

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-crypto-hash \
    --crypto-hash-algorithm blake2b \
    --crypto-hash-salt "my-salt" \
    --crypto-hash-length 16 \
    --output ./test_blake2b

nm ./test_blake2b/test | grep -v ' U '
```

### Test 5: Constant Obfuscation Only

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-constant-obfuscate \
    --output ./test_constant

./test_constant/test  # Should still work
```

### Test 6: All Passes Combined

```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --enable-constant-obfuscate \
    --crypto-hash-algorithm blake2b \
    --crypto-hash-salt "production-salt-2024" \
    --crypto-hash-length 16 \
    --output ./test_all

./test_all/test
strings ./test_all/test | grep -i "password\|secret"
nm ./test_all/test | grep -v ' U ' | grep "validate"
```

---

## 📊 Performance Testing

### Measure Compilation Time

```bash
# Baseline
time python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c --output ./perf_baseline

# With obfuscation
time python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --output ./perf_obfuscated
```

### Measure Binary Size

```bash
du -h perf_baseline/test
du -h perf_obfuscated/test
```

### Measure Runtime Performance

```bash
# Create test with loop
cat > perf_test.c << 'EOF'
#include <stdio.h>
int main() {
    long sum = 0;
    for (int i = 0; i < 100000000; i++) {
        sum += i;
    }
    printf("Sum: %ld\n", sum);
    return 0;
}
EOF

# Compile both versions
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile perf_test.c --output ./perf_base
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile perf_test.c \
    --enable-string-encrypt \
    --enable-crypto-hash \
    --output ./perf_obf

# Time execution
time ./perf_base/perf_test
time ./perf_obf/perf_test
```

---

## 🚨 Troubleshooting Commands

### Check MLIR Build Errors

```bash
cd mlir-obs
./build.sh 2>&1 | tee build.log
cat build.log | grep -i "error"
```

### Check LLVM Installation

```bash
llvm-config --version
llvm-config --cmakedir
llvm-config --libdir
ls -la $(llvm-config --libdir)/cmake/mlir
```

### Check MLIR Library Dependencies

```bash
ldd mlir-obs/build/lib/libMLIRObfuscation.so  # Linux
otool -L mlir-obs/build/lib/libMLIRObfuscation.dylib  # macOS
```

### Check Python Module

```bash
python3 -c "import cmd.llvm-obfuscator.cli.obfuscate; print('OK')"
python3 -m cmd.llvm-obfuscator.cli.obfuscate --help
```

### Verify MLIR Passes Registered

```bash
cd mlir-obs/test_output
mlir-opt test.mlir --load-pass-plugin=../build/lib/libMLIRObfuscation.so --help | grep -E "string-encrypt|symbol-obfuscate|crypto-hash|constant-obfuscate"
```

### Check ClangIR Installation

```bash
which clangir
clangir --version
clangir --help | grep -i "emit-cir"
mlir-opt --help | grep "cir-to-llvm"
```

---

## 🎯 Expected Results Quick Reference

| Test | Command | Expected Result |
|------|---------|----------------|
| **Environment** | `clang --version` | Version 22.x |
| **MLIR Build** | `ls mlir-obs/build/lib/` | `libMLIRObfuscation.*` exists |
| **MLIR Test** | `./mlir-obs/test.sh` | "✅ All tests completed!" |
| **String Encrypt** | `strings output/binary \| grep SECRET` | Empty output |
| **Symbol Obfuscate** | `nm output/binary \| grep validate` | Empty output |
| **Crypto Hash** | `nm output/binary \| grep -v ' U '` | Hex names like `f_abc123` |
| **Binary Execution** | `./output/binary` | Correct output, no crash |
| **ClangIR** | `clangir --version` | Version info or "command not found" (optional) |

---

## 📝 Clean Up Commands

### Remove Test Outputs

```bash
rm -rf test_output_* output_* test_* perf_* baseline obfuscated
```

### Clean MLIR Build

```bash
cd mlir-obs
rm -rf build test_output
./build.sh
```

### Full Clean

```bash
cd /path/to/oaas
git clean -fdx  # WARNING: Removes ALL untracked files
# OR manually:
rm -rf mlir-obs/build mlir-obs/test_output test_output_* output_* *.log
```

---

## 💡 Pro Tips

### 1. Test in Order
Always test in this order:
1. Environment check → 2. MLIR build → 3. MLIR tests → 4. Individual passes → 5. Combined passes

### 2. Save Logs
Add `2>&1 | tee output.log` to save logs:
```bash
python3 -m cmd.llvm-obfuscator.cli.obfuscate compile test.c --enable-string-encrypt --output ./out 2>&1 | tee test.log
```

### 3. Use Docker for Clean Environment
If local tests fail, try Docker:
```bash
docker-compose -f docker-compose.test.yaml run --rm test bash
```

### 4. Check Intermediate Files
MLIR files are saved in temp directories - check them to debug:
```bash
ls -la /tmp/mlir_obfuscate_*
cat /tmp/mlir_obfuscate_*/test.mlir
```

### 5. Compare Outputs
Always compare before/after to verify obfuscation worked:
```bash
diff <(nm baseline/binary) <(nm obfuscated/binary)
diff <(strings baseline/binary) <(strings obfuscated/binary)
```

---

## 📚 Further Reading

- **Complete Guide**: See `TESTING_GUIDE.md`
- **MLIR Usage**: See `MLIR_INTEGRATION_GUIDE.md`
- **ClangIR Details**: See `CLANGIR_PIPELINE_GUIDE.md`
- **Main README**: See `README.md`

---

**Need help?** Check `TESTING_GUIDE.md` section "Troubleshooting" for detailed debugging steps.
