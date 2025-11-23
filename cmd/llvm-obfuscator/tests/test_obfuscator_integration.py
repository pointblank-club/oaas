"""Integration tests for the complete obfuscation pipeline."""

import pytest
import subprocess
import tempfile
from pathlib import Path
import json

from core import (
    LLVMObfuscator,
    ObfuscationConfig,
    ObfuscationLevel,
    Platform,
    AdvancedConfiguration,
    SymbolObfuscationConfiguration,
    UPXConfiguration,
    OutputConfiguration,
)


class TestObfuscationPipeline:
    """Test complete obfuscation pipeline."""
    
    @pytest.fixture
    def simple_c_source(self, tmp_path):
        """Create a simple C source file."""
        source = tmp_path / "test.c"
        source.write_text("""
#include <stdio.h>
#include <string.h>

const char* SECRET_PASSWORD = "AdminPass2024!";
const char* API_KEY = "sk_live_secret_12345";

int validate_user(const char* input) {
    if (strcmp(input, SECRET_PASSWORD) == 0) {
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <password>\\n", argv[0]);
        return 1;
    }
    
    if (validate_user(argv[1])) {
        printf("Access granted!\\n");
        return 0;
    } else {
        printf("Access denied!\\n");
        return 1;
    }
}
""")
        return source
    
    @pytest.fixture
    def complex_c_source(self, tmp_path):
        """Create a more complex C source file."""
        source = tmp_path / "complex.c"
        source.write_text("""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[50];
    float balance;
} Account;

const char* MASTER_KEY = "MasterKey2024";
const char* DB_PASSWORD = "db_pass_secret";

Account* create_account(int id, const char* name, float balance) {
    Account* acc = (Account*)malloc(sizeof(Account));
    if (!acc) return NULL;
    
    acc->id = id;
    strncpy(acc->name, name, 49);
    acc->name[49] = '\\0';
    acc->balance = balance;
    
    return acc;
}

int authenticate(const char* key) {
    return strcmp(key, MASTER_KEY) == 0;
}

float get_balance(Account* acc) {
    if (!acc) return 0.0f;
    return acc->balance;
}

void free_account(Account* acc) {
    if (acc) free(acc);
}

int main() {
    if (!authenticate("MasterKey2024")) {
        printf("Authentication failed\\n");
        return 1;
    }
    
    Account* acc = create_account(1001, "John Doe", 1500.50f);
    if (acc) {
        printf("Account ID: %d\\n", acc->id);
        printf("Balance: $%.2f\\n", get_balance(acc));
        free_account(acc);
    }
    
    return 0;
}
""")
        return source
    
    def test_basic_obfuscation(self, simple_c_source, tmp_path):
        """Test basic obfuscation (Level 1)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.LOW,
            platform=Platform.LINUX,
            output=OutputConfiguration(
                directory=output_dir,
                report_formats=["json"]
            )
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(simple_c_source, config)
        
        assert result is not None
        assert "output_file" in result
        assert Path(result["output_file"]).exists()
    
    def test_symbol_obfuscation(self, simple_c_source, tmp_path):
        """Test symbol obfuscation layer."""
        output_dir = tmp_path / "output_symbols"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.MEDIUM,
            platform=Platform.LINUX,
            advanced=AdvancedConfiguration(
                symbol_obfuscation=SymbolObfuscationConfiguration(
                    enabled=True,
                    algorithm="sha256",
                    hash_length=12
                )
            ),
            output=OutputConfiguration(
                directory=output_dir,
                report_formats=["json"]
            )
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(simple_c_source, config)
        
        assert result is not None
        assert "symbol_obfuscation" in result
        
        # Check that symbols were reduced
        baseline = result.get("baseline_metrics", {})
        output = result.get("output_attributes", {})
        
        if baseline.get("symbols_count", 0) > 0:
            assert output.get("symbols_count", 999) < baseline["symbols_count"]
    
    def test_string_encryption(self, simple_c_source, tmp_path):
        """Test string encryption layer."""
        output_dir = tmp_path / "output_strings"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.MEDIUM,
            platform=Platform.LINUX,
            advanced=AdvancedConfiguration(
                string_encryption=True
            ),
            output=OutputConfiguration(
                directory=output_dir,
                report_formats=["json"]
            )
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(simple_c_source, config)
        
        assert result is not None
        
        # Check string obfuscation results
        string_obf = result.get("string_obfuscation", {})
        assert string_obf.get("encrypted_strings", 0) > 0
        
        # Verify secrets not visible in binary
        binary_path = Path(result["output_file"])
        if binary_path.exists():
            with open(binary_path, "rb") as f:
                binary_content = f.read()
            
            # These secrets should NOT be visible
            assert b"AdminPass2024!" not in binary_content
            assert b"sk_live_secret_12345" not in binary_content
    
    def test_upx_packing(self, simple_c_source, tmp_path):
        """Test UPX packing layer."""
        import shutil
        if not shutil.which("upx"):
            pytest.skip("UPX not installed")
        
        output_dir = tmp_path / "output_upx"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.MEDIUM,
            platform=Platform.LINUX,
            advanced=AdvancedConfiguration(
                upx_packing=UPXConfiguration(
                    enabled=True,
                    compression_level="best",
                    use_lzma=True
                )
            ),
            output=OutputConfiguration(
                directory=output_dir,
                report_formats=["json"]
            )
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(simple_c_source, config)
        
        assert result is not None
        
        # Check UPX results
        upx_result = result.get("upx_packing", {})
        if upx_result.get("status") == "success":
            assert upx_result["original_size"] > upx_result["packed_size"]
            assert upx_result["compression_ratio"] > 0
    
    def test_full_pipeline(self, complex_c_source, tmp_path):
        """Test complete obfuscation pipeline with all layers."""
        import shutil
        
        output_dir = tmp_path / "output_full"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.HIGH,
            platform=Platform.LINUX,
            compiler_flags=["-O2"],
            advanced=AdvancedConfiguration(
                symbol_obfuscation=SymbolObfuscationConfiguration(
                    enabled=True,
                    algorithm="sha256",
                    hash_length=12
                ),
                string_encryption=True,
                upx_packing=UPXConfiguration(
                    enabled=shutil.which("upx") is not None,
                    compression_level="best"
                )
            ),
            output=OutputConfiguration(
                directory=output_dir,
                report_formats=["json"]
            )
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(complex_c_source, config)
        
        assert result is not None
        assert "output_file" in result
        
        binary_path = Path(result["output_file"])
        assert binary_path.exists()
        
        # Test execution
        try:
            exec_result = subprocess.run(
                [str(binary_path)],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Should run without crashing
            assert exec_result.returncode in [0, 1]  # May fail auth, that's OK
        except subprocess.TimeoutExpired:
            pytest.fail("Obfuscated binary timed out")
    
    def test_correctness_preservation(self, simple_c_source, tmp_path):
        """Test that obfuscation preserves program semantics."""
        # Compile baseline
        baseline_binary = tmp_path / "baseline"
        subprocess.run(
            ["clang", str(simple_c_source), "-O2", "-o", str(baseline_binary)],
            check=True
        )
        
        # Run baseline
        baseline_result = subprocess.run(
            [str(baseline_binary), "AdminPass2024!"],
            capture_output=True,
            text=True
        )
        
        # Obfuscate and compile
        output_dir = tmp_path / "output_correctness"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.MEDIUM,
            platform=Platform.LINUX,
            advanced=AdvancedConfiguration(
                symbol_obfuscation=SymbolObfuscationConfiguration(enabled=True),
                string_encryption=True
            ),
            output=OutputConfiguration(directory=output_dir)
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(simple_c_source, config)
        
        obf_binary = Path(result["output_file"])
        
        # Run obfuscated
        obf_result = subprocess.run(
            [str(obf_binary), "AdminPass2024!"],
            capture_output=True,
            text=True
        )
        
        # Outputs should match
        assert baseline_result.returncode == obf_result.returncode
        assert baseline_result.stdout == obf_result.stdout
    
    def test_report_generation(self, simple_c_source, tmp_path):
        """Test report generation."""
        output_dir = tmp_path / "output_report"
        output_dir.mkdir()
        
        config = ObfuscationConfig(
            level=ObfuscationLevel.MEDIUM,
            platform=Platform.LINUX,
            output=OutputConfiguration(
                directory=output_dir,
                report_formats=["json", "html"]
            )
        )
        
        obfuscator = LLVMObfuscator()
        result = obfuscator.obfuscate(simple_c_source, config)
        
        assert result is not None
        assert "report_paths" in result
        
        # Check JSON report exists
        if "json" in result["report_paths"]:
            json_report = Path(result["report_paths"]["json"])
            assert json_report.exists()
            
            # Validate JSON structure
            with open(json_report) as f:
                report_data = json.load(f)
            
            assert "obfuscation_level" in report_data
            assert "output_attributes" in report_data
            assert "baseline_metrics" in report_data

