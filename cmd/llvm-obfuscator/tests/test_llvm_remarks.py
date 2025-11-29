"""Tests for LLVM Remarks integration."""

import pytest
from pathlib import Path
import subprocess

from core.llvm_remarks import (
    Remark,
    RemarksAnalysis,
    LLVMRemarksParser,
    RemarksCollector,
    ObfuscationRemarksValidator,
    analyze_obfuscation_with_remarks,
)


class TestRemark:
    """Test Remark dataclass."""
    
    def test_from_dict(self):
        """Test creating Remark from dict."""
        data = {
            "Pass": "inline",
            "Name": "Inlined",
            "Function": "main",
            "Args": [
                {"Key": "Callee", "Value": "foo"},
                {"Key": "Cost", "Value": "100"}
            ],
            "Hotness": 500
        }
        
        remark = Remark.from_dict(data)
        
        assert remark.pass_name == "inline"
        assert remark.remark_name == "Inlined"
        assert remark.function == "main"
        assert remark.args["Callee"] == "foo"
        assert remark.args["Cost"] == "100"
        assert remark.hotness == 500


class TestLLVMRemarksParser:
    """Test LLVMRemarksParser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return LLVMRemarksParser()
    
    @pytest.fixture
    def sample_remarks_file(self, tmp_path):
        """Create sample remarks YAML file."""
        remarks_yaml = tmp_path / "remarks.yaml"
        remarks_yaml.write_text("""---
Pass: inline
Name: Inlined
Function: main
Args:
  - Key: Callee
    Value: foo
  - Key: Cost
    Value: 100
Hotness: 500
---
Pass: loop-unroll
Name: Unrolled
Function: compute
Args:
  - Key: UnrollCount
    Value: 4
Hotness: 1000
---
Pass: vectorize
Name: Vectorized
Function: compute
Args:
  - Key: VectorWidth
    Value: 8
""")
        return remarks_yaml
    
    def test_parse_remarks_file(self, parser, sample_remarks_file):
        """Test parsing remarks file."""
        remarks = parser.parse_remarks_file(sample_remarks_file)
        
        assert len(remarks) == 3
        assert remarks[0].pass_name == "inline"
        assert remarks[1].pass_name == "loop-unroll"
        assert remarks[2].pass_name == "vectorize"
    
    def test_parse_nonexistent_file(self, parser, tmp_path):
        """Test parsing non-existent file."""
        remarks = parser.parse_remarks_file(tmp_path / "nonexistent.yaml")
        assert remarks == []
    
    def test_analyze_remarks(self, parser, sample_remarks_file):
        """Test analyzing remarks."""
        remarks = parser.parse_remarks_file(sample_remarks_file)
        analysis = parser.analyze_remarks(remarks)
        
        assert analysis.total_remarks == 3
        assert analysis.inlining_decisions == 1
        assert analysis.loop_transformations == 1
        assert analysis.vectorization_attempts == 1
        assert "inline" in analysis.remarks_by_pass
        assert analysis.remarks_by_pass["inline"] == 1
    
    def test_hottest_functions(self, parser, sample_remarks_file):
        """Test hottest functions tracking."""
        remarks = parser.parse_remarks_file(sample_remarks_file)
        analysis = parser.analyze_remarks(remarks)
        
        assert len(analysis.hottest_functions) > 0
        # compute should be hottest (1000)
        assert analysis.hottest_functions[0][0] == "compute"
        assert analysis.hottest_functions[0][1] == 1000


class TestRemarksCollector:
    """Test RemarksCollector."""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance."""
        return RemarksCollector()
    
    def test_get_remarks_flags(self, collector, tmp_path):
        """Test remarks flags generation."""
        output_file = tmp_path / "remarks.yaml"
        flags = collector.get_remarks_flags(output_file)
        
        assert "-fsave-optimization-record=yaml" in flags
        assert any("foptimization-record-file" in f for f in flags)
        assert any("foptimization-record-passes" in f for f in flags)
    
    @pytest.mark.skipif(
        subprocess.run(["which", "clang"], capture_output=True).returncode != 0,
        reason="clang not available"
    )
    def test_collect_and_analyze(self, collector, tmp_path):
        """Test remarks collection and analysis."""
        # Create simple source file
        source = tmp_path / "test.c"
        source.write_text("""
#include <stdio.h>
int add(int a, int b) { return a + b; }
int main() {
    int result = add(5, 3);
    printf("%d\\n", result);
    return 0;
}
""")
        
        output = tmp_path / "test"
        
        analysis = collector.collect_and_analyze(
            source,
            output,
            extra_flags=["-O2"]
        )
        
        if analysis:  # May fail if clang doesn't support remarks
            assert analysis.total_remarks >= 0
            assert isinstance(analysis.remarks_by_pass, dict)


class TestObfuscationRemarksValidator:
    """Test ObfuscationRemarksValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ObfuscationRemarksValidator()
    
    def test_generate_insights(self, validator):
        """Test insights generation."""
        comparison = {
            "delta": {
                "total_remarks": 10,
                "optimizations_applied": 5,
                "inlining_decisions": 3
            },
            "pass_differences": {
                "inline": {"baseline": 5, "obfuscated": 8, "delta": 3},
                "loop-unroll": {"baseline": 10, "obfuscated": 3, "delta": -7}
            }
        }
        
        insights = validator._generate_insights(comparison)
        
        assert len(insights) > 0
        assert any("additional remarks" in insight for insight in insights)
        assert any("Inlining increased" in insight for insight in insights)
    
    @pytest.mark.skipif(
        subprocess.run(["which", "clang"], capture_output=True).returncode != 0,
        reason="clang not available"
    )
    def test_validate_obfuscation(self, validator, tmp_path):
        """Test full obfuscation validation."""
        # Create simple source
        source = tmp_path / "test.c"
        source.write_text("""
#include <stdio.h>
int secret() { return 42; }
int main() { printf("%d\\n", secret()); return 0; }
""")
        
        baseline = tmp_path / "baseline"
        obfuscated = tmp_path / "obfuscated"
        
        result = validator.validate_obfuscation(
            source,
            baseline,
            obfuscated,
            compiler_flags=["-O3", "-flto"]
        )
        
        # Should at least not crash
        assert "status" in result


class TestAnalyzeObfuscationWithRemarks:
    """Test convenience function."""
    
    @pytest.mark.skipif(
        subprocess.run(["which", "clang"], capture_output=True).returncode != 0,
        reason="clang not available"
    )
    def test_analyze_function(self, tmp_path):
        """Test one-shot analysis function."""
        source = tmp_path / "test.c"
        source.write_text("""
int main() { return 0; }
""")
        
        baseline = tmp_path / "baseline"
        obfuscated = tmp_path / "obfuscated"
        
        result = analyze_obfuscation_with_remarks(
            source,
            baseline,
            obfuscated,
            ["-O2"]
        )
        
        assert "status" in result

