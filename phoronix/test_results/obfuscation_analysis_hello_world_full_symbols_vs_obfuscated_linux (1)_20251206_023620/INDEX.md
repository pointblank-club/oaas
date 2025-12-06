# Obfuscation Analysis Test Results

**Test Date:** Sat 06 Dec 2025 02:36:21 AM IST

**Timestamp:** 20251206_023620

**Test Name:** hello_world_full_symbols_vs_obfuscated_linux (1)

---

## Test Binaries

- **Baseline:** ./phoronix/option1_binaries/hello_world_full_symbols
  - Size: 83920 bytes
  - Type: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=865e9be8d51e5df37b0248b8ec0bcdfdf8071fb1, for GNU/Linux 3.2.0, with debug_info, not stripped

- **Obfuscated:** /home/incharaj/Downloads/obfuscated_linux (1)
  - Size: 14800 bytes
  - Type: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 3.2.0, stripped

---

## Report Structure

```
./phoronix/test_runs//obfuscation_analysis_hello_world_full_symbols_vs_obfuscated_linux (1)_20251206_023620/
├── metrics/
│   ├── metrics.json          # Raw metrics data
│   ├── metrics.csv           # Spreadsheet format
│   └── metrics.md            # Markdown format
├── security/
│   └── security_analysis.json # Security & decompilation analysis
├── reports/
│   ├── final_report.json     # Complete aggregated report
│   ├── final_report.md       # Markdown report
│   ├── final_report.html     # Interactive HTML report
│   └── final_report.csv      # CSV format
├── logs/
│   ├── execution.log         # Full execution log
│   ├── metrics.log           # Metrics collection log
│   ├── security.log          # Security analysis log
│   └── report.log            # Report generation log
└── INDEX.md                  # This file
```

---

## Quick Results

### Metrics
- See `metrics/metrics.json` for detailed binary metrics
- See `metrics/metrics.csv` for spreadsheet view

### Security Analysis
- See `security/security_analysis.json` for decompilation analysis

### Final Report
- **JSON:** `reports/final_report.json` - Programmatic access
- **Markdown:** `reports/final_report.md` - Human-readable
- **HTML:** `reports/final_report.html` - Interactive view
- **CSV:** `reports/final_report.csv` - Spreadsheet import

---

## Execution Log

See `logs/execution.log` for complete execution details.

---

## How to Use These Results

1. **For quick review:** Open `reports/final_report.md`
2. **For detailed analysis:** Use `reports/final_report.json`
3. **For visualization:** Open `reports/final_report.html` in browser
4. **For spreadsheets:** Import `reports/final_report.csv` to Excel

---

Generated with Obfuscation Test Suite
