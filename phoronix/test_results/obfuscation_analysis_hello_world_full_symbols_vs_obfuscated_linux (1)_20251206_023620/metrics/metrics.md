# Obfuscation Metrics Report

**Config:** test_hello_world_full_symbols_vs_obfuscated_linux (1)

**Timestamp:** 2025-12-06T02:36:20.836918

## Baseline Metrics

| Metric | Value |
|--------|-------|
| file_size_bytes | 83920 |
| text_section_size | 0 |
| num_functions | 9 |
| num_basic_blocks | 4 |
| instruction_count | 314 |
| text_entropy | 0.0 |
| cyclomatic_complexity | 1.0 |
| stripped | True |
| pie_enabled | True |

## Obfuscated Binaries Metrics

### obfuscated_linux (1)

| Metric | Value |
|--------|-------|
| file_size_bytes | 14800 |
| text_section_size | 0 |
| num_functions | 0 |
| num_basic_blocks | 1 |
| instruction_count | 836 |
| text_entropy | 0.0 |
| cyclomatic_complexity | 1.0 |
| stripped | True |
| pie_enabled | True |

## Comparison

### obfuscated_linux (1)

| Metric | Δ Value |
|--------|----------|
| file_size_increase_bytes | -69120 |
| file_size_increase_percent | -82.36 |
| text_section_increase_bytes | 0 |
| function_count_delta | -9 |
| basic_block_count_delta | -3 |
| instruction_count_delta | 522 |
| entropy_increase | 0.0 |
| complexity_increase | 0.0 |
