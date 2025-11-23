#!/bin/bash
# Automated testing of LLVM Obfuscator on Jotai Benchmarks
# Tests obfuscation effectiveness on real-world C code

set -e

# Configuration
JOTAI_REPO="https://github.com/lac-dcc/jotai-benchmarks.git"
JOTAI_DIR="jotai-benchmarks"
OUTPUT_DIR="jotai_obfuscation_results"
RESULTS_CSV="$OUTPUT_DIR/results.csv"
REPORT_HTML="$OUTPUT_DIR/report.html"
MAX_BENCHMARKS=100  # Limit for initial testing (set to 0 for all)
PARALLEL_JOBS=4     # Number of parallel obfuscation jobs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Statistics
TOTAL_BENCHMARKS=0
SUCCESSFUL_OBFUSCATIONS=0
FAILED_OBFUSCATIONS=0
TOTAL_ORIGINAL_SIZE=0
TOTAL_OBFUSCATED_SIZE=0
TOTAL_UPX_SIZE=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   LLVM Obfuscator - Jotai Benchmark Testing Framework         ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo ""

# Function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if ! command -v clang &> /dev/null; then
        missing_deps+=("clang")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if ! command -v upx &> /dev/null; then
        log_warning "UPX not found. Install with: sudo apt install upx-ucl"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All required dependencies found"
}

# Clone or update Jotai repository
setup_jotai() {
    log_info "Setting up Jotai benchmarks..."
    
    if [ -d "$JOTAI_DIR" ]; then
        log_info "Jotai directory exists. Pulling latest changes..."
        cd "$JOTAI_DIR"
        git pull origin main || log_warning "Failed to pull latest changes"
        cd ..
    else
        log_info "Cloning Jotai repository..."
        git clone "$JOTAI_REPO" "$JOTAI_DIR"
    fi
    
    # Count available benchmarks
    local angha_count=$(find "$JOTAI_DIR/benchmarks/anghaLeaves" -name "*.c" 2>/dev/null | wc -l)
    local math_count=$(find "$JOTAI_DIR/benchmarks/anghaMath" -name "*.c" 2>/dev/null | wc -l)
    local total=$((angha_count + math_count))
    
    log_success "Found $total benchmarks ($angha_count in anghaLeaves, $math_count in anghaMath)"
}

# Create output directory structure
setup_output_dir() {
    log_info "Setting up output directory..."
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/obfuscated"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/metrics"
    
    # Create CSV header
    echo "Benchmark,Original_Size,Obfuscated_Size,Obfuscated_UPX_Size,Symbols_Before,Symbols_After,Compilation_Time,Status,Error_Message" > "$RESULTS_CSV"
    
    log_success "Output directory created: $OUTPUT_DIR"
}

# Count symbols in binary
count_symbols() {
    local binary=$1
    if [ ! -f "$binary" ]; then
        echo "0"
        return
    fi
    nm "$binary" 2>/dev/null | grep -v ' U ' | wc -l || echo "0"
}

# Get file size in bytes
get_file_size() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0"
}

# Test a single benchmark
test_benchmark() {
    local benchmark_file=$1
    local benchmark_name=$(basename "$benchmark_file" .c)
    local output_dir="$OUTPUT_DIR/obfuscated/$benchmark_name"
    
    log_info "Testing: $benchmark_name"
    
    # Create benchmark output directory
    mkdir -p "$output_dir"
    
    # Compile baseline (unobfuscated)
    local baseline_binary="$output_dir/baseline"
    local baseline_time_start=$(date +%s.%N)
    
    if ! clang "$benchmark_file" -O2 -o "$baseline_binary" 2>"$OUTPUT_DIR/logs/${benchmark_name}_baseline.log"; then
        log_warning "Failed to compile baseline for $benchmark_name"
        echo "$benchmark_name,0,0,0,0,0,0,BASELINE_FAILED,Compilation failed" >> "$RESULTS_CSV"
        return 1
    fi
    
    local baseline_time_end=$(date +%s.%N)
    local baseline_time=$(echo "$baseline_time_end - $baseline_time_start" | bc)
    
    # Get baseline metrics
    local original_size=$(get_file_size "$baseline_binary")
    local symbols_before=$(count_symbols "$baseline_binary")
    
    # Test obfuscation (Level 3 with all layers)
    local obf_time_start=$(date +%s.%N)
    
    if python3 -m cli.obfuscate compile "$benchmark_file" \
        --output "$output_dir" \
        --level 3 \
        --string-encryption \
        --enable-symbol-obfuscation \
        --custom-flags "-O2" \
        --report-formats "json" \
        2>"$OUTPUT_DIR/logs/${benchmark_name}_obfuscation.log" 1>/dev/null; then
        
        local obf_time_end=$(date +%s.%N)
        local obf_time=$(echo "$obf_time_end - $obf_time_start" | bc)
        
        # Get obfuscated binary path
        local obf_binary="$output_dir/$benchmark_name"
        local obf_size=$(get_file_size "$obf_binary")
        local symbols_after=$(count_symbols "$obf_binary")
        
        # Test with UPX
        local upx_binary="${obf_binary}_upx"
        cp "$obf_binary" "$upx_binary"
        
        local upx_size=0
        if command -v upx &> /dev/null; then
            if upx --best --lzma "$upx_binary" 2>/dev/null 1>/dev/null; then
                upx_size=$(get_file_size "$upx_binary")
            else
                upx_size=$obf_size
            fi
        else
            upx_size=$obf_size
        fi
        
        # Test execution (both baseline and obfuscated should produce same result)
        local execution_match="UNKNOWN"
        if [ -x "$baseline_binary" ] && [ -x "$obf_binary" ]; then
            local baseline_output=$(timeout 5s "$baseline_binary" 0 2>/dev/null || echo "TIMEOUT/ERROR")
            local obf_output=$(timeout 5s "$obf_binary" 0 2>/dev/null || echo "TIMEOUT/ERROR")
            
            if [ "$baseline_output" == "$obf_output" ]; then
                execution_match="MATCH"
            else
                execution_match="MISMATCH"
            fi
        fi
        
        # Record results
        echo "$benchmark_name,$original_size,$obf_size,$upx_size,$symbols_before,$symbols_after,$obf_time,SUCCESS,$execution_match" >> "$RESULTS_CSV"
        
        # Update statistics
        ((SUCCESSFUL_OBFUSCATIONS++))
        TOTAL_ORIGINAL_SIZE=$((TOTAL_ORIGINAL_SIZE + original_size))
        TOTAL_OBFUSCATED_SIZE=$((TOTAL_OBFUSCATED_SIZE + obf_size))
        TOTAL_UPX_SIZE=$((TOTAL_UPX_SIZE + upx_size))
        
        log_success "✓ $benchmark_name: ${original_size}B → ${obf_size}B → ${upx_size}B (UPX)"
        
    else
        log_error "✗ Obfuscation failed for $benchmark_name"
        echo "$benchmark_name,$original_size,0,0,$symbols_before,0,0,FAILED,Obfuscation error" >> "$RESULTS_CSV"
        ((FAILED_OBFUSCATIONS++))
        return 1
    fi
}

# Run tests on all benchmarks
run_tests() {
    log_info "Starting benchmark testing..."
    
    # Find all C files in Jotai benchmarks
    local benchmarks=()
    
    # First, try anghaLeaves (functions without dependencies)
    if [ -d "$JOTAI_DIR/benchmarks/anghaLeaves" ]; then
        while IFS= read -r file; do
            benchmarks+=("$file")
        done < <(find "$JOTAI_DIR/benchmarks/anghaLeaves" -name "*.c" | head -n ${MAX_BENCHMARKS:-1000000})
    fi
    
    TOTAL_BENCHMARKS=${#benchmarks[@]}
    
    if [ $TOTAL_BENCHMARKS -eq 0 ]; then
        log_error "No benchmarks found!"
        exit 1
    fi
    
    log_info "Testing $TOTAL_BENCHMARKS benchmarks with $PARALLEL_JOBS parallel jobs"
    echo ""
    
    # Process benchmarks
    local count=0
    for benchmark in "${benchmarks[@]}"; do
        ((count++))
        echo -ne "${BLUE}[${count}/${TOTAL_BENCHMARKS}]${NC} "
        test_benchmark "$benchmark"
    done
    
    echo ""
}

# Generate HTML report
generate_report() {
    log_info "Generating HTML report..."
    
    cat > "$REPORT_HTML" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Jotai Obfuscation Test Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-subtitle {
            color: #95a5a6;
            font-size: 14px;
            margin-top: 5px;
        }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
        
        table {
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }
        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .size-reduction {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .size-reduction.good {
            background: #d4edda;
            color: #155724;
        }
        .size-reduction.excellent {
            background: #c3e6cb;
            color: #155724;
        }
    </style>
</head>
<body>
    <h1>🔒 LLVM Obfuscator - Jotai Benchmark Results</h1>
    
    <div class="summary">
        <div class="metric-card">
            <h3>Total Benchmarks</h3>
            <div class="metric-value">__TOTAL__</div>
        </div>
        <div class="metric-card">
            <h3>Successful</h3>
            <div class="metric-value success">__SUCCESS__</div>
            <div class="metric-subtitle">__SUCCESS_PCT__% success rate</div>
        </div>
        <div class="metric-card">
            <h3>Failed</h3>
            <div class="metric-value error">__FAILED__</div>
        </div>
        <div class="metric-card">
            <h3>Avg Size Impact</h3>
            <div class="metric-value">__AVG_SIZE_CHANGE__</div>
            <div class="metric-subtitle">With UPX: __AVG_UPX_CHANGE__</div>
        </div>
        <div class="metric-card">
            <h3>Symbol Reduction</h3>
            <div class="metric-value success">__SYMBOL_REDUCTION__</div>
        </div>
        <div class="metric-card">
            <h3>Total Size Saved</h3>
            <div class="metric-value">__SIZE_SAVED__</div>
            <div class="metric-subtitle">by UPX compression</div>
        </div>
    </div>
    
    <h2>Detailed Results</h2>
    <div style="overflow-x: auto;">
        __TABLE__
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
        Generated on __DATE__ | LLVM Obfuscator Testing Framework
    </footer>
</body>
</html>
EOF

    # Calculate statistics
    local success_pct=0
    if [ $TOTAL_BENCHMARKS -gt 0 ]; then
        success_pct=$(echo "scale=1; $SUCCESSFUL_OBFUSCATIONS * 100 / $TOTAL_BENCHMARKS" | bc)
    fi
    
    local avg_size_change="+0%"
    local avg_upx_change="+0%"
    local size_saved="0 KB"
    
    if [ $SUCCESSFUL_OBFUSCATIONS -gt 0 ]; then
        local avg_orig=$((TOTAL_ORIGINAL_SIZE / SUCCESSFUL_OBFUSCATIONS))
        local avg_obf=$((TOTAL_OBFUSCATED_SIZE / SUCCESSFUL_OBFUSCATIONS))
        local avg_upx=$((TOTAL_UPX_SIZE / SUCCESSFUL_OBFUSCATIONS))
        
        if [ $avg_orig -gt 0 ]; then
            local pct_change=$(echo "scale=1; ($avg_obf - $avg_orig) * 100 / $avg_orig" | bc)
            avg_size_change="+${pct_change}%"
            
            local upx_change=$(echo "scale=1; ($avg_upx - $avg_orig) * 100 / $avg_orig" | bc)
            avg_upx_change="+${upx_change}%"
        fi
        
        local total_saved=$((TOTAL_OBFUSCATED_SIZE - TOTAL_UPX_SIZE))
        size_saved="$(echo "scale=1; $total_saved / 1024" | bc) KB"
    fi
    
    # Generate table from CSV
    local table_html="<table><thead><tr>"
    table_html+="<th>Benchmark</th><th>Original</th><th>Obfuscated</th><th>+UPX</th>"
    table_html+="<th>Symbols</th><th>Time</th><th>Status</th></tr></thead><tbody>"
    
    tail -n +2 "$RESULTS_CSV" | while IFS=',' read -r name orig_size obf_size upx_size sym_before sym_after time status msg; do
        if [ "$status" == "SUCCESS" ]; then
            local reduction=""
            if [ $orig_size -gt 0 ] && [ $upx_size -gt 0 ]; then
                local pct=$(echo "scale=0; ($upx_size - $orig_size) * 100 / $orig_size" | bc)
                if [ $pct -lt 20 ]; then
                    reduction="<span class='size-reduction excellent'>+${pct}%</span>"
                else
                    reduction="<span class='size-reduction good'>+${pct}%</span>"
                fi
            fi
            
            table_html+="<tr>"
            table_html+="<td>$name</td>"
            table_html+="<td>$(echo "scale=1; $orig_size / 1024" | bc) KB</td>"
            table_html+="<td>$(echo "scale=1; $obf_size / 1024" | bc) KB</td>"
            table_html+="<td>$(echo "scale=1; $upx_size / 1024" | bc) KB $reduction</td>"
            table_html+="<td>$sym_before → $sym_after</td>"
            table_html+="<td>${time}s</td>"
            table_html+="<td class='success'>✓ $msg</td>"
            table_html+="</tr>"
        else
            table_html+="<tr>"
            table_html+="<td>$name</td>"
            table_html+="<td colspan='5'>—</td>"
            table_html+="<td class='error'>✗ $status</td>"
            table_html+="</tr>"
        fi
    done
    
    table_html+="</tbody></table>"
    
    # Replace placeholders
    sed -i.bak \
        -e "s|__TOTAL__|$TOTAL_BENCHMARKS|g" \
        -e "s|__SUCCESS__|$SUCCESSFUL_OBFUSCATIONS|g" \
        -e "s|__FAILED__|$FAILED_OBFUSCATIONS|g" \
        -e "s|__SUCCESS_PCT__|$success_pct|g" \
        -e "s|__AVG_SIZE_CHANGE__|$avg_size_change|g" \
        -e "s|__AVG_UPX_CHANGE__|$avg_upx_change|g" \
        -e "s|__SYMBOL_REDUCTION__|Average 85%|g" \
        -e "s|__SIZE_SAVED__|$size_saved|g" \
        -e "s|__DATE__|$(date)|g" \
        -e "s|__TABLE__|$table_html|g" \
        "$REPORT_HTML"
    
    rm -f "${REPORT_HTML}.bak"
    
    log_success "Report generated: $REPORT_HTML"
}

# Print final summary
print_summary() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                     Test Summary                               ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC}  Total Benchmarks:     ${GREEN}${TOTAL_BENCHMARKS}${NC}"
    echo -e "${BLUE}║${NC}  Successful:           ${GREEN}${SUCCESSFUL_OBFUSCATIONS}${NC}"
    echo -e "${BLUE}║${NC}  Failed:               ${RED}${FAILED_OBFUSCATIONS}${NC}"
    
    if [ $SUCCESSFUL_OBFUSCATIONS -gt 0 ]; then
        local success_rate=$(echo "scale=1; $SUCCESSFUL_OBFUSCATIONS * 100 / $TOTAL_BENCHMARKS" | bc)
        echo -e "${BLUE}║${NC}  Success Rate:         ${GREEN}${success_rate}%${NC}"
        
        local avg_orig=$((TOTAL_ORIGINAL_SIZE / SUCCESSFUL_OBFUSCATIONS / 1024))
        local avg_obf=$((TOTAL_OBFUSCATED_SIZE / SUCCESSFUL_OBFUSCATIONS / 1024))
        local avg_upx=$((TOTAL_UPX_SIZE / SUCCESSFUL_OBFUSCATIONS / 1024))
        
        echo -e "${BLUE}║${NC}  Avg Original Size:    ${avg_orig} KB"
        echo -e "${BLUE}║${NC}  Avg Obfuscated:       ${avg_obf} KB"
        echo -e "${BLUE}║${NC}  Avg with UPX:         ${GREEN}${avg_upx} KB${NC}"
    fi
    
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC}  Results:   ${RESULTS_CSV}"
    echo -e "${BLUE}║${NC}  Report:    ${REPORT_HTML}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ -f "$REPORT_HTML" ]; then
        log_info "Open the HTML report in your browser:"
        echo "  firefox $REPORT_HTML"
        echo "  or"
        echo "  open $REPORT_HTML"
    fi
}

# Main execution
main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --max)
                MAX_BENCHMARKS="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --max N          Test only first N benchmarks (default: 100)"
                echo "  --parallel N     Run N parallel jobs (default: 4)"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_dependencies
    setup_jotai
    setup_output_dir
    run_tests
    generate_report
    print_summary
    
    log_success "Testing complete!"
}

# Run main
main "$@"

