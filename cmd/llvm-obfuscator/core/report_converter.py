"""Convert JSON reports to PDF and Markdown formats on-the-fly."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


def _safe_float(value, default=0.0):
    """Convert value to float, handling None gracefully."""
    return default if value is None else float(value) if isinstance(value, (int, float, str)) else default


def _safe_int(value, default=0):
    """Convert value to int, handling None gracefully."""
    return default if value is None else int(value) if isinstance(value, (int, float, str)) else default


def _safe_str(value, default='N/A'):
    """Convert value to string, handling None gracefully."""
    return str(value) if value is not None else default


def _safe_list_str(value, default='None', separator=', '):
    """Convert list to comma-separated string, handling None and empty lists gracefully."""
    if value is None:
        return default
    if isinstance(value, list):
        return separator.join(value) if value else default
    if isinstance(value, str):
        return value if value else default
    return default


def format_percentage(value, decimals=1):
    """Format percentage with consistent decimal places."""
    try:
        num = float(value) if value is not None else 0
        return f"{num:.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_entropy(value, decimals=4):
    """Format entropy value with consistent decimal places."""
    try:
        num = float(value) if value is not None else 0
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_time(value, decimals=2):
    """Format time value in milliseconds with consistent decimal places."""
    try:
        num = float(value) if value is not None else 0
        return f"{num:.{decimals}f} ms"
    except (ValueError, TypeError):
        return "N/A"


def format_timestamp_human_readable(timestamp_str):
    """Convert ISO timestamp to human-readable format (IST)."""
    try:
        from datetime import datetime
        # Parse ISO format: 2025-12-08T10:21:43.678471
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').split('.')[0])
        # Format as: 8 Dec 2025, 3:30 PM IST
        return dt.strftime('%d %b %Y, %I:%M %p')
    except:
        return timestamp_str


# Matplotlib imports for chart generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, charts will be skipped")


def json_to_markdown(report: Dict[str, Any]) -> str:
    """Convert JSON report to Markdown format."""
    md = "# LLVM Obfuscation Report\n\n"

    input_params = report.get("input_parameters", {})
    source_file = input_params.get('source_file', 'N/A')
    md += f"## 📄 Source File: `{source_file}`\n\n"
    md += f"**Job ID:** {report.get('job_id', 'N/A')}\n"
    md += f"**Timestamp:** {input_params.get('timestamp', 'N/A')}\n"
    md += f"**Platform:** {input_params.get('platform', 'N/A')}\n"
    md += f"**Obfuscation Level:** {input_params.get('obfuscation_level', 'N/A')}\n\n"

    # Baseline vs Obfuscated Metrics
    md += "## Baseline Metrics\n"
    baseline = report.get("baseline_metrics", {})
    md += f"- **File Size:** {baseline.get('file_size', 'N/A')} bytes\n"
    md += f"- **Functions:** {baseline.get('functions', 'N/A')}\n"
    md += f"- **Entropy:** {format_entropy(baseline.get('entropy'))}\n\n"

    md += "## Obfuscated Metrics\n"
    obfuscated = report.get("output_attributes", {})
    md += f"- **File Size:** {obfuscated.get('file_size', 'N/A')} bytes\n"
    md += f"- **Functions:** {obfuscated.get('functions_count', 'N/A')}\n"
    md += f"- **Entropy:** {_safe_float(obfuscated.get('entropy')):.4f}\n\n"

    # Comparison
    md += "## Comparison\n"
    comparison = report.get("comparison", {})
    md += f"- **Size Change:** {comparison.get('size_change', 'N/A')} bytes ({comparison.get('size_change_percent', 'N/A')}%)\n"
    md += f"- **Entropy Increase:** {_safe_float(comparison.get('entropy_increase')):.4f}\n"
    md += f"- **RE Effort:** {report.get('estimated_re_effort', 'N/A')}\n\n"

    # Applied Passes
    md += "## Applied Obfuscation Passes\n"
    applied = report.get("input_parameters", {}).get("applied_passes", [])
    if applied:
        for pass_name in applied:
            md += f"- {pass_name}\n"
    else:
        md += "- None\n"

    # Phoronix Benchmarking Metrics (if available)
    phoronix_info = report.get('phoronix', {})
    if phoronix_info and phoronix_info.get('key_metrics'):
        key_metrics = phoronix_info.get('key_metrics', {})
        md += "\n## 📊 Obfuscation Impact Metrics\n\n"

        instr_delta = key_metrics.get('instruction_count_delta')
        instr_percent = key_metrics.get('instruction_count_increase_percent')
        perf_overhead = key_metrics.get('performance_overhead_percent')

        if instr_delta is not None:
            md += f"**Code Expansion:**\n"
            md += f"- Instruction Count Increase: +{instr_delta} instructions (+{instr_percent}%)\n\n"

        if perf_overhead is not None:
            md += f"**Performance Overhead:**\n"
            md += f"- Runtime Slowdown: +{perf_overhead:.1f}%\n\n"

        md += "**Note:** Instruction count reflects code expansion from obfuscation passes. "
        md += "Performance overhead is based on runtime measurements if available.\n"

    md += "\n---\n"
    return md


def safe_get(data: Any, *keys, default='N/A') -> Any:
    """Safely navigate nested dict structure."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
        if data is None:
            return default
    return data


def format_bytes(size: int) -> str:
    """Format bytes to KB/MB."""
    try:
        size = int(size) if size else 0
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"
    except:
        return "N/A"


def format_percentage(value: float) -> str:
    """Format percentage with + or - sign."""
    try:
        value = float(value) if value else 0
        if value > 0:
            return f"+{value:.2f}%"
        else:
            return f"{value:.2f}%"
    except:
        return "N/A"


def get_score_emoji(score: float) -> str:
    """Get emoji based on obfuscation score."""
    try:
        score = float(score) if score else 0
        if score >= 80:
            return "🟢"
        elif score >= 60:
            return "🟡"
        else:
            return "🔴"
    except:
        return "⚪"


# ============= Chart Generation Functions =============

def create_control_flow_chart(report: Dict[str, Any]) -> bytes:
    """Create control flow analysis bar chart (Baseline vs Obfuscated)."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        control_flow = report.get('control_flow_metrics', {})
        if not control_flow:
            return None

        baseline = control_flow.get('baseline', {})
        obfuscated = control_flow.get('obfuscated', {})

        metrics = ['Basic Blocks', 'CFG Edges', 'Cyclomatic\nComplexity']
        baseline_vals = [
            baseline.get('basic_blocks', 0),
            baseline.get('cfg_edges', 0),
            baseline.get('cyclomatic_complexity', 0)
        ]
        obfuscated_vals = [
            obfuscated.get('basic_blocks', 0),
            obfuscated.get('cfg_edges', 0),
            obfuscated.get('cyclomatic_complexity', 0)
        ]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#1f6feb', alpha=0.8)
        bars2 = ax.bar(x + width/2, obfuscated_vals, width, label='Obfuscated', color='#2ea043', alpha=0.8)

        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title('Control Flow Analysis', fontsize=12, fontweight='bold', color='#1f6feb')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer.read()
    except Exception as e:
        logger.error(f"Error creating control flow chart: {e}")
        return None


def create_instruction_distribution_charts(report: Dict[str, Any]) -> bytes:
    """Create side-by-side pie charts for instruction distribution."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        instr_metrics = report.get('instruction_metrics', {})
        if not instr_metrics:
            return None

        baseline_dist = instr_metrics.get('baseline', {}).get('instruction_distribution', {})
        obfuscated_dist = instr_metrics.get('obfuscated', {}).get('instruction_distribution', {})

        if not baseline_dist or not obfuscated_dist:
            return None

        labels = ['LOAD', 'STORE', 'CALL', 'BR', 'PHI', 'ARITHMETIC', 'OTHER']
        baseline_vals = [baseline_dist.get(label, 0) for label in labels]
        obfuscated_vals = [obfuscated_dist.get(label, 0) for label in labels]

        # Skip chart if all values are zero (baseline compilation failed)
        if sum(baseline_vals) == 0 or sum(obfuscated_vals) == 0:
            return None

        colors = ['#1f6feb', '#2ea043', '#d29922', '#da3633', '#8957e5', '#00bcd4', '#9e9e9e']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

        # Baseline pie chart
        wedges1, texts1, autotexts1 = ax1.pie(baseline_vals, labels=labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90, textprops={'fontsize': 8})
        ax1.set_title('Baseline Instructions', fontsize=11, fontweight='bold', color='#1f6feb')
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Obfuscated pie chart
        wedges2, texts2, autotexts2 = ax2.pie(obfuscated_vals, labels=labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90, textprops={'fontsize': 8})
        ax2.set_title('Obfuscated Instructions', fontsize=11, fontweight='bold', color='#2ea043')
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        fig.suptitle('Instruction Distribution Comparison', fontsize=12, fontweight='bold', color='#1f6feb', y=0.98)
        fig.tight_layout()
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer.read()
    except Exception as e:
        logger.error(f"Error creating instruction distribution charts: {e}")
        return None


def create_comparison_progress_bars(report: Dict[str, Any]) -> bytes:
    """Create progress bars for before/after metrics comparison."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        baseline_metrics = report.get('baseline_metrics', {})
        output_attrs = report.get('output_attributes', {})
        comparison = report.get('comparison', {})

        if not all([baseline_metrics, output_attrs, comparison]):
            return None

        # Calculate percentages for progress bars
        metrics_data = []

        # Symbol reduction
        symbol_baseline = baseline_metrics.get('symbols_count', 1)
        symbol_obf = output_attrs.get('symbols_count', symbol_baseline)
        symbol_reduction = ((symbol_baseline - symbol_obf) / symbol_baseline * 100) if symbol_baseline > 0 else 0
        metrics_data.append(('Symbol Reduction', symbol_reduction, '#8957e5'))

        # File size change (negative is good)
        size_change = abs(_safe_float(comparison.get('size_change_percent'), 0))
        metrics_data.append(('File Size Change', min(100, size_change), '#d29922'))

        # Entropy increase
        entropy_inc = _safe_float(comparison.get('entropy_increase'), 0)
        entropy_pct = min(100, (entropy_inc / 8.0 * 100))  # Entropy max ~8 bits
        metrics_data.append(('Entropy Increase', entropy_pct, '#2ea043'))

        fig, ax = plt.subplots(figsize=(6, 3))
        y_pos = np.arange(len(metrics_data))
        values = [m[1] for m in metrics_data]
        labels_list = [m[0] for m in metrics_data]
        colors_list = [m[2] for m in metrics_data]

        bars = ax.barh(y_pos, values, color=colors_list, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list, fontsize=10)
        ax.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax.set_title('Obfuscation Effectiveness', fontsize=12, fontweight='bold', color='#1f6feb')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 2, bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
                   va='center', fontsize=9, fontweight='bold')

        fig.tight_layout()
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer.read()
    except Exception as e:
        logger.error(f"Error creating comparison progress bars: {e}")
        return None


def json_to_pdf(report: Dict[str, Any]) -> bytes:
    """Convert JSON report to beautiful PDF format using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        )
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        # Fallback to Markdown if reportlab not available
        md = json_to_markdown(report)
        return md.encode('utf-8')

    # Create PDF in memory
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=1*inch, bottomMargin=0.75*inch
    )

    story = []
    styles = getSampleStyleSheet()

    # ============= PAGE 1: Executive Summary & Core Metrics =============

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f6feb'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph("🛡️ OAAS Obfuscation Report", title_style))
    story.append(Spacer(1, 0.08*inch))

    # Timestamp in readable format (Job ID removed)
    input_params = report.get('input_parameters', {})
    timestamp = input_params.get('timestamp', 'N/A')
    formatted_timestamp = format_timestamp_human_readable(timestamp) if timestamp != 'N/A' else 'N/A'
    source_file = input_params.get('source_file', 'Unknown')
    story.append(Paragraph(f"<b>Generated:</b> {formatted_timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.05*inch))

    # Source filename prominently displayed
    source_style = ParagraphStyle(
        'SourceFile',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1f6feb'),
        fontName='Helvetica-Bold',
        spaceAfter=6
    )
    story.append(Paragraph(f"📄 <b>Source File:</b> <font color='#28a745'>{_safe_str(source_file)}</font>", source_style))
    story.append(Spacer(1, 0.08*inch))

    # ✅ NEW: Platform and binary format metadata (for Windows score fix transparency)
    metadata = report.get('metadata', {})
    if metadata:
        platform = _safe_str(metadata.get('platform'), 'unknown').upper()
        binary_format = _safe_str(metadata.get('binary_format'), 'unknown')
        extraction_method = _safe_str(metadata.get('metric_extraction_method'), 'unknown')

        platform_info = f"📊 <b>Target Platform:</b> {platform} ({binary_format})"
        story.append(Paragraph(platform_info, source_style))
        story.append(Spacer(1, 0.03*inch))

        # Add extraction method note in smaller text
        extraction_note = f"<font size='8' color='#666666'>Metrics extracted using: {extraction_method}</font>"
        story.append(Paragraph(extraction_note, styles['Normal']))
        story.append(Spacer(1, 0.08*inch))

    # Horizontal line separator
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 0.12*inch))

    # Warning banners (if applicable)
    baseline_status = report.get('baseline_status', 'success')
    warnings = report.get('warnings', [])

    if baseline_status == 'failed':
        warning_data = [['⚠️ Baseline Compilation Failed - Some metrics unavailable']]
        warning_table = Table(warning_data, colWidths=[7*inch])
        warning_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffc107')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor('#ff9800')),
        ]))
        story.append(warning_table)
        story.append(Spacer(1, 0.15*inch))

    if warnings:
        # Filter out informational warnings about exception handling that don't indicate failures
        # Use regex for more robust matching to handle variations in warning text
        exception_handling_patterns = re.compile(
            r"(c\+\+\s+)?exception\s+handling|"
            r"flattening\s+(?:pass\s+)?disabled|"
            r"exception-aware\s+obfuscation|"
            r"invoke.*landingpad|"
            r"known\s+to\s+crash\s+on\s+exception",
            re.IGNORECASE
        )
        display_warnings = [w for w in warnings if not exception_handling_patterns.search(w)]

        for warning in display_warnings[:3]:  # Show first 3 warnings (excluding informational ones)
            warn_data = [[f"⚠️ {warning}"]]
            warn_table = Table(warn_data, colWidths=[7*inch])
            warn_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffe0e0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#dc3545')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor('#dc3545')),
            ]))
            story.append(warn_table)

        # Only add spacer if we actually displayed warnings
        if display_warnings:
            story.append(Spacer(1, 0.15*inch))

    # Overall Obfuscation Score - Displayed on First Page (metric-driven score)
    overall_index = _safe_float(report.get('overall_protection_index', 0))

    # Create a summary metrics table for page 1 with Overall Obfuscation Score
    summary_metrics = [
        ['OVERALL OBFUSCATION SCORE'],
        [f'{overall_index:.1f}/100'],
        ['Metric-driven: Symbol Reduction + Function Hiding + Entropy + Techniques']
    ]
    summary_table = Table(summary_metrics, colWidths=[7*inch])
    summary_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        # Score row
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0,1), (-1,1), colors.HexColor('#1f6feb')),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,1), 18),
        ('TOPPADDING', (0,1), (-1,1), 8),
        ('BOTTOMPADDING', (0,1), (-1,1), 8),
        ('ALIGN', (0,1), (-1,1), 'CENTER'),
        # Description row
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0,2), (-1,2), colors.HexColor('#666666')),
        ('FONTNAME', (0,2), (-1,2), 'Helvetica'),
        ('FONTSIZE', (0,2), (-1,2), 9),
        ('TOPPADDING', (0,2), (-1,2), 4),
        ('BOTTOMPADDING', (0,2), (-1,2), 6),
        # All cells
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.08*inch))

    # Quick Metrics Row
    symbol_red = float(report.get('symbol_reduction', 0)) if report.get('symbol_reduction') else 0
    func_red = float(report.get('function_reduction', 0)) if report.get('function_reduction') else 0
    entropy_inc = float(report.get('entropy_increase', 0)) if report.get('entropy_increase') else 0
    re_effort = report.get('estimated_re_effort', 'Unknown')

    quick_metrics = [
        ['Symbol Reduction', 'Function Hiding', 'Entropy Increase', 'RE Difficulty'],
        [f'{symbol_red:.1f}%', f'{func_red:.1f}%', f'{entropy_inc:.4f}', str(re_effort)]
    ]
    quick_table = Table(quick_metrics, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
    quick_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
    ]))
    story.append(quick_table)
    story.append(Spacer(1, 0.05*inch))

    # Add Comparison Progress Bars Chart
    comparison_chart = create_comparison_progress_bars(report)
    if comparison_chart:
        try:
            chart_img = Image(BytesIO(comparison_chart), width=5.5*inch, height=2.2*inch)
            story.append(chart_img)
            story.append(Spacer(1, 0.08*inch))
        except Exception as e:
            logger.error(f"Error embedding comparison chart: {e}")

    # Input Parameters - COMPREHENSIVE TABLE WITH ALL SETTINGS
    story.append(PageBreak())  # Dedicated page for input parameters
    story.append(Paragraph("Input Parameters & Configuration Details", styles['Heading2']))
    story.append(Spacer(1, 0.15*inch))

    # GLOBAL SETTINGS
    story.append(Paragraph("Global Settings", styles['Heading3']))
    global_data = [
        ['Parameter', 'Value'],
        ['Source File', str(input_params.get('source_file', 'N/A'))],
        ['Platform', str(input_params.get('platform', 'N/A'))],
        ['Architecture', str(input_params.get('architecture', 'N/A'))],
        ['Obfuscation Level', str(input_params.get('obfuscation_level', 'N/A'))],
        ['MLIR Frontend', str(input_params.get('mlir_frontend', 'clang'))],
        ['Compiler Flags', _safe_list_str(input_params.get('compiler_flags'), 'None')],
    ]
    global_table = Table(global_data, colWidths=[2.5*inch, 4.5*inch])
    global_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(global_table)
    story.append(Spacer(1, 0.15*inch))

    # OLLVM PASSES
    story.append(Paragraph("OLLVM Obfuscation Passes", styles['Heading3']))
    passes_data = [
        ['Pass', 'Enabled', 'Pass', 'Enabled'],
        ['Flattening', "✓" if input_params.get('pass_flattening', False) else "✗", 'String Encrypt', "✓" if input_params.get('pass_string_encrypt', False) else "✗"],
        ['Substitution', "✓" if input_params.get('pass_substitution', False) else "✗", 'Symbol Obf', "✓" if input_params.get('pass_symbol_obfuscate', False) else "✗"],
        ['Bogus CF', "✓" if input_params.get('pass_bogus_control_flow', False) else "✗", 'Const Obf', "✓" if input_params.get('pass_constant_obfuscate', False) else "✗"],
        ['Split', "✓" if input_params.get('pass_split', False) else "✗", '', ''],
        ['Linear MBA', "✓" if input_params.get('pass_linear_mba', False) else "✗", '', ''],
    ]
    passes_table = Table(passes_data, colWidths=[1.5*inch, 1.0*inch, 1.5*inch, 1.0*inch])
    passes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(passes_table)
    story.append(Spacer(1, 0.12*inch))

    # REQUESTED vs APPLIED PASSES
    passes_list_data = [
        ['Requested Passes', _safe_list_str(input_params.get('requested_passes'), 'None')],
        ['Applied Passes', _safe_list_str(input_params.get('applied_passes'), 'None')],
    ]
    passes_list_table = Table(passes_list_data, colWidths=[2.0*inch, 5.0*inch])
    passes_list_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(passes_list_table)
    story.append(Spacer(1, 0.15*inch))

    # ADVANCED FEATURES
    story.append(Paragraph("Advanced Features & Configuration", styles['Heading3']))
    advanced_data = [
        ['Feature', 'Value', 'Feature', 'Value'],
        ['Crypto Hash', "✓" if input_params.get('crypto_hash_enabled', False) else "✗", 'UPX Packing', "✓" if input_params.get('upx_enabled', False) else "✗"],
        ['Indirect Calls', "✓" if input_params.get('indirect_calls_enabled', False) else "✗", 'Remarks', "✓" if input_params.get('remarks_enabled', True) else "✗"],
        ['IR Metrics', "✓" if input_params.get('ir_metrics_enabled', True) else "✗", 'Preserve IR', "✓" if input_params.get('preserve_ir', True) else "✗"],
    ]
    advanced_table = Table(advanced_data, colWidths=[1.5*inch, 1.0*inch, 1.5*inch, 1.0*inch])
    advanced_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(advanced_table)
    story.append(Spacer(1, 0.15*inch))

    # ============= LAYER-BASED CONFIGURATION DETAILS =============

    # LAYER 1: SYMBOL OBFUSCATION (Crypto Hash)
    if input_params.get('pass_symbol_obfuscate', False) or input_params.get('crypto_hash_enabled', False):
        story.append(Paragraph("Layer 1: Symbol Obfuscation Configuration", styles['Heading3']))
        symbol_obf_data = [
            ['Setting', 'Value'],
            ['Crypto Hash Enabled', "✓ Yes" if input_params.get('crypto_hash_enabled', False) else "✗ No"],
            ['Algorithm', str(input_params.get('crypto_hash_algorithm', 'N/A'))],
            ['Salt', str(input_params.get('crypto_hash_salt', '')) or '(none)'],
            ['Hash Length', str(input_params.get('crypto_hash_length', 12))],
        ]
        symbol_obf_table = Table(symbol_obf_data, colWidths=[2.5*inch, 4.5*inch])
        symbol_obf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8e44ad')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(symbol_obf_table)
        story.append(Spacer(1, 0.12*inch))

    # LAYER 2: STRING ENCRYPTION
    if input_params.get('pass_string_encrypt', False):
        story.append(Paragraph("Layer 2: String Encryption Configuration", styles['Heading3']))
        string_enc_data = [
            ['Setting', 'Value'],
            ['String Encryption Enabled', "✓ Yes" if input_params.get('pass_string_encrypt', False) else "✗ No"],
            ['Min String Length', str(input_params.get('string_min_length', 'N/A'))],
            ['Encrypt Format Strings', "✓ Yes" if input_params.get('string_encrypt_format_strings', False) else "✗ No"],
        ]
        string_enc_table = Table(string_enc_data, colWidths=[2.5*inch, 4.5*inch])
        string_enc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(string_enc_table)
        story.append(Spacer(1, 0.12*inch))

    # LAYER 2: FAKE LOOPS INSERTION
    fake_loops_config = report.get('fake_loops_inserted', {})
    fake_loops_count = fake_loops_config.get('count', 0) if fake_loops_config else 0
    if fake_loops_count > 0:
        story.append(Paragraph("Layer 2: Fake Loops Configuration", styles['Heading3']))
        fake_loops_data = [
            ['Setting', 'Value'],
            ['Total Fake Loops Inserted', str(fake_loops_count)],
            ['Loop Types', ", ".join(fake_loops_config.get('types', [])) or 'None'],
            ['Locations', ", ".join(fake_loops_config.get('locations', [])) or 'None'],
        ]
        fake_loops_table = Table(fake_loops_data, colWidths=[2.5*inch, 4.5*inch])
        fake_loops_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(fake_loops_table)
        story.append(Spacer(1, 0.12*inch))

    # LAYER 2.5: INDIRECT CALLS
    if input_params.get('indirect_calls_enabled', False):
        story.append(Paragraph("Layer 2.5: Indirect Calls Configuration", styles['Heading3']))
        indirect_calls_data = [
            ['Setting', 'Value'],
            ['Indirect Calls Enabled', "✓ Yes" if input_params.get('indirect_calls_enabled', False) else "✗ No"],
            ['Obfuscate Stdlib', "✓ Yes" if input_params.get('indirect_calls_obfuscate_stdlib', True) else "✗ No"],
            ['Obfuscate Custom Calls', "✓ Yes" if input_params.get('indirect_calls_obfuscate_custom', True) else "✗ No"],
        ]
        indirect_calls_table = Table(indirect_calls_data, colWidths=[2.5*inch, 4.5*inch])
        indirect_calls_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d35400')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(indirect_calls_table)
        story.append(Spacer(1, 0.12*inch))

    # LAYER 4: COMPILER FLAGS
    compiler_flags = input_params.get('compiler_flags', [])
    if compiler_flags:
        story.append(Paragraph("Layer 4: Compiler Flags Configuration", styles['Heading3']))
        flags_data = [
            ['Compiler Flag', 'Status'],
        ]
        for flag in compiler_flags:
            flags_data.append([str(flag), '✓ Applied'])

        flags_table = Table(flags_data, colWidths=[3.0*inch, 4.0*inch])
        flags_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(flags_table)
        story.append(Spacer(1, 0.12*inch))

    # LAYER 5: UPX PACKING
    if input_params.get('upx_enabled', False):
        story.append(Paragraph("Layer 5: UPX Packing Configuration", styles['Heading3']))
        upx_data = [
            ['Setting', 'Value'],
            ['UPX Packing Enabled', "✓ Yes" if input_params.get('upx_enabled', False) else "✗ No"],
            ['Compression Level', str(input_params.get('upx_compression_level', 'best'))],
            ['Use LZMA', "✓ Yes" if input_params.get('upx_use_lzma', True) else "✗ No"],
            ['Preserve Original', "✓ Yes" if input_params.get('upx_preserve_original', False) else "✗ No"],
        ]
        upx_table = Table(upx_data, colWidths=[2.5*inch, 4.5*inch])
        upx_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(upx_table)
        story.append(Spacer(1, 0.12*inch))

    # Baseline Compilation Details (if available)
    baseline_compiler = report.get('baseline_compiler', {})
    if baseline_compiler:
        story.append(Paragraph("<b>Baseline Compilation Details</b>", styles['Heading2']))
        story.append(Spacer(1, 0.08*inch))

        compiler_data = [
            ['Compiler', baseline_compiler.get('compiler', 'N/A')],
            ['Version', baseline_compiler.get('version', 'N/A')],
            ['Optimization Level', baseline_compiler.get('optimization_level', 'N/A')],
            ['Compilation Method', baseline_compiler.get('compilation_method', 'N/A')],
        ]
        compiler_table = Table(compiler_data, colWidths=[2*inch, 5*inch])
        compiler_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ('LEFTPADDING', (0, 0), (0, -1), 20),  # Blue left border effect
        ]))
        story.append(compiler_table)

    # Spacing before next section (no forced page break to allow natural continuation)
    story.append(Spacer(1, 0.1*inch))

    # ============= CONTINUATION: Before/After Comparison & Attributes =============

    story.append(Paragraph("Metrics Comparison & Output Details", styles['Heading1']))
    story.append(Spacer(1, 0.05*inch))

    # Before/After Comparison - BAR CHART
    story.append(Paragraph("<b>Before/After Metrics</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    baseline_metrics = report.get('baseline_metrics', {})
    output_attrs = report.get('output_attributes', {})
    comparison = report.get('comparison', {})

    if MATPLOTLIB_AVAILABLE:
        try:
            # Create comparison bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)

            # Symbols chart
            baseline_symbols = int(baseline_metrics.get('symbols_count', 0))
            output_symbols = int(output_attrs.get('symbols_count', 0))

            ax1.bar(['Baseline', 'Obfuscated'], [baseline_symbols, output_symbols],
                   color=['#3498DB', '#E74C3C'], edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('Count', fontweight='bold', fontsize=10)
            ax1.set_title('Symbol Count Comparison', fontweight='bold', fontsize=11)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            for i, (baseline, obf) in enumerate([(baseline_symbols, output_symbols)]):
                ax1.text(0, baseline, f'{baseline}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                ax1.text(1, obf, f'{obf}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            # Functions chart
            baseline_functions = int(baseline_metrics.get('functions_count', 0))
            output_functions = int(output_attrs.get('functions_count', 0))

            ax2.bar(['Baseline', 'Obfuscated'], [baseline_functions, output_functions],
                   color=['#3498DB', '#E74C3C'], edgecolor='black', linewidth=1.5)
            ax2.set_ylabel('Count', fontweight='bold', fontsize=10)
            ax2.set_title('Function Count Comparison', fontweight='bold', fontsize=11)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            for i, (baseline, obf) in enumerate([(baseline_functions, output_functions)]):
                ax2.text(0, baseline, f'{baseline}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                ax2.text(1, obf, f'{obf}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            plt.tight_layout()

            # Convert to image
            img_buffer_comparison = BytesIO()
            plt.savefig(img_buffer_comparison, format='png', bbox_inches='tight', dpi=100)
            img_buffer_comparison.seek(0)
            plt.close(fig)

            from reportlab.platypus import Image as RLImage
            comparison_chart = RLImage(img_buffer_comparison, width=6.5*inch, height=2.5*inch)
            story.append(comparison_chart)
            story.append(Spacer(1, 0.08*inch))

        except Exception as e:
            logger.warning(f"Failed to create comparison chart: {e}")
            # Fallback to table
            comparison_data = [
                ['Metric', 'Baseline', 'Obfuscated', 'Change'],
                ['Symbols',
                 str(baseline_metrics.get('symbols_count', 'N/A')),
                 str(output_attrs.get('symbols_count', 'N/A')),
                 f"-{_safe_float(comparison.get('symbols_removed_percent'), 0):.1f}%"],
                ['Functions',
                 str(baseline_metrics.get('functions_count', 'N/A')),
                 str(output_attrs.get('functions_count', 'N/A')),
                 f"-{_safe_float(comparison.get('functions_removed_percent'), 0):.1f}%"],
            ]
            comparison_table = Table(comparison_data, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
            comparison_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(comparison_table)
            story.append(Spacer(1, 0.08*inch))
    else:
        # Fallback if matplotlib not available
        comparison_data = [
            ['Metric', 'Baseline', 'Obfuscated', 'Change'],
            ['Symbols',
             str(baseline_metrics.get('symbols_count', 'N/A')),
             str(output_attrs.get('symbols_count', 'N/A')),
             f"-{_safe_float(comparison.get('symbols_removed_percent'), 0):.1f}%"],
            ['Functions',
             str(baseline_metrics.get('functions_count', 'N/A')),
             str(output_attrs.get('functions_count', 'N/A')),
             f"-{_safe_float(comparison.get('functions_removed_percent'), 0):.1f}%"],
        ]
        comparison_table = Table(comparison_data, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
        ]))
        story.append(comparison_table)
        story.append(Spacer(1, 0.08*inch))

    # Output Attributes
    story.append(Paragraph("<b>Output File Attributes</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    output_data = [
        ['File Size', format_bytes(output_attrs.get('file_size', 0))],
        ['Binary Format', output_attrs.get('binary_format', 'N/A')],
        ['Symbol Count', str(output_attrs.get('symbols_count', 'N/A'))],
        ['Function Count', str(output_attrs.get('functions_count', 'N/A'))],
        ['Binary Entropy', format_entropy(output_attrs.get('entropy', 0))],
        ['Obfuscation Methods', _safe_list_str(output_attrs.get('obfuscation_methods'), 'Default pipeline')],
    ]
    output_table = Table(output_data, colWidths=[2*inch, 5*inch])
    output_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
    ]))
    story.append(output_table)
    story.append(Spacer(1, 0.08*inch))

    # Bogus Code Generation
    bogus_info = report.get('bogus_code_info', {})
    # Handle both dict and string formats
    if bogus_info and (isinstance(bogus_info, dict) and any(bogus_info.values()) or isinstance(bogus_info, str)):
        story.append(Paragraph("<b>Bogus Code Generation</b>", styles['Heading2']))
        story.append(Spacer(1, 0.08*inch))

        if isinstance(bogus_info, dict):
            bogus_data = [
                ['Dead Code Blocks', str(bogus_info.get('dead_code_blocks', 0))],
                ['Opaque Predicates', str(bogus_info.get('opaque_predicates', 0))],
                ['Junk Instructions', str(bogus_info.get('junk_instructions', 0))],
                ['Code Bloat %', f"{_safe_float(bogus_info.get('code_bloat_percentage'), 0):.2f}%"],
            ]
        else:
            # String format - just display it
            story.append(Paragraph(str(bogus_info), styles['Normal']))
            story.append(Spacer(1, 0.08*inch))
            bogus_data = None

        if bogus_data:
            bogus_table = Table(bogus_data, colWidths=[2*inch, 5*inch])
            bogus_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ffe0e0')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(bogus_table)
            story.append(Spacer(1, 0.08*inch))

    # String Obfuscation - with real metrics
    string_obf = report.get('string_obfuscation', {})
    if isinstance(string_obf, dict):
        is_enabled = string_obf.get('enabled', False)
        total_strings = string_obf.get('total_strings', 0)
        encrypted_strings = string_obf.get('encrypted_strings', 0)
        encryption_pct = string_obf.get('encryption_percentage', 0.0)
        method = string_obf.get('method', 'MLIR string-encrypt pass')

        # Only show if enabled or has actual metrics
        if is_enabled or total_strings > 0:
            story.append(Paragraph("<b>🔐 String Obfuscation</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            if MATPLOTLIB_AVAILABLE and total_strings > 0:
                try:
                    # Create pie chart for string encryption
                    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

                    unencrypted_strings = max(0, total_strings - encrypted_strings)
                    sizes = [encrypted_strings, unencrypted_strings]
                    labels = [f'Encrypted\n({encrypted_strings:,})', f'Unencrypted\n({unencrypted_strings:,})']
                    colors_string = ['#28a745', '#E8E8E8']

                    wedges, texts, autotexts = ax.pie(
                        sizes,
                        labels=labels,
                        autopct='%1.1f%%',
                        colors=colors_string,
                        startangle=90,
                        textprops={'fontsize': 10, 'weight': 'bold'}
                    )

                    for autotext in autotexts:
                        autotext.set_color('white' if autotext.xy[0] > 0 else 'black')
                        autotext.set_fontsize(11)
                        autotext.set_weight('bold')

                    ax.set_title(f'String Encryption Rate: {encryption_pct:.1f}%', fontsize=12, weight='bold', pad=20)
                    plt.tight_layout()

                    # Convert to image
                    img_buffer_string = BytesIO()
                    plt.savefig(img_buffer_string, format='png', bbox_inches='tight', dpi=100)
                    img_buffer_string.seek(0)
                    plt.close(fig)

                    from reportlab.platypus import Image as RLImage
                    string_chart = RLImage(img_buffer_string, width=3.5*inch, height=2.8*inch)
                    story.append(string_chart)
                    story.append(Spacer(1, 0.08*inch))

                except Exception as e:
                    logger.warning(f"Failed to create string encryption chart: {e}")
                    # Fallback to table
                    string_data = [
                        ['Total Strings', f"{total_strings:,}"],
                        ['Encrypted', f"{encrypted_strings:,}"],
                        ['Rate', f"{encryption_pct:.1f}%"],
                    ]
                    string_table = Table(string_data, colWidths=[2.5*inch, 4.5*inch])
                    string_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('PADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                    ]))
                    story.append(string_table)
                    story.append(Spacer(1, 0.08*inch))
            else:
                # Fallback if matplotlib not available or no strings
                string_data = [
                    ['Total Strings', f"{total_strings:,}"],
                    ['Encrypted', f"{encrypted_strings:,}"],
                    ['Rate', f"{encryption_pct:.1f}%"],
                    ['Method', method],
                ]
                string_table = Table(string_data, colWidths=[2.5*inch, 4.5*inch])
                string_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                ]))
                story.append(string_table)
                story.append(Spacer(1, 0.08*inch))

    # Symbol Obfuscation - with real metrics
    symbol_obf = report.get('symbol_obfuscation', {})
    if isinstance(symbol_obf, dict):
        is_enabled = symbol_obf.get('enabled', False)
        symbols_obfuscated = symbol_obf.get('symbols_obfuscated', 0)
        reduction_pct = symbol_obf.get('reduction_percentage', 0.0)
        algorithm = symbol_obf.get('algorithm', 'MLIR symbol-obfuscate pass')

        # Show if enabled
        if is_enabled or symbols_obfuscated > 0:
            story.append(Paragraph("<b>🔤 Symbol Obfuscation</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            symbol_data = [
                ['Symbols Obfuscated', f"{symbols_obfuscated:,}"],
                ['Symbol Reduction', f"{reduction_pct:.1f}%"],
                ['Algorithm', algorithm],
            ]
            symbol_table = Table(symbol_data, colWidths=[2.5*inch, 4.5*inch])
            symbol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8957e5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0e6ff')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(symbol_table)
            story.append(Spacer(1, 0.08*inch))

    # Cycles Completed
    cycles = report.get('cycles_completed', {})
    if cycles and cycles.get('total_cycles'):
        story.append(Paragraph("<b>Obfuscation Cycles</b>", styles['Heading2']))
        story.append(Spacer(1, 0.08*inch))

        cycles_per_cycle = cycles.get('per_cycle_metrics', [])

        # Create cycle duration chart if we have multiple cycles
        if len(cycles_per_cycle) > 0 and MATPLOTLIB_AVAILABLE:
            try:
                cycle_nums = [str(c.get('cycle', i+1)) for i, c in enumerate(cycles_per_cycle)]
                durations = [int(c.get('duration_ms', 0)) for c in cycles_per_cycle]

                fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=100)
                bars = ax.bar(cycle_nums, durations, color='#3498DB', edgecolor='black', linewidth=1.5)

                # Add value labels on bars
                for bar, duration in zip(bars, durations):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{duration}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)

                ax.set_ylabel('Duration (ms)', fontweight='bold', fontsize=11)
                ax.set_xlabel('Cycle', fontweight='bold', fontsize=11)
                ax.set_title('Obfuscation Cycle Execution Time', fontweight='bold', fontsize=12)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                plt.tight_layout()

                img_buffer_cycles = BytesIO()
                plt.savefig(img_buffer_cycles, format='png', bbox_inches='tight', dpi=100)
                img_buffer_cycles.seek(0)
                plt.close(fig)

                from reportlab.platypus import Image as RLImage
                cycles_chart = RLImage(img_buffer_cycles, width=5.5*inch, height=2.3*inch)
                story.append(cycles_chart)
                story.append(Spacer(1, 0.08*inch))

            except Exception as e:
                logger.warning(f"Failed to create cycles chart: {e}")
                # Fallback to table
                cycles_data = [['Cycle', 'Duration (ms)']]
                for cycle_info in cycles_per_cycle:
                    cycles_data.append([
                        str(cycle_info.get('cycle', 'N/A')),
                        str(cycle_info.get('duration_ms', 0))
                    ])
                if len(cycles_data) > 1:
                    cycles_table = Table(cycles_data, colWidths=[2*inch, 3*inch])
                    cycles_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('PADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                    ]))
                    story.append(cycles_table)
        else:
            # Fallback if no cycles or matplotlib unavailable
            cycles_data = [['Cycle', 'Duration (ms)']]
            for cycle_info in cycles_per_cycle:
                cycles_data.append([
                    str(cycle_info.get('cycle', 'N/A')),
                    str(cycle_info.get('duration_ms', 0))
                ])
            if len(cycles_data) > 1:
                cycles_table = Table(cycles_data, colWidths=[2*inch, 3*inch])
                cycles_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                ]))
                story.append(cycles_table)

    # ============= ADVANCED METRICS SECTION (if available) =============

    control_flow = report.get('control_flow_metrics')
    instruction_metrics = report.get('instruction_metrics')
    binary_structure = report.get('binary_structure')
    pattern_resistance = report.get('pattern_resistance')

    if any([control_flow, instruction_metrics, binary_structure, pattern_resistance]):
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Advanced Analysis Dashboard", styles['Heading1']))
        story.append(Spacer(1, 0.05*inch))

        # Control Flow Metrics
        if control_flow:
            story.append(Paragraph("<b>Control Flow Analysis</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            cf_baseline = control_flow.get('baseline', {})
            cf_obfuscated = control_flow.get('obfuscated', {})
            cf_comparison = control_flow.get('comparison', {})

            # Control Flow comparison chart
            if MATPLOTLIB_AVAILABLE:
                try:
                    basic_blocks_baseline = int(cf_baseline.get('basic_blocks', 0))
                    basic_blocks_obf = int(cf_obfuscated.get('basic_blocks', 0))
                    cyclomatic_baseline = int(cf_baseline.get('cyclomatic_complexity', 0))
                    cyclomatic_obf = int(cf_obfuscated.get('cyclomatic_complexity', 0))

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)

                    # Basic Blocks comparison
                    ax1.bar(['Baseline', 'Obfuscated'], [basic_blocks_baseline, basic_blocks_obf],
                           color=['#3498DB', '#E74C3C'], edgecolor='black', linewidth=1.5)
                    ax1.set_ylabel('Count', fontweight='bold', fontsize=10)
                    ax1.set_title('Basic Blocks Comparison', fontweight='bold', fontsize=11)
                    ax1.grid(axis='y', alpha=0.3, linestyle='--')
                    ax1.text(0, basic_blocks_baseline, f'{basic_blocks_baseline}', ha='center', va='bottom', fontweight='bold')
                    ax1.text(1, basic_blocks_obf, f'{basic_blocks_obf}', ha='center', va='bottom', fontweight='bold')

                    # Cyclomatic Complexity comparison
                    ax2.bar(['Baseline', 'Obfuscated'], [cyclomatic_baseline, cyclomatic_obf],
                           color=['#3498DB', '#E74C3C'], edgecolor='black', linewidth=1.5)
                    ax2.set_ylabel('Complexity', fontweight='bold', fontsize=10)
                    ax2.set_title('Cyclomatic Complexity Comparison', fontweight='bold', fontsize=11)
                    ax2.grid(axis='y', alpha=0.3, linestyle='--')
                    ax2.text(0, cyclomatic_baseline, f'{cyclomatic_baseline}', ha='center', va='bottom', fontweight='bold')
                    ax2.text(1, cyclomatic_obf, f'{cyclomatic_obf}', ha='center', va='bottom', fontweight='bold')

                    plt.tight_layout()

                    img_buffer_cf = BytesIO()
                    plt.savefig(img_buffer_cf, format='png', bbox_inches='tight', dpi=100)
                    img_buffer_cf.seek(0)
                    plt.close(fig)

                    from reportlab.platypus import Image as RLImage
                    cf_chart = RLImage(img_buffer_cf, width=6.5*inch, height=2.5*inch)
                    story.append(cf_chart)
                    story.append(Spacer(1, 0.08*inch))

                except Exception as e:
                    logger.warning(f"Failed to create control flow chart: {e}")
                    # Fallback to table
                    cf_data = [
                        ['Metric', 'Baseline', 'Obfuscated', 'Change'],
                        ['Basic Blocks', str(basic_blocks_baseline), str(basic_blocks_obf),
                         f"+{cf_comparison.get('basic_blocks_added', 0)}"],
                        ['Cyclomatic', str(cyclomatic_baseline), str(cyclomatic_obf),
                         f"+{_safe_float(cf_comparison.get('complexity_increase_percent'), 0):.1f}%"],
                    ]
                    cf_table = Table(cf_data, colWidths=[1.7*inch, 1.6*inch, 1.6*inch, 1.6*inch])
                    cf_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('PADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                    ]))
                    story.append(cf_table)
                    story.append(Spacer(1, 0.08*inch))
            else:
                # Fallback if matplotlib unavailable
                cf_data = [
                    ['Metric', 'Baseline', 'Obfuscated', 'Change'],
                    ['Basic Blocks', str(cf_baseline.get('basic_blocks', 0)), str(cf_obfuscated.get('basic_blocks', 0)),
                     f"+{cf_comparison.get('basic_blocks_added', 0)}"],
                    ['Cyclomatic', str(cf_baseline.get('cyclomatic_complexity', 0)),
                     str(cf_obfuscated.get('cyclomatic_complexity', 0)),
                     f"+{_safe_float(cf_comparison.get('complexity_increase_percent'), 0):.1f}%"],
                ]
                cf_table = Table(cf_data, colWidths=[1.7*inch, 1.6*inch, 1.6*inch, 1.6*inch])
                cf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                ]))
                story.append(cf_table)
                story.append(Spacer(1, 0.08*inch))

            # Add Control Flow Chart
            cf_chart = create_control_flow_chart(report)
            if cf_chart:
                try:
                    cf_chart_img = Image(BytesIO(cf_chart), width=5.5*inch, height=2.8*inch)
                    story.append(cf_chart_img)
                    story.append(Spacer(1, 0.12*inch))
                except Exception as e:
                    logger.error(f"Error embedding control flow chart: {e}")

        # Instruction Metrics
        if instruction_metrics:
            story.append(Paragraph("<b>Instruction-Level Metrics</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            instr_baseline = instruction_metrics.get('baseline', {})
            instr_obfuscated = instruction_metrics.get('obfuscated', {})
            instr_comparison = instruction_metrics.get('comparison', {})

            instr_data = [
                ['Metric', 'Baseline', 'Obfuscated', 'Change %'],
                ['Total Instructions', str(instr_baseline.get('total_instructions', 0)),
                 str(instr_obfuscated.get('total_instructions', 0)),
                 f"+{_safe_float(instr_comparison.get('instruction_growth_percent'), 0):.1f}%"],
                ['Arithmetic Complexity', f"{_safe_float(instr_baseline.get('arithmetic_complexity_score'), 0):.2f}",
                 f"{_safe_float(instr_obfuscated.get('arithmetic_complexity_score'), 0):.2f}",
                 f"+{_safe_float(instr_comparison.get('arithmetic_complexity_increase'), 0):.1f}"],
                ['MBA Expressions', str(instr_baseline.get('mba_expression_count', 0)),
                 str(instr_obfuscated.get('mba_expression_count', 0)),
                 f"+{instr_comparison.get('mba_expressions_added', 0)}"],
            ]
            instr_table = Table(instr_data, colWidths=[2*inch, 1.6*inch, 1.6*inch, 1.6*inch])
            instr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(instr_table)
            story.append(Spacer(1, 0.08*inch))

            # Add Instruction Distribution Charts
            instr_chart = create_instruction_distribution_charts(report)
            if instr_chart:
                try:
                    instr_chart_img = Image(BytesIO(instr_chart), width=6*inch, height=2.8*inch)
                    story.append(instr_chart_img)
                    story.append(Spacer(1, 0.12*inch))
                except Exception as e:
                    logger.error(f"Error embedding instruction chart: {e}")

        # Binary Structure
        if binary_structure:
            story.append(Paragraph("<b>Binary Structure Analysis</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            binary_data = [
                ['Section Count', str(binary_structure.get('section_count', 0))],
                ['Import Table Entries', str(binary_structure.get('import_table', {}).get('imported_symbols', 0))],
                ['Relocation Count', str(binary_structure.get('relocations', {}).get('relocation_count', 0))],
                ['Code to Data Ratio', f"{_safe_float(binary_structure.get('code_to_data_ratio'), 0):.2f}"],
            ]
            binary_table = Table(binary_data, colWidths=[3*inch, 4*inch])
            binary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ffe0e0')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(binary_table)
            story.append(Spacer(1, 0.12*inch))

        # Pattern Resistance
        if pattern_resistance:
            story.append(Paragraph("<b>Pattern Resistance & RE Difficulty</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            pattern_data = [
                ['Visible String Count', str(pattern_resistance.get('string_analysis', {}).get('visible_string_count', 0))],
                ['String Entropy', f"{_safe_float(pattern_resistance.get('string_analysis', {}).get('string_entropy'), 0):.2f}"],
                ['Opcode Distribution Entropy', f"{_safe_float(pattern_resistance.get('code_analysis', {}).get('opcode_distribution_entropy'), 0):.2f}"],
                ['Decompiler Confusion Score', f"{_safe_float(pattern_resistance.get('reverse_engineering_difficulty', {}).get('decompiler_confusion_score'), 0):.1f}"],
            ]
            pattern_table = Table(pattern_data, colWidths=[3*inch, 4*inch])
            pattern_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8957e5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4ecf7')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(pattern_table)

    # ============= TEST SUITE RESULTS SECTION (if available) =============

    test_results = report.get('test_results')
    if test_results:
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Functional Correctness & Reliability Testing", styles['Heading1']))
        story.append(Spacer(1, 0.05*inch))

        # Reliability Status Banner
        reliability_status = report.get('reliability_status', {})
        reliability_level = reliability_status.get('level', 'UNKNOWN')

        if reliability_level == 'HIGH':
            banner_color = colors.HexColor('#e8f5e9')
            header_color = colors.HexColor('#28a745')
            text_color = colors.HexColor('#1b5e20')
            status_emoji = '✅'
        elif reliability_level == 'MEDIUM':
            banner_color = colors.HexColor('#fff9c4')
            header_color = colors.HexColor('#ffc107')
            text_color = colors.HexColor('#f57f17')
            status_emoji = '⚠️'
        else:
            banner_color = colors.HexColor('#ffebee')
            header_color = colors.HexColor('#dc3545')
            text_color = colors.HexColor('#b71c1c')
            status_emoji = '❌'

        reliability_data = [[f'{status_emoji} Reliability Status: {reliability_level}']]
        reliability_table = Table(reliability_data, colWidths=[7*inch])
        reliability_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), banner_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), text_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BORDER', (0, 0), (-1, -1), 2, header_color),
        ]))
        story.append(reliability_table)
        story.append(Spacer(1, 0.15*inch))

        # Test Metadata
        metadata = report.get('metadata', {})
        test_data = [
            ['Timestamp', metadata.get('timestamp', 'N/A')],
            ['Program', metadata.get('program', 'N/A')],
            ['Functional Correctness', 'PASSED ✓' if report.get('functional_correctness_passed') else 'FAILED ✗'],
        ]
        test_table = Table(test_data, colWidths=[2*inch, 5*inch])
        test_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
        ]))
        story.append(test_table)
        story.append(Spacer(1, 0.12*inch))

        # Functional Results
        functional = test_results.get('functional', {})
        if functional:
            story.append(Paragraph("<b>Functional Test Results</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            same_behavior = 'YES ✓' if functional.get('same_behavior') else 'NO ✗'
            func_data = [
                ['Same Behavior', same_behavior],
                ['Tests Passed', f"{functional.get('passed', 0)}/{functional.get('test_count', 0)}"],
                ['Tests Failed', str(functional.get('failed', 0))],
            ]
            func_table = Table(func_data, colWidths=[3*inch, 4*inch])
            func_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(func_table)
            story.append(Spacer(1, 0.12*inch))

        # Binary Properties
        binary_props = test_results.get('binary_properties', {})
        if binary_props:
            story.append(Paragraph("<b>Binary Properties Comparison</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            binary_props_data = [
                ['Baseline Size', format_bytes(binary_props.get('baseline_size_bytes', 0))],
                ['Obfuscated Size', format_bytes(binary_props.get('obf_size_bytes', 0))],
                ['Size Change', format_percentage(binary_props.get('size_increase_percent', 0))],
                ['Baseline Entropy', format_entropy(binary_props.get('baseline_entropy', 0))],
                ['Obfuscated Entropy', format_entropy(binary_props.get('obf_entropy', 0))],
            ]
            binary_props_table = Table(binary_props_data, colWidths=[3*inch, 4*inch])
            binary_props_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(binary_props_table)
            story.append(Spacer(1, 0.12*inch))

        # Performance Analysis
        performance = test_results.get('performance', {})
        if performance:
            story.append(Paragraph("<b>Performance Analysis</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            # Determine acceptable status based on overhead
            baseline_ms = _safe_float(performance.get('baseline_ms'), 0)
            obf_ms = _safe_float(performance.get('obf_ms'), 0)
            overhead_pct = _safe_float(performance.get('overhead_percent'), 0)

            if baseline_ms == 0 and obf_ms == 0:
                acceptable_str = 'N/A (not measured)'
            elif overhead_pct < 50:
                acceptable_str = 'YES ✓'
            else:
                acceptable_str = 'NO ✗'

            perf_data = [
                ['Baseline Time', format_time(baseline_ms)],
                ['Obfuscated Time', format_time(obf_ms)],
                ['Overhead', format_percentage(overhead_pct)],
                ['Acceptable', acceptable_str],
            ]
            perf_table = Table(perf_data, colWidths=[3*inch, 4*inch])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ffe0e0')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(perf_table)

    # ============= PHORONIX BENCHMARKING SECTION (if available) =============
    phoronix_info = report.get('phoronix', {})
    if phoronix_info and phoronix_info.get('key_metrics'):
        key_metrics = phoronix_info.get('key_metrics', {})
        if key_metrics.get('available'):
            story.append(Spacer(1, 0.2*inch))

            # Section heading
            bench_heading = ParagraphStyle(
                'BenchHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1f6feb'),
                spaceAfter=12,
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph("📊 Obfuscation Impact Analysis", bench_heading))
            story.append(Spacer(1, 0.08*inch))

            # Build metrics table with only reliable metrics
            bench_data = [
                ['Metric', 'Value', 'Interpretation'],
            ]

            # Instruction Count Increase (RELIABLE - directly measurable)
            instr_delta = key_metrics.get('instruction_count_delta')
            instr_percent = key_metrics.get('instruction_count_increase_percent')
            if instr_delta is not None:
                instr_interp = "Code expansion due to obfuscation passes"
                bench_data.append([
                    'Instruction Count Increase',
                    f"+{instr_delta} ({instr_percent}%)",
                    instr_interp
                ])

            # Performance Overhead (if available)
            perf_overhead = key_metrics.get('performance_overhead_percent')
            if perf_overhead is not None:
                if perf_overhead < 5:
                    perf_interp = "Minimal performance impact"
                elif perf_overhead < 20:
                    perf_interp = "Acceptable performance cost"
                elif perf_overhead < 50:
                    perf_interp = "Significant performance overhead"
                else:
                    perf_interp = "High performance cost"
                bench_data.append([
                    'Performance Overhead',
                    f"+{perf_overhead}%",
                    perf_interp
                ])

            bench_table = Table(bench_data, colWidths=[2*inch, 1.5*inch, 3.5*inch])
            bench_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17a2b8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e7f7ff')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(bench_table)
            story.append(Spacer(1, 0.1*inch))

            # Information note about metrics
            note_style = ParagraphStyle(
                'NoteStyle',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#666666'),
                fontName='Helvetica-Oblique',
                leftIndent=20
            )
            note_text = (
                "<b>Note:</b> Displayed metrics are based on static code analysis and represent measurable "
                "changes from obfuscation passes. Instruction count increase reflects code expansion from "
                "obfuscation techniques. Performance overhead reflects runtime slowdown if measured. "
                "For comprehensive reverse engineering difficulty assessment, manual analysis with "
                "decompilation tools (Ghidra, IDA) is recommended."
            )
            story.append(Paragraph(note_text, note_style))

    # ============= PROTECTION SCORE DETAILS SECTION (Integrated) =============
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("Protection Score Analysis", styles['Heading1']))
    story.append(Spacer(1, 0.05*inch))

    # PRCS Framework Score
    score = _safe_float(report.get('obfuscation_score', 0))

    # Score interpretation for 0-100 scale
    if score >= 85:
        score_interpretation = "Exceptional obfuscation with very high reverse engineering difficulty"
    elif score >= 75:
        score_interpretation = "Strong obfuscation with high reverse engineering difficulty"
    elif score >= 65:
        score_interpretation = "Solid obfuscation with moderate to high reverse engineering difficulty"
    elif score >= 50:
        score_interpretation = "Reasonable obfuscation with moderate reverse engineering difficulty"
    else:
        score_interpretation = "Basic obfuscation with minimal to moderate reverse engineering difficulty"

    # Create protection score table (displayed in table format, not big)
    protection_score_data = [
        ['Metric', 'Score', 'Details'],
        ['PRCS Score', f'{score:.1f}/100', score_interpretation],
        ['Standard', 'OWASP MASTG', 'Potency 30% | Resilience 35% | Cost 20% | Stealth 15%'],
        ['Scale', '0-100', 'Max score is practically impossible to achieve'],
    ]
    protection_score_table = Table(protection_score_data, colWidths=[2.0*inch, 1.5*inch, 3.5*inch])
    protection_score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f5f5f5')),
        ('BACKGROUND', (0, 2), (-1, 2), colors.white),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f5f5f5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(protection_score_table)
    story.append(Spacer(1, 0.1*inch))

    # Score Gauge Chart
    if MATPLOTLIB_AVAILABLE:
        try:
            # Create a visual gauge for the score
            fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)

            # Create horizontal bar gauge
            max_score = 100
            current_score = score

            # Determine color based on score
            if current_score >= 85:
                gauge_color = '#27AE60'  # Green
            elif current_score >= 65:
                gauge_color = '#F39C12'  # Orange
            else:
                gauge_color = '#E74C3C'  # Red

            # Draw background bar
            ax.barh(0, max_score, color='#ECF0F1', height=0.5)
            # Draw score bar
            ax.barh(0, current_score, color=gauge_color, height=0.5, edgecolor='black', linewidth=2)

            # Add score text
            ax.text(current_score/2, 0, f'{current_score:.1f}',
                   ha='center', va='center', fontsize=20, fontweight='bold', color='white')

            # Add min/max labels
            ax.text(0, -0.4, '0', ha='center', fontsize=9, fontweight='bold')
            ax.text(100, -0.4, '100', ha='center', fontsize=9, fontweight='bold')

            # Remove axes
            ax.set_xlim(-5, 105)
            ax.set_ylim(-1, 1)
            ax.axis('off')

            plt.tight_layout()

            # Convert to image
            img_buffer_gauge = BytesIO()
            plt.savefig(img_buffer_gauge, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            img_buffer_gauge.seek(0)
            plt.close(fig)

            from reportlab.platypus import Image as RLImage
            gauge_chart = RLImage(img_buffer_gauge, width=4.5*inch, height=1.5*inch)
            story.append(gauge_chart)
        except Exception as e:
            logger.warning(f"Failed to create score gauge: {e}")
            pass

    story.append(Spacer(1, 0.15*inch))

    # PRCS Component Breakdown - PIE CHART
    story.append(Paragraph("<b>PRCS Framework Weights</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    if MATPLOTLIB_AVAILABLE:
        try:
            # Create pie chart for PRCS weights
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

            components = ['Potency', 'Resilience', 'Cost', 'Stealth']
            weights = [30, 35, 20, 15]
            colors_prcs = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']

            wedges, texts, autotexts = ax.pie(
                weights,
                labels=components,
                autopct='%1.0f%%',
                colors=colors_prcs,
                startangle=90,
                textprops={'fontsize': 10, 'weight': 'bold'}
            )

            # Style the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
                autotext.set_weight('bold')

            ax.set_title('PRCS Component Weights', fontsize=12, weight='bold', pad=20)
            plt.tight_layout()

            # Convert to image
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            img_buffer.seek(0)
            plt.close(fig)

            from reportlab.platypus import Image as RLImage
            prcs_chart = RLImage(img_buffer, width=4*inch, height=2.8*inch)
            story.append(prcs_chart)
        except Exception as e:
            logger.warning(f"Failed to create PRCS pie chart: {e}")
            # Fallback to simple table
            prcs_data = [
                ['Component', 'Weight'],
                ['Potency', '30%'],
                ['Resilience', '35%'],
                ['Cost', '20%'],
                ['Stealth', '15%'],
            ]
            prcs_table = Table(prcs_data, colWidths=[3*inch, 2*inch])
            prcs_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(prcs_table)
    else:
        # Fallback if matplotlib not available
        prcs_data = [
            ['Component', 'Weight'],
            ['Potency', '30%'],
            ['Resilience', '35%'],
            ['Cost', '20%'],
            ['Stealth', '15%'],
        ]
        prcs_table = Table(prcs_data, colWidths=[3*inch, 2*inch])
        prcs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
        ]))
        story.append(prcs_table)

    story.append(Spacer(1, 0.1*inch))

    # Protection Metrics Chart (Bar Chart)
    story.append(Paragraph("<b>Obfuscation Metrics</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    if MATPLOTLIB_AVAILABLE:
        try:
            # Extract metrics
            symbol_red = _safe_float(report.get('symbol_reduction', 0), 0)
            func_red = _safe_float(report.get('function_reduction', 0), 0)
            entropy_inc = _safe_float(report.get('entropy_increase', 0), 0)

            # Normalize entropy to 0-100 scale for visualization (capped at 100)
            entropy_normalized = min(entropy_inc * 100, 100)

            # Create bar chart
            fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)

            metrics_names = ['Symbol\nReduction', 'Function\nReduction', 'Entropy\nIncrease']
            metrics_values = [symbol_red, func_red, entropy_normalized]
            bars_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

            bars = ax.bar(metrics_names, metrics_values, color=bars_colors, edgecolor='black', linewidth=1.5)

            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold'
                )

            ax.set_ylabel('Value (%)', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 110)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            plt.tight_layout()

            # Convert to image
            img_buffer_metrics = BytesIO()
            plt.savefig(img_buffer_metrics, format='png', bbox_inches='tight', dpi=100)
            img_buffer_metrics.seek(0)
            plt.close(fig)

            from reportlab.platypus import Image as RLImage
            metrics_chart = RLImage(img_buffer_metrics, width=4.5*inch, height=2.5*inch)
            story.append(metrics_chart)
        except Exception as e:
            logger.warning(f"Failed to create metrics bar chart: {e}")
            # Fallback to simple display
            metrics_display = f"""<font size=8>
            <b>Symbol Reduction:</b> {_safe_float(report.get('symbol_reduction', 0)):.1f}%<br/>
            <b>Function Reduction:</b> {_safe_float(report.get('function_reduction', 0)):.1f}%<br/>
            <b>Entropy Increase:</b> {_safe_float(report.get('entropy_increase', 0)):.4f}
            </font>"""
            story.append(Paragraph(metrics_display, styles['Normal']))
    else:
        # Fallback if matplotlib not available
        metrics_display = f"""<font size=8>
        <b>Symbol Reduction:</b> {_safe_float(report.get('symbol_reduction', 0)):.1f}%<br/>
        <b>Function Reduction:</b> {_safe_float(report.get('function_reduction', 0)):.1f}%<br/>
        <b>Entropy Increase:</b> {_safe_float(report.get('entropy_increase', 0)):.4f}
        </font>"""
        story.append(Paragraph(metrics_display, styles['Normal']))

    story.append(Spacer(1, 0.1*inch))

    # Research References
    story.append(Paragraph("<b>Framework References</b>", styles['Heading2']))
    story.append(Spacer(1, 0.05*inch))

    references_text = (
        "<font size=7><b>Standard:</b> OWASP MASTG (Potency, Resilience, Cost, Stealth)<br/>"
        "<b>Scale:</b> 0-100 (100 is practically impossible to achieve)<br/>"
        "<b>Rating:</b> 85+ Exceptional | 75+ Strong | 65+ Solid | 50+ Reasonable | &lt;50 Basic</font>"
    )
    story.append(Paragraph(references_text, styles['Normal']))
    story.append(Spacer(1, 0.05*inch))

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.read()
