"""Convert JSON reports to PDF and Markdown formats on-the-fly."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from io import BytesIO

logger = logging.getLogger(__name__)


def json_to_markdown(report: Dict[str, Any]) -> str:
    """Convert JSON report to Markdown format."""
    md = "# LLVM Obfuscation Report\n\n"

    input_params = report.get("input_parameters", {})
    md += f"**Job ID:** {report.get('job_id', 'N/A')}\n"
    md += f"**Timestamp:** {input_params.get('timestamp', 'N/A')}\n"
    md += f"**Source File:** {input_params.get('source_file', 'N/A')}\n"
    md += f"**Platform:** {input_params.get('platform', 'N/A')}\n"
    md += f"**Obfuscation Level:** {input_params.get('obfuscation_level', 'N/A')}\n\n"

    # Baseline vs Obfuscated Metrics
    md += "## Baseline Metrics\n"
    baseline = report.get("baseline_metrics", {})
    md += f"- **File Size:** {baseline.get('file_size', 'N/A')} bytes\n"
    md += f"- **Functions:** {baseline.get('functions', 'N/A')}\n"
    md += f"- **Entropy:** {baseline.get('entropy', 'N/A'):.4f}\n\n"

    md += "## Obfuscated Metrics\n"
    obfuscated = report.get("output_attributes", {})
    md += f"- **File Size:** {obfuscated.get('file_size', 'N/A')} bytes\n"
    md += f"- **Functions:** {obfuscated.get('functions_count', 'N/A')}\n"
    md += f"- **Entropy:** {obfuscated.get('entropy', 'N/A'):.4f}\n\n"

    # Comparison
    md += "## Comparison\n"
    comparison = report.get("comparison", {})
    md += f"- **Size Change:** {comparison.get('size_change', 'N/A')} bytes ({comparison.get('size_change_percent', 'N/A')}%)\n"
    md += f"- **Entropy Increase:** {comparison.get('entropy_increase', 'N/A'):.4f}\n"
    md += f"- **RE Effort:** {report.get('estimated_re_effort', 'N/A')}\n\n"

    # Applied Passes
    md += "## Applied Obfuscation Passes\n"
    applied = report.get("input_parameters", {}).get("applied_passes", [])
    if applied:
        for pass_name in applied:
            md += f"- {pass_name}\n"
    else:
        md += "- None\n"

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


def json_to_pdf(report: Dict[str, Any]) -> bytes:
    """Convert JSON report to beautiful PDF format using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
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
    story.append(Paragraph("🛡️ LLVM Obfuscation Report", title_style))
    story.append(Spacer(1, 0.15*inch))

    # Job ID and Timestamp in small text
    job_id = report.get('job_id', 'N/A')
    input_params = report.get('input_parameters', {})
    timestamp = input_params.get('timestamp', 'N/A')
    story.append(Paragraph(f"<b>Job ID:</b> {job_id} | <b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))

    # Horizontal line separator
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e0e0e0')))
    story.append(Spacer(1, 0.2*inch))

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
        for warning in warnings[:3]:  # Show first 3 warnings
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
        story.append(Spacer(1, 0.15*inch))

    # Key Metrics Score Card (Large, prominent)
    score = float(report.get('obfuscation_score', 0)) if report.get('obfuscation_score') else 0
    score_emoji = get_score_emoji(score)

    score_grade = "Excellent Protection" if score >= 80 else "Good Protection" if score >= 60 else "Moderate Protection"

    score_data = [
        ['OBFUSCATION SCORE'],
        [f'{score:.1f}/100'],
        [f'{score_emoji}'],
        [score_grade]
    ]
    score_table = Table(score_data, colWidths=[7*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, 1), 36),
        ('FONTSIZE', (0, 2), (-1, 2), 28),
        ('FONTSIZE', (0, 3), (-1, 3), 13),
        ('PADDING', (0, 0), (-1, -1), 16),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('BORDER', (0, 0), (-1, -1), 2, colors.HexColor('#1f6feb')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.2*inch))

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
    story.append(Spacer(1, 0.2*inch))

    # Input Parameters
    story.append(Paragraph("<b>Input Parameters</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    input_data = [
        ['Source File', input_params.get('source_file', 'N/A')],
        ['Platform', input_params.get('platform', 'N/A')],
        ['Obfuscation Level', str(input_params.get('obfuscation_level', 'N/A'))],
        ['Requested Passes', ', '.join(input_params.get('requested_passes', []))],
        ['Applied Passes', ', '.join(input_params.get('applied_passes', []))],
    ]
    input_table = Table(input_data, colWidths=[2*inch, 5*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.15*inch))

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

    # Add page break
    story.append(PageBreak())

    # ============= PAGE 2: Before/After Comparison & Attributes =============

    story.append(Paragraph("Metrics Comparison & Output Details", styles['Title']))
    story.append(Spacer(1, 0.15*inch))

    # Before/After Comparison
    story.append(Paragraph("<b>Before/After Metrics</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    baseline_metrics = report.get('baseline_metrics', {})
    output_attrs = report.get('output_attributes', {})
    comparison = report.get('comparison', {})

    comparison_data = [
        ['Metric', 'Baseline', 'Obfuscated', 'Change'],
        ['File Size',
         format_bytes(baseline_metrics.get('file_size', 0)),
         format_bytes(output_attrs.get('file_size', 0)),
         format_percentage(comparison.get('size_change_percent', 0))],
        ['Symbols',
         str(baseline_metrics.get('symbols_count', 'N/A')),
         str(output_attrs.get('symbols_count', 'N/A')),
         f"-{comparison.get('symbols_removed_percent', 0):.1f}%"],
        ['Functions',
         str(baseline_metrics.get('functions_count', 'N/A')),
         str(output_attrs.get('functions_count', 'N/A')),
         f"-{comparison.get('functions_removed_percent', 0):.1f}%"],
        ['Entropy',
         f"{baseline_metrics.get('entropy', 0):.4f}",
         f"{output_attrs.get('entropy', 0):.4f}",
         f"+{comparison.get('entropy_increase', 0):.4f}"],
    ]
    comparison_table = Table(comparison_data, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 0.15*inch))

    # Output Attributes
    story.append(Paragraph("<b>Output File Attributes</b>", styles['Heading2']))
    story.append(Spacer(1, 0.08*inch))

    output_data = [
        ['File Size', format_bytes(output_attrs.get('file_size', 0))],
        ['Binary Format', output_attrs.get('binary_format', 'N/A')],
        ['Symbol Count', str(output_attrs.get('symbols_count', 'N/A'))],
        ['Function Count', str(output_attrs.get('functions_count', 'N/A'))],
        ['Binary Entropy', f"{output_attrs.get('entropy', 0):.4f}"],
        ['Obfuscation Methods', ', '.join(output_attrs.get('obfuscation_methods', []))],
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
    story.append(Spacer(1, 0.15*inch))

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
                ['Code Bloat %', f"{bogus_info.get('code_bloat_percentage', 0):.2f}%"],
            ]
        else:
            # String format - just display it
            story.append(Paragraph(str(bogus_info), styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
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
            story.append(Spacer(1, 0.15*inch))

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

            string_data = [
                ['Total Strings Found', f"{total_strings:,}"],
                ['Strings Encrypted', f"{encrypted_strings:,}"],
                ['Encryption Rate', f"{encryption_pct:.1f}%"],
                ['Method', method],
            ]
            string_table = Table(string_data, colWidths=[2.5*inch, 4.5*inch])
            string_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(string_table)
            story.append(Spacer(1, 0.15*inch))

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
            story.append(Spacer(1, 0.15*inch))

    # Cycles Completed
    cycles = report.get('cycles_completed', {})
    if cycles and cycles.get('total_cycles'):
        story.append(Paragraph("<b>Obfuscation Cycles</b>", styles['Heading2']))
        story.append(Spacer(1, 0.08*inch))

        cycles_per_cycle = cycles.get('per_cycle_metrics', [])
        cycles_data = [['Cycle', 'Passes Applied', 'Duration (ms)']]
        for cycle_info in cycles_per_cycle:
            cycles_data.append([
                str(cycle_info.get('cycle', 'N/A')),
                ', '.join(cycle_info.get('passes_applied', [])),
                str(cycle_info.get('duration_ms', 0))
            ])

        if len(cycles_data) > 1:
            cycles_table = Table(cycles_data, colWidths=[1*inch, 4.5*inch, 1.5*inch])
            cycles_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(cycles_table)

    # ============= PAGE 3: Advanced Metrics (if available) =============

    control_flow = report.get('control_flow_metrics')
    instruction_metrics = report.get('instruction_metrics')
    binary_structure = report.get('binary_structure')
    pattern_resistance = report.get('pattern_resistance')

    if any([control_flow, instruction_metrics, binary_structure, pattern_resistance]):
        story.append(PageBreak())
        story.append(Paragraph("Advanced Analysis Dashboard", styles['Title']))
        story.append(Spacer(1, 0.15*inch))

        # Control Flow Metrics
        if control_flow:
            story.append(Paragraph("<b>Control Flow Analysis</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            cf_baseline = control_flow.get('baseline', {})
            cf_obfuscated = control_flow.get('obfuscated', {})
            cf_comparison = control_flow.get('comparison', {})

            cf_data = [
                ['Metric', 'Baseline', 'Obfuscated', 'Change %'],
                ['Basic Blocks', str(cf_baseline.get('basic_blocks', 0)), str(cf_obfuscated.get('basic_blocks', 0)),
                 f"+{cf_comparison.get('basic_blocks_added', 0)}"],
                ['CFG Edges', str(cf_baseline.get('cfg_edges', 0)), str(cf_obfuscated.get('cfg_edges', 0)),
                 f"+{cf_comparison.get('cfg_edges_added', 0)}"],
                ['Cyclomatic Complexity', str(cf_baseline.get('cyclomatic_complexity', 0)),
                 str(cf_obfuscated.get('cyclomatic_complexity', 0)),
                 f"+{cf_comparison.get('complexity_increase_percent', 0):.1f}%"],
            ]
            cf_table = Table(cf_data, colWidths=[2*inch, 1.6*inch, 1.6*inch, 1.6*inch])
            cf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ebf5fb')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ]))
            story.append(cf_table)
            story.append(Spacer(1, 0.12*inch))

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
                 f"+{instr_comparison.get('instruction_growth_percent', 0):.1f}%"],
                ['Arithmetic Complexity', f"{instr_baseline.get('arithmetic_complexity_score', 0):.2f}",
                 f"{instr_obfuscated.get('arithmetic_complexity_score', 0):.2f}",
                 f"+{instr_comparison.get('arithmetic_complexity_increase', 0):.1f}"],
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
            story.append(Spacer(1, 0.12*inch))

        # Binary Structure
        if binary_structure:
            story.append(Paragraph("<b>Binary Structure Analysis</b>", styles['Heading2']))
            story.append(Spacer(1, 0.08*inch))

            binary_data = [
                ['Section Count', str(binary_structure.get('section_count', 0))],
                ['Import Table Entries', str(binary_structure.get('import_table', {}).get('imported_symbols', 0))],
                ['Relocation Count', str(binary_structure.get('relocations', {}).get('relocation_count', 0))],
                ['Code to Data Ratio', f"{binary_structure.get('code_to_data_ratio', 0):.2f}"],
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
                ['String Entropy', f"{pattern_resistance.get('string_analysis', {}).get('string_entropy', 0):.2f}"],
                ['Opcode Distribution Entropy', f"{pattern_resistance.get('code_analysis', {}).get('opcode_distribution_entropy', 0):.2f}"],
                ['Decompiler Confusion Score', f"{pattern_resistance.get('reverse_engineering_difficulty', {}).get('decompiler_confusion_score', 0):.1f}"],
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

    # ============= PAGE 4: Test Suite Results (if available) =============

    test_results = report.get('test_results')
    if test_results:
        story.append(PageBreak())
        story.append(Paragraph("Functional Correctness & Reliability Testing", styles['Title']))
        story.append(Spacer(1, 0.15*inch))

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
                ['Baseline Entropy', f"{binary_props.get('baseline_entropy', 0):.4f}"],
                ['Obfuscated Entropy', f"{binary_props.get('obf_entropy', 0):.4f}"],
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

            perf_data = [
                ['Baseline Time', f"{performance.get('baseline_ms', 0):.2f} ms"],
                ['Obfuscated Time', f"{performance.get('obf_ms', 0):.2f} ms"],
                ['Overhead', format_percentage(performance.get('overhead_percent', 0))],
                ['Acceptable', 'YES ✓' if performance.get('acceptable') else 'NO ✗'],
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

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.read()
