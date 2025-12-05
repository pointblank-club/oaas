"""Convert JSON reports to PDF and Markdown formats on-the-fly."""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


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
    obfuscated = report.get("obfuscated_metrics", {})
    md += f"- **File Size:** {obfuscated.get('file_size', 'N/A')} bytes\n"
    md += f"- **Functions:** {obfuscated.get('functions', 'N/A')}\n"
    md += f"- **Entropy:** {obfuscated.get('entropy', 'N/A'):.4f}\n\n"

    # Comparison
    md += "## Comparison\n"
    comparison = report.get("comparison_metrics", {})
    md += f"- **Size Change:** {comparison.get('size_change', 'N/A')} bytes ({comparison.get('size_change_percent', 'N/A')}%)\n"
    md += f"- **Entropy Increase:** {comparison.get('entropy_increase', 'N/A'):.4f}\n"
    md += f"- **RE Effort:** {comparison.get('estimated_re_effort', 'N/A')}\n\n"

    # Applied Passes
    md += "## Applied Obfuscation Passes\n"
    applied = report.get("applied_passes", [])
    if applied:
        for pass_name in applied:
            md += f"- {pass_name}\n"
    else:
        md += "- None\n"

    md += "\n---\n"
    return md


def json_to_pdf(report: Dict[str, Any]) -> bytes:
    """Convert JSON report to PDF format using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from io import BytesIO
    except ImportError:
        # Fallback to text if reportlab not available
        md = json_to_markdown(report)
        return md.encode('utf-8')

    # Create PDF in memory
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=1*inch, bottomMargin=0.75*inch)

    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("LLVM Obfuscation Report", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Basic Info
    input_params = report.get("input_parameters", {})
    info_data = [
        ["Job ID:", report.get('job_id', 'N/A')],
        ["Timestamp:", input_params.get('timestamp', 'N/A')],
        ["Source File:", input_params.get('source_file', 'N/A')],
        ["Platform:", input_params.get('platform', 'N/A')],
        ["Level:", str(input_params.get('obfuscation_level', 'N/A'))],
    ]

    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8eef7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))

    # Metrics Comparison
    story.append(Paragraph("Metrics Comparison", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))

    baseline = report.get("baseline_metrics", {})
    obfuscated = report.get("obfuscated_metrics", {})
    comparison = report.get("comparison_metrics", {})

    metrics_data = [
        ["Metric", "Baseline", "Obfuscated", "Change"],
        ["File Size (bytes)",
         str(baseline.get('file_size', 'N/A')),
         str(obfuscated.get('file_size', 'N/A')),
         str(comparison.get('size_change', 'N/A'))],
        ["Functions",
         str(baseline.get('functions', 'N/A')),
         str(obfuscated.get('functions', 'N/A')),
         str(comparison.get('function_reduction', 'N/A'))],
        ["Entropy",
         f"{baseline.get('entropy', 0):.4f}",
         f"{obfuscated.get('entropy', 0):.4f}",
         f"{comparison.get('entropy_increase', 0):.4f}"],
    ]

    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # Applied Passes
    story.append(Paragraph("Applied Obfuscation Passes", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    applied = report.get("applied_passes", [])
    if applied:
        for pass_name in applied:
            story.append(Paragraph(f"• {pass_name}", styles['Normal']))
    else:
        story.append(Paragraph("None", styles['Normal']))

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.read()


def load_json_report(json_path: Path) -> Dict[str, Any]:
    """Load JSON report from file."""
    with open(json_path, 'r') as f:
        return json.load(f)
