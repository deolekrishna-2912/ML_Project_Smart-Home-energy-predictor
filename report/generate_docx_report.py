"""
generate_docx_report.py — Smart Home Appliance Energy Predictor (Phase 5)
========================================================================
Converts the structured academic markdown report (`report/report_draft.md`)
into a professionally formatted and styled Microsoft Word (.docx) document
(`report/mini_project_report.docx`) using python-docx.

Includes:
- Professional academic typography (Calibri, navy headings, 1-inch margins)
- Styled tables with dark navy header rows and alternating row borders
- Embedded high-resolution figures from figures/ with academic captions
- Callout styling for Executive Summary and Viva Q&A blocks
- Mathematical notation and formatted bulleted/numbered lists
"""

import os
import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "report"
FIGURES_DIR = BASE_DIR / "figures"
MARKDOWN_PATH = REPORT_DIR / "report_draft.md"
DOCX_OUTPUT_PATH = REPORT_DIR / "mini_project_report.docx"

# Palette
PRIMARY_COLOR = RGBColor(26, 54, 93)      # Deep Navy
SECONDARY_COLOR = RGBColor(43, 84, 126)   # Slate Navy
BODY_COLOR = RGBColor(40, 44, 52)         # Off-black / Charcoal
MUTED_COLOR = RGBColor(100, 110, 125)     # Slate Gray
ACCENT_BLUE = RGBColor(0, 102, 204)       # Link / Highlight Blue


def set_cell_background(cell, fill_hex):
    """Set background fill color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner padding for table cells."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_formatted_runs(paragraph, text, base_font_size=11, is_italic=False, text_color=BODY_COLOR):
    """Parses markdown bold **text** and regular text into styled runs."""
    tokens = re.split(r'(\*\*.*?\*\*)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.size = Pt(base_font_size)
            run.font.color.rgb = text_color
            run.font.name = "Calibri"
        else:
            run = paragraph.add_run(token)
            run.bold = False
            run.font.size = Pt(base_font_size)
            run.font.italic = is_italic
            run.font.color.rgb = text_color
            run.font.name = "Calibri"


def create_styled_table(doc, headers, rows):
    """Creates an academically formatted table."""
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], "1A365D")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)

    # Data Rows
    for r_idx, row in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_fill = "F7FAFC" if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_formatted_runs(p, str(val).strip(), base_font_size=9.5)
            set_cell_background(row_cells[c_idx], bg_fill)
            set_cell_margins(row_cells[c_idx], top=90, bottom=90, left=120, right=120)

    # Add space after table
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(8)


def add_image_if_exists(doc, filename, caption_text, width_inches=5.8):
    """Embeds an image from the figures directory if it exists."""
    img_path = FIGURES_DIR / filename
    if img_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(img_path), width=Inches(width_inches))

        p_caption = doc.add_paragraph()
        p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_caption.paragraph_format.space_before = Pt(2)
        p_caption.paragraph_format.space_after = Pt(10)
        run = p_caption.add_run(f"Figure: {caption_text}")
        run.italic = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = MUTED_COLOR
        run.font.name = "Calibri"


def build_academic_report():
    print(f"Reading markdown source: {MARKDOWN_PATH}")
    with open(MARKDOWN_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = docx.Document()

    # Configure Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = BODY_COLOR

    # Parse lines
    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_lines = []
    in_mermaid = False

    while i < len(lines):
        line = lines[i].strip()

        # Skip mermaid diagrams
        if line.startswith("```mermaid"):
            in_mermaid = True
            i += 1
            continue
        if in_mermaid:
            if line.startswith("```"):
                in_mermaid = False
            i += 1
            continue

        # Handle Tables
        if line.startswith("|") and line.endswith("|"):
            table_lines.append(line)
            i += 1
            continue
        elif table_lines:
            # Process accumulated table
            if len(table_lines) >= 3:
                raw_headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                raw_rows = []
                for row_line in table_lines[2:]:
                    cells = [c.strip() for c in row_line.split("|")[1:-1]]
                    raw_rows.append(cells)
                create_styled_table(doc, raw_headers, raw_rows)
            table_lines = []

        if not line:
            i += 1
            continue

        # Document Main Title (# )
        if line.startswith("# ") and not line.startswith("## "):
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(title_text)
            run.bold = True
            run.font.size = Pt(22)
            run.font.color.rgb = PRIMARY_COLOR
            run.font.name = "Calibri"

        # Subtitle (## )
        elif line.startswith("## "):
            subtitle_text = line[3:].strip()
            # If Section 1 to 9 or Executive Summary
            is_main_sec = bool(re.match(r'^(Section \d|Executive Summary)', subtitle_text))
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(subtitle_text)
            run.bold = True
            run.font.size = Pt(15 if is_main_sec else 13)
            run.font.color.rgb = PRIMARY_COLOR if is_main_sec else SECONDARY_COLOR
            run.font.name = "Calibri"

            # Contextual Figure Injections under specific sections
            if "Section 4" in subtitle_text:
                add_image_if_exists(doc, "fig1_target_distribution.png", "Distribution of Target Energy Consumption (Appliances Wh)")
                add_image_if_exists(doc, "fig2_correlation_heatmap.png", "Environmental Telemetry Correlation Heatmap")
                add_image_if_exists(doc, "fig3_time_series_trend.png", "Time Series Energy Consumption and Environmental Trends")
                add_image_if_exists(doc, "fig4_feature_relationships.png", "Feature Relationships and Appliance Consumption Correlations")
            elif "Section 6" in subtitle_text:
                add_image_if_exists(doc, "fig5_mae_comparison.png", "Test Set Mean Absolute Error (MAE in Wh) Across 4 Models")
                add_image_if_exists(doc, "fig6_rmse_comparison.png", "Test Set Root Mean Squared Error (RMSE in Wh) Across 4 Models")
                add_image_if_exists(doc, "fig7_actual_vs_predicted.png", "Parity Scatter Plot: Actual vs. Predicted Wh (HistGradientBoosting)")

        # Sub-heading (### )
        elif line.startswith("### "):
            subhead_text = line[4:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(subhead_text)
            run.bold = True
            run.font.size = Pt(12)
            run.font.color.rgb = SECONDARY_COLOR
            run.font.name = "Calibri"

        # Horizontal Divider
        elif line == "---":
            # Add subtle divider
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)

        # Bullet List (- or *)
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_formatted_runs(p, bullet_text, base_font_size=10.5)

        # Numbered List (1. , 2. )
        elif re.match(r'^\d+\.\s', line):
            num_match = re.match(r'^\d+\.\s', line)
            list_text = line[num_match.end():].strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_formatted_runs(p, list_text, base_font_size=10.5)

        # Q&A Viva Blocks
        elif line.startswith("**Q") or line.startswith("### Q"):
            q_clean = line.replace("### ", "").replace("**", "")
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(f"📌 {q_clean}")
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = PRIMARY_COLOR

        elif line.startswith("**Answer:**"):
            ans_clean = line[11:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            run_lbl = p.add_run("Answer: ")
            run_lbl.bold = True
            run_lbl.font.color.rgb = SECONDARY_COLOR
            add_formatted_runs(p, ans_clean, base_font_size=10.5)

        # Regular Body Paragraphs
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, line, base_font_size=10.5)

        i += 1

    # Check if dangling table
    if table_lines and len(table_lines) >= 3:
        raw_headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
        raw_rows = []
        for row_line in table_lines[2:]:
            cells = [c.strip() for c in row_line.split("|")[1:-1]]
            raw_rows.append(cells)
        create_styled_table(doc, raw_headers, raw_rows)

    doc.save(str(DOCX_OUTPUT_PATH))
    print(f"Academic report generated successfully: {DOCX_OUTPUT_PATH}")


if __name__ == "__main__":
    build_academic_report()
