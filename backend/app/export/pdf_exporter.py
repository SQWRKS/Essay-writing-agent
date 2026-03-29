import json
import os
from app.models import Project


def export_project_pdf(project: Project, output_dir: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    content = {}
    if project.content:
        try:
            content = json.loads(project.content)
        except Exception:
            content = {}

    filename = f"project_{project.id[:8]}.pdf"
    filepath = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=letter, topMargin=1 * inch, bottomMargin=1 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=11, leading=16, alignment=TA_JUSTIFY)

    story = [
        Paragraph(project.title, title_style),
        Spacer(1, 0.3 * inch),
        Paragraph(f"Topic: {project.topic}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]

    sections = content.get("sections", {})
    for sec_key, sec_content in sections.items():
        heading = sec_key.replace("_", " ").title()
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(heading, heading_style))
        for para in sec_content.split("\n\n"):
            if para.strip():
                safe_para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe_para, body_style))
                story.append(Spacer(1, 0.1 * inch))

    metadata = content.get("metadata", {})
    bibliography = metadata.get("bibliography", "")
    if bibliography:
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Bibliography", heading_style))
        for line in bibliography.split("\n"):
            if line.strip():
                safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe_line, body_style))
                story.append(Spacer(1, 0.05 * inch))

    doc.build(story)
    return filepath
