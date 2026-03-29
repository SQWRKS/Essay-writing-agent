import json
import os
from app.models import Project


def export_project_docx(project: Project, output_dir: str) -> str:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    content = {}
    if project.content:
        try:
            content = json.loads(project.content)
        except Exception:
            content = {}

    doc = Document()

    title = doc.add_heading(project.title, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f"Topic: {project.topic}")
    doc.add_paragraph(f"Status: {project.status}")
    doc.add_paragraph()

    sections = content.get("sections", {})
    for sec_key, sec_content in sections.items():
        heading = sec_key.replace("_", " ").title()
        doc.add_heading(heading, level=1)
        for para in sec_content.split("\n\n"):
            if para.strip():
                p = doc.add_paragraph(para.strip())
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    metadata = content.get("metadata", {})
    bibliography = metadata.get("bibliography", "")
    if bibliography:
        doc.add_heading("Bibliography", level=1)
        for line in bibliography.split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip())

    filename = f"project_{project.id[:8]}.docx"
    filepath = os.path.join(output_dir, filename)
    doc.save(filepath)
    return filepath
