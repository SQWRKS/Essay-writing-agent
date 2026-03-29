import json
import os
from app.models import Project


def export_project_txt(project: Project, output_dir: str) -> str:
    content = {}
    if project.content:
        try:
            content = json.loads(project.content)
        except Exception:
            content = {}

    lines = [
        f"Title: {project.title}",
        f"Topic: {project.topic}",
        f"Status: {project.status}",
        "",
        "=" * 60,
        "",
    ]

    sections = content.get("sections", {})
    for sec_key, sec_content in sections.items():
        lines.append(sec_key.replace("_", " ").title())
        lines.append("-" * 40)
        lines.append(sec_content)
        lines.append("")

    metadata = content.get("metadata", {})
    figures = metadata.get("figures", [])
    if figures:
        lines.append("Figures")
        lines.append("-" * 40)
        for idx, fig in enumerate(figures, 1):
            title = fig.get("title", f"Figure {idx}")
            description = fig.get("description", "")
            ref = fig.get("url") or fig.get("path") or ""
            lines.append(f"{idx}. {title}")
            if description:
                lines.append(f"   {description}")
            if ref:
                lines.append(f"   {ref}")
            lines.append("")

    bibliography = metadata.get("bibliography", "")
    if bibliography:
        lines.append("Bibliography")
        lines.append("-" * 40)
        lines.append(bibliography)
        lines.append("")

    text = "\n".join(lines)
    filename = f"project_{project.id[:8]}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath
