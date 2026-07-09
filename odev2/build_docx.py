from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    return text.strip()


def split_table_row(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [clean_inline(cell.strip()) for cell in line.split("|")]


def is_table_sep(line: str) -> bool:
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return bool(parts) and all(re.fullmatch(r":?-{3,}:?", p or "") for p in parts)


def set_cell_font(cell, size=7.2):
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(size)


def add_table(doc: Document, rows: list[str]) -> None:
    header = split_table_row(rows[0])
    body = [split_table_row(r) for r in rows[2:]]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.autofit = True
    for i, val in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = val
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(7.2)
    for row in body:
        cells = table.add_row().cells
        for i in range(len(header)):
            cells[i].text = row[i] if i < len(row) else ""
            set_cell_font(cells[i])
    doc.add_paragraph()


def add_image(doc: Document, md_dir: Path, line: str) -> None:
    match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line.strip())
    if not match:
        return
    img_path = (md_dir / match.group(1)).resolve()
    if not img_path.exists():
        doc.add_paragraph(f"[Gorsel bulunamadi: {img_path}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(img_path), width=Inches(4.6))


def add_paragraph(doc: Document, text: str) -> None:
    text = clean_inline(text)
    if text:
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(6)


def build_docx(md_path: Path, docx_path: Path) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(10)
    for name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)]:
        styles[name].font.name = "Calibri"
        styles[name].font.size = Pt(size)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    md_dir = md_path.parent
    i = 0
    in_code = False
    code_lines: list[str] = []
    para_lines: list[str] = []

    def flush_para():
        nonlocal para_lines
        if para_lines:
            add_paragraph(doc, " ".join(para_lines))
            para_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                code_lines = []
                in_code = False
            else:
                flush_para()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("#"):
            flush_para()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = clean_inline(stripped[level:].strip())
            doc.add_heading(title, level=min(level, 3))
            i += 1
            continue

        if stripped.startswith("![](") or stripped.startswith("!["):
            flush_para()
            add_image(doc, md_dir, stripped)
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            flush_para()
            table_rows = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_rows.append(lines[i])
                i += 1
            add_table(doc, table_rows)
            continue

        if stripped.startswith("- "):
            flush_para()
            p = doc.add_paragraph(clean_inline(stripped[2:]), style="List Bullet")
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        if re.match(r"\d+\.\s+", stripped):
            flush_para()
            item = re.sub(r"^\d+\.\s+", "", stripped)
            p = doc.add_paragraph(clean_inline(item), style="List Number")
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        para_lines.append(stripped)
        i += 1

    flush_para()
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(docx_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", default="odev2/PROJE_ILERLEME_RAPORU_2.md")
    parser.add_argument("--docx", default="odev2/PROJE_ILERLEME_RAPORU_2.docx")
    args = parser.parse_args()
    build_docx(Path(args.md), Path(args.docx))
    print(f"DOCX written: {args.docx}")


if __name__ == "__main__":
    main()
