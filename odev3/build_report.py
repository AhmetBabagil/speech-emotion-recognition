'''Build Google-Docs-ready HTML and DOCX reports from real experiment outputs.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DISPLAY = {'cremad': 'CREMA-D', 'meld': 'MELD'}
DRIVE_URL = (
    'https://drive.google.com/drive/folders/'
    '1Hbp4WtCGFZjpvQCDxFmqtOmeqq-SMPvW?usp=sharing'
)
GITHUB_URL = (
    'https://github.com/AhmetBabagil/'
    'speech-emotion-recognition/tree/feat/speech-emotion-recognition'
)
DATASET_LINKS = (
    ('CREMA-D', 'https://github.com/CheyneyComputerScience/CREMA-D'),
    ('CREMA-D ses aynası', 'https://huggingface.co/datasets/AbstractTTS/CREMA-D'),
    ('MELD', 'https://github.com/declare-lab/MELD'),
    ('MELD ham veri', 'http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'),
)


@dataclass(frozen=True)
class ReportPaths:
    '''Files created for one diagnostic or final report build.'''

    html: Path
    docx: Path


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Paragraph:
    text: str


@dataclass(frozen=True)
class Notice:
    text: str


@dataclass(frozen=True)
class BulletList:
    items: tuple[str, ...]


@dataclass(frozen=True)
class LinkList:
    items: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TableBlock:
    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    caption: str | None = None


@dataclass(frozen=True)
class ImageBlock:
    path: Path
    alt_text: str
    caption: str


ReportBlock = Heading | Paragraph | Notice | BulletList | LinkList | TableBlock | ImageBlock


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return 'Evet' if value else 'Hayır'
    if isinstance(value, float):
        return f'{value:.4f}'
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '-'
    return str(value)


def _relative(target: str | Path, report_path: Path) -> str:
    return Path(os.path.relpath(Path(target), report_path.parent)).as_posix()


def _html_table(
    headers: Iterable[str],
    rows: Iterable[Iterable[Any]],
    caption: str | None = None,
) -> str:
    head = ''.join(f'<th>{escape(str(header))}</th>' for header in headers)
    body = []
    for row in rows:
        cells = ''.join(f'<td>{escape(_cell(value))}</td>' for value in row)
        body.append(f'<tr>{cells}</tr>')
    caption_html = f'<caption>{escape(caption)}</caption>' if caption else ''
    return (
        f'<table>{caption_html}<thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def _render_html(blocks: Iterable[ReportBlock], path: Path, title: str) -> Path:
    '''Render report blocks as a standalone UTF-8 HTML document.'''

    body: list[str] = []
    for block in blocks:
        if isinstance(block, Heading):
            level = min(max(block.level, 1), 4)
            body.append(f'<h{level}>{escape(block.text)}</h{level}>')
        elif isinstance(block, Paragraph):
            body.append(f'<p>{escape(block.text)}</p>')
        elif isinstance(block, Notice):
            body.append(f'<div class="notice">{escape(block.text)}</div>')
        elif isinstance(block, BulletList):
            items = ''.join(f'<li>{escape(item)}</li>' for item in block.items)
            body.append(f'<ul>{items}</ul>')
        elif isinstance(block, LinkList):
            items = ''.join(
                f'<li>{escape(label)}: '
                f'<a href="{escape(url, quote=True)}">{escape(url)}</a></li>'
                for label, url in block.items
            )
            body.append(f'<ul>{items}</ul>')
        elif isinstance(block, TableBlock):
            body.append(_html_table(block.headers, block.rows, block.caption))
        elif isinstance(block, ImageBlock):
            if block.path.is_file():
                source = escape(_relative(block.path, path), quote=True)
                body.append(
                    '<figure>'
                    f'<img src="{source}" alt="{escape(block.alt_text, quote=True)}">'
                    f'<figcaption>{escape(block.caption)}</figcaption>'
                    '</figure>'
                )
            else:
                body.append(
                    f'<div class="missing">Eksik görsel: {escape(str(block.path))}</div>'
                )
        else:
            raise TypeError(f'Unsupported report block: {type(block).__name__}')

    stylesheet = '''
        @page { size: A4; margin: 18mm; }
        body { font-family: Arial, sans-serif; line-height: 1.5; color: #172033;
               max-width: 1100px; margin: 0 auto; padding: 24px; }
        h1 { color: #123c69; border-bottom: 3px solid #2f80ed; padding-bottom: 8px; }
        h2 { color: #174f85; margin-top: 34px; border-bottom: 1px solid #b8cce4; }
        h3 { color: #2468a2; margin-top: 24px; }
        p, li { font-size: 10.5pt; }
        table { width: 100%; border-collapse: collapse; margin: 16px 0 24px;
                font-size: 8.5pt; page-break-inside: auto; }
        caption { text-align: left; font-weight: bold; margin-bottom: 7px; }
        th, td { border: 1px solid #8ca6bf; padding: 5px 6px; text-align: left; }
        th { background: #dceaf7; }
        tr:nth-child(even) td { background: #f6f9fc; }
        figure { text-align: center; page-break-inside: avoid; margin: 22px 0; }
        img { max-width: 92%; height: auto; }
        figcaption { font-size: 9pt; color: #475569; margin-top: 6px; }
        .notice { border-left: 5px solid #d97706; background: #fff7ed;
                  padding: 12px; margin: 16px 0; font-weight: bold; }
        .missing { color: #b91c1c; border: 1px dashed #b91c1c; padding: 8px; }
        a { color: #075ea8; }
    '''
    document = (
        '<!doctype html><html lang="tr"><head><meta charset="utf-8">'
        f'<title>{escape(title)}</title><style>{stylesheet}</style></head><body>'
        f'{"".join(body)}</body></html>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding='utf-8')
    return path


def _set_docx_cell_font(cell, size: float) -> None:
    from docx.shared import Pt

    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.name = 'Arial'
            run.font.size = Pt(size)


def _render_docx(blocks: Iterable[ReportBlock], path: Path, title: str) -> Path:
    '''Render the same report blocks as a self-contained Word document.'''

    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    document.core_properties.title = title
    document.core_properties.author = 'Ahmet Babagil'
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    document.styles['Normal'].font.name = 'Arial'
    document.styles['Normal'].font.size = Pt(10)
    for style_name, size in (
        ('Title', 18),
        ('Heading 1', 15),
        ('Heading 2', 13),
        ('Heading 3', 11),
    ):
        document.styles[style_name].font.name = 'Arial'
        document.styles[style_name].font.size = Pt(size)

    for block in blocks:
        if isinstance(block, Heading):
            if block.level == 1 and not document.paragraphs:
                document.add_heading(block.text, level=0)
            else:
                document.add_heading(block.text, level=min(max(block.level - 1, 1), 3))
        elif isinstance(block, Paragraph):
            paragraph = document.add_paragraph(block.text)
            paragraph.paragraph_format.space_after = Pt(6)
        elif isinstance(block, Notice):
            paragraph = document.add_paragraph()
            run = paragraph.add_run(block.text)
            run.bold = True
            run.font.color.rgb = RGBColor(180, 83, 9)
        elif isinstance(block, BulletList):
            for item in block.items:
                document.add_paragraph(item, style='List Bullet')
        elif isinstance(block, LinkList):
            for label, url in block.items:
                document.add_paragraph(f'{label}: {url}', style='List Bullet')
        elif isinstance(block, TableBlock):
            if block.caption:
                caption = document.add_paragraph()
                caption_run = caption.add_run(block.caption)
                caption_run.bold = True
            table = document.add_table(rows=1, cols=len(block.headers))
            table.style = 'Table Grid'
            font_size = 6.0 if len(block.headers) >= 10 else 8.0
            for index, header in enumerate(block.headers):
                cell = table.rows[0].cells[index]
                cell.text = str(header)
                for run in cell.paragraphs[0].runs:
                    run.bold = True
                _set_docx_cell_font(cell, font_size)
            for row in block.rows:
                cells = table.add_row().cells
                for index in range(len(block.headers)):
                    cells[index].text = _cell(row[index] if index < len(row) else None)
                    _set_docx_cell_font(cells[index], font_size)
            document.add_paragraph()
        elif isinstance(block, ImageBlock):
            if block.path.is_file():
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.add_run().add_picture(str(block.path), width=Inches(6.1))
                caption = document.add_paragraph(block.caption)
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if caption.runs:
                    caption.runs[0].italic = True
                    caption.runs[0].font.size = Pt(9)
            else:
                document.add_paragraph(f'Eksik görsel: {block.path}')
        else:
            raise TypeError(f'Unsupported report block: {type(block).__name__}')

    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    return path


def _report_paths(
    output_root: Path,
    diagnostic: bool,
    html_path: str | Path | None,
    docx_path: str | Path | None,
) -> ReportPaths:
    if diagnostic:
        default_html = output_root / 'RAPOR_DIAGNOSTIK.html'
        default_docx = output_root / 'RAPOR_DIAGNOSTIK.docx'
    else:
        default_html = Path('odev3/PROJE_ILERLEME_RAPORU_3.html')
        default_docx = Path('odev3/PROJE_ILERLEME_RAPORU_3.docx')
    return ReportPaths(
        html=Path(html_path) if html_path else default_html,
        docx=Path(docx_path) if docx_path else default_docx,
    )
