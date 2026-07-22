'''Tests for HTML and Word reports generated from experiment artifacts.'''

from pathlib import Path

from odev3.build_report import (
    Heading,
    LinkList,
    Paragraph,
    TableBlock,
    _render_docx,
    _render_html,
)


def test_html_renderer_preserves_turkish_text_and_escapes_content(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / 'report.html'
    blocks = [
        Heading(1, 'Ödev 3 – Türkçe Rapor'),
        Paragraph('MLP giriş boyutu 64×64 ve değer < 5000.'),
        LinkList((('Proje', 'https://example.com/?a=1&b=2'),)),
        TableBlock(
            ('Deney', 'Macro-F1'),
            ((1, 0.45678),),
            'Validation sonuçları',
        ),
    ]

    _render_html(blocks, output_path, 'Ödev 3')
    html = output_path.read_text(encoding='utf-8')

    assert '<meta charset="utf-8">' in html
    assert 'Ödev 3 – Türkçe Rapor' in html
    assert 'değer &lt; 5000' in html
    assert '0.4568' in html
    assert '<caption>Validation sonuçları</caption>' in html
    assert 'href="https://example.com/?a=1&amp;b=2"' in html


def test_docx_renderer_writes_headings_paragraphs_and_tables(tmp_path: Path) -> None:
    from docx import Document

    output_path = tmp_path / 'report.docx'
    blocks = [
        Heading(1, 'YAP 470 / BİL 570 – Proje İlerleme Raporu 3'),
        Paragraph('Konuşmadan duygu tanıma çalışması.'),
        TableBlock(
            ('Deney', 'Doğruluk', 'Macro-F1'),
            ((1, 0.61, 0.55),),
            'CREMA-D validation sonuçları',
        ),
    ]

    _render_docx(blocks, output_path, 'Proje İlerleme Raporu 3')
    document = Document(output_path)
    paragraph_text = '\n'.join(paragraph.text for paragraph in document.paragraphs)

    assert output_path.stat().st_size > 0
    assert 'Proje İlerleme Raporu 3' in paragraph_text
    assert 'Konuşmadan duygu tanıma çalışması.' in paragraph_text
    assert len(document.tables) == 1
    assert document.tables[0].cell(1, 2).text == '0.5500'
