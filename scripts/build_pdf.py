"""Render docs/RAPOR.md to a submittable PDF (docs/RAPOR.pdf).

Pipeline: Markdown -> styled HTML -> PDF via headless Chrome/Edge
(`--print-to-pdf`). Chrome handles UTF-8 (Turkish characters) and embeds the
figures correctly. Falls back across Chrome/Edge install locations.

    python scripts/build_pdf.py
    python scripts/build_pdf.py --md docs/RAPOR.md --pdf docs/RAPOR.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Calibri", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 19pt; border-bottom: 2px solid #2a4d69; padding-bottom: 4px; color: #2a4d69; }
h2 { font-size: 14pt; color: #2a4d69; border-bottom: 1px solid #cdd9e5; padding-bottom: 2px;
     margin-top: 18px; }
h3 { font-size: 12pt; color: #34495e; margin-top: 14px; }
table { border-collapse: collapse; margin: 10px 0; width: auto; font-size: 9.8pt; }
th, td { border: 1px solid #b8c4d0; padding: 4px 9px; text-align: left; }
th { background: #eaf0f6; font-weight: 600; }
tr:nth-child(even) td { background: #f6f9fc; }
code { background: #f0f2f4; padding: 1px 4px; border-radius: 3px; font-size: 9.2pt;
       font-family: "Consolas", monospace; }
pre { background: #f6f8fa; padding: 8px 10px; border-radius: 5px; overflow-x: auto;
      border: 1px solid #e1e4e8; font-size: 9pt; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #7a9cc6; margin: 8px 0; padding: 2px 12px;
             color: #44515e; background: #f7fafd; }
img { max-width: 88%; display: block; margin: 8px auto; border: 1px solid #dde3ea; }
a { color: #1a5fb4; text-decoration: none; }
strong { color: #1a1a1a; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><style>{css}</style></head>
<body>{body}</body></html>"""


def find_browser() -> str | None:
    for name in ("chrome", "chrome.exe", "msedge", "msedge.exe"):
        p = shutil.which(name)
        if p:
            return p
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default="docs/RAPOR.md")
    ap.add_argument("--pdf", default="docs/RAPOR.pdf")
    args = ap.parse_args()

    md_path = Path(args.md).resolve()
    pdf_path = Path(args.pdf).resolve()
    docs_dir = md_path.parent  # so relative figures/ paths resolve

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )
    html = HTML_TEMPLATE.format(css=CSS, body=body)
    # Write the HTML next to the markdown so relative figures/ paths work.
    html_path = docs_dir / (md_path.stem + ".html")
    html_path.write_text(html, encoding="utf-8")

    browser = find_browser()
    if not browser:
        print("No Chrome/Edge found. HTML written to:", html_path,
              "\nOpen it and 'Print -> Save as PDF' manually.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser, "--headless", "--disable-gpu", "--no-sandbox",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        print("Rendering PDF via", Path(browser).name, "...")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0 or not pdf_path.exists():
            print("Browser print failed:", res.stderr[:400])
            print("HTML is at:", html_path, "-- you can print it manually.")
            sys.exit(1)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"PDF written: {pdf_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
