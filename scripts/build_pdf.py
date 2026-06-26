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
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Calibri", "Helvetica Neue", Arial, sans-serif;
       font-size: 10.6pt; line-height: 1.5; color: #20262e; max-width: 100%;
       text-rendering: optimizeLegibility; -webkit-font-smoothing: antialiased; }
p { margin: 6px 0 9px; text-align: justify; }
h1 { font-size: 21pt; color: #1f3b57; margin: 0 0 4px; letter-spacing: -0.2px;
     border-bottom: 3px solid #1f3b57; padding-bottom: 8px; }
h2 { font-size: 14.5pt; color: #1f3b57; margin: 22px 0 8px; padding: 4px 0 4px 10px;
     border-left: 4px solid #2f74c0; background: linear-gradient(90deg,#eef4fb,transparent); }
h3 { font-size: 11.8pt; color: #2c4a66; margin: 15px 0 6px; }
hr { border: none; border-top: 1px solid #dfe6ee; margin: 16px 0; }
ul { margin: 6px 0 10px; padding-left: 20px; }
li { margin: 3px 0; }
table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 9.7pt;
        box-shadow: 0 1px 2px rgba(31,59,87,0.06); }
th, td { border: 1px solid #c9d6e3; padding: 6px 10px; text-align: left; }
th { background: #1f3b57; color: #fff; font-weight: 600; border-color: #1f3b57; }
tr:nth-child(even) td { background: #f3f7fb; }
td:nth-child(n+2), th:nth-child(n+2) { text-align: center; }
code { background: #eef1f4; padding: 1px 5px; border-radius: 3px; font-size: 9.1pt;
       font-family: "Consolas", "Courier New", monospace; color: #344; }
pre { background: #f6f8fa; padding: 9px 11px; border-radius: 6px; overflow-x: auto;
      border: 1px solid #e1e4e8; font-size: 8.9pt; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #8fb3d9; margin: 10px 0; padding: 6px 14px;
             color: #3c4a59; background: #f5f9fd; border-radius: 0 4px 4px 0; }
img { max-width: 62%; display: block; margin: 14px auto 4px;
      border: 1px solid #d4dde7; border-radius: 4px; box-shadow: 0 2px 6px rgba(31,59,87,0.12); }
/* caption = the paragraph right after an image paragraph */
p:has(> img) + p { text-align: center; font-size: 9pt; color: #5b6b7a; margin-top: 0; }
p:has(> img) + p em { font-style: italic; }
a { color: #1a5fb4; text-decoration: none; }
strong { color: #16202b; }
h2 + p, h3 + p { margin-top: 4px; }
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
