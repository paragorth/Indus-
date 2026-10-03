"""Render PAPER-DRAFT.md to docs/Garg_2026_Indus_seals_credentials.pdf.

Pipeline: python-markdown (tables, sane lists) -> HTML with a print stylesheet ->
headless Chromium via Playwright (same browser build as tools/pdf_page_shot.py), A4.

Usage: python3 tools/build_paper_pdf.py [input.md] [output.pdf]
"""
import os, sys, pathlib
import markdown
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'PAPER-DRAFT.md'
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'docs' / 'Garg_2026_Indus_seals_credentials.pdf'
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'DejaVu Serif', 'Liberation Serif', Georgia, serif; font-size: 10.2pt; line-height: 1.38; color: #111; }
h1 { font-size: 16pt; line-height: 1.25; margin: 0 0 8pt 0; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt 0; border-bottom: 0.5pt solid #999; padding-bottom: 2pt; }
h3 { font-size: 10.8pt; margin: 12pt 0 4pt 0; }
p { margin: 0 0 6pt 0; text-align: justify; }
ul { margin: 0 0 6pt 0; padding-left: 16pt; }
li { margin-bottom: 2.5pt; text-align: justify; }
em { font-style: italic; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; font-size: 9.2pt; }
th, td { border: 0.5pt solid #777; padding: 2pt 5pt; text-align: left; }
th { background: #eee; }
a { color: #111; text-decoration: none; }
h1 + p em { font-size: 9.2pt; color: #333; }
"""

def main():
    md = SRC.read_text(encoding='utf-8')
    body = markdown.markdown(md, extensions=['tables', 'sane_lists'])
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>PAPER-DRAFT</title><style>{CSS}</style></head><body>{body}</body></html>'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        kw = {'args': ['--no-sandbox']}
        if os.path.exists(CHROME):
            kw['executable_path'] = CHROME
        b = p.chromium.launch(**kw)
        pg = b.new_page()
        pg.set_content(html, wait_until='load')
        pg.pdf(path=str(OUT), format='A4', print_background=True,
               margin={'top': '20mm', 'bottom': '20mm', 'left': '18mm', 'right': '18mm'})
        b.close()
    print('wrote', OUT, OUT.stat().st_size, 'bytes')

if __name__ == '__main__':
    main()
