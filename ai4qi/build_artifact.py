"""Build a copy of the web app that can be published as a private claude.ai Artifact.

    python3 build_artifact.py OUTDIR

Differences from app/: no document skeleton tags (the host adds them), AI4QI_EMBED mode
(template copied as CSV text instead of downloaded, no print button, no figures), audit
records in 1,000-record shards, no templates/ or figures/ folders, no service worker.
"""
import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app")
OUT = sys.argv[1]
SHARD = 1000

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "data", "audits"))
for f in ("app.js", "styles.css", "config.json", "export.js"):
    if os.path.exists(os.path.join(APP, f)):
        shutil.copy(os.path.join(APP, f), OUT)
for f in ("proposed.json", "cards.json", "standards.json", "version.json"):
    shutil.copy(os.path.join(APP, "data", f), os.path.join(OUT, "data"))
lib = json.load(open(os.path.join(APP, "data", "library.json"), encoding="utf-8"))
rows = []
for f in glob.glob(os.path.join(APP, "data", "audits", "*.json")):
    rows += json.load(open(f, encoding="utf-8"))


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "figures"}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o


shards = {}
for r in rows:
    shards.setdefault(r["id"] // SHARD, []).append(strip(r))
for k, v in shards.items():
    json.dump(v, open(os.path.join(OUT, "data", "audits", f"{k}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
lib["shard"] = SHARD
json.dump(lib, open(os.path.join(OUT, "data", "library.json"), "w", encoding="utf-8"),
          ensure_ascii=False, separators=(",", ":"))

html = open(os.path.join(APP, "index.html"), encoding="utf-8").read()
html = re.sub(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", html, flags=re.I)
html = re.sub(r'<meta charset[^>]*>\s*|<meta name="viewport"[^>]*>\s*|<link rel="(manifest|apple-touch-icon)"[^>]*>\s*', "", html)
html = html.replace('<script src="app.js"></script>', '<script>window.AI4QI_EMBED = true;</script>\n<script src="app.js"></script>')
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
n = sum(len(fs) for _, _, fs in os.walk(OUT))
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"{n} files, {size / 1e6:.1f} MB in {OUT}")
