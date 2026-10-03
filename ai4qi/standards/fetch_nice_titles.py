"""Fetch the titles of every NICE guideline the library cites, for human-readable source names.

    python3 standards/fetch_nice_titles.py      # writes standards/nice_titles.json (cached; only new codes fetched)
"""
import html
import json
import re
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app" / "data"
OUT = HERE / "nice_titles.json"
CODE = re.compile(r"\bNICE\s+(NG|CG|QS|PH|NM|TA|DG|IPG|MTG|SC)\s?(\d+)", re.I)


def main():
    have = json.loads(OUT.read_text()) if OUT.exists() else {}
    codes = set()
    for f in ("proposed.json", "standards.json"):
        data = json.loads((APP / f).read_text())
        for x in data:
            st = x.get("standard", x)
            for m in CODE.finditer(st.get("source", "")):
                codes.add((m.group(1) + m.group(2)).lower())
    s = requests.Session()
    s.headers["User-Agent"] = "ai4qi-audit-library/1.0 (standard titles)"
    for c in sorted(codes - set(have)):
        try:
            r = s.get(f"https://www.nice.org.uk/guidance/{c}", timeout=30)
            m = re.search(r"<title>([^<]*)</title>", r.text)
            t = html.unescape(m.group(1)) if m else ""
            t = re.sub(r"^\s*Overview\s*\|\s*", "", t)
            t = re.sub(r"\s*\|\s*(Guidance|Quality standards?|Advice)\s*\|\s*NICE\s*$", "", t, flags=re.I).strip()
            if t and "NICE" != t:
                have[c] = t
                print(c, t)
        except requests.RequestException as e:
            print(c, "failed", e)
        time.sleep(0.5)
    OUT.write_text(json.dumps(dict(sorted(have.items())), indent=1, ensure_ascii=False))
    print(len(have), "titles")


if __name__ == "__main__":
    main()
