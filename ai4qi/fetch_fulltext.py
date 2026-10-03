"""Download PubMed Central full text (JATS XML) for screened-in papers that have no usable abstract.

    python3 fetch_fulltext.py [--pass nonortho]

Many conference abstracts (e.g. BJPsych Open supplements) are indexed in Europe PMC as articles
with full text but no abstract field. prepare_batches.py reads a paper whose full text is cached.
"""
import argparse
import os
import time
import urllib.request

import extract
import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pas", default="nonortho")
    a = ap.parse_args()
    keys, recs, _ = extract._todo(a.pas, 0)
    want = [recs[k]["pmcid"] for k in keys if recs[k].get("pmcid") and len(recs[k].get("abstract") or "") < 200]
    os.makedirs(pipeline.FT_DIR, exist_ok=True)
    got = 0
    for i, pmc in enumerate(want):
        out = os.path.join(pipeline.FT_DIR, f"{pmc}.xml")
        if os.path.exists(out):
            got += 1
            continue
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=40) as f:
                    data = f.read()
                if data.strip().startswith(b"<"):
                    with open(out, "wb") as g:
                        g.write(data)
                    got += 1
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.15)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(want)}: {got} full texts", flush=True)
    print(f"done: {got} of {len(want)} full texts cached")


if __name__ == "__main__":
    main()
