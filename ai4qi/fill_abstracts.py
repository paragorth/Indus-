"""Fill in missing abstracts for screened-in papers, from Europe PMC, then re-screen them.

    python3 fill_abstracts.py [--pass nonortho] [--limit 0]

Semantic Scholar and Crossref often return a title with no abstract, so prepare_batches.py skips the
paper (title-only records carry no results). This looks each one up in Europe PMC by PMID or DOI,
stores the abstract (and PMCID, journal, year when missing), and re-applies the screening rules.
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request

import classify
import extract
import pipeline

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def lookup(rec):
    if rec.get("pmcid"):
        q = f"PMCID:{rec['pmcid']}"
    elif rec.get("pmid"):
        q = f"EXT_ID:{rec['pmid']} AND SRC:MED"
    elif rec.get("doi"):
        q = f'DOI:"{rec["doi"]}"'
    else:
        return None
    url = EPMC + "?" + urllib.parse.urlencode({"query": q, "resultType": "core", "format": "json", "pageSize": 1})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as f:
                d = json.load(f)
            res = (d.get("resultList") or {}).get("result") or []
            return res[0] if res else None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pas", default="nonortho")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    keys, recs, _ = extract._todo(a.pas, 0)
    keys = [k for k in keys if len(recs[k].get("abstract") or "") < 200 and (recs[k].get("pmid") or recs[k].get("doi") or recs[k].get("pmcid"))]
    if a.limit:
        keys = keys[:a.limit]
    allrecs = pipeline.load("records.json", {})
    scr = pipeline.load("screen.json", {})
    filled = kept = 0
    for i, k in enumerate(keys):
        r = allrecs.get(k)
        if not r:
            continue
        hit = lookup(r)
        time.sleep(0.15)
        abst = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", (hit or {}).get("abstractText") or "")).strip()
        if len(abst) < 200:
            continue
        r["abstract"] = abst
        for src, dst in (("pmcid", "pmcid"), ("pmid", "pmid")):
            if hit.get(src) and not r.get(dst):
                r[dst] = hit[src]
        if not r.get("journal") and (hit.get("journalInfo") or {}).get("journal"):
            r["journal"] = hit["journalInfo"]["journal"].get("title", "")
        if not r.get("year") and hit.get("pubYear"):
            r["year"] = hit["pubYear"]
        filled += 1
        keep, why, kind = classify.screen(r)
        old = scr.get(k, {})
        scr[k] = dict(old, keep=keep, reason=why, audit_kind=kind)
        kept += keep
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(keys)} looked up, {filled} abstracts found, {kept} still kept", flush=True)
    pipeline.save("records.json", allrecs)
    pipeline.save("screen.json", scr)
    print(f"done: {len(keys)} looked up, {filled} abstracts found, {kept} kept after re-screening")


if __name__ == "__main__":
    main()
