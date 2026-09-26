"""Harvest conference-abstract audits from Crossref (abstract text deposited by publishers).

Why: most UK closed-loop audits are published only as meeting abstracts (ASiT/ASGBI in
BJS supplements, RCPsych in BJPsych Open, BGS in Age and Ageing, and so on), which PubMed
and Europe PMC do not index.  Crossref holds their DOI, title, authors and abstract.

    python3 crossref_harvest.py            # all journals in JOURNALS
    python3 crossref_harvest.py "British Journal of Surgery"

Records land in work/records.json under key "DOI:<doi>" with metadata_source "Crossref",
and in work/hits_crossref.json; the screen/extract/merge stages then treat them like any
other hit.  Licence comes from the Crossref licence field when present.
"""
import json
import re
import sys
import time

import requests

import pipeline
import sources

# (container title as Crossref has it, queries, first year)
JOURNALS = [
    ("Orthopaedic Proceedings", ["audit", "re-audit", "closed loop", "quality improvement"], 2012),
    ("The Bone & Joint Journal", ["audit", "re-audit", "quality improvement"], 2013),
    ("Bone & Joint Open", ["audit", "quality improvement"], 2020),
    ("British Journal of Surgery", ["closed loop audit", "re-audit", "audit cycle", "quality improvement audit", "audit"], 2015),
    ("BJPsych Open", ["audit", "re-audit", "quality improvement"], 2018),
    ("Age and Ageing", ["audit", "re-audit", "quality improvement"], 2014),
    ("Future Healthcare Journal", ["audit", "quality improvement"], 2014),
    ("Clinical Medicine", ["audit", "quality improvement"], 2014),
    ("Rheumatology", ["audit", "re-audit"], 2014),
    ("Irish Journal of Medical Science", ["audit", "re-audit", "closed loop"], 2012),
    ("British Journal of Dermatology", ["audit", "re-audit"], 2014),
    ("Archives of Disease in Childhood", ["audit", "re-audit", "quality improvement"], 2014),
    ("Emergency Medicine Journal", ["audit", "re-audit"], 2014),
    ("Thorax", ["audit", "re-audit"], 2014),
    ("Gut", ["audit", "re-audit"], 2014),
    ("Heart", ["audit", "re-audit"], 2014),
    ("BJA Open", ["audit"], 2022),
    ("British Journal of Anaesthesia", ["audit", "re-audit"], 2014),
    ("Anaesthesia", ["audit", "re-audit"], 2014),
    ("European Psychiatry", ["audit", "re-audit"], 2016),
    ("Journal of Clinical Urology", ["audit", "re-audit"], 2014),
    ("Clinical Oncology", ["audit", "re-audit"], 2014),
    ("Journal of Hospital Infection", ["audit", "re-audit"], 2014),
    ("BMJ Open Quality", ["audit"], 2014),
    ("BMJ Leader", ["audit"], 2017),
    ("British Journal of Oral and Maxillofacial Surgery", ["audit", "re-audit"], 2014),
    ("Journal of Neurology, Neurosurgery & Psychiatry", ["audit"], 2014),
    ("The Surgeon", ["audit"], 2010),
    ("Bulletin of the Royal College of Surgeons of England", ["audit"], 2010),
    ("Postgraduate Medical Journal", ["audit", "re-audit"], 2010),
]
MAX_PER_QUERY = 4000
S = requests.Session()
S.headers["User-Agent"] = "ai4qi-audit-library/1.0 (clinical audit library research)"


def fetch(container, query, year):
    cursor, out = "*", []
    while len(out) < MAX_PER_QUERY:
        r = None
        for attempt in range(4):
            try:
                r = S.get("https://api.crossref.org/works", timeout=90, params={
                    "query.bibliographic": query, "rows": 1000, "cursor": cursor,
                    "filter": f"container-title:{container},has-abstract:true,from-pub-date:{year}",
                    "select": "DOI,title,abstract,container-title,issued,author,volume,issue,page,license,type"})
                if r.status_code == 200:
                    break
            except requests.RequestException:
                r = None
            time.sleep(2 ** attempt)
        if r is None or r.status_code != 200:
            sources.log_failure("crossref", f"{container}|{query}", getattr(r, "status_code", "connection error"))
            break
        m = r.json()["message"]
        out += m["items"]
        if len(m["items"]) < 1000:
            break
        cursor = m["next-cursor"]
    return out


def to_record(it):
    abst = re.sub(r"<[^>]+>", " ", it.get("abstract") or "")
    abst = re.sub(r"\s+", " ", abst).strip()
    authors = [" ".join(x for x in (a.get("family"), (a.get("given") or "")[:1]) if x) for a in it.get("author", [])]
    affs = [aff.get("name", "") for a in it.get("author", []) for aff in a.get("affiliation", [])]
    lic = ""
    for l in it.get("license") or []:
        u = l.get("URL", "")
        m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)", u)
        if m:
            lic = "cc0" if m.group(2) == "zero" else "cc " + m.group(2).replace("-", "-")
            lic = lic.replace("cc by-", "cc by-") if lic != "cc0" else lic
            break
    return {
        "pmid": "", "pmcid": "", "doi": it["DOI"].lower(),
        "title": html_unescape((it.get("title") or [""])[0]), "authors": authors,
        "journal": html_unescape((it.get("container-title") or [""])[0]), "journal_abbrev": "",
        "year": str((it.get("issued", {}).get("date-parts") or [[""]])[0][0] or ""),
        "volume": it.get("volume", ""), "issue": it.get("issue", ""), "pages": it.get("page", ""),
        "abstract": html_unescape(abst), "pub_types": ["conference abstract or article (Crossref)"],
        "affiliations": list(dict.fromkeys(a for a in affs if a)),
        "licence": lic, "open_access": bool(lic), "metadata_source": "Crossref", "s3_checked": True,
    }


def html_unescape(s):
    import html
    return html.unescape(s or "")


def main(only=None):
    recs = pipeline.load("records.json", {})
    hits = pipeline.load("hits_crossref.json", {})
    for container, queries, year in JOURNALS:
        if only and container != only:
            continue
        got = {}
        for q in queries:
            for it in fetch(container, q, year):
                got[it["DOI"].lower()] = it
        new = 0
        for doi, it in got.items():
            k = f"DOI:{doi}"
            h = hits.setdefault(k, {"pmid": "", "labels": [], "engines": ["crossref"], "journal": container})
            if k not in recs:
                recs[k] = to_record(it)
                new += 1
        print(f"  {container:52s} {len(got):5d} records ({new} new)", flush=True)
        pipeline.save("records.json", recs)
        pipeline.save("hits_crossref.json", hits)
    print(f"crossref: {len(hits)} hits in total")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
