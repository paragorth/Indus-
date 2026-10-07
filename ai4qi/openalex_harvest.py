"""Harvest clinical audit papers from OpenAlex (about 250 million works, including many journals outside
PubMed and Europe PMC).

    python3 openalex_harvest.py [first_year] [last_year]

Uses the free public API. If an environment variable OPENALEX_API_KEY is set it is sent as api_key;
otherwise requests are anonymous and slowed down to respect OpenAlex's rate limits (it waits and
retries on 429/503). Records go into work/records.json and work/hits_crossref.json through the same
Store as extra_harvest.py (clinical-wording filter, skip anything already held by DOI/PMID/PMCID), so
`pipeline.py screen-crossref` screens them like other hits.
"""
import os
import re
import sys
import time

import requests

from extra_harvest import Store, _record, _clean

S = requests.Session()
S.headers["User-Agent"] = "ai4qi-audit-library/1.0 (clinical audit library research)"
KEY = os.environ.get("OPENALEX_API_KEY", "")
QUERIES = [
    ("title audit", "title.search:audit|audits|audited|re-audit"),
    ("abstract audit cycle", 'abstract.search:"re-audit"|"audit cycle"|"closed loop audit"|"clinical audit"'),
]


def get(params):
    if KEY:
        params = dict(params, api_key=KEY)
    for attempt in range(12):
        try:
            r = S.get("https://api.openalex.org/works", params=params, timeout=90)
        except requests.RequestException:
            time.sleep(10)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 503, 502, 500):
            m = re.search(r"retry in (\d+)s", r.text)
            time.sleep(int(m.group(1)) + 2 if m else min(300, 15 * (attempt + 1)))
            continue
        return None
    return None


def abstract(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))


def main(first=1990, last=2026):
    st = Store()
    for y in range(first, last + 1):
        for label, flt in QUERIES:
            cursor, got = "*", 0
            while cursor:
                d = get({"filter": f"{flt},publication_year:{y},has_abstract:true", "per-page": 200, "cursor": cursor,
                         "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location,authorships,ids,type,biblio,best_oa_location"})
                if not d:
                    break
                for w in d.get("results") or []:
                    if w.get("type") in ("review", "editorial", "erratum", "letter", "paratext"):
                        continue
                    title = w.get("title") or ""
                    if re.search(r"\bauditory\b", title, re.I):
                        continue
                    ids = w.get("ids") or {}
                    doi = (w.get("doi") or "").replace("https://doi.org/", "")
                    pmid = (ids.get("pmid") or "").rsplit("/", 1)[-1]
                    pmcid = (ids.get("pmcid") or "").rsplit("/", 1)[-1]
                    src = ((w.get("primary_location") or {}).get("source") or {})
                    lic = ((w.get("best_oa_location") or {}).get("license") or "").replace("-", " ")
                    b = w.get("biblio") or {}
                    rec = _record(title, abstract(w.get("abstract_inverted_index")), doi=doi, pmid=pmid,
                                  pmcid=pmcid if pmcid.upper().startswith("PMC") else "",
                                  journal=src.get("display_name") or "", year=w.get("publication_year"),
                                  authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
                                  affs=[i.get("display_name", "") for a in w.get("authorships") or [] for i in a.get("institutions") or []],
                                  volume=b.get("volume") or "", issue=b.get("issue") or "",
                                  pages="-".join(x for x in (b.get("first_page"), b.get("last_page")) if x),
                                  licence=lic if lic.startswith("cc") else "", source="OpenAlex")
                    key = (f"PMID:{pmid}" if pmid else f"DOI:{doi.lower()}" if doi else "OA:" + w["id"].rsplit("/", 1)[-1])
                    st.add(key, rec, f"openalex {label} {y}")
                    got += 1
                cursor = (d.get("meta") or {}).get("next_cursor")
                time.sleep(1.0 if KEY else 3.0)
            print(f"  {y} {label}: {got} results, {st.new} new so far", flush=True)
        st.save()
    print(f"openalex: {st.new} new records in total", flush=True)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*a) if a else main()
