"""Harvest audit papers from sources beyond PubMed/Europe PMC and the named Crossref journals.

    python3 extra_harvest.py doaj            # DOAJ open-access journals, "audit" in the title, by year
    python3 extra_harvest.py s2              # Semantic Scholar bulk search for audit-cycle wording
    python3 extra_harvest.py crossref_all    # Crossref across all journals, audit query, by year
    python3 extra_harvest.py all

Records land in work/records.json under "DOI:<doi>" (or "S2:<id>" when there is no DOI) and in
work/hits_crossref.json, so `pipeline.py screen-crossref` screens them by content like other
conference and journal hits. Anything already in records.json by DOI, PMID or PMCID is skipped.
"""
import html
import re
import sys
import time

import requests

import pipeline
import sources

S = requests.Session()
S.headers["User-Agent"] = "ai4qi-audit-library/1.0 (clinical audit library research)"
YEARS = range(1990, 2027)


def _get(url, params, kind):
    for attempt in range(5):
        try:
            r = S.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            sources.log_failure(kind, str(params)[:120], r.status_code)
            return None
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
    sources.log_failure(kind, str(params)[:120], "retries exhausted")
    return None


def _clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _record(title, abstract, doi="", pmid="", pmcid="", journal="", year="", authors=(), affs=(),
            volume="", issue="", pages="", licence="", source=""):
    return {"pmid": pmid or "", "pmcid": pmcid or "", "doi": (doi or "").lower(), "title": _clean(title),
            "authors": list(authors), "journal": _clean(journal), "journal_abbrev": "", "year": str(year or ""),
            "volume": volume or "", "issue": issue or "", "pages": pages or "", "abstract": _clean(abstract),
            "pub_types": [f"journal article ({source})"], "affiliations": list(dict.fromkeys(a for a in affs if a)),
            "licence": licence, "open_access": bool(licence), "metadata_source": source, "s3_checked": True}


CLINICAL = re.compile(r"\bpatients?\b|hospital|\bwards?\b|clinical|nursing|surg|\bNHS\b|prescri|admission|"
                      r"discharge|theatre|anaesth|obstet|paediatr|pediatr|psychiatr|dental|infection|radiolog|patholog", re.I)
NOT_CLINICAL = re.compile(r"auditor|audit fee|financial|accounting|blockchain|internal audit function|tax|energy audit|"
                          r"information system|cyber|supply chain|firm", re.I)


class Store:
    def __init__(self):
        self.recs = pipeline.load("records.json", {})
        self.hits = pipeline.load("hits_crossref.json", {})
        self.seen = set()
        for r in self.recs.values():
            for f in ("doi", "pmid", "pmcid"):
                if r.get(f):
                    self.seen.add(f"{f}:{str(r[f]).lower()}")
        self.new = 0

    def add(self, key, rec, label):
        ids = [f"{f}:{str(rec[f]).lower()}" for f in ("doi", "pmid", "pmcid") if rec.get(f)]
        if any(i in self.seen for i in ids) or key in self.recs:
            return
        if len(rec["abstract"]) < 200 or not rec["title"]:
            return
        if not CLINICAL.search(rec["title"] + " " + rec["abstract"]) or NOT_CLINICAL.search(rec["title"]):
            return                                   # financial, IT and other non-clinical audits
        self.recs[key] = rec
        self.hits[key] = {"pmid": rec.get("pmid", ""), "labels": [label], "engines": [rec["metadata_source"]],
                          "journal": rec["journal"]}
        self.seen.update(ids)
        self.new += 1

    def save(self):
        pipeline.save("records.json", self.recs)
        pipeline.save("hits_crossref.json", self.hits)


def doaj(st):
    for y in YEARS:
        got, page = 0, 1
        while True:
            q = f'bibjson.title:(audit OR audits OR "re-audit" OR audited) AND bibjson.year:{y}'
            d = _get(f"https://doaj.org/api/search/articles/{requests.utils.quote(q)}",
                     {"pageSize": 100, "page": page}, "doaj")
            items = (d or {}).get("results") or []
            for it in items:
                b = it.get("bibjson", {})
                doi = next((i.get("id", "") for i in b.get("identifier", []) if i.get("type", "").lower() == "doi"), "")
                j = b.get("journal", {})
                rec = _record(b.get("title"), b.get("abstract"), doi=doi, journal=j.get("title"), year=b.get("year"),
                              authors=[a.get("name", "") for a in b.get("author", [])],
                              affs=[a.get("affiliation", "") for a in b.get("author", [])],
                              volume=j.get("volume", ""), issue=j.get("number", ""),
                              pages="-".join(x for x in (b.get("start_page"), b.get("end_page")) if x),
                              licence="cc by" if j.get("license") else "", source="DOAJ")
                key = f"DOI:{doi.lower()}" if doi else f"DOAJ:{it.get('id')}"
                st.add(key, rec, f"doaj {y}")
                got += 1
            if len(items) < 100 or page >= 10:          # DOAJ serves at most 1,000 results per query
                break
            page += 1
            time.sleep(0.5)
        print(f"  doaj {y}: {got} results, {st.new} new so far", flush=True)
    st.save()


S2_QUERIES = ['"clinical audit"', '"re-audit" | reaudit', '"audit cycle" | "closed loop audit" | "closed-loop audit"',
              'audit + ("quality improvement" | PDSA) + (guideline | standard | NICE)',
              'audit + (compliance | adherence) + (guideline | standard | protocol) + (hospital | ward | patients)']


def s2(st):
    for q in S2_QUERIES:
        token, got = None, 0
        while True:
            p = {"query": q, "fields": "title,abstract,externalIds,year,venue,journal,authors,publicationTypes,openAccessPdf"}
            if token:
                p["token"] = token
            d = _get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", p, "semanticscholar")
            if not d:
                break
            for it in d.get("data") or []:
                ex = it.get("externalIds") or {}
                if "Review" in (it.get("publicationTypes") or []):
                    continue
                j = it.get("journal") or {}
                rec = _record(it.get("title"), it.get("abstract"), doi=ex.get("DOI", ""), pmid=ex.get("PubMed", ""),
                              pmcid=("PMC" + str(ex["PubMedCentral"])) if ex.get("PubMedCentral") else "",
                              journal=j.get("name") or it.get("venue", ""), year=it.get("year"),
                              authors=[a.get("name", "") for a in it.get("authors") or []],
                              volume=j.get("volume", ""), pages=j.get("pages", ""),
                              licence="open access" if it.get("openAccessPdf") else "", source="Semantic Scholar")
                if rec["licence"] == "open access":
                    rec["licence"] = ""                  # a free PDF is not a reuse licence
                key = (f"PMID:{rec['pmid']}" if rec["pmid"] else f"DOI:{rec['doi']}" if rec["doi"]
                       else f"S2:{it.get('paperId')}")
                st.add(key, rec, "semantic scholar")
                got += 1
            token = d.get("token")
            print(f"  s2 {q[:40]}: {got} results, {st.new} new so far", flush=True)
            if not token:
                break
            time.sleep(1.2)                               # shared anonymous rate limit
        st.save()


def crossref_all(st):
    for y in YEARS:
        cursor, got = "*", 0
        for _ in range(3):                                # top 3,000 by relevance per year
            d = _get("https://api.crossref.org/works", {
                "query.bibliographic": "audit re-audit audit cycle closed loop audit clinical audit",
                "filter": f"has-abstract:true,type:journal-article,from-pub-date:{y},until-pub-date:{y}",
                "rows": 1000, "cursor": cursor,
                "select": "DOI,title,abstract,container-title,issued,author,volume,issue,page,license"}, "crossref-all")
            if not d:
                break
            m = d["message"]
            for it in m["items"]:
                title = (it.get("title") or [""])[0]
                abst = _clean(it.get("abstract"))
                if not re.search(r"\baudit(?!ory|ion|ive)", f"{title} {abst}", re.I):
                    continue
                lic = ""
                for l in it.get("license") or []:
                    mm = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)", l.get("URL", ""))
                    if mm:
                        lic = "cc0" if mm.group(2) == "zero" else "cc " + mm.group(2)
                        break
                rec = _record(title, abst, doi=it["DOI"], journal=(it.get("container-title") or [""])[0],
                              year=(it.get("issued", {}).get("date-parts") or [[""]])[0][0],
                              authors=[" ".join(x for x in (a.get("family"), (a.get("given") or "")[:1]) if x) for a in it.get("author", [])],
                              affs=[f.get("name", "") for a in it.get("author", []) for f in a.get("affiliation", [])],
                              volume=it.get("volume", ""), issue=it.get("issue", ""), pages=it.get("page", ""),
                              licence=lic, source="Crossref")
                st.add(f"DOI:{it['DOI'].lower()}", rec, f"crossref all {y}")
                got += 1
            if len(m["items"]) < 1000:
                break
            cursor = m["next-cursor"]
        print(f"  crossref all {y}: {got} audit-worded, {st.new} new so far", flush=True)
        if y % 5 == 0:
            st.save()
    st.save()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    st = Store()
    for name, fn in (("doaj", doaj), ("s2", s2), ("crossref_all", crossref_all)):
        if which in (name, "all"):
            fn(st)
            print(f"{name}: {st.new} new records in total", flush=True)
