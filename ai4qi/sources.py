"""HTTP access to Europe PMC, PubMed E-utilities and the PMC open-access bucket.

Every network failure is logged to work/failures.jsonl with the reason, so the
final report can say exactly what could not be fetched and why.  Only these
public APIs are contacted; publisher websites are never scraped, so there is
nothing to "retry" against a blocked publisher.
"""
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
TOOL = "ai4qi-audit-library"

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
os.makedirs(WORK, exist_ok=True)
_fail_lock = threading.Lock()


def log_failure(kind, ident, reason):
    with _fail_lock, open(os.path.join(WORK, "failures.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "id": ident, "reason": str(reason)[:300],
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


class Limiter:
    """Minimum spacing between calls (NCBI allows 3 requests/second without a key)."""

    def __init__(self, per_second):
        self.gap = 1.0 / per_second
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = self.last + self.gap - now
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


ncbi_limit = Limiter(2.8)
epmc_limit = Limiter(8)
s3_limit = Limiter(10)
session = requests.Session()
session.headers["User-Agent"] = f"{TOOL}/1.0 (clinical audit library research)"


def get(url, limiter, params=None, data=None, tries=4, kind="http", ident=None, binary=False):
    """GET/POST with backoff on 429/5xx and connection errors.  4xx other than 429
    is treated as final (not found / not allowed) and never retried."""
    last = None
    for attempt in range(tries):
        limiter.wait()
        try:
            if data is not None:
                r = session.post(url, data=data, timeout=60)
            else:
                r = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            last = f"connection error: {e.__class__.__name__}"
        else:
            if r.status_code == 200:
                return r.content if binary else r.text
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        time.sleep(2 ** attempt)
    log_failure(kind, ident or url, last)
    return None


# ---------------------------------------------------------------- Europe PMC

def epmc_available():
    t = get(f"{EPMC}/search", epmc_limit, {"query": "audit", "format": "json", "pageSize": 1},
            tries=3, kind="epmc-health")
    return t is not None


def epmc_search(query, max_results=5000):
    """All core records for a query (cursorMark paging).  Returns list or None if down."""
    out, cursor = [], "*"
    while True:
        t = get(f"{EPMC}/search", epmc_limit,
                {"query": query, "format": "json", "resultType": "core", "pageSize": 1000,
                 "cursorMark": cursor}, kind="epmc-search", ident=query[:120])
        if t is None:
            return out or None
        d = json.loads(t)
        res = d.get("resultList", {}).get("result", [])
        out.extend(res)
        nxt = d.get("nextCursorMark")
        if not res or not nxt or nxt == cursor or len(out) >= max_results:
            return out
        cursor = nxt


def epmc_by_pmids(pmids):
    """Core records for PubMed IDs, 100 per query."""
    out = {}
    pmids = list(pmids)
    for i in range(0, len(pmids), 100):
        chunk = pmids[i:i + 100]
        q = "SRC:MED AND (" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ")"
        for r in epmc_search(q) or []:
            if r.get("pmid"):
                out[r["pmid"]] = r
    return out


def epmc_fulltext(pmcid):
    return get(f"{EPMC}/{pmcid}/fullTextXML", epmc_limit, kind="epmc-fulltext", ident=pmcid,
               tries=2)


# ---------------------------------------------------------------- PubMed

def pubmed_search(term):
    t = get(f"{EUTILS}/esearch.fcgi", ncbi_limit,
            {"db": "pubmed", "term": term, "retmode": "json", "retmax": 9999, "tool": TOOL},
            kind="pubmed-search", ident=term[:120])
    if t is None:
        return None
    return json.loads(t)["esearchresult"]["idlist"]


def _text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def pubmed_fetch(pmids):
    """Parse PubMed XML into dicts shaped like the fields we keep."""
    out = {}
    pmids = list(pmids)
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        t = get(f"{EUTILS}/efetch.fcgi", ncbi_limit,
                data={"db": "pubmed", "id": ",".join(chunk), "retmode": "xml", "tool": TOOL},
                kind="pubmed-fetch", ident=f"{chunk[0]}..({len(chunk)})")
        if t is None:
            continue
        root = ET.fromstring(t)
        for art in root.findall(".//PubmedArticle"):
            pmid = _text(art.find(".//MedlineCitation/PMID"))
            a = art.find(".//Article")
            ids = {x.get("IdType"): _text(x) for x in art.findall(".//PubmedData/ArticleIdList/ArticleId")}
            authors, affs = [], []
            for au in a.findall(".//AuthorList/Author"):
                name = " ".join(x for x in (_text(au.find("LastName")), _text(au.find("Initials"))) if x)
                authors.append(name or _text(au.find("CollectiveName")))
                affs += [_text(x) for x in au.findall(".//AffiliationInfo/Affiliation")]
            abst = []
            for p in a.findall(".//Abstract/AbstractText"):
                lab = p.get("Label")
                abst.append(f"{lab}: {_text(p)}" if lab else _text(p))
            year = _text(a.find(".//Journal/JournalIssue/PubDate/Year")) or \
                _text(a.find(".//Journal/JournalIssue/PubDate/MedlineDate"))[:4]
            out[pmid] = {
                "pmid": pmid, "pmcid": ids.get("pmc", ""), "doi": ids.get("doi", "").lower(),
                "title": _text(a.find("ArticleTitle")), "authors": authors,
                "journal": _text(a.find(".//Journal/Title")),
                "journal_abbrev": _text(a.find(".//Journal/ISOAbbreviation")),
                "year": year, "volume": _text(a.find(".//JournalIssue/Volume")),
                "issue": _text(a.find(".//JournalIssue/Issue")),
                "pages": _text(a.find(".//Pagination/MedlinePgn")),
                "abstract": "\n".join(abst),
                "pub_types": [_text(x) for x in a.findall(".//PublicationTypeList/PublicationType")],
                "affiliations": list(dict.fromkeys(affs)),
            }
    return out


# ---------------------------------------------------------------- PMC open-access bucket

def pmc_s3_meta(pmcid):
    """Latest-version metadata (licence, OA flag, xml and media URLs) or None."""
    t = get(f"{PMC_S3}/", s3_limit, {"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
            kind="pmc-s3-list", ident=pmcid)
    if not t:
        return None
    versions = re.findall(rf"<Prefix>({pmcid}\.(\d+))/</Prefix>", t)
    if not versions:
        return None
    ver = max(versions, key=lambda v: int(v[1]))[0]
    j = get(f"{PMC_S3}/{ver}/{ver}.json", s3_limit, kind="pmc-s3-meta", ident=ver)
    if not j:
        return None
    m = json.loads(j)

    def http(u):
        return u and u.split("?")[0].replace("s3://pmc-oa-opendata/", PMC_S3 + "/")
    m["xml_http"] = http(m.get("xml_url"))
    m["media_http"] = [http(u) for u in m.get("media_urls") or []]
    return m


def s3_fetch(url, binary=False):
    return get(url, s3_limit, kind="pmc-s3-file", ident=url, binary=binary, tries=3)
