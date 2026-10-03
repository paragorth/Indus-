"""Download every Bone & Joint Publishing record with an abstract from Crossref
(prefix 10.1302: Bone Joint J, Bone & Joint Open, Orthopaedic Proceedings and the
old JBJS Br meeting supplements) so conference abstracts, which PubMed does not
index, can be screened for audits locally."""
import json, os, time, requests
from sources import WORK, log_failure
OUT = os.path.join(WORK, "crossref_10.1302.jsonl")
S = requests.Session(); S.headers["User-Agent"] = "ai4qi-audit-library/1.0 (mailto:not-provided)"
cursor, n = "*", 0
with open(OUT, "w", encoding="utf-8") as f:
    while True:
        for attempt in range(4):
            try:
                r = S.get("https://api.crossref.org/prefixes/10.1302/works", timeout=90, params={
                    "filter": "has-abstract:true", "rows": 1000, "cursor": cursor,
                    "select": "DOI,title,abstract,container-title,issued,author,volume,issue,page,type"})
                if r.status_code == 200: break
            except requests.RequestException as e:
                r = None
            time.sleep(2 ** attempt)
        if r is None or r.status_code != 200:
            log_failure("crossref", cursor, getattr(r, "status_code", "connection error")); break
        m = r.json()["message"]
        for it in m["items"]:
            f.write(json.dumps(it) + "\n"); n += 1
        if not m["items"]: break
        cursor = m["next-cursor"]
        print(n, flush=True)
print("done", n)
