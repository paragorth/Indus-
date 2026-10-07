"""Harvest audit papers from sources beyond PubMed/Europe PMC and the named Crossref journals.

    python3 extra_harvest.py doaj            # DOAJ open-access journals, "audit" in the title, by year
    python3 extra_harvest.py s2              # Semantic Scholar bulk search for audit-cycle wording
    python3 extra_harvest.py crossref_all    # Crossref across all journals, audit query, by year
    python3 extra_harvest.py thin            # Crossref + Semantic Scholar per thin specialty
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


# Wave 16 (Sep 2026): wider wording, to reach 20,000 audits.
S2_QUERIES_2 = ['"retrospective audit" + (hospital | patients)', '"prospective audit" + (hospital | patients)',
                'audit + (NICE | "Royal College" | BTS | BSG | RCOG | RCEM | RCR | BAUS | "national standard")',
                '"quality improvement project" + (audit | compliance | documentation)',
                '"compliance audit" | "documentation audit" | "prescribing audit"',
                'audit + (UK | NHS | Ireland | HSE) + (standard | guideline)']


S2_QUERIES_3 = ['audit + (anaesthesia | anaesthetic | theatre | perioperative)', 'audit + (antibiotic | antimicrobial | sepsis)',
                'audit + (prescribing | medication | "drug chart" | pharmacist)', 'audit + (discharge | handover | "clinical documentation")',
                'audit + (paediatric | neonatal | children) + (guideline | standard)', 'audit + (maternity | obstetric | midwifery)',
                'audit + (radiology | imaging | "radiation dose" | "request forms")', 'audit + ("mental health" | psychiatric | dementia | delirium)',
                'audit + ("general practice" | "primary care" | GP)', 'audit + ("emergency department" | "emergency medicine")',
                'audit + (stroke | cardiology | "heart failure" | "atrial fibrillation")', 'audit + (diabetes | "blood glucose" | insulin)',
                'audit + (cancer | oncology | chemotherapy | "two week wait")', 'audit + (VTE | thromboprophylaxis | anticoagulation)',
                'audit + ("acute kidney injury" | fluid | electrolyte)', 'audit + (pain | analgesia | opioid)',
                'audit + (falls | "pressure ulcer" | nutrition | "malnutrition")', 'audit + (dental | oral | orthodontic)',
                'audit + (consent | "operation note" | "surgical safety checklist")', 'audit + (endoscopy | colonoscopy | gastroenterology)',
                'audit + (respiratory | asthma | COPD | oxygen)', 'audit + (infection control | "hand hygiene" | cannula)']


S2_QUERIES_4 = ['audit + (ophthalmology | cataract | glaucoma | eye)', 'audit + (dermatology | skin | melanoma)', 'audit + (urology | catheter | prostate)',
                'audit + (ENT | otolaryngology | tonsillectomy | tracheostomy)', 'audit + (haematology | anaemia | "iron deficiency")',
                'audit + (renal | dialysis | nephrology)', 'audit + (rheumatology | "rheumatoid arthritis" | gout | osteoporosis)',
                'audit + (neurology | epilepsy | "Parkinson" | "multiple sclerosis")', 'audit + (orthopaedic | fracture | arthroplasty | "hip fracture")',
                'audit + ("general surgery" | appendicectomy | cholecystectomy | hernia)', 'audit + (caesarean | labour | postpartum | antenatal)',
                'audit + (neonatal | NICU | preterm | newborn)', 'audit + (ICU | "intensive care" | ventilated | "critical care")',
                'audit + (palliative | "end of life" | DNACPR | "advance care")', 'audit + (physiotherapy | rehabilitation | "occupational therapy")',
                'audit + (vascular | amputation | "diabetic foot")', 'audit + (plastic surgery | "hand surgery" | burns)',
                'audit + (breast | mammography | "breast surgery")', 'audit + (colorectal | "bowel cancer" | stoma)',
                'audit + (hepatology | cirrhosis | "liver disease" | alcohol)', 'audit + (sexual health | HIV | contraception)',
                'audit + (blood transfusion | "blood products" | "patient blood management")']


# The s2d run stopped at the neonatal query (killed before saving): the rest of S2_QUERIES_4.
S2_QUERIES_5 = S2_QUERIES_4[S2_QUERIES_4.index('audit + (neonatal | NICU | preterm | newborn)'):]


def s2(st, queries=None):
    for q in (queries or S2_QUERIES):
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


def crossref_all(st, years=None, pages=3):
    for y in (years or YEARS):
        cursor, got = "*", 0
        for _ in range(pages):                            # top 1,000 x pages by relevance per year (default 3,000)
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


THIN_TERMS = {   # specialties with the fewest audits in the library (specialties.py gap report)
    "podiatry": "podiatry foot diabetic foot ulcer", "endocrine surgery": "thyroidectomy parathyroidectomy adrenalectomy",
    "occupational therapy": "occupational therapy", "sport medicine": "sports medicine exercise injury",
    "prison": "prison custody secure hospital", "hepatology": "cirrhosis liver disease hepatitis",
    "transplant": "transplant transplantation", "midwifery": "midwife midwifery labour intrapartum",
    "speech therapy": "speech and language therapy dysphagia", "genetics": "genetic testing clinical genetics",
    "allergy": "allergy anaphylaxis immunology", "clinical pharmacology": "therapeutic drug monitoring medication",
    "nuclear medicine": "nuclear medicine PET scintigraphy", "audiology": "audiology hearing aid newborn hearing",
    "hepatobiliary": "cholecystectomy pancreatitis hepatobiliary", "physiotherapy": "physiotherapy",
    "rehabilitation": "rehabilitation", "paediatric surgery": "paediatric surgery appendicitis children",
    "occupational medicine": "occupational health staff", "ambulance": "ambulance paramedic prehospital",
    "cardiothoracic": "cardiac surgery thoracic surgery", "public health": "screening vaccination immunisation",
    "neurosurgery": "neurosurgery head injury", "maxillofacial": "maxillofacial oral surgery mandible",
    "dietetics": "nutrition dietitian malnutrition", "upper gi": "oesophagectomy gastrectomy bariatric",
    "plastic surgery": "plastic surgery burns hand surgery", "dermatology": "dermatology skin cancer",
    "neurology": "epilepsy multiple sclerosis Parkinson", "breast": "breast surgery mastectomy",
}


def thin(st):
    for name, terms in THIN_TERMS.items():
        got = 0
        for q in (f"clinical audit re-audit {terms}", f"audit compliance guideline {terms}"):
            cursor = "*"
            for _ in range(2):
                d = _get("https://api.crossref.org/works", {
                    "query.bibliographic": q, "filter": "has-abstract:true,type:journal-article",
                    "rows": 1000, "cursor": cursor,
                    "select": "DOI,title,abstract,container-title,issued,author,volume,issue,page,license"}, "crossref-thin")
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
                    st.add(f"DOI:{it['DOI'].lower()}", rec, f"thin crossref {name}")
                    got += 1
                if len(m["items"]) < 1000:
                    break
                cursor = m["next-cursor"]
        token = None
        while True:                                       # Semantic Scholar, same specialty
            p = {"query": f"audit + ({' | '.join(terms.split()[:4])})",
                 "fields": "title,abstract,externalIds,year,venue,journal,authors,publicationTypes"}
            if token:
                p["token"] = token
            d = _get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", p, "semanticscholar-thin")
            if not d:
                break
            for it in d.get("data") or []:
                ex = it.get("externalIds") or {}
                if "Review" in (it.get("publicationTypes") or []):
                    continue
                if not re.search(r"\baudit(?!ory|ion|ive)", f"{it.get('title') or ''} {it.get('abstract') or ''}", re.I):
                    continue
                j = it.get("journal") or {}
                rec = _record(it.get("title"), it.get("abstract"), doi=ex.get("DOI", ""), pmid=ex.get("PubMed", ""),
                              pmcid=("PMC" + str(ex["PubMedCentral"])) if ex.get("PubMedCentral") else "",
                              journal=j.get("name") or it.get("venue", ""), year=it.get("year"),
                              authors=[a.get("name", "") for a in it.get("authors") or []],
                              volume=j.get("volume", ""), pages=j.get("pages", ""), source="Semantic Scholar")
                key = (f"PMID:{rec['pmid']}" if rec["pmid"] else f"DOI:{rec['doi']}" if rec["doi"]
                       else f"S2:{it.get('paperId')}")
                st.add(key, rec, f"thin s2 {name}")
                got += 1
            token = d.get("token")
            if not token or got > 6000:
                break
            time.sleep(1.2)
        print(f"  thin {name}: {got} results, {st.new} new so far", flush=True)
        st.save()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    st = Store()
    for name, fn in (("doaj", doaj), ("s2", s2), ("crossref_all", crossref_all), ("thin", thin),
                     ("s2b", lambda st: s2(st, S2_QUERIES_2)), ("s2c", lambda st: s2(st, S2_QUERIES_3)),
                     ("s2d", lambda st: s2(st, S2_QUERIES_4)), ("s2e", lambda st: s2(st, S2_QUERIES_5)),
                     ("crossref_deep", lambda st: crossref_all(st, range(2008, 2027), 8))):
        if which == name or (which == "all" and name not in ("s2b", "s2c", "s2d", "s2e", "crossref_deep")):
            fn(st)
            print(f"{name}: {st.new} new records in total", flush=True)
