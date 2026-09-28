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
    # third pass: more UK meeting-abstract journals
    ("International Journal of Surgery", ["audit", "re-audit", "closed loop"], 2012),
    ("BJS Open", ["audit", "re-audit"], 2017),
    ("Diabetic Medicine", ["audit", "re-audit"], 2012),
    ("BJOG: An International Journal of Obstetrics &amp; Gynaecology", ["audit", "re-audit"], 2012),
    ("BJOG: An International Journal of Obstetrics & Gynaecology", ["audit", "re-audit"], 2012),
    ("British Journal of Haematology", ["audit", "re-audit"], 2012),
    ("Transfusion Medicine", ["audit", "re-audit"], 2012),
    ("BMJ Supportive &amp; Palliative Care", ["audit", "re-audit"], 2012),
    ("BMJ Supportive & Palliative Care", ["audit", "re-audit"], 2012),
    ("European Journal of Surgical Oncology", ["audit", "re-audit"], 2012),
    ("Journal of Plastic, Reconstructive &amp; Aesthetic Surgery", ["audit", "re-audit"], 2012),
    ("Journal of Plastic, Reconstructive & Aesthetic Surgery", ["audit", "re-audit"], 2012),
    ("Clinical Otolaryngology", ["audit", "re-audit"], 2012),
    ("The Journal of Laryngology &amp; Otology", ["audit", "re-audit"], 2012),
    ("The Journal of Laryngology & Otology", ["audit", "re-audit"], 2012),
    ("Eye", ["audit", "re-audit"], 2012),
    ("International Journal of Pharmacy Practice", ["audit", "re-audit"], 2012),
    ("Clinical Radiology", ["audit", "re-audit"], 2012),
    ("Journal of Clinical Pathology", ["audit", "re-audit"], 2012),
    ("Journal of Infection", ["audit", "re-audit"], 2012),
    ("Physiotherapy", ["audit", "re-audit"], 2012),
    ("Journal of Obstetrics and Gynaecology", ["audit", "re-audit"], 2012),
    ("Journal of Hand Surgery (European Volume)", ["audit", "re-audit"], 2012),
    ("Lung Cancer", ["audit", "re-audit"], 2012),
    ("Scottish Medical Journal", ["audit", "re-audit"], 2010),
    ("The Ulster Medical Journal", ["audit", "re-audit"], 2010),
    ("Journal of Neurology", ["audit", "re-audit"], 2014),
    ("British Journal of Hospital Medicine", ["audit", "re-audit"], 2010),
    ("BJPsych Bulletin", ["audit", "re-audit"], 2014),
]
# fourth pass (Sep 2026, growth to 20,000): journals for thin specialties and allied professions.
# Selected with: python3 crossref_harvest.py --wave4
JOURNALS4 = [(j, ["audit", "re-audit", "quality improvement"], 2010) for j in [
    "British Dental Journal", "Faculty Dental Journal", "Dental Update", "Primary Dental Journal", "Journal of Orthodontics",
    "British Journal of General Practice", "BJGP Open", "Education for Primary Care",
    "BMJ Sexual &amp; Reproductive Health", "BMJ Sexual & Reproductive Health", "Sexually Transmitted Infections", "HIV Medicine", "International Journal of STD &amp; AIDS", "International Journal of STD & AIDS",
    "Journal of Perioperative Practice", "British Journal of Pain", "Journal of the Intensive Care Society", "Clinical Nutrition ESPEN",
    "Proceedings of the Nutrition Society", "Journal of Human Nutrition and Dietetics", "International Journal of Audiology",
    "British Journal of Occupational Therapy", "International Journal of Language &amp; Communication Disorders", "International Journal of Language & Communication Disorders",
    "Radiography", "Occupational Medicine", "British Journal of Sports Medicine", "Journal of Public Health", "Public Health",
    "Emergency Nurse", "British Paramedic Journal", "Journal of Paramedic Practice", "Journal of Neonatal Nursing", "Midwifery", "British Journal of Midwifery",
    "British Journal of Nursing", "Nursing Standard", "Nursing Older People", "Journal of Clinical Nursing", "Journal of Wound Care", "Journal of Tissue Viability",
    "European Journal of Hospital Pharmacy", "Journal of Oncology Pharmacy Practice", "The Pharmaceutical Journal", "JAC-Antimicrobial Resistance",
    "Journal of Antimicrobial Chemotherapy", "Infection Prevention in Practice", "Journal of Infection Prevention", "Clinical Infection in Practice",
    "Acute Medicine", "Frontline Gastroenterology", "Journal of Crohn's and Colitis", "Endocrine Abstracts", "Clinical Endocrinology", "Practical Diabetes",
    "Journal of Renal Care", "Nephrology Dialysis Transplantation", "Kidney International Reports", "Journal of Cystic Fibrosis",
    "International Journal of Stroke", "European Stroke Journal", "Seizure", "Developmental Medicine &amp; Child Neurology", "Developmental Medicine & Child Neurology",
    "British Journal of Neurosurgery", "BJU International", "Journal of Pediatric Urology", "Colorectal Disease", "Annals of The Royal College of Surgeons of England",
    "Clinical and Experimental Dermatology", "Palliative Medicine", "European Geriatric Medicine", "Journal of Psychopharmacology",
    "Irish Journal of Psychological Medicine", "Journal of Intellectual Disability Research", "Advances in Mental Health and Intellectual Disabilities",
    "The Journal of Forensic Psychiatry &amp; Psychology", "The Journal of Forensic Psychiatry & Psychology", "Archives of Disease in Childhood - Fetal and Neonatal Edition",
    "BMJ Paediatrics Open", "Child: Care, Health and Development", "Vox Sanguinis", "Cytopathology", "Histopathology", "Annals of Clinical Biochemistry",
    "British Journal of Biomedical Science", "The Clinical Teacher", "BMJ Simulation &amp; Technology Enhanced Learning", "BMJ Simulation & Technology Enhanced Learning",
    "European Journal of Vascular and Endovascular Surgery", "Journal of Vascular Societies Great Britain &amp; Ireland", "Journal of Vascular Societies Great Britain & Ireland",
    "Interactive CardioVascular and Thoracic Surgery", "Journal of Cardiothoracic Surgery", "British Journal of Ophthalmology", "Ophthalmic and Physiological Optics",
    "Journal of Medical Imaging and Radiation Sciences", "The British Journal of Radiology", "BJR|Open", "Journal of Clinical Urology",
    "Transplantation", "Clinical Transplantation", "International Journal of Obstetric Anesthesia", "Pediatric Anesthesia", "Anaesthesia Reports",
    "Journal of Evaluation in Clinical Practice", "BMJ Open", "BMJ Quality &amp; Safety", "BMJ Quality & Safety", "Cureus",
]]
# fifth pass (Sep 2026): congress-abstract journals missed so far (resuscitation, trauma, burns, EU public health,
# nursing, epilepsy, obstetrics). Selected with: python3 crossref_harvest.py --wave5
JOURNALS5 = [(j, ["audit", "re-audit", "quality improvement"], 2010) for j in [
    "Resuscitation", "Resuscitation Plus", "Injury", "Burns", "Burns Open", "The Breast", "Clinical Nutrition",
    "European Journal of Public Health", "Family Practice", "Neuro-Oncology", "Neuro-Oncology Advances",
    "European Heart Journal - Quality of Care and Clinical Outcomes", "European Heart Journal", "Europace",
    "Rheumatology Advances in Practice", "Journal of Pediatric Surgery", "Surgery (Oxford)", "Annals of Medicine and Surgery",
    "International Journal of Surgery Open", "Intensive and Critical Care Nursing", "Women and Birth",
    "European Journal of Obstetrics &amp; Gynecology and Reproductive Biology", "European Journal of Obstetrics & Gynecology and Reproductive Biology",
    "Epilepsy &amp; Behavior", "Epilepsy & Behavior", "Journal of the Neurological Sciences", "Parkinsonism &amp; Related Disorders",
    "Parkinsonism & Related Disorders", "Journal of the Royal College of Physicians of Edinburgh", "JRSM Open",
    "European Journal of Cardiovascular Nursing", "Journal of Burn Care &amp; Research", "Journal of Burn Care & Research",
    "Clinical Medicine Insights", "QJM: An International Journal of Medicine", "Journal of Clinical Neuroscience",
    "European Urology", "European Urology Supplements", "Journal of Clinical Oncology", "Annals of Oncology",
    "European Journal of Cancer", "Radiotherapy and Oncology", "Journal of Medical Imaging and Radiation Oncology",
    "Anaesthesia and Intensive Care", "Canadian Journal of Anesthesia", "Pediatric Critical Care Medicine",
    "Critical Care", "Intensive Care Medicine Experimental", "European Journal of Anaesthesiology",
    "Journal of Perinatal Medicine", "Archives of Disease in Childhood - Education &amp; Practice Edition",
    "Journal of Paediatrics and Child Health", "Paediatrics and Child Health", "Irish Medical Journal",
    "Journal of Oral and Maxillofacial Surgery", "British Journal of Oral &amp; Maxillofacial Surgery", "Oral Surgery",
    "Journal of Dentistry", "International Journal of Paediatric Dentistry", "Gerodontology",
    "Journal of Psychiatric and Mental Health Nursing", "BJPsych Advances", "Progress in Neurology and Psychiatry",
    "Journal of Affective Disorders", "Schizophrenia Research", "European Neuropsychopharmacology",
]]
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
    js = JOURNALS4 if only == "--wave4" else JOURNALS5 if only == "--wave5" else JOURNALS
    for container, queries, year in js:
        if only and not only.startswith("--wave") and container != only:
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
