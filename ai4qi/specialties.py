"""Canonical specialty names for the Ai4Qi library.

Readers write free-text specialties ("Critical care", "Otolaryngology", "Old age psychiatry"); this maps
each one to a single UK-style name so the app's filters and the gap report stay clean. Orthopaedic
subspecialties ("Orthopaedics – Hip fracture") are kept as they are. Seed entries (ids 1–1145) are never
changed; `pipeline.py merge` applies this to entries found by the pipeline.

    python3 specialties.py            # gap report: every canonical specialty and its count
"""
import re

CANONICAL = {
    "Surgery": ["General surgery", "Colorectal surgery", "Upper GI surgery", "Hepatobiliary surgery", "Breast surgery",
                "Endocrine surgery", "Vascular surgery", "Plastic surgery", "Neurosurgery", "Urology", "ENT",
                "Cardiothoracic surgery", "Paediatric surgery", "Transplant surgery", "Maxillofacial surgery", "Ophthalmology"],
    "Medicine": ["Acute and general medicine", "Care of the elderly", "Cardiology", "Respiratory", "Gastroenterology",
                 "Hepatology", "Renal medicine", "Endocrinology and diabetes", "Neurology", "Stroke", "Rheumatology",
                 "Dermatology", "Infectious diseases and microbiology", "Haematology", "Oncology", "Palliative care",
                 "Sexual health", "Clinical genetics", "Immunology and allergy", "Rehabilitation medicine",
                 "Sport and exercise medicine", "Occupational medicine", "Clinical pharmacology"],
    "Acute care": ["Emergency medicine", "Anaesthesia", "Intensive care", "Pre-hospital and ambulance care"],
    "Women and children": ["Obstetrics and gynaecology", "Paediatrics", "Neonatology"],
    "Mental health": ["Psychiatry"],
    "Diagnostics": ["Radiology", "Nuclear medicine", "Pathology", "Audiology"],
    "Community": ["General practice", "Public health", "Prison and secure healthcare"],
    "Professions": ["Pharmacy", "Nursing", "Midwifery", "Physiotherapy", "Occupational therapy",
                    "Speech and language therapy", "Nutrition and dietetics", "Podiatry", "Dentistry"],
    "Across the organisation": ["Governance", "Medical education"],
}
ALL = [s for group in CANONICAL.values() for s in group]

# (pattern, canonical); first match wins. Patterns run on the lower-cased label.
RULES = [
    (r"^orthopaed|^spine$|trauma and orthop|podiatric surgery|sports orthop|veterinary orthop", None),   # kept as given
    (r"paediatric surgery|paediatric neurosurg|paediatric urolog|cleft", "Paediatric surgery"),
    (r"paediatric dent|restorative|orthodont|oral surgery|oral medicine|special care dent|primary care dent|dental|dentistry", "Dentistry"),
    (r"paediatric (emergency|anaesth|radiolog|audiolog|ent|palliative|theatre)", None),                    # handled below by the adult rule
    (r"neonat", "Neonatology"),
    (r"paediatric|child health|^paediatrics", "Paediatrics"),
    (r"obstetric anaes", "Anaesthesia"),
    (r"midwif", "Midwifery"),
    (r"obstet|gynae|gynec|maternity", "Obstetrics and gynaecology"),
    (r"psychiat|mental health|camhs|learning disab|neuropsycholog", "Psychiatry"),
    (r"colorectal", "Colorectal surgery"), (r"upper gi|oesophag|bariatric", "Upper GI surgery"),
    (r"hepatobil|hpb|pancrea.*surg|liver surg", "Hepatobiliary surgery"), (r"breast", "Breast surgery"),
    (r"endocrine surg|thyroid surg|parathyroid", "Endocrine surgery"), (r"vascular", "Vascular surgery"),
    (r"plastic|burns|hand surgery", "Plastic surgery"), (r"neurosurg", "Neurosurgery"), (r"\burolog", "Urology"),
    (r"\bent\b|otolaryng|otorhino|head and neck", "ENT"), (r"cardiothorac|cardiac surg|thoracic surg", "Cardiothoracic surgery"),
    (r"transplant", "Transplant surgery"), (r"maxillofac|oral and maxillo", "Maxillofacial surgery"),
    (r"ophthalm|\beye\b", "Ophthalmology"),
    (r"major trauma|trauma surg|general surg|^surgery|surgical", "General surgery"),
    (r"emergency|a&e|urgent care", "Emergency medicine"),
    (r"pre-?hospital|ambulance|paramedic", "Pre-hospital and ambulance care"),
    (r"intensive|critical care|\bicu\b", "Intensive care"),
    (r"anaesth|anesth|pre-?operative assess|perioperative|theatre|pain medicine", "Anaesthesia"),
    (r"acute medicine|general medicine|internal medicine|^medicine$|acute and general", "Acute and general medicine"),
    (r"elderly|geriatr|older people|frailty|falls", "Care of the elderly"),
    (r"cardiol|heart", "Cardiology"), (r"respirat|pulmon|thorac", "Respiratory"),
    (r"hepatol|liver", "Hepatology"), (r"gastro|endoscop|\bibd\b", "Gastroenterology"),
    (r"renal|nephro|dialysis|kidney", "Renal medicine"), (r"endocrin|diabet", "Endocrinology and diabetes"),
    (r"stroke", "Stroke"), (r"neurolog|epilep", "Neurology"), (r"rheumat", "Rheumatology"), (r"dermat", "Dermatology"),
    (r"infection control|infection prevention|infectious|microbio|virolog|antimicrob|tropical|travel", "Infectious diseases and microbiology"),
    (r"transfusion|haematol|hematol|laboratory haem", "Haematology"),
    (r"oncolog|cancer|radiotherap", "Oncology"), (r"palliat|end of life|hospice", "Palliative care"),
    (r"sexual health|genitourin|\bhiv\b|sexual assault|contracep", "Sexual health"),
    (r"genetic", "Clinical genetics"), (r"immunol|allerg", "Immunology and allergy"),
    (r"rehabilit", "Rehabilitation medicine"), (r"\bsports?\b", "Sport and exercise medicine"),
    (r"occupational (medicine|health)", "Occupational medicine"), (r"clinical pharmacol", "Clinical pharmacology"),
    (r"nuclear medicine", "Nuclear medicine"),
    (r"radiolog|imaging|radiograph|medical physics", "Radiology"),
    (r"patholog|biochem|clinical chem|cytolog|mortuary", "Pathology"),
    (r"audiolog|audiovestib|hearing", "Audiology"),
    (r"general practice|primary care|family medicine|community care", "General practice"),
    (r"public health|screening programme", "Public health"),
    (r"prison|secure", "Prison and secure healthcare"),
    (r"pharmac", "Pharmacy"),
    (r"physiother", "Physiotherapy"), (r"occupational therap", "Occupational therapy"),
    (r"speech|language therap|dysphag", "Speech and language therapy"),
    (r"dietet|nutrition", "Nutrition and dietetics"), (r"podiatr", "Podiatry"),
    (r"nursing|tissue viab|wound|pressure ulcer", "Nursing"),
    (r"medical education|training|teaching|simulation", "Medical education"),
    (r"governance|patient safety|cross-specialty|outpatient|estates|switchboard|informatic|telehealth|therapies|ward care|multispecialty|not stated|not reported", "Governance"),
    (r"vascular science|vascular lab", "Vascular surgery"),
]
# Labels that mean the entry is not a human clinical audit; merge drops these.
NOT_CLINICAL = re.compile(r"veterinar|not clinical|research governance|meta-research|research policy|clinical trials research", re.I)


def canonical(label):
    s = (label or "").strip()
    if not s or s.lower() == "not reported":
        return s
    low = s.lower()
    if low.startswith("paediatric ") and re.match(r"paediatric (emergency|anaesth|radiolog|audiolog|ent|palliative|theatre)", low):
        low = low[len("paediatric "):]                  # a paediatric emergency audit files under Emergency medicine…
        if low.startswith("emergency"):
            return "Paediatrics"                        # …except that trainees look for it under Paediatrics
    for pat, out in RULES:
        if re.search(pat, low):
            return s if out is None else out
    return s


def gap_report(audits):
    import collections
    c = collections.Counter(canonical(a.get("specialty", "")) for a in audits)
    rows = []
    for group, names in CANONICAL.items():
        for n in names:
            rows.append((group, n, c.get(n, 0)))
    ortho = sum(v for k, v in c.items() if k.startswith("Orthopaedics"))
    other = {k: v for k, v in c.items() if k not in ALL and not k.startswith("Orthopaedics")}
    return rows, ortho, other


if __name__ == "__main__":
    import json
    import os
    lib = json.load(open(os.path.join(os.path.dirname(__file__), "ai4qi-library.json"), encoding="utf-8"))["audits"]
    rows, ortho, other = gap_report(lib)
    for g, n, k in sorted(rows, key=lambda r: r[2]):
        print(f"{k:5d}  {n:38s} {g}")
    print(f"{ortho:5d}  Orthopaedics (all subspecialties)")
    if other:
        print("unmapped:", other)
