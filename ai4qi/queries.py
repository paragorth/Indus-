"""Search definitions for the Ai4Qi audit library.

Each query is AUDIT_TERMS AND <area terms>.  The same area terms are also
used afterwards to label each paper with a subspecialty/specialty.
"""

# Closed-loop / audit-cycle wording.  "quality improvement project" only
# counts when "audit" is also in the title/abstract.
AUDIT_PHRASES = [
    "closed loop audit", "closed-loop audit", "completed audit cycle",
    "complete audit cycle", "full audit cycle", "re-audit", "reaudit",
    "re-audited", "audit cycle", "audit cycles", "second audit cycle",
    "second cycle", "audit loop", "closing the loop", "close the loop",
    "two-cycle audit", "two cycle audit", "clinical audit",
]


def pubmed_audit_clause():
    terms = " OR ".join(f'"{p}"[tiab]' for p in AUDIT_PHRASES)
    return f'({terms} OR ("quality improvement project"[tiab] AND audit[tiab]))'


def epmc_audit_clause():
    terms = " OR ".join(f'TITLE_ABS:"{p}"' for p in AUDIT_PHRASES)
    return f'({terms} OR (TITLE_ABS:"quality improvement project" AND TITLE_ABS:audit))'


# Trauma & orthopaedics, by subspecialty.  Order matters for labelling:
# the first subspecialty whose terms match the title wins, then abstract.
ORTHO = {
    "hip fracture": ["hip fracture", "hip fractures", "neck of femur", "femoral neck fracture",
                     "proximal femoral fracture", "fractured neck of femur", "NOF fracture",
                     "intertrochanteric", "hemiarthroplasty"],
    "foot and ankle": ["ankle fracture", "ankle", "foot", "achilles", "hallux", "calcaneal",
                       "diabetic foot"],
    "hand and wrist": ["hand surgery", "hand trauma", "hand injury", "hand injuries", "wrist",
                       "distal radius", "scaphoid", "carpal tunnel", "finger", "metacarpal"],
    "paediatric": ["paediatric orthopaedic", "pediatric orthopedic", "paediatric fracture",
                   "pediatric fracture", "supracondylar", "developmental dysplasia of the hip",
                   "DDH", "slipped capital femoral", "clubfoot", "children's fracture"],
    "spine": ["spine", "spinal", "cauda equina", "lumbar", "cervical spine", "vertebral",
              "scoliosis", "discectomy"],
    "arthroplasty": ["arthroplasty", "joint replacement", "hip replacement", "knee replacement",
                     "total hip", "total knee", "periprosthetic", "TKA", "THA", "TKR", "THR"],
    "shoulder and elbow": ["shoulder", "elbow", "clavicle", "proximal humerus", "humeral",
                           "rotator cuff", "olecranon"],
    "knee": ["knee", "anterior cruciate", "ACL", "meniscus", "meniscal", "patella", "tibial plateau"],
    "trauma": ["fracture clinic", "fracture", "fractures", "orthopaedic trauma", "orthopedic trauma",
               "major trauma", "trauma", "polytrauma", "open fracture", "plaster cast",
               "virtual fracture clinic", "compartment syndrome"],
    "theatre and perioperative": ["orthopaedic theatre", "orthopedic surgery", "orthopaedic surgery",
                                  "orthopaedic", "orthopedic", "orthopaedics", "orthopedics",
                                  "trauma and orthopaedic", "venous thromboembolism",
                                  "surgical site infection", "consent", "tourniquet",
                                  "WHO checklist", "theatre", "operating theatre", "operating room"],
}

# Terms that anchor a paper to T&O at all (used in the ortho pass so that e.g.
# a "consent" audit only counts when an orthopaedic word is also present).
ORTHO_ANCHORS = ["orthopaed", "orthoped", "fracture", "arthroplasty", "trauma and orth",
                 "t&o", "spinal surgery", "spine surgery", "joint replacement", "hip ", "knee",
                 "shoulder", "ankle", "wrist", "elbow", "scaphoid", "achilles", "cauda equina",
                 "plaster", "cast ", "neck of femur", "hemiarthroplasty", "fracture clinic"]

NON_ORTHO = {
    "general surgery": ["general surgery", "general surgical", "appendicectomy", "appendectomy",
                        "cholecystectomy", "hernia", "colorectal", "laparotomy", "emergency laparotomy",
                        "breast surgery", "vascular surgery", "acute surgical"],
    "urology": ["urology", "urological", "prostate", "catheter", "renal colic", "cystoscopy",
                "haematuria", "hematuria", "kidney stone"],
    "ENT": ["ENT", "otolaryngology", "otorhinolaryngology", "tonsillectomy", "epistaxis",
            "head and neck", "ear nose and throat", "tracheostomy"],
    "obstetrics and gynaecology": ["obstetric", "obstetrics", "gynaecology", "gynecology",
                                   "maternity", "caesarean", "cesarean", "labour ward", "antenatal",
                                   "postpartum", "pregnancy", "pregnant"],
    "paediatrics": ["paediatric", "pediatric", "neonatal", "neonate", "children", "child",
                    "infant", "NICU"],
    "emergency medicine": ["emergency department", "emergency medicine", "accident and emergency",
                           "A&E", "ED "],
    "acute and general medicine": ["acute medicine", "acute medical", "general medicine",
                                   "medical admissions", "internal medicine", "sepsis",
                                   "venous thromboembolism", "VTE", "falls", "delirium",
                                   "fluid balance", "NEWS2", "discharge summary", "handover"],
    "cardiology": ["cardiology", "cardiac", "heart failure", "atrial fibrillation", "myocardial",
                   "acute coronary", "ECG", "anticoagulation"],
    "respiratory": ["respiratory", "asthma", "COPD", "pneumonia", "oxygen prescribing",
                    "pulmonary", "non-invasive ventilation", "spirometry", "tuberculosis"],
    "gastroenterology": ["gastroenterology", "endoscopy", "colonoscopy", "hepatology", "liver",
                         "inflammatory bowel", "upper GI bleed", "gastrointestinal bleed"],
    "anaesthesia": ["anaesthesia", "anesthesia", "anaesthetic", "anesthetic", "PONV",
                    "postoperative nausea", "airway", "preoperative fasting", "pre-operative fasting",
                    "analgesia", "pain management"],
    "intensive care": ["intensive care", "critical care", "ICU", "ITU", "ventilator",
                       "mechanically ventilated"],
    "radiology": ["radiology", "radiological", "imaging", "CT ", "MRI", "X-ray", "radiograph",
                  "ultrasound", "radiation dose", "IR(ME)R"],
    "psychiatry": ["psychiatry", "psychiatric", "mental health", "antipsychotic", "clozapine",
                   "lithium", "dementia", "self-harm", "CAMHS"],
    "general practice": ["general practice", "primary care", "GP practice", "general practitioner",
                         "family medicine", "family practice"],
    "pharmacy": ["pharmacy", "pharmacist", "prescribing", "medication", "antibiotic", "antimicrobial",
                 "drug chart", "medicines reconciliation", "opioid"],
    "nursing": ["nursing", "nurse", "nurses", "pressure ulcer", "pressure injury", "care home"],
    "dentistry": ["dental", "dentistry", "oral surgery", "orthodontic", "maxillofacial",
                  "periodontal", "caries"],
    "ophthalmology": ["ophthalmology", "ophthalmic", "eye", "cataract", "glaucoma", "retinal",
                      "intravitreal"],
    "dermatology": ["dermatology", "skin cancer", "melanoma", "psoriasis", "eczema", "dermatological",
                    "skin lesion"],
    "oncology": ["oncology", "cancer", "chemotherapy", "radiotherapy", "tumour", "tumor",
                 "palliative"],
    "haematology": ["haematology", "hematology", "blood transfusion", "transfusion", "anaemia",
                    "anemia", "sickle cell", "thrombosis", "warfarin"],
    # wider coverage (added in the third search pass)
    "vascular surgery": ["vascular surgery", "aortic aneurysm", "carotid", "varicose", "peripheral arterial", "amputation"],
    "colorectal surgery": ["colorectal", "bowel cancer", "colonoscopy", "stoma", "anastomotic", "haemorrhoid"],
    "breast surgery": ["breast surgery", "breast cancer", "mastectomy", "breast clinic", "mammography"],
    "plastic surgery and burns": ["plastic surgery", "burns", "burn injury", "skin cancer excision", "hand trauma", "flap"],
    "neurosurgery": ["neurosurgery", "neurosurgical", "subarachnoid", "hydrocephalus", "craniotomy", "brain tumour"],
    "cardiothoracic surgery": ["cardiac surgery", "cardiothoracic", "coronary artery bypass", "thoracic surgery", "lung resection"],
    "maxillofacial surgery": ["maxillofacial", "oral surgery", "mandibular fracture", "facial trauma", "third molar"],
    "neurology": ["neurology", "epilepsy", "seizure", "multiple sclerosis", "parkinson", "migraine", "headache"],
    "stroke": ["stroke", "transient ischaemic attack", "thrombolysis", "thrombectomy"],
    "care of the elderly": ["elderly", "older people", "geriatric", "frailty", "falls", "dementia", "delirium"],
    "renal medicine": ["renal", "kidney", "dialysis", "acute kidney injury", "nephrology"],
    "diabetes and endocrinology": ["diabetes", "diabetic", "insulin", "hypoglycaemia", "thyroid", "endocrin", "DKA"],
    "rheumatology": ["rheumatology", "rheumatoid", "gout", "methotrexate", "giant cell arteritis", "lupus"],
    "infection and microbiology": ["antimicrobial", "antibiotic", "sepsis", "infection control", "microbiology", "C. difficile", "MRSA"],
    "palliative care": ["palliative", "end of life", "end-of-life", "hospice", "DNACPR", "anticipatory"],
    "neonatology": ["neonatal", "neonate", "newborn", "NICU", "preterm", "jaundice"],
    "sexual health": ["sexual health", "genitourinary", "HIV", "chlamydia", "gonorrhoea", "syphilis", "contraception"],
    "pathology and laboratory": ["pathology", "histopathology", "laboratory", "blood test", "phlebotomy", "cytology"],
    "physiotherapy and rehabilitation": ["physiotherapy", "rehabilitation", "occupational therapy", "speech and language"],
    "learning disability and CAMHS": ["learning disability", "intellectual disability", "CAMHS", "child and adolescent mental health", "autism"],
    "patient safety and governance": ["patient safety", "incident reporting", "handover", "medication error", "never event", "record keeping"],
}


def _pm_terms(terms):
    return "(" + " OR ".join(f'"{t.strip()}"[tiab]' for t in terms) + ")"


def _epmc_terms(terms):
    return "(" + " OR ".join(f'TITLE_ABS:"{t.strip()}"' for t in terms) + ")"


def query_set(which):
    """Return [(label, pubmed_query, epmc_query)] for 'ortho' or 'nonortho'."""
    areas = ORTHO if which == "ortho" else NON_ORTHO
    out = []
    for label, terms in areas.items():
        if which == "ortho" and label == "theatre and perioperative":
            # Perioperative words are generic: require an orthopaedic anchor too.
            anchor = ["orthopaedic", "orthopedic", "orthopaedics", "orthopedics", "trauma and orthopaedic",
                      "fracture", "arthroplasty"]
            pm = f"{pubmed_audit_clause()} AND {_pm_terms(terms)} AND {_pm_terms(anchor)}"
            ep = f"{epmc_audit_clause()} AND {_epmc_terms(terms)} AND {_epmc_terms(anchor)}"
        else:
            pm = f"{pubmed_audit_clause()} AND {_pm_terms(terms)}"
            ep = f"{epmc_audit_clause()} AND {_epmc_terms(terms)}"
        out.append((label, pm, ep))
    return out


# ---------------------------------------------------------------------------
# Broadening queries (added after the first ortho run found 790 hits):
#  - Europe PMC searches open-access FULL TEXT when no field is given, which
#    catches audits that only say "re-audit"/"closed loop" in the body.
#  - "audit" in the title, and QI projects that mention audit anywhere.
# The area terms here are restricted to the TITLE so that full-text matches
# are not triggered by passing mentions.
STRONG = ['"closed loop audit"', '"closed-loop audit"', '"re-audit"', "reaudit", '"re-audited"',
          '"completed audit cycle"', '"complete audit cycle"', '"audit cycle"', '"second audit cycle"',
          '"audit loop"']


def broad_query_set(which):
    areas = ORTHO if which == "ortho" else NON_ORTHO
    out = []
    for label, terms in areas.items():
        ep_title = "(" + " OR ".join(f'TITLE:"{t.strip()}"' for t in terms) + ")"
        pm_title = "(" + " OR ".join(f'"{t.strip()}"[ti]' for t in terms) + ")"
        if which == "ortho" and label == "theatre and perioperative":
            ep_title += ' AND (TITLE:orthopaed* OR TITLE:orthoped* OR TITLE:fracture* OR TITLE:arthroplasty)'
            pm_title += ' AND (orthopaed*[ti] OR orthoped*[ti] OR fracture*[ti] OR arthroplasty[ti])'
        out.append((label + " (full text)", None, f"({' OR '.join(STRONG)}) AND {ep_title}"))
        out.append((label + " (audit in title)", f"audit*[ti] AND {pm_title}", f"TITLE:audit* AND {ep_title}"))
        out.append((label + " (QI + audit)",
                    f'("quality improvement"[tiab] OR PDSA[tiab]) AND audit*[tiab] AND {pm_title}',
                    f'(TITLE:"quality improvement" OR TITLE:"improvement project" OR TITLE:PDSA) AND audit* AND {ep_title}'))
    return out
