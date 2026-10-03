# New orthopaedic audit design (Ai4Qi)

Goal: design NEW, easy-to-deploy orthopaedic audits for UK trainees, grounded in CURRENT standards,
that are not already common in our corpus (or have never had a closed loop in it).

## Step 1 — current standards (fetch them; do not rely on memory)
Use WebFetch (or curl) on the official pages for your area, e.g.:
- BOAST list: https://www.boa.ac.uk/standards-guidance/boasts.html (open each relevant BOAST page)
- NICE: https://www.nice.org.uk/guidance/<code>/chapter/Recommendations (NG124 hip fracture, NG38 non-complex fractures,
  NG37 complex fractures, NG41 spinal injury, NG39 major trauma, NG157 joint replacement, NG226 osteoarthritis,
  NG89 VTE, NG125 surgical site infection, NG180 perioperative care, NG45 preoperative tests, NG24 transfusion,
  CG103/NG... delirium, CG161 falls, QS as relevant)
- NHFD KPIs (https://www.nhfd.co.uk), GIRFT orthopaedics, BSSH hand trauma standards, NatSSIPs 2 (2023),
  RCEM, Royal Osteoporosis Society clinical standards for FLS, IR(ME)R, BAPRAS/BOA open fracture, NJR.
Quote the recommendation wording EXACTLY (copy it) with its number and URL. If a page cannot be fetched, say so
and do not invent wording. Prefer guidance updated 2020–2026.

## Step 2 — check the corpus for gaps
cd /home/user/Indus-/ai4qi && python3 query.py "<keywords>" --specialty orthopaed -n 15   (and without --specialty)
python3 query.py --topics --specialty orthopaed
Classify each idea: "new" (nothing similar in corpus) or "under-audited" (in corpus but no closed loop / different
question). Skip ideas the corpus shows are heavily overdone (operation notes, consent completeness, time to theatre,
supracondylar documentation) unless you target a specific item nobody measured.

## Step 3 — only keep EASY-to-deploy audits
Data retrievable from notes/EPR, theatre system, PACS, drug chart, NHFD/NJR extract or clinic letters; no patient
contact, no ethics, no new equipment; 30–50 cases collectable in ≤2 weeks in a typical district general hospital.

## Step 4 — write each audit (JSON), exactly these keys
{
 "area": "<your area>",
 "question": "ONE plain measurable question (who, what, pass). No jargon, no 'X and Y'.",
 "why": "1–2 lines: the problem and the gap (cite corpus ids like [1892] or say 'no audit in corpus').",
 "standard": {"source": "e.g. NICE NG124 rec 1.6.1 (2011, updated 2023)", "wording": "exact quote", "url": "..."},
 "pass": "what counts as meeting the standard for one patient",
 "population": "inclusion / exclusion",
 "sample": "e.g. 40 consecutive adults ... over the last 3 months",
 "data_source": "where each field comes from",
 "template": [{"field": "snake_case_name", "type": "yes/no | date | datetime | number | text | choice", "options": [], "note": ""}],
 "timeline": "Wk 1–2 collect; Wk 3 analyse/present; Wk 4 change; Wk 5–12 embed; Wk 13–14 re-audit",
 "change": "ONE fix, preferably a form/template or system change (corpus: teaching alone rarely works)",
 "target": "e.g. ≥90%",
 "reaudit": "when, same template/sample",
 "close_loop": "where to present; what to embed if it worked",
 "evidence": ["[id] where: before → after (fix)", "..."],
 "novelty": "new | under-audited",
 "effort": "e.g. ~20 min per 10 patients from EPR"
}
Template: 6–12 fields, always include hospital_number_pseudonymised and date fields needed for timings,
and one field per pass criterion. British spelling. Short sentences.

Write a JSON array of your audits (at least the number you are asked for) to your output path, validate it loads
with python and every item has every key. Keep scratch files in a scratchpad folder named after your area.
Do not modify other files; do not commit. Reply with the count and the list of questions.
