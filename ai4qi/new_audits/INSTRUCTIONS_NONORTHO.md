# New non-orthopaedic audit design (Ai4Qi)

Goal: design NEW, easy-to-deploy audits in YOUR specialty area (given in your task) for UK trainees, grounded in CURRENT standards,
that are not already common in our corpus (or have never had a closed loop in it).

## Step 1 — current standards (fetch them; do not rely on memory)
Use WebFetch (or curl) on the official pages for your area: NICE guidance and quality standards
(https://www.nice.org.uk/guidance/<code>/chapter/Recommendations), royal college and specialty society standards
(RCEM, RCP, RCPsych, RCOG, RCPCH, RCR, RCoA/AAGBI, ASGBI, BAUS, ENT UK, BAPRAS, RCOphth, BSG, BTS, BSH/SHOT,
BASHH, BGS, FSRH etc.), NCEPOD, national audits (NELA, NBOCA, SSNAP, NaDIA, NACAP, NCAP, NMPA, NNAP), GIRFT,
CPOC, NHS England specifications, MHRA drug safety updates, UKHSA, SIGN if UK-relevant.
Quote the recommendation wording EXACTLY (copy it) with its number and URL. If a page cannot be fetched, do not
invent wording; pick another standard. Prefer guidance updated 2020–2026. Never click through CAPTCHAs or bot checks,
never disable TLS verification.

## Step 2 — check the corpus for gaps
cd /home/user/Indus-/ai4qi && python3 query.py "<keywords>" --specialty "<specialty>" -n 15   (and without --specialty)
python3 query.py --topics --specialty "<specialty>"
Classify each idea: "new" (nothing similar in corpus) or "under-audited" (in corpus but no closed loop / different
question). Skip ideas the corpus shows are heavily overdone (operation notes, consent completeness, discharge summaries, handover,
antipsychotic physical health monitoring, VTE risk assessment, lithium monitoring) unless you target a specific item nobody measured.

## Step 3 — only keep EASY-to-deploy audits
Data retrievable from notes/EPR, theatre system, PACS, drug chart, national audit extract or clinic letters; no patient
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
 "pitfalls": ["3–5 items: 'what can go wrong or who objects → how to prevent it' (data gaps, small numbers, changing definitions between cycles, staff pushback, the fix not used, gains fading)"],
 "pearls": ["3–5 practical tips that make it succeed (who to recruit, what to agree before cycle 1, how to keep the change after rotation)"],
 "novelty": "new | under-audited",
 "effort": "e.g. ~20 min per 10 patients from EPR"
}
Template: 6–12 fields, always include hospital_number_pseudonymised and date fields needed for timings,
and one field per pass criterion. British spelling. Short sentences.

Write a JSON array of your audits (at least the number you are asked for) to your output path, validate it loads
with python and every item has every key. Keep scratch files in a scratchpad folder named after your area.
Do not modify other files; do not commit. Reply with the count and the list of questions.

## Finished-product wording
Every text field goes straight into a professional application. No process talk in any field ("I searched",
"not in corpus", "could not fetch", "made up"). In "why", describe the gap as a fact ("No closed-loop audit of X
has been published in the library" is fine; cite ids where they exist). Evidence lines cite only real library
ids; if none are close, give the nearest related entries. If there is no national standard, source is
"Local standard" and the wording is your proposed local standard. Put any caveats (blocked pages, uncertain
wording) in your final reply to me, not in the JSON.
