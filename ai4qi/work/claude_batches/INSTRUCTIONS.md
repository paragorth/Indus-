# In-session extraction instructions (Ai4Qi audit library)

You extract structured data from published clinical audits and quality improvement projects, for a library that UK trainees use to plan audits.

For EACH paper file listed in your batch manifest:
1. Read the whole file with the Read tool (use offset/limit for long files; read all of it).
2. Produce one JSON object matching `work/claude_batches/schema.json` exactly (all keys required, no extra keys, all values non-empty strings, enums exactly as listed).

Rules:
- Use only what the text states. Never infer, estimate or fill from general knowledge. If a field is not stated, write exactly "not reported".
- Results must keep the numbers exactly as the paper gives them (percentages, n/N, medians, times, p values). Cycle 1 = baseline; cycle 2 = re-audit after the change. If more than two cycles, put the final re-audit in cycle 2 and mention the others in other_findings. If there was no second measurement, all cycle2 fields are "not reported" and loop_closed is "no" (or "not reported" if unclear whether one was done).
- sample_size: digits only (e.g. "58"), or "not reported". If cycles have several denominators, give the main one and put detail in result.
- is_audit: "yes" if it measures practice against a standard/guideline/target (audit or audit-based QI); "no" for case series, outcome studies, registry analyses, trials, surveys of opinion, technique papers; "unclear" otherwise.
- specialty: "Orthopaedics – <subspecialty>" using one of: Hip fracture, Trauma, Foot and ankle, Hand and wrist, Paediatric, Spine, Arthroplasty, Shoulder and elbow, Knee, Theatre and perioperative, Ward care, Outpatients, Imaging, Infection, Bone health, Governance, Training. If the paper is not orthopaedic at all, give its real specialty (e.g. "Anaesthesia", "Emergency medicine").
- topic: short, generic, reusable name so audits of the same thing group together (e.g. "Operation note quality", "Hip fracture time to theatre", "VTE prophylaxis in lower limb immobilisation", "Consent documentation", "Fracture clinic follow-up imaging"). Reuse these existing names exactly when they fit: Hip fracture time to theatre; Hip fracture time to ward; Operation note quality; Supracondylar fracture nerve and vessel documentation. Keep topics at the level of "the clinical question audited", not the hospital.
- country: from the affiliations/setting text; "not reported" if absent.
- standard: the guideline/standard audited against, with target if stated.
- intervention: what was changed between cycles.
- fix_type: form or template = proforma, sticker, checklist, electronic template; system change = new pathway, list, rota, IT workflow, staffing, protocol; education only = teaching, posters, emails, reminders; mixed = more than one of these; "not reported" if no intervention described.
- other_findings: one or two sentences of anything else useful (e.g. items that stayed poor), or "not reported".
- British spelling, plain concise sentences.

Output: write a single JSON file at the output path given in your task: an object mapping each paper's KEY (from the first line of the file, e.g. "PMID:12345678") to its extraction object. Write it with the Write tool. Then run
`cd /home/user/Indus-/ai4qi && python3 -c "import json,validate_claude as v,extract;d=json.load(open('<output path>'));bad={k:v.check(x,extract.SCHEMA) for k,x in d.items()};print({k:e for k,e in bad.items() if e} or 'all valid', len(d))"`
and fix any errors. Every paper in the manifest must appear. Reply with just: number written, number is_audit yes/no/unclear.
