# Topic card instructions (Ai4Qi audit library)

You write short topic summaries for a clinical audit library used by UK trainees and reviewed by consultants.
For each topic file you are given (JSON: topic name + every detailed audit in the library on that topic), read it fully and write ONE JSON file to
`/home/user/Indus-/ai4qi/work/topics_out/cards/<same file name>.json` with exactly these keys:

{"topic": "<topic name exactly as in the input>", "usual_baseline": "...", "fix_that_works": "...", "fix_that_fails": "...", "consultant_advice": "..."}

Using ONLY the audits in the file:
- usual_baseline: what cycle 1 usually finds, with typical numbers.
- fix_that_works: interventions followed by real improvement at re-audit, with the numbers, citing audits by library id in brackets, e.g. [1136].
- fix_that_fails: interventions followed by little or no improvement, or items that stayed poor, with ids. If none is reported, write "No failed fix reported in these audits" — do not invent one.
- consultant_advice: 2–3 sentences a consultant would tell a trainee choosing this audit: is it overdone, what would be more useful to measure.
Style (match the existing seed cards): plain British English, short sentences, numbers kept exactly, no hedging filler, no knowledge from outside the audits. Each field at most ~70 words.
If "is_seed_topic" is true, an earlier card exists; write a fresh card from all the audits in the file.
Do not modify any other files; do not commit. Reply with the list of files written.
Keep scratch files only in a folder named after your batch (e.g. <scratchpad>/cards_3/).
