# Topic grouping instructions (Ai4Qi audit library)

Input: a JSON file with "existing_canonical" (canonical topic names already in use) and "topics" (topic names written by readers of individual audits, each with a count n and the specialties it appeared in).

Task: map EVERY name in "topics" to a canonical topic name so that audits of the same clinical question group together.
- Reuse an existing canonical name exactly when it fits (e.g. "Consent documentation", "VTE prophylaxis in orthopaedics", "Operation note quality", "Discharge summary quality").
- Otherwise create a short, generic, reusable canonical name at the level of "the clinical question audited" (e.g. "Clozapine monitoring", "Antipsychotic physical health monitoring", "Emergency laparotomy (NELA standards)", "Sepsis six in the emergency department"). Merge near-duplicates and different wordings of the same thing; do NOT merge different clinical questions.
- Keep the specialty context in the name only when the same topic in another specialty is a different audit (e.g. "Operation note quality" is shared across surgery; "VTE risk assessment" is shared).
- British spelling.

Output: write a JSON object {"<topic name exactly as given>": "<canonical name>", ...} covering every input name, to the output path you are given. Validate with python that every input name is a key. Do not modify other files; do not commit. Reply with the number mapped and the number of distinct canonical names.
