# Ai4Qi corpus — how to use it

This folder is Claude's reference corpus for the Ai4Qi site: a library of real clinical audits and
closed-loop QI projects, used to (1) build the site and (2) answer users' questions. The repository
root CLAUDE.md is about a different project (the Indus script) and does not apply here.

## What is in it

- `ai4qi-library.json`: `audits` (3,713 entries, numbered `id`), `topic_knowledge` (topic cards), `about`.
  - ids 1–1145: the original library supplied by the user. Never edit these; treat them as ground truth.
  - ids 1146+: found by the pipeline (PubMed, Europe PMC, Crossref conference abstracts) and read by
    Claude. `status` says how solid each is: `published, detailed` (read and confirmed an audit),
    `published, audit status unclear` (thin abstract), seed statuses such as `recurring topic, not
    individually linked` (no source; a common audit idea, not evidence).
  - `detail`: setting, country, cycle1 {period, n, result}, intervention, cycle2, loop_closed, fix_type.
    "not reported" means the paper did not say; never fill it in.
  - `paper`: authors, journal, year, DOI/PMID/PMCID, abstract, licence, UK/Ireland flag, citation.
- Topic cards: 3 seed cards (written by the user's team; authoritative) and 99 drafts with
  `"status": "Draft, needs consultant sign-off"` and `evidence_ids`.
- `ai4qi-library.csv` (flat copy), `index.html` (browse/filter), `figures/` (CC BY/CC0 only).

## Answering a question

Run the query tool first; answer from what it returns, not from general knowledge.

    python3 query.py "free-text question words" [--specialty psychiatry] [--closed] [-n 15]
    python3 query.py --topic "Operation note quality"     # card + every audit on the topic
    python3 query.py --topics --specialty surgery         # what topics exist
    python3 query.py --id 1136                            # full record
    add --json for structured output

Rules:
1. Cite every factual claim with library ids in brackets, e.g. [1132], and give the source link or
   citation for the key ones. If the corpus has nothing relevant, say so plainly.
2. Prefer detailed, closed-loop, UK/Ireland audits. Quote numbers exactly as stored.
3. Say how strong the evidence is: how many audits, how many closed the loop. Five single-cycle
   audits are weaker than two closed loops.
4. Topic cards: present seed cards as the team's judgement; present draft cards as
   "draft, not yet consultant-reviewed". If a draft and a seed card disagree, show both.
5. Known duplicates: some projects appear twice, once as a conference abstract and once as a paper
   (e.g. [583]/[1270], [1130]/[1544], [1132]/[1511], [1131]/[1518], [1873]/[1895]). Count a pair
   once when saying how many audits exist.
6. Typical user questions and where to look: "what should I audit in X?" → `--topics --specialty X`
   then the cards' consultant advice (it names overdone topics and gaps); "what fix works for Y?" →
   `--topic Y` and the fix_type / cycle2 fields; "has anyone audited Z?" → free-text query.

## Using content on the public site

- Facts, numbers, standards and citations can be summarised in our own words with a link.
- Reproduce abstract text or figures verbatim only when `paper.licence` is `cc by` or `cc0`
  (figures in `figures/` already satisfy this and carry a `credit` line that must be shown).
  Everything else: summarise and link.
- Draft topic cards must show their draft status until a consultant signs them off.

## Updating the corpus

See README.md. Pipeline: `pipeline.py` (search/fetch/screen/figures/merge/output),
`crossref_harvest.py` (conference abstracts), in-session reading via
`work/claude_batches/INSTRUCTIONS.md` + `validate_claude.py --ingest`, topics via `topics_local.py`.
`pipeline.py merge` rebuilds new entries from the seed each time, so ids 1146+ can shift if
inputs change; cite by id only within one version of the library.
