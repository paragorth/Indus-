# Ai4Qi corpus — how to use it

This folder is Claude's quick reference for anything about clinical audit and QI on the Ai4Qi site:
a library of real audits and closed-loop QI projects. For audit and QI questions, look here FIRST,
before using pre-trained knowledge or searching the web. Only if the corpus has nothing relevant,
say so, and then (if still needed) fall back to general knowledge, clearly labelled as not from the corpus. The repository
root CLAUDE.md is about a different project (the Indus script) and does not apply here.

## What is in it

- `ai4qi-library.json`: `audits` (9,702 entries, numbered `id`), `topic_knowledge` (topic cards), `about`.
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
- `new_audits/`: 171 PROPOSED orthopaedic audits (ONA-001…), designed from current UK standards
  (NICE, BOAST, NHFD, BSCOS, ROS, RCEM, GIRFT/NHS England, fetched 2026) and gaps in the corpus. Not run
  anywhere yet: never cite them as evidence, only as ideas. Each has one plain question, the exact
  standard wording + URL, pass definition, sample, timeline, change, re-audit and a ready template
  `new_audits/templates/ONA-xxx.csv`. List: `new_audits/catalogue.md`. Search:
  `python3 query.py "words" --proposed`. Rebuild after editing `new_audits/parts/*.json`:
  `python3 new_audits/compile.py`.
- `new_audits/nonortho_new_audits.json`: 401 PROPOSED non-orthopaedic audits (NNA-001…) across 18 areas,
  same structure plus `pitfalls` and `pearls`; templates `new_audits/templates/NNA-xxx.csv`; list
  `new_audits/catalogue_nonortho.md`. `query.py --proposed` searches both sets.
- `standards/standards.json`: exact standard wording with URLs, and which ONA audits use it. Quote
  standards from here before quoting from memory.
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
   (e.g. [583]/[1270], [1130]/[1532], [1132]/[1499], [1131]/[1506], [1850]/[1872]). Count a pair
   once when saying how many audits exist.
6. Typical user questions and where to look: "what should I audit in X?" → `--topics --specialty X`
   then the cards' consultant advice (it names overdone topics and gaps); "what fix works for Y?" →
   `--topic Y` and the fix_type / cycle2 fields; "has anyone audited Z?" → free-text query.

## Output for the Ai4Qi application

Answers go straight into the product, so write them as a finished professional document:
- No working notes or process talk ("the corpus doesn't have it", "I searched", "I couldn't open",
  "made up just now", "not corpus"). Provenance appears only as citations: library ids [1234],
  proposed ids ONA-xxx, and linked standards.
- Where there is no national standard, write "Local standard" in the Standard line; that is content,
  not process. Never present a local target as national.
- Evidence lines cite only real library entries; if none fit, write the closest related entries, or
  omit the Evidence section instead of explaining its absence.
- Process notes for the developer (blocked sites, uncertain wording) go in the commit message or
  new_audits/NOTES.md, never in the answer.

## Build an audit in the app

The app's main box builds a new protocol for any theme the user types (`#/build?q=…`), in the answer
shape above, using the closest published audits, standards and proposed audits as material; evidence
lines that cite ids not in the library are dropped. A request with no theme ("a quick closed-loop
audit") goes to `#/suggest` (ready-made proposed audits, shortest timeline first). Inside Claude the
page uses the `sample` capability; on the hosted site it uses `backend/supabase/functions/build-audit`
(`build_url` in `app/config.json`). Built protocols land in the `built_audits` table: review and add
good ones to the `generated.json` parts.

## Demo mode and the worked example

`#/demo` (or account menu → Demo mode) adds dashed "Demo:" buttons that fill fictitious details and
records for any audit, one click per stage, for live demonstrations; `#/demo/off` ends it. The worked
example (NNA-074 sepsis, fictitious hospital) with the filled Excel sheets, results code, final
presentation and screenshots is in `demo/` (see `demo/README.md`).

## Keep growing the proposed library

Every new audit designed in a conversation is added to `new_audits/parts/generated.json` (orthopaedic) or
`new_audits/nonortho_parts/generated.json` (everything else) (same keys
as the other parts, plus `pitfalls` and `pearls` lists), then run `python3 new_audits/compile.py`,
commit and push. `generated` is compiled last, so existing ONA ids never change. Check for a near-duplicate
with `python3 query.py "words" --proposed` first; if one exists, improve that entry instead.

## Feedback on proposed audits

`new_audits/feedback.json` holds thumbs up/down with reasons (Too generic, Too specific, Poor framing,
Too complex, Not relevant to my specialty, Not an important topic to audit), from the app and from
reviewers. `compile.py` attaches the totals to each audit and `query.py --proposed` ranks down-voted
ones lower. Before proposing a new audit, check the reasons given for similar down-voted ones and
avoid repeating the problem (e.g. "Not an important topic" → pick higher-volume, higher-harm topics).
The app collects feedback on the device and posts it to `feedback_url` in `app/config.json` once a
collector is set; add received entries to `feedback.json`.

## Answer shape (for "can I audit X?")

Short. No preamble. The corpus gives examples; you may also propose new audits nobody has done (say so).
Each option is ONE plain, measurable question: who, what standard, what counts as a pass. No jargon
in the title; define any term. One audit = one question (never "X and Y"). Cite the actual standard
wording and do not overstate it (e.g. NICE says "regularly", not "daily"). UK/Ireland examples first.
Exactly this order:

**Best option:** one audit, one line. **Alternative:** one audit, one line.
**Why:** 1–2 lines (gap or problem the corpus shows).
**How:** standard; sample; data-collection template (the fields, as a list); timeline (weeks).
**Change:** the one fix to put in (prefer form/template or system change; the corpus shows teaching alone rarely works).
**Re-audit:** when, same template, same sample size, target.
**Close the loop:** what to present where, and what to embed if it worked.
**Evidence:** 2–4 lines, each "[id] where: before → after (fix)". Mark draft cards as draft.
**Pitfalls:** 3–4 lines, each "what can go wrong or who will object → how to prevent it" (data gaps,
small numbers, moving the goalposts between cycles, staff pushback, the fix not being used).
Use the topic card's fix_that_fails and failed cycle2 results in the corpus where they exist.
**Pearls:** 3–4 short tips that make it succeed (who to recruit, what to lock down before cycle 1,
how to keep the change alive after the audit team rotates).
If a proposed audit (ONA-xxx) fits, offer it with its template path and label it "proposed, not yet run".

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
Library ids are permanent: `work/id_map.json` maps each paper's record key to its id, and new papers
get the next free number, so an id cited today stays valid.
