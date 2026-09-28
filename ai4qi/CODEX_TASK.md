# Task for Codex: turn user-suggested audits into proposed audits

Repository `paragorth/Indus-`, folder `ai4qi/`. Branch from `claude/ai4qi-audit-library-pipeline-8yph4w`
into a new branch `codex/ai4qi-suggestions`, and open a PR back into that branch. Read `ai4qi/CLAUDE.md`
and `ai4qi/new_audits/compile.py` first.

## Background

Ai4Qi is a free clinical audit app for UK and Irish doctors. When a user types a topic, the app builds
an audit protocol. On the hosted site each build is saved in the Supabase table `public.built_audits`
(see `backend/supabase/migrations/003_*.sql`). The columns are `topic`, `topic_key` (`B-…`), `reused`,
`protocol` (jsonb, null when reused) and `created_at`. The protocol shape is produced by
`normaliseBuilt()` in `ai4qi/app/app.js` (around line 828).

These suggestions have not been published anywhere, so they belong in the **proposed** audit
collection (ids `NNA-xxx` and `ONA-xxx`), not in the published library.

## What to build

1. **`ai4qi/import_suggestions.py`** (Python 3.11, standard library plus `requests`):
   - Inputs:
     - `--from-supabase` reads rows where `reused = false` and `protocol is not null`, using the env
       vars `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` via the PostgREST API. Never print or log
       the key.
     - `--from-file rows.json` takes a JSON array of rows exported from the dashboard, or bare
       protocols.
     - `--since YYYY-MM-DD` filters by date.
     - `--dry-run` prints what would be added without writing anything.
   - Map each protocol to the proposed-audit schema:
     - Keep `KEYS` in `compile.py`, plus `pitfalls` and `pearls`.
     - Add `origin: "user suggestion"`, `suggested_topic` (the typed topic) and `suggested_on` (the date).
     - Drop the app-only fields `id`, `variant`, `built`, `alternative`, `resources` and `similar`.
     - Keep `evidence` lines only when every `[id]` they cite exists in `ai4qi/ai4qi-library.json`.
     - Set `novelty: "user suggested"` and `effort` from the protocol, or "not stated".
   - **Validate** and skip, with a reason, any protocol missing: question, standard source or
     wording, pass, a template with at least 3 fields including a yes/no pass field, target, or
     timeline. Skip questions that join two audits with " and ".
   - **Route:** orthopaedic topics (area or question mentions orthopaedic, fracture, arthroplasty,
     spine, joint replacement and so on) go to `new_audits/parts/suggested.json`. Everything else
     goes to `new_audits/nonortho_parts/suggested.json`.
   - **Dedupe:**
     - Skip exact duplicates of an existing question, normalised as `compile.py` does.
     - Skip near-duplicates: token Jaccard ≥ 0.6 against every existing proposed question in
       `new_audits/ortho_new_audits.json`, `new_audits/nonortho_new_audits.json` and the
       suggested files. Report each as "near-duplicate of NNA-xxx".
     - When several users built the same `topic_key`, keep only the most recent protocol.
   - **Privacy:** never write `user_id`, emails, or anything that is not in the protocol. Also refuse
     any protocol whose text contains an NHS number pattern (`\b\d{3}\s?\d{3}\s?\d{4}\b`), an email
     address or a UK postcode.
   - Append to the suggested files, sorted by date. Existing entries are never rewritten. Then run
     `python3 new_audits/compile.py`.
   - Print a summary: read, added (with new ids), skipped by reason.

2. **`compile.py`:** add `"suggested"` as the LAST entry of both area orders, after `generated`, so
   existing ids never shift. Carry `origin`, `suggested_topic` and `suggested_on` through to the
   compiled JSON. The catalogue markdown should label these "Suggested by a user – not yet run".
   Change nothing else. The permanent id maps must keep every existing id.

3. **Tests:** add `ai4qi/tests/test_import_suggestions.py` (pytest) with fixtures covering:
   - a valid protocol;
   - a duplicate and a near-duplicate;
   - one with evidence citing a missing id;
   - one with an NHS-number-like string;
   - an orthopaedic one.

   Also check that compiling twice gives the same ids.

4. Add a short "User suggestions" section to `ai4qi/README.md` explaining how to run the importer
   (weekly by hand, or as a scheduled GitHub Action with the two secrets).

## Do not touch

- `ai4qi/app/` (the app side is being changed elsewhere).
- `ai4qi/work/`, `ai4qi/ai4qi-library.*`.
- Any existing file under `new_audits/parts/` or `new_audits/nonortho_parts/`.
- The id maps, except through `compile.py`.

Other rules:
- Never commit secrets or keys.
- Never disable TLS verification.
- Do not put model names in commit messages.
- Use British spelling in user-facing text.

## Done when

- `python3 import_suggestions.py --from-file tests/fixtures/rows.json --dry-run` prints a correct
  summary.
- pytest passes.
- `python3 new_audits/compile.py` runs cleanly, and every existing NNA/ONA id is unchanged.
