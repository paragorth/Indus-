# Ai4Qi clinical audit library pipeline

Grows `ai4qi-library.json` with real published audits and closed-loop quality
improvement work, keeping the original library's schema and every entry already in it.

## Files

| File | What it is |
|---|---|
| `ai4qi-library.json` | The library: `about`, `topic_knowledge` (topic cards) and `audits` (numbered entries). |
| `ai4qi-library.csv` | Flat copy of every entry, one row each. |
| `index.html` | Open it in a browser (no server needed) to filter by library, specialty, topic, fix type and status, and to review the topic cards. |
| `figures/PMCxxxx/` | Figures from CC BY and CC0 papers only, with caption, licence and credit line stored on the entry. |
| `seed/` | The library as supplied (ids 1–1145) and the script that rebuilt it from the pasted copy. |
| `work/` | Caches: search hits, metadata, screening decisions, extractions, failure log. |

## Schema

Seed entries are unchanged. New entries use the same keys (`specialty`, `title`, `standard`,
`finding`, `change`, `source`, `status`, `id`, and `topic`/`detail` when detailed). `detail` has
the seed shape: `setting`, `country`, `cycle1 {period, n, result}`, `intervention`,
`cycle2 {period, n, result}`, `loop_closed`, `other_findings`, `citation`, `fix_type`,
`fix_type_note`. Anything a paper doesn't state is `"not reported"`.

New entries also carry:

- `paper`: authors, journal, year, DOI, PMID, PMCID, abstract, licence, open-access flag,
  UK/Ireland flag, full citation, and how the paper was found.
- `figures` or `figure_link`: downloaded figures (CC BY/CC0 only) or just a link.

When a search hit matches a seed entry (same PMID, PMCID or DOI in its source link), the seed
entry only gains a `paper` block. Nothing in it is overwritten.

Topic cards: the three seed cards are kept word for word. New cards are marked
`"status": "Draft, needs consultant sign-off"` and list the `evidence_ids` they were written from.

## Running

```
pip install requests anthropic
python3 pipeline.py search  --pass ortho
python3 pipeline.py fetch   --pass ortho
python3 pipeline.py figures --pass ortho
python3 pipeline.py extract --pass ortho [--batch]   # needs ANTHROPIC_API_KEY or `ant auth login`
python3 pipeline.py merge
python3 pipeline.py output
python3 pipeline.py report
# then the same with --pass nonortho, and finally:
python3 pipeline.py topics && python3 pipeline.py output
```

Sources are the Europe PMC REST API, PubMed E-utilities (kept under 3 requests a second) and the
PMC open-access bucket, which serves licences, JATS XML and figure images when Europe PMC is down.
Publisher websites are never contacted.

Extraction uses `AI4QI_MODEL` (default `claude-opus-5`) with structured JSON output. Direct calls
send `fallbacks: "default"`, so if the model refuses a request the API re-runs it on a fallback
model. `--batch` uses the Message Batches API at half the price.

## Moving it to another server or repository

The `ai4qi/` folder is self-contained: no absolute paths, no database, no API keys.

- **Read-only use (the site's reference):** copy `ai4qi-library.json`, `query.py`, `new_audits/`
  (`ortho_new_audits.json`, `templates/`), `standards/`, `figures/`, `index.html`, `CLAUDE.md`.
  Needs Python 3.8+ and nothing else. `index.html` opens in any browser offline.
- **Rebuilding or growing it:** copy the whole folder (including `seed/` and `work/`), then
  `pip install -r requirements.txt`. Files git ignores (`work/fulltext/`, `work/claude_in/`,
  `work/claude_out/`, raw dumps) are caches; the pipeline re-downloads them if missing.
- **Own GitHub repository with history:** from the repo root,
  `git subtree split -P ai4qi -b ai4qi-only`, then push branch `ai4qi-only` to the new repository's `main`.
- Tracked size is about 85 MB, of which about 30 MB is figures; all files are under GitHub's 100 MB per-file limit.

## Web app

`app/` is a static single-page site (plain HTML, CSS and JavaScript; no build step) for searching the
library, reading proposed audits as printable protocols, and browsing topic cards and standards.

Rebuild its data after any change to the library, the proposed audits or the standards:

```
python3 build_app_data.py        # writes app/data/, app/templates/ and app/figures/ (standard library only)
```

Serve it locally:

```
cd app && python3 -m http.server 8000    # then open http://localhost:8000/
```

To publish, copy the whole `app/` folder to any static host (GitHub Pages, S3, nginx). It must be served
over HTTP; opening `index.html` straight from disk will not load the data.
