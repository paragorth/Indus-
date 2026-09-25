# Indus

Computational epigraphy of the Indus script: an audited corpus, a positional grammar, a stroke morphology, a two-field reading of the tablets, an identity directory with shipments, an offline seal reader, two frozen pre-registrations, and a blind validator that any proposed dictionary must pass. There is no dictionary here: the corpus does not contain one, and this repository records every attempt to find it and why each failed.

**Start here:** `docs/INDUS-DECIPHERMENT-ATTEMPT.md` (the full record), then `docs/INDUS-CONTINUATION-PROMPT.md` (the state of the project in one paste).

## Contents
- `data/im77/` — Mahadevan's corpus, machine-readable: 2,906 texts, 14,153 tokens, with direction, field symbol, object type and excavation level.
- `data/indus-repro-package.zip` — the Zenodo package (DOI 10.5281/zenodo.21497936): the open database, `run_all.py`, all reported numbers.
- `data/derived/` — the extended sign bridge (143 signs → Mahadevan numbers), the merged 5,369-object corpus in reading order, the identity directory (386 recurring identities with sealing sites), affiliations, sign dossiers.
- `code/` — `p3lib.py`, `p3_engine.py` (Part 3 engine), `read_texts.py` (template reader for the 1977 scan; needs the sign-list page images, not included), `validator.py` (the blind test for keys), `fetch_external.sh`.
- `prereg/` — the two frozen prediction sets (July 2026: P1–P9; September 2026: Q1–Q14) with falsifiers.
- `reader/indus-reader.html` — open it in any browser: CISI number or sign sequence → sign shapes, slots, document type, identity, shipments.
- `papers/` — Paper A (the corpus is not one statistical population) and Paper B (functional sign classes with out-of-sample validation).

## Reproduce
```
pip install -r requirements.txt
(cd code && ./fetch_external.sh)
unzip data/indus-repro-package.zip -d data/repro
python data/repro/repro/scripts/run_all.py
```

## Rules this project keeps
Coherence is not evidence; every number is computed, never narrated. Sealings are deduplicated before any emblem or object test. Mohenjo-daro seals first; every other stratum is a transfer test. A dictionary counts only if it passes `code/validator.py` on unseen texts — two published decipherments do not.

## Key search (September 2026)
`SEARCH.md` describes a blind search for a sign-to-sound key: Tamil and Sanskrit, syllabic, logographic and mixed keys, simulated annealing with a search-matched fake-lexicon null and planted-language positive controls. Results log: `RESULTS.md`. **Correction (25 Sept):** passing keys do not recover planted readings, and when only the common signs are decoded, wrong keys beat the true key under every objective tested. Only a small (80-sign) syllabary makes the true key optimal; at 150+ signs wrong keys win even with the exact vocabulary (STRATEGIES.md S107, S113–S127). Tests for Q1–Q14: `prereg/test_prereg.py` (`PREREG_RESULTS.md`).

## Outside anchors and the two-hour strategy push (25 September 2026)
- `ANCHORS.md`: foreign-found texts, cuneiform Meluhha attestations, Linear Elamite, the Gulf seals; §18 reinterprets the "foreign grammar" as the round-seal tradition.
- `STRATEGIES.md`: 100+ data-only strategies, each with a control and a verdict, including corrections. Summary at the end.
- `FUNCTIONAL-DICTIONARY.md`: what each of the 120 commonest signs *does* (function class, positions, counts), computed from data. No sounds or meanings.
- `GRAMMAR.md`: the resulting data-only frame grammar (opener, middle, closer, suffix), with the controls that separate it from artefacts.
- `prereg/preregistration-3-frozen.json`: predictions R1–R13 from this work, frozen for new data.
- `CLEANROOM.md`, `MYTHS.md`, `RELATED-ARTIFACTS.md`: analysis without published readings; myth and scene parallels; non-Indus artefacts with related motifs.
