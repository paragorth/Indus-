# Indus script: blind key search

Can any simple sign-to-sound key read the Indus corpus as Tamil or Sanskrit,
measured on held-out texts against controls that a fitted key can't game?

```
./scripts/fetch_data.sh          # public data -> data/raw/ (not committed)
python validator.py --published  # fixed-key test of Yajnadevam's published key
python run_all.py                # everything; logs to RESULTS.md, keys to results/
python prereg/test_prereg.py     # preregistered predictions vs new_data/*.csv
```

## Data (all public)

| role | source |
|---|---|
| corpus | Yajnadevam's `inscriptions.csv` (5,680 objects, Wells sign numbers), github.com/yajnadevam/lipi |
| published key | the same repo's `xlits.csv` |
| Tamil lexicon | TamilVU dictionary headwords (63,896), from open-tamil. **Stand-in for DEDR**, which is not reachable from this environment |
| Sanskrit lexicon | Monier-Williams `<k1>` headwords, sanskrit-lexicon/csl-orig |

The Mahadevan (IM77) corpus, the merged reading-order corpus, and
`preregistration-2-frozen.json` weren't available. Bonta's map isn't implemented.

## Method

* Texts are deduplicated and reversed into reading order. Damage, spaces and line breaks
  split segments. The split is 50/50 train / held-out by text.
* Both languages go into one reduced alphabet (`indus_core.py`).
* Keys are fitted by simulated annealing on the training half (`search.py`). Score = 2–5-sign
  windows whose rendered string is a lexicon form.
* **Two tests, both required** (`experiment.py`):
  1. *Fixed-key* (`validator.py`, as specified): held-out rate vs scrambled order and vs 30
     fake lexicons (letters shuffled within words, CV template kept, same size).
  2. *Search-matched null*: the identical search re-run against each of 30 fake lexicons.
     gap-over-fake = (mean real held-out − mean fake held-out) / sd(fake).
  Test 1 alone is not enough. A key fitted to a *fake* lexicon passes it,
  because held-out texts repeat the same formulae the key was fitted to.
* **Positive controls** plant Tamil or Sanskrit under a secret key into a corpus with
  the real length distribution, and run the same pipeline. A search type whose
  control fails is reported as having no power, not as evidence either way.

Never read any output of this pipeline as a translation.
