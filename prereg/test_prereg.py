"""Run frozen preregistered predictions against any corpus dropped into new_data/.

    python prereg/test_prereg.py [corpus.csv ...]

With no arguments it tests every *.csv in new_data/; if new_data/ is empty it
tests Yajnadevam's corpus (data/raw/inscriptions.csv) to show the harness works.

Predictions come from preregistration-2-frozen.json at the repo root.  That
file was not available when this harness was written, so each prediction must
carry a machine-checkable "test" object in one of the forms below; anything
else is reported as NOT CHECKABLE rather than guessed at.

  {"type": "terminal_share",  "sign": "740", "min": 0.5}
        share of the sign's occurrences that are text-final (reading order)
  {"type": "initial_share",   "sign": "861", "min": 0.4}
  {"type": "rank_at_most",    "sign": "740", "rank": 1}
        frequency rank (1 = most frequent) among non-damage signs
  {"type": "bigram_ratio_min","pair": ["002", "740"], "min": 5.0}
        observed / expected count of the adjacent pair (reading order)
  {"type": "mean_length_between", "min": 3.0, "max": 6.0}
        mean signs per text

Frozen-file shape expected: {"predictions": [{"id": ..., "statement": ..., "test": {...}}, ...]}
Corpus files: Yajnadevam CSV format (a `text` column with 3-digit sign codes,
stored left-to-right, reversed here to reading order).
"""
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import indus_core as C  # noqa: E402

FROZEN = os.path.join(ROOT, "preregistration-2-frozen.json")

# Used only when the frozen file is absent, to show the harness runs.
# These are NOT the preregistered predictions.
EXAMPLE = {"predictions": [
    {"id": "example-1", "statement": "the jar sign is the most frequent sign",
     "test": {"type": "rank_at_most", "sign": "740", "rank": 1}},
    {"id": "example-2", "statement": "the jar sign is mostly text-final",
     "test": {"type": "terminal_share", "sign": "740", "min": 0.5}},
    {"id": "example-3", "statement": "mean text length is 3-6 signs",
     "test": {"type": "mean_length_between", "min": 3.0, "max": 6.0}},
    {"id": "example-4", "statement": "free-text claim with no test object",
     "test": None},
]}


def texts_of(path):
    return [segs for _, segs in C.load_corpus(path)]


def check(test, texts):
    flat = [s for segs in texts for s in segs]
    freq = Counter(x for s in flat for x in s)
    t = test["type"]
    if t in ("terminal_share", "initial_share"):
        sg = test["sign"]
        pos = -1 if t == "terminal_share" else 0
        # use each whole text's first/last segment only
        n = freq[sg]
        edge = sum(1 for segs in texts if segs and segs[pos] and segs[pos][pos] == sg)
        v = edge / n if n else 0.0
        return v >= test["min"], v
    if t == "rank_at_most":
        ranks = [s for s, _ in freq.most_common()]
        r = ranks.index(test["sign"]) + 1 if test["sign"] in ranks else None
        return r is not None and r <= test["rank"], r
    if t == "bigram_ratio_min":
        a, b = test["pair"]
        big = Counter((s[i], s[i + 1]) for s in flat for i in range(len(s) - 1))
        nb = sum(big.values())
        tot = sum(freq.values())
        exp = nb * (freq[a] / tot) * (freq[b] / tot) if tot else 0
        v = big[(a, b)] / exp if exp else 0.0
        return v >= test["min"], v
    if t == "mean_length_between":
        v = sum(sum(len(s) for s in segs) for segs in texts) / max(1, len(texts))
        return test["min"] <= v <= test["max"], v
    raise KeyError(t)


def main(paths):
    if os.path.exists(FROZEN):
        spec, label = json.load(open(FROZEN)), "preregistration-2-frozen.json"
    else:
        spec, label = EXAMPLE, "EXAMPLE predictions (frozen file not found)"
    if not paths:
        paths = sorted(glob.glob(os.path.join(ROOT, "new_data", "*.csv"))) or \
            [os.path.join(C.RAW, "inscriptions.csv")]
    print(f"predictions: {label}")
    fails = 0
    for p in paths:
        texts = texts_of(p)
        print(f"\ncorpus: {os.path.relpath(p, ROOT)}  ({len(texts)} objects)")
        for pr in spec["predictions"]:
            test = pr.get("test")
            if not isinstance(test, dict) or "type" not in test:
                print(f"  NOT CHECKABLE  {pr['id']}: {pr.get('statement', '')}")
                continue
            try:
                ok, v = check(test, texts)
            except KeyError:
                print(f"  NOT CHECKABLE  {pr['id']}: unknown test type {test['type']!r}")
                continue
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL':13s}  {pr['id']}: {pr.get('statement', '')}  (value={v})")
    return fails


if __name__ == "__main__":
    main(sys.argv[1:])
