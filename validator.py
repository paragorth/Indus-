"""Blind test for a sign -> string key.

  rate(key, texts, lexicon) = fraction of 2-5-sign windows (inside one segment)
                              whose rendered string is a lexicon form.

Fixed-key test (as specified):
  (a) held-out real rate
  (b) same texts, sign order shuffled within each segment (mean of 5 shuffles)
  (c) the key against 30 fake lexicons (letters shuffled within each word)
  pass  <=>  a > b  and  (a - mean(c)) / sd(c) > 3

That test is necessary but NOT sufficient for a key that was *searched*: a key
fitted to lexicon L on the training half re-finds the same formulaic sign
n-grams in the held-out half, so it beats fake lexicons it never saw almost by
construction.  run_all.py therefore adds the search-matched null (the same
search re-run against each fake lexicon) and requires both.

CLI:  python validator.py --published   (Yajnadevam's xlits.csv vs Sanskrit)
"""
import argparse
import csv
import os
import random
import statistics

import indus_core as C

WMIN, WMAX = 2, 5


def windows(texts, allowed=None):
    """All 2-5-sign windows; returns (scorable windows, total window count).
    A window with a sign outside `allowed` counts in the denominator only."""
    out, total = [], 0
    for segs in texts:
        for s in segs:
            for n in range(WMIN, WMAX + 1):
                for i in range(len(s) - n + 1):
                    w = tuple(s[i:i + n])
                    total += 1
                    if allowed is None or all(x in allowed for x in w):
                        out.append(w)
    return out, total


def scramble(texts, seed):
    rng = random.Random(seed)
    out = []
    for segs in texts:
        # shuffle all signs of a text, then cut back into the same segment lengths
        flat = [x for s in segs for x in s]
        rng.shuffle(flat)
        new, k = [], 0
        for s in segs:
            new.append(flat[k:k + len(s)]); k += len(s)
        out.append(new)
    return out


def rate(key, texts, formset, minlen):
    ws, total = windows(texts, key)
    hits = 0
    for w in ws:
        r = "".join(key[x] for x in w)
        if len(r) >= minlen and r in formset:
            hits += 1
    return hits / total if total else 0.0


def fixed_key_test(key, heldout, lex, mode, n_fake=30, n_scramble=5, fake_seed0=10_000):
    minlen = C.MINLEN[mode]
    fs = C.forms(lex, mode)
    real = rate(key, heldout, fs, minlen)
    scr = statistics.mean(rate(key, scramble(heldout, 100 + i), fs, minlen) for i in range(n_scramble))
    fakes = [rate(key, heldout, C.fake_forms(lex, mode, fake_seed0 + i), minlen)
             for i in range(n_fake)]
    mu, sd = statistics.mean(fakes), statistics.pstdev(fakes)
    z = (real - mu) / sd if sd > 0 else (float("inf") if real > mu else 0.0)
    return {
        "real": real, "scrambled": scr, "fake_mean": mu, "fake_sd": sd, "z_fixed": z,
        "pass_fixed": bool(real > scr and z > 3),
    }


def published_key(path=None):
    """Yajnadevam's xlits.csv (his transliteration is SLP1)."""
    path = path or os.path.join(C.RAW, "xlits.csv")
    key = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s, x = r["sign"].strip(), r["xlit"]
            if s in (C.DAMAGE, C.SPACE):
                continue
            key[s] = C.slp1_to_latin(x.strip())
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", action="store_true")
    ap.add_argument("--fakes", type=int, default=30)
    a = ap.parse_args()
    texts = C.unique_texts(C.load_corpus())
    _, held = C.split_texts(texts)
    key = published_key()
    for lang in ("sanskrit", "tamil"):
        lex = C.load_lexicon(lang)
        for mode in ("full", "skel"):
            k = key if mode == "full" else {s: C.skeleton(v) for s, v in key.items()}
            for name, tx in (("held-out", held), ("all", texts)):
                r = fixed_key_test(k, tx, lex, mode, n_fake=a.fakes)
                print(f"yajnadevam  {lang:8s} {mode:4s} {name:8s} "
                      + " ".join(f"{kk}={v:.4f}" if isinstance(v, float) else f"{kk}={v}"
                                 for kk, v in r.items()))


if __name__ == "__main__":
    main()
