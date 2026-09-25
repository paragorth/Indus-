"""Attempts to break a passing condition (see RESULTS.md, 'Break tests').

  english      same search, English lexicon (a language nobody proposes)
  sanskrit     same search, Sanskrit lexicon
  iid-tamil    Tamil lexicon on a structure-free corpus: every sign drawn
               independently from the real sign frequencies, same text lengths
  im77-tamil   Tamil lexicon on Mahadevan's independently transcribed corpus
"""
import random
import sys
from multiprocessing import Pool

import indus_core as C
import run_all as RA


def iid_corpus(texts, seed=0):
    rng = random.Random(seed)
    f = C.sign_freq(texts)
    signs, w = list(f), list(f.values())
    return [[rng.choices(signs, w, k=len(s)) for s in segs] for segs in texts]


def main(which, N=80, restarts=20, fakes=30):
    RA.log(f"## Break tests for `tamil-full-{N}` ({RA.now()})")
    texts = C.unique_texts(C.load_corpus())
    with Pool(4) as pool:
        for w in which:
            lang, tx = "tamil", texts
            if w in ("english", "sanskrit"):
                lang = w
            elif w == "iid-tamil":
                tx = iid_corpus(texts)
            elif w == "im77-tamil":
                tx = C.unique_texts(C.load_im77_corpus())
            tr, he = C.split_texts(tx)
            top = [s for s, _ in C.sign_freq(tr).most_common()]
            RA.run_condition(pool, f"break-{w}-full-{N}", lang, "full", N, tr, he, top,
                             restarts, fakes, 400 * N, control=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["english", "iid-tamil", "sanskrit", "im77-tamil"])
