#!/usr/bin/env python3
"""LA-69 cycle 1: score every document of the calibration sets, planted notes and Linear A on the
13 base reader-independence features; per-feature calibration (NOTE vs FINAL, PLANT vs source)."""
import os, sys, json, random, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la69_common as C

rng = random.Random(69)
NMAX = 400
OUT = os.path.join(C.CK, 'c1_feats.pkl')


def cap(ds, n):
    ds = list(ds); rng.shuffle(ds); return ds[:n]


def main():
    S = C.calib_sets()
    rows = []          # (corpus, cls, doc meta, feats)
    for cname, st in S.items():
        ref = st['ref']
        excl = None
        for cls in ('NOTE', 'FINAL'):
            docs = cap(st[cls], NMAX)
            fs = C.score_docs(docs, ref, rng)
            rows += [(cname, cls, d, f) for d, f in zip(docs, fs)]
            if cls == 'FINAL':
                pl = C.plant_notes(docs, rng)
                src = {('PL:' + str(d['id'])): d for d in docs}
                fs = C.score_docs(pl, ref, rng)
                rows += [(cname, 'PLANT', d, f) for d, f in zip(pl, fs)]
        print(cname, 'done', flush=True)
    la = C.la_docs()
    fs = C.score_docs(la, la, rng, M=min(300, len(la) - 1))
    rows += [('LA', 'LA', d, f) for d, f in zip(la, fs)]
    # LA planted notes (from LA itself) and LA lines-shuffled-across-documents null
    pl = C.plant_notes(la, rng)
    rows += [('LA', 'LA_PLANT', d, f) for d, f in zip(pl, C.score_docs(pl, la, rng, M=min(300, len(la) - 1)))]
    for name, f in [('OB', C.ob_docs), ('PC', C.pc_docs), ('KH', C.kh_docs)]:
        ds = cap(f(), NMAX)
        rows += [(name, 'ARCH', d, ff) for d, ff in zip(ds, C.score_docs(ds, f(), rng))]
        print(name, len(ds), flush=True)
    pickle.dump([(a, b, {k: v for k, v in d.items() if k != 'toks'} | {'nT': sum(1 for x in d['toks'] if x[0] == 'T')}, f)
                 for a, b, d, f in rows], open(OUT, 'wb'))


if __name__ == '__main__':
    main()
