"""v42: v31 feature battery (32 line-free features, 100-token samples) for
  clean: Codex Seraphinianus in three unit schemes (raw strokes, repeats collapsed, Ponzi's bigram scheme);
         Voynich with repeats collapsed (to match);
  n18 / n09: every v31 training corpus, generator and the Voynich passed through the OCR-like channel at 18% / 9%
         character error (the CS transliteration's own validation error is ~18%).
Clean training features are re-used from data/v31_ckpt/feats_N100.json (same library, same sampling).
Output data/v42_ckpt/feats.json."""
import os, sys, json, random, re, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L, v31_lib as L31

N = 100
MAXS = {'LANG': 8, 'INVENT': 8, 'GEN': 8, 'MAGIC': 60, 'GIBB': 6, 'TEST': 80}
SB = [('iii', 'w'), ('rrg', 'Q'), ('ii', 'v'), ('rg', 'q'), ('ff', 'F'), ('rj', 'J'), ('El', 'V'), ('in', 'm'),
      ('ee', 'U'), ('LB', 'K'), ('Lj', 'k')]


def sb(w):
    for a, b in SB: w = w.replace(a, b)
    return w


def jobs():
    C = L.all_corpora()
    J = []
    def add(name, cls, docs, var):
        S = L31.samples(docs, N=N, maxs=MAXS[cls])
        for i, s in enumerate(S): J.append((name, cls, var, i, s))
    for k in ('S_CS', 'S_CS1', 'S_CS2'):
        cls, d = C[k]
        add(k, cls, d, 'raw'); add(k, cls, L.map_docs(d, L.collapse), 'col'); add(k, cls, L.map_docs(d, sb), 'sb')
    for k in ('V_ZL', 'V_IT'):
        add(k, 'TEST', L.map_docs(C[k][1], L.collapse), 'col')
    for rate, var in ((0.18, 'n18'), (0.09, 'n09')):
        for k, (cls, d) in C.items():
            if k.startswith('S_'): continue
            add(k, cls, L.noise_docs(d, rate, seed=zlib.crc32(k.encode()) & 0xffff), var)
    return J


def work(a):
    name, cls, var, i, lines = a
    rng = random.Random(zlib.crc32(f"{name}|{var}|{i}".encode()))
    F = L31.features(lines, rng)
    LF = L31.line_features(lines, rng)
    return {'corpus': name, 'cls': cls, 'var': var, 'i': i, 'F': {k: float(v) for k, v in F.items()},
            'LF': {k: float(v) for k, v in LF.items()} if LF else None}


if __name__ == '__main__':
    J = jobs(); print('jobs', len(J), flush=True)
    with Pool(2) as p: R = p.map(work, J, chunksize=8)
    R31 = L31.load('feats_N100.json')
    for r in R31:
        r['var'] = 'clean'
    L.save('feats.json', R + R31)
    print('done', len(R) + len(R31))
