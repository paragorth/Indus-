"""v39 cycle 2: DATA-DRIVEN merges.  The interchangeability index I (v35) is computed for all unit pairs on one
half of the pages (discovery); units are merged agglomeratively (union of pairs with I >= threshold, thresholds
0.9 .. 0.5) and the battery is run on the OTHER half (held out).  Same on IT2a (second transcription).
Positive control: Latin / German / Italian with a planted free homophone split (5 letters, Voynich-matched): the
index must rank the planted pairs first and merging at the same thresholds must restore the language profile.
Negative control: unplanted languages merged at the same thresholds (merging distinct letters must NOT make them
more language-like).  Usage: python3 v39_cycle2.py [nworkers]"""
import sys, time, random
from multiprocessing import Pool
import numpy as np
import v39_lib as L
from v39_cycle1 import base as base1, plant_spec, EXCL

THR = (0.9, 0.8, 0.7, 0.6, 0.5)
_B = {}


def corpus(name):
    if name in _B: return _B[name]
    if name == 'V': P = L.voynich('ZL3b')
    elif name == 'VI': P = L.voynich('IT2a')
    elif name.startswith('P'):          # planted language
        lg = name[1:]
        P0 = base1(lg)
        letters, ratios = plant_spec(P0)
        P, _ = L.plant_split(P0, letters, ratios, seed=0, mode='free')
        _B[name + '_planted'] = [chr(0xE000 + i) + c for i, c in enumerate(letters)]
    else: P = base1(name)
    rng = random.Random(39)
    idx = list(range(len(P))); rng.shuffle(idx)
    h = set(idx[:len(P) // 2])
    disc = [p for i, p in enumerate(P) if i in h]; held = [p for i, p in enumerate(P) if i not in h]
    _B[name] = (disc, held)
    return _B[name]


def merges(name):
    disc, held = corpus(name)
    A, I = L.interchange_index(disc, minc=30)
    n = len(A)
    pairs = sorted(((float(I[i, j]), A[i], A[j]) for i in range(n) for j in range(i + 1, n)), reverse=True)
    return pairs


def job(spec):
    name, thr = spec
    fn = f'c2_{name}|{thr}.json'
    if L.load(fn) is not None: return spec, 'cached'
    t0 = time.time()
    disc, held = corpus(name)
    pairs = merges(name)
    sel = [(a, b) for v, a, b in pairs if v >= thr] if thr is not None else []
    mp = L.merge_map(sel)
    P = L.apply_map(held, mp)
    excl = EXCL.get(name.lstrip('P'), ())
    r = L.battery(P, exclude=excl)
    r['npairs'] = len(sel); r['nmerged'] = len(mp); r['map'] = mp
    r['top'] = [(round(v, 3), a, b) for v, a, b in pairs[:12]]
    if name.startswith('P'): r['planted'] = _B.get(name + '_planted')
    r['sec'] = round(time.time() - t0, 1)
    L.save(fn, r)
    return spec, r['sec']


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    S = [(n, t) for n in ('V', 'VI', 'Pla', 'Pde', 'Pit', 'la', 'de', 'it') for t in (None,) + THR]
    t0 = time.time()
    with Pool(nw) as P:
        for k, s in P.imap_unordered(job, S):
            print(k, s, round(time.time() - t0), flush=True)
