"""X-4: the Voynich v31 alphabet-free feature battery (32 features per 100-word sample) on every X-4 corpus and
on generators fitted to it. Used by cycle 3 (classifier) and cycle 4 (forgery battery).

Per corpus: one 3,000-word subsample of whole documents (seed 0; Linear A has 3,679 words in all), documents
concatenated in order, cut into 30 consecutive 100-word samples. Generators TRI / SLOT / CPV / WBG are fitted to
that same subsample and run twice (independent runs a, b: the forgery-vs-forgery floor).
Out: data/x4_ckpt/feats.json
"""
import os, sys, json, random, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X
import v31_lib as L

N, MAXS, BUDGET = 100, 30, 3000


def work(a):
    name, cond, i, lines = a
    rng = random.Random(zlib.crc32(f'{name}{cond}{i}'.encode()))
    return {'corpus': name, 'cond': cond, 'i': i, 'F': {k: float(v) for k, v in L.features(lines, rng).items()}}


def jobs():
    C = X.corpora(); J = []
    for name in X.LIST + ['VOY'] + X.PROSE:
        base = X.subsample(C[name], BUDGET, 0)
        variants = {'real': base}
        for g, fn in X.GENS.items():
            for run in 'ab':
                variants[f'{g}_{run}'] = fn(base, random.Random(zlib.crc32(f'{name}{g}{run}'.encode())))
        for cond, docs in variants.items():
            lines = [l for d in docs for l in d if l]
            S = L.samples([lines], N=N, maxs=MAXS)
            for i, s in enumerate(S): J.append((name, cond, i, s))
    return J


if __name__ == '__main__':
    J = jobs(); print(len(J), 'samples', flush=True)
    with Pool(int(os.environ.get('W', '2'))) as p:
        R = p.map(work, J, chunksize=16)
    X.save('feats.json', R)
    print('done', len(R))
