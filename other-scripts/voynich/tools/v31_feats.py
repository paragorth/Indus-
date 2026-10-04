"""v31: sample every corpus (training classes, generators, Voynich and other test objects) and compute the
feature battery. Output data/v31_ckpt/feats.json (one record per sample)."""
import os, sys, json, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_lib as L, v31_gen as G

N = int(os.environ.get('V31_N', 100))
MAXS = {'LANG': 8, 'INVENT': 8, 'GEN': 8, 'MAGIC': 60, 'GIBB': 6, 'TEST': 80}


def voynich_docs(name, part=None):
    import v21_lib as V
    P = V.voynich_pages(name, minw=10)
    if part: P = [p for p in P if p['sec'].endswith(part)]
    docs = {}
    for p in P: docs.setdefault(p['sec'], []).extend(l for pa in p['paras'] for l in pa)   # one document per section
    return list(docs.values())


def forged_docs(kind, seed=0):
    import v21_lib as V
    P = V.voynich_pages('ZL3b')
    rng = random.Random(seed)
    if kind == 'junction':
        F = V.Forger(P, scope='sec', pos=True, chain=True, name='F3')
    elif kind == 'selfcite':
        F = V.Forger(P, scope='sec', pos=True, chain=True, cite=0.15, cite_window=40, cite_edit=0.5, name='F10')
    else:
        F = V.SlotForger(P)
    Q = F.forge(P, rng)
    docs = {}
    for p in Q: docs.setdefault(p['sec'], []).extend(l for pa in p['paras'] for l in pa)
    return list(docs.values())


def corpora():
    C = json.load(open(L.CORPORA))
    out = {k: (v['cls'], v['docs']) for k, v in C.items()}
    for k, d in G.build(C).items(): out[k] = ('GEN', [d])
    out['V_ZL'] = ('TEST', voynich_docs('ZL3b'))
    out['V_IT'] = ('TEST', voynich_docs('IT2a'))
    out['V_ZL_A'] = ('TEST', voynich_docs('ZL3b', 'A'))
    out['V_ZL_B'] = ('TEST', voynich_docs('ZL3b', 'B'))
    for kind in ('junction', 'selfcite', 'slot'):
        out['T_forge_' + kind] = ('TEST', forged_docs(kind))
    return out


def work(args):
    name, cls, i, lines = args
    rng = random.Random(hash((name, i)) & 0xffffffff)
    F = L.features(lines, rng)
    LF = L.line_features(lines, rng)
    return {'corpus': name, 'cls': cls, 'i': i, 'F': {k: float(v) for k, v in F.items()},
            'LF': {k: float(v) for k, v in LF.items()} if LF else None,
            'nlines': len(lines)}


def main():
    C = corpora()
    jobs = []
    for name, (cls, docs) in C.items():
        S = L.samples(docs, N=N, maxs=MAXS[cls])
        for i, s in enumerate(S): jobs.append((name, cls, i, s))
        print(f'{name:22s} {cls:6s} samples {len(S)}', flush=True)
    with Pool(2) as pool:
        R = pool.map(work, jobs, chunksize=8)
    L.save(f'feats_N{N}.json', R)
    print('samples', len(R))


if __name__ == '__main__':
    main()
