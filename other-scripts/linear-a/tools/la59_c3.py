#!/usr/bin/env python3
"""LA-59 cycle 3: change the ear. Same unconstrained fit as cycle 1 with four keys:
  mn     Miller-Nicely (MAPCLUS / Shepard normalisation) + five-vowel formant key
  semds  Miller-Nicely (Hubert normalisation, rank-mapped to the same values) + formant key
  diach  attested sound changes (Index Diachronica 10.2, 1,433 one-to-one segment changes) for C and V
  flat   identity-only key: every off-diagonal similarity replaced by its mean (no graded hearing at all)
If graded human confusability carries Linear A's alternations, mn/semds/diach must beat flat and shuffled
versions of themselves, in-sample and on held-out sign pairs (3 splits). Shuffled diach keys x 6 as null.
usage: la59_c3.py LA|LBs"""
import sys, os, json, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *

tag = sys.argv[1]
rng = np.random.default_rng(zlib.crc32((tag + 'c3').encode()))
o = json.load(open(os.path.join(CK, f'c1_{tag}.json')))
signs = o['signs']; A = np.array(o['A']); W = {tuple(k.split('|')): v for k, v in o['W'].items()}
SV, _ = vowel_key(); dk = json.load(open(os.path.join(CK, 'diachronica_key.json')))
keys = {}
keys['mn'] = (consonant_key(), SV)
keys['semds'] = (consonant_key('semds'), SV)
SCd = consonant_key(); SCd[:16, :16] = np.array(dk['SC']); keys['diach'] = (SCd, np.array(dk['SV']))
def flat(S):
    T = S.copy(); off = ~np.eye(len(S), dtype=bool); T[off] = S[off].mean(); return T
keys['flat'] = (flat(consonant_key()), flat(SV))
t0 = time.time()
splits = [split_W(W, rng) for _ in range(3)]
splits = [(graph_from_W(tr, signs=signs)[1], graph_from_W(te, signs=signs)[1]) for tr, te in splits]

def run(SC, SVk, restarts=4):
    K = kernel(SC, SVk, 0.0)
    best = max((anneal(A, K, rng, 30) for _ in range(restarts)), key=lambda x: x[1])
    held = []
    for Atr, Ate in splits:
        s, g, l = max((anneal(Atr, K, rng, 30) for _ in range(2)), key=lambda x: x[1])
        held.append(gain(Ate, s, K, lam=l))
    return dict(g=best[1], lam=best[2], s=best[0].tolist(), held=held)

out = dict(tag=tag, signs=signs, keys={})
for name, (SC, SVk) in keys.items():
    out['keys'][name] = run(SC, SVk)
    r = out['keys'][name]
    print(tag, name, 'gain', round(r['g'], 4), 'held', np.round(r['held'], 4), round(time.time() - t0), 's', flush=True)
nul = []
for k in range(6):
    SCs = SCd.copy(); SCs[:16, :16] = shuffle_offdiag(SCd[:16, :16], rng)
    nul.append(run(SCs, shuffle_offdiag(keys['diach'][1], rng)))
    print(tag, 'diach-shuffled', k, round(nul[-1]['g'], 4), np.round(nul[-1]['held'], 4), round(time.time() - t0), 's', flush=True)
out['null_diach'] = [dict(g=x['g'], held=x['held']) for x in nul]
json.dump(out, open(os.path.join(CK, f'c3_{tag}.json'), 'w'))
print(tag, 'done', round(time.time() - t0), 's')
