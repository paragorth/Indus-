#!/usr/bin/env python3
"""LA-59 cycle 2: the syllabary constraint (each sign a distinct C x V state) + held-out alternations.
usage: la59_c2.py LA|LBs|LB [keyname: mn|semds|diach]
Fit = Gibbs annealing with swaps (anneal_inj), 20 sweeps, best of 2 restarts (real: 4 restarts, reported as the
expected best of 2 so it matches the nulls). Nulls: shuffled keys (same values, human structure destroyed);
rewired graphs (double-edge swaps, pair weights kept). Planted: realistic hidden syllabary (distinct states
from a grid of ~n/5+1 consonants x 5 vowels), graph drawn from the kernel on the real degrees and weight, sound
share 1.0 and 0.5. Held-out: half the sign pairs train, half test (3 splits); test gain under the train fit (lam
from train), real key vs shuffled keys; 100,000 random injective guesses scored on train, top 20 re-scored on test."""
import sys, os, json, time, zlib, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *

tag = sys.argv[1]; keyname = sys.argv[2] if len(sys.argv) > 2 else 'mn'
rng = np.random.default_rng(zlib.crc32((tag + keyname + 'c2').encode()))
o = json.load(open(os.path.join(CK, f'c1_{tag}.json')))
signs = o['signs']; A = np.array(o['A']); W = {tuple(k.split('|')): v for k, v in o['W'].items()}
if keyname == 'mn':
    SC = consonant_key(); SV, _ = vowel_key()
elif keyname == 'semds':
    SC = consonant_key('semds'); SV, _ = vowel_key()
else:
    dk = json.load(open(os.path.join(CK, 'diachronica_key.json')))
    SC = consonant_key(); SC[:16, :16] = np.array(dk['SC']); SV = np.array(dk['SV'])
K = kernel(SC, SV, 0.0)
t0 = time.time(); SW = 20
out = dict(tag=tag, key=keyname, signs=signs)

def fitbest(A_, K_, restarts):
    res = [anneal_inj(A_, K_, rng, SW) for _ in range(restarts)]
    res.sort(key=lambda x: -x[1])
    return res

def shufK():
    SCs = SC.copy(); SCs[:16, :16] = shuffle_offdiag(SC[:16, :16], rng)
    return kernel(SCs, shuffle_offdiag(SV, rng), 0.0)

def log(*a):
    print(tag, keyname, *a, round(time.time() - t0), 's', flush=True)

R = fitbest(A, K, 4)
gs = [r[1] for r in R]
out['real'] = [dict(s=r[0].tolist(), g=r[1], lam=r[2]) for r in R]
out['real_eb2'] = float(np.mean([max(c) for c in itertools.combinations(gs, 2)]))
log('real', np.round(gs, 4))
out['null_key'] = [fitbest(A, shufK(), 2)[0][1] for _ in range(8)]
log('shufkey', np.round(out['null_key'], 4))
out['null_rewire'] = [fitbest(rewire(A, rng), K, 2)[0][1] for _ in range(8)]
log('rewired', np.round(out['null_rewire'], 4))

d = A.sum(1); ne = int(round(A.sum() / 2)); pl = []
for share in (1.0, 0.5):
    for rep in range(3):
        st = planted_grid(len(signs), rng)
        B = sample_planted(d, st, K, ne, rng, share)
        sf, g, l = fitbest(B, K, 2)[0]
        tc = [x // 5 for x in st]; tv = [x % 5 for x in st]
        pl.append(dict(share=share, g=g, g_true=gain(B, st, K),
                       acc_c=float(np.mean(sf // 5 == st // 5)), acc_v=float(np.mean(sf % 5 == st % 5)),
                       acc_place=float(np.mean([PLACE.get(C_LABELS[a // 5]) == PLACE.get(C_LABELS[b // 5]) for a, b in zip(sf, st) if C_LABELS[b // 5] in PLACE])),
                       co_c=co_assign_agreement(sf, tc, lambda x: x // 5), co_v=co_assign_agreement(sf, tv, lambda x: x % 5)))
        log('planted', {k: round(v, 3) for k, v in pl[-1].items()})
out['planted'] = pl

ho = []
for sp in range(3):
    tr, te = split_W(W, rng)
    _, Atr = graph_from_W(tr, signs=signs); _, Ate = graph_from_W(te, signs=signs)
    s, g, l = fitbest(Atr, K, 2)[0]
    h = gain(Ate, s, K, lam=l)
    hk = []
    for k in range(4):
        Ks = shufK(); ss, gs_, ls = fitbest(Atr, Ks, 2)[0]
        hk.append(gain(Ate, ss, Ks, lam=ls))
    Sr = np.argsort(rng.random((100000, 90)), 1)[:, :len(signs)]
    Gtr = gain_batch(Atr, Sr, K)
    top = Sr[np.argsort(-Gtr)[:20]]
    hr = [gain(Ate, x, K) for x in top]; hm = [gain(Ate, x, K) for x in Sr[np.argsort(-Gtr)[50000:50020]]]
    ho.append(dict(train=g, lam=l, held=h, held_shufkey=hk, rand_top_train=float(Gtr.max()), rand_top_held=float(np.mean(hr)),
                   rand_mid_held=float(np.mean(hm))))
    log('heldout', {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in ho[-1].items()})
out['heldout'] = ho
json.dump(out, open(os.path.join(CK, f'c2_{tag}_{keyname}.json'), 'w'))
log('done')
