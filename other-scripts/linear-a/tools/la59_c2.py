#!/usr/bin/env python3
"""LA-59 cycle 2: held-out alternations.
usage: la59_c2.py LA|LBs|LB [nsplit] [nkey]
The consensus graph of cycle 1 is split by sign pair (half the pairs train, half test). The fit on the train half
(real human key; shuffled keys) is scored on the test half (held-out gain, lam from train). Nulls: shuffled-key
fits (same values, human structure destroyed); test graph rewired (degrees kept) scored with the real fit;
massive random guessing: 100,000 random assignments scored on train, the top 50 re-scored on test."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *

tag = sys.argv[1]; nsplit = int(sys.argv[2]) if len(sys.argv) > 2 else 4; nkey = int(sys.argv[3]) if len(sys.argv) > 3 else 6
keyname = sys.argv[4] if len(sys.argv) > 4 else 'mn'
import zlib
rng = np.random.default_rng(zlib.crc32((tag + keyname + 'c2').encode()))
o = json.load(open(os.path.join(CK, f'c1_{tag}.json')))
signs = o['signs']; W = {tuple(k.split('|')): v for k, v in o['W'].items()}
if keyname == 'mn':
    SC = consonant_key(); SV, _ = vowel_key()
elif keyname == 'semds':
    SC = consonant_key('semds'); SV, _ = vowel_key()
else:   # diachronica
    dk = json.load(open(os.path.join(CK, 'diachronica_key.json')))
    SC = consonant_key(); SC[:16, :16] = np.array(dk['SC']); SV = np.array(dk['SV'])
K = kernel(SC, SV, 0.0)
t0 = time.time()

def fitbest(A_, K_, restarts):
    best = None
    for r in range(restarts):
        s, g, l = anneal(A_, K_, rng, 30)
        if best is None or g > best[1]:
            best = (s, g, l)
    return best

res = []
for sp in range(nsplit):
    tr, te = split_W(W, rng)
    _, Atr = graph_from_W(tr, signs=signs); _, Ate = graph_from_W(te, signs=signs)
    s, g, l = fitbest(Atr, K, 4)
    h = gain(Ate, s, K, lam=l); hfree = gain(Ate, s, K)
    hk = []
    for k in range(nkey):
        SCs = SC.copy(); SCs[:16, :16] = shuffle_offdiag(SC[:16, :16], rng); SVs = shuffle_offdiag(SV, rng)
        Ks = kernel(SCs, SVs, 0.0)
        ss, gs, ls = fitbest(Atr, Ks, 4)
        hk.append(gain(Ate, ss, Ks, lam=ls))
    hr = [gain(rewire(Ate, rng), s, K, lam=l) for _ in range(20)]
    Sr = rng.integers(0, 90, (100000, len(signs)))
    Gtr = gain_batch(Atr, Sr, K)
    top = Sr[np.argsort(-Gtr)[:50]]
    hrand = [gain(Ate, x, K) for x in top]
    hmed = [gain(Ate, x, K) for x in Sr[np.argsort(-Gtr)[50000:50050]]]
    r = dict(split=sp, train_gain=g, lam=l, heldout=h, heldout_freelam=hfree, heldout_shufkey=hk, heldout_rewired=hr,
             rand_top_train=float(Gtr.max()), rand_top_heldout=float(np.mean(hrand)), rand_mid_heldout=float(np.mean(hmed)),
             s=s.tolist())
    res.append(r)
    print(tag, keyname, sp, 'train', round(g, 4), 'held', round(h, 4), 'shufkey', np.round(hk, 4), 'rewired', round(np.mean(hr), 4),
          'rand top/mid', round(r['rand_top_heldout'], 4), round(r['rand_mid_heldout'], 4), round(time.time() - t0), 's', flush=True)
json.dump(res, open(os.path.join(CK, f'c2_{tag}_{keyname}.json'), 'w'))
print('done')
