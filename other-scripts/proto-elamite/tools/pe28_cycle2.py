"""pe28 cycle 2: HELD-OUT test of propagated rates.  NS random 70/30 splits of tablets.  The crossword
solver (anchors + propagation, minv in {2, 3}) is fitted on the 70%; on the 30% every target line with a
window whose keys are all fixed (>= 1 new, non-anchor key) gets predictions; a hit = the written value
equals a prediction exactly.  Nulls on the held-out part: (i) target values re-dealt among held-out
targets of the same (system, final sign) (NR times; keeps the number of windows / predictions per line);
(ii) random rates: each new fitted rate replaced by a random value from the same notation grid (NR times).
Pooled over splits: hits vs null means, p from the pooled null.  Per new key: held-out hits pooled over the
splits in which it was fitted.  Same pipeline on Ur III (anchor gurusz -> sze-bi = 60) and on the planted
network as controls."""
import os, sys, json, time
import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe28_common import *  # noqa

NS = int(os.environ.get('NS', 20))
NR = int(os.environ.get('NR', 200))
t0 = time.time()


def heldout(E, ntab, anchors, minv, gsys=None, seed=0, label=''):
    rng = np.random.default_rng(seed)
    real, n_sc, nul_d, nul_r = 0, 0, np.zeros(NR), np.zeros(NR)
    perkey = defaultdict(lambda: [0, 0, 0.0])   # fitted-splits, held-out hits, null-hit expectation
    for s in range(NS):
        test = set(rng.choice(ntab, int(0.3 * ntab), replace=False).tolist())
        Etr = [e for e in E if e['t'] not in test]
        Ete = [e for e in E if e['t'] in test]
        fx, inf = propagate(Etr, anchors, minv=minv, gsys=gsys)
        n, h, det = predict(Ete, fx, anchors)
        real += h; n_sc += n
        # per key: hits of windows using that key
        for k in inf:
            perkey[k][0] += 1
        def key_hits(Eh, rates):
            c = Counter()
            for e in Eh:
                for w in e['W']:
                    if all(k in rates for k in w) and any(k not in anchors for k in w):
                        if sum(rates[k] * x for k, x in w.items()) == e['y']:
                            for k in w:
                                if k not in anchors:
                                    c[k] += 1
                            break
            return c
        kh = key_hits(Ete, fx)
        for k, c in kh.items():
            perkey[k][1] += c
        for r in range(NR):
            Ed = redeal(Ete, rng)
            nul_d[r] += predict(Ed, fx, anchors)[1]
            if r < 20:
                for k, c in key_hits(Ed, fx).items():
                    perkey[k][2] += c / 20
            fr = dict(fx)
            for k in inf:
                g = grid_for(k[2]) if gsys is None else G_UR
                fr[k] = g[int(rng.integers(len(g)))]
            nul_r[r] += predict(Ete, fr, anchors)[1]
        print(label, minv, s, n, h, round(time.time() - t0), flush=True)
    out = {'minv': minv, 'splits': NS, 'scored_lines': n_sc, 'hits': real,
           'null_redeal_mean': float(nul_d.mean()), 'p_redeal': float((1 + (nul_d >= real).sum()) / (1 + NR)),
           'null_randrate_mean': float(nul_r.mean()), 'p_randrate': float((1 + (nul_r >= real).sum()) / (1 + NR))}
    pk = sorted(((keystr(k), v[0], v[1], round(v[2], 2)) for k, v in perkey.items() if v[1] > 0),
                key=lambda a: -a[2])
    out['per_key'] = pk[:40]
    print(label, out['hits'], out['scored_lines'], out['null_redeal_mean'], out['p_redeal'],
          out['null_randrate_mean'], out['p_randrate'], flush=True)
    for r in pk[:12]:
        print('    ', r, flush=True)
    return out


res = {}
# Ur III control
US = ur_seqs()
rng = np.random.default_rng(0)
idx = rng.choice(len(US), 800, replace=False)
EU = build([US[i] for i in idx], ur=True)
res['ur3'] = heldout(EU, 800, {('gurusz', 'sze-bi', 'UR'): Fr(60)}, 3, gsys='UR', seed=1, label='UR3')
# PE
S = pe_seqs()
E = build(S)
A = anchors_pe()
for mv in (3, 2):
    res[f'pe_minv{mv}'] = heldout(E, len(S), A, mv, seed=2, label='PE')
json.dump(res, open(os.path.join(CK, 'cycle2.json'), 'w'), indent=1, default=str)
print('done', round(time.time() - t0))
