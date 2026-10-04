"""pe31 cycle 1b: does a Bayesian merge-evidence statistic (EV) pin individual
homophone pairs better than the cycle-1 z-scores?  Same controls and plants."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe31_lib as L

MINC = 6


def ev(data, truth_fn, label):
    alph, cnt = L.alphabet(data, MINC)
    Sd = L.scores(data, alph)
    Sd['EV'] = L.merge_evidence(data, alph)
    T = truth_fn(alph); n = len(alph); iu = np.triu_indices(n, 1)
    lab = np.array([tuple(sorted((alph[i], alph[j]))) in T for i, j in zip(*iu)])
    res = dict(corpus=label, n_truth=int(lab.sum()))
    variants = {'EVraw': Sd['EV'][iu], 'EVz': L.freq_z(Sd['EV'], cnt, alph)[iu],
                'EVz+MCP': (L.freq_z(Sd['EV'], cnt, alph) + L.combo(Sd, cnt, alph, (1, 1, 1, 0)))[iu],
                'MCP': L.combo(Sd, cnt, alph, (1, 1, 1, 0))[iu]}
    for k, Z in variants.items():
        K = max(10, int(lab.sum())); top = np.argsort(-Z)[:K]
        res[k] = dict(auc=round(L.auc(Z, lab), 3), hits=int(lab[top].sum()), K=K,
                      top1pct=int(lab[np.argsort(-Z)[:max(1, len(Z) // 100)]].sum()))
    return res


out = []
for name in ('UR3_SEAL', 'OB_SEAL', 'LINB'):
    r = ev(L.control(name), lambda A, nm=name: L.truth_pairs(nm, A), name); out.append(r); print(json.dumps(r), flush=True)
pe = L.pe_names(variants=False)
alph, cnt = L.alphabet(pe, 1)
cand = [s for s in alph if 16 <= cnt[s] <= 80]
for mode in ('token', 'tablet'):
    for rep in range(3):
        rng = random.Random(100 + rep); ps = set(rng.sample(cand, 10))
        d = L.plant(pe, ps, mode, rng)
        r = ev(d, lambda A, ps=ps: {tuple(sorted((s, s + "'"))) for s in ps if s + "'" in A and s in A}, f'plant-{mode}-{rep}')
        out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(L.CK, 'cycle1b.json'), 'w'), indent=1)
