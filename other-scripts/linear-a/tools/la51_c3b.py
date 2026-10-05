#!/usr/bin/env python3
"""la51 cycle 3b: per-class null for the leave-one-deposit-out room reading (LA and LB-small).
N2 (class vectors shuffled across deposits, 100 reps) and, for RIT, a support-blind check:
the same AUC with all documents put in one stratum (shows how much rides on the support)."""
import sys, json, os, numpy as np
import la51_common as L, la51_engine as E
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else 'la', '0']
import importlib
c3 = None
corpus = sys.argv[1]
exec(open(os.path.join(os.path.dirname(__file__), 'la51_c3.py')).read().split("M = get_M()")[0])
M = get_M()
def per_class(dep_of, C):
    a, S, T, W = loo(M, dep_of, C)
    out = {}
    for c in range(len(L.CLASSES)):
        idx = [i for i, w in enumerate(W) if w[1] == c]
        if idx and len(set(T[i] for i in idx)) > 1: out[L.CLASSES[c]] = auc([S[i] for i in idx], [T[i] for i in idx])
    return out
real = per_class(M['dep_of'], M['C'])
nul = {k: [] for k in real}
for r in range(100):
    rng = np.random.default_rng(900 + r)
    Cn = E.null_cross_site(M, rng)
    pc = per_class(M['dep_of'], Cn)
    for k in real:
        if k in pc and not np.isnan(pc[k]): nul[k].append(pc[k])
res = {k: dict(auc=float(real[k]), null_mean=float(np.mean(nul[k])), P=float((np.sum(np.array(nul[k]) >= real[k]) + 1) / (len(nul[k]) + 1)), reps=len(nul[k])) for k in real}
for k, v in res.items(): print(corpus, k, {a: round(b, 3) for a, b in v.items()})
json.dump(res, open(os.path.join(L.CK, f'c3b_{corpus}.json'), 'w'), indent=1)
