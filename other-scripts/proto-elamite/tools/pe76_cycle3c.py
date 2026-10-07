"""pe76 cycle 3c: within ONE volume (one copyist / editor), does habit sharing over the BLOCK null decay with
Sb museum-number distance?  bins |dSb| <= 10, 11-50, 51-200, > 200; 30 BLOCK draws."""
import os, json, collections
import numpy as np
import pe76_common as P, common
import pe76_bank2 as B2
from pe76_bank3 import block
T = common.load(); n = len(T); ids = [t['id'] for t in T]
meta = json.load(open(os.path.join(P.DATA, 'pe8_meta.json')))
vol = np.array([meta[p]['pub_vol'] if p in meta else '' for p in ids])
mus = np.array([meta[p]['mus_num'] if (p in meta and meta[p].get('mus_prefix') == 'Sb' and meta[p].get('mus_num')) else -1 for p in ids], float)
real = B2.real_tokens(T)


def share(tok):
    tab = collections.defaultdict(set)
    for i, s in tok:
        tab[s].add(i)
    X = P.to_matrix([v for v in tab.values() if 2 <= len(v) <= 60], n)
    return (X @ X.T).tocsr()


hasv = np.zeros(n, bool)
for i, _ in real:
    hasv[i] = True
bins = [(0, 10), (11, 50), (51, 200), (201, 1e9)]
groups = collections.defaultdict(list)
for v in ['MDP 06', 'MDP 17', 'MDP 26', 'MDP 26S', 'TCL 32']:
    el = np.where(hasv & (vol == v) & (mus > 0))[0]
    for x, a in enumerate(el):
        for b in el[x + 1:]:
            d = abs(mus[a] - mus[b])
            for lo, hi in bins:
                if lo <= d <= hi:
                    groups[(v, lo)].append((a, b)); groups[('ALL', lo)].append((a, b))
rng = np.random.default_rng(7)
for g in groups:
    if len(groups[g]) > 40000:
        groups[g] = [groups[g][i] for i in rng.choice(len(groups[g]), 40000, replace=False)]


def rates(M):
    return {g: float((np.asarray(M[np.array([p[0] for p in pr]), np.array([p[1] for p in pr])]).ravel() > 0).mean()) for g, pr in groups.items()}


R = rates(share(real)); N = [rates(share(block(real, np.random.default_rng(9500 + k)))) for k in range(30)]
res = {}
for g in sorted(groups, key=str):
    nl = np.array([x[g] for x in N])
    res['%s |dSb|>=%d' % g] = dict(n=len(groups[g]), excess=round(R[g] - nl.mean(), 4), z=round((R[g] - nl.mean()) / (nl.std() + 1e-9), 2))
# trend test pooled: near-bin excess minus far-bin excess vs null
dr = (R[('ALL', 0)] - np.mean([x[('ALL', 0)] for x in N])) - (R[('ALL', 201)] - np.mean([x[('ALL', 201)] for x in N]))
dn = np.array([(x[('ALL', 0)] - x[('ALL', 201)]) - np.mean([y[('ALL', 0)] - y[('ALL', 201)] for y in N]) for x in N])
res['ALL near-minus-far excess'] = dict(real=round(dr, 4), null_sd=round(float(dn.std()), 4), z=round(dr / (dn.std() + 1e-9), 2))
print(json.dumps(res, indent=0))
json.dump(res, open(os.path.join(P.CKPT, 'cycle3c.json'), 'w'), indent=1)
