"""pe76 cycle 3b (after freeze): museum/volume sharing scored against the BLOCK null (keeps each tablet's
own consistency), 50 draws; difference same-volume-far minus different-volume with its null distribution."""
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
    names = sorted(k for k, v in tab.items() if 2 <= len(v) <= 60)
    X = P.to_matrix([tab[k] for k in names], n)
    return (X @ X.T).tocsr()


hasv = np.zeros(n, bool)
for i, _ in real:
    hasv[i] = True
rng = np.random.default_rng(76)
el = np.where(hasv & (vol != ''))[0]
groups = collections.defaultdict(list)
for x, a in enumerate(el):
    for b in el[x + 1:]:
        if vol[a] == vol[b]:
            g = 'near' if (mus[a] > 0 and mus[b] > 0 and abs(mus[a] - mus[b]) <= 10) else 'same_vol_far'
        else:
            g = 'diff_vol'
        groups[g].append((a, b))
for g in groups:
    if len(groups[g]) > 60000:
        groups[g] = [groups[g][i] for i in rng.choice(len(groups[g]), 60000, replace=False)]
big = ['MDP 06', 'MDP 17', 'MDP 26', 'MDP 26S', 'TCL 32', 'MDP 31']
for i, a in enumerate(big):
    A = [k for k in el if vol[k] == a]
    pr = [(x, y) for j, x in enumerate(A) for y in A[j + 1:]]
    if len(pr) > 20000:
        pr = [pr[j] for j in rng.choice(len(pr), 20000, replace=False)]
    groups['within ' + a] = pr


def rates(M):
    out = {}
    for g, pr in groups.items():
        A = np.array([p[0] for p in pr]); B = np.array([p[1] for p in pr])
        out[g] = float((np.asarray(M[A, B]).ravel() > 0).mean())
    return out


R = rates(share(real))
N = [rates(share(block(real, np.random.default_rng(9000 + k)))) for k in range(50)]
res = {}
for g in groups:
    nl = np.array([x[g] for x in N])
    res[g] = dict(n=len(groups[g]), real=R[g], block_null=float(nl.mean()), excess=float(R[g] - nl.mean()),
                  z=float((R[g] - nl.mean()) / (nl.std() + 1e-9)), p_high=float((np.sum(nl >= R[g]) + 1) / 51))
d_real = R['same_vol_far'] - R['diff_vol']
d_null = np.array([x['same_vol_far'] - x['diff_vol'] for x in N])
res['gradient_samefar_minus_diff'] = dict(real=d_real, null_mean=float(d_null.mean()), null_sd=float(d_null.std()),
                                          p=float((np.sum(d_null >= d_real) + 1) / 51))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(P.CKPT, 'cycle3b.json'), 'w'), indent=1)
