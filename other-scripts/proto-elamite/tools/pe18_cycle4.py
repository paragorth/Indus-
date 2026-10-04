"""pe18 cycle 4: is the B / N23 'plateau-likeness' an accent or a genre?
4a goods-matched contrast: each Susa tablet with system s vs its 5 nearest Susa tablets without s
   (same size bin, nearest by Jaccard over entry-final class signs); null = 500 pseudo-target sets
   drawn from tablets without s with the same size profile, matched the same way.
4b plateau model refitted WITHOUT plateau capacity tablets (does the excess survive?).
4c driver signs: which sign tokens carry the LLR on the s tablets; where they sit on the plateau.
4d planted: genre-matched contrast must find a plant of 30% plateau tokens on 23 random tablets.
"""
import json, sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe18_common import *

rng = np.random.default_rng(1804)
R, CLASS = tablets()
SU = [t for t in R if t['region'] == 'SUSA']
PL = [t for t in R if t['region'] == 'PLAT']
sb = np.array([size_bin(t) for t in SU])
fins = [set(t['finals']) for t in SU]
N = len(SU)
_voc = sorted(set().union(*fins)); _ix = {v: j for j, v in enumerate(_voc)}
FM = np.zeros((N, len(_voc)))
for i, f in enumerate(fins):
    for v in f:
        FM[i, _ix[v]] = 1
_in = FM @ FM.T; _rs = FM.sum(1); _un = _rs[:, None] + _rs[None, :] - _in
JM = np.where(_un > 0, _in / np.maximum(_un, 1), 1.0)


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def matched_stat(score, tgt, pool_mask, k=5):
    v = []
    for i in tgt:
        if np.isnan(score[i]):
            continue
        cand = np.where(pool_mask & (sb == sb[i]) & ~np.isnan(score))[0]
        cand = cand[cand != i]
        if len(cand) < k:
            continue
        sims = JM[i, cand] + rng.random(len(cand)) * 1e-6
        nn = cand[np.argsort(-sims)[:k]]
        v.append(score[i] - score[nn].mean())
    return float(np.mean(v)) if v else np.nan


def run(score, has, nnull=2000):
    tgt = np.where(has)[0]
    obs = matched_stat(score, tgt, ~has)
    others = np.where(~has)[0]
    nl = []
    for _ in range(nnull):
        pseudo = np.concatenate([rng.choice(others[sb[others] == b], (sb[tgt] == b).sum(), replace=False) for b in range(4) if (sb[tgt] == b).sum()])
        pm = ~has.copy(); pm[pseudo] = False
        nl.append(matched_stat(score, pseudo, pm))
    nl = np.array(nl)
    return {'n': int(len(tgt)), 'obs': round(obs, 4), 'null_mean': round(float(np.nanmean(nl)), 4),
            'z': round(float((obs - np.nanmean(nl)) / np.nanstd(nl)), 2),
            'p_high': float((1 + (nl >= obs).sum()) / (nnull + 1))}


out = {}
m = Model(PL, SU); S = scores(m, SU, True); comb = combine(S, S)
PLnc = [t for t in PL if not t['cap']]
m2 = Model(PLnc, SU); S2 = scores(m2, SU, True); comb2 = combine(S2, S2)
for s in ['B', 'N23', 'DEC', 'SEX', 'C@']:
    has = np.array([s in t['sys'] for t in SU])
    r = {}
    r['matched_combined'] = run(comb, has)
    r['matched_nocls'] = run(S[:, FAM.index('nocls')], has)
    r['matched_toks'] = run(S[:, FAM.index('toks')], has)
    r['noncap_plateau_combined'] = run(comb2, has)
    out[s] = r
    print('4ab', s, r, flush=True)
# 4c drivers
pc = collections.Counter(x for t in PL for x in t['toks'])
sc = collections.Counter(x for t in SU for x in t['toks'])
Np, Ns = sum(pc.values()), sum(sc.values()); V = len(set(pc) | set(sc)) + 1
for s in ['B', 'N23']:
    contrib = collections.Counter(); cnt = collections.Counter()
    for t in SU:
        if s in t['sys']:
            for x in t['toks']:
                l = np.log((pc[x] + .5) / (Np + .5 * V)) - np.log((sc[x] + .5) / (Ns + .5 * V))
                contrib[x] += l; cnt[x] += 1
    top = contrib.most_common(10)
    rows = []
    for x, c in top:
        where = collections.Counter(t['site'] for t in PL if x in t['toks'])
        capw = sum(1 for t in PL if x in t['toks'] and t['cap'])
        rows.append({'sign': x, 'llr_sum': round(float(c), 2), 'n_on_s_tablets': cnt[x], 'plateau_count': pc[x],
                     'susa_count': sc[x], 'plateau_sites': dict(where), 'plateau_tablets_capacity': capw})
    out['drivers_' + s] = rows
    print('4c', s, rows, flush=True)
# 4d planted
ptok = [x for t in PL for x in t['toks']]
zs = []
for rep in range(8):
    grp = rng.choice(N, 23, replace=False)
    SU2 = []
    for i, t in enumerate(SU):
        if i in set(grp.tolist()):
            t = dict(t); t['toks'] = [ptok[rng.integers(len(ptok))] if rng.random() < .3 else x for x in t['toks']]
            t['nocls'] = [x for x in t['toks'] if x not in CLASS]
        SU2.append(t)
    mm = Model(PL, SU2); SS = scores(mm, SU2, True); cc = combine(SS, SS)
    has = np.zeros(N, bool); has[grp] = True
    zs.append(run(cc, has, 300)['z'])
out['planted'] = {'z': zs, 'mean': round(float(np.mean(zs)), 2)}
print('4d', out['planted'], flush=True)
json.dump(out, open(os.path.join(CK, 'c4.json'), 'w'), indent=1, default=str)
