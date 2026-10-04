"""pe18 cycle 2: are sealed tablets dispatch documents whose number system marks the counterparty?
2a seal rate Susa vs plateau (size-matched).
2b Susa: seal x system; null = seal flags permuted within (publication volume x size bin).
2c Susa: seal x plateau-likeness (families + combined), same null.
2d interaction: plateau-likeness gap of system s in sealed minus unsealed tablets.
2e same seal, same system? pairs of tablets impressed with the same identified seal (PESnnnn)
   share a number system more than pairs from the same volume?
2f planted: (i) seal prob doubled for top-quartile plateau-likeness; (ii) seal OR ~2.5 with DEC.
"""
import json, sys, os, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe18_common import *

rng = np.random.default_rng(182)
R, CLASS = tablets()
SU = [t for t in R if t['region'] == 'SUSA']
PL = [t for t in R if t['region'] == 'PLAT']
sb = np.array([size_bin(t) for t in SU])
seal = np.array([t['sealed'] for t in SU])
vol = np.array([t['vol'] for t in SU])
VS = np.array(['%s_%d' % (v, b) for v, b in zip(vol, sb)])
out = {}
# 2a
sbp = np.array([size_bin(t) for t in PL]); sp = np.array([t['sealed'] for t in PL])
exp = sum(seal[sb == b].mean() * (sbp == b).sum() for b in range(4))
nul = np.zeros(20000)
for b in range(4):
    k = (sbp == b).sum()
    if k:
        idx = np.where(sb == b)[0]
        nul += np.array([seal[rng.choice(idx, k, replace=False)].sum() for _ in range(20000)])
out['2a'] = {'susa_rate': round(float(seal.mean()), 3), 'plat_obs': int(sp.sum()), 'plat_n': len(PL),
             'plat_exp': round(float(exp), 2), 'p_low': float((1 + (nul <= sp.sum()).sum()) / 20001),
             'by_site': {st: [int(sum(t['sealed'] for t in PL if t['site'] == st)), sum(1 for t in PL if t['site'] == st)] for st in PLATEAU},
             'susa_by_size': {b: [int(seal[sb == b].sum()), int((sb == b).sum())] for b in range(4)},
             'susa_volumes_with_seal_notes': sorted(collections.Counter(vol[seal]).items(), key=lambda x: -x[1])[:15]}
print('2a', out['2a'], flush=True)


def lor(a, b):
    """log odds ratio of a (bool) given b (bool), +0.5 smoothing"""
    n11 = (a & b).sum() + .5; n10 = (a & ~b).sum() + .5; n01 = (~a & b).sum() + .5; n00 = (~a & ~b).sum() + .5
    return float(np.log(n11 * n00 / (n10 * n01)))


def permtest(stat, lab, strata, n=5000):
    obs = stat(lab)
    nl = np.array([stat(strata_perm(lab, strata, rng)) for _ in range(n)])
    z = (obs - nl.mean()) / (nl.std() + 1e-12)
    p = float((1 + (np.abs(nl - nl.mean()) >= abs(obs - nl.mean())).sum()) / (n + 1))
    return obs, z, p


# 2b seal x system
r2b = {}
LS = label_sets(SU)
for name, (lab, elig) in LS.items():
    e = elig
    obs, z, p = permtest(lambda S: lor(S[e], lab[e]), seal, VS, 3000)
    r2b[name] = {'n_sys': int(lab[e].sum()), 'sealed_sys': int((seal & lab & e).sum()),
                 'rate_sys': round(float(seal[e & lab].mean()), 3), 'rate_rest': round(float(seal[e & ~lab].mean()), 3),
                 'logOR': round(obs, 3), 'z': round(float(z), 2), 'p': round(p, 4)}
    print('2b', name, r2b[name], flush=True)
out['2b'] = r2b

# 2c seal x plateau-likeness
m = Model(PL, SU); S = scores(m, SU, True); comb = combine(S, S)
r2c = {}
for fi, f in enumerate(FAM + ['combined']):
    sc = comb if f == 'combined' else S[:, fi]
    ok = ~np.isnan(sc)
    st = lambda L: float(sc[ok & L].mean() - sc[ok & ~L].mean())
    obs, z, p = permtest(st, seal, VS, 3000)
    r2c[f] = {'diff': round(obs, 4), 'd': round(obs / np.nanstd(sc), 3), 'z': round(float(z), 2), 'p': round(p, 4)}
    print('2c', f, r2c[f], flush=True)
out['2c'] = r2c

# 2d interaction
r2d = {}
for name, (lab, elig) in LS.items():
    ok = elig & ~np.isnan(comb)
    def st(L, lab=lab, ok=ok):
        a = ok & L; b = ok & ~L
        if (a & lab).sum() < 2 or (a & ~lab).sum() < 2:
            return 0.0
        return float((comb[a & lab].mean() - comb[a & ~lab].mean()) - (comb[b & lab].mean() - comb[b & ~lab].mean()))
    obs, z, p = permtest(st, seal, VS, 2000)
    r2d[name] = {'n_sealed_sys': int((ok & seal & lab).sum()), 'interaction': round(obs, 3), 'z': round(float(z), 2), 'p': round(p, 4)}
    print('2d', name, r2d[name], flush=True)
out['2d'] = r2d

# 2e same seal -> same system?
def sysvec(t):
    return frozenset(s for s in t['sys'] if s != 'AMB')
ids = collections.defaultdict(list)
for i, t in enumerate(SU):
    for s in set(t['seals']):
        ids[s].append(i)
pairs = [(a, b) for v in ids.values() if 2 <= len(v) <= 40 for a, b in itertools.combinations(v, 2)]
def jacc(a, b):
    A, B = sysvec(SU[a]), sysvec(SU[b])
    return len(A & B) / len(A | B) if (A | B) else np.nan
obs = np.nanmean([jacc(a, b) for a, b in pairs]) if pairs else np.nan
# null: same-volume random pairs, matched counts per volume
byvol = collections.defaultdict(list)
for i, t in enumerate(SU):
    byvol[t['vol']].append(i)
nl = []
for _ in range(2000):
    v = []
    for a, b in pairs:
        pool = byvol[SU[a]['vol']]
        x, y = rng.choice(pool, 2, replace=False) if len(pool) > 1 else (a, b)
        v.append(jacc(x, y))
    nl.append(np.nanmean(v))
nl = np.array(nl)
out['2e'] = {'n_seal_ids': len(ids), 'n_ids_multi': sum(1 for v in ids.values() if len(v) >= 2), 'n_pairs': len(pairs),
             'obs_jaccard': round(float(obs), 3), 'null_mean': round(float(nl.mean()), 3),
             'z': round(float((obs - nl.mean()) / (nl.std() + 1e-12)), 2), 'p_high': float((1 + (nl >= obs).sum()) / 2001)}
print('2e', out['2e'], flush=True)

# 2f planted
r2f = {}
q = np.nanpercentile(comb, 75)
for kind in ['plateau', 'dec']:
    zs = []
    for rep in range(20):
        s2 = seal.copy()
        if kind == 'plateau':
            tgt = np.where((comb >= q) & ~seal)[0]
            k = int(seal[comb >= q].sum())  # double the sealed count in the top quartile
            s2[rng.choice(tgt, min(k, len(tgt)), replace=False)] = True
            ok = ~np.isnan(comb)
            st = lambda L: float(comb[ok & L].mean() - comb[ok & ~L].mean())
        else:
            lab, elig = LS['DEC_vs_SEX']
            tgt = np.where(lab & ~seal)[0]
            s2[rng.choice(tgt, int(0.15 * lab.sum()), replace=False)] = True
            st = lambda L, lab=lab, elig=elig: lor(L[elig], lab[elig])
        o, z, p = permtest(st, s2, VS, 500)
        zs.append(z)
    r2f[kind] = {'z_mean': round(float(np.mean(zs)), 2), 'power_z2': float(np.mean(np.array(zs) >= 2)),
                 'power_z3': float(np.mean(np.array(zs) >= 3))}
    print('2f', kind, r2f[kind], flush=True)
out['2f'] = r2f
json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1, default=str)
