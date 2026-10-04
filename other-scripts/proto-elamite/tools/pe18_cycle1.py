"""pe18 cycle 1: is a number system a highland 'accent'?
1a direct: plateau vs Susa system rates, size-matched.
1b validity: does a plateau-likeness score (vocab, no-class vocab, variant forms, header, format)
   separate HELD-OUT plateau tablets from Susa (AUC, 30 random plateau halves)?
1c main: Susa tablets carrying system s vs other Susa tablets, plateau-likeness difference;
   nulls = labels permuted within (size bin x capacity flag) and within (size x cap x dominant class sign).
   DEC is contrasted with SEX (both need counted values >= 60).  Replicated with plateau half models.
1d planted register: random Susa label group of the DEC group's size, r of its tokens replaced by
   draws from the plateau distribution; power of 1c.
"""
import json, sys, os, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe18_common import *

rng = np.random.default_rng(18)
R, CLASS = tablets()
SU = [t for t in R if t['region'] == 'SUSA']
PL = [t for t in R if t['region'] == 'PLAT']
A = 0.5
FAM = ['toks', 'nocls', 'forms', 'hdr', 'fmt']


def items(t, f):
    if f == 'hdr':
        return [t['hdr']] if t['hdr'] else []
    if f == 'forms':
        return t['forms']
    return t[f]


class Model:
    def __init__(self, plat, susa):
        self.pc, self.sc, self.st = {}, {}, {}
        for f in ['toks', 'nocls', 'hdr']:
            self.pc[f] = collections.Counter(x for t in plat for x in items(t, f))
            self.sc[f] = collections.Counter(x for t in susa for x in items(t, f))
        # forms: conditional on base
        self.pf = collections.Counter(x for t in plat for x in t['forms'])
        self.pb = collections.Counter(b for t in plat for b, _ in t['forms'])
        self.sf = collections.Counter(x for t in susa for x in t['forms'])
        self.sb = collections.Counter(b for t in susa for b, _ in t['forms'])
        self.nforms = collections.Counter(b for (b, _) in set(self.sf) | set(self.pf))
        F = np.array([t['fmt'] for t in susa]); P = np.array([t['fmt'] for t in plat])
        mu, sd = F.mean(0), F.std(0) + 1e-6
        self.mu, self.sd = mu, sd
        Fz, Pz = (F - mu) / sd, (P - mu) / sd
        self.fm, self.fs = Fz.mean(0), Fz.std(0) + 0.1
        self.pm, self.ps = Pz.mean(0), Pz.std(0) + 0.1
        self.V = {f: len(set(self.pc[f]) | set(self.sc[f])) + 1 for f in self.pc}

    def score(self, t, f, loo=False):
        if f == 'fmt':
            z = (np.array(t['fmt']) - self.mu) / self.sd
            lp = -0.5 * (((z - self.pm) / self.ps) ** 2) - np.log(self.ps)
            ls = -0.5 * (((z - self.fm) / self.fs) ** 2) - np.log(self.fs)
            return float((lp - ls).mean())
        xs = items(t, f)
        if not xs:
            return np.nan
        if f == 'forms':
            own = collections.Counter(xs) if loo else collections.Counter()
            ownb = collections.Counter(b for b, _ in xs) if loo else collections.Counter()
            v = []
            for b, s in xs:
                k = self.nforms[b] + 1
                pp = (self.pf[(b, s)] + A) / (self.pb[b] + A * k)
                ps = (self.sf[(b, s)] - own[(b, s)] + A) / (self.sb[b] - ownb[b] + A * k)
                v.append(math.log(pp / ps))
            return float(np.mean(v))
        pc, sc, V = self.pc[f], self.sc[f], self.V[f]
        Np, Ns = sum(pc.values()), sum(sc.values())
        own = collections.Counter(xs) if loo else collections.Counter()
        n_own = len(xs) if loo else 0
        v = [math.log((pc[x] + A) / (Np + A * V)) - math.log((sc[x] - own[x] + A) / (Ns - n_own + A * V)) for x in xs]
        return float(np.mean(v))


def scores(model, tabs, loo):
    S = np.array([[model.score(t, f, loo) for f in FAM] for t in tabs])
    return S


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def combine(S, ref):
    """z-score each family on Susa reference rows; average available."""
    mu = np.nanmean(ref, 0); sd = np.nanstd(ref, 0) + 1e-9
    Z = (S - mu) / sd
    return np.nanmean(Z, 1)


out = {}
# ---------------- 1a direct rates
sb = np.array([size_bin(t) for t in SU]); sbp = np.array([size_bin(t) for t in PL])
r1a = {}
for s in SYSTEMS:
    has = np.array([s in t['sys'] for t in SU]); hp = np.array([s in t['sys'] for t in PL])
    exp = sum(has[sb == b].mean() * (sbp == b).sum() for b in range(4) if (sb == b).any())
    obs = int(hp.sum())
    # null: draw 74 Susa tablets with plateau's size distribution, 20,000 times
    nul = np.zeros(20000)
    for b in range(4):
        k = (sbp == b).sum()
        if k:
            idx = np.where(sb == b)[0]
            nul += np.array([has[rng.choice(idx, k, replace=False)].sum() for _ in range(20000)])
    p_hi = float((1 + (nul >= obs).sum()) / 20001); p_lo = float((1 + (nul <= obs).sum()) / 20001)
    r1a[s] = {'plat_obs': obs, 'plat_exp': round(float(exp), 2), 'ratio': round(obs / exp, 2) if exp else None,
              'p_high': p_hi, 'p_low': p_lo}
    print('1a', s, r1a[s], flush=True)
# DEC share among marked (DEC or SEX) tablets
def decshare(X):
    d = sum(1 for t in X if 'DEC' in t['sys'] and 'SEX' not in t['sys'])
    x = sum(1 for t in X if 'SEX' in t['sys'] and 'DEC' not in t['sys'])
    return d, x
r1a['DEC_vs_SEX'] = {'SUSA': decshare(SU), 'PLAT': decshare(PL),
                     'by_site': {st: decshare([t for t in PL if t['site'] == st]) for st in PLATEAU}}
print('1a DEC/SEX', r1a['DEC_vs_SEX'], flush=True)
out['1a'] = r1a

# ---------------- 1b validity: held-out plateau halves
aucs = collections.defaultdict(list)
for rep in range(30):
    perm = rng.permutation(len(PL)); half = len(PL) // 2
    PA = [PL[i] for i in perm[:half]]; PB = [PL[i] for i in perm[half:]]
    m = Model(PA, SU)
    Ss = scores(m, SU, True); Sp = scores(m, PB, False)
    for j, f in enumerate(FAM):
        aucs[f].append(auc(Sp[:, j], Ss[:, j]))
    aucs['combined'].append(auc(combine(Sp, Ss), combine(Ss, Ss)))
r1b = {f: {'mean_auc': round(float(np.nanmean(v)), 3), 'sd': round(float(np.nanstd(v)), 3)} for f, v in aucs.items()}
print('1b', r1b, flush=True)
out['1b'] = r1b

# ---------------- 1c main
mfull = Model(PL, SU)
S_full = scores(mfull, SU, True)
perm = rng.permutation(len(PL)); half = len(PL) // 2
mA = Model([PL[i] for i in perm[:half]], SU); mB = Model([PL[i] for i in perm[half:]], SU)
S_A = scores(mA, SU, True); S_B = scores(mB, SU, True)
cap = np.array([t['cap'] for t in SU])
topdom = [d for d, _ in collections.Counter(t['dom'] for t in SU).most_common(10)]
dom = np.array([t['dom'] if t['dom'] in topdom else 'other' for t in SU])
ST1 = np.array(['%d_%d' % (b, c) for b, c in zip(sb, cap)])
ST2 = np.array(['%s_%s' % (a, d) for a, d in zip(ST1, dom)])


def contrast(score, lab, elig, strata, nperm=5000):
    sc = score[elig]; lb = lab[elig]; st = strata[elig]
    ok = ~np.isnan(sc); sc, lb, st = sc[ok], lb[ok], st[ok]
    if lb.sum() < 5 or (~lb).sum() < 5:
        return None
    obs = sc[lb].mean() - sc[~lb].mean()
    d = obs / (sc.std() + 1e-9)
    nul = np.array([(lambda L: sc[L].mean() - sc[~L].mean())(strata_perm(lb, st, rng)) for _ in range(nperm)])
    z = (obs - nul.mean()) / (nul.std() + 1e-12)
    p = float((1 + (np.abs(nul - nul.mean()) >= abs(obs - nul.mean())).sum()) / (nperm + 1))
    return {'n1': int(lb.sum()), 'n0': int((~lb).sum()), 'diff': round(float(obs), 4), 'd': round(float(d), 3),
            'z': round(float(z), 2), 'p': round(p, 4)}


def label_sets(rows):
    L = {}
    dec = np.array(['DEC' in t['sys'] and 'SEX' not in t['sys'] for t in rows])
    sex = np.array(['SEX' in t['sys'] and 'DEC' not in t['sys'] for t in rows])
    L['DEC_vs_SEX'] = (dec, dec | sex)
    for s in ['C@', 'B', 'S@', 'N23', 'FRAC', 'C', 'SEX']:
        has = np.array([s in t['sys'] for t in rows])
        L[s + '_vs_rest'] = (has, np.array([t['n_num'] > 0 for t in rows]))
    return L


LS = label_sets(SU)
r1c = {}
for name, (lab, elig) in LS.items():
    r1c[name] = {}
    for sname, S in [('full', S_full), ('halfA', S_A), ('halfB', S_B)]:
        comb = combine(S, S)
        for fi, f in enumerate(FAM + ['combined']):
            sc = comb if f == 'combined' else S[:, fi]
            for stn, stv in [('size_cap', ST1), ('size_cap_dom', ST2)]:
                if sname != 'full' and f != 'combined':
                    continue
                r = contrast(sc, lab, elig, stv, 3000)
                r1c[name]['%s|%s|%s' % (sname, f, stn)] = r
    print('1c', name, {k: (v['d'], v['z'], v['p']) for k, v in r1c[name].items() if v and 'combined' in k}, flush=True)
out['1c'] = r1c

# ---------------- 1d planted register
lab, elig = LS['DEC_vs_SEX']
ptok = [x for t in PL for x in t['toks']]; pform = [x for t in PL for x in t['forms']]
cntelig = np.where(elig)[0]
r1d = {}
for rate in [0.05, 0.1, 0.2, 0.3]:
    zs = []
    for rep in range(10):
        grp = rng.choice(cntelig, lab.sum(), replace=False)
        planted = set(grp.tolist())
        SU2 = []
        for i, t in enumerate(SU):
            if i in planted:
                t = dict(t)
                t['toks'] = [ptok[rng.integers(len(ptok))] if rng.random() < rate else x for x in t['toks']]
                t['nocls'] = [x for x in t['toks'] if x not in CLASS]
                t['forms'] = [pform[rng.integers(len(pform))] if rng.random() < rate else x for x in t['forms']]
            SU2.append(t)
        m2 = Model(PL, SU2); S2 = scores(m2, SU2, True)
        L2 = np.zeros(len(SU), bool); L2[grp] = True
        r = contrast(combine(S2, S2), L2, elig, ST1, 1000)
        zs.append(r['z'])
    r1d[str(rate)] = {'z_mean': round(float(np.mean(zs)), 2), 'z_min': round(float(np.min(zs)), 2),
                      'power_z3': float(np.mean(np.array(zs) >= 3))}
    print('1d', rate, r1d[str(rate)], flush=True)
out['1d'] = r1d
json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
