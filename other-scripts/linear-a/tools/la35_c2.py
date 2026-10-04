"""LA-35 cycle 2: massive random-rule search with held-out tablets.
A rule = (set S1 of endings -> number class '1', set S2 -> '>=2'). 3,000 random rules per split
(sizes 1-3 per side, drawn from endings seen in >= 2 stem families) + every single-ending rule.
Train half of tablets: rule score = sum z(S1) - sum z(S2), z = within-family (CMH) excess of
class '1' for that ending. Keep the top 50; held-out score = their mean test score.
Also M1: Spearman(train z_e, test z_e); M3: held-out AUC of class '1' from the train ending effect,
family-centred. Nulls: numbers shuffled within (tablet, commodity) [N1] and (site, commodity) [N2].
Controls: LB at LA size, planted rule in LA."""
import json, sys, time, random
import numpy as np
from la35_common import *
from la35_c1 import plant, lb_sub

DEFS = [('pre2_E1', 2, 'E1', None), ('pre1_E1', 1, 'E1', None), ('pre2_R', 2, 'R', None)]


def ending_z(E, idxs, cls, d):
    sub = [E[i] for i in idxs]
    Dz = Design(sub, d[1], d[2], d[3])
    if len(Dz.idx) < 4: return {}, Dz, sub
    O, dO, V = Dz.ending_table(cls[idxs], 2)
    z = {Dz.ends[j]: dO[j, 0] / np.sqrt(V[j, 0]) for j in range(Dz.ne) if V[j, 0] > 1e-9}
    return z, Dz, sub


def spearman(a, b):
    if len(a) < 4: return 0.0
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, y):
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    s = np.concatenate([pos, neg]); r = np.argsort(np.argsort(s)) + 1.0
    # ties: average
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def one_split(E, cls, d, tr_docs, rng, nrules=1500, top=50):
    tr = [i for i, e in enumerate(E) if e['doc'] in tr_docs]
    te = [i for i, e in enumerate(E) if e['doc'] not in tr_docs]
    ztr, _, _ = ending_z(E, tr, cls, d)
    zte, Dte, subte = ending_z(E, te, cls, d)
    common = sorted(set(ztr) & set(zte))
    m1 = spearman([ztr[k] for k in common], [zte[k] for k in common])
    pool = [k for k in ztr]
    if len(pool) < 2: return m1, 0.0, 0.5, []
    rules = []
    for _ in range(nrules):
        a = rng.choice(pool, size=min(len(pool) - 1, rng.integers(1, 4)), replace=False)
        rest = [k for k in pool if k not in set(a)]
        b = rng.choice(rest, size=min(len(rest), rng.integers(1, 4)), replace=False)
        rules.append((tuple(a), tuple(b)))
    rules += [((k,), ()) for k in pool] + [((), (k,)) for k in pool]
    sc = [sum(ztr[x] for x in a) - sum(ztr[x] for x in b) for a, b in rules]
    order = np.argsort(sc)[::-1][:top]
    tes = [sum(zte.get(x, 0) for x in rules[o][0]) - sum(zte.get(x, 0) for x in rules[o][1]) for o in order]
    m2 = float(np.mean(tes))
    # M3: family-centred prediction on test entries
    sc_e, y, fam = [], [], []
    for k, i in enumerate(Dte.idx):
        en = Dte.ends[Dte.e[k]]
        sc_e.append(ztr.get(en, 0.0)); y.append(int(cls[te[i]] == 0)); fam.append(Dte.f[k])
    sc_e = np.array(sc_e); y = np.array(y); fam = np.array(fam)
    if len(sc_e):
        for f in set(fam.tolist()):
            m = fam == f; sc_e[m] -= sc_e[m].mean()
        keep = sc_e != 0
        m3 = auc(sc_e[keep], y[keep]) if keep.sum() > 4 else 0.5
    else: m3 = 0.5
    toprules = [rules[o] for o in order[:5]]
    return m1, m2, m3, toprules


def evaluate(E, nsplit, seed, d):
    rng = np.random.default_rng(seed)
    cls = np.array([SCHEMES['K2'](e['v']) for e in E])
    docs = sorted({e['doc'] for e in E})
    M = []; tops = Counter()
    for s in range(nsplit):
        tr = set(rng.choice(docs, size=len(docs) // 2, replace=False).tolist())
        m1, m2, m3, tr_ = one_split(E, cls, d, tr, rng)
        M.append((m1, m2, m3))
        for a, b in tr_[:3]:
            for x in a: tops['%s>1' % x] += 1
            for x in b: tops['%s>=2' % x] += 1
    return np.array(M).mean(0), tops


def shuffled(E, key, rng):
    groups = strata(E, key)
    vals = np.array([e['v'] for e in E]); v2 = permute_vals(vals, groups, rng)
    out = [dict(e) for e in E]
    for e, v in zip(out, v2): e['v'] = int(v)
    return out


def contest(name, E, d, nsplit, nnull, seed):
    real, tops = evaluate(E, nsplit, seed, d)
    rng = np.random.default_rng(seed + 1)
    nul = {}
    for nl, key in (('N1', lambda e: (e['doc'], e['com'])), ('N2', lambda e: (e['site'], e['com']))):
        sims = np.array([evaluate(shuffled(E, key, rng), max(4, nsplit // 4), seed + 10 + k, d)[0] for k in range(nnull)])
        p = ((sims >= real[None, :]).sum(0) + 1) / (nnull + 1)
        nul[nl] = dict(mean=sims.mean(0).round(4).tolist(), sd=sims.std(0).round(4).tolist(), p=p.round(4).tolist())
    r = dict(name=name, d=d[0], real=real.round(4).tolist(), null=nul, tops=tops.most_common(8))
    print(json.dumps(r), flush=True)
    return r


if __name__ == '__main__':
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    NN = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    LA = la_entries(); LB = lb_entries()
    out = []
    for d in DEFS:
        out.append(contest('LA', LA, d, NS, NN, 1))
    d = DEFS[0]
    for s in range(3):
        P, xy = plant(LA, 0.3, 200 + s)
        r = contest('PLANT0.3 %s' % (xy,), P, d, NS, NN // 2, 30 + s); out.append(r)
    for s in range(4):
        out.append(contest('LBsub%d' % s, lb_sub(LB, len(LA), s), d, NS, NN // 2, 50 + s))
    json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
