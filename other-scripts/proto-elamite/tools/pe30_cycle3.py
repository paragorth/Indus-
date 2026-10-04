"""pe30 cycle 3: which entries do the failing totals leave out? (roles)

(a) ROLE ENRICHMENT. For every tablet of tier 1+2 that does not close with all
    entries, the single-feature rules EX(f) that close it name the candidate
    left-out entries. Per feature class (position, size, bare, unit, system,
    duplicate, damage, relation to the total, sign) and per feature: number of
    failing tablets rescued. Null: features permuted among the entries WITHIN
    each tablet (values fixed, so the arithmetic of the miss is kept and only the
    identity of the left-out entry is random), 400 permutations; per-feature
    p-values Bonferroni-corrected over the features that rescue >= 2.
    Planted check: on baseline-closing tablets, the entry with a planted
    feature class (last entry) is dropped from the total; does the class test fire?
(b) CARRIED FROM ANOTHER TABLET. For failing count tablets, positive residual
    d = total - sum (any hypothesis / value set). Rate at which d equals the
    written total of another tablet in the corpus, vs totals replaced by random
    numbers of the same size (200x).
(c) TIER 3: 12 tablets with exactly one lost entry numeral. If the lost entry
    is counted, total - sum(known) must be > 0. Overshoot (known entries already
    exceed the total) would demand an exclusion. Rate vs random totals (200x).
Usage: python3 pe30_cycle3.py
"""
import time
from pe30_common import *

CLASSES = {'P_': 'position', 'Z_': 'size', 'N0': 'bare', 'N1': 'nsigns', 'N2': 'nsigns', 'N3': 'nsigns',
           'U_': 'unit', 'Y_': 'system', 'D_': 'duplicate', 'X_': 'damage', 'T_': 'total-rel',
           'B_': 'sign', 'S_': 'sign', 'F_': 'sign', 'L_': 'sign', 'K_': 'marker'}


def cls_of(f):
    return CLASSES.get(f[:2], 'other')


def rescued(C, fails, fl):
    sub = [C[k] for k in fails]
    sc = Scorer(sub, fl, pairs=False)
    ex = sc.M[1:1 + len(fl)]                 # EX1 rows, (F, nfail)
    per_feat = ex.sum(1)
    per_cls = collections.Counter()
    for c in set(map(cls_of, fl)):
        rows = [i for i, f in enumerate(fl) if cls_of(f) == c]
        per_cls[c] = int(ex[rows].any(0).sum())
    return per_feat, per_cls, int(ex.any(0).sum())


def permute_within(C, rng):
    out = []
    for c in C:
        keys = []
        for h in c['hyps']:
            for p in h:
                for fl in p['F']:
                    if tuple(fl) not in keys:
                        keys.append(tuple(fl))
        perm = [keys[i] for i in rng.permutation(len(keys))]
        mp = dict(zip(keys, perm))
        out.append(dict(c, hyps=[[dict(p, F=[list(mp[tuple(fl)]) for fl in p['F']]) for p in h] for h in c['hyps']]))
    return out


def residuals(c):
    ds = set()
    for h in c['hyps']:
        if len(h) != 1 or h[0]['cls'] != 'cnt':
            continue
        p = h[0]
        for v in range(len(p['T'])):
            d = Fr(p['T'][v]) - sum(Fr(e[v]) for e in p['E'])
            if d > 0:
                ds.add(d)
    return ds


def totals_pool(C):
    pool = collections.defaultdict(set)
    for k, c in enumerate(C):
        for h in c['hyps']:
            for p in h:
                for t in p['T']:
                    pool[Fr(t)].add(k)
    return pool


def carried_rate(C, Cres, fails, pool):
    hit = 0; n = 0
    for k in fails:
        ds = residuals(Cres[k])
        if not ds:
            continue
        n += 1
        if any(pool.get(d, set()) - {k} for d in ds):
            hit += 1
    return hit, n


def overshoot(C3):
    n = 0; over = 0
    for c in C3:
        ok_any = False; over_all = True
        for h in c['hyps']:
            for p in h:
                if not any('X_lost' in fl for fl in p['F']):
                    continue
                for v in range(len(p['T'])):
                    s = sum(Fr(e[v]) for e in p['E'])
                    ok_any = True
                    if Fr(p['T'][v]) - s > 0:
                        over_all = False
        if ok_any:
            n += 1; over += over_all
    return over, n


if __name__ == '__main__':
    t0 = time.time()
    rng = np.random.default_rng(30)
    C = load_cases(2)
    fl, cnt = feature_list(C)
    sc0 = Scorer(C, fl, pairs=False)
    fails = [k for k in range(len(C)) if not sc0.M[0, k]]
    out = {'n': len(C), 'n_fail': len(fails)}
    pf, pc, anyr = rescued(C, fails, fl)
    NP = 400
    nf = np.zeros((NP, len(fl))); ncl = collections.defaultdict(list); nany = []
    for r in range(NP):
        D = permute_within([C[k] for k in fails], rng)
        Dfull = list(C)
        for j, k in enumerate(fails):
            Dfull[k] = D[j]
        a, b, c_ = rescued(Dfull, fails, fl)
        nf[r] = a; nany.append(c_)
        for kk in pc:
            ncl[kk].append(b[kk])
    out['any_single_rescue'] = {'real': anyr, 'null_mean': float(np.mean(nany))}
    out['per_class'] = {k: {'real': pc[k], 'null_mean': float(np.mean(ncl[k])),
                            'p_hi': float((1 + sum(x >= pc[k] for x in ncl[k])) / (1 + NP))}
                        for k in sorted(pc, key=lambda k: -pc[k])}
    tested = [i for i in range(len(fl)) if pf[i] >= 2]
    feats = []
    for i in tested:
        p = float((1 + (nf[:, i] >= pf[i]).sum()) / (1 + NP))
        feats.append((fl[i], int(pf[i]), float(nf[:, i].mean()), p, min(1.0, p * len(tested))))
    feats.sort(key=lambda x: x[3])
    out['per_feature_top'] = feats[:15]
    out['n_features_tested'] = len(tested)
    # which tablets / entries: for the top 5 features, the rescued tablets
    ex = Scorer([C[k] for k in fails], fl, pairs=False).M[1:1 + len(fl)]
    out['rescued_by'] = {f[0]: [C[fails[j]]['id'] for j in np.where(ex[fl.index(f[0])])[0]] for f in feats[:8]}
    print(json.dumps({k: out[k] for k in ('n_fail', 'any_single_rescue', 'per_class')}, default=str), flush=True)
    print(feats[:10], flush=True)
    # planted class check: drop the last entry on half of the closing tablets
    clos = [k for k in range(len(C)) if sc0.M[0, k]]
    pick = set(rng.choice(clos, len(clos) // 2, replace=False).tolist())
    Cp = list(C)
    sub_by = dict(zip([k for k in sorted(pick)], plant([C[k] for k in sorted(pick)], lambda x: 'P_last' in x)[0]))
    for k in pick:
        Cp[k] = sub_by[k]
    scp = Scorer(Cp, fl, pairs=False)
    fp = [k for k in range(len(Cp)) if not scp.M[0, k]]
    _, pcp, _ = rescued(Cp, fp, fl)
    nclp = collections.defaultdict(list)
    for r in range(100):
        D = permute_within([Cp[k] for k in fp], rng)
        Df = list(Cp)
        for j, k in enumerate(fp):
            Df[k] = D[j]
        b = rescued(Df, fp, fl)[1]
        for kk in pcp:
            nclp[kk].append(b[kk])
    out['plant_last'] = {k: {'real': pcp[k], 'null_mean': float(np.mean(nclp[k])),
                             'p_hi': float((1 + sum(x >= pcp[k] for x in nclp[k])) / 101)} for k in pcp}
    out['plant_last']['n_planted'] = len(pick)
    print('plant', out['plant_last'], flush=True)
    # (b) carried from another tablet
    pool = totals_pool(C)
    h, n = carried_rate(C, C, fails, pool)
    rr = []
    for r in range(200):
        Cr = rand_totals(C, rng)
        scr = Scorer(Cr, fl[:1], pairs=False, consts=(1,))
        fr = [k for k in range(len(Cr)) if not scr.M[0, k]]
        hh, nn = carried_rate(C, Cr, fr, pool)
        rr.append(hh / max(nn, 1))
    out['carried'] = {'hits': h, 'n': n, 'rate': h / max(n, 1), 'rand_mean': float(np.mean(rr)),
                      'p_hi': float((1 + sum(x >= h / max(n, 1) for x in rr)) / 201)}
    print('carried', out['carried'], flush=True)
    # (c) tier 3
    C3 = load_cases(3)
    o, n3 = overshoot(C3)
    ro = []
    for r in range(200):
        a, b = overshoot(rand_totals(C3, rng)); ro.append(a / max(b, 1))
    out['tier3'] = {'n': n3, 'overshoot': o, 'rand_rate_mean': float(np.mean(ro)),
                    'p_hi': float((1 + sum(x >= o / max(n3, 1) for x in ro)) / 201),
                    'ids': [c['id'] for c in C3]}
    print('tier3', out['tier3'], flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(DATA, 'pe30_cycle3.json'), 'w'), indent=1, default=str)
