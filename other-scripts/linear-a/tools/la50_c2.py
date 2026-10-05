#!/usr/bin/env python3
"""la50 cycle 2: which goods appear together, which never do. Seasonal snapshot test.

If a tablet records one moment of the farming year, goods harvested close together in the
calendar should share tablets more than expected, and goods half a year apart should avoid each
other. Statistic S(cal) = weighted correlation, over pairs of known-season items, between the
co-occurrence lift log((n_ab + .5) / (E_ab + .5)) and -circular distance of their months.
E_ab from 1,000 incidence-matrix swaps (curveball; row and column totals kept).
Nulls: 10^5 random calendars; every permutation of Aegean months; northern calendar.
Control: Linear B KN, PY. Planted: synthetic archives whose tablets each have a season and
draw goods with seasonal weight (LA tablet sizes and item frequencies).
"""
import itertools, json
from la50_common import *

rng = np.random.default_rng(502)
OUT = os.path.join(CK, 'c2.log'); open(OUT, 'w').close()
NR = 100000


def incidence(docs, items):
    rows = []
    for d in docs:
        s = [items.index(x) for x in set(d['flat']) if x in items]
        if len(s) >= 2: rows.append(s)
    return rows


def curveball(rows, n_items, nswap):
    rows = [list(r) for r in rows]
    R = len(rows)
    for _ in range(nswap):
        i, j = rng.integers(0, R, 2)
        if i == j: continue
        a, b = set(rows[i]), set(rows[j])
        sa, sb = list(a - b), list(b - a)
        if not sa or not sb: continue
        pool = sa + sb; rng.shuffle(pool)
        k = len(sa)
        both = a & b
        rows[i] = list(both | set(pool[:k])); rows[j] = list(both | set(pool[k:]))
    return rows


def cooc(rows, n):
    C = np.zeros((n, n))
    for r in rows:
        for p in r:
            for q in r:
                if p != q: C[p, q] += 1
    return C


def lift_matrix(rows, n, n_null=1000):
    O = cooc(rows, n)
    E = np.zeros((n, n)); cur = rows
    for k in range(n_null):
        cur = curveball(cur, n, 5 * len(rows))
        E += cooc(cur, n)
    E /= n_null
    return np.log((O + 0.5) / (E + 0.5)), O, E


def stat(M, L, iu, wts):
    """M[n_cal, k]; L lift matrix; iu index pairs; correlation of lift with -distance."""
    a, b = iu
    y = L[a, b]
    D = -circ_d(M[:, a], M[:, b])
    yw = y - (y * wts).sum() / wts.sum()
    Dw = D - (D * wts).sum(1, keepdims=True) / wts.sum()
    num = (Dw * yw * wts).sum(1)
    den = np.sqrt((Dw ** 2 * wts).sum(1) * (yw ** 2 * wts).sum()) + 1e-12
    return num / den


def analyse(name, docs, quiet=False, n_null=1000):
    items = sorted({x for d in docs for x in d['flat'] if x in AEGEAN})
    from collections import Counter
    cnt = Counter(x for d in docs for x in set(d['flat']) if x in items)
    items = [x for x in items if cnt[x] >= 3]
    rows = incidence(docs, items)
    n = len(items)
    L, O, E = lift_matrix(rows, n, n_null)
    iu = np.triu_indices(n, 1)
    wts = np.sqrt(E[iu] + 0.5)
    aeg = cal_vec(AEGEAN, items)
    sa = stat(aeg[None], L, iu, wts)[0]
    sn = stat(cal_vec(NORTH, items)[None], L, iu, wts)[0]
    R = rng.uniform(0, 12, (NR, n))
    sr = stat(R, L, iu, wts)
    perms = np.array(list(itertools.permutations(aeg))) if n <= 8 else aeg[np.array([rng.permutation(n) for _ in range(20000)])]
    sp = stat(perms, L, iu, wts)
    res = dict(name=name, items=items, n_rows=len(rows), S=float(sa), north=float(sn),
               P_random=float((sr >= sa).mean()), P_perm=float((sp >= sa).mean()),
               P_north=float((sr >= sn).mean()),
               pairs={'%s-%s' % (items[i], items[j]): [float(O[i, j]), round(float(E[i, j]), 2)] for i, j in zip(*iu)})
    if not quiet:
        log(OUT, '%s: tablets %d items %s | S(Aegean) %.3f P_random %s P_perm %s | north %.3f P_random %s'
            % (name, len(rows), ','.join(items), sa, fmt(res['P_random']), fmt(res['P_perm']), sn, fmt(res['P_north'])))
        log(OUT, '   obs/exp: ' + '; '.join('%s %d/%.1f' % (k, v[0], v[1]) for k, v in res['pairs'].items()))
    return res


def planted(docs, strength, reps=10):
    """same tablet sizes; each tablet has a month; item weight base_freq * exp(strength*cos)"""
    items = [x for x in ('GRA', 'FIC', 'VIN', 'OLE', 'OLIV')]
    from collections import Counter
    cnt = Counter(x for d in docs for x in set(d['flat']) if x in items)
    base = np.array([cnt[x] for x in items], float); base /= base.sum()
    sizes = [len([x for x in set(d['flat']) if x in items]) for d in docs]
    sizes = [s for s in sizes if s >= 1]
    m = cal_vec(AEGEAN, items)
    out = []
    for r in range(reps):
        D2 = []
        for s in sizes:
            mo = rng.uniform(0, 12)
            wv = base * np.exp(strength * np.cos(2 * np.pi * (m - mo) / 12))
            wv /= wv.sum()
            ch = rng.choice(len(items), size=s, replace=False, p=wv)
            D2.append({'flat': [items[c] for c in ch]})
        res = analyse('PLANT', D2, quiet=True, n_null=200)
        out.append((res['S'], res['P_random'], res['P_perm']))
    o = np.array(out)
    log(OUT, 'PLANTED strength %.1f (%d reps): S %.3f, P_random median %s (<0.05 in %d/%d), P_perm median %s'
        % (strength, reps, o[:, 0].mean(), fmt(np.median(o[:, 1])), (o[:, 1] < 0.05).sum(), reps, fmt(np.median(o[:, 2]))))
    return o.tolist()


if __name__ == '__main__':
    la = load_la(); kn = load_lb(('KN',)); py = load_lb(('PY',)); lb = kn + py
    ht = [d for d in la if d['id'].startswith('HT')]
    R = [analyse(nm, d) for nm, d in [('LA', la), ('LA-HT', ht), ('LB-KN+PY', lb), ('LB-KN', kn), ('LB-PY', py)]]
    P = {s: planted(la, s) for s in (0.5, 1.0, 2.0)}
    json.dump({'results': R, 'planted': P}, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
