#!/usr/bin/env python3
"""LA-37 cycle 3: does doublet use split by writer (spelling habit) or by word position (sound rule)?
For a pair (a, b), take all tokens of a and b in 2+ sign words; Cramer's V of the a/b label against
  site, scribe (named only), word position (initial / medial / final), next sign, previous sign.
Each V is placed as a percentile among ALL sign pairs of the same corpus (frequency-matched bins), so the
base rate (different signs differ) is removed. Control: Linear B doublets (a/a2, a/a3, ra/ra2, ra/ra3, ro/ro2,
pu/pu2, ta/ta2, o/wo) in KN+PY, the same statistic. Random-pair control: the percentile distribution is uniform
by construction, so the group mean of percentiles over k pairs is tested against random k-sets.
Second test (within word): word types written both ways (minimal pairs) and whether both spellings occur in
one document, one scribe, one site.
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import la37_common as K

LOG = os.path.join(K.CK, 'c3.log')


def cramer(lab, cat):
    cats = sorted(set(cat))
    if len(cats) < 2:
        return np.nan
    ci = {c: i for i, c in enumerate(cats)}
    M = np.zeros((2, len(cats)))
    for l, c in zip(lab, cat):
        M[l, ci[c]] += 1
    M = M[:, M.sum(0) > 0]
    n = M.sum()
    E = np.outer(M.sum(1), M.sum(0)) / n
    chi = ((M - E) ** 2 / np.where(E > 0, E, 1)).sum()
    return float(np.sqrt(chi / n / max(1, min(M.shape) - 1)))


def tokens(units):
    T = collections.defaultdict(list)
    for r in units:
        w = r['w']; L = len(w)
        for p, s in enumerate(w):
            pos = 'I' if p == 0 else ('F' if p == L - 1 else 'M')
            T[s].append(dict(site=r['site'], scribe=r['scribe'], pos=pos,
                             nxt=w[p + 1] if p + 1 < L else '#', prv=w[p - 1] if p > 0 else '#', doc=r['doc']))
    return T


FACT = ('site', 'scribe', 'pos', 'nxt', 'prv')


def pairV(T, a, b):
    out = {}
    ta, tb = T[a], T[b]
    for f in FACT:
        xs = [(0, t[f]) for t in ta] + [(1, t[f]) for t in tb]
        if f == 'scribe':
            xs = [x for x in xs if x[1]]
        if len(xs) < 6 or len({l for l, _ in xs}) < 2:
            out[f] = np.nan; continue
        out[f] = cramer([l for l, _ in xs], [c for _, c in xs])
    return out


def all_pairs(units, fmin):
    al, cnt = K.alphabet(units, fmin)
    T = tokens(units)
    iu = np.triu_indices(len(al), 1)
    V = {f: np.array([pairV(T, al[a], al[b])[f] for a, b in zip(*iu)]) for f in FACT}
    fm = {}
    for f in FACT:
        v = np.where(np.isnan(V[f]), -1, V[f])
        fm[f] = np.where(np.isnan(V[f]), np.nan, K.freq_matched_pct(v, iu, al, cnt))
    return al, iu, V, fm, T


def minimal(units, a, b):
    """word types written both ways (a <-> b in one slot) and where both spellings occur."""
    by = collections.defaultdict(list)
    for r in units:
        by[r['w']].append(r)
    res = []
    for w in by:
        for p, s in enumerate(w):
            if s != a:
                continue
            w2 = w[:p] + (b,) + w[p + 1:]
            if w2 in by:
                A, B = by[w], by[w2]
                res.append(dict(w1='-'.join(w), w2='-'.join(w2), n1=len(A), n2=len(B), pos=p, L=len(w),
                                doc=bool({r['doc'] for r in A} & {r['doc'] for r in B}),
                                scribe=bool(({r['scribe'] for r in A} & {r['scribe'] for r in B}) - {''}),
                                site=bool({r['site'] for r in A} & {r['site'] for r in B}),
                                sites1=sorted({r['site'][:4] for r in A}), sites2=sorted({r['site'][:4] for r in B})))
    return res


def group_test(fm, idx, nperm=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for f in FACT:
        v = fm[f]; ok = ~np.isnan(v)
        sel = [j for j in idx if ok[j]]
        if not sel:
            out[f] = None; continue
        obs = float(np.mean(v[sel]))
        pool = v[ok]
        null = np.array([rng.choice(pool, len(sel), replace=False).mean() for _ in range(nperm)])
        out[f] = (round(obs, 3), len(sel), round(float((1 + (null >= obs).sum()) / (nperm + 1)), 4))
    return out


if __name__ == '__main__':
    t0 = time.time()
    lines = []
    # Linear B control
    B = K.lb_units()
    for r in B:
        r['site'] = r['site']  # KN / PY
    al, iu, V, fm, T = all_pairs(B, 10)
    pidx = {frozenset((al[a], al[b])): j for j, (a, b) in enumerate(zip(*iu))}
    core = [pidx[p] for p in K.CORE if p in pidx]
    lines.append('LB doublets (core, in alphabet): ' + ', '.join('~'.join(sorted(p)) for p in K.CORE if p in pidx))
    for p in K.CORE:
        if p in pidx:
            j = pidx[p]
            lines.append('  LB ' + '~'.join(sorted(p)) + ' ' + json.dumps({f: (None if np.isnan(fm[f][j]) else round(float(fm[f][j]), 2)) for f in FACT}))
            a, b = sorted(p)
            mm = minimal(B, a, b) + minimal(B, b, a)
            if mm:
                lines.append(f'    minimal pairs {len(mm)}: same doc {sum(m["doc"] for m in mm)}, same hand {sum(m["scribe"] for m in mm)}, '
                             f'same site {sum(m["site"] for m in mm)}; positions {collections.Counter(("I" if m["pos"] == 0 else ("F" if m["pos"] == m["L"] - 1 else "M")) for m in mm)}')
    lines.append('LB core doublets group percentile (mean, k, P vs random k-sets): ' + json.dumps(group_test(fm, core)))
    labs = np.array([K.lb_label(al[a], al[b]) for a, b in zip(*iu)], dtype=object)
    for g in ('sameC', 'sameV'):
        lines.append(f'LB {g} group: ' + json.dumps(group_test(fm, list(np.where(labs == g)[0]))))
    # Linear A
    c2 = json.load(open(os.path.join(K.CK, 'c2.json')))
    U = K.la_units()
    al, iu, V, fm, T = all_pairs(U, 8)
    assert al == c2['alph']
    Tla = np.array(c2['T'])
    stab = np.array(c2['stab'])
    c2b = [x for x in json.load(open(os.path.join(K.CK, 'c2b.json'))) if x['kind'] == 'LA'][0]
    repl = np.array(c2b['all'])
    sets = dict(replicated_ge_0_2=list(np.where(repl >= 0.2)[0]), stable_ge_0_8=list(np.where(stab >= 0.8)[0]),
                top20=list(np.argsort(-Tla)[:20]))
    for k, idx in sets.items():
        lines.append(f'LA {k} (n {len(idx)}) group percentile: ' + json.dumps(group_test(fm, idx)))
    cand = sorted(set(sets['replicated_ge_0_2']) | set(sets['stable_ge_0_8']), key=lambda j: -Tla[j])
    for j in cand:
        a, b = al[iu[0][j]], al[iu[1][j]]
        mm = minimal(U, a, b) + minimal(U, b, a)
        lines.append(f'  LA {a}~{b} T {Tla[j]:.3f} ' + json.dumps({f: (None if np.isnan(fm[f][j]) else round(float(fm[f][j]), 2)) for f in FACT})
                     + f' | minimal pairs {len(mm)}: same doc {sum(m["doc"] for m in mm)}, same scribe {sum(m["scribe"] for m in mm)}, same site {sum(m["site"] for m in mm)}; '
                     + '; '.join(f'{m["w1"]}/{m["w2"]}({m["n1"]},{m["n2"]};{",".join(m["sites1"])}/{",".join(m["sites2"])})' for m in mm[:6]))
    json.dump(dict(lines=lines), open(os.path.join(K.CK, 'c3.json'), 'w'))
    for l in lines:
        K.log(LOG, l)
    K.log(LOG, f'done {time.time() - t0:.0f}s')
