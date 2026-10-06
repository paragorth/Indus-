"""pe70 cycle 2: does one seal mark one party's dossier with its own vocabulary?

Groups = tablets impressed with the same seal id (PE: 12 seals after merging id pairs that name the same
tablet set; Ur III: CDLI seal_id).  Statistics (content tokens only, seal legend never used):
  COH  : mean idf-weighted Jaccard of tablet pairs within a seal group
  OWN  : number of (seal, token) pairs where the token is on >= 2 tablets of the group and its hypergeometric
         tail p (group size, corpus df) < 1e-3  ('own vocabulary')
Nulls: seal ids shuffled among tablets with a seal id (group sizes kept), (a) freely, (b) within volume.
Controls: Ur III (Umma + Girsu) full and drawn at PE size (12 groups with PE's group sizes); validation of
own-vocabulary tokens = share that appear in the group's seal legend (the sealer's name / title) vs the
share of random group tokens.  Planted: a df-3..8 PE sign added to every tablet of one real seal group.
usage: pe70_c2.py -> data/pe70_ckpt/c2.json
"""
import json, os, collections, math
import numpy as np
from scipy.stats import hypergeom
from pe70_common import get, CK

NPERM = 2000


def groups_of(R, min_size=2):
    g = collections.defaultdict(set)
    for i, r in enumerate(R):
        for s in r['seals']:
            g[s].add(i)
    G, seen = {}, set()
    for s, v in sorted(g.items()):
        k = frozenset(v)
        if len(v) >= min_size and k not in seen:
            seen.add(k); G[s] = sorted(v)
    return G


def idf(R):
    df = collections.Counter(t for r in R for t in r['toks'])
    N = len(R)
    return df, {t: math.log(N / c) for t, c in df.items()}


def wjac(a, b, w):
    i = a & b; u = a | b
    return sum(w[t] for t in i) / (sum(w[t] for t in u) + 1e-9)


def coh(G, R, w):
    v = []
    for s, ix in G.items():
        S = [set(R[i]['toks']) for i in ix]
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                v.append(wjac(S[a], S[b], w))
    return float(np.mean(v)) if v else np.nan


def own(G, R, df, N, thr=1e-3):
    out = []
    for s, ix in G.items():
        c = collections.Counter(t for i in ix for t in R[i]['toks'])
        for t, k in c.items():
            if k >= 2:
                p = hypergeom.sf(k - 1, N, df[t], len(ix))
                if p < thr:
                    out.append((s, t, k, df[t], float(p)))
    return out


def shuffle_groups(G, pool_idx, vol, rng, within_vol):
    """reassign the groups' tablets: draw new tablet sets of the same sizes from tablets with a seal id."""
    newG = {}
    if not within_vol:
        P = list(rng.permutation(pool_idx))
        for s, ix in G.items():
            newG[s] = [P.pop() for _ in ix]
        return newG
    byv = collections.defaultdict(list)
    for i in pool_idx:
        byv[vol[i]].append(i)
    for v in byv:
        byv[v] = list(rng.permutation(byv[v]))
    for s, ix in G.items():
        new = []
        for i in ix:
            L = byv[vol[i]]
            new.append(L.pop() if L else i)
        newG[s] = new
    return newG


def test(R, G, rng, nperm=NPERM, label=''):
    df, w = idf(R); N = len(R)
    pool_idx = sorted({i for r_i, r in enumerate(R) for i in [r_i] if r['seals']})
    vol = [r['vol'] for r in R]
    rc = coh(G, R, w); ro = own(G, R, df, N)
    out = dict(label=label, groups=len(G), tablets=sum(len(v) for v in G.values()), coh=rc, own=len(ro))
    for nm, wv in (('free', False), ('vol', True)):
        nc, no = [], []
        for _ in range(nperm):
            G2 = shuffle_groups(G, pool_idx, vol, rng, wv)
            nc.append(coh(G2, R, w)); no.append(len(own(G2, R, df, N)))
        nc, no = np.array(nc), np.array(no)
        out[nm] = dict(coh_null=float(np.nanmean(nc)), p_coh=float((nc >= rc).mean()), own_null=float(no.mean()),
                       p_own=float((no >= len(ro)).mean()))
    out['own_list'] = sorted(ro, key=lambda x: x[4])[:30]
    return out


def legend_validation(R, G, ownl, rng):
    """share of own-vocabulary tokens that occur in the group's seal legend vs random tokens of the group"""
    hit, n, base_hit, base_n = 0, 0, 0, 0
    for s, t, k, d, p in ownl:
        leg = set(x for i in G[s] for x in R[i]['legend'])
        if not leg:
            continue
        n += 1; hit += t in leg
    for s, ix in G.items():
        leg = set(x for i in ix for x in R[i]['legend'])
        if not leg:
            continue
        toks = [t for i in ix for t in R[i]['toks']]
        for t in rng.choice(toks, min(5, len(toks)), replace=False):
            base_n += 1; base_hit += t in leg
    return dict(own_in_legend=hit, own_with_legend=n, rate=hit / max(1, n), base_rate=base_hit / max(1, base_n))


def main():
    rng = np.random.default_rng(702)
    pe, ur = get('pe'), get('ur3')
    res = {}
    G = groups_of(pe)
    print('PE groups', {s: len(v) for s, v in G.items()}, flush=True)
    import sys
    CTRL = 'controls' in sys.argv
    if not CTRL:
        res['pe'] = test(pe, G, rng, label='PE')
        print('PE', {k: v for k, v in res['pe'].items() if k != 'own_list'}, flush=True)
    # planted
    res['plant'] = []
    df = collections.Counter(t for r in pe for t in r['toks'])
    rare = sorted(t for t, c in df.items() if 3 <= c <= 8)
    for k in range(0 if CTRL else 10):
        s = rng.choice(sorted(G)); t = rng.choice(rare)
        P = [dict(r) for r in pe]
        for i in G[s]:
            P[i]['toks'] = sorted(set(P[i]['toks']) | {t})
        o = test(P, G, rng, nperm=300, label='plant %d' % k)
        rec = any(x[0] == s and x[1] == t for x in o['own_list'])
        res['plant'].append(dict(seal=s, size=len(G[s]), tok=t, recovered=rec, p_coh=o['vol']['p_coh'], p_own=o['vol']['p_own']))
        print('plant', res['plant'][-1], flush=True)
    # Ur III full (by site)
    for site in ('Umma', 'Girsu'):
        U = [r for r in ur if r['vol'] == site]
        GU = groups_of(U)
        GU = {s: v for s, v in GU.items() if len(v) <= 40}
        o = test(U, GU, rng, nperm=200, label='UR3 ' + site)
        o['legend'] = legend_validation(U, GU, own(GU, U, *idf(U)[:1], len(U)), rng)
        res['ur3_' + site] = o
        print(o['label'], {k: v for k, v in o.items() if k != 'own_list'}, flush=True)
    # Ur III at PE size: 12 groups with PE group sizes, embedded in 1,581 tablets with ~97 seal-id tablets
    sizes = sorted(len(v) for v in G.values())
    res['ur3_pe'] = []
    U = [r for r in ur if r['vol'] == 'Umma']
    GU = groups_of(U)
    for k in range(20):
        cand = [s for s, v in GU.items() if len(v) >= max(sizes)]
        pick = rng.choice(cand, len(sizes), replace=False)
        idx = []
        for s, m in zip(pick, sizes):
            idx += list(rng.choice(GU[s], m, replace=False))
        single = [i for i, r in enumerate(U) if r['seals'] and i not in set(idx)]
        idx_single = list(rng.choice(single, 97 - len(idx), replace=False))
        rest = [i for i, r in enumerate(U) if not r['seals']]
        idx_rest = list(rng.choice(rest, 1581 - 97, replace=False))
        sub = []
        for i in idx + idx_single:
            r = dict(U[i]); sub.append(r)
        newG, j = {}, 0
        for s, m in zip(pick, sizes):
            newG[s] = list(range(j, j + m)); j += m
        for r in sub[:len(idx)]:
            pass
        # restrict seal ids so only the chosen groups recur
        for q, r in enumerate(sub):
            r['seals'] = [s for s in newG if q in newG[s]] or ['single%d' % q]
        sub += [dict(U[i], seals=[]) for i in idx_rest]
        o = test(sub, newG, rng, nperm=300, label='UR3 PE-size %d' % k)
        o['legend'] = legend_validation(sub, newG, o['own_list'], rng)
        res['ur3_pe'].append({kk: vv for kk, vv in o.items() if kk != 'own_list'})
        print(o['label'], res['ur3_pe'][-1], flush=True)
    json.dump(res, open(os.path.join(CK, 'c2_ctrl.json' if CTRL else 'c2.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
