#!/usr/bin/env python3
"""la34 cycle 3: use blind form distances (no published hands) on ALL units with >= 3 syllabogram drawings.
Distance = arm D (sign-agnostic ductus, scale-free) and arm M (sign-matched form, scale-free); every test is
within site, with a within-site permutation null (units' attributes shuffled among units of the same site).
 T1 spelling: unit pairs that write the same word identically vs pairs where one writes a one-sign variant of
    the other's word (and the pair shares no identical word).  If variants mark different writers, variant pairs
    are farther apart in form.  Planted check: pairs of the same published hand vs different hands, same test.
 T2 commodity: pairs sharing a main commodity logogram (base sign: GRA VIN OLE OLIV CYP VIR FIC ...) vs not.
 T3 support: same support type vs different.
 T4 vocabulary: Mantel-type rank correlation of form distance with word-type Jaccard distance.
 T5 cross-site: cross-site pairs closer than the 5th percentile of same-site different-hand pairs, vs the
    same-site same-hand rate at that threshold (sensitivity) -> candidate travelling hands (grade C at best)."""
import numpy as np, json, os, collections, re
from scipy.stats import spearmanr
from la34_common import load, CK
from la34_img import occ_table, unit_D, unit_M
from la34_score import standardize, dist

occ, meta, cid = load()
feat = json.load(open(os.path.join(CK, 'feat.json')))
rng = np.random.default_rng(3434)


def unit_words(u):
    W = set(); logo = set(); sup = meta[u]['support']
    for d in cid:
        pass
    return W


def build():
    o_, names, Z, P, codes = occ_table(feat, scale_free=True, occ=occ)
    nocc = collections.Counter(o['unit'] for o in o_)
    U = sorted(u for u in nocc if nocc[u] >= 3)
    _, V, _ = unit_D(o_, Z, 'unit')
    allU = sorted({o['unit'] for o in o_}); ix = {u: i for i, u in enumerate(allU)}
    DD = dist(V[[ix[u] for u in U]])
    DM = unit_M(o_, Z, P, codes, U, 'unit')
    # words / logograms / support from lineara.xyz documents in each unit
    dids = collections.defaultdict(set)
    for o in occ:
        if o['did']: dids[o['unit']].add(o['did'])
    words = {u: {w for d in dids[u] for w in cid[d]['words'] if w.count('-') >= 1 and '#' not in w} for u in U}
    logos = {u: {re.split(r'[+\[]', l)[0] for d in dids[u] for l in cid[d]['logograms']} for u in U}
    return U, DD, DM, words, logos


def perm_site(U, sites):
    p = np.arange(len(U))
    for s in set(sites):
        ix = np.where(np.array(sites) == s)[0]; p[ix] = rng.permutation(ix)
    return p


def pair_test(Dm, U, sites, attr, kind, nperm=1000):
    """attr: list per unit; kind(a,b) -> 1 (group A), 0 (group B), None (skip). stat = mean dist B - mean dist A"""
    n = len(U); sites = np.array(sites)
    iu = np.triu_indices(n, 1)
    ss = sites[iu[0]] == sites[iu[1]]
    fin = np.isfinite(Dm[iu])
    def stat(at):
        g = np.array([kind(at[a], at[b]) if (s_ and f_) else None for a, b, s_, f_ in zip(iu[0], iu[1], ss, fin)], object)
        A = Dm[iu][g == 1]; B = Dm[iu][g == 0]
        if len(A) < 3 or len(B) < 3: return np.nan, len(A), len(B)
        return float(np.mean(B) - np.mean(A)), len(A), len(B)
    real, na, nb = stat(attr)
    null = []
    for _ in range(nperm):
        p = perm_site(U, sites); null.append(stat([attr[i] for i in p])[0])
    null = np.array(null); null = null[np.isfinite(null)]
    return {'stat': real, 'nA': na, 'nB': nb, 'null_mean': float(null.mean()), 'p_closer': float((1 + (null >= real).sum()) / (1 + len(null))),
            'p_farther': float((1 + (null <= real).sum()) / (1 + len(null)))}


def one_sign(a, b):
    x, y = a.split('-'), b.split('-')
    return len(x) == len(y) and len(x) >= 2 and sum(i != j for i, j in zip(x, y)) == 1


def main():
    U, DD, DM, words, logos = build()
    sites = [meta[u]['site'] for u in U]
    out = {'units': len(U)}
    print('units', len(U), flush=True)
    for nm, Dm in (('D', DD), ('M', DM)):
        r = {}
        # T1 spelling: group A = share identical word; group B = only one-sign variants, no identical word
        def k_spell(a, b):
            if a & b: return 1
            if any(one_sign(x, y) for x in a for y in b): return 0
            return None
        # positive check: does the same machinery separate published hands? (A=same hand, B=different)
        hands = [meta[u]['scribe'] or None for u in U]
        r['T0_hands'] = pair_test(Dm, U, sites, hands, lambda a, b: None if (a is None or b is None) else (1 if a == b else 0), 500)
        r['T1_spelling'] = pair_test(Dm, U, sites, [words[u] for u in U], k_spell, 500)
        main_l = {'GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR', 'FIC', 'HIDE', 'AROM', 'BOS', 'OVIS', 'CAP', 'SUS'}
        r['T2_commodity'] = pair_test(Dm, U, sites, [logos[u] & main_l for u in U],
                                      lambda a, b: None if not (a and b) else (1 if a & b else 0), 500)
        r['T3_support'] = pair_test(Dm, U, sites, [meta[u]['support'] or None for u in U],
                                    lambda a, b: None if (a is None or b is None) else (1 if a == b else 0), 500)
        # T4 vocabulary Mantel within site
        n = len(U); iu = np.triu_indices(n, 1); sa = np.array(sites)
        ok = (sa[iu[0]] == sa[iu[1]]) & np.isfinite(Dm[iu])
        def jac(a, b): return 1 - len(a & b) / len(a | b) if (a | b) else np.nan
        Wl = [words[u] for u in U]
        def mant(W):
            j = np.array([jac(W[a], W[b]) for a, b in zip(iu[0][ok], iu[1][ok])])
            f = np.isfinite(j)
            return spearmanr(Dm[iu][ok][f], j[f]).correlation
        real = mant(Wl); null = []
        for _ in range(200):
            p = perm_site(U, sites); null.append(mant([Wl[i] for i in p]))
        r['T4_vocab_mantel'] = {'rho': float(real), 'null_mean': float(np.mean(null)), 'p': float((1 + np.sum(np.array(null) >= real)) / 201)}
        # T5 cross-site closeness
        same = (sa[iu[0]] == sa[iu[1]]); fin = np.isfinite(Dm[iu])
        hh = np.array([h or '' for h in hands])
        lab = (hh[iu[0]] != '') & (hh[iu[1]] != '')
        dh = Dm[iu][same & fin & lab & (hh[iu[0]] != hh[iu[1]])]; sh = Dm[iu][same & fin & lab & (hh[iu[0]] == hh[iu[1]])]
        thr = np.percentile(dh, 5)
        cross = Dm[iu][~same & fin]
        cand = [(U[a], U[b], float(Dm[a, b])) for a, b in zip(iu[0], iu[1]) if sa[a] != sa[b] and np.isfinite(Dm[a, b]) and Dm[a, b] < thr]
        cand.sort(key=lambda x: x[2])
        r['T5_cross'] = {'thr_p5_diffhand': float(thr), 'samehand_rate': float((sh < thr).mean()), 'cross_rate': float((cross < thr).mean()),
                         'ncross': int(len(cross)), 'top': cand[:15]}
        out[nm] = r
        print(nm, json.dumps({k: (v if k != 'T5_cross' else {kk: vv for kk, vv in v.items() if kk != 'top'}) for k, v in r.items()}), flush=True)
        print(nm, 'cross candidates', cand[:8], flush=True)
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
