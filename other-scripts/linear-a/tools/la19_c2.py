#!/usr/bin/env python3
"""LA-19 cycle 2: do the inferred 'languages' have support the model never saw?
For a fitted corpus (consensus partition from c1 at K_sel and at K=2), tokens of each word type carry attributes
(site, support, following commodity logogram, number size, position in the document / line, what follows, hand).
Statistic: mutual information (bits) between component and attribute over word TOKENS.
Null: component labels permuted among word types inside strata of (length, token-frequency bin), so that a split
that only tracks word length or frequency cannot pass. 2,000 permutations; Holm over attributes.
Positive control: LB personnel names (component vs site KN/PY; component vs Greek-looking / non-Greek-looking label).
Planted control: an attribute created to follow the true component of a planted mixture.
usage: la19_c2.py
"""
import sys, time
from la19_common import *


def mi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)))
    np.add.at(M, (ia, ib), 1)
    P = M / M.sum(); pa = P.sum(1, keepdims=True); pb = P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pa @ pb)[nz])).sum()), M, ua, ub


def strata_perm_test(type_comp, tok_type, tok_attr, strata, nperm=2000, seed=0):
    rnd = np.random.RandomState(seed)
    type_comp = np.asarray(type_comp); tok_type = np.asarray(tok_type)
    obs, M, ua, ub = mi(type_comp[tok_type], tok_attr)
    groups = collections.defaultdict(list)
    for i, s in enumerate(strata): groups[s].append(i)
    groups = [np.array(v) for v in groups.values()]
    null = np.zeros(nperm)
    for r in range(nperm):
        z = type_comp.copy()
        for gidx in groups: z[gidx] = type_comp[rnd.permutation(gidx)]
        null[r] = mi(z[tok_type], tok_attr)[0]
    p = (1 + (null >= obs).sum()) / (nperm + 1)
    # residuals for the table
    E = M.sum(1, keepdims=True) * M.sum(0, keepdims=True) / M.sum()
    return dict(mi=round(obs, 4), null_mean=round(float(null.mean()), 4), null_q95=round(float(np.quantile(null, 0.95)), 4),
                p=round(float(p), 4), comps=ua.tolist(), cats=[str(x) for x in ub],
                table=M.astype(int).tolist(), z=np.round((M - E) / np.sqrt(E + 1e-9), 2).tolist())


def holm(ps):
    o = np.argsort(ps); m = len(ps); out = np.zeros(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); out[i] = run
    return out


def la_attrs(t):
    s = t['site']
    site = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}.get(s, 'OTH')
    sup = t['support']
    sup = sup if sup in ('Tablet', 'Nodule', 'Roundel') else ('Vessel/object' if sup in (
        'Stone vessel', 'Clay vessel', 'Metal object', 'Stone object', 'Inked inscription', 'Architecture', 'ivory object',
        'clay vessel', 'Graffito', 'Triton', 'Loom weight') else 'Other admin')
    lg = t['logo'] if t['logo'] in ('GRA', 'OLE', 'VIN', 'VIR', 'CYP', 'OLIV', 'NI', 'FIC') else ('other' if t['logo'] else 'none')
    n = t['num']
    nb = 'none' if n is None else '1' if n == 1 else '2-9' if n < 10 else '10-99' if n < 100 else '100+'
    pos = 'docfirst' if t['docfirst'] else 'linefirst' if t['linefirst'] else 'mid'
    hand = t['scribe'] if (t['scribe'] and t['site'] == 'Haghia Triada') else None
    return dict(site=site, support=sup, commodity=lg, numsize=nb, position=pos, follows=t['nxt'], hand=hand)


def strata_of(words, freq):
    out = []
    for w in words:
        L = min(len(w), 5); f = freq[w]; fb = 0 if f == 1 else 1 if f == 2 else 2 if f <= 5 else 3
        out.append((L, fb))
    return out


def char_signs(words, comp, K, top=8):
    """signs over-represented in each component: overall, initial, final (log2 ratio vs rest, with counts)."""
    res = {}
    for k in range(K):
        e = {}
        for pos in ('any', 'initial', 'final'):
            ck = collections.Counter(); cr = collections.Counter()
            for w, c in zip(words, comp):
                S = list(w) if pos == 'any' else [w[0]] if pos == 'initial' else [w[-1]]
                (ck if c == k else cr).update(S)
            nk, nr = sum(ck.values()), sum(cr.values())
            sc = []
            for s in set(ck) | set(cr):
                a, b = ck[s], cr[s]
                if a + b < 5: continue
                lr = math.log2(((a + 0.5) / (nk + 1)) / ((b + 0.5) / (nr + 1)))
                sc.append((lr, s, a, b))
            sc.sort(reverse=True)
            e[pos] = [('%s' % s, a, b, round(lr, 2)) for lr, s, a, b in sc[:top]]
        res[k] = e
    return res


def la_analysis(K_list=None, nperm=2000):
    T = la_tokens()
    row = json.load(open(os.path.join(CK, 'c1_LA.json')))
    words = [tuple(w.split('-')) for w in row['words']]
    ix = {w: i for i, w in enumerate(words)}
    freq = collections.Counter(t['w'] for t in T)
    strata = strata_of(words, freq)
    tok_type = np.array([ix[t['w']] for t in T])
    A = [la_attrs(t) for t in T]
    out = {}
    for K, e in row['stab'].items():
        if 'map' not in e: continue
        comp = np.array(e['map']); K = int(K)
        r = dict(K=K, sizes=e['sizes'], pmax=e.get('pmax_mean'), tests={})
        names = ['site', 'support', 'commodity', 'numsize', 'position', 'follows', 'hand']
        for a in names:
            keep = np.array([x[a] is not None for x in A])
            r['tests'][a] = strata_perm_test(comp, tok_type[keep], np.array([x[a] for x in A], dtype=object)[keep].astype(str),
                                             strata, nperm, seed=K * 31 + len(a))
        ph = holm(np.array([r['tests'][a]['p'] for a in names]))
        for a, h in zip(names, ph): r['tests'][a]['p_holm'] = round(float(h), 4)
        # restricted to Hagia Triada tablets (removes the site / support confound)
        keep = np.array([x['site'] == 'HT' and x['support'] == 'Tablet' for x in A])
        for a in ('commodity', 'numsize', 'position', 'follows', 'hand'):
            k2 = keep & np.array([x[a] is not None for x in A])
            r['tests']['HTtab_' + a] = strata_perm_test(comp, tok_type[k2], np.array([x[a] for x in A], dtype=object)[k2].astype(str),
                                                        strata, nperm, seed=K * 37 + len(a))
        r['signs'] = char_signs(words, comp, K)
        # example words per component (most frequent)
        ex = collections.defaultdict(list)
        for w in sorted(words, key=lambda w: -freq[w]): ex[int(comp[ix[w]])].append('-'.join(w))
        r['examples'] = {k: v[:25] for k, v in ex.items()}
        out[K] = r
    return out


def lb_analysis(name='LBpers', nperm=2000):
    row = json.load(open(os.path.join(CK, 'c1_%s.json' % name)))
    words = [tuple(w.split('-')) for w in row['words']]
    site = dict(lb_names())
    freq = collections.Counter({w: 1 for w in words})
    strata = strata_of(words, freq)
    out = {}
    for K, e in row['stab'].items():
        if 'map' not in e: continue
        comp = np.array(e['map']); K = int(K)
        r = dict(K=K, sizes=e['sizes'], tests={})
        tt = np.arange(len(words))
        sl = np.array([site.get(w, 'OTHER') for w in words])
        keep = np.isin(sl, ['KN', 'PY'])
        r['tests']['site'] = strata_perm_test(comp, tt[keep], sl[keep], strata, nperm, seed=1)
        gl = np.array([lb_greekness(w) for w in words]); k2 = gl != '?'
        r['tests']['greek_label'] = strata_perm_test(comp, tt[k2], gl[k2], strata, nperm, seed=2)
        r['signs'] = char_signs(words, comp, K)
        out[K] = r
    return out


if __name__ == '__main__':
    res = dict(LA=la_analysis())
    for nm in ('LBpers', 'LBpers_0', 'LBpers_1', 'LBpers_2'):
        if os.path.exists(os.path.join(CK, 'c1_%s.json' % nm)): res[nm] = lb_analysis(nm)
    dump(os.path.join(OUT, 'c2.json'), res)
    for nm, R in res.items():
        for K, r in R.items():
            print(nm, 'K', K, 'sizes', r['sizes'], {a: (v['mi'], v['null_mean'], v['p'], v.get('p_holm')) for a, v in r['tests'].items()})
