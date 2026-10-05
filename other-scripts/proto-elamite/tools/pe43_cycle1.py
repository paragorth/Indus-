"""pe43 cycle 1: does the variant/compound tree carry meaning families?

1A variant inheritance: a derived simple form (X~b, X@g) vs its own root: co-occurrence profile
   and context (SYS = rarest numeral code on the line, POS = slot) similarity, against frequency-matched
   other roots (= variant labels shuffled among signs).
1B compound kinship: roots joined in one compound |A+B|: same lexical family (PC) / same usage (all)
   vs degree-preserving shuffled pairs.
1C random-phylogeny search (pe43_search): thousands of random forests over roots, hill-climbed on
   train co-occurrence; families scored on held-out tablets; PC families vs lexical lists; PLANT vs truth;
   PE families vs contexts. Nulls: token-shuffled corpus (bases shuffled across tablets), random partitions
   of the same sizes, variant labels shuffled among signs.
usage: python3 pe43_cycle1.py [PE|PC|PLANT|ALL]
"""
import json, sys, os
from collections import Counter, defaultdict
import numpy as np
from pe43_common import *
from pe43_search import search, consensus

RNG = np.random.default_rng(43)


def profiles(T, units_of_token, keys):
    """context profiles: dict unit -> Counter over keys('SYS'/'POS')"""
    P = {k: defaultdict(Counter) for k in keys}
    for ti, f, sy, pos in token_contexts(T):
        for u in units_of_token(f):
            P['SYS'][u][sy] += 1
            P['POS'][u][pos] += 1
    return P


def js_sim(c1, c2):
    keys = set(c1) | set(c2)
    if not keys:
        return np.nan
    p = np.array([c1.get(k, 0) for k in keys], float); q = np.array([c2.get(k, 0) for k in keys], float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    def kl(a, b):
        z = a > 0
        return (a[z] * np.log2(a[z] / b[z])).sum()
    return 1 - 0.5 * (kl(p, m) + kl(q, m))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def part_A(T, name):
    """variant inheritance"""
    tabs_of = defaultdict(set)
    for ti, t in enumerate(T):
        for l in t['lines']:
            for f in l['forms']:
                tabs_of[f].add(ti)
    roots_tabs = defaultdict(set)    # tablets with the BARE root
    for f, s in tabs_of.items():
        if not f.startswith('|') and not is_derived(f):
            roots_tabs[f] |= s
    TB = tablet_bases(T)
    roots = [r for r in roots_tabs if len(roots_tabs[r]) >= 5]
    idx = {r: i for i, r in enumerate(roots)}
    def cvec(tabs, excl):
        v = np.zeros(len(roots))
        for ti in tabs:
            for b in TB[ti]:
                if b in idx and b not in excl:
                    v[idx[b]] += 1
        return v
    rv = {r: cvec(roots_tabs[r], {r}) for r in roots}
    P = profiles(T, lambda f: [f] if not f.startswith('|') else [], ['SYS', 'POS'])
    rfreq = {r: len(roots_tabs[r]) for r in roots}
    order = sorted(roots, key=lambda r: rfreq[r])
    res = []
    for f, s in tabs_of.items():
        if f.startswith('|') or not is_derived(f) or len(s) < 3:
            continue
        r = root(f)
        if r not in idx:
            continue
        vv = cvec(s, {r})
        # frequency-matched decoys: 20 roots nearest in frequency (not r)
        k = order.index(r)
        dec = [x for x in order[max(0, k - 15):k + 16] if x != r][:30]
        own = [cos(vv, rv[r]), js_sim(P['SYS'][f], P['SYS'][r]), js_sim(P['POS'][f], P['POS'][r])]
        oth = [[cos(vv, rv[d]) for d in dec], [js_sim(P['SYS'][f], P['SYS'][d]) for d in dec],
               [js_sim(P['POS'][f], P['POS'][d]) for d in dec]]
        pct = [np.nanmean(np.array(o) < w) if not np.isnan(w) else np.nan for w, o in zip(own, oth)]
        # best other root (speciation candidate)
        best = dec[int(np.nanargmax([cos(vv, rv[d]) for d in dec]))] if dec else None
        res.append({'form': f, 'root': r, 'n': len(s), 'own': own, 'pct': pct, 'best_other': best})
    out = {'n_forms': len(res)}
    for j, k in enumerate(['COOC', 'SYS', 'POS']):
        p = np.array([x['pct'][j] for x in res], float); p = p[~np.isnan(p)]
        out[k] = {'mean_pct': float(p.mean()) if len(p) else None, 'share_top_half': float((p > 0.5).mean()) if len(p) else None,
                  'z_vs_0.5': float((p.mean() - 0.5) / (p.std() / np.sqrt(len(p)))) if len(p) > 2 else None, 'n': int(len(p))}
    # speciated forms: own-root co-occurrence percentile < 0.2 with n >= 5
    out['speciated'] = sorted([(x['form'], x['n'], round(x['pct'][0], 2), x['best_other']) for x in res
                               if x['n'] >= 5 and x['pct'][0] < 0.2], key=lambda z: -z[1])[:25]
    return out, res


def part_B(T, name, lexf=None):
    pairs = set()
    for t in T:
        for l in t['lines']:
            for f in l['forms']:
                if f.startswith('|'):
                    b = sorted(set(bases(f)))
                    for i in range(len(b)):
                        for j in range(i + 1, len(b)):
                            pairs.add((b[i], b[j]))
    pairs = sorted(pairs)
    TB = tablet_bases(T)
    roots = sorted({b for s in TB for b in s})
    ntab = Counter(b for s in TB for b in s)
    P = profiles(T, bases, ['SYS', 'POS'])
    # co-occurrence: tablets of each root
    tabs = defaultdict(set)
    for ti, s in enumerate(TB):
        for b in s:
            tabs[b].add(ti)
    def stats(PP):
        o = {}
        jac = [len(tabs[a] & tabs[b]) / max(1, len(tabs[a] | tabs[b])) for a, b in PP]
        o['jac'] = float(np.mean(jac))
        o['SYS'] = float(np.nanmean([js_sim(P['SYS'][a], P['SYS'][b]) for a, b in PP]))
        o['POS'] = float(np.nanmean([js_sim(P['POS'][a], P['POS'][b]) for a, b in PP]))
        if lexf:
            L = [(lexf[a] == lexf[b]) for a, b in PP if a in lexf and b in lexf]
            o['lex_same'] = float(np.mean(L)) if L else np.nan; o['lex_n'] = len(L)
        return o
    real = stats(pairs)
    # degree-preserving shuffle: permute the second members
    ends = [p[0] for p in pairs] + [p[1] for p in pairs]
    nulls = []
    for r in range(200):
        e = list(ends); RNG.shuffle(e)
        PP = [(e[2 * i], e[2 * i + 1]) for i in range(len(pairs)) if e[2 * i] != e[2 * i + 1]]
        nulls.append(stats(PP))
    out = {'n_pairs': len(pairs), 'real': real}
    for k in real:
        if k == 'lex_n':
            continue
        v = np.array([x[k] for x in nulls], float); v = v[~np.isnan(v)]
        out[k + '_null'] = [float(v.mean()), float(v.std())]
        out[k + '_p'] = float((v >= real[k]).mean()) if len(v) else None
    return out


def part_C(T, name, lexf=None, truth=None, min_tab=4, vshuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    n = len(T)
    perm = rng.permutation(n)
    te = set(perm[:int(0.3 * n)].tolist())
    vmap = None
    if vshuffle:
        forms = Counter(f for t in T for l in t['lines'] for f in l['forms'])
        der = [f for f in forms if not f.startswith('|') and is_derived(f)]
        rts = sorted({root(f) for f in forms if not f.startswith('|')})
        vmap = {f: rts[rng.integers(len(rts))] for f in der}
    TB = tablet_bases(T, vmap)
    cnt = Counter(b for s in TB for b in s)
    roots = sorted(b for b in cnt if cnt[b] >= min_tab)
    idx = {b: i for i, b in enumerate(roots)}
    tr_sets = [TB[i] for i in range(n) if i not in te]
    te_sets = [TB[i] for i in range(n) if i in te]
    A_tr, A_te = cooc(tr_sets, idx), cooc(te_sets, idx)
    keep, randQ, starts = search(A_tr, rng)
    lab, C = consensus(keep, len(roots))
    o = {'n_roots': len(roots), 'n_tab': n}
    o['Q_tr_best'] = float(keep[0][0]); o['Q_rand_start_best'] = float(randQ[0]); o['Q_rand_start_med'] = float(np.median(randQ))
    o['Q_te_climbed'] = [float(modularity(A_te, l)) for _, l, _ in keep[:5]]
    o['Q_te_consensus'] = float(modularity(A_te, lab))
    o['Q_te_randstarts_top'] = float(np.mean([modularity(A_te, forest_labels(p)) for _, p in starts[:20]]))
    sizes = Counter(lab)
    o['n_fam'] = len(sizes); o['fam_sizes'] = sorted(sizes.values(), reverse=True)[:15]
    # random partitions of the same sizes on held-out graph
    rq = [modularity(A_te, rng.permutation(lab)) for _ in range(200)]
    o['Q_te_randpart'] = [float(np.mean(rq)), float(np.std(rq))]
    fams = defaultdict(list)
    for b, l in zip(roots, lab):
        fams[int(l)].append(b)
    o['families'] = sorted([sorted(v, key=lambda b: -cnt[b]) for v in fams.values() if len(v) >= 2], key=len, reverse=True)
    if lexf:
        tl = [lexf.get(b) for b in roots]
        pp, tot = pair_precision(lab, tl)
        nl = [pair_precision(rng.permutation(lab), tl)[0] for _ in range(1000)]
        base_rate = pair_precision(np.zeros(len(lab), int), tl)[0]
        o['lex'] = {'pair_prec': pp, 'pairs': tot, 'null': [float(np.mean(nl)), float(np.std(nl))],
                    'p': float((np.array(nl) >= pp).mean()), 'base_rate': base_rate,
                    'n_labelled': int(sum(x is not None for x in tl))}
        # per family lexical composition
        comp = []
        for v in o['families'][:20]:
            c = Counter(lexf[b] for b in v if b in lexf)
            comp.append((v[:8], dict(c)))
        o['lex_comp'] = comp
    if truth:
        tl = [truth['fam'].get(b) for b in roots]
        o['ari'] = float(ari(list(lab), tl))
        o['ari_best_climb'] = float(ari(list(keep[0][1]), tl))
        o['ari_null'] = float(np.mean([ari(list(rng.permutation(lab)), tl) for _ in range(50)]))
    # contexts: within-family profile similarity vs same-size random partitions
    P = profiles(T, bases, ['SYS', 'POS'])
    for key in ['SYS', 'POS']:
        S = np.full((len(roots), len(roots)), np.nan)
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                S[i, j] = js_sim(P[key][roots[i]], P[key][roots[j]])
        def within(l):
            m = (l[:, None] == l[None, :])
            v = S[m & ~np.isnan(S)]
            return float(v.mean())
        w = within(lab)
        nl = [within(rng.permutation(lab)) for _ in range(300)]
        o['ctx_' + key] = {'within': w, 'null': [float(np.mean(nl)), float(np.std(nl))],
                           'z': float((w - np.mean(nl)) / np.std(nl))}
    return o


def shuffle_tokens(T, seed):
    """token-shuffled corpus: every form token re-drawn from the pooled token list (sizes kept)."""
    rng = np.random.default_rng(seed)
    pool = [f for t in T for l in t['lines'] for f in l['forms']]
    rng.shuffle(pool)
    k = 0; out = []
    for t in T:
        L = []
        for l in t['lines']:
            m = len(l['forms'])
            L.append(dict(l, forms=pool[k:k + m])); k += m
        out.append(dict(t, lines=L))
    return out


def run(name):
    lexf = truth = None
    if name == 'PE':
        T = load_pe()
    elif name == 'PC':
        T = load_pc(); lexf = lex_families()
    else:
        T, truth = make_plant(seed=1)
    R = {'A': part_A(T, name)[0], 'B': part_B(T, name, lexf)}
    print(name, 'A', json.dumps(R['A'])[:600], flush=True)
    print(name, 'B', json.dumps(R['B']), flush=True)
    R['C'] = part_C(T, name, lexf, truth)
    print(name, 'C', json.dumps({k: v for k, v in R['C'].items() if k not in ('families', 'lex_comp')}), flush=True)
    R['C_vshuf'] = part_C(T, name, lexf, truth, vshuffle=True)
    print(name, 'Cvshuf', json.dumps({k: v for k, v in R['C_vshuf'].items() if k not in ('families', 'lex_comp')}), flush=True)
    R['C_tokshuf'] = part_C(shuffle_tokens(T, 5), name, lexf, truth)
    print(name, 'Ctokshuf', json.dumps({k: v for k, v in R['C_tokshuf'].items() if k not in ('families', 'lex_comp')}), flush=True)
    json.dump(R, open(os.path.join(CK, f'c1_{name}.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    for nm in (['PLANT', 'PC', 'PE'] if w == 'ALL' else [w]):
        run(nm)
