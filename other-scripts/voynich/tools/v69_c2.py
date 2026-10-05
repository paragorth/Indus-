"""v69 cycle 2: power, the carefully written words, and where they sit.
 P  power: plant a care bump of d SD (0.05-0.3) on the rarest 25% of Voynich tokens (corpus freq),
    or on first mentions, then rerun T1/T3 against the same stratified nulls -> smallest detectable d;
    compare with the Latin effect size measured in c1.
 W  per-type mean care (types with >= 4 tokens on the pages), z against content-stratified
    permutations of care (type means recomputed on each permutation); top and bottom types.
 S  where the top-3% care tokens sit (line position, paragraph-first line, line index on page,
    page, rarity), each against the same share among all tokens; Latin top-3% for comparison.
 H  hand split: f58r/v (Currier A) vs Q20 (B) T1/T3 already in c1 (V_Q20); here the f58 pair alone.
Out: data/v69_ckpt/c2.json"""
import sys, os, json, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v69_lib as X
import v69_c1 as C1

RNG = np.random.default_rng(692)


def rarity(D, freq):
    tot = sum(freq.values())
    return np.array([-math.log((freq.get(w, 0) + 1) / tot) for w in D.word])


def power(D, freq, nperm=300):
    rar = rarity(D, freq)
    rare = rar >= np.percentile(rar, 75)
    Sc = D.strata('content'); Sp = D.strata('place')
    base = D.care.copy()
    # first-mention mask
    byp = collections.defaultdict(list)
    for i in np.lexsort((D.k, D.li, D.page)):
        byp[(D.page[i], D.word[i])].append(i)
    firstm = np.zeros(D.N, bool)
    for v in byp.values():
        if len(v) >= 2:
            firstm[v[0]] = True
    out = {}
    for d in [0.05, 0.1, 0.2, 0.3]:
        c = base + d * rare
        obs = C1.corr(c, rar)
        nul = [C1.corr(D.perm(c, Sc), rar) for _ in range(nperm)]
        z1 = (obs - np.mean(nul)) / np.std(nul)
        c = base + d * firstm
        obs = C1.first_mention(D, c)[0]
        nul = [C1.first_mention(D, D.perm(c, Sp))[0] for _ in range(nperm)]
        z3 = (obs - np.mean(nul)) / np.std(nul)
        out[str(d)] = {'T1_z': round(float(z1), 2), 'T3_z': round(float(z3), 2)}
        print('power', d, out[str(d)], flush=True)
    return out


def careful_types(D, freq, nperm=400, minn=4):
    idx = collections.defaultdict(list)
    for i, w in enumerate(D.word):
        idx[w].append(i)
    types = [w for w, v in idx.items() if len(v) >= minn]
    T = [np.array(idx[w]) for w in types]
    obs = np.array([D.care[t].mean() for t in T])
    Sc = D.strata('content')
    nul = np.zeros((nperm, len(T)))
    for p in range(nperm):
        cp = D.perm(D.care, Sc)
        nul[p] = [cp[t].mean() for t in T]
    z = (obs - nul.mean(0)) / (nul.std(0) + 1e-9)
    # how many |z|>2.5 vs expected under the null (null z of one permutation vs the rest)
    zn = (nul[0] - nul[1:].mean(0)) / (nul[1:].std(0) + 1e-9)
    order = np.argsort(-z)
    rows = [{'w': types[i], 'n': len(T[i]), 'care': round(float(obs[i]), 3), 'z': round(float(z[i]), 2),
             'freq': freq.get(types[i], 0)} for i in order]
    return {'ntypes': len(types), 'n_z_gt_2.5': int((z > 2.5).sum()), 'n_z_lt_-2.5': int((z < -2.5).sum()),
            'null_n_gt_2.5': int((zn > 2.5).sum()), 'null_n_lt_-2.5': int((zn < -2.5).sum()),
            'corr_typez_rarity': round(C1.corr(z, -np.log([freq.get(t, 0) + 1 for t in types])), 3),
            'top': rows[:25], 'bottom': rows[-15:]}


def where(D, freq, q=97):
    rar = rarity(D, freq)
    top = D.care >= np.percentile(D.care, q)
    k = D.k; nw = np.array([r['nw'] for r in D.R])
    ps = np.array([bool(r['para_start']) for r in D.R])
    pages = np.array(D.folio)
    feats = {'line_first': k == 0, 'line_last': k == nw - 1, 'para_first_line': ps,
             'top5_lines': D.li < 5, 'rare_q75': rar >= np.percentile(rar, 75), 'hapax': np.array([freq.get(w, 0) <= 1 for w in D.word])}
    out = {}
    for f, m in feats.items():
        out[f] = {'top': round(float(m[top].mean()), 3), 'all': round(float(m.mean()), 3)}
    pc = collections.Counter(pages[top]); pa = collections.Counter(pages)
    out['pages_ratio'] = {p: round(pc[p] / max(1, pa[p]) / (top.mean()), 2) for p in sorted(pa)}
    out['tokens'] = [(D.folio[i], int(D.li[i]), int(D.k[i]), D.word[i], round(float(D.care[i]), 2))
                     for i in np.argsort(-D.care)[:40]]
    return out


def instrument(D):
    # raw (page-z, not residualised) features at the last word of a line vs line-interior words of the same
    # glyph count: scribes are known to compress at line ends; a working instrument should see it
    nw = np.array([r['nw'] for r in D.R]); last = D.k == nw - 1; inner = (D.k > 0) & ~last
    out = {}
    for f in ['wpg', 'hgt', 'upright', 'ncomp', 'gapcv', 'irr', 'swid', 'dark']:
        v = D.raw[f]; ds = []
        for g in np.unique(D.g):
            a = v[last & (D.g == g)]; b = v[inner & (D.g == g)]
            if len(a) > 10 and len(b) > 10:
                ds.append((a.mean() - b.mean(), len(a)))
        d = sum(x * n for x, n in ds) / sum(n for _, n in ds)
        out[f] = round(float(d), 3)
    return out


if __name__ == '__main__':
    vf, _ = X.voynich_freq(); lf = X.latin_freq()
    V = C1.Data('V'); L = C1.Data('L')
    res = {'V_instr': instrument(V), 'L_instr': instrument(L)}
    print('instr', res, flush=True)
    res.update({'V_power': power(V, vf), 'V_types': careful_types(V, vf), 'L_types': careful_types(L, lf),
           'V_where': where(V, vf), 'L_where': where(L, lf)})
    F = C1.Data('V', lambda f: f in ('f58r', 'f58v'))
    C1.NP = 400
    res['V_f58'] = C1.run(F, vf, 'V_f58')
    json.dump(res, open(os.path.join(X.CK, 'c2.json'), 'w'), indent=1)
    for k in ['V_types', 'L_types']:
        r = res[k]
        print(k, {a: b for a, b in r.items() if a not in ('top', 'bottom')})
        print(' top', [(x['w'], x['n'], x['z'], x['freq']) for x in r['top'][:15]])
        print(' bot', [(x['w'], x['n'], x['z'], x['freq']) for x in r['bottom'][-8:]])
    for k in ['V_where', 'L_where']:
        print(k, {a: b for a, b in res[k].items() if a not in ('tokens', 'pages_ratio')})
        print(' ', res[k]['tokens'][:20])
    print('V pages', res['V_where']['pages_ratio'])
