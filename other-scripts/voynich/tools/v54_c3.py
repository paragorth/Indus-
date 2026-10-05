"""v54 cycle 3: (a) stemma test: do the variant readings of near-repeat witnesses cluster by hand (and quire)
beyond section? Hand labels permuted among pages within a section (200x). (b) transcriber disagreements (ZL vs
IT2a, same line, same word count) as a third set of witnesses: are the scribal variant classes the same as the
reading-noise classes?"""
import sys, os, json, collections, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V

def variant_sides(pages):
    toks, pairs, _ = V.families(pages)
    rows = []   # (op key, page of side holding first variant, page of other side)
    for i, j, *_ in pairs:
        pi, pj = toks[i][1], toks[j][1]
        if pi == pj: continue
        for k in range(3):
            a, b = toks[i + k][2], toks[j + k][2]
            if a == b: continue
            for o in V.align_ops(a, b):
                if o[0] != 'S': continue
                x, y = o[1], o[2]
                # which side carries x?
                side_x = pi if (x in a and y not in a) or (a.count(x) > b.count(x)) else pj
                side_y = pj if side_x == pi else pi
                rows.append((o[:3], side_x, side_y))
    return rows

def mi_stat(rows, lab):
    """Sum over ops (n>=20) of n * MI(label of witness ; variant chosen)."""
    by = collections.defaultdict(list)
    for k, px, py in rows: by[k] += [(lab[px], 0), (lab[py], 1)]
    tot = 0.0
    for k, v in by.items():
        if len(v) < 40: continue
        n = len(v); c = collections.Counter(v); cl = collections.Counter(a for a, _ in v); cv = collections.Counter(b for _, b in v)
        tot += sum(m * math.log2(m * n / (cl[a] * cv[b])) for (a, b), m in c.items())
    return tot

def stemma(name, pages, var, strat='sec', nperm=200, seed=1):
    rows = variant_sides(pages)
    lab = [p['vars'].get(var) for p in pages]
    obs = mi_stat(rows, lab)
    rng = random.Random(seed); null = []
    groups = collections.defaultdict(list)
    for i, p in enumerate(pages): groups[p['vars'].get(strat)].append(i)
    for _ in range(nperm):
        L = list(lab)
        for g in groups.values():
            vals = [lab[i] for i in g]; rng.shuffle(vals)
            for i, v in zip(g, vals): L[i] = v
        null.append(mi_stat(rows, L))
    m = sum(null) / len(null); sd = (sum((x - m) ** 2 for x in null) / len(null)) ** .5
    p = (1 + sum(x >= obs for x in null)) / (1 + len(null))
    return dict(name=name, var=var, strat=strat, nrows=len(rows), obs=round(obs, 1), null=round(m, 1), z=round((obs - m) / (sd or 1), 2), p=round(p, 4))

def transcriber():
    import vlib
    a = json.load(open(os.path.join(V.ROOT, 'data', 'derived', 'ZL3b_lines.json')))
    b = {(r['folio'], r['n']): r for r in json.load(open(os.path.join(V.ROOT, 'data', 'derived', 'IT2a_lines.json')))}
    ops = collections.Counter(); occ = collections.Counter(); nw = 0; nd = 0
    for r in a:
        s = b.get((r['folio'], r['n']))
        if not s or len(s['words']) != len(r['words']) or r['ltype'] != 'P': continue
        for x, y in zip(r['words'], s['words']):
            if '?' in x or '?' in y: continue
            gx, gy = ''.join(vlib.glyphs(x)), ''.join(vlib.glyphs(y))
            nw += 1
            for c in gx + gy: occ[c] += 1
            if gx != gy and abs(len(gx) - len(gy)) <= 2:
                nd += 1
                for o in V.align_ops(gx, gy): ops[o[:-1]] += 1
    return ops, occ, nw, nd

def rank_corr(d1, d2):
    keys = sorted(set(d1) | set(d2))
    def rk(d):
        s = sorted(keys, key=lambda k: d.get(k, 0)); return {k: i for i, k in enumerate(s)}
    r1, r2 = rk(d1), rk(d2); n = len(keys); m = (n - 1) / 2
    num = sum((r1[k] - m) * (r2[k] - m) for k in keys); den = sum((r1[k] - m) ** 2 for k in keys)
    return num / den

if __name__ == '__main__':
    out = dict(stemma=[], trans={})
    zl = V.voynich('ZL3b'); it = V.voynich('IT2a')
    bru, M = V.brumati()
    for nm, P, vars_ in [('BRU_planted', bru, [('hand', 'sec')]), ('ZL', zl, [('hand', 'sec'), ('quire', 'sec'), ('lang', 'sec'), ('sec', None)]),
                         ('IT', it, [('hand', 'sec'), ('quire', 'sec')]),
                         ('ZL_markov', V.null_markov(zl, 3), [('hand', 'sec'), ('quire', 'sec'), ('sec', None)]),
                         ('ZL_markov_hand', V.null_markov(zl, 3, key='hand'), [('hand', 'sec')])]:
        for var, strat in vars_:
            s = stemma(nm, P, var, strat if strat else '__none__'); out['stemma'].append(s); print(json.dumps(s), flush=True)
    # transcriber classes vs scribal collation classes vs line-shuffle chance classes
    tops, tocc, nw, nd = transcriber()
    def norm(ops, occ):
        d = {}
        for k, n in ops.items():
            if k[0] == 'S':
                if occ[k[1]] >= 50 and occ[k[2]] >= 50: d[k] = n / math.sqrt(occ[k[1]] * occ[k[2]])
            elif occ[k[1]] >= 50: d[k] = n / occ[k[1]]
        return d
    T = norm(tops, tocc)
    toks, pairs, _ = V.families(zl); col = V.collate(toks, pairs); S = norm(col['ops'], col['occ'])
    toks, pairs, _ = V.families(V.null_lineshuf(zl, 5)); coln = V.collate(toks, pairs); Sn = norm(coln['ops'], coln['occ'])
    # excess of real collation over chance pairing, per op
    E = {k: S.get(k, 0) / (Sn.get(k, 0) + 1e-3) for k in set(S) | set(Sn)}
    keys = [k for k in set(S) | set(T) | set(Sn) if (col['ops'][k] + coln['ops'][k] >= 20)]
    sub = lambda d: {k: d.get(k, 0) for k in keys}
    out['trans'] = dict(words=nw, disagree=nd, top=[(round(v, 4), tops[k], k) for k, v in sorted(T.items(), key=lambda x: -x[1])[:15]],
                        rho_trans_scribal=round(rank_corr(sub(T), sub(S)), 3), rho_trans_chance=round(rank_corr(sub(T), sub(Sn)), 3),
                        rho_scribal_chance=round(rank_corr(sub(S), sub(Sn)), 3), rho_trans_excess=round(rank_corr(sub(T), sub(E)), 3),
                        excess_top=[(round(E[k], 2), col['ops'][k], coln['ops'][k], k) for k in sorted(keys, key=lambda k: -E.get(k, 0))[:15]],
                        excess_bottom=[(round(E[k], 2), col['ops'][k], coln['ops'][k], k) for k in sorted(keys, key=lambda k: E.get(k, 0))[:10]],
                        nkeys=len(keys), pairs_real=len(pairs))
    print(json.dumps(out['trans']), flush=True)
    json.dump(out, open(os.path.join(V.CK, 'c3.json'), 'w'))
