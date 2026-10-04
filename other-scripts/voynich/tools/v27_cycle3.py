"""v27 cycle 3: TWO WITNESSES. If body lines are shuffled cards of a once-continuous text, two INDEPENDENT
continuity witnesses must point to the same hidden successors, without ever looking at the written order:
  witness A: a junction hypothesis (end feature of the last word of line i -> start feature of the first word
             of line j, its table trained only on within-line adjacent word pairs);
  witness B: lexical similarity of line INTERIORS (first and last word of each line excluded).
Per paragraph (3..12 lines, head fixed, exact DP) the best path under A and under B are compared: shared
undirected edges minus the random-path expectation. MASSIVE RANDOM GUESSING: 2,000 random junction hypotheses
(random end/start features, random glyph-class merges). Nulls: cross-page cards (same section), F5
resynthesis; planted: F3 + hidden continuity (junction and copy); positive: Latin / Italian herbals with
lines shuffled. Survivors (beating every null hypothesis score) are re-tested on held-out pages and IT2a.
"""
import sys, os, json, random, time, math
from multiprocessing import Pool
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_lib as L

NH = int(os.environ.get('V27_NH', 2000))
FEATS = ['l1', 'l2', 'l3', 'f1', 'f2', 'len', 'fl', 'w']


def feat(w, kind, cmap):
    m = lambda s: ''.join(cmap.get(c, c) for c in s)
    if kind == 'l1': return m(w[-1:])
    if kind == 'l2': return m(w[-2:])
    if kind == 'l3': return m(w[-3:])
    if kind == 'f1': return m(w[:1])
    if kind == 'f2': return m(w[:2])
    if kind == 'len': return str(min(len(w), 8))
    if kind == 'fl': return m(w[:1] + w[-1:])
    return w


def random_hyp(rng, units):
    k = rng.choice([0, 0, 2, 3, 4, 6, 10])
    cmap = {}
    if k:
        for u in units: cmap[u] = chr(65 + rng.randrange(k))
    return dict(a=rng.choice(['last', 'last', 'pen']), ea=rng.choice(FEATS), b=rng.choice(['first', 'first', 'second']),
                sb=rng.choice(FEATS), cmap=cmap, k=k)


def hyp_tables(C, h):
    """P(start feature of word k+1 | end feature of word k) from WITHIN-LINE pairs; LLR table."""
    big = defaultdict(Counter); uni = Counter()
    PC = C if isinstance(C, Counter) else pair_counts(C)
    fa, fb = {}, {}
    for (x, y), c in PC.items():
        a = fa.get(x)
        if a is None: a = fa[x] = feat(x, h['ea'], h['cmap'])
        b = fb.get(y)
        if b is None: b = fb[y] = feat(y, h['sb'], h['cmap'])
        big[a][b] += c; uni[b] += c
    N = sum(uni.values()); V = len(uni) + 1
    tot = {a: sum(c.values()) for a, c in big.items()}
    cache = {}
    def llr(a, b):
        k = (a, b)
        if k in cache: return cache[k]
        pu = (uni.get(b, 0) + 0.5) / (N + 0.5 * V)
        c = big.get(a); n = tot.get(a, 0)
        pb = ((c.get(b, 0) if c else 0) + 5 * pu) / (n + 5)
        cache[k] = v = math.log(pb / pu)
        return v
    return llr


def pair_counts(C):
    PC = Counter()
    for p in C:
        for pa in p['paras']:
            for l in pa:
                for x, y in zip(l, l[1:]): PC[(x, y)] += 1
    return PC


def para_list(C, which=None):
    out = []
    for pi, p in enumerate(C):
        if which is not None and (pi % 2) != which: continue
        for pa in p['paras']:
            if 3 <= len(pa) <= 15: out.append(pa)
    return out


def interior_L(sc, pa):
    ins = [l[1:-1] if len(l) > 2 else [] for l in pa]
    n = len(pa); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = sum(sc.idf.get(w, 0) for w in set(ins[i]) & set(ins[j]))
            M[i, j] = M[j, i] = v
    return M


def pick(l, which):
    if which == 'last': return l[-1]
    if which == 'pen': return l[-2] if len(l) > 1 else l[-1]
    if which == 'first': return l[0]
    return l[1] if len(l) > 1 else l[0]


def agreement(PS, pathsB, h, llr, split=None, rng=None):
    """shared undirected edges between best A-path and B-path, minus random expectation. Returns per-paragraph
    (shared, expectation, variance) so that page-half splits are sums over the same list."""
    rng = rng or np.random.RandomState(0)
    per = []
    for pa, pb in zip(PS, pathsB):
        n = len(pa)
        ends = [feat(pick(l, h['a']), h['ea'], h['cmap']) for l in pa]
        sts = [feat(pick(l, h['b']), h['sb'], h['cmap']) for l in pa]
        ue = sorted(set(ends)); us = sorted(set(sts))
        T = np.array([[llr(a, b) for b in us] for a in ue])
        ie = [ue.index(x) for x in ends]; js = [us.index(x) for x in sts]
        M = T[ie][:, js]; np.fill_diagonal(M, 0.0)
        M += 1e-7 * rng.rand(n, n)            # random tie-breaking
        _, p = L.best_path(M)
        ea = {frozenset(e) for e in zip(p, p[1:])}
        eb = {frozenset(e) for e in zip(pb, pb[1:])}
        q = 1.0 / (n - 1) + (n - 2) * 2.0 / (n - 1)
        per.append((len(ea & eb), q, q * (1 - q / (n - 1))))
    return per


def zsum(per, idx=None):
    sel = per if idx is None else [per[i] for i in idx]
    sh = sum(x[0] for x in sel); ex = sum(x[1] for x in sel); var = sum(x[2] for x in sel)
    return [sh, round(ex, 2), round((sh - ex) / math.sqrt(var), 3)]


def build(nm, ver, s):
    C = L.corpus(nm)
    if ver == 'real': return C
    if ver == 'cross': return L.cross_page(C, 200 + s)
    if ver == 'F5': return L.forge(C, 'F5', 100 + s)[0]
    if ver == 'plant': return L.plant(C, 300 + s, rho_j=0.3, rho_l=0.3)
    if ver == 'plantJ': return L.plant(C, 300 + s, rho_j=0.5)


VERS = [('LA', 'real', 0), ('IT', 'real', 0), ('LA', 'cross', 1), ('IT', 'cross', 1),
        ('ZL', 'real', 0), ('ZL', 'cross', 1), ('ZL', 'cross', 2), ('ZL', 'F5', 1), ('ZL', 'plant', 1),
        ('IT2a', 'real', 0), ('IT2a', 'cross', 1)]


def prep(v):
    nm, ver, s = v
    C = build(nm, ver, s)
    sc = L.Scorer(C)
    rng = np.random.RandomState(7)
    PS, pathsB, parity = [], [], []
    for pi, p in enumerate(C):
        for pa in p['paras']:
            if not (3 <= len(pa) <= 12): continue
            MB = interior_L(sc, pa)
            if MB.max() <= 0: continue        # no witness B in this paragraph
            MB = MB + 1e-7 * rng.rand(*MB.shape)
            PS.append(pa); pathsB.append(L.best_path(MB)[1]); parity.append(pi % 2)
    return C, (PS, pathsB, parity)


def worker(args):
    v, hs = args
    fn = os.path.join(L.CK, 'c3_%s_%s_%d.json' % v)
    if os.path.exists(fn): return json.load(open(fn))
    C, D = prep(v)
    units = sorted({c for p in C for pa in p['paras'] for l in pa for w in l for c in w})
    res = []
    PCC = pair_counts(C)
    for hi, hd in enumerate(hs):
        h = random_hyp(random.Random(hd), units)
        llr = hyp_tables(PCC, h)
        PS, pB, par = D
        per = agreement(PS, pB, h, llr, rng=np.random.RandomState(hd))
        r = {'hi': hi, 'None': zsum(per), '0': zsum(per, [i for i, x in enumerate(par) if x == 0]),
             '1': zsum(per, [i for i, x in enumerate(par) if x == 1])}
        res.append(r)
        if hi % 200 == 0: print(v, hi, r, flush=True)
    # the full-word junction witness (the cycle-1 J scorer) as hypothesis -1
    sc = L.Scorer(C)
    PS, pB, _ = D
    sh = ex = var = 0
    jr = np.random.RandomState(11)
    for pa, pb in zip(PS, pB):
        n = len(pa); _, p = L.best_path(sc.matrix(pa, 'J') + 1e-7 * jr.rand(n, n))
        sh += len({frozenset(e) for e in zip(p, p[1:])} & {frozenset(e) for e in zip(pb, pb[1:])})
        q = 1.0 / (n - 1) + (n - 2) * 2.0 / (n - 1); ex += q; var += q * (1 - q / (n - 1))
    out = dict(v=v, hyps=res, Jword=[sh, ex, (sh - ex) / math.sqrt(var)], npara=len(PS))
    json.dump(out, open(fn, 'w'))
    return out


if __name__ == '__main__':
    hs = list(range(NH))
    t = time.time()
    with Pool(2) as P:
        R = P.map(worker, [(v, hs) for v in VERS], chunksize=1)
    print('done', round(time.time() - t))
    for r in R:
        z = np.array([h['None'][2] for h in r['hyps']])
        print(r['v'], r['npara'], 'Jword z %.2f' % r['Jword'][2], 'hyp z mean %.2f sd %.2f max %.2f q99 %.2f' % (
            z.mean(), z.std(), z.max(), np.quantile(z, .99)))
