"""v27 cycle 3: TWO WITNESSES. If body lines are shuffled cards of a once-continuous text, two INDEPENDENT
continuity witnesses must point to the same hidden successors, without ever looking at the written order:
  witness A: a junction hypothesis (end feature of the last word of line i -> start feature of the first word
             of line j, its table trained only on within-line adjacent word pairs);
  witness B: lexical similarity of line INTERIORS (first and last word of each line excluded).
Per paragraph (3..15 lines, head fixed, exact DP) the best path under A and under B are compared: shared
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
    for p in C:
        for pa in p['paras']:
            for l in pa:
                for x, y in zip(l, l[1:]):
                    a = feat(x, h['ea'], h['cmap']); b = feat(y, h['sb'], h['cmap'])
                    big[a][b] += 1; uni[b] += 1
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


def agreement(PS, pathsB, h, llr):
    """shared undirected edges between best A-path and B-path, minus random expectation, summed over paragraphs."""
    sh = 0; ex = 0.0; var = 0.0
    for pa, pb in zip(PS, pathsB):
        n = len(pa)
        ends = [feat(pick(l, h['a']), h['ea'], h['cmap']) for l in pa]
        sts = [feat(pick(l, h['b']), h['sb'], h['cmap']) for l in pa]
        M = np.array([[llr(ends[i], sts[j]) if i != j else 0.0 for j in range(n)] for i in range(n)])
        _, p = L.best_path(M)
        ea = {frozenset(e) for e in zip(p, p[1:])}
        eb = {frozenset(e) for e in zip(pb, pb[1:])}
        sh += len(ea & eb)
        q = 1.0 / (n - 1) + (n - 2) * 2.0 / (n - 1)
        ex += q; var += q * (1 - q / (n - 1))
    return sh, ex, (sh - ex) / math.sqrt(var)


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
    out = {}
    for which in (None, 0, 1):
        PS = para_list(C, which)
        pathsB = [L.best_path(interior_L(sc, pa))[1] for pa in PS]
        out[which] = (PS, pathsB)
    return C, out


def worker(args):
    v, hs = args
    fn = os.path.join(L.CK, 'c3_%s_%s_%d.json' % v)
    if os.path.exists(fn): return json.load(open(fn))
    C, D = prep(v)
    units = sorted({c for p in C for pa in p['paras'] for l in pa for w in l for c in w})
    res = []
    for hi, hd in enumerate(hs):
        h = random_hyp(random.Random(hd), units)
        llr = hyp_tables(C, h)
        r = {'hi': hi}
        for which in ((None, 0, 1) if hi < NH else (None,)):
            PS, pB = D[which]
            sh, ex, z = agreement(PS, pB, h, llr)
            r[str(which)] = [sh, round(ex, 2), round(z, 3)]
        res.append(r)
    # the full-word junction witness (the cycle-1 J scorer) as hypothesis -1
    sc = L.Scorer(C)
    PS, pB = D[None]
    sh = ex = var = 0
    for pa, pb in zip(PS, pB):
        n = len(pa); _, p = L.best_path(sc.matrix(pa, 'J'))
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
