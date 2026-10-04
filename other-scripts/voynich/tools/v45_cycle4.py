"""v45 cycle 4: massive random guessing over residual extractors, held-out pages.

Each hypothesis = a random way to read the residual: a word feature (whole word, prefix k, suffix k, core =
word minus first and last unit, glyph bigrams), a token filter (all, only the top-q surprise words, no
line-initial words, no paragraph-first lines), a residual weighting (observed minus rule-expected, Pearson;
or surprise-weighted counts minus their mean), and centred or plain cosine. Score = partial Mantel r with the
drawing composite (v38 confounds) on the discovery half of leaves; the top 20 are re-scored on the held-out
half. The whole search is rerun on visuals permuted within strata (NNULL times) for the null.
Corpora: V (Voynich herbal pages), GEN0 and PL (no message), GEw / GEl (Gerard planted, positive).
Usage: python3 v45_cycle4.py NAME
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v45_lib import *
from v38_lib import Partial, upper, zs, perm
import v38_cycle2
v38_cycle2.CACHE = os.path.join(SCR, 'v38img')
from v38_cycle1 import voynich_setup
from v38_cycle2 import gerard_setup

NH = int(os.environ.get('NH', 1500)); NNULL = int(os.environ.get('NNULL', 25)); TOPK = 20


def feat(w, f, k):
    if f == 'word': return [w]
    if f == 'pre': return [w[:k]]
    if f == 'suf': return [w[-k:]]
    if f == 'core': return [w[1:-1] or '_']
    return [w[i:i + 2] for i in range(len(w) - 1)] or [w]


def page_vectors(R, ids, spec, thr):
    f, k, filt, wt = spec['f'], spec['k'], spec['filt'], spec['wt']
    rows = []
    for pid in ids:
        recs = R[pid]['recs']; E = R[pid]['E']
        O = Counter(); X = Counter()
        keepw = Counter()
        for r in recs:
            w, qi, li, kk, n, lp, H, rank, u, pm, nc = r
            if filt == 'surp' and (-lp - H) < thr[spec['q']]: continue
            if filt == 'noinit' and kk == 0: continue
            if filt == 'nopf' and li == 0: continue
            keepw[w] += 1
            for x in feat(w, f, k):
                O[x] += 1; X[x] += (-lp - H)
        Ef = Counter()
        if wt == 'pearson':
            frac = 1.0 if filt in ('all',) else (sum(keepw.values()) / max(1, len(recs)))
            for w, e in E.items():
                for x in feat(w, f, k): Ef[x] += e * frac
        rows.append((O, X, Ef))
    voc = Counter()
    for O, X, Ef in rows: voc.update(set(O))
    voc = [x for x, c in voc.items() if c >= 2]
    vi = {x: j for j, x in enumerate(voc)}
    M = np.zeros((len(ids), len(voc)))
    for i, (O, X, Ef) in enumerate(rows):
        for x, c in O.items():
            j = vi.get(x)
            if j is None: continue
            M[i, j] = (c - Ef[x]) / math.sqrt(Ef[x] + 0.5) if wt == 'pearson' else X[x]
        if wt == 'pearson':
            for x, e in Ef.items():
                j = vi.get(x)
                if j is not None and x not in O: M[i, j] = -e / math.sqrt(e + 0.5)
    if wt == 'surpw': M = M - M.mean(0)
    elif spec['centre']: M = M - M.mean(0)
    return cos(M)


def rand_spec(rng):
    f = str(rng.choice(['word', 'word', 'pre', 'suf', 'core', 'bi']))
    return dict(f=f, k=int(rng.integers(1, 4)), filt=str(rng.choice(['all', 'surp', 'noinit', 'nopf'])),
                q=str(rng.choice(['0.5', '0.7', '0.85'])), wt=str(rng.choice(['pearson', 'pearson', 'surpw'])),
                centre=bool(rng.random() < 0.5))


def sub_r(A, B, part, idx):
    return float(np.mean(zs(part.res(upper(A[np.ix_(idx, idx)]))) * zs(part.res(upper(B[np.ix_(idx, idx)])))))


if __name__ == '__main__':
    nm = sys.argv[1]
    rng = np.random.default_rng(454)
    R = residual(nm)
    allrec = [r for v in R.values() for r in v['recs']]
    ex = np.array([-r[5] - r[6] for r in allrec])
    thr = {q: float(np.quantile(ex, float(q))) for q in ('0.5', '0.7', '0.85')}
    if nm.startswith('GE') and not nm.startswith('GEN'):
        keys, words, vis, conf = gerard_setup(); strata = None
        ids = ['ge' + k for k in keys]; leaves = [int(k) // 2 for k in keys]; src = 'gerard'
    else:
        pages, keys, words, vis, conf, strata = voynich_setup()
        sel = [i for i, k in enumerate(keys) if k in R]
        keys = [keys[i] for i in sel]; conf = {a: b[np.ix_(sel, sel)] for a, b in conf.items()}
        strata = [strata[i] for i in sel]; ids = keys; leaves = [pages[i]['leafnum'] for i in sel]; src = 'voynich'
    v2 = json.load(open(os.path.join(vlib.DATA, 'derived', 'v38_vis2_%s.json' % src)))
    F = np.concatenate([np.array([v2[k][f] for k in keys], float) for f in ('effb0', 'dinov2')], 1)
    F = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1)
    Vm = cos(F)
    n = len(ids)
    ul = sorted(set(leaves)); rng.shuffle(ul); A = set(ul[:len(ul) // 2])
    iA = np.array([i for i in range(n) if leaves[i] in A]); iB = np.array([i for i in range(n) if leaves[i] not in A])
    pA = Partial([C[np.ix_(iA, iA)] for C in conf.values()], len(iA))
    pB = Partial([C[np.ix_(iB, iB)] for C in conf.values()], len(iB))
    specs = [rand_spec(rng) for _ in range(NH)]
    cache = {}
    for h, s in enumerate(specs):
        key = json.dumps(s, sort_keys=True)
        if key not in cache: cache[key] = page_vectors(R, ids, s, thr)
    print(nm, 'distinct text views', len(cache), flush=True)

    def search(V):
        sc = sorted(((sub_r(V, cache[json.dumps(s, sort_keys=True)], pA, iA), h) for h, s in enumerate(specs)), reverse=True)[:TOPK]
        rep = [sub_r(V, cache[json.dumps(specs[h], sort_keys=True)], pB, iB) for _, h in sc]
        return sc[0][0], float(np.mean(rep)), [h for _, h in sc]

    bA, mB, top = search(Vm)
    nulls = []
    for k in range(NNULL):
        p = perm(n, rng, strata)
        nulls.append(search(Vm[np.ix_(p, p)])[:2])
    nb = np.array(nulls)
    out = dict(name=nm, n=n, NH=NH, views=len(cache), bestA=bA, meanB=mB,
               null_bestA=[float(nb[:, 0].mean()), float(nb[:, 0].std())], null_meanB=[float(nb[:, 1].mean()), float(nb[:, 1].std())],
               zA=float((bA - nb[:, 0].mean()) / nb[:, 0].std()), zB=float((mB - nb[:, 1].mean()) / nb[:, 1].std()),
               pB=float((1 + (nb[:, 1] >= mB).sum()) / (1 + NNULL)),
               top=[specs[h] for h in top[:5]],
               top_f=dict(Counter(specs[h]['f'] for h in top)), top_filt=dict(Counter(specs[h]['filt'] for h in top)),
               top_wt=dict(Counter(specs[h]['wt'] for h in top)))
    jsave(f'c4_{nm}.json', out)
    print(json.dumps(out), flush=True)
