"""LA-78 engine: random hypothesis pool, time-travel scoring, controls."""
import sys, os, json, math, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la78_common import load, CK, sha  # noqa

ALPHA = 0.5


def tokens(docs, min_syl=2):
    T = []
    for di, d in enumerate(docs):
        for w in d['words']:
            if w['clean'] and len(w['syl']) >= min_syl and len(w['syl']) == len(w['s']):
                T.append(dict(doc=di, syl=w['syl'], word='-'.join(w['syl']), y1=w['y1'], init=w['init'], q=w['q']))
    return T


# ------------------------------------------------------------------ hypotheses
def make_pool(T, n_per=600, seed=0):
    rng = random.Random(seed)
    signs = sorted({s for t in T for s in t['syl']})
    S = {s: i for i, s in enumerate(signs)}
    first = np.array([S[t['syl'][0]] for t in T]); last = np.array([S[t['syl'][-1]] for t in T])
    ln = np.array([min(len(t['syl']), 4) - 2 for t in T])  # 0,1,2
    words = sorted({t['word'] for t in T}); W = {w: i for i, w in enumerate(words)}
    wid = np.array([W[t['word']] for t in T])
    pool = []
    for fam in ('FIN', 'INI', 'PAIR', 'LENFIN', 'WORD'):
        for k in range(n_per):
            K = rng.choice([2, 3, 4]) if fam not in ('PAIR', 'LENFIN') else 2
            part = np.array([rng.randrange(K) for _ in signs])
            if fam == 'FIN': f = part[last]
            elif fam == 'INI': f = part[first]
            elif fam == 'PAIR': f = part[first] * 2 + part[last]
            elif fam == 'LENFIN': f = part[last] * 3 + ln
            else:
                wp = np.array([rng.randrange(K) for _ in words])
                f = wp[wid]; part = wp
            pool.append(dict(name=f'{fam}{K}_{k}', fam=fam, K=K, f=f.astype(np.int64), part=part.tolist()))
    return pool, signs, words, dict(first=first, last=last, ln=ln, wid=wid)


def word_feature(T, wordset_classes, default):
    return np.array([wordset_classes.get(t['word'], default) for t in T], dtype=np.int64)


# ------------------------------------------------------------------ scoring
MODE = {'mode': 'raw', 'grp': None}   # raw | prior | site


def _ll(f, y, ytr_mask, yte_mask, ny, word_backoff=None):
    """bits per test token gained over the baseline; classes fitted on train tokens.
    mode raw:   baseline = train marginal p(y).
    mode prior: both model and baseline are re-weighted to the test-period marginal q(y)
                (label-shift correction; same 3 parameters for every hypothesis).
    mode site:  as prior, but q is the test token's own site marginal (site-matched)."""
    ftr, ytr = f[ytr_mask], y[ytr_mask]
    fte, yte = f[yte_mask], y[yte_mask]
    if len(yte) == 0:
        return np.nan
    nc = int(f.max()) + 1
    cnt = np.zeros((nc, ny)); np.add.at(cnt, (ftr, ytr), 1)
    marg = np.bincount(ytr, minlength=ny) + ALPHA
    pm = marg / marg.sum()
    P = (cnt + 2.0 * pm) / (cnt.sum(1, keepdims=True) + 2.0)
    mode = MODE['mode']
    if mode == 'raw':
        return float(np.mean(np.log2(P[fte, yte]) - np.log2(pm[yte])))
    if mode == 'prior':
        q = np.bincount(yte, minlength=ny) + ALPHA; q = q / q.sum()
        Q = np.broadcast_to(q, (len(yte), ny))
    else:
        g = MODE['grp'][yte_mask]
        Q = np.zeros((len(yte), ny))
        for s in np.unique(g):
            m = g == s
            q = np.bincount(yte[m], minlength=ny) + ALPHA; Q[m] = q / q.sum()
    A = P[fte] * Q / pm
    A = A / A.sum(1, keepdims=True)
    idx = np.arange(len(yte))
    return float(np.mean(np.log2(A[idx, yte]) - np.log2(Q[idx, yte])))


def score(f, Y, tr, te):
    """sum over observables of bits/token gain."""
    return sum(_ll(f, y, tr, te, ny) for y, ny in Y)


def cv_score(f, Y, tr, docid, k=5, seed=0):
    rng = np.random.RandomState(seed)
    ds = np.unique(docid[tr]); fold = dict(zip(ds, rng.randint(0, k, len(ds))))
    fo = np.array([fold.get(d, -1) for d in docid])
    vals = []
    for j in range(k):
        te = tr & (fo == j); tr2 = tr & (fo != j)
        vals.append(score(f, Y, tr2, te) * te.sum())
    return float(np.nansum(vals) / tr.sum())


def loso_score(f, Y, tr, grp):
    """leave-one-site-out inside the training years: each past site in turn plays the 'new find'."""
    tot = 0.0; n = 0
    for g in np.unique(grp[tr]):
        te = tr & (grp == g); tr2 = tr & (grp != g)
        if te.sum() < 5 or tr2.sum() < 50: continue
        v = score(f, Y, tr2, te)
        if not np.isnan(v): tot += v * te.sum(); n += te.sum()
    return tot / max(n, 1)


def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ok = ~(np.isnan(a) | np.isnan(b))
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(pool, Y, year, docid, windows, extra_mask=None, grp=None):
    """returns arrays: cv[h,w], fut[h,w], ins[h,w] (+ loso[h,w] if grp given)"""
    H = len(pool); nW = len(windows)
    cv = np.zeros((H, nW)); fut = np.zeros((H, nW)); ins = np.zeros((H, nW)); lo = np.zeros((H, nW))
    for wi, (a, b) in enumerate(windows):
        tr = year <= a; te = (year > a) & (year <= b)
        if extra_mask is not None:
            te = te & extra_mask[wi]
        for h, p in enumerate(pool):
            fut[h, wi] = score(p['f'], Y, tr, te)
            ins[h, wi] = score(p['f'], Y, tr, tr)
            cv[h, wi] = cv_score(p['f'], Y, tr, docid)
            if grp is not None: lo[h, wi] = loso_score(p['f'], Y, tr, grp)
    if grp is not None: return cv, fut, ins, lo
    return cv, fut, ins
