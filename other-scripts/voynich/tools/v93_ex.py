"""v93 cycle 2: THE MEANING IS IN WHAT EXCLUDES WHAT (ecologists' checkerboards on the Voynich).

Change of medium: pages are islands, word types are species. Diamond's assembly rules ask which species pairs never
share an island beyond what fixed island sizes and fixed species frequencies allow (checkerboard pairs, tested with a
curveball null that keeps both margins). In a text that names things, sets of ALTERNATIVES (hot / cold, root / leaf,
one colour per plant) should form checkerboards; spelling habit and copying make attraction.

Random guessing: thousands of random word sets (k = 2..4) and random unit/representation choices scored for
exclusion on train units; survivors re-tested on held-out units against the same set in shuffles and generators.
"""
import os, sys, json, random, math, zlib
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v93_lib as L
import v72_lib as V


def herbal_plant(src='v21_IT', seed=931, cut=180):
    """each herbal entry -> one page (lines of 8 words), verbose payload code + v72 surface."""
    T = L.texts()
    pages = []
    for i, u in enumerate(T[src]['units']):
        ws = [L._clean(t) for t in u['tok']]; ws = [w for w in ws if w][:cut]
        if len(ws) < 40: continue
        pages.append(dict(id='%s%03d' % (src[:4], i), sec='S%d' % (i * 4 // len(T[src]['units'])), lang='-', hand='-',
                          lines=[dict(w=ws[k:k + 8], ps=(k == 0), lang='x') for k in range(0, len(ws), 8)]))
    return L._encode(pages, None, seed)


def corpus(name):
    if name in ('HERB_IT', 'HERB_LA', 'HERB_GE', 'HERB_KO'):
        src = dict(HERB_IT='v21_IT', HERB_LA='v21_LA', HERB_GE='gerard_pages', HERB_KO='konrad_plants')[name]
        return herbal_plant(src)
    if name.endswith('~gen'):                      # copy-and-vary generator fitted to the plant (kill for the plant)
        return V.gen_selfcit(corpus(name[:-4]), seed=5)
    return L.corpus(name)


def units(pages, level):
    """-> list of (unit id, stratum, split, words)."""
    out = []
    isv = 'sec' in pages[0] and pages[0]['id'].startswith('f')
    for p in pages:
        sp = L.split_of(p['id'], isv)
        strat = (p.get('sec', '-'), p.get('lang', '-'))
        if level == 'page':
            ws = [w for l in p['lines'] for w in l['w']]
            if len(ws) >= 30: out.append((p['id'], strat, sp, ws))
        else:
            for k, pa in enumerate(L.paragraphs(p)):
                ws = [w for l in pa for w in l]
                if len(ws) >= 25: out.append(('%s.%d' % (p['id'], k), strat, sp, ws))
    return out


def presence(U, rep, vocab=None, dfmin=3, dfmax=0.4):
    docs = [set(L.rep(w, rep) for w in ws) for _, _, _, ws in U]
    if vocab is None:
        df = Counter(w for d in docs for w in d)
        vocab = sorted(w for w, c in df.items() if c >= dfmin and c <= dfmax * len(docs))
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.uint8)
    for r, d in enumerate(docs):
        for w in d:
            if w in vi: X[r, vi[w]] = 1
    return X, vocab


def curveball(X, rng, strata, steps=None):
    """swap randomisation keeping row and column sums, rows only exchanged inside a stratum."""
    X = X.copy(); rows = [list(np.nonzero(X[i])[0]) for i in range(X.shape[0])]
    rs = [set(r) for r in rows]
    groups = defaultdict(list)
    for i, s in enumerate(strata): groups[s].append(i)
    groups = [g for g in groups.values() if len(g) >= 2]
    steps = steps or 3 * X.shape[0]
    for _ in range(steps):
        g = groups[rng.randrange(len(groups))]
        a, b = rng.sample(g, 2)
        A, B = rs[a], rs[b]
        oa = list(A - B); ob = list(B - A)
        if not oa or not ob: continue
        pool = oa + ob; rng.shuffle(pool)
        na = set(pool[:len(oa)]); nb = set(pool[len(oa):])
        rs[a] = (A & B) | na; rs[b] = (A & B) | nb
    Y = np.zeros_like(X)
    for i, r in enumerate(rs):
        if r: Y[i, list(r)] = 1
    return Y


def cooc_z(X, strata, R=40, seed=0):
    """observed pair co-occurrence and curveball null mean/sd (V x V)."""
    rng = random.Random(seed)
    Xf = X.astype(np.float32)
    obs = Xf.T @ Xf
    s1 = np.zeros_like(obs); s2 = np.zeros_like(obs)
    for _ in range(R):
        Y = curveball(X, rng, strata).astype(np.float32)
        c = Y.T @ Y; s1 += c; s2 += c * c
    mu = s1 / R; sd = np.sqrt(np.maximum(s2 / R - mu * mu, 0)) + 0.5
    return obs, mu, sd


def set_score(obs, mu, sd, S):
    """exclusion score of a word set = mean over member pairs of (mu - obs) / sd (positive = checkerboard)."""
    z = [(mu[a, b] - obs[a, b]) / sd[a, b] for i, a in enumerate(S) for b in S[i + 1:]]
    return float(np.mean(z))
