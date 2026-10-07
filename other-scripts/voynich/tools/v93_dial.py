"""v93 cycle 3: IT IS A DIALOGUE? (two speakers alternate; same-speaker turns share idiolect: I/you, address forms)

Zig-zag statistic on a sequence of turns u_t (random unit definitions): s_k = mean cosine(u_t, u_t+k);
Z = s2 - (s1 + s3) / 2. Alternating speakers make lag 2 (same speaker) beat lags 1 and 3 (other speaker);
an ordinary text, a copy generator or a shuffle decays monotonically.
Planted: Petrarch's Secretum (Augustine / Petrarch, 536 turns, Gutenberg 49450), turns as paragraphs, v72 code + surface;
kills: the same turns in shuffled order, one speaker only, non-dialogue plants, Voynich shuffles and generators.
"""
import os, sys, json, random, math, zlib
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v93_lib as L
import v72_lib as V


def petrarch(kind, seed=933, cap=120):
    T = json.load(open(os.path.join(L.CK, 'petrarch_turns.json')))['turns']
    turns = [t for t in T if len(t['w']) >= 4]
    if kind == 'PE_SHUF':
        rng = random.Random(5); turns = turns[:]; rng.shuffle(turns)
    elif kind == 'PE_MONO':
        turns = [t for t in turns if t['s'] == 'S. Augustine']
    paras = [[t['w'][:cap][i:i + 8] for i in range(0, len(t['w'][:cap]), 8)] for t in turns]
    pages = L._paginate(paras, kind)
    k = 0; spk = [t['s'] for t in turns]
    for p in pages:
        for l in p['lines']: l['lang'] = 'en'
        p['spk'] = []
    return L._encode(pages, None, seed)


def corpus(name):
    if name.startswith('PE_'): return petrarch(name)
    return L.corpus(name)


def unit_seq(pages, unit, isv):
    """-> list of (page index, split, words) in reading order."""
    out = []
    for pi, p in enumerate(pages):
        sp = L.split_of(p['id'], isv)
        if unit == 'para':
            for pa in L.paragraphs(p): out.append((pi, sp, [w for l in pa for w in l]))
        elif unit.startswith('run'):
            K = int(unit[3:]); Ls = [l['w'] for l in p['lines']]
            for i in range(0, len(Ls), K): out.append((pi, sp, [w for l in Ls[i:i + K] for w in l]))
        elif unit.startswith('chunk'):
            N = int(unit[5:]); ws = [w for l in p['lines'] for w in l['w']]
            for i in range(0, len(ws) - N // 2, N): out.append((pi, sp, ws[i:i + N]))
    return out


def vectors(seq, rep, band, weight):
    docs = [[L.rep(w, rep) for w in ws] for _, _, ws in seq]
    tf = Counter(w for d in docs for w in d)
    order = [w for w, _ in tf.most_common()]
    lo, hi = band
    voc = order[lo:hi]
    vi = {w: i for i, w in enumerate(voc)}
    X = np.zeros((len(docs), len(voc)), np.float32)
    for r, d in enumerate(docs):
        for w in d:
            j = vi.get(w)
            if j is not None: X[r, j] += 1
    if weight == 'bin': X = (X > 0).astype(np.float32)
    elif weight == 'tfidf':
        df = (X > 0).sum(0) + 1; X = X * np.log(len(docs) / df)[None, :].astype(np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return X / nrm, voc


def zigzag(seq, X, scope, splits):
    """mean lag-k cosine for k = 1, 2, 3 over windows t..t+3 that start in a unit of the given split(s)."""
    pg = np.array([s[0] for s in seq]); sp = np.array([s[1] for s in seq])
    n = len(seq); ok = np.ones(n - 3, bool)
    if scope == 'page': ok &= (pg[:-3] == pg[3:])
    else: ok &= (pg[3:] - pg[:-3] <= 1)
    ok &= np.isin(sp[:-3], splits)
    nz = np.linalg.norm(X, axis=1) > 0
    ok &= nz[:-3] & nz[1:-2] & nz[2:-1] & nz[3:]
    idx = np.nonzero(ok)[0]
    if len(idx) < 20: return None
    c = [np.sum(X[idx] * X[idx + k], 1) for k in (1, 2, 3)]
    d = c[1] - (c[0] + c[2]) / 2
    s = [float(x.mean()) for x in c]
    return dict(s=s, Z=float(d.mean()), z=float(d.mean() / (d.std() + 1e-9) * math.sqrt(len(d))), n=int(len(idx)))


def random_hyp(rng):
    return dict(unit=rng.choice(['para', 'para', 'run1', 'run2', 'run3', 'run4', 'chunk20', 'chunk40']),
                rep=rng.choice(['whole', 'core', 'frame', 'pre2', 'suf2']),
                band=rng.choice([(0, 20), (0, 50), (0, 200), (0, 1000), (20, 500), (50, 2000)]),
                weight=rng.choice(['bin', 'tf', 'tfidf']), scope=rng.choice(['page', 'global']))


def word_contrib(seq, X, voc, scope, splits, k=25):
    """which words make lag 2 beat lags 1 and 3 (summed product contributions)."""
    pg = np.array([s[0] for s in seq]); sp = np.array([s[1] for s in seq]); n = len(seq)
    ok = (pg[:-3] == pg[3:]) if scope == 'page' else (pg[3:] - pg[:-3] <= 1)
    ok &= np.isin(sp[:-3], splits); idx = np.nonzero(ok)[0]
    c = (X[idx] * X[idx + 2]).sum(0) - 0.5 * ((X[idx] * X[idx + 1]).sum(0) + (X[idx] * X[idx + 3]).sum(0))
    o = np.argsort(-c)[:k]
    return [(voc[i], float(c[i])) for i in o]
