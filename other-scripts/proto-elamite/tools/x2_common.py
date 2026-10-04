#!/usr/bin/env python3
"""X-2 engine: let two commodity-account scripts translate each other.

Pipeline (identical for every pair of corpora):
 1. Every entry occurrence of an item (commodity sign 'c:X' or designation word/sign 'w:X') gets
    SHARED contexts, script-free: quantity bin, sub-unit flag, round-ten flag, line kind
    (entry / header / trailer / inner unnumbered), total line, relative position in the list,
    list length, number of commodities on the document, document has a total, designation
    length, multi-commodity line, quantity relative to the document median;
    and INTERNAL contexts, script-specific: other commodities on the document, the entry's
    commodity, the entry's other designation tokens.
 2. Item x context counts -> PPMI (context smoothing 0.75) -> truncated SVD (d dims):
    item vectors U S^1/2, context vectors V S^1/2 (unit-normalised).
 3. Alignment of space A to space B, four ways:
    seed  : orthogonal Procrustes on the shared-context vectors (the 'numerals' of UMT),
            then 5 rounds of self-learning on mutual nearest neighbours (CSLS) of same-type items
    unsup : no anchors; R random orthogonal starts + self-learning; best start by objective
    gw    : entropic Gromov-Wasserstein between within-script cosine-distance matrices of the
            items, R random initial couplings, best by GW loss (anchor-free)
    prof  : no learning: cosine of the shared-context PPMI profiles (baseline)
    freq  : frequency rank only (baseline)
 4. Output: similarity matrix commodity(A) x commodity(B) and word(A) x word(B).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, math, random
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DX = os.path.join(HERE, '..', 'data', 'x2')


BIG = os.environ.get('X2_BIG', os.path.join(os.environ.get('X2_SCRATCH', '/tmp/x2_scratch'), 'x2'))


def load(name):
    # the Ur III corpus (105 MB) is kept outside the repository; rebuild with x2_build.py UR3
    f = os.path.join(DX, 'corpus_%s.json' % name)
    if not os.path.exists(f):
        f = os.path.join(BIG, 'corpus_%s.json' % name)
    return json.load(open(f))


def jsonl(path):
    """Read a checkpoint .jsonl, or its gzipped copy."""
    import gzip
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        return [json.loads(l) for l in gzip.open(path + '.gz', 'rt')]
    return [json.loads(l) for l in open(path)]


def qbin(q):
    if q is None:
        return 'none'
    for lim, lab in ((0, '0'), (1, '1'), (2, '2'), (4, '3-4'), (9, '5-9'), (19, '10-19'), (49, '20-49'),
                     (99, '50-99'), (499, '100-499')):
        if q <= lim:
            return lab
    return '500+'


def lenbin(n):
    for lim, lab in ((1, '1'), (3, '2-3'), (7, '4-7'), (15, '8-15')):
        if n <= lim:
            return lab
    return '16+'


def occurrences(docs, minc=8, minw=8, minw_docs=3, maxw=150, maxc=40):
    """Return items (list), per-item list of (shared ctx list, internal ctx list), item counts."""
    cc = Counter(); wc = Counter(); wd = defaultdict(set)
    for d in docs:
        for e in d['entries']:
            if e['com']:
                cc[e['com']] += 1
            for w in e['des']:
                wc[w] += 1; wd[w].add(d['id'])
    coms = sorted([c for c, n in cc.items() if n >= minc], key=lambda c: -cc[c])[:maxc]
    words = [w for w, n in wc.most_common() if n >= minw and len(wd[w]) >= minw_docs and w not in cc][:maxw]
    cset, wset = set(coms), set(words)
    occ = defaultdict(list)
    for d in docs:
        ents = d['entries']
        num = [i for i, e in enumerate(ents) if e['kind'] == 'E']
        rank = {i: r for r, i in enumerate(num)}
        nn = len(num)
        qs = [ents[i]['q'] for i in num if ents[i]['q']]
        med = float(np.median(qs)) if qs else None
        dcom = sorted({e['com'] for e in ents if e['com']})
        ncom = len(dcom)
        has_tot = any(e['tot'] for e in ents)
        doc_sh = ['dl:' + lenbin(nn), 'dc:' + str(min(ncom, 3)), 'dt:' + str(int(has_tot))]
        for i, e in enumerate(ents):
            sh = list(doc_sh)
            sh.append('k:' + e['kind'])
            if e['kind'] == 'E':
                sh.append('q:' + qbin(e['q']))
                sh.append('fr:' + str(int(e['frac'])))
                if e['q'] and e['q'] >= 10 and e['q'] % 10 == 0:
                    sh.append('r10')
                if e['tot']:
                    sh.append('tot')
                r = rank[i]
                if nn == 1:
                    sh.append('pos:single')
                elif r == 0:
                    sh.append('pos:first')
                elif r == nn - 1:
                    sh.append('pos:last')
                else:
                    sh.append('pos:%d' % min(2, int(3 * r / nn)))
                if e['multi']:
                    sh.append('mu')
                if med and e['q']:
                    lr = math.log(e['q'] / med)
                    sh.append('rel:' + ('hi' if lr > 0.7 else 'lo' if lr < -0.7 else 'eq'))
                sh.append('hasc:' + str(int(e['com'] is not None)))
            sh.append('ds:' + str(min(len(e['des']), 3)))
            if e['com'] in cset:
                it = ['C:' + c for c in dcom if c != e['com'] and c in cset]
                it += ['W:' + w for w in e['des'] if w in wset]
                occ['c:' + e['com']].append((sh, it))
            for j, w in enumerate(e['des']):
                if w in wset:
                    shw = sh + ['wp:' + ('first' if j == 0 else 'mid' if j < len(e['des']) - 1 else 'last')]
                    it = (['C:' + e['com']] if e['com'] in cset else ['C:none'])
                    it += ['W:' + x for x in e['des'] if x != w and x in wset]
                    occ['w:' + w].append((shw, it))
    items = ['c:' + c for c in coms] + ['w:' + w for w in words]
    items = [x for x in items if occ[x]]
    return items, occ


SHARED_PREFIX = ('dl:', 'dc:', 'dt:', 'k:', 'q:', 'fr:', 'r10', 'tot', 'pos:', 'mu', 'rel:', 'hasc:', 'ds:', 'wp:')


def is_shared(c):
    return c.startswith(SHARED_PREFIX)


def matrix(items, occ, internal_weight=1.0):
    ctx = sorted({c for it in items for sh, inn in occ[it] for c in sh + inn})
    ci = {c: j for j, c in enumerate(ctx)}
    M = np.zeros((len(items), len(ctx)))
    for i, it in enumerate(items):
        for sh, inn in occ[it]:
            for c in sh:
                M[i, ci[c]] += 1
            for c in inn:
                M[i, ci[c]] += internal_weight
    return M, ctx


def ppmi(M, alpha=0.75):
    tot = M.sum()
    pr = M.sum(1, keepdims=True) / tot
    pc = M.sum(0) ** alpha
    pc = pc / pc.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.log((M / tot) / (pr * pc[None, :]))
    P[~np.isfinite(P)] = 0
    return np.maximum(P, 0)


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def embed(items, occ, d=12, internal_weight=1.0):
    M, ctx = matrix(items, occ, internal_weight)
    P = ppmi(M)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    d = min(d, len(S))
    X = U[:, :d] * np.sqrt(S[:d])
    C = Vt[:d].T * np.sqrt(S[:d])
    sh = [j for j, c in enumerate(ctx) if is_shared(c)]
    prof = unit(P[:, sh])
    return {'items': items, 'X': unit(X), 'C': C, 'ctx': ctx, 'prof': prof, 'prof_ctx': [ctx[j] for j in sh],
            'freq': np.array([len(occ[it]) for it in items], float), 'Cw': M.sum(0)}


def procrustes(A, B, w=None):
    if w is not None:
        A = A * w[:, None]; B = B * w[:, None]
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def csls(XA, XB, k=3):
    S = XA @ XB.T
    k1 = min(k, S.shape[1]); k2 = min(k, S.shape[0])
    ra = np.sort(S, 1)[:, -k1:].mean(1)
    rb = np.sort(S, 0)[-k2:, :].mean(0)
    return 2 * S - ra[:, None] - rb[None, :]


def typed_blocks(ea, eb):
    ta = np.array([it[0] for it in ea['items']]); tb = np.array([it[0] for it in eb['items']])
    return ta, tb


def mutual_pairs(S, ta, tb):
    pairs = []
    for t in ('c', 'w'):
        ia = np.where(ta == t)[0]; ib = np.where(tb == t)[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        sub = S[np.ix_(ia, ib)]
        fa = sub.argmax(1); fb = sub.argmax(0)
        for i, j in enumerate(fa):
            if fb[j] == i:
                pairs.append((ia[i], ib[j]))
    return pairs


def refine(ea, eb, W, rounds=5, seedA=None, seedB=None, seedw=None):
    ta, tb = typed_blocks(ea, eb)
    obj = 0
    for _ in range(rounds):
        S = csls(ea['X'] @ W, eb['X'])
        pairs = mutual_pairs(S, ta, tb)
        if not pairs:
            break
        A = ea['X'][[p[0] for p in pairs]]; B = eb['X'][[p[1] for p in pairs]]
        obj = float(np.mean([S[p] for p in pairs]))
        if seedA is not None:
            A = np.vstack([A, seedA]); B = np.vstack([B, seedB])
            w = np.concatenate([np.ones(len(pairs)), seedw])
        else:
            w = None
        W = procrustes(A, B, w)
    return W, obj


def shared_seed(ea, eb):
    ca = {c: j for j, c in enumerate(ea['ctx'])}; cb = {c: j for j, c in enumerate(eb['ctx'])}
    common = [c for c in ca if c in cb and is_shared(c)]
    A = unit(ea['C'][[ca[c] for c in common]]); B = unit(eb['C'][[cb[c] for c in common]])
    w = np.sqrt(np.minimum(ea['Cw'][[ca[c] for c in common]], eb['Cw'][[cb[c] for c in common]]))
    w = w / w.mean()
    return A, B, w, common


def rand_orth(d, rng):
    Q, R = np.linalg.qr(rng.normal(size=(d, d)))
    return Q * np.sign(np.diag(R))


def pad(ea, eb):
    d = min(ea['X'].shape[1], eb['X'].shape[1])
    for e in (ea, eb):
        e['X'] = unit(e['X'][:, :d]); e['C'] = e['C'][:, :d]
    return d


def align_seed(ea, eb):
    A, B, w, _ = shared_seed(ea, eb)
    W = procrustes(A, B, w)
    W, obj = refine(ea, eb, W, seedA=A, seedB=B, seedw=w)
    return csls(ea['X'] @ W, eb['X']), obj


def align_unsup(ea, eb, rng, R=30):
    d = ea['X'].shape[1]
    best = None
    for _ in range(R):
        W, obj = refine(ea, eb, rand_orth(d, rng), rounds=6)
        if best is None or obj > best[1]:
            best = (W, obj)
    return csls(ea['X'] @ best[0], eb['X']), best[1]


def gw(CA, CB, p, q, rng, eps=5e-3, outer=40, inner=60, T0=None):
    """Entropic Gromov-Wasserstein (square loss), Peyre et al. 2016 projected iterations."""
    T = np.outer(p, q) if T0 is None else T0
    constC = ((CA ** 2) @ p)[:, None] + ((CB ** 2) @ q)[None, :]
    for _ in range(outer):
        G = constC - 2 * CA @ T @ CB.T
        K = np.exp(-(G - G.min()) / eps)
        u = np.ones_like(p)
        for _ in range(inner):
            v = q / (K.T @ u + 1e-300)
            u = p / (K @ v + 1e-300)
        T = u[:, None] * K * v[None, :]
    loss = float(np.sum((constC - 2 * CA @ T @ CB.T) * T))
    return T, loss


def align_gw(ea, eb, rng, R=10, which='c'):
    ta, tb = typed_blocks(ea, eb)
    ia = np.where(ta == which)[0]; ib = np.where(tb == which)[0]
    XA = ea['X'][ia]; XB = eb['X'][ib]
    CA = 1 - XA @ XA.T; CB = 1 - XB @ XB.T
    p = np.sqrt(ea['freq'][ia]); p /= p.sum()
    q = np.sqrt(eb['freq'][ib]); q /= q.sum()
    best = None
    for r in range(R):
        T0 = None
        if r > 0:
            T0 = np.outer(p, q) * rng.uniform(0.2, 1.8, size=(len(p), len(q)))
            T0 = T0 / T0.sum()
        T, loss = gw(CA, CB, p, q, rng, T0=T0)
        if best is None or loss < best[1]:
            best = (T, loss)
    S = np.full((len(ea['items']), len(eb['items'])), -1.0)
    T = best[0]
    S[np.ix_(ia, ib)] = T / T.sum(1, keepdims=True)
    return S, -best[1]


def align_prof(ea, eb):
    ca = {c: j for j, c in enumerate(ea['prof_ctx'])}; cb = {c: j for j, c in enumerate(eb['prof_ctx'])}
    common = [c for c in ca if c in cb]
    A = unit(ea['prof'][:, [ca[c] for c in common]]); B = unit(eb['prof'][:, [cb[c] for c in common]])
    return A @ B.T, 0.0


def align_freq(ea, eb):
    ta, tb = typed_blocks(ea, eb)
    S = np.zeros((len(ea['items']), len(eb['items'])))
    for t in ('c', 'w'):
        ia = np.where(ta == t)[0]; ib = np.where(tb == t)[0]
        fa = np.log(ea['freq'][ia] / ea['freq'][ia].sum()); fb = np.log(eb['freq'][ib] / eb['freq'][ib].sum())
        S[np.ix_(ia, ib)] = -np.abs(fa[:, None] - fb[None, :])
    return S, 0.0


METHODS = ('seed', 'unsup', 'gw', 'prof', 'freq')


def run_pair(docsA, docsB, rng, methods=METHODS, d=12, minc=8, R_unsup=30, R_gw=10, internal_weight=1.0):
    itA, ocA = occurrences(docsA, minc=minc)
    itB, ocB = occurrences(docsB, minc=minc)
    ea = embed(itA, ocA, d, internal_weight); eb = embed(itB, ocB, d, internal_weight)
    pad(ea, eb)
    out = {}
    for m in methods:
        if m == 'seed':
            S, obj = align_seed(ea, eb)
        elif m == 'unsup':
            S, obj = align_unsup(ea, eb, rng, R_unsup)
        elif m == 'gw':
            S, obj = align_gw(ea, eb, rng, R_gw)
        elif m == 'prof':
            S, obj = align_prof(ea, eb)
        else:
            S, obj = align_freq(ea, eb)
        out[m] = (S, obj)
    return ea['items'], eb['items'], out


# ------------------------------------------------------------------ data perturbations
def boot(docs, rng):
    return [docs[i] for i in rng.integers(0, len(docs), len(docs))]


def shuffle_commodities(docs, rng):
    """Null: commodity labels permuted across all entries carrying one (numbers, positions, words kept)."""
    labs = [e['com'] for d in docs for e in d['entries'] if e['com']]
    rng.shuffle(labs)
    it = iter(labs)
    out = []
    for d in docs:
        es = []
        for e in d['entries']:
            e2 = dict(e)
            if e['com']:
                e2['com'] = next(it)
            es.append(e2)
        out.append({'id': d['id'], 'site': d['site'], 'entries': es})
    return out


def shuffle_words(docs, rng):
    toks = [w for d in docs for e in d['entries'] for w in e['des']]
    rng.shuffle(toks)
    it = iter(toks)
    out = []
    for d in docs:
        es = []
        for e in d['entries']:
            e2 = dict(e); e2['des'] = [next(it) for _ in e['des']]
            es.append(e2)
        out.append({'id': d['id'], 'site': d['site'], 'entries': es})
    return out


def shuffle_all(docs, rng):
    return shuffle_words(shuffle_commodities(docs, rng), rng)


# ------------------------------------------------------------------ evaluation
GOLD_FILE = os.path.join(DX, 'gold_LB_UR3.json')


def gold_eval(itA, itB, S, gold, rng=None, nperm=0, pre='c:'):
    """MRR / P@1 over gold sources present; optional permutation null over target labels."""
    ia = {it: i for i, it in enumerate(itA)}
    cb = [j for j, it in enumerate(itB) if it.startswith(pre)]
    labs = [itB[j][2:] for j in cb]
    srcs = [(s, set(t)) for s, t in gold.items() if pre + s in ia and any(x in labs for x in t)]
    if not srcs:
        return None

    def score(labs_):
        rr = []; p1 = 0
        for s, tg in srcs:
            row = S[ia[pre + s], cb]
            order = np.argsort(-row)
            rank = next(r for r, j in enumerate(order) if labs_[j] in tg) + 1
            rr.append(1 / rank); p1 += rank == 1
        return float(np.mean(rr)), p1
    mrr, p1 = score(labs)
    res = {'n': len(srcs), 'mrr': mrr, 'p1': p1, 'ntargets': len(cb)}
    if nperm:
        null = []
        for _ in range(nperm):
            l2 = list(labs); rng.shuffle(l2)
            null.append(score(l2)[0])
        null = np.array(null)
        res['null_mrr'] = float(null.mean()); res['p'] = float((1 + (null >= mrr).sum()) / (1 + nperm))
    return res


# ------------------------------------------------------------------ cycle-2 methods
def cooc(e, items, occ):
    """Item x item internal co-occurrence (PPMI, row-stochastic)."""
    ix = {it[2:]: i for i, it in enumerate(items)}
    K = np.zeros((len(items), len(items)))
    for i, it in enumerate(items):
        for sh, inn in occ[it]:
            for c in inn:
                j = ix.get(c[2:])
                if j is not None and j != i:
                    K[i, j] += 1
    K = ppmi(K + 1e-9, alpha=1.0)
    s = K.sum(1, keepdims=True); s[s == 0] = 1
    return K / s


def align_flood(ea, eb, KA, KB, lam=0.5, iters=15):
    """Similarity flooding: S = (1-lam) S0 + lam KA S KB^T, S0 = shared-profile cosine."""
    S0, _ = align_prof(ea, eb)
    S0 = (S0 - S0.mean()) / (S0.std() + 1e-12)
    S = S0.copy()
    for _ in range(iters):
        P = KA @ S @ KB.T
        P = (P - P.mean()) / (P.std() + 1e-12)
        S = (1 - lam) * S0 + lam * P
    return S, 0.0


def align_joint(ea, eb, d=8):
    ca = {c: j for j, c in enumerate(ea['prof_ctx'])}; cb = {c: j for j, c in enumerate(eb['prof_ctx'])}
    common = [c for c in ca if c in cb]
    A = ea['prof'][:, [ca[c] for c in common]]; B = eb['prof'][:, [cb[c] for c in common]]
    Z = np.vstack([A, B]); Z = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    d = min(d, len(S))
    Y = unit(U[:, :d] * S[:d])
    return Y[:len(A)] @ Y[len(A):].T, 0.0


METHODS2 = ('prof', 'joint', 'flood', 'freq')


def run_pair2(docsA, docsB, methods=METHODS2, minc=8, lam=0.5):
    itA, ocA = occurrences(docsA, minc=minc)
    itB, ocB = occurrences(docsB, minc=minc)
    ea = embed(itA, ocA, 12); eb = embed(itB, ocB, 12)
    out = {}
    for m in methods:
        if m == 'prof':
            out[m] = align_prof(ea, eb)[0]
        elif m == 'joint':
            out[m] = align_joint(ea, eb)[0]
        elif m == 'flood':
            out[m] = align_flood(ea, eb, cooc(ea, itA, ocA), cooc(eb, itB, ocB), lam)[0]
        elif m == 'freq':
            out[m] = align_freq(ea, eb)[0]
    return itA, itB, out


def top_partner(itA, itB, S, pre='c:'):
    """For each A item of type pre: best B item of the same type and margin (top1 - top2)."""
    ia = [i for i, it in enumerate(itA) if it.startswith(pre)]
    jb = [j for j, it in enumerate(itB) if it.startswith(pre)]
    out = {}
    if not jb:
        return out
    for i in ia:
        row = S[i, jb]
        o = np.argsort(-row)
        out[itA[i][2:]] = (itB[jb[o[0]]][2:], float(row[o[0]] - (row[o[1]] if len(o) > 1 else 0)))
    return out
