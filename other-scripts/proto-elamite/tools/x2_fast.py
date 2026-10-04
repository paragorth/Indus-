#!/usr/bin/env python3
"""X-2 fast path: precompute per-document (item, context) counts once, then every bootstrap
resample is a weighted sum (sparse matrix product). Reproduces x2_common.run_pair2 item
selection (minc, maxc, minw, minw_docs, maxw) and the prof / joint / flood / freq similarities.
Candidate vocabulary is limited to the 120 commonest commodities and 400 commonest words of
the corpus (the 40 / 150 kept per resample are always drawn from these)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from collections import Counter, defaultdict
from scipy import sparse
import x2_common as X


def precompute(docs, maxc_pool=120, maxw_pool=400):
    cc = Counter(); wc = Counter()
    for d in docs:
        for e in d['entries']:
            if e['com']:
                cc[e['com']] += 1
            for w in e['des']:
                wc[w] += 1
    coms = [c for c, _ in cc.most_common(maxc_pool)]
    words = [w for w, _ in wc.most_common(maxw_pool + len(cc)) if w not in cc][:maxw_pool]
    items = ['c:' + c for c in coms] + ['w:' + w for w in words]
    ii = {it: i for i, it in enumerate(items)}
    ctx_idx = {}
    rows_D, cols_D, vals_D = [], [], []
    rows_n, cols_n = [], []
    for di, d in enumerate(docs):
        it, occ = occurrences_doc(d, set(coms), set(words))
        pair = Counter(); cnt = Counter()
        for item, lst in occ.items():
            if item not in ii:
                continue
            for sh, inn in lst:
                cnt[ii[item]] += 1
                for c in sh + inn:
                    j = ctx_idx.setdefault(c, len(ctx_idx))
                    pair[(ii[item], j)] += 1
        for (i, j), v in pair.items():
            rows_D.append(di); cols_D.append((i, j)); vals_D.append(v)
        for i, v in cnt.items():
            rows_n.append((di, i, v))
    nC = len(ctx_idx); nI = len(items)
    D = sparse.csr_matrix((vals_D, (rows_D, [i * nC + j for i, j in cols_D])), shape=(len(docs), nI * nC))
    N = sparse.csr_matrix(([v for _, _, v in rows_n], ([r for r, _, _ in rows_n], [i for _, i, _ in rows_n])),
                          shape=(len(docs), nI))
    ctx = [None] * nC
    for c, j in ctx_idx.items():
        ctx[j] = c
    return {'items': items, 'ctx': ctx, 'D': D, 'N': N, 'Nb': (N > 0).astype(float), 'nC': nC, 'nI': nI}


def occurrences_doc(d, cset, wset):
    """Same context rules as X.occurrences, for one document and a fixed vocabulary."""
    import math
    occ = defaultdict(list)
    ents = d['entries']
    num = [i for i, e in enumerate(ents) if e['kind'] == 'E']
    rank = {i: r for r, i in enumerate(num)}
    nn = len(num)
    qs = [ents[i]['q'] for i in num if ents[i]['q']]
    med = float(np.median(qs)) if qs else None
    dcom = sorted({e['com'] for e in ents if e['com']})
    ncom = len(dcom)
    has_tot = any(e['tot'] for e in ents)
    doc_sh = ['dl:' + X.lenbin(nn), 'dc:' + str(min(ncom, 3)), 'dt:' + str(int(has_tot))]
    for i, e in enumerate(ents):
        sh = list(doc_sh)
        sh.append('k:' + e['kind'])
        if e['kind'] == 'E':
            sh.append('q:' + X.qbin(e['q']))
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
    return None, occ


def select(pre, w, minc=8, maxc=40, minw=8, minw_docs=3, maxw=150):
    cnt = w @ pre['N']; dcnt = (w > 0).astype(float) @ pre['Nb']
    items = pre['items']
    ci = [i for i, it in enumerate(items) if it[0] == 'c' and cnt[i] >= minc]
    ci = sorted(ci, key=lambda i: -cnt[i])[:maxc]
    present_com = {items[i][2:] for i, it in enumerate(items) if it[0] == 'c' and cnt[i] > 0}
    wi = [i for i, it in enumerate(items) if it[0] == 'w' and cnt[i] >= minw and dcnt[i] >= minw_docs
          and it[2:] not in present_com]
    wi = sorted(wi, key=lambda i: -cnt[i])[:maxw]
    sel = ci + wi
    M = np.asarray((w @ pre['D'])).reshape(pre['nI'], pre['nC'])[sel]
    names = {items[i][2:] for i in sel}
    keepc = [j for j, c in enumerate(pre['ctx']) if X.is_shared(c) or c == 'C:none' or c[2:] in names]
    M = M[:, keepc]
    ctx = [pre['ctx'][j] for j in keepc]
    nz = M.sum(0) > 0
    M = M[:, nz]; ctx = [c for c, k in zip(ctx, nz) if k]
    return [items[i] for i in sel], M, ctx, cnt[sel]


def space(items, M, ctx, freq):
    P = X.ppmi(M)
    sh = [j for j, c in enumerate(ctx) if X.is_shared(c)]
    e = {'items': items, 'prof': X.unit(P[:, sh]), 'prof_ctx': [ctx[j] for j in sh], 'freq': np.asarray(freq, float)}
    ix = {it[2:]: i for i, it in enumerate(items)}
    K = np.zeros((len(items), len(items)))
    for j, c in enumerate(ctx):
        if c.startswith(('C:', 'W:')):
            t = ix.get(c[2:])
            if t is not None:
                K[:, t] += M[:, j]
    np.fill_diagonal(K, 0)
    K = X.ppmi(K + 1e-9, alpha=1.0)
    s = K.sum(1, keepdims=True); s[s == 0] = 1
    e['K'] = K / s
    return e


def run_w(preA, wA, preB, wB, methods=X.METHODS2, lam=0.5):
    ea = space(*select(preA, wA)); eb = space(*select(preB, wB))
    out = {}
    for m in methods:
        if m == 'prof':
            out[m] = X.align_prof(ea, eb)[0]
        elif m == 'joint':
            out[m] = X.align_joint(ea, eb)[0]
        elif m == 'flood':
            out[m] = X.align_flood(ea, eb, ea['K'], eb['K'], lam)[0]
        elif m == 'freq':
            out[m] = X.align_freq(ea, eb)[0]
    return ea['items'], eb['items'], out


def boot_w(n, rng):
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
