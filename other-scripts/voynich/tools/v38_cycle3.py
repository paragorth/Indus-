"""v38 cycle 3: massive random guessing over feature choices, with held-out pages.
Each hypothesis = random visual sub-space (random dims from the concatenated, standardised
families, random weights, cosine or Euclidean) x random text view (word types in a random
document-frequency band, or glyph n-grams of random order, or word prefixes/suffixes of random
length, binary or tf-idf). Score = partial Mantel r on pairs inside a discovery half of leaves;
the top K are re-scored on the held-out half. The whole search is rerun on page-permuted visuals
(within strata) to give the null for the best discovery score and for the replication.
Run for Voynich and for the Gerard control.
Usage: python3 v38_cycle3.py voynich|gerard
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *

NH = int(os.environ.get('NH', 3000))
NNULL = int(os.environ.get('NNULL', 30))
TOPK = 20


def text_view(words, spec):
    kind = spec['kind']
    if kind == 'word':
        toks = words
    elif kind == 'pre':
        toks = [[w[:spec['k']] for w in ws] for ws in words]
    elif kind == 'suf':
        toks = [[w[-spec['k']:] for w in ws] for ws in words]
    else:
        toks = [[('<' + w + '>')[i:i + spec['k']] for w in ws for i in range(len(w) + 3 - spec['k'])]
                for ws in words]
    pr = text_profiles(toks)
    B, X, dfa = pr['B'], pr['X'], pr['dfa']
    n = len(words)
    lo, hi = spec['lo'], max(spec['lo'] + 1, int(spec['hi'] * n))
    sel = (dfa >= lo) & (dfa <= hi)
    if sel.sum() < 5:
        sel = dfa >= 2
    M = (B if spec['bin'] else np.log1p(X) * np.log(n / dfa))[:, sel]
    return cos_sim(M - (M.mean(0) if spec['centre'] else 0))


def random_spec(rng, Fall, fam_slices):
    fams = [f for f in fam_slices if rng.random() < 0.5] or [rng.choice(list(fam_slices))]
    dims = np.concatenate([np.arange(*fam_slices[f]) for f in fams])
    k = int(rng.integers(3, min(60, len(dims)) + 1))
    d = np.sort(rng.choice(dims, k, replace=False))
    w = rng.gamma(1.0, 1.0, k)
    vs = dict(fams=[str(f) for f in fams], dims=d.tolist(), w=w.round(3).tolist(), metric=str(rng.choice(['cos', 'euc'])))
    kind = str(rng.choice(['word', 'word', 'pre', 'suf', 'gram']))
    ts = dict(kind=kind, k=int(rng.integers(2, 5)) if kind != 'word' else 0,
              lo=int(rng.integers(1, 4)), hi=float(rng.choice([0.05, 0.1, 0.2, 0.5, 1.0])),
              bin=bool(rng.random() < 0.5), centre=bool(rng.random() < 0.5))
    return vs, ts


def vis_matrix(Fall, vs):
    F = Fall[:, vs['dims']] * np.array(vs['w'])
    return cos_sim(F) if vs['metric'] == 'cos' else -np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1))


def sub_r(A, B, part_sub, idx):
    a = zs(part_sub.res(upper(A[np.ix_(idx, idx)])))
    b = zs(part_sub.res(upper(B[np.ix_(idx, idx)])))
    return float(np.mean(a * b))


def search(Fall, fam_slices, words, conf, iA, iB, rng, nh, perm_rows=None, cacheT=None):
    pA = Partial([C[np.ix_(iA, iA)] for C in conf.values()], len(iA))
    pB = Partial([C[np.ix_(iB, iB)] for C in conf.values()], len(iB))
    F = Fall if perm_rows is None else Fall[perm_rows]
    res = []
    for h in range(nh):
        vs, ts = specs[h]
        key = json.dumps(ts, sort_keys=True)
        if key not in cacheT:
            cacheT[key] = text_view(words, ts)
        Vm = vis_matrix(F, vs)
        res.append((sub_r(Vm, cacheT[key], pA, iA), h))
    res.sort(reverse=True)
    top = res[:TOPK]
    rep = []
    for rA, h in top:
        vs, ts = specs[h]
        Vm = vis_matrix(F, vs)
        rep.append(sub_r(Vm, cacheT[json.dumps(ts, sort_keys=True)], pB, iB))
    return dict(bestA=top[0][0], topA=[t[0] for t in top], repB=rep, meanB=float(np.mean(rep)),
                best_h=top[0][1], top_h=[t[1] for t in top])


if __name__ == '__main__':
    src = sys.argv[1]
    rng = np.random.default_rng(383 if src == 'voynich' else 384)
    if src == 'voynich':
        from v38_cycle1 import voynich_setup
        pages, keys, words, vis, conf, strata = voynich_setup()
        leaves = [p['leafnum'] for p in pages]
    else:
        from v38_cycle2 import gerard_setup
        keys, words, vis, conf = gerard_setup()
        strata = None
        leaves = [int(k) // 2 for k in keys]
    n = len(keys)
    blocks, sl, c = [], {}, 0
    for f in VIS_FAMS:
        M = np.array([vis[k][f] for k in keys], float)
        if M.std() == 0:
            continue
        if M.shape[1] > 40:
            U, S, _ = np.linalg.svd(M - M.mean(0), full_matrices=False)
            M = U[:, :30] * S[:30]
        M = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
        blocks.append(M); sl[f] = (c, c + M.shape[1]); c += M.shape[1]
    Fall = np.concatenate(blocks, 1)
    ul = sorted(set(leaves))
    rng.shuffle(ul)
    A_leaves = set(ul[:len(ul) // 2])
    iA = np.array([i for i in range(n) if leaves[i] in A_leaves]); iB = np.array([i for i in range(n) if leaves[i] not in A_leaves])
    specs = [random_spec(rng, Fall, sl) for _ in range(NH)]
    cacheT = {}
    real = search(Fall, sl, words, conf, iA, iB, rng, NH, None, cacheT)
    print(src, 'real bestA %.4f meanB(top20) %.4f' % (real['bestA'], real['meanB']), flush=True)
    nulls = []
    for k in range(NNULL):
        p = perm(n, rng, strata)
        nr = search(Fall, sl, words, conf, iA, iB, rng, NH, p, cacheT)
        nulls.append((nr['bestA'], nr['meanB']))
        print(' null', k, '%.4f %.4f' % nulls[-1], flush=True)
        json.dump(dict(real=real, nulls=nulls, specs_best=[specs[h] for h in real['top_h'][:5]], nA=len(iA), nB=len(iB)),
                  open(os.path.join(CK, 'c3_%s.json' % src), 'w'), default=str)
    bA = np.array([a for a, _ in nulls]); mB = np.array([b for _, b in nulls])
    pA_ = (1 + (bA >= real['bestA']).sum()) / (1 + len(nulls))
    pB_ = (1 + (mB >= real['meanB']).sum()) / (1 + len(nulls))
    zB = (real['meanB'] - mB.mean()) / mB.std()
    fam_count = Counter(f for h in real['top_h'] for f in specs[h][0]['fams'])
    txt_count = Counter(specs[h][1]['kind'] for h in real['top_h'])
    summ = dict(src=src, n=n, nA=len(iA), nB=len(iB), NH=NH, NNULL=NNULL, bestA=real['bestA'], null_bestA=[float(bA.mean()), float(bA.max())],
                p_bestA=pA_, meanB=real['meanB'], null_meanB=[float(mB.mean()), float(mB.std())], zB=float(zB), p_meanB=pB_,
                fam_count=dict(fam_count), txt_count=dict(txt_count), repB=real['repB'])
    json.dump(dict(summary=summ, real=real, nulls=nulls), open(os.path.join(CK, 'c3_%s.json' % src), 'w'), default=str)
    print(json.dumps(summ))
