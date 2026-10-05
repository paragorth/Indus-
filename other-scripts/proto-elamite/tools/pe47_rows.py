"""pe47 cycle 3: low-rank (multiplicative) factorisation of the hidden table.

If each tablet is a slice (one row = one unit in one period) of a table whose cells follow
   log X[u,c,p] = row_u + col_c + per_p     (rank-1 multiplicative / CP-1 in log space),
then within a tablet every entry is its commodity norm times ONE tablet scale, and tablets that
are slices of the same row (same unit) share that scale.
(a) ROW SCALE: hide one entry; predict log v from col_c (commodity mean, train tablets) + the
    tablet scale estimated from the tablet's OTHER entries. Gain in held-out squared error over
    col_c alone. Nulls: N1 values shuffled within (last token, system) across tablets; N2 entries
    re-dealt between tablets.
(b) ROW IDENTITY (random search): 3,000 random header-token -> row assignments plus ridge on all
    header tokens; a hypothesis groups tablets into rows; score = held-out (tablet-disjoint)
    prediction of tablet scales from row means. Null: headers shuffled among tablets (same search).
    Read-off: header tokens whose own row predicts scale beyond the shuffled headers.
"""
import collections, json, math, random, sys
import numpy as np
from pe47_common import build_pe, build_ur3, sample_like, ent_lengths, plant
from pe47_search import null_shuffle_values, null_permute_entries


def col_means(corpus, ids):
    s = collections.defaultdict(list)
    for i in ids:
        for e in corpus[i]['ents']:
            s[(e[2], e[0][-1])].append(math.log(e[1]))
    sysm = collections.defaultdict(list)
    for (sy, l), L in s.items():
        sysm[sy] += L
    return {k: (np.mean(L), len(L)) for k, L in s.items()}, {k: np.mean(L) for k, L in sysm.items()}


def colpred(cm, sm, e):
    k = (e[2], e[0][-1])
    if k in cm:
        m, n = cm[k]
        return (n * m + 2 * sm.get(e[2], 0)) / (n + 2)
    return sm.get(e[2], 0.0)


def row_scale_test(corpus, seed):
    r = random.Random(seed)
    ids = list(range(len(corpus)))
    r.shuffle(ids)
    A, B = ids[:len(ids) // 2], ids[len(ids) // 2:]
    cm, sm = col_means(corpus, A)
    se0 = se1 = 0.0; n = 0
    for i in B:
        E = corpus[i]['ents']
        if len(E) < 3:
            continue
        res = [math.log(e[1]) - colpred(cm, sm, e) for e in E]
        for j, e in enumerate(E):
            others = [res[k] for k in range(len(E)) if k != j and E[k][2] == e[2]]
            if len(others) < 2:
                continue
            sc = sum(others) / (len(others) + 1.0)   # shrunk tablet scale
            se0 += res[j] ** 2; se1 += (res[j] - sc) ** 2; n += 1
    return dict(n=n, mse_col=se0 / max(1, n), mse_row=se1 / max(1, n), gain=(se0 - se1) / max(1, se0))


def tablet_scales(corpus, cm, sm, ids):
    S = {}
    for i in ids:
        res = [math.log(e[1]) - colpred(cm, sm, e) for e in corpus[i]['ents']]
        if len(res) >= 2:
            S[i] = float(np.mean(res))
    return S


def row_identity(corpus, seed, n_hyp=3000, shuffle_headers=False):
    r = random.Random(seed)
    ctx = [list(t['ctx']) for t in corpus]
    if shuffle_headers:
        r.shuffle(ctx)
    ids = list(range(len(corpus)))
    r.shuffle(ids)
    A, B = ids[:len(ids) // 2], ids[len(ids) // 2:]
    cm, sm = col_means(corpus, A)
    SA = tablet_scales(corpus, cm, sm, A)
    SB = tablet_scales(corpus, cm, sm, B)
    tf = collections.Counter(w for i in A for w in set(ctx[i]))
    toks = [w for w, n in tf.items() if n >= 3]
    varB = np.var(list(SB.values())) if SB else 1.0
    muA = float(np.mean(list(SA.values())))

    def evaluate(rowtok):
        # rowtok: set of header tokens that index rows; a tablet's row = its row tokens (sorted tuple)
        g = collections.defaultdict(list)
        for i, s in SA.items():
            k = tuple(sorted(w for w in ctx[i] if w in rowtok))
            if k:
                g[k].append(s)
        gm = {k: (sum(L) + 2 * muA) / (len(L) + 2) for k, L in g.items()}
        err = err0 = 0.0
        for i, s in SB.items():
            k = tuple(sorted(w for w in ctx[i] if w in rowtok))
            p = gm.get(k, muA)
            err += (s - p) ** 2; err0 += (s - muA) ** 2
        return 1 - err / max(1e-9, err0)
    # inner CV on A for the random search, then test on B once
    best = []
    A1 = set(A[:len(A) // 2])
    for h in range(n_hyp):
        k = r.choice([1, 2, 3, 5, 8, 13, 21])
        rt = set(r.sample(toks, min(k, len(toks))))
        # inner score: train on A1, test on A\A1
        g = collections.defaultdict(list)
        for i, s in SA.items():
            if i in A1:
                kk = tuple(sorted(w for w in ctx[i] if w in rt))
                if kk:
                    g[kk].append(s)
        gm = {kk: (sum(L) + 2 * muA) / (len(L) + 2) for kk, L in g.items()}
        e1 = e0 = 0.0
        for i, s in SA.items():
            if i not in A1:
                kk = tuple(sorted(w for w in ctx[i] if w in rt))
                e1 += (s - gm.get(kk, muA)) ** 2; e0 += (s - muA) ** 2
        best.append((1 - e1 / max(1e-9, e0), rt))
    best.sort(key=lambda x: -x[0])
    held = [evaluate(rt) for _, rt in best[:10]]
    # single-token read-off on held-out half
    single = {}
    for w in toks:
        single[w] = evaluate({w})
    top = sorted(single.items(), key=lambda x: -x[1])[:15]
    return dict(inner_best=best[0][0], held_top10=held, held_best=held[0], best_rowtok=sorted(best[0][1]),
                single_top=top, n_tok=len(toks), varB=float(varB))


def pair1(corpus):
    """Tablet pairs joined by a value seen exactly twice in the corpus (value >= 5, same system),
    validated by token Jaccard and (Ur III) same king-year."""
    from pe47_links import toks, jac
    occ = collections.defaultdict(list)
    for ti, t in enumerate(corpus):
        for e in t['ents']:
            if e[1] >= 5:
                occ[(e[2], round(e[1], 4))].append(ti)
    P = [tuple(L) for L in occ.values() if len(L) == 2 and L[0] != L[1]]
    TK = [toks(t) for t in corpus]
    out = dict(n_pairs=len(P), jac=float(np.mean([jac(TK[a], TK[b]) for a, b in P])) if P else None)
    # same-last-token pairs only
    if 'year' in corpus[0]['meta']:
        def same(a, b):
            ma, mb = corpus[a]['meta'], corpus[b]['meta']
            return ma['king'] == mb['king'] and ma['year'] == mb['year'] and ma['year'] not in ('', '00')
        out['sameyear'] = float(np.mean([same(a, b) for a, b in P])) if P else None
    return out


def get(name, seed):
    pe = build_pe()
    if name == 'PE':
        return pe
    if name == 'PLANT':
        return plant(len(pe), ent_lengths(pe), 1000 + seed, U=30, C=20, P=6)
    if name == 'UR3D':
        return sample_like(build_ur3('drehem'), len(pe), 77 + seed)
    if name == 'UR3U':
        return sample_like(build_ur3('umma'), len(pe), 77 + seed)


def job(a):
    name, null, seed, part = a
    c = get(name, 0)
    if null == 'N1':
        c = null_shuffle_values(c, 700 + seed)
    elif null == 'N2':
        c = null_permute_entries(c, 700 + seed)
    if part == 'pair1':
        v = pair1(c)
    elif part == 'scale':
        v = row_scale_test(c, seed)
    else:
        v = row_identity(c, seed, shuffle_headers=(null == 'HSHUF'))
    v.update(name=name, null=null, seed=seed, part=part)
    return v


if __name__ == '__main__':
    from multiprocessing import Pool
    out = sys.argv[1]
    jobs = []
    for name in ('PLANT', 'UR3D', 'UR3U', 'PE'):
        for s in range(5):
            for null in ('real', 'N1', 'N2'):
                jobs.append((name, null, s, 'scale'))
            for null in ('real', 'HSHUF'):
                jobs.append((name, null, s, 'rows'))
        jobs.append((name, 'real', 0, 'pair1'))
        for s in range(20):
            jobs.append((name, 'N1', s, 'pair1'))
    res = []
    with Pool(2) as p:
        for v in p.imap_unordered(job, jobs):
            res.append(v)
            json.dump(res, open(out, 'w'))
    print('done', len(res))
