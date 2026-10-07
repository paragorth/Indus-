"""pe75 cycle 2: does the clay chosen BEFORE writing (h, w, t, aspect, thickness-for-its-area) know
something about the account that the finished text amount and the header do not?

Per tablet: B = finished text amount (log lines, log glyphs, obverse lines, reverse lines) - the
'line-count baseline' that is trivially known after writing; S = start-of-text features (header
present, header sign one-hot top 8, first numeral system); P = physical (log h, log w, log t,
aspect, thickness residual on area, area residual on glyphs = slack).
Targets known only after writing: total present, reverse used, sealed, >1 number system,
largest number >= 60, and presence of each of the 40 commonest signs (the reverse test:
which signs does blank room / extra clay predict?).
Massive random guessing: 3,000 hypotheses (target x random subset of P), fit ridge-logistic on a
fit half of the training tablets, select top 1% by validation gain over B+S, re-test on held-out
tablets.  Nulls: P rows shuffled among tablets within line-count bins (20 runs).  Planted target:
reverse-use-like Bernoulli with +0.8 logit per sd of thickness residual.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json, sys, collections, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_common as C
from pe18_common import seal_map

NH = 3000


def build():
    R = [r for r in C.pe_table() if r['complete_cat'] and r['h'] and r['w'] and r['t']]
    S = seal_map()
    n = len(R)
    lg = lambda a: np.log(np.asarray(a, float) + 1)
    B = np.c_[lg([r['n_lines'] for r in R]), lg([r['glyphs'] for r in R]), [r['obv'] for r in R], [r['rev'] for r in R]]
    hc = collections.Counter(r['header'][0] for r in R if r['header'])
    top = [s for s, _ in hc.most_common(8)]
    Sx = np.c_[[bool(r['header']) for r in R]] * 1.0
    Sx = np.c_[Sx, [[1.0 if r['header'] and r['header'][0] == s else 0 for s in top] for r in R],
               [[1.0 if r['first_sys'] == s else 0 for s in ('SDB', 'C', 'NONE')] for r in R]]
    h, w, t = (np.log([r[k] for r in R]) for k in 'hwt')
    area = h + w
    tres = t - np.polyval(np.polyfit(area, t, 1), area)
    slack = area - np.polyval(np.polyfit(B[:, 1], area, 1), B[:, 1])
    P = np.c_[h, w, t, h - w, tres, slack]
    Y = {'total': [r['has_total'] for r in R], 'reverse': [r['rev'] > 0 for r in R],
         'sealed': [S.get(r['id'], [False])[0] for r in R], 'multisys': [len(r['systems']) > 1 for r in R],
         'big60': [r['max_num'] >= 60 for r in R]}
    sc = collections.Counter(s for r in R for s in r['signs'])
    for s, c in sc.most_common(40):
        Y['sign:' + s] = [s in r['signs'] for r in R]
    Y = {k: np.array(v, bool) for k, v in Y.items() if 25 <= sum(v) <= n - 25}
    rng = np.random.default_rng(75020)
    Xs = std(np.c_[B, Sx]); bb = irls(Xs, Y['reverse'] * 1.0)
    lo = np.c_[np.ones(n), Xs] @ bb
    zt = (P[:, 4] - P[:, 4].mean()) / P[:, 4].std()
    Y['PLANT'] = rng.random(n) < 1 / (1 + np.exp(-(lo + 0.8 * zt)))
    lines = np.array([r['n_lines'] for r in R])
    return R, B, Sx, P, Y, np.digitize(lines, [3, 5, 7, 10, 14])


def std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def irls(X, y, lam=1.0, it=25):
    X = np.c_[np.ones(len(X)), X]
    b = np.zeros(X.shape[1]); R = lam * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - R @ b
        Hm = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-6 * np.eye(len(b))
        d = np.linalg.solve(Hm, g); b += d
        if np.abs(d).max() < 1e-6:
            break
    return b


def llb(b, X, y):
    p = 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ b, -30, 30)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.mean(np.where(y, np.log2(p), np.log2(1 - p)))


_B0 = {}


def gain(Xb, Xp, y, fit, ev):
    """held-out bits/tablet gained by adding physical columns Xp to the text+header baseline (ridge logistic, IRLS)"""
    if y[fit].all() or (~y[fit]).all():
        return 0.0
    key = (y.tobytes(), fit.tobytes(), ev.tobytes())
    if key not in _B0:
        _B0[key] = llb(irls(Xb[fit], y[fit] * 1.0), Xb[ev], y[ev])
    X1 = np.c_[Xb, Xp]
    return llb(irls(X1[fit], y[fit] * 1.0), X1[ev], y[ev]) - _B0[key]


def pipeline(args):
    tag, seed, B, Sx, P, Y, H = args
    rng = np.random.default_rng(seed)
    n = len(B)
    Xb = std(np.c_[B, Sx]); Pz = std(P)
    sp = rng.permutation(n)
    fit, val, te = sp[: n // 3], sp[n // 3: 2 * n // 3], sp[2 * n // 3:]
    trall = np.r_[fit, val]
    sc = np.array([gain(Xb, Pz[:, list(cols)], Y[tg], fit, val) for tg, cols in H])
    top = np.argsort(-sc)[: len(H) // 100]
    te_g = [gain(Xb, Pz[:, list(H[i][1])], Y[H[i][0]], trall, te) for i in top]
    return tag, float(np.mean(te_g)), [(H[i][0], list(H[i][1]), round(sc[i] * 1000, 1), round(te_g[j] * 1000, 1)) for j, i in enumerate(top)]


def main():
    R, B, Sx, P, Y, lb = build()
    rng = np.random.default_rng(75021)
    tg = [k for k in Y if k != 'PLANT']
    H = []
    for _ in range(NH):
        k = rng.integers(1, 4)
        H.append((tg[rng.integers(len(tg))], tuple(sorted(rng.choice(P.shape[1], k, replace=False)))))
    H.append(('PLANT', (4,)))
    jobs = [('real', 75100 + i, B, Sx, P, Y, H) for i in range(3)]
    for i in range(20):
        r2 = np.random.default_rng(9000 + i)
        perm = np.arange(len(B))
        for b in np.unique(lb):
            idx = np.where(lb == b)[0]
            perm[idx] = r2.permutation(idx)
        jobs.append(('null', 75100 + i % 3, B, Sx, P[perm], Y, H))
    with Pool(2) as Pp:
        res = Pp.map(pipeline, jobs)
    real = [r for r in res if r[0] == 'real']
    null = [r[1] for r in res if r[0] == 'null']
    out = {'n': len(B), 'targets': {k: int(v.sum()) for k, v in Y.items()},
           'real_heldout_mbit': [r[1] * 1000 for r in real], 'null_heldout_mbit': [x * 1000 for x in null],
           'p': float(np.mean([x >= np.mean([r[1] for r in real]) for x in null])),
           'survivors': real[0][2]}
    # per-target held-out gain from ALL physical features (10 splits) vs null; includes PLANT
    def tgain(args):
        pass
    per = {}
    Xb = std(np.c_[B, Sx]); Pz = std(P)
    for k, y in Y.items():
        g = []; gn = []
        for s in range(10):
            sp = np.random.default_rng(500 + s).permutation(len(B)); tr, te = sp[: len(B) // 2], sp[len(B) // 2:]
            g.append(gain(Xb, Pz, y, tr, te))
            r2 = np.random.default_rng(700 + s); perm = np.arange(len(B))
            for b in np.unique(lb):
                idx = np.where(lb == b)[0]; perm[idx] = r2.permutation(idx)
            gn.append(gain(Xb, Pz[perm], y, tr, te))
        per[k] = (round(1000 * np.mean(g), 1), round(1000 * np.mean(gn), 1), round(1000 * np.std(gn) / np.sqrt(10), 1),
                  round(float(np.mean(np.array(g) > np.array(gn))), 2))
    out['per_target'] = per
    json.dump(out, open(os.path.join(C.CK, 'cycle2.json'), 'w'), indent=1, default=int)
    print({k: v for k, v in out.items() if k not in ('survivors', 'per_target')})
    for s in out['survivors'][:12]:
        print(s)
    for k, v in sorted(per.items(), key=lambda kv: -kv[1][0] + kv[1][1]):
        print(k, v)


if __name__ == '__main__':
    main()
