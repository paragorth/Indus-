"""pe75 cycle 1: were PE tablets cut to a length unit?  Massive random guessing of quanta.

Hypothesis H = (quantum q in [6,30] mm, dimension set from {h,w,t}, offset 0 or random, tablet
subset chosen by a START-of-text feature).  Score = Kendall cosine quantogram z = sqrt(2/N) sum cos(2pi(x-e)/q)
on a random train half of tablets; top 1% re-scored on the held-out half.
Nulls: (a) KDE resample (dims redrawn from a smooth density, bandwidth 4 mm: no quantum, same shape),
(b) dims shuffled across tablets (kills any link between text-subset and quantum).
Planted: dims rebuilt as k*17 mm*(1+N(0,.03)) + N(0,s) with s in {1,2,3} mm on a fraction f of tablets.
Comparison archives from the same CDLI catalogue: proto-cuneiform (Uruk IV-III), Ur III (sample).
"""
import json, os, sys, hashlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_common as C

NH = 5000
QMIN, QMAX = 6.0, 30.0


def subsets(R):
    S = {'all': np.ones(len(R), bool)}
    import collections
    hc = collections.Counter(r['header'][0] for r in R if r['header'])
    for s, n in hc.most_common(8):
        if n >= 25:
            S['hdr:' + s] = np.array([bool(r['header']) and r['header'][0] == s for r in R])
    S['hdr:none'] = np.array([not r['header'] for r in R])
    for fs in ('SDB', 'C'):
        S['sys:' + fs] = np.array([r['first_sys'] == fs for r in R])
    return {k: v for k, v in S.items() if v.sum() >= 25}


def make_hyps(rng, subnames):
    H = []
    dsets = [('h',), ('w',), ('t',), ('h', 'w'), ('h', 'w', 't')]
    for _ in range(NH):
        q = rng.uniform(QMIN, QMAX)
        ds = dsets[rng.integers(len(dsets))]
        e = 0.0 if rng.random() < 0.5 else rng.uniform(0, q)
        sub = subnames[rng.integers(len(subnames))] if rng.random() < 0.5 else 'all'
        H.append((q, ds, e, sub))
    return H


def zscore(D, idx, q, ds, e):
    x = np.concatenate([D[d][idx] for d in ds])
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return 0.0
    return float(np.sqrt(2.0 / len(x)) * np.cos(2 * np.pi * (x - e) / q).sum())


def run_pipeline(D, S, H, seed):
    rng = np.random.default_rng(seed)
    n = len(D['h'])
    tr = rng.random(n) < 0.5
    sc = np.array([zscore(D, np.where(tr & S[h[3]])[0], *h[:3]) for h in H])
    top = np.argsort(-sc)[:max(1, len(H) // 100)]
    te = [zscore(D, np.where(~tr & S[H[i][3]])[0], *H[i][:3]) for i in top]
    return float(np.mean(te)), [(H[i][0], H[i][1], H[i][3], sc[i], te[j]) for j, i in enumerate(top[:10])]


def kde(D, rng, bw=4.0):
    return {k: (v + rng.normal(0, bw, len(v)))[rng.permutation(len(v))] for k, v in D.items()}


def plant(D, rng, s, f, q0=17.0):
    out = {}
    m = rng.random(len(D['h'])) < f
    for k, v in D.items():
        kk = np.maximum(1, np.round(v / q0))
        p = np.round(kk * q0 * (1 + rng.normal(0, .03, len(v))) + rng.normal(0, s, len(v)))
        out[k] = np.where(m & ~np.isnan(v), p, v)
    return out


def quantogram(x, qs):
    x = x[~np.isnan(x)]
    return np.array([np.sqrt(2.0 / len(x)) * np.cos(2 * np.pi * x / q).sum() for q in qs])


def job(args):
    kind, seed, D, S, H = args
    rng = np.random.default_rng(seed)
    if kind == 'kde':
        D = kde(D, rng)
    elif kind == 'shuf':
        p = rng.permutation(len(D['h']))
        D = {k: v[p] for k, v in D.items()}
    elif kind.startswith('plant'):
        _, s, f = kind.split(':')
        D = plant(D, rng, float(s), float(f))
    return kind, run_pipeline(D, S, H, seed + 7)[0]


def main():
    R = [r for r in C.pe_table() if r['complete_cat'] and r['h'] and r['w'] and r['t']]
    D = {k: np.array([r[k] for r in R], float) for k in 'hwt'}
    S = subsets(R)
    rng = np.random.default_rng(75001)
    H = make_hyps(rng, [k for k in S if k != 'all'])
    real, top = run_pipeline(D, S, H, 75002)
    reals = [run_pipeline(D, S, H, 75002 + i)[0] for i in range(5)]
    jobs = [('kde', 1000 + i, D, S, H) for i in range(30)] + [('shuf', 2000 + i, D, S, H) for i in range(30)]
    for s in (1, 2, 3):
        for f in (0.3, 0.6):
            jobs += [('plant:%d:%.1f' % (s, f), 3000 + 10 * s + i, D, S, H) for i in range(3)]
    with Pool(2) as P:
        res = P.map(job, jobs)
    out = {'n': len(R), 'subsets': {k: int(v.sum()) for k, v in S.items()}, 'real_heldout': reals,
           'top': [(round(a, 2), list(b), c, round(d, 2), round(e, 2)) for a, b, c, d, e in top]}
    import collections
    agg = collections.defaultdict(list)
    for k, v in res:
        agg[k].append(v)
    out['nulls'] = {k: v for k, v in agg.items()}
    # full quantograms, all archives
    qs = np.arange(QMIN, QMAX + 1e-9, 0.05)
    qg = {}
    cat = C.catalogue()
    arch = {'PE': D}
    for name, pre in (('PC', 'Uruk'), ('UR3', 'Ur III')):
        rows = [v for v in cat.values() if v['period'].startswith(pre) and v['h'] and v['w'] and v['t']
                and not C.BAD_PRES.search(v['pres'] or '')]
        rr = np.random.default_rng(5).permutation(len(rows))[:3000]
        rows = [rows[i] for i in rr]
        arch[name] = {k: np.array([r[k] for r in rows], float) for k in 'hwt'}
    for name, A in arch.items():
        for d in 'hwt':
            z = quantogram(A[d], qs)
            nm = []
            for i in range(30):
                zz = quantogram(kde({d: A[d]}, np.random.default_rng(i))[d], qs)
                nm.append(zz.max())
            qg[name + ':' + d] = {'n': int((~np.isnan(A[d])).sum()), 'best_q': float(qs[z.argmax()]),
                                  'best_z': float(z.max()), 'null_max95': float(np.percentile(nm, 95)),
                                  'z_at_16_17': float(z[(qs >= 15.5) & (qs <= 17.5)].max())}
    out['quantogram'] = qg
    json.dump(out, open(os.path.join(C.CK, 'cycle1.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
