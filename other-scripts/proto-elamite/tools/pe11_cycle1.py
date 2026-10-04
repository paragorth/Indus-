"""pe11 cycle 1: CALIBRATION.  Can a blind ring search rebuild the year at all?
(a) Ur III Drehem livestock tablets with KNOWN months (real positive control):
    ground-truth month R2 per feature; blind search at PE size (n = 638) and n = 3000;
    recovery = share of tablets placed within +-1 month (best of 24 alignments) vs a
    permuted-month null; loop excess D vs copula / shuffle / random-walk nulls.
(b) Simulated herd office built from PE's own tablets (entry counts, sign mix,
    value distributions) with lambing, culling, harvest-stock and herd-size cycles at
    amplitudes a and seasonal fraction f; same search, same nulls.
"""
import json, os, pickle, sys, time
import numpy as np
from multiprocessing import Pool
from pe11_common import *  # noqa

NREP = 10


def sim_tabs(P, pool, a, f, seed):
    rng = np.random.default_rng(seed)
    ids = sorted(P)
    vals = defaultdict(list)
    sysmaj = {}
    sc = defaultdict(Counter)
    gl = Counter()
    for t in P.values():
        for e in t['entries']:
            vals[e[0]].append(e[2])
            sc[e[0]][e[1]] += 1
            gl[e[0]] += 1
    for s in sc:
        sysmaj[s] = sc[s].most_common(1)[0][0]
    signs = list(gl)
    gp = np.array([gl[s] for s in signs], float)
    gp /= gp.sum()
    idx = {s: i for i, s in enumerate(signs)}
    lamb, cull, grain = 'M263', 'M346', 'M297'
    out, months = {}, []
    for tid in ids:
        m = int(rng.integers(0, S))
        seasonal = rng.random() < f
        ma = a if seasonal else 0.0
        lm = 1 + 4 * ma * np.exp(-((m - 2) % S) / 2)
        cm = 1 + 3 * ma * np.exp(-((m - 8) % S) / 1.5)
        st = 1 + 3 * ma * (1 - ((m - 5) % S) / S)
        hf = 1 + ma * np.exp(-((m - 2) % S) / 4)
        E = P[tid]['entries']
        loc = np.zeros(len(signs))
        for e in E:
            loc[idx[e[0]]] += 1
        p = 0.5 * loc / loc.sum() + 0.5 * gp
        p[idx[lamb]] *= lm
        p[idx[cull]] *= cm
        p /= p.sum()
        ents = []
        for e in E:
            s = signs[rng.choice(len(signs), p=p)]
            v = vals[s][rng.integers(len(vals[s]))]
            sy = sysmaj[s]
            v = v * (st if s == grain else (hf if sy == 'S' else 1.0))
            v = max(1.0, round(v)) if sy == 'S' else v
            ents.append((s, sy, float(v), e[3], e[4] if sy == 'S' else False))
        hdr = 'HSPR' if (m in (2, 3, 4) and seasonal and rng.random() < 0.5) else P[tid]['hdr']
        out[tid] = {'hdr': hdr, 'entries': ents, 'signs': P[tid]['signs'], 'design': ''}
        months.append(m)
    return out, np.array(months)


def task(args):
    kind, key, X, m, seed, extra = args
    fn = os.path.join(CKPT, 'c1_%s.json' % key)
    if os.path.exists(fn):
        return json.load(open(fn))
    rng = np.random.default_rng(seed)
    if kind != 'real':
        X = NULLS[kind](X, rng)
    r = loop_score(X, rng, restarts=6, sweeps=25)
    res = {'key': key, 'kind': kind, 'R2c': r['R2c'], 'R2l': r['R2l'], 'D': r['D']}
    if m is not None and kind == 'real':
        acc, _ = align_acc(r['z'], m)
        nacc = [align_acc(r['z'], rng.permutation(m))[0] for _ in range(100)]
        res.update(acc=acc, acc_null=float(np.mean(nacc)), acc_null95=float(np.quantile(nacc, 0.95)),
                   cc=circ_corr(r['z'], m))
    res.update(extra)
    json.dump(res, open(fn, 'w'))
    return res


def month_r2(X, m):
    Xs = standardize(X)
    mu = np.stack([Xs[m == k].mean(0) for k in range(S)])
    return float(1 - ((Xs - mu[m]) ** 2).sum() / Xs.size), [float(1 - ((Xs[:, j] - mu[m, j]) ** 2).sum() / len(Xs)) for j in range(Xs.shape[1])]


def main():
    P, U = pickle.load(open(os.path.join(CKPT, 'frames.pkl'), 'rb'))
    pool = pe_pool(P)
    rng = np.random.default_rng(11)
    jobs, meta = [], {}
    # (a) Ur III
    Xu, nu, idu = ur3_features(U)
    mu = np.array([U[i]['month'] for i in idu])
    tot, per = month_r2(Xu, mu)
    meta['ur3_truth'] = {'n': len(idu), 'R2_month_all': tot, 'per_feature': dict(zip(nu, per))}
    for n in (638, 2000):
        for rep in range(3):
            sel = rng.choice(len(idu), n, replace=False)
            X, m = Xu[sel], mu[sel]
            meta['ur3_truth']['n%d_r%d' % (n, rep)] = month_r2(X, m)[0]
            jobs.append(('real', 'ur3_n%d_r%d_real' % (n, rep), X, m, rep, {'set': 'ur3_n%d' % n}))
            for k in ('copula', 'shuffle', 'rw'):
                for q in range(NREP if rep == 0 else 3):
                    jobs.append((k, 'ur3_n%d_r%d_%s%d' % (n, rep, k, q), X, None, 1000 + q, {'set': 'ur3_n%d' % n}))
    # (b) simulation
    for a in (0.25, 0.5, 1.0):
        for f in (1.0, 0.5):
            Ps, ms = sim_tabs(P, pool, a, f, seed=int(a * 100 + f * 10))
            X, _, _ = pe_features(Ps, pool)
            meta['sim_a%g_f%g' % (a, f)] = month_r2(X, ms)[0]
            jobs.append(('real', 'sim_a%g_f%g_real' % (a, f), X, ms, 7, {'set': 'sim_a%g_f%g' % (a, f)}))
            for k in ('copula', 'shuffle', 'rw'):
                for q in range(NREP):
                    jobs.append((k, 'sim_a%g_f%g_%s%d' % (a, f, k, q), X, None, 2000 + q, {'set': 'sim_a%g_f%g' % (a, f)}))
    print(len(jobs), 'jobs', flush=True)
    t = time.time()
    with Pool(2) as pl:
        res = []
        for i, r in enumerate(pl.imap_unordered(task, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(i, round(time.time() - t), flush=True)
    summ = {}
    sets = sorted(set(r['set'] for r in res))
    for s in sets:
        R = [r for r in res if r['set'] == s]
        reals = [r for r in R if r['kind'] == 'real']
        row = {'real_D': [r['D'] for r in reals], 'real_R2c': [r['R2c'] for r in reals]}
        for k in ('copula', 'shuffle', 'rw'):
            N = [r for r in R if r['kind'] == k]
            row[k + '_D'] = (float(np.mean([r['D'] for r in N])), float(np.std([r['D'] for r in N])), float(np.max([r['D'] for r in N])))
            row[k + '_R2c'] = (float(np.mean([r['R2c'] for r in N])), float(np.std([r['R2c'] for r in N])), float(np.max([r['R2c'] for r in N])))
        if 'acc' in reals[0]:
            row['acc'] = [(r['acc'], r['acc_null'], r['acc_null95'], r['cc']) for r in reals]
        summ[s] = row
    save('pe11_cycle1.json', {'meta': meta, 'summary': summ})
    print(json.dumps(meta, indent=1, default=float)[:3000])
    for s, row in summ.items():
        print(s, json.dumps(row, default=lambda x: round(x, 4)))


if __name__ == '__main__':
    main()
