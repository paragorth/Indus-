#!/usr/bin/env python3
"""LA-30 cycle 1: hidden-currency rates fitted on half the tablets, tested on the other half.

Real Linear A KU-RO / PO-TO-KU-RO sections (32). Per replicate: 10 random tablet splits;
in each, ~2x10^4 random rate vectors (simple rationals p/q <= 12, half the types left at 1)
x random fraction values, then 16 annealed chains x 300 steps with 'solve' moves.
Statistic: mean held-out gain = balanced test sections (best train vector) - balanced
test sections under the plain sum (all rates 1, site fraction values).
Nulls (same search): N1 commodity labels shuffled across all entries; N2 totals replaced
by random integers of the same size (log-uniform in [T/2, 2T], fraction signs kept).
Planted: random rate vector (half the types != 1) generates all totals; 30 % of sections
then get random totals (damage). Usage: la30_c1.py MODE FRACMODE NREP
"""
import sys, json, os, time
from multiprocessing import Pool
from la30_common import *

MODE = sys.argv[1] if len(sys.argv) > 1 else 'exact'
FRAC = sys.argv[2] if len(sys.argv) > 2 else 'free'
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 40
NSPLIT = 10
SECS = la_sections(('KU-RO', 'PO-TO-KU-RO'))


def make(kind, seed):
    rng = np.random.default_rng(seed)
    secs = json.loads(json.dumps(SECS))
    truth = None
    if kind == 'N1':
        pool = [e[0] for s in secs for e in s['entries']]
        rng.shuffle(pool)
        k = 0
        for s in secs:
            for i, e in enumerate(s['entries']):
                s['entries'][i] = [pool[k], e[1], e[2]]; k += 1
    elif kind == 'N2':
        for s in secs:
            T = max(s['total'][0], 1)
            s['total'] = [int(round(np.exp(rng.uniform(np.log(T / 2), np.log(2 * T))))), s['total'][1]]
    elif kind == 'P':
        des = Design(secs)
        R = rng_rates(rng, 1, len(des.types), 0.5)
        F0 = np.array([SITE.get(l, 1 / 16) for l in des.letters])
        Sx, _ = des.S_T(R, F0[None])
        for a, s in enumerate(secs):
            v = Sx[0, a]
            if rng.random() < 0.3:
                v = np.exp(rng.uniform(np.log(max(v, 1) / 2), np.log(2 * max(v, 1))))
                v = round(v)
            if MODE == 'round':
                v = round(v)
            s['total'] = [float(v), []]
        truth = dict(zip(des.types, R[0].tolist()))
    return secs, truth


def run(job):
    kind, seed = job
    out = os.path.join(CK, f'c1_{MODE}_{FRAC}_{kind}_{seed}.json')
    if os.path.exists(out):
        return json.load(open(out))
    secs, truth = make(kind, seed)
    des = Design(secs)
    rng = np.random.default_rng(1000 + seed)
    tabs = sorted({s['id'] for s in secs})
    mixed = des.ntypes_per_sec >= 2
    gains = []; gmix = []; trs = []
    for sp in range(NSPLIT):
        rng.shuffle(tabs)
        tr = set(tabs[:len(tabs) // 2])
        itr = np.array([i for i, s in enumerate(secs) if s['id'] in tr])
        ite = np.array([i for i, s in enumerate(secs) if s['id'] not in tr])
        srch = Search(des, mode=MODE, fracmode=FRAC, seed=seed * 100 + sp)
        k, R, F = srch.fit(itr, n_random=20000, n_chain=12, n_steps=250)
        _, _, bb = srch.score(R[None], F[None], ite)
        _, _, b0 = srch.score(np.ones((1, srch.K)), srch.F0[None], ite)
        gains.append(int(bb.sum() - b0.sum()))
        gmix.append(int((bb[0] & mixed[ite]).sum() - (b0[0] & mixed[ite]).sum()))
        _, _, btr = srch.score(np.ones((1, srch.K)), srch.F0[None], itr)
        trs.append(float(k) - int(btr.sum()))
    res = {'kind': kind, 'seed': seed, 'gain': float(np.mean(gains)), 'gain_mixed': float(np.mean(gmix)),
           'train_gain': float(np.mean(trs)), 'gains': gains}
    if truth is not None:
        srch = Search(des, mode=MODE, fracmode=FRAC, seed=seed)
        k, R, F = srch.fit(np.arange(len(secs)), n_random=20000, n_chain=12, n_steps=250)
        nz = [t for t, v in truth.items() if v != 1]
        res['recovered'] = sum(1 for t, v in zip(des.types, R) if t in nz and abs(v - truth[t]) < 1e-9)
        res['n_planted'] = len(nz)
    json.dump(res, open(out, 'w'))
    return res


if __name__ == '__main__':
    jobs = [('real', s) for s in range(4)] + [(k, s) for s in range(NREP) for k in ('N1', 'N2')] + \
           [('P', s) for s in range(NREP // 2)]
    t = time.time()
    with Pool(2) as p:
        R = p.map(run, jobs, chunksize=1)
    by = {}
    for r in R:
        by.setdefault(r['kind'], []).append(r)
    summ = {}
    real = np.mean([r['gain'] for r in by['real']])
    realm = np.mean([r['gain_mixed'] for r in by['real']])
    for k, L in by.items():
        g = np.array([r['gain'] for r in L]); gm = np.array([r['gain_mixed'] for r in L])
        summ[k] = {'n': len(L), 'gain_mean': float(g.mean()), 'gain_sd': float(g.std()),
                   'gain_mixed_mean': float(gm.mean()), 'train_gain': float(np.mean([r['train_gain'] for r in L])),
                   'P_ge_real': float((1 + (g >= real).sum()) / (1 + len(g))),
                   'P_mixed_ge_real': float((1 + (gm >= realm).sum()) / (1 + len(gm)))}
        if k == 'P':
            summ[k]['recovered'] = int(sum(r['recovered'] for r in L)); summ[k]['planted'] = int(sum(r['n_planted'] for r in L))
            summ[k]['P_planted_gain_gt_N1_95'] = float(np.mean(g > np.percentile([r['gain'] for r in by['N1']], 95)))
    summ['time_s'] = time.time() - t
    json.dump(summ, open(os.path.join(CK, f'c1_{MODE}_{FRAC}_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
