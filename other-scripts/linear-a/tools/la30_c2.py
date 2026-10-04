#!/usr/bin/env python3
"""LA-30 cycle 2: Ur III positive control and planted-rate recovery at Linear A size.

Ur III merchant (dam-gar3) accounts, CDLI ATF (scratchpad): each goods line is followed by
'ku3-bi <silver>'. Section = one text's lines in the 12 commonest commodities; total = sum of
the scribe's own silver values for those lines; the silver values are then hidden. The
'true' rate of a commodity = its median written price. Same search as cycle 1 (continuous
log-uniform rates 1e-4..1 plus solve moves; no fractions). Tolerances rel5, rel10.
Held-out: half the texts fit, half tested. Nulls: N1 commodity labels shuffled across
entries, N2 totals multiplied by log-uniform [1/2, 2]. Also a full fit: fitted rates vs
median prices. Second control: 'Ur III planted' = the same sections with totals recomputed
from the median prices (a perfectly fixed currency), to separate method power from real
price scatter. Usage: la30_c2.py MODE NREP [NSEC]  (NSEC = 32: a random Linear A-sized subset per replicate)
"""
import sys, json, os, time
from multiprocessing import Pool
from la30_common import *

MODE = sys.argv[1] if len(sys.argv) > 1 else 'rel10'
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NSPLIT = 6
NSEC = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = all 58 sections; 32 = Linear A size
TAG = f'_n{NSEC}' if NSEC else ''
U = ur3_sections()
c = Counter(e[0] for u in U for e in u['entries'])
TOP = [k for k, _ in c.most_common(12)]
MED = {k: float(np.median([e[2] / e[1] for u in U for e in u['entries'] if e[0] == k])) for k in TOP}
BASE = []
for u in U:
    E = [e for e in u['entries'] if e[0] in TOP]
    if len(E) >= 2:
        BASE.append({'id': u['id'], 'entries': [[e[0], e[1], []] for e in E],
                     'silver': [e[2] for e in E], 'total': [sum(e[2] for e in E), []]})


def make(kind, seed):
    rng = np.random.default_rng(seed)
    secs = json.loads(json.dumps(BASE))
    if NSEC:
        sub = np.random.default_rng(10 ** 6 + seed).choice(len(secs), NSEC, replace=False)
        secs = [secs[i] for i in sorted(sub)]
    if kind in ('N1',):
        pool = [e[0] for s in secs for e in s['entries']]
        rng.shuffle(pool); k = 0
        for s in secs:
            for e in s['entries']:
                e[0] = pool[k]; k += 1
    if kind == 'N2':
        for s in secs:
            s['total'][0] *= float(np.exp(rng.uniform(np.log(.5), np.log(2))))
    if kind == 'fixed':  # planted: one fixed price vector = medians
        for s in secs:
            s['total'][0] = sum(MED[e[0]] * e[1] for e in s['entries'])
    if kind == 'fixedN1':
        for s in secs:
            s['total'][0] = sum(MED[e[0]] * e[1] for e in s['entries'])
        pool = [e[0] for s in secs for e in s['entries']]
        rng.shuffle(pool); k = 0
        for s in secs:
            for e in s['entries']:
                e[0] = pool[k]; k += 1
    return secs


def run(job):
    kind, seed = job
    out = os.path.join(CK, f'c2_{MODE}{TAG}_{kind}_{seed}.json')
    if os.path.exists(out):
        return json.load(open(out))
    secs = make(kind, seed)
    des = Design(secs, types=TOP, letters=[])
    rng = np.random.default_rng(500 + seed)
    tabs = [s['id'] for s in secs]
    gains = []; rec = []
    for sp in range(NSPLIT):
        rng.shuffle(tabs)
        tr = set(tabs[:len(tabs) // 2])
        itr = np.array([i for i, s in enumerate(secs) if s['id'] in tr])
        ite = np.array([i for i, s in enumerate(secs) if s['id'] not in tr])
        srch = Search(des, mode=MODE, grid=None, logrange=(1e-4, 1), fracmode='site', seed=seed * 100 + sp)
        k, R, F = srch.fit(itr, n_random=20000, n_chain=12, n_steps=300)
        _, _, bb = srch.score(R[None], F[None], ite)
        gains.append(int(bb.sum()))
        # rate recovery: |log(fitted/median)| < log(1.25) for types present in train
        pres = des.I[itr].sum(0) > 0
        rec.append(float(np.mean([abs(np.log(R[j] / MED[t])) < np.log(1.25) for j, t in enumerate(TOP) if pres[j]])))
    res = {'kind': kind, 'seed': seed, 'gain': float(np.mean(gains)), 'rec': float(np.mean(rec)), 'n_test': len(ite)}
    json.dump(res, open(out, 'w'))
    return res


if __name__ == '__main__':
    print('sections', len(BASE), 'types', TOP)
    des = Design(BASE, types=TOP, letters=[])
    Rm = np.array([[MED[t] for t in TOP]])
    for m in ('rel5', 'rel10'):
        s = Search(des, mode=m, grid=None, logrange=(1e-4, 1))
        print(m, 'median prices balance', int(s.score(Rm, s.F0[None], np.arange(len(BASE)))[0][0]), 'of', len(BASE))
    jobs = [(k, s) for s in range(NREP) for k in ('real', 'N1', 'N2', 'fixed', 'fixedN1')]
    t = time.time()
    with Pool(2) as p:
        R = p.map(run, jobs, chunksize=1)
    by = {}
    for r in R:
        by.setdefault(r['kind'], []).append(r)
    summ = {}
    for k, L in by.items():
        g = np.array([r['gain'] for r in L])
        summ[k] = {'n': len(L), 'heldout_balanced_mean': float(g.mean()), 'sd': float(g.std()),
                   'rate_recovered_within_25pct': float(np.mean([r['rec'] for r in L])), 'n_test': L[0]['n_test']}
    for a, b in (('real', 'N1'), ('real', 'N2'), ('fixed', 'fixedN1')):
        ga = np.array([r['gain'] for r in by[a]]); gb = np.array([r['gain'] for r in by[b]])
        summ[f'{a}_vs_{b}'] = {'diff': float(ga.mean() - gb.mean()),
                               'P_null_mean_ge': float(np.mean([np.mean(np.random.default_rng(i).choice(gb, len(ga))) >= ga.mean() for i in range(2000)]))}
    # full fit, real
    srch = Search(des, mode=MODE, grid=None, logrange=(1e-4, 1), seed=7)
    k, R, F = srch.fit(np.arange(len(BASE)), n_random=100000, n_chain=16, n_steps=500)
    summ['fullfit'] = {'balanced': float(k), 'rates_vs_median': {t: [float(R[j]), MED[t]] for j, t in enumerate(TOP)}}
    summ['time_s'] = time.time() - t
    json.dump(summ, open(os.path.join(CK, f'c2_{MODE}{TAG}_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
