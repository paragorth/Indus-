"""pe58 cycle 2: PE exchange rates with the calibrated estimator, plus matched-power controls.
A Ur III rate lines: lattice test vs quantities shuffled across the corpus (keeps round numbers).
B planted ladder / arbitrary / physical factors on the PE skeleton, k=17 (PE-sized), 6 seeds each.
C PE: factor of every pe52 weighted sign (300 random calibrated specs x bootstraps), pairwise rates among them,
  lattice tests (p,q<=6 and <=4) vs jitter, empirical (frequent unweighted signs) and shuffled-quantity nulls,
  spike (rate-likeness) per sign vs quantities shuffled within tablet, physics point-distance test."""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe58_lib import *
from multiprocessing import Pool

FLOOR = 0.12


def lat_both(x, pool):
    return {f'p{pm}': lattice_test(x, pool, n=10000, pmax=pm) for pm in (6, 4)}


def job(args):
    kind, seed = args
    if kind == 'UR3':
        C, voc = ur3_ta()
        S = frequent(C)
        tab, use = pair_table(C, S)
        pool = []
        for sd in range(10):
            _, u = pair_table(qshuf_sys(C, 500 + sd), S); pool += u
        return dict(job='UR3', nuse=len(use), npool=len(pool), lat=lat_both(use, pool))
    pe = pe_corpus()
    W = json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))['signs']
    if kind in ('ladder', 'arb', 'phys'):
        C, f = planted(pe, seed, kind, k=17, exclude=set(W))
        E = Est(C, W=set(f))
        est = {s: E.factor(s, n=100, seed=seed, spec2=True) for s in f}
        est = {s: e for s, e in est.items() if e}
        x = [e['med'] for e in est.values() if abs(e['med']) >= FLOOR]
        rng = random.Random(seed)
        oth = [s for s in frequent(C, 20, 8) if s not in f and s not in W]
        oth = rng.sample(oth, min(40, len(oth)))
        pool = [e['med'] for e in (E.factor(s, n=60, seed=seed, spec2=True) for s in oth) if e and abs(e['med']) >= FLOOR]
        return dict(job=f'PLANT_{kind}_{seed}', mae=float(np.mean([abs(e['med'] - math.log(f[s])) for s, e in est.items()])),
                    k=len(x), npool=len(pool), lat=lat_both(x, pool),
                    phys=phys_dist_test({s: e['med'] for s, e in est.items() if abs(e['med']) >= FLOOR}, n=5000, seed=seed))
    if kind == 'PE':
        E = Est(pe, W=set(W))
        est = {s: E.factor(s, n=300, seed=1, spec2=True) for s in W}
        sp = {s: E.spike(s) for s in W}
        return dict(job='PE', est=est, spike=sp)
    if kind == 'PEpairs':
        tab, use = pair_table(pe, sorted(W), minpair=3)
        return dict(job='PEpairs', tab={f'{a}|{b}': v for (a, b), v in tab.items()}, use=use)
    if kind == 'PEemp':
        E = Est(pe, W=set(W))
        oth = [s for s in frequent(pe, 15, 5) if s not in W]
        return dict(job='PEemp', est={s: E.factor(s, n=100, seed=2, spec2=True) for s in oth})
    if kind == 'PEshuf':
        C = qshuf_sys(pe, 900 + seed) if seed < 100 else qshuf_tab(pe, 900 + seed)
        E = Est(C, W=set(W))
        out = {'est': {s: E.factor(s, n=60, seed=seed, spec2=True) for s in W}}
        if seed >= 100:
            out['spike'] = {s: E.spike(s) for s in W}
        _, use = pair_table(C, sorted(W), minpair=3)
        out['use'] = use
        return dict(job=f'PEshuf_{seed}', **out)


if __name__ == '__main__':
    jobs = [('PE', 0), ('PEpairs', 0), ('PEemp', 0), ('UR3', 0)]
    jobs += [('PEshuf', s) for s in range(5)] + [('PEshuf', 100 + s) for s in range(10)]
    jobs += [(k, s) for k in ('ladder', 'arb', 'phys') for s in range(6)]
    with Pool(2) as P:
        res = P.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), default=str)
    print('done', len(res))
