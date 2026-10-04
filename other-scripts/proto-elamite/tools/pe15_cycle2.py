"""pe15 cycle 2: hidden 'offices' by stochastic block model.

For each network, a bipartite degree-corrected SBM is fitted by annealed Metropolis MCMC
with many restarts over a grid of (Kc, Kr).  The number of blocks is chosen by
WHOLE-TABLET hold-out: 5 folds of tablets; held-out entries whose consumer and resource
were seen in training are scored by log2 P(resource | consumer).  Competitors:
  DEG  = K=1 (resource popularity only, the configuration-model predictor)
  HIST = the consumer's own past resources (Dirichlet-smoothed, c tuned on folds)
  SBM  = best (Kc,Kr)
Nulls: the same pipeline on (a) events re-paired at random (fixed weighted degrees)
and (b) resources shuffled among entries of the same tablet.  Controls with known
offices (UR3 provenience, Drehem receiving official, Linear B series, planted modules):
NMI between consumer blocks and the truth vs block-label permutations.
"""
import json, os, sys, math, random
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe15_common import *

GRID = [(1, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (6, 4), (6, 6), (8, 4), (8, 6), (10, 6), (12, 8)]
RESTARTS = int(os.environ.get('RESTARTS', 6))
SWEEPS = int(os.environ.get('SWEEPS', 150))
CS = [0.1, 0.3, 1, 3, 10]


def folds_of(ev, k, seed):
    tabs = sorted({e[2] for e in ev})
    random.Random(seed).shuffle(tabs)
    return {t: i % k for i, t in enumerate(tabs)}


def cv(ev, seed, grid=GRID):
    """Returns mean held-out bits/event per model, and n scored."""
    fo = folds_of(ev, 5, seed)
    tot = defaultdict(float)
    n = 0
    for f in range(5):
        tr = [e for e in ev if fo[e[2]] != f]
        te = [e for e in ev if fo[e[2]] == f]
        W, C, R = matrix(tr, min_c=1, min_r=1)
        ci = {c: i for i, c in enumerate(C)}; ri = {r: i for i, r in enumerate(R)}
        te = [(ci[e[0]], ri[e[1]]) for e in te if e[0] in ci and e[1] in ri]
        if not te:
            continue
        ti = np.array([a for a, b in te]); tj = np.array([b for a, b in te])
        n += len(te)
        for c in CS:
            P = hist_prob(W, c)
            tot['HIST_%g' % c] += -np.log2(P[ti, tj]).sum()
        for (kc, kr) in grid:
            kc2, kr2 = min(kc, W.shape[0]), min(kr, W.shape[1])
            if kc == 1 and kr == 1:
                bc = np.zeros(W.shape[0], int); br = np.zeros(W.shape[1], int)
            else:
                bc, br, L = sbm_fit(W, kc2, kr2, restarts=RESTARTS, sweeps=SWEEPS, seed=seed * 10 + f)
            P = cond_prob(W, bc, br, kc2, kr2)
            tot['SBM_%d_%d' % (kc, kr)] += -np.log2(P[ti, tj]).sum()
            # SBM + own history: Dirichlet prior centred on the block profile
            for c in (1, 3):
                Q = (W + c * P) / (W.sum(1, keepdims=True) + c)
                tot['SBMH%g_%d_%d' % (c, kc, kr)] += -np.log2(Q[ti, tj]).sum()
    return {k: v / max(n, 1) for k, v in tot.items()}, n


def best_of(res, prefix):
    ks = [k for k in res if k.startswith(prefix)]
    k = min(ks, key=lambda x: res[x])
    return k, res[k]


def shuffle_global(ev, rng):
    rs = [e[1] for e in ev]
    rng.shuffle(rs)
    return [(e[0], r, e[2]) for e, r in zip(ev, rs)]


def shuffle_tablet(ev, rng):
    by = defaultdict(list)
    for e in ev:
        by[e[2]].append(e)
    out = []
    for t, L in by.items():
        rs = [e[1] for e in L]
        rng.shuffle(rs)
        out += [(e[0], r, t) for e, r in zip(L, rs)]
    return out


def job(args):
    name, ev, truth, seed, kind = args
    rng = random.Random(seed)
    if kind == 'glob':
        ev = shuffle_global(ev, rng)
    elif kind == 'tab':
        ev = shuffle_tablet(ev, rng)
    res, n = cv(ev, seed)
    kD = res['SBM_1_1']
    kH, vH = best_of(res, 'HIST')
    kS, vS = min(((k, v) for k, v in res.items() if k.startswith('SBM_') and k != 'SBM_1_1'), key=lambda x: x[1])
    kSH, vSH = best_of(res, 'SBMH')
    out = {'net': name, 'kind': kind, 'seed': seed, 'n': n, 'DEG': kD, 'HIST': vH, 'HIST_k': kH,
           'SBM': vS, 'SBM_k': kS, 'SBMH': vSH, 'SBMH_k': kSH, 'all': res}
    # full fit at chosen K, truth matching
    kc, kr = map(int, kS.split('_')[1:])
    W, C, R = matrix(ev, min_c=2, min_r=2)
    bc, br, L = sbm_fit(W, min(kc, W.shape[0]), min(kr, W.shape[1]), restarts=RESTARTS * 3, sweeps=SWEEPS * 2, seed=seed + 99)
    bc2, br2, L2 = sbm_fit(W, min(kc, W.shape[0]), min(kr, W.shape[1]), restarts=RESTARTS * 3, sweeps=SWEEPS * 2, seed=seed + 7777)
    out['stability_nmi_c'] = float(nmi(bc, bc2)); out['stability_nmi_r'] = float(nmi(br, br2))
    out['blocks'] = {'C': C, 'R': R, 'bc': bc.tolist(), 'br': br.tolist(), 'W': W.tolist() if W.size < 60000 else None}
    if truth and kind == 'real':
        lab = []
        idx = []
        for i, c in enumerate(C):
            if c in truth:
                lab.append(truth[c]); idx.append(i)
        if len(set(lab)) > 1:
            g = bc[idx]
            obs = nmi(lab, g)
            nr = np.random.default_rng(seed)
            perm = [nmi(lab, nr.permutation(g)) for _ in range(200)]
            # degree-only partition with the same number of blocks
            deg = W.sum(1)[idx]
            q = np.searchsorted(np.quantile(deg, np.linspace(0, 1, len(set(g)) + 1)[1:-1]), deg)
            out['truth'] = {'nmi': float(obs), 'perm_mean': float(np.mean(perm)), 'perm_max': float(np.max(perm)),
                            'deg_nmi': float(nmi(lab, q)), 'n_lab': len(lab), 'n_classes': len(set(lab))}
    print('%-10s %-4s s%d n %5d  DEG %.3f HIST %.3f SBM %.3f (%s) SBM+H %.3f (%s) gain SBM-DEG %+.3f SBMH-HIST %+.3f stab %.2f/%.2f %s' % (
        name, kind, seed, n, kD, vH, vS, kS, vSH, kSH, kD - vS, vH - vSH, out['stability_nmi_c'], out['stability_nmi_r'],
        ('truth NMI %.3f (perm %.3f max %.3f; degree-split %.3f)' % (out['truth']['nmi'], out['truth']['perm_mean'], out['truth']['perm_max'], out['truth']['deg_nmi'])) if 'truth' in out else ''), flush=True)
    return out


def consumer_truth(ev, tablet_label):
    by = defaultdict(Counter)
    for c, r, t in ev:
        if tablet_label.get(t):
            by[c][tablet_label[t]] += 1
    return {c: v.most_common(1)[0][0] for c, v in by.items()}


if __name__ == '__main__':
    Dall = load_nets()
    D, T = Dall['nets'], Dall['truth']
    which = sys.argv[1:] or ['PE_SIGN', 'PE_STR', 'PE_HDR', 'PE_HRES', 'UR3_STR', 'UR3_SIGN', 'UR3_DAB', 'LB_STR', 'LB_SIGN', 'PLANT_MOD', 'PLANT_NULL']
    jobs = []
    for nm in which:
        reps = 1
        if nm in ('PLANT_MOD', 'PLANT_NULL'):
            ev, tr = planted(480, 41, 6700, 5, 0, mix=0.15 if nm == 'PLANT_MOD' else 1.0)
            truth = tr
        elif nm.startswith('UR3_STR') or nm.startswith('UR3_SIGN'):
            base = 'UR3_STR' if nm.startswith('UR3_STR') else 'UR3_SIGN'
            ev = subsample_tablets(D[base], len(D['PE_' + base.split('_')[1]]), 0)
            truth = consumer_truth(ev, {p: v['prov'] for p, v in T['UR3'].items()})
        elif nm == 'UR3_DAB':
            ev = D[nm]
            truth = consumer_truth(ev, {p: v['office'] for p, v in T['UR3'].items()})
        elif nm.startswith('LB'):
            ev = D[nm]
            truth = consumer_truth(ev, {p: v['series'] for p, v in T['LB'].items()})
        else:
            ev, truth = D[nm], None
        ev = [tuple(e) for e in ev]
        jobs.append((nm, ev, truth, 1, 'real'))
        jobs.append((nm, ev, truth, 2, 'glob'))
        jobs.append((nm, ev, truth, 3, 'tab'))
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    tag = os.environ.get('TAG', 'c2')
    json.dump(res, open(os.path.join(CK, tag + '.json'), 'w'))
