"""pe15 cycle 1: food-web structure.  NODF (vs curveball fixed-degree null), Bluethgen H2'
(vs weighted event-shuffle null), Barber modularity Q (vs curveball), degree distributions.
Controls are cut to PE size by whole tablets (5 draws).  Planted modular and planted
no-module (configuration) networks of PE_SIGN size calibrate the statistics."""
import json, os, sys, math
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe15_common import *

NN = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def pl_alpha(deg, xmin=2):
    d = np.asarray([x for x in deg if x >= xmin], float)
    return 1 + len(d) / np.sum(np.log(d / (xmin - 0.5))) if len(d) else float('nan')


def analyse(args):
    name, ev, seed = args
    W, C, R = matrix(ev, min_c=2, min_r=2)
    B = (W > 0).astype(int)
    rng = np.random.default_rng(seed)
    obs_n = nodf(B)
    nulls = [nodf(curveball(B, seed * 7919 + k, nswap=20 * B.shape[0])) for k in range(NN)]
    h2o, _ = h2(W)
    h2n = [h2(shuffle_events(W, rng))[0] for _ in range(NN)]
    # stricter weighted null: resources shuffled among the entries of the same tablet
    h2t = []
    by = {}
    for e in ev:
        by.setdefault(e[2], []).append(e)
    for k in range(NN):
        sh = []
        for t, L in by.items():
            rs = [e[1] for e in L]
            rng.shuffle(rs)
            sh += [(e[0], r, t) for e, r in zip(L, rs)]
        Wt, Ct, Rt = matrix(sh, min_c=2, min_r=2)
        h2t.append(h2(Wt)[0])
    q, _ = brim(W, restarts=30, seed=seed)
    qn = [brim(curveball(B, seed * 31 + k, nswap=20 * B.shape[0]), restarts=30, seed=k)[0] for k in range(max(10, NN // 5))]
    cdeg = B.sum(1)
    out = {'net': name, 'seed': seed, 'nC': W.shape[0], 'nR': W.shape[1], 'events': int(W.sum()),
           'links': int(B.sum()), 'connectance': float(B.mean()),
           'nodf': obs_n, 'nodf_null': float(np.mean(nulls)), 'nodf_sd': float(np.std(nulls)),
           'nodf_z': float((obs_n - np.mean(nulls)) / (np.std(nulls) + 1e-9)),
           'h2': float(h2o), 'h2_null': float(np.mean(h2n)), 'h2_z': float((h2o - np.mean(h2n)) / (np.std(h2n) + 1e-9)),
           'h2_tab_null': float(np.mean(h2t)), 'h2_tab_z': float((h2o - np.mean(h2t)) / (np.std(h2t) + 1e-9)),
           'Q': float(q), 'Q_null': float(np.mean(qn)), 'Q_z': float((q - np.mean(qn)) / (np.std(qn) + 1e-9)),
           'cons_links_share1': float(np.mean(cdeg == 1)), 'cons_alpha': float(pl_alpha(cdeg)),
           'res_deg_max_share': float(B.sum(0).max() / B.shape[0])}
    print('%-10s s%d %4dx%-3d ev %5d C %.3f NODF %5.1f (null %5.1f, z %+6.1f) H2\' %.3f (null %.3f z %+6.1f; tablet-null %.3f z %+5.1f) Q %.3f (null %.3f z %+5.1f) one-link %.2f' % (
        name, seed, W.shape[0], W.shape[1], W.sum(), B.mean(), obs_n, np.mean(nulls), out['nodf_z'], h2o, np.mean(h2n), out['h2_z'], out['h2_tab_null'], out['h2_tab_z'],
        q, np.mean(qn), out['Q_z'], out['cons_links_share1']), flush=True)
    return out


if __name__ == '__main__':
    D = load_nets()['nets']
    jobs = []
    for k in ('PE_STR', 'PE_SIGN', 'PE_HDR', 'PE_HRES', 'UR3_DAB', 'LB_STR', 'LB_SIGN'):
        jobs.append((k, D[k], 0))
    for s in range(5):
        jobs.append(('UR3_STR', subsample_tablets(D['UR3_STR'], len(D['PE_STR']), s), s))
        jobs.append(('UR3_SIGN', subsample_tablets(D['UR3_SIGN'], len(D['PE_SIGN']), s), s))
        jobs.append(('LB_STRsub', subsample_tablets(D['LB_STR'], 2000, s), s))
    for s in range(3):
        ev, _ = planted(480, 41, 6700, 5, s)
        jobs.append(('PLANT_MOD', ev, s))
        ev, _ = planted(480, 41, 6700, 5, s, mix=1.0)
        jobs.append(('PLANT_NULL', ev, s))
    with Pool(2) as p:
        res = p.map(analyse, jobs, chunksize=1)
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
