"""pe40 cycle 1: calibrate the rota detector.
 planted rota on PE sizes (periods 6/12/24, handover overlap, 30% noise), planted blocks (no cycle),
 Ur III Drehem (Amar-Suen 5, players = rare words, truth = month), then real PE (rare name signs)."""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): _o.environ[_v]='1'
import sys, json, time
import numpy as np
from multiprocessing import Pool
from pe40_common import *
from pe40_eval import evaluate
sys.path.insert(0, HERE)


def job(args):
    name, X, truth, period, seed = args
    t = time.time()
    r = evaluate(X, truth, period, seed=seed)
    r['name'] = name; r['sec'] = round(time.time() - t)
    return r


def ur3_month():
    from pe29_common import ur3_tablets
    U = ur3_tablets(cache=os.path.join(CK, 'ur3_AS05.json'))
    rounds = [(u['id'], [w for w in u['bases'] if not w[0].isdigit()]) for u in U]
    month = {u['id']: int(u['date'].split('.')[2]) for u in U}
    F = filter_players(rounds, 2, None, 0.05)
    X, P = incidence(F)
    return X, [month[t] - 1 for t, _ in F]


if __name__ == '__main__':
    R = filter_players(pe_rounds('sign'))
    X, P = incidence(R)
    sizes = X.sum(1).astype(int); npl = X.shape[1]
    jobs = []
    for per in (6, 12, 24):
        Xp, st = plant_rota(len(X), sizes, npl, per, overlap=1, noise=0.3, rng=np.random.default_rng(per))
        jobs.append((f'PLANT_ROTA_p{per}', Xp, st, per, 1))
    for k in (12, 24):
        Xb, st = plant_block(len(X), sizes, npl, k, noise=0.3, rng=np.random.default_rng(100 + k))
        jobs.append((f'PLANT_BLOCK_k{k}', Xb, st, k, 1))
    Xu, mu = ur3_month()
    jobs.append(('UR3_AS5_month', Xu, mu, 12, 1))
    jobs.append(('PE_signs', X, None, None, 1))
    with Pool(2) as pool:
        out = pool.map(job, jobs)
    json.dump(out, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1)
    keys = ['n', 'p', 'RING', 'LINE', 'BLOCK', 'KNN', 'FREQ', 'R-L', 'R-B', 'R-K', 'RING_2nd', 'LINE_2nd', 'BLOCK_2nd', 'FREQ_2nd', 'n_2nd', 'cc_circ', 'cc_line', 'cc_spec', 'sec']
    for r in out:
        print(r['name'], ' '.join(f"{k}={r[k]:.3f}" if isinstance(r.get(k), float) else f"{k}={r.get(k)}" for k in keys if k in r))
