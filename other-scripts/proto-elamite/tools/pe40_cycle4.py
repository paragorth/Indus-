"""pe40 cycle 4: the one part of cycle 1 that passed - the SPECTRAL CIRCULAR ANGLE (atan2 of Laplacian
eigenvectors 2 and 3) - used directly as the rota clock.  Held-out prediction of hidden player occurrences by a
circular kernel on the spectral angle (RINGs) vs a linear kernel on the Fiedler order (LINEs) vs BLOCK / KNN.
Ring shape statistic: angle uniformity (1 - |mean resultant|) and radial CV of the 2-D embedding.
Controls: planted rota (6/12/24), planted blocks (12/24), Ur III AS5 (month).  Nulls: 10 curveball shuffles.
Calendar: PE spectral angle vs museum number (circular-linear correlation, 1,000 label shuffles)."""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'): _o.environ[_v] = '1'
import sys, json, time
import numpy as np
from multiprocessing import Pool
from pe40_common import *
from pe40_eval import BW, KS


def spec_angle(W):
    d = W.sum(1) + 1e-9; Dm = 1 / np.sqrt(d)
    L = np.eye(len(W)) - (Dm[:, None] * W * Dm[None, :])
    ev, V = small_eig(L, 4)
    x, y = V[:, 1] * Dm, V[:, 2] * Dm
    return np.arctan2(y, x), np.hypot(x, y), V[:, 1] * Dm, ev


def ang_kernel(a, Xtr, bw, dp):
    D = np.abs(a[:, None] - a[None, :]); D = np.minimum(D, 2 * np.pi - D)
    K = np.exp(-D / bw); np.fill_diagonal(K, 0)
    return (K @ Xtr) * (Xtr.sum(0) + 1) ** dp


def lin_kernel(f, Xtr, bw, dp):
    r = np.argsort(np.argsort(f)) / len(f) * 2 * np.pi  # rank scale comparable to angle
    D = np.abs(r[:, None] - r[None, :])
    K = np.exp(-D / bw); np.fill_diagonal(K, 0)
    return (K @ Xtr) * (Xtr.sum(0) + 1) ** dp


ABW = (0.05, 0.15, 0.4, 1.0)


def run(a):
    name, X, truth, period, seed = a
    rng = np.random.default_rng(seed)
    g = giant(X); X = X[g][:, X[g].sum(0) >= 2]
    tr = None if truth is None else np.asarray(truth)[g]
    W = weights(X)
    ang, rad, fied, ev = spec_angle(W)
    z = np.exp(1j * ang).mean()
    out = dict(name=name, n=len(X), unif=float(1 - abs(z)), radcv=float(rad.std() / rad.mean()),
               gap23=float(abs(ev[2] - ev[1]) / (ev[3] - ev[1] + 1e-12)))
    if tr is not None:
        b = 2 * np.pi * tr / period
        out['cc'] = float(max(abs(np.exp(1j * (ang - b)).mean()), abs(np.exp(1j * (-ang - b)).mean())))
    res = collections.defaultdict(list)
    for s in range(3):
        Xtr, hid = split_hidden(X, rng)
        a2, _, f2, _ = spec_angle(weights(Xtr))
        res['RINGs'].append(max(ranks(ang_kernel(a2, Xtr, bw, dp), Xtr, hid) for bw in ABW for dp in (0, .5, 1)))
        res['LINEs'].append(max(ranks(lin_kernel(f2, Xtr, bw, dp), Xtr, hid) for bw in ABW for dp in (0, .5, 1)))
        res['BLOCK'].append(max(ranks(block_score(Xtr, k, rng), Xtr, hid) for k in KS))
        res['KNN'].append(ranks(knn_score(Xtr), Xtr, hid))
    for k, v in res.items():
        out[k] = float(np.mean(v))
    out['RS-LS'] = out['RINGs'] - out['LINEs']; out['RS-B'] = out['RINGs'] - out['BLOCK']
    if name == 'PE':
        out['ang'] = ang.tolist(); out['g'] = g.tolist()
    return out


if __name__ == '__main__':
    R = filter_players(pe_rounds('sign')); X, Pl = incidence(R)
    sizes = X.sum(1).astype(int); npl = X.shape[1]
    jobs = []
    for per in (6, 12, 24):
        Xp, st = plant_rota(len(X), sizes, npl, per, 1, 0.3, np.random.default_rng(per))
        jobs.append((f'PLANT_ROTA_p{per}', Xp, st, per, 1))
    for k in (12, 24):
        Xb, st = plant_block(len(X), sizes, npl, k, 0.3, np.random.default_rng(100 + k))
        jobs.append((f'PLANT_BLOCK_k{k}', Xb, st, k, 1))
    from pe40_cycle1 import ur3_month
    Xu, mu = ur3_month()
    jobs.append(('UR3_AS5', Xu, mu, 12, 1))
    jobs.append(('PE', X, None, None, 1))
    rng = np.random.default_rng(9)
    for i in range(10):
        jobs.append((f'NULL{i}', curveball(X, rng), None, None, 200 + i))
    out = []
    with Pool(2) as pool:
        for o in pool.imap(run, jobs):
            out.append(o)
            print('ROW', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ('ang', 'g')}), flush=True)
    # calendar vs museum number
    pe = [o for o in out if o['name'] == 'PE'][0]
    cat = json.load(open(os.path.join(HERE, '..', 'data', 'pe17_ckpt', 'pe_cat.json')))
    from pe29_common import _musparse
    ids = [R[i][0] for i in pe['g']]
    ang = np.array(pe['ang']); mus = []
    for t in ids:
        pre, no = _musparse(cat.get(t, {}).get('museum_no', ''))
        mus.append(no if pre and pre.startswith('Sb') else np.nan)
    mus = np.array(mus, float); ok = ~np.isnan(mus)

    def cl(a, x):  # circular-linear correlation (Mardia)
        x = np.argsort(np.argsort(x)).astype(float)
        rc = np.corrcoef(x, np.cos(a))[0, 1]; rs = np.corrcoef(x, np.sin(a))[0, 1]; rcs = np.corrcoef(np.cos(a), np.sin(a))[0, 1]
        return np.sqrt((rc ** 2 + rs ** 2 - 2 * rc * rs * rcs) / (1 - rcs ** 2))
    obs = cl(ang[ok], mus[ok]); rr = np.random.default_rng(1)
    nul = np.array([cl(ang[ok], rr.permutation(mus[ok])) for _ in range(1000)])
    cal = dict(n=int(ok.sum()), r=float(obs), null_mean=float(nul.mean()), p=float((1 + (nul >= obs).sum()) / 1001))
    for o in out:
        o.pop('ang', None); o.pop('g', None)
    json.dump(dict(rows=out, calendar=cal), open(os.path.join(CK, 'cycle4.json'), 'w'), indent=1)
    for o in out:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}))
    print('calendar', cal)
