"""la73 cycle 1: does archive time depth leave a recoverable trace in who-co-occurs-with-whom?
(1a) HT reference table; (1b) planted worlds recovered blind; (1c) Ur III dated archives of known
window (3 months ... 40 years) at HT tablet count, real names, plus name-shuffled copies."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la73_abc as A, la73_common as C, la73_prep as P
import numpy as np
from scipy.stats import spearmanr

def ur3_draws(rng, n=158):
    U = P.ur3(); out = []
    for site in ['Umma', 'Puzriš-Dagan', 'Girsu']:
        X = [x for x in U if x['site'] == site]; t = np.array([x['t'] for x in X])
        for win in [3, 12, 36, 120, 480]:
            starts = [s for s in range(t.min(), t.max() - win + 2) if np.sum((t >= s) & (t < s + win)) >= n]
            if not starts:
                continue
            for rep in range(3):
                s0 = starts[rng.integers(len(starts))]
                pool = np.flatnonzero((t >= s0) & (t < s0 + win))
                pick = rng.choice(pool, n, replace=False)
                out.append(dict(site=site, win=win, start=int(s0), docs=[X[i]['w'] for i in pick]))
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(73)
    res = {}
    D = P.la_docs('Haghia Triada'); L = np.array([len(x['w']) for x in D])
    th, s, d = A.table('ht', 60000, L, seed=1)
    print('table', s.shape, flush=True)
    # (1b) planted
    pth, ps, pd = A.table('ht_planted', 400, L, seed=99, chunks=8)
    rows = {k: [] for k in ['logR', 'R_eff', 'logNp', 'phi', 'loc', 'logK', 'kappa', 'logm', 'log_persons_ever', 'log_surv_frac', 'log_tablets_ever']}
    for i in range(len(ps)):
        o = A.abc(ps[i], th, s, d)
        truth = dict(zip(C.PNAMES, pth[i])); truth['R_eff'] = pd[i, 0]
        truth['log_persons_ever'] = np.log10(pd[i, 1]); truth['log_surv_frac'] = np.log10(max(pd[i, 2], 1e-9))
        truth['log_tablets_ever'] = np.log10(pd[i, 3])
        for k in rows:
            rows[k].append((truth[k], *o[k]))
    plant = {}
    for k, v in rows.items():
        v = np.array(v); prior_sd = np.std(v[:, 0])
        r = np.corrcoef(v[:, 0], v[:, 1])[0, 1]; cov = np.mean((v[:, 0] >= v[:, 2]) & (v[:, 0] <= v[:, 3]))
        rmse = np.sqrt(np.mean((v[:, 0] - v[:, 1]) ** 2))
        plant[k] = dict(r=round(r, 3), cov80=round(cov, 3), rmse_over_priorsd=round(rmse / prior_sd, 3))
        print('planted', k, plant[k], flush=True)
    res['planted'] = plant
    # (1c) Ur III
    dr = ur3_draws(rng)
    pool = [np.array([len(x) for x in g['docs']]) for g in dr]
    uth, us, ud = A.table('ur3', 40000, pool, seed=2)
    U = []
    for g in dr:
        so = C.stats(C.to_int_docs(g['docs']))
        o = A.abc(so, uth, us, ud)
        flat = [w for x in g['docs'] for w in x]; rng.shuffle(flat); it = iter(flat)
        sh = [[next(it) for _ in x] for x in g['docs']]
        osh = A.abc(C.stats(C.to_int_docs(sh)), uth, us, ud)
        U.append(dict(site=g['site'], win=g['win'], logR=o['logR'], R_eff=o['R_eff'], shuf_logR=osh['logR'],
                      shuf_R_eff=osh['R_eff'], contig=float(so[C.SNAMES.index('contig')]), l3l2=float(so[C.SNAMES.index('l3l2')])))
        print('ur3', g['site'], g['win'], [round(x, 2) for x in o['R_eff']], 'shuf', [round(x, 2) for x in osh['R_eff']], flush=True)
    w = [np.log10(u['win']) for u in U]
    res['ur3'] = U
    res['ur3_spearman_Reff'] = spearmanr(w, [u['R_eff'][0] for u in U])
    res['ur3_spearman_logR'] = spearmanr(w, [u['logR'][0] for u in U])
    res['ur3_spearman_shuf'] = spearmanr(w, [u['shuf_R_eff'][0] for u in U])
    res['ur3_real_minus_shuf'] = float(np.mean([u['R_eff'][0] - u['shuf_R_eff'][0] for u in U]))
    for k in ('ur3_spearman_Reff', 'ur3_spearman_logR', 'ur3_spearman_shuf'):
        res[k] = [float(res[k][0]), float(res[k][1])]; print(k, res[k])
    json.dump(res, open(os.path.join(A.CK, 'c1.json'), 'w'), indent=1, default=float)
