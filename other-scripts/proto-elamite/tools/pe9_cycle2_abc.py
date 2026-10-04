"""pe9 cycle 2 analysis: rejection ABC on the reference tables of pe9_cycle2.py.

For each observed corpus: distance = Euclidean on MAD-scaled summary statistics,
accept the nearest 0.1% of the table.  Reports model posterior (share of accepted
sims per mode), bag size K / copies c / eps / session S posteriors, and a
posterior-predictive check (observed stat vs 5-95% of accepted sims).
Controls: (a) 300 pseudo-observed sims drawn from the table, leave-one-out ->
model confusion and K recovery; (b) planted corpora (LM, POLYA, BAG_WOR K 12 c 2)
and PE middles shuffled across tablets, scored against the PE table;
(c) Ur III and Linear B against their own tables.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe9_common import *  # noqa
from pe9_cycle1 import pe_fit_tablets, shuffle_across, planted  # noqa
from pe9_cycle2 import load_table  # noqa

ACC = 0.001


def abc(P, S, obs, scale, excl=None):
    d = np.sqrt((((S - obs) / scale) ** 2).sum(1))
    if excl is not None:
        d[excl] = np.inf
    n = max(50, int(ACC * len(d)))
    idx = np.argpartition(d, n)[:n]
    A = P[idx]
    out = {'model_post': {MODES[m]: float(np.mean(A[:, 0] == m)) for m in range(5)},
           'base_bigram': float(np.mean(A[:, 1] == 1)), 'dist_max': float(d[idx].max())}
    for m in (2, 3, 4):
        B = A[A[:, 0] == m]
        if len(B) >= 10:
            out[MODES[m] + '_K'] = np.percentile(B[:, 3], [5, 50, 95]).round(1).tolist()
            out[MODES[m] + '_eps'] = np.percentile(B[:, 5], [5, 50, 95]).round(2).tolist()
            if m == 3:
                out['BAG_WOR_c'] = {str(int(c)): float(np.mean(B[:, 4] == c)) for c in (1, 2, 3, 5, 10)}
    B = A[A[:, 0] == 1]
    if len(B) >= 10:
        out['POLYA_theta'] = np.percentile(B[:, 6], [5, 50, 95]).round(2).tolist()
    out['S_post'] = {str(int(s)): float(np.mean(A[:, 7] == s)) for s in (1, 2, 4, 8)}
    lo, hi = np.percentile(S[idx], 5, 0), np.percentile(S[idx], 95, 0)
    out['ppc_outside'] = [STAT_NAMES[k] + ' obs %.3f vs %.3f-%.3f' % (obs[k], lo[k], hi[k])
                          for k in range(len(obs)) if not lo[k] <= obs[k] <= hi[k]]
    return out, idx


def obs_stats(tabs, vocab=None):
    E = Enc(tabs, vocab)
    return stats(E.tok, E.ss, E.ts, E.V, 15, 0)


if __name__ == '__main__':
    C = load_corpora()
    res = {}
    fit = pe_fit_tablets(C)[0]
    vocab = Enc(fit).vocab
    for name in ('PE', 'UR3', 'LINB'):
        try:
            P, S = load_table(name)
        except ValueError:
            continue
        ok = np.isfinite(S).all(1)
        P, S = P[ok], S[ok]
        scale = np.median(np.abs(S - np.median(S, 0)), 0) * 1.4826 + 1e-9
        tabs = fit if name == 'PE' else C[name]['tablets']
        obs = {name: obs_stats(tabs)}
        if name == 'PE':
            for s in range(3):
                obs['PE_SHUF%d' % s] = obs_stats(shuffle_across(fit, s))
            for kind in ('LM', 'POLYA', 'BAG'):
                for s in range(2):
                    obs['PLANT_%s%d' % (kind, s)] = obs_stats(planted(fit, kind, 100 + s))
        for k, o in obs.items():
            r, _ = abc(P, S, o, scale)
            r['n_table'] = int(len(P))
            r['obs'] = dict(zip(STAT_NAMES, np.round(o, 4).tolist()))
            res[k] = r
            print(k, json.dumps({x: r[x] for x in r if x not in ('obs',)}), flush=True)
        # leave-one-out recovery on the table
        rng = np.random.RandomState(7)
        pick = rng.choice(len(P), 300, replace=False)
        conf = np.zeros((5, 5))
        kt, kh, ct, ch = [], [], [], []
        for i in pick:
            r, idx = abc(P, S, S[i], scale, excl=np.array([i]))
            mp = max(r['model_post'], key=r['model_post'].get)
            conf[int(P[i, 0]), MODES.index(mp)] += 1
            if P[i, 0] == 3 and P[i, 5] < 0.5:
                B = P[idx][P[idx][:, 0] == 3]
                if len(B) >= 5:
                    kt.append(np.log(P[i, 3])); kh.append(np.median(np.log(B[:, 3])))
                    ct.append(P[i, 4]); ch.append(np.median(B[:, 4]))
        res[name + '_LOO'] = {'confusion_rows_true_cols_map': conf.astype(int).tolist(),
                              'acc': float(np.trace(conf) / conf.sum()),
                              'BAG_WOR_logK_corr': float(np.corrcoef(kt, kh)[0, 1]) if len(kt) > 3 else None,
                              'BAG_WOR_c_corr': float(np.corrcoef(ct, ch)[0, 1]) if len(ct) > 3 else None,
                              'n_bag_eps_lt_half': len(kt)}
        print(name + '_LOO', json.dumps(res[name + '_LOO']), flush=True)
    json.dump(res, open(os.path.join(DATA, 'pe9_cycle2.json'), 'w'), indent=1)
