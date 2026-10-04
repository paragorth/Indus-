"""pe9 cycle 1b: the 'use profile' -- after a sign has been used k times on a tablet,
is it boosted (inexhaustible bag / tablet topic) or suppressed (depleted token)?
Model pe9_common.score_profile: r = f1, f2, f3 after 1, 2, >=3 earlier uses,
fw inside the current string; base = bigram (a = 0.5) fitted on training folds.
Variants: FREE (f1,f2,f3,fw free), FLAT (f1=f2=f3, fw free: inexhaustible),
LM (all 1).  5-fold CV (contiguous blocks), Nelder-Mead on training ll, session S
in {1,2,4} chosen on training.  Same corpora as cycle 1.  Bootstrap over tablets
for the f-profile (200x, full data).
"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe9_common import *  # noqa
from pe9_cycle1 import pe_fit_tablets, shuffle_across, planted  # noqa


def fit(E, tr, variant, a=0.5):
    w, B = fit_base(E.tok, E.ss, E.ts, tr, E.V, a)
    best = None
    for S in (1, 2, 4):
        sess = sess_flags(E.T, S)

        def nll(x):
            if variant == 'FLAT':
                f = (np.exp(x[0]),) * 3 + (np.exp(x[1]),)
            else:
                f = tuple(np.exp(x))
            return -score_profile(E.tok, E.ss, E.ts, tr, sess, E.V, w, B, 1, *f).sum()
        x0 = np.zeros(2 if variant == 'FLAT' else 4)
        r = minimize(nll, x0, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 2000})
        if best is None or r.fun < best[0]:
            xf = r.x
            f = (np.exp(xf[0]),) * 3 + (np.exp(xf[1]),) if variant == 'FLAT' else tuple(np.exp(xf))
            best = (r.fun, S, f)
    return best, w, B


def cv(tabs, nfold=5):
    E = Enc(tabs)
    folds = np.array_split(np.arange(E.T), nfold)
    res = {}
    for variant in ('LM', 'FLAT', 'FREE'):
        tot, pars = 0.0, []
        for f in folds:
            tr = np.array(sorted(set(range(E.T)) - set(f.tolist())), np.int64)
            te = np.array(f, np.int64)
            if variant == 'LM':
                w, B = fit_base(E.tok, E.ss, E.ts, tr, E.V, 0.5)
                S, fv = 1, (1., 1., 1., 1.)
            else:
                (_, S, fv), w, B = fit(E, tr, variant)
            tot += score_profile(E.tok, E.ss, E.ts, te, sess_flags(E.T, S), E.V, w, B, 1, *fv).sum()
            pars.append({'S': S, 'f': [float(x) for x in fv]})
        res[variant] = {'bits_per_sign': -tot / LN2 / len(E.tok), 'pars': pars}
    # full-data FREE fit + tablet bootstrap of the profile (S fixed at full-data best)
    allt = np.arange(E.T, dtype=np.int64)
    (_, S, fv), _, _ = fit(E, allt, 'FREE')
    res['FULL'] = {'S': S, 'f': [float(x) for x in fv]}
    rng = np.random.RandomState(5)
    boots = []
    for b in range(200):
        idx = rng.randint(E.T, size=E.T)
        bt = Enc([tabs[i] for i in idx], vocab=E.vocab)
        w, B = fit_base(bt.tok, bt.ss, bt.ts, np.arange(bt.T, dtype=np.int64), bt.V, 0.5)
        sess = sess_flags(bt.T, 1)
        r = minimize(lambda x: -score_profile(bt.tok, bt.ss, bt.ts, np.arange(bt.T, dtype=np.int64), sess,
                                              bt.V, w, B, 1, *np.exp(x)).sum(),
                     np.log(np.array(fv)), method='Nelder-Mead', options={'xatol': 1e-2, 'fatol': 1e-2})
        boots.append([float(v) for v in np.exp(r.x)])
    bo = np.array(boots)
    res['BOOT'] = {'q05': np.percentile(bo, 5, 0).tolist(), 'q50': np.percentile(bo, 50, 0).tolist(),
                   'q95': np.percentile(bo, 95, 0).tolist(),
                   'P_f2_lt_f1': float(np.mean(bo[:, 1] < bo[:, 0])),
                   'P_f3_lt_f2': float(np.mean(bo[:, 2] < bo[:, 1]))}
    return res


def job(arg):
    name, tabs = arg
    ck = os.path.join(CKPT, 'c1b_' + name + '.json')
    if os.path.exists(ck):
        return name, json.load(open(ck))
    r = cv(tabs)
    json.dump(r, open(ck, 'w'))
    print(name, {k: round(v['bits_per_sign'], 3) for k, v in r.items() if 'bits_per_sign' in v},
          'f', [round(x, 2) for x in r['FULL']['f']], 'S', r['FULL']['S'], flush=True)
    return name, r


if __name__ == '__main__':
    C = load_corpora()
    fitt = pe_fit_tablets(C)[0]
    jobs = [('PE', fitt)] + [('PE_SHUF%d' % s, shuffle_across(fitt, s)) for s in range(3)]
    jobs += [('UR3', C['UR3']['tablets']), ('LINB', C['LINB']['tablets'])]
    for kind in ('LM', 'POLYA', 'BAG'):
        for s in range(2):
            jobs.append(('PLANT_%s%d' % (kind, s), planted(fitt, kind, 100 + s)))
    with Pool(2) as pool:
        out = dict(pool.map(job, jobs, chunksize=1))
    json.dump(out, open(os.path.join(DATA, 'pe9_cycle1b.json'), 'w'), indent=1)
