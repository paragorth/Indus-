"""v75 cycle 3 report: rejection ABC over random designation grammars (see v75_c3.py)."""
import os, sys, pickle, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X

D = pickle.load(open(os.path.join(X.CK, 'c3_sims.pkl'), 'rb'))
TF = D['TYPEF']; FI = [TF.index(f) for f in D['FIT']]; HI = [TF.index(f) for f in D['HOLD']]
P = [p for p, _ in D['sims']]; S = np.array([s for _, s in D['sims']])
ok = np.isfinite(S).all(1); P = [p for p, o in zip(P, ok) if o]; S = S[ok]
med = np.median(S, 0); mad = np.median(np.abs(S - med), 0) * 1.4826 + 1e-6


def par(p):
    f = p['fields']
    return dict(F=p['F'], logV=float(np.log10(sum(x['V'] for x in f))), logV1=float(np.log10(f[0]['V'])),
                s=float(np.mean([x['s'] for x in f])), rho=p['rho'], c=p['c'], d=p['d'],
                a=float(np.mean([x['a'] for x in f])), popt=float(np.mean([x['p'] for x in f])))


PP = [par(p) for p in P]
KEYS = list(PP[0])
NACC = max(50, len(S) // 100)


def abc(t, exclude=None):
    d = np.sqrt((((S[:, FI] - t[FI]) / mad[FI]) ** 2).sum(1))
    if exclude is not None: d[exclude] = np.inf
    acc = np.argsort(d)[:NACC]
    hold = np.abs(np.median(S[acc][:, HI], 0) - t[HI]) / mad[HI]
    rng = np.random.default_rng(0); pri = rng.choice(len(S), NACC, replace=False)
    hold0 = np.abs(np.median(S[pri][:, HI], 0) - t[HI]) / mad[HI]
    fitd = float(np.median(d[acc]))
    post = {k: np.array([PP[i][k] for i in acc]) for k in KEYS}
    return acc, dict(fit_dist=fitd, hold_err=float(hold.mean()), hold_err_prior=float(hold0.mean()),
                     hold_by=dict(zip(D['HOLD'], [round(float(x), 2) for x in hold])),
                     post={k: (round(float(np.median(v)), 3), round(float(np.quantile(v, .1)), 3), round(float(np.quantile(v, .9)), 3)) for k, v in post.items()},
                     F_dist={int(f): round(float(np.mean(post['F'] == f)), 3) for f in range(1, 7)})


out = {}
# calibration: planted grammars (truth known), also leave-one-out on 100 random sims
cal = []
for p, s in D['planted'] + [D['sims'][i] for i in range(0, 2000, 20)]:
    s = np.array(s)
    if not np.isfinite(s).all(): continue
    acc, r = abc(s)
    tp = par(p)
    cov = {k: int(r['post'][k][1] <= tp[k] <= r['post'][k][2]) for k in KEYS}
    cal.append(dict(r=r, true=tp, cov=cov))
out['calibration'] = dict(n=len(cal), hold_err=float(np.median([c['r']['hold_err'] for c in cal])),
                          hold_err_q90=float(np.quantile([c['r']['hold_err'] for c in cal], .9)),
                          fit_dist_q90=float(np.quantile([c['r']['fit_dist'] for c in cal], .9)),
                          coverage={k: float(np.mean([c['cov'][k] for c in cal])) for k in KEYS},
                          recov_corr={k: float(np.corrcoef([c['true'][k] for c in cal], [c['r']['post'][k][0] for c in cal])[0, 1]) for k in KEYS})
for name, t in D['targets'].items():
    out[name] = abc(np.array(t))[1]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(X.CK, 'c3_report.json'), 'w'), indent=1)
