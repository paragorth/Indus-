"""v75: chunk-level test of the designation lean. For the survivors of a run, each Voynich-side chunk gets the share of
survivor models that put it in a designation kind (not GEN, not LANG). Voynich chunks vs each generator's chunks:
mean share, Mann-Whitney U p (one-sided, Voynich higher). Usage: V75_FEATS=... python3 v75_chunkvote.py TAG"""
import os, sys, pickle, json
import numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_cls as C, v75_lib as X

tag = sys.argv[1]
R = pickle.load(open(os.path.join(X.CK, 'run_%s.pkl' % tag), 'rb'))
os.environ['V75_FEATS'] = R.get('featfile', 'feats')
out = {}
for rep in ('A', 'B'):
    f, rows, F = C.load(rep)
    tr = [i for i, r in enumerate(rows) if r['kind'] != '?' and (R['with_gen'] or r['kind'] != 'GEN')]
    te = [i for i, r in enumerate(rows) if r['kind'] == '?']
    share = np.zeros(len(te)); idx = defaultdict(lambda: defaultdict(int))
    for mi in R['surv']:
        m = R['models'][mi]
        fit = C.Fitted(m, F[tr], [rows[i]['kind'] for i in tr], [rows[i]['base'] for i in tr])
        pred, _ = fit.predict(F[te])
        for j, p in enumerate(pred):
            share[j] += p not in ('GEN', 'LANG')
            idx[j][p] += 1
    share /= len(R['surv'])
    grp = defaultdict(list)
    for j, i in enumerate(te):
        r = rows[i]; grp[(r['base'], 'VOY' if r['role'].startswith('voy') else r['role'][5:])].append(share[j])
    for base in sorted({b for b, _ in grp}):
        v = np.array(grp[(base, 'VOY')])
        res = dict(voy=round(float(v.mean()), 3), n=len(v))
        for (b, g), x in grp.items():
            if b != base or g == 'VOY': continue
            p = mannwhitneyu(v, x, alternative='greater').pvalue
            res[g] = (round(float(np.mean(x)), 3), float('%.2g' % p))
        out['%s %s' % (rep, base)] = res
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(X.CK, 'chunkvote_%s.json' % tag), 'w'), indent=1)
