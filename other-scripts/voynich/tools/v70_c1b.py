"""v70 cycle 1b: domain-free comparison. The cycle-1 classifier is fitted on synthetic-hand output; real scans
shift every alphabet (k-means and random alike). Here the score is the PAIRED difference in classifier logit
between a k-means ink alphabet and the random-Voronoi alphabet at the same (theta, K, seed, merges), on held-out
page-half B, so scan domain, alphabet size and merges cancel. Also per-feature paired differences."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import CK, BFEATS
from v70_c1_report import fv, LANG, GENS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
data = {n: json.load(open(os.path.join(CK, f'c1_{n}.json'))) for n in LANG + GENS + ['V', 'L', 'SV']}
X, y, g = [], [], []
for n in LANG + GENS:
    for r in data[n]['rows']:
        for h in 'AB':
            X.append(fv(r[h])); y.append(n in LANG); g.append(n)
X, y, g = np.array(X), np.array(y), np.array(g)
out = {}
for n in data:
    tr = g != n
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight='balanced')).fit(X[tr], y[tr])
    rows = data[n]['rows']
    key = lambda r: (r['theta'], r['K'], r['seed'], r['m'])
    km = {key(r): r for r in rows if r['kind'] == 'km'}; rn = {key(r): r for r in rows if r['kind'] == 'rnd'}
    ks = sorted(set(km) & set(rn))
    d = clf.decision_function(np.array([fv(km[k]['B']) for k in ks])) - clf.decision_function(np.array([fv(rn[k]['B']) for k in ks]))
    feats = {f: float(np.median([km[k]['B'][f] - rn[k]['B'][f] for k in ks])) for f in BFEATS + ['keff']}
    out[n] = {'dlogit_med': float(np.median(d)), 'frac_pos': float(np.mean(d > 0)), 'n': len(ks), 'dfeat': feats}
    print(n, round(out[n]['dlogit_med'], 2), round(out[n]['frac_pos'], 2), {f: round(v, 3) for f, v in feats.items()})
json.dump(out, open(os.path.join(CK, 'c1b.json'), 'w'), indent=1)
