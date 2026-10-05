"""pe49: supervised ceiling -- how much scribe identity the habit features carry at all
(cross-validated logistic regression on habits; content features for comparison;
chance from label-permuted fits)."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pe49_common as P
from pe49_cycle1 import features, plant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def acc(X, y, rng, perm=False):
    if perm:
        y = rng.permutation(y)
    return float(cross_val_score(LogisticRegression(C=0.3, max_iter=2000), X, y, cv=5).mean())


rng = np.random.default_rng(0)
out = {}
for which in ('ur3', 'plant', 'plant0'):
    if which == 'ur3':
        T, _ = P.load_ur3(); y = np.unique([t['group'] for t in T], return_inverse=True)[1]
    else:
        T, y, _ = plant(P.load_pe(), np.random.default_rng(5 if which == 'plant' else 6), follow=0.6 if which == 'plant' else 0.0)
    R, names, strata = features(T)
    X = np.nan_to_num(R)
    Xc, _ = P.content_matrix(T)
    r = {'habit': acc(X, y, rng), 'habit_perm': [acc(X, y, rng, True) for _ in range(3)],
         'content': acc(Xc, y, rng), 'chance': float(np.bincount(y).max() / len(y))}
    out[which] = r
    print(which, r, flush=True)
json.dump(out, open(os.path.join(P.CK, 'ceiling.json'), 'w'))
