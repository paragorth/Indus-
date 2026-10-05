"""pe44: SUPERVISED upper bound on Ur III.  How much does one quantity's fingerprint (FEATS minus raw
size) say about its kind when the labels ARE used?  Logistic regression trained on tablet half A,
tested on half B, both sides size-matched in log2 bins and restricted to one system.
Control: labels shuffled within half A before training."""
import json, math, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa
from pe44_cycle1 import build, KIND

u = ur3_tabs()
tabs = {t: d['recs'] for t, d in u.items()}
X, meta = corpus_matrix(tabs)
lab = np.array([tabs[t][i]['lab'] or '' for t, i in meta])
sysa = np.array([tabs[t][i]['sys'] for t, i in meta])
import random
rr = random.Random(1)
half = {t: rr.random() < 0.5 for t in sorted(tabs)}
inA = np.array([half[t] for t, i in meta])
cols = [FEATS.index(f) for f in FEATS if f != 'LMAG']
lv = np.floor(X[:, FEATS.index('LMAG')] / math.log(2))
rng = np.random.default_rng(2)


def matched(idxP, idxN):
    pos, neg = [], []
    for b in np.unique(lv[np.concatenate([idxP, idxN])]):
        P = idxP[lv[idxP] == b]; N = idxN[lv[idxN] == b]
        n = min(len(P), len(N))
        if n:
            pos += list(rng.choice(P, n, replace=False)); neg += list(rng.choice(N, n, replace=False))
    return np.array(pos, int), np.array(neg, int)


tests = [('PLANNED(cap) vs GRAIN', ['RATION', 'DEBIT', 'ESTIMATE'], ['GRAIN'], 'UR_CAP'),
         ('RATION vs GRAIN', ['RATION'], ['GRAIN'], 'UR_CAP'),
         ('DEBIT(cap) vs GRAIN', ['DEBIT'], ['GRAIN'], 'UR_CAP'),
         ('ESTIMATE vs GRAIN', ['ESTIMATE'], ['GRAIN'], 'UR_CAP'),
         ('DEBIT(cnt) vs LIVESTOCK+PEOPLE', ['DEBIT'], ['LIVESTOCK', 'PEOPLE'], 'UR_CNT'),
         ('DEFICIT(cnt) vs LIVESTOCK+PEOPLE', ['DEFICIT'], ['LIVESTOCK', 'PEOPLE'], 'UR_CNT'),
         ('DEFICIT(cap) vs GRAIN', ['DEFICIT'], ['GRAIN'], 'UR_CAP')]
res = []
for name, P, N, s in tests:
    out = {'test': name}
    for side, mask in (('A', inA), ('B', ~inA)):
        iP = np.where(mask & (sysa == s) & np.isin(lab, P))[0]
        iN = np.where(mask & (sysa == s) & np.isin(lab, N))[0]
        out[side] = matched(iP, iN)
    (pA, nA), (pB, nB) = out.pop('A'), out.pop('B')
    if min(len(pA), len(pB)) < 30:
        res.append({'test': name, 'n': [len(pA), len(pB)], 'skip': True}); print(res[-1]); continue
    XA = X[np.concatenate([pA, nA])][:, cols]; yA = np.r_[np.ones(len(pA)), np.zeros(len(nA))]
    XB = X[np.concatenate([pB, nB])][:, cols]; yB = np.r_[np.ones(len(pB)), np.zeros(len(nB))]
    clf = LogisticRegression(max_iter=2000).fit(XA, yA)
    pb = clf.predict_proba(XB)[:, 1]
    a = auc(pb[yB == 1], pb[yB == 0])
    sh = []
    for _ in range(20):
        c2 = LogisticRegression(max_iter=2000).fit(XA, rng.permutation(yA))
        q = c2.predict_proba(XB)[:, 1]
        sh.append(auc(q[yB == 1], q[yB == 0]))
    coef = dict(zip([FEATS[c] for c in cols], np.round(clf.coef_[0], 2).tolist()))
    res.append({'test': name, 'nA': int(len(pA)), 'nB': int(len(pB)), 'auc': a, 'shuf_mean': float(np.mean(sh)),
                'shuf_max': float(np.max(sh)), 'coef': coef})
    print(res[-1], flush=True)
json.dump(res, open(os.path.join(CK, 'sup.json'), 'w'), indent=1)
