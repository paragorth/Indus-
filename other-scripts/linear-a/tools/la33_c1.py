#!/usr/bin/env python3
"""LA-33 cycle 1: archive-level regime classifier trained on simulated archives, checked on
planted archives, Ur III (rations vs deliveries), Linear B (personnel/census vs disbursement
series), shuffles; then applied to Linear A groups (site, scribe, commodity)."""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from la33_common import *

rng = np.random.default_rng(33)
out = {}


def load(name):
    d = json.load(open(os.path.join(CK, name + '.json')))
    R = d['rows']
    X = np.array([r[3:] for r in R]); y = np.array([r[0] for r in R]); w = np.array([r[2] for r in R])
    return X, y, w


X, y, w = load('sims_c1'); Xt, yt, wt = load('sims_c1_test')
clf = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=2, random_state=0).fit(X, y)
reg = RandomForestRegressor(400, min_samples_leaf=3, n_jobs=2, random_state=0).fit(X, w)
cls = list(clf.classes_)
P = clf.predict_proba(Xt)
m = yt != 'M'
auc_sp = roc_auc_score(yt[m] == 'S', P[m, cls.index('S')])
acc3 = np.mean(clf.predict(Xt) == yt)
wh = reg.predict(Xt)
out['planted'] = {'auc_S_vs_P': auc_sp, 'acc3': acc3, 'w_mae': float(np.mean(np.abs(wh - wt))),
                  'w_corr': float(np.corrcoef(wh, wt)[0, 1]),
                  'imp': dict(zip(FEAT, np.round(reg.feature_importances_, 3).tolist()))}
print('planted', out['planted'], flush=True)


def score(docs):
    f = feats(docs, rng)[None]
    p = clf.predict_proba(f)[0]
    return float(reg.predict(f)[0]), dict(zip(cls, p.round(3).tolist())), f[0]


def sub_scores(docs, T, reps):
    res = []
    for _ in range(reps):
        idx = rng.choice(len(docs), size=min(T, len(docs)), replace=False)
        res.append(score([docs[i] for i in idx])[0])
    return res

# ------------------------------------------------------------------ Ur III
U = ur_docs()
G = defaultdict(list)
for d in U: G[(d['site'], d['kind'])].append(d['ents'])
ur = {}
for T in (30, 60, 150):
    sc = defaultdict(list)
    for (site, kind), docs in G.items():
        if len(docs) < max(40, T) or site in ('uncertain', ''): continue
        sc[(site, kind)] = sub_scores(docs, T, 25)
    lab = [k[1] == 'S' for k, v in sc.items() for _ in v]; val = [x for v in sc.values() for x in v]
    auc = roc_auc_score(lab, val)
    # label-shuffle control within site
    sh = []
    for rep in range(10):
        lab2, val2 = [], []
        for site in set(k[0] for k in sc):
            if (site, 'S') not in sc or (site, 'D') not in sc: continue
            pool = G[(site, 'S')] + G[(site, 'D')]
            nS = len(G[(site, 'S')])
            perm = rng.permutation(len(pool))
            gs = [pool[i] for i in perm[:nS]]; gd = [pool[i] for i in perm[nS:]]
            for _ in range(5):
                for g, l in ((gs, True), (gd, False)):
                    idx = rng.choice(len(g), size=min(T, len(g)), replace=False)
                    lab2.append(l); val2.append(score([g[i] for i in idx])[0])
        sh.append(roc_auc_score(lab2, val2))
    # entry shuffle inside each group
    es = []
    for (site, kind), docs in sc.items():
        g = G[(site, kind)]
        idx = rng.choice(len(g), size=min(T, len(g)), replace=False)
        es.append((kind, score(shuffle_entries([list(g[i]) for i in idx], random.Random(int(rng.integers(1e9)))))[0]))
    ur[T] = {'auc_S_vs_D': auc, 'label_shuffle_auc': [float(np.mean(sh)), float(np.max(sh))],
             'groups': {f'{k[0]}|{k[1]}': [float(np.mean(v)), float(np.std(v))] for k, v in sc.items()},
             'entry_shuffled': es}
    print('UR', T, round(auc, 3), 'shuffle', np.round([np.mean(sh), np.max(sh)], 3), flush=True)
    for k, v in sorted(sc.items()): print('   ', k, round(np.mean(v), 3))
out['ur'] = ur

# raw features (no simulator): matched T = 60
raw = defaultdict(list)
for (site, kind), docs in G.items():
    if len(docs) < 60 or site in ('uncertain', ''): continue
    for _ in range(10):
        idx = rng.choice(len(docs), size=60, replace=False)
        raw[kind].append(feats([docs[i] for i in idx], rng))
out['ur_raw_feats'] = {k: dict(zip(FEAT, np.mean(v, 0).round(3).tolist())) for k, v in raw.items()}
print('UR raw', out['ur_raw_feats'], flush=True)

# ------------------------------------------------------------------ Linear B
B = lb_docs()
GB = defaultdict(list)
for d in B:
    if d['kind'] != '?': GB[(d['series'], d['kind'])].append(d['ents'])
lb = {}
for k, docs in GB.items():
    if len(docs) < 8: continue
    lb[k] = sub_scores(docs, min(40, len(docs)), 15 if len(docs) > 40 else 1)
lab = [k[1] == 'S' for k, v in lb.items() for _ in v]; val = [x for v in lb.values() for x in v]
lbm = {k: float(np.mean(v)) for k, v in lb.items()}
auc_lb = roc_auc_score([k[1] == 'S' for k in lbm], list(lbm.values()))
# label shuffle over series
shl = []
keys = list(lbm)
for _ in range(2000):
    perm = rng.permutation([k[1] == 'S' for k in keys])
    shl.append(roc_auc_score(perm, [lbm[k] for k in keys]))
out['lb'] = {'auc_series': auc_lb, 'p_perm': float(np.mean(np.array(shl) >= auc_lb)),
             'series': {f'{k[0]}|{k[1]}': v for k, v in sorted(lbm.items())}}
print('LB auc', round(auc_lb, 3), 'p', out['lb']['p_perm'])
for k, v in sorted(lbm.items(), key=lambda x: x[1]): print('   ', k, round(v, 3))

# ------------------------------------------------------------------ Linear A
L = la_docs()
groups = {'LA all': [d for d in L], 'HT': [d for d in L if d['site'] == 'HT'],
          'KH': [d for d in L if d['site'] == 'KH'], 'ZA': [d for d in L if d['site'] == 'ZA'],
          'PH': [d for d in L if d['site'] == 'PH']}
for s, c in Counter(d['scribe'] for d in L if d['site'] == 'HT' and d['scribe']).items():
    if c >= 8: groups['HT ' + s] = [d for d in L if d['scribe'] == s]
for c, n in Counter(d['com'] for d in L).items():
    if n >= 7: groups['com ' + c] = [d for d in L if d['com'] == c]
la = {}
for g, ds in groups.items():
    docs = [d['ents'] for d in ds]
    if len(docs) < 8: continue
    wfull, pr, f = score(docs)
    subs = sub_scores(docs, int(0.8 * len(docs)), 30)
    halves = []
    for _ in range(20):
        perm = rng.permutation(len(docs)); h = len(docs) // 2
        halves.append((score([docs[i] for i in perm[:h]])[0], score([docs[i] for i in perm[h:]])[0]))
    shuf = [score(shuffle_entries([list(x) for x in docs], random.Random(i)))[0] for i in range(10)]
    la[g] = {'T': len(docs), 'w': wfull, 'proba': pr, 'sub80': [float(np.percentile(subs, 5)), float(np.percentile(subs, 95))],
             'half_diff': float(np.mean([abs(a - b) for a, b in halves])), 'entry_shuffled_w': float(np.mean(shuf)),
             'feat': dict(zip(FEAT, f.round(3).tolist()))}
    print('LA', g, la[g]['T'], round(wfull, 3), pr, np.round(la[g]['sub80'], 3), 'shuf', round(np.mean(shuf), 3), flush=True)
out['la'] = la
json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1, default=float)
