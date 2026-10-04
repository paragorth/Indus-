#!/usr/bin/env python3
"""LA-33 cycle 2: tablet-level regime and word regimes.
A per-tablet classifier (features of the tablet inside its archive) is trained on mixed simulated
archives where each tablet's regime is known; checked on Ur III archives of mixed kinds
(site pools of ration and delivery texts) and on entry-shuffled archives; then applied to
Linear A tablets. Words are scored by the regime of the tablets they are written on;
a permutation test asks whether words 'live' in one regime; Ur III checks whether demand
words are more often places ({ki}) or gods ({d}) than scheduled words."""
import json, os
from collections import Counter, defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from la33_common import *

rng = np.random.default_rng(332)
TF = ['logn', 'mode', 'cv', 'recur', 'maxjacc', 'consist', 'ones', 'relamt', 'nrel']


def tab_feats(docs):
    """per-tablet features inside an archive (list of entry lists)."""
    occ = defaultdict(list)
    for i, d in enumerate(docs):
        for r, x, _ in d: occ[r].append((i, x))
    sets = [set(r for r, _, _ in d) for d in docs]
    inv = defaultdict(set)
    for i, s in enumerate(sets):
        for r in s: inv[r].add(i)
    allA = np.array([x for d in docs for _, x, _ in d])
    med = np.median(allA); mn = np.mean([len(d) for d in docs])
    F = []
    for i, d in enumerate(docs):
        a = np.array([x for _, x, _ in d])
        mode = Counter(a).most_common(1)[0][1] / len(a) if len(a) > 1 else np.nan
        cv = a.std() / a.mean() if len(a) > 1 else np.nan
        rec = np.mean([len(inv[r]) >= 2 for r, _, _ in d])
        nb = Counter(j for r in sets[i] for j in inv[r] if j != i)
        mj = max((c / len(sets[i] | sets[j]) for j, c in nb.items()), default=0.0)
        cons = [any(abs(y - x) < 1e-9 for j, y in occ[r] if j != i) for r, x, _ in d if len(occ[r]) > 1]
        F.append([math.log(len(d)), mode, cv, rec, mj, np.mean(cons) if cons else np.nan,
                  np.mean(a == 1), math.log(np.median(a) / med), math.log(len(d) / mn)])
    F = np.array(F, float)
    return np.where(np.isnan(F), -1.0, F)


# ---------------------------------------------------------------- training on mixed simulated archives
fn = os.path.join(CK, 'c2_train.json')
if os.path.exists(fn):
    d = json.load(open(fn)); XT, yT = np.array(d['X']), np.array(d['y'])
else:
    XT, yT = [], []
    for k in range(1500):
        T = int(math.exp(rng.uniform(math.log(20), math.log(250))))
        docs, lab, w = sim_archive(T, 'M', rng, w=rng.uniform(0.1, 0.9))
        XT.append(tab_feats(docs)); yT.extend(lab)
    XT = np.vstack(XT); yT = np.array(yT)
    json.dump({'X': XT.tolist(), 'y': yT.tolist()}, open(fn, 'w'))
clf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=2, random_state=1).fit(XT, yT)
out = {'imp': dict(zip(TF, clf.feature_importances_.round(3).tolist()))}
# planted held-out
pl = []
for k in range(150):
    docs, lab, w = sim_archive(int(math.exp(rng.uniform(math.log(20), math.log(250)))), 'M', rng, w=rng.uniform(0.1, 0.9))
    p = clf.predict_proba(tab_feats(docs))[:, 1]
    if 0 < np.mean(lab) < 1: pl.append(roc_auc_score(lab, p))
out['planted_auc'] = float(np.mean(pl))
print('planted tablet AUC', round(np.mean(pl), 3), flush=True)

# ---------------------------------------------------------------- Ur III mixed archives
U = ur_docs()
bysite = defaultdict(list)
for d in U: bysite[d['site']].append(d)
ur = {}
for site, ds in bysite.items():
    S = [d for d in ds if d['kind'] == 'S']; Dd = [d for d in ds if d['kind'] == 'D']
    if len(S) < 40 or len(Dd) < 40: continue
    aucs, sh, aucs1 = [], [], []
    for rep in range(10):
        pick = [S[i] for i in rng.choice(len(S), 40, replace=False)] + [Dd[i] for i in rng.choice(len(Dd), 80, replace=False)]
        docs = [p['ents'] for p in pick]; lab = np.array([p['kind'] == 'S' for p in pick])
        p = clf.predict_proba(tab_feats(docs))[:, 1]
        aucs.append(roc_auc_score(lab, p))
        multi = np.array([len(x) >= 2 for x in docs])
        if lab[multi].any() and (~lab[multi]).any(): aucs1.append(roc_auc_score(lab[multi], p[multi]))
        sd = shuffle_entries([list(x) for x in docs], random.Random(rep))
        sh.append(roc_auc_score(lab, clf.predict_proba(tab_feats(sd))[:, 1]))
    ur[site] = {'auc': float(np.mean(aucs)), 'auc_multi_entry': float(np.mean(aucs1)) if aucs1 else None,
                'entry_shuffled_auc': float(np.mean(sh))}
    print('UR', site, ur[site], flush=True)
out['ur'] = ur

# Ur III word classes by regime (known kinds): place {ki} / god {d} share among recipients
cls = {'S': Counter(), 'D': Counter()}
for d in U:
    for r, _, _ in d['ents']:
        c = 'GN' if '{ki}' in r else ('DN' if r.startswith('{d}') else 'other')
        cls[d['kind']][c] += 1
out['ur_wordclass'] = {k: {c: v / sum(cnt.values()) for c, v in cnt.items()} for k, cnt in cls.items()}
print('UR word classes', out['ur_wordclass'])
# Ur III: do recipient words live in one regime? (share of words on >=2 texts that are on only one kind)
occk = defaultdict(set)
for d in U:
    for r, _, _ in d['ents']: occk[(d['site'], r)].add(d['kind'])
multi = [v for v in occk.values()]
out['ur_switch_share'] = float(np.mean([len(v) == 2 for v in multi]))

# ---------------------------------------------------------------- Linear A
L = la_docs()
arch = defaultdict(list)
for d in L: arch[d['site'] if d['site'] in ('HT', 'KH', 'ZA') else 'other'].append(d)
tabs = []
for site, ds in arch.items():
    docs = [d['ents'] for d in ds]
    F = tab_feats(docs)
    p = clf.predict_proba(F)[:, 1]
    # entry-shuffled reference for each tablet (mean over 20 shuffles)
    ps = np.zeros(len(docs))
    for r in range(20):
        ps += clf.predict_proba(tab_feats(shuffle_entries([list(x) for x in docs], random.Random(r))))[:, 1] / 20
    for d, pi, si, f in zip(ds, p, ps, F):
        tabs.append({'id': d['id'], 'site': site, 'com': d['com'], 'n': len(d['ents']), 'p': float(pi), 'p_shuf': float(si),
                     'words': [e[0] for e in d['ents']], 'amts': [e[1] for e in d['ents']]})
P = np.array([t['p'] for t in tabs]); PS = np.array([t['p_shuf'] for t in tabs])
out['la_tab'] = {'mean_p': float(P.mean()), 'mean_p_shuf': float(PS.mean()), 'frac_gt_.5': float(np.mean(P > .5)),
                 'frac_gt_.5_shuf': float(np.mean(PS > .5)), 'sd_p': float(P.std()), 'sd_p_shuf': float(PS.std())}
print('LA tablets', out['la_tab'], flush=True)
bycom = defaultdict(list)
for t in tabs: bycom[t['com']].append(t['p'])
out['la_com'] = {c: [len(v), float(np.mean(v))] for c, v in bycom.items() if len(v) >= 4}
bysite2 = defaultdict(list)
for t in tabs: bysite2[t['site']].append(t['p'])
out['la_site'] = {c: [len(v), float(np.mean(v))] for c, v in bysite2.items()}
print('LA by commodity', out['la_com']); print('LA by site', out['la_site'])

# word regimes: words on >= 2 tablets
wt = defaultdict(list)
for i, t in enumerate(tabs):
    for w_ in set(t['words']): wt[w_].append(i)
W = {w_: ix for w_, ix in wt.items() if len(ix) >= 2}


def wvar(Pv):
    """mean within-word variance of tablet scores (low = words live in one regime)"""
    return float(np.mean([np.var(Pv[ix]) for ix in W.values()]))


obs = wvar(P)
null = []
for r in range(2000):
    perm = rng.permutation(len(P))
    # permute scores among tablets of the same site
    Pp = P.copy()
    for site in set(t['site'] for t in tabs):
        ix = [i for i, t in enumerate(tabs) if t['site'] == site]
        Pp[ix] = P[rng.permutation(ix)]
    null.append(wvar(Pp))
out['la_word_cohesion'] = {'n_words': len(W), 'obs_within_var': obs, 'null_mean': float(np.mean(null)),
                           'p': float(np.mean(np.array(null) <= obs))}
print('word cohesion', out['la_word_cohesion'], flush=True)
words = []
for w_, ix in W.items():
    ps = P[ix]
    words.append({'w': w_, 'k': len(ix), 'mean': float(ps.mean()), 'min': float(ps.min()), 'max': float(ps.max()),
                  'switch': bool(ps.min() < 0.3 and ps.max() > 0.7), 'tabs': [tabs[i]['id'] for i in ix]})
words.sort(key=lambda x: -x['mean'])
out['la_words'] = words
out['la_tabs'] = tabs
json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1, default=float)
for x in words: print('  %-16s k=%d mean %.2f [%.2f-%.2f]%s' % (x['w'], x['k'], x['mean'], x['min'], x['max'], ' SWITCH' if x['switch'] else ''))
