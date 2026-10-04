#!/usr/bin/env python3
"""LA-33 cycle 3: (a) size-matched site comparison with an Ur III same-kind spread null;
(b) an outside prediction: in Ur III, are totals (szu-nigin) more frequent on scheduled texts?
If so, Linear A tablets that the tablet-level model calls schedule-like should carry KU-RO /
PO-TO-KU-RO more often (totals are not used by any feature: totals were removed from entries);
(c) list-length law per group (Poisson vs negative binomial, likelihood ratio) with Ur III as
reference; (d) commodity regime test against site-stratified permutation."""
import json, os, re
from collections import Counter, defaultdict
import numpy as np
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
from la33_common import *

rng = np.random.default_rng(3333)
out = {}
c2 = json.load(open(os.path.join(CK, 'c2.json')))
tabs = c2['la_tabs']

# ---------------------------------------------------------------- (b) totals as an outside prediction
# Ur III: szu-nigin on scheduled vs demand texts (multi-entry texts only)
texts = {}; pid = None; buf = []
for raw in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
    if raw.startswith('&P'):
        if pid: texts[pid] = ''.join(buf)
        pid = raw.split()[0][1:]; buf = []; continue
    if pid: buf.append(raw)
if pid: texts[pid] = ''.join(buf)
U = ur_docs()
tot = defaultdict(list)
for d in U:
    if len(d['ents']) < 2: continue
    tot[d['kind']].append('szu-nigin' in texts.get(d['id'], ''))
out['ur_totals'] = {k: [len(v), float(np.mean(v))] for k, v in tot.items()}
print('UR totals share (multi-entry)', out['ur_totals'], flush=True)
del texts

C = {x['id']: x for x in json.load(open(os.path.join(D, 'corpus.json')))}
has_tot = np.array([any(w in ('KU-RO', 'PO-TO-KU-RO') for w in C[t['id']]['words']) for t in tabs])
P = np.array([t['p'] for t in tabs]); n = np.array([t['n'] for t in tabs])
site = np.array([t['site'] for t in tabs])
auc = roc_auc_score(has_tot, P)
# control 1: permute P within site x list-length band (length predicts totals by itself)
band = np.minimum(n, 6)
null = []
for r in range(5000):
    Pp = P.copy()
    for s in set(site):
        for b in set(band):
            ix = np.where((site == s) & (band == b))[0]
            if len(ix) > 1: Pp[ix] = P[rng.permutation(ix)]
    null.append(roc_auc_score(has_tot, Pp))
null = np.array(null)
# control 2: shuffled-entry scores (from c2) predict totals?
auc_sh = roc_auc_score(has_tot, np.array([t['p_shuf'] for t in tabs]))
out['la_totals'] = {'n_tabs': len(tabs), 'with_total': int(has_tot.sum()), 'auc': auc,
                    'null_mean': float(null.mean()), 'p_upper': float(np.mean(null >= auc)), 'p_lower': float(np.mean(null <= auc)),
                    'auc_shuffled_scores': auc_sh,
                    'share_total_top_third': float(has_tot[P >= np.quantile(P, 2 / 3)].mean()),
                    'share_total_bottom_third': float(has_tot[P <= np.quantile(P, 1 / 3)].mean())}
print('LA totals', out['la_totals'], flush=True)

# ---------------------------------------------------------------- (a) size-matched sites
from la33_c1_model import score_w  # noqa: E402  (shared model helper)
L = la_docs()
bys = defaultdict(list)
for d in L: bys[d['site']].append(d['ents'])
sm = {}
for T in (30, 47):
    r = {}
    for s in ('HT', 'KH', 'ZA'):
        docs = bys[s]
        if len(docs) < T: continue
        r[s] = [score_w([docs[i] for i in rng.choice(len(docs), T, replace=False)], rng) for _ in range(30)]
    sm[T] = {s: [float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))] for s, v in r.items()}
    print('size-matched', T, sm[T], flush=True)
out['la_sites_matched'] = sm
# Ur III: spread between sites of the same kind at T = 47 (how different can same-kind sites be?)
G = defaultdict(list)
for d in U: G[(d['site'], d['kind'])].append(d['ents'])
spread = {}
for kind in 'SD':
    ms = []
    for (s, k), docs in G.items():
        if k != kind or len(docs) < 47 or s in ('uncertain', ''): continue
        ms.append(np.mean([score_w([docs[i] for i in rng.choice(len(docs), 47, replace=False)], rng) for _ in range(10)]))
    spread[kind] = [float(min(ms)), float(max(ms)), len(ms)]
out['ur_same_kind_spread_T47'] = spread
print('UR same-kind site spread', spread, flush=True)

# ---------------------------------------------------------------- (c) list-length law
def nb_lr(nv):
    x = np.asarray(nv) - 1  # lengths >= 1: shift
    lam = x.mean()
    llp = stats.poisson.logpmf(x, lam).sum()
    def nll(th):
        r = math.exp(th); p = r / (r + lam)
        return -stats.nbinom.logpmf(x, r, p).sum()
    res = optimize.minimize_scalar(nll, bounds=(-6, 8), method='bounded')
    return float(2 * (-res.fun - llp)), float(math.exp(res.x)), float(x.var() / max(x.mean(), 1e-9))


ll = {}
for s in ('HT', 'KH', 'ZA'):
    ll['LA ' + s] = nb_lr([len(x) for x in bys[s]])
for (s, k), docs in G.items():
    if len(docs) >= 150 and s in ('Umma', 'Girsu', 'Puzriš-Dagan', 'Ur'):
        idx = rng.choice(len(docs), 150, replace=False)
        ll[f'UR {s} {k}'] = nb_lr([len(docs[i]) for i in idx])
out['length_law'] = ll
for k, v in ll.items(): print('length law', k, 'LR %.1f  NB r %.2f  disp(shifted) %.2f' % v)

# ---------------------------------------------------------------- (d) commodity regime, site-stratified permutation
com = np.array([t['com'] for t in tabs])
res = {}
for c in [c for c, k in Counter(com).items() if k >= 6]:
    obs = P[com == c].mean()
    nl = []
    for r in range(5000):
        Pp = P.copy()
        for s in set(site):
            ix = np.where(site == s)[0]; Pp[ix] = P[rng.permutation(ix)]
        nl.append(Pp[com == c].mean())
    nl = np.array(nl)
    res[c] = {'n': int((com == c).sum()), 'mean_p': float(obs), 'null': float(nl.mean()),
              'p_high': float(np.mean(nl >= obs)), 'p_low': float(np.mean(nl <= obs))}
    print('commodity', c, res[c], flush=True)
out['la_commodity'] = res
json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=float)
