#!/usr/bin/env python3
"""LA-19 cycle 4: 'foreign words' as outliers rather than as a component.
If a minority tongue is too small to form a stable component, its words should still be phonotactically
surprising under the pooled K=1 model. Per-word held-out surprisal (5-fold CV, K=1, from cycle 1) per sign, by the
site(s) and support where the word type occurs; null = surprisal permuted among types inside length strata (5,000).
Controls: LB names (KN-only vs PY-only; Greek-looking vs non-Greek-looking label), planted 80/20 mixtures
(minority words must be more surprising), K=1-resampled LA (no site exists: attributes assigned at random).
"""
from la19_common import *


def test(surp, labels, lens, target, nperm=5000, seed=0):
    surp = np.asarray(surp); labels = np.asarray(labels); lens = np.minimum(np.asarray(lens), 6)
    m = labels == target
    if m.sum() < 3: return None
    obs = surp[m].mean() - surp[~m].mean()
    rnd = np.random.RandomState(seed); null = np.zeros(nperm)
    groups = [np.where(lens == L)[0] for L in np.unique(lens)]
    for r in range(nperm):
        s = surp.copy()
        for g in groups: s[g] = surp[rnd.permutation(g)]
        null[r] = s[m].mean() - s[~m].mean()
    return dict(n=int(m.sum()), diff=round(float(obs), 4), z=round(float((obs - null.mean()) / null.std()), 2),
                p_hi=round(float((1 + (null >= obs).sum()) / (nperm + 1)), 4))


def load(name):
    r = json.load(open(os.path.join(CK, 'c1_%s.json' % name)))
    W = [tuple(w.split('-')) for w in r['words']]
    lp = np.array(r['lpw']['1']); L = np.array([len(w) for w in W])
    return r, W, -lp / (L + 1), L


R = {}
r, W, s, L = load('LA')
T = la_tokens(); sites = collections.defaultdict(set); sups = collections.defaultdict(set)
SN = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}
for t in T:
    sites[t['w']].add(SN.get(t['site'], 'OTH')); sups[t['support']] if False else sups[t['w']].add(t['support'])
for st in ('HT', 'KH', 'ZA', 'PH', 'KN', 'OTH'):
    lab = np.array([('Y' if sites[w] == {st} else 'N') for w in W])
    R['LA_site_only_' + st] = test(s, lab, L, 'Y', seed=1)
for sp in ('Tablet', 'Nodule', 'Roundel', 'Stone vessel', 'Clay vessel', 'Metal object'):
    lab = np.array([('Y' if sp in sups[w] else 'N') for w in W])
    R['LA_support_' + sp] = test(s, lab, L, 'Y', seed=2)
r, W, s, L = load('LBpers'); site = dict(lb_names())
for st in ('KN', 'PY'):
    R['LB_site_' + st] = test(s, np.array([site.get(w) for w in W], dtype=object).astype(str), L, st, seed=3)
R['LB_nonGreek_label'] = test(s, np.array([lb_greekness(w) for w in W]), L, 'N', seed=4)
for nm in ('PM_fin_arb_80', 'PM_jpn_tur_80', 'PM_ell_eus_80', 'PM_sux_akk_80', 'PM_haw_kat_80'):
    r, W, s, L = load(nm)
    R['PLANT_minority_' + nm] = test(s, np.array(r['truth']), L, 1, seed=5)
for i in range(3):
    r, W, s, L = load('LA_K1res_%d' % i)
    rnd = np.random.RandomState(i); lab = np.where(rnd.rand(len(W)) < 0.15, 'Y', 'N')
    R['K1res_random15_%d' % i] = test(s, lab, L, 'Y', seed=6)
dump(os.path.join(OUT, 'c4.json'), R)
for k, v in R.items(): print(k, v)
