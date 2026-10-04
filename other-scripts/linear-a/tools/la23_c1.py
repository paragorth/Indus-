"""LA-23 cycle 1: blind separation of person-like entry words, calibrated on Linear B.

Features are identity-free (position, number size, repetition, document context, length).
Calibration: classifiers trained on LA-sized Linear B draws from one site, scored on LA-sized
draws from the other site (AUC, precision of the p>=0.5 set). Blind two-component Gaussian
mixture as an unsupervised alternative. Controls: (i) LB with word tokens shuffled across all
word slots (positions broken) must lose the separation; (ii) label-permuted training.
Output: data/la23_ckpt/c1.json (LA person probabilities), rows for loops/la23_cycle1.txt.
"""
import sys, json
from la23_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(seed('la23c1'))
LA = la_docs(site=None)
LAocc = occurrences(LA)
NOCC = len(LAocc)
LB = lb_docs()
LBL = lb_name_label()
by_site = {s: [d for d in LB if d['site'] == s] for s in ('KN', 'PY')}


def draw(docs, target, rng):
    idx = rng.permutation(len(docs)); out, n = [], 0
    for i in idx:
        d = docs[i]; k = sum(1 for ln in d['lines'] for t, _ in ln if t == 'W')
        if k == 0: continue
        out.append(d); n += k
        if n >= target: break
    return out


def shuffle_words(docs, rng):
    ws = [v for d in docs for ln in d['lines'] for k, v in ln if k == 'W']
    rng.shuffle(ws); it = iter(ws); out = []
    for d in docs:
        out.append(dict(d, lines=[[(k, next(it)) if k == 'W' else (k, v) for k, v in ln] for ln in d['lines']]))
    return out


def featset(docs, shuffle=False, rng=None):
    if shuffle: docs = shuffle_words(docs, rng)
    T, X = type_features(occurrences(docs))
    y = np.array([t in LBL for t in T], int)
    return T, X, y


def models():
    return {'logit': lambda: LogisticRegression(max_iter=2000, C=1.0),
            'gbm': lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=3)}


res = collections.defaultdict(list)
NREP = 40
for rep in range(NREP):
    for tr, te in (('KN', 'PY'), ('PY', 'KN')):
        Ttr, Xtr, ytr = featset(draw(by_site[tr], NOCC, rng))
        Tte, Xte, yte = featset(draw(by_site[te], NOCC, rng))
        sc = StandardScaler().fit(Xtr)
        for mn, mk in models().items():
            m = mk().fit(sc.transform(Xtr), ytr); p = m.predict_proba(sc.transform(Xte))[:, 1]
            res[(mn, 'real')].append((roc_auc_score(yte, p), yte[p >= 0.5].mean() if (p >= 0.5).any() else np.nan,
                                      (p >= 0.5).mean(), yte.mean()))
            # control: permuted training labels
            m2 = mk().fit(sc.transform(Xtr), rng.permutation(ytr)); p2 = m2.predict_proba(sc.transform(Xte))[:, 1]
            res[(mn, 'labelperm')].append((roc_auc_score(yte, p2), np.nan, np.nan, np.nan))
        # control: positions broken (words shuffled across slots) in the test draw
        Tsh, Xsh, ysh = featset(draw(by_site[te], NOCC, rng), shuffle=True, rng=rng)
        m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        res[('logit', 'shuffled_test')].append((roc_auc_score(ysh, m.predict_proba(sc.transform(Xsh))[:, 1]), np.nan, np.nan, np.nan))
        # blind GMM on test draw: pick the component with more f_entry*f_small? choose by
        # lower log_occ (rarer) -> 'person'; report AUC
        gm = GaussianMixture(2, covariance_type='diag', random_state=rep).fit(StandardScaler().fit_transform(Xte))
        post = gm.predict_proba(StandardScaler().fit_transform(Xte))
        comp = int(np.argmin(gm.means_[:, FEATS.index('log_occ')]))
        res[('gmm_blind', 'real')].append((roc_auc_score(yte, post[:, comp]), np.nan, np.nan, np.nan))
    if rep % 10 == 9: print('rep', rep + 1, flush=True)

summ = {}
for k, v in res.items():
    a = np.array(v, float)
    summ['%s|%s' % k] = dict(auc=float(np.nanmean(a[:, 0])), auc_lo=float(np.nanpercentile(a[:, 0], 5)),
                             prec=float(np.nanmean(a[:, 1])) if not np.all(np.isnan(a[:, 1])) else None,
                             frac_pos=float(np.nanmean(a[:, 2])) if not np.all(np.isnan(a[:, 2])) else None,
                             base=float(np.nanmean(a[:, 3])) if not np.all(np.isnan(a[:, 3])) else None)
for k, v in summ.items(): print(k, v)

# ---- final models on all LB (both sites, LA-sized draws pooled as ensemble) -> LA
TL, XL = type_features(LAocc)
PL = []
coefs = []
for rep in range(60):
    s = 'KN' if rep % 2 == 0 else 'PY'
    Ttr, Xtr, ytr = featset(draw(by_site[s], NOCC, rng))
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    g = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=3).fit(sc.transform(Xtr), ytr)
    PL.append(0.5 * m.predict_proba(sc.transform(XL))[:, 1] + 0.5 * g.predict_proba(sc.transform(XL))[:, 1])
    coefs.append(m.coef_[0])
PL = np.array(PL)
pm = PL.mean(0); psd = PL.std(0)
# shuffled LA: words shuffled across slots
TLs, XLs = type_features(occurrences(shuffle_words(LA, rng)))
PLs = []
for rep in range(20):
    s = 'KN' if rep % 2 == 0 else 'PY'
    Ttr, Xtr, ytr = featset(draw(by_site[s], NOCC, rng)); sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    g = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=3).fit(sc.transform(Xtr), ytr)
    PLs.append(0.5 * m.predict_proba(sc.transform(XLs))[:, 1] + 0.5 * g.predict_proba(sc.transform(XLs))[:, 1])
PLs = np.array(PLs).mean(0)

# blind GMM directly on LA
Z = StandardScaler().fit_transform(XL)
gm = GaussianMixture(2, covariance_type='diag', random_state=0).fit(Z)
comp = int(np.argmin(gm.means_[:, FEATS.index('log_occ')]))
pg = gm.predict_proba(Z)[:, comp]
from scipy.stats import spearmanr
rho = spearmanr(pm, pg).correlation

# HT-only stats
HT = la_docs(); HTw = set(o['w'] for o in occurrences(HT))
idx = {t: i for i, t in enumerate(TL)}
ht_p = np.array([pm[idx[w]] for w in HTw])
out = dict(summary=summ, feats=FEATS, coef_mean=np.mean(coefs, 0).tolist(),
           la=[dict(w=t, p=float(pm[i]), sd=float(psd[i]), p_gmm=float(pg[i]), ht=t in HTw) for i, t in enumerate(TL)],
           la_shuffled_mean_p=float(PLs.mean()), la_shuffled_frac_ge05=float((PLs >= 0.5).mean()),
           la_frac_ge05=float((pm >= 0.5).mean()), ht_frac_ge05=float((ht_p >= 0.5).mean()), ht_n=len(HTw),
           rho_clf_gmm=float(rho))
json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
print('LA frac p>=.5', out['la_frac_ge05'], 'HT', out['ht_frac_ge05'], 'n HT types', len(HTw),
      'shuffled LA frac', out['la_shuffled_frac_ge05'], 'rho clf/gmm', rho)
print('coef', dict(zip(FEATS, np.round(np.mean(coefs, 0), 2))))
top = sorted(out['la'], key=lambda r: -r['p'])
print('top', [(r['w'], round(r['p'], 2)) for r in top[:40]])
print('bottom', [(r['w'], round(r['p'], 2)) for r in top[-25:]])
