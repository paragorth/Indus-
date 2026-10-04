"""pe26 cycle 2: rank Susa tablets by how plateau-like their CLAY looks; freeze + hash BEFORE scoring;
then score against the hXRF labels (data/pe17_xrf_labels.json) and against the pe17 text ranking.
Model: ensemble of 2,000 random-feature-subset logistic models (massive random guessing) on within-batch-centred
clay features, NMI only (one museum): Susa (0) vs Yahya+Malyan+Ozbaki (1). Every tablet is scored out-of-fold
(5-fold, 20 repeats), so no tablet scores itself. Louvre Susa tablets are scored with the NMI model on their own
batch-centred features (transfer)."""
import sys, json, os, hashlib, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe26_common import *
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
rng = np.random.default_rng(2626)
rows = [r for r in load() if r['group'] == 'PE' and r.get('is_grey', 0) < 0.5]
bat = [r['batch'] for r in rows]
Xw_all = within_batch_centre(X_of(rows, CLAY), bat)
idx = {r['id']: i for i, r in enumerate(rows)}
train = [i for i, r in enumerate(rows) if r['museum'] == 'NMI' and r['site'] in ('Susa', 'Yahya', 'Malyan', 'Ozbaki')]
y = np.array([0 if rows[i]['site'] == 'Susa' else 1 for i in train])
Xt = Xw_all[train]
NM = int(os.environ.get('NMODELS', 2000)); K = len(CLAY)
subsets = [rng.choice(K, size=rng.integers(3, 9), replace=False) for _ in range(NM)]

def fit_ens(Xtr, ytr, models):
    out = []
    for s in models:
        sc = StandardScaler().fit(Xtr[:, s])
        m = LogisticRegression(C=0.3, max_iter=500, class_weight='balanced').fit(sc.transform(Xtr[:, s]), ytr)
        out.append((s, sc, m))
    return out

def score(ens, X):
    return np.mean([m.decision_function(sc.transform(X[:, s])) for s, sc, m in ens], 0)

# out-of-fold scores, repeated
oof = np.zeros(len(train)); cnt = np.zeros(len(train))
for rep in range(int(os.environ.get('REPS', 10))):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Xt, y):
        sub = [subsets[j] for j in rng.choice(NM, size=200, replace=False)]
        ens = fit_ens(Xt[tr], y[tr], sub); oof[te] += score(ens, Xt[te]); cnt[te] += 1
oof /= cnt
auc = roc_auc_score(y, oof)
# null: within-batch permuted labels
nul = []
for i in range(int(os.environ.get('NNULL', 30))):
    yp = perm_within_batch(y, [bat[j] for j in train], rng); s2 = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + i).split(Xt, yp):
        sub = [subsets[j] for j in rng.choice(NM, size=100, replace=False)]
        s2[te] = score(fit_ens(Xt[tr], yp[tr], sub), Xt[te])
    nul.append(roc_auc_score(yp, s2))
print('OOF AUC NMI Susa vs plateau', round(auc, 3), 'null q95', round(float(np.percentile(nul, 95)), 3), 'p', (np.sum(np.array(nul) >= auc) + 1) / (len(nul) + 1))
# Louvre Susa transfer scores with the full ensemble
ens_full = fit_ens(Xt, y, subsets[:500])
louvre = [i for i, r in enumerate(rows) if r['museum'] == 'Louvre' and r['site'] == 'Susa']
s_lou = score(ens_full, Xw_all[louvre]) if louvre else []
susa = [dict(id=rows[i]['id'], museum='NMI', batch=rows[i]['batch'], clay_plateau_score=round(float(oof[k]), 4)) for k, i in enumerate(train) if rows[i]['site'] == 'Susa']
susa += [dict(id=rows[i]['id'], museum='Louvre', batch=rows[i]['batch'], clay_plateau_score=round(float(s), 4)) for i, s in zip(louvre, s_lou)]
for m in ('NMI', 'Louvre'):
    L = sorted([s for s in susa if s['museum'] == m], key=lambda s: -s['clay_plateau_score'])
    for k, s in enumerate(L): s['rank_in_museum'] = k + 1; s['pct_in_museum'] = round(1 - k / max(len(L) - 1, 1), 4)
plateau = [dict(id=rows[i]['id'], site=rows[i]['site'], batch=rows[i]['batch'], clay_plateau_score_oof=round(float(oof[k]), 4)) for k, i in enumerate(train) if rows[i]['site'] != 'Susa']
frozen = dict(note='pe26 blind clay-appearance ranking (CDLI thumbnails, within-photo-batch centred colour+texture); frozen before scoring against hXRF and pe17',
              oof_auc_nmi=round(float(auc), 4), null_q95=round(float(np.percentile(nul, 95)), 4), susa=susa, plateau=plateau)
h = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
frozen['sha256_of_content_without_this_field'] = h
json.dump(frozen, open(os.path.join(D, 'pe26_frozen_clay_ranking.json'), 'w'), indent=0)
print('FROZEN sha256', h)

# ---------- scoring (only after freeze) ----------
lab = json.load(open(os.path.join(D, 'pe17_xrf_labels.json')))
res = dict(sha256=h, oof_auc=auc, null=nul)
st11 = 'P009157'
nmi = [s for s in susa if s['museum'] == 'NMI']
hit = [s for s in nmi if s['id'] == st11]
res['ST11'] = hit[0] if hit else 'no colour photo'
print('ST-11', res['ST11'], 'of', len(nmi))
# Yahya-found tablets: Susa-clay ones should look LESS plateau-like (lower score) than other Yahya tablets; Malyan-clay ones: no clear direction vs Yahya
pl = {p['id']: p for p in plateau}
yah = [p for p in plateau if p['site'] == 'Yahya']
for P, clay in [('P009536', 'Susa'), ('P009537', 'Susa'), ('P009545', 'Malyan'), ('P009540', 'Malyan')]:
    if P in pl:
        sc = pl[P]['clay_plateau_score_oof']; r = sum(q['clay_plateau_score_oof'] < sc for q in yah)
        print(P, clay, 'score', sc, 'rank-from-bottom among', len(yah), 'Yahya:', r + 1); res[P] = dict(score=sc, rank_from_bottom=r + 1, n=len(yah))
    else:
        print(P, clay, 'no colour photo'); res[P] = 'no colour photo'
# vs pe17 text ranking
t = json.load(open(os.path.join(D, 'pe17_frozen_ranking.json')))
tx = {s['id']: s['plateau_score'] for s in t['susa']}
for m in ('NMI', 'Louvre', 'all'):
    L = [s for s in susa if (m == 'all' or s['museum'] == m) and s['id'] in tx]
    if len(L) > 10:
        a = np.array([s['clay_plateau_score'] for s in L]); b = np.array([tx[s['id']] for s in L])
        rho = spearmanr(a, b).correlation
        bl = [s['batch'] for s in L]; nr = []
        for i in range(2000):
            pb = perm_within_batch(np.arange(len(a)), bl, rng); nr.append(spearmanr(a[pb], b).correlation)
        p = (np.sum(np.abs(nr) >= abs(rho)) + 1) / 2001
        print('pe17 vs clay', m, 'n', len(L), 'rho', round(rho, 3), 'p(two-sided, within-batch perm)', round(p, 4))
        res['pe17_' + m] = dict(n=len(L), rho=rho, p=p)
json.dump(res, open(os.path.join(CK, 'cycle2.json'), 'w'), indent=1, default=float)
