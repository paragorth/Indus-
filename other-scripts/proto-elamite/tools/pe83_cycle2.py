"""pe83 cycle 2: breakage at the top, random confound models, double ML, document type, period/deposit."""
import sys, os, json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P
import pe81_engine as E
import common

rng = np.random.default_rng(832)
F1 = json.load(open(os.path.join(P.CK, 'c1_feats.json'))); F2 = json.load(open(os.path.join(P.CK, 'c1d_feats.json')))
cat = P.catalogue()
R = [dict(r, **F1[r['id']], **F2[r['id']]) for r in P.rows() if F1.get(r['id']) and F2.get(r['id'])]
for r in R:
    l1 = r['t']['lines'][0]
    r['l1lac'] = float(l1['lacuna']); r['l1dam'] = float(l1['damaged'])
    c = cat[r['id']]
    r['pres'] = c['object_preservation'] or 'none'; r['surf'] = c['surface_preservation'] or 'none'
    r['kind'] = ('black' if r['bg'] < 30 else 'white' if r['bg'] > 200 else 'grey') + ('g' if r['sat'] < 6 else 'c')
TAG = ''
if os.environ.get('PE83_INTACT'):
    R = [r for r in R if r['l1lac'] == 0 and r['l1dam'] == 0]; TAG = '_intact'
X, names, _ = E.features([('S', r['t']) for r in R])
hd = np.array([r['hd'] for r in R]); n = len(R)
col = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in R], float)
Z0 = P.design(R)
out = {'n': n}
CF = {'cf': col('cf'), 'cf_top': col('cf_top'), 'rev_end': col('rev_img_lower')}


def boot_ci(y, h, Z, B=2000):
    ok = np.isfinite(y); y, h, Z = y[ok], h[ok], Z[ok]; rs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        try:
            rs.append(P.partial(y[i], h[i], Z[i])['r'])
        except Exception:
            pass
    return [round(float(np.nanquantile(rs, .025)), 3), round(float(np.nanquantile(rs, .975)), 3)]


# C2.1 intact top
intact = (col('l1lac') == 0) & (col('l1dam') == 0)
subs = {'a_line1_intact': intact, 'b_intact_complete': intact & np.array([r['comp'] for r in R]),
        'c_intact_smooth_top': intact & (col('top_rough') <= np.nanmedian(col('top_rough'))), 'd_intact_no_x': intact & (col('dx') == 0)}
Zl = P.design(R, [col('l1lac'), col('l1dam')])
out['C2.1_full_with_line1_flags'] = {k: P.partial(v, hd, Zl, 1000, rng) for k, v in CF.items()}
out['C2.1_header_rate_line1_lacuna_vs_not'] = [round(hd[col('l1lac') == 1].mean(), 3), round(hd[col('l1lac') == 0].mean(), 3)]
for nm, s in subs.items():
    out['C2.1_' + nm] = {k: dict(P.partial(v[s], hd[s], Z0[s], 1000, rng), ci=boot_ci(v[s], hd[s], Z0[s], 1000)) for k, v in CF.items()}
print(json.dumps(out, indent=1), flush=True)

# confound pool
def dum(key):
    ks = sorted(set(r[key] for r in R))[1:]
    return {f'{key}={k}': np.array([r[key] == k for r in R], float) for k in ks}
pool = {k: col(k) for k in ('la', 'asp', 'tw', 'nl', 'dx', 'rev', 'l1lac', 'l1dam', 'n_entries', 'numonly', 'corner_light', 'grad_tb',
                            'grad_lr', 'bright', 'rim_dark', 'keystone', 'lr_asym', 'tilt', 'top_rough', 'bot_rough', 'top_dark', 'Hpx')}
for key in ('pres', 'surf', 'coll', 'batch', 'kind'):
    pool.update(dum(key))
for nm in ('n_distinct_signs', 'sys_C', 'sys_B', 'sys_SDB', 'sys_C*', 'compound_share', 'log_sum_S'):
    pool[nm] = X[:, names.index(nm)]
pool = {k: np.nan_to_num(v, nan=np.nanmedian(v)) for k, v in pool.items()}
PK = list(pool)
out['C2.2_pool_size'] = len(PK)
rs = {k: [] for k in CF}
for it in range(1000):
    ks = list(rng.choice(PK, rng.integers(3, len(PK) + 1), replace=False))
    extra = [pool[k] for k in ks]
    for _ in range(rng.integers(0, 7)):
        a, b = rng.choice(PK, 2, replace=False); extra.append((pool[a] - pool[a].mean()) * (pool[b] - pool[b].mean()))
    for _ in range(rng.integers(0, 3)):
        a = rng.choice(PK); qs = np.unique(np.quantile(pool[a], np.linspace(0, 1, 6)[1:-1]))
        bins = np.digitize(pool[a], qs)
        extra += [(bins == b).astype(float) for b in np.unique(bins)[1:]]
    Z = np.column_stack([np.ones(n)] + extra)
    for k, v in CF.items():
        rs[k].append(P.partial(v, hd, Z)['r'])
out['C2.2_random_models'] = {k: dict(median=round(float(np.nanmedian(v)), 3), q05=round(float(np.nanquantile(v, .05)), 3),
                                     min=round(float(np.nanmin(v)), 3), share_below_0_10=round(float(np.mean(np.array(v) < .10)), 3))
                             for k, v in rs.items()}
print(json.dumps(out['C2.2_random_models'], indent=1), flush=True)

# double ML
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def dml(y, h, W, model='gb', folds=5, seed=0):
    ok = np.isfinite(y); y, h, W = rankdata(y[ok]) / ok.sum(), h[ok], W[ok]
    ry = np.zeros(len(y)); rh = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(W):
        if model == 'gb':
            my = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(W[tr], y[tr])
            mh = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(W[tr], h[tr])
        else:
            my = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(W[tr], y[tr]); mh = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(W[tr], h[tr])
        ry[te] = y[te] - my.predict(W[te]); rh[te] = h[te] - mh.predict(W[te])
    return round(float(np.corrcoef(ry, rh)[0, 1]), 3)


Wpool = np.column_stack([pool[k] for k in PK])
out['C2.2_dml_gb'] = {k: [dml(v, hd, Wpool, 'gb', seed=s) for s in range(3)] for k, v in CF.items()}
# planted controls
sxl = (pool['la'] - pool['la'].mean()) * (pool['nl'] - pool['nl'].mean())
lat = rankdata(sxl) / n + rankdata(pool['top_rough']) / n * -1 + rng.normal(0, 0.25, n)
fake1 = (lat > np.median(lat)).astype(float)
out['C2.2_planted_confound_only'] = dict(linear_pe81=P.partial(CF['cf_top'], fake1, Z0)['r'], dml_gb=dml(CF['cf_top'], fake1, Wpool, 'gb'))
lat2 = rankdata(CF['cf_top']) / n + rng.normal(0, 0.55, n)
fake2 = (lat2 > np.median(lat2)).astype(float)
out['C2.2_planted_true_link'] = dict(linear_pe81=P.partial(CF['cf_top'], fake2, Z0)['r'], dml_gb=dml(CF['cf_top'], fake2, Wpool, 'gb'))
print(json.dumps({k: out[k] for k in out if k.startswith('C2.2_')}, indent=1), flush=True)

# C2.3 document type: whole text profile minus header (line 1 left out of the profile features)
X2, n2, _ = E.features([('S', dict(r['t'], lines=r['t']['lines'][1:] or r['t']['lines'][:0])) for r in R])
keep = [i for i, nm in enumerate(n2) if nm != 'header' and X2[:, i].std() > 0]
Wtxt = np.column_stack([Wpool, X2[:, keep]])
out['C2.3_doc_type_ridge'] = {k: dml(v, hd, Wtxt, 'ridge') for k, v in CF.items()}
out['C2.3_doc_type_gb'] = {k: dml(v, hd, Wtxt, 'gb') for k, v in CF.items()}
sysn = ['sys_' + s for s in E.SYSTEMS]
maj = np.array([sysn[int(np.argmax(X[i, [names.index(s) for s in sysn]]))] if X[i, [names.index(s) for s in sysn]].sum() > 0 else 'none' for i in range(n)])
out['C2.3_within_major_system'] = {}
for s in sorted(set(maj)):
    m = maj == s
    if m.sum() >= 60 and 5 < hd[m].sum() < m.sum() - 5:
        out['C2.3_within_major_system'][s] = {k: P.partial(v[m], hd[m], Z0[m], 500, rng) for k, v in CF.items() if k != 'rev_end'}
print(json.dumps({k: out[k] for k in out if k.startswith('C2.3')}, indent=1), flush=True)


# C2.4 deposit / period
def within(groups, y, B=2000):
    g = np.array(groups, dtype=object); okg = np.array([x is not None for x in groups]) & np.isfinite(y)
    keys = [k for k in set(g[okg]) if (g[okg] == k).sum() >= 3 and 0 < hd[okg & (g == k)].sum() < (okg & (g == k)).sum()]
    m = okg & np.isin(g, keys)
    res = lambda v, Z: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    a = res(rankdata(y[m]), Z0[m]); b = res(hd[m], Z0[m]); gg = g[m]
    for k in keys:
        s = gg == k; a[s] -= a[s].mean(); b[s] -= b[s].mean()
    r = float(np.corrcoef(a, b)[0, 1]); cnt = 0
    for _ in range(B):
        bp = b.copy()
        for k in keys:
            s = np.where(gg == k)[0]; bp[s] = rng.permutation(bp[s])
        cnt += np.corrcoef(a, bp)[0, 1] >= r
    return dict(n=int(m.sum()), groups=len(keys), r=round(r, 3), p=round((cnt + 1) / (B + 1), 5))


out['C2.4_within_Sb5'] = {k: within([r['sb'] // 5 if r['sb'] else None for r in R], v) for k, v in CF.items()}
out['C2.4_within_volume'] = {k: within([r['t'].get('volume') for r in R], v) for k, v in CF.items()}
sus = np.array(['Susa' in r['site'] for r in R])
out['C2.4_susa'] = {k: P.partial(v[sus], hd[sus], Z0[sus], 1000, rng) for k, v in CF.items()}
out['C2.4_not_susa'] = {k: P.partial(v[~sus], hd[~sus], Z0[~sus], 1000, rng) for k, v in CF.items()}
print(json.dumps({k: out[k] for k in out if k.startswith('C2.4')}, indent=1), flush=True)
json.dump(out, open(os.path.join(P.CK, 'c2' + TAG + '.json'), 'w'), indent=1)
