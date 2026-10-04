"""pe26 shared loader: features + labels + photo-batch keys, and the batch-controlled tests."""
import json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CK = os.path.join(D, 'pe26_ckpt')
COLOUR = ['L_p10', 'L_p25', 'L_med', 'L_p75', 'L_p90', 'L_iqr', 'a_p10', 'a_p25', 'a_med', 'a_p75', 'a_p90', 'a_iqr',
          'b_p10', 'b_p25', 'b_med', 'b_p75', 'b_p90', 'b_iqr', 'chroma', 'hue', 'a_over_L', 'b_over_L', 'r_chrom', 'g_chrom', 'rg_corr']
TEXTURE = ['hp_std', 'hp_rel', 'grad_med', 'grad_p90', 'mottle', 'dark_frac'] + [f'lbp{i}' for i in range(10)]
CLAY = COLOUR + TEXTURE
BATCHCOV = ['bg_L', 'bg_a', 'bg_b', 'fg_frac', 'H', 'W']

def site_of(prov):
    p = prov or ''
    for k, s in [('Susa', 'Susa'), ('Yahya', 'Yahya'), ('Malyan', 'Malyan'), ('Sialk', 'Sialk'), ('Sofalin', 'Sofalin'),
                 ('Ozbaki', 'Ozbaki'), ('Palum', 'Palum'), ('Girsu', 'Girsu'), ('Jemdet', 'JemdetNasr'), ('Uruk (mod', 'Uruk'),
                 ('Larsa', 'Larsa'), ('Kish', 'Kish'), ('Persepolis', 'Persepolis')]:
        if k in p: return s
    return 'other' if p else 'unknown'

def museum_of(c):
    c = c or ''
    if 'Louvre' in c: return 'Louvre'
    if 'Tehran' in c: return 'NMI'
    return 'other'

def batch_key(m):
    d = (m.get('date') or '')[:10]
    return '|'.join([m.get('make') or '-', m.get('model') or '-', (m.get('software') or '-')[:20], d or 'nodate'])

def load():
    rows = []; seen = set()
    cat = json.load(open(os.path.join(D, 'pe17_ckpt', 'pe_cat.json')))
    mp = json.load(open(os.path.join(CK, 'meta_pe.json'))); fp = json.load(open(os.path.join(CK, 'feat_pe.json')))
    for p, f in fp.items():
        if not f: continue
        c = cat[p]
        rows.append(dict(id=p, group='PE', site=site_of(c['provenience']), museum=museum_of(c['collection']),
                         museum_no=c['museum_no'], period='Proto-Elamite', batch=batch_key(mp[p]), **f))
    for suf in ('', '2'):
        if not os.path.exists(os.path.join(CK, f'feat_ctrl{suf}.json')): continue
        cc = json.load(open(os.path.join(CK, f'controls_cat{suf}.json')))
        mc = json.load(open(os.path.join(CK, f'meta_ctrl{suf}.json'))); fc = json.load(open(os.path.join(CK, f'feat_ctrl{suf}.json')))
        for p, f in fc.items():
            if not f or p in seen: continue
            c = cc[p]; seen.add(p)
            rows.append(dict(id=p, group='CTRL', site=site_of(c['provenience']), museum=museum_of(c['collection']),
                             museum_no=c['museum_no'], period=c['period'][:12], batch=batch_key(mc[p]), **f))
    return rows

def X_of(rows, cols):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    med = np.nanmedian(X, 0); i = np.where(np.isnan(X)); X[i] = np.take(med, i[1]); return X

def within_batch_centre(X, batches):
    X = X.copy(); b = np.array(batches)
    for k in set(batches):
        i = b == k; X[i] -= np.median(X[i], 0)
    return X

def cv_auc(X, y, groups, C=0.3, seed=0, nfold=5):
    """grouped CV: every batch is wholly in train or test. Returns out-of-fold scores and AUC."""
    g = np.array(groups); ug = sorted(set(groups)); rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ug)); fold = {ug[j]: k % nfold for k, j in enumerate(perm)}
    f = np.array([fold[x] for x in g]); s = np.full(len(y), np.nan)
    for k in range(nfold):
        te = f == k; tr = ~te
        if te.sum() == 0 or len(set(y[tr])) < 2: continue
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=C, max_iter=500, class_weight='balanced', solver='liblinear').fit(sc.transform(X[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(X[te]))
    ok = ~np.isnan(s)
    auc = roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) == 2 else float('nan')
    return auc, s

def strat_cv_auc(X, y, seed=0, nfold=5, C=0.3):
    """ordinary stratified CV (batches mixed), for comparison."""
    from sklearn.model_selection import StratifiedKFold
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(nfold, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=C, max_iter=500, class_weight='balanced', solver='liblinear').fit(sc.transform(X[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(X[te]))
    return roc_auc_score(y, s), s

def perm_within_batch(y, batches, rng):
    y = y.copy(); b = np.array(batches)
    for k in set(batches):
        i = np.where(b == k)[0]; y[i] = y[rng.permutation(i)]
    return y
