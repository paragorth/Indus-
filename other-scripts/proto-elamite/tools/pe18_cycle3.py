"""pe18 cycle 3: massive random search over feature sets that predict a tablet's number system.
Target: main non-ambiguous system of each Susa tablet (C, C@, DEC, SEX, B, S@, N23, FRAC).
Families: GOODS (class signs present: pe6-style baseline), HDR, VOC (non-class signs), VAR (variant
forms), FMT, SITE (plateau-likeness scores), SEAL.  Model: multinomial logistic regression.
Search half (60% of Susa tablets, 5-fold CV) / reserve (40%, frozen).  Score = CV bits saved per
tablet by GOODS+set over GOODS alone.  FWER from the same search on labels permuted within
(size bin x capacity flag x dominant class sign) -- keeps goods->system, kills anything beyond.
Planted: a real variant form switched on in 30% of DEC tablets.  Transfer: plateau tablets.
"""
import json, sys, os, collections, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe18_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

NSETS = int(os.environ.get('NSETS', 2000))
R, CLASS = tablets()
ORDER = ['DEC', 'SEX', 'B', 'S@', 'N23', 'FRAC', 'C@', 'C']


def main_sys(t):
    c = {s: n for s, n in t['sys'].items() if s != 'AMB'}
    if not c:
        return None
    mx = max(c.values())
    return [s for s in ORDER if c.get(s, 0) == mx][0]


SU = [t for t in R if t['region'] == 'SUSA' and main_sys(t)]
PL = [t for t in R if t['region'] == 'PLAT' and main_sys(t)]
CLS = sorted({main_sys(t) for t in SU})
y_all = np.array([CLS.index(main_sys(t)) for t in SU])
m = Model([t for t in R if t['region'] == 'PLAT'], [t for t in R if t['region'] == 'SUSA'])


def feats(rows, loo):
    goods = sorted(CLASS)
    hdrs = [h for h, _ in collections.Counter(t['hdr'] for t in SU if t['hdr']).most_common(25)]
    voc = [s for s, _ in collections.Counter(x for t in SU for x in set(t['nocls'])).most_common(150)]
    var = [s for s, _ in collections.Counter(x for t in SU for x in set(t['forms'])).most_common(120)]
    F, names, fam = [], [], []
    def add(n, f, col):
        names.append(n); fam.append(f); F.append(col)
    for g in goods:
        add('G:' + g, 'GOODS', [float(g in t['finals']) for t in rows])
    for h in hdrs:
        add('H:' + h, 'HDR', [float(t['hdr'] == h) for t in rows])
    add('H:none', 'HDR', [float(t['hdr'] is None) for t in rows])
    for v in voc:
        add('V:' + v, 'VOC', [float(v in t['nocls']) for t in rows])
    for v in var:
        add('R:%s' % v[1], 'VAR', [float(v in t['forms']) for t in rows])
    fm = np.array([t['fmt'] for t in rows])
    for j in range(fm.shape[1]):
        add('F:%d' % j, 'FMT', list(fm[:, j]))
    S = scores(m, rows, loo)
    S = np.nan_to_num(S, nan=0.0)
    for j, f in enumerate(FAM):
        add('S:' + f, 'SITE', list(S[:, j]))
    add('SEAL', 'SEAL', [float(t['sealed']) for t in rows])
    X = np.array(F).T
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return X, names, np.array(fam)


X_all, NAMES, FAMS = feats(SU, True)
mu, sd = X_all.mean(0), X_all.std(0) + 1e-9
Xz = (X_all - mu) / sd
GOODS = np.where(FAMS == 'GOODS')[0]
rng0 = np.random.default_rng(1818)
idx = rng0.permutation(len(SU)); ns = int(0.6 * len(SU))
SRCH, RES = idx[:ns], idx[ns:]
sb = np.array([size_bin(t) for t in SU]); cap = np.array([t['cap'] for t in SU])
topdom = [d for d, _ in collections.Counter(t['dom'] for t in SU).most_common(10)]
STR = np.array(['%d_%d_%s' % (b, c, t['dom'] if t['dom'] in topdom else 'o') for b, c, t in zip(sb, cap, SU)])
K = len(CLS)


def bits(model, X, y):
    P = model.predict_proba(X)
    full = np.full((len(y), K), 1e-4)
    full[:, model.classes_] = P
    full /= full.sum(1, keepdims=True)
    return float(-np.log2(full[np.arange(len(y)), y]).mean())


def fit(X, y):
    return LogisticRegression(C=0.3, max_iter=300).fit(X, y)


def cv_bits(cols, y, rows):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    yy = y[rows]; ys = np.where(np.bincount(yy, minlength=K)[yy] >= 5, yy, -1)
    tot = 0.0
    for tr, te in skf.split(rows, ys):
        mdl = fit(Xz[rows[tr]][:, cols], yy[tr])
        tot += bits(mdl, Xz[rows[te]][:, cols], yy[te]) * len(te)
    return tot / len(rows)


NONG = np.where(FAMS != 'GOODS')[0]
FAMLIST = ['HDR', 'VOC', 'VAR', 'FMT', 'SITE', 'SEAL']


def draw_sets(n, seed):
    r = np.random.default_rng(seed); out = []
    for _ in range(n):
        fams = r.choice(FAMLIST, r.integers(1, 3), replace=False)
        pool = np.where(np.isin(FAMS, fams))[0]
        k = int(min(len(pool), r.integers(1, 16)))
        out.append(sorted(r.choice(pool, k, replace=False).tolist()))
    return out


SETS = draw_sets(NSETS, 7)


def run_search(args):
    tag, y = args
    base = cv_bits(GOODS, y, SRCH)
    res = []
    for s in SETS:
        res.append(base - cv_bits(np.concatenate([GOODS, s]), y, SRCH))
    return tag, base, res


def reserve_gain(cols, y, n_null=200, seed=0):
    r = np.random.default_rng(seed)
    def g(yv):
        b = bits(fit(Xz[SRCH][:, GOODS], yv[SRCH]), Xz[RES][:, GOODS], yv[RES])
        c = bits(fit(Xz[SRCH][:, cols], yv[SRCH]), Xz[RES][:, cols], yv[RES])
        return b - c
    cols = np.concatenate([GOODS, cols])
    obs = g(y)
    nl = []
    for _ in range(n_null):
        y2 = y.copy(); y2[RES] = strata_perm(y[RES], STR[RES], r)
        nl.append(g(y2))
    nl = np.array(nl)
    return obs, float((1 + (nl >= obs).sum()) / (n_null + 1)), float(nl.mean())


if __name__ == '__main__':
    out = {'classes': {c: int((y_all == i).sum()) for i, c in enumerate(CLS)}, 'n_susa': len(SU), 'n_plat': len(PL),
           'n_features': len(NAMES), 'nsets': NSETS}
    print(out, flush=True)
    # family-level gains on the reserve
    fam_res = {}
    for f in FAMLIST + ['ALLNONGOODS']:
        cols = NONG if f == 'ALLNONGOODS' else np.where(FAMS == f)[0]
        o, p, nm = reserve_gain(cols, y_all, 200, 1)
        fam_res[f] = {'reserve_gain_bits': round(o, 4), 'null_mean': round(nm, 4), 'p': round(p, 4)}
        print('3a family', f, fam_res[f], flush=True)
    b_goods = bits(fit(Xz[SRCH][:, GOODS], y_all[SRCH]), Xz[RES][:, GOODS], y_all[RES])
    pri = np.bincount(y_all[SRCH], minlength=K) + 0.5; pri = pri / pri.sum()
    b_prior = float(-np.log2(pri[y_all[RES]]).mean())
    out['3a'] = {'reserve_bits_prior': round(b_prior, 4), 'reserve_bits_goods': round(b_goods, 4), 'families': fam_res}
    # searches: real, 3 nulls, planted
    rng = np.random.default_rng(99)
    jobs = [('real', y_all)]
    for k in range(3):
        y2 = strata_perm(y_all, STR, rng); jobs.append(('null%d' % k, y2))
    # planted: real variant form with 3-8% prevalence switched on in 30% of DEC tablets
    VARC = np.where(FAMS == 'VAR')[0]
    prev = X_all[:, VARC].mean(0)
    cand = VARC[(prev > 0.03) & (prev < 0.08)]
    pc = int(rng.choice(cand))
    dec = np.where(y_all == CLS.index('DEC'))[0]
    on = rng.choice(dec, int(0.3 * len(dec)), replace=False)
    Xp = X_all.copy(); Xp[on, pc] = 1.0
    planted_col = (Xp[:, pc] - Xp[:, pc].mean()) / (Xp[:, pc].std() + 1e-9)
    out['planted_feature'] = NAMES[pc]
    with Pool(2) as P:
        R3 = dict((tag, (b, r)) for tag, b, r in P.map(run_search, jobs))
    # planted search: swap column in place, single process
    Xz_saved = Xz[:, pc].copy(); Xz[:, pc] = planted_col
    _, bp, rp = run_search(('planted', y_all))
    Xz[:, pc] = Xz_saved
    fwer = max(max(R3['null%d' % k][1]) for k in range(3))
    real = np.array(R3['real'][1]); order = np.argsort(-real)
    out['3b'] = {'real_best': round(float(real.max()), 4), 'null_maxes': [round(float(max(R3['null%d' % k][1])), 4) for k in range(3)],
                 'null_q95_each': [round(float(np.quantile(R3['null%d' % k][1], .95)), 4) for k in range(3)],
                 'real_q95': round(float(np.quantile(real, .95)), 4), 'fwer_line': round(fwer, 4),
                 'n_survivors': int((real > fwer).sum()), 'goods_cv_bits': round(R3['real'][0], 4)}
    print('3b', out['3b'], flush=True)
    rp = np.array(rp); po = np.argsort(-rp)
    out['3c_planted'] = {'best': round(float(rp.max()), 4), 'n_over_fwer': int((rp > fwer).sum()),
                         'top10_contain_planted': int(sum(pc in SETS[i] for i in po[:10])),
                         'sets_with_planted_mean_gain': round(float(np.mean([rp[i] for i in range(len(SETS)) if pc in SETS[i]] or [0])), 4),
                         'sets_without_mean_gain': round(float(np.mean([rp[i] for i in range(len(SETS)) if pc not in SETS[i]])), 4)}
    print('3c planted', out['3c_planted'], flush=True)
    # re-test top 15 real sets on the reserve
    top = []
    for i in order[:15]:
        o, p, nm = reserve_gain(np.array(SETS[i]), y_all, 100, i)
        top.append({'set': [NAMES[j] for j in SETS[i]], 'search_gain': round(float(real[i]), 4),
                    'reserve_gain': round(o, 4), 'reserve_p': round(p, 4)})
        print('3d', top[-1], flush=True)
    out['3d_top'] = top
    # feature frequency among top 2% of real vs null sets
    def enrich(res):
        q = np.quantile(res, .98); c = collections.Counter()
        for i, v in enumerate(res):
            if v >= q:
                c.update(SETS[i])
        return c
    er = enrich(real); en = collections.Counter()
    for k in range(3):
        en.update(enrich(R3['null%d' % k][1]))
    out['3e_enriched'] = [(NAMES[j], c, round(en[j] / 3, 1)) for j, c in er.most_common(20)]
    print('3e', out['3e_enriched'], flush=True)
    # transfer to plateau
    yP = np.array([CLS.index(main_sys(t)) if main_sys(t) in CLS else -1 for t in PL])
    XP, _, _ = feats(PL, False); XPz = (XP - mu) / sd
    ok = yP >= 0
    allrows = np.arange(len(SU))
    tr = {}
    for nm, cols in [('GOODS', GOODS), ('GOODS+ALL', np.arange(len(NAMES))), ('GOODS+best', np.concatenate([GOODS, SETS[order[0]]])),
                     ('GOODS+SITE', np.concatenate([GOODS, np.where(FAMS == 'SITE')[0]]))]:
        mdl = fit(Xz[:, cols], y_all)
        tr[nm] = round(bits(mdl, XPz[ok][:, cols], yP[ok]), 4)
    pri = np.bincount(y_all, minlength=K) + .5; pri /= pri.sum()
    tr['prior'] = round(float(-np.log2(pri[yP[ok]]).mean()), 4)
    tr['n'] = int(ok.sum()); tr['labels'] = dict(collections.Counter(CLS[i] for i in yP[ok]))
    out['3f_transfer'] = tr
    print('3f', tr, flush=True)
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=str)
