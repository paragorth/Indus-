"""pe70 cycle 1: what separates sealed from unsealed tablets?

A. every feature (signs/words, header, number system, dominant entry-final sign, size and structure)
   vs the sealed flag; |z| against the family-wise max-|z| of label permutations within size bands
   (and within volume x size band for PE).
B. one full logistic model, 5-fold CV AUC, real vs label-permuted (same strata); and the AUC gain of
   'content' over a size+structure-only model.
C. 4,000 random classifiers (random 1-8 feature subsets, logistic, train half); survivors = top 2% by
   train AUC; their held-out AUC, real vs nulls.
Controls: Ur III (Umma + Girsu, words opaque; full and PE-size draws), proto-cuneiform (Uruk IV-III),
planted PE sign on 30% of sealed tablets.
usage: pe70_c1.py  -> data/pe70_ckpt/c1.json
"""
import json, sys, os, collections
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pe70_common import get, CK
from pe70_feats import matrix, strata, perm_within, lor_z, num_z
import warnings
warnings.filterwarnings('ignore')

NPERM = 200
CTRL = 'controls' in sys.argv
NRAND = 4000


def univariate(Xb, B, Xn, NN, y, st, rng, nperm=NPERM):
    z = np.concatenate([lor_z(Xb, y), num_z(Xn, y)])
    names = B + NN
    mx = np.array([np.abs(np.concatenate([lor_z(Xb, yp), num_z(Xn, yp)])).max()
                   for yp in (perm_within(y, st, rng) for _ in range(nperm))])
    thr = float(np.quantile(mx, .95))
    order = np.argsort(-np.abs(z))
    hits = [(names[i], round(float(z[i]), 2)) for i in order if abs(z[i]) > thr]
    return dict(thr=thr, n_hits=len(hits), hits=hits[:40], top=[(names[i], round(float(z[i]), 2)) for i in order[:15]])


def cv_auc(X, y, seed=0, C=0.1):
    if y.sum() < 10:
        return np.nan
    sk = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in sk.split(X, y):
        m = LogisticRegression(C=C, max_iter=300, class_weight='balanced').fit(X[tr], y[tr])
        p[te] = m.decision_function(X[te])
    return roc_auc_score(y, p)


def full_model(Xb, Xn, y, st, rng, nperm=20):
    Xall = np.hstack([Xb, Xn]); Xs = Xn
    real = cv_auc(Xall, y); real_s = cv_auc(Xs, y)
    nul, nul_gain = [], []
    for k in range(nperm):
        yp = perm_within(y, st, rng)
        a = cv_auc(Xall, yp, k); s = cv_auc(Xs, yp, k)
        nul.append(a); nul_gain.append(a - s)
    nul = np.array(nul); nul_gain = np.array(nul_gain)
    return dict(auc=real, auc_struct=real_s, gain=real - real_s, null_auc=float(nul.mean()), null_auc_sd=float(nul.std()),
                p_auc=float((nul >= real).mean()), null_gain=float(nul_gain.mean()), null_gain_sd=float(nul_gain.std()),
                p_gain=float((nul_gain >= real - real_s).mean()))


def _rand_job(args):
    X, y, tr, te, subsets = args
    out = []
    for S in subsets:
        try:
            m = LogisticRegression(C=1.0, max_iter=200, class_weight='balanced').fit(X[tr][:, S], y[tr])
            a_tr = roc_auc_score(y[tr], m.decision_function(X[tr][:, S]))
            a_te = roc_auc_score(y[te], m.decision_function(X[te][:, S]))
        except Exception:
            a_tr, a_te = 0.5, 0.5
        out.append((a_tr, a_te))
    return out


def random_search(X, y, names, labels, rng, pool, nrand=NRAND):
    """labels: dict name -> label vector (real + nulls).  same subsets and split for every label set."""
    n, p = X.shape
    tr = rng.random(n) < 0.5; te = ~tr
    tr, te = np.where(tr)[0], np.where(te)[0]
    subsets = [rng.choice(p, rng.integers(1, 9), replace=False) for _ in range(nrand)]
    res = {}
    for nm, yy in labels.items():
        half = nrand // 2
        parts = pool.map(_rand_job, [(X, yy, tr, te, subsets[:half]), (X, yy, tr, te, subsets[half:])])
        A = np.array(parts[0] + parts[1])
        top = np.argsort(-A[:, 0])[:max(1, nrand // 50)]
        fc = collections.Counter(names[j] for i in top for j in subsets[i])
        res[nm] = dict(surv_test_auc=float(A[top, 1].mean()), all_test_auc=float(A[:, 1].mean()),
                       best_train=float(A[top, 0].mean()), surv_feats=fc.most_common(12))
    return res


def run_corpus(R, kind, rng, pool, label, nulls=('band',), plant=None, do_rand=True, nperm=NPERM):
    Xb, B, Xn, NN = matrix(R, kind)
    y = np.array([r['sealed'] for r in R], dtype=int)
    if plant is not None:
        Xb, B, y = plant(Xb, B, y, rng)
    out = dict(label=label, n=len(R), sealed=int(y.sum()), n_feat=len(B) + len(NN))
    for how in nulls:
        st = strata(R, how)
        out['uni_' + how] = univariate(Xb, B, Xn, NN, y, st, rng, nperm)
        out['full_' + how] = full_model(Xb, Xn, y, st, rng)
    if do_rand:
        st = strata(R, nulls[-1])
        labels = {'real': y}
        for k in range(3):
            labels['null%d' % k] = perm_within(y, st, rng)
        X = np.hstack([Xb, Xn])
        out['rand'] = random_search(X, y, B + NN, labels, rng, pool)
    print(label, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('thr', 'n_hits', 'auc', 'auc_struct', 'gain', 'null_auc', 'p_auc', 'p_gain', 'null_gain')}) for k, v in out.items() if k != 'rand'}, default=float), flush=True)
    if 'rand' in out:
        print('   rand', {k: round(v['surv_test_auc'], 3) for k, v in out['rand'].items()}, flush=True)
    return out


def make_plant(frac=0.3):
    def plant(Xb, B, y, rng):
        df = Xb.sum(0)
        cand = [j for j in range(len(B)) if B[j].startswith('S:') and 20 <= df[j] <= 60]
        j = rng.choice(cand)
        idx = np.where((y == 1) & (Xb[:, j] == 0))[0]
        add = rng.choice(idx, int(frac * y.sum()), replace=False) if len(idx) >= int(frac * y.sum()) else idx
        Xb = Xb.copy(); Xb[add, j] = 1
        B = list(B); B[j] = B[j] + '*PLANT'
        return Xb, B, y
    return plant


def main():
    rng = np.random.default_rng(70)
    pe, ur, pc = get('pe'), get('ur3'), get('pc')
    res = {}
    with Pool(2) as pool:
        if not CTRL:
            res['pe'] = run_corpus(pe, 'pe', rng, pool, 'PE', nulls=('band', 'volband'))
            res['plant'] = [run_corpus(pe, 'pe', rng, pool, 'PE planted %d' % k, nulls=('volband',), plant=make_plant(),
                                       do_rand=(k < 2), nperm=100) for k in range(5)]
        res['pc'] = run_corpus(pc, 'pc', rng, pool, 'PC', nulls=('band', 'volband'))
        idx = rng.choice(len(ur), 4000, replace=False)
        res['ur3_full'] = run_corpus([ur[i] for i in idx], 'ur3', rng, pool, 'UR3 4000', nulls=('band', 'volband'))
        res['ur3_pe'] = []
        sealed = [i for i, r in enumerate(ur) if r['sealed']]; uns = [i for i, r in enumerate(ur) if not r['sealed']]
        for k in range(5):
            ii = list(rng.choice(sealed, 192, replace=False)) + list(rng.choice(uns, 1389, replace=False))
            res['ur3_pe'].append(run_corpus([ur[i] for i in ii], 'ur3', rng, pool, 'UR3 PE-size %d' % k,
                                            nulls=('volband',), do_rand=(k < 2), nperm=100))
    json.dump(res, open(os.path.join(CK, 'c1_ctrl.json' if CTRL else 'c1.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
