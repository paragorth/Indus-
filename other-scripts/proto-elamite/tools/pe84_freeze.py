"""pe84 cycle 3c: meter-only classifier for 'total-only reverse' (<= 2 lines, all with numerals), trained on transliterated
Susa tablets, frozen on the 87 untransliterated Tehran tablets. Features are measured at scales relative to tablet width
(PE84_FIXW=40 for all tablets, because the Tehran tablets have no catalogue dimensions).
usage: python3 pe84_freeze.py feats_w40.json feats_teh_w40.json -> data/pe84_frozen_tehran_reverse.json (+ sha256 print)"""
import sys, os, json, hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe84_common as C

KEYS = C.PHYS_CORE


def vec(p):
    o, r = p.get('ob'), p.get('rv')
    v = []
    for k in KEYS:
        a = o.get(k) if o else None
        b = r.get(k) if r else None
        v += [a if a is not None else np.nan, b if b is not None else np.nan,
              (b - a) if (a is not None and b is not None) else np.nan]
    v.append(float(r is None))
    v.append((r['area_cm2'] / o['area_cm2']) if (r and o) else np.nan)
    return v


def main():
    PH = json.load(open(sys.argv[1])); TE = json.load(open(sys.argv[2]))
    R = C.rows(sys.argv[1])
    X = np.array([vec(PH[r['id']]) for r in R], float)
    y = np.array([r['tx']['rv_total_only'] for r in R])
    half = np.array([r['half'] for r in R])
    med = np.nanmedian(X, 0)
    fill = lambda A: np.where(np.isfinite(A), A, med)
    X = fill(X)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    out = {'n_train': len(R), 'base_rate': round(float(y.mean()), 3)}
    rng = np.random.default_rng(8403)
    cv = []
    for seed in range(5):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            p[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        cv.append(roc_auc_score(y, p))
    out['cv_auc'] = round(float(np.mean(cv)), 3)
    for a, b in (('A', 'B'), ('B', 'A')):
        m = mk().fit(X[half == a], y[half == a])
        out['auc_%s_to_%s' % (a, b)] = round(float(roc_auc_score(y[half == b], m.predict_proba(X[half == b])[:, 1])), 3)
    sh = []
    for k in range(20):
        ys = rng.permutation(y); p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, ys):
            p[te] = mk().fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
        sh.append(roc_auc_score(ys, p))
    out['shuffled_cv_auc_max'] = round(float(max(sh)), 3); out['shuffled_cv_auc_median'] = round(float(np.median(sh)), 3)
    # content control: does the meter add anything beyond 'reverse present' and reverse/obverse area ratio?
    Xs = X[:, -2:]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        p[te] = mk().fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    out['cv_auc_presence_and_area_only'] = round(float(roc_auc_score(y, p)), 3)
    ii = [3 * KEYS.index('ink'), 3 * KEYS.index('ink') + 1, 3 * KEYS.index('ink') + 2]
    Xi = X[:, ii + [X.shape[1] - 2, X.shape[1] - 1]]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, y):
        p[te] = mk().fit(Xi[tr], y[tr]).predict_proba(Xi[te])[:, 1]
    out['cv_auc_ink_amount_only'] = round(float(roc_auc_score(y, p)), 3)
    m = mk().fit(X, y)
    ids = sorted(k for k, v in TE.items() if v and 'err' not in v and v.get('ob'))
    XT = fill(np.array([vec(TE[k]) for k in ids], float))
    pt = m.predict_proba(XT)[:, 1]
    thr = float(np.quantile(m.predict_proba(X)[:, 1], 1 - y.mean()))
    frozen = dict(what='pe84 cycle 3c: P(reverse is total-only: <= 2 lines, every line with numerals) for untransliterated '
                       'Tehran Susa tablets, from impression physique only (CDLI photos, scales relative to tablet width)',
                  trained=out, threshold_at_base_rate=round(thr, 4),
                  per_tablet={k: round(float(v), 4) for k, v in zip(ids, pt)},
                  predicted_total_only=[k for k, v in zip(ids, pt) if v >= thr],
                  rule='Would support (B): AUC >= 0.70 against the transliterations once published. Would kill: AUC < 0.60. '
                       'Also: the predicted set has a higher total-only rate than the rest.',
                  frozen='2026-10-07')
    path = os.path.join(C.common.DATA, 'pe84_frozen_tehran_reverse.json')
    s = json.dumps(frozen, indent=1, sort_keys=True)
    open(path, 'w').write(s)
    print(json.dumps(out, indent=1))
    print('n_tehran', len(ids), 'predicted', len(frozen['predicted_total_only']))
    print('sha256', hashlib.sha256(s.encode()).hexdigest())


if __name__ == '__main__':
    main()
