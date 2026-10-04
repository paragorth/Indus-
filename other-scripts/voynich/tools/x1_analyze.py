"""X1: does an 'undeciphered signature' exist?  Univariate exact-permutation scan, cluster test, LOCO classifier.

usage: python3 x1_analyze.py REGIME [--u indus,linear_a,proto_elamite,voynich] [--drop name,...] [--nperm 300]
         [--gran sign|all] [--tag TAG]
Writes voynich/data/results/x1_analysis_<REGIME><TAG>.json and prints a report.

Controls:
  * exact label permutation: every 4-subset of the U+D corpora relabelled 'undeciphered' (C(n,4) labelings)
    gives the null for (a) the number of features on which the labelled four are perfectly separated,
    (b) the number with AUC >= 0.95, (c) the cluster statistic (within-four distance / four-to-rest distance)
  * classifier: leave-one-corpus-out L2 logistic regression on all non-size features; held-out score =
    mean P(U) over the corpus' windows; AUC across corpora; null = the same pipeline on random 4-subsets.
"""
import sys, os, json, itertools, argparse, random, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'data', 'results')


def load(regime):
    return json.load(open(os.path.join(RES, f'x1_features_{regime}.json')))


def matrix(data, names, feats):
    M = np.array([[np.nanmedian([r.get(f, np.nan) for r in data[n]['reps']]) for f in feats] for n in names])
    return M


def auc_vec(R, lab):
    """R: corpora x features ranks (1..n, average ties); lab: bool vector. AUC of U above D per feature."""
    k = lab.sum(); m = len(lab) - k
    s = R[lab].sum(0)
    return (s - k * (k + 1) / 2) / (k * m)


def rankdata_cols(M):
    from scipy.stats import rankdata
    return np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])


def loco_scores(data, names, feats, ulab, C=0.05):
    Xs, ys, gs = [], [], []
    for i, n in enumerate(names):
        for r in data[n]['reps']:
            Xs.append([r.get(f, np.nan) for f in feats]); ys.append(int(ulab[i])); gs.append(i)
    X = np.array(Xs, float); y = np.array(ys); g = np.array(gs)
    out = np.zeros(len(names))
    for i in range(len(names)):
        tr = g != i
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(C=C, class_weight='balanced', solver='liblinear', max_iter=500))
        model.fit(X[tr], y[tr])
        out[i] = model.predict_proba(X[g == i])[:, 1].mean()
    return out


def auc_scores(sc, lab):
    u = sc[lab]; d = sc[~lab]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in u for b in d]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('regime')
    ap.add_argument('--u', default='indus,linear_a,proto_elamite,voynich')
    ap.add_argument('--drop', default='')
    ap.add_argument('--gran', default='all')
    ap.add_argument('--nperm', type=int, default=300)
    ap.add_argument('--tag', default='')
    ap.add_argument('--families', default='')  # restrict to these families (comma list)
    a = ap.parse_args()
    data = load(a.regime)
    U = a.u.split(',')
    drop = set(x for x in a.drop.split(',') if x)
    names = [n for n in data if (n in U or data[n]['class'] == 'D') and n not in drop
             and (a.gran == 'all' or n in U or data[n]['gran'] == a.gran)]
    names = U + sorted(n for n in names if n not in U)
    tests = [n for n in data if n not in names]
    feats = sorted(set(f for n in names for r in data[n]['reps'] for f in r))
    feats = [f for f in feats if not f.startswith('size_')]
    if a.families:
        fam = set(a.families.split(','))
        feats = [f for f in feats if f.split('_')[0] in fam]
    M = matrix(data, names, feats)
    # impute per feature median
    med = np.nanmedian(M, 0)
    keep = ~np.isnan(med)
    M = M[:, keep]; feats = [f for f, k in zip(feats, keep) if k]
    M = np.where(np.isnan(M), med[keep], M)
    n = len(names); lab = np.array([x in U for x in names]); names = np.array(names, dtype=object)
    print(f'regime {a.regime}: {n} corpora ({lab.sum()} U), {len(feats)} features; test-only: {tests}')
    R = rankdata_cols(M)
    auc = auc_vec(R, lab)
    sep = np.abs(auc - 0.5) * 2  # 1 = perfect
    perfect = int((sep >= 0.999).sum()); near = int((sep >= 0.9).sum())
    # within-corpus noise -> margin in pooled sd units for perfect separators
    sd = np.array([[np.nanstd([r.get(f, np.nan) for r in data[nm]['reps']]) for f in feats] for nm in names])
    rows = []
    for j in np.argsort(-sep):
        u = M[lab, j]; d = M[~lab, j]
        if auc[j] >= 0.5:
            gap = u.min() - d.max()
        else:
            gap = d.min() - u.max()
        scale = np.nanstd(M[:, j]) + 1e-12
        rows.append({'feat': feats[j], 'auc': float(auc[j]), 'gap_sd': float(gap / scale),
                     'u': [float(x) for x in u], 'd_min': float(d.min()), 'd_max': float(d.max()),
                     'd_med': float(np.median(d)), 'nearest_d': names[~lab][np.argmin(np.abs(d - (u.min() if auc[j] >= .5 else u.max())))] if True else ''})
    # exact enumeration of all k-subsets
    k = int(lab.sum())
    combs = list(itertools.combinations(range(n), k))
    Rsum_cache = R  # n x F
    nperf, nnear, clus = [], [], []
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    for c in combs:
        s = R[list(c)].sum(0)
        au = (s - k * (k + 1) / 2) / (k * (n - k))
        sp = np.abs(au - 0.5) * 2
        nperf.append(int((sp >= 0.999).sum())); nnear.append(int((sp >= 0.9).sum()))
        idx = list(c); rest = [i for i in range(n) if i not in c]
        w = D[np.ix_(idx, idx)][np.triu_indices(k, 1)].mean(); b = D[np.ix_(idx, rest)].mean()
        clus.append(w / b)
    nperf = np.array(nperf); nnear = np.array(nnear); clus = np.array(clus)
    ridx = combs.index(tuple(range(k)))
    names = list(names)
    p_perf = float((nperf >= perfect).mean()); p_near = float((nnear >= near).mean())
    p_clus = float((clus <= clus[ridx]).mean())
    print(f'perfect separators {perfect} (null mean {nperf.mean():.2f}, 95th {np.percentile(nperf,95):.0f}, P = {p_perf:.4f}); '
          f'|AUC-.5|*2 >= 0.9: {near} (null mean {nnear.mean():.1f}, 95th {np.percentile(nnear,95):.0f}, P = {p_near:.4f})')
    print(f'cluster ratio within-U / U-to-D = {clus[ridx]:.3f} (null mean {clus.mean():.3f}, P = {p_clus:.4f})')
    # which subsets beat the real one
    best = np.argsort(-nnear)[:5]
    top_sub = [([names[i] for i in combs[b]], int(nnear[b])) for b in best]
    print('top labelings by near-separator count:', top_sub)
    print('top features:')
    for r in rows[:25]:
        print(f"  {r['feat']:32s} AUC {r['auc']:.3f} gap {r['gap_sd']:+.2f}sd U {[round(x,3) for x in r['u']]} D [{r['d_min']:.3f},{r['d_max']:.3f}] med {r['d_med']:.3f} nearestD {r['nearest_d']}")
    sys.stdout.flush()
    # LOCO classifier
    sc = loco_scores(data, names, feats, lab)
    real_auc = auc_scores(sc, lab)
    rng = random.Random(5)
    null = []
    pool = [c for i, c in enumerate(combs) if i != ridx]
    for c in rng.sample(pool, min(a.nperm, len(pool))):
        l2 = np.zeros(n, bool); l2[list(c)] = True
        null.append(auc_scores(loco_scores(data, names, feats, l2), l2))
    null = np.array(null) if null else np.array([np.nan])
    p_cls = float(((null >= real_auc).sum() + 1) / (len(null) + 1))
    order = np.argsort(-sc)
    print(f'LOCO logistic AUC {real_auc:.3f} (null mean {np.nanmean(null):.3f}, 95th {np.nanpercentile(null,95):.3f}, P = {p_cls:.4f})')
    print('  held-out P(U):', ', '.join(f'{names[i]} {sc[i]:.2f}' for i in order))
    # test-only corpora scored by a model trained on all
    Xs, ys = [], []
    for i, nm in enumerate(names):
        for r in data[nm]['reps']:
            Xs.append([r.get(f, np.nan) for f in feats]); ys.append(int(lab[i]))
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000)).fit(np.array(Xs, float), ys)
    tsc = {}
    for t in tests:
        X = np.array([[r.get(f, np.nan) for f in feats] for r in data[t]['reps']], float)
        tsc[t] = float(model.predict_proba(X)[:, 1].mean())
    print('  test-only P(U):', {k: round(v, 2) for k, v in tsc.items()})
    coef = model[-1].coef_[0]
    topc = np.argsort(-np.abs(coef))[:12]
    print('  top coefficients:', [(feats[j], round(float(coef[j]), 2)) for j in topc])
    out = {'regime': a.regime, 'U': U, 'names': names, 'n_feats': len(feats), 'perfect': perfect,
           'perfect_null_mean': float(nperf.mean()), 'p_perfect': p_perf, 'near': near,
           'near_null_mean': float(nnear.mean()), 'p_near': p_near, 'cluster_ratio': float(clus[ridx]),
           'cluster_null_mean': float(clus.mean()), 'p_cluster': p_clus, 'loco_auc': real_auc,
           'loco_null_mean': float(np.nanmean(null)), 'loco_null_95': float(np.nanpercentile(null, 95)), 'p_loco': p_cls,
           'heldout': {names[i]: float(sc[i]) for i in range(n)}, 'test_scores': tsc,
           'features': rows, 'top_labelings': top_sub,
           'top_coef': [(feats[j], float(coef[j])) for j in topc]}
    json.dump(out, open(os.path.join(RES, f'x1_analysis_{a.regime}{a.tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
