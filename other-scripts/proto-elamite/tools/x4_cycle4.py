"""X-4 cycle 4: forgery battery (Voynich v21 adapted). Can a generator fitted to the corpus be told from it?

For every corpus: 30 real 100-word samples (3,000-word subsample) vs 30 samples of each fitted generator
(TRI sign trigram, SLOT independent words, CPV copy-and-vary, WBG word bigram; x4_feats.py), 32 v31 features.
Discriminators: logistic regression (C 0.5) and random forest (200 trees, depth 4), 5-fold stratified CV x 10
repeats, AUC. Floor: generator run a vs independent run b (same generator, same fit): should be ~0.5.
Ceiling check: a language should be caught by every generator; a generated text by none of its own kind.
Also the random-feature version (300 random 4-feature subsets, logistic): median and 90th-percentile AUC, so a
single dominant feature cannot carry the verdict.
"""
import os, sys, json, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X

GENS = ['TRI', 'SLOT', 'CPV', 'WBG']


def auc_cv(A, B, kind, reps=10, cols=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    Xm = np.vstack([A, B]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    if cols is not None: Xm = Xm[:, cols]
    out = []
    for r in range(reps):
        sc = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Xm, y):
            mu = Xm[tr].mean(0); sd = Xm[tr].std(0); sd[sd == 0] = 1
            m = (LogisticRegression(C=0.5, max_iter=2000) if kind == 'lr' else
                 RandomForestClassifier(200, max_depth=4, random_state=r, n_jobs=1))
            m.fit((Xm[tr] - mu) / sd, y[tr]); sc[te] = m.predict_proba((Xm[te] - mu) / sd)[:, 1]
        out.append(roc_auc_score(y, sc))
    return float(np.mean(out))


def main():
    F = X.load('feats.json'); keys = sorted(F[0]['F'])
    by = defaultdict(list)
    for r in F: by[(r['corpus'], r['cond'])].append([r['F'][k] for k in keys])
    by = {k: np.nan_to_num(np.array(v, float)) for k, v in by.items()}
    res = X.load('c4.json') or {}
    rng = random.Random(4)
    subsets = [sorted(rng.sample(range(len(keys)), 4)) for _ in range(300)]
    for name in X.LIST + ['VOY'] + X.PROSE:
        if name in res: continue
        R = {}
        real = by[(name, 'real')]
        for g in GENS:
            a, b = by[(name, g + '_a')], by[(name, g + '_b')]
            R[g] = {'lr': auc_cv(real, a, 'lr'), 'rf': auc_cv(real, a, 'rf'),
                    'floor_lr': auc_cv(a, b, 'lr'), 'floor_rf': auc_cv(a, b, 'rf')}
            rs = [auc_cv(real, a, 'lr', reps=2, cols=c) for c in subsets[:100]]
            R[g]['rand4_med'] = float(np.median(rs)); R[g]['rand4_p90'] = float(np.percentile(rs, 90))
        res[name] = R; X.save('c4.json', res)
        print(name, len(real), {g: (round(v['lr'], 2), round(v['rf'], 2), round(v['floor_rf'], 2), round(v['rand4_med'], 2))
                                for g, v in R.items()}, flush=True)


if __name__ == '__main__':
    main()
