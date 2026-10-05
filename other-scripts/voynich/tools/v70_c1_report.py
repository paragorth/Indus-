"""v70 cycle 1 report: calibrated language score for every ink alphabet.

Classifier: logistic regression on the battery (BFEATS + log effective alphabet size), trained on
synthetic-hand alphabets of six natural languages (label 1) vs three generators (label 0),
all alphabets (k-means and random). Leave-one-corpus-out accuracy is the calibration.
Held-out selection: the best alphabet per corpus is chosen on page-half A and scored on half B;
the same selection is done on the matched random-Voronoi nulls.
"""
import sys, os, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import CK, BFEATS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LANG = ['SL_la', 'SL_lac', 'SL_it', 'SL_es', 'SL_de', 'SL_cs']
GENS = ['SG_self', 'SG_grille', 'SG_mk3']
TEST = ['V', 'L', 'SV']


def fv(b):
    return [b[f] for f in BFEATS] + [np.log(b['keff'])]


def load(name):
    fn = os.path.join(CK, f'c1_{name}.json')
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    data = {n: load(n) for n in LANG + GENS + TEST}
    have = [n for n in data if data[n]]
    Xs, ys, gs = [], [], []
    for n in LANG + GENS:
        if not data[n]:
            continue
        for r in data[n]['rows']:
            for h in 'AB':
                Xs.append(fv(r[h])); ys.append(1 if n in LANG else 0); gs.append(n)
    X, y, g = np.array(Xs), np.array(ys), np.array(gs)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    out = {'loco': {}}
    for n in set(gs):
        clf = mk().fit(X[g != n], y[g != n])
        out['loco'][n] = float(clf.predict_proba(X[g == n])[:, 1].mean())
    clf = mk().fit(X, y)
    out['coef'] = dict(zip(BFEATS + ['logkeff'], map(float, clf[-1].coef_[0])))
    for n in have:
        rows = data[n]['rows']
        res = {}
        for kind in ('km', 'rnd'):
            R = [r for r in rows if r['kind'] == kind]
            pa = clf.predict_proba(np.array([fv(r['A']) for r in R]))[:, 1]
            pb = clf.predict_proba(np.array([fv(r['B']) for r in R]))[:, 1]
            i = int(np.argmax(pa))
            res[kind] = {'median': float(np.median((pa + pb) / 2)), 'q90': float(np.percentile((pa + pb) / 2, 90)),
                         'bestA_heldB': float(pb[i]), 'best': {k: R[i][k] for k in ('theta', 'K', 'seed', 'm')},
                         'frac_gt_half': float(np.mean((pa + pb) / 2 > 0.5))}
            if 'nmi' in R[0]:
                res[kind]['nmi_med'] = float(np.median([r['nmi'] for r in R]))
                res[kind]['nmi_best'] = float(max(r['nmi'] for r in R))
                lr = [r for r in R if 'lig_recall' in r]
                if lr:
                    res[kind]['lig_f'] = float(np.median([2 * r['lig_recall'] * r['lig_prec'] / max(1e-9, r['lig_recall'] + r['lig_prec']) for r in lr]))
                sp = [r['split_rebuilt'] for r in R if 'split_rebuilt' in r]
                if sp:
                    res[kind]['split_rebuilt'] = float(np.mean(sp))
        ref = data[n]['ref']
        res['ref_truth_text'] = float(clf.predict_proba(np.array([fv(ref['A']), fv(ref['B'])]))[:, 1].mean())
        res['ref_feats'] = {k: round(v, 3) for k, v in ref['B'].items()}
        out[n] = res
    json.dump(out, open(os.path.join(CK, 'c1_report.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
