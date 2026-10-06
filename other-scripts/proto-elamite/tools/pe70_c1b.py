"""pe70 cycle 1b: are the sealed-tablet separators a few dossiers, a volume, or the archive?

For each cycle-1 separator: z of sealed association (a) all tablets, (b) one tablet per seal id
(duplicates of the same seal dropped), (c) without MDP 26S, (d) only tablets with dimensions and not
fragments.  p from 5,000 label permutations within volume x size band (Mantel-Haenszel-like stratified
count).  Plus leave-one-volume-out AUC of the full logistic model vs the same with labels permuted.
usage: pe70_c1b.py -> data/pe70_ckpt/c1b.json
"""
import json, os, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pe70_common import get, CK
from pe70_feats import matrix, strata, perm_within, lor_z, num_z
import warnings
warnings.filterwarnings('ignore')

FEATS = ['S:|M153+M342|', 'H:M157', 'S:M157', 'S:M288', 'E:M388', 'S:M340', 'S:M388', 'S:M391', 'E:M288', 'Y:C',
         'Y:C@', 'Y:DEC', 'Y:SEX', 'N:num_share', 'N:single_share', 'N:has_hdr', 'N:frag', 'N:ent_len', 'N:log_lines',
         'N:reverse', 'N:log_area', 'N:log_area_missing']


def stat(X, names, y, st, rng, nperm=5000):
    out = {}
    for f in FEATS:
        if f not in names:
            continue
        x = X[:, names.index(f)]
        binary = set(np.unique(x)) <= {0, 1}
        if binary:
            obs = float((x * y).sum())
            exp_ = sum(x[st == s].mean() * y[st == s].sum() for s in np.unique(st))
            nul = np.array([(x * perm_within(y, st, rng)).sum() for _ in range(nperm)])
            out[f] = dict(n_feat=int(x.sum()), sealed_with=int(obs), expected=round(float(exp_), 1),
                          p_hi=float((nul >= obs).mean()), p_lo=float((nul <= obs).mean()))
        else:
            obs = float(x[y == 1].mean())
            nul = np.array([x[perm_within(y, st, rng) == 1].mean() for _ in range(nperm // 5)])
            out[f] = dict(sealed_mean=round(obs, 3), null_mean=round(float(nul.mean()), 3),
                          p_hi=float((nul >= obs).mean()), p_lo=float((nul <= obs).mean()))
    return out


def lovo(R, X, y, rng, nperm=20):
    vols = np.array([r['vol'] for r in R])
    keep = [v for v, c in collections.Counter(vols[y == 1]).items() if c >= 10]

    def auc(yy):
        sc, ys = [], []
        for v in keep:
            te = vols == v; tr = ~te
            m = LogisticRegression(C=0.1, max_iter=300, class_weight='balanced').fit(X[tr], yy[tr])
            sc.append(roc_auc_score(yy[te], m.decision_function(X[te])))
        return float(np.mean(sc)), [round(s, 3) for s in sc]
    real, per = auc(y)
    st = strata(R, 'volband')
    nul = [auc(perm_within(y, st, rng))[0] for _ in range(nperm)]
    return dict(vols=keep, auc=real, per_vol=per, null=float(np.mean(nul)), null_sd=float(np.std(nul)),
                p=float((np.array(nul) >= real).mean()))


def main():
    rng = np.random.default_rng(704)
    R = get('pe')
    Xb, B, Xn, NN = matrix(R, 'pe')
    X = np.hstack([Xb, Xn]); names = B + NN
    y = np.array([r['sealed'] for r in R], int)
    res = {}
    seen = set(); keep1 = []
    for i, r in enumerate(R):
        if r['seals'] and seen & set(r['seals']):
            continue
        seen |= set(r['seals']); keep1.append(i)
    subsets = {'all': list(range(len(R))), 'one_per_seal': keep1,
               'no_26S': [i for i, r in enumerate(R) if r['vol'] != 'MDP 26S'],
               'dims_nonfrag': [i for i, r in enumerate(R) if r['area'] and r['pres'] != 'fragment'],
               'no_seal_groups': [i for i, r in enumerate(R) if not r['seals'] or i in keep1 and not any(
                   sum(s in rr['seals'] for rr in R) >= 2 for s in r['seals'])]}
    for nm, ix in subsets.items():
        ix = np.array(ix)
        st = strata([R[i] for i in ix], 'volband')
        res[nm] = dict(n=len(ix), sealed=int(y[ix].sum()), feats=stat(X[ix], names, y[ix], st, rng))
        print(nm, res[nm]['n'], res[nm]['sealed'], {f: (v.get('sealed_with', v.get('sealed_mean')), v.get('expected', v.get('null_mean')),
                                                       round(min(v['p_hi'], v['p_lo']), 4)) for f, v in res[nm]['feats'].items()}, flush=True)
    res['lovo'] = lovo(R, X, y, rng)
    print('LOVO', res['lovo'], flush=True)
    json.dump(res, open(os.path.join(CK, 'c1b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
