"""pe17: random-hypothesis classifier ensemble (plateau vs Susa), cross-fitted.

One hypothesis = (feature families, random feature subset, model type, regularisation).
For each hypothesis:
  * validation: leave-group-out (Malyan / Yahya / Other plateau) vs an out-of-fold Susa block -> AUC per group
  * Susa scores: 5-fold cross-fit, all (training) plateau tablets as positives -> out-of-fold score per Susa tablet
  * plateau scores: each plateau tablet scored by the model that did not see its group
Susa negatives are weighted so that their length-bin distribution matches the positives' (size control).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from pe17_common import auc

GROUPS = ['Malyan', 'Yahya', 'Other']


def group_of(site):
    return site if site in ('Malyan', 'Yahya') else ('Susa' if site == 'Susa' else 'Other')


def weights(ypos_bins, yneg_bins):
    """weights for negatives so their bin mix matches the positives'"""
    nb = 6
    pp = np.bincount(ypos_bins, minlength=nb) / max(len(ypos_bins), 1)
    pn = np.bincount(yneg_bins, minlength=nb) / max(len(yneg_bins), 1)
    w = np.where(pn[yneg_bins] > 0, pp[yneg_bins] / np.maximum(pn[yneg_bins], 1e-12), 0)
    return w / w.sum() * len(ypos_bins)


def fit_score(kind, C, Xp, Xn, wn, Xs):
    if kind == 'cent':
        mp = Xp.mean(0)
        mn = (Xn * wn[:, None]).sum(0) / wn.sum()
        sd = np.sqrt(0.5 * (Xp.var(0) + ((Xn - mn) ** 2 * wn[:, None]).sum(0) / wn.sum())) + 0.25
        return Xs @ ((mp - mn) / sd)
    if kind == 'nb':
        a = 0.5
        pp = (Xp.clip(0, 1).sum(0) + a) / (len(Xp) + 2 * a)
        pn = ((Xn.clip(0, 1) * wn[:, None]).sum(0) + a) / (wn.sum() + 2 * a)
        w1 = np.log(pp / pn) - np.log((1 - pp) / (1 - pn))
        return Xs.clip(0, 1) @ w1
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    sw = np.r_[np.ones(len(Xp)), wn]
    m = LogisticRegression(C=C, penalty='l1' if kind == 'lr1' else 'l2', solver='liblinear', max_iter=200)
    m.fit(X, y, sample_weight=sw)
    return m.decision_function(Xs)


def draw_hyp(rng, fam):
    fams = np.array(['SIGN', 'VAR', 'NUM', 'FMT', 'BIG'])
    while True:
        use = fams[rng.random(5) < 0.45]
        if len(use):
            break
    pool = np.where(np.isin(fam, use))[0]
    k = int(min(len(pool), rng.integers(3, 81)))
    cols = np.sort(rng.choice(pool, k, replace=False))
    kind = rng.choice(['cent', 'nb', 'lr2', 'lr1'], p=[0.3, 0.3, 0.25, 0.15])
    C = float(10 ** rng.uniform(-2, 1))
    return dict(fams=list(use), cols=cols, kind=str(kind), C=C)


def run_hyp(h, X, grp, lbin, folds):
    """grp: array of 'Susa'/'Malyan'/'Yahya'/'Other'; folds: Susa fold id (0..4) or -1"""
    Xh = X[:, h['cols']]
    susa = np.where(grp == 'Susa')[0]
    plat = np.where(grp != 'Susa')[0]
    out = {'val': {}, 'susa': np.full(len(X), np.nan), 'plat': np.full(len(X), np.nan)}
    for gi, g in enumerate(GROUPS):
        held = np.where(grp == g)[0]
        if len(held) == 0:
            continue
        tr_p = np.setdiff1d(plat, held)
        te_s = susa[folds[susa] == gi]
        tr_s = susa[folds[susa] != gi]
        wn = weights(lbin[tr_p], lbin[tr_s])
        sc = fit_score(h['kind'], h['C'], Xh[tr_p], Xh[tr_s], wn, Xh[np.r_[held, te_s]])
        out['val'][g] = auc(sc[:len(held)], sc[len(held):])
        # z-score against the out-of-fold Susa block so plateau scores are comparable across groups
        ss = sc[len(held):]
        out['plat'][held] = (sc[:len(held)] - ss.mean()) / (ss.std() + 1e-9)
    for f in range(5):
        te = susa[folds[susa] == f]
        tr = susa[folds[susa] != f]
        wn = weights(lbin[plat], lbin[tr])
        sc = fit_score(h['kind'], h['C'], Xh[plat], Xh[tr], wn, Xh[te])
        trs = fit_score(h['kind'], h['C'], Xh[plat], Xh[tr], wn, Xh[tr[:200]])
        out['susa'][te] = (sc - trs.mean()) / (trs.std() + 1e-9)
    return out


def ensemble(X, fam, grp, lbin, H, seed, top=0.10):
    rng = np.random.default_rng(seed)
    susa = np.where(grp == 'Susa')[0]
    folds = np.full(len(X), -1)
    folds[susa] = rng.permutation(np.arange(len(susa)) % 5)
    hyps, VAL, SS, PL = [], [], [], []
    for i in range(H):
        h = draw_hyp(rng, fam)
        o = run_hyp(h, X, grp, lbin, folds)
        hyps.append(h)
        VAL.append([o['val'].get(g, np.nan) for g in GROUPS])
        SS.append(o['susa'])
        PL.append(o['plat'])
    VAL = np.array(VAL); SS = np.array(SS); PL = np.array(PL)
    sel = np.nanmean(VAL[:, :2], 1)
    ns = max(1, int(top * H))
    surv = np.argsort(-sel)[:ns]
    return dict(hyps=hyps, VAL=VAL, SS=SS, PL=PL, surv=surv, sel=sel)
