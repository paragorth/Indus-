"""v38 cycle 6: read the text off the picture. Ridge regression from the plant embedding
(top-20 PCs of ResNet-18 + DINO) to the page's word profile (tf-idf of the 300 commonest mid-
frequency words), fitted leave-one-out with every page within 3 binding positions or on the
same bifolio also removed from training. Page-level confounds (quire, language, hand, log
length; drawing area) are regressed out of both sides first. Score: for each page, the rank of
its own text among same-stratum pages >= 3 positions away, by correlation with the prediction
(0.5 = chance). Null: visual vectors permuted within strata, model refitted (200 perms).
Controls: Gerard (same pipeline), planted Voynich (6%), page-shuffled Voynich.
Writes data/v38_ckpt/c6.json and loops/v38_cycle6.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
from v38_cycle1 import voynich_setup
from v38_cycle2 import gerard_setup

NPERM = int(os.environ.get('NPERM', 200))


def design(cats, cont):
    cols = [np.ones(len(cont[0]))]
    for c in cats:
        u = sorted(set(map(str, c)))
        for x in u[1:]:
            cols.append(np.array([str(v) == x for v in c], float))
    cols += [np.asarray(v, float) for v in cont]
    return np.column_stack(cols)


def resid(M, X):
    return M - X @ np.linalg.lstsq(X, M, rcond=None)[0]


def text_matrix(words, k=300):
    df = Counter(w for ws in words for w in set(ws))
    n = len(words)
    voc = [w for w, c in df.most_common() if 3 <= c <= 0.5 * n][:k]
    vi = {w: i for i, w in enumerate(voc)}
    M = np.zeros((n, len(voc)))
    for i, ws in enumerate(words):
        for w in ws:
            if w in vi:
                M[i, vi[w]] += 1
    M = np.log1p(M) * np.log(n / np.array([df[w] for w in voc]))
    return M


def score(Vf, Tm, order, groups, strata, lam=10.0):
    n = len(order)
    ranks = []
    for i in range(n):
        excl = (np.abs(order - order[i]) < 3) | (groups == groups[i])
        tr = ~excl
        A = Vf[tr]; B = Tm[tr]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)
        p = Vf[i] @ W
        cand = np.where((strata == strata[i]) & (np.abs(order - order[i]) >= 3))[0]
        if len(cand) < 5:
            continue
        cs = [np.corrcoef(p, Tm[j])[0, 1] for j in cand]
        own = np.corrcoef(p, Tm[i])[0, 1]
        ranks.append(np.mean(np.array(cs) < own))
    return float(np.mean(ranks))


def run(Vf, Tm, order, groups, strata, rng):
    obs = score(Vf, Tm, order, groups, strata)
    null = [score(Vf[perm(len(order), rng, list(strata))], Tm, order, groups, strata) for _ in range(NPERM)]
    null = np.array(null)
    return obs, float((obs - null.mean()) / null.std()), float((1 + (null >= obs).sum()) / (1 + NPERM))


def prep_vis(vis, keys, X):
    F = np.concatenate([np.array([vis[k][f] for k in keys], float) for f in ['r18', 'dino']], 1)
    F = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1)
    U, S, _ = np.linalg.svd(F - F.mean(0), full_matrices=False)
    V = resid(U[:, :20] * S[:20], X)
    return V / V.std(0)


if __name__ == '__main__':
    rng = np.random.default_rng(3806)
    rows, out = [], {}
    pages, keys, words, vis, conf, strata = voynich_setup()
    order = np.array([p['order'] for p in pages], float)
    groups = np.array(['%s-%s' % (p['quire'], p['bifolio']) for p in pages])
    st = np.array(['B' if s.startswith('B') else s for s in strata])
    X = design([[p['quire'] for p in pages], [p['lang'] for p in pages], [p['hand'] for p in pages]],
               [np.log([len(w) for w in words]), np.log([vis[k]['area'] + 1e-3 for k in keys])])
    Vf = prep_vis(vis, keys, X)
    Tm = resid(text_matrix(words), X)
    r = run(Vf, Tm, order, groups, st, rng)
    out['voynich'] = r
    allw = Counter(w for ws in words for w in ws)
    pool = [w for w, c in allw.items() if 5 <= c <= 40]
    Z = np.array([vis[k]['dino'] for k in keys], float)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pw = plant_text(words, U[:, :10] * S[:10], 0.06, rng, pool)
    rp = run(Vf, resid(text_matrix(pw), X), order, groups, st, rng)
    out['planted6'] = rp
    sp = rng.permutation(len(words))
    rs = run(Vf, resid(text_matrix([words[i] for i in sp]), X), order, groups, st, rng)
    out['shuffled'] = rs
    gk, gw, gvis, gconf = gerard_setup()
    go = np.array([int(k) for k in gk], float) / 8.
    gX = design([], [np.log([len(w) for w in gw]), np.log([gvis[k]['area'] for k in gk])])
    gr = run(prep_vis(gvis, gk, gX), resid(text_matrix(gw), gX), go, np.array([str(int(k) // 2) for k in gk]),
             np.array(['g'] * len(gk)), rng)
    out['gerard'] = gr
    f = lambda t: 'mean rank %.3f (chance 0.500), z %+.1f, p %.3f' % t
    rows.append(('V-38.24', 'READ THE TEXT OFF THE PICTURE: leave-one-out ridge from plant embedding (20 PCs, ResNet-18 + DINO) to the page\'s word profile (300 mid-frequency words), neighbours within 3 pages and the same bifolio left out of training, quire/language/hand/length/area regressed out; own text ranked among same-stratum pages >= 3 apart; null = visual vectors permuted within strata, refitted (%d)' % NPERM,
                 'Voynich: %s' % f(r), 'PASS' if r[2] < 0.01 else ('weak' if r[2] < 0.05 else 'no prediction')))
    rows.append(('V-38.25', 'Controls for V-38.24: Gerard (same pipeline); planted Voynich (6%% DINO-driven vocabulary); page-shuffled Voynich',
                 'Gerard: %s; planted: %s; shuffled: %s' % (f(gr), f(rp), f(rs)), ''))
    json.dump(out, open(os.path.join(CK, 'c6.json'), 'w'))
    with open(os.path.join(LOOPS, 'v38_cycle6.txt'), 'w') as fh:
        fh.write('# v38 cycle 6 - predict a page\'s words from its picture (held-out, leave-one-out)\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for x in rows:
            fh.write('| %s | %s | %s | %s |\n' % x)
    print(open(os.path.join(LOOPS, 'v38_cycle6.txt')).read())
