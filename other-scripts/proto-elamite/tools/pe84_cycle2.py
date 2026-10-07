"""pe84 cycle 2: STYLUS CENSUS. Is the stroke-width / relief spectrum a stylus fingerprint, and do tablets written with
'the same stylus' share vocabulary beyond find lot, volume and size?
 2a meter check: obverse vs reverse of one tablet closer than two tablets of the same photo batch and volume (rank AUC).
 2b 3,000 random stylus metrics (random subsets of 12 spectrum features, random weights, raw or rank space); per metric a
    pair-level partial correlation (stylus distance vs vocabulary Jaccard distance, partialled on size gap, line-count gap,
    museum-number gap, same volume, brightness gap, s_orig gap); select top 2% on tablet half A (pairs inside A), re-test
    on half B. Null: vocabularies permuted within strata (20 runs, same metrics). Planted: 25 'scribes' = groups of 4
    stylus-nearest tablets each given 2 shared private signs; must replicate.
usage: python3 pe84_cycle2.py feats.json -> data/pe84_ckpt/c2.json"""
import sys, os, json, re
import numpy as np
from scipy.stats import rankdata, norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe84_common as C
import common

rng = np.random.default_rng(842)
SPEC = ('E04', 'E08', 'E16', 'G1', 'G2', 'G3', 'G4', 'wid', 'depth', 'coh', 'diag', 'ncomp')


def vocab(t):
    return {common.base(g) for l in t['lines'] for g in l['signs'] if common.is_sign(g)}


def spec_matrix(R, face='ob'):
    X = np.array([[r['ph'].get('%s_%s' % (face, k)) if r['ph'].get('%s_%s' % (face, k)) is not None else np.nan
                   for k in SPEC] for r in R], float)
    # spectrum shape: granulometry normalised by its sum (scale-free), relief ratios
    g = X[:, 3:7]; X[:, 3:7] = g / np.nansum(g, 1, keepdims=True)
    return X


def zscore(X):
    m = np.nanmean(X, 0); s = np.nanstd(X, 0) + 1e-9
    Z = (X - m) / s
    Z[~np.isfinite(Z)] = 0
    return Z


def meter_check(R):
    have = [i for i, r in enumerate(R) if 'rv_E08' in r['ph']]
    Ob = zscore(spec_matrix([R[i] for i in have], 'ob')); Rv = zscore(spec_matrix([R[i] for i in have], 'rv'))
    batch = np.array([R[i]['batch'] + R[i]['vol'] for i in have])
    wins = []
    for j in range(len(have)):
        same = np.linalg.norm(Ob[j] - Rv[j])
        pool = [k for k in range(len(have)) if k != j and batch[k] == batch[j]]
        if len(pool) < 5:
            continue
        others = np.linalg.norm(Ob[j] - Rv[pool], axis=1)
        wins.append((others > same).mean())
    return dict(n=len(wins), auc=round(float(np.mean(wins)), 3))


def pairs_for(idx, R):
    I, J = np.triu_indices(len(idx), 1)
    return np.array(idx)[I], np.array(idx)[J]


def run_metrics(R, V, metrics, pairsets, covs, K=5):
    """per metric and half: k-NN vocabulary lift = mean covariate-residualised Jaccard similarity between each tablet and
    its K stylus-nearest tablets in the same half, as a z against random neighbours."""
    out = []
    Xob = zscore(spec_matrix(R, 'ob'))
    Xr = zscore(np.apply_along_axis(rankdata, 0, Xob))
    RJ, IDX, SD = {}, {}, {}
    for h, (I, J) in pairsets.items():
        sim = np.array([len(V[i] & V[j]) / max(1, len(V[i] | V[j])) for i, j in zip(I, J)])
        Q = np.linalg.qr(covs[h])[0]
        res = sim - Q @ (Q.T @ sim)
        idx = np.unique(np.concatenate([I, J])); pos = {v: k for k, v in enumerate(idx)}
        M = np.zeros((len(idx), len(idx)))
        a = np.array([pos[v] for v in I]); b = np.array([pos[v] for v in J])
        M[a, b] = res; M[b, a] = res
        RJ[h] = M; IDX[h] = idx; SD[h] = res.std() / np.sqrt(len(idx) * K)
    for (sub, w, rank) in metrics:
        X = (Xr if rank else Xob)[:, sub] * w
        rs = []
        for h in pairsets:
            Xh = X[IDX[h]]
            D = ((Xh[:, None, :] - Xh[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(D, np.inf)
            nn = np.argpartition(D, K, axis=1)[:, :K]
            lift = np.take_along_axis(RJ[h], nn, 1).mean()
            rs.append(float(lift / SD[h]))
        out.append(rs)
    return np.array(out)


def pair_covs(R, I, J):
    la = np.array([r['la'] for r in R]); nl = np.array([r['nl'] for r in R]); br = np.array([r['bright'] for r in R])
    so = np.array([r['s_orig'] for r in R])
    sb = np.array([r['sb'] if r['sb'] is not None else np.nan for r in R], float)
    vol = np.array([r['vol'] for r in R]); bat = np.array([r['batch'] for r in R])
    nv = np.array([len(x) for x in VOC])
    sbg = np.abs(sb[I] - sb[J]); sbg = np.where(np.isfinite(sbg), np.log1p(sbg), np.nanmedian(np.log1p(sbg)))
    return np.column_stack([np.ones(len(I)), np.abs(la[I] - la[J]), la[I] + la[J], np.abs(nl[I] - nl[J]), nl[I] + nl[J],
                            sbg, (vol[I] == vol[J]).astype(float), (bat[I] == bat[J]).astype(float), np.abs(br[I] - br[J]), np.abs(so[I] - so[J]),
                            np.log1p(nv[I]) + np.log1p(nv[J]), np.abs(np.log1p(nv[I]) - np.log1p(nv[J]))])


def score(rs, n_sel):
    """replications: top 2% by lift z in the selection half, re-tested on the other half (z > 1.645)."""
    rep = 0
    for sel, tst in ((0, 1), (1, 0)):
        cut = np.quantile(rs[:, sel], 0.98)
        rep += int(((rs[:, sel] >= cut) & (rs[:, tst] > 1.645)).sum())
    return int(rep)


VOC = None


def main():
    global VOC
    R = C.rows(sys.argv[1])
    R = [r for r in R if r['ph'].get('ob_E08') is not None and r['tx']['n_distinct'] >= 3]
    VOC = [vocab(r['t']) for r in R]
    out = {'n': len(R), 'meter': meter_check(R)}
    print('meter', out['meter'], flush=True)
    half = np.array([r['half'] for r in R])
    pairsets, covs, ntab = {}, {}, []
    for h in 'AB':
        idx = list(np.where(half == h)[0])
        I, J = pairs_for(idx, R)
        pairsets[h] = (I, J); covs[h] = pair_covs(R, I, J); ntab.append(len(idx))
    M = int(os.environ.get('PE84_M', 3000)); NN = int(os.environ.get('PE84_NULLS', 20))
    metrics = []
    for _ in range(M):
        k = rng.integers(2, len(SPEC) + 1)
        sub = np.sort(rng.choice(len(SPEC), size=k, replace=False))
        metrics.append((sub, rng.uniform(0.2, 1.0, size=k), bool(rng.integers(0, 2))))
    rs = run_metrics(R, VOC, metrics, pairsets, covs)
    out['real'] = dict(rep=score(rs, ntab), medA=round(float(np.median(rs[:, 0])), 4), medB=round(float(np.median(rs[:, 1])), 4),
                       maxA=round(float(rs[:, 0].max()), 4), maxB=round(float(rs[:, 1].max()), 4))
    # feature enrichment among the top 5% (both halves)
    top = np.where((rs[:, 0] >= np.quantile(rs[:, 0], .95)) & (rs[:, 1] >= np.quantile(rs[:, 1], .5)))[0]
    out['real']['top_feature_share'] = {SPEC[f]: round(float(np.mean([f in metrics[k][0] for k in top])), 2) for f in range(len(SPEC))}
    print('real', out['real'], flush=True)
    # null: vocabularies permuted within strata
    s = C.strata(R); nulls = []
    for k in range(NN):
        p = C.permute_within(s, rng)
        VOCp = [VOC[i] for i in p]
        VOC_save = VOC; VOC = VOCp
        covp = {h: pair_covs(R, *pairsets[h]) for h in 'AB'}
        VOC = VOC_save
        rsn = run_metrics(R, VOCp, metrics, pairsets, covp)
        nulls.append(dict(rep=score(rsn, ntab), medA=round(float(np.median(rsn[:, 0])), 4), medB=round(float(np.median(rsn[:, 1])), 4)))
        print('null', k, nulls[-1], flush=True)
    out['null'] = nulls
    # planted scribes: 25 groups of 4 stylus-nearest tablets (full spectrum), 2 private signs each
    X = zscore(spec_matrix(R, 'ob')); used = set(); VOCp = [set(v) for v in VOC]
    order = rng.permutation(len(R)); g = 0
    for i in order:
        if g >= 25 or i in used:
            continue
        d = np.linalg.norm(X - X[i], axis=1); d[list(used | {i})] = np.inf
        nb = [i] + list(np.argsort(d)[:3])
        if any(half[j] != half[i] for j in nb):
            nb = [i] + [j for j in np.argsort(d) if half[j] == half[i] and j not in used][:3]
        for j in nb:
            used.add(j); VOCp[j] |= {'PL%da' % g, 'PL%db' % g}
        g += 1
    VOC_save = VOC; VOC = VOCp
    covp = {h: pair_covs(R, *pairsets[h]) for h in 'AB'}
    VOC = VOC_save
    rsp = run_metrics(R, VOCp, metrics, pairsets, covp)
    out['planted'] = dict(rep=score(rsp, ntab), medA=round(float(np.median(rsp[:, 0])), 4), medB=round(float(np.median(rsp[:, 1])), 4))
    print('planted', out['planted'], flush=True)
    json.dump(out, open(os.path.join(C.CK, 'c2.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
