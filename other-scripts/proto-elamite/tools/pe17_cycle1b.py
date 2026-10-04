"""pe17 cycle 1b: per-site affinity of every Susa tablet (itinerant-scribe fingerprint), frozen blind.

For each plateau site s in (Malyan, Yahya, Sialk, Sofalin): 800 random hypotheses 'site s vs Susa'
(5-fold cross-fitted over Susa, length-weighted negatives). Affinity_s(tablet) = mean z-score.
Validation per site: leave-one-tablet-out AUC of site s tablets vs out-of-fold Susa (does the site have
its own fingerprint at all?). Control: site labels permuted among plateau tablets (pooled-plateau
fingerprint only; then the per-site AUC for the TRUE site should drop).
Distance decay: mean of the top-1% Susa affinities per site vs distance from Susa.
Run before the hXRF table was read; output hashed into data/pe17_frozen_affinity.json.
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe17_common import load, build, length_bin, auc, sha, DATA, DIST_KM
from pe17_engine import draw_hyp, fit_score, weights

tabs = load()
X, names, fam = build(tabs)
site = np.array([t['site'] for t in tabs])
lbin = np.array([length_bin(t) for t in tabs])
ids = [t['id'] for t in tabs]
SITES = ['Malyan', 'Yahya', 'Sialk', 'Sofalin']
H = int(os.environ.get('H', 800))


def site_run(args):
    s, lab, seed = args
    rng = np.random.default_rng(seed)
    sus = np.where(lab == 'Susa')[0]
    pos = np.where(lab == s)[0]
    folds = rng.permutation(np.arange(len(sus)) % 5)
    AFF = np.zeros((H, len(X))); VAL = np.zeros(H)
    for k in range(H):
        h = draw_hyp(rng, fam)
        kind = h['kind'] if h['kind'] in ('cent', 'nb') else 'cent'
        Xh = X[:, h['cols']]
        # LOO over site tablets, against Susa fold 0
        te_s = sus[folds == 0]; tr_s = sus[folds != 0]
        wn = weights(lbin[pos], lbin[tr_s])
        lo = []
        for i in range(len(pos)):
            tr_p = np.delete(pos, i)
            wn2 = weights(lbin[tr_p], lbin[tr_s])
            lo.append(fit_score(kind, 1.0, Xh[tr_p], Xh[tr_s], wn2, Xh[[pos[i]]])[0])
        ss = fit_score(kind, 1.0, Xh[pos], Xh[tr_s], wn, Xh[te_s])
        VAL[k] = auc(lo, ss)
        for f in range(5):
            te = sus[folds == f]; tr = sus[folds != f]
            wn = weights(lbin[pos], lbin[tr])
            sc = fit_score(kind, 1.0, Xh[pos], Xh[tr], wn, Xh[te])
            AFF[k, te] = (sc - sc.mean()) / (sc.std() + 1e-9)
    top = np.argsort(-VAL)[:max(1, H // 10)]
    return s, seed, VAL, AFF[top].mean(0), AFF.mean(0)


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(s, site, 5 + i) for i, s in enumerate(SITES)]
    rng = np.random.default_rng(99)
    plat = np.where(site != 'Susa')[0]
    for r in range(3):
        lab = site.copy(); lab[plat] = site[rng.permutation(plat)]
        jobs += [(s, lab, 100 + 10 * r + i) for i, s in enumerate(SITES)]
    with Pool(2) as pool:
        out = pool.map(site_run, jobs, chunksize=1)
    res = {'real': {}, 'perm': {}}
    aff = {}
    for j, (s, seed, VAL, A_top, A_all) in enumerate(out):
        key = 'real' if j < len(SITES) else 'perm'
        res[key].setdefault(s, []).append(dict(val_mean=float(np.nanmean(VAL)), val_top10=float(np.nanmean(np.sort(VAL)[-H // 10:]))))
        if key == 'real':
            aff[s] = A_top
    sus = np.where(site == 'Susa')[0]
    decay = {s: float(np.mean(np.sort(aff[s][sus])[-15:])) for s in SITES}
    frozen = dict(note='pe17 per-site affinity of Susa tablets (z vs Susa); frozen before reading hXRF table',
                  susa={ids[i]: {s: round(float(aff[s][i]), 4) for s in SITES} for i in sus})
    frozen['sha256_of_content_without_this_field'] = sha(frozen)
    json.dump(frozen, open(os.path.join(DATA, 'pe17_frozen_affinity.json'), 'w'))
    out2 = dict(res=res, top15_affinity=decay, dist=DIST_KM, sha256=frozen['sha256_of_content_without_this_field'],
                n_site={s: int((site == s).sum()) for s in SITES}, secs=time.time() - t0)
    json.dump(out2, open(os.path.join(DATA, 'pe17_cycle1b.json'), 'w'), indent=1)
    print(json.dumps(out2, indent=1))
