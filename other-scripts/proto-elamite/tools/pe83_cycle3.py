"""pe83 cycle 3 analysis: line-art outlines vs header; credibility; planted and shuffled controls."""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P

LA = json.load(open(os.path.join(P.CK, 'lineart_feats.json')))
F1 = json.load(open(os.path.join(P.CK, 'c1_feats.json')))
rng = np.random.default_rng(834)
R = [dict(r, **LA[r['id']]) for r in P.rows() if LA.get(r['id'])]
hd = np.array([r['hd'] for r in R]); Z = P.design(R)
col = lambda k, RR=R: np.array([np.nan if r.get(k) is None else r[k] for r in RR], float)
out = {'n': len(R), 'n_closed': int(col('closed').sum())}
out['cred_aspect_rho'] = round(float(spearmanr(np.log(col('la_aspect')), col('asp'), nan_policy='omit').correlation), 3)
both = [r for r in R if F1.get(r['id'])]
out['cred_vs_photo'] = {k: round(float(spearmanr(col(k, both), [F1[r['id']][p] for r in both], nan_policy='omit').correlation), 3)
                        for k, p in (('la_cf', 'cf'), ('la_cf_top', 'cf_top'), ('la_cf_bot', 'cf_bot'))}
for k in ('la_cf', 'la_cf_top', 'la_cf_bot', 'la_hull', 'la_hull_top', 'la_hull_bot'):
    out[k] = P.partial(col(k), hd, Z, 2000, rng)
good = (col('closed') == 1) & (np.abs(np.log(col('la_aspect')) - col('asp')) < 0.25)
out['credible_subset_n'] = int(good.sum())
for k in ('la_cf', 'la_cf_top', 'la_cf_bot', 'la_hull_top'):
    out['cred_' + k] = P.partial(col(k)[good], hd[good], Z[good], 2000, rng)
# drawings for tablets WITHOUT a CDLI photo-outline in pe81 (fully independent tablets)
nophoto = np.array([not F1.get(r['id']) for r in R])
out['no_photo_tablets_n'] = int(nophoto.sum())
# shuffled and planted
s = P.strata(Z); y = col('la_cf_top'); ok = np.isfinite(y); nul = []
for _ in range(1000):
    hp = hd.copy()
    for k in np.unique(s):
        m = np.where(s == k)[0]; hp[m] = rng.permutation(hp[m])
    nul.append(abs(P.partial(y, hp, Z)['r']))
out['shuffle_abs_r_q95'] = round(float(np.quantile(nul, .95)), 3)
from scipy.stats import rankdata
yy = np.where(ok, y, np.nanmedian(y)); rec = []
for _ in range(50):
    lat = rankdata(yy) / len(yy) + rng.normal(0, 0.45, len(yy)); fake = (lat > np.median(lat)).astype(float)
    rec.append(P.partial(y, fake, Z)['r'])
out['planted_recovered_r_median'] = round(float(np.median(rec)), 3)
lat = rankdata(yy) / len(yy) + rng.normal(0, 0.45, len(yy)); fake = (lat > np.median(lat)).astype(float)
out['planted_raw_r'] = round(float(np.corrcoef(rankdata(yy), fake)[0, 1]), 3)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(P.CK, 'c3.json'), 'w'), indent=1)
