"""pe83 cycle 1b: analysis of photo-kind, lighting, perspective, ink-proof metrics and planted band."""
import sys, os, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P

F = json.load(open(os.path.join(P.CK, 'c1_feats.json')))
R = [dict(r, **F[r['id']]) for r in P.rows() if F.get(r['id'])]
rng = np.random.default_rng(831)
hd = np.array([r['hd'] for r in R])
col = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in R], float)
Z0 = P.design(R)
out = {'n': len(R), 'baseline_cf': P.partial(col('cf'), hd, Z0, 2000, rng)}
# photo kind
kind = []
for r in R:
    bgk = 'black' if r['bg'] < 30 else ('white' if r['bg'] > 200 else 'grey')
    kind.append(f"{bgk}/{'grey' if r['sat'] < 6 else 'col'}")
for r, k in zip(R, kind):
    r['kind'] = k
tab = {}
for k in sorted(set(kind)):
    s = [r for r in R if r['kind'] == k]
    tab[k] = dict(n=len(s), header_rate=round(np.mean([r['hd'] for r in s]), 3), median_cf=round(float(np.median([r['cf'] for r in s])), 3))
out['C1.1a_kind_table'] = tab
for key in ('batch', 'coll'):
    out[f'C1.1a_{key}_table'] = {k: dict(n=sum(r[key] == k for r in R), header_rate=round(np.mean([r['hd'] for r in R if r[key] == k]), 3),
                                         median_cf=round(float(np.median([r['cf'] for r in R if r[key] == k])), 3)) for k in sorted(set(r[key] for r in R))}
dummies = lambda key: [np.array([r[key] == k for r in R], float) for k in sorted(set(r[key] for r in R))[1:]]
photo_cov = dummies('kind') + dummies('batch') + dummies('coll') + [col('touch'), col('rect') > 0.97, col('Hpx'), col('Wpx'), col('nblob'), col('side_wide')]
Zp = P.design(R, photo_cov)
out['C1.1b_all_photo_covariates'] = P.partial(col('cf'), hd, Zp, 2000, rng)
clean = np.array([r['kind'] == 'black/col' and r['touch'] == 0 and r['rect'] <= 0.97 and r['side_wide'] == 0 for r in R])
out['C1.1c_homogeneous_black_colour_untouched'] = P.partial(col('cf')[clean], hd[clean], Z0[clean], 2000, rng)
for k in ('Louvre', 'Tehran'):
    s = np.array([r['coll'] == k for r in R])
    out[f'C1.1d_within_{k}'] = P.partial(col('cf')[s], hd[s], Z0[s], 2000, rng)
for k in sorted(set(r['batch'] for r in R)):
    s = np.array([r['batch'] == k for r in R])
    out[f'C1.1e_within_batch_{k}'] = P.partial(col('cf')[s], hd[s], Z0[s], 1000, rng)
# lighting
light = [col('corner_light'), col('grad_tb'), col('grad_lr'), col('bright'), col('rim_dark')]
ok = np.all([np.isfinite(x) for x in light], 0)
Zl = P.design(R, [np.nan_to_num(x) for x in light])
out['C1.2_lighting_covariates'] = P.partial(np.where(ok, col('cf'), np.nan), hd, Zl, 2000, rng)
out['C1.2_header_vs_light_r'] = {n: round(float(np.corrcoef(x[ok], hd[ok])[0, 1]), 3) for n, x in zip(('corner_light', 'grad_tb', 'grad_lr', 'bright', 'rim_dark'), light)}
# perspective
pers = [col('keystone'), col('lr_asym'), col('tilt')]
out['C1.3_perspective_covariates'] = P.partial(col('cf'), hd, P.design(R, pers), 2000, rng)
sc = np.abs(col('keystone')) + col('tilt') / 10
low = sc <= np.quantile(sc, 1 / 3)
out['C1.3_least_keystone_tilt_third'] = P.partial(col('cf')[low], hd[low], Z0[low], 2000, rng)
# ink
for k in ('cf_top', 'cf_bot', 'cf_hull', 'cf_hull_bot', 'cf_close6', 'cf_rev', 'cf_edges'):
    out[f'C1.4_{k}'] = P.partial(col(k), hd, Z0, 2000, rng)
# planted ink band: shift of each metric
pl = {}
for a, b in (('cf', 'pl_cf'), ('cf_bot', 'pl_bot'), ('cf_close6', 'pl_close6'), ('cf_hull', 'pl_hull')):
    d = col(b) - col(a); pl[a] = dict(mean_shift=round(float(np.nanmean(d)), 4), sd_cf=round(float(np.nanstd(col(a))), 4))
out['C1.4_planted_band_shift'] = pl
# planted: fake header = the band artefact: tablets where we 'paint' a band get pl_cf, others cf; label = painted
paint = rng.random(len(R)) < 0.5
MP = {'cf': 'pl_cf', 'cf_bot': 'pl_bot', 'cf_close6': 'pl_close6', 'cf_hull': 'pl_hull'}
out['C1.4_planted_band_as_label'] = {m: P.partial(np.where(paint, col(MP[m]), col(m)), paint.astype(float), Z0) for m in MP}
# C1.6 neighbours / crops
s = (col('touch') == 0) & (col('side_wide') == 0) & (col('rect') <= 0.97)
out['C1.6_no_touch_no_neighbour_no_tile'] = P.partial(col('cf')[s], hd[s], Z0[s], 2000, rng)
# everything at once
Zall = P.design(R, photo_cov + [np.nan_to_num(x) for x in light] + pers)
out['C1.ALL_covariates'] = P.partial(np.where(ok, col('cf'), np.nan), hd, Zall, 2000, rng)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(P.CK, 'c1b.json'), 'w'), indent=1)
