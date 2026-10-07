"""pe83 cycle 1a: per-tablet photo-kind, lighting, perspective and alternative corner metrics (incl. ink-proof ones and a
planted header-band artefact). -> data/pe83_ckpt/c1_feats.json"""
import sys, os, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P
from scipy import ndimage


def measure(pid):
    try:
        A = P.load_img(pid)
    except Exception:
        return pid, None
    g = A.mean(2); H, W = g.shape
    f = {}
    brd = np.concatenate([g[:3].ravel(), g[-3:].ravel(), g[:, :3].ravel(), g[:, -3:].ravel()])
    f['bg'] = float(np.median(brd)); f['sat'] = float(np.abs(A[..., 0] - A[..., 2]).mean()); f['Hpx'] = H; f['Wpx'] = W
    B = P.blobs(g)
    f['nblob'] = len(B)
    ob = P.pick_obverse(B, H)
    if ob is None:
        return pid, None
    sl, m = ob; h, w = m.shape
    f['touch'] = float(sl[0].start <= 1 or sl[1].start <= 1 or sl[0].stop >= H - 1 or sl[1].stop >= W - 1)
    f['rect'] = float(m.mean())
    # a second big blob at the same height beside the obverse that is not an edge view (edge views are narrow)
    side = [b for b in B if b is not ob and abs((b[0][0].start + b[0][0].stop) / 2 - (sl[0].start + sl[0].stop) / 2) < h * 0.25]
    f['side_wide'] = float(any((b[0][1].stop - b[0][1].start) > 0.6 * w for b in side))
    cf = P.corner_metrics(m)
    f['cf'] = float(np.mean(cf)); f['cf_top'] = float((cf[0] + cf[1]) / 2); f['cf_bot'] = float((cf[2] + cf[3]) / 2)
    # lighting
    sub = g[sl]; k = max(2, int(0.15 * min(h, w)))
    cen = sub[h // 3: 2 * h // 3, w // 3: 2 * w // 3][m[h // 3: 2 * h // 3, w // 3: 2 * w // 3]].mean()
    cb = []
    for ys, xs in ((slice(0, k), slice(0, k)), (slice(0, k), slice(w - k, w)), (slice(h - k, h), slice(0, k)), (slice(h - k, h), slice(w - k, w))):
        mm = m[ys, xs]
        cb.append(sub[ys, xs][mm].mean() / cen if mm.any() else np.nan)
    f['corner_light'] = float(np.nanmean(cb)) if np.isfinite(cb).any() else None
    f['grad_tb'] = float((sub[: h // 2][m[: h // 2]].mean() - sub[h // 2:][m[h // 2:]].mean()) / cen)
    f['grad_lr'] = float((sub[:, : w // 2][m[:, : w // 2]].mean() - sub[:, w // 2:][m[:, w // 2:]].mean()) / cen)
    f['bright'] = float(sub[m].mean())
    # rim shadow: fraction of dark pixels in a 3px ring just inside the mask
    er = ndimage.binary_erosion(m, iterations=3)
    ring = m & ~er
    f['rim_dark'] = float((sub[ring] < 70).mean())
    # perspective
    rows = m.sum(1)
    wt, wm, wb = rows[int(0.1 * h)], rows[h // 2], rows[min(h - 1, int(0.9 * h))]
    f['keystone'] = float((wt - wb) / max(1, wm))
    f['lr_asym'] = float(abs(m[:, : w // 2].sum() - m[:, w - w // 2:].sum()) / m.sum())
    ys, xs = np.nonzero(m); C = np.cov(np.vstack([xs, ys]))
    ev, evec = np.linalg.eigh(C); v = evec[:, 1]
    ang = np.degrees(np.arctan2(v[1], v[0])) % 180
    f['tilt'] = float(min(abs(ang - 90), abs(ang), abs(ang - 180)))
    # ink-proof variants
    f['cf_close6'] = float(np.mean(P.corner_metrics(P.close_mask(m))))
    hf = P.hull_fill(m)
    f['cf_hull'] = float(np.mean(hf)) if hf else None
    f['cf_hull_bot'] = float((hf[2] + hf[3]) / 2) if hf else None
    rv = P.pick_reverse(B, H, ob)
    if rv is not None:
        rc = P.corner_metrics(rv[1]); f['cf_rev'] = float(np.mean(rc)); f['rev_aspect'] = rv[1].shape[0] / rv[1].shape[1]
    else:
        f['cf_rev'] = None; f['rev_aspect'] = None
    # edge views: narrow blobs left/right of the obverse
    ev_ = [b for b in B if b is not ob and b[0][0].start < sl[0].stop and b[0][0].stop > sl[0].start and
           (b[0][1].stop <= sl[1].start + 3 or b[0][1].start >= sl[1].stop - 3)]
    f['cf_edges'] = float(np.mean([np.mean(P.corner_metrics(b[1], 0.2)) for b in ev_])) if ev_ else None
    # planted header-band artefact: darken top 12% of obverse clay to 10 and re-measure
    g2 = g.copy(); band = int(0.12 * h)
    tgt = np.zeros_like(m); tgt[:band] = m[:band]
    g2[sl][tgt] = 10.0
    B2 = P.blobs(g2); ob2 = P.pick_obverse(B2, H)
    if ob2 is not None:
        c2 = P.corner_metrics(ob2[1]); f['pl_cf'] = float(np.mean(c2)); f['pl_bot'] = float((c2[2] + c2[3]) / 2)
        f['pl_close6'] = float(np.mean(P.corner_metrics(P.close_mask(ob2[1]))))
        h2 = P.hull_fill(ob2[1]); f['pl_hull'] = float(np.mean(h2)) if h2 else None
    return pid, f


if __name__ == '__main__':
    R = P.rows()
    ids = [r['id'] for r in R if os.path.exists(os.path.join(P.TN, r['id'] + '.jpg'))]
    print('tablets', len(R), 'with thumbnails', len(ids), flush=True)
    with Pool(2) as pool:
        res = dict(pool.map(measure, ids, chunksize=20))
    json.dump(res, open(os.path.join(P.CK, 'c1_feats.json'), 'w'))
    print('measured', sum(1 for v in res.values() if v))
