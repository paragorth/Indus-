"""pe83 C1.4 follow-up: where is the square corner? reverse-face corners split by physical end, top-edge view, ink darkness
of the top strip. -> c1d_feats.json + analysis c1d.json"""
import sys, os, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P


def measure(pid):
    A = P.load_img(pid); g = A.mean(2); H, W = g.shape
    B = P.blobs(g); ob = P.pick_obverse(B, H)
    if ob is None:
        return pid, None
    sl, m = ob; h, w = m.shape; f = {}
    sub = g[sl]; k = max(2, int(0.12 * h))
    top = sub[:k][m[:k]]; bot = sub[-k:][m[-k:]]; mid = sub[h // 3: 2 * h // 3][m[h // 3: 2 * h // 3]]
    f['top_dark'] = float((top < 0.6 * np.median(mid)).mean()); f['bot_dark'] = float((bot < 0.6 * np.median(mid)).mean())
    f['top_lum'] = float(top.mean() / mid.mean()); f['bot_lum'] = float(bot.mean() / mid.mean())
    # mask edge roughness along the top and bottom (ink nicks cut the mask): count of mask-boundary direction changes
    def rough(rows):
        prof = np.array([np.argmax(m[:, j]) if m[:, j].any() else np.nan for j in range(w)]) if rows == 'top' else \
               np.array([h - 1 - np.argmax(m[::-1, j]) if m[:, j].any() else np.nan for j in range(w)])
        pr = prof[int(.2 * w): int(.8 * w)]
        return float(np.nanstd(np.diff(pr)))
    f['top_rough'] = rough('top'); f['bot_rough'] = rough('bot')
    rv = P.pick_reverse(B, H, ob)
    if rv is not None:
        c = P.corner_metrics(rv[1])
        # reverse image: upper corners sit next to the bottom edge view; in CDLI fat cross the reverse is turned about the
        # horizontal axis, so its upper image corners are the obverse BOTTOM end and its lower image corners the obverse TOP end
        f['rev_img_upper'] = float((c[0] + c[1]) / 2); f['rev_img_lower'] = float((c[2] + c[3]) / 2)
    # top edge view (blob directly above obverse): fill of its two end boxes (slab end vs lens end)
    tp = [b for b in B if b[0][0].stop <= sl[0].start + 3 and b[0][1].start < sl[1].stop and b[0][1].stop > sl[1].start]
    if tp:
        tm = max(tp, key=lambda b: b[1].sum())[1]
        th, tw = tm.shape; kk = max(2, int(0.25 * th))
        f['topedge_end_fill'] = float(np.mean([tm[:, :kk].mean(), tm[:, -kk:].mean()])); f['topedge_rect'] = float(tm.mean())
    bt = [b for b in B if b[0][0].start >= sl[0].stop - 3 and b[0][1].start < sl[1].stop and b[0][1].stop > sl[1].start]
    if bt:
        # first blob below obverse = bottom edge view (smaller than reverse)
        bb = min(bt, key=lambda b: b[0][0].start)
        bm = bb[1]
        if bm.sum() < 0.6 * m.sum():
            bh, bw = bm.shape; kk = max(2, int(0.25 * bh))
            f['botedge_end_fill'] = float(np.mean([bm[:, :kk].mean(), bm[:, -kk:].mean()])); f['botedge_rect'] = float(bm.mean())
    return pid, f


if __name__ == '__main__':
    R = P.rows()
    with Pool(2) as pool:
        res = dict(pool.map(measure, [r['id'] for r in R], chunksize=20))
    json.dump(res, open(os.path.join(P.CK, 'c1d_feats.json'), 'w'))
    F1 = json.load(open(os.path.join(P.CK, 'c1_feats.json')))
    R = [dict(r, **F1[r['id']], **res[r['id']]) for r in R if res.get(r['id']) and F1.get(r['id'])]
    hd = np.array([r['hd'] for r in R]); Z = P.design(R); rng = np.random.default_rng(8317)
    col = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in R], float)
    out = {}
    for k in ('rev_img_upper', 'rev_img_lower', 'topedge_end_fill', 'topedge_rect', 'botedge_end_fill', 'botedge_rect',
              'top_dark', 'bot_dark', 'top_lum', 'top_rough', 'bot_rough'):
        out[k] = P.partial(col(k), hd, Z, 1000, rng)
    ink = [np.nan_to_num(col(k)) for k in ('top_dark', 'top_lum', 'top_rough', 'bot_dark', 'bot_rough')]
    out['cf_with_ink_covariates'] = P.partial(col('cf'), hd, P.design(R, ink), 1000, rng)
    out['cf_top_with_ink_covariates'] = P.partial(col('cf_top'), hd, P.design(R, ink), 1000, rng)
    # cleanest top: the third of tablets with the least top-strip darkness and roughness
    sc = np.argsort(np.argsort(col('top_dark'))) + np.argsort(np.argsort(col('top_rough')))
    lo = sc <= np.quantile(sc, 1 / 3)
    out['cf_top_cleanest_top_third'] = P.partial(col('cf_top')[lo], hd[lo], Z[lo], 1000, rng)
    # orientation check: obverse top-vs-bottom asymmetry and reverse lower-vs-upper asymmetry correlate?
    a = col('cf_top') - col('cf_bot'); b = col('rev_img_lower') - col('rev_img_upper'); ok = np.isfinite(a) & np.isfinite(b)
    out['asym_obv_vs_rev_r'] = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(P.CK, 'c1d.json'), 'w'), indent=1)
