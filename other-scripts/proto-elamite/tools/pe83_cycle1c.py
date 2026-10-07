"""pe83 C1.5: 400 random segmentation pipelines x all tablets; header partial r per pipeline vs stratified-permutation nulls."""
import sys, os, json
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P

NPIPE = 200
rng0 = np.random.default_rng(8315)
PIPES = [dict(ch=int(rng0.integers(4)), thr=float(rng0.uniform(15, 90)), blur=float(rng0.choice([0, 0.7, 1.2, 2.0])),
              op=int(rng0.integers(0, 4)), cl=int(rng0.integers(0, 7)), c=float(rng0.uniform(0.08, 0.25)),
              metric=str(rng0.choice(['box', 'chamfer']))) for _ in range(NPIPE)]


def metric(m, c, kind):
    h, w = m.shape
    if kind == 'box':
        return float(np.mean(P.corner_metrics(m, c)))
    if kind == 'hull':
        hf = P.hull_fill(m, c)
        return float(np.mean(hf)) if hf else np.nan
    # chamfer: distance from each bbox corner to the nearest mask pixel, relative to min side (negated so high = square)
    dt = ndimage.distance_transform_edt(~m)
    d = [dt[0, 0], dt[0, -1], dt[-1, 0], dt[-1, -1]]
    return float(-np.mean(d) / min(h, w))


def run(pid):
    A = P.load_img(pid); g0 = A.mean(2); H, W = g0.shape
    B = P.blobs(g0); ob = P.pick_obverse(B, H)
    if ob is None:
        return pid, None
    sl = ob[0]; h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    y0, y1 = max(0, sl[0].start - int(.12 * h)), min(H, sl[0].stop + int(.12 * h))
    x0, x1 = max(0, sl[1].start - int(.12 * w)), min(W, sl[1].stop + int(.12 * w))
    crop = A[y0:y1, x0:x1]
    chans = [crop.mean(2), crop[..., 0], crop[..., 1], crop[..., 2]]
    cy, cx = (sl[0].start + sl[0].stop) / 2 - y0, (sl[1].start + sl[1].stop) / 2 - x0
    out = []
    for p in PIPES:
        g = chans[p['ch']]
        if p['blur']:
            g = ndimage.gaussian_filter(g, p['blur'])
        m = g > p['thr']
        if p['op']:
            m = ndimage.binary_opening(m, iterations=p['op'])
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m)
        if n == 0:
            out.append((np.nan, np.nan)); continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab == k)
        mm = (lab == k)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if p['cl']:
            mm = P.close_mask(mm, p['cl'])
        if mm.shape[0] < 20 or mm.shape[1] < 20:
            out.append((np.nan, np.nan)); continue
        out.append((metric(mm, p['c'], p['metric']), mm.shape[0] / mm.shape[1]))
    return pid, out


if __name__ == '__main__':
    R = P.rows()
    ck = os.path.join(P.CK, 'c1c_raw.json')
    if os.path.exists(ck):
        res = json.load(open(ck))
    else:
        part = ck + '.part'
        res = json.load(open(part)) if os.path.exists(part) else {}
        todo = [r['id'] for r in R if r['id'] not in res]
        with Pool(2) as pool:
            for i, (pid, o) in enumerate(pool.imap_unordered(run, todo, chunksize=5)):
                res[pid] = o
                if i % 100 == 99:
                    json.dump(res, open(part, 'w')); print('done', len(res), flush=True)
        json.dump(res, open(ck, 'w'))
    R = [r for r in R if res.get(r['id'])]
    M = np.array([[v[0] for v in res[r['id']]] for r in R]); AS = np.array([[v[1] for v in res[r['id']]] for r in R])
    hd = np.array([r['hd'] for r in R]); Z = P.design(R); s = P.strata(Z); rng = np.random.default_rng(8316)
    catasp = np.exp(np.array([r['asp'] for r in R]))
    NN = 20
    perms = []
    for _ in range(NN):
        hp = hd.copy()
        for k in np.unique(s):
            ix = np.where(s == k)[0]; hp[ix] = rng.permutation(hp[ix])
        perms.append(hp)
    rows = []
    for j, p in enumerate(PIPES):
        y = M[:, j]
        if np.isfinite(y).sum() < 600:
            continue
        r = P.partial(y, hd, Z)['r']
        nul = [P.partial(y, hp, Z)['r'] for hp in perms]
        ok = np.isfinite(AS[:, j])
        cred = spearmanr(AS[ok, j], catasp[ok]).correlation
        rows.append(dict(p, r=r, null_max=float(np.max(np.abs(nul))), cred=round(float(cred), 3), n=int(np.isfinite(y).sum())))
    rr = np.array([x['r'] for x in rows]); cr = np.array([x['cred'] for x in rows]); nm = np.array([x['null_max'] for x in rows])
    good = cr > 0.85
    summ = dict(pipelines=len(rows), credible=int(good.sum()), r_median=round(float(np.median(rr)), 3),
                r_q05=round(float(np.quantile(rr, .05)), 3), r_q95=round(float(np.quantile(rr, .95)), 3),
                credible_r_median=round(float(np.median(rr[good])), 3), credible_r_min=round(float(rr[good].min()), 3),
                share_below_0_10=round(float((rr < 0.10).mean()), 3), credible_share_below_0_10=round(float((rr[good] < 0.10).mean()), 3),
                negative=int((rr < 0).sum()), null_absmax_median=round(float(np.median(nm)), 3),
                beats_own_null=round(float((rr > nm).mean()), 3),
                by_metric={k: round(float(np.median(rr[np.array([x['metric'] == k for x in rows])])), 3) for k in ('box', 'chamfer')},
                worst5=sorted(rows, key=lambda x: x['r'])[:5])
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(P.CK, 'c1c.json'), 'w'), indent=1)
