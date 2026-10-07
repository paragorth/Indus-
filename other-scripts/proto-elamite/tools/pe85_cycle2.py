"""pe85 cycle 2: structure tests (neighbours, dossiers, seal units, header families, tag face) + frozen outside findspot test."""
import sys, os, json, re
import numpy as np
from scipy.stats import kruskal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe85_common as C
import pe83_common as P
import common
from pe85_cycle1 import setup, corr

rng = np.random.default_rng(852)
out = {}


def group_sim(y, groups, vol, B=5000, planted=None):
    """mean |y_i - y_j| within groups (>=2) vs groups re-drawn at random from the same volume, same sizes."""
    gid = [g for g in set(groups) if g is not None and sum(1 for x in groups if x == g) >= 2]
    idx = {g: np.where(np.array([x == g for x in groups]))[0] for g in gid}
    def stat(v):
        d = []
        for g, m in idx.items():
            vv = v[m]; d += [abs(a - b) for i, a in enumerate(vv) for b in vv[i + 1:]]
        return float(np.mean(d))
    s0 = stat(y); cnt = 0
    byvol = {v: np.where(vol == v)[0] for v in set(vol)}
    for _ in range(B):
        d = []
        for g, m in idx.items():
            vs = vol[m[0]]; pool = byvol[vs]
            mm = rng.choice(pool, len(m), replace=False) if len(pool) >= len(m) else rng.choice(len(y), len(m), replace=False)
            vv = y[mm]; d += [abs(a - b) for i, a in enumerate(vv) for b in vv[i + 1:]]
        cnt += np.mean(d) <= s0
    return dict(groups=len(gid), tablets=int(sum(len(m) for m in idx.values())), within=round(s0, 3), p_more_similar=round((cnt + 1) / (B + 1), 4))


def neighbour(y, R, vol, B=2000):
    """correlation of y between publication-order neighbours inside a volume; null: order shuffled within volume."""
    order = {}
    for v in set(vol):
        ii = [i for i in range(len(R)) if vol[i] == v and np.isfinite(R[i]['pub'])]
        order[v] = sorted(ii, key=lambda i: R[i]['pub'])
    def stat(perm):
        a, b = [], []
        for v, ii in perm.items():
            for i, j in zip(ii[:-1], ii[1:]):
                a.append(y[i]); b.append(y[j])
        return corr(np.array(a), np.array(b))
    s0 = stat(order); null = []
    for _ in range(B):
        null.append(stat({v: list(rng.permutation(ii)) for v, ii in order.items()}))
    return dict(r=round(s0, 3), null_q95=round(float(np.quantile(null, .95)), 3), p=round((np.sum(np.array(null) >= s0) + 1) / (B + 1), 4))


for intact in (True, False):
    tg = 'intact' if intact else 'full'
    R, hd, Z, A, vol, strata = setup(intact)
    y = C.resid(C.col(R, 'cf_top'), Z); yb = C.resid(C.col(R, 'cf_bot'), Z)
    y = (y - np.nanmean(y)) / np.nanstd(y); yb = (yb - np.nanmean(yb)) / np.nanstd(yb)
    res = {'n': len(R)}
    # 2a neighbours (filing batches / joints)
    res['neighbour_top'] = neighbour(y, R, vol); res['neighbour_bot'] = neighbour(yb, R, vol)
    yp = y + 0.5 * np.repeat(rng.normal(size=len(R) // 4 + 1), 4)[:len(R)][np.argsort(np.argsort([(r['vol'], r['pub']) for r in R], axis=0)[:, 0])] if False else None
    # planted: neighbours share a block effect of 0.4 sd in blocks of 4 consecutive tablets
    pl = y.copy()
    for v in set(vol):
        ii = sorted([i for i in range(len(R)) if vol[i] == v and np.isfinite(R[i]['pub'])], key=lambda i: R[i]['pub'])
        for k in range(0, len(ii), 4):
            pl[ii[k:k + 4]] += 0.4 * rng.normal()
    res['neighbour_planted'] = neighbour(pl, R, vol, B=500)
    # 2b dossiers and seal units
    doss = [r['doss'] for r in R]; units = [r['seal_unit'] if r['seal_unit'] and not str(r['seal_unit']).startswith('T:') else None for r in R]
    res['dossier_top'] = group_sim(y, doss, vol); res['dossier_bot'] = group_sim(yb, doss, vol)
    res['sealunit_top'] = group_sim(y, units, vol); res['sealunit_bot'] = group_sim(yb, units, vol)
    pld = y.copy()
    for d in set(x for x in doss if x):
        m = [i for i, x in enumerate(doss) if x == d]; pld[m] = pld[m] * 0.3 + rng.normal()
    res['dossier_planted'] = group_sim(pld, doss, vol, B=1000)
    # 2c header families (office style): among headed tablets, does top residual vary by first header sign beyond random groupings?
    H = [((common.header(r['t']) or ['-'])[0]) for r in R]
    fam = np.array([h if hd[i] == 1 else None for i, h in enumerate(H)], object)
    big = [h for h in set(fam[fam != None]) if (fam == h).sum() >= 8]
    def kw(v, labels):
        gs = [v[(labels == h)] for h in big]
        return kruskal(*gs).statistic
    lab = np.array([h if h in big else None for h in fam], object); m = lab != None
    s0 = kw(y[m], lab[m]); null = [kw(y[m], rng.permutation(lab[m])) for _ in range(2000)]
    res['header_family'] = dict(families=big, n=int(m.sum()), H=round(float(s0), 2), p=round((np.sum(np.array(null) >= s0) + 1) / 2001, 4),
                                means={h: round(float(y[lab == h].mean()), 3) for h in big})
    # header effect size by family relative to headless baseline
    res['header_family_vs_headless'] = {h: round(float(y[lab == h].mean() - y[hd == 0].mean()), 3) for h in big}
    # 2d tag face: edge views (pe83 c1d topedge_rect / topedge_end_fill) by tag, among headed tablets
    for k in ('topedge_rect', 'topedge_end_fill', 'botedge_rect'):
        v = C.col(R, k); mm = (hd == 1) & np.isfinite(v)
        tagv = C.col(R, 'tag')[mm]
        if tagv.sum() >= 5:
            d0 = v[mm][tagv == 1].mean() - v[mm][tagv == 0].mean()
            nl = [np.mean(v[mm][p == 1]) - np.mean(v[mm][p == 0]) for p in (rng.permutation(tagv) for _ in range(5000))]
            res['tagface_' + k] = dict(n_tag=int(tagv.sum()), n=int(mm.sum()), diff=round(float(d0), 3), p_two=round(float((np.sum(np.abs(nl) >= abs(d0)) + 1) / 5001), 4))
    # tag effect on top residual among headed, and within M157
    for sub, mm in (('headed', hd == 1), ('M157', C.col(R, 'h157') == 1)):
        tagv = C.col(R, 'tag')[mm]; v = y[mm]
        d0 = v[tagv == 1].mean() - v[tagv == 0].mean()
        nl = [np.mean(v[p == 1]) - np.mean(v[p == 0]) for p in (rng.permutation(tagv) for _ in range(5000))]
        res['tag_top_' + sub] = dict(n_tag=int(tagv.sum()), n=int(mm.sum()), diff_sd=round(float(d0), 3), p_two=round(float((np.sum(np.abs(nl) >= abs(d0)) + 1) / 5001), 4))
    out[tg] = res
    print(tg, json.dumps(res, indent=0), flush=True)

# 2e frozen outside test: findspots (line art)
cat = P.catalogue(); LA = json.load(open(os.path.join(P.CK, 'lineart_feats.json'))); T = {t['id']: t for t in common.load()}
S = []
for k, r in cat.items():
    if r['findspot_square'] and LA.get(k) and k in T and T[k]['lines']:
        t = T[k]; l1 = t['lines'][0]
        site = 'Malyan' if 'Mal' in r['provenience'] else 'Susa'
        sq = re.sub(r'\s*\(.*', '', r['findspot_square']).replace('-', '').replace(' ', '')
        S.append(dict(id=k, site=site, sq=sq, lev=r['stratigraphic_level'], hd=common.header(t) is not None, ok=not l1['lacuna'] and not l1['damaged'],
                      top=LA[k]['la_cf_top'], bot=LA[k]['la_cf_bot']))
o = {'n': len(S), 'frozen_sha': __import__('hashlib').sha256(open(os.path.join(C.CK, 'frozen_outside.json'), 'rb').read()).hexdigest()}
ok = [s for s in S if s['ok']]
g1 = [s['top'] - s['bot'] for s in ok if s['hd']]; g0 = [s['top'] - s['bot'] for s in ok if not s['hd']]
gap = float(np.mean(g1) - np.mean(g0)); pool = np.array(g1 + g0); nl = []
for _ in range(10000):
    p = rng.permutation(pool); nl.append(p[:len(g1)].mean() - p[len(g1):].mean())
o['O1'] = dict(n_headed=len(g1), n_unheaded=len(g0), gap=round(gap, 3), p_one=round(float((np.sum(np.array(nl) >= gap) + 1) / 10001), 4))
top = np.array([s['top'] for s in S]); site = np.array([s['site'] for s in S]); sq = np.array([s['sq'] for s in S])
def same_diff(lbl):
    a, b = [], []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if site[i] != site[j]:
                continue
            (a if lbl[i] == lbl[j] else b).append(abs(top[i] - top[j]))
    return np.mean(a) - np.mean(b), len(a)
d0, npairs = same_diff(sq); nl = []
for _ in range(10000):
    l = sq.copy()
    for s_ in set(site):
        m = np.where(site == s_)[0]; l[m] = l[rng.permutation(m)]
    nl.append(same_diff(l)[0])
o['O2'] = dict(same_pairs=npairs, same_minus_diff=round(float(d0), 3), p_one=round(float((np.sum(np.array(nl) <= d0) + 1) / 10001), 4))
lev = np.array([s['lev'] or '?' for s in S]); kk = {}
for s_ in set(site):
    m = (site == s_) & (lev != '?')
    gs = [top[m & (lev == L)] for L in set(lev[m]) if (m & (lev == L)).sum() >= 2]
    if len(gs) >= 2:
        kk[s_] = round(float(kruskal(*gs).pvalue), 3)
o['O3'] = kk
out['outside'] = o
print(json.dumps(o, indent=0))
json.dump(out, open(os.path.join(C.CK, 'c2.json'), 'w'), indent=1)
