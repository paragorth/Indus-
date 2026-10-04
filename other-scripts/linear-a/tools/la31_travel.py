"""LA-31 travel-time matrices: walking (Tobler on a 0.01 deg DEM), calm-water rowing, and
wind-driven sailing (square-sail boat polar on daily ERA5 winds, one Dijkstra voyage set per
day, 2014-2018), plus combined land-sea routes. Raw DEM / wind stay in the scratchpad; only the
derived site x site matrices are written to data/la31/travel.json.

Usage: python3 la31_travel.py [max_days]
"""
import os, sys, json, math, glob, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la31_common import SITES, OUT, CKPT, SCR  # noqa

RES_F = 0.01      # fine (walking) grid
BLK = 5           # sailing cell = 5 x 5 fine cells = 0.05 deg
EMBARK_H = 2.0    # loading / launching penalty at a port, hours
KMH_KN = 1.852


def load_dem():
    H = np.load(os.path.join(SCR, 'dem_001.npy')).astype(np.float64)
    m = json.load(open(os.path.join(SCR, 'dem_001.json')))
    return H, m['lon0'], m['lat0']


def grid_graph(mask, cost_fn, offsets):
    """Directed graph over True cells of mask; cost_fn(i0,j0,i1,j1,di,dj) -> hours (vector)."""
    idx = -np.ones(mask.shape, np.int64); ii, jj = np.nonzero(mask)
    idx[ii, jj] = np.arange(len(ii))
    rows, cols, w = [], [], []
    R, C = mask.shape
    for di, dj in offsets:
        i1 = ii + di; j1 = jj + dj
        ok = (i1 >= 0) & (i1 < R) & (j1 >= 0) & (j1 < C)
        a, b, c, d = ii[ok], jj[ok], i1[ok], j1[ok]
        ok2 = mask[c, d]
        a, b, c, d = a[ok2], b[ok2], c[ok2], d[ok2]
        cost = cost_fn(a, b, c, d, di, dj)
        g = np.isfinite(cost)
        rows.append(idx[a[g], b[g]]); cols.append(idx[c[g], d[g]]); w.append(cost[g])
    n = len(ii)
    return csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)), idx


OFF8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
OFF16 = OFF8 + [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def walking(H, lon0, lat0):
    land = H > 0
    lat = lat0 - np.arange(H.shape[0]) * RES_F
    kx = RES_F * 111.32 * np.cos(np.radians(lat)); ky = RES_F * 111.32

    def cost(a, b, c, d, di, dj):
        dist = np.hypot(dj * kx[a], di * ky)               # km
        s = (H[c, d] - H[a, b]) / 1000.0 / dist             # slope dh/dx
        v = 6.0 * np.exp(-3.5 * np.abs(s + 0.05))          # Tobler, km/h
        return dist / v
    G, idx = grid_graph(land, cost, OFF8)
    return G, idx, land


def site_cell(lat, lon, lat0, lon0, mask):
    i = int(round((lat0 - lat) / RES_F)); j = int(round((lon - lon0) / RES_F))
    if mask[i, j]:
        return i, j
    ii, jj = np.nonzero(mask[max(0, i - 30):i + 31, max(0, j - 30):j + 31])
    k = np.argmin((ii - min(i, 30)) ** 2 + (jj - min(j, 30)) ** 2)
    return max(0, i - 30) + ii[k], max(0, j - 30) + jj[k]


def wind_days():
    recs = []
    for f in sorted(glob.glob(os.path.join(SCR, 'wind', 'chunk_*.json'))):
        recs += json.load(open(f))
    P = np.array([[r['req'][1], r['req'][0]] for r in recs])     # lon, lat
    days = recs[0]['daily']['time']
    S = np.array([r['daily']['wind_speed_10m_mean'] for r in recs], float).T / KMH_KN   # knots, day x pt
    D = np.array([r['daily']['wind_direction_10m_dominant'] for r in recs], float).T     # FROM, deg
    S = np.nan_to_num(S); D = np.nan_to_num(D)
    # vector of the wind's ORIGIN direction (pointing to where it comes from)
    U = S * np.sin(np.radians(D)); V = S * np.cos(np.radians(D))
    months = np.array([int(t[5:7]) for t in days])
    return P, U, V, months


def sail_speed(heading_deg, wu, wv):
    """Square-sail boat polar with oar fallback. wu, wv: vector pointing to where the wind comes
    FROM (east, north components, knots). Returns knots."""
    tws = np.hypot(wu, wv)
    wfrom = np.degrees(np.arctan2(wu, wv))
    twa = np.abs((heading_deg - wfrom + 180) % 360 - 180)   # 0 = head to wind, 180 = dead run
    g = np.interp(twa, [0, 70, 100, 150, 180], [0, 0, 0.75, 1.0, 0.9])
    sail = np.minimum(0.4 * tws, 7.0) * g
    head = tws * np.cos(np.radians(twa))                     # headwind component (>0 against)
    oar = np.clip(2.0 - 0.15 * np.maximum(head, 0), 0.5, 2.0)
    v = np.maximum(sail, oar)
    v = np.where(tws > 27, np.minimum(v, 0.7), v)             # gale: mostly waiting
    return v


def main(max_days=None):
    t0 = time.time()
    H, lon0, lat0 = load_dem()
    codes = list(SITES)
    # ------------- walking
    Gw, idxw, land = walking(H, lon0, lat0)
    sc = {c: site_cell(SITES[c][1], SITES[c][2], lat0, lon0, land) for c in codes}
    src = [idxw[sc[c]] for c in codes]
    Tw_cells = dijkstra(Gw, directed=True, indices=src)       # hours, site -> every land cell
    K = len(codes)
    Twalk = np.array([[Tw_cells[a, src[b]] for b in range(K)] for a in range(K)])
    print('walking done', round(time.time() - t0), 's', flush=True)
    # walking time TO each site (reverse graph) for disembarking legs
    Tw_in = dijkstra(Gw.T.tocsr(), directed=True, indices=src)
    # ------------- sailing grid
    R, C = H.shape[0] // BLK, H.shape[1] // BLK
    Hb = H[:R * BLK, :C * BLK].reshape(R, BLK, C, BLK)
    seafrac = (Hb <= 0).mean(axis=(1, 3))
    sea = seafrac >= 0.6
    clat = lat0 - (np.arange(R) * BLK + (BLK - 1) / 2) * RES_F
    clon = lon0 + (np.arange(C) * BLK + (BLK - 1) / 2) * RES_F
    sidx = -np.ones((R, C), np.int64); si, sj = np.nonzero(sea); sidx[si, sj] = np.arange(len(si))
    ns = len(si)
    # port links: each site to coastal sea cells reachable on foot within 12 h (plus embark)
    coast = np.zeros_like(sea)
    for di, dj in OFF8:
        sh = np.roll(np.roll(~sea, di, 0), dj, 1); coast |= sea & sh
    ci, cj = np.nonzero(coast)
    # the land fine cell nearest to the centre of each coastal block
    port_land = []
    for a, b in zip(ci, cj):
        blk = land[a * BLK:(a + 1) * BLK + 3, max(0, b * BLK - 3):(b + 1) * BLK + 3]
        bi, bj = np.nonzero(blk)
        if len(bi) == 0:
            for r in (6, 10):
                blk = land[max(0, a * BLK - r):(a + 1) * BLK + r, max(0, b * BLK - r):(b + 1) * BLK + r]
                bi, bj = np.nonzero(blk)
                if len(bi):
                    bi = bi + max(0, a * BLK - r); bj = bj + max(0, b * BLK - r); break
        else:
            bi = bi + a * BLK; bj = bj + max(0, b * BLK - 3)
        if len(bi) == 0:
            port_land.append(-1); continue
        k = np.argmin((bi - (a * BLK + 2)) ** 2 + (bj - (b * BLK + 2)) ** 2)
        port_land.append(idxw[bi[k], bj[k]])
    port_land = np.array(port_land)
    pl_ok = port_land >= 0
    out_leg = np.full((K, len(ci)), np.inf); in_leg = np.full((K, len(ci)), np.inf)
    out_leg[:, pl_ok] = Tw_cells[:, port_land[pl_ok]] + EMBARK_H
    in_leg[:, pl_ok] = Tw_in[:, port_land[pl_ok]] + EMBARK_H
    out_leg[out_leg > 12 + EMBARK_H] = np.inf; in_leg[in_leg > 12 + EMBARK_H] = np.inf
    # site with no port within 12 h (should not happen): nearest coastal cell by distance
    for k in range(K):
        if not np.isfinite(out_leg[k]).any():
            d = (clat[ci] - SITES[codes[k]][1]) ** 2 + (clon[cj] - SITES[codes[k]][2]) ** 2
            j = np.argmin(d); out_leg[k, j] = in_leg[k, j] = EMBARK_H + 24
    print('sites without walk-port fallback handled; sea cells', ns, 'coast', len(ci), flush=True)
    cnode = sidx[ci, cj]
    # sea edges geometry
    kx = 0.05 * 111.32 * np.cos(np.radians(clat)); ky = 0.05 * 111.32
    E_a, E_b, E_dist, E_head = [], [], [], []
    for di, dj in OFF16:
        i1 = si + di; j1 = sj + dj
        ok = (i1 >= 0) & (i1 < R) & (j1 >= 0) & (j1 < C)
        ok[ok] &= sea[i1[ok], j1[ok]]
        # for knight moves require the two intermediate cells to be sea (no land cutting)
        if abs(di) == 2 or abs(dj) == 2:
            mi = np.clip(si + np.sign(di), 0, R - 1); mj = np.clip(sj + np.sign(dj), 0, C - 1)
            mi2 = np.clip(si + (di - np.sign(di) if abs(di) == 2 else 0), 0, R - 1)
            mj2 = np.clip(sj + (dj - np.sign(dj) if abs(dj) == 2 else 0), 0, C - 1)
            ok &= sea[mi, mj] & sea[mi2, mj2]
        a = sidx[si[ok], sj[ok]]; b = sidx[i1[ok], j1[ok]]
        dx = dj * kx[si[ok]]; dy = -di * ky
        E_a.append(a); E_b.append(b); E_dist.append(np.hypot(dx, dy) / KMH_KN)   # nautical miles
        E_head.append(np.degrees(np.arctan2(dx, dy)) % 360 + 0 * a)
    E_a = np.concatenate(E_a); E_b = np.concatenate(E_b)
    E_dist = np.concatenate(E_dist); E_head = np.concatenate(E_head)
    # virtual site nodes: ns + k (out), ns + K + k (in)
    N = ns + 2 * K
    vo_a, vo_b, vo_w, vi_a, vi_b, vi_w = [], [], [], [], [], []
    for k in range(K):
        g = np.isfinite(out_leg[k]); vo_a += [ns + k] * g.sum(); vo_b += list(cnode[g]); vo_w += list(out_leg[k, g])
        g = np.isfinite(in_leg[k]); vi_a += list(cnode[g]); vi_b += [ns + K + k] * g.sum(); vi_w += list(in_leg[k, g])
    VA = np.array(vo_a + vi_a); VB = np.array(vo_b + vi_b); VW = np.array(vo_w + vi_w)

    def solve(edge_hours):
        G = csr_matrix((np.concatenate([edge_hours, VW]), (np.concatenate([E_a, VA]), np.concatenate([E_b, VB]))),
                       shape=(N, N))
        d = dijkstra(G, directed=True, indices=np.arange(ns, ns + K))
        T = d[:, ns + K:ns + 2 * K]
        return T
    # calm-water rowing (2.5 kn), no wind
    Tcalm_sea = solve(E_dist / 2.5)
    # wind interpolation weights (barycentric) onto sea cells
    P, U, V, months = wind_days()
    tri = Delaunay(P)
    q = np.c_[clon[sj], clat[si]]
    s = tri.find_simplex(q)
    s = np.where(s < 0, tri.find_simplex(np.clip(q, P.min(0) + 1e-6, P.max(0) - 1e-6)), s)
    Tm = tri.transform[s]; bc = np.einsum('nij,nj->ni', Tm[:, :2], q - Tm[:, 2])
    W = np.c_[bc, 1 - bc.sum(1)]; verts = tri.simplices[s]
    nd = len(months) if max_days is None else min(max_days, len(months))
    dsel = np.arange(len(months)) if max_days is None else np.linspace(0, len(months) - 1, nd).astype(int)
    Tdays = np.zeros((len(dsel), K, K), np.float32)
    for n, d in enumerate(dsel):
        u = (U[d][verts] * W).sum(1); v = (V[d][verts] * W).sum(1)
        spd = sail_speed(E_head, u[E_a], v[E_a])
        Tdays[n] = solve(E_dist / spd)
        if n % 100 == 0:
            print('day', n, '/', len(dsel), round(time.time() - t0), 's', flush=True)
    np.save(os.path.join(CKPT, 'Tdays.npy'), Tdays)
    np.save(os.path.join(CKPT, 'Tdays_month.npy'), months[dsel])
    # combined = min(walking, sea-with-walking-legs)
    res = dict(codes=codes, walk=Twalk.tolist(), calm=np.minimum(Twalk, Tcalm_sea).tolist(),
               calm_sea_only=Tcalm_sea.tolist(), n_days=int(len(dsel)))
    seasons = dict(annual=range(1, 13), sailing=range(4, 11), etesian=(7, 8), spring=(4, 5, 6),
                   autumn=(9, 10), winter=(11, 12, 1, 2, 3))
    mo = months[dsel]
    for name, ms in seasons.items():
        sel = np.isin(mo, list(ms))
        med = np.median(Tdays[sel], 0)
        res['wind_' + name] = np.minimum(Twalk, med).tolist()
        res['windsea_' + name] = med.tolist()
        res['n_' + name] = int(sel.sum())
    for m in range(1, 13):
        res[f'windsea_m{m:02d}'] = np.median(Tdays[mo == m], 0).tolist()
    json.dump(res, open(os.path.join(OUT, 'travel.json'), 'w'))
    print('done', round(time.time() - t0), 's')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
