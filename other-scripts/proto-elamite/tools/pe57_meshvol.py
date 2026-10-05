"""pe57: interior (brim-full) capacity of a vessel from a 3D mesh (GLB/OBJ/PLY).
Method: find the rotation axis among the PCA axes (the one whose height slices are most circular),
orient the mouth up (the end whose extreme slice has the larger radius), find the interior floor
(highest vertex near the axis), then integrate the cavity cross-section from the floor to the rim:
per height bin and per 10-degree sector, the inner wall is the smallest radius among wall vertices;
area = sum 0.5 r^2 dtheta. Missing sectors are filled from neighbours. Returns litres if the mesh
is in metres. Also returns rim diameter, height, a wall-thickness check and a circularity score.
"""
import numpy as np, trimesh, sys, json

def load_vertices(path):
    s = trimesh.load(path, force='scene')
    vs = []
    for name in s.graph.nodes_geometry:
        T, g = s.graph[name]
        geo = s.geometry[g]
        if not hasattr(geo, 'vertices') or len(geo.vertices) == 0:
            continue
        v = np.asarray(geo.vertices, float)
        v = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        vs.append(v)
    return np.vstack(vs)

def circularity(v, axis, c):
    h = (v - c) @ axis
    P = (v - c) - np.outer(h, axis)
    r = np.linalg.norm(P, axis=1)
    bins = np.linspace(h.min(), h.max(), 21)
    cv = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (h >= a) & (h < b)
        if m.sum() < 50: continue
        rr = r[m]; q = np.percentile(rr, 90)
        rr = rr[rr > 0.6 * q]
        if len(rr) > 20: cv.append(np.std(rr) / np.mean(rr))
    return np.median(cv) if cv else 9.

def capacity(v, nb=80, ns=36):
    c0 = v.mean(0)
    _, _, vt = np.linalg.svd(v - c0, full_matrices=False)
    best = None
    for ax in vt:
        # recentre on axis: centre = mean of the upper-percentile radius ring approx -> use bbox centre in plane
        sc = circularity(v, ax, c0)
        if best is None or sc < best[0]:
            best = (sc, ax)
    circ, ax = best
    # refine centre: least-squares circle centre of the rim-region points
    h = (v - c0) @ ax
    # orient mouth up: compare radius near both ends
    P = (v - c0) - np.outer(h, ax)
    e1 = np.cross(ax, [1, 0, 0]);
    if np.linalg.norm(e1) < 1e-3: e1 = np.cross(ax, [0, 1, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    x, y = P @ e1, P @ e2
    lo, hi = np.percentile(h, [1, 99])
    def ring_r(sel):
        return np.percentile(np.hypot(x[sel], y[sel]), 95) if sel.sum() > 10 else 0
    rlo = ring_r(h < lo + 0.05 * (hi - lo)); rhi = ring_r(h > hi - 0.05 * (hi - lo))
    if rlo > rhi:
        h = -h; x = -x  # flip (keep right-handed)
    # recentre x, y with a circle fit on mid-height points
    H0, H1 = h.min(), h.max()
    mid = (h > H0 + 0.3 * (H1 - H0)) & (h < H0 + 0.9 * (H1 - H0))
    A = np.c_[2 * x[mid], 2 * y[mid], np.ones(mid.sum())]
    sol, *_ = np.linalg.lstsq(A, x[mid] ** 2 + y[mid] ** 2, rcond=None)
    x = x - sol[0]; y = y - sol[1]
    r = np.hypot(x, y); th = np.arctan2(y, x)
    R = np.percentile(r[h > H1 - 0.05 * (H1 - H0)], 95)  # rim outer radius
    # interior floor: highest vertex within 0.25 R of the axis
    near = r < 0.25 * R
    if near.sum() < 10:
        return None
    floor = np.percentile(h[near], 99.5)
    base_bottom = np.percentile(h[near], 0.5)
    rim = np.percentile(h, 99.7)
    if rim - floor < 0.15 * (H1 - H0):
        return None
    edges = np.linspace(floor, rim, nb + 1)
    sect = np.linspace(-np.pi, np.pi, ns + 1)
    vol = 0.
    wall = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (h >= a) & (h < b)
        rin = np.full(ns, np.nan)
        for k in range(ns):
            mm = m & (th >= sect[k]) & (th < sect[k + 1])
            if mm.sum() >= 2:
                rr = r[mm]
                rin[k] = np.min(rr)
                wall.append(np.max(rr) - np.min(rr))
        if np.all(np.isnan(rin)):
            continue
        # fill gaps from neighbours (circular)
        idx = np.arange(ns); ok = ~np.isnan(rin)
        rin = np.interp(idx, np.r_[idx[ok] - ns, idx[ok], idx[ok] + ns], np.r_[rin[ok], rin[ok], rin[ok]])
        vol += 0.5 * np.sum(rin ** 2) * (2 * np.pi / ns) * (b - a)
    return {'vol_m3': float(vol), 'litres_if_m': float(vol * 1000), 'rim_diam': float(2 * R),
            'height': float(H1 - H0), 'depth': float(rim - floor), 'base_thick': float(floor - base_bottom),
            'wall_med': float(np.median(wall)) if wall else None, 'circ': float(circ), 'nverts': int(len(v))}

def cone_check():
    """synthetic truncated cone shell (r 0.09 at top, 0.045 at base, depth 0.09, wall 0.01, base 0.015)."""
    pts = []
    rng = np.random.default_rng(0)
    for _ in range(60000):
        t = rng.uniform(); th = rng.uniform(-np.pi, np.pi); z = 0.015 + 0.09 * t
        ri = 0.045 + 0.045 * t; pts.append([ri * np.cos(th), ri * np.sin(th), z]); pts.append([(ri + 0.01) * np.cos(th), (ri + 0.01) * np.sin(th), z])
    for _ in range(20000):
        rr = 0.045 * np.sqrt(rng.uniform()); th = rng.uniform(-np.pi, np.pi)
        pts.append([rr * np.cos(th), rr * np.sin(th), 0.015]); pts.append([1.1 * rr * np.cos(th), 1.1 * rr * np.sin(th), 0.0])
    v = np.array(pts)
    true = np.pi * 0.09 / 3 * (0.09 ** 2 + 0.09 * 0.045 + 0.045 ** 2)
    return capacity(v), true * 1000

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        print(cone_check())
    else:
        for p in sys.argv[1:]:
            print(p, json.dumps(capacity(load_vertices(p))))
