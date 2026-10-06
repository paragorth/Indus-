"""pe57 shared: vessel data sets, mode finding (KDE + Gaussian mixture on log volume, bootstrap),
and the frozen pe55 verdict rule."""
import csv, json, os
import numpy as np
CK = os.path.join(os.path.dirname(__file__), '..', 'data', 'pe57_ckpt')
WIN_A = (0.5, 1.0); WIN_B = (4.0, 9.0); SUPPORT = (2.8 / 1.2, 2.8 * 1.2)

def abu_salabikh(kind=None):
    rows = list(csv.DictReader(open(os.path.join(CK, 'vessels_published.csv'))))
    return np.array([float(r['volume_l']) for r in rows if kind is None or r['vessel'].endswith(kind)])

def mesh_set(tag, bowls_only=False, dmin=0.03, dmax=1.5):
    """3D-model capacities; drops unscaled models (rim diameter outside dmin-dmax metres) and failures."""
    d = json.load(open(os.path.join(CK, f'mesh_{tag}.json')))
    out = []
    for k, r in d.items():
        c = r['cap']
        if not c or 'error' in c or not (dmin <= c['rim_diam'] <= dmax):
            continue
        if bowls_only and 'bowl' not in r['title'].lower():
            continue
        out.append(c['litres_if_m'])
    return np.array(out)

def kde_modes(x, bw=None, grid=None, rel=0.1):
    lx = np.log10(x)
    n = len(lx)
    if bw is None:
        s = min(np.std(lx, ddof=1), (np.percentile(lx, 75) - np.percentile(lx, 25)) / 1.34)
        bw = max(0.9 * s * n ** -0.2, 0.02)
    g = np.linspace(-2, 2.5, 901) if grid is None else grid
    f = np.exp(-0.5 * ((g[:, None] - lx[None]) / bw) ** 2).sum(1)
    pk = [i for i in range(1, len(g) - 1) if f[i] > f[i - 1] and f[i] >= f[i + 1] and f[i] > rel * f.max()]
    return 10 ** g[pk], bw

def gmm_modes(x, kmax=4, seed=0):
    from sklearn.mixture import GaussianMixture
    lx = np.log10(x)[:, None]
    best = None
    for k in range(1, min(kmax, len(x) // 4) + 1):
        g = GaussianMixture(k, n_init=1, random_state=seed).fit(lx)
        b = g.bic(lx)
        if best is None or b < best[0]:
            best = (b, g)
    g = best[1]
    o = np.argsort(g.means_[:, 0])
    return 10 ** g.means_[o, 0], g.weights_[o]

def verdict(modes):
    inA = any(WIN_A[0] <= m <= WIN_A[1] for m in modes)
    inB = any(WIN_B[0] <= m <= WIN_B[1] for m in modes)
    sup = any(SUPPORT[0] <= m <= SUPPORT[1] for m in modes)
    return {'survive': bool(inA or inB), 'inA': bool(inA), 'inB': bool(inB), 'support28': bool(sup)}

def bootstrap(x, B=2000, seed=1, use='kde'):
    r = np.random.default_rng(seed)
    hits = {'survive': 0, 'inA': 0, 'inB': 0, 'support28': 0}
    allm = []
    for _ in range(B):
        xb = r.choice(x, len(x), replace=True)
        m = kde_modes(xb)[0] if use == 'kde' else gmm_modes(xb, seed=int(r.integers(1e6)))[0]
        allm.extend(m)
        for k, v in verdict(m).items():
            hits[k] += v
    allm = np.array(allm)
    return {k: v / B for k, v in hits.items()}, allm
