"""S-DARK-14: WORKSHOP SIGNATURES.
Physical features of seals (boss code, material, colour, shape, cross-section, h/v/th) in
data/raw/inscriptions.csv define candidate workshops (clusters within each city, text NOT used).
Tests: (a) do clusters differ in sign vocabulary / closer / text length more than random clusters of
the same sizes (labels permuted within site x object type)? (d) do clusters map onto find areas?
(b) do allograph variants (raw form vs merged form, sign_allographs_levels.json) track clusters
more than site? (c) do seals within a cluster share more text (middles) than across clusters?
Usage: python3 tools/dark_loop14.py CYCLE [NPERM]
"""
import csv, json, math, sys, collections, re
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
CYCLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rng = np.random.default_rng(1400 + CYCLE)

KEY = ['cisi', 'site', 'type', 'symbol', 'cult', 'material', 'shape', 'area-section', 'block-house',
       'room-grid', 'complete', 'dir.', 'time', 'period', 'phase']
NUMRE = re.compile(r'(\d{3})')
LEVELS = ['seq_raw', 'seq_strong', 'seq_all']
BIG = ('Mohenjo-daro', 'Harappa')


def load():
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    canon = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    by = collections.defaultdict(list)
    for t in canon:
        by[tuple(t.get(f, '') for f in KEY)].append(t)
    seals, nseal = [], 0
    for r in rows:
        if not r['type'].startswith('SEAL'):
            continue
        nseal += 1
        k = tuple(r.get(f, '') for f in KEY)
        nums = sorted(int(x) for x in NUMRE.findall(r['text']))
        hit = None
        for t in by.get(k, []):
            if sorted(t['seq_raw']) == nums:
                hit = t
                break
        if hit is None and len(by.get(k, [])) == 1:
            hit = by[k][0]
        if hit is None:
            continue
        r['_c'] = hit
        seals.append(r)
    return seals, nseal


def fnum(s):
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except Exception:
        return np.nan


def features(r, with_emblem=False):
    cat = {}
    for f in ['boss', 'material', 'color', 'shape', 'cross-section']:
        v = r[f].strip()
        cat[f] = None if v in ('-', '', '?') else v.lower()
    h, v, th = fnum(r['h']), fnum(r['v']), fnum(r['th'])
    if np.isnan(h) and fnum(r['horizontal(mm)']) > 0:
        h = fnum(r['horizontal(mm)']) / 10
    if np.isnan(v) and fnum(r['vertical(mm)']) > 0:
        v = fnum(r['vertical(mm)']) / 10
    if np.isnan(th) and fnum(r['thickness(mm)']) > 0:
        th = fnum(r['thickness(mm)']) / 10
    num = {'h': math.log(h) if not np.isnan(h) else np.nan,
           'v': math.log(v) if not np.isnan(v) else np.nan,
           'th': math.log(th) if not np.isnan(th) else np.nan}
    if with_emblem:
        sym = r['symbol']
        cat['emblem'] = None if sym in ('-', '', 'None') else sym.split(':')[0]
        cat['style'] = sym.split(':')[1] if ':' in sym else None
        cat['cult'] = None if r['cult'] in ('-', '') else r['cult']
    return cat, num


def gower(items):
    n = len(items)
    cats = list(items[0][0].keys())
    nums = list(items[0][1].keys())
    D = np.zeros((n, n)); W = np.zeros((n, n))
    for f in cats:
        vals = np.array([it[0][f] if it[0][f] is not None else '' for it in items], object)
        known = vals != ''
        m = known[:, None] & known[None, :]
        D += m * (vals[:, None] != vals[None, :])
        W += m
    for f in nums:
        vals = np.array([it[1][f] for it in items], float)
        known = ~np.isnan(vals)
        rg = (np.nanmax(vals) - np.nanmin(vals)) if known.sum() > 1 else 1.0
        m = known[:, None] & known[None, :]
        diff = np.abs(np.nan_to_num(vals)[:, None] - np.nan_to_num(vals)[None, :]) / (rg or 1.0)
        D += m * np.minimum(diff, 1.0)
        W += m
    with np.errstate(invalid='ignore', divide='ignore'):
        G = np.where(W > 0, D / W, 1.0)
    np.fill_diagonal(G, 0)
    return squareform(G, checks=False)


def cluster_city(seals, k, with_emblem=False, min_known=3):
    items, keep = [], []
    for i, r in enumerate(seals):
        cat, num = features(r, with_emblem)
        nk = sum(v is not None for v in cat.values()) + sum(not np.isnan(v) for v in num.values())
        if nk >= min_known:
            items.append((cat, num)); keep.append(i)
    if len(keep) < 20:
        return None, None
    Z = linkage(gower(items), method='average')
    return np.array(keep), fcluster(Z, k, criterion='maxclust')


def mi(a, b):
    n = len(a)
    ca = collections.Counter(a); cb = collections.Counter(b); cab = collections.Counter(zip(a, b))
    return sum(c / n * math.log2(c * n / (ca[x] * cb[y])) for (x, y), c in cab.items())


def gstat_vocab(lab, sets, signs):
    lab = np.asarray(lab); labs = np.unique(lab); total = 0.0
    for s in signs:
        pres = np.array([s in st for st in sets]); p = pres.mean()
        if p == 0 or p == 1:
            continue
        for L in labs:
            m = lab == L; nL = m.sum(); k = pres[m].sum()
            for obs, exp in ((k, nL * p), (nL - k, nL * (1 - p))):
                if obs > 0:
                    total += 2 * obs * math.log(obs / exp)
    return total


def kruskal(lab, y):
    from scipy.stats import kruskal as kw
    groups = [np.asarray(y)[np.asarray(lab) == L] for L in np.unique(lab)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    try:
        return kw(*groups).statistic
    except ValueError:
        return 0.0


def perm_p(stat_fn, lab, strata, nperm):
    obs = stat_fn(lab)
    lab = np.asarray(lab).copy(); strata = np.asarray(strata)
    idx = {s: np.where(strata == s)[0] for s in np.unique(strata)}
    null = np.empty(nperm)
    for i in range(nperm):
        p = lab.copy()
        for s, ix in idx.items():
            p[ix] = lab[rng.permutation(ix)]
        null[i] = stat_fn(p)
    pval = (1 + (null >= obs).sum()) / (nperm + 1)
    sd = null.std() or 1.0
    return obs, pval, (obs - null.mean()) / sd, null.mean()


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0] * m; run = 0
    for rank, i in enumerate(order):
        run = max(run, ps[i] * (m - rank)); adj[i] = min(1.0, run)
    return adj


def describe_clusters(seals, keep, lab):
    out = []
    for L in np.unique(lab):
        ix = keep[lab == L]
        rs = [seals[i] for i in ix]
        def top(f):
            c = collections.Counter(r[f] for r in rs if r[f] not in ('-', ''))
            return ','.join(f'{a}:{b}' for a, b in c.most_common(2))
        hs = [fnum(r['h']) for r in rs]; hs = [x for x in hs if not np.isnan(x)]
        out.append(f'   c{L} n={len(ix)} boss[{top("boss")}] col[{top("color")}] shape[{top("shape")}] '
                   f'xs[{top("cross-section")}] h~{np.median(hs):.2f}cm(n={len(hs)}) emblem[{top("symbol")}]')
    return '\n'.join(out)


def city_groups(seals):
    g = collections.defaultdict(list)
    for i, r in enumerate(seals):
        g[r['site'] if r['site'] in BIG else 'OTHER'].append(i)
    return g


def middles(seq):
    """signs after the opener slot and before the closer (drop first and last) for len>=3."""
    return tuple(seq[1:-1]) if len(seq) >= 3 else tuple()
