"""pe70 feature matrices and label nulls, shared by all cycles."""
import collections
import numpy as np
from pe70_common import bands


def matrix(R, kind, min_df=5):
    n = len(R)
    items = []
    for r in R:
        f = set('S:' + t for t in r['toks'])
        if kind == 'pe':
            if r['hdr']:
                f.add('H:' + r['hdr'])
            f |= set('Y:' + s for s in r['sys'])
            if r['finals']:
                f.add('E:' + collections.Counter(r['finals']).most_common(1)[0][0])
        if kind == 'pc':
            f |= set('Y:' + c for c in r['codes'])
        items.append(f)
    df = collections.Counter(x for f in items for x in f)
    B = sorted(x for x, c in df.items() if min_df <= c <= n - min_df)
    idx = {x: i for i, x in enumerate(B)}
    Xb = np.zeros((n, len(B)), dtype=np.float32)
    for i, f in enumerate(items):
        for x in f:
            if x in idx:
                Xb[i, idx[x]] = 1
    num, names = [], []

    def add(name, v):
        v = np.array([np.nan if x is None else x for x in v], dtype=float)
        miss = np.isnan(v)
        v[miss] = np.nanmedian(v) if (~miss).any() else 0
        num.append((v - v.mean()) / (v.std() + 1e-9)); names.append(name)
        if miss.mean() > 0.02:
            num.append(miss.astype(float) - miss.mean()); names.append(name + '_missing')
    add('N:log_lines', [np.log1p(r['n_lines']) for r in R])
    add('N:log_area', [np.log(r['area']) if r['area'] else None for r in R])
    add('N:thick', [r['thick'] for r in R])
    add('N:h/w', [r['hw'] for r in R])
    add('N:n_types', [np.log1p(len(r['toks'])) for r in R])
    if kind == 'pe':
        lab = ['ent_len', 'single_share', 'n_cols', 'reverse', 'num_off_obv', 'num_share', 'damaged', 'has_hdr']
        for j, l in enumerate(lab):
            add('N:' + l, [r['fmt'][j + 1] for r in R])
        add('N:frag', [float(r['pres'] == 'fragment') for r in R])
    if kind == 'pc':
        add('N:num_share', [r['n_num'] / max(1, r['n_lines']) for r in R])
    Xn = np.array(num).T if num else np.zeros((n, 0))
    return Xb, B, Xn, names


def strata(R, how):
    b = bands(R)
    if how == 'band':
        return b
    return np.array([r['vol'] + '|' + x for r, x in zip(R, b)])


def perm_within(y, st, rng):
    y2 = y.copy()
    for s in np.unique(st):
        i = np.where(st == s)[0]
        y2[i] = y[rng.permutation(i)]
    return y2


def lor_z(Xb, y):
    y = y.astype(float)
    a = y @ Xb; b = y.sum() - a
    c = (1 - y) @ Xb; d = (1 - y).sum() - c
    lor = np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
    se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    return lor / se


def num_z(Xn, y):
    """point-biserial t-ish z for each numeric column"""
    if Xn.shape[1] == 0:
        return np.zeros(0)
    y = y.astype(float)
    r = ((Xn - Xn.mean(0)) * (y - y.mean())[:, None]).mean(0) / (Xn.std(0) * y.std() + 1e-12)
    return r * np.sqrt(len(y))
