#!/usr/bin/env python3
"""la34 image arms: build per-occurrence matrices and unit distance matrices.
  D  ductus (sign-agnostic): each scalar feature z-scored WITHIN its sign code, averaged over a unit's
     occurrences -> unit vector; Euclidean distance.
  M  sign-matched form: for each sign code shared by two units, distance between the units' mean
     [z-scalars + within-code PCA of the 16x16 grid]; pair distance = mean over shared codes (NaN if none).
Units can be tablets (sides merged) or single documents (sides apart, for the side-pair positive control)."""
import numpy as np, json, os, collections
from la34_common import load, CK

SCAL = ['aspect', 'S_density', 'S_width', 'skel_len', 'ends', 'junc', 'cx', 'cy', 'axis_sin2', 'axis_cos2', 'ecc',
        'skew_x', 'skew_y', 'shear_xy'] + [f'dir{i}' for i in range(8)]


def occ_table(feat, roles=('syllabogram',), min_code=6, scale_free=False, occ=None):
    occ = [o for o in occ if o['role'] in roles and feat.get(o['file'])]
    cnt = collections.Counter(o['code'] for o in occ)
    occ = [o for o in occ if o['code'] and cnt[o['code']] >= min_code]
    names = [n for n in SCAL if not (scale_free and n.startswith('S_'))]
    X = np.array([[feat[o['file']][n] for n in names] for o in occ])
    G = np.array([feat[o['file']]['grid'] for o in occ])
    codes = np.array([o['code'] for o in occ])
    Z = np.zeros_like(X); P = np.zeros((len(occ), 8))
    for c in np.unique(codes):
        ix = codes == c
        mu, sd = X[ix].mean(0), X[ix].std(0); sd[sd < 1e-9] = 1
        Z[ix] = (X[ix] - mu) / sd
        g = G[ix] - G[ix].mean(0)
        u, s, vt = np.linalg.svd(g, full_matrices=False)
        k = min(8, len(s)); pc = g @ vt[:k].T
        pc = pc / (pc.std(0) + 1e-9)
        P[ix, :k] = pc
    return occ, names, Z, P, codes


def unit_D(occ, Z, key='unit', w=None):
    by = collections.defaultdict(list)
    for i, o in enumerate(occ): by[o[key]].append(i)
    U = sorted(by)
    V = np.array([Z[by[u]].mean(0) for u in U])
    if w is not None: V = V * w
    return U, V, {u: len(by[u]) for u in U}


def unit_M(occ, Z, P, codes, units, key='unit', wz=1.0, wp=1.0):
    F = np.hstack([Z * wz, P * wp])
    means = collections.defaultdict(dict)
    by = collections.defaultdict(list)
    for i, o in enumerate(occ): by[(o[key], codes[i])].append(i)
    for (u, c), ix in by.items(): means[u][c] = F[ix].mean(0)
    n = len(units); Dm = np.full((n, n), np.nan)
    for a in range(n):
        ma = means.get(units[a], {})
        for b in range(a + 1, n):
            mb = means.get(units[b], {})
            sh = set(ma) & set(mb)
            if sh:
                Dm[a, b] = Dm[b, a] = np.mean([np.linalg.norm(ma[c] - mb[c]) for c in sh])
    np.fill_diagonal(Dm, 0)
    return Dm


def labelled(meta, units, nocc, min_occ, key_meta=None):
    U = [u for u in units if meta[u]['scribe'] and nocc.get(u, 0) >= min_occ]
    c = collections.Counter(meta[u]['scribe'] for u in U)
    return [u for u in U if c[meta[u]['scribe']] >= 2]
