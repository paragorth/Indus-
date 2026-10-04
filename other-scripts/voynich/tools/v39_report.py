"""v39 report helpers: load cycle checkpoints, compare a merge with its random-merge null, language deltas."""
import os, glob, json
import numpy as np
import v39_lib as L

MEAS = ['h1', 'h2', 'h3', 'h21', 'wl_mu', 'wl_sd', 'ttr', 'hapax', 'zipf', 'arrow', 'gap', 'pL', 'pG', 'pB', 'auc']


def all_results(prefix='c1_'):
    out = {}
    for f in glob.glob(os.path.join(L.CK, prefix + '*.json')):
        k = os.path.basename(f)[len(prefix):-5]
        out[k] = json.load(open(f))
    return out


def fmt(x, m):
    if m == 'arrow': return f'{int(x)}'
    return f'{x:.2f}' if abs(x) >= 0.1 or m in ('pL', 'pG', 'pB') else f'{x:.3f}'


def compare(R, key, null_keys, meas=MEAS):
    """value, null mean, null sd, z, empirical two-sided p (vs null)"""
    out = {}
    for m in meas:
        v = R[key][m]
        nv = np.array([R[k][m] for k in null_keys if k in R and m in R[k]], float)
        mu, sd = nv.mean(), nv.std(ddof=1) if len(nv) > 1 else 0
        z = (v - mu) / sd if sd > 0 else 0.0
        p = (1 + (np.abs(nv - mu) >= abs(v - mu)).sum()) / (len(nv) + 1)
        out[m] = dict(v=v, mu=mu, sd=sd, z=z, p=p, n=len(nv))
    return out


def line(R, key, meas=MEAS):
    return ', '.join(f'{m} {fmt(R[key][m], m)}' for m in meas if m in R[key])
