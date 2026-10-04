"""v35 report helper: rows from cached res_/post_ pickles (r and permutation p exactly as v25/v28)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v35_lib as V, v28_lib as X, v25_lib as L

M3 = ('ppmi', 'svd', 'potts')


def triple(res, key):
    v = [res['summ'].get((key, m)) for m in M3]
    if any(x is None for x in v):
        return None
    return ' / '.join(f'{r:+.2f}' for r, p, p2, pr in v) + ' (p ' + ' / '.join(f'{p:.4f}' for r, p, p2, pr in v) + ')'


def imgmean(res):
    ks = sorted({k for k, _ in res['summ'] if k.startswith('img:')})
    return [float(np.mean([res['summ'][(k, m)][0] for k in ks])) for m in M3], ks


def summary(name):
    res = X.load(f'res_{name}.pkl')
    if res is None:
        return 'not run'
    im, ks = imgmean(res)
    parts = [f'[{res["n"]} units, {res["tok"]} tokens] image combo mean over fonts ppmi/svd/potts {im[0]:+.2f} / {im[1]:+.2f} / {im[2]:+.2f}']
    for k in ks:
        parts.append(f'{k} {triple(res, k)}')
    if ('hand', 'ppmi') in res['summ']:
        parts.append(f'hand strokes {triple(res, "hand")}')
    pr = [res['summ'][(k, m)][3] for k in ks for m in M3]
    parts.append(f'partial r given position profile (image) {min(pr):+.2f} to {max(pr):+.2f}')
    return '; '.join(parts)


def post_summary(name, K=23):
    r = X.load(f'post_{name}.pkl')
    if r is None:
        return ''
    out = []
    keys = sorted({k for (t, k, m) in r})
    for k in keys:
        if ('topK', k, 'ppmi') in r:
            out.append(f'{k} top-{K} units ' + ' / '.join(f'{r[("topK", k, m)][0]:+.2f}' for m in M3) +
                       ' (p ' + ' / '.join(f'{r[("topK", k, m)][1]:.3f}' for m in M3) + '); random ' + str(K) +
                       '-subsets mean ' + ' / '.join(f'{r[("randK", k, m)][0]:+.2f}' for m in M3))
        for t in sorted({t for (t, kk, m) in r if kk == k and t not in ('topK', 'randK')}):
            out.append(f'{k} {t} ({r[(t, k, "ppmi")][2]} units) ' + ' / '.join(f'{r[(t, k, m)][0]:+.2f}' for m in M3) +
                       ' (p ' + ' / '.join(f'{r[(t, k, m)][1]:.3f}' for m in M3) + ')')
    return '; '.join(out)


PIPE = ('Exact v25 pipeline (behaviour ppmi / svd / potts on L1 R1 L2 R2 + edges; Mantel Spearman shape vs behaviour '
        'similarity; null = 5,000 random glyph-to-shape permutations; image combo = v25 zone+frame+topo descriptors)')
