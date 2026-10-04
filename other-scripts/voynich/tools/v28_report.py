"""v28 report helper: turn cached res_<corpus>.pkl into loop rows (r and permutation p exactly as v25)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v28_lib as X


def cell(res, key, m):
    v = res['summ'].get((key, m))
    if v is None:
        return None
    r, p, p2, pr = v
    return f'{r:+.2f} (p {p:.4f}; strat p {p2:.3f}; |pos {pr:+.2f})'


def row(rid, name, label, verdict=''):
    res = X.load(f'res_{name.replace("/", "-")}.pkl')
    if res is None:
        return (rid, label, 'not run', verdict)
    keys = sorted({k for k, _ in res['summ']})
    parts = []
    for k in keys:
        if not (k.startswith('img:') or k == 'hand'):
            continue
        parts.append(f'{k}: ppmi {cell(res, k, "ppmi")}, svd {cell(res, k, "svd")}, potts {cell(res, k, "potts")}')
    imgs = [res['summ'][(k, m)][0] for k in keys if k.startswith('img:') for m in ('ppmi', 'svd', 'potts')]
    summary = f'[{res["n"]} units, {res["tok"]} tokens] image combo r range {min(imgs):+.2f} to {max(imgs):+.2f}. ' + ' | '.join(parts)
    return (rid, f'{label}. Corpus key {name}. Exact v25 pipeline (behaviour ppmi/svd/potts on L1 R1 L2 R2; Mantel Spearman; null 5,000 random glyph-to-shape permutations; freq-stratified null; partial given word-position profile)', summary, verdict)


def imgrange(name):
    res = X.load(f'res_{name.replace("/", "-")}.pkl')
    v = [r for (k, m), (r, p, p2, pr) in res['summ'].items() if k.startswith('img:')]
    ps = [p for (k, m), (r, p, p2, pr) in res['summ'].items() if k.startswith('img:')]
    return min(v), max(v), np.median(v), max(ps)
