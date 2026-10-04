#!/usr/bin/env python3
"""LA-21 report: la21_report.py TAG [top_frac]  -> stability, truth scores (controls), LA groups, outside check."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la21_common import *

def load(tag, job):
    z = np.load(os.path.join(CK, f'{tag}_{job}.npz'))
    return z

def analyse(z, top=1.0):
    rows, cols, sc = z['rows'], z['cols'], z['sc']
    if top < 1.0:
        k = max(5, int(len(sc) * top)); o = np.argsort(-sc[:, 0])[:k]; rows, cols = rows[o], cols[o]
    signs = [str(s) for s in z['signs']]
    PR, PC, pv = coassign(rows, cols)
    out = dict(nrun=len(rows), score=float(sc[:, 0].mean()), ll=float(sc[:, 1].mean()), **stability(PR, PC))
    tf = truth_lb if str(z['truth']) == 'lb' else truth_syll
    out.update(truth_scores(signs, PR, PC, tf))
    return out, signs, PR, PC, pv

if __name__ == '__main__':
    tag = sys.argv[1]; top = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    jobs = sorted(os.path.basename(p)[len(tag) + 1:-4] for p in glob.glob(os.path.join(CK, f'{tag}_*.npz')))
    res = {}
    for j in jobs:
        o, signs, PR, PC, pv = analyse(load(tag, j), top)
        res[j] = o
        print(f"{j:10s} n={o['nrun']} rowdec {o['rowdec']:.3f} coldec {o['coldec']:.3f} row70 {o['row70']:3d} col70 {o['col70']:3d} | "
              f"rowAUC {o['row_auc']:.3f} cRowAUC {o['crow_auc']:.3f} colAUC {o['col_auc']:.3f} prec70 r{o['row_prec70']} c{o['col_prec70']} "
              f"(base {o['base_row']:.3f}/{o['base_col']:.3f}) score {o['score']:.0f}")
        if j.startswith('LA') and not j.startswith('LAsh'):
            print('  row groups (>=0.6):', [(' '.join(g), round(m, 2)) for g, m in groups(signs, PR, .6)])
            print('  col groups (>=0.6):', [(' '.join(g), round(m, 2)) for g, m in groups(signs, PC, .6)])
            o_ = np.argsort(-pv)[:10]
            print('  pure-vowel row:', [(signs[i], round(float(pv[i]), 2)) for i in o_])
    json.dump(res, open(os.path.join(CK, f'report_{tag}_{top}.json'), 'w'), indent=1)
