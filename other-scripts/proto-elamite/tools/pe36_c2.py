#!/usr/bin/env python3
"""pe36 cycle 2: held-out replication. Each corpus (controls matched to 2x PE-name evidence) split into two tablet
halves; grids fitted blind on each half; Spearman rho of the row co-assignment matrices over shared signs, and how
often pairs that share a row (>= 0.5) in half A also do in half B. Null: the same halves with within-string shuffles."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe36_common import *

CORP = ['PEN', 'PEA', 'PECLS', 'OB', 'UR3', 'LINB', 'PC']


def rep(c, va, vb):
    ra, ca, _, sa, _ = load_job(f'{c}_{va}'); rb, cb, _, sb, _ = load_job(f'{c}_{vb}')
    PRa, PCa, _ = coassign(ra, ca); PRb, PCb, _ = coassign(rb, cb)
    sh = [s for s in sa if s in sb]; ia = [sa.index(s) for s in sh]; ib = [sb.index(s) for s in sh]
    iu = np.triu_indices(len(sh), 1)
    out = dict(shared=len(sh))
    for nm, A, B in (('row', PRa, PRb), ('col', PCa, PCb)):
        a = A[np.ix_(ia, ia)][iu]; b = B[np.ix_(ib, ib)][iu]
        rho = float(np.corrcoef(rank(a), rank(b))[0, 1])
        sel = a >= .5
        out[nm] = dict(rho=rho, npairs=int(sel.sum()), carry=float(b[sel].mean()) if sel.any() else float('nan'), base=float(b.mean()))
    tf = TRUTH.get(c)
    if tf:
        out['truthA'] = truth_scores(sa, PRa, PCa, tf); out['truthB'] = truth_scores(sb, PRb, PCb, tf)
    return out


if __name__ == '__main__':
    res = {}
    for c in CORP:
        try:
            r = rep(c, 'h0', 'h1')
        except FileNotFoundError:
            continue
        nulls = []
        for s in (1, 2, 3):
            try: nulls.append(rep(c, f'h0shW{s}', f'h1shW{s}'))
            except FileNotFoundError: pass
        res[c] = dict(real=r, null=nulls)
        nr = [x['row']['rho'] for x in nulls]
        print(f"{c:6s} shared {r['shared']}: ROW rho {r['row']['rho']:.3f} (shuffled halves {', '.join(f'{x:.3f}' for x in nr)}); "
              f"{r['row']['npairs']} A-pairs carry {r['row']['carry']:.3f} vs base {r['row']['base']:.3f} "
              f"(null carry {', '.join(f'{x['row']['carry']:.3f}' for x in nulls)}) | COL rho {r['col']['rho']:.3f} "
              f"(null {', '.join(f'{x['col']['rho']:.3f}' for x in nulls)})")
        if 'truthA' in r:
            print(f"       truth halves cRowAUC {r['truthA']['crow_auc']:.3f}/{r['truthB']['crow_auc']:.3f}; colAUC {r['truthA']['col_auc']:.3f}/{r['truthB']['col_auc']:.3f}")
    json.dump(res, open(os.path.join(CK, 'report_c2.json'), 'w'), indent=1)
