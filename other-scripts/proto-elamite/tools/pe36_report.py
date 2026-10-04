#!/usr/bin/env python3
"""pe36 cycle-1 report: grid stability of each corpus vs its within-string shuffles (100-restart blocks),
truth AUCs for the controls with known values, PE stable rows."""
import sys, os, glob, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe36_common import *

CORP = ['PEN', 'PEA', 'PENUM', 'PECLS', 'OB', 'OBSYL', 'UR3', 'LINB', 'PC']
KEYS = ['rowdec', 'row70', 'coldec', 'col70']


def block_stats(pre, cname, variant, tf=None):
    out = []
    for f in sorted(glob.glob(os.path.join(CK, f'{pre}{cname}_{variant}__*.npz'))):
        z = np.load(f); signs = [str(s) for s in z['signs']]
        PR, PC, pv = coassign(z['rows'], z['cols'])
        o = stability(PR, PC); o['score'] = float(z['sc'][:, 0].mean())
        if tf: o.update(truth_scores(signs, PR, PC, tf))
        out.append(o)
    return out


def pooled(pre, cname, variant):
    rows, cols, sc, signs, meta = load_job(f'{pre}{cname}_{variant}')
    PR, PC, pv = coassign(rows, cols)
    return signs, PR, PC, pv, meta, len(rows)


def summarize(pre=''):
    res = {}
    for c in CORP:
        tf = TRUTH.get(c)
        try:
            real = block_stats(pre, c, 'real', tf)
        except Exception:
            real = []
        seen = set(); sh2 = []
        for f in sorted(glob.glob(os.path.join(CK, f'{pre}{c}_realshW*__*.npz'))):
            v = re.search(r'_(realshW\d+)__', f).group(1)
            if v in seen: continue
            seen.add(v); sh2 += block_stats(pre, c, v, tf)
        sh = sh2
        if not real:
            continue
        r = {'n_real_blocks': len(real), 'n_sh_blocks': len(sh)}
        for k in KEYS + (['crow_auc', 'col_auc'] if tf else []):
            a = np.array([o[k] for o in real], float); b = np.array([o[k] for o in sh], float)
            r[k] = dict(real=float(a.mean()), real_sd=float(a.std()), sh=float(b.mean()) if len(b) else None,
                        sh_sd=float(b.std()) if len(b) else None,
                        z=float((a.mean() - b.mean()) / (b.std() + 1e-9)) if len(b) > 1 else None,
                        exceed=int((b >= a.mean()).sum()) if len(b) else None)
        if tf:
            r['prec70'] = [o['prec70'] for o in real]; r['ntruth'] = real[0]['ntruth']; r['npos_row'] = real[0]['npos_row']
        res[c] = r
    return res


if __name__ == '__main__':
    pre = sys.argv[1] if len(sys.argv) > 1 else ''
    res = summarize(pre)
    for c, r in res.items():
        line = f"{c:6s} blocks {r['n_real_blocks']}/{r['n_sh_blocks']} | " + ' | '.join(
            f"{k} {r[k]['real']:.3f} vs sh {r[k]['sh']:.3f}+-{r[k]['sh_sd']:.3f} (z {r[k]['z']:.1f})" for k in KEYS if r[k]['sh'] is not None)
        print(line)
        if 'crow_auc' in r:
            print(f"       truth n={r['ntruth']} same-consonant pairs={r['npos_row']}: cRowAUC {r['crow_auc']['real']:.3f} (sh {r['crow_auc']['sh']:.3f}+-{r['crow_auc']['sh_sd']:.3f}) "
                  f"colAUC {r['col_auc']['real']:.3f} (sh {r['col_auc']['sh']:.3f}) prec70 {r['prec70']}")
    for c in ('PEN', 'PEA', 'OB', 'UR3', 'LINB', 'PC'):
        try:
            signs, PR, PCm, pv, meta, n = pooled(pre, c, 'real')
        except FileNotFoundError:
            continue
        g = groups(signs, PR, .6)
        print(f'{c} pooled {n} restarts: row groups >= 0.6:', [(' '.join(x), round(m, 2)) for x, m in g])
        o_ = np.argsort(-pv)[:8]
        print(f'   pure-vowel row:', [(signs[i], round(float(pv[i]), 2)) for i in o_])
    json.dump(res, open(os.path.join(CK, f'report_{pre}c1.json'), 'w'), indent=1)
