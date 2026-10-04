#!/usr/bin/env python3
"""pe36 cycle 3: (a) universals ablation (ll_: tier likelihood only) vs full score; (b) grid-size sensitivity;
(c) which signs sit in stable rows: per-sign row stability (mean of its two strongest row-mate frequencies),
calibrated on the controls (do CV syllables get higher stability than logograms / other signs?), then applied to PE:
is stability carried by the frequent simple signs?"""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe36_common import *
from pe36_report import summarize


def per_sign(PR):
    P = PR.copy(); np.fill_diagonal(P, 0)
    return np.sort(P, 1)[:, -2:].mean(1)


def spearman(a, b):
    return float(np.corrcoef(rank(np.asarray(a, float)), rank(np.asarray(b, float)))[0, 1])


def sign_calib(c, pre=''):
    rows, cols, _, signs, meta = load_job(f'{pre}{c}_real')
    PR, PC, pv = coassign(rows, cols); st = per_sign(PR)
    tf = TRUTH.get(c)
    freq = meta['I'] + meta['B'].sum(0)
    out = dict(rho_freq=spearman(st, freq))
    if tf:
        lab = [tf(s) is not None for s in signs]
        out['auc_CV_vs_other'] = auc(st, lab); out['nCV'] = int(sum(lab))
    nulls = []
    for f in sorted(glob.glob(os.path.join(CK, f'{pre}{c}_realshW*__*.npz'))):
        z = np.load(f); PRn, _, _ = coassign(z['rows'], z['cols']); sn = per_sign(PRn)
        o = dict(rho_freq=spearman(sn, freq), mean=float(sn.mean()))
        if tf: o['auc'] = auc(sn, lab)
        nulls.append(o)
    out['mean_stab'] = float(st.mean()); out['null'] = nulls
    return out, signs, st, freq


if __name__ == '__main__':
    R = {}
    print('== (a) ablation: tier likelihood only (ll_) vs full universals (cycle 1)')
    full = summarize(''); ll = summarize('ll_')
    for c in ll:
        f, l = full[c], ll[c]
        s = f"{c:5s} row70 full {f['row70']['real']:.1f} (sh {f['row70']['sh']:.1f}) | ll {l['row70']['real']:.1f} (sh {l['row70']['sh']:.1f}); rowdec full z {f['rowdec']['z']:.1f}, ll z {l['rowdec']['z']:.1f}"
        if 'crow_auc' in l:
            s += f" | cRowAUC full {f['crow_auc']['real']:.3f} ll {l['crow_auc']['real']:.3f} (ll sh {l['crow_auc']['sh']:.3f})"
        print(s)
    R['ablation'] = dict(full=full, ll=ll)
    print('== (b) grid size')
    R['grid'] = {}
    for g in ('g13x4_', 'g18x5_', 'g15x3_'):
        r = summarize(g); R['grid'][g] = r
        for c, x in r.items():
            s = f"{g} {c:4s} row70 {x['row70']['real']:.1f} (sh {x['row70']['sh']:.1f}) rowdec z {x['rowdec']['z']:.1f} coldec z {x['coldec']['z']:.1f}"
            if 'crow_auc' in x: s += f" cRowAUC {x['crow_auc']['real']:.3f} (sh {x['crow_auc']['sh']:.3f})"
            print(s)
    print('== (c) per-sign row stability')
    R['signs'] = {}
    for c in ('OB', 'OBSYL', 'UR3', 'LINB', 'PC', 'PEN', 'PEA'):
        o, signs, st, freq = sign_calib(c)
        R['signs'][c] = o
        s = f"{c:5s} mean stab {o['mean_stab']:.3f} (null {np.mean([n['mean'] for n in o['null']]):.3f}); rho(stab, freq) {o['rho_freq']:.2f} (null {np.mean([n['rho_freq'] for n in o['null']]):.2f})"
        if 'auc_CV_vs_other' in o:
            s += f"; AUC CV syllables vs other signs {o['auc_CV_vs_other']:.3f} (n CV {o['nCV']}/65; null {np.mean([n['auc'] for n in o['null']]):.3f})"
        print(s)
        if c in ('PEN', 'PEA'):
            comp = [s_.startswith('|') for s_ in signs]
            print(f"      compounds among top-65: {sum(comp)}; AUC simple vs compound {auc(st, [not x for x in comp]) if any(comp) else float('nan'):.3f}")
            o_ = np.argsort(-st)
            print('      most stable:', [(signs[i], round(float(st[i]), 2), int(freq[i])) for i in o_[:15]])
            # frequency tercile
            fr = rank(freq); top = fr >= 2 * len(fr) / 3; bot = fr < len(fr) / 3
            print(f"      stab by frequency tercile: top {st[top].mean():.3f}, mid {st[~top & ~bot].mean():.3f}, low {st[bot].mean():.3f}")
            R['signs'][c]['most_stable'] = [(signs[i], float(st[i]), int(freq[i])) for i in o_[:15]]
    json.dump(R, open(os.path.join(CK, 'report_c3.json'), 'w'), indent=1, default=float)
