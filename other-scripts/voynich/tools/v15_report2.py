"""Summarise cycle-2 decoding runs: real vs nulls per (stream, scheme, language).

For each cell: held-out score of the real stream, mean/sd/max over the within-line
nulls ('line') and the global nulls ('glob'); z_line; margin over the best null.
Then per source the best cell, and a family-wise view (max z over the cells of a
source vs the distribution of max z in the controls).
"""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15_lib as V

D = os.path.join(V.RES, sys.argv[1] if len(sys.argv) > 1 else 'cycle2')
rows = []
for p in sorted(glob.glob(os.path.join(D, '[!_]*.json'))):
    r = json.load(open(p))
    real = [x for x in r['runs'] if x['kind'] == 'real'][0]
    nl = np.array([x['test'] for x in r['runs'] if x['kind'] == 'line'])
    ng = np.array([x['test'] for x in r['runs'] if x['kind'] == 'glob'])
    ntr = np.array([x['train'] for x in r['runs'] if x['kind'] == 'line'])
    alln = np.concatenate([nl, ng])
    z = (real['test'] - nl.mean()) / (nl.std(ddof=1) + 1e-9) if len(nl) > 1 else float('nan')
    zg = (real['test'] - ng.mean()) / (ng.std(ddof=1) + 1e-9) if len(ng) > 1 else float('nan')
    rows.append(dict(src=r['src'], unit=r['unit'], scheme=r['scheme'], lang=r['lang'], real=real['test'],
                     real_tr=real['train'], rand_tr=real['rand_train'], null_line=float(nl.mean()) if len(nl) else np.nan,
                     null_tr=float(ntr.mean()) if len(ntr) else np.nan,
                     null_glob=float(ng.mean()) if len(ng) else np.nan, z=z, zg=zg,
                     margin=real['test'] - alln.max(), sample=real['sample'], nsym=real['n_sym']))

by = defaultdict(list)
for r in rows:
    by[(r['src'], r['unit'])].append(r)
for k, rs in sorted(by.items()):
    rs.sort(key=lambda r: -r['z'] if r['z'] == r['z'] else 0)
    zs = np.array([r['z'] for r in rs])
    print(f"== {k[0]} [{k[1]}] cells {len(rs)}; max z {np.nanmax(zs):+.1f}; cells z>3: {(zs > 3).sum()}; "
          f"cells beating every null: {sum(r['margin'] > 0 for r in rs)}")
    for r in rs[:4]:
        print(f"   {r['scheme']:8s} {r['lang']:3s} test {r['real']:.3f} null(line) {r['null_line']:.3f} "
              f"null(glob) {r['null_glob']:.3f} z {r['z']:+.1f} zg {r['zg']:+.1f} margin {r['margin']:+.3f} "
              f"train {r['real_tr']:.3f} (null {r['null_tr']:.3f}; best random {r['rand_tr']:.3f})  {r['sample'][:50]}")
json.dump(rows, open(os.path.join(D, '_summary.json'), 'w'), indent=0)

# ---- calibrated verdict: plaintext reference per language and family-wise view
refs = {}
for lang in V.LANGS:
    lm = V.get_lm(lang)
    y = lm.x[-20000:]
    q = np.bincount(y, minlength=lm.A) / len(y); m = q > 0
    refs[lang] = lm.score_idx(y) - V.LAMBDA * float((q[m] * (np.log2(q[m]) - np.log2(lm.uni[m]))).sum())
print('\nplaintext reference J (own text tail; in-sample model):', {k: round(v, 2) for k, v in refs.items()})
neg = [r for r in rows if r['src'] in ('Caesar', 'Manzoni')]
negmax = defaultdict(lambda: -99)
for r in neg:
    negmax[(r['scheme'], r['lang'])] = max(negmax[(r['scheme'], r['lang'])], r['z'])
print('negative-control max z over all cells:', round(max(r['z'] for r in neg), 1) if neg else None)
for k, rs in sorted(by.items()):
    dec = [r for r in rs if r['real'] >= refs[r['lang']] - 0.6]
    above_neg = [r for r in rs if r['z'] == r['z'] and r['z'] > negmax.get((r['scheme'], r['lang']), -99) and r['z'] > 3]
    print(f"{k[0]:13s} {k[1]:4s} decoded (J >= ref-0.6): {len(dec):2d}/{len(rs)}  z>3 and above both negative controls: "
          f"{len(above_neg)}  " + ' '.join(f"{r['scheme']}/{r['lang']}" for r in dec[:6]))
