"""Summarise v22 cycle 2 checkpoints."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v22_lib as L

R = {}
for f in glob.glob(os.path.join(L.CK, 'c2_*.json')):
    if f.endswith('c2_all.json'): continue
    d = json.load(open(f)); R[d['name']] = d
K = ['best_rigid', 'best_mix', 'H2', 'H3', 'H4', 'H6', 'H9', 'best_hmm', 'best_poe']
for n in sorted(R):
    d = R[n]
    if 'best_hmm' in d:
        print(n, ' '.join(f"{k}={d[k]:.0f}" for k in K if k in d), d.get('best_rigid_k'), d.get('best_hmm_k'),
              f"ari4={d['ari4']:.2f}" if 'ari4' in d else '')
    elif 'S' in d and 'stages' not in d:
        print(n, {k: round(v, 1) for k, v in d.items() if ':' in k})
    elif 'stages' in d:
        print(n, 'S', d['S'], 'top lls', [round(x) for x in d['top_ll']], 'ARI top-1 vs next', [round(a, 2) for a in d['ari_top']],
              'visits', round(d['visits_mean'], 2), d['visits_hist'])
        print('   base', {k: round(v, 3) for k, v in d['base'].items()})
        for s in sorted(d['stages'], key=lambda z: (z['rel_mean'] if z['rel_mean'] is not None else 9)):
            print('   ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
