"""Summarise v22 cycle 1 checkpoints (millibits per token, held-out, gain over the position-free model M0)."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v22_lib as L

R = {}
for f in glob.glob(os.path.join(L.CK, 'c1_*.json')):
    if f.endswith('c1_all.json'): continue
    d = json.load(open(f)); R[d['name']] = d

KEYS = ['best_rigid', 'best_mix', 'H2', 'H4', 'H6', 'H9', 'best_hmm']


def grp(prefix):
    return [R[k] for k in sorted(R) if k.startswith(prefix)]


def fmt(d):
    return ' '.join(f"{k}={d.get(k, float('nan')):.0f}" for k in KEYS)


lines = []
for c in ('ZL', 'LA', 'IT'):
    if f'{c}_real' not in R: continue
    re_ = R[f'{c}_real']
    lines.append(f'{c} real: {fmt(re_)} (rigid best {re_["best_rigid_k"]}, hmm best {re_["best_hmm_k"]})')
    for m in ('lineshuf_', 'lineshufhead_', 'wordshuf_', 'markov_', 'plant0.25_', 'plant0.1_'):
        g = grp(f'{c}_{m}')
        if not g: continue
        mean = {k: np.mean([x[k] for x in g]) for k in KEYS}
        sd = {k: (np.std([x[k] for x in g], ddof=1) if len(g) > 1 else float('nan')) for k in KEYS}
        mx = max(x['best_hmm'] for x in g)
        extra = ''
        if 'ari4' in g[0]: extra = ' ARI4=' + ','.join(f"{x['ari4']:.2f}" for x in g)
        lines.append(f'  {m[:-1]} n={len(g)}: ' + ' '.join(f"{k}={mean[k]:.0f}+-{sd[k]:.0f}" for k in KEYS) + f' max_hmm={mx:.0f}{extra}')
        if m in ('lineshuf_', 'lineshufhead_', 'wordshuf_', 'markov_'):
            ex = re_['best_hmm'] - mean['best_hmm']; exr = re_['best_rigid'] - mean['best_rigid']
            z = ex / sd['best_hmm'] if sd['best_hmm'] == sd['best_hmm'] and sd['best_hmm'] > 0 else float('nan')
            lines.append(f'    excess real-null: HMM {ex:+.0f} (z {z:.1f}), rigid {exr:+.0f}, mix {re_["best_mix"] - mean["best_mix"]:+.0f}; '
                         f'program-beyond-rigid {ex - exr:+.0f}')
print('\n'.join(lines))
json.dump(lines, open(os.path.join(L.CK, 'report1.json'), 'w'), indent=1)
