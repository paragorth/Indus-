"""summarise pe7 cycle 1 checkpoints: real vs shuffled-element null per corpus."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import CKPT
R = {}
for p in glob.glob(os.path.join(CKPT, 'c1_*.json')):
    k = os.path.basename(p)[3:-5]
    corp, rep, kind = k.rsplit('_', 2)
    R.setdefault(corp, {}).setdefault(kind[:4], []).append(json.load(open(p)))
for corp, d in R.items():
    if 'real' not in d or 'null' not in d:
        continue
    row = [corp, 'reps %d' % len(d['real'])]
    for m in ('delta', 'ri', 'ci', 'support_mean', 'support_ge50'):
        a = np.array([x[m] for x in d['real']]); b = np.array([x[m] for x in d['null']])
        diff = a.mean() - b.mean()
        row.append('%s %.3f/%.3f (d %+.3f, z %.1f)' % (m, a.mean(), b.mean(), diff, diff / (np.sqrt(a.var() / len(a) + b.var() / len(b)) + 1e-9)))
    rt = np.array([x['rates'] for x in d['real']]).mean(0)
    row.append('rates m/s/i/d %.2f/%.2f/%.2f/%.2f' % tuple(rt))
    print(' | '.join(row))
