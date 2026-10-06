"""pe58 cycle 3 summary."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe58_lib import *
R = {r['job']: r for r in json.load(open(os.path.join(CK, 'c3.json')))}
W = json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))['signs']
for k in ('SPIKE_UR3', 'SPIKE_PLANT0'):
    print(k, {s: (round(v['z'], 2), round(v['conc'], 3), round(v['null'], 3), round(v['mode'], 2)) for s, v in R[k]['res'].items()})
tr = json.load(open(os.path.join(CK, 'c3_PLANT0_truth.json')))
print('plant truth', {s: round(math.log(v), 2) for s, v in tr.items()})
for k in ('ONE_PE', 'ONE_UR3', 'ONE_PLANT0'):
    print(k, R[k]['obs'], round(R[k]['null'], 4), round(R[k]['sd'], 4), R[k]['p'])
for n in ('PE', 'UR3', 'PLANT0'):
    for sc in ('tab', 'corpus'):
        r = R[f'MP_{n}_{sc}']['res']
        zs = [v['z_mode'] for v in r.values()]
        z1 = [v['z_one'] for v in r.values()]
        print(f'\nMP {n} {sc}: keys {len(r)}; z_mode>=3: {sum(z>=3 for z in zs)}, mean z_mode {np.mean(zs):.2f}; z_one>=3: {sum(z>=3 for z in z1)}')
        for k, v in sorted(r.items(), key=lambda kv: -kv[1]['z_mode'])[:12]:
            tag = ''
            if n == 'PLANT0':
                parts = k.split('|')[1:]
                tl = [math.log(tr[p]) if p in tr else 0 for p in parts]
                tag = f" truth {tl[0] - (tl[1] if len(tl) > 1 else 0):+.2f}" if any(tl) else ' truth 0'
            print(f"  {k:28s} n {v['n']:3d} mode {v['mode']:+.3f} x{math.exp(v['mode']):.2f} at_mode {v['at_mode']:.2f} (null {v['null_at_mode']:.2f}) z {v['z_mode']:.1f} | 1:1 {v['at_one']:.2f} (null {v['null_at_one']:.2f}) z {v['z_one']:.1f}{tag}")
# split-half
sp = [R[f'SPLIT_{s}'] for s in range(20)]
def corr(sel):
    rs = []
    for x in sp:
        a = [(v[0], v[1]) for s, v in x['res'].items() if sel(s) and v[0] is not None and v[1] is not None]
        if len(a) > 4:
            a = np.array(a); rs.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    return float(np.mean(rs)), float(np.std(rs)), len(rs)
print('\nsplit-half r weighted', corr(lambda s: s in W), 'B only', corr(lambda s: s in W and W[s]['both']), 'unweighted', corr(lambda s: s not in W))
sg = defaultdict(list)
for x in sp:
    for s, v in x['res'].items():
        if s in W and v[0] is not None and v[1] is not None:
            sg[s].append(np.sign(v[0]) == np.sign(v[1]) and abs(v[0]) > 0.05)
print('sign agreement across halves (W):', {s: round(np.mean(v), 2) for s, v in sorted(sg.items(), key=lambda kv: -np.mean(kv[1]))})
