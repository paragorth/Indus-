"""pe58 cycle 2 summary: PE factors, lattice tests with three nulls, spike test, physics, controls."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe58_lib import *
R = {r['job']: r for r in json.load(open(os.path.join(CK, 'c2.json')))}
W = json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))['signs']
FL = 0.12
def fmt(d): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
print('UR3', R['UR3']['nuse'], R['UR3']['npool'], {k: fmt(v) for k, v in R['UR3']['lat'].items()})
for k in ('ladder', 'arb', 'phys'):
    for s in range(6):
        r = R[f'PLANT_{k}_{s}']
        print(k, s, 'mae', round(r['mae'], 3), 'k', r['k'], 'pool', r['npool'], 'p6 jit/emp', round(r['lat']['p6']['p_jit'], 3), round(r['lat']['p6'].get('p_emp', -1), 3),
              'p4', round(r['lat']['p4']['p_jit'], 3), round(r['lat']['p4'].get('p_emp', -1), 3), 'phys p', round(r['phys']['p'], 3))
est = R['PE']['est']; sp = R['PE']['spike']
sh = [R[f'PEshuf_{s}'] for s in range(5)]; st = [R[f'PEshuf_{100+s}'] for s in range(10)]
print('\nsign occ both | factor med [lo,hi] iqr x | shuf_sys med | spike conc vs tabshuf')
rows = []
for s in sorted(W, key=lambda s: -abs(est[s]['med']) if est[s] else 0):
    e = est[s]
    if not e: print(s, 'none'); continue
    ns = [x['est'][s]['med'] for x in sh if x['est'][s]]
    c = sp[s][0] if sp[s] else None
    nc = [x['spike'][s][0] for x in st if x['spike'][s]]
    z = (c - np.mean(nc)) / (np.std(nc) + 1e-9) if c is not None and nc else None
    rows.append((s, e['med']))
    print(f"{s:16s} {W[s]['occ']:4d} {str(W[s]['both']):5s} | {e['med']:+.3f} [{e['lo']:+.2f},{e['hi']:+.2f}] {e['iqr']:.2f} x{math.exp(e['med']):.2f} | {np.mean(ns):+.3f} | {c if c is None else round(c,3)} vs {np.mean(nc):.3f} z {z if z is None else round(z,2)} mode {sp[s][1] if sp[s] else None}")
x = [m for s, m in rows if abs(m) >= FL]
xb = [est[s]['med'] for s in W if W[s]['both'] and est[s] and abs(est[s]['med']) >= FL]
emp = [e['med'] for e in R['PEemp']['est'].values() if e and abs(e['med']) >= FL]
shp = [x_['est'][s]['med'] for x_ in sh for s in W if x_['est'][s] and abs(x_['est'][s]['med']) >= FL]
print('\nPE all W: k', len(x), 'emp pool', len(emp), 'shuf pool', len(shp))
for pm in (6, 4):
    print(' p', pm, 'emp', fmt(lattice_test(x, emp, pmax=pm)), '\n     shuf', fmt(lattice_test(x, shp, pmax=pm)))
    print(' B-only', fmt(lattice_test(xb, emp, pmax=pm)))
pu = R['PEpairs']['use']; psh = [u for x_ in sh for u in x_['use']]
print('pairs', len(R['PEpairs']['tab']), 'usable', len(pu), 'shuf usable', len(psh))
for pm in (6, 4):
    print(' pairs p', pm, fmt(lattice_test(pu, psh, pmax=pm)) if pu else None)
for k, v in sorted(R['PEpairs']['tab'].items(), key=lambda kv: -kv[1]['ntab'])[:25]:
    print(' ', k, 'ntab', v['ntab'], 'mode', round(v['mode'], 3), 'x', round(math.exp(v['mode']), 2), 'ci', round(v['mlo'], 2), round(v['mhi'], 2), 'med', round(v['med'], 3))
print('phys all', fmt({k: v for k, v in phys_dist_test({s: est[s]['med'] for s in W if est[s] and abs(est[s]['med']) >= FL}).items()}))
print('phys B', fmt(phys_dist_test({s: est[s]['med'] for s in W if W[s]['both'] and est[s] and abs(est[s]['med']) >= FL})))
print('phys emp signs', fmt(phys_dist_test({s: v for s, v in zip(range(len(emp)), emp)})))
