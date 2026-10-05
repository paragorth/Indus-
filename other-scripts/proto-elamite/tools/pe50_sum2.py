"""Summarise pe50 cycle 2."""
import json, os, collections
import numpy as np
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe50_ckpt')
R = json.load(open(os.path.join(CK, 'cycle2.json')))
f = lambda r: f"I {r['I']:.2f} (null 97.5% {r['I_null_hi']:.2f}) R_sb {r['R_sb']:.2f} R_ss {r['R_ss']:.2f} R_bb {r['R_bb']:.2f} n_s {r['n_s']} n_b {r['n_b']}"
print('== plants')
g = collections.defaultdict(list)
for m, rep, r in R['plant']:
    g[m].append(r)
for m, L in g.items():
    I = [r['I'] for r in L]
    print(m, 'I med %.2f [%.2f-%.2f]' % (np.median(I), min(I), max(I)), 'R_sb %.2f R_ss %.2f R_bb %.2f' % tuple(np.median([[r['R_sb'], r['R_ss'], r['R_bb']] for r in L], 0)),
          'I>null', sum(r['I'] > r['I_null_hi'] for r in L), '/', len(L),
          'flagged U %d/%d S %d/%d' % (sum(r['flag_U'] for r in L), sum(r['nU'] for r in L), sum(r['flag_S'] for r in L), sum(r['nS'] for r in L)))
print('== Ur III')
for a, d, rep, r in R['ur3']:
    fl = [k for k, v in r['scan'].items() if v['flag']]
    if rep in ('full', 0, 1, 2):
        print(a, d, rep, f(r), 'flagged', fl)
    elif rep == 7:
        pass
for a in ('DREHEM', 'UMMA'):
    for d in ('OFF', 'DENT'):
        L = [r for aa, dd, rep, r in R['ur3'] if aa == a and dd == d and rep != 'full']
        print(' ', a, d, 'PE-size I med %.2f, >null %d/%d' % (np.median([r['I'] for r in L]), sum(r['I'] > r['I_null_hi'] for r in L), len(L)),
              'flag counts', collections.Counter(k for r in L for k, v in r['scan'].items() if v['flag']).most_common(5))
print('== PE')
for d, r in R['pe']:
    print(d, f(r))
    for nm in ('scan_final', 'scan_hdr'):
        sc = r.get(nm) or {}
        top = sorted(sc.items(), key=lambda kv: -kv[1]['exp_big'])[:12]
        print('  ', nm, ' '.join(f"{k}:{v['obs_big']}/{v['exp_big']:.1f}(n{v['n_small']}){'*' if v['flag'] else ''}" for k, v in top))
        fl = [k for k, v in sc.items() if v['flag']]
        print('   flagged', fl)
