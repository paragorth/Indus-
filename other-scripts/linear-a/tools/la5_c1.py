#!/usr/bin/env python3
"""LA-5 cycle 1: type-level form battery (dedup >= 2-sign types), every population drawn to the same m.
m = half the Linear A admin type count, without replacement: the spread of half-samples equals the bootstrap spread
of the full LA sample, so the LA 95% band is a CI, and every other population is measured at the same m.
Statistics (la5_common.battery): phonotactic gaps among the top-15 signs, held-out adjacency gain beyond position class, top-10 initial / final coverage, positional
asymmetry, one-sign prefix / suffix extension share; each as excess over G (global shuffle), P (position-class shuffle),
M2 (order-2 Markov regeneration).
Output: ../data/la5_c1.json, ../data/la5_c1.txt
"""
import sys, json, random, collections
sys.path.insert(0, __import__('os').path.dirname(__file__))
import la5_common as C

ND = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rnd = random.Random(51)
la = C.la_docs(); lb = C.lb_docs()
wa = C.words_of(la); wb = C.words_of(lb)
T = lambda ws, f=lambda s: True: sorted(set(w for _, s, w in ws if f(s)))
pops = {
    'LA': T(wa), 'LA_HT': T(wa, lambda s: s == 'Haghia Triada'), 'LA_nonHT': T(wa, lambda s: s != 'Haghia Triada'),
    'LB': T(wb), 'LB_KN': T(wb, lambda s: s == 'KN'), 'LB_PY': T(wb, lambda s: s == 'PY'),
}
pops.update(C.comparators())
r0 = random.Random(7)
for base in ('LA', 'LB'):
    m1 = C.random_id_code(pops[base], r0); m2 = C.slot_code(pops[base], r0)
    pops[base + '_randID'] = sorted(set(m1.values())); pops[base + '_slot'] = sorted(set(m2.values()))
M = len(pops['LA']) // 2
small = len(pops['LA_nonHT'])
out = {'m': M, 'n_types': {k: len(v) for k, v in pops.items()}, 'pops': {}}
lines = [f'LA-5 cycle 1: m = {M} types per draw, {ND} draws; LA_nonHT has {small} types (drawn at m = {small // 2} for its own row)',
         'types: ' + ', '.join(f'{k} {len(v)}' for k, v in pops.items())]
KEYS = ['gap', 'gap_exG', 'gap_exP', 'gap_exM2', 'gain', 'gain_exG', 'gain_exP', 'gain_exM2', 'init10_exG', 'fin10_exG', 'pos_exG', 'pos_exP',
        'pre', 'suf', 'pre_exM2', 'suf_exM2']
for name, types in pops.items():
    m = M if len(types) >= M * 1.2 else len(types) // 2
    acc = collections.defaultdict(list)
    for d in range(ND):
        s = rnd.sample(types, m)
        for k, v in C.battery(s, rnd, nnull=4).items(): acc[k].append(v)
    out['pops'][name] = {k: C.summ(v) for k, v in acc.items()}
    out['pops'][name]['m'] = m
    lines.append(f'\n{name} (m={m})')
    lines.append('  ' + '  '.join(f'{k} {out["pops"][name][k][0]:+.3f} [{out["pops"][name][k][1]:+.3f},{out["pops"][name][k][2]:+.3f}]' for k in KEYS))
    print(lines[-2], lines[-1], flush=True)
json.dump(out, open(C.LAD + '/la5_c1.json', 'w'), indent=1)
open(C.LAD + '/la5_c1.txt', 'w').write('\n'.join(lines) + '\n')
