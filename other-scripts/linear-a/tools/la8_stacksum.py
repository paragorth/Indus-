#!/usr/bin/env python3
"""LA-8: summarise the fsa vs pda (one-counter stack) arms. Gain = best fsa fitness - best pda fitness (bits)."""
import glob, json, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la8')
rows = {}
for p in glob.glob(os.path.join(OUT, 'stack_*.json')):
    j = json.load(open(p)); b = j['best']
    key = j['corpus'] + (' (true classes)' if j['truth'] else '')
    rows.setdefault(key, {}).setdefault(j['arm'], []).append((b['f'], b, os.path.basename(p), j['gen']))
for k in sorted(rows):
    r = rows[k]
    if 'fsa' not in r or 'pda' not in r: continue
    f = min(x[0] for x in r['fsa']); bp = min(r['pda'], key=lambda x: x[0])
    b = bp[1]
    print('%-22s fsa %.1f  pda %.1f  gain %.1f bits  | pda best: S=%d push %d pop %d depth-mass %.0f ablate-stack %s | gens %s' % (
        k, f, bp[0], f - bp[0], b['S'], b['push'], b['pop'], b['depth_mass'], round(b.get('ablate_stack_f', float('nan')), 1),
        sorted(x[3] for x in r['fsa'] + r['pda'])))
