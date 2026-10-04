#!/usr/bin/env python3
"""LA-5 cycle 4: which single statistic decides? Calibration over the cycle 1 and cycle 3 outputs (no new sampling).
For each statistic: language group = LB, LB_KN, LB_PY, LBpers; code group = planted LA/LB randID and slot codes, plus
HTS and ICD where the statistic exists (form statistics only; value-tied statistics need sign values).
separation = gap between the groups' 95% bands (lowest language lower bound minus highest code upper bound, or the
reverse if language is lower), divided by the mean band half-width. A statistic is DECISIVE only if the bands do not
overlap (separation > 0). LA's position: f = (LA - code mean) / (language mean - code mean), with f at LA's CI ends;
f near 1 = language-like, near 0 = code-like. Replication rows: LA_HT, LA_nonHT, LA_noLB, LA_valued.
Output ../data/la5_c4.txt
"""
import json, os, math
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
c1 = json.load(open(os.path.join(D, 'la5_c1.json')))['pops']
c3 = json.load(open(os.path.join(D, 'la5_c3.json')))
LANG = ['LB', 'LB_KN', 'LB_PY', 'LBpers']
CODE = ['LA_randID', 'LA_slot', 'LB_randID', 'LB_slot', 'HTS', 'ICD']
REP = ['LA', 'LA_HT', 'LA_nonHT', 'LA_noLB', 'LA_valued']
def get(src, pop, k):
    if src == 1: return tuple(c1[pop][k]) if pop in c1 and k in c1[pop] else None
    return tuple(c3[pop]['half'][k]) if pop in c3 and k in c3[pop]['half'] else None
STATS = [(1, 'gain_exP'), (1, 'gap_exP'), (1, 'fin10_exG'), (1, 'pos_exG'), (3, 'echo'), (3, 'cecho'), (3, 'vinit'), (3, 'kober')]
out = ['LA-5 cycle 4: calibration of every candidate statistic (language = LB, LB_KN, LB_PY, LBpers; code = planted randID/slot on LA and LB, HTS, ICD)']
rows = []
for src, k in STATS:
    L = [get(src, p, k) for p in LANG]; Cc = [get(src, p, k) for p in CODE]; L = [x for x in L if x]; Cc = [x for x in Cc if x and not math.isnan(x[0])]
    lm = sum(x[0] for x in L) / len(L); cm = sum(x[0] for x in Cc) / len(Cc)
    hw = sum((x[2] - x[1]) / 2 for x in L + Cc) / len(L + Cc)
    sep = (min(x[1] for x in L) - max(x[2] for x in Cc)) if lm > cm else (min(x[1] for x in Cc) - max(x[2] for x in L))
    f = lambda v: (v - cm) / (lm - cm)
    line = f'\n{k} (cycle {src}): language mean {lm:.3f} [{min(x[1] for x in L):.3f}..{max(x[2] for x in L):.3f}]  code mean {cm:.3f} [{min(x[1] for x in Cc):.3f}..{max(x[2] for x in Cc):.3f}] (n code pops {len(Cc)})  separation {sep / hw:+.2f} half-widths -> {"DECISIVE" if sep > 0 else "overlap"}'
    for p in REP:
        x = get(src, p, k)
        if x: line += f'\n   {p:10s} {x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]  f = {f(x[0]):.2f} [{min(f(x[1]), f(x[2])):.2f}, {max(f(x[1]), f(x[2])):.2f}]'
    out.append(line); rows.append((sep / hw, k)); print(line)
out.append('\nranking by separation: ' + ', '.join(f'{k} {s:+.2f}' for s, k in sorted(rows, reverse=True)))
print(out[-1])
open(os.path.join(D, 'la5_c4.txt'), 'w').write('\n'.join(out) + '\n')
