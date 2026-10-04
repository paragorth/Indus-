#!/usr/bin/env python3
"""LA-32 cycle 4: follow-up on the cycle-3 survivor (all one-sign pairs, words >= 2 signs).
Full pool, both scores, by locality class; LA vs 20 LB draws of equal size (shape z distribution);
shape score restricted to sign pairs inside the la21 65-sign set (matched to the sound test);
shape score after removing pairs whose two signs share an LB-value consonant (outside check: is the
shape excess just same-series look-alikes?)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la32_common import *
rng = np.random.default_rng(3204)
out = []
def log(s):
    print(s, flush=True); out.append(s)
M = pickle.load(open(os.path.join(CK, 'mats.pkl'), 'rb'))
iLA = {s: i for i, s in enumerate(M['la_signs'])}; iLB = {s: i for i, s in enumerate(M['lb_signs'])}
VLA, SLA, VLB, SLB = M['VIS_LA'], M['SND_LA'], M['VIS_LB'], M['SND_LB']
WA = la_words()
EA = one_sign_pairs(WA, 2, None)
EAs = one_sign_pairs(WA, 2, 'site')
log(f'# LA-32 cycle 4: full one-sign pool (>= 2 signs): {len(EA)} pairs any site, {len(EAs)} within site')
in65 = set(s for s in M['la_signs'] if not np.all(np.isnan(SLA[iLA[s]])))
classes = [('all', EA), ('within site', EAs), ('same doc', [e for e in EAs if e['samedoc']]),
           ('same hand', [e for e in EAs if e['samescribe']]), ('cross-site only', [e for e in EA if e['g'] == 'all' and not any(
               x['w1'] == e['w1'] and x['w2'] == e['w2'] for x in [])]),
           ('2-sign', [e for e in EA if e['L'] == 2]), ('>=3 signs', [e for e in EA if e['L'] >= 3]),
           ('both in 65', [e for e in EA if e['a'] in in65 and e['b'] in in65])]
res = {}
for name, E in classes[:3] + classes[3:4] + classes[5:]:
    rv = score_edges(E, VLA, iLA, 2000, rng); rs = score_edges(E, SLA, iLA, 2000, rng)
    log(f'  LA {name:12s} shape {fmt(rv)} | sound {fmt(rs)}')
    res[name] = (rv, rs)
nocon = [e for e in EA if not (M['CON_LA'][iLA[e['a']], iLA[e['b']]] == 1)] if True else EA
rv = score_edges(nocon, VLA, iLA, 2000, rng); log(f'  LA without LB-same-consonant pairs* shape {fmt(rv)}')
res['nocon'] = rv
EB = one_sign_pairs(lb_words(), 2, 'site')
zb = []; zbs = []
for d in range(20):
    sub = [EB[i] for i in rng.choice(len(EB), len(EA), replace=False)]
    zb.append(score_edges(sub, VLB, iLB, 500, rng)['z']); zbs.append(score_edges(sub, SLB, iLB, 500, rng)['z'])
log(f'  LB 20 draws of {len(EA)} pairs: shape z mean {np.mean(zb):.2f} (range {min(zb):.2f} to {max(zb):.2f}); sound z mean {np.mean(zbs):.2f} ({min(zbs):.2f} to {max(zbs):.2f})')
log(f'  LA shape z {res["all"][0]["z"]:.2f} exceeds {np.mean(np.array(zb) < res["all"][0]["z"]):.2f} of LB draws')
# pairs carrying the shape excess
cnt = collections.Counter(tuple(sorted((e['a'], e['b']))) for e in EA if e['a'] in iLA and e['b'] in iLA)
top = sorted(cnt.items(), key=lambda kv: -kv[1] * (VLA[iLA[kv[0][0]], iLA[kv[0][1]]] - 0.5))[:12]
log('  pairs contributing most shape excess (count x (shape rank - 0.5)): ' + ', '.join(f'{a}~{b} {n} ({VLA[iLA[a], iLA[b]]:.2f})' for (a, b), n in top))
json.dump({k: v for k, v in res.items()}, open(os.path.join(CK, 'c4_res.json'), 'w'), indent=1, default=str)
json.dump(dict(zb=zb, zbs=zbs), open(os.path.join(CK, 'c4_lb.json'), 'w'))
open(os.path.join(CK, 'c4_report.txt'), 'w').write('\n'.join(out))
