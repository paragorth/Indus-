"""pe80 cycle 3, PE side: size scale of counted signs (E2, tablet fixed effects), split-half
reliability vs within-tablet shuffle, 2,000 random half splits; Ur III reliability at PE size
for comparison; frozen ranking with predicted mass via the Ur III slope."""
import os, json, random, hashlib, collections, math
import numpy as np
from scipy.stats import spearmanr
import pe80_common as pc
from pe80_c3 import pe_lines, ur3_lines, scales, shuffle_within_tablet

def split_rel(L, nsplit, seed, minn=10):
    by = collections.defaultdict(list)
    for x in L: by[x[0]].append(x)
    tabs = sorted(by); rng = random.Random(seed); out = []
    for _ in range(nsplit):
        rng.shuffle(tabs); h = len(tabs) // 2
        A = [x for t in tabs[:h] for x in by[t]]; B = [x for t in tabs[h:] for x in by[t]]
        _, ea, _ = scales(A, minn); _, eb, _ = scales(B, minn)
        ks = [w for w in ea if w in eb]
        if len(ks) >= 5:
            out.append(spearmanr([ea[k] for k in ks], [eb[k] for k in ks]).correlation)
    return np.array(out)

P = pe_lines(); print('PE counted lines', len(P), 'tablets', len({t for t, _, _ in P}))
E1, E2, c = scales(P, minn=15)
rows = sorted(E2.items(), key=lambda x: x[1])
res = {}
res['pe_rel'] = split_rel(P, 300, 1)
res['pe_rel_shuf'] = split_rel(shuffle_within_tablet(P, random.Random(2)), 300, 3)
U = ur3_lines()
# Ur III at PE size
by = collections.defaultdict(list)
for x in U: by[x[0]].append(x)
tabs = list(by); random.Random(4).shuffle(tabs); sub = []
for t in tabs:
    sub += by[t]
    if len(sub) >= len(P): break
res['ur3_rel_pe_size'] = split_rel(sub, 300, 5)
for k, v in res.items(): print(k, len(v), 'median rho', round(float(np.median(v)), 3), 'q05', round(float(np.quantile(v, .05)), 3))
# Ur III slope: log count scale vs log mass (full)
mass = json.load(open(os.path.join(pc.DATA, 'pe80_mass_table_frozen.json')))['mass_kg']
_, U2, _ = scales(U)
ks = [w for w in U2 if w in mass]
x = np.array([U2[w] for w in ks]); y = np.log10([mass[w] for w in ks])
b, a = np.polyfit(x, y, 1)
print('Ur III fit log10 mass = %.2f + %.2f * scale' % (a, b))
# centre PE scales on PE median vs Ur III median (unit-free: only ranks are claimed)
med_shift = np.median(list(U2.values())) - np.median(list(E2.values()))
frozen = dict(note='pe80 cycle 3 frozen ranking: PE counted signs (final sign of SDB entry lines) by count scale E2 (log10, tablet fixed effects). Predicted relative mass uses the Ur III slope; only the ORDER is claimed (C).',
              ur3_fit=dict(a=a, b=b, n=len(ks)),
              signs=[dict(sign=w, n=c[w], scale=round(s, 3), pred_log10_kg=round(a + b * (s + med_shift), 2)) for w, s in rows])
fn = os.path.join(pc.DATA, 'pe80_size_ranking_frozen.json')
json.dump(frozen, open(fn, 'w'), indent=1)
print('sha256', hashlib.sha256(open(fn, 'rb').read()).hexdigest())
for r in frozen['signs']: print(r)
json.dump({k: v.tolist() for k, v in res.items()}, open(os.path.join(pc.CK, 'c3_pe_rel.json'), 'w'))
