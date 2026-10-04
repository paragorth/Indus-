"""v8 cycle 3b:
 V-3.3 blind conjugate identification: merge recto+verso into leaves; inside quire x homogeneous group,
       is a leaf's most similar other leaf its conjugate (same $B)? Chance = mean 1/(k-1).
       Controls: Isidore in book order (expect chance or below) and sheet-face order (expect above).
 V-3.4 jumps inside homogeneous runs (no quire/hand/language/section change at the boundary, window of
       2+2 pages entirely inside one group): which are in the sharpest 5% of all such boundaries, and
       how a planted 3-letter key change in Isidore ranks among the same kind of boundaries.
"""
import sys, os, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
from v8_lib import *
from v8_cycle3 import Vall, Lp, Ls, groupkey  # reuses cycle-3 objects (re-runs it quietly)

def leaves(pages):
    L = defaultdict(lambda: None)
    out = {}
    for p in pages:
        key = (p['quire'], p['leafnum'], p['leaf'])
        if key not in out:
            out[key] = dict(id=f"f{p['leafnum']}", quire=p['quire'], bifolio=p['bifolio'], leaf=p['leaf'],
                            lang=p['lang'], hand=p['hand'], illus=p['illus'], lines=[])
        out[key]['lines'] += p['lines']
    return list(out.values())


def conj_id(pages, label):
    Lv = leaves(pages)
    groups = defaultdict(list)
    for i, l in enumerate(Lv):
        groups[(l['quire'], l['lang'], l['hand'], l['illus'])].append(i)
    hits, chance, n = 0, 0.0, 0
    for g, ids in groups.items():
        if len(ids) < 3:
            continue
        sub = [Lv[i] for i in ids]
        X, _ = matrix(sub, 2); S = hellinger_sim(X); np.fill_diagonal(S, np.nan)
        m = np.nanmean(S, 1); R = S - m[:, None] - m[None, :] + np.nanmean(S)
        for a in range(len(sub)):
            partners = [b for b in range(len(sub)) if b != a and sub[b]['bifolio'] == sub[a]['bifolio'] and sub[b]['leaf'] != sub[a]['leaf']]
            if not partners:
                continue
            best = int(np.nanargmax(R[a]))
            hits += best in partners; chance += len(partners) / (len(sub) - 1); n += 1
    # binomial-ish z
    p0 = chance / n
    z = (hits - chance) / np.sqrt(n * p0 * (1 - p0))
    print(label, 'leaves tested', n, 'hits', hits, 'expected', round(chance, 1), 'z', round(z, 2))
    return n, hits, chance, z

rows = []
cv = conj_id(Vall, 'Voynich ZL')
cit = conj_id(voynich_pages(min_tokens=20, name='IT2a'), 'Voynich IT2a')
cb = conj_id(Lp, 'Latin book order')
cs = conj_id(Ls, 'Latin sheet-face order')
rows.append(('V-3.3', 'Blind conjugate identification (prediction test): recto+verso merged into leaves; within quire x language x hand x illustration, is each leaf\'s most similar leaf (double-centred Hellinger) its conjugate? Chance = sum of 1/(k-1). Controls: Isidore poured in book order and in sheet-face order',
             f'ZL: {cv[1]}/{cv[0]} hits vs {cv[2]:.1f} expected (z {cv[3]:+.1f}); IT2a: {cit[1]}/{cit[0]} vs {cit[2]:.1f} (z {cit[3]:+.1f}); control book order {cb[1]}/{cb[0]} vs {cb[2]:.1f} (z {cb[3]:+.1f}); control sheet-face {cs[1]}/{cs[0]} vs {cs[2]:.1f} (z {cs[3]:+.1f})',
             None))

# ---------------- V-3.4 ----------------
def homog_boundaries(pages):
    X, _ = matrix(pages, 2); S = hellinger_sim(X); np.fill_diagonal(S, np.nan)
    m = np.nanmean(S, 1); R = S - m[:, None] - m[None, :] + np.nanmean(S)
    out = []
    for k in range(2, len(pages) - 1):
        win = pages[k - 2:k + 2]
        if len({(p['quire'], p['lang'], p['hand'], p['illus']) for p in win}) != 1:
            continue
        v = [R[i, j] for i in (k - 2, k - 1) for j in (k, k + 1)]
        out.append((k, np.mean(v)))
    return out

hb = homog_boundaries(Vall)
vals = np.array([v for _, v in hb]); thr = np.percentile(vals, 5)
sharp = [(Vall[k - 1]['id'] + '|' + Vall[k]['id'], round(v, 3), 'same leaf' if Vall[k - 1]['leafnum'] == Vall[k]['leafnum'] else ('same bifolio' if Vall[k - 1]['bifolio'] == Vall[k]['bifolio'] else 'bifolio change')) for k, v in hb if v <= thr]
print('homogeneous boundaries', len(hb), 'sharpest 5%:', sharp)
# kinds among all homogeneous boundaries
kinds = defaultdict(list)
for k, v in hb:
    kd = 'same leaf' if Vall[k - 1]['leafnum'] == Vall[k]['leafnum'] else 'leaf change'
    kinds[kd].append(v)
print({k: (len(v), round(float(np.mean(v)), 3)) for k, v in kinds.items()})
# calibration with planted key changes at homogeneous positions of the Latin pages (same template)
det = []
rng = random.Random(4)
cand = [k for k, _ in hb]
for t in range(30):
    k0 = rng.choice(cand)
    lets = set(random.Random(t).sample(list('aeioustnrlm'), 3))
    Lpp = [dict(p) for p in Lp]
    for i in range(k0, len(Lpp)):
        Lpp[i]['lines'] = [[''.join(ch.upper() if ch in lets else ch for ch in w) for w in l] for l in Lpp[i]['lines']]
    hl = dict(homog_boundaries(Lpp))
    vv = np.array(list(hl.values()))
    det.append(np.mean(vv < hl[k0]))
det = np.array(det)
print('planted change quantile median', np.median(det), 'in lowest 5%', np.mean(det <= 0.05))
rows.append(('V-3.4', 'Jumps inside homogeneous runs: boundaries in binding order whose 2+2-page window lies in one quire x language x hand x illustration group; boundary score as V-3.2. Calibration: a 3-letter partial key change planted at one such position in Isidore pages with the same layout (30 trials)',
             f'{len(hb)} homogeneous boundaries; sharpest 5%: ' + ', '.join(f'{a} {b:+.3f} ({c})' for a, b, c in sharp) +
             '; mean score same-leaf ' + f"{np.mean(kinds['same leaf']):+.3f} (n={len(kinds['same leaf'])}) vs leaf change {np.mean(kinds['leaf change']):+.3f} (n={len(kinds['leaf change'])})" +
             f' || planted key change: median quantile {np.median(det):.3f}, in lowest 5% in {np.mean(det <= 0.05) * 100:.0f}% of trials',
             None))
import pickle
pickle.dump(dict(rows=rows, cv=cv, cit=cit, cb=cb, cs=cs, sharp=sharp, det=det), open(os.path.join(DATA, 'results', 'v8_cycle3b.pkl'), 'wb'))
for r in rows:
    print(r)
