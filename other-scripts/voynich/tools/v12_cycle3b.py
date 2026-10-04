"""v12 cycle 3b follow-ups (single process, cheap).
1. The ch/sh swap that is enriched at lag 2 but not at lag 1: by start position i of the pair (i, i+2), which word pairs,
   what sits in the middle, and is the lag-1 ch/sh swap depleted? 50 within-line shuffles.
2. Parity entropy with the first (and last) word of the line removed: is there a low-entropy 'filler' position class?"""
import sys, os, json, math, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V

def shuf(lines, seed):
    rng = random.Random(seed); out = []
    for l in lines:
        ws = list(l['words']); rng.shuffle(ws); out.append(dict(l, words=ws))
    return out

def cs_pairs(lines, lag):
    pos = Counter(); pairs = Counter(); mids = Counter(); same_mid = 0; n = 0
    for l in lines:
        ws = l['words']
        for i in range(len(ws) - lag):
            a, b = ws[i], ws[i + lag]
            if V.hamming1(a, b):
                k = [j for j in range(len(a)) if a[j] != b[j]][0]
                if {a[k], b[k]} == {'C', 'S'}:
                    pos[min(i, 5)] += 1; pairs[(a, b)] += 1; n += 1
                    if lag == 2:
                        mids[ws[i + 1]] += 1
                        same_mid += V.hamming1(a, ws[i + 1]) or V.hamming1(ws[i + 1], b)
    return n, pos, pairs, mids, same_mid

def H(c):
    t = sum(c.values()); return -sum(v / t * math.log2(v / t) for v in c.values())

def par_ent(lines, drop_first=True, drop_last=False):
    o, e = Counter(), Counter()
    for l in lines:
        ws = l['words']
        for i, w in enumerate(ws):
            if drop_first and i == 0: continue
            if drop_last and i == len(ws) - 1: continue
            (o if i % 2 == 0 else e)[w] += 1
    # equalise sample size (entropy plug-in depends on n)
    rng = random.Random(0); lo = list(o.elements()); le = list(e.elements()); m = min(len(lo), len(le))
    return H(Counter(rng.sample(lo, m))), H(Counter(rng.sample(le, m)))

out = {}
for name in ['Voynich-ZL', 'Voynich-IT']:
    C = V.corpus(name); r = {}
    for lag in (1, 2, 3):
        n, pos, pairs, mids, sm = cs_pairs(C, lag)
        nulls = [cs_pairs(shuf(C, s), lag) for s in range(1, 51)]
        mn = sum(x[0] for x in nulls) / 50; sd = math.sqrt(sum((x[0] - mn) ** 2 for x in nulls) / 49)
        posn = {i: sum(x[1][i] for x in nulls) / 50 for i in range(6)}
        r[f'lag{lag}'] = {'obs': n, 'null': mn, 'z': (n - mn) / sd, 'pos_obs': dict(pos), 'pos_null': posn,
                          'top_pairs': [(f'{a} {b}', c) for (a, b), c in pairs.most_common(12)],
                          'top_mid': mids.most_common(10), 'mid_similar': sm}
        print(name, f'lag{lag} ch/sh Hamming-1 pairs {n} vs {mn:.1f} (z {(n-mn)/sd:.1f}); by start pos obs',
              [pos[i] for i in range(6)], 'null', [round(posn[i], 1) for i in range(6)])
        if lag == 2:
            print('   top pairs', r['lag2']['top_pairs'][:10]); print('   middles', mids.most_common(8), 'middle similar to an end', sm)
    for df, dl in ((True, False), (True, True)):
        o = par_ent(C, df, dl); nl = [par_ent(shuf(C, s), df, dl) for s in range(1, 21)]
        d = o[0] - o[1]; dn = [x[0] - x[1] for x in nl]; m = sum(dn) / 20; sd = math.sqrt(sum((x - m) ** 2 for x in dn) / 19)
        print(name, f'parity entropy drop_first={df} drop_last={dl}: H(pos 3,5,..) {o[0]:.3f} H(pos 2,4,..) {o[1]:.3f} diff {d:+.3f} null {m:+.3f} (z {(d-m)/sd:.1f})')
        r[f'parent_{df}_{dl}'] = {'obs': o, 'diff': d, 'null': m, 'z': (d - m) / sd}
    out[name] = r
json.dump(out, open(os.path.join(V.CK, 'c3b.json'), 'w'), indent=1, default=str)
