"""Fair seed-agreement comparison with identical data across seeds:
REAL vs REALSHUFX0/X1 (one fixed numeral shuffle each), 4-seed (all 600 steps)
and 6-seed versions; PLANT precision by agreement level for reference."""
import sys, os, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe39_common import CK
MINN = 15


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CK, pattern)))]


def stable(R, k):
    tg = collections.defaultdict(list); ns = {}; gn = collections.defaultdict(list)
    for r in R:
        for s, t in r['lexAB'].items():
            if t[4] >= MINN:
                tg[s].append(t[0]); ns[s] = t[4]
                if s in r['gainAB'] and r['gainAB'][s][0] == t[0]:
                    gn[s].append(r['gainAB'][s][1])
    out = []
    for s, ts in tg.items():
        m, c = collections.Counter(ts).most_common(1)[0]
        if c >= k:
            g = gn.get(s)
            out.append((s, m, c, ns[s], round(sum(g) / len(g), 3) if g else None))
    return sorted(out, key=lambda x: (-x[2], -x[3]))


res = {}
sets4 = {'REAL': load('c3_REAL_prof_1[0-3].json'), 'SHUFX0': load('c3_REALSHUFX0_prof_1[0-3].json'),
         'SHUFX1': load('c3_REALSHUFX1_prof_1[0-3].json'), 'PLANT': load('c3_PLANT_prof_1[0-3].json')}
sets6 = {'REAL': load('c[23]_REAL_prof_*.json'), 'SHUFX0': load('c3_REALSHUFX0_prof_*.json'),
         'SHUFX1': load('c3_REALSHUFX1_prof_*.json'), 'PLANT': load('c[23]_PLANT_prof_*.json')}
for name, S in (('4 seeds', sets4), ('6 seeds', sets6)):
    print('==', name)
    for st, R in S.items():
        line = '%-7s n=%d' % (st, len(R))
        for k in range(2, len(R) + 1):
            St = stable(R, k)
            if st == 'PLANT':
                c = sum(x[0].split('|', 1)[1] == x[1].split('|', 1)[1] for x in St)
                line += '  k>=%d: %d (correct %d)' % (k, len(St), c)
            else:
                line += '  k>=%d: %d' % (k, len(St))
        print(line)
        res['%s/%s' % (name, st)] = {k: stable(R, k) for k in range(2, len(R) + 1)}
for st in ('REAL', 'SHUFX0', 'SHUFX1'):
    print(st, 'stable >= 3/6:', stable(sets6[st], 3)[:20])
json.dump(res, open(os.path.join(CK, 'calib2_c3.json'), 'w'))
