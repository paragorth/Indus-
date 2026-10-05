"""Pool prof runs (tags c2, c3); calibrate seed agreement as a filter on the
known-answer control; apply the same filter to real and number-shuffled PE."""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe39_common import CK

MINN = 15
runs = collections.defaultdict(list)
for tag in ('c2', 'c3'):
    for fn in glob.glob(os.path.join(CK, '%s_*_prof_*.json' % tag)):
        r = json.load(open(fn))
        runs[r['set']].append(r)
out = {}
for st, R in sorted(runs.items()):
    n = len(R)
    tg = collections.defaultdict(list); ns = {}
    gains = collections.defaultdict(list)
    for r in R:
        for s, t in r['lexAB'].items():
            if t[4] >= MINN:
                tg[s].append(t[0]); ns[s] = t[4]
                if s in r['gainAB'] and r['gainAB'][s][0] == t[0]:
                    gains[s, t[0]].append(r['gainAB'][s][1])
    rows = []
    for s, ts in tg.items():
        m, c = collections.Counter(ts).most_common(1)[0]
        g = gains.get((s, m))
        rows.append((s, m, c, len(ts), ns[s], round(float(np.mean(g)), 3) if g else None,
                     (s.split('|', 1)[1] == m.split('|', 1)[1]) if st.startswith('PLANT') else None))
    print('%s: %d seeds, %d sources' % (st, n, len(rows)))
    res = {}
    for k in range(2, n + 1):
        S = [x for x in rows if x[2] >= k]
        line = '  agree >= %d/%d: %d stable' % (k, n, len(S))
        if st.startswith('PLANT'):
            line += ', correct %d' % sum(x[6] for x in S)
            Sg = [x for x in S if x[5] is not None and x[5] > 0.1]
            line += '; with gain > 0.1: %d, correct %d' % (len(Sg), sum(x[6] for x in Sg))
        print(line)
        res[k] = [list(x) for x in S]
    out[st] = {'n_seeds': n, 'by_k': res}
    for x in sorted(rows, key=lambda x: -x[2])[:12]:
        print('    ', x)
json.dump(out, open(os.path.join(CK, 'calib_c3.json'), 'w'))
