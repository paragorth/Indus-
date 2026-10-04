"""pe23 cycle 4e: does the top-edge '1(N34)' mark state the tablet's own sum (a target or quota)?
For tablets whose countable entries are all clean CNT, compare the sum of entries with the edge
value under N34 = 300, 60 or 1000; control = the same test on length-matched tablets without an
edge mark (closeness of their sum to the same targets)."""
import sys, json, math
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pe23_common import *
rng = np.random.default_rng(2325)
corp = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
rows = []
for t in corp:
    edge = [l for l in t['lines'] if l['surface'] in ('top', 'edge', 'left') and l['numerals'] and not l['signs']]
    ent = [l for l in t['lines'] if l['surface'] in ('obverse', 'reverse') and l['numerals'] and l['signs']]
    if not ent:
        continue
    ok = all(not l['lacuna'] and '...' not in l['raw'] and
             all(n is not None and not c.startswith('n') for n, c in l['numerals']) and
             {c.split('@')[0] for _, c in l['numerals']} <= CNTSET for l in ent)
    if not ok:
        continue
    obv = [l for l in ent if l['surface'] == 'obverse']
    mark = ' '.join(f'{n}({c})' for l in edge for n, c in l['numerals']) if edge else None
    rows.append({'tab': t['id'], 'mark': mark, 'nent': len(obv), 'obv': obv})
R = {}
for N34 in (300, 60, 1000):
    V = {'N01': 1, 'N14': 10, 'N45': 100, 'N34': N34}
    s = lambda ls: sum(n * V[c.split('@')[0]] for l in ls for n, c in l['numerals'])
    m = [r for r in rows if r['mark'] == '1(N34)']
    c = [r for r in rows if r['mark'] is None and r['nent'] >= 2]
    def close(r, tgt):
        v = s(r['obv'])
        return v > 0 and abs(v - tgt) / tgt <= 0.1
    def under(r, tgt):
        return 0 < s(r['obv']) <= tgt
    a = np.mean([close(r, N34) for r in m]); b = np.mean([close(r, N34) for r in c])
    a2 = np.mean([under(r, N34) for r in m]); b2 = np.mean([under(r, N34) for r in c])
    # length-matched control: resample controls to the marked tablets' length distribution
    lm = []
    byn = defaultdict(list)
    for r in c:
        byn[min(r['nent'], 20)].append(r)
    for _ in range(500):
        pick = [byn[min(r['nent'], 20)][rng.integers(len(byn[min(r['nent'], 20)]))] for r in m if byn[min(r['nent'], 20)]]
        lm.append(np.mean([close(r, N34) for r in pick]))
    lm = np.array(lm)
    R[N34] = {'n_marked': len(m), 'n_ctrl': len(c), 'within10pct_marked': float(a), 'within10pct_ctrl': float(b),
              'len_matched_ctrl_mean': float(lm.mean()), 'p': float((lm >= a).mean()),
              'sum_le_target_marked': float(a2), 'sum_le_target_ctrl': float(b2),
              'sums_marked': sorted(s(r['obv']) for r in m)}
    print(N34, {k: v for k, v in R[N34].items() if k != 'sums_marked'})
print('sums on 1(N34)-marked tablets (N34=300):', R[300]['sums_marked'])
json.dump(R, open(os.path.join(CK, 'c4e.json'), 'w'), indent=1, default=float)
