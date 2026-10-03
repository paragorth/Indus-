"""Loop 41 cycle 2 (b): identical-reading objects only (Wells = IM77, cost-0 aligned pairs from loop 24) and fragile
signs removed (W617 and W388 dropped, W700 dropped, 741/742 merged into 740). Every headline statistic with its own null.
Also the original S349 null (R fixed) vs cycle 1's both-sides-shuffled null, and a Markov order-1 null for P2 and nesting.
Output: data/derived/dark/loop41_cycle2.txt"""
import sys, random, json, collections
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *
out = open(DARK + 'loop41_cycle2.txt', 'w')
def P(*a):
    print(*a); print(*a, file=out); out.flush()
C = load('canonical')
pairs = [p for p in json.load(open(DARK + 'loop24_pairs.json')) if p['accepted']]
agreed = {p['widx'] for p in pairs if p['cost'] == 0}; matched = {p['widx'] for p in pairs}
P('# Loop 41 cycle 2: identical-reading objects and fragile signs')
P(f'aligned pairs accepted {len(pairs)}; identical (cost 0) {len(agreed)}; matched {len(matched)}')
rnd = random.Random(412); rows = []
for lvl in ('seq_raw', 'seq_all'):
    full = dedup(C, lvl, 'die')
    sets = {'FULL die': full,
            'MATCHED die': [(r, s) for r, s in full if C.index(r) in matched] if False else [(C[i], tuple(C[i][lvl])) for i in sorted(matched) if C[i][lvl]],
            'AGREED die': [(C[i], tuple(C[i][lvl])) for i in sorted(agreed) if C[i][lvl]],
            'FULL die defragile': defragile(full),
            'AGREED die defragile': defragile([(C[i], tuple(C[i][lvl])) for i in sorted(agreed) if C[i][lvl]])}
    # AGREED objects are one copy each already (one Wells object per pair); apply die dedup on text too
    for k in ('MATCHED die', 'AGREED die', 'AGREED die defragile'):
        seen = set(); dd = []
        for r, s in sets[k]:
            key = (r['cisi'] if r['cisi'] not in ('-', '') else id(r), s)
            if key in seen: continue
            seen.add(key); dd.append((r, s))
        sets[k] = dd
    for name, T in sets.items():
        res = all_stats(T, rnd, nnull=40, label=f'{lvl} {name}')
        # extra: size-matched FULL control for AGREED (AGREED is 90% seals, Mohenjo-daro heavy): same site x type counts
        if name.startswith('AGREED') and not name.endswith('defragile'):
            need = collections.Counter((r['site'], otype(r['type'])) for r, _ in T)
            pool = collections.defaultdict(list)
            for r, s in (defragile(full) if 'defragile' in name else full): pool[(r['site'], otype(r['type']))].append((r, s))
            ctrl = []
            for key, k in need.items(): ctrl += rnd.sample(pool[key], min(k, len(pool[key])))
            res2 = all_stats(ctrl, rnd, nnull=40, label=f'{lvl} FULL matched to AGREED mix'); rows.append(res2); P(line(res2))
        rows.append(res); P(line(res))
json.dump(rows, open(DARK + 'loop41_cycle2.json', 'w'), indent=1, default=str)
