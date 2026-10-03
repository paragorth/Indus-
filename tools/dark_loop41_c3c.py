"""Loop 41 cycle 3c: (f) Ur III-side name-calibration sweep; Indus ranges read from loop41_cycle3_namecalib.json. Appends to loop41_cycle3.txt"""
import sys, random, json, collections, statistics as st
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *
out = open(DARK + 'loop41_cycle3.txt', 'a')
def P(*a):
    print(*a); print(*a, file=out); out.flush()
C = load('canonical'); rnd = random.Random(4133)
# fast n-gram generator: per-state (keys, weights) precomputed, 'E' excluded
def fast_model(ms, order):
    m = markov_fit(ms, order); tab = {}
    for h, cnt in m.items():
        items = [(k, v) for k, v in cnt.items() if k != 'E']
        if items: tab[h] = (tuple(k for k, _ in items), tuple(v for _, v in items))
    uni = collections.Counter(x for s in ms for x in s); U = (tuple(uni), tuple(uni.values()))
    return tab, U
def fast_gen(model, L, rnd, order):
    tab, U = model; out = []; h = ('S',) * order
    for _ in range(L):
        ks, ws = tab.get(h, U); x = rnd.choices(ks, ws)[0]; out.append(x); h = (h + (x,))[1:]
    return tuple(out)
def ngram_null(ms, order, rnd):
    model = fast_model(ms, order)
    return [fast_gen(model, len(s), rnd, order) for s in ms]
def unigram_null(ms, rnd):
    pool = [x for s in ms for x in s]
    return [tuple(rnd.choice(pool) for _ in s) for s in ms]
def ratio_for(ms, n, order, rnd, draws=6):
    rs = []; ns = []
    for _ in range(draws):
        sub = rnd.sample(ms, min(n, len(ms)))
        rs.append(uniq(sub))
        g = unigram_null(sub, rnd) if order == 0 else ngram_null(sub, order, rnd)
        ns.append(uniq(g))
    return st.mean(rs), st.mean(ns), st.mean(rs) / st.mean(ns)
P('\n## (f) Ur III side (loop18 cache, 25,291 CDLI seal impressions with a legend): same generators, same granularity choices, n matched to Indus (1,744) and full')
ur = json.load(open(DARK + 'loop18_ur3_impressions.json'))
units = {'impressions_line1': [tuple(l['line1'][0]) for l in ur if l.get('line1')],
         'impressions_fulllegend': [tuple(x for ln in l['legend'] for x in ln) for l in ur],
         'legends_line1': list({tuple(x for ln in l['legend'] for x in ln): tuple(l['line1'][0]) for l in ur if l.get('line1')}.values()),
         }
P(f'  (one copy per distinct full legend is unique by construction, so that unit is not a test; cache object kinds: {collections.Counter(l["obj"] for l in ur).most_common(4)})')
urres = []
for uname, ms in units.items():
    ms = [m for m in ms if len(m) >= 1]
    if not ms: continue
    for minlen in (1, 2):
        mm = [m for m in ms if len(m) >= minlen]
        for order in (0, 1, 2):
            for n in (1744, len(mm)):
                if n > len(mm): continue
                u, b, ratio = ratio_for(mm, n, order, rnd, draws=8 if n > 5000 else 12)
                urres.append(dict(unit=uname, minlen=minlen, order=order, n=n, u=u, b=b, ratio=ratio))
                P(f'  {uname:24s} min{minlen} order{order} n={n:5d}: unique {u:.3f} vs null {b:.3f} ratio {ratio:.3f}')
json.dump(urres, open(DARK + 'loop41_cycle3_ur3.json', 'w'), indent=0)
ur_r = [x['ratio'] for x in urres]; results = json.load(open(DARK + 'loop41_cycle3_namecalib.json')); rat = [x['ratio'] for x in results]
P(f'  Ur III ratio range {min(ur_r):.3f}-{max(ur_r):.3f}; Indus range {min(rat):.3f}-{max(rat):.3f}; overlap: {max(ur_r) >= min(rat)}')
# matched-choice comparison: same order, same minlen, n=1744, Indus seals die/frame18 vs Ur III seal_objects_line1 / legends_line1
P('  matched-choice table (n=1744): ')
for order in (0, 1, 2):
    for minlen in (1, 2):
        ind = [x['ratio'] for x in results if x['regime'] == 'die' and x['types'] == 'SEAL' and x['strip'] == 'frame18' and x['minlen'] == minlen and x['order'] == order and x['n'] <= 1744]
        u3 = {x['unit']: x['ratio'] for x in urres if x['minlen'] == minlen and x['order'] == order and x['n'] == 1744}
        P(f'    order{order} min{minlen}: Indus seals(die, frame18) {min(ind):.2f}-{max(ind):.2f} | Ur III ' + ', '.join(f'{k} {v:.2f}' for k, v in u3.items()))
