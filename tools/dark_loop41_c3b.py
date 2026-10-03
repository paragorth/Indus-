"""Loop 41 cycle 3b: (f) Indus-side name-calibration sweep (split from c3 for speed). Appends to loop41_cycle3.txt"""
import sys, random, json, collections, statistics as st
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *
out = open(DARK + 'loop41_cycle3.txt', 'a')
def P(*a):
    print(*a); print(*a, file=out); out.flush()
C = load('canonical'); rnd = random.Random(4133)
# ---------------------------------------------------------------- (f) name calibration range
P('\n## (f) Name-calibration ratio (unique share observed / unique share under a generator trained on the same strings), Indus side, every defensible choice')
CL11 = set(CL)
def mid_factory(strip):
    def f(seq):
        s = list(seq)
        if strip in ('frame18', 'frame11', 'closer_only', 'opener_only', 'frame18_drop'):
            cl = CLOSE18 if strip.startswith('frame18') else CL11
            if strip != 'opener_only':
                if s and s[-1] in (SUF18 if strip.startswith('frame18') else SUF): s = s[:-1]
                if s and s[-1] in cl: s = s[:-1]
            if strip != 'closer_only':
                if s and s[0] in (OP18 if strip.startswith('frame18') else OP): s = s[1:]
                if s and s[0] in MARK: s = s[1:]
            if strip == 'frame18_drop': s = [x for x in s if x not in CLOSE18 | OP18 | SUF18 | MARK]
        return tuple(s)
    return f
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
results = []
for lvl in ('seq_raw', 'seq_all'):
    for regime in ('rows', 'die', 'site_text'):
        T = dedup(C, lvl, regime)
        for types in ('SEAL', 'SEAL+TAG', 'ALL'):
            TT = [(r, s) for r, s in T if types == 'ALL' or r['type'].startswith('SEAL') or (types == 'SEAL+TAG' and r['type'].startswith('TAG'))]
            for strip in ('frame18', 'frame11', 'closer_only', 'opener_only', 'none', 'frame18_drop'):
                f = mid_factory(strip)
                for minlen in (1, 2, 3):
                    ms = [m for m in (f(s) for _, s in TT) if len(m) >= minlen]
                    if len(ms) < 100: continue
                    for order in (0, 1, 2):
                        for n in sorted({len(ms), min(1744, len(ms))}):
                            u, b, ratio = ratio_for(ms, n, order, rnd)
                            results.append(dict(level=lvl, regime=regime, types=types, strip=strip, minlen=minlen, order=order, n=n, u=u, b=b, ratio=ratio))
json.dump(results, open(DARK + 'loop41_cycle3_namecalib.json', 'w'), indent=0)
rat = [r['ratio'] for r in results]
P(f'  {len(results)} combinations; ratio range {min(rat):.3f}-{max(rat):.3f}; median {st.median(rat):.3f}')
for key in ('level', 'regime', 'types', 'strip', 'minlen', 'order'):
    P(f'  by {key}: ' + '; '.join(f'{v}: {min(x["ratio"] for x in results if x[key]==v):.2f}-{max(x["ratio"] for x in results if x[key]==v):.2f} (med {st.median([x["ratio"] for x in results if x[key]==v]):.2f})' for v in sorted({x[key] for x in results}, key=str)))
low = sorted(results, key=lambda x: x['ratio'])[:6]
P('  lowest ratios: ' + ' | '.join(f"{x['level']}/{x['regime']}/{x['types']}/{x['strip']}/min{x['minlen']}/order{x['order']}/n{x['n']} u={x['u']:.2f} b={x['b']:.2f} r={x['ratio']:.2f}" for x in low))
# the headline choice
hd = [x for x in results if x['level'] == 'seq_raw' and x['regime'] == 'rows' and x['types'] == 'SEAL' and x['strip'] == 'frame18' and x['minlen'] == 2 and x['order'] == 1]
P('  headline choice (rows/SEAL/frame18/min2/bigram): ' + '; '.join(f"n{x['n']} u={x['u']:.3f} b={x['b']:.3f} r={x['ratio']:.3f}" for x in hd))

