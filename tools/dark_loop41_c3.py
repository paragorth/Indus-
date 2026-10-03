"""Loop 41 cycle 3: (c) strict site x type x length stratification and Markov nulls for every order/nesting claim;
(f) the name-calibration ratio across all defensible granularity choices, Indus and Ur III sides.
Output: data/derived/dark/loop41_cycle3.txt"""
import sys, random, json, collections, statistics as st, itertools
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *
out = open(DARK + 'loop41_cycle3.txt', 'w')
def P(*a):
    print(*a); print(*a, file=out); out.flush()
C = load('canonical'); rnd = random.Random(413)
P('# Loop 41 cycle 3: strict stratification (c) and name-calibration range (f)')

def strat_key(r, s): return (r['site'], otype(r['type']), len(s))
def strat_permute(T, rnd):
    """tokens permuted across texts within site x type x length (positions free)."""
    groups = collections.defaultdict(list)
    for i, (r, s) in enumerate(T): groups[strat_key(r, s)].append(i)
    TT = list(T)
    for idx in groups.values():
        pool = [x for i in idx for x in T[i][1]]; rnd.shuffle(pool); k = 0
        for i in idx:
            L = len(T[i][1]); TT[i] = (T[i][0], tuple(pool[k:k + L])); k += L
    return TT
def strat_permute_pos(T, rnd):
    """tokens permuted across texts within site x type x length, POSITION KEPT (kills everything positional by construction;
    shows what survives as co-occurrence only)."""
    groups = collections.defaultdict(list)
    for i, (r, s) in enumerate(T): groups[strat_key(r, s)].append(i)
    TT = list(T)
    for idx in groups.values():
        L = len(T[idx[0]][1]); cols = []
        for p in range(L):
            col = [T[i][1][p] for i in idx]; rnd.shuffle(col); cols.append(col)
        for j, i in enumerate(idx): TT[i] = (T[i][0], tuple(cols[p][j] for p in range(L)))
    return TT
def markov_strat(T, rnd, order=1):
    groups = collections.defaultdict(list)
    for r, s in T: groups[(r['site'], otype(r['type']))].append(s)
    models = {g: markov_fit(v, order) for g, v in groups.items()}
    unis = {g: collections.Counter(x for s in v for x in s) for g, v in groups.items()}
    return [(r, markov_gen(models[(r['site'], otype(r['type']))], len(s), rnd, order, unis[(r['site'], otype(r['type']))])) for r, s in T]

def closer_count(T):
    texts = [s for _, s in T if len(s) >= 2]
    tok = collections.Counter(x for s in texts for x in s); fin = collections.Counter(); jw = collections.Counter(); je = collections.defaultdict(float)
    bylen = collections.defaultdict(list)
    for s in texts: bylen[len(s)].append(740 in s)
    jr = {L: sum(v) / len(v) for L, v in bylen.items()}
    for s in texts:
        t = list(s)
        while len(t) > 1 and t[-1] in SUF: t.pop()
        fin[t[-1]] += 1
        for x in set(s):
            if 740 in s: jw[x] += 1
            je[x] += jr[len(s)]
    return sum(1 for x, n in tok.items() if n >= 15 and x != 740 and x not in SUF and fin[x] / n >= 0.4 and (jw[x] / je[x] if je[x] else 9) <= 0.5)

def nulls_for(T, rnd, nn=25):
    """every claim statistic under four nulls: within-text shuffle, strata permutation, strata permutation position-kept, Markov-1 (site x type), Markov-2."""
    stats = {
        'closers': closer_count,
        'nest': lambda T: nest_stat([s for _, s in T])[0],
        'anagram_diff': lambda T: (lambda g, ag, p, d: d / p if p else float('nan'))(*anagram(T)),
        'fixed_share': lambda T: (lambda M, f, fr: f / M if M else float('nan'))(*fixed_pairs(T)),
        'mid_chg': lambda T: middle_closer_change(T, rnd, nnull=1)[0],
        'w2_twice': lambda T: w2_rules(T, rnd, nnull=1)[0][0],
        'w2_741': lambda T: w2_rules(T, rnd, nnull=1)[0][1],
        'opener_first': lambda T: frame_rates(T)['opener_first'],
        'closer_last': lambda T: frame_rates(T)['closer_last'],
        'fish_rev_share': lambda T: (lambda ok, rev: rev / (ok + rev) if ok + rev else float('nan'))(*fish_order(T)),
        'qual_ov': lambda T: qualifier_overlap(T, rnd, nnull=1)[0],
        'name_u': lambda T: uniq([m for m in (name_middle(s) for r, s in T if r['type'].startswith('SEAL')) if m]),
    }
    gens = {'shuffle': shuffled, 'strata': strat_permute, 'strata_pos': strat_permute_pos,
            'markov1': lambda T, r: markov_strat(T, r, 1), 'markov2': lambda T, r: markov_strat(T, r, 2)}
    obs = {k: f(T) for k, f in stats.items()}
    res = {k: {} for k in stats}
    for gname, g in gens.items():
        draws = collections.defaultdict(list)
        for _ in range(nn):
            G = g(T, rnd)
            for k, f in stats.items(): draws[k].append(f(G))
        for k in stats:
            v = sorted(x for x in draws[k] if x == x)
            res[k][gname] = (v[len(v) // 2], v[0], v[-1]) if v else (float('nan'),) * 3
    return obs, res

for lvl in ('seq_raw', 'seq_all'):
    T = dedup(C, lvl, 'die')
    obs, res = nulls_for(T, rnd, nn=25)
    P(f'\n## {lvl} die regime n={len(T)}: observed | null median [min,max] for shuffle / strata(site x type x length) / strata position-kept / Markov-1 / Markov-2 (site x type, lengths kept)')
    for k in obs:
        P(f'  {k:15s} obs {fmt(obs[k]):>8s} | ' + ' | '.join(f'{g} {fmt(res[k][g][0])} [{fmt(res[k][g][1])},{fmt(res[k][g][2])}]' for g in res[k]))

# held-out P1/P2 with Markov fitted on FIT (MD+H) only, generated held-out texts of the held-out lengths: what reuse does a big-city chain predict for a NEW text?
P('\n## S349 held-out P1/P2: generators fitted on Mohenjo-daro + Harappa only (no held-out leakage), held-out lengths kept')
for lvl in ('seq_raw', 'seq_all'):
    T = dedup(C, lvl, 'die'); A, H = heldout_split([(r, s) for r, s in T if len(s) >= 3])
    big = [(r, s) for r, s in A if r['site'] in BIG]; R = runs_of({s for _, s in big})
    ht = [s for _, s in H]; at = [s for _, s in A] + ht
    def p2(texts): return sum(any(s[i:i + 3] in R for i in range(len(s) - 2)) for s in texts) / len(texts)
    def p1(texts, allt):
        sub = set()
        for t in allt:
            if len(t) < 4: continue
            for n in (3, 4, 5):
                for i in range(len(t) - n + 1):
                    if n < len(t): sub.add(t[i:i + n])
        sm = [t for t in texts if 3 <= len(t) <= 5]; return sum(t in sub for t in sm) / max(1, len(sm))
    o1, o2 = p1(ht, at), p2(ht)
    for order in (1, 2):
        m = markov_fit([s for _, s in big], order); uni = collections.Counter(x for _, s in big for x in s)
        n1 = []; n2 = []
        for _ in range(40):
            gh = [markov_gen(m, len(s), rnd, order, uni) for s in ht]
            n1.append(p1(gh, [s for _, s in A] + gh)); n2.append(p2(gh))
        n1.sort(); n2.sort()
        P(f'  {lvl} held-out n={len(ht)} Markov-{order} fitted on MD+H: P1 {o1:.3f} vs {n1[20]:.3f} [{n1[0]:.3f},{n1[-1]:.3f}] = {o1/n1[20]:.1f}x | P2 {o2:.3f} vs {n2[20]:.3f} [{n2[0]:.3f},{n2[-1]:.3f}] = {o2/n2[20]:.1f}x')
    # and the honest small-site baseline: do OTHER IM77 small sites (Lothal, Kalibangan, Chanhu-daro) reuse MD+H runs at the same rate?
    for site in ('Lothal', 'Kalibangan', 'Chanhu-daro'):
        ss = [s for r, s in A if r['site'] == site]
        if len(ss) >= 20: P(f'  {lvl} in-sample small site {site} n={len(ss)}: P2 reuse of MD+H runs {p2(ss):.3f}')

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
def ngram_null(ms, order, rnd):
    m = markov_fit(ms, order); uni = collections.Counter(x for s in ms for x in s)
    return [markov_gen(m, len(s), rnd, order, uni) for s in ms]
def unigram_null(ms, rnd):
    pool = [x for s in ms for x in s]
    return [tuple(rnd.choice(pool) for _ in s) for s in ms]
def ratio_for(ms, n, order, rnd, draws=12):
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
                        for n in (len(ms), min(1744, len(ms))):
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

P('\n## (f) Ur III side (loop18 cache, 25,291 CDLI seal impressions with a legend): same generators, same granularity choices, n matched to Indus (1,744) and full')
ur = json.load(open(DARK + 'loop18_ur3_impressions.json'))
units = {'impressions_line1': [tuple(l['line1'][0]) for l in ur if l.get('line1')],
         'impressions_fulllegend': [tuple(x for ln in l['legend'] for x in ln) for l in ur],
         'legends_line1': list({tuple(x for ln in l['legend'] for x in ln): tuple(l['line1'][0]) for l in ur if l.get('line1')}.values()),
         'legends_full': list({tuple(x for ln in l['legend'] for x in ln) for l in ur}),
         'seal_objects_line1': [tuple(l['line1'][0]) for l in ur if l.get('line1') and l['obj'] == 'seal']}
urres = []
for uname, ms in units.items():
    ms = [m for m in ms if len(m) >= 1]
    for minlen in (1, 2):
        mm = [m for m in ms if len(m) >= minlen]
        for order in (0, 1, 2):
            for n in (1744, len(mm)):
                if n > len(mm): continue
                u, b, ratio = ratio_for(mm, n, order, rnd, draws=8 if n > 5000 else 12)
                urres.append(dict(unit=uname, minlen=minlen, order=order, n=n, u=u, b=b, ratio=ratio))
                P(f'  {uname:24s} min{minlen} order{order} n={n:5d}: unique {u:.3f} vs null {b:.3f} ratio {ratio:.3f}')
json.dump(urres, open(DARK + 'loop41_cycle3_ur3.json', 'w'), indent=0)
ur_r = [x['ratio'] for x in urres]; P(f'  Ur III ratio range {min(ur_r):.3f}-{max(ur_r):.3f}; Indus range {min(rat):.3f}-{max(rat):.3f}; overlap: {max(ur_r) >= min(rat)}')
# matched-choice comparison: same order, same minlen, n=1744, Indus seals die/frame18 vs Ur III seal_objects_line1 / legends_line1
P('  matched-choice table (n=1744): ')
for order in (0, 1, 2):
    for minlen in (1, 2):
        ind = [x['ratio'] for x in results if x['regime'] == 'die' and x['types'] == 'SEAL' and x['strip'] == 'frame18' and x['minlen'] == minlen and x['order'] == order and x['n'] <= 1744]
        u3 = {x['unit']: x['ratio'] for x in urres if x['minlen'] == minlen and x['order'] == order and x['n'] == 1744}
        P(f'    order{order} min{minlen}: Indus seals(die, frame18) {min(ind):.2f}-{max(ind):.2f} | Ur III ' + ', '.join(f'{k} {v:.2f}' for k, v in u3.items()))
