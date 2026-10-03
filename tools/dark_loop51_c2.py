#!/usr/bin/env python3
"""Loop 51 cycle 2: S149 tested directly. Kalibangan K-69..75/K-156 (91-56-2-417-890-892) vs Bahrain Gulf seal
no. 10 = Janabiyah (71-31-831-55-121-99-55, Wells; S276), plus the W56 link (H-14, L-115).

(a) Twin status under four equivalence modes: exact Wells (level), M collapse (bridge + S-DARK-27 proposals),
    M + composite split (56 = 55 55, 91 = 90 90, 93 = 1 90 from xlits), and sign-set overlap.
    Statistics: edit distance, longest common substring, shared sign types (multiset), shared bigrams.
(b) Chance: every pair of home texts from two different sites with the same two lengths (6 and 7; also 5-8),
    the same statistics -> what fraction of unrelated cross-site pairs is at least as similar as K-J?
(c) Does any text anywhere (home, foreign, IM77 West Asian in M space) share the opening pair of K
    (twins + grid) or its rare core 2-417 / 417-890 / 890-892? Person + 12/24-grid co-occurrence, foreign vs home.
(d) For every foreign text: nearest home text under each mode; is it a twin (d = 0), near twin (d <= 1 or
    normalised <= 0.25) or chance (what share of cross-site home pairs of those lengths is as close?).
usage: python3 tools/dark_loop51_c2.py LEVEL
"""
import sys, os, json, random, collections, itertools, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dark_loop51_common import *
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
rnd = random.Random(52)
OUT = f'data/derived/dark/loop51_c2_{LV}.txt'; LOG = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

objs = load(LV)
home = [o for o in objs if not o['foreign'] and not o['border'] and o['site'] != 'Unknown']
foreign = dedup([o for o in objs if o['foreign'] and len(o['seq']) >= 2])
K = next(o for o in objs if o['id'] == 'K-69')['seq']
J = next(o for o in objs if o['site'] == 'Janabiyah')['seq']
P(f'# S-DARK-51 cycle 2, level {LV}: S149 twin test. K = {K}, J = {J}')
MODES = {'exact': mapper('W'), 'M': mapper('M'), 'MC': mapper('MC')}

def lcs_len(a, b):
    best = 0; prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]: cur[j] = prev[j - 1] + 1; best = max(best, cur[j])
        prev = cur
    return best
def shared_types(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sum((ca & cb).values())
def shared_bigrams(a, b):
    sa = set(zip(a, a[1:])); sb = set(zip(b, b[1:])); return len(sa & sb)
def stats(a, b):
    return dict(edit=edit(a, b), lcs=lcs_len(a, b), types=shared_types(a, b), bigrams=shared_bigrams(a, b),
                same_first=a[0] == b[0])

P('\n== (a) K vs J under each mode')
for name, f in MODES.items():
    a, b = f(K), f(J)
    st = stats(a, b)
    P(f'  {name:5s}: K={a}\n         J={b}\n         edit {st["edit"]} (norm {st["edit"]/max(len(a),len(b)):.2f}), LCS {st["lcs"]}, shared types {st["types"]}, shared bigrams {st["bigrams"]}, same first sign {st["same_first"]}')

P('\n== (b) chance: cross-site home pairs with the same lengths')
by_len_site = collections.defaultdict(list)
for o in home: by_len_site[(len(o['seq']), home_label(o['site']))].append(o)
def cross_pairs(la, lb, nmax=20000):
    A = [o for (L, s), v in by_len_site.items() if L == la for o in v]
    B = [o for (L, s), v in by_len_site.items() if L == lb for o in v]
    pairs = [(x, y) for x in A for y in B if home_label(x['site']) != home_label(y['site'])]
    if len(pairs) > nmax: pairs = rnd.sample(pairs, nmax)
    return pairs
for name, f in MODES.items():
    a, b = f(K), f(J); kj = stats(a, b)
    for (la, lb) in [(6, 7), (None, None)]:
        if la: pairs = cross_pairs(6, 7)
        else:
            pairs = []
            for x in range(5, 9):
                for y in range(5, 9): pairs += cross_pairs(x, y, nmax=3000)
        n = len(pairs); cnt = collections.Counter()
        for x, y in pairs:
            st = stats(f(x['seq']), f(y['seq']))
            if st['types'] >= kj['types']: cnt['types>=KJ'] += 1
            if st['edit'] <= kj['edit']: cnt['edit<=KJ'] += 1
            if st['lcs'] >= max(1, kj['lcs']): cnt['lcs>=KJ'] += 1
            if st['types'] >= 2 and st['same_first']: cnt['same first + >=2 types'] += 1
            if st['edit'] <= 1: cnt['edit<=1'] += 1
            if st['lcs'] >= 2: cnt['lcs>=2'] += 1
        lab = '6 x 7' if la else '5-8 x 5-8'
        P(f'  {name:5s} lengths {lab:9s}: {n} cross-site pairs; share as similar as K-J: ' +
          ', '.join(f'{k} {v/n:.3f}' for k, v in sorted(cnt.items())) + f'   (K-J: edit {kj["edit"]}, lcs {kj["lcs"]}, types {kj["types"]})')

P('\n== (c) who else carries the K units; person + grid co-occurrence')
def has_run(seq, run):
    return any(tuple(seq[i:i + len(run)]) == tuple(run) for i in range(len(seq) - len(run) + 1))
for run in [(91, 56), (56, 2), (2, 417), (417, 890), (890, 892), (91, 55), (55, 55), (90, 55), (55, 90), (90, 56)]:
    hits = [(o['id'], o['site'], o['seq']) for o in dedup(objs) if has_run(o['seq'], run)]
    P(f'  run {run}: {len(hits)} distinct (site,text): {hits[:8]}')
# person + 12/24-grid co-occurrence, foreign vs home (distinct texts per site)
GRID = {55, 56}
def pg(o): return any(x in PERSON for x in o['seq']) and any(x in GRID for x in o['seq'])
hd = dedup(home); fd = foreign
fh = sum(pg(o) for o in fd); hh = sum(pg(o) for o in hd)
P(f'  person(90/91/93/71) + grid(55/56) in the same text: foreign {fh}/{len(fd)} vs home {hh}/{len(hd)} distinct texts: '
  f'{[(o["site"], o["seq"]) for o in fd if pg(o)]} | home {[(o["id"], o["site"], o["seq"]) for o in hd if pg(o)]}')
# Fisher one-sided
from math import comb, log, exp
def fisher_hi(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    p = 0.0
    for k in range(a, min(r1, c1) + 1):
        p += comb(c1, k) * comb(n - c1, r1 - k) / comb(n, r1)
    return p
P(f'  Fisher one-sided P = {fisher_hi(fh, len(fd) - fh, hh, len(hd) - hh):.2e}')
# in M space, IM77 West Asian lines: anything with the K opening (M for 91? unbridged) -> report bridge status
P(f'  bridge status of the K signs: ' + ', '.join(f'W{x}->{("M%d" % W2M[x]) if x in W2M else "unbridged"}{"(proposed)" if x in PROPOSED else ""}' for x in K))
P(f'  bridge status of the J signs: ' + ', '.join(f'W{x}->{("M%d" % W2M[x]) if x in W2M else "unbridged"}{"(proposed)" if x in PROPOSED else ""}' for x in J))

P('\n== (d) nearest home text for every foreign text, by mode; chance = share of cross-site home pairs of the same two lengths at least as close')
cache = {}
def chance_edit(la, lb, d, f, name):
    key = (la, lb, d, name)
    if key in cache: return cache[key]
    pairs = cross_pairs(la, lb, nmax=4000)
    if not pairs: cache[key] = (float('nan'), 0); return cache[key]
    # for a text of length la: nearest neighbour over OTHER sites' texts of length lb -> not the same thing as pair share.
    # we compute the pair share (random unrelated pair at least as close) and the NN share (a random text of length la
    # finds a text of length lb at another site within d)
    hit = sum(1 for x, y in pairs if edit(f(x['seq']), f(y['seq'])) <= d)
    cache[key] = (hit / len(pairs), len(pairs)); return cache[key]
summary = collections.Counter()
for o in sorted(foreign, key=lambda o: (o['group'], o['site'])):
    line = f"  {o['group']:5s} {o['site']:18s} {o['shape']:8s} {'-'.join(map(str, o['seq'])):30s}"
    for name, f in MODES.items():
        q = f(o['seq']); best = None
        for h in home:
            d = edit(q, f(h['seq']), cap=len(q))
            if best is None or d < best[0]: best = (d, h)
            if d == 0: break
        d, h = best
        nn_count = 0
        if d == 0:
            nn_count = sum(1 for hh in home if f(hh['seq']) == q)
        # NN share: how many home texts (any length) within d of a random cross-site home text of this length?
        verdict = 'TWIN' if d == 0 else 'near' if d <= 1 or d / max(len(q), len(h['seq'])) <= 0.25 else 'chance'
        if name == 'MC': summary[verdict] += 1
        line += f" | {name} d={d} -> {h['site'][:5]} {'-'.join(map(str, h['seq']))}{(' x%d' % nn_count) if d == 0 else ''} [{verdict}]"
    P(line)
P(f'  verdicts under MC mode: {dict(summary)} of {len(foreign)}')
# calibration: NN distance of a random home text of the same length to the nearest text at ANOTHER site
P('\n  calibration: for home texts of length L (sample 60 per L), nearest text at another site: share with d = 0, d <= 1')
f = MODES['MC']
for L in range(2, 9):
    cand = [o for o in home if len(o['seq']) == L]
    if not cand: continue
    samp = rnd.sample(cand, min(60, len(cand))); z = 0; one = 0
    for o in samp:
        q = f(o['seq']); best = 99
        for h in home:
            if home_label(h['site']) == home_label(o['site']): continue
            d = edit(q, f(h['seq']), cap=2)
            if d < best: best = d
            if best == 0: break
        z += best == 0; one += best <= 1
    fz = sum(1 for o in foreign if len(o['seq']) == L);
    P(f'    L={L}: home cross-site twin share {z/len(samp):.2f}, within 1 {one/len(samp):.2f} (n={len(samp)}); foreign texts of this length: {fz}')
P(f'\nwritten {OUT}')
