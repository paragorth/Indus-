#!/usr/bin/env python3
"""Loop 11 cycle 3.
(A) Allograph test for the unmerged high-shape-similarity substitution candidates from
    cycle 2: context sharing (left/right neighbour cosine) vs frequency-matched sign pairs,
    and split by site / medium (an allograph by local hand should be site- or medium-split;
    a different word should not).
(B) Optional signs (indels) with a slot-aware null: which signs drop out, in which position,
    and is the shorter text the frequent one (additions to a base) or the longer one (omissions)?
(C) Confusion rate vs frequency, a-side only, with a position-matched null.
Usage: python3 tools/dark11_cycle3.py LEVEL [NPERM] [SEED]
"""
import json, sys, random, collections, itertools, math
import numpy as np
from scipy.stats import spearmanr

ROOT = '/home/user/Indus-'
LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 33
rng = random.Random(SEED)
corpus = json.load(open(f'{ROOT}/data/derived/merged-corpus-canonical.json'))
bridge = json.load(open(f'{ROOT}/data/derived/bridge_extended.json'))
allo = json.load(open(f'{ROOT}/data/derived/sign_allographs_levels.json'))
variants = json.load(open(f'{ROOT}/data/derived/sign_variant_classes.json'))
glyph_signs = json.load(open(f'{ROOT}/data/derived/glyph_sim_signs.json'))
glyph_sim = np.load(f'{ROOT}/data/derived/glyph_sim.npy')
gidx = {s: i for i, s in enumerate(glyph_signs)}

# transitive union of every merge / variant-class statement
parent = {}
def find(x):
    while parent.get(x, x) != x:
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for m in allo['merges']: union(m['form'], m['into'])
for tier in ('tierA', 'tierB'):
    for f, into in variants[tier].items(): union(int(f), into)
def classed(a, b): return find(a) == find(b)

SLOT = {}
for s in (861, 817, 820): SLOT[s] = 'opener'
for s in (2, 60): SLOT[s] = 'marker'
for s in (235, 31): SLOT[s] = 'mid-initial'
for s in (740, 390, 405, 406, 407, 156, 151, 527, 526, 520, 595): SLOT[s] = 'closer'
for s in (400, 90): SLOT[s] = 'suffix'
for s in range(3, 60): SLOT.setdefault(s, 'numeral')

objs = []
for r in corpus:
    if r['complete'] != 'Y': continue
    seq = tuple(r[LEVEL])
    if len(seq) < 3: continue
    objs.append(dict(cisi=r['cisi'], site=r['site'], tc=r['type'].split(':')[0], area=r['area-section'], seq=seq))
texts = collections.defaultdict(list)
for o in objs: texts[o['seq']].append(o)
sc = collections.Counter(s for o in objs for s in o['seq'])
# whole corpus (any length, any completeness) for context profiles
allseqs = [tuple(r[LEVEL]) for r in corpus if len(r[LEVEL]) >= 2]
allobjs = [(tuple(r[LEVEL]), r['site'], r['type'].split(':')[0]) for r in corpus if len(r[LEVEL]) >= 1]
sc_all = collections.Counter(s for q in allseqs for s in q)

def find_subs(texts, minfreq):
    freq = {t for t, l in texts.items() if len(l) >= minfreq}
    by_del = collections.defaultdict(list)
    for T in texts:
        for p in range(len(T)): by_del[T[:p] + T[p + 1:]].append((T, p))
    seen, subs, indels = set(), [], []
    for key, lst in by_del.items():
        for (T1, p1), (T2, p2) in itertools.combinations(lst, 2):
            if p1 != p2 or T1 == T2: continue
            if T1 not in freq and T2 not in freq: continue
            k = frozenset((T1, T2))
            if k in seen: continue
            seen.add(k); subs.append((T1, T2, p1, T1[p1], T2[p1]))
        if key in texts and len(key) >= 3:
            for (T, p) in lst:
                if T not in freq and key not in freq: continue
                if (T, key) in seen: continue
                seen.add((T, key)); indels.append((T, key, p, T[p]))
    return subs, indels

out = []; P = out.append
P(f'LOOP 11 cycle 3  level={LEVEL} nperm={NPERM} seed={SEED}')

# ---------------- (A) allograph test ----------------
subs2, indels2 = find_subs(texts, 2)
cands = collections.Counter()
for (T1, T2, p, a, b) in subs2:
    if not classed(a, b) and a in gidx and b in gidx and glyph_sim[gidx[a], gidx[b]] >= 0.80:
        cands[(min(a, b), max(a, b))] += 1

def ctx_vec(s):
    L = collections.Counter(); R = collections.Counter()
    for q in allseqs:
        for i, x in enumerate(q):
            if x != s: continue
            L[q[i - 1] if i > 0 else '^'] += 1
            R[q[i + 1] if i < len(q) - 1 else '$'] += 1
    return L, R
def cos(c1, c2):
    keys = set(c1) | set(c2)
    v1 = np.array([c1.get(k, 0) for k in keys], float); v2 = np.array([c2.get(k, 0) for k in keys], float)
    if v1.sum() == 0 or v2.sum() == 0: return float('nan')
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
CTX = {}
def ctx(s):
    if s not in CTX: CTX[s] = ctx_vec(s)
    return CTX[s]
def ctx_sim(a, b):
    La, Ra = ctx(a); Lb, Rb = ctx(b)
    return np.nanmean([cos(La, Lb), cos(Ra, Rb)])
def dist_profile(s):
    site = collections.Counter(); med = collections.Counter()
    for q, st, tc in allobjs:
        n = q.count(s)
        if n: site[st if st in ('Harappa', 'Mohenjo-daro') else 'other'] += n; med[tc if tc in ('SEAL', 'TAB') else 'other'] += n
    return site, med
def split_score(a, b):
    """1 - cosine between the site profiles, and between medium profiles (0 = same distribution)"""
    sa, ma = dist_profile(a); sb, mb = dist_profile(b)
    return 1 - cos(sa, sb), 1 - cos(ma, mb)
signs_by_freq = sorted(sc_all, key=lambda s: sc_all[s])
def freq_matched(s, k=20):
    i = signs_by_freq.index(s); lo = max(0, i - k); hi = min(len(signs_by_freq), i + k)
    pool = [x for x in signs_by_freq[lo:hi] if x != s]
    return rng.choice(pool)

P('(A) ALLOGRAPH TEST for unmerged substitution pairs with glyph similarity >= 0.80 (any class), tokens >= 5 each')
P('    ctx = mean cosine of left/right neighbour profiles (whole corpus); null = 500 frequency-matched sign pairs; site/medium split = 1 - cosine of distribution profiles (higher = more split)')
P('    pair           frames glyph  ctx    ctx-null(P)        site-split  med-split  tokens      slots            M           verdict')
null_ctx_all = []
for _ in range(500):
    x = rng.choice([s for s in sc_all if sc_all[s] >= 5]); y = freq_matched(x)
    null_ctx_all.append(ctx_sim(x, y))
null_split = [split_score(rng.choice([s for s in sc_all if sc_all[s] >= 5]), 0) for _ in range(0)]
rowsA = []
for (a, b), n in cands.most_common():
    if sc_all[a] < 5 or sc_all[b] < 5: continue
    c = ctx_sim(a, b)
    # pair-specific null: partners frequency-matched to b
    nl = [ctx_sim(a, freq_matched(b)) for _ in range(200)]
    pv = (1 + sum(1 for x in nl if x >= c)) / (len(nl) + 1)
    ss, ms = split_score(a, b)
    # split null: frequency-matched pairs
    nls = [split_score(a, freq_matched(b)) for _ in range(100)]
    ps = (1 + sum(1 for x in nls if x[0] >= ss)) / (len(nls) + 1); pm = (1 + sum(1 for x in nls if x[1] >= ms)) / (len(nls) + 1)
    verdict = ('ALLOGRAPH-LIKE' if pv < 0.05 and ss < 0.15 and ms < 0.15 else 'local/medium variant?' if pv < 0.05 else 'different word')
    g = glyph_sim[gidx[a], gidx[b]]
    P(f'    {a:>4}<->{b:<4}    {n:3d}   {g:.2f}  {c:.2f}   {np.mean(nl):.2f} (P={pv:.3f})   {ss:.2f} (P={ps:.2f})  {ms:.2f} (P={pm:.2f})  {sc_all[a]:4d}/{sc_all[b]:<4d}  {SLOT.get(a,"-")}/{SLOT.get(b,"-"):12s} M{bridge.get(str(a),"?")}/M{bridge.get(str(b),"?")}  {verdict}')
    rowsA.append((a, b, n, float(g), float(c), pv, ss, ms, verdict))
P(f'    reference: frequency-matched random pairs ctx mean {np.nanmean(null_ctx_all):.2f} (95% {np.nanpercentile(null_ctx_all,2.5):.2f}..{np.nanpercentile(null_ctx_all,97.5):.2f})')
# how do the KNOWN strong allographs score on the same test (calibration)?
P('    calibration, strong Wells merges on the same test:')
for m in allo['merges']:
    if m['level'] != 'strong': continue
    a, b = m['form'], m['into']
    if sc_all[a] < 5 or sc_all[b] < 5: continue
    c = ctx_sim(a, b); nl = [ctx_sim(a, freq_matched(b)) for _ in range(200)]
    pv = (1 + sum(1 for x in nl if x >= c)) / (len(nl) + 1); ss, ms = split_score(a, b)
    P(f'      {a}<->{b} ctx {c:.2f} (null {np.mean(nl):.2f}, P={pv:.3f}) site-split {ss:.2f} med-split {ms:.2f} tokens {sc_all[a]}/{sc_all[b]}')

# ---------------- (B) optional signs ----------------
P('(B) OPTIONAL SIGNS: indel frames (minfreq 3), direction and position')
subs3, indels3 = find_subs(texts, 3)
longer_freq = sum(1 for (T, S, p, d) in indels3 if len(texts[T]) > len(texts[S]))
shorter_freq = sum(1 for (T, S, p, d) in indels3 if len(texts[T]) < len(texts[S]))
P(f'  {len(indels3)} indel frames: longer text is the commoner one in {longer_freq}, shorter commoner in {shorter_freq}, tie {len(indels3)-longer_freq-shorter_freq}')
pos = collections.Counter('initial' if p == 0 else 'final' if p == len(T) - 1 else 'medial' for (T, S, p, d) in indels3)
P(f'  position of the extra sign: {dict(pos)}  (expected if uniform over positions: initial = final = {sum(1/len(T) for (T,S,p,d) in indels3):.1f}, medial = {sum((len(T)-2)/len(T) for (T,S,p,d) in indels3):.1f})')
# slot-aware null: for each indel frame, which sign of the longer text could be removed?  Compare the observed removed sign
# with the token-frequency null restricted to the same position class (initial/medial/final) in texts of that length and object class
del_frames = collections.Counter(d for (T, S, p, d) in indels3)
pos_tok = collections.defaultdict(collections.Counter)
for o in objs:
    L = len(o['seq'])
    for p, s in enumerate(o['seq']):
        pos_tok[('initial' if p == 0 else 'final' if p == L - 1 else 'medial', o['tc'])][s] += 1
def pos_of(T, p): return 'initial' if p == 0 else 'final' if p == len(T) - 1 else 'medial'
exp = collections.Counter()
for (T, S, p, d) in indels3:
    dist = pos_tok[(pos_of(T, p), texts[T][0]['tc'])]; tot = sum(dist.values())
    for s, n in dist.items(): exp[s] += n / tot
P('  per-sign deletions vs position+object-class matched expectation (signs with >=2 frames):')
P('    sign frames  exp  ratio  P(binom-ish perm)  tokens  M      pos')
sim_tot = collections.defaultdict(list)
for it in range(1000):
    cnt = collections.Counter()
    for (T, S, p, d) in indels3:
        dist = pos_tok[(pos_of(T, p), texts[T][0]['tc'])]
        items = list(dist.items()); tot = sum(n for _, n in items); x = rng.random() * tot
        for s, n in items:
            x -= n
            if x <= 0: cnt[s] += 1; break
    for s in del_frames: sim_tot[s].append(cnt[s])
dp = collections.defaultdict(collections.Counter)
for (T, S, p, d) in indels3: dp[d][pos_of(T, p)] += 1
optional = []
for d, n in del_frames.most_common():
    if n < 2: break
    pv = (1 + sum(1 for x in sim_tot[d] if x >= n)) / 1001
    P(f'    {d:>4} {n:5d} {exp[d]:5.1f} {n/exp[d] if exp[d] else float("inf"):5.2f}   {pv:.3f}            {sc[d]:5d}  M{bridge.get(str(d),"?")!s:9s} {dict(dp[d])}')
    optional.append((d, n, float(exp[d]), pv))
# are optional signs the GRAMMAR deletables / qualifiers?  Set test with the position-matched null
known_opt = {235, 31, 400, 90, 2, 60, 1, 100, 705, 706, 33, 240, 233, 231}
ko = sum(del_frames[s] for s in known_opt)
sim = []
for it in range(2000):
    s_ = 0
    for (T, S, p, d) in indels3:
        dist = pos_tok[(pos_of(T, p), texts[T][0]['tc'])]
        items = list(dist.items()); tot = sum(n for _, n in items); x = rng.random() * tot
        for s, n in items:
            x -= n
            if x <= 0: s_ += (s in known_opt); break
    sim.append(s_)
P(f'  known deletable/qualifier set (GRAMMAR mid-initial, suffix, markers, fish qualifiers, U-stroke 3, W100): {ko} of {len(indels3)} deletions vs position-matched null {np.mean(sim):.1f} (max {max(sim)}) P={(1+sum(1 for x in sim if x>=ko))/2001:.4f}')
# the jar: how often is the jar closer itself optional?
P(f'  jar W740 deleted in {del_frames[740]} frames (expected {exp[740]:.1f}); opener (817/820/861) deleted in {sum(del_frames[s] for s in (817,820,861))} (expected {sum(exp[s] for s in (817,820,861)):.1f})')
P('  indel frames (longer -> shorter, [deleted]):')
for (T, S, p, d) in sorted(indels3, key=lambda x: -len(texts[x[0]]) - len(texts[x[1]]))[:40]:
    P(f'    {"-".join(str(s) if i != p else "[" + str(s) + "]" for i, s in enumerate(T))}  n={len(texts[T])} vs {"-".join(map(str,S))} n={len(texts[S])}')

# ---------------- (C) frequency effect, cleaner ----------------
P('(C) CONFUSION RATE vs FREQUENCY, both sides, position-matched null (signs >= 10 tokens)')
subs3_ev = collections.Counter()
for (_, _, _, a, b) in subs3: subs3_ev[a] += 1; subs3_ev[b] += 1
signs_c = [s for s in sc if sc[s] >= 10]
freqs = np.log(np.array([sc[s] for s in signs_c]))
rate = np.array([subs3_ev[s] / sc[s] for s in signs_c])
rho = spearmanr(freqs, rate)[0]
# null: for each frame, both a and b drawn from the position+class distribution (frame positions fixed)
nr = []
for it in range(300):
    ev = collections.Counter()
    for (T1, T2, p, a, b) in subs3:
        dist = pos_tok[(pos_of(T1, p), texts[T1][0]['tc'])]; items = list(dist.items()); tot = sum(n for _, n in items)
        for _ in range(2):
            x = rng.random() * tot
            for s, n in items:
                x -= n
                if x <= 0: ev[s] += 1; break
    nr.append(spearmanr(freqs, np.array([ev[s] / sc[s] for s in signs_c]))[0])
P(f'  rho obs {rho:.3f}; null mean {np.mean(nr):.3f} (95% {np.percentile(nr,2.5):.3f}..{np.percentile(nr,97.5):.3f}); P(obs<=null) = {(1+sum(1 for x in nr if x<=rho))/(len(nr)+1):.4f}')
# binned view
bins = [(10, 30), (30, 100), (100, 300), (300, 5000)]
for lo, hi in bins:
    ss = [s for s in signs_c if lo <= sc[s] < hi]
    P(f'    tokens {lo}-{hi}: {len(ss)} signs, confusion events per 100 tokens {100*sum(subs3_ev[s] for s in ss)/sum(sc[s] for s in ss):.2f}')
json.dump(dict(level=LEVEL, allograph=rowsA, optional=optional, rho=float(rho), rho_null=float(np.mean(nr)),
               indel_dir=dict(longer_freq=longer_freq, shorter_freq=shorter_freq, n=len(indels3))),
          open(f'{ROOT}/data/derived/dark/loop11_c3_{LEVEL}.json', 'w'), indent=1)
print('\n'.join(out))
