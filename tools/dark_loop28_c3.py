"""S-DARK-28 cycle 3: predictions.  (a) Rare derived signs (1-4 tokens) should appear in the
contexts of their base if derivation is an operator layer: for each rare derived sign, the
share of its tokens whose left / right neighbour and position bin fall inside the base's
majority sets, vs 1,000 draws of a frequency-matched random base.  (b) Holes: frequent bases
(>= 30 tokens) with no attested derivative of a given type; rare signs that graphically
contain such a base (template coverage >= 0.85, dark_loop28_c1b) are the candidate fillers,
and their contexts are scored the same way.  (c) IM77: signs with no Wells counterpart
(unbridged) scored against numerically adjacent Mahadevan signs (shape families) as a weak
graphic proxy, vs random adjacency.
usage: python3 tools/dark_loop28_c3.py seq_raw|seq_strong|seq_all
"""
import sys, json, random, collections, csv, time
import numpy as np
sys.path.insert(0, 'tools')
random.seed(28); np.random.seed(28)
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
OUT = 'data/derived/dark/'; NPERM = 1000
E = json.load(open(OUT + 'loop28_edges_all.json'))
M = json.load(open('data/derived/merged-corpus-canonical.json'))
NUM = {1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 55, 56}
TEXTS = [[x for x in (r.get(LV) or []) if x not in (0, 999)] for r in M if r['complete'] == 'Y']
TEXTS = [t for t in TEXTS if t]
freq = collections.Counter(x for t in TEXTS for x in t)
def posbin(i, n):
    if n == 1: return 'alone'
    return 'first' if i == 0 else 'last' if i == n - 1 else 'mid'
CTX = collections.defaultdict(lambda: dict(L=collections.Counter(), R=collections.Counter(), pos=collections.Counter(), toks=[]))
for t in TEXTS:
    n = len(t)
    for i, x in enumerate(t):
        l = t[i - 1] if i > 0 else '#'; r = t[i + 1] if i < n - 1 else '#'
        c = CTX[x]; c['L'][l] += 1; c['R'][r] += 1; c['pos'][posbin(i, n)] += 1; c['toks'].append((l, r, posbin(i, n)))
def majority(cnt, share=0.6):
    tot = sum(cnt.values()); acc = 0; s = set()
    for k, v in cnt.most_common():
        s.add(k); acc += v
        if acc / tot >= share: break
    return s
MAJ = {x: dict(L=majority(c['L']), R=majority(c['R']), pos=majority(c['pos'])) for x, c in CTX.items() if len(c['toks']) >= 10}
def score(d, b):
    """share of d's tokens whose left, right, position fall in b's majority sets (3 numbers)"""
    m = MAJ[b]; toks = CTX[d]['toks']
    return np.array([np.mean([l in m['L'] for l, r, p in toks]), np.mean([r in m['R'] for l, r, p in toks]), np.mean([p in m['pos'] for l, r, p in toks])])
BASES = sorted(MAJ)
def pool_for(b, exclude):
    fb = freq[b]
    return [r for r in BASES if r != b and r not in exclude and fb / 2 <= freq[r] <= fb * 2]
rep = [f'# LOOP 28 cycle 3 ({LV}): do derivatives appear where their base appears? ({time.strftime("%Y-%m-%dT%H:%M")})']
rep.append(f'texts {len(TEXTS)}; edges {len(E)}; bases with >= 10 tokens {len(BASES)}')
linked = collections.defaultdict(set)
for e in E: linked[e['derived']].add(e['base']); linked[e['base']].add(e['derived'])
# (a) rare derived signs
rep.append('\n## (a) rare derived signs (1-4 tokens) in the contexts of their base (edges of mid/high confidence)')
res = collections.defaultdict(list)
for e in E:
    d, b = e['derived'], e['base']
    if e.get('conf') == 'low' or b not in MAJ or not (1 <= freq.get(d, 0) <= 4) or d in NUM: continue
    pool = pool_for(b, linked[d])
    if len(pool) < 5: continue
    obs = score(d, b); nul = np.array([score(d, r) for r in random.choices(pool, k=200)])
    res[e['type']].append((d, b, obs, nul.mean(0), freq[d]))
    res['ALL'].append((d, b, obs, nul.mean(0), freq[d]))
rep.append(f'{"type":10s} {"n":>3s} | left-ctx hit obs/null | right-ctx obs/null | position obs/null | P (perm, mean of 3)')
for t in ['enclosure', 'roof', 'strokes', 'doubling', 'ligature', 'ALL']:
    rs = res.get(t)
    if not rs: continue
    O = np.array([r[2] for r in rs]); N = np.array([r[3] for r in rs])
    # permutation P: per edge draw a random base score; compare mean of means
    obs_m = O.mean(); sims = []
    for _ in range(NPERM):
        sims.append(np.mean([random.choice(r[3]) if False else r[3].mean() + np.random.randn() * 0 for r in rs]))
    # exact null: redo draws (cheap): one random base per edge
    sims = []
    for _ in range(300):
        sims.append(np.mean([score(r[0], random.choice(pool_for(r[1], linked[r[0]]))).mean() for r in rs]))
    sims = np.array(sims)
    rep.append(f'{t:10s} {len(rs):3d} | {O[:,0].mean():.2f}/{N[:,0].mean():.2f} | {O[:,1].mean():.2f}/{N[:,1].mean():.2f} | {O[:,2].mean():.2f}/{N[:,2].mean():.2f} | P = {np.mean(sims >= obs_m):.3f} (obs {obs_m:.2f}, null {sims.mean():.2f})')
rep.append('   detail: derived <- base (freq) left/right/pos hit vs null')
for t in ['enclosure', 'roof', 'strokes', 'doubling', 'ligature']:
    for d, b, o, n, f in res.get(t, []):
        rep.append(f'   {t:9s} W{d:<4d} <- W{b:<4d} ({f}) {o[0]:.2f}/{n[0]:.2f} {o[1]:.2f}/{n[1]:.2f} {o[2]:.2f}/{n[2]:.2f} ctx: ' + ' '.join(f'{l}_{r}' for l, r, p in CTX[d]['toks'][:4]))
# (b) holes
rep.append('\n## (b) holes: frequent bases (>= 30 tokens) without an attested derivative of each type, and rare signs that contain them graphically')
try:
    import dark_loop28_c1b as C
    have_c1b = True
except Exception as ex:
    have_c1b = False; rep.append(f'   (c1b import failed: {ex})')
if have_c1b:
    FB = [b for b in BASES if freq[b] >= 30 and b not in NUM]
    RARE = [w for w in C.SIGNS if 1 <= freq.get(w, 0) <= 4 and w not in NUM]
    types_of = collections.defaultdict(set)
    for e in E: types_of[e['base']].add(e['type'])
    for t in ['enclosure', 'strokes', 'doubling', 'ligature']:
        holes = [b for b in FB if t not in types_of[b]]
        rep.append(f'   {t}: {len(holes)} of {len(FB)} frequent bases have no attested {t} derivative: ' + ' '.join(f'W{b}' for b in holes[:40]) + (' ...' if len(holes) > 40 else ''))
    # candidate fillers: rare signs containing a frequent base at coverage >= 0.85 (template search restricted to FB)
    fill = []
    for w in RARE:
        hits = C.analyse(C.G[w], w, 0.85, bases=FB)
        for h in C.best_hits(hits):
            if h['type'] == 'same': continue
            b = h['base']
            if h['type'].rstrip('+') in types_of[b]: continue   # not a hole
            pool = pool_for(b, linked[w] | {b})
            if len(pool) < 5: continue
            o = score(w, b); n = np.array([score(w, r) for r in random.choices(pool, k=200)]).mean(0)
            fill.append((w, b, h['type'], h['score'], h['explained'], freq[w], o, n))
    rep.append(f'   rare signs graphically containing a frequent base in a hole (coverage >= 0.85): {len(fill)}')
    if fill:
        O = np.array([f[6] for f in fill]); N = np.array([f[7] for f in fill])
        rep.append(f'   their context hit rates (left/right/pos) {O[:,0].mean():.2f}/{O[:,1].mean():.2f}/{O[:,2].mean():.2f} vs null {N[:,0].mean():.2f}/{N[:,1].mean():.2f}/{N[:,2].mean():.2f}')
        for w, b, t, cov, ex, f, o, n in sorted(fill, key=lambda z: -z[4])[:60]:
            rep.append(f'     W{w:<4d} contains W{b:<4d} {t:11s} cov {cov:.2f} expl {ex:.2f} freq {f} | hits {o[0]:.2f}/{o[1]:.2f}/{o[2]:.2f} vs {n[0]:.2f}/{n[1]:.2f}/{n[2]:.2f} | ctx ' + ' '.join(f'{l}_{r}' for l, r, p in CTX[w]['toks'][:3]))
# (c) IM77 unbridged signs vs numerically adjacent M signs
rep.append('\n## (c) IM77 signs with no Wells counterpart: do they take the contexts of their shape-family neighbours (|dM| <= 2)?')
BR = {int(k): v for k, v in json.load(open('data/derived/bridge_extended.json')).items()}
BR[798] = [53]; BR[806] = [389]
bridged = set(m for v in BR.values() for m in v)
IM = []
with open('data/im77/im77_corpus_lines.csv') as f:
    for r in csv.DictReader(f):
        s = [int(x) for x in r['signs_clean'].split() if x.isdigit() and int(x) > 0]
        if s: IM.append(s)
ifreq = collections.Counter(x for t in IM for x in t)
ICTX = collections.defaultdict(lambda: dict(L=collections.Counter(), R=collections.Counter(), pos=collections.Counter(), toks=[]))
for t in IM:
    n = len(t)
    for i, x in enumerate(t):
        l = t[i - 1] if i > 0 else '#'; r = t[i + 1] if i < n - 1 else '#'
        c = ICTX[x]; c['L'][l] += 1; c['R'][r] += 1; c['pos'][posbin(i, n)] += 1; c['toks'].append((l, r, posbin(i, n)))
IMAJ = {x: dict(L=majority(c['L']), R=majority(c['R']), pos=majority(c['pos'])) for x, c in ICTX.items() if len(c['toks']) >= 10}
def iscore(d, b):
    m = IMAJ[b]; toks = ICTX[d]['toks']
    return np.array([np.mean([l in m['L'] for l, r, p in toks]), np.mean([r in m['R'] for l, r, p in toks]), np.mean([p in m['pos'] for l, r, p in toks])])
unb = [m for m in ifreq if m not in bridged and m not in range(86, 122) and 2 <= ifreq[m] <= 10]
pairs = []
for m in unb:
    nb = [b for b in IMAJ if b != m and abs(b - m) <= 2 and b not in range(86, 122)]
    for b in nb:
        pool = [r for r in IMAJ if abs(r - m) > 5 and r not in range(86, 122) and ifreq[b] / 2 <= ifreq[r] <= ifreq[b] * 2]
        if len(pool) < 5: continue
        pairs.append((m, b, iscore(m, b), np.array([iscore(m, r) for r in random.choices(pool, k=100)]).mean(0)))
if pairs:
    O = np.array([p[2] for p in pairs]); N = np.array([p[3] for p in pairs])
    rep.append(f'   {len(unb)} unbridged IM77 signs (2-10 tokens), {len(pairs)} (sign, adjacent-number base) pairs: context hits {O[:,0].mean():.2f}/{O[:,1].mean():.2f}/{O[:,2].mean():.2f} vs random bases {N[:,0].mean():.2f}/{N[:,1].mean():.2f}/{N[:,2].mean():.2f}')
    diff = (O - N).mean(1)
    rep.append(f'   pairs where the adjacent base beats the random base on the mean of 3: {np.mean(diff > 0):.2f} (sign test P = {min(1, 2 * sum(np.random.binomial(len(diff), 0.5, 2000) >= (diff > 0).sum()) / 2000):.3f})')
open(OUT + f'loop28_cycle3_{LV}.txt', 'w').write('\n'.join(rep) + '\n')
print('\n'.join(l for l in rep if not l.startswith('     ')))
