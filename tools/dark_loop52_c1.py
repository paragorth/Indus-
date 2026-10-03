"""Loop 52, cycle 1: HOW MANY SIGNS? A merge graph from four independent evidence types.

E1 contextual allography: the choice between a and b is predicted by the previous or next sign
   (MI vs labels permuted within site x object type, 300x), S-DARK-46 method on every candidate pair.
E2 free interchange: left+right neighbour profiles of a and b are more alike than those of
   frequency-matched partners (S-DARK-24.2 / 11.3 method), plus minimal pairs (texts identical but a<->b).
E3 graphic derivation: loop28 edge of type strokes or doubling (S-DARK-28: paradigm-mates), or
   glyph_sim_fine at or above the level of the S268 strong merges.
E4 transcriber disagreement: Wells's a and b are read as the same Mahadevan sign on the aligned objects
   (loop24_positions), or directly substituted (Wells a where Mahadevan reads the M partner of b) >= 2x.
Vetoes: a and b adjacent in >= 2 texts (S268 rule 4); both signs stroke numerals W1-60 (value matters).
Merge where >= k evidence types agree (k = 1, 2, 3); connected components; projected onto seq_raw,
seq_strong, seq_all. Control: the same number of edges drawn at random among candidate pairs.
Writes data/derived/dark/loop52_cycle1.txt and loop52_merges.json.
"""
import json, collections, itertools, random, sys
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
rng = np.random.default_rng(52)
random.seed(52)

corpus = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
levels = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))['merges']
glyph_signs = json.load(open(ROOT + 'data/derived/glyph_sim_signs.json'))
G = np.load(ROOT + 'data/derived/glyph_sim_fine.npy')
gidx = {s: i for i, s in enumerate(glyph_signs)}
edges28 = json.load(open(ROOT + 'data/derived/dark/loop28_edges.json'))
pos24 = json.load(open(ROOT + 'data/derived/dark/loop24_positions.json'))
bridge = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
props = json.load(open(ROOT + 'data/derived/dark/bridge_proposals.json'))['proposals']

MIN_TOK = 8
NUMERAL = set(range(1, 61))

tok = collections.Counter(s for r in corpus for s in r['seq_raw'])
signs = sorted(tok)
freq_signs = [s for s in signs if tok[s] >= MIN_TOK]
strata = [r['site'] + '|' + r['type'].split(':')[0] for r in corpus]

# ------------------------------------------------------------------ neighbour profiles (seq_raw)
BOUND = -1
left = collections.defaultdict(collections.Counter)
right = collections.defaultdict(collections.Counter)
for r in corpus:
    q = r['seq_raw']
    for i, s in enumerate(q):
        left[s][q[i - 1] if i > 0 else BOUND] += 1
        right[s][q[i + 1] if i + 1 < len(q) else BOUND] += 1
ctx_keys = sorted(set(k for s in freq_signs for k in left[s]) | set(k for s in freq_signs for k in right[s]))
kidx = {k: i for i, k in enumerate(ctx_keys)}
K = len(ctx_keys)
prof = {}
for s in freq_signs:
    v = np.zeros(2 * K)
    for k, c in left[s].items(): v[kidx[k]] += c
    for k, c in right[s].items(): v[K + kidx[k]] += c
    prof[s] = v / np.linalg.norm(v)
F = np.array([prof[s] for s in freq_signs])
COS = F @ F.T
fi = {s: i for i, s in enumerate(freq_signs)}
logf = np.log(np.array([tok[s] for s in freq_signs]))


def e2_test(a, b):
    """cosine of neighbour profiles; P = share of frequency-matched partners (within x2 of the sign) scoring >= observed, max over the two signs."""
    ia, ib = fi[a], fi[b]
    c = COS[ia, ib]
    ps = []
    for i, j in ((ia, ib), (ib, ia)):
        m = (np.abs(logf - logf[j]) <= np.log(2)) & (np.arange(len(freq_signs)) != i) & (np.arange(len(freq_signs)) != j)
        if m.sum() < 10:
            m = np.argsort(np.abs(logf - logf[j]))[:12]
            mm = np.zeros(len(freq_signs), bool); mm[m] = True; mm[i] = False; mm[j] = False; m = mm
        ps.append((np.sum(COS[i, m] >= c) + 1) / (m.sum() + 1))
    return float(c), float(max(ps))


# minimal pairs: texts identical except one position a<->b
texts_by_key = collections.defaultdict(set)
for r in corpus:
    q = tuple(r['seq_raw'])
    if len(q) >= 2:
        for i in range(len(q)):
            texts_by_key[(q[:i], q[i + 1:])].add(q[i])
minpair = collections.Counter()
for key, fills in texts_by_key.items():
    if len(fills) >= 2:
        for a, b in itertools.combinations(sorted(fills), 2):
            minpair[(a, b)] += 1

# adjacency veto
adjacent = collections.Counter()
for r in corpus:
    q = r['seq_raw']
    for i in range(len(q) - 1):
        a, b = sorted((q[i], q[i + 1]))
        if a != b:
            adjacent[(a, b)] += 1

# ------------------------------------------------------------------ E1: form predicted by neighbour
# token table in seq_raw: sign, prev, next, stratum
tok_rows = []
for ti, r in enumerate(corpus):
    q = r['seq_raw']
    for i, s in enumerate(q):
        tok_rows.append((s, q[i - 1] if i > 0 else BOUND, q[i + 1] if i + 1 < len(q) else BOUND, strata[ti]))
by_sign = collections.defaultdict(list)
for row in tok_rows:
    by_sign[row[0]].append(row)
strat_idx = {s: i for i, s in enumerate(sorted(set(strata)))}


def mi(lab, ctx):
    n = len(lab)
    nc = ctx.max() + 1
    tab = np.bincount(lab * nc + ctx, minlength=2 * nc).reshape(2, nc).astype(float)
    pl = tab.sum(1, keepdims=True) / n; pc = tab.sum(0, keepdims=True) / n; p = tab / n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = p * np.log(p / (pl * pc))
    return float(np.nansum(t))


def e1_test(a, b, nperm=300):
    rows = by_sign[a] + by_sign[b]
    lab = np.array([0] * len(by_sign[a]) + [1] * len(by_sign[b]))
    st = np.array([strat_idx[r[3]] for r in rows])
    out = {}
    for side, col in (('prev', 1), ('next', 2)):
        vals = [r[col] for r in rows]
        keys = {k: i for i, k in enumerate(sorted(set(vals)))}
        ctx = np.array([keys[v] for v in vals])
        obs = mi(lab, ctx)
        null = np.empty(nperm)
        perm = lab.copy()
        groups = [np.where(st == g)[0] for g in np.unique(st)]
        for p in range(nperm):
            for g in groups:
                perm[g] = rng.permutation(perm[g])
            null[p] = mi(perm, ctx)
        out[side] = (obs, float((np.sum(null >= obs) + 1) / (nperm + 1)))
    return out


# ------------------------------------------------------------------ E3: graphic
sim28 = {}
for e in edges28:
    a, b = sorted((e['base'], e['derived']))
    sim28.setdefault((a, b), set()).add(e['type'])
# calibrate the fine-similarity cut on the strong merges
strong_pairs = [(m['form'], m['into']) for m in levels if m['level'] == 'strong']
fine_strong = [G[gidx[a], gidx[b]] for a, b in strong_pairs if a in gidx and b in gidx]
FINE_CUT = float(np.percentile(fine_strong, 25))


def fine(a, b):
    if a in gidx and b in gidx:
        return float(G[gidx[a], gidx[b]])
    return float('nan')


# ------------------------------------------------------------------ E4: transcriber
WM = collections.defaultdict(collections.Counter)
for p in pos24:
    if p['cls'] in ('agree', 'sub', 'unbridged') and p['w'] and p['m']:
        WM[p['w']][p['m']] += 1
dominant = {w: c.most_common(1)[0][0] for w, c in WM.items() if sum(c.values()) >= 2}
subs = collections.Counter()
m2w = collections.defaultdict(set)
for w, ms in bridge.items():
    for m in ms: m2w[m].add(int(w))
for p in pos24:
    if p['cls'] == 'sub' and p['w'] and p['m']:
        for w2 in m2w.get(p['m'], ()):
            if w2 != p['w']:
                subs[tuple(sorted((p['w'], w2)))] += 1


def e4_test(a, b):
    reasons = []
    if a in dominant and b in dominant and dominant[a] == dominant[b]:
        # both read as the same M sign at least twice each
        m = dominant[a]
        if WM[a][m] >= 2 and WM[b][m] >= 2:
            reasons.append('same-M%d(%d,%d)' % (m, WM[a][m], WM[b][m]))
    if subs[(a, b)] >= 2:
        reasons.append('sub x%d' % subs[(a, b)])
    return reasons


# ------------------------------------------------------------------ candidate universe
cand = set()
for a, b in itertools.combinations(freq_signs, 2):
    f = fine(a, b)
    if (not np.isnan(f) and f >= 0.5) or (a, b) in sim28 or minpair[(a, b)] >= 1 or e4_test(a, b) or COS[fi[a], fi[b]] >= 0.6:
        cand.add((a, b))
# rare signs (< MIN_TOK) can only carry E3 + E4 (no usage test is possible); include them for the graphic/transcriber graph
rare_cand = set()
for (a, b) in list(sim28) + list(subs):
    if (tok[a] < MIN_TOK or tok[b] < MIN_TOK) and tok[a] > 0 and tok[b] > 0:
        rare_cand.add(tuple(sorted((a, b))))
for a in signs:
    if tok[a] < MIN_TOK and a in gidx:
        for b in signs:
            if b != a and b in gidx and G[gidx[a], gidx[b]] >= FINE_CUT:
                rare_cand.add(tuple(sorted((a, b))))
print('candidate pairs (both >= %d tokens): %d; rare-sign candidates: %d; FINE_CUT %.3f' % (MIN_TOK, len(cand), len(rare_cand), FINE_CUT), flush=True)

# ------------------------------------------------------------------ score every candidate
rows = []
for n, (a, b) in enumerate(sorted(cand)):
    ev = {}
    cos, p2 = e2_test(a, b)
    ev['E2'] = bool(p2 <= 0.05 and cos >= 0.5)
    e1 = e1_test(a, b)
    ev['E1'] = bool(min(e1['prev'][1], e1['next'][1]) <= 0.025)
    f = fine(a, b)
    g28 = sim28.get((a, b), set())
    ev['E3'] = bool((g28 & {'strokes', 'doubling'}) or (not np.isnan(f) and f >= FINE_CUT))
    e4 = e4_test(a, b)
    ev['E4'] = bool(e4)
    veto = []
    if adjacent[(a, b)] >= 2: veto.append('adjacent x%d' % adjacent[(a, b)])
    if a in NUMERAL and b in NUMERAL: veto.append('both numerals')
    rows.append({'a': a, 'b': b, 'na': tok[a], 'nb': tok[b], 'cos': round(cos, 3), 'P_E2': round(p2, 3), 'minpairs': minpair[(a, b)],
                 'MI_prev_P': round(e1['prev'][1], 3), 'MI_next_P': round(e1['next'][1], 3), 'fine': None if np.isnan(f) else round(f, 3),
                 'loop28': sorted(g28), 'E4': e4, 'adjacent': adjacent[(a, b)], 'veto': veto, 'ev': ev, 'n_ev': sum(ev.values())})
    if n % 200 == 0:
        print('  scored', n, flush=True)
for a, b in sorted(rare_cand):
    f = fine(a, b); g28 = sim28.get((a, b), set()); e4 = e4_test(a, b)
    ev = {'E1': False, 'E2': False, 'E3': bool((g28 & {'strokes', 'doubling'}) or (not np.isnan(f) and f >= FINE_CUT)), 'E4': bool(e4)}
    veto = []
    if adjacent[(a, b)] >= 2: veto.append('adjacent x%d' % adjacent[(a, b)])
    if a in NUMERAL and b in NUMERAL: veto.append('both numerals')
    rows.append({'a': a, 'b': b, 'na': tok[a], 'nb': tok[b], 'cos': None, 'P_E2': None, 'minpairs': minpair[(a, b)], 'MI_prev_P': None, 'MI_next_P': None,
                 'fine': None if np.isnan(f) else round(f, 3), 'loop28': sorted(g28), 'E4': e4, 'adjacent': adjacent[(a, b)], 'veto': veto, 'ev': ev,
                 'n_ev': sum(ev.values()), 'rare': True})

# ------------------------------------------------------------------ partitions
def canonical_map(level):
    allowed = {'raw': set(), 'strong': {'strong'}, 'all': {'strong', 'probable'}}[level]
    return {m['form']: m['into'] for m in levels if m['level'] in allowed}


class UF:
    def __init__(self): self.p = {}
    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def partition(level, k, use_rows=None, exclude_types=()):
    uf = UF()
    for s in signs: uf.f(s)
    for f_, h in canonical_map(level).items(): uf.u(f_, h)
    for r in (use_rows if use_rows is not None else rows):
        if r['veto']: continue
        ne = sum(v for t, v in r['ev'].items() if t not in exclude_types)
        if ne >= k:
            uf.u(r['a'], r['b'])
    cls = collections.defaultdict(set)
    for s in signs: cls[uf.f(s)].add(s)
    return cls


def inventory(cls):
    return len(cls)


def chao1(counts):
    c = np.array(list(counts)); S = len(c); f1 = int((c == 1).sum()); f2 = int((c == 2).sum())
    return S + (f1 * (f1 - 1) / (2 * (f2 + 1)) if f2 == 0 else f1 * f1 / (2 * f2))


def merged_counts(cls):
    return [sum(tok[s] for s in members) for members in cls.values()]


out = []
out.append('# Loop 52 cycle 1: four-evidence merge graph (seq_raw signs, %d types, %d tokens)' % (len(signs), sum(tok.values())))
out.append('candidate pairs with both signs >= %d tokens: %d; rare-sign candidates (E3/E4 only): %d; fine-similarity cut (25th pct of S268 strong merges) %.3f' % (MIN_TOK, len(cand), len(rare_cand), FINE_CUT))
usable = [r for r in rows if not r.get('rare')]
for t in ('E1', 'E2', 'E3', 'E4'):
    out.append('  %s positive: %d of %d testable pairs (%d after vetoes)' % (t, sum(r['ev'][t] for r in rows), len(rows), sum(r['ev'][t] for r in rows if not r['veto'])))
pairs_k = {k: [r for r in rows if not r['veto'] and r['n_ev'] >= k] for k in (1, 2, 3, 4)}
out.append('pairs by evidence count (after vetoes): ' + ', '.join('>=%d: %d' % (k, len(v)) for k, v in pairs_k.items()))
out.append('vetoed pairs with >= 2 evidence types: %d (adjacent %d, numerals %d)' % (sum(1 for r in rows if r['veto'] and r['n_ev'] >= 2),
           sum(1 for r in rows if any(v.startswith('adjacent') for v in r['veto']) and r['n_ev'] >= 2), sum(1 for r in rows if 'both numerals' in r['veto'] and r['n_ev'] >= 2)))
# co-occurrence of evidence types among usable pairs (independence check)
out.append('\n## agreement between evidence types (testable pairs, no veto)')
ok = [r for r in usable if not r['veto']]
for t1, t2 in itertools.combinations(('E1', 'E2', 'E3', 'E4'), 2):
    n11 = sum(r['ev'][t1] and r['ev'][t2] for r in ok); n1 = sum(r['ev'][t1] for r in ok); n2 = sum(r['ev'][t2] for r in ok)
    exp = n1 * n2 / max(1, len(ok))
    out.append('  %s & %s: %d observed vs %.1f if independent (ratio %.2f)' % (t1, t2, n11, exp, n11 / exp if exp else float('nan')))

out.append('\n## inventory ladder (types with >= 1 token; Chao1 on merged counts)')
out.append('| start | evidence k | classes | merged signs | largest class | Chao1 | f1 | f2 |')
ladder = {}
for level in ('raw', 'strong', 'all'):
    for k in (None, 3, 2, 1):
        if k is None:
            cls = partition(level, 99)
            label = 'canonical %s' % level
        else:
            cls = partition(level, k)
            label = 'canonical %s + graph >= %d' % (level, k)
        mc = merged_counts(cls)
        big = max(len(v) for v in cls.values())
        ladder[(level, k)] = {'classes': len(cls), 'chao1': chao1(mc), 'largest': big, 'f1': sum(1 for c in mc if c == 1), 'f2': sum(1 for c in mc if c == 2)}
        out.append('| %s | %s | %d | %d | %d | %.0f | %d | %d |' % (label, k if k else '-', len(cls), len(signs) - len(cls), big, chao1(mc), ladder[(level, k)]['f1'], ladder[(level, k)]['f2']))

# excluding each evidence type in turn (k >= 2) and E1 alone (context-only, the S268-rejected route)
out.append('\n## sensitivity (start all): drop one evidence type, k >= 2')
for t in ('E1', 'E2', 'E3', 'E4'):
    cls = partition('all', 2, exclude_types=(t,))
    out.append('  without %s: %d classes (Chao1 %.0f, largest %d)' % (t, len(cls), chao1(merged_counts(cls)), max(len(v) for v in cls.values())))
cls = partition('all', 1, exclude_types=('E2', 'E3', 'E4'))
out.append('  E1 alone (context-only, S268 rejected): %d classes, largest %d' % (len(cls), max(len(v) for v in cls.values())))
cls = partition('all', 1, exclude_types=('E1', 'E2', 'E4'))
out.append('  E3 alone (graphic-only): %d classes, largest %d' % (len(cls), max(len(v) for v in cls.values())))

# control: the same number of edges among random candidate pairs
out.append('\n## control: k>=2 edge count drawn at random from the candidate universe (200x), start all')
n_edges = len(pairs_k[2])
pool = [r for r in rows if not r['veto']]
ctrl = []
for _ in range(200):
    pick = random.sample(pool, n_edges)
    fake = [dict(r, ev={'E1': True, 'E2': True, 'E3': False, 'E4': False}) for r in pick]
    cls = partition('all', 2, use_rows=fake)
    ctrl.append((len(cls), max(len(v) for v in cls.values())))
ctrl = np.array(ctrl)
out.append('  real: %d classes, largest %d; random same-size edge sets: %.1f +/- %.1f classes, largest %.1f (max %d)' % (
    ladder[('all', 2)]['classes'], ladder[('all', 2)]['largest'], ctrl[:, 0].mean(), ctrl[:, 0].std(), ctrl[:, 1].mean(), ctrl[:, 1].max()))

# list classes at k>=2, start all
cls2 = partition('all', 2)
multi = sorted([sorted(v, key=lambda s: -tok[s]) for v in cls2.values() if len(v) > 1], key=lambda v: -sum(tok[s] for s in v))
out.append('\n## merged classes at k >= 2 (start all): %d classes with >= 2 members, %d signs' % (len(multi), sum(len(v) for v in multi)))
for v in multi:
    out.append('  ' + '/'.join('%d(%d)' % (s, tok[s]) for s in v))
out.append('\n## k >= 2 edges with their evidence')
for r in sorted(pairs_k[2], key=lambda r: -(r['na'] + r['nb'])):
    out.append('  %d-%d n=%d/%d ev=%s cos=%s P_E2=%s minpairs=%d MI_P=%s/%s fine=%s loop28=%s E4=%s' % (
        r['a'], r['b'], r['na'], r['nb'], ''.join(t for t in ('E1', 'E2', 'E3', 'E4') if r['ev'][t]), r['cos'], r['P_E2'], r['minpairs'], r['MI_prev_P'], r['MI_next_P'], r['fine'], r['loop28'], r['E4']))
out.append('\n## vetoed pairs with >= 2 evidence types')
for r in sorted([r for r in rows if r['veto'] and r['n_ev'] >= 2], key=lambda r: -(r['na'] + r['nb'])):
    out.append('  %d-%d n=%d/%d ev=%s veto=%s' % (r['a'], r['b'], r['na'], r['nb'], ''.join(t for t in ('E1', 'E2', 'E3', 'E4') if r['ev'][t]), r['veto']))

open(OUT + 'loop52_cycle1.txt', 'w').write('\n'.join(out) + '\n')
json.dump({'note': 'loop 52 merge graph; evidence E1 contextual allography, E2 free interchange, E3 graphic (loop28 strokes/doubling or fine sim), E4 transcriber (loop24)',
           'min_tokens': MIN_TOK, 'fine_cut': FINE_CUT, 'pairs': rows,
           'classes': {'%s_k%s' % (lv, k): [sorted(v) for v in partition(lv, k).values() if len(v) > 1] for lv in ('raw', 'strong', 'all') for k in (1, 2, 3)},
           'ladder': {'%s_k%s' % (lv, k): v for (lv, k), v in ladder.items()}},
          open(OUT + 'loop52_merges.json', 'w'), indent=0)
print('\n'.join(out[:40]))
