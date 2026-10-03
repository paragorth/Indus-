#!/usr/bin/env python3
"""Loop 40: are long texts concatenations of short attested texts?

Cycle 1  decomposition census: every complete text of >= 5 signs, all segmentations into 2-3 contiguous units
         each attested as a complete text on another object; exact, S262 variants, M-level (bridge) collapse;
         by site and object type; nulls: within-slot shuffle, within-text shuffle, S366 best model, order-2 chain;
         random-concatenation control (join two attested shorts; cut at closer boundary?).
Cycle 2  where is the cut; precedence matrix of unit classes; consistency (binomial, BH), cycles, longest chain;
         comparison with the S-DARK-19 sign-level order and the S-DARK-21 P2 reused runs.
Cycle 3  find spots: do the two units of a decomposable text occur as separate single objects at the same site /
         area / house more than chance (units permuted among decomposable texts of the same site).
Cycle 4  the 324 IM77-only new texts (M numbers, matched in M space): share of long ones decomposing into
         Wells-attested units vs shuffles; unit order against the precedence learned on Mohenjo-daro + Harappa.

usage: python3 tools/dark_loop40.py CYCLE [LEVEL] [--nperm N]
"""
import sys, os, json, csv, random, collections, math, importlib
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, 'tools')
OUT = 'data/derived/dark/'
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'seq_raw'
NPERM = int(sys.argv[sys.argv.index('--nperm') + 1]) if '--nperm' in sys.argv else 200
rnd = random.Random(40)
BIG = {'Mohenjo-daro', 'Harappa'}
C = json.load(open('data/derived/merged-corpus-canonical.json'))
BR = json.load(open('data/derived/bridge_extended.json'))
PROP = json.load(open('data/derived/dark/bridge_proposals.json'))
L27 = json.load(open('data/derived/dark/loop27_sets.json'))
LOG = open(OUT + f'loop40_cycle{CY}_{LV}.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

# ------------------------------------------------------------------ corpus
def otype(t):
    t = t.split(':')[0]
    return 'SEAL' if t == 'SEAL' else 'TAB' if t == 'TAB' else 'OTHER'
RAW2LV = collections.defaultdict(collections.Counter)
for r in C:
    for a, b in zip(r['seq_raw'], r[LV]): RAW2LV[a][b] += 1
def canon(w):
    c = RAW2LV.get(w)
    return c.most_common(1)[0][0] if c else w
OBJ = []
for r in C:
    s = r[LV]
    if not s or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
    OBJ.append(dict(id=r['cisi'], site=r['site'], ot=otype(r['type']), typ=r['type'], seq=tuple(s),
                    area=r['area-section'], block=r['block-house'], room=r['room-grid'],
                    big=r['site'] in BIG))
P(f'# LOOP 40 cycle {CY} level {LV}: {len(OBJ)} complete direction-recorded texts; '
  f'{sum(len(o["seq"]) >= 5 for o in OBJ)} of >= 5 signs')

# frame classes (S289 closer paradigm, S310/S331 parser, as loop 33)
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]; CLS = set(CL)
NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}; TREE = {390, 405, 407}
def slots(s):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s and s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK: lab[1] = 'MARKER'; i = 2
    while j - 1 > i and s[j - 1] in SUF: lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CLS: lab[j - 1] = 'CLOSER'; j -= 1
    return lab
def shuffle_slots(s, R):
    lab = slots(s); t = list(s)
    idx = [k for k, l in enumerate(lab) if l == 'NAME']
    vals = [t[k] for k in idx]; R.shuffle(vals)
    for k, v in zip(idx, vals): t[k] = v
    return tuple(t)
def shuffle_all(s, R):
    t = list(s); R.shuffle(t); return tuple(t)

# ------------------------------------------------------------------ sign equivalence modes
S262 = {405: 390, 406: 390, 407: 390, 158: 154, 156: 154, 318: 320, 525: 527, 526: 527}
W2M = {}
for w, ms in BR.items():
    if ms: W2M[int(w)] = ms[0]
for w, ms in L27['bridge'].items():
    if ms and int(w) not in W2M: W2M[int(w)] = ms[0]
for row in PROP.get('proposals', PROP.get('rows', [])) if isinstance(PROP, dict) else []:
    try:
        w, m = int(row['W']), int(row['M'])
        if w not in W2M: W2M[w] = m
    except Exception: pass
def mapper(mode):
    if mode == 'exact': return lambda s: tuple(s)
    if mode == 'var': return lambda s: tuple(S262.get(x, x) for x in s)
    if mode == 'M': return lambda s: tuple(('M', W2M[x]) if x in W2M else ('W', x) for x in s)
MODES = ['exact', 'var', 'M']

# ------------------------------------------------------------------ decomposition
def attested(objs, f):
    """complete text (mapped) -> list of objects carrying it"""
    A = collections.defaultdict(list)
    for o in objs: A[f(o['seq'])].append(o)
    return A
def decomps(t, A, minlen=2, self_id=None, kmax=3):
    """all segmentations of mapped text t into 2..kmax contiguous units each attested as a complete text
    on an object other than self_id. returns list of tuples of cut positions."""
    n = len(t); out = []
    def ok(u):
        objs = A.get(u)
        if not objs: return False
        return any(o['id'] != self_id for o in objs)
    for i in range(minlen, n - minlen + 1):
        if ok(t[:i]) and ok(t[i:]): out.append((i,))
    if kmax >= 3:
        for i in range(minlen, n - 2 * minlen + 1):
            if not ok(t[:i]): continue
            for j in range(i + minlen, n - minlen + 1):
                if ok(t[i:j]) and ok(t[j:]): out.append((i, j))
    return out
def boundary_kind(t, i):
    """what stands at cut i: 'closer' (sign before is a paradigm closer or a suffix), 'opener' (sign after is an
    opener), 'both', or 'none'"""
    a = t[i - 1]; b = t[i]
    a = a[1] if isinstance(a, tuple) else a; b = b[1] if isinstance(b, tuple) else b
    c = a in CLS or a in SUF or (isinstance(t[i - 1], tuple) and t[i - 1][0] == 'M' and a in M_CLS)
    o = b in OPEN or (isinstance(t[i], tuple) and t[i][0] == 'M' and b in M_OPEN)
    return 'both' if c and o else 'closer' if c else 'opener' if o else 'none'
M_CLS = {W2M[w] for w in CLS | SUF if w in W2M}; M_OPEN = {W2M[w] for w in OPEN if w in W2M}

def census(texts, A, f, minlen, label, selfids=None):
    """texts: list of raw tuples. returns dict with shares"""
    n = len(texts); d2 = d3 = cl = 0; cuts = []
    for k, s in enumerate(texts):
        t = f(s); sid = selfids[k] if selfids else None
        D = decomps(t, A, minlen, sid)
        if D:
            d3 += 1
            if any(len(c) == 1 for c in D): d2 += 1
            # best cut: 2-way preferred, then cut at a closer boundary, then the rarest unit most attested
            best = sorted(D, key=lambda c: (len(c), -sum(boundary_kind(t, i) != 'none' for i in c),
                                             -min(len(A[t[a:b]]) for a, b in zip((0,) + c, c + (len(t),)))))[0]
            cuts.append((s, t, best))
            if all(boundary_kind(t, i) != 'none' for i in best): cl += 1
    return dict(n=n, d2=d2, d3=d3, cl=cl, cuts=cuts, label=label)
def fmt(c):
    n = c['n'] or 1
    return f"{c['label']}: n={c['n']} 2-way {c['d2']} ({c['d2']/n:.3f}) 2-or-3-way {c['d3']} ({c['d3']/n:.3f}) " \
           f"cut at frame boundary {c['cl']}/{c['d3']}"
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
def pval(obs, null, side='hi'):
    if side == 'hi': return (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)
    return (sum(1 for x in null if x <= obs) + 1) / (len(null) + 1)

LONG = [o for o in OBJ if len(o['seq']) >= 5]

# ================================================================== cycle 1
def cycle1():
    P('\n## Cycle 1: decomposition census (units = complete texts attested on another object, any site)')
    for mode in MODES:
        f = mapper(mode); A = attested(OBJ, f)
        P(f'\n### matching mode {mode}: {len(A)} distinct complete texts as unit inventory')
        for minlen in (1, 2):
            texts = [o['seq'] for o in LONG]; ids = [o['id'] for o in LONG]
            real = census(texts, A, f, minlen, f'REAL minlen={minlen}', ids)
            P(fmt(real))
            if minlen == 2 and mode == 'exact':
                P('  by site:')
                for site, nn in collections.Counter(o['site'] for o in LONG).most_common(8):
                    sub = [o for o in LONG if o['site'] == site]
                    c = census([o['seq'] for o in sub], A, f, minlen, '    ' + site, [o['id'] for o in sub]); P(fmt(c))
                P('  by object type:')
                for ot in ('SEAL', 'TAB', 'OTHER'):
                    sub = [o for o in LONG if o['ot'] == ot]
                    c = census([o['seq'] for o in sub], A, f, minlen, '    ' + ot, [o['id'] for o in sub]); P(fmt(c))
                P('  by length:')
                for L in (5, 6, 7, 8):
                    sub = [o for o in LONG if (len(o['seq']) == L if L < 8 else len(o['seq']) >= 8)]
                    c = census([o['seq'] for o in sub], A, f, minlen, f'    len {L}{"+" if L == 8 else ""}', [o['id'] for o in sub]); P(fmt(c))
                P('  MD+H only (units from all sites):')
                sub = [o for o in LONG if o['big']]
                P(fmt(census([o['seq'] for o in sub], A, f, minlen, '    MD+H', [o['id'] for o in sub])))
                P('  held-out sites, units restricted to MD+H texts:')
                Abig = attested([o for o in OBJ if o['big']], f)
                sub = [o for o in LONG if not o['big'] and o['site'] != 'Unknown']
                P(fmt(census([o['seq'] for o in sub], Abig, f, minlen, '    held-out', [o['id'] for o in sub])))
                # units restricted to SHORT texts (<= 4 signs) only
                Ash = attested([o for o in OBJ if len(o['seq']) <= 4], f)
                P(fmt(census(texts, Ash, f, minlen, '  units restricted to texts of <= 4 signs', ids)))
            # nulls: within-slot and within-text shuffles of the long texts
            nulls = {'slot': [], 'text': []}
            R = random.Random(1 + minlen)
            for rep in range(NPERM):
                for kind, fn in (('slot', shuffle_slots), ('text', shuffle_all)):
                    sh = [fn(s, R) for s in texts]
                    c = census(sh, A, f, minlen, kind, ids); nulls[kind].append((c['d2'], c['d3'], c['cl']))
            for kind in nulls:
                d2 = [x[0] for x in nulls[kind]]; d3 = [x[1] for x in nulls[kind]]; cl = [x[2] for x in nulls[kind]]
                P(f'  null {kind}-shuffle ({NPERM}x): 2-way median {q(d2,.5)} [{q(d2,.025)}-{q(d2,.975)}] P={pval(real["d2"], d2):.4f}; '
                  f'2-or-3-way median {q(d3,.5)} [{q(d3,.025)}-{q(d3,.975)}] P={pval(real["d3"], d3):.4f}; '
                  f'ratio real/null {real["d3"]/max(1,q(d3,.5)):.2f}; cut-at-boundary share null {sum(cl)/max(1,sum(d3)):.2f} vs real {real["cl"]/max(1,real["d3"]):.2f}')
        # examples
        f = mapper(mode); A = attested(OBJ, f)
        real = census([o['seq'] for o in LONG], A, f, 2, 'ex', [o['id'] for o in LONG])
        P('  examples (text | units | copies of each unit as a complete text):')
        ex = sorted(real['cuts'], key=lambda x: -min(len(A[x[1][a:b]]) for a, b in zip((0,) + x[2], x[2] + (len(x[1]),))))
        seen = set()
        for s, t, c in ex:
            if s in seen: continue
            seen.add(s)
            units = [t[a:b] for a, b in zip((0,) + c, c + (len(t),))]
            P('    ' + '-'.join(map(str, s)) + ' = ' + ' | '.join('-'.join(str(x[1] if isinstance(x, tuple) else x) for x in u) + f'(x{len(A[u])})' for u in units)
              + '  boundary: ' + ','.join(boundary_kind(t, i) for i in c))
            if len(seen) >= 25: break
    # generated nulls (exact mode, minlen 2 and 1): S366 best model and order-2 chain
    P('\n### generated corpora as null (exact matching; units = REAL attested texts)')
    sys.argv = ['strat_adequacy.py', '--seqkey', LV, '--quick']
    SA = importlib.import_module('strat_adequacy')
    f = mapper('exact'); A = attested(OBJ, f)
    meta = [(site, cls) for site, cls, _ in SA.FIT]
    realbig = {ml: census([o['seq'] for o in LONG if o['big']], A, f, ml, 'real', [o['id'] for o in LONG if o['big']]) for ml in (1, 2)}
    nrep = max(5, NPERM // 20)
    for name, model in (('S366 best (textreuse+closerdep+type+open+site)', SA.SlotModel(SA.FIT, mech=['textreuse', 'closerdep', 'type', 'open', 'site'])),
                        ('S366 M0 slot grammar (no reuse)', SA.SlotModel(SA.FIT)),
                        ('order-2 chain, no slots', SA.PlainMarkov(SA.FIT, k=2))):
        res = {1: [], 2: []}; nl = []
        R = random.Random(366)
        for rep in range(nrep):
            G = model.generate_corpus(meta, R)
            longg = [tuple(s) for _, _, s in G if len(s) >= 5]; nl.append(len(longg))
            for ml in (1, 2):
                c = census(longg, A, f, ml, name)
                res[ml].append((c['d3'] / max(1, c['n']), c['cl'] / max(1, c['d3']), c['d2'] / max(1, c['n'])))
        for ml in (1, 2):
            rb = realbig[ml]
            P(f'  {name} ({nrep} corpora, {sum(nl)//len(nl)} long texts each) minlen={ml}: decomposable share '
              f'{sum(x[0] for x in res[ml])/nrep:.3f} [{min(x[0] for x in res[ml]):.3f}-{max(x[0] for x in res[ml]):.3f}] '
              f'(2-way {sum(x[2] for x in res[ml])/nrep:.3f}) vs REAL MD+H {rb["d3"]/rb["n"]:.3f} (2-way {rb["d2"]/rb["n"]:.3f}); '
              f'cut-at-boundary {sum(x[1] for x in res[ml])/nrep:.2f} vs real {rb["cl"]/max(1,rb["d3"]):.2f}')
    # random concatenation control
    P('\n### random-concatenation control: join two attested short texts (2-4 signs, drawn by object count)')
    shorts = [o['seq'] for o in OBJ if 2 <= len(o['seq']) <= 4]
    R = random.Random(7); nj = 5000; kinds = collections.Counter(); att = 0; forb = 0
    realcuts = census([o['seq'] for o in LONG], A, f, 2, 'r', [o['id'] for o in LONG])['cuts']
    for _ in range(nj):
        a = R.choice(shorts); b = R.choice(shorts); t = a + b
        kinds[boundary_kind(t, len(a))] += 1
        if t in A: att += 1
    P(f'  join point: {dict(kinds)} -> frame boundary share {(nj - kinds["none"])/nj:.2f}; joined text itself attested {att}/{nj} = {att/nj:.4f}')
    P(f'  real decomposable long texts: cut kinds {dict(collections.Counter(boundary_kind(t, c[0]) for s, t, c in realcuts if len(c) == 1))}')
    # the base rate: share of short attested texts ending in a closer/suffix
    endc = sum(1 for s in shorts if s[-1] in CLS or s[-1] in SUF) / len(shorts)
    P(f'  base rate: share of 2-4-sign complete texts ending in a paradigm closer or suffix = {endc:.2f}; starting with an opener {sum(1 for s in shorts if s[0] in OPEN)/len(shorts):.2f}')
    json.dump(dict(level=LV, cuts=[(list(s), [list(x) if isinstance(x, tuple) else x for x in t], list(c)) for s, t, c in realcuts]),
              open(OUT + f'loop40_c1_cuts_{LV}.json', 'w'))

# ================================================================== cycle 2: precedence
def unit_class(u):
    """functional class of a unit (W numbers): OPENER-unit, JAR, ARROW, C<x> (other paradigm closer), COUNT
    (numeral + counted good), SUFFIXED (unit ends in a suffix after a closer), PLAIN"""
    u = tuple(x[1] if isinstance(x, tuple) else x for x in u)
    core = list(u)
    suf = False
    while len(core) > 1 and core[-1] in SUF: core.pop(); suf = True
    if u[0] in OPEN and (len(u) <= 3 or (len(u) > 1 and u[1] in MARK)) and core[-1] not in CLS: return 'OPENER-unit'
    if core[-1] == 740: return 'JAR' + ('+suf' if suf else '')
    if core[-1] == 520: return 'ARROW'
    if core[-1] in CLS: return 'C' + str(core[-1])
    if len(core) >= 2 and core[-2] in NUM and core[-1] in TREE | {900, 585, 575}: return 'COUNT'
    if u[0] in OPEN: return 'OPENER-unit'
    return 'PLAIN'
def precedence(cuts, label, bigonly=None):
    """cuts: list of (s, t, c). returns pair counter {(A,B): n} for A before B over consecutive units"""
    pairs = collections.Counter(); upairs = collections.Counter(); cls_pos = collections.defaultdict(collections.Counter)
    for s, t, c in cuts:
        units = [t[a:b] for a, b in zip((0,) + c, c + (len(t),))]
        cl = [unit_class(u) for u in units]
        for k, x in enumerate(cl): cls_pos[x][k] += 1
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                pairs[(cl[i], cl[j])] += 1
                upairs[(units[i], units[j])] += 1
    CLOSED = lambda x: x.startswith('JAR') or x.startswith('ARROW') or x.startswith('C')
    two_cred = sum(1 for s, t, c in cuts if sum(CLOSED(unit_class(t[a:b])) for a, b in zip((0,) + c, c + (len(t),))) >= 2)
    op_plus = sum(1 for s, t, c in cuts if unit_class(t[:c[0]]) == 'OPENER-unit')
    P(f'\n### {label}: {len(cuts)} decomposable texts; with >= 2 closer-bearing units (true stacks) {two_cred}; '
      f'opener-phrase + one text {op_plus}; unit classes by position (0 = first):')
    for x, cnt in sorted(cls_pos.items(), key=lambda kv: -sum(kv[1].values())):
        P(f'    {x:12s} ' + ' '.join(f'pos{k}:{cnt[k]}' for k in sorted(cnt)))
    # order consistency per class pair
    P('  class precedence (A before B : B before A), one-sided binomial; BH at 0.05 over pairs with n >= 5')
    rows = []
    done = set()
    for (a, b), n in pairs.most_common():
        if (b, a) in done or (a, b) in done or a == b: continue
        done.add((a, b)); m = pairs[(b, a)]; tot = n + m
        if tot < 5: continue
        k = min(n, m); p = sum(math.comb(tot, i) for i in range(k + 1)) / 2 ** tot
        rows.append((a, b, n, m, p))
    rows.sort(key=lambda r: r[4]); nfix = 0; fixed = []
    for rank, (a, b, n, m, p) in enumerate(rows, 1):
        bh = p <= 0.05 * rank / max(1, len(rows))
        if bh: nfix += 1; fixed.append((a, b) if n >= m else (b, a))
        P(f'    {a:12s} -> {b:12s} {n:3d} : {m:3d}  P={p:.4f} {"FIXED" if bh else "free" if p > 0.3 else "?"}')
    # cycle check and longest chain over fixed edges
    nodes = set(x for e in fixed for x in e); G = collections.defaultdict(set)
    for a, b in fixed: G[a].add(b)
    def has_cycle():
        col = {}
        def dfs(u):
            col[u] = 1
            for v in G[u]:
                if col.get(v) == 1: return True
                if v not in col and dfs(v): return True
            col[u] = 2; return False
        return any(u not in col and dfs(u) for u in list(nodes))
    cyc = has_cycle()
    memo = {}
    def longest(u):
        if u in memo: return memo[u]
        memo[u] = 1 + max([longest(v) for v in G[u]] + [0]); return memo[u]
    chain = max([longest(u) for u in nodes] + [0]) if not cyc else -1
    P(f'  fixed class pairs {nfix} of {len(rows)} tested; cycle in fixed edges: {cyc}; longest chain {chain} classes')
    # same-unit-pair order at the identity level
    both = 0; cons = 0; seen = set()
    for (u, v), n in upairs.items():
        if (v, u) in seen or (u, v) in seen: continue
        seen.add((u, v)); m = upairs[(v, u)]
        if n + m >= 2: both += 1; cons += (m == 0 or n == 0)
    P(f'  identity level: unit pairs seen >= 2 times {both}, always in the same order {cons} ({cons/max(1,both):.2f})')
    return pairs, fixed

def cycle2():
    P('\n## Cycle 2: where is the cut, and which unit comes first')
    f = mapper('exact'); A = attested(OBJ, f)
    for minlen in (2, 1):
        real = census([o['seq'] for o in LONG], A, f, minlen, 'real', [o['id'] for o in LONG])
        cuts2 = [x for x in real['cuts'] if len(x[2]) == 1]
        kinds = collections.Counter(boundary_kind(t, c[0]) for s, t, c in cuts2)
        P(f'\n### minlen {minlen}: 2-way cuts {len(cuts2)}: boundary kinds {dict(kinds)} -> at a frame boundary {sum(v for k, v in kinds.items() if k != "none")/max(1,len(cuts2)):.2f}')
        # null: for the same texts, a random cut position with both sides >= minlen
        R = random.Random(2); nb = []
        for rep in range(NPERM):
            nb.append(sum(boundary_kind(t, R.randint(minlen, len(t) - minlen)) != 'none' for s, t, c in cuts2) / max(1, len(cuts2)))
        P(f'  random cut in the same texts: boundary share {q(nb,.5):.2f} [{q(nb,.025):.2f}-{q(nb,.975):.2f}]')
        # where the cut falls when it is NOT at a frame boundary
        nonb = [(s, t, c) for s, t, c in cuts2 if boundary_kind(t, c[0]) == 'none']
        P(f'  non-boundary cuts {len(nonb)}: sign before cut {collections.Counter(t[c[0]-1] for s, t, c in nonb).most_common(8)}; sign after {collections.Counter(t[c[0]] for s, t, c in nonb).most_common(8)}')
        pairs, fixed = precedence(real['cuts'], f'all sites, minlen {minlen}')
        pb, fb = precedence([x for x in real['cuts'] if x[0] in set(o['seq'] for o in LONG if o['big'])], f'MD+H only, minlen {minlen}')
        ph, fh = precedence([x for x in real['cuts'] if x[0] in set(o['seq'] for o in LONG if not o['big'])], f'held-out sites, minlen {minlen}')
        agree = sum(1 for e in fh if e in fb); contra = sum(1 for e in fh if (e[1], e[0]) in fb)
        P(f'  held-out fixed edges also fixed on MD+H: {agree}/{len(fh)}; contradicting MD+H: {contra}')
        # direction of MD+H fixed pairs on held-out texts
        ok = bad = 0
        for a, b in fb: ok += ph[(a, b)]; bad += ph[(b, a)]
        P(f'  MD+H fixed class order applied to held-out decompositions: conforming {ok}, violating {bad}')
        if minlen == 2:
            json.dump(dict(fixed_big=fb, fixed_all=fixed), open(OUT + f'loop40_c2_order_{LV}.json', 'w'))
    # comparison with S-DARK-21 P2 reuse: which units are reused most
    real = census([o['seq'] for o in LONG], A, f, 2, 'real', [o['id'] for o in LONG])
    ucount = collections.Counter(); upos = collections.defaultdict(collections.Counter)
    for s, t, c in real['cuts']:
        units = [t[a:b] for a, b in zip((0,) + c, c + (len(t),))]
        for k, u in enumerate(units): ucount[u] += 1; upos[u]['first' if k == 0 else 'last' if k == len(units) - 1 else 'mid'] += 1
    P('\n### most used units in decompositions (unit, class, uses, position, copies as a complete text):')
    for u, n in ucount.most_common(25):
        P(f'    {"-".join(map(str,u)):18s} {unit_class(u):12s} uses {n:3d} {dict(upos[u])} copies {len(A[u])}')

# ================================================================== cycle 3: find spots
def cycle3():
    P('\n## Cycle 3: do the two units of a decomposable text occur as separate objects at the same place?')
    f = mapper('exact'); A = attested(OBJ, f)
    for minlen in (2, 1):
        real = census([o['seq'] for o in LONG], A, f, minlen, 'real', [o['id'] for o in LONG])
        byseq = collections.defaultdict(list)
        for o in LONG: byseq[o['seq']].append(o)
        # one record per host object: (host, unit1, unit2) for 2-way cuts
        recs = []
        for s, t, c in real['cuts']:
            if len(c) != 1: continue
            u1, u2 = t[:c[0]], t[c[0]:]
            for host in byseq[s]: recs.append((host, u1, u2))
        P(f'\n### minlen {minlen}: {len(recs)} host objects with a 2-way decomposition')
        def place_stats(recs, level):
            """share of hosts whose unit1 AND unit2 both occur as separate objects at the host's level-place
            (site / area / block); the pair statistic: share of (u1 object, u2 object) pairs sharing the place"""
            both = 0; n = 0; pairs = 0; same = 0
            for host, u1, u2 in recs:
                hp = host[level]
                if level != 'site' and hp in ('--', '-', ''): continue
                o1 = [o for o in A[u1] if o['id'] != host['id'] and (o['site'] == host['site'])]
                o2 = [o for o in A[u2] if o['id'] != host['id'] and (o['site'] == host['site'])]
                n += 1
                if any(o[level] == hp for o in o1) and any(o[level] == hp for o in o2): both += 1
                for a in o1:
                    for b in o2:
                        if level != 'site' and (a[level] in ('--', '-', '') or b[level] in ('--', '-', '')): continue
                        pairs += 1; same += a[level] == b[level]
            return n, both, pairs, same
        for level in ('site', 'area', 'block'):
            n, both, pairs, same = place_stats(recs, level)
            # null: permute unit2 among hosts of the same site (keeps unit1 and host place)
            R = random.Random(3); nb = []; ns = []
            for rep in range(NPERM):
                bysite = collections.defaultdict(list)
                for host, u1, u2 in recs: bysite[host['site']].append(u2)
                for k in bysite: R.shuffle(bysite[k])
                it = {k: iter(v) for k, v in bysite.items()}
                prec = [(host, u1, next(it[host['site']])) for host, u1, u2 in recs]
                n_, b_, p_, s_ = place_stats(prec, level); nb.append(b_ / max(1, n_)); ns.append(s_ / max(1, p_))
            P(f'  {level:5s}: hosts with both units present as separate objects at the same {level} {both}/{n} = {both/max(1,n):.3f} '
              f'vs permuted-unit2 {q(nb,.5):.3f} [{q(nb,.025):.3f}-{q(nb,.975):.3f}] P={pval(both/max(1,n), nb):.3f}; '
              f'(u1-object, u2-object) pairs sharing the {level}: {same}/{pairs} = {same/max(1,pairs):.3f} vs {q(ns,.5):.3f} [{q(ns,.025):.3f}-{q(ns,.975):.3f}] P={pval(same/max(1,pairs), ns):.3f}')
        for site in ('Mohenjo-daro', 'Harappa'):
            sub = [r for r in recs if r[0]['site'] == site]
            for level in ('area', 'block'):
                n, both, pairs, same = place_stats(sub, level)
                R = random.Random(4); nb = []; ns = []
                for rep in range(NPERM):
                    u2s = [r[2] for r in sub]; R.shuffle(u2s)
                    n_, b_, p_, s_ = place_stats([(h, u1, u2) for (h, u1, _), u2 in zip(sub, u2s)], level)
                    nb.append(b_ / max(1, n_)); ns.append(s_ / max(1, p_))
                P(f'  {site} {level}: both at host place {both}/{n} = {both/max(1,n):.3f} vs {q(nb,.5):.3f} [{q(nb,.025):.3f}-{q(nb,.975):.3f}] P={pval(both/max(1,n), nb):.3f}; '
                  f'pairs same {level} {same}/{pairs} = {same/max(1,pairs):.3f} vs {q(ns,.5):.3f} [{q(ns,.025):.3f}-{q(ns,.975):.3f}] P={pval(same/max(1,pairs), ns):.3f}')
        # object type of host vs units
        ot = collections.Counter()
        for host, u1, u2 in recs:
            t1 = collections.Counter(o['ot'] for o in A[u1]).most_common(1)[0][0]; t2 = collections.Counter(o['ot'] for o in A[u2]).most_common(1)[0][0]
            ot[(host['ot'], t1, t2)] += 1
        P(f'  host type x dominant type of unit1 x unit2: {ot.most_common(10)}')

# ================================================================== cycle 4: IM77-only new texts
def load_im77_new():
    rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    want = set((a, b) for a, b in L27['new'])
    by = collections.defaultdict(list)
    for r in rows:
        k = (r['text_no'], r['side'])
        if k in want and r['signs_clean'].strip():
            by[k].append((int(r['line']), [int(x) for x in r['signs_clean'].split()], r['object_type'], r['site']))
    out = []
    for k, lines in by.items():
        lines.sort(); s = tuple(('M', x) for ln in lines for x in ln[1])
        ot = {'seal': 'SEAL', 'miniature tablet': 'TAB', 'copper tablet': 'TAB'}.get(lines[0][2], 'OTHER')
        out.append(dict(id='IM' + k[0] + '/' + k[1], site=lines[0][3], ot=ot, seq=s, typ=lines[0][2]))
    return out
def cycle4():
    P('\n## Cycle 4: the 324 IM77-only texts (M space): do their long texts decompose into Wells-attested units?')
    fM = mapper('M'); A = attested(OBJ, fM)            # Wells texts mapped to M space (unbridged W stay W-tagged)
    new = load_im77_new(); longn = [o for o in new if len(o['seq']) >= 5]
    P(f'  new texts {len(new)}, >= 5 signs {len(longn)}: sites {collections.Counter(o["site"] for o in longn).most_common(6)}, types {collections.Counter(o["ot"] for o in longn)}')
    ident = lambda s: tuple(s)
    # also allow the IM77 overlap+new texts themselves as units? no: units must be attested complete texts in the Wells corpus
    for minlen in (2, 1):
        texts = [o['seq'] for o in longn]
        real = census(texts, A, ident, minlen, f'IM77-new minlen={minlen}')
        P(fmt(real))
        R = random.Random(5); nulls = {'slot': [], 'text': []}
        for rep in range(NPERM):
            for kind in nulls:
                if kind == 'slot':
                    sh = []
                    for s in texts:
                        # slot shuffle in M space: fix first sign if opener-class, last if closer/suffix-class
                        t = list(s); i = 0; j = len(t)
                        if t[0][1] in M_OPEN: i = 1
                        if j - 1 > i and t[j - 1][1] in M_CLS: j -= 1
                        if j - 1 > i and t[j - 1][1] in M_CLS: j -= 1
                        mid = t[i:j]; R.shuffle(mid); sh.append(tuple(t[:i] + mid + t[j:]))
                else:
                    sh = [shuffle_all(s, R) for s in texts]
                c = census(sh, A, ident, minlen, kind); nulls[kind].append(c['d3'])
        for kind in nulls:
            d = nulls[kind]
            P(f'  null {kind}-shuffle: {q(d,.5)} [{q(d,.025)}-{q(d,.975)}] of {len(texts)}; P={pval(real["d3"], d):.4f}; ratio {real["d3"]/max(0.5,q(d,.5)):.1f}x')
        # length-matched comparison with Wells MD+H long texts, units from the same inventory
        realW = census([o['seq'] for o in LONG if o['big']], fM, fM, minlen, 'Wells MD+H long (M space)', [o['id'] for o in LONG if o['big']]) if False else None
        # (units in M space for Wells texts: use A directly)
        wl = [o for o in LONG if o['big']]
        cw = census([o['seq'] for o in wl], A, fM, minlen, 'Wells MD+H long texts, same inventory', [o['id'] for o in wl]); P(fmt(cw))
        # examples and order check vs MD+H precedence
        order = json.load(open(OUT + f'loop40_c2_order_{LV}.json'))['fixed_big'] if os.path.exists(OUT + f'loop40_c2_order_{LV}.json') else []
        fixed = set(tuple(e) for e in order)
        ok = bad = 0; kinds = collections.Counter()
        P('  decompositions (M numbers; units with copies in Wells):')
        def W_of_unit(u):  # for unit_class: convert M-tagged unit back to a W representative via the inventory
            objs = A.get(u)
            return objs[0]['seq'] if objs else tuple(x[1] for x in u)
        for s, t, c in real['cuts']:
            units = [t[a:b] for a, b in zip((0,) + c, c + (len(t),))]
            cl = [unit_class(W_of_unit(u)) for u in units]
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    if (cl[i], cl[j]) in fixed: ok += 1
                    elif (cl[j], cl[i]) in fixed: bad += 1
            for i in c: kinds[boundary_kind(t, i)] += 1
            P('    ' + '-'.join(str(x[1]) for x in s) + ' = ' + ' | '.join('-'.join(str(x[1]) for x in u) + f'(x{len(A[u])},{k})' for u, k in zip(units, cl)))
        P(f'  cut kinds {dict(kinds)}; unit-class pairs conforming to the MD+H fixed order {ok}, violating {bad} (fixed edges: {sorted(fixed)})')
        # by site / type
        for site, nn in collections.Counter(o['site'] for o in longn).most_common(5):
            sub = [o['seq'] for o in longn if o['site'] == site]
            c = census(sub, A, ident, minlen, f'    {site}'); P(fmt(c))
        for ot in ('SEAL', 'TAB', 'OTHER'):
            sub = [o['seq'] for o in longn if o['ot'] == ot]
            if sub: c = census(sub, A, ident, minlen, f'    {ot}'); P(fmt(c))

if __name__ == '__main__':
    {1: cycle1, 2: cycle2, 3: cycle3, 4: cycle4}[CY]()
    P('\n# done')
