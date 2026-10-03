"""S-DARK-64: DO CO-DEPOSITED SEALS SHARE NAME ELEMENTS?
If the middle is a personal designation built from elements, households / lineages should share ELEMENTS (patronymic or
clan elements) even when whole middles differ (S-DARK-29 tested whole middles: null). Element = NON-ADJACENT middle sign
(S-DARK-26 / S-DARK-45 definition: NAME-labelled tokens of the S310/S331 parser, excluding openers, markers, numerals,
titles, closers, suffixes and the sign directly before the closer = S303 adjacent qualifier).
  cycle 1: same-room pairs (Mohenjo-daro, Harappa; seals and tablets separately): number of shared element types per pair,
           share of pairs with >= 1 shared element, shared RARE (< 5 texts) / MID (5-19) / FREQUENT (>= 20) elements and a
           rarity weight sum(-log2 p_e); vs (a) random same-site pairs matched on object type x length, (b) same-area
           different-room pairs, (c) room labels permuted within site x type (and x length), NP x; identical texts excluded;
           by element and by room.
  cycle 2: element 'surnames': elements with >= 3 texts found in >= 3 rooms of ONE area with >= 60% of their tokens there,
           vs room permutation; element x area and element x room MI excess and loyalty (S-DARK-45 statistics at the
           area grain); position fixedness (edge share) of local elements; Ur III legends by site as the comparator
           (loop48 corpora; Linear B has no find spots in DAMOS).
  cycle 3: emblem: within a room, do pairs sharing an element share the emblem beyond emblem permutation (S-DARK-34
           independence baseline); site-wide: pairs sharing a RARE element vs matched null on emblem and material.
  cycle 4: replication: Chanhu-daro and Kalibangan (room), Lothal / Dholavira (area), IM77 locus field (M space, loop45
           parser), share of the evidence resting on Mohenjo-daro.
Usage: python3 tools/dark_loop64.py <cycle 1|2|3|4> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json, sys, random, collections, math, csv, re, itertools, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
CY = int(sys.argv[1]); LV = sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rnd = random.Random(64)
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)
def bad(v): return v is None or v.strip() in ('-', '--', '- -', '')
def oclass(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'TAG': 'tablet'}.get(t, 'other')
def emb(s):
    s = (s or '').strip()
    if s in ('', '-', 'None'): return None
    return s.split(':')[0]
OBJ = []
for r in C:
    s = r[LV]
    if not s or len(s) < 2 or r['complete'] != 'Y': continue
    oc = oclass(r['type'])
    if oc == 'other': continue
    area = None if bad(r['area-section']) else r['area-section'].strip()
    room = None
    if not bad(r['room-grid']): room = (area or '', (r['block-house'] or '').strip(), r['room-grid'].strip())
    OBJ.append(dict(cisi=r['cisi'], site=r['site'], oc=oc, typ=r['type'], seq=tuple(s), area=area, room=room,
                    mat=None if bad(r['material']) else r['material'].strip().capitalize(), emb=emb(r['symbol'])))
# ---------------- frame parser (S310 parse_all.py + S331 openers, as in dark_loop29/45) ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
left = collections.defaultdict(collections.Counter)
for o in OBJ:
    s = list(o['seq'])
    while len(s) > 1 and s[-1] in SUF: s.pop()
    if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
QUAL = {}
for c, cnt in left.items():
    tot = sum(cnt.values()); acc = 0; q = set()
    for a, n in cnt.most_common():
        if acc / tot >= 0.6: break
        q.add(a); acc += n
    QUAL[c] = q
def parse(s):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CL:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab
def elements(seq, lab):
    """non-adjacent middle elements: NAME tokens, minus the sign directly before the closer (S303 adjacent qualifier).
    returns list in order (positions kept)"""
    cl = [i for i, l in enumerate(lab) if l == 'CLOSER']
    adj = None
    if cl:
        ci = cl[0]
        if ci - 1 >= 0 and lab[ci - 1] in ('NAME', 'COUNT', 'TITLE'): adj = ci - 1
    return [seq[i] for i, l in enumerate(lab) if l == 'NAME' and i != adj and seq[i] not in FRAME]
FRAME = OPEN | MARK | MJAR | SUF | set(CL) | NUM   # frame-class signs are never elements, wherever they stand
for o in OBJ:
    lab = parse(o['seq']); o['lab'] = lab
    o['ell'] = elements(o['seq'], lab)
    o['el'] = frozenset(o['ell'])
    o['L'] = len(o['seq'])
    o['head'] = next((s for s, l in zip(o['seq'], lab) if l == 'CLOSER'), None)
def fmt(seq): return '-'.join(str(a) for a in seq)
def mean(v): return sum(v) / len(v) if v else float('nan')
def pval(obs, null, hi=True):
    v = [x for x in null if x == x]
    if not v or obs != obs: return float('nan')
    return (sum(1 for x in v if (x >= obs if hi else x <= obs)) + 1) / (len(v) + 1)
def ub(k, n): return f'{k}/{n}' if k > 0 else (f'0/{n} (< {3 / n:.3f})' if n else '0/0')
P(f'== S-DARK-64 cycle {CY} level {LV} nperm {NP}; objects {len(OBJ)} (complete, >= 2 signs, seals + tablets/sealings); with room {sum(1 for o in OBJ if o["room"])}, with >= 1 element {sum(1 for o in OBJ if o["el"])}')

# ---------------- element frequency classes per pool (distinct texts) ----------------
def freq_table(pool):
    seen = set(); f = collections.Counter()
    for o in pool:
        k = (o['site'], o['oc'], o['seq'])
        if k in seen: continue
        seen.add(k)
        for e in o['el']: f[e] += 1
    return f
def fclass(f, e):
    n = f.get(e, 0)
    return 'rare' if n < 5 else 'mid' if n < 20 else 'freq'
CLASSES = ('rare', 'mid', 'freq')
def pstat(a, b, F, ntexts):
    sh = a['el'] & b['el']
    d = dict(n=len(sh), any=int(bool(sh)), w=sum(-math.log2(max(F.get(e, 1), 1) / ntexts) for e in sh),
             ident=int(a['seq'] == b['seq']), midsame=int(a['el'] == b['el'] and bool(a['el'])))
    for c in CLASSES: d[c] = sum(1 for e in sh if fclass(F, e) == c)
    d['emb'] = float(a['emb'] == b['emb']) if (a['emb'] and b['emb']) else float('nan')
    d['mat'] = float(a['mat'] == b['mat']) if (a['mat'] and b['mat']) else float('nan')
    return d, sh
KEYS = ['n', 'any', 'w', 'rare', 'mid', 'freq', 'midsame']
def agg(pairs, F, ntexts, skip_ident=True):
    acc = collections.defaultdict(float); k = 0; byel = collections.Counter(); emb = []
    for a, b in pairs:
        if skip_ident and a['seq'] == b['seq']: continue
        if a['cisi'] == b['cisi'] and a['cisi'] != '-': continue
        d, sh = pstat(a, b, F, ntexts); k += 1
        for key in KEYS: acc[key] += d[key]
        for e in sh: byel[e] += 1
        if d['emb'] == d['emb']: emb.append(d['emb'])
    return {key: (acc[key] / k if k else float('nan')) for key in KEYS}, k, byel, mean(emb)
def group_pairs(pool, key):
    G = collections.defaultdict(list)
    for o in pool:
        if o[key]: G[o[key]].append(o)
    return [(a, b) for g in G.values() for a, b in itertools.combinations(g, 2)], G
def permuted(pool, key, strat):
    S = collections.defaultdict(list)
    for o in pool: S[strat(o)].append(o)
    lab = {}
    for k, os_ in S.items():
        labs = [o[key] for o in os_]; rnd.shuffle(labs)
        for o, l in zip(os_, labs): lab[id(o)] = l
    G = collections.defaultdict(list)
    for o in pool:
        if lab[id(o)]: G[lab[id(o)]].append(o)
    return [(a, b) for g in G.values() for a, b in itertools.combinations(g, 2)]
def cap(pairs, n=6000): return rnd.sample(pairs, n) if len(pairs) > n else pairs
def report(label, obs, k, nulls):
    P(f'  {label}: {k} pairs')
    for key in KEYS:
        line = f'    {key:8s} obs {obs[key]:.4f}'
        if key != 'w': line += f' ({ub(round(obs[key] * k), k)})'
        for nm, nl in nulls.items():
            m = mean(nl[key]); line += f' | {nm} {m:.4f} x{(obs[key] / m if m > 0 else float("nan")):.2f} P={pval(obs[key], nl[key]):.3f}'
        P(line)

# ======================================================================= cycle 1
def cycle1(site, oc, show_el=True):
    pool = [o for o in OBJ if o['site'] == site and o['oc'] == oc and o['room']]
    allsite = [o for o in OBJ if o['site'] == site and o['oc'] == oc]
    F = freq_table(allsite); ntexts = len({(o['seq']) for o in allsite})
    pairs, G = group_pairs(pool, 'room')
    rooms = [g for g in G.values() if len(g) >= 2]
    if len(pairs) < 3:
        P(f'\n--- {site} {oc}s by room: {len(pool)} objects, {len(pairs)} same-room pairs: too few'); return None
    obs, k, byel, embobs = agg(pairs, F, ntexts)
    P(f'\n--- {site} {oc}s by room: {len(pool)} objects with a room ({len(allsite)} at the site), {len(rooms)} rooms with >= 2, {len(pairs)} same-room pairs, {k} after dropping identical texts; element classes at the site: rare < 5 texts, mid 5-19, freq >= 20 ({sum(1 for e in F if fclass(F, e) == "rare")}/{sum(1 for e in F if fclass(F, e) == "mid")}/{sum(1 for e in F if fclass(F, e) == "freq")} elements)')
    nulls = {}
    # (a) random same-site pairs matched on type x length (pairwise redraw for each observed pair)
    S = collections.defaultdict(list)
    for o in allsite: S[(o['typ'], min(o['L'], 7))].append(o)
    na = collections.defaultdict(list)
    for _ in range(NP):
        rp = []
        for a, b in pairs:
            if a['seq'] == b['seq']: continue
            A = S[(a['typ'], min(a['L'], 7))]; B = S[(b['typ'], min(b['L'], 7))]
            for _t in range(30):
                x = rnd.choice(A); y = rnd.choice(B)
                if x is not y and x['seq'] != y['seq']: break
            else: continue
            rp.append((x, y))
        st, _, _, _ = agg(rp, F, ntexts)
        for key in KEYS: na[key].append(st[key])
    nulls['a:matched'] = na
    # (b) same-area, different-room pairs
    GA = collections.defaultdict(list)
    for o in pool:
        if o['area']: GA[o['area']].append(o)
    bp = [(a, b) for g in GA.values() for a, b in itertools.combinations(g, 2) if a['room'] != b['room']]
    bp = cap(bp)
    stb, kb, _, _ = agg(bp, F, ntexts)
    nulls['b:area-diff-room'] = {key: [stb[key]] for key in KEYS}
    # (c) room labels permuted within type (and type x length)
    nc = collections.defaultdict(list); ncL = collections.defaultdict(list)
    byel_null = collections.Counter(); byroom_null = collections.defaultdict(float)
    for _ in range(NP):
        pp = permuted(pool, 'room', lambda o: o['typ'])
        st, _, be, _ = agg(pp, F, ntexts)
        for key in KEYS: nc[key].append(st[key])
        byel_null.update(be)
        pp = permuted(pool, 'room', lambda o: (o['typ'], min(o['L'], 7)))
        st, _, _, _ = agg(pp, F, ntexts)
        for key in KEYS: ncL[key].append(st[key])
    nulls['c:room-perm(type)'] = nc; nulls['c:room-perm(type x L)'] = ncL
    report('same-room pairs (identical texts dropped)', obs, k, nulls)
    P(f'    (b) same-area different-room pairs used: {kb}')
    # with identical texts kept, for reference
    obs2, k2, _, _ = agg(pairs, F, ntexts, skip_ident=False)
    P(f'    identical texts kept: {k2} pairs, shared n {obs2["n"]:.3f}, any {obs2["any"]:.3f}, rare {obs2["rare"]:.3f}')
    # excess by class: pairs sharing a class element
    P('    excess shared-element tokens by frequency class (obs count - null(c) mean count over all same-room pairs):')
    for c in CLASSES:
        oc_ = obs[c] * k; nm = mean(nc[c]) * k
        P(f'      {c:5s} obs {oc_:.1f} null {nm:.2f} excess {oc_ - nm:+.2f} P={pval(obs[c], nc[c]):.3f}')
    if show_el:
        rows = []
        for e, n in byel.items():
            exp = byel_null[e] / NP
            rows.append((n - exp, e, n, exp))
        rows.sort(reverse=True)
        P('    elements carrying same-room sharing (obs pairs, null(c) expected, site text freq):')
        for ex, e, n, exp in rows[:12]:
            P(f'      W{e:<4d} obs {n:2d} exp {exp:5.2f} excess {ex:+.2f}  freq {F.get(e, 0)} ({fclass(F, e)})')
        # rooms
        P('    rooms with most shared-element pairs (room, n objects, shared-element pairs, texts):')
        rr = []
        for key, g in G.items():
            if len(g) < 2: continue
            s = 0; ex = []
            for a, b in itertools.combinations(g, 2):
                if a['seq'] == b['seq']: continue
                sh = a['el'] & b['el']
                if sh: s += 1; ex.append((fmt(a['seq']), fmt(b['seq']), sorted(sh)))
            rr.append((s, len(g), key, ex))
        rr.sort(reverse=True)
        for s, n, key, ex in rr[:6]:
            if s == 0: break
            P(f'      {key} n={n} sharing pairs={s}: ' + '; '.join(f'{x} ~ {y} share {z}' for x, y, z in ex[:3]))
    return obs, k, nulls

if CY == 1:
    for site in ('Mohenjo-daro', 'Harappa'):
        for oc in ('seal', 'tablet'):
            cycle1(site, oc)

# ======================================================================= cycle 2
def mi_loyal(T, lab_of, mintok=5, cut=0.8):
    """T: list of (element list, label). returns MI/H(label) %, loyalty share, n elements, loyal-to-nondominant count"""
    joint = collections.Counter(); ne = collections.Counter(); nl = collections.Counter(); n = 0
    for els, l in T:
        for e in els: joint[(e, l)] += 1; ne[e] += 1; nl[l] += 1; n += 1
    if n == 0: return float('nan'), float('nan'), 0, 0
    mi = sum(c / n * math.log2(c / n / (ne[e] / n * nl[l] / n)) for (e, l), c in joint.items())
    H = -sum(c / n * math.log2(c / n) for c in nl.values())
    dom = nl.most_common(1)[0][0]
    ok = [e for e, c in ne.items() if c >= mintok]
    loy = 0; minor = 0
    for e in ok:
        best = max(((joint[(e, l)], l) for l in nl), key=lambda x: x[0])
        if best[0] / ne[e] >= cut:
            loy += 1
            if best[1] != dom: minor += 1
    return 100 * mi / H if H > 0 else float('nan'), (loy / len(ok) if ok else float('nan')), len(ok), minor
def locality_census(pool, F, label, show=True):
    """elements with >= 3 texts, in >= 3 distinct rooms of one area, >= 60% of their tokens in that area"""
    byel = collections.defaultdict(list)
    for o in pool:
        for e in o['el']: byel[e].append(o)
    found = []
    for e, os_ in byel.items():
        if len(os_) < 3: continue
        A = collections.defaultdict(set); cntA = collections.Counter()
        for o in os_:
            if o['area']: A[o['area']].add(o['room']); cntA[o['area']] += 1
        for a, rooms in A.items():
            if len(rooms) >= 3 and cntA[a] / len(os_) >= 0.6: found.append((e, a, len(rooms), cntA[a], len(os_)))
    return found
def edge_share(pool, els):
    """share of tokens of the given elements that stand at the first or last position of the element list (length >= 2)"""
    k = n = 0
    for o in pool:
        L = o['ell']
        if len(L) < 2: continue
        for i, e in enumerate(L):
            if e in els: n += 1; k += int(i == 0 or i == len(L) - 1)
    return k / n if n else float('nan'), n
def cycle2(site, oc):
    pool = [o for o in OBJ if o['site'] == site and o['oc'] == oc and o['room'] and o['area']]
    if len(pool) < 20: return
    allsite = [o for o in OBJ if o['site'] == site and o['oc'] == oc]
    F = freq_table(allsite)
    # collapse identical texts within a room (stock copies) to one
    seen = set(); P2 = []
    for o in pool:
        k = (o['room'], o['seq'])
        if k in seen: continue
        seen.add(k); P2.append(o)
    pool = P2
    nA = len({o['area'] for o in pool}); nR = len({o['room'] for o in pool})
    P(f'\n--- {site} {oc}s with area + room: {len(pool)} (one per text per room), {nA} areas, {nR} rooms')
    obsA = mi_loyal([(o['el'], o['area']) for o in pool], None)
    obsR = mi_loyal([(o['el'], o['room']) for o in pool], None, mintok=3)
    found = locality_census(pool, F, site)
    nullA = []; nullR = []; nullF = []; nullLoyA = []; nullMinA = []
    for _ in range(NP):
        S = collections.defaultdict(list)
        for o in pool: S[(o['typ'], min(o['L'], 7))].append(o)
        lab = {}
        for k, os_ in S.items():
            labs = [(o['area'], o['room']) for o in os_]; rnd.shuffle(labs)
            for o, l in zip(os_, labs): lab[id(o)] = l
        PP = [dict(o, area=lab[id(o)][0], room=lab[id(o)][1]) for o in pool]
        a = mi_loyal([(o['el'], o['area']) for o in PP], None); r = mi_loyal([(o['el'], o['room']) for o in PP], None, mintok=3)
        nullA.append(a[0]); nullLoyA.append(a[1]); nullMinA.append(a[3]); nullR.append(r[0])
        nullF.append(len(locality_census(PP, F, site, show=False)))
    P(f'  element x AREA MI {obsA[0]:.1f}% of H(area) vs permuted (type x L) {mean(nullA):.1f}% (excess {obsA[0] - mean(nullA):+.1f}, P={pval(obsA[0], nullA):.3f}); loyalty (>= 5 tokens, >= 80% one area) {obsA[1]:.3f} vs {mean(nullLoyA):.3f} (P={pval(obsA[1], nullLoyA):.3f}), n elements {obsA[2]}, loyal to a NON-dominant area {obsA[3]} vs {mean(nullMinA):.2f}')
    P(f'  element x ROOM MI {obsR[0]:.1f}% of H(room) vs permuted {mean(nullR):.1f}% (excess {obsR[0] - mean(nullR):+.1f}, P={pval(obsR[0], nullR):.3f})')
    P(f'  local lineage candidates (>= 3 texts, >= 3 rooms of one area, >= 60% of tokens there): {len(found)} vs permuted {mean(nullF):.2f} (P={pval(len(found), nullF):.3f})')
    for e, a, nr, na, n in sorted(found, key=lambda x: -x[3])[:15]:
        es, en = edge_share(pool, {e})
        P(f'    W{e:<4d} area {a}: {nr} rooms, {na}/{n} tokens ({100 * na / n:.0f}%), site freq {F.get(e, 0)} ({fclass(F, e)}); edge share {es:.2f} (n={en})')
    allE, allN = edge_share(pool, set(e for o in pool for e in o['el']))
    locE, locN = edge_share(pool, {e for e, *_ in found})
    P(f'  edge share (first or last of the element list, lists >= 2): local candidates {locE:.3f} (n={locN}) vs all elements {allE:.3f} (n={allN})')
    # shared-area rate for pairs sharing a rare element vs any pair (direct 'area' version of cycle 1)
    pairs_rare = [(a, b) for a, b in itertools.combinations(pool, 2) if a['seq'] != b['seq'] and any(fclass(F, e) == 'rare' for e in a['el'] & b['el'])]
    pairs_any = [(a, b) for a, b in itertools.combinations(pool, 2) if a['seq'] != b['seq'] and (a['el'] & b['el'])]
    base = [(a, b) for a, b in itertools.combinations(pool, 2) if a['seq'] != b['seq']]
    for nm, pr in (('rare element', pairs_rare), ('any element', pairs_any)):
        if not pr: continue
        sa = mean([a['area'] == b['area'] for a, b in pr]); sr = mean([a['room'] == b['room'] for a, b in pr])
        ba = mean([a['area'] == b['area'] for a, b in base]); br = mean([a['room'] == b['room'] for a, b in base])
        P(f'  pairs sharing a {nm}: {len(pr)}; same area {sa:.3f} vs all pairs {ba:.3f} (x{sa / ba:.2f}); same room {ub(sum(a["room"] == b["room"] for a, b in pr), len(pr))} vs {br:.4f}')
def ur3_control():
    TITLES = {'dub-sar', 'arad2', 'arad2-zu', 'arad', 'ensi2', 'ugula', 'nu-banda3', 'gudu4', 'dam-gar3', 'lunga', 'sipa', 'szabra',
              'dumu', 'lugal', 'kiszib3', 'sukkal', 'ra2-gaba', 'szu-i', 'szagina', 'nu-banda3-gu4', 'ka-guru7', 'sanga', 'szandana',
              'i3-du8', 'muhaldim', 'szesz', 'dam', 'mu-ni', 'ki-ag2', 'ir11-zu', 'ir11', 'nin', 'szar2-ra-ab-du', 'sa12-du5'}
    T = [json.loads(l) for l in open(ROOT + 'data/derived/dark/loop48_corpora/ur3_words.jsonl')]
    T = [t for t in T if t['site'] not in ('uncertain',)]
    # one per distinct legend per site (stock copies collapsed)
    seen = set(); U = []
    for t in T:
        k = (t['site'], tuple(t['seq']))
        if k in seen: continue
        seen.add(k)
        els = [w for w in t['seq'] if w not in TITLES]
        U.append(dict(el=frozenset(els), ell=els, area=t['site'], L=len(t['seq']), typ='legend'))
    sites = collections.Counter(u['area'] for u in U)
    P(f'\n--- Ur III seal legends by SITE (loop48 ur3_words, one per distinct legend per site): {len(U)} legends, sites {dict(sites.most_common(6))}; elements = name words (titles removed)')
    obs = mi_loyal([(u['el'], u['area']) for u in U], None)
    null = []; nloy = []; nmin = []
    for _ in range(min(NP, 200)):
        labs = [u['area'] for u in U]; rnd.shuffle(labs)
        a = mi_loyal([(u['el'], l) for u, l in zip(U, labs)], None)
        null.append(a[0]); nloy.append(a[1]); nmin.append(a[3])
    P(f'  name-word x SITE MI {obs[0]:.1f}% of H(site) vs permuted {mean(null):.1f}% (excess {obs[0] - mean(null):+.1f}, P={pval(obs[0], null):.3f}); loyalty {obs[1]:.3f} vs {mean(nloy):.3f}; n elements {obs[2]}; loyal to a NON-dominant site {obs[3]} vs {mean(nmin):.1f}')
    # loyal non-dominant examples and their edge share
    joint = collections.Counter(); ne = collections.Counter()
    for u in U:
        for e in u['el']: joint[(e, u['area'])] += 1; ne[e] += 1
    dom = sites.most_common(1)[0][0]
    loc = []
    for e, n in ne.items():
        if n < 5: continue
        best = max(((joint[(e, s)], s) for s in sites), key=lambda x: x[0])
        if best[0] / n >= 0.8 and best[1] != dom: loc.append((n, e, best[1]))
    loc.sort(reverse=True)
    P('  examples of site-loyal name words (non-dominant site): ' + '; '.join(f'{e} ({s}, {n})' for n, e, s in loc[:12]))
    locE, locN = edge_share(U, {e for _, e, _ in loc}); allE, allN = edge_share(U, set(ne))
    P(f'  edge share of site-loyal words {locE:.3f} (n={locN}) vs all name words {allE:.3f} (n={allN}) (Ur III legends put the owner name first, so edge share is high for everything)')
    # pair-level: do same-site legend pairs share a name word (titles removed) more than permuted-site pairs? (sampled)
    def same_site_share(U, labs, nsamp=20000):
        G = collections.defaultdict(list)
        for u, l in zip(U, labs): G[l].append(u)
        pr = [(a, b) for g in G.values() for a, b in itertools.combinations(g, 2)] if sum(len(g) ** 2 for g in G.values()) < 2e6 else None
        if pr is None:
            pr = []
            for g in G.values():
                m = int(nsamp * len(g) / len(U)) + 1
                for _ in range(m): a, b = rnd.sample(g, 2) if len(g) >= 2 else (None, None); pr.append((a, b)) if a else None
        pr = cap(pr, nsamp)
        return mean([bool(a['el'] & b['el']) for a, b in pr])
    o = same_site_share(U, [u['area'] for u in U])
    nn = []
    for _ in range(50):
        labs = [u['area'] for u in U]; rnd.shuffle(labs); nn.append(same_site_share(U, labs))
    P(f'  same-site legend pairs sharing >= 1 name word: {o:.4f} vs site labels permuted {mean(nn):.4f} (x{o / mean(nn):.2f}, P={pval(o, nn):.3f})')
if CY == 2:
    for site in ('Mohenjo-daro', 'Harappa'):
        for oc in ('seal', 'tablet'): cycle2(site, oc)
    ur3_control()
    P('\n  Linear B: DAMOS items carry only the tablet heading (site prefix KN/PY/...), no room or find-spot: palace-room comparison not available.')

# ======================================================================= cycle 3
def cycle3(site, oc):
    pool = [o for o in OBJ if o['site'] == site and o['oc'] == oc and o['room'] and o['emb']]
    allsite = [o for o in OBJ if o['site'] == site and o['oc'] == oc]
    F = freq_table(allsite); ntexts = len({o['seq'] for o in allsite})
    pairs, G = group_pairs(pool, 'room')
    pairs = [(a, b) for a, b in pairs if a['seq'] != b['seq']]
    if len(pairs) < 3: return
    sh = [(a, b) for a, b in pairs if a['el'] & b['el']]; no = [(a, b) for a, b in pairs if not (a['el'] & b['el'])]
    e_sh = mean([a['emb'] == b['emb'] for a, b in sh]) if sh else float('nan'); e_no = mean([a['emb'] == b['emb'] for a, b in no]) if no else float('nan')
    P(f'\n--- {site} {oc}s: same-room non-identical pairs with emblem {len(pairs)}; sharing >= 1 element {len(sh)}: same emblem {ub(sum(a["emb"] == b["emb"] for a, b in sh), len(sh))} = {e_sh:.3f}; sharing none {len(no)}: {e_no:.3f}')
    null = []
    for _ in range(NP):
        embs = collections.defaultdict(list)
        for o in pool: embs[o['typ']].append(o['emb'])
        for k in embs: rnd.shuffle(embs[k])
        idx = collections.Counter(); lab = {}
        for o in pool: lab[id(o)] = embs[o['typ']][idx[o['typ']]]; idx[o['typ']] += 1
        null.append(mean([lab[id(a)] == lab[id(b)] for a, b in sh]) if sh else float('nan'))
    P(f'  emblem permuted within type ({NP}x): same-emblem share among element-sharing room pairs {mean(null):.3f} (P={pval(e_sh, null):.3f})')
    # site-wide: pairs sharing a RARE element (any find spot) vs matched null on emblem / material
    A = [o for o in allsite if o['emb']]
    rare_pairs = []
    byel = collections.defaultdict(list)
    for o in A:
        for e in o['el']:
            if fclass(F, e) == 'rare': byel[e].append(o)
    seenp = set()
    for e, os_ in byel.items():
        for a, b in itertools.combinations(os_, 2):
            if a['seq'] == b['seq']: continue
            k = (id(a), id(b))
            if k in seenp: continue
            seenp.add(k); rare_pairs.append((a, b))
    if rare_pairs:
        obs_e = mean([a['emb'] == b['emb'] for a, b in rare_pairs]); obs_m = mean([a['mat'] == b['mat'] for a, b in rare_pairs if a['mat'] and b['mat']])
        S = collections.defaultdict(list)
        for o in A: S[(o['typ'], min(o['L'], 7))].append(o)
        ne_ = []; nm_ = []
        for _ in range(NP):
            rp = []
            for a, b in rare_pairs:
                x = rnd.choice(S[(a['typ'], min(a['L'], 7))]); y = rnd.choice(S[(b['typ'], min(b['L'], 7))])
                if x is not y: rp.append((x, y))
            ne_.append(mean([a['emb'] == b['emb'] for a, b in rp])); nm_.append(mean([a['mat'] == b['mat'] for a, b in rp if a['mat'] and b['mat']]))
        P(f'  site-wide pairs sharing a RARE element (< 5 texts), non-identical: {len(rare_pairs)}; same emblem {obs_e:.3f} vs matched null {mean(ne_):.3f} (P={pval(obs_e, ne_):.3f}); same material {obs_m:.3f} vs {mean(nm_):.3f} (P={pval(obs_m, nm_):.3f})')
        # by emblem class: which emblems carry it
        cnt = collections.Counter((a['emb']) for a, b in rare_pairs if a['emb'] == b['emb'])
        base = collections.Counter(o['emb'] for o in A)
        P('    same-emblem rare pairs by emblem: ' + ', '.join(f'{e} {n} (site share {base[e] / len(A):.2f})' for e, n in cnt.most_common(6)))
if CY == 3:
    for site in ('Mohenjo-daro', 'Harappa'):
        for oc in ('seal', 'tablet'): cycle3(site, oc)

# ======================================================================= cycle 4
MCL = {342: 'C740', 211: 'C520', 12: 'C151', 15: 'C156', 254: 'C527', 60: 'C226', 245: 'C617', 328: 'C700', 66: 'C236'}
MOPEN = {267, 391, 293, 150}; MMARK = {99, 100, 123}; MMJAR = {343, 344, 345, 346}; MSUF = {176, 1}
MNUM = set()
def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    objs = collections.OrderedDict()
    for r in rows:
        if r['line'] == '9' or not r['signs_clean'].strip(): continue
        key = (r['text_no'], r['side'])
        o = objs.setdefault(key, dict(cisi='IM' + r['text_no'] + '.' + r['side'], site=r['site'], ot=r['object_type'], seq=[], locus=r['locus'].strip(), level=r['level'].strip(), emb=r['fs80'].strip(), mat=None))
        o['seq'].extend(int(x) for x in r['signs_clean'].split() if x != '0')
    T = []
    tmap = {'seal': 'seal', 'sealing': 'tablet', 'miniature tablet': 'tablet', 'copper tablet': 'tablet'}
    for key, o in objs.items():
        if len(o['seq']) < 2 or o['ot'] not in tmap: continue
        s = o['seq']; n = len(s); i = 0; j = n; lab = ['NAME'] * n
        if s[0] in MOPEN:
            lab[0] = 'OPENER'; i = 1
            if n > 1 and s[1] in MMARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] == 293 and n > 2 and s[2] in MMJAR: lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in MSUF and (s[j - 2] in MCL or s[j - 2] in MSUF): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in MCL:
            lab[j - 1] = 'CLOSER'; ci = j - 1
            if ci - 1 >= i: lab[ci - 1] = 'TITLE'
        ell = [s[k] for k in range(n) if lab[k] == 'NAME']
        room = (o['locus'],) if o['locus'] not in ('', '0', '-') else None
        T.append(dict(cisi=o['cisi'], site=o['site'], oc=tmap[o['ot']], typ=o['ot'], seq=tuple(s), lab=lab, ell=ell, el=frozenset(ell), L=n,
                      area=o['level'] if o['level'] not in ('', '-') else None, room=room, emb=o['emb'] or None, mat=None, head=None))
    return T
def small_site(pool, allsite, label):
    F = freq_table(allsite); ntexts = len({o['seq'] for o in allsite})
    pairs, G = group_pairs(pool, 'room')
    k0 = len(pairs); pairs_ni = [(a, b) for a, b in pairs if a['seq'] != b['seq']]
    if len(pairs_ni) < 2:
        P(f'\n--- {label}: {len(pool)} objects with a room, {k0} same-room pairs ({len(pairs_ni)} non-identical): too few'); return None
    obs, k, byel, _ = agg(pairs, F, ntexts)
    nc = collections.defaultdict(list)
    for _ in range(NP):
        st, _, _, _ = agg(permuted(pool, 'room', lambda o: o['typ']), F, ntexts)
        for key in KEYS: nc[key].append(st[key])
    P(f'\n--- {label}: {len(pool)} objects with a room ({len(allsite)} at site), {len([g for g in G.values() if len(g) >= 2])} rooms with >= 2, {k0} same-room pairs, {k} non-identical')
    report('same-room pairs', obs, k, {'c:room-perm': nc})
    return obs, k, nc
if CY == 4:
    for site in ('Chanhu-daro', 'Kalibangan', 'Lothal', 'Dholavira'):
        for oc in ('seal', 'tablet'):
            allsite = [o for o in OBJ if o['site'] == site and o['oc'] == oc]
            pool = [o for o in allsite if o['room']]
            if len(pool) >= 4: small_site(pool, allsite, f'{site} {oc}s by room')
            pool = [dict(o, room=o['area']) for o in allsite if o['area']]
            if len(pool) >= 4: small_site(pool, allsite, f'{site} {oc}s by AREA')
    I = load_im77()
    P(f'\n== IM77 (M space, loop45 frame; elements = NAME tokens minus the sign before the closer): {len(I)} seals + tablets/sealings >= 2 signs; with locus {sum(1 for o in I if o["room"])}')
    for site in ('Mohenjodaro', 'Harappa', 'Chanhudaro', 'Kalibangan'):
        for oc in ('seal', 'tablet'):
            allsite = [o for o in I if o['site'] == site and o['oc'] == oc]
            pool = [o for o in allsite if o['room']]
            if len(pool) >= 4: small_site(pool, allsite, f'IM77 {site} {oc}s by LOCUS')
    # how much rests on Mohenjo-daro: same-room pair counts by site (Wells)
    cnt = collections.Counter()
    for o in OBJ:
        if o['room']: cnt[(o['site'], o['oc'])] += 1
    pc = collections.Counter()
    for (site, oc), n in cnt.items():
        pool = [o for o in OBJ if o['site'] == site and o['oc'] == oc and o['room']]
        pairs, _ = group_pairs(pool, 'room'); pc[(site, oc)] = len([1 for a, b in pairs if a['seq'] != b['seq']])
    tot = sum(pc.values())
    P('\n== share of all same-room non-identical pairs by site x type (Wells): ' + ', '.join(f'{s} {oc} {n} ({100 * n / tot:.0f}%)' for (s, oc), n in pc.most_common()))

os.makedirs(ROOT + 'data/derived/dark', exist_ok=True)
open(ROOT + f'data/derived/dark/loop64_c{CY}_{LV}.txt', 'w').write('\n'.join(OUT) + '\n')
