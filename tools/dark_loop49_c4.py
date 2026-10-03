"""S-DARK-49 cycle 4: REPLICATION on held-out sites and on IM77 (Mahadevan 1977; ~70% the same objects as Wells, so 'transcription-robust',
not 'replicated', except for the held-out sites and IM77-only texts).
(A) Field profiles (cycle 1 metrics) on the Wells texts from sites other than Mohenjo-daro / Harappa (parser slots and partial-order classes),
    and on IM77 texts with the class sets mapped W -> M through data/derived/bridge_extended.json (W34 -> M95 is a contextual proposal, flagged).
(B) Time-stamp test (cycle 2) on IM77's own excavation levels (column `level`, feet below datum, negative) at Mohenjodaro and Harappa:
    peak share in 4-ft bands against levels shuffled within object type and within object type x locus (NPERM x), BH over signs; and the
    cross-transcription check: Spearman rho between a sign's mean within-city level percentile in IM77 and its mean depth percentile in Wells
    (cycle 2 json), per city, over bridged signs.
(C) Cycle structure (cycle 3b) on IM77 sides (column `side`): cross-side pairs of numerals (M86-M121 values) and of closer-paradigm signs,
    neighbour share under the natural / frequency ordering against sides reassigned within site x object type.
Usage: python3 tools/dark_loop49_c4.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import sys, json, csv, math, collections, random, statistics
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import OPEN, CL, FISH, NUM, learn_qual, make_parser, pval
ROOT = '/home/user/Indus-/'
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rnd = random.Random(494)
OUT = open(ROOT + f'data/derived/dark/loop49_c4_{LV}.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

# ---------------------------------------------------------------- metrics (as cycle 1)
def metrics(cnt):
    f = sorted(cnt.values(), reverse=True); N = sum(f); K = len(f)
    if K == 0: return None
    p = [x / N for x in f]; D = 1 / sum(x * x for x in p); H = -sum(x * math.log(x) for x in p if x > 0)
    if K >= 3:
        xs = [math.log(i + 1) for i in range(K)]; ys = [math.log(x) for x in f]; mx = sum(xs) / K; my = sum(ys) / K
        sxx = sum((x - mx) ** 2 for x in xs); slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float('nan')
    else: slope = float('nan')
    return dict(K=K, N=N, D=D, ED=D / K, EH=(H / math.log(K)) if K > 1 else 1.0, slope=slope, top=p[0])
def band(m):
    if 8 <= m['K'] <= 40 and m['ED'] >= 0.6 and (math.isnan(m['slope']) or m['slope'] > -0.5): return 'CALENDAR-band'
    if m['ED'] < 0.3 or (not math.isnan(m['slope']) and m['slope'] <= -0.7): return 'name-band'
    return 'between'
def fmt(m): return f"K={m['K']:4d} N={m['N']:5d} D/K={m['ED']:.2f} H/logK={m['EH']:.2f} slope={m['slope']:+.2f} top={m['top']:.2f} [{band(m)}]"

# ---------------------------------------------------------------- (A1) Wells held-out sites
def otype(t):
    t0 = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t0, 'other')
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
T = [dict(site=r['site'], ot=otype(r['type']), seq=list(r[LV])) for r in C if r[LV] and len(r[LV]) >= 2 and r['complete'] == 'Y' and r['dir.'].strip() != '-']
parse = make_parser(learn_qual([t['seq'] for t in T if t['site'] in ('Mohenjo-daro', 'Harappa')]))  # qualifiers learnt on MD+H only
HO = [t for t in T if t['site'] not in ('Mohenjo-daro', 'Harappa')]
say(f'# S-DARK-49 cycle 4, level {LV}, nperm {NPERM}: replication. Wells held-out-site texts {len(HO)} ({collections.Counter(t["site"] for t in HO).most_common(6)})')
PO = collections.OrderedDict([('PO0 openers', set(OPEN)), ('PO1 marked jar / leaf', {741, 742, 745, 803, 806}), ('PO2 tall numerals', {31, 32, 33, 34}),
                              ('PO3 350-798-415', {350, 798, 415}), ('PO4 fish words', set(FISH)), ('PO5 titles 705/706/255/435/690', {705, 706, 255, 435, 690}),
                              ('PO6 closer paradigm', set(CL)), ('numeral values', set(NUM))])
say('\n== (A1) Wells held-out sites: slot and class profiles')
F = collections.defaultdict(collections.Counter); multi = collections.Counter(); ntx = collections.Counter()
for t in HO:
    s = t['seq']; lab = parse(s); per = collections.Counter()
    for w, l in zip(s, lab): F['slot ' + l][w] += 1
    for nm, S in PO.items():
        h = [w for w in s if w in S]
        for w in h: F['class ' + nm][w] += 1
        if h: ntx[nm] += 1
        if len(set(h)) >= 2: multi[nm] += 1
for name in sorted(F):
    m = metrics(F[name]); say(f'{name:40s} {fmt(m)}' + (f'  texts with >= 2 distinct members {multi[name[6:]]}/{ntx[name[6:]]}' if name.startswith('class') else ''))
    say(f"{'':40s}   top: {', '.join(f'{w}:{n}' for w, n in F[name].most_common(10))}")

# ---------------------------------------------------------------- IM77 load
B = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
W2M = {int(w): ms for w, ms in B.items()}
def toM(S, extra=()):
    out = set()
    for w in S: out.update(W2M.get(w, []))
    out.update(extra); return out
PO_M = collections.OrderedDict([(k, toM(S)) for k, S in PO.items()])
PO_M['PO2 tall numerals'] |= {95}; PO_M['numeral values'] |= {95}  # W34 -> M95 contextual proposal (bridge_proposals.json), flagged
PO_M['PO6 closer paradigm'] |= {162, 169}  # M162/169 tree closers (S289 IM77 paradigm M15, M254, M12, M211, M60)
MVAL = {97: 1, 98: 1, 102: 3, 103: 3, 104: 4, 105: 4, 106: 5, 107: 5, 108: 6, 109: 6, 110: 7, 112: 7, 114: 8, 86: 1, 87: 2, 89: 3, 95: 4, 121: 12, 99: 2, 100: 2}
rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
TX = collections.OrderedDict()
for r in rows:
    if r['direction'] in ('no text',): continue
    toks = [int(x) for x in r['signs_reading_order'].split() if x.isdigit() and int(x) != 0]
    if not toks: continue
    t = TX.setdefault(r['text_no'], dict(site=r['site'], ot=r['object_type'], level=None, locus=r['locus'], sides=collections.defaultdict(list)))
    t['sides'][r['side']].extend(toks)
    try:
        lv = float(r['level'])
        if lv < 0: t['level'] = -lv
    except ValueError: pass
for t in TX.values(): t['all'] = [w for k in sorted(t['sides']) for w in t['sides'][k]]
say(f'\nIM77: {len(TX)} texts with signs; sites {collections.Counter(t["site"] for t in TX.values()).most_common()}; with a negative level: MD {sum(1 for t in TX.values() if t["site"]=="Mohenjodaro" and t["level"])}, HP {sum(1 for t in TX.values() if t["site"]=="Harappa" and t["level"])}')
say('class sets in M numbers: ' + '; '.join(f'{k}: {sorted(v)}' for k, v in PO_M.items()))

# (A2) IM77 class profiles, all sites and small sites
say('\n== (A2) IM77 class profiles (bridge-mapped; M95 = W34 is a proposal)')
for subset, TT in (('all IM77', list(TX.values())), ('IM77 small sites (Lothal, Kalibangan, Chanhudaro, other)', [t for t in TX.values() if t['site'] not in ('Mohenjodaro', 'Harappa')])):
    say(f'-- {subset}: {len(TT)} texts')
    for nm, S in PO_M.items():
        c = collections.Counter(w for t in TT for w in t['all'] if w in S)
        if not c: say(f'   {nm:36s} no tokens'); continue
        n2 = sum(1 for t in TT if len(set(w for w in t['all'] if w in S)) >= 2); n1 = sum(1 for t in TT if any(w in S for w in t['all']))
        say(f'   {nm:36s} {fmt(metrics(c))}  >=2 distinct members {n2}/{n1}; top: {", ".join(f"M{w}:{n}" for w, n in c.most_common(8))}')
    allc = collections.Counter(w for t in TT for w in t['all']); say(f'   {"all IM77 sign tokens (reference)":36s} {fmt(metrics(allc))}')

# ---------------------------------------------------------------- (B) IM77 levels: time-stamp test
say('\n== (B) IM77 excavation levels (ft below datum, 4-ft bands): concentration of each sign (>= 8 dated texts) vs levels shuffled within object type / object type x locus')
def iqr(v):
    v = sorted(v); n = len(v); return v[int(0.75 * (n - 1))] - v[int(0.25 * (n - 1))]
def peak(v): return collections.Counter(int(x // 4) for x in v).most_common(1)[0][1] / len(v)
def bh(ps, q=0.05):
    m = len(ps); srt = sorted(range(m), key=lambda i: ps[i]); thr = 0
    for k, i in enumerate(srt):
        if ps[i] <= q * (k + 1) / m: thr = k + 1
    return set(srt[:thr])
c2 = json.load(open(ROOT + f'data/derived/dark/loop49_c2_{LV}.json'))
IM_PCT = {}
for site, wkey in (('Mohenjodaro', 'md'), ('Harappa', 'hp')):
    D = [t for t in TX.values() if t['site'] == site and t['level']]
    if len(D) < 30: say(f'-- {site}: {len(D)} dated texts, skipped'); continue
    vals = [t['level'] for t in D]
    cnt = collections.Counter(w for t in D for w in set(t['all']))
    elems = [w for w, n in cnt.items() if n >= 8]
    idx = {w: [i for i, t in enumerate(D) if w in t['all']] for w in elems}
    obs = {w: peak([vals[i] for i in idx[w]]) for w in elems}
    nulls = {w: [] for w in elems}; nulls2 = {w: [] for w in elems}
    for it in range(NPERM):
        for nd, keyf in ((nulls, lambda t: t['ot']), (nulls2, lambda t: (t['ot'], t['locus']))):
            groups = collections.defaultdict(list)
            for i, t in enumerate(D): groups[keyf(t)].append(i)
            nv = list(vals)
            for g in groups.values():
                vs = [vals[i] for i in g]; rnd.shuffle(vs)
                for i, x in zip(g, vs): nv[i] = x
            for w in elems: nd[w].append(peak([nv[i] for i in idx[w]]))
    P1 = {w: pval(obs[w], nulls[w]) for w in elems}; P2 = {w: pval(obs[w], nulls2[w]) for w in elems}
    s1 = bh([P1[w] for w in elems]); s2 = bh([P2[w] for w in elems])
    say(f'-- {site}: {len(D)} dated texts, {len(elems)} signs; BH-significant peak concentration: type-null {len(s1)}, type x locus null {len(s2)}; raw P<0.05: {sum(p<0.05 for p in P1.values())} / {sum(p<0.05 for p in P2.values())} (expected {0.05*len(elems):.1f})')
    say(f'   sample bands: {sorted(collections.Counter(int(x // 4) for x in vals).items())}')
    for w in sorted(elems, key=lambda w: P2[w])[:8]:
        v = [vals[i] for i in idx[w]]
        say(f'   M{w:<4d} n={len(v):3d} mean={statistics.mean(v):6.2f} peak={obs[w]:.2f} (null {sum(nulls[w])/NPERM:.2f}) P_type={P1[w]:.3f} P_locus={P2[w]:.3f}')
    # cross-transcription: mean level percentile per sign vs Wells mean depth percentile (cycle 2 json), bridged
    srt = sorted(vals)
    def pct(x): return (sum(1 for v in srt if v < x) + 0.5 * sum(1 for v in srt if v == x)) / len(srt)
    IM_PCT[site] = {w: statistics.mean(pct(vals[i]) for i in idx[w]) for w in elems}
    wells = c2[wkey]
    wvals = {}
    for wstr, r in wells.items():
        for m in W2M.get(int(wstr), []):
            if m in IM_PCT[site]: wvals.setdefault(m, []).append(r['mean'])
    shared = sorted(wvals)
    if len(shared) >= 8:
        x = [IM_PCT[site][m] for m in shared]; y = [statistics.mean(wvals[m]) for m in shared]
        def rk(v):
            s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
            for k, i in enumerate(s): r[i] = k
            return r
        rx, ry = rk(x), rk(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
        rho = sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        say(f'   cross-transcription: {len(shared)} bridged signs dated in both; Spearman rho (IM77 mean level percentile vs Wells mean depth) = {rho:+.3f} (same objects largely: transcription-robustness, not replication)')
if 'Mohenjodaro' in IM_PCT and 'Harappa' in IM_PCT:
    sh = [m for m in IM_PCT['Mohenjodaro'] if m in IM_PCT['Harappa']]
    x = [IM_PCT['Mohenjodaro'][m] for m in sh]; y = [IM_PCT['Harappa'][m] for m in sh]
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(s): r[i] = k
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    rho = sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    # null: shuffle levels within type at both cities
    null = []
    for it in range(min(NPERM, 500)):
        pc = {}
        for site in ('Mohenjodaro', 'Harappa'):
            D = [t for t in TX.values() if t['site'] == site and t['level']]; vals = [t['level'] for t in D]
            groups = collections.defaultdict(list)
            for i, t in enumerate(D): groups[t['ot']].append(i)
            nv = list(vals)
            for g in groups.values():
                vs = [vals[i] for i in g]; rnd.shuffle(vs)
                for i, v_ in zip(g, vs): nv[i] = v_
            srt = sorted(nv)
            def pct(x_): return (sum(1 for v in srt if v < x_) + 0.5 * sum(1 for v in srt if v == x_)) / len(srt)
            pc[site] = {m: statistics.mean(pct(nv[i]) for i, t in enumerate(D) if m in t['all']) for m in sh}
        x = [pc['Mohenjodaro'][m] for m in sh]; y = [pc['Harappa'][m] for m in sh]
        rx, ry = rk(x), rk(y); null.append(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)))
    ns = sorted(null)
    say(f'-- IM77 cross-city: {len(sh)} signs dated >= 8 in both cities; Spearman rho of mean level percentile MD vs HP = {rho:+.3f}; null (levels shuffled within type) [{ns[int(0.025*len(ns))]:+.3f}, {ns[int(0.975*len(ns))-1]:+.3f}] P_hi={pval(rho, null):.3f}')

# ---------------------------------------------------------------- (C) IM77 sides: cross-side neighbour pairs
say('\n== (C) IM77 multi-sided texts: cross-side distinct-member pairs; neighbour share (|i-j|=1) vs sides reassigned within site x object type')
MS = [t for t in TX.values() if sum(1 for k, v in t['sides'].items() if v) >= 2]
say(f'texts with >= 2 inscribed sides: {len(MS)} ({collections.Counter((t["site"], t["ot"]) for t in MS).most_common(6)})')
def pairs(objs, S):
    P = []
    for o in objs:
        fm = [set(w for w in v if w in S) for v in o]
        for a in range(len(fm)):
            for b in range(a + 1, len(fm)):
                for wa in fm[a]:
                    for wb in fm[b]:
                        if wa != wb: P.append((wa, wb))
    return P
def d1(P, rank, K, cyc):
    if not P: return None
    c = 0
    for a, b in P:
        d = abs(rank[a] - rank[b]); d = min(d, K - d) if cyc else d
        c += d == 1
    return c / len(P)
allc = collections.Counter(w for t in TX.values() for w in t['all'])
for nm, S, natural in (('numerals (M86-M121 values; M97/98/99/100 = 1/2 markers included)', set(MVAL), lambda w: (MVAL[w], w)),
                       ('tall numerals M86/87/89/95', {86, 87, 89, 95}, lambda w: MVAL[w]),
                       ('closer paradigm (M)', PO_M['PO6 closer paradigm'], None),
                       ('fish words (M)', PO_M['PO4 fish words'], None)):
    S = set(w for w in S if allc[w]); K = len(S)
    sides = [[v for v in t['sides'].values() if v] for t in MS]
    P = pairs(sides, S); nobj = sum(1 for o in sides if sum(1 for v in o if any(w in S for w in v)) >= 2)
    say(f'-- {nm}: K={K}; texts with the set on >= 2 sides {nobj}; cross-side pairs {len(P)}' + (f'; upper bound < {3/len(MS):.4f}' if nobj == 0 else ''))
    if not P: continue
    orderings = ([('natural', {w: i for i, w in enumerate(sorted(S, key=natural))})] if natural else []) + [('frequency', {w: i for i, w in enumerate(sorted(S, key=lambda w: -allc[w]))})]
    groups = collections.defaultdict(list)
    for t in MS: groups[(t['site'], t['ot'])].append(t)
    for oname, rank in orderings:
        for cyc in (False, True):
            sh = d1(P, rank, K, cyc); null = []
            for it in range(min(NPERM, 500)):
                fake = []
                for g in groups.values():
                    pool = [v for t in g for v in t['sides'].values() if v]; rnd.shuffle(pool); p = 0
                    for t in g:
                        k = sum(1 for v in t['sides'].values() if v); fake.append(pool[p:p + k]); p += k
                x = d1(pairs(fake, S), rank, K, cyc); null.append(x if x is not None else 0.0)
            ns = sorted(null)
            say(f'   ordering {oname:10s} {"cyclic" if cyc else "linear"}: neighbour share {sh:.2f} vs null {sum(null)/len(null):.2f} [{ns[int(0.025*len(ns))]:.2f}-{ns[int(0.975*len(ns))-1]:.2f}] P_hi={pval(sh, null):.3f}')
    say(f'   commonest cross-side pairs: {collections.Counter(tuple(sorted(p)) for p in P).most_common(6)}')
    if natural:
        say(f'   values differing by exactly 1: {sum(1 for a, b in P if abs(MVAL[a]-MVAL[b]) == 1)} of {len(P)}; same value on two sides: {sum(1 for o in sides for a in range(len(o)) for b in range(a+1, len(o)) for w in set(o[a]) & set(o[b]) & S)}')
