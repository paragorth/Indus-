"""pe66 cycle 3: the remaining testable C guesses, plus stronger re-tests of cycle 1-2 survivors.

H2 pe54 C+  three recurring herd names carry close counts (leave-one-pair-out, model-free closeness, fresh decoys)
D2 pe48 C   Yahya lexicon (presence-share metric) and Susa avoidance with slot-matched decoys
I2 x4  C    combinatorial designations: add proto-cuneiform lines as the designation calibrator
V2 pe43 C   M005~a header slot vs variant decoys
T  pe24 C   failing totals are off by about one entry (omission rescue vs random totals)
P  pe47 C   tablet pairs joined by a number seen exactly twice share signs (kill: a genre-matched null removes it)
U  pe23 C   M157 entries are rounder even without edge marks (kill: p > 0.1 on new tablets)
F2 pe15 C   two header families
Z  pe16 C   capacity companion line is 0.2-0.67 of the main line
"""
from pe66_lib import *
import pe66_c2 as C2

OUT = []
RES = {}


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ H2 herd names
def test_H2():
    sys.path.insert(0, HERE)
    import pe54_lib as L
    R, X, tabs, strings = L.pe_corpus()
    X = np.asarray(X, float)
    N = len(strings)
    def close(i, j):
        m = (X[i] > 0) & (X[j] > 0)   # unwritten classes = missing (the pe54 'zm' convention)
        if m.sum() == 0:
            return None, 0
        return float(np.mean(np.abs(np.log1p(X[i, m]) - np.log1p(X[j, m])))), int(m.sum())
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if strings[i] == strings[j] and tabs[i] != tabs[j] and not strings[i].startswith('?')]
    pool = collections.defaultdict(list)
    for i in range(N):
        for j in range(i + 1, N):
            if tabs[i] != tabs[j] and strings[i] != strings[j]:
                d, k = close(i, j)
                if d is not None:
                    pool[k].append(d)
    rng = np.random.default_rng(66051)
    per = []
    for i, j in pairs:
        d, k = close(i, j)
        if d is None:
            continue
        p = np.array(pool[k]); per.append((strings[i], tabs[i], tabs[j], k, d, float(np.mean(p <= d))))
    def test(sel, n=20000):
        obs = np.mean([x[4] for x in sel])
        dr = np.zeros(n)
        for x in sel:
            p = np.array(pool[x[3]]); dr += p[rng.integers(len(p), size=n)]
        dr /= len(sel)
        return float(obs), float((1 + np.sum(dr <= obs)) / (n + 1))
    allp = test(per) if per else (None, None)
    loo = [(per[k][0], test(per[:k] + per[k + 1:])) for k in range(len(per))] if len(per) > 1 else []
    # zero-as-observed convention (unwritten = 0)
    def close0(i, j):
        m = (X[i] > 0) | (X[j] > 0)
        return float(np.mean(np.abs(np.log1p(X[i, m]) - np.log1p(X[j, m])))), int(m.sum())
    RES['H2'] = dict(per=per, all=allp, loo=loo)
    OUT.append(row('PE-66.3h', 'pe54 C+ "the three herd names that recur on two tablets with counts carry closer counts than chance" (kill: new cross-tablet pairs no closer than chance; none exist). Strongest now: model-free closeness (mean |log(1+x) difference| over jointly written herd classes, pe54 herd records) vs 20,000 random cross-tablet pairs with the SAME number of jointly written classes (seed 66051); leave-one-pair-out',
                   'pairs %d: %s. All pairs p %.3f. Leave-one-out: %s' % (
                       len(per), '; '.join('%s (%s/%s, %d classes) dist %.2f, null rank %.2f' % x for x in per), allp[1] if allp[1] is not None else float('nan'),
                       '; '.join('without %s p %.3f' % (s, r[1]) for s, r in loo)),
                   ('survives (C+ kept): closeness holds without any single pair' if allp[1] is not None and allp[1] < 0.05 and all(r[1] < 0.05 for _, r in loo) else
                    'KILLED in the model-free form / rests on one pair: demote to C-' if allp[1] is not None else 'untestable')))
    say(OUT[-1])


# ------------------------------------------------------------------ D2 Yahya
def test_D2():
    SUSA = sorted(t['id'] for t in T if site(t) == 'Susa')
    YAH = sorted(t['id'] for t in T if site(t) == 'Yahya')
    TS = {i: {base(s) for l in TAB[i]['lines'] for s in signs_of(l)} for i in TAB}
    L = {i: sum(len(signs_of(l)) for l in TAB[i]['lines']) for i in TAB}
    cand = [s for s in TABN if len(TABN[s]) >= 3 and s not in ('MXXX', 'X') and not s.startswith('MX')]
    def share(g, s):
        return sum(1 for i in g if s in TS[i]) / len(g)
    def held(group, rest, rng, top=4):
        g = list(group); rng.shuffle(g)
        A, B = g[:len(g) // 2], g[len(g) // 2:]
        rs = {s: share(rest, s) for s in cand}
        topk = sorted(cand, key=lambda s: -(share(A, s) - rs[s]))[:top]
        return float(np.mean([share(B, s) - rs[s] for s in topk])), topk
    rng = np.random.default_rng(66052)
    real = [held(YAH, SUSA, rng) for _ in range(40)]
    rm = float(np.mean([r[0] for r in real]))
    tops = collections.Counter(s for r in real for s in r[1])
    srt = sorted(SUSA, key=lambda i: L[i])
    pseudo = []
    for rep in range(40):
        r2 = np.random.default_rng(66400 + rep)
        pick = set()
        for y in YAH:
            cc = [i for i in srt if i not in pick and abs(L[i] - L[y]) <= max(2, 0.2 * L[y])] or [i for i in srt if i not in pick]
            pick.add(cc[r2.integers(len(cc))])
        rest = [i for i in SUSA if i not in pick]
        pseudo.append(float(np.mean([held(sorted(pick), rest, r2)[0] for _ in range(4)])))
    p_site = pval_ge(rm, pseudo)
    # Susa avoidance with decoys matched on frequency AND header-line share
    named = ['M056', 'M044', 'M219', 'M136']
    def hshare(s):
        h, n = C2.hdrshare(s, False)
        return h / max(1, n)
    def cooc(sset):
        return sum(1 for i in SUSA if len(TS[i] & set(sset)) >= 2)
    w = np.array([len(TS[i]) for i in SUSA], float); w /= w.sum()
    def nullc(sset, nrep, seed):
        r3 = np.random.default_rng(seed)
        cnts = [sum(1 for i in SUSA if s in TS[i]) for s in sset]
        out = []
        for _ in range(nrep):
            M = np.zeros(len(SUSA), int)
            for c in cnts:
                M[r3.choice(len(SUSA), size=c, replace=False, p=w)] += 1
            out.append(int((M >= 2).sum()))
        return np.asarray(out)
    o = cooc(named); nl = nullc(named, 2000, 66053)
    p_av = float((1 + np.sum(nl <= o)) / 2001)
    hs = {s: hshare(s) for s in named}
    pool = [s for s in TOK if TOK[s] >= 5 and s not in named and s.startswith(('M', '|'))]
    hsp = {s: hshare(s) for s in pool}
    dps = []
    for rep in range(30):
        r4 = np.random.default_rng(66500 + rep)
        ds = []
        for s in named:
            c = [d for d in pool if d not in ds and 0.5 * TOK[s] <= TOK[d] <= 2 * TOK[s] and abs(hsp[d] - hs[s]) <= 0.1]
            if not c:
                c = sorted([d for d in pool if d not in ds], key=lambda d: abs(hsp[d] - hs[s]) + abs(math.log(TOK[d] / TOK[s])))[:5]
            ds.append(c[r4.integers(len(c))])
        oo = cooc(ds); nn = nullc(ds, 300, 66600 + rep)
        dps.append(float((1 + np.sum(nn <= oo)) / 301))
    dpass = sum(1 for x in dps if x <= max(p_av, 0.01))
    RES['D2'] = dict(real=rm, pseudo=pseudo, p_site=p_site, tops=tops.most_common(8), o=o, null=float(nl.mean()), p_av=p_av, hs=hs, dps=dps)
    OUT.append(row('PE-66.3d', 'pe48 C Yahya lexicon, re-test with an unbiased metric (cycle 1 log-odds rewarded rarity): presence share on the held-out Yahya half minus Susa share, for the 4 signs chosen on the other half (40 splits, seed 66052), vs 40 size-matched Susa pseudo-sites; unclassified MXXX excluded. Susa avoidance re-tested with decoy 4-sets matched on frequency AND header-line share (30 sets), since header signs exclude each other by slot',
                   'held-out presence excess: Yahya %.3f vs pseudo-sites %.3f (p %.3f); chosen most often: %s. Header shares of the 4: %s. Susa tablets with >= 2 of the 4: %d vs null %.1f (p %.4f); slot-matched decoy sets as avoidant: %d of 30' % (
                       rm, float(np.mean(pseudo)), p_site, ', '.join('%s %d' % x for x in tops.most_common(6)),
                       ', '.join('%s %.2f' % kv for kv in hs.items()), o, float(nl.mean()), p_av, dpass),
                   'site lexicon: %s; avoidance: %s' % ('replicates beyond pseudo-sites' if p_site < 0.05 else 'not beyond pseudo-sites (killed as a held-out-replicable lexicon)',
                                                       'survives slot-matched decoys' if p_av < 0.01 and dpass <= 3 else 'explained by slot / not beyond decoys')))
    say(OUT[-1])


# ------------------------------------------------------------------ I2 proto-cuneiform calibrator
def test_I2():
    d = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    docs = []
    for t in d:
        names = []
        for l in t['lines']:
            sg = [s for s in l['signs'] if s != 'x' and not re.match(r'^N\d', s)]
            sg = [re.sub(r'~[a-z0-9]+', '', s) for s in sg]
            if len(sg) >= 2:
                names.append(sg)
        if names:
            docs.append({'id': t['id'], 'names': names})
    rng = np.random.default_rng(66054)
    idx = rng.choice(len(docs), size=min(696, len(docs)), replace=False)
    sub = [docs[i] for i in idx]
    res = []
    for sd in (66021, 66022, 66023):
        A = half(sd, [x['id'] for x in sub])
        tr = [n for x in sub if x['id'] in A for n in x['names']]
        te = [n for x in sub if x['id'] not in A for n in x['names']]
        r = np.random.default_rng(sd)
        res.append((C2.bigram_gain(tr, te, r)[0], C2.bigram_gain(tr, te, r, internal=True)[0], len(tr) + len(te)))
    prev = json.load(open(os.path.join(CK, 'c2_I.json')))['I']['res']
    pe = [(x[0][0], x[1][0]) for x in prev['PE']]; ur = [(x[0][0], x[1][0]) for x in prev['UR3']]
    m = lambda L, k: float(np.mean([x[k] for x in L]))
    RES['I2'] = dict(pc=res)
    pcf, pci = m(res, 0), m(res, 1)
    pef, pei = m(pe, 0), m(pe, 1)
    urf, uri = m(ur, 0), m(ur, 1)
    pos = (pei - pci) / max(1e-9, uri - pci)
    OUT.append(row('PE-66.3i', 'x4 C combinatorial designations, cycle 2 follow-up: the same held-out order-gain test on proto-cuneiform lines (pe2 corpus, 696 random tablets, lines with >= 2 non-numeral signs, seed 66054) as the DESIGNATION calibrator (PC admin entries are designations, not running language), against Ur III personal names as the LANGUAGE calibrator',
                   'bits/sign full / internal: proto-cuneiform %.3f / %.3f; PE %.3f / %.3f; Ur III names %.3f / %.3f. PE internal order sits at %.0f%% of the way from proto-cuneiform to Ur III' % (pcf, pci, pef, pei, urf, uri, 100 * pos),
                   'PE internal order is %s' % ('at or below the designation level: survives (C)' if pos <= 0.25 else
                                               ('between designation and language: undecided, C- (cycle 2 "killed" was set without a designation calibrator)' if pos < 0.6 else 'language-like: KILLED'))))
    say(OUT[-1])


# ------------------------------------------------------------------ V2 M005~a header slot
def test_V2():
    def hs_var(v):
        h, n = C2.hdrshare(v, True)
        return h, n
    hv = hs_var('M005~a'); ho = C2.hdrshare('M005', False)
    diff = hv[0] / hv[1] - ho[0] / max(1, ho[1])
    dd = []
    for v, n in VTOK.items():
        if '~' not in v or v.startswith('|') or n < 15 or v == 'M005~a':
            continue
        a = hs_var(v); b = C2.hdrshare(base(v), False)
        if a[1] and b[1] >= 5:
            dd.append((v, a[0] / a[1] - b[0] / b[1]))
    k = sum(1 for _, x in dd if x >= diff)
    # held-out halves
    hh = []
    for sd in (66021, 66022, 66023):
        B = set(TAB) - half(sd)
        h = n = 0
        for t in T:
            if t['id'] not in B:
                continue
            first = next((k2 for k2, x in enumerate(t['lines']) if x['surface'] == 'obverse'), -1)
            for k2, l in enumerate(t['lines']):
                for s in signs_of(l):
                    if s == 'M005~a':
                        n += 1; h += k2 == first and not l['numerals']
        hh.append((h, n))
    RES['V2'] = dict(hv=hv, ho=ho, k=k, nd=len(dd), hh=hh)
    OUT.append(row('PE-66.3v', 'pe43 C "M005~a takes the header slot, unlike M005" (cycle 2 tested only its number system). Header-line share of M005~a minus that of bare M005, against every other variant with >= 15 tokens (difference to its own bare root); held-out halves (seeds 66021-23)',
                   'M005~a header %d/%d vs M005 %d/%d (difference %.2f); variants with an equal or larger header shift: %d of %d. Held-out halves: %s' % (
                       hv[0], hv[1], ho[0], ho[1], diff, k, len(dd), ', '.join('%d/%d' % x for x in hh)),
                   'survives: header shift larger than %d of %d variant decoys and present in every half' % (len(dd) - k, len(dd)) if k <= 0.05 * len(dd) and all(h > 0 for h, n in hh) else 'not beyond variant decoys: demote'))
    say(OUT[-1])


# ------------------------------------------------------------------ T failing totals off by one entry
def test_T():
    import pe42_common as P
    cases = P.build_pe()
    maps = P.PE_MAPS
    def closes(h, tv_override=None):
        for mname in (['cap'] if h[0]['cls'] == 'cap' else ['dec2', 'dec6', 'sex2', 'sex6', 'cap']):
            m = maps[mname]
            ok = True
            for k, p in enumerate(h):
                tv = P.val(p['T']['nums'], m) if tv_override is None else tv_override[mname][k]
                ev = [P.val(e['nums'], m) for e in p['E']]
                if tv is None or any(v is None for v in ev) or tv != sum(ev):
                    ok = False; break
            if ok:
                return True
        return False
    def rescue(h, tvs=None):
        # omitting 1 entry of any one pair closes?
        for mname in (['cap'] if h[0]['cls'] == 'cap' else ['dec2', 'dec6', 'sex2', 'sex6', 'cap']):
            m = maps[mname]
            vals = []
            bad = False
            for k, p in enumerate(h):
                tv = P.val(p['T']['nums'], m) if tvs is None else tvs[k]
                ev = [P.val(e['nums'], m) for e in p['E']]
                if tv is None or any(v is None for v in ev):
                    bad = True; break
                vals.append((tv, ev))
            if bad:
                continue
            for k, (tv, ev) in enumerate(vals):
                others = all(t2 == sum(e2) for j, (t2, e2) in enumerate(vals) if j != k)
                if others and any(tv == sum(ev) - x for x in ev):
                    return True
        return False
    fail = [c for c in cases if c['tier'] == 1 and not any(closes(h) for h in c['hyps'])]
    obs = sum(1 for c in fail if any(rescue(h) for h in c['hyps']))
    rng = np.random.default_rng(66055)
    # null: replace each failing tablet's total(s) by a random total of the same notation size drawn from other failing tablets
    allT = [(c['id'], h) for c in fail for h in c['hyps'][:1]]
    nl = []
    for rep in range(200):
        hits = 0
        for c in fail:
            h = c['hyps'][0]
            donor = allT[rng.integers(len(allT))][1]
            tvs = None
            # value the donor's first total in the case's own map family
            ok = False
            for mname in (['cap'] if h[0]['cls'] == 'cap' else ['dec2']):
                tv = P.val(donor[0]['T']['nums'], maps[mname])
                if tv is not None:
                    tvs = [tv] + [P.val(p['T']['nums'], maps[mname]) for p in h[1:]]
                    ok = True
            if ok and None not in tvs and rescue(h, tvs):
                hits += 1
        nl.append(hits)
    RES['T'] = dict(nfail=len(fail), obs=obs, null=float(np.mean(nl)))
    p = pval_ge(obs, nl)
    OUT.append(row('PE-66.3t', 'pe24 C "failing totals are off by about one entry" (omitting 1-3 entries rescued 6/26). Clean (tier 1) totalled tablets that close under no value set (pe42 case builder); rescue = omitting ONE entry closes the total under any value set. Null: each tablet\'s total replaced by the total of another failing tablet (200x, seed 66055)',
                   'failing tablets %d; rescued by one omission %d vs null %.1f (p %.3f)' % (len(fail), obs, float(np.mean(nl)), p),
                   'survives (beyond swapped totals)' if p < 0.05 else 'KILLED: one-entry omissions rescue no more failing totals than random totals do'))
    say(OUT[-1])


# ------------------------------------------------------------------ P pairs joined by a number seen twice
def test_P():
    key = collections.defaultdict(set)
    for tid, E in ENT.items():
        for e in E:
            if e['clean'] and e['sys'] in ('SDB', 'C'):
                v = e['cnt'] if e['sys'] == 'SDB' else e['cap']
                if v and v >= (20 if e['sys'] == 'SDB' else 240):
                    key[(e['sys'], v)].add(tid)
    pairs = [tuple(sorted(s)) for k, s in key.items() if len(s) == 2]
    TS = {i: {base(s) for l in TAB[i]['lines'] for s in signs_of(l)} for i in TAB}
    jac = lambda a, b: len(TS[a] & TS[b]) / max(1, len(TS[a] | TS[b]))
    def genre(i):
        E = ENT[i]
        cap = sum(e['sys'] in ('C', 'C*') for e in E) > len(E) / 2 if E else False
        h = header(TAB[i])
        hd = 'M157' if h and 'M157' in h else ('h' if h else '-')
        n = len(E)
        return (cap, hd, 0 if n < 4 else (1 if n < 10 else 2), site(TAB[i]))
    G = collections.defaultdict(list)
    for i in TAB:
        G[genre(i)].append(i)
    obs = float(np.mean([jac(a, b) for a, b in pairs]))
    rng = np.random.default_rng(66056)
    nl = []
    for _ in range(2000):
        s = []
        for a, b in pairs:
            ga, gb = G[genre(a)], G[genre(b)]
            x = ga[rng.integers(len(ga))]; y = gb[rng.integers(len(gb))]
            while y == x:
                y = gb[rng.integers(len(gb))]
            s.append(jac(x, y))
        nl.append(np.mean(s))
    # unmatched null for reference
    ids = sorted(TAB)
    nu = [np.mean([jac(ids[rng.integers(len(ids))], ids[rng.integers(len(ids))]) for _ in pairs]) for _ in range(500)]
    p = pval_ge(obs, nl)
    RES['P'] = dict(n=len(pairs), obs=obs, null=float(np.mean(nl)), p=p, unmatched=float(np.mean(nu)))
    OUT.append(row('PE-66.3p', 'pe47 C "tablet pairs joined by a (rare) number seen exactly twice share signs" (kill: a genre-matched null removes the excess). Numbers >= 20 counted units or >= 2 N01 of capacity that occur on exactly 2 tablets; sign Jaccard of the two tablets vs random pairs drawn from the SAME genres (capacity/count tablet x header M157/other/none x length bin x site), 2,000x, seed 66056',
                   'pairs %d; Jaccard %.3f vs genre-matched %.3f (p %.3f); unmatched random pairs %.3f' % (len(pairs), obs, float(np.mean(nl)), p, float(np.mean(nu))),
                   'survives the genre-matched null' if p < 0.05 else 'KILLED as stated: the genre-matched null removes the excess'))
    say(OUT[-1])


# ------------------------------------------------------------------ U M157 roundness
def test_U():
    def one_denom(nums):
        return len([1 for n, c in nums if n]) == 1
    rows = []
    for tid, E in ENT.items():
        h = header(TAB[tid])
        m157 = bool(h and 'M157' in h)
        edge = any(l['surface'] == 'top' and l['numerals'] for l in TAB[tid]['lines'])
        if edge:
            continue
        for e in E:
            if e['sys'] == 'SDB' and e['cnt'] and e['cnt'] >= 10 and e['clean']:
                rows.append((tid, m157, int(math.log10(e['cnt'])), one_denom(e['nums'])))
    res = []
    for sd in (66021, 66022, 66023):
        B = set(TAB) - half(sd)
        R = [r for r in rows if r[0] in B]
        def stat(R, lab):
            # within size bins: M157 share round minus other share round, weighted
            num = den = 0.0
            for b in set(r[2] for r in R):
                x = [r for r in R if r[2] == b]
                a = [r[3] for r in x if lab[r[0]]]; o = [r[3] for r in x if not lab[r[0]]]
                if a and o:
                    num += len(a) * (np.mean(a) - np.mean(o)); den += len(a)
            return num / max(1, den)
        lab = {r[0]: r[1] for r in R}
        obs = stat(R, lab)
        tids = sorted(lab); vals = [lab[t] for t in tids]
        rng = np.random.default_rng(sd + 11)
        nl = []
        for _ in range(500):
            pv = rng.permutation(vals)
            nl.append(stat(R, dict(zip(tids, pv))))
        res.append((obs, pval_ge(obs, nl), sum(1 for t in tids if lab[t])))
    RES['U'] = res
    OUT.append(row('PE-66.3u', 'pe23 C "M157 entries are slightly rounder even without edge marks" (kill: p > 0.1 on new tablets). Count entries >= 10 on tablets without a top-edge numeral; one-denomination share on M157-headed vs other tablets within size decades; M157 label permuted among tablets (500x) on 3 fresh held-out halves (seeds 66021-23)',
                   '; '.join('half: excess %.3f p %.3f (M157 tablets %d)' % x for x in res),
                   'survives (p <= 0.1 in >= 2 of 3 halves)' if sum(1 for x in res if x[1] <= 0.1) >= 2 else 'KILLED as stated: p > 0.1 on held-out halves'))
    say(OUT[-1])


# ------------------------------------------------------------------ F2 header families
def test_F2():
    F1 = ['M157', '|M327+M342|', 'M388', '|M377+M320+M377|']
    F2 = ['M327', 'M217', 'M247', 'M377', 'M136']
    allh = F1 + F2
    def hkey(t):
        h = header(t)
        if not h:
            return None
        for s in allh:
            if s in h:
                return s
        return None
    TS = {i: collections.Counter(s for e in ENT[i] for s in e['bsigns']) for i in TAB}
    res = []
    for sd in (66021, 66022, 66023):
        B = sorted(set(TAB) - half(sd))
        lab = {i: hkey(TAB[i]) for i in B}
        lab = {i: h for i, h in lab.items() if h and TS[i]}
        prof = {h: collections.Counter() for h in allh}
        for i, h in lab.items():
            prof[h].update(TS[i])
        hs = [h for h in allh if sum(prof[h].values()) >= 10]
        def vec(c):
            keys = sorted(set(k for h in hs for k in prof[h]))
            v = np.array([c[k] for k in keys], float)
            return v / (np.linalg.norm(v) + 1e-9)
        V = {h: vec(prof[h]) for h in hs}
        def score(part):
            w = []; b = []
            for x in range(len(hs)):
                for y in range(x + 1, len(hs)):
                    s = float(V[hs[x]] @ V[hs[y]])
                    (w if part[hs[x]] == part[hs[y]] else b).append(s)
            return np.mean(w) - np.mean(b) if w and b else 0
        part = {h: (h in F1) for h in hs}
        obs = score(part)
        rng = np.random.default_rng(sd + 13)
        nl = []
        vals = [part[h] for h in hs]
        for _ in range(2000):
            pv = rng.permutation(vals)
            nl.append(score(dict(zip(hs, pv))))
        res.append((obs, pval_ge(obs, nl), len(hs)))
    RES['F2'] = res
    OUT.append(row('PE-66.3f', 'pe15 C "two header families {M157, [M327+M342], M388, [M377+M320+M377]} vs {M327, M217, M247, M377, M136}". Entry-sign profile of the tablets each header heads; within-family minus between-family cosine, vs header labels permuted between the two families (2,000x), 3 fresh held-out halves (seeds 66021-23)',
                   '; '.join('excess %.3f p %.3f (headers %d)' % x for x in res),
                   'survives' if sum(1 for x in res if x[1] < 0.05) >= 2 else 'KILLED: the two families do not separate the content of held-out tablets'))
    say(OUT[-1])


# ------------------------------------------------------------------ Z companion line
def test_Z():
    pairs = []
    for tid, E in ENT.items():
        for a, b in zip(E, E[1:]):
            if b['i'] != a['i'] + 1 or a['sys'] != 'C' or b['sys'] != 'C' or not a['cap'] or not b['cap']:
                continue
            comp = len(b['bsigns']) == 1 and len(a['bsigns']) >= 2
            pairs.append((tid, comp, b['final'], b['cap'] / a['cap']))
    inr = lambda r: 0.2 <= r <= 0.67
    cp = [p for p in pairs if p[1]]; op = [p for p in pairs if not p[1]]
    a = sum(inr(p[3]) for p in cp) / max(1, len(cp)); o = sum(inr(p[3]) for p in op) / max(1, len(op))
    rng = np.random.default_rng(66057)
    lab = np.array([p[1] for p in pairs]); r = np.array([inr(p[3]) for p in pairs])
    nl = []
    for _ in range(5000):
        pl = rng.permutation(lab)
        nl.append(r[pl].mean() if pl.any() else 0)
    p = pval_ge(a, nl)
    smaller = sum(1 for x in cp if x[3] < 1) / max(1, len(cp))
    RES['Z'] = dict(nc=len(cp), no=len(op), a=a, o=o, p=p)
    OUT.append(row('PE-66.3z', 'pe16 C "in capacity records the companion line is 0.2-0.67 of the main line (one adult- and one dependant-sized ration)". Adjacent capacity pairs where a multi-sign (name) line is followed by a one-sign companion line, vs all other adjacent capacity pairs (decoy pairs); share of ratios in 0.2-0.67; labels permuted (5,000x, seed 66057)',
                   'companion pairs %d: in range %.2f (smaller than the main line %.2f); other pairs %d: %.2f; p %.3f' % (len(cp), a, smaller, len(op), o, p),
                   'survives beyond decoy pairs' if p < 0.05 else 'KILLED: companion lines fall in 0.2-0.67 no more often than any adjacent capacity pair'))
    say(OUT[-1])


# ------------------------------------------------------------------ T2 same-size discrepancy null
def test_T2():
    import pe42_common as P
    cases = P.build_pe()
    maps = P.PE_MAPS
    def vals(h, mname):
        m = maps[mname]; out = []
        for p in h:
            tv = P.val(p['T']['nums'], m); ev = [P.val(e['nums'], m) for e in p['E']]
            if tv is None or any(v is None for v in ev):
                return None
            out.append((tv, ev))
        return out
    def mlist(h):
        return ['cap'] if h[0]['cls'] == 'cap' else ['dec2', 'dec6', 'sex2', 'sex6', 'cap']
    def closes(h):
        return any((v := vals(h, m)) and all(t == sum(e) for t, e in v) for m in mlist(h))
    fail = [c for c in cases if c['tier'] == 1 and not any(closes(h) for h in c['hyps'])]
    # each failing tablet: its first hypothesis under its first valid map, single-pair structure
    items = []
    for c in fail:
        for h in c['hyps']:
            for m in mlist(h):
                v = vals(h, m)
                if v and len(v) == 1:
                    items.append((c['id'], m, v[0][0], v[0][1]))
                    break
            else:
                continue
            break
    def rescued(t, ev):
        return any(t == sum(ev) - x for x in ev)
    # rescue under ANY map (as the stated test); null uses the item's map and the same relative discrepancy
    obs = sum(1 for c in fail if any(rescued(t, e) for h in c['hyps'] for m in mlist(h) if (v := vals(h, m)) and len(v) == 1 for t, e in v))
    obs1 = sum(1 for _, m, t, ev in items if rescued(t, ev))
    rel = [float((t - sum(ev)) / sum(ev)) for _, _, t, ev in items if sum(ev)]
    rng = np.random.default_rng(66058)
    nl = []
    from fractions import Fraction as Fr
    for _ in range(2000):
        h = 0
        for _, m, t, ev in items:
            r = rel[rng.integers(len(rel))]
            s = sum(ev)
            # discrepancy of the same relative size, snapped to the tablet's smallest unit
            unit = min(maps[m][c] for e in [ev] for c in ['N01'] if c in maps[m])
            unit = min([unit] + [x for x in ev if x > 0])
            d = Fr(round(float(s) * r / float(unit))) * unit if unit else 0
            h += rescued(s + d, ev)
        nl.append(h)
    p = pval_ge(obs1, nl)
    RES['T2'] = dict(nfail=len(fail), items=len(items), obs=obs, obs1=obs1, null=float(np.mean(nl)), p=p)
    OUT.append(row('PE-66.3t2', 'pe24 C off-by-one-entry, harder null: the same failing tablets, but the null keeps each discrepancy\'s SIZE (a relative discrepancy drawn from another failing tablet, applied to this tablet\'s own entry sum and snapped to its units, 2,000x, seed 66058); one-pair tablets in their first valid map',
                   'one-pair failing tablets %d: rescued by omitting one entry %d vs same-size null %.2f (p %.3f); any map / structure: %d of %d' % (len(items), obs1, float(np.mean(nl)), p, obs, len(fail)),
                   'survives the same-size null' if p < 0.05 else 'KILLED: a discrepancy of that size equals some entry by chance as often'))
    say(OUT[-1])


if __name__ == '__main__':
    which = sys.argv[1:] or ['H2', 'D2', 'I2', 'V2', 'T', 'P', 'U', 'F2', 'Z']
    for w in which:
        globals()['test_' + w]()
    tag = '_'.join(which)
    dump(RES, 'c3_%s.json' % tag)
    open(os.path.join(CK, 'c3_%s.rows' % tag), 'w').write('\n'.join(OUT) + '\n')


