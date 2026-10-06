"""pe66 cycle 1: kill tests for the explicitly named C guesses with numeric kill lines.

A  pe42  M388 tablets are read in capacity (N14 = 6 N01).       kill: an M388 tablet that closes only as a count
B  pe44  P009258 M297 = deliveries against a norm of 10.          kill: M297 series with no common centre
C  pe56  M297 heads closing totals equal to the running sum.    kill: M297 total lines that do not sum
D  pe48  Yahya lexicon M056/M044/M219/M136 (+ Susa avoidance).   strongest: site-held-out, pseudo-site decoys
F  pe58  M009 : M102 = 1 : 2 within tablets.                       strongest: within-tablet relabel null + decoy pairs + decoy ratios
G  pe27/pe63  48 N39C per counted unit (M056 -> M288; dossier D2). strongest: outside the discovery tablets, decoy rates
H  pe28  [M036+1(N30D)] = 5 N01 per counted unit.                 kill: other multiples after 2-4 units
L  pe15/pe38/pe41  [M327+M342] is a header/office sign.           strongest: held-out halves, decoys, line-order null
O  pe15  grain-office signs M081 M265 M266 M296 M112 M286 M248.   kill: < 30% capacity
"""
import itertools
from pe66_lib import *
from fractions import Fraction as Fr

OUT = []
RES = {}
K = 30


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ A  M388 capacity
def test_A():
    sys.path.insert(0, HERE)
    import pe42_common as P
    cases = P.build_pe()
    ok = {'N01', 'N14', 'N45', 'N34', 'N48'}
    tabs = []
    for c in cases:
        hy = [h for h in c['hyps'] if all(p['cls'] == 'cnt' for p in h)]
        if not hy:
            continue
        codes = {cc for h in hy for p in h for l in [p['T']] + p['E'] for _, cc in l['nums']}
        if not codes <= ok or not (codes - {'N01'}):
            continue  # must be SDB-only and discriminating (some N14+)
        def closes(m):
            for h in hy:
                good = True
                for p in h:
                    tv = P.val(p['T']['nums'], m)
                    ev = [P.val(e['nums'], m) for e in p['E']]
                    if tv is None or any(v is None for v in ev) or tv != sum(ev):
                        good = False; break
                if good:
                    return True
            return False
        cn = any(closes(P.PE_MAPS[m]) for m in ('dec2', 'dec6', 'sex2', 'sex6'))
        cp = closes(P.PE_MAPS['cap'])
        sg = {base(s) for l in TAB[c['id']]['lines'] for s in signs_of(l)}
        tabs.append((c['id'], cn, cp, sg))
    def score(sign):
        rel = [(i, cn, cp) for i, cn, cp, sg in tabs if sign in sg]
        co = [i for i, cn, cp in rel if cn and not cp]
        ca = [i for i, cn, cp in rel if cp and not cn]
        return len(rel), ca, co
    n, ca, co = score('M388')
    allco = sum(1 for _, cn, cp, _ in tabs if cn and not cp)
    allca = sum(1 for _, cn, cp, _ in tabs if cp and not cn)
    dec = decoys('M388', K, 66001, lo=0.5, hi=2.0)
    dres = []
    for d in dec:
        dn, dca, dco = score(d)
        if dn:
            dres.append((d, dn, len(dca), len(dco)))
    # decoys "passing" = at least one cap-only tablet and no count-only tablet
    dpass = sum(1 for _, dn, a, b in dres if a >= 1 and b == 0)
    dshare = [a / (a + b) for _, _, a, b in dres if a + b]
    RES['A'] = dict(n=n, cap_only=ca, cnt_only=co, all_cap_only=allca, all_cnt_only=allco, decoys=dres)
    killed = len(co) > 0
    verdict = ('KILLED as stated: %d M388 tablet(s) close only as counts (%s)' % (len(co), ', '.join(co)) if killed else
               'survives the stated kill; vs decoys: %d/%d decoy signs also pass' % (dpass, len(dres)))
    OUT.append(row('PE-66.1a', 'pe42 C "M388 tablets are read in capacity". Totalled tablets written only in N01/N14+ (pe42 case builder, SDB-only, at least one N14 or higher so the two readings differ); closure under the 4 count value sets (N14 = 10) vs the capacity chain (N14 = 6 N01). Stated kill: an M388 tablet that closes only as a count. Decoys: %d signs of 0.5-2x M388 frequency (seed 66001), same test' % len(dec),
                   'discriminating tablets %d (cap-only %d, count-only %d overall). M388 on %d: capacity-only %d (%s), count-only %d (%s). Decoy signs with >=1 cap-only and 0 count-only: %d/%d; decoy cap-only share median %.2f' % (
                       len(tabs), allca, allco, n, len(ca), ' '.join(ca), len(co), ' '.join(co), dpass, len(dres), float(np.median(dshare)) if dshare else float('nan')),
                   verdict))
    say(OUT[-1])


# ------------------------------------------------------------------ B  M297 norm of 10
def series(sign, minn=4):
    out = []
    for tid, E in ENT.items():
        for sy, key in (('SDB', 'cnt'), ('C', 'cap')):
            v = [e[key] for e in E if e['final'] == sign and e['sys'] == sy and e[key] is not None and e[key] > 0]
            if len(v) >= minn:
                out.append((tid, sy, v))
    return out


def cv(v):
    v = np.asarray(v, float)
    return v.std() / v.mean()


def series_test(sign, rng, nnull=2000):
    S = series(sign)
    if not S:
        return None
    pool = {sy: [e[k] for E in ENT.values() for e in E if e['final'] == sign and e['sys'] == sy and e[k] and e[k] > 0]
            for sy, k in (('SDB', 'cnt'), ('C', 'cap'))}
    obs = float(np.median([cv(v) for _, _, v in S]))
    null = []
    for _ in range(nnull):
        null.append(np.median([cv(rng.choice(pool[sy], size=len(v))) for _, sy, v in S]))
    p = float((1 + np.sum(np.asarray(null) <= obs)) / (1 + nnull))
    near10 = sum(1 for _, sy, v in S if sy == 'SDB' and 8 <= np.median(v) <= 12)
    return dict(n=len(S), obs=obs, null=float(np.mean(null)), p=p, near10=near10,
                nsdb=sum(1 for _, sy, _ in S if sy == 'SDB'), series=[(t, sy, v) for t, sy, v in S])


def test_B():
    rng = np.random.default_rng(66002)
    r = series_test('M297', rng)
    S = [s for s in r['series'] if s[0] != 'P009258']
    # leave P009258 out (the discovery tablet)
    pool = [e['cnt'] for E in ENT.values() for e in E if e['final'] == 'M297' and e['sys'] == 'SDB' and e['cnt']]
    sdb = [v for t, sy, v in S if sy == 'SDB']
    obs_lo = float(np.median([cv(v) for v in sdb])) if sdb else float('nan')
    nl = [np.median([cv(rng.choice(pool, size=len(v))) for v in sdb]) for _ in range(2000)] if sdb else [0]
    p_lo = float((1 + np.sum(np.asarray(nl) <= obs_lo)) / 2001)
    near10 = sum(1 for v in sdb if 8 <= np.median(v) <= 12)
    meds = [float(np.median(v)) for v in sdb]
    # decoys: same series test on matched-frequency signs with series
    dec = decoys('M297', 40, 66003, lo=0.4, hi=2.5)
    dr = []
    for d in dec:
        x = series_test(d, np.random.default_rng(66004), 500)
        if x and x['n'] >= 2:
            dr.append((d, x['n'], x['p'], x['near10'], x['nsdb']))
    dpass = sum(1 for d in dr if d[2] < 0.05)
    RES['B'] = dict(all=r, held=dict(n=len(sdb), cv=obs_lo, p=p_lo, near10=near10, meds=meds), decoys=dr)
    common_centre = p_lo < 0.05
    at10 = near10 / max(1, len(sdb))
    verdict = ('M297 series DO have a common centre within tablets (low spread, p %.3g) but the centre is 10 in only %d of %d other SDB series (medians %s): the "norm of 10" is KILLED; the narrower "fixed per-tablet allotment" stays (= pe44 B-)' % (p_lo, near10, len(sdb), meds)
               if common_centre and at10 < 0.5 else
               ('survives: series cluster and centre near 10 in %d/%d' % (near10, len(sdb)) if common_centre else
                'KILLED as stated: M297 series show no common centre (p %.2f)' % p_lo))
    OUT.append(row('PE-66.1b', 'pe44 C "P009258 M297 = 8,10,7,10,8,9: deliveries against a norm of 10". Every other tablet with >= 4 M297 entries of one system (P009258 left out); spread = median coefficient of variation; null = values drawn from all M297 values of that system (2,000x, seed 66002). Stated support: other M297 series scatter around 10; kill: no common centre. Decoys: 40 signs of 0.4-2.5x frequency with series (seed 66003), same test',
                   'M297 series %d (SDB %d, held-out of P009258). SDB median CV %.2f vs null %.2f (p %.3g). Series medians: %s; within 8-12: %d of %d. Decoys with series: %d; with p < 0.05: %d' % (
                       r['n'], len(sdb), obs_lo, float(np.mean(nl)), p_lo, meds, near10, len(sdb), len(dr), dpass),
                   verdict))
    say(OUT[-1])


# ------------------------------------------------------------------ C  M297 closing totals
def running_sum_hits(sign):
    hits = tot = 0
    for tid, E in ENT.items():
        for k, e in enumerate(E):
            if e['final'] != sign or k < 2 or e['sys'] not in ('SDB', 'C'):
                continue
            key = 'cnt' if e['sys'] == 'SDB' else 'cap'
            if e[key] is None:
                continue
            prev = [x for x in E[:k] if x['sys'] == e['sys']]
            if len(prev) < 2 or any(x[key] is None for x in prev):
                continue
            # only lines that could be totals: on the reverse or the last numeral line
            if not (e['surf'] != 'obverse' or k == len(E) - 1):
                continue
            tot += 1
            if e[key] == sum(x[key] for x in prev):
                hits += 1
    return hits, tot


def test_C():
    h, n = running_sum_hits('M297')
    dec = decoys('M297', 40, 66005, lo=0.4, hi=2.5)
    dr = [(d,) + running_sum_hits(d) for d in dec]
    dr = [x for x in dr if x[2] > 0]
    rates = [a / b for _, a, b in dr]
    allh = sum(a for _, a, _ in dr); alln = sum(b for _, _, b in dr)
    RES['C'] = dict(h=h, n=n, decoys=dr)
    fails = n - h
    OUT.append(row('PE-66.1c', 'pe56 C "M297 heads closing totals equal to the running sum". Every M297-final line that is a candidate total (reverse or last numeral line, >= 2 earlier lines of the same system, all readable): equals the sum of all earlier same-system lines? Stated kill: M297 total lines that do not sum. Decoys: 40 signs of 0.4-2.5x frequency (seed 66005), same test',
                   'M297 candidate totals %d: sum %d, do not sum %d. Decoys with candidates %d: pooled %d/%d sum (median rate %.2f)' % (n, h, fails, len(dr), allh, alln, float(np.median(rates)) if rates else float('nan')),
                   'KILLED as stated: %d M297 total-position lines do not equal the running sum (%d do); rate %.2f vs decoy pooled %.2f' % (fails, h, h / max(1, n), allh / max(1, alln)) if fails > h else
                   'survives: %d of %d sum' % (h, n)))
    say(OUT[-1])


# ------------------------------------------------------------------ D  Yahya lexicon
def tabsigns(tid):
    return {base(s) for l in TAB[tid]['lines'] for s in signs_of(l)}


def test_D():
    SUSA = sorted(t['id'] for t in T if site(t) == 'Susa')
    YAH = sorted(t['id'] for t in T if site(t) == 'Yahya')
    TS = {i: tabsigns(i) for i in TAB}
    L = {i: sum(len(signs_of(l)) for l in TAB[i]['lines']) for i in TAB}
    cand = [s for s in TABN if len(TABN[s]) >= 3]

    def enrich(group, rest, signs):
        g = len(group); r = len(rest)
        out = {}
        for s in signs:
            a = sum(1 for i in group if s in TS[i]); b = sum(1 for i in rest if s in TS[i])
            out[s] = math.log((a + 0.5) / (g - a + 0.5)) - math.log((b + 0.5) / (r - b + 0.5))
        return out

    def held(group, rest, rng, top=4):
        g = list(group); rng.shuffle(g)
        A, B = g[:len(g) // 2], g[len(g) // 2:]
        eA = enrich(A, rest, cand)
        topk = sorted(cand, key=lambda s: -eA[s])[:top]
        eB = enrich(B, rest, topk)
        # presence on B: tablets of B carrying any top sign
        return float(np.mean([eB[s] for s in topk])), topk

    rng = np.random.default_rng(66006)
    real = [held(YAH, SUSA, rng) for _ in range(60)]
    real_m = float(np.mean([r[0] for r in real]))
    tops = collections.Counter(s for r in real for s in r[1])
    # pseudo-sites: 27 Susa tablets matched on sign count (nearest by rank), rest = remaining Susa
    srt = sorted(SUSA, key=lambda i: L[i])
    pseudo = []
    for rep in range(40):
        r2 = np.random.default_rng(66100 + rep)
        pick = set()
        for y in YAH:
            cands = [i for i in srt if i not in pick and abs(L[i] - L[y]) <= max(2, 0.2 * L[y])]
            if not cands:
                cands = [i for i in srt if i not in pick]
            pick.add(cands[r2.integers(len(cands))])
        rest = [i for i in SUSA if i not in pick]
        pseudo.append(float(np.mean([held(sorted(pick), rest, r2)[0] for _ in range(6)])))
    p_site = pval_ge(real_m, pseudo)
    # the named set on Yahya vs Susa, and Susa avoidance
    named = ['M056', 'M044', 'M219', 'M136']
    e = enrich(YAH, SUSA, named)
    def cooc(sset, ids):
        return sum(1 for i in ids if len(TS[i] & set(sset)) >= 2)
    obs_c = cooc(named, SUSA)
    # null: place each sign's Susa tablets at random among Susa tablets, prob ∝ distinct sign types
    w = np.array([len(TS[i]) for i in SUSA], float); w /= w.sum()
    def null_cooc(sset, nrep=2000, seed=66007):
        r3 = np.random.default_rng(seed)
        cnts = [sum(1 for i in SUSA if s in TS[i]) for s in sset]
        out = []
        for _ in range(nrep):
            M = np.zeros(len(SUSA), int)
            for c in cnts:
                M[r3.choice(len(SUSA), size=c, replace=False, p=w)] += 1
            out.append(int((M >= 2).sum()))
        return np.asarray(out)
    nc = null_cooc(named)
    p_av = float((1 + np.sum(nc <= obs_c)) / (1 + len(nc)))
    # decoy 4-sets matched per sign on Susa frequency
    dav = []
    for rep in range(30):
        ds = [decoys(s, 1, 66200 + 10 * rep + j, exclude=named)[0] for j, s in enumerate(named)]
        o = cooc(ds, SUSA); n2 = null_cooc(ds, 400, 66300 + rep)
        dav.append(float((1 + np.sum(n2 <= o)) / (1 + len(n2))))
    d_av_pass = sum(1 for x in dav if x <= p_av)
    RES['D'] = dict(real=real_m, pseudo=pseudo, p_site=p_site, tops=tops.most_common(10), enrich=e,
                    obs_c=obs_c, null_c=float(nc.mean()), p_av=p_av, decoy_av=dav)
    OUT.append(row('PE-66.1d', 'pe48 C "Yahya has its own small lexicon M056/M044/M219/M136; at Susa they avoid each other". Frozen kill needs 40 new Yahya tablets (none exist). Strongest now: (i) SITE HELD-OUT: pick the 4 most Yahya-enriched signs on a random half of the 27 Yahya tablets, score their enrichment on the other half (60 splits, seed 66006); DECOY SITES: 40 pseudo-sites of 27 Susa tablets matched on sign count (seeds 66100+), same pipeline. (ii) Susa avoidance (Susa tablets carrying >= 2 of the 4) vs a null placing each sign on Susa tablets in proportion to their sign-type count (2,000x); 30 decoy 4-sets matched per sign on frequency',
                   '(i) held-out log-odds enrichment of half-selected signs: Yahya %.2f vs pseudo-sites mean %.2f (p %.3f); signs most often selected: %s. Named set log-odds vs Susa: %s. (ii) Susa tablets with >= 2 of the 4: %d vs null %.1f (p %.3f); decoy 4-sets at least as avoidant: %d/30' % (
                       real_m, float(np.mean(pseudo)), p_site, ', '.join('%s %d' % x for x in tops.most_common(6)),
                       ', '.join('%s %.1f' % (k, v) for k, v in e.items()), obs_c, float(nc.mean()), p_av, d_av_pass),
                   'site lexicon %s; Susa avoidance %s' % ('replicates on held-out Yahya tablets beyond pseudo-sites' if p_site < 0.05 else 'does NOT beat size-matched pseudo-sites',
                                                          'survives (beyond decoys)' if p_av < 0.05 and d_av_pass <= 3 else 'fails / not beyond decoys')))
    say(OUT[-1])


# ------------------------------------------------------------------ F  M009 : M102 = 1 : 2
def ratio_hits(a, b, ratios=(Fr(2),)):
    """tablets where both a and b head clean same-system entries; hit if mean(b)/mean(a) in ratios."""
    tabs = []
    for tid, E in ENT.items():
        for sy, key in (('SDB', 'cnt'), ('C', 'cap')):
            va = [e[key] for e in E if e['final'] == a and e['sys'] == sy and e[key]]
            vb = [e[key] for e in E if e['final'] == b and e['sys'] == sy and e[key]]
            if va and vb:
                pool = [e[key] for e in E if e['sys'] == sy and e[key]]
                tabs.append((tid, va, vb, pool))
    return tabs


def r_of(va, vb):
    return Fr(sum(vb), len(vb)) / Fr(sum(va), len(va))


def test_F():
    rng = np.random.default_rng(66008)
    tabs = ratio_hits('M009', 'M102')
    obs = sum(1 for _, va, vb, _ in tabs if r_of(va, vb) == 2)
    def null(tabs, n=2000):
        out = []
        for _ in range(n):
            h = 0
            for _, va, vb, pool in tabs:
                x = rng.choice(pool, size=len(va) + len(vb))
                h += r_of(list(x[:len(va)]), list(x[len(va):])) == 2
            out.append(h)
        return np.asarray(out)
    nl = null(tabs)
    p = pval_ge(obs, nl)
    other = {str(r): sum(1 for _, va, vb, _ in tabs if r_of(va, vb) == r) for r in (Fr(1, 2), Fr(1), Fr(3, 2), Fr(3), Fr(4))}
    # decoy pairs
    dp = []
    for j in range(30):
        a = decoys('M009', 1, 66400 + j, exclude=['M102'])[0]
        b = decoys('M102', 1, 66500 + j, exclude=['M009', a])[0]
        tb = ratio_hits(a, b)
        if len(tb) >= 3:
            o = sum(1 for _, va, vb, _ in tb if r_of(va, vb) == 2)
            dp.append((a, b, len(tb), o, pval_ge(o, null(tb, 300))))
    dpass = sum(1 for x in dp if x[4] <= p)
    RES['F'] = dict(n=len(tabs), obs=obs, null=float(nl.mean()), p=p, other=other, decoys=dp)
    OUT.append(row('PE-66.1f', 'pe58 C- "M009 : M102 = 1 : 2 within tablets (n 6)". Tablets where both head readable entries of one system; hit = mean M102 / mean M009 exactly 2. Null: the two signs\' entries replaced by random entries of the same tablet and system (2,000x, seed 66008). Decoy ratios 1/2, 1, 3/2, 3, 4 on the same tablets; 30 decoy pairs matched on both frequencies (seeds 66400/66500), same test',
                   'tablets %d; ratio exactly 2: %d vs null %.2f (p %.3f). Other ratios: %s. Decoy pairs with >= 3 tablets: %d; as or more significant: %d' % (
                       len(tabs), obs, float(nl.mean()), p, other, len(dp), dpass),
                   'survives (C-): 1:2 beats the within-tablet null and decoys' if p < 0.05 and dpass <= 0.1 * max(1, len(dp)) else 'KILLED: 1:2 is not beyond the within-tablet null / decoy pairs'))
    say(OUT[-1])


# ------------------------------------------------------------------ G  48 N39C per counted unit
DISC = set(['P008796', 'P008797', 'P008798', 'P008799', 'P008800', 'P008801', 'P008802'])


def count_cap_pairs(exclude=()):
    P = []
    for tid, E in ENT.items():
        if tid in exclude:
            continue
        for k in range(len(E) - 1):
            a, b = E[k], E[k + 1]
            if b['i'] != a['i'] + 1:
                continue
            if a['sys'] == 'SDB' and a['cnt'] and b['sys'] == 'C' and b['cap'] and a['cnt'] == int(a['cnt']):
                P.append((tid, a['final'], b['final'], int(a['cnt']), b['cap']))
    return P


def test_G():
    allP = count_cap_pairs()
    disc = DISC | {tid for tid, fa, fb, n, m in allP if fa == 'M056' and fb == 'M288'}
    P = count_cap_pairs(disc)
    rng = np.random.default_rng(66009)
    n = np.array([x[3] for x in P]); m = np.array([x[4] for x in P])
    rates = list(range(20, 121))
    obs = {r: int(np.sum(m == r * n)) for r in rates}
    NN = 5000
    null = {r: [] for r in rates}
    for _ in range(NN):
        mp = rng.permutation(m)
        for r in rates:
            null[r].append(int(np.sum(mp == r * n)))
    ex = {r: (obs[r] + 0.5) / (np.mean(null[r]) + 0.5) for r in rates}
    p48 = pval_ge(obs[48], null[48])
    rk = sorted(rates, key=lambda r: -ex[r])
    rank48 = rk.index(48) + 1
    # single-unit and multi-unit separately (n >= 2 is the informative part)
    mm = n >= 2
    obs2 = int(np.sum(m[mm] == 48 * n[mm]))
    nl2 = [int(np.sum(rng.permutation(m[mm]) == 48 * n[mm])) for _ in range(NN)]
    p2 = pval_ge(obs2, nl2)
    # by following sign
    fb = collections.Counter(x[2] for x in P if x[4] == 48 * x[3])
    RES['G'] = dict(nP=len(P), disc=sorted(disc), obs=obs, ex=ex, p48=p48, rank48=rank48, obs2=obs2, p2=p2, fb=fb.most_common())
    OUT.append(row('PE-66.1g', 'pe27 C / pe63 C+ "48 N39C per counted unit" (M056 -> M288 2(N39B); dossier D2). All adjacent pairs count line (n) -> capacity line (m N39C), with the discovery tablets removed (D2 P008796-802 and every M056 -> M288 tablet: %d). Null: m permuted among pairs (5,000x, seed 66009). Decoy rates: every integer 20-120 N39C per unit, ranked by obs/null' % len(disc),
                   'pairs %d. rate 48: %d hits vs null %.2f (p %.3f), rank %d of 101 rates by excess; top rates %s. With n >= 2 only: %d hits (p %.3f). Following signs of 48-hits: %s' % (
                       len(P), obs[48], float(np.mean(null[48])), p48, rank48,
                       ', '.join('%d (%d vs %.1f)' % (r, obs[r], np.mean(null[r])) for r in rk[:6]), obs2, p2, fb.most_common(6)),
                   'survives outside its discovery tablets' if p48 < 0.01 and rank48 <= 5 and obs2 >= 2 else
                   ('weak: single-unit coincidences only' if p48 < 0.05 else 'KILLED outside the discovery tablets: 48 per unit is not beyond the permutation null / decoy rates')))
    say(OUT[-1])


# ------------------------------------------------------------------ H  [M036+1(N30D)] = 5 N01 per unit
def test_H():
    S = '|M036+1(N30D)|'
    P = []
    for tid, E in ENT.items():
        for k in range(1, len(E)):
            a, b = E[k - 1], E[k]
            if b['final'] == S and b['i'] == a['i'] + 1 and a['sys'] == 'SDB' and a['cnt'] and b['sys'] == 'SDB' and b['cnt']:
                P.append((tid, a['final'], int(a['cnt']), b['cnt']))
    multi = [x for x in P if 2 <= x[2] <= 4]
    hits = [x for x in multi if x[3] == 5 * x[2]]
    oth = [x for x in multi if x[3] != 5 * x[2]]
    single = [x for x in P if x[2] == 1]
    s5 = sum(1 for x in single if x[3] == 5)
    # decoy: same pair rule for other final signs following counts, rate 5 per unit with n 2-4
    dec = decoys(S, 40, 66010, counter=VTOK, lo=0.4, hi=2.5)
    dr = []
    for d in dec:
        Q = [(e1['cnt'], e2['cnt']) for E in ENT.values() for e1, e2 in zip(E, E[1:]) if e2['signs'][-1] == d
             and e2['i'] == e1['i'] + 1 and e1['sys'] == 'SDB' and e2['sys'] == 'SDB' and e1['cnt'] and e2['cnt'] and 2 <= e1['cnt'] <= 4]
        if Q:
            dr.append((d, len(Q), sum(1 for a, b in Q if b == 5 * a)))
    RES['H'] = dict(P=P, multi=multi, hits=hits, decoys=dr)
    OUT.append(row('PE-66.1h', 'pe28 C "[M036+1(N30D)] = 5(N01) per M263-type unit". Every [M036+1(N30D)] line directly after a count line (both readable counts). Stated support: 10-20 after 2-4 units; kill: other multiples. Decoys: 40 signs of matched frequency (seed 66010), share at 5 per unit after 2-4 units',
                   'pairs %d; after 1 unit: %d of %d are 5; after 2-4 units: %d lines, 5x: %d (%s), other: %d (%s). Decoys with 2-4-unit pairs: %d, pooled 5x hits %d of %d' % (
                       len(P), s5, len(single), len(multi), len(hits), [(x[0], x[2], x[3]) for x in hits], len(oth), [(x[0], x[2], x[3]) for x in oth],
                       len(dr), sum(x[2] for x in dr), sum(x[1] for x in dr)),
                   ('KILLED as stated: other multiples after 2-4 units' if len(oth) > len(hits) else
                    ('survives the stated test' if hits else 'untestable-now: no line after 2-4 units'))))
    say(OUT[-1])


# ------------------------------------------------------------------ L  [M327+M342] header
def header_share(sign, tids=None):
    h = n = 0
    for t in T:
        if tids is not None and t['id'] not in tids:
            continue
        hd = header(t)
        for li, l in enumerate(t['lines']):
            for s in signs_of(l):
                if base(s) == sign:
                    n += 1
                    if hd and li == next(k for k, x in enumerate(t['lines']) if x['surface'] == 'obverse'):
                        h += 1
    return h, n


def test_L():
    S = '|M327+M342|'
    res = []
    for sd in (66011, 66012, 66013):
        B = set(TAB) - half(sd)
        h, n = header_share(S, B)
        dec = decoys(S, 40, sd + 100)
        ds = [header_share(d, B) for d in dec]
        ds = [a / b for a, b in ds if b >= 3]
        res.append((h, n, h / max(1, n), float(np.mean(ds)), sum(1 for x in ds if x >= h / max(1, n)), len(ds)))
    hall, nall = header_share(S)
    # order null: header position vs uniform line position on the same tablets
    exp = 0
    for t in T:
        k = sum(1 for l in t['lines'] for s in signs_of(l) if base(s) == S)
        if k:
            nl = sum(1 for l in t['lines'] if signs_of(l))
            exp += k / max(1, nl)
    RES['L'] = dict(res=res, hall=hall, nall=nall, exp=exp)
    OUT.append(row('PE-66.1l', 'pe15/pe38/pe41 C "[M327+M342] is a header (tablet-type / office) sign". Share of its tokens in the unnumbered first obverse line, on 3 fresh held-out halves (seeds 66011-13); decoys: 40 signs of 0.67-1.5x frequency per half; line-order null = expected share if its line were placed at random among the tablet\'s lines',
                   'all data: %d of %d tokens in the header line (line-order null %.1f). Held-out halves: %s; decoys with an equal or higher header share: %s' % (
                       hall, nall, exp, ', '.join('%d/%d = %.2f (decoy mean %.2f)' % (a, b, c, d) for a, b, c, d, _, _ in res),
                       ', '.join('%d/%d' % (e, f) for *_, e, f in res)),
                   'survives: header share far above decoys and the order null in every half' if all(r[2] > 0.5 and r[4] <= 2 for r in res) else 'KILLED / weak: not header-specific beyond decoys'))
    say(OUT[-1])


# ------------------------------------------------------------------ O  grain-office signs
def cshare(sign, tids=None):
    c = n = 0
    for tid, E in ENT.items():
        if tids is not None and tid not in tids:
            continue
        for e in E:
            if sign in e['bsigns']:
                n += 1
                c += e['sys'] in ('C', 'C*')
    return c, n


def test_O():
    G = ['M081', 'M265', 'M266', 'M296', 'M112', 'M286', 'M248']
    out = []
    for s in G:
        c, n = cshare(s)
        B = set(TAB) - half(66014)
        cb, nb = cshare(s, B)
        dec = decoys(s, 30, 66015, exclude=G)
        dsh = [cshare(d) for d in dec]
        dpass = sum(1 for a, b in dsh if b and a / b >= 0.3)
        out.append((s, c, n, cb, nb, dpass, sum(1 for a, b in dsh if b)))
    base_c = sum(e['sys'] in ('C', 'C*') for E in ENT.values() for e in E) / sum(len(E) for E in ENT.values())
    RES['O'] = out
    kill = [s for s, c, n, cb, nb, _, _ in out if n and c / n < 0.3]
    surv = [s for s, c, n, cb, nb, dp, dn in out if n >= 3 and c / n >= 0.3 and (nb == 0 or cb / nb >= 0.3) and dp / max(1, dn) < 0.2]
    OUT.append(row('PE-66.1o', 'pe15 C "new grain-office signs M081 M265 M266 M296 M112 M286 M248". Capacity share of entries containing each sign, all data and the held-out half (seed 66014). Stated kill: < 30%% capacity. Decoys: 30 signs of matched frequency per sign (seed 66015), share >= 30%%; base capacity share %.2f' % base_c,
                   '; '.join('%s %d/%d (half %d/%d; decoys >= 30%%: %d/%d)' % x for x in out),
                   'survive: %s; killed (< 30%%): %s; the rest pass but not beyond decoys or n < 3' % (' '.join(surv) or 'none', ' '.join(kill) or 'none')))
    say(OUT[-1])


if __name__ == '__main__':
    which = sys.argv[1:] or list('ABCDFGHLO')
    for w in which:
        globals()['test_' + w]()
    tag = ''.join(which)
    dump(RES, 'c1_%s.json' % tag)
    open(os.path.join(CK, 'c1_%s.rows' % tag), 'w').write('\n'.join(OUT) + '\n')
