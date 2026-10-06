#!/usr/bin/env python3
"""LA-67 cycle 1: kill tests for the grade-C guesses the sweep was asked about first (cheap, exact
statistics).  Each test: stated would-kill line on existing data where it can be checked literally, plus
the strongest existing-data test (fresh seeds, matched decoys, held-out tablet halves, permutation nulls).
Decoys go through the identical criterion; their survival share is the false-survival rate."""
from la67_lib import *

OUT = os.path.join(LOOPS, 'la67_cycle1.txt')
TABLETLIKE = {'Tablet', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar', 'Label'}
NH = 20
RES = {}


def tablets(D):
    return [d for d in D if d['support'] in TABLETLIKE]


# ---------------------------------------------------------------- headings (la40)
def head_score(D, w):
    hit = n = 0
    for d in D:
        L = lines(d)
        for li, ln in enumerate(L):
            for j, t in enumerate(ln):
                if t[0] in 'WL' and t[1] == w:
                    n += 1
                    nxt = ln[j + 1] if j + 1 < len(ln) else None
                    nxt2 = ln[j + 2] if j + 2 < len(ln) else None
                    counted = (nxt is not None and nxt[0] == 'N') or (nxt is not None and nxt[0] == 'L' and nxt2 is not None and nxt2[0] == 'N')
                    hit += (li == 0) and not counted
    return (hit / n if n else None), n


def t_headings(D):
    T = tablets(D)
    cnt = counts(T, 'W')
    out = {}
    for w in ['KU-RE', 'U', 'A', 'DA', 'KA-NA']:
        pool = [x for x in cnt if is_single(x) == is_single(w) and cnt[x] >= 2]
        dec = matched_decoys(cnt, w, pool, 40, exclude={'KU-RE', 'U', 'A', 'DA', 'KA-NA'})
        s, n = head_score(T, w)
        ds = [head_score(T, x)[0] for x in dec]
        p = pct_rank(s, ds)
        # halves: held-out replication in both halves
        rep = 0; tested = 0; drep = Counter(); dtest = Counter()
        for k in range(NH):
            A, B = halves(T, k)
            sa = [head_score(A, x)[0] for x in [w] + dec]; sb = [head_score(B, x)[0] for x in [w] + dec]
            for i, x in enumerate([w] + dec):
                if sa[i] is None or sb[i] is None:
                    continue
                refa = [v for jj, v in enumerate(sa) if jj not in (0, i) and v is not None]
                refb = [v for jj, v in enumerate(sb) if jj not in (0, i) and v is not None]
                ok = pct_rank(sa[i], refa) <= 0.1 and pct_rank(sb[i], refb) <= 0.1
                if i == 0:
                    tested += 1; rep += ok
                else:
                    dtest[x] += 1; drep[x] += ok
        dsurv = []
        for x in dec:
            px = pct_rank(head_score(T, x)[0], [v for y, v in zip(dec, ds) if y != x])
            r = drep[x] / dtest[x] if dtest[x] else 0
            dsurv.append(px <= 0.05 and dtest[x] >= 5 and r >= 0.5)
        surv = p <= 0.05 and tested >= 5 and rep / tested >= 0.5
        out[w] = dict(n=n, score=round(s, 2), dec_mean=round(float(np.mean([x for x in ds if x is not None])), 2), p=round(p, 3),
                      halves='%d/%d' % (rep, tested), dec_false=round(float(np.mean(dsurv)), 2), survive=bool(surv))
    RES['headings'] = out
    row = ('| LA-67.1a | la40 C: KU-RE, U, A, DA, KA-NA are heading words. Kill line: counted entry heads. Test on tablets: heading score = share of '
           'occurrences in the first line AND not followed by a number (directly or after one logogram); vs 40 frequency-matched decoys of the same kind '
           '(single sign / multi-sign); held-out: in %d fresh tablet halves, top 10%% of decoys in BOTH halves. Survive = P <= 0.05 and replication >= 0.5 of '
           'testable halves (>= 5). Same rule applied to each decoy | %s | %s |' % (
               NH, '; '.join('%s n %d score %.2f (decoys %.2f) P %.3f halves %s decoy false-survival %.2f' % (
                   w, v['n'], v['score'], v['dec_mean'], v['p'], v['halves'], v['dec_false']) for w, v in out.items()),
               '; '.join('%s %s' % (w, 'SURVIVES' if v['survive'] else 'KILLED' if v['n'] >= 3 else 'KILLED (n<3, cannot beat decoys)') for w, v in out.items())))
    wlog(OUT, row)


# ---------------------------------------------------------------- OLE+U/MI/DI entries (la45)
def after_word_share(D, x):
    a = n = 0
    for d in D:
        for ln in lines(d):
            for j, t in enumerate(ln):
                if t[0] == 'L' and t[1] == x:
                    n += 1; a += j > 0 and ln[j - 1][0] == 'W'
    return (a / n if n else None), n


def t_ole(D):
    A_ = admin(D)
    cnt = counts(A_, 'L')
    tg = ['OLE+U', 'OLE+MI', 'OLE+DI']
    pool = [x for x in cnt if cnt[x] >= 5 and x not in tg]
    ref = {x: after_word_share(A_, x)[0] for x in pool}
    out = {}
    for w in tg:
        s, n = after_word_share(A_, w)
        p = (np.sum(np.array(list(ref.values())) <= s) + 1) / (len(ref) + 1)
        # held-out halves: target in lowest 25% of decoy logograms in both halves
        rep = tested = 0
        for k in range(NH):
            H = halves(A_, k)
            ok = True; have = True
            for h in H:
                sh, nh = after_word_share(h, w)
                if not nh:
                    have = False; break
                r = [after_word_share(h, x)[0] for x in pool]
                r = [v for v in r if v is not None]
                ok &= (np.sum(np.array(r) <= sh) + 1) / (len(r) + 1) <= 0.25
            if have:
                tested += 1; rep += ok
        out[w] = dict(n=n, share=round(s, 2), p=round(float(p), 3), halves='%d/%d' % (rep, tested), lit=int(round(s * n)))
    vals = list(ref.values())
    lowfalse = float(np.mean([(np.sum(np.array(vals[:i] + vals[i + 1:]) <= v) + 1) / len(vals) <= 0.05 for i, v in enumerate(vals)]))
    RES['ole'] = out
    row = ('| LA-67.1b | la45 C: OLE+U, OLE+MI, OLE+DI act as entries, not commodities. Kill line: they appear in commodity slots after a word. Literal check on '
           'administrative documents + share of occurrences directly after a word on the same line, vs %d decoy logograms (n >= 5; mean share %.2f); P = share of '
           'decoys as low or lower; %d fresh tablet halves (lowest quarter in both halves). Decoy false-survival at P <= 0.05: %.2f | %s | %s |' % (
               len(pool), np.mean(vals), NH, lowfalse,
               '; '.join('%s n %d after-word %d (%.2f) P %.3f halves %s' % (w, v['n'], v['lit'], v['share'], v['p'], v['halves']) for w, v in out.items()),
               '; '.join('%s %s' % (w, 'KILLED (literal: %d commodity-slot uses after a word already in the corpus, e.g. HT 2 A-KA-RU OLE+U 20)' % v['lit'] if v['lit'] > 0 and v['p'] > 0.05
                                     else 'survives literal; low after-word share P %.3f' % v['p'] if v['lit'] == 0 else 'mixed: %d after-word uses but rarer than decoys (P %.3f)' % (v['lit'], v['p']))
                         for w, v in out.items())))
    wlog(OUT, row)


# ---------------------------------------------------------------- *308 fraction-bound (la63)
def numbered(D, x):
    out = []
    for i, d in enumerate(D):
        for ln in lines(d):
            for j, t in enumerate(ln):
                if t[0] in 'WL' and t[1] == x and j + 1 < len(ln) and ln[j + 1][0] == 'N':
                    out.append((i, ln[j + 1]))
    return out


def t_308(D):
    A_ = admin(D)
    cnt = counts(A_)
    def frac_share(docs, x):
        e = numbered(docs, x)
        return (sum(1 for _, t in e if t[2]) / len(e) if e else None), len(e), sum(1 for _, t in e if not t[2] and t[1] > 0)
    s, n, bare = frac_share(A_, '*308')
    pool = [x for x in cnt if len(numbered(A_, x)) >= 4 and x != '*308']
    ref = [frac_share(A_, x)[0] for x in pool]
    p_dec = pct_rank(s, ref)
    # within-document number shuffle (keeps each document's fraction density)
    rng = random.Random(seed('308-perm'))
    def perm_docs():
        P_ = []
        for d in A_:
            nums = [t for t in d['toks'] if t[0] == 'N']; rng.shuffle(nums)
            it = iter(nums)
            P_.append(dict(d, toks=[next(it) if t[0] == 'N' else t for t in d['toks']]))
        return P_
    null = {x: [] for x in ['*308'] + pool}
    for _ in range(1000):
        P_ = perm_docs()
        for x in null:
            null[x].append(frac_share(P_, x)[0])
    p_perm = pct_rank(s, null['*308'])
    obs = dict(zip(pool, ref))
    dfalse = float(np.mean([pct_rank(obs[x], null[x]) <= 0.05 and pct_rank(obs[x], [v for y, v in obs.items() if y != x]) <= 0.05 for x in pool]))
    RES['308'] = dict(n=n, share=s, bare=bare, p_dec=p_dec, p_perm=p_perm, dfalse=dfalse)
    surv = bare == 0 and p_perm <= 0.05 and p_dec <= 0.05
    row = ('| LA-67.1c | la63 C: *308 is a fraction-bound commodity sign. Kill line: *308 with a bare integer. Literal check on all administrative documents; '
           'fraction share of its numbers vs %d decoy signs/logograms with >= 4 numbered entries; 1,000 within-document number shuffles (fresh seed) | '
           '*308 numbered %d, with fraction %d (%.2f), bare integer %d. Decoys: mean share %.2f, P %.3f; within-document shuffle P %.3f. Decoy false-survival (both tests) %.2f | %s |' % (
               len(pool), n, int(round(s * n)), s, bare, np.nanmean([r for r in ref if r is not None]), p_dec, p_perm, dfalse,
               'SURVIVES (no bare integer; fraction-bound beyond decoys and beyond its documents)' if surv else
               ('KILLED (bare integer present)' if bare else 'WEAK: no bare integer, but the fraction binding is explained by its documents or matched by decoys')))
    wlog(OUT, row)


# ---------------------------------------------------------------- SI / NI / TA2 commodity links (la63)
def doc_sets(D):
    return [(d['site'], set(t[1] for t in d['toks'] if t[0] == 'W'), set(re.sub(r'\+.*', '', t[1]) for t in d['toks'] if t[0] == 'L')) for d in D]


def lift_test(S, w, c, rng, nperm=500):
    """site-stratified: share of docs with w that contain commodity c, vs w's docs re-dealt within site."""
    idx = [i for i, s in enumerate(S) if w in s[1]]
    if not idx:
        return None, None, 0
    obs = sum(c in S[i][2] for i in idx) / len(idx)
    bysite = defaultdict(list)
    for i, s in enumerate(S):
        bysite[s[0]].append(i)
    null = []
    for _ in range(nperm):
        k = 0
        for st, m in Counter(S[i][0] for i in idx).items():
            k += sum(c in S[j][2] for j in rng.sample(bysite[st], m))
        null.append(k / len(idx))
    return obs, pct_rank(obs, null), len(idx)


def best_com(S, w):
    c = Counter(x for s in S if w in s[1] for x in s[2])
    return c.most_common(1)[0][0] if c else None


def t_sicom(D):
    A_ = [d for d in admin(D) if any(t[0] == 'L' for t in d['toks'])]
    S = doc_sets(A_)
    cnt = Counter(w for s in S for w in s[1])
    pool = [w for w in cnt if is_single(w) and cnt[w] >= 3]
    rng = random.Random(seed('sicom'))
    out = {}
    tg = {'SI': 'CYP', 'NI': None, 'TA₂': None}
    dec_all = set()
    res_items = {}
    for w in list(tg) + sorted(set(x for t in tg for x in matched_decoys(cnt, t, pool, 15, exclude=set(tg)))):
        if w in res_items:
            continue
        rep = tested = 0
        for k in range(NH):
            Ah, Bh = halves(A_, k)
            SA, SB = doc_sets(Ah), doc_sets(Bh)
            c = tg.get(w) or best_com(SA, w)
            if c is None or not any(w in s[1] for s in SB):
                continue
            o, p, n = lift_test(SB, w, c, rng, 200)
            tested += 1; rep += p <= 0.05
        o, p, n = lift_test(S, w, tg.get(w) or best_com(S, w), rng, 1000)
        res_items[w] = dict(n=n, com=tg.get(w) or best_com(S, w), share=o, p=p, rep=rep, tested=tested,
                            surv=bool(tested >= 5 and rep / tested >= 0.5))
    decs = [w for w in res_items if w not in tg]
    dfalse = float(np.mean([res_items[w]['surv'] for w in decs]))
    RES['sicom'] = res_items
    row = ('| LA-67.1d | la63 C: SI (with CYP), NI and TA2 predict the commodity logogram. Kill line (SI): SI heading non-CYP lists as often. Test on administrative documents '
           'with a logogram: share of the sign\'s documents holding the commodity (SI: CYP, fixed in advance; NI, TA2: commodity chosen on half A) vs the sign\'s documents '
           're-dealt within site; held-out = P <= 0.05 on half B in >= 0.5 of %d fresh halves. %d frequency-matched single-sign decoys through the same rule | %s; decoy false-survival %.2f | %s |' % (
               NH, len(decs), '; '.join('%s n %d %s share %.2f P %.3f held-out %d/%d' % (w, v['n'], v['com'], v['share'], v['p'], v['rep'], v['tested']) for w, v in res_items.items() if w in tg),
               dfalse, '; '.join('%s %s' % (w, 'SURVIVES' if res_items[w]['surv'] else 'KILLED') for w in tg)))
    wlog(OUT, row)


# ---------------------------------------------------------------- TA-I as AROM (la60/la61)
def adj_logo(D):
    """(word, logogram, tablet) for a word followed directly by a logogram on the same line."""
    out = []
    for d in D:
        for ln in lines(d):
            for j in range(len(ln) - 1):
                if ln[j][0] == 'W' and ln[j + 1][0] == 'L':
                    out.append((ln[j][1], re.sub(r'\+.*', '', ln[j + 1][1]), d['tab'], d['site']))
    return out


def t_tai(D):
    A_ = admin(D)
    E = adj_logo(A_)
    rng = random.Random(seed('tai'))
    def stat(E, w):
        tabs = defaultdict(set)
        for x, l, t, s in E:
            if x == w:
                tabs[l].add(t)
        if not tabs:
            return 0, None
        l = max(tabs, key=lambda k: (len(tabs[k]), k))
        return len(tabs[l]), l
    words = Counter(x for x, _, t, _ in set((x, l, t, s) for x, l, t, s in E))
    cand = [w for w in words if len(set(t for x, _, t, _ in E if x == w)) >= 2]
    obs = {w: stat(E, w) for w in cand}
    bysite = defaultdict(list)
    for k, e in enumerate(E):
        bysite[e[3]].append(k)
    NULL = defaultdict(list)
    for _ in range(1000):
        L = [e[1] for e in E]
        for st, ks in bysite.items():
            v = [L[k] for k in ks]; rng.shuffle(v)
            for k, x in zip(ks, v):
                L[k] = x
        Ep = [(e[0], L[k], e[2], e[3]) for k, e in enumerate(E)]
        for w in cand:
            tabs = set(e[2] for e in Ep if e[0] == w and e[1] == obs[w][1])
            NULL[w].append(len(tabs))
    P = {w: pct_rank(obs[w][0], NULL[w]) for w in cand}
    dec = [w for w in cand if w != 'TA-I']
    dfalse = float(np.mean([P[w] <= 0.05 for w in dec]))
    RES['tai'] = dict(obs=obs.get('TA-I'), p=P.get('TA-I'), dfalse=dfalse, ndec=len(dec),
                      sig=sorted([(w, obs[w], round(P[w], 3)) for w in dec if P[w] <= 0.05], key=lambda z: z[2])[:8])
    o = obs.get('TA-I')
    row = ('| LA-67.1e | la60/la61 C: TA-I goes with AROM. Kill (la60 line): < 1 in 3 right on new uses. Existing-data test: number of distinct tablets on which the word '
           'stands directly before its modal logogram vs 1,000 site-stratified shuffles of the logograms among all word->logogram slots (fresh seed); decoys = all %d words '
           'with word->logogram slots on >= 2 tablets, same test | TA-I before %s on %d tablets (HT 9 both sides, HT 39), P %.3f; decoys passing P <= 0.05: %.2f (%s) | %s |' % (
               len(dec), o[1], o[0], P['TA-I'], dfalse, ', '.join('%s-%s %d' % (w, x[1], x[0]) for w, x, _ in RES['tai']['sig']),
               'SURVIVES the shuffle (2 tablets; HT 9a/9b count once), but the decoy rate shows how easily a word-logogram pair on 2 tablets passes' if P['TA-I'] <= 0.05 else 'KILLED'))
    wlog(OUT, row)


# ---------------------------------------------------------------- Zakros template words (la65)
def t_zatpl(D):
    tabs = defaultdict(set)
    for d in D:
        for t in d['toks']:
            if t[0] == 'W':
                tabs[t[1]].add(d['tab'])
    site = {d['tab']: d['site'] for d in D}
    lit = {w: sorted(tabs[w]) for w in ['*28B-NU-MA-RE', 'SI-PI-KI']}
    # pairs of words (>= 3 tablets each) with identical tablet sets: real vs site-stratified curveball-like re-dealing
    W3 = [w for w in tabs if len(tabs[w]) >= 3 and '-' in w]
    def ident_pairs(T):
        by = Counter(frozenset(T[w]) for w in W3)
        return sum(c * (c - 1) // 2 for c in by.values())
    obs = ident_pairs(tabs)
    rng = random.Random(seed('zatpl'))
    st_tabs = defaultdict(list)
    for t, s in site.items():
        st_tabs[s].append(t)
    null = []
    for _ in range(2000):
        T = {}
        for w in W3:
            ss = Counter(site[t] for t in tabs[w])
            T[w] = set(x for s, m in ss.items() for x in rng.sample(st_tabs[s], m))
        null.append(ident_pairs(T))
    p = pct_rank(obs, null)
    pairs = [(a, b) for i, a in enumerate(W3) for b in W3[i + 1:] if tabs[a] == tabs[b]]
    RES['zatpl'] = dict(lit=lit, obs=obs, null=float(np.mean(null)), p=p, pairs=pairs)
    off = [t for w in lit for t in lit[w] if t not in ('ZA4', 'ZA5', 'ZA15')]
    row = ('| LA-67.1f | la65 C: *28B-NU-MA-RE and SI-PI-KI are fixed line words of the one Zakros wine template. Kill line: either word on a tablet of another '
           'template or commodity. Literal check over ALL documents; decoy test: pairs of multi-sign words (on >= 3 tablets) with identical tablet sets, real vs 2,000 '
           're-dealings of each word\'s tablets within site (fresh seed) | tablets: %s; off-template uses %d. Identical-set pairs %d (%s) vs null %.2f, P %.4f | %s |' % (
               '; '.join('%s %s' % (w, ','.join(v)) for w, v in lit.items()), len(off), obs, '; '.join('%s=%s' % p_ for p_ in pairs[:4]), np.mean(null), p,
               'SURVIVES (no off-template use; co-fixed pair beyond chance)' if not off and p <= 0.05 else 'KILLED' if off else 'SURVIVES literal only (pair not beyond chance)'))
    wlog(OUT, row)


# ---------------------------------------------------------------- HT95b grain (la54)
def t_ht95b(D):
    A_ = admin(D)
    S = [(d['tab'], d['id'], set(t[1] for t in d['toks'] if t[0] == 'W'), Counter(re.sub(r'\+.*', '', t[1]) for t in d['toks'] if t[0] == 'L')) for d in A_]
    def predict(i):
        """commodity vote from the words of doc i, using only OTHER tablets (both sides of doc i's tablet removed)."""
        tab, _, ws, _ = S[i]
        v = Counter()
        for t, _, w2, L in S:
            if t == tab or not L:
                continue
            sh = len(ws & w2)
            if sh:
                for c in L:
                    v[c] += sh / len(L)
        if not v:
            return None, 0
        c, x = v.most_common(1)[0]
        return c, x / sum(v.values())
    i95 = [i for i, s in enumerate(S) if s[1] == 'HT95b'][0]
    c95, conf95 = predict(i95)
    # calibration on documents whose commodity is known and single (hidden): accuracy at >= this confidence
    hits = []
    for i, s in enumerate(S):
        if len(s[3]) == 1 and s[1] != 'HT95b':
            c, conf = predict(i)
            if c is not None:
                hits.append((conf, c == next(iter(s[3]))))
    hi = [h for cf, h in hits if cf >= conf95 - 1e-9]
    base = Counter(next(iter(s[3])) for s in S if len(s[3]) == 1)
    RES['ht95b'] = dict(c=c95, conf=conf95, acc_hi=float(np.mean(hi)) if hi else None, n_hi=len(hi), n_cal=len(hits))
    row = ('| LA-67.1g | la54 C: HT 95b is a grain document. Kill line: a non-grain sign there (HT 95b has no logogram, so the literal test is empty). Existing-data test: '
           'commodity voted by the words of HT 95b from OTHER tablets only (HT 95a removed); calibration = the same vote on %d documents with one known commodity '
           '(hidden; own tablet removed); decoy base rate GRA %.2f | vote %s, confidence %.2f; documents voted at >= this confidence are right %.2f (%d of them) | %s |' % (
               len(hits), base['GRA'] / sum(base.values()), c95, conf95, np.mean(hi) if hi else float('nan'), len(hi),
               'SURVIVES (grain vote from HT 86 words, at a confidence where held-out votes are mostly right; but it rests on one near-copy list, HT 86/95)' if c95 == 'GRA' and hi and np.mean(hi) >= 0.6 else 'KILLED or uninformative'))
    wlog(OUT, row)


# ---------------------------------------------------------------- D = 1/5, B = 1/3 (la27)
def t_frac(D):
    mx = Counter()
    for d in D:
        for t in d['toks']:
            if t[0] == 'N' and t[2]:
                for L in ('D', 'B'):
                    k = len(re.findall(L + '(?!D)', t[2])) if L == 'B' else t[2].count('D')
                    mx[(L, k)] += 1
    five = mx[('D', 5)] + sum(v for (L, k), v in mx.items() if L == 'D' and k > 5)
    b3 = sum(v for (L, k), v in mx.items() if L == 'B' and k >= 3)
    RES['frac'] = dict(five=five, b3=b3, dist={'%s%d' % k: v for k, v in mx.items()})
    row = ('| LA-67.1h | la27 C: D = 1/5 and B = 1/3. Kill line: HT 115a re-read and five Ds in one amount anywhere (for B: three Bs). Literal count over all amounts '
           '(the separate DD sign counts as two) | repetitions %s; amounts with >= 5 D: %d; with >= 3 B: %d | SURVIVES the literal check; the decisive test (re-reading HT 115a DDDD) '
           'needs a photograph: untestable-now beyond this |' % (', '.join('%s:%d' % kv for kv in sorted(RES['frac']['dist'].items())), five, b3))
    wlog(OUT, row)


if __name__ == '__main__':
    D = docs_all()
    wlog(OUT, '# LA-67 cycle 1: kill tests for the grade-C guesses named in the task (6 Oct 2026). Scripts tools/la67_lib.py, la67_c1.py.')
    for f in (t_headings, t_ole, t_308, t_sicom, t_tai, t_zatpl, t_ht95b, t_frac):
        f(D)
        print(f.__name__, 'done', flush=True)
    json.dump(RES, open(os.path.join(CK, 'c1.json'), 'w'), default=str, ensure_ascii=False, indent=1)
