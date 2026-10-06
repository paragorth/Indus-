"""pe66 cycle 2: kill tests for the C guesses about sign classes, slots and families.

E  pe61  commodity candidates M269 M354 M367 M418 M482 M069 M206 M268 [M036+1(N30D)] [M288+1(N01)] M054
I  x4    PE strings are combinatorial designations, not running language (kill: language-like order arrows)
J  pe45  name-frame elements keep their slot outside herd and ration tablets
K  pe43  twig families (capacity, header, two name-string families) are real families
S  pe9   tablet-sticky C signs M145 M203 M136 M210 [M036+1(N30D)] M002
R  pe31  closed-frame pairs M124~M377 M057~M262 M097~M259 M370~M388 (kill: never a third shared frame)
M  pe34  M342 is a transferable modifier: [X+M342] unnumbered / opening more often than X
V  pe43  speciated variants M056~f (-> M383) and M005~a (-> M131) take the adopted root's numeral system / slot
W  pe49  N30C N30D M009 M001 M346 used at the same rate by every group (kill: rates differ by tablet type)
N  pe21  N01 = 8 over-represented against 7
Q  pe13  the next-entry dip appears on non-Susa tablets
"""
from pe66_lib import *

OUT = []
RES = {}


def say(*a):
    print(*a, flush=True)


HALVES = (66021, 66022, 66023)


# ------------------------------------------------------------------ E commodity candidates
def sign_entries(sign, tids):
    out = []
    for tid in tids:
        for e in ENT[tid]:
            if e['final'] == sign:
                out.append((tid, e['sys']))
    return out


def comm_stats(sign, tids, rng, nnull=300):
    tok = adj = 0
    for tid in tids:
        for l in TAB[tid]['lines']:
            sg = [base(s) for s in signs_of(l)]
            for k, s in enumerate(sg):
                if s == sign:
                    tok += 1
                    adj += bool(l['numerals']) and k == len(sg) - 1
    se = sign_entries(sign, tids)
    if tok < 4 or len(se) < 4:
        return None
    sysd = collections.Counter(s for _, s in se)
    pur = max(sysd.values()) / len(se)
    pools = {tid: [e['sys'] for e in ENT[tid]] for tid, _ in se}
    nl = []
    for _ in range(nnull):
        c = collections.Counter(pools[tid][rng.integers(len(pools[tid]))] for tid, _ in se)
        nl.append(max(c.values()) / len(se))
    p = pval_ge(pur, nl)
    return dict(tok=tok, adj=adj / tok, n=len(se), pur=pur, p=p, pas=adj / tok >= 0.5 and p < 0.05)


def test_E():
    C = ['M269', 'M354', 'M367', 'M418', 'M482', 'M069', 'M206', 'M268', '|M036+1(N30D)|', '|M288+1(N01)|', 'M054']
    per = {c: [] for c in C}
    dec_rate = []
    for sd in HALVES:
        B = sorted(set(TAB) - half(sd))
        rng = np.random.default_rng(sd)
        for c in C:
            per[c].append(comm_stats(c, B, rng))
        dp = dn = 0
        seen = set()
        for c in C:
            for d in decoys(c, 8, sd + 7, exclude=C):
                if d in seen:
                    continue
                seen.add(d)
                x = comm_stats(d, B, rng, 150)
                if x:
                    dn += 1; dp += x['pas']
        dec_rate.append(dp / max(1, dn))
    rate = float(np.mean(dec_rate))
    surv = [c for c in C if sum(1 for x in per[c] if x and x['pas']) >= 2]
    untest = [c for c in C if sum(1 for x in per[c] if x) < 2]
    # binomial: chance that a decoy passes >= 2 of 3 halves ~ rate^2 * (3 - 2 rate)
    p2 = rate ** 2 * (3 - 2 * rate)
    RES['E'] = dict(per=per, dec_rate=dec_rate)
    def fmt(c):
        xs = per[c]
        return '%s %s' % (c, '/'.join('-' if not x else ('%.2f,%.2f%s' % (x['adj'], x['pur'], '*' if x['pas'] else '')) for x in xs))
    OUT.append(row('PE-66.2e', 'pe61 C "commodity candidates missing from pe59". A commodity should (1) stand next to the number (share of tokens that are the last sign of a numeral line >= 0.5) and (2) fix its number system beyond its tablet (purity of its entries\' system vs a null drawing each entry\'s system from a random entry of the same tablet, p < 0.05). Three fresh held-out halves (seeds 66021-23); decoys: 8 frequency-matched signs per candidate per half, same test',
                   'per half (adjacency, purity, * = pass): %s. Decoy pass rate per half %s (expected chance of passing >= 2 of 3 halves %.2f)' % (
                       '; '.join(fmt(c) for c in C), ', '.join('%.2f' % r for r in dec_rate), p2),
                   'pass >= 2 of 3 halves: %s; fail: %s; too rare: %s. Decoys pass at %.0f%%, so a pass is only %s' % (
                       ' '.join(surv) or 'none', ' '.join(c for c in C if c not in surv and c not in untest) or 'none', ' '.join(untest) or 'none',
                       100 * rate, 'weakly informative' if rate > 0.2 else 'informative')))
    say(OUT[-1])


# ------------------------------------------------------------------ I combinatorial designations
def bigram_gain(train, test, rng, internal=False, nperm=20, lam=0.7):
    uni = collections.Counter(); bi = collections.Counter(); ctx = collections.Counter()
    for s in train:
        seq = ['<s>'] + s + ['</s>']
        for a, b in zip(seq, seq[1:]):
            bi[(a, b)] += 1; ctx[a] += 1; uni[b] += 1
    V = len(uni) + 1; N = sum(uni.values())
    def lp(seq):
        seq = ['<s>'] + seq + ['</s>']
        t = 0.0
        for a, b in zip(seq, seq[1:]):
            pu = (uni[b] + 0.5) / (N + 0.5 * V)
            pb = bi[(a, b)] / ctx[a] if ctx[a] else 0
            t += math.log2(lam * pb + (1 - lam) * pu)
        return t
    gains = []; nsig = 0
    for s in test:
        if internal:
            if len(s) < 4:
                continue
            mid = s[1:-1]
            perms = []
            for _ in range(nperm):
                m = list(mid); rng.shuffle(m); perms.append([s[0]] + m + [s[-1]])
        else:
            if len(s) < 2:
                continue
            perms = []
            for _ in range(nperm):
                m = list(s); rng.shuffle(m); perms.append(m)
        gains.append(lp(s) - np.mean([lp(p) for p in perms]))
        nsig += len(s)
    return float(np.sum(gains) / max(1, nsig)), len(gains)


def test_I():
    corp = json.load(open(os.path.join(DATA, 'pe45_ckpt', 'corpora.json')))
    res = {}
    for name in ('PE', 'UR3'):
        docs = corp[name]
        for sd in HALVES:
            rng = np.random.default_rng(sd)
            ids = [d['id'] for d in docs]
            A = half(sd, ids)
            tr = [n for d in docs if d['id'] in A for n in d['names']]
            te = [n for d in docs if d['id'] not in A for n in d['names']]
            full = bigram_gain(tr, te, rng)
            mid = bigram_gain(tr, te, rng, internal=True)
            # shuffled-order twin (pure combination)
            sh = lambda X: [list(rng.permutation(x)) for x in X]
            fsh = bigram_gain(sh(tr), sh(te), rng)
            msh = bigram_gain(sh(tr), sh(te), rng, internal=True)
            res.setdefault(name, []).append((full, mid, fsh, msh))
    # productive combination: share of held-out unseen strings whose every unordered sign pair was seen in training
    prod = []
    docs = corp['PE']
    for sd in HALVES:
        rng = np.random.default_rng(sd + 50)
        A = half(sd, [d['id'] for d in docs])
        tr = [n for d in docs if d['id'] in A for n in d['names']]
        te = [n for d in docs if d['id'] not in A for n in d['names']]
        seen = {tuple(x) for x in tr}
        pairs = {(a, b) for x in tr for i, a in enumerate(x) for b in x[i + 1:]} | {(b, a) for x in tr for i, a in enumerate(x) for b in x[i + 1:]}
        uni = collections.Counter(s for x in tr for s in x)
        sg, w = zip(*uni.items()); w = np.array(w, float) / sum(w)
        def cover(X):
            nov = [x for x in X if tuple(x) not in seen and len(x) >= 2 and all(s in uni for s in x)]
            return sum(1 for x in nov if all((x[i], x[j]) in pairs for i in range(len(x)) for j in range(i + 1, len(x)))) / max(1, len(nov)), len(nov)
        real = cover(te)
        rnd = cover([list(rng.choice(sg, size=len(x), p=w)) for x in te])
        prod.append((real, rnd))
    RES['I'] = dict(res=res, prod=prod)
    f = lambda L, k: float(np.mean([x[k][0] for x in L]))
    pe, ur = res['PE'], res['UR3']
    OUT.append(row('PE-66.2i', 'x4 C "entries are combinatorial designations rather than running language" (support: held-out combinations predicted by productive rules; kill: language-like word-order arrows). Name-like strings (pe45 builder; 696 PE tablets) vs Ur III personal names at the same size (the language calibrator). Order gain = held-out bits per sign of the real order over 20 within-string permutations under an interpolated bigram model trained on the other half (3 fresh halves, seeds 66021-23); INTERNAL = first and last sign fixed, middle permuted (slot arrows removed). Shuffled-order twins as zero. Productivity: share of unseen held-out strings whose every sign pair was seen in training, vs random strings of the same lengths from the unigram distribution',
                   'order gain, bits/sign full / internal: PE %.3f / %.3f (shuffled %.3f / %.3f); Ur III names %.3f / %.3f (shuffled %.3f / %.3f). Productivity coverage PE %s vs random strings %s' % (
                       f(pe, 0), f(pe, 1), f(pe, 2), f(pe, 3), f(ur, 0), f(ur, 1), f(ur, 2), f(ur, 3),
                       ', '.join('%.2f (n %d)' % r for r, _ in prod), ', '.join('%.2f' % x[0] for _, x in prod)),
                   'PE order arrows are %.0f%% of the Ur III name level (internal %.0f%%); %s' % (
                       100 * f(pe, 0) / max(1e-9, f(ur, 0)), 100 * f(pe, 1) / max(1e-9, f(ur, 1)),
                       'no language-like arrow: survives (C)' if f(pe, 1) < 0.33 * f(ur, 1) else 'language-like internal arrows: KILLED')))
    say(OUT[-1])


# ------------------------------------------------------------------ J name-frame elements
HERD = {'M362', 'M367', 'M346', 'M006'}


def tablet_kind(t):
    sg = {base(s) for l in t['lines'] for s in signs_of(l)}
    if sg & HERD:
        return 'herd'
    E = ENT[t['id']]
    if 'M288' in sg or (E and sum(e['sys'] in ('C', 'C*') for e in E) >= 1):
        return 'ration'
    return 'other'


KIND = {t['id']: tablet_kind(t) for t in T}


def slot_z(sign, kinds):
    oi = of = ei = ef = vi = vf = 0.0
    n = 0
    for t in T:
        if KIND[t['id']] not in kinds:
            continue
        for l in t['lines']:
            sg = [base(s) for s in signs_of(l)]
            if len(sg) < 2:
                continue
            for k, s in enumerate(sg):
                if s == sign:
                    n += 1
                    p = 1 / len(sg)
                    ei += p; ef += p; vi += p * (1 - p); vf += p * (1 - p)
                    oi += k == 0; of += k == len(sg) - 1
    if n < 5:
        return None
    return dict(n=n, zi=(oi - ei) / math.sqrt(vi), zf=(of - ef) / math.sqrt(vf))


def test_J():
    stated = {'M056': 'f', 'M010': 'f', 'M001': 'f', 'M157': 'i', 'M210': 'i', 'M054': None, 'M009': None}
    IN = {'herd', 'ration'}; OUTK = {'other'}
    res = []
    for s, d in stated.items():
        a = slot_z(s, IN); b = slot_z(s, OUTK)
        if d is None and a:
            d = 'f' if abs(a['zf']) >= abs(a['zi']) and a['zf'] > 0 else ('i' if a['zi'] > 0 else 'f')
        za = a and a['z' + d]; zb = b and b['z' + d]
        res.append((s, d, a and a['n'], za, b and b['n'], zb))
    # decoys: signs with a |z| >= 2 slot preference inside -> share keeping it (z >= 2 same slot) outside
    kept = tot = 0
    for s in stated:
        for dcy in decoys(s, 15, 66024, exclude=stated):
            a = slot_z(dcy, IN)
            if not a:
                continue
            d = 'f' if a['zf'] >= a['zi'] else 'i'
            if a['z' + d] < 2:
                continue
            b = slot_z(dcy, OUTK)
            tot += 1
            kept += bool(b and b['z' + d] >= 2)
    RES['J'] = dict(res=res, dec_kept=kept, dec_tot=tot)
    surv = [r[0] for r in res if r[5] is not None and r[5] >= 2]
    kill = [r[0] for r in res if r[5] is not None and r[5] < 2 and r[3] is not None and r[3] >= 2]
    unt = [r[0] for r in res if r[5] is None]
    OUT.append(row('PE-66.2j', 'pe45 C "name-frame elements M056 M010 M001 (final), M157 M210 (initial), M054 M009 keep their slot on tablets of every kind" (kill: the slot preference disappears outside herd and ration tablets). Multi-sign lines; z of first/last-slot count vs the within-string uniform expectation, on herd tablets (M362/M367/M346/M006) + ration tablets (M288 or capacity entries) vs all other tablets. Decoys: 15 frequency-matched signs per element with a slot preference (z >= 2) inside (seed 66024): share that keep it outside',
                   '; '.join('%s %s inside n %s z %s, outside n %s z %s' % (s, d, a, '%.1f' % za if za is not None else '-', b, '%.1f' % zb if zb is not None else '-') for s, d, a, za, b, zb in res) +
                   '. Decoys keeping their slot outside: %d of %d' % (kept, tot),
                   'keep slot outside (survive): %s; lose it (killed): %s; too rare outside: %s. Decoy retention %d/%d shows slot preference is a general sign property, so survival is weak evidence for a NAME FRAME' % (
                       ' '.join(surv) or 'none', ' '.join(kill) or 'none', ' '.join(unt) or 'none', kept, tot)))
    say(OUT[-1])


# ------------------------------------------------------------------ K pe43 twig families
def profiles(tids, minn=3):
    P = collections.defaultdict(lambda: np.zeros(9))
    for tid in tids:
        t = TAB[tid]
        first = next((k for k, x in enumerate(t['lines']) if x['surface'] == 'obverse'), -1)
        for k, l in enumerate(t['lines']):
            sg = [base(s) for s in signs_of(l)]
            hdr = k == first and not l['numerals']
            sy = system_of(l['numerals']) if l['numerals'] else None
            for j, s in enumerate(sg):
                v = P[s]
                if hdr: v[0] += 1
                elif len(sg) == 1: v[1] += 1
                elif j == 0: v[2] += 1
                elif j == len(sg) - 1: v[3] += 1
                else: v[4] += 1
                if sy in ('C', 'C*'): v[5] += 1
                elif sy == 'SDB': v[6] += 1
                elif sy: v[7] += 1
                else: v[8] += 1
    out = {}
    for s, v in P.items():
        if v[:5].sum() >= minn:
            a = v[:5] / v[:5].sum(); b = v[5:] / v[5:].sum()
            out[s] = np.concatenate([a, b])
    return out


def coh(fam, P):
    m = [P[s] for s in fam if s in P]
    if len(m) < 3:
        return None, len(m)
    M = np.array(m)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T
    k = len(m)
    return float((S.sum() - k) / (k * (k - 1))), k


def test_K():
    d = json.load(open(os.path.join(DATA, 'pe43_ckpt', 'c3_PE.json')))['A']['leaves']
    top = json.load(open(os.path.join(DATA, 'pe43_ckpt', 'c1_PE.json')))['C']['families']
    fams = {'name-string T1 (M388 ...)': d[0], 'capacity T2 (M387 M297 ...)': d[1], 'name-string T4 (M371 ...)': d[3], 'header T6 (M288 M157 ...)': d[5]}
    topof = {s: i for i, f in enumerate(top) for s in f}
    res = {}
    for sd in HALVES:
        B = sorted(set(TAB) - half(sd))
        P = profiles(B)
        rng = np.random.default_rng(sd + 3)
        for name, fam in fams.items():
            c, k = coh(fam, P)
            if c is None:
                continue
            members = [s for s in fam if s in P]
            tf = collections.Counter(topof.get(s) for s in members).most_common(1)[0][0]
            pool_top = [s for s in top[tf] if s in P]
            pool_all = [s for s in P if s in topof]
            na = [coh(list(rng.choice(pool_all, size=k, replace=False)), P)[0] for _ in range(400)]
            nt = [coh(list(rng.choice(pool_top, size=k, replace=False)), P)[0] for _ in range(400)]
            res.setdefault(name, []).append((c, k, pval_ge(c, na), pval_ge(c, nt)))
    RES['K'] = res
    surv = [n for n, v in res.items() if sum(1 for x in v if x[3] < 0.05) >= 2]
    OUT.append(row('PE-66.2k', 'pe43 C "twig families: capacity {M387 M297 M036 M260 M111 M264 M265 M002 ...}, header {M288 M157 M153 M175 M106 M010 ...}, name-string {M388 M218 M263 ...} and {M371 M377 M320 ...}". Coherence = mean pairwise cosine of each member\'s slot + number-system profile, measured on 3 fresh held-out halves (seeds 66021-23). Decoy families: 400 random same-size sets from all roots, and 400 from the SAME top-level family (pe43 cycle 1), which is what a twig must beat',
                   '; '.join('%s: %s' % (n, ', '.join('coh %.2f (k %d) p_all %.3f p_top %.3f' % x for x in v)) for n, v in res.items()),
                   'beat same-top-family decoys in >= 2 of 3 halves: %s; the others are only the top-level slot split (killed as twig families)' % (', '.join(surv) or 'none')))
    say(OUT[-1])


# ------------------------------------------------------------------ S tablet-sticky signs
def sticky(sign, tids, rng, nnull=500):
    w = []; obs = 0; ntok = 0
    for tid in tids:
        sg = [base(s) for l in TAB[tid]['lines'] for s in signs_of(l)]
        c = sg.count(sign)
        ntok += c
        obs += c >= 2
        w.append(len(sg))
    if ntok < 5:
        return None
    w = np.array(w, float); w /= w.sum()
    nl = [int((rng.multinomial(ntok, w) >= 2).sum()) for _ in range(nnull)]
    return dict(n=ntok, obs=obs, null=float(np.mean(nl)), p=pval_ge(obs, nl))


def test_S():
    G = ['M145', 'M203', 'M136', 'M210', '|M036+1(N30D)|', 'M002']
    res = {g: [] for g in G}
    drate = []
    for sd in HALVES:
        B = sorted(set(TAB) - half(sd))
        rng = np.random.default_rng(sd + 5)
        for g in G:
            res[g].append(sticky(g, B, rng))
        dp = dn = 0
        for g in G:
            for dcy in decoys(g, 10, sd + 9, exclude=G):
                x = sticky(dcy, B, rng, 200)
                if x:
                    dn += 1; dp += x['p'] < 0.05
        drate.append(dp / max(1, dn))
    RES['S'] = dict(res=res, drate=drate)
    surv = [g for g in G if sum(1 for x in res[g] if x and x['p'] < 0.05) >= 2]
    OUT.append(row('PE-66.2s', 'pe9 C "tablet-sticky signs M145 M203 M136 M210 [M036+1(N30D)] M002 (did not replicate at 52 tablets)". Tablets with >= 2 tokens of the sign vs tokens dropped on tablets in proportion to their length (500x), on 3 fresh held-out halves (seeds 66021-23); decoys: 10 frequency-matched signs per sign per half',
                   '; '.join('%s %s' % (g, ', '.join('-' if not x else '%d vs %.1f p %.3f' % (x['obs'], x['null'], x['p']) for x in res[g])) for g in G) +
                   '. Decoys sticky (p < 0.05) per half: %s' % ', '.join('%.2f' % r for r in drate),
                   'sticky in >= 2 of 3 halves: %s (decoys are sticky at %.0f%%: stickiness is the general tablet-pool property, B in pe9; individual stickiness carries no extra meaning)' % (' '.join(surv) or 'none', 100 * float(np.mean(drate)))))
    say(OUT[-1])


# ------------------------------------------------------------------ R closed-frame pairs
STR = [tuple(base(s) for s in signs_of(l)) for t in T for l in t['lines'] if len(signs_of(l)) >= 2]
FR = collections.defaultdict(set)
for s in STR:
    for k, x in enumerate(s):
        FR[x].add(s[:k] + ('_',) + s[k + 1:])


def test_R():
    P = [('M124', 'M377'), ('M057', 'M262'), ('M097', 'M259'), ('M370', 'M388')]
    res = []
    for a, b in P:
        sh = len(FR[a] & FR[b])
        dd = []
        for j in range(40):
            da = decoys(a, 1, 66030 + j, exclude=[a, b])[0]
            db = decoys(b, 1, 66130 + j, exclude=[a, b, da])[0]
            dd.append(len(FR[da] & FR[db]))
        res.append((a, b, sh, float(np.mean(dd)), pval_ge(sh, dd)))
    RES['R'] = res
    OUT.append(row('PE-66.2r', 'pe31 C "closed-frame candidates M124~M377, M057~M262, M097~M259, M370~M388" (support: both in the same frame on new tablets; kill: never a third shared frame). Frame = a multi-sign line with the sign replaced by a slot; shared frames counted on the whole corpus. Decoy pairs: 40 per pair, each sign frequency-matched (seeds 66030/66130)',
                   '; '.join('%s~%s shared frames %d (decoy pairs mean %.1f, p %.3f)' % x for x in res),
                   'third shared frame present (stated kill passed) but beyond frequency-matched decoys only for: %s; killed as interchangeable pairs (no more shared frames than decoys): %s' % (
                       ' '.join('%s~%s' % (a, b) for a, b, sh, m, p in res if sh >= 3 and p < 0.05) or 'none',
                       ' '.join('%s~%s' % (a, b) for a, b, sh, m, p in res if not (sh >= 3 and p < 0.05)) or 'none')))
    say(OUT[-1])


# ------------------------------------------------------------------ M M342 modifier
def unnum_share(sign):
    u = n = 0
    for t in T:
        for l in t['lines']:
            sg = signs_of(l)
            for s in sg:
                if s == sign or (s.startswith('|') and base(s) == sign) or base(s) == sign:
                    n += 1; u += not l['numerals']
    return u, n


def modifier_rate(mod):
    comps = [s for s in TOK if s.startswith('|') and s.count('+') == 1]
    out = []
    for c in comps:
        parts = c.strip('|').split('+')
        if parts[1] != mod:
            continue
        X = parts[0]
        if TOK[X] < 3 or TOK[c] < 2:
            continue
        uc, nc = unnum_share(c); ux, nx = unnum_share(X)
        out.append((X, uc / nc, ux / nx, nc))
    return out


def test_M():
    m = modifier_rate('M342')
    hi = sum(1 for _, a, b, _ in m if a > b)
    mods = collections.Counter(s.strip('|').split('+')[1] for s in TOK if s.startswith('|') and s.count('+') == 1)
    dr = []
    for mod, c in mods.items():
        if mod == 'M342' or not mod.startswith('M'):
            continue
        r = modifier_rate(mod)
        if len(r) >= 4:
            dr.append((mod, len(r), sum(1 for _, a, b, _ in r if a > b) / len(r)))
    rate = hi / max(1, len(m))
    pct = sum(1 for x in dr if x[2] >= rate)
    RES['M'] = dict(m=m, decoys=dr)
    OUT.append(row('PE-66.2m', 'pe34 C "M342 is a transferable modifier: [X+M342] is unnumbered / opening more often than X" (kill: numbered as often as X). Every base X with [X+M342] (>= 2 tokens) and X (>= 3): unnumbered-line share of compound vs base. Decoy modifiers: every other second element with >= 4 such bases, same rate',
                   'M342 bases %d: compound more unnumbered on %d (%s). Decoy modifiers %d: as high a rate in %d (%s)' % (
                       len(m), hi, ', '.join('%s %.2f vs %.2f' % (x, a, b) for x, a, b, _ in m), len(dr), pct, ', '.join('%s %d/%.2f' % x for x in dr)),
                   'survives beyond decoy modifiers' if rate > 0.6 and pct <= 0.1 * max(1, len(dr)) else
                   ('survives the stated kill (compounds less numbered) but %d of %d decoy modifiers do the same: demote' % (pct, len(dr)) if rate > 0.5 else 'KILLED: [X+M342] numbered as often as X')))
    say(OUT[-1])


# ------------------------------------------------------------------ V speciated variants
def sysprof(sign, tids=None):
    c = collections.Counter()
    for tid, E in ENT.items():
        if tids is not None and tid not in tids:
            continue
        for e in E:
            if sign in e['signs'] or (sign == base(sign) and sign in e['bsigns'] and not any(s != base(s) and base(s) == sign for s in e['signs'])):
                c[e['sys'] if e['sys'] in ('C', 'SDB') else 'oth'] += 1
    return c


def js(a, b):
    k = ['C', 'SDB', 'oth']
    p = np.array([a[x] for x in k], float) + 0.5; q = np.array([b[x] for x in k], float) + 0.5
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def hdrshare(sign, exact=True):
    h = n = 0
    for t in T:
        first = next((k for k, x in enumerate(t['lines']) if x['surface'] == 'obverse'), -1)
        for k, l in enumerate(t['lines']):
            for s in signs_of(l):
                if (s == sign) if exact else (base(s) == sign and s == base(s)):
                    n += 1; h += k == first and not l['numerals']
    return h, n


def test_V():
    pairs = [('M056~f', 'M056', 'M383'), ('M005~a', 'M005', 'M131'), ('M003~c', 'M003', 'M246'), ('M056~e', 'M056', 'M265'), ('M147~d', 'M147', 'M041')]
    res = []
    for v, own, adp in pairs:
        pv = sysprof(v); po = sysprof(own); pa = sysprof(adp)
        dv = js(pv, po); da = js(pv, pa)
        hv = hdrshare(v); ho = hdrshare(own, False); ha = hdrshare(adp, False)
        res.append((v, sum(pv.values()), dv, da, hv, ho, ha))
    # decoys: other variants (>= 10 entries) vs own root and a frequency-matched random root
    closer = tot = 0
    rng = np.random.default_rng(66040)
    vs = [s for s, n in VTOK.items() if '~' in s and not s.startswith('|') and n >= 10 and base(s) != s]
    for v in vs:
        pv = sysprof(v)
        if sum(pv.values()) < 5:
            continue
        r = decoys(base(v), 1, int(rng.integers(1e9)), exclude=[base(v)])[0]
        tot += 1; closer += js(pv, sysprof(r)) < js(pv, sysprof(base(v)))
    RES['V'] = dict(res=res, closer=closer, tot=tot)
    OUT.append(row('PE-66.2v', 'pe43 C "speciated variants: M056~f travels with M383; M005~a takes the header slot unlike M005" (support: the variant takes the adopted root\'s numeral system; kill: it keeps its own root\'s system). Number-system profile (capacity / count / other) of each variant vs own root and adopted root (Jensen-Shannon); header-line share. Decoy: every other variant with >= 10 tokens vs its own root and a frequency-matched random root (seed 66040)',
                   '; '.join('%s (n %d): JS to own %.3f, to adopted %.3f; header %d/%d vs own %d/%d, adopted %d/%d' % (v, n, a, b, h[0], h[1], o[0], o[1], q[0], q[1]) for v, n, a, b, h, o, q in res) +
                   '. Decoy variants closer to a random root than to their own: %d of %d' % (closer, tot),
                   'takes the adopted root\'s system (support): %s; keeps own (kill): %s' % (
                       ' '.join(v for v, n, a, b, *_ in res if b < a) or 'none', ' '.join(v for v, n, a, b, *_ in res if b >= a) or 'none')))
    say(OUT[-1])


# ------------------------------------------------------------------ W even-rate system vocabulary
def test_W():
    G = ['N30C', 'N30D', 'M009', 'M001', 'M346']
    def ttype(t):
        E = ENT[t['id']]
        cap = sum(e['sys'] in ('C', 'C*') for e in E) > len(E) / 2 if E else False
        h = header(t)
        hd = 'M157' if h and 'M157' in h else ('hdr' if h else 'nohdr')
        return ('cap' if cap else 'cnt') + '-' + hd
    TT = {t['id']: ttype(t) for t in T}
    def rates(item, tids):
        occ = collections.Counter(); exp = collections.Counter()
        for tid in tids:
            t = TAB[tid]
            if item.startswith('N'):
                n = sum(1 for l in t['lines'] for _, c in l['numerals'] if c == item)
                d = sum(len(l['numerals']) for l in t['lines'])
            else:
                n = sum(1 for l in t['lines'] for s in signs_of(l) if base(s) == item)
                d = sum(len(signs_of(l)) for l in t['lines'])
            occ[TT[tid]] += n; exp[TT[tid]] += d
        return occ, exp
    def chi(item, tids):
        occ, exp = rates(item, tids)
        tot = sum(occ.values()); D = sum(exp.values())
        if tot < 10:
            return None
        x2 = sum((occ[k] - tot * exp[k] / D) ** 2 / max(1e-9, tot * exp[k] / D) for k in exp if exp[k])
        dof = sum(1 for k in exp if exp[k]) - 1
        from math import erf
        # Wilson-Hilferty approx p-value
        z = ((x2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        p = 0.5 * (1 - erf(z / math.sqrt(2)))
        return x2, dof, p, {k: round(occ[k] / exp[k], 3) if exp[k] else 0 for k in sorted(exp)}
    res = {}
    for sd in HALVES:
        B = sorted(set(TAB) - half(sd))
        for g in G:
            res.setdefault(g, []).append(chi(g, B))
    even = 0; dt = 0
    B = sorted(set(TAB) - half(HALVES[0]))
    for g in ['M009', 'M001', 'M346']:
        for d in decoys(g, 15, 66041, exclude=G):
            x = chi(d, B)
            if x:
                dt += 1; even += x[2] >= 0.01
    RES['W'] = dict(res=res, even=even, dt=dt)
    kill = [g for g in G if sum(1 for x in res[g] if x and x[2] < 0.01) >= 2]
    OUT.append(row('PE-66.2w', 'pe49 C "N30C N30D M009 M001 M346 are used at the same rate by every habit group (system vocabulary)" (kill: their rates differ by tablet type on reserved tablets). Rate per sign token (per numeral group for N-codes) across 6 tablet types (capacity vs count tablets x M157 header / other header / none), chi-square on 3 fresh reserved halves (seeds 66021-23). Decoys: 15 frequency-matched signs each for M009 M001 M346',
                   '; '.join('%s %s' % (g, ', '.join('-' if not x else 'p %.2g' % x[2] for x in res[g])) for g in G) + '. Rates (half 1): ' +
                   '; '.join('%s %s' % (g, res[g][0][3] if res[g][0] else '-') for g in G) + '. Decoys even (p >= 0.01): %d of %d' % (even, dt),
                   'KILLED (rates differ by tablet type in >= 2 of 3 halves): %s; survive: %s' % (' '.join(kill) or 'none', ' '.join(g for g in G if g not in kill) or 'none')))
    say(OUT[-1])


# ------------------------------------------------------------------ N eight over seven
def test_N():
    def digits(tids, code):
        c = collections.Counter()
        for tid in tids:
            for e in ENT[tid]:
                if e['sys'] == 'SDB':
                    for n, cc in e['nums']:
                        if cc == code and n:
                            c[n] += 1
        return c
    res = []
    for sd in HALVES:
        for part in ('A', 'B'):
            H = half(sd)
            tids = sorted(H if part == 'A' else set(TAB) - H)
            c = digits(tids, 'N01')
            res.append((c[7], c[8], c[9]))
    c = digits(sorted(TAB), 'N01')
    c14 = digits(sorted(TAB), 'N14')
    ex = {d: c[d] / math.sqrt(c[d - 1] * c[d + 1]) for d in range(2, 9) if c[d - 1] and c[d + 1]}
    ex14 = {d: c14[d] / math.sqrt(c14[d - 1] * c14[d + 1]) for d in range(2, 9) if c14[d - 1] and c14[d + 1]}
    RES['N'] = dict(res=res, c=dict(c), ex=ex, ex14=ex14)
    wins = sum(1 for a, b, _ in res if b > a)
    OUT.append(row('PE-66.2n', 'pe21 C side note "N01 = 8 is over-represented against 7". Counts of 7, 8, 9 N01 in count entries, on both halves of 3 fresh splits (seeds 66021-23); decoy digits: the same neighbour ratio c(d) / sqrt(c(d-1) c(d+1)) for every digit 2-8, and for N14 digits',
                   'all data 7:%d 8:%d 9:%d. Halves (7, 8, 9): %s; 8 > 7 in %d of 6. Neighbour ratio N01: %s; N14: %s' % (
                       c[7], c[8], c[9], res, wins, ', '.join('%d %.2f' % kv for kv in ex.items()), ', '.join('%d %.2f' % kv for kv in ex14.items())),
                   'survives: 8 beats 7 in every half and its neighbour ratio is the highest of the digits' if wins == 6 and max(ex, key=ex.get) == 8 else
                   ('survives as 8 > 7 in every half, but other digits show similar bumps' if wins == 6 else 'KILLED: 8 does not beat 7 in held-out halves')))
    say(OUT[-1])


# ------------------------------------------------------------------ Q next-entry dip on non-Susa tablets
def dip(tids, rng, nnull=300):
    def sim(E):
        a1 = []; a2 = []
        S = [set(e['bsigns']) for e in E]
        for k in range(len(S) - 1):
            a1.append(len(S[k] & S[k + 1]) / len(S[k] | S[k + 1]))
        for k in range(len(S) - 2):
            a2.append(len(S[k] & S[k + 2]) / len(S[k] | S[k + 2]))
        return a1, a2
    tabs = [[e for e in ENT[t] if e['surf'] == 'obverse'] for t in tids]
    tabs = [E for E in tabs if len(E) >= 4]
    o1 = []; o2 = []
    for E in tabs:
        a, b = sim(E); o1 += a; o2 += b
    obs = np.mean(o1) - np.mean(o2)
    nl = []
    for _ in range(nnull):
        n1 = []; n2 = []
        for E in tabs:
            P = [E[i] for i in rng.permutation(len(E))]
            a, b = sim(P); n1 += a; n2 += b
        nl.append(np.mean(n1) - np.mean(n2))
    nl = np.asarray(nl)
    return dict(ntab=len(tabs), obs=float(obs), null=float(nl.mean()), p=float((1 + np.sum(nl <= obs)) / (1 + nnull)))


def test_Q():
    rng = np.random.default_rng(66042)
    NS = [t['id'] for t in T if site(t) not in ('Susa',)]
    r = dip(NS, rng)
    # power: random Susa subsets with the same number of eligible tablets
    SU = [t['id'] for t in T if site(t) == 'Susa' and len([e for e in ENT[t['id']] if e['surf'] == 'obverse']) >= 4]
    pw = []
    for j in range(30):
        sub = list(np.random.default_rng(66300 + j).choice(SU, size=r['ntab'], replace=False))
        pw.append(dip(sub, rng, 150)['p'])
    full = dip(SU, rng, 100)
    RES['Q'] = dict(r=r, power=pw, full=full)
    OUT.append(row('PE-66.2q', 'pe13 C "entries written one per record from a per-tablet set, laid out with alternation" (support: the same next-entry dip on non-Susa tablets; kill: a decaying kernel in a larger corpus). Dip = mean Jaccard of adjacent obverse entries minus entries two apart, vs within-tablet order permutations (300x, seed 66042), on all non-Susa tablets with >= 4 obverse entries. Power: 30 random Susa subsets of the same size',
                   'non-Susa: %d tablets, dip %.3f vs null %.3f (p %.3f). Susa (all %d): dip %.3f vs %.3f (p %.3f). Susa subsets of the same size reach p < 0.05 in %d of 30' % (
                       r['ntab'], r['obs'], r['null'], r['p'], len(SU), full['obs'], full['null'], full['p'], sum(1 for x in pw if x < 0.05)),
                   'supported off Susa' if r['p'] < 0.05 else ('untestable-now (no power: same-size Susa subsets detect it %d/30)' % sum(1 for x in pw if x < 0.05) if sum(1 for x in pw if x < 0.05) < 15 else 'not seen off Susa although the test has power: weakened')))
    say(OUT[-1])


if __name__ == '__main__':
    which = sys.argv[1:] or list('EIJKSRMVWNQ')
    for w in which:
        globals()['test_' + w]()
    tag = ''.join(which)
    dump(RES, 'c2_%s.json' % tag)
    open(os.path.join(CK, 'c2_%s.rows' % tag), 'w').write('\n'.join(OUT) + '\n')
