#!/usr/bin/env python3
"""LA-67 cycle 2: kill tests for the remaining grade-C guesses in FINDINGS.md that existing data can
reach.  Same rules as cycle 1: literal would-kill check first, then fresh-seed nulls, matched decoys and
held-out tablet halves; decoys go through the identical criterion (false-survival rate)."""
from la67_lib import *
import la67_c1 as C1
from fractions import Fraction

OUT = os.path.join(LOOPS, 'la67_cycle2.txt')
NH = 20
RES = {}
FV = {'J': Fraction(1, 2), 'E': Fraction(1, 4), 'F': Fraction(1, 8), 'K': Fraction(1, 16), 'D': Fraction(1, 5), 'B': Fraction(1, 3),
      'A': Fraction(1, 6), 'H': Fraction(1, 6), 'JE': Fraction(3, 4)}


def val(t):
    v = Fraction(t[1])
    s = t[2]
    if not s:
        return v
    if s == 'JE':
        return v + FV['JE']
    for ch in s:
        if ch not in FV:
            return None
        v += FV[ch]
    return v


def entries(d):
    """(word, number token, line index) for a word directly followed (maybe via one logogram) by a number."""
    out = []
    for li, ln in enumerate(lines(d)):
        for j, t in enumerate(ln):
            if t[0] != 'W':
                continue
            if j + 1 < len(ln) and ln[j + 1][0] == 'N':
                out.append((t[1], ln[j + 1], li))
            elif j + 2 < len(ln) and ln[j + 1][0] == 'L' and ln[j + 2][0] == 'N':
                out.append((t[1], ln[j + 2], li))
    return out


def logo_bases(d):
    return set(re.sub(r'\+.*', '', t[1]) for t in d['toks'] if t[0] == 'L')


# ---------------------------------------------------------------- la12 cross-tablet sums
def t_la12(D):
    HT = [d for d in admin(D) if d['site'] == 'HT']
    written = defaultdict(set)
    for d in HT:
        for t in d['toks']:
            if t[0] == 'N' and t[1] >= 20:
                written[t[1]].add(d['tab'])
    def matches(x, tab):
        return any(t != tab for t in written.get(x, ()))
    sums = []
    for d in HT:
        ints = [t[1] for t in d['toks'] if t[0] == 'N']
        if len(ints) >= 3 and sum(ints) >= 20:
            sums.append((sum(ints), d['tab']))
        # per-logogram block sums
        cur = None; acc = []
        for t in d['toks']:
            if t[0] == 'L':
                if cur and len(acc) >= 2 and sum(acc) >= 20:
                    sums.append((sum(acc), d['tab']))
                cur, acc = t[1], []
            elif t[0] == 'N' and cur:
                acc.append(t[1])
        if cur and len(acc) >= 2 and sum(acc) >= 20:
            sums.append((sum(acc), d['tab']))
    real = np.mean([matches(s, t) for s, t in sums])
    dec = {k: np.mean([matches(s + k, t) for s, t in sums]) for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)}
    p = pct_rank(real, list(dec.values()))
    RES['la12'] = dict(n=len(sums), real=real, dec=dec)
    row = ('| LA-67.2a | la12 C: HT 116a GRA lines (109) = HT 1 KU-PA3-NU 109; HT 27a VIR 140 = all numbers of HT 27b. Kill line (stated: probably coincidence). Test: every HT '
           'side-sum and logogram-block sum (>= 20; n %d) checked for an equal written number (>= 20) on ANOTHER HT tablet; decoys = the same sums shifted by +-1..5 | '
           'real sums matched %.2f; shifted decoy sums %.2f-%.2f (mean %.2f); P %.2f | KILLED: a written match for an internal sum is as common for true sums as for wrong ones, so the two leads are what chance gives |' % (
               len(sums), real, min(dec.values()), max(dec.values()), np.mean(list(dec.values())), p))
    wlog(OUT, row)


# ---------------------------------------------------------------- la10 consonant repetition inside words
def cons(s):
    if s.startswith('*') or not re.match(r'^[A-Z]+[₂₃]?$', s):
        return None
    s = s.rstrip('₂₃')
    return re.sub(r'[AEIOU]+$', '', s)


def t_la10(D):
    W = [t[1].split('-') for d in D for t in d['toks'] if t[0] == 'W' and '-' in t[1]]
    rng = random.Random(seed('la10'))
    site_w = defaultdict(list)
    for d in D:
        for t in d['toks']:
            if t[0] == 'W' and '-' in t[1]:
                site_w['HT' if d['site'] == 'HT' else 'nonHT'].append(t[1].split('-'))
    out = {}
    for grp, ws in site_w.items():
        ws = [list(x) for x in set(tuple(w) for w in ws)]
        def rate(ws):
            a = n = 0
            for w in ws:
                for x, y in zip(w, w[1:]):
                    cx, cy = cons(x), cons(y)
                    if cx is None or cy is None or cx == '' or cy == '':
                        continue
                    n += 1; a += cx == cy
            return a / n
        obs = rate(ws)
        null = []
        for _ in range(500):
            sh = []
            for w in ws:
                w2 = w[:]; rng.shuffle(w2); sh.append(w2)
            null.append(rate(sh))
        # global sign shuffle (signs re-dealt across words, lengths kept)
        allsig = [s for w in ws for s in w]
        null2 = []
        for _ in range(500):
            rng.shuffle(allsig); it = iter(allsig)
            null2.append(rate([[next(it) for _ in w] for w in ws]))
        out[grp] = dict(obs=obs, within=float(np.mean(null)), p_within=float((np.sum(np.array(null) <= obs) + 1) / 501),
                        glob=float(np.mean(null2)), p_glob=float((np.sum(np.array(null2) <= obs) + 1) / 501), types=len(ws))
    RES['la10'] = out
    surv = all(v['p_glob'] <= 0.05 for v in out.values())
    row = ('| LA-67.2b | la10 C: repeated consonants avoided inside roots (and consonant-final stems). Uses LB-derived consonants (the guess itself does). Test: share of adjacent '
           'sign pairs in a word type with the same consonant, in HT and non-HT word types separately, vs signs re-dealt across word types (lengths kept) and vs within-word '
           'shuffles (500 each, fresh seed). Consonant-final stems: untestable (no word-final consonant is written) | %s | Repetition avoidance %s; consonant-final stems UNTESTABLE-NOW |' % (
               '; '.join('%s (%d types) %.3f vs re-dealt %.3f P %.3f, within-word %.3f P %.3f' % (g, v['types'], v['obs'], v['glob'], v['p_glob'], v['within'], v['p_within']) for g, v in out.items()),
               'SURVIVES in both halves (agrees with la38/la44 B)' if surv else 'KILLED in at least one half'))
    wlog(OUT, row)


# ---------------------------------------------------------------- one-sign substitutions across groups (la13 KA/JA, la16 KU~WA, MA~ME)
def subs_pairs(wa, wb, pos=None):
    """count of (x in group a, y in group b) substitutions between word types of equal length differing in one slot."""
    C = Counter()
    ib = defaultdict(set)
    for w in wb:
        s = w.split('-')
        for i in range(len(s)):
            ib[(len(s), i, tuple(s[:i] + ['_'] + s[i + 1:]))].add(s[i])
    for w in wa:
        s = w.split('-')
        for i in range(len(s)):
            if pos is not None and i != pos:
                continue
            for y in ib.get((len(s), i, tuple(s[:i] + ['_'] + s[i + 1:])), ()):
                if y != s[i]:
                    C[(s[i], y)] += 1
    return C


def t_subs(D):
    docs = [d for d in D]
    rng = random.Random(seed('subs'))
    def groups(lab):
        g = defaultdict(set)
        for d, l in zip(docs, lab):
            for t in d['toks']:
                if t[0] == 'W' and '-' in t[1]:
                    g[l].add(t[1])
        return g
    lab = ['HT' if d['site'] == 'HT' else 'X' for d in docs]
    tab_site = {}
    for d in docs:
        tab_site[d['tab']] = d['site']
    targets = {'la13 HT KA vs other JA (2-sign words, first slot)': (('KA', 'JA'), 2, 0),
               'la16 KU~WA (2-sign words)': (('KU', 'WA'), 2, None), 'la16 MA~ME (2-sign words)': (('MA', 'ME'), 2, None)}
    def stat(lab, key, L, pos):
        g = groups(lab)
        a = [w for w in g['HT'] if len(w.split('-')) == L]; b = [w for w in g['X'] if len(w.split('-')) == L]
        C = subs_pairs(a, b, pos)
        if pos is None:
            return C[key] + C[key[::-1]], C
        return C[key], C
    out = {}
    tabs = sorted(set(d['tab'] for d in docs))
    for name, (key, L, pos) in targets.items():
        obs, Creal = stat(lab, key, L, pos)
        null = []
        for _ in range(300):
            # relabel whole tablets, keeping the HT tablet count
            perm = tabs[:]; rng.shuffle(perm)
            nht = sum(1 for t in tabs if tab_site[t] == 'HT')
            ht = set(perm[:nht])
            null.append(stat(['HT' if d['tab'] in ht else 'X' for d in docs], key, L, pos)[0])
        # decoy sign pairs with the same real count: how many pairs reach the same count
        vals = sorted(Creal.values(), reverse=True)
        out[name] = dict(obs=obs, null=float(np.mean(null)), p=pct_rank(obs, null), rank='%d of %d pairs >= it' % (sum(v >= obs for v in Creal.values()), len(Creal)))
    RES['subs'] = out
    row = ('| LA-67.2c | la13 C (HT KA vs others JA) and la16 C (KU~WA, MA~ME): one-sign substitutions between HT and non-HT word types. Test: count of the substitution '
           'between word types of the same length, vs 300 relabelings of whole tablets as HT / non-HT (HT tablet count kept; fresh seed); decoys = every other sign pair '
           'through the same count | %s | %s |' % (
               '; '.join('%s: %d (relabelled %.2f, P %.3f; %s)' % (k, v['obs'], v['null'], v['p'], v['rank']) for k, v in out.items()),
               '; '.join('%s %s' % (k.split(' ')[1] if 'KA' not in k else 'KA/JA', 'SURVIVES' if v['p'] <= 0.05 else 'KILLED') for k, v in out.items())))
    wlog(OUT, row)


# ---------------------------------------------------------------- la18 SI- prefix within HT
def occ_feats(D):
    F = defaultdict(lambda: np.zeros(6))
    for d in D:
        sup = d['support']
        for ln in lines(d):
            for j, t in enumerate(ln):
                if t[0] != 'W':
                    continue
                nx = ln[j + 1] if j + 1 < len(ln) else None
                f = np.array([sup == 'Tablet', sup == 'Nodule' or sup == 'Roundel', d['lib'], nx is not None and nx[0] == 'N',
                              nx is not None and nx[0] == 'L', j == 0], float)
                F[t[1]] += f
    return F


def t_la18(D):
    HT = [d for d in D if d['site'] == 'HT']
    F = occ_feats(HT)
    cnt = Counter(t[1] for d in HT for t in d['toks'] if t[0] == 'W')
    pairs = [(w, w[3:]) for w in cnt if w.startswith('SI-') and w[3:] in cnt]
    def cos(a, b):
        a, b = F[a], F[b]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    obs = np.mean([cos(a, b) for a, b in pairs]) if pairs else None
    rng = random.Random(seed('la18'))
    words = list(cnt)
    null = []
    for _ in range(2000):
        s = []
        for a, b in pairs:
            ca = [w for w in words if abs(math.log(cnt[w] / cnt[a])) < 0.7 and w != b]
            s.append(cos(rng.choice(ca), b))
        null.append(np.mean(s))
    p = pct_rank(obs, null) if pairs else None
    RES['la18'] = dict(pairs=pairs, obs=obs, null=float(np.mean(null)), p=p)
    row = ('| LA-67.2d | la18 C: SI- is an alternating prefix (SI-X and X share contexts). Kill line: q > 0.1 within Hagia Triada alone. Test within HT: mean context cosine '
           '(support, libation object, followed by number, by logogram, line-initial) of the SI-X / X pairs vs 2,000 decoy pairings of X with a frequency-matched HT word '
           '(fresh seed); one test, so P is q | HT pairs %d (%s); cosine %.3f vs decoys %.3f, P %.3f | %s |' % (
               len(pairs), ', '.join('%s/%s' % p_ for p_ in pairs), obs, np.mean(null), p,
               'SURVIVES (P <= 0.1 within HT)' if p is not None and p <= 0.1 else 'KILLED (P > 0.1 within HT alone)'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la20 *307 top rank
def t_la20(D):
    pre = Counter(); post = Counter(); where = []
    for d in D:
        seq = [t[1] for t in d['toks'] if t[0] in 'WL']
        if '*307' not in seq:
            continue
        i7 = seq.index('*307')
        for j, x in enumerate(seq):
            b = re.sub(r'\+.*', '', x)
            if b in ('OLE', 'VIN', 'GRA', 'CYP', 'OLIV'):
                (pre if j < i7 else post)[b] += 1
                if j < i7 and b in ('OLE', 'VIN'):
                    where.append(d['id'])
    RES['la20'] = dict(before=dict(pre), after=dict(post), where=where)
    row = ('| LA-67.2e | la20 C: *307 is a top-rank (grain-class) commodity. Kill line: *307 written after OLE or VIN. Literal check over all documents | '
           'commodities written BEFORE *307: %s; after: %s; OLE/VIN before *307 on %s | %s |' % (
               dict(pre), dict(post), ', '.join(where) or 'none',
               'KILLED (literal: OLE precedes *307 on %s, the two Petras records of one transfer, la48)' % ', '.join(where) if where else 'SURVIVES literal'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la24 HT9a/b scaling
def t_la24(D):
    A_ = [d for d in admin(D)]
    E = {d['id']: [(w, val(n)) for w, n, _ in entries(d)] for d in A_}
    rng = random.Random(seed('la24'))
    ids = [i for i in E if len(E[i]) >= 3]
    def best_pairs(E):
        res = {}
        for i in range(len(ids)):
            a = {w: v for w, v in E[ids[i]] if v}
            for j in range(i + 1, len(ids)):
                b = {w: v for w, v in E[ids[j]] if v}
                sh = [w for w in a if w in b and a[w] and b[w]]
                if len(sh) < 3:
                    continue
                r = Counter(b[w] / a[w] for w in sh if a[w] is not None and b[w] is not None)
                if not r:
                    continue
                k, c = r.most_common(1)[0]
                if c >= 3 and k != 1:
                    res[(ids[i], ids[j])] = (c, str(k))
        return res
    obs = best_pairs(E)
    null = []
    for _ in range(300):
        Ep = {}
        for i, e in E.items():
            vs = [v for _, v in e]; rng.shuffle(vs)
            Ep[i] = [(w, v) for (w, _), v in zip(e, vs)]
        null.append(len(best_pairs(Ep)))
    p = pct_rank(len(obs), null)
    # J = 1/3 instead of 1/2 (the alternative that breaks it)
    RES['la24'] = dict(obs={'%s|%s' % k: v for k, v in obs.items()}, null=float(np.mean(null)), p=p)
    row = ('| LA-67.2f | la24 C: HT 9b repeats three of HT 9a\'s recipients at exactly 4/5 (needs J = 1/2). Kill line: a re-reading of the J signs on HT 9a (needs a photograph). '
           'Existing-data test: all document pairs with >= 3 shared entry words whose amounts share one exact ratio != 1 (conventional fraction values), vs 300 within-document '
           'shuffles of the amounts (fresh seed) | real pairs %d (%s); shuffled %.2f, P %.3f | %s; the decisive re-reading is UNTESTABLE-NOW |' % (
               len(obs), '; '.join('%s %s x%s on %d' % (k.split('|')[0], k.split('|')[1], v[1], v[0]) for k, v in RES['la24']['obs'].items()), np.mean(null), p,
               'SURVIVES (same-ratio pairs beyond shuffles)' if p <= 0.05 else 'KILLED on existing data (a 3-word common ratio arises in shuffled amounts as often)'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la28 nodule signs unnumbered in HT ledgers
def counted(ln_list, li, j):
    ln = ln_list[li]
    for t in ln[j + 1:]:
        if t[0] == 'N':
            return True
        if t[0] == 'W':
            return False
    if li + 1 < len(ln_list) and ln_list[li + 1] and ln_list[li + 1][0][0] == 'N':
        return True
    return False


def t_la28(D):
    T = [d for d in C1.tablets(D) if d['site'] == 'HT']
    rec = defaultdict(list)
    for d in T:
        L = lines(d)
        for li, ln in enumerate(L):
            for j, t in enumerate(ln):
                if t[0] == 'W' and is_single(t[1]):
                    rec[t[1]].append(counted(L, li, j))
    tg = ['KA', 'KU', 'SI', 'I']
    obs = 1 - np.mean([x for w in tg for x in rec[w]])
    cnt = {w: len(v) for w, v in rec.items()}
    pool = [w for w in rec if w not in tg and cnt[w] >= 2]
    rng = random.Random(seed('la28'))
    null = []
    for _ in range(2000):
        s = []
        for w in tg:
            c = [x for x in pool if abs(math.log(cnt[x] / cnt[w])) < 0.7] or pool
            s += rec[rng.choice(c)]
        null.append(1 - np.mean(s))
    p = pct_rank(obs, null)
    RES['la28'] = dict(obs=obs, null=float(np.mean(null)), p=p, n={w: cnt.get(w, 0) for w in tg})
    row = ('| LA-67.2g | la28 C: KA, KU, SI, I (the HT nodule signs) stand in HT ledgers more often without a number. Kill line: the effect disappearing under a stricter '
           'heading/entry parse. Stricter parse: a sign counts as numbered if a number follows before the next word on its line or opens the next line; vs 2,000 decoy sets of '
           'frequency-matched single signs (fresh seed) | n %s; unnumbered share %.2f vs decoys %.2f, P %.3f | %s |' % (
               RES['la28']['n'], obs, np.mean(null), p, 'SURVIVES the stricter parse' if p <= 0.05 else 'KILLED (gone under the stricter parse)'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la33 grain lists are rations
def t_la33(D):
    A_ = admin(D)
    rng = random.Random(seed('la33'))
    out = {}
    for com in ('GRA', 'OLE', 'CYP', 'VIN'):
        docs = [d for d in A_ if com in logo_bases(d)]
        for excl in (False, True):
            ds = [d for d in docs if not (excl and d['tab'] in ('HT86', 'HT95'))]
            E = [(w, float(n[1]), d['tab']) for d in ds for w, n, _ in entries(d) if n[1] > 0 and w not in ('KU-RO', 'KI-RO', 'PO-TO-KU-RO')]
            def disp(E):
                by = defaultdict(dict)
                for w, v, t in E:
                    by[w].setdefault(t, v)
                r = [abs(math.log(max(x.values()) / min(x.values()))) for x in by.values() if len(x) >= 2]
                return (float(np.mean(r)) if r else None), len(r)
            o, n = disp(E)
            null = []
            for _ in range(1000):
                vs = [v for _, v, _ in E]; rng.shuffle(vs)
                null.append(disp([(w, v, t) for (w, _, t), v in zip(E, vs)])[0])
            out['%s%s' % (com, '-noHT86/95' if excl else '')] = dict(obs=o, n=n, null=float(np.mean([x for x in null if x is not None])) if n else None,
                                                                     p=float((np.sum(np.array([x for x in null if x is not None]) <= o) + 1) / (len(null) + 1)) if n else None)
    RES['la33'] = out
    g = out['GRA-noHT86/95']
    row = ('| LA-67.2h | la33 C: grain lists are rations (would support: near-equal amounts for recurring entry words). Test: mean |log ratio| of amounts for entry words recurring on '
           '2+ tablets within lists of one commodity, vs 1,000 shuffles of amounts among that commodity\'s entries (fresh seed); with and without the HT 86/95 near-copy; decoys = '
           'OLE, CYP, VIN lists through the same test | %s | %s |' % (
               '; '.join('%s words %d disp %s (null %s) P %s' % (k, v['n'], '%.2f' % v['obs'] if v['obs'] is not None else '-', '%.2f' % v['null'] if v['null'] else '-',
                                                              '%.3f' % v['p'] if v['p'] is not None else '-') for k, v in out.items()),
               ('SURVIVES (GRA recurring words near-equal beyond shuffles without HT 86/95)' if g['p'] is not None and g['p'] <= 0.05 else
                'UNTESTABLE-NOW / KILLED: without HT 86/95 too few recurring GRA words' if not g['n'] or g['n'] < 3 else 'KILLED (no near-equal amounts beyond shuffles without HT 86/95)')))
    wlog(OUT, row)


# ---------------------------------------------------------------- la37 word-final alternations
def t_la37(D):
    types = sorted(set(t[1] for d in D for t in d['toks'] if t[0] == 'W' and '-' in t[1]))
    by = defaultdict(set)
    for w in types:
        s = w.split('-')
        for i in range(len(s)):
            by[(len(s), i, tuple(s[:i] + ['_'] + s[i + 1:]))].add(s[i])
    fin = Counter(); med = Counter()
    for (L, i, _), xs in by.items():
        xs = sorted(xs)
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                (fin if i == L - 1 else med)[(xs[a], xs[b])] += 1
    allp = set(fin) | set(med)
    share = {p: fin[p] / (fin[p] + med[p]) for p in allp if fin[p] + med[p] >= 3}
    tg = [('SI', 'TI'), ('TE', 'TI'), ('ME', 'RE')]
    out = {}
    for p in tg:
        out['%s~%s' % p] = dict(fin=fin[p], med=med[p], share=share.get(p), p=pct_rank(share.get(p), [v for q, v in share.items() if q not in tg]) if p in share else None)
    pooled = sum(fin[p] for p in tg) / max(1, sum(fin[p] + med[p] for p in tg))
    dvals = [v for q, v in share.items() if q not in tg]
    RES['la37'] = dict(out=out, pooled=pooled, dec_mean=float(np.mean(dvals)), n_dec=len(dvals))
    row = ('| LA-67.2i | la37 C: SI~TI, TI~TE, RE~ME alternate word-finally. Kill line: the same pairs alternating word-medially as often. Literal counts over word types '
           '(one-sign differences, same length) and final share vs %d decoy sign pairs with >= 3 alternations (mean final share %.2f) | %s; pooled final share %.2f | %s |' % (
               len(dvals), np.mean(dvals), '; '.join('%s final %d / non-final %d (share %s, P %s)' % (k, v['fin'], v['med'], '%.2f' % v['share'] if v['share'] is not None else '-',
                                                                                            '%.3f' % v['p'] if v['p'] is not None else '-') for k, v in out.items()), pooled,
               '; '.join('%s %s' % (k, 'SURVIVES' if v['fin'] > v['med'] and v['p'] is not None and v['p'] <= 0.1 else 'survives literal only (final > medial, not beyond decoys)' if v['fin'] > v['med'] else 'KILLED') for k, v in out.items())))
    wlog(OUT, row)


# ---------------------------------------------------------------- la48 standing quotas
def t_la48(D):
    A_ = admin(D)
    rng = random.Random(seed('la48'))
    E = [(w, (n[1], n[2]), d['tab'], d['site']) for d in A_ for w, n, _ in entries(d) if w not in ('KU-RO', 'KI-RO', 'PO-TO-KU-RO')]
    def rep(E, excl=False):
        by = defaultdict(lambda: defaultdict(set))
        for w, v, t, s in E:
            if excl and t in ('HT86', 'HT95'):
                continue
            by[w][v].add(t)
        return sum(1 for w in by for v, ts in by[w].items() if len(ts) >= 2)
    res = {}
    for excl in (False, True):
        o = rep(E, excl)
        null = []
        bys = defaultdict(list)
        for k, e in enumerate(E):
            bys[e[3]].append(k)
        for _ in range(1000):
            V = [e[1] for e in E]
            for s, ks in bys.items():
                v = [V[k] for k in ks]; rng.shuffle(v)
                for k, x in zip(ks, v):
                    V[k] = x
            null.append(rep([(e[0], V[k], e[2], e[3]) for k, e in enumerate(E)], excl))
        res['noHT86/95' if excl else 'all'] = (o, float(np.mean(null)), pct_rank(o, null))
    RES['la48'] = res
    a = res['noHT86/95']
    row = ('| LA-67.2j | la48 C: entry words are not standing accounts with fixed quotas. Kill line: a word with the same quota on new tablets. Existing-data test: number of '
           '(word, exact amount) pairs on 2+ different tablets vs 1,000 within-site shuffles of amounts among entries (fresh seed), with and without HT 86/95 | %s | %s |' % (
               '; '.join('%s: %d vs %.2f, P %.3f' % (k, *v) for k, v in res.items()),
               'KILLED (repeated quotas beyond shuffles)' if a[2] <= 0.05 else 'SURVIVES (no repeated quotas beyond shuffles once the HT 86/95 copy is out)'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la50 staple order; figs vs olives
def t_la50(D):
    cls = {'GRA': 'G', 'OLE': 'O', 'OLIV': 'O', 'NI': 'F', 'VIN': 'W'}
    LB = 'GOFW'
    pairs = []
    for d in D:
        seq = []
        for t in d['toks']:
            if t[0] in 'WL':
                b = re.sub(r'\+.*', '', t[1])
                if b in cls and cls[b] not in seq:
                    seq.append(cls[b])
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                pairs.append((d['tab'], LB.index(seq[i]) < LB.index(seq[j]), seq[i] + seq[j]))
    conc = np.mean([c for _, c, _ in pairs])
    tabs = sorted(set(t for t, _, _ in pairs))
    rep = 0
    for k in range(NH):
        r = random.Random(seed('la50-%d' % k)); u = tabs[:]; r.shuffle(u); h = set(u[:len(u) // 2])
        a = [c for t, c, _ in pairs if t in h]; b = [c for t, c, _ in pairs if t not in h]
        rep += (np.mean(a) > 0.5) and (np.mean(b) > 0.5)
    ow = sum(1 for _, c, s in pairs if s == 'WO')
    # decoy: all 24 orders, concordance in full data
    import itertools
    allc = {}
    for perm in itertools.permutations('GOFW'):
        allc[''.join(perm)] = np.mean([perm.index(s[0]) < perm.index(s[1]) for _, _, s in pairs])
    rank = sorted(allc.values(), reverse=True).index(allc['GOFW']) + 1
    # figs (NI) and olives (OLIV) on one tablet: site-stratified permutation
    T = defaultdict(lambda: [set(), None])
    for d in D:
        for t in d['toks']:
            if t[0] in 'WL':
                T[d['tab']][0].add(re.sub(r'\+.*', '', t[1]))
        T[d['tab']][1] = d['site']
    F = [t for t, v in T.items() if 'NI' in v[0]]; O = [t for t, v in T.items() if 'OLIV' in v[0]]
    obs = len(set(F) & set(O))
    rng = random.Random(seed('la50-fo'))
    bys = defaultdict(list)
    for t, v in T.items():
        if v[0]:
            bys[v[1]].append(t)
    null = []
    for _ in range(5000):
        f2 = set()
        for s, m in Counter(T[t][1] for t in F).items():
            f2 |= set(rng.sample(bys[s], m))
        null.append(len(f2 & set(O)))
    pfo = float((np.sum(np.array(null) <= obs) + 1) / 5001)
    RES['la50'] = dict(conc=conc, n=len(pairs), rep=rep, ow=ow, rank=rank, fo=(obs, float(np.mean(null)), pfo))
    row = ('| LA-67.2k | la50 C: LA shares the LB staple order grain > olive good > figs > wine; and figs (NI) and olives (OLIV) share fewer tablets than chance. Kill lines: olive goods '
           'after VIN on 3+ baskets; chance level co-occurrence. Tests: literal WINE-before-OLIVE count; pair concordance with GOFW and its rank among all 24 orders; replication in %d fresh '
           'tablet halves (both halves > 0.5); NI/OLIV shared tablets vs 5,000 within-site re-dealings of NI tablets | %d ordered pairs, concordance %.2f, rank %d of 24, halves %d/%d; '
           'VIN before olive good %d; NI & OLIV tablets %d vs %.2f, P(low) %.3f | order: %s; figs/olives: %s |' % (
               NH, len(pairs), conc, rank, rep, NH, ow, obs, np.mean(null), pfo,
               'KILLED (literal: %d VIN-before-olive baskets already)' % ow if ow >= 3 else ('SURVIVES' if rank == 1 and rep >= 0.7 * NH else 'weak (not rank 1 or not replicated)'),
               'SURVIVES' if pfo <= 0.05 else 'KILLED (within-site re-dealing gives as few)'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la57 syllabic commodity words
def t_la57(D):
    tg = ['DI-DE-RU', 'DA-SI-*118', '*28B-NU-MA-RE', 'U-*325-ZA', 'TE-TU']
    out = {}
    for w in tg:
        nn = nl = head = 0
        for d in D:
            L = lines(d)
            for li, ln in enumerate(L):
                for j, t in enumerate(ln):
                    if t[0] == 'W' and t[1] == w:
                        nx = ln[j + 1] if j + 1 < len(ln) else None
                        nn += nx is not None and nx[0] == 'N'
                        nl += nx is not None and nx[0] == 'L'
                        rest = [x for l2 in L[li + 1:] for x in l2]
                        head += li == 0 and not (nx and nx[0] == 'N') and sum(x[0] == 'L' for x in rest) >= 2
        out[w] = (nn, nl, head)
    RES['la57'] = out
    row = ('| LA-67.2l | la57 C-: DI-DE-RU, DA-SI-*118, *28B-NU-MA-RE, U-*325-ZA, TE-TU are syllabic commodity words. Kill line: one of them heading a list of logograms; '
           'support: one in a logogram slot before a number. Literal check over all documents (before a number / before a logogram / heading a list with >= 2 logograms) | %s | %s |' % (
               '; '.join('%s %d/%d/%d' % (w, *v) for w, v in out.items()),
               'KILLED (heads a logogram list: %s)' % ', '.join(w for w, v in out.items() if v[2]) if any(v[2] for v in out.values()) else
               'survives literal, but every use is word + number (entry, name-like: la63); none in a logogram slot. Stays C-'))
    wlog(OUT, row)


# ---------------------------------------------------------------- la60 entry words return with the same commodity
def t_la60(D):
    A_ = [d for d in admin(D) if logo_bases(d)]
    tg = ['KU-PA₃-NU', 'MA-DI', 'SA-RU', 'SA-RO', 'DA-RE', 'TA-I']
    words = Counter()
    for d in A_:
        for w in set(t[1] for t in d['toks'] if t[0] == 'W' and '-' in t[1]):
            words[w] += 1
    cand = [w for w, c in words.items() if c >= 2]
    hit = defaultdict(lambda: [0, 0]); site_hit = [0, 0]
    sited = defaultdict(Counter)
    for k in range(NH):
        tr, te = halves(A_, k)
        site_def = defaultdict(Counter)
        for d in tr:
            for b in logo_bases(d):
                site_def[d['site']][b] += 1
        for w in cand:
            c = Counter(b for d in tr if any(t[1] == w for t in d['toks']) for b in logo_bases(d))
            if not c:
                continue
            g = c.most_common(1)[0][0]
            for d in te:
                if any(t[1] == w for t in d['toks']):
                    hit[w][0] += g in logo_bases(d); hit[w][1] += 1
                    sd = site_def[d['site']].most_common(1)
                    site_hit[0] += bool(sd) and sd[0][0] in logo_bases(d); site_hit[1] += 1
    rate = {w: v[0] / v[1] for w, v in hit.items() if v[1] >= 3}
    tr_ = {w: rate.get(w) for w in tg}
    pooled_t = sum(hit[w][0] for w in tg) / max(1, sum(hit[w][1] for w in tg))
    dec = [w for w in rate if w not in tg]
    pooled_d = sum(hit[w][0] for w in dec) / max(1, sum(hit[w][1] for w in dec))
    # decoy sets of 6 words: pooled rate distribution
    rng = random.Random(seed('la60'))
    ds = []
    for _ in range(2000):
        s = rng.sample(dec, min(6, len(dec)))
        ds.append(sum(hit[w][0] for w in s) / max(1, sum(hit[w][1] for w in s)))
    p = pct_rank(pooled_t, ds)
    RES['la60'] = dict(t=tr_, pooled_t=pooled_t, pooled_d=pooled_d, site=site_hit[0] / max(1, site_hit[1]), p=p)
    row = ('| LA-67.2m | la60 C: entry words return with the same commodity (KU-PA3-NU, MA-DI, SA-RU, SA-RO, DA-RE, TA-I). Kill line: < 1 in 3 right on new uses. '
           'Existing-data test: commodity chosen on a training half, hit = held-out document with the word holds it; %d fresh halves; decoys = all %d other words on 2+ commodity '
           'documents (2,000 random sets of 6) | named words: %s; pooled %.2f; decoy words pooled %.2f (sets of 6: P %.3f); site-default hit %.2f | %s |' % (
               NH, len(dec), ', '.join('%s %s' % (w, '%.2f' % v if v is not None else 'n<3') for w, v in tr_.items()), pooled_t, pooled_d, p, site_hit[0] / max(1, site_hit[1]),
               'SURVIVES the 1-in-3 line and beats decoy words' if pooled_t >= 1 / 3 and p <= 0.05 else
               ('survives the 1-in-3 line but NOT beyond decoy words (any recurring word does as well)' if pooled_t >= 1 / 3 else 'KILLED (< 1 in 3)')))
    wlog(OUT, row)


# ---------------------------------------------------------------- la14 consecutive entries share first sign outside HT
def t_la14(D):
    A_ = [d for d in admin(D) if d['site'] != 'HT']
    rng = random.Random(seed('la14'))
    def stat(docs, pos, shuffle=False):
        a = n = 0
        for d in docs:
            ws = [w for w, _, _ in entries(d) if '-' in w]
            if shuffle:
                rng.shuffle(ws)
            for x, y in zip(ws, ws[1:]):
                n += 1; a += x.split('-')[pos] == y.split('-')[pos]
        return a / n if n else None
    out = {}
    for pos, name in ((0, 'first'), (-1, 'last'), (1, 'second')):
        o = stat(A_, pos)
        null = [stat(A_, pos, True) for _ in range(2000)]
        rep = 0
        for k in range(NH):
            H = halves(A_, k)
            ok = True
            for h in H:
                oh = stat(h, pos); nh = [stat(h, pos, True) for _ in range(200)]
                ok &= pct_rank(oh, nh) <= 0.1
            rep += ok
        out[name] = (o, float(np.mean(null)), pct_rank(o, null), rep)
    RES['la14'] = out
    f = out['first']
    row = ('| LA-67.2n | la14 C: outside HT, consecutive entries share their first sign. Kill: (none stated) - tested by fresh within-document order shuffles (2,000), %d fresh '
           'tablet halves (P <= 0.1 in both), and decoy features (last sign, second sign) through the same test | %s | %s |' % (
               NH, '; '.join('%s sign: %.3f vs %.3f, P %.4f, halves %d/%d' % (k, v[0], v[1], v[2], v[3], NH) for k, v in out.items()),
               'SURVIVES (first sign specific, replicated)' if f[2] <= 0.05 and f[3] >= NH // 2 and out['last'][2] > 0.05 else
               ('KILLED (decoy features do the same: generic list similarity)' if out['last'][2] <= 0.05 or out['second'][2] <= 0.05 else 'KILLED (not replicated in halves)' if f[2] <= 0.05 else 'KILLED')))
    wlog(OUT, row)


if __name__ == '__main__':
    D = docs_all()
    wlog(OUT, '# LA-67 cycle 2: kill tests for the remaining grade-C guesses reachable on existing data (6 Oct 2026). Script tools/la67_c2.py.')
    fs = [t_la12, t_la10, t_subs, t_la18, t_la20, t_la24, t_la28, t_la33, t_la37, t_la48, t_la50, t_la57, t_la60, t_la14]
    only = sys.argv[1:] or None
    for f in fs:
        if only and f.__name__ not in only:
            continue
        f(D); print(f.__name__, 'done', flush=True)
        json.dump(RES, open(os.path.join(CK, 'c2_%s.json' % f.__name__), 'w'), default=str, ensure_ascii=False)
