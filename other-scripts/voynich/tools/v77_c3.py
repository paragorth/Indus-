"""v77 cycle 3: kill tests for the remaining text-only C guesses (one function each; Voynich ZL and IT2a, held-out
leaves where a selection is involved, and the same statistic on fitted generator text).
M1  v6/v10 line-marker sequence (first glyph of line L predicts first glyph of L+1, one glyph wide)
M2  v62 fixed 2-8 line spacing of the chain on even folios (stated kill: vanishes without paragraph-first lines)
M3  v20 op- words least over-dispersed (stated kill: evenness vanishes without paragraph-first lines)
M4  v52 only word edges act like fields (stated kill: goes away if r/s+a- and y+q- are read as misplaced spaces)
M5  v12 ch- and sh- words one word apart, not adjacent (position-preserving null)
M6  v22/v34 each body line self-contained (stated kill: words split across line breaks)
M7  v40 k~t, ch~sh one unit chosen by position and run (stated kill: a section where the shape keeps one form
    regardless of position and run)
M8  v43 one kind of text, quire M the repetitive extreme (stated kill: an M-like island in another hand without the
    repetition shift)
M9  v29 two gallows in one word share the bench
M10 v35/v39 p~f and benched pairs as variants of fewer units (type collapse vs frequency-matched merges)
M11 r3 a marker sign opening ~30-40% of lines (line-initial y/s/t/d words = ordinary word + marker)
M12 v23 Voynich compresses better reversed (stated: the glyph chain also produces it)
M13 v15 length rhythm at period 8 is layout
M14 v21 near_far as scribe variation (would support: a page-wide self-citation window closes it); ch/sh runs as a
    persisting pen or table state
M15 v30 A/B rewrite symmetric
M16 v41 hand 1 drift only between quires
M17 v62 the line is a unit of likeness (would support: line modes predict something outside the line)
"""
import sys, os, json, math, random, collections, bz2, lzma
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v77_lib as L

LOG = open(os.path.join(L.CK, 'c3.log'), 'a')


def log(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def MI(pairs):
    c = Counter(pairs); n = sum(c.values())
    a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs)
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in c.items())


def paragraphs(p):
    out, cur = [], []
    for l in p['lines']:
        if l['ps'] and cur: out.append(cur); cur = []
        cur.append(l)
    if cur: out.append(cur)
    return out


# M1 ---------------------------------------------------------------
def m1(pages, rng, nsh=50):
    def collect(P):
        f1, rest = [], []
        for p in P:
            for par in paragraphs(p):
                for a, b in zip(par, par[1:]):
                    wa, wb = a['w'][0], b['w'][0]
                    f1.append((wa[0], wb[0]))
                    if len(wa) > 1 and len(wb) > 1 and wa[0] == wb[0]:
                        rest.append((wa[1:3], wb[1:3]))
        return f1, rest
    f1, rest = collect(pages)
    o1, orr = MI(f1), MI(rest)
    n1, nr = [], []
    for _ in range(nsh):
        Q = []
        for p in pages:
            nl = []
            for par in paragraphs(p):
                par = [dict(l) for l in par]; heads = [l['w'] for l in par]; rng.shuffle(heads)
                for l, h in zip(par, heads): l['w'] = h
                nl += par
            Q.append(dict(p, lines=nl))
        a, b = collect(Q); n1.append(MI(a)); nr.append(MI(b))
    return dict(mi1=o1 - np.mean(n1), z1=(o1 - np.mean(n1)) / (np.std(n1) + 1e-9), mirest=orr - np.mean(nr),
                zrest=(orr - np.mean(nr)) / (np.std(nr) + 1e-9))


# M2 ---------------------------------------------------------------
def cls3(w):
    g = w[0]
    if g in 'CSktpfKTPF': return 0
    if g in 'oelari': return 1
    return 2


def m2(pages, rng, nsh=100):
    res = {}
    for parity in (0, 1):
        for drop in (False, True):
            def score(labs_pages):
                s = 0
                for labs in labs_pages:
                    n = len(labs)
                    for g in range(2, 9):
                        for i in range(n - 2 * g):
                            if labs[i] == labs[i + g] == labs[i + 2 * g] and labs[i + 1] != labs[i]: s += 1
                return s
            LP = []
            for p in pages:
                m = __import__('re').match(r'f(\d+)', p['id'])
                if not m or int(m.group(1)) % 2 != parity: continue
                LP.append([cls3(l['w'][0]) for l in p['lines'] if not (drop and l['ps'])])
            o = score(LP); nul = []
            for _ in range(nsh):
                Q = [rng.sample(x, len(x)) for x in LP]; nul.append(score(Q))
            res['%s_%s' % ('even' if parity == 0 else 'odd', 'nops' if drop else 'all')] = float((o - np.mean(nul)) / (np.std(nul) + 1e-9))
    return res


# M3 ---------------------------------------------------------------
def m3(pages, rng):
    res = {}
    for drop in (False, True):
        cnt = defaultdict(Counter); tot = Counter()
        for p in pages:
            for l in p['lines']:
                if drop and l['ps']: continue
                for w in l['w']:
                    tot[p['id']] += 1; cnt[w[:2]][p['id']] += 1
        ids = [i for i in tot if tot[i] >= 20]; N = sum(tot[i] for i in ids)
        disp = {}
        for k, c in cnt.items():
            n = sum(c[i] for i in ids)
            if n < 30: continue
            chi = sum((c[i] - n * tot[i] / N) ** 2 / (n * tot[i] / N) for i in ids)
            disp[k] = chi / (len(ids) - 1)
        order = sorted(disp, key=lambda k: disp[k])
        res['nops' if drop else 'all'] = dict(op_n=sum(cnt['op'].values()), op_disp=disp.get('op'), op_rank=(order.index('op') + 1) if 'op' in disp else None,
                                              nclass=len(order), least=order[:3])
    return res


# M4 ---------------------------------------------------------------
def m4(pages, rng, nsh=20):
    """share of the junction coupling (last glyph -> next first glyph, excess over within-line shuffles) that sits
    in the two 'misplaced space' cells r/s.a- and y.q-; if they carry most of it, the edge-field effect goes away
    when they are read as misplaced spaces."""
    CELLS = {('r', 'a'), ('s', 'a'), ('y', 'q')}
    def contrib(P):
        pr = [(a[-1], b[0]) for p in P for l in p['lines'] for a, b in zip(l['w'], l['w'][1:])]
        c = Counter(pr); n = sum(c.values()); A = Counter(x for x, _ in pr); B = Counter(y for _, y in pr)
        tot = 0.0; sub = 0.0
        for (x, y), v in c.items():
            t = v / n * math.log2(v * n / (A[x] * B[y])); tot += t
            if (x, y) in CELLS: sub += t
        return tot, sub
    def shuf(P):
        return [dict(p, lines=[dict(l, w=rng.sample(l['w'], len(l['w']))) for l in p['lines']]) for p in P]
    t, s_ = contrib(pages)
    nul = [contrib(shuf(pages)) for _ in range(nsh)]
    tn = np.mean([x[0] for x in nul]); sn = np.mean([x[1] for x in nul])
    return dict(mi_excess=float(t - tn), cells_excess=float(s_ - sn), frac_in_cells=float((s_ - sn) / (t - tn)))


# M5 ---------------------------------------------------------------
def colshuf(P, rng):
    out = []
    for p in P:
        L_ = [list(l['w']) for l in p['lines']]
        mx = max(len(x) for x in L_)
        for k in range(mx):
            rows = [r for r in range(len(L_)) if len(L_[r]) > k]
            vals = [L_[r][k] for r in rows]; rng.shuffle(vals)
            for r, v in zip(rows, vals): L_[r][k] = v
        out.append(dict(p, lines=[dict(l, w=x) for l, x in zip(p['lines'], L_)]))
    return out


def m5(pages, rng, nsh=100):
    def cnt(P, half=None):
        c1 = c2 = 0
        for p, li, l in L.lines_of(P, half):
            w = l['w']; k = [x[0] if x[0] in 'CS' else None for x in w]
            for i in range(len(w) - 1):
                if k[i] and k[i + 1] and k[i] != k[i + 1]: c1 += 1
                if i + 2 < len(w) and k[i] and k[i + 2] and k[i] != k[i + 2]: c2 += 1
        return c1, c2
    res = {}
    for half in (None, 0, 1):
        a1, a2 = cnt(pages, half); n1, n2 = [], []
        for _ in range(nsh if half is None else nsh // 2):
            b1, b2 = cnt(colshuf(pages, rng), half); n1.append(b1); n2.append(b2)
        res['h%s' % half] = dict(z_lag1=(a1 - np.mean(n1)) / (np.std(n1) + 1e-9), z_lag2=(a2 - np.mean(n2)) / (np.std(n2) + 1e-9))
    return res


# M6 ---------------------------------------------------------------
def m6(pages, rng, nsh=200):
    voc = Counter(w for p in pages for l in p['lines'] for w in l['w'][1:-1])
    V = {w for w, c in voc.items() if c >= 2}
    pairs = []
    for p in pages:
        for a, b in zip(p['lines'], p['lines'][1:]):
            if b['ps']: continue
            pairs.append((p['id'], a['w'][-1], b['w'][0]))
    obs = sum((x + y) in V for _, x, y in pairs)
    bypage = defaultdict(list)
    for pid, x, y in pairs: bypage[pid].append(y)
    nul = []
    for _ in range(nsh):
        s = 0
        for pid, ys in bypage.items():
            xs = [x for q, x, y in pairs if q == pid]; ys2 = rng.sample(ys, len(ys))
            s += sum((x + y) in V for x, y in zip(xs, ys2))
        nul.append(s)
    # truncated-word test: line-final words not in the vocabulary but completed by the next line-initial word
    return dict(n=len(pairs), joins=obs, null=float(np.mean(nul)), z=float((obs - np.mean(nul)) / (np.std(nul) + 1e-9)))


# M7 ---------------------------------------------------------------
def m7(pages, rng):
    out = {}
    for twin, (A, B) in (('k~t', ('k', 't')), ('ch~sh', ('C', 'S'))):
        rows = []
        for p in pages:
            run = None
            for l in p['lines']:
                n = len(l['w'])
                for i, w in enumerate(l['w']):
                    for j, g in enumerate(w):
                        if g in (A, B):
                            shape = w[:j] + '*' + w[j + 1:]
                            pos = 0 if i == 0 else (2 if i == n - 1 else 1)
                            rows.append((p['sec'], p.get('hand', '-'), L.leaf_half(p['id']), shape, pos, l['ps'], run, g == A))
                            run = g == A
        res = {}
        groups = defaultdict(list)
        for r in rows: groups[r[0]].append(r)
        for sec, R in groups.items():
            if len(R) < 300: continue
            ll = {'shape': [], 'full': []}
            for tr in (0, 1):
                trR = [r for r in R if r[2] == tr]; teR = [r for r in R if r[2] != tr]
                cs = defaultdict(lambda: [0.5, 0.5]); cf = defaultdict(lambda: [0.5, 0.5]); cx = defaultdict(lambda: [0.5, 0.5])
                for r in trR:
                    cs[r[3]][r[7]] += 1; cx[(r[4], r[5], r[6])][r[7]] += 1
                for r in teR:
                    a = cs[r[3]]; ps = a[1] / (a[0] + a[1])
                    b = cx[(r[4], r[5], r[6])]; px = b[1] / (b[0] + b[1])
                    # combine shape and context by summing log-odds against the overall rate
                    base = sum(v[1] for v in cs.values()) / sum(v[0] + v[1] for v in cs.values())
                    lo = math.log(ps / (1 - ps)) + math.log(px / (1 - px)) - math.log(base / (1 - base))
                    pf = 1 / (1 + math.exp(-lo))
                    y = r[7]
                    ll['shape'].append(math.log2(ps if y else 1 - ps)); ll['full'].append(math.log2(pf if y else 1 - pf))
            # shape consistency: share of shape tokens taking the shape's majority form (types with >= 5 tokens)
            sh = defaultdict(Counter)
            for r in R: sh[r[3]][r[7]] += 1
            cons = [max(c.values()) / sum(c.values()) for c in sh.values() if sum(c.values()) >= 5]
            res[sec] = dict(n=len(R), gain_bits=float(np.mean(ll['full']) - np.mean(ll['shape'])), shape_consistency=float(np.mean(cons)) if cons else None)
        out[twin] = res
    return out


# M8 ---------------------------------------------------------------
def m8(pages, rng):
    toks = {p['id']: [w for l in p['lines'] for w in l['w']] for p in pages}
    M = [p['id'] for p in pages if p['quire'] == 'M']
    O = [p['id'] for p in pages if p['quire'] != 'M']
    def rep(ws):
        n = len(ws) - 1
        return sum(a == b for a, b in zip(ws, ws[1:])) / max(n, 1), sum(a == c for a, c in zip(ws, ws[2:])) / max(n - 1, 1)
    # M-likeness: log-likelihood ratio of a page under the M unigram vs the rest (leave-page-out for M pages)
    cM = Counter(w for i in M for w in toks[i]); cO = Counter(w for i in O for w in toks[i])
    V = len(set(cM) | set(cO)) + 1
    def llr(i):
        a = cM.copy(); b = cO.copy()
        if i in M: a.subtract(toks[i])
        else: b.subtract(toks[i])
        na, nb = sum(a.values()), sum(b.values())
        return np.mean([math.log((a[w] + .5) / (na + .5 * V)) - math.log((b[w] + .5) / (nb + .5 * V)) for w in toks[i]])
    S = {i: llr(i) for i in toks}
    thr = np.quantile([S[i] for i in M], 0.25)
    hands = {p['id']: p.get('hand', '-') for p in pages}
    island = [i for i in O if S[i] >= thr and hands[i] != '2']
    island2 = [i for i in O if S[i] >= thr]
    def mean_rep(ids):
        if not ids: return None
        r = [rep(toks[i]) for i in ids]
        return [float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))]
    far = [i for i in O if S[i] < np.median([S[j] for j in O])]
    return dict(M=mean_rep(M), island_other_hand=mean_rep(island), island_ids=island, island_any=mean_rep(island2),
                n_island_any=len(island2), rest_low=mean_rep(far))


# M9 ---------------------------------------------------------------
BEN = {'K': 'k', 'T': 't', 'P': 'p', 'F': 'f'}


def m9(pages, rng):
    two = []
    for p in pages:
        for l in p['lines']:
            for w in l['w']:
                g = [c for c in w if c in 'ktpfKTPF']
                if len(g) == 2: two.append(tuple(c in 'KTPF' for c in g))
    n = len(two); b1 = np.mean([x[0] for x in two]); b2 = np.mean([x[1] for x in two])
    agree = np.mean([x[0] == x[1] for x in two]); exp = b1 * b2 + (1 - b1) * (1 - b2)
    both = sum(x[0] and x[1] for x in two)
    return dict(n=n, both_benched=both, exp_both=float(n * b1 * b2), agree=float(agree), exp_agree=float(exp),
                z=float((agree - exp) / math.sqrt(exp * (1 - exp) / max(n, 1))))


# M10 --------------------------------------------------------------
def m10(pages, rng, nrand=300):
    toks = [w for p in pages for l in p['lines'] for w in l['w']]
    gc = Counter(c for w in toks for c in w)
    T0 = len(set(toks))
    def types_after(a, b):
        return len({w.replace(b, a) for w in set(toks)})
    res = {}
    glyphs = [g for g, c in gc.items() if c >= 20]
    for a, b in (('p', 'f'), ('P', 'F'), ('K', 'T'), ('k', 't'), ('C', 'S')):
        obs = T0 - types_after(a, b)
        fa, fb = gc[a], gc[b]
        nul = []
        for _ in range(nrand):
            # frequency-matched pair: glyphs within factor 2 of each twin's count
            ca = [g for g in glyphs if 0.5 * fa <= gc[g] <= 2 * fa and g not in (a, b)] or glyphs
            cb = [g for g in glyphs if 0.5 * fb <= gc[g] <= 2 * fb and g not in (a, b)] or glyphs
            x, y = rng.choice(ca), rng.choice(cb)
            if x == y: continue
            nul.append(T0 - types_after(x, y))
        res['%s=%s' % (a, b)] = dict(collapse=obs, null_mean=float(np.mean(nul)), z=float((obs - np.mean(nul)) / (np.std(nul) + 0.5)),
                                     n_a=fa, n_b=fb)
    return res


# M11 --------------------------------------------------------------
def m11(pages, rng):
    mid = Counter(w for p in pages for l in p['lines'] for w in l['w'][1:-1])
    common = {w for w, c in mid.items() if c >= 5}
    res = {}
    for g in 'ysdto':
        li = [l['w'][0] for p in pages for l in p['lines'] if not l['ps'] and l['w'][0][0] == g and len(l['w'][0]) > 1]
        mi = [w for p in pages for l in p['lines'] for w in l['w'][1:-1] if w[0] == g and len(w) > 1]
        a = np.mean([w[1:] in common for w in li]) if li else float('nan')
        b = np.mean([w[1:] in common for w in mi]) if mi else float('nan')
        res[g] = dict(n_init=len(li), strip_common_init=float(a), strip_common_mid=float(b), ratio=float(a / b) if b else None)
    nl = sum(1 for p in pages for l in p['lines'] if not l['ps'])
    res['share_lines_ysdt'] = sum(1 for p in pages for l in p['lines'] if not l['ps'] and l['w'][0][0] in 'ysdt') / nl
    return res


# M12 --------------------------------------------------------------
def m12(pages, rng):
    lines = [' '.join(l['w']) for p in pages for l in p['lines']]
    fw = '\n'.join(lines).encode(); rv = '\n'.join(x[::-1] for x in lines).encode()
    return dict(bz2=(len(bz2.compress(rv, 9)) - len(bz2.compress(fw, 9))) / len(bz2.compress(fw, 9)),
                xz=(len(lzma.compress(rv, preset=9)) - len(lzma.compress(fw, preset=9))) / len(lzma.compress(fw, preset=9)))


# M13 --------------------------------------------------------------
def m13(pages, rng, nsh=30):
    def acf(P):
        x = np.array([len(w) for p in P for l in p['lines'] for w in l['w']], float); x -= x.mean()
        v = (x * x).mean()
        return np.array([(x[:-k] * x[k:]).mean() / v for k in range(1, 16)])
    a = acf(pages)
    sh = np.mean([acf([dict(p, lines=[dict(l, w=rng.sample(l['w'], len(l['w']))) for l in p['lines']]) for p in pages]) for _ in range(nsh)], 0)
    ll = np.mean([len(l['w']) for p in pages for l in p['lines']])
    k = int(np.argmax(a[5:12])) + 6
    return dict(mean_line_len=float(ll), peak_lag=k, acf_peak=float(a[k - 1]), acf_peak_withinline_shuffle=float(sh[k - 1]),
                acf_6_10=[round(float(v), 4) for v in a[5:10]], shuf_6_10=[round(float(v), 4) for v in sh[5:10]])


# M14 --------------------------------------------------------------
def ed1(a, b):
    if a == b or abs(len(a) - len(b)) > 1: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def m14(pages, rng):
    near = far = nn = nf = 0
    for p in pages:
        L_ = p['lines']
        for i in range(len(L_)):
            for j in range(i + 2, min(len(L_), i + 6)):
                for a in L_[i]['w']:
                    for b in L_[j]['w']:
                        nn += 1; near += ed1(a, b)
    # same-section other-page baseline
    bysec = defaultdict(list)
    for p in pages: bysec[p['sec']].append(p)
    for p in pages:
        oth = [q for q in bysec[p['sec']] if q['id'] != p['id']]
        if not oth: continue
        q = rng.choice(oth)
        for i in range(min(len(p['lines']), len(q['lines']))):
            for j in range(i + 2, min(len(q['lines']), i + 6)):
                for a in p['lines'][i]['w']:
                    for b in q['lines'][j]['w']:
                        nf += 1; far += ed1(a, b)
    adj = sum(a == b for p in pages for l in p['lines'] for a, b in zip(l['w'], l['w'][1:])) / sum(max(0, len(l['w']) - 1) for p in pages for l in p['lines'])
    # ch/sh runs among C/S-initial words in page reading order (Wald-Wolfowitz z)
    zs = []
    seq = []
    for p in pages:
        s = [w[0] for l in p['lines'] for w in l['w'] if w[0] in 'CS']
        seq.append(s)
    R = sum(1 + sum(a != b for a, b in zip(s, s[1:])) for s in seq if s)
    n1 = sum(x == 'C' for s in seq for x in s); n2 = sum(x == 'S' for s in seq for x in s)
    # expected runs per page under within-page permutation
    ER = 0; VR = 0
    for s in seq:
        a = s.count('C'); b = s.count('S'); n = a + b
        if n < 2: ER += n > 0; continue
        e = 1 + 2 * a * b / n; v = 2 * a * b * (2 * a * b - n) / (n * n * (n - 1))
        ER += e; VR += v
    return dict(near_far=near / nn / (far / nf), adj_repeat=adj, chsh_runs_z=float((R - ER) / math.sqrt(VR)))


# M15 --------------------------------------------------------------
def m15(pages, rng):
    A = [w for p in pages if p['lang'] == 'A' for l in p['lines'] for w in l['w']]
    B = [w for p in pages if p['lang'] == 'B' for l in p['lines'] for w in l['w']]
    n = min(len(A), len(B)) // 2
    def model(ws):
        c = defaultdict(Counter)
        for w in ws:
            x = '^^' + w + '$'
            for j in range(2, len(x)): c[x[j - 2:j]][x[j]] += 1
        return c
    def H(ws, c, V=40):
        s = n_ = 0
        for w in ws:
            x = '^^' + w + '$'
            for j in range(2, len(x)):
                k = c.get(x[j - 2:j]); t = sum(k.values()) if k else 0
                s -= math.log2(((k[x[j]] if k else 0) + 0.1) / (t + 0.1 * V)); n_ += 1
        return s / n_
    out = []
    for rep in range(4):
        a = rng.sample(A, 2 * n); b = rng.sample(B, 2 * n)
        mA, mB = model(a[:n]), model(b[:n])
        dAB = H(b[n:], mA) - H(b[n:], mB)   # cost of reading B with A's model
        dBA = H(a[n:], mB) - H(a[n:], mA)
        out.append((dAB, dBA))
    o = np.array(out)
    return dict(A_to_B=float(o[:, 0].mean()), B_to_A=float(o[:, 1].mean()), asym=float((o[:, 0] - o[:, 1]).mean()),
                asym_sd=float((o[:, 0] - o[:, 1]).std()))


# M16 --------------------------------------------------------------
def m16(pages, rng, nperm=500):
    # spelling clock: share of B-type endings (-edy/-dy and e-runs before y) among y-final words, per page
    P = [p for p in pages if p.get('hand') == '1']
    def clock(p):
        ws = [w for l in p['lines'] for w in l['w'] if w.endswith('y')]
        return np.mean([w.endswith('edy') or w.endswith('dy') for w in ws]) if ws else np.nan
    x = np.array([clock(p) for p in P]); q = np.array([p['quire'] for p in P])
    ok = ~np.isnan(x); x, q = x[ok], q[ok]
    def between_share(x, q):
        m = x.mean(); ss = ((x - m) ** 2).sum()
        sb = sum(((x[q == k].mean() - m) ** 2) * (q == k).sum() for k in set(q))
        return sb / ss
    obs = between_share(x, q)
    nul = [between_share(x, np.array(rng.sample(list(q), len(q)))) for _ in range(nperm)]
    # within-quire trend in page order: mean Spearman inside quires
    rs = []
    for k in set(q):
        xx = x[q == k]
        if len(xx) >= 5:
            r = np.corrcoef(np.argsort(np.argsort(xx)), np.arange(len(xx)))[0, 1]; rs.append(r)
    return dict(n=int(len(x)), between_share=float(obs), null_mean=float(np.mean(nul)), p=float(np.mean(np.array(nul) >= obs)),
                within_quire_trend=float(np.mean(rs)) if rs else None)


# M17 --------------------------------------------------------------
def m17(pages, rng):
    """does a page's mix of line modes (first-glyph class of each line's words) predict its illustration section
    beyond Currier language and hand? nearest-centroid, leaf halves, compared with lang|hand-only and generators
    fitted per lang|hand (no section information)."""
    def modes(p):
        c = np.zeros(3)
        for l in p['lines']:
            k = Counter(cls3(w) for w in l['w']).most_common(1)[0][0]; c[k] += 1
        return c / c.sum()
    P = [p for p in pages if p['sec'] in ('H', 'S', 'B', 'P')]
    X = np.array([modes(p) for p in P]); y = np.array([p['sec'] for p in P]); g = np.array(['%s%s' % (p['lang'], p['hand']) for p in P])
    h = np.array([L.leaf_half(p['id']) for p in P])
    acc, accg = [], []
    for tr in (0, 1):
        cents = {(s, gg): X[(y == s) & (g == gg) & (h == tr)].mean(0) for s in set(y) for gg in set(g) if ((y == s) & (g == gg) & (h == tr)).sum() >= 2}
        prior = {gg: Counter(y[(g == gg) & (h == tr)]).most_common(1)[0][0] for gg in set(g) if ((g == gg) & (h == tr)).sum()}
        for i in np.nonzero(h != tr)[0]:
            cand = [(np.abs(X[i] - c).sum(), s) for (s, gg), c in cents.items() if gg == g[i]]
            if not cand or g[i] not in prior: continue
            acc.append(min(cand)[1] == y[i]); accg.append(prior[g[i]] == y[i])
    return dict(acc_modes=float(np.mean(acc)), acc_langhand_only=float(np.mean(accg)), n=len(acc))


TESTS = dict(m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, m6=m6, m7=m7, m8=m8, m9=m9, m10=m10, m11=m11, m12=m12, m13=m13,
             m14=m14, m15=m15, m16=m16, m17=m17)


def job(arg):
    lab, gen, seed, name, key = arg
    rng = random.Random(seed + 7)
    if gen is None: P = L.voy(name)
    elif key == 'lh':
        P = L.generate(name, gen, seed, key=lambda p: '%s|%s' % (p['lang'], p.get('hand', '-')))
    elif gen.startswith('SCPAGE'):
        import v72_lib as V72
        P = L._ungroup(V72.gen_selfcit(L._grouped(L.voy(name), L.gkey), seed=seed, window=100000))
    else: P = L.generate(name, gen, seed)
    out = dict(label=lab, key=key)
    for t, f in TESTS.items():
        if key == 'lh' and t != 'm17': continue
        try:
            out[t] = f(P, rng)
        except Exception as e:
            out[t] = dict(error=repr(e))
    L.jsave('c3_%s_%s.json' % (lab.replace(':', '_'), key), out)
    log('done', lab, key)
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [('ZL3b', None, 0, 'ZL3b', 'sl'), ('IT2a', None, 0, 'IT2a', 'sl')]
    for g in ('STACK', 'SELFCIT', 'SC10', 'MK2', 'JUNC'):
        jobs.append(('ZL:%s:774' % g, g, 774, 'ZL3b', 'sl'))
    jobs.append(('ZL:STACK:775', 'STACK', 775, 'ZL3b', 'sl'))
    jobs.append(('ZL:SCPAGE:774', 'SCPAGE', 774, 'ZL3b', 'sl'))
    for g in ('STACK', 'SELFCIT', 'MK2'):
        jobs.append(('ZL:%s:776' % g, g, 776, 'ZL3b', 'lh'))
    with Pool(2) as pool:
        R = list(pool.imap_unordered(job, jobs))
    L.jsave('c3_all.json', R)
