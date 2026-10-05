"""v60 hot or cold, wet or dry: the complexion slot.

External side: parse real herbals (Old French Circa instans 'Livre des simples medecines', Banckes' Herbal 1552 (Macer /
Circa instans tradition), the English Macer 1543, Lyte's Dodoens 1578, Gerard 1633, Pechey 1694) into entries and read
each entry's complexion (hot/cold/temperate x dry/moist x degree 1-4) and where in the entry it is stated.

Corpus format (shared by Voynich and controls): list of entries; entry = dict(id, page, strat, toks=[str,...]) where a
token is a string of single-character units (Voynich: vlib.glyphs units; controls: opaque symbols).
"""
import os, sys, re, json, random, math, collections, html
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v60_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write('| %s | %s | %s | %s |\n' % (rid, method, result, verdict))


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'))


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ external herbals
def _xml_text(s):
    s = re.sub(r'<note[^>]*>.*?</note>', ' ', s, flags=re.S)
    s = re.sub(r'<g ref="char:EOL(un)?hyphen"/>', '', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s).replace('ſ', 's').replace('̄', '').replace('̃', '')
    return re.sub(r'\s+', ' ', s)


def _tok(s):
    s = s.lower()
    s = s.replace('y e ', 'the ').replace('y t ', 'that ')
    return re.findall(r"[a-zà-ÿ]+|\.[ivx]+\.", s)


HOT = re.compile(r'^(hote?|hoate|whote|hotte|hotter|heateth|heating|chau[zdtsx]|chaudes?|chaut|chaud|[ce]h?auz|cliauz|calid\w*)$')
COLD = re.compile(r'^(colde?|coolde?|cooleth|cooling|froi[tzd]?|froides?|froiz|froyde|colder|frigid\w*)$')
DRY = re.compile(r'^(drye?|drie|drieth|drying|s[eè]s|sec|seches?|sez|ses|seiche|sech|secs|sic+\w*)$')
MOIST = re.compile(r'^(moyste?|moist|moisteth|moistening|moiste|moistes|humide?|humid\w*|wette?|moites?|moistz)$')
TEMP = re.compile(r'^(temperate|temperat|attemper\w*|tempre\w*|tempere\w*)$')
DEG = [re.compile(r'^(first|fyrst|firste|fyrste|premier|permer|primer|premer|prime|\.i\.)$'),
       re.compile(r'^(second|seconde|secunde|segont|segunt|secont|\.ii\.)$'),
       re.compile(r'^(third|thirde|thyrde|thyrd|tierz|tiers|tierce|\.iii\.)$'),
       re.compile(r'^(fourth|fourthe|fowerth|fourt|quart|quarte|\.iiii\.|\.iv\.)$')]


def complexion(toks):
    """First complexion statement in an entry: (q1 in H/C/T, q2 in D/M/-, degree 1-4 or 0, rel position) or None."""
    n = len(toks)
    for i, t in enumerate(toks):
        q1 = 'H' if HOT.match(t) else 'C' if COLD.match(t) else 'T' if TEMP.match(t) else None
        if q1 is None:
            continue
        win = toks[i + 1:i + 9]
        q2 = '-'
        for u in win:
            if DRY.match(u): q2 = 'D'; break
            if MOIST.match(u): q2 = 'M'; break
        # require a degree word or the second quality nearby (avoids 'cold dropsy' etc.)
        deg = 0
        for u in toks[i + 1:i + 14]:
            for k, rx in enumerate(DEG):
                if rx.match(u): deg = k + 1; break
            if deg: break
        if q2 == '-' and not deg and q1 != 'T':
            continue
        return dict(q1=q1, q2=q2, deg=deg, pos=i / max(1, n - 1), i=i)
    return None


def _divs(t, typ):
    out = []
    for m in re.finditer(r'<div[^>]*type="%s"[^>]*>' % typ, t):
        out.append(m.start())
    return [t[a:b] for a, b in zip(out, out[1:] + [len(t)])]


def herbal_entries(src):
    """Return list of entries: dict(name, toks, cx) for one source."""
    if src == 'CI':                                   # Old French Circa instans, numbered entries
        t = open(os.path.join(CK, 'b31364317.txt'), errors='ignore').read()
        parts = re.split(r'\n\s*(\d{1,3})\.\s+(?=[A-Z])', t)
        paras, last = [], 0
        for k in range(1, len(parts) - 1, 2):
            num = int(parts[k])
            if last < num <= last + 12:
                paras.append(parts[k + 1][:3000]); last = num
            elif paras:
                paras[-1] += ' ' + parts[k + 1][:3000]
        ents = []
        for p in paras:
            tk = _tok(p)
            if ents and not any(w in ('est', 'sunt', 'es', 'esft', 'ert') for w in tk[1:4]):
                ents[-1]['toks'] += tk
            else:
                ents.append(dict(name=p[:25], toks=tk))
    else:
        f = dict(BAN='A03040', MAC='A03046', GER='A01622', LYT='A20579', PEC='A53912')[src]
        typ = dict(BAN='herb', MAC='section', GER='chapter', LYT='chapter', PEC='entry')[src]
        t = open(os.path.join(CK, f + '.xml')).read()
        t = t[t.find('<body'):]
        ents = []
        for d in _divs(t, typ):
            s = _xml_text(d)
            ents.append(dict(name=s[:30], toks=_tok(s)))
    ents = [e for e in ents if len(e['toks']) >= 15]
    for e in ents:
        e['cx'] = complexion(e['toks'])
    return ents


SOURCES = ['CI', 'BAN', 'MAC', 'LYT', 'GER', 'PEC']


def external_table(srcs=SOURCES):
    res = {}
    for s in srcs:
        E = herbal_entries(s)
        cx = [e['cx'] for e in E if e['cx']]
        q = collections.Counter(c['q1'] + c['q2'] for c in cx)
        dg = collections.Counter(c['deg'] for c in cx)
        res[s] = dict(n_entries=len(E), n_cx=len(cx), cells=dict(q), deg=dict(dg),
                      pos_mean=float(np.mean([c['pos'] for c in cx])) if cx else None,
                      pos_q=[float(x) for x in np.percentile([c['pos'] for c in cx], [10, 25, 50, 75, 90])] if cx else None)
    return res


# ------------------------------------------------------------------ Voynich
def voynich_entries(name='ZL3b', secs=('H', 'P'), unit='para'):
    import vlib
    L = vlib.load_voynich(name)
    ents, cur = [], None
    for r in L:
        if r['illus'] not in secs:
            cur = None
            continue
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if w and '?' not in w and '*' not in w]
        key = r['folio'] if unit == 'page' else None
        if cur is None or (unit == 'para' and r['para_start']) or (unit == 'page' and cur['page'] != r['folio']):
            cur = dict(id='%s.%d' % (r['folio'], r['n']), page=r['folio'], strat='%s%s%s' % (r['illus'], r['lang'] or '-', r['hand']),
                       toks=[], lines=[])
            ents.append(cur)
        cur['toks'] += ws
        cur['lines'].append(ws)
    return [e for e in ents if len(e['toks']) >= 8]


# ------------------------------------------------------------------ opaque verbose encoding with padding
SYMS = list('abcdefghiklmnopqrstuvxyzBDEFGHJLMNQRUVWXZ')


def encode_entries(ents, seed=60, pad=0.35, max_len=None, page_of=None):
    """Letters -> 1-3 opaque symbols (fixed random code); words kept; filler words inserted at rate `pad`
    (fillers drawn Zipf-like from a pool of 60 opaque words). Returns corpus entries."""
    rng = random.Random(seed)
    letters = sorted(set(c for e in ents for t in e['toks'] for c in t))
    code = {}
    for c in letters:
        code[c] = ''.join(rng.choice(SYMS) for _ in range(rng.choice([1, 1, 2, 2, 3])))
    pool = [''.join(rng.choice(SYMS) for _ in range(rng.randint(3, 6))) for _ in range(60)]
    pw = [1.0 / (k + 1) for k in range(60)]
    out = []
    for j, e in enumerate(ents):
        toks = e['toks'][:max_len] if max_len else e['toks']
        tt, so = [], []
        for t in toks:
            while rng.random() < pad:
                tt.append(rng.choices(pool, pw)[0]); so.append(None)
            tt.append(''.join(code[c] for c in t)); so.append(t)
        out.append(dict(id='e%d' % j, page='p%d' % (j if page_of is None else page_of[j]), strat='X', toks=tt,
                        cx=e.get('cx'), src_toks=toks, src_of=so))
    return out


def plain_entries(path_kind):
    """Negative-control texts as entries (paragraph-like units)."""
    if path_kind == 'CURY':
        t = open(os.path.join(CK, 'cury.txt'), errors='ignore').read()
        a = t.find('I. FOR TO MAKE FURMENTY'); b = t.find('*** END')
        body = t[a:b]
        parts = re.split(r'\n(?=[IVXLC]+\. [A-Z])', body)
        ents = [dict(name=p[:30], toks=_tok(re.sub(r'\[\d+\]', ' ', p))) for p in parts]
    elif path_kind == 'AST':
        t = open(os.path.join(CK, 'A00700.xml')).read()
        t = t[t.find('<body'):]
        ps = re.findall(r'<p>(.*?)</p>', t, flags=re.S)
        ents = [dict(name='', toks=_tok(_xml_text(p))) for p in ps]
    ents = [e for e in ents if len(e['toks']) >= 15]
    for e in ents:
        e['cx'] = complexion(e['toks'])
    return ents


# ------------------------------------------------------------------ nulls
def markov_null(ents, seed=1):
    """Unit-trigram word resynthesis trained per stratum; entry lengths kept (no word identity, no phrase, no slot)."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for e in ents: by[e['strat']] += e['toks']
    tabs = {}
    for k, ws in by.items():
        tri = collections.defaultdict(collections.Counter)
        for w in ws:
            s = '\x01\x01' + w + '\x02'
            for i in range(2, len(s)): tri[s[i - 2:i]][s[i]] += 1
        tabs[k] = {c: (list(v.keys()), list(v.values())) for c, v in tri.items()}
    out = []
    for e in ents:
        tab = tabs[e['strat']]
        nt = []
        for _ in e['toks']:
            ctx, w = '\x01\x01', ''
            while True:
                ks, vs = tab[ctx]; c = rng.choices(ks, vs)[0]
                if c == '\x02' or len(w) > 20: break
                w += c; ctx = ctx[1] + c
            nt.append(w or ks[0])
        out.append(dict(e, toks=nt))
    return out


def wordshuf_null(ents, seed=1):
    """Tokens shuffled across entries within stratum: word frequencies kept, entry membership destroyed."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for e in ents: by[e['strat']] += e['toks']
    for v in by.values(): rng.shuffle(v)
    it = {k: iter(v) for k, v in by.items()}
    return [dict(e, toks=[next(it[e['strat']]) for _ in e['toks']]) for e in ents]


# ------------------------------------------------------------------ items (word classes / word-part classes)
def items_of(tok):
    g = tok
    out = {'w:' + g}
    for k in (1, 2, 3):
        if len(g) > k:
            out.add('p:' + g[:k]); out.add('s:' + g[-k:])
    for k in (2, 3, 4):
        for i in range(1, len(g) - k):
            out.add('m:' + g[i:i + k])
    return out


def presence(ents, rmin=0.06, rmax=0.75, max_items=2500):
    n = len(ents)
    cnt = collections.Counter()
    per = []
    for e in ents:
        s = collections.Counter()
        for t in e['toks']:
            for it in items_of(t): s[it] += 1
        per.append(s)
        for it in s: cnt[it] += 1
    keep = [it for it, c in cnt.items() if rmin * n <= c <= rmax * n]
    keep.sort(key=lambda it: -cnt[it])
    keep = keep[:max_items]
    idx = {it: j for j, it in enumerate(keep)}
    N = np.zeros((n, len(keep)), dtype=np.float32)
    for i, s in enumerate(per):
        for it, c in s.items():
            j = idx.get(it)
            if j is not None: N[i, j] = c
    return keep, N


def nested(a, b):
    sa, sb = a[2:], b[2:]
    return sa in sb or sb in sa


# ------------------------------------------------------------------ lattice fit
# 8 relabelings of a 2x2 table (axes swap x flip rows x flip cols)
def syms(T):
    T = np.asarray(T, dtype=float).reshape(2, 2)
    out = []
    for M in (T, T.T):
        for fr in (False, True):
            for fc in (False, True):
                X = M[::-1] if fr else M
                X = X[:, ::-1] if fc else X
                out.append(X.reshape(4))
    return out


def fit_score(T, Q):
    """Log-likelihood gain (nats) of the observed 2x2 cell counts under external Q vs uniform, best relabeling.
    T: (..., 4) counts with cell order [r0c0, r0c1, r1c0, r1c1]; Q (4,)."""
    lq = np.log(np.asarray(Q) * 4.0)
    best = None
    for perm in _PERMS:
        v = T[..., perm] @ lq
        best = v if best is None else np.maximum(best, v)
    return best


def _perm_list():
    base = np.arange(4)
    ps = []
    for X in syms(base):
        ps.append(np.array(X, dtype=int))
    return ps


_PERMS = _perm_list()


# ------------------------------------------------------------------ cycle 1 engine: order-free 2x2 lattice search
def qvec(cells):
    v = np.array([cells.get('HD', 0), cells.get('HM', 0), cells.get('CD', 0), cells.get('CM', 0)], float) + 0.5
    return v / v.sum()


def strat_expect(X, strat):
    """Expected co-presence under independence within strata: sum_s n_a,s n_b,s / n_s."""
    E = np.zeros((X.shape[1], X.shape[1]))
    for s in set(strat):
        m = np.array([t == s for t in strat])
        Xs = X[m]
        c = Xs.sum(0)
        E += np.outer(c, c) / max(1, m.sum())
    return E


def lattice_search(ents, Q, train, test, K=3000, top=50, min_cov=0.3, min_used=0.25, seed=0):
    keep, N = presence([ents[i] for i in train])
    # presence on test with same items
    idx = {it: j for j, it in enumerate(keep)}
    Nt = np.zeros((len(test), len(keep)), np.float32)
    for r, i in enumerate(test):
        for t in ents[i]['toks']:
            for it in items_of(t):
                j = idx.get(it)
                if j is not None: Nt[r, j] += 1
    X = (N > 0).astype(np.float64); Xt = (Nt > 0).astype(np.float64)
    st = [ents[i]['strat'] for i in train]; stt = [ents[i]['strat'] for i in test]
    O = X.T @ X
    E = strat_expect(X, st)
    Z = (E - O) / np.sqrt(E + 1.0)
    n = X.shape[0]
    cov = (X.sum(0)[:, None] + X.sum(0)[None, :] - O) / n
    I = len(keep)
    iu = np.triu_indices(I, 1)
    z = Z[iu]; c = cov[iu]
    ok = (c >= min_cov) & (z >= 2.0)
    a_, b_ = iu[0][ok], iu[1][ok]
    zz = z[ok]
    nn = np.array([not nested(keep[a], keep[b]) for a, b in zip(a_, b_)], bool)
    a_, b_, zz = a_[nn], b_[nn], zz[nn]
    order = np.argsort(-zz)[:K]
    a_, b_, zz = a_[order], b_[order], zz[order]
    P = len(a_)
    if P < 2:
        return dict(n_pairs=int(P), top=[], n_items=I)
    A1 = (X[:, a_] * (1 - X[:, b_])); A2 = (X[:, b_] * (1 - X[:, a_]))
    T = np.stack([A1.T @ A1, A1.T @ A2, A2.T @ A1, A2.T @ A2], -1)        # (P, P, 4)
    used = T.sum(-1)
    lq = np.log(Q * 4.0)
    best = np.full((P, P), -1e9); bperm = np.zeros((P, P), int)
    for k, perm in enumerate(_PERMS):
        v = T[..., perm] @ lq
        upd = v > best
        best[upd] = v[upd]; bperm[upd] = k
    # forbid same pair / nested cross items / low usage
    mask = np.triu(np.ones((P, P), bool), 1) & (used >= min_used * n)
    S = keep
    nest = np.zeros((P, P), bool)
    strs = [(S[a], S[b]) for a, b in zip(a_, b_)]
    itemset = sorted(set(a_) | set(b_))
    nm = {}
    for x in itemset:
        for y in itemset:
            nm[(x, y)] = nested(S[x], S[y]) or x == y
    for i in range(P):
        for j in range(i + 1, P):
            if nm[(a_[i], a_[j])] or nm[(a_[i], b_[j])] or nm[(b_[i], a_[j])] or nm[(b_[i], b_[j])]:
                nest[i, j] = True
    mask &= ~nest
    score = np.where(mask, best, -1e9)
    n_hyp = int(mask.sum())
    flat = np.argsort(-score, axis=None)[:top]
    res = []
    # test side
    Ot = Xt.T @ Xt; Et = strat_expect(Xt, stt); Zt = (Et - Ot) / np.sqrt(Et + 1.0)
    B1 = (Xt[:, a_] * (1 - Xt[:, b_])); B2 = (Xt[:, b_] * (1 - Xt[:, a_]))
    for f in flat:
        i, j = divmod(int(f), P)
        if score[i, j] < -1e8: break
        perm = _PERMS[bperm[i, j]]
        tt = np.array([B1[:, i] @ B1[:, j], B1[:, i] @ B2[:, j], B2[:, i] @ B1[:, j], B2[:, i] @ B2[:, j]])
        res.append(dict(items=[S[a_[i]], S[b_[i]], S[a_[j]], S[b_[j]]], train_T=T[i, j].tolist(), perm=int(bperm[i, j]),
                        train_ll=float(best[i, j]), train_ll_per=float(best[i, j] / max(1, used[i, j])),
                        z1=float(zz[i]), z2=float(zz[j]), test_T=tt.tolist(),
                        test_ll=float(tt[perm] @ lq), test_ll_per=float((tt[perm] @ lq) / max(1, tt.sum())),
                        test_z1=float(Zt[a_[i], b_[i]]), test_z2=float(Zt[a_[j], b_[j]]),
                        test_used=float(tt.sum() / max(1, len(test)))))
    return dict(n_items=I, n_pairs=int(P), n_hyp=n_hyp, top=res)


def summarize(r):
    t = r['top']
    if not t:
        return dict(n_hyp=r.get('n_hyp', 0), med_test_llp=float('nan'), frac_excl=0.0, best_train_llp=float('nan'))
    return dict(n_hyp=r['n_hyp'], n_pairs=r['n_pairs'],
                best_train_llp=t[0]['train_ll_per'],
                med_train_llp=float(np.median([x['train_ll_per'] for x in t])),
                med_test_llp=float(np.median([x['test_ll_per'] for x in t])),
                frac_excl=float(np.mean([(x['test_z1'] >= 2) and (x['test_z2'] >= 2) for x in t])),
                med_test_used=float(np.median([x['test_used'] for x in t])))


# ------------------------------------------------------------------ engine v2: locked slot pairs (quality1 -> quality2)
def token_items(ents, keep_items=None, rmin=0.05, rmax=0.9, max_items=1200):
    """Token x item boolean matrix (items = word types and word parts), entry boundaries."""
    if keep_items is None:
        cnt = collections.Counter()
        tokc = collections.Counter()
        for e in ents:
            s = set()
            for t in e['toks']:
                its = items_of(t); s |= its; tokc.update(its)
            cnt.update(s)
        n = len(ents)
        keep_items = [it for it, c in cnt.items() if rmin * n <= c <= rmax * n]
        keep_items.sort(key=lambda it: -tokc[it])
        keep_items = keep_items[:max_items]
    idx = {it: j for j, it in enumerate(keep_items)}
    rows, bounds = [], [0]
    for e in ents:
        for t in e['toks']:
            rows.append([idx[it] for it in items_of(t) if it in idx])
        bounds.append(len(rows))
    M = np.zeros((len(rows), len(keep_items)), np.float32)
    for r, js in enumerate(rows):
        M[r, js] = 1
    return keep_items, M, bounds


def same_token_ok(a, b):
    return (a[:2], b[:2]) == ('p:', 's:')


def lock_pairs(keep, M, bounds, w=3, top=3000, zmin=3.0):
    I = len(keep)
    obs = np.zeros((I, I)); exp = np.zeros((I, I))
    for e in range(len(bounds) - 1):
        Me = M[bounds[e]:bounds[e + 1]]
        n = Me.shape[0]
        if n < 2: continue
        c = Me.sum(0)
        for k in range(1, w + 1):
            if n > k: obs += Me[:-k].T @ Me[k:]
        npairs = sum(n - k for k in range(1, w + 1) if n > k)
        exp += np.outer(c, c) * npairs / (n * n)
    st = np.array([[same_token_ok(a, b) for b in keep] for a in keep])
    obs0 = M.T @ M
    rate = M.mean(0)
    exp0 = np.outer(rate, rate) * M.shape[0]
    obs = np.where(st, obs + obs0, obs); exp = np.where(st, exp + exp0, exp)
    Z = (obs - exp) / np.sqrt(exp + 1.0)
    np.fill_diagonal(Z, -1e9)
    nest = np.array([[nested(a, b) for b in keep] for a in keep])
    Z[nest] = -1e9
    flat = np.argsort(-Z, axis=None)[:top]
    out = []
    for f in flat:
        a, b = divmod(int(f), I)
        if Z[a, b] < zmin: break
        out.append((a, b, float(Z[a, b])))
    return out


def event_tensor(M, bounds, A, B, keep, w=3):
    """E[e, i, j] = entry e has item A[i] followed within w tokens (or in the same token, prefix->suffix) by B[j]."""
    n = len(bounds) - 1
    E = np.zeros((n, len(A), len(B)), np.uint8)
    st = np.array([[same_token_ok(keep[a], keep[b]) for b in B] for a in A], np.float32)
    for e in range(n):
        Me = M[bounds[e]:bounds[e + 1]]
        if Me.shape[0] < 1: continue
        Ma, Mb = Me[:, A], Me[:, B]
        acc = (Ma.T @ Mb) * st
        for k in range(1, w + 1):
            if Me.shape[0] > k: acc += Ma[:-k].T @ Mb[k:]
        E[e] = acc > 0
    return E


def kl_best(T, Q):
    """T (..., 4) counts -> (KL(T^||Q) under the best of the 8 relabelings, relabel index)."""
    tot = T.sum(-1, keepdims=True)
    P = (T + 1e-9) / np.maximum(tot, 1e-9)
    best = np.full(T.shape[:-1], 1e9); bi = np.zeros(T.shape[:-1], int)
    lq = np.log(Q)
    lP = np.log(P)
    for k, perm in enumerate(_PERMS):
        v = (P * (lP - lq[perm])).sum(-1)
        upd = v < best
        best[upd] = v[upd]; bi[upd] = k
    return best, bi


from scipy.special import gammaln


def logbf(T, Q, bi):
    """log Bayes factor of counts T under external Q (relabeling bi) against a Dirichlet(1,1,1,1) table."""
    T = np.asarray(T, float)
    lq = np.log(Q)[np.array(_PERMS)[bi]]
    ll = (T * lq).sum(-1)
    n = T.sum(-1)
    lm = gammaln(4.0) - gammaln(n + 4.0) + gammaln(T + 1.0).sum(-1)
    return ll - lm


def kl_fixed(T, Q, k):
    P = np.asarray(T, float) + 1e-9; P = P / P.sum()
    return float((P * (np.log(P) - np.log(Q)[_PERMS[k]])).sum())


def quad_search(E, locks_idx, Q, min_cov=0.3, min_exact=0.5, top=50, max_quads=4_000_000, seed=0, nestA=None, nestB=None):
    """Pairs of lock pairs (a->c) and (b->d); per entry the 2x2 cells [ac, ad, bc, bd]. Gates: coverage (entries with any
    cell) and exactness (share of covered entries with exactly one cell). Rank by KL(T^||Q), best relabeling."""
    rng = np.random.default_rng(seed)
    L_ = np.array(locks_idx, int)
    nL = len(L_)
    ii, jj = np.triu_indices(nL, 1)
    if len(ii) > max_quads:
        sel = rng.choice(len(ii), max_quads, replace=False); ii, jj = ii[sel], jj[sel]
    a, c = L_[ii, 0], L_[ii, 1]; b, d = L_[jj, 0], L_[jj, 1]
    ok = (a != b) & (c != d)
    if nestA is not None:
        ok &= ~nestA[a, b] & ~nestB[c, d]
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    R = dict(kl=[], bi=[], cov=[], ex=[], T=[])
    for s in range(0, len(a), 100_000):
        sa, sb, sc, sd = a[s:s + 100_000], b[s:s + 100_000], c[s:s + 100_000], d[s:s + 100_000]
        X = np.stack([E[:, sa, sc], E[:, sa, sd], E[:, sb, sc], E[:, sb, sd]], -1)
        k = X.sum(-1, dtype=np.int16)
        one = (k == 1)
        cov = (k > 0).mean(0); ex = one.sum(0) / np.maximum((k > 0).sum(0), 1)
        T = (X * one[..., None]).sum(0, dtype=np.int32).astype(float)
        kl, bi = kl_best(T, Q)
        kl = -logbf(T, Q, bi)          # rank by (minus) log Bayes factor: Q vs any 2x2 table (Dirichlet 1)
        R['kl'].append(kl); R['bi'].append(bi); R['cov'].append(cov); R['ex'].append(ex); R['T'].append(T)
    if not R['kl']:
        return dict(n_hyp=0, n_pass=0, top=[])
    for k_ in R: R[k_] = np.concatenate(R[k_])
    good = (R['cov'] >= min_cov) & (R['ex'] >= min_exact) & (R['T'].sum(-1) >= 20)
    score = np.where(good, R['kl'], 1e9)
    order = np.argsort(score)[:top]
    out = []
    for o in order:
        if score[o] >= 1e8: break
        out.append(dict(a=int(a[o]), b=int(b[o]), c=int(c[o]), d=int(d[o]), kl=float(R['kl'][o]), perm=int(R['bi'][o]),
                        cov=float(R['cov'][o]), ex=float(R['ex'][o]), T=R['T'][o].tolist()))
    return dict(n_hyp=int(len(a)), n_pass=int(good.sum()), top=out)


def entry_values(E, q):
    X = np.stack([E[:, q['a'], q['c']], E[:, q['a'], q['d']], E[:, q['b'], q['c']], E[:, q['b'], q['d']]], -1).astype(int)
    k = X.sum(-1)
    return X, k


def rescore(E, q, Q):
    X, k = entry_values(E, q)
    T = (X * (k == 1)[:, None]).sum(0).astype(float)
    return dict(T=T.tolist(), kl=kl_fixed(T, Q, q['perm']) if T.sum() else 9.0,
                lbf=float(logbf(T[None], Q, np.array([q['perm']]))[0]), cov=float((k > 0).mean()),
                ex=float((k == 1).sum() / max(1, (k > 0).sum())))


def run_slot_search(ents, Q, train, test, w=3, n_locks=3000, top=50, seed=0, max_items=1200):
    """Full pipeline: items and locks learned on train entries, quadruples ranked on train, re-scored on test."""
    tr = [ents[i] for i in train]; te = [ents[i] for i in test]
    keep, M, bnd = token_items(tr, max_items=max_items)
    locks = lock_pairs(keep, M, bnd, w=w, top=n_locks)
    if len(locks) < 2:
        return dict(n_locks=len(locks), n_hyp=0, top=[])
    A = sorted(set(l[0] for l in locks)); B = sorted(set(l[1] for l in locks))
    ai = {x: i for i, x in enumerate(A)}; bi = {x: i for i, x in enumerate(B)}
    li = [(ai[l[0]], bi[l[1]]) for l in locks]
    E = event_tensor(M, bnd, A, B, keep, w)
    nestA = np.array([[nested(keep[x], keep[y]) for y in A] for x in A])
    nestB = np.array([[nested(keep[x], keep[y]) for y in B] for x in B])
    R = quad_search(E, li, Q, top=top, seed=seed, nestA=nestA, nestB=nestB)
    _, Mt, bt = token_items(te, keep_items=keep)
    Et = event_tensor(Mt, bt, A, B, keep, w)
    for q in R['top']:
        q['items'] = [keep[A[q['a']]], keep[A[q['b']]], keep[B[q['c']]], keep[B[q['d']]]]
        q['test'] = rescore(Et, q, Q)
    R['n_locks'] = len(locks)
    R['_ctx'] = (keep, A, B, E, Et)
    return R


def summ(R):
    t = R['top']
    if not t:
        return dict(n_locks=R.get('n_locks', 0), n_hyp=R.get('n_hyp', 0), n_pass=R.get('n_pass', 0), best_kl=None,
                    med_test_kl=None, med_test_cov=None)
    return dict(n_locks=R['n_locks'], n_hyp=R['n_hyp'], n_pass=R['n_pass'], best_lbf=round(-t[0]['kl'], 2),
                med_test_lbf=round(float(np.median([q['test']['lbf'] for q in t])), 2),
                max_test_lbf=round(float(np.max([q['test']['lbf'] for q in t])), 2),
                med_train_kl=round(float(np.median([q['kl'] for q in t])), 4),
                med_test_kl=round(float(np.median([q['test']['kl'] for q in t])), 4),
                best_test_kl=round(float(t[0]['test']['kl']), 4),
                med_test_cov=round(float(np.median([q['test']['cov'] for q in t])), 3),
                med_test_ex=round(float(np.median([q['test']['ex'] for q in t])), 3))


def decode_item(ents, item, k=4):
    """For encoded controls: the source words most often carrying an item."""
    c = collections.Counter()
    for e in ents:
        for t, so in zip(e['toks'], e.get('src_of') or []):
            if item in items_of(t): c[so or '<pad>'] += 1
    return [w for w, _ in c.most_common(k)]
