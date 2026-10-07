"""v91: the sparse-channel steganography search.

The generator-like bulk is treated as cover. A 'mask' picks (a) carrier positions, (b) one or two
variant-choice features of the carrier words, (c) a chunk size k and frame offset (k consecutive carrier
symbols form one super-symbol, e.g. 5 bits -> a letter). The extracted super-symbol stream is scored by
held-out bits it carries BEYOND a context model of each choice (hand+language+section, line position,
the rest of the word = 'skeleton', the same feature on the previous token in the line, or on the previous
line's first word for line-initial tokens): a unigram tilt (the message's own symbol frequencies) plus a
bigram tilt (message-like sequence). Fitted on one folio half, scored on the other, both ways.
"""
import os, sys, json, math, random, hashlib
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v91_ckpt')
os.makedirs(CK, exist_ok=True)

GALL = set('ktpfKTPF')


def G(w):
    return vlib.glyphs(w)


# ---------------- features: word -> (symbol or None, skeleton) ----------------
def _first_idx(g, S):
    for i, c in enumerate(g):
        if c in S:
            return i
    return -1


def _run_at(g, ch):
    i = _first_idx(g, {ch})
    if i < 0:
        return -1, 0
    j = i
    while j < len(g) and g[j] == ch:
        j += 1
    return i, j - i


def f_chsh(g):
    i = _first_idx(g, {'C', 'S'})
    if i < 0: return None, None
    return (0 if g[i] == 'C' else 1), ''.join(g[:i]) + '#' + ''.join(g[i + 1:])


def f_kt(g):
    i = _first_idx(g, {'k', 't'})
    if i < 0: return None, None
    return (0 if g[i] == 'k' else 1), ''.join(g[:i]) + '#' + ''.join(g[i + 1:])


def f_pf(g):
    i = _first_idx(g, {'p', 'f'})
    if i < 0: return None, None
    return (0 if g[i] == 'p' else 1), ''.join(g[:i]) + '#' + ''.join(g[i + 1:])


BEN = {'K': 'k', 'T': 't', 'P': 'p', 'F': 'f'}


def f_bench(g):
    i = _first_idx(g, GALL)
    if i < 0: return None, None
    c = g[i]
    return (1 if c in BEN else 0), ''.join(g[:i]) + '#' + BEN.get(c, c) + ''.join(g[i + 1:])


def f_e(g):
    i, n = _run_at(g, 'e')
    if i < 0: return None, None
    return min(n, 3) - 1, ''.join(g[:i]) + '#' + ''.join(g[i + n:])


def f_i(g):
    i, n = _run_at(g, 'i')
    if i < 0: return None, None
    return min(n, 3) - 1, ''.join(g[:i]) + '#' + ''.join(g[i + n:])


def f_q(g):
    h = g[1:] if g and g[0] == 'q' else g
    if len(h) >= 2 and h[0] == 'o' and h[1] in GALL:
        return (1 if g[0] == 'q' else 0), ''.join(h)
    return None, None


def f_ao(g):
    i = _first_idx(g, {'a', 'o'})
    if i < 0: return None, None
    return (0 if g[i] == 'a' else 1), ''.join(g[:i]) + '#' + ''.join(g[i + 1:])


FIN = {'y': 0, 'n': 1, 'l': 2, 'r': 3, 'm': 4, 's': 5}


def f_fin(g):
    if len(g) < 2: return None, None
    return FIN.get(g[-1], 6), ''.join(g[:-1])


def f_rlm(g):
    if len(g) < 2 or g[-1] not in 'rlm': return None, None
    return 'rlm'.index(g[-1]), ''.join(g[:-1])


def f_par(g):
    return len(g) % 2, (g[0], g[-1])


def f_len3(g):
    return len(g) % 3, (g[0], g[-1])


FIRST = {'q': 0, 'o': 1, 'C': 2, 'S': 3, 'd': 4, 'y': 5, 's': 6, 'a': 7, 'l': 8}


def f_first(g):
    if len(g) < 2: return None, None
    c = g[0]
    v = FIRST.get(c, 9 if c in GALL else None)
    if v is None: return None, None
    return v, ''.join(g[1:])


FEATS = [('chsh', f_chsh, 2), ('kt', f_kt, 2), ('pf', f_pf, 2), ('bench', f_bench, 2), ('e', f_e, 3),
         ('i', f_i, 3), ('q', f_q, 2), ('ao', f_ao, 2), ('fin', f_fin, 7), ('rlm', f_rlm, 3), ('par', f_par, 2),
         ('len3', f_len3, 3), ('first', f_first, 10)]
FNAMES = [f[0] for f in FEATS]
FA = {f[0]: f[2] for f in FEATS}


# ---------------- planting toggles (word -> word with feature value v) ----------------
def set_chsh(w, v):
    g = G(w); i = _first_idx(g, {'C', 'S'}); g[i] = 'CS'[v]; return ung(g)


def set_q(w, v):
    g = G(w)
    h = g[1:] if g[0] == 'q' else g
    return ung((['q'] if v else []) + h)


def set_kt(w, v):
    g = G(w); i = _first_idx(g, {'k', 't'}); g[i] = 'kt'[v]; return ung(g)


def set_ao(w, v):
    g = G(w); i = _first_idx(g, {'a', 'o'}); g[i] = 'ao'[v]; return ung(g)


SETTERS = {'chsh': set_chsh, 'q': set_q, 'kt': set_kt, 'ao': set_ao}
UNG = {'C': 'ch', 'S': 'sh', 'T': 'cth', 'K': 'ckh', 'P': 'cph', 'F': 'cfh'}


def ung(g):
    return ''.join(UNG.get(c, c) for c in g)


# ---------------- corpora ----------------
def voy_lines(name):
    L = vlib.load_voynich(name)
    out = []
    for r in L:
        out.append({'folio': r['folio'], 'grp': (r['hand'], r['lang'] or '-', r['illus']), 'ps': bool(r['para_start']),
                    'pe': bool(r['para_end']), 'words': list(r['words'])})
    return out


def shuffle_within_line(lines, seed):
    rng = random.Random(seed)
    out = []
    for r in lines:
        w = list(r['words']); rng.shuffle(w); r = dict(r); r['words'] = w; out.append(r)
    return out


def permute_lines_in_page(lines, seed):
    rng = random.Random(seed)
    byf = defaultdict(list)
    order = []
    for r in lines:
        if r['folio'] not in byf: order.append(r['folio'])
        byf[r['folio']].append(r)
    out = []
    for f in order:
        ls = list(byf[f]); rng.shuffle(ls); out.extend(ls)
    return out


def markov_cover(lines, seed):
    """Junction resynthesis: line-initial word from P(w | grp, para-start), next word from
    P(w | grp, last glyph of previous word); line lengths and metadata kept."""
    rng = random.Random(seed)
    ini = defaultdict(list); nxt = defaultdict(list); allw = defaultdict(list)
    for r in lines:
        ws = r['words']; g = r['grp']
        ini[(g, r['ps'])].append(ws[0])
        for a, b in zip(ws, ws[1:]):
            nxt[(g, G(a)[-1] if a else '')].append(b)
        allw[g].extend(ws)
    out = []
    for r in lines:
        g = r['grp']; n = len(r['words'])
        ws = [rng.choice(ini[(g, r['ps'])])]
        for _ in range(n - 1):
            pool = nxt.get((g, G(ws[-1])[-1]))
            ws.append(rng.choice(pool) if pool else rng.choice(allw[g]))
        rr = dict(r); rr['words'] = ws; out.append(rr)
    return out


# ---------------- token table ----------------
NPRED_FIXED = None


def tokenize(lines):
    toks = []
    prev_first = {}
    for li, r in enumerate(lines):
        ws = r['words']; n = len(ws)
        for wi, w in enumerate(ws):
            toks.append({'li': li, 'wi': wi, 'n': n, 'w': w, 'folio': r['folio'], 'grp': r['grp'], 'ps': r['ps'],
                         'pe': r['pe']})
    return toks


def split_folios(folios, salt='v91'):
    fs = sorted(set(folios))
    h = {f: int(hashlib.sha256((salt + f).encode()).hexdigest(), 16) for f in fs}
    order = sorted(fs, key=lambda f: h[f])
    n = len(order)
    test = set(order[: n // 2]); tr = order[n // 2:]
    A = set(tr[: len(tr) // 2]); B = set(tr[len(tr) // 2:])
    return A, B, test


class Corpus:
    def __init__(self, lines, salt='v91', fit_scope='train', type_salts=None):
        self.lines = lines
        toks = tokenize(lines)
        self.toks = toks
        N = len(toks); self.N = N
        folios = [t['folio'] for t in toks]
        A, B, T = split_folios(folios, salt)
        self.split = np.array([0 if f in A else (1 if f in B else 2) for f in folios], dtype=np.int8)
        gl = [G(t['w']) for t in toks]
        self.sym = {}; self.P = {}
        # previous-token-in-line index, or previous line's first token for line-initial tokens
        prev = np.full(N, -1, dtype=np.int64)
        lstart = {}
        for i, t in enumerate(toks):
            if t['wi'] == 0: lstart[t['li']] = i
        for i, t in enumerate(toks):
            if t['wi'] > 0: prev[i] = i - 1
            elif t['li'] - 1 in lstart and not t['ps']: prev[i] = lstart[t['li'] - 1]
        self.prev = prev
        pos = np.array([(0 if t['wi'] == 0 else (3 if t['wi'] == t['n'] - 1 else (1 if t['wi'] == 1 else 2)))
                        + 4 * int(t['ps']) for t in toks])
        fitm = self.split != 2 if fit_scope == 'train' else np.ones(N, bool)
        for name, fn, Af in FEATS:
            res = [fn(g) for g in gl]
            s = np.array([-1 if v is None else v for v, _ in res], dtype=np.int16)
            sk = [k for _, k in res]
            self.sym[name] = s
            self.P[name] = self._ctx_model(s, sk, pos, Af, fitm)
        self.pos = pos
        self.preds = self._preds(type_salts or [])

    def _ctx_model(self, s, sk, pos, Af, fitm):
        toks = self.toks; N = self.N
        pv = np.array([s[j] if j >= 0 else -2 for j in self.prev])
        keys = []
        for i in range(N):
            g = toks[i]['grp']
            keys.append([None, (pos[i], g[0], g[1], g[2]), (sk[i], pos[i] // 4 * 4 + min(pos[i] % 4, 3)),
                         (sk[i], pos[i], g[1], int(pv[i])), (sk[i], pos[i], g[0], g[1], g[2], int(pv[i]))])
        cnt = [defaultdict(lambda: np.zeros(Af)) for _ in range(5)]
        for i in range(N):
            if s[i] < 0 or not fitm[i]: continue
            for lv in range(5):
                cnt[lv][keys[i][lv]][s[i]] += 1
        P = np.zeros((N, Af))
        base = cnt[0][None] + 1.0
        base = base / base.sum()
        beta = [0, 4.0, 2.0, 2.0, 2.0]
        for i in range(N):
            if s[i] < 0: continue
            p = base
            for lv in range(1, 5):
                c = cnt[lv].get(keys[i][lv])
                if c is None: continue
                if fitm[i]:
                    c = c.copy(); c[s[i]] -= 1  # leave-one-out on fit tokens
                p = (c + beta[lv] * p) / (c.sum() + beta[lv])
            P[i] = p
        return P

    def _preds(self, type_salts):
        toks = self.toks; N = self.N
        wi = np.array([t['wi'] for t in toks]); n = np.array([t['n'] for t in toks])
        ps = np.array([t['ps'] for t in toks]); pe = np.array([t['pe'] for t in toks])
        L = np.array([len(G(t['w'])) for t in toks])
        prevlast = np.array(['^' if t['wi'] == 0 else G(toks[i - 1]['w'])[-1] for i, t in enumerate(toks)])
        # line index within paragraph, token index within paragraph
        lip = np.zeros(N, int); tip = np.zeros(N, int)
        cl = -1; cur = -1; k = 0; lastli = -1
        for i, t in enumerate(toks):
            if t['ps'] and t['wi'] == 0:
                cl = 0; k = 0
            elif t['wi'] == 0:
                cl += 1
            lip[i] = cl; tip[i] = k; k += 1
        P = {'all': np.ones(N, bool), 'w0': wi == 0, 'w1': wi == 1, 'w2': wi == 2, 'w3': wi == 3, 'last': wi == n - 1,
             'last2': wi == n - 2, 'inner': (wi > 0) & (wi < n - 1), 'weven': wi % 2 == 0, 'wodd': wi % 2 == 1,
             'paraline': ps, 'paralast': pe, 'notpara': ~ps, 'lpeven': lip % 2 == 0, 'lpodd': lip % 2 == 1,
             'py': prevlast == 'y', 'pn': prevlast == 'n', 'pl': prevlast == 'l', 'pr': prevlast == 'r',
             'pother': ~np.isin(prevlast, ['y', 'n', 'l', 'r', '^']), 'len3': L <= 3, 'len45': (L >= 4) & (L <= 5),
             'len6': L >= 6, 'tp3a': tip % 3 == 0, 'tp3b': tip % 3 == 1, 'tp3c': tip % 3 == 2, 'tp5': tip % 5 == 0,
             'tp7': tip % 7 == 0}
        for s in type_salts:
            hv = np.array([int(hashlib.md5((s + t['w']).encode()).hexdigest()[:8], 16) for t in toks])
            P['h2_' + s] = hv % 2 == 0
            P['h4_' + s] = hv % 4 == 0
        return P


TYPE_SALTS = ['s%02d' % i for i in range(6)]


def feature_sets():
    fs = [(f,) for f in FNAMES]
    bins = [f for f in FNAMES if FA[f] == 2]
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            fs.append((bins[i], bins[j]))
    for b in ['chsh', 'kt', 'q', 'ao']:
        for t in ['e', 'i', 'rlm', 'len3']:
            fs.append((b, t))
    return fs


def mask_space(pred_names):
    fixed = [p for p in pred_names if not p.startswith('h')]
    hp = [p for p in pred_names if p.startswith('h')]
    combos = [(p,) for p in pred_names]
    for i in range(len(fixed)):
        for j in range(i + 1, len(fixed)):
            combos.append((fixed[i], fixed[j]))
    for h in hp:
        for p in ['all', 'w0', 'last', 'inner', 'paraline', 'wodd', 'weven']:
            combos.append((h, p))
    return combos


# ---------------- scoring ----------------
KAP = 2.0
LN2 = math.log(2)


def stream(C, preds, fset):
    m = np.ones(C.N, bool)
    for p in preds: m &= C.preds[p]
    for f in fset: m &= C.sym[f] >= 0
    idx = np.nonzero(m)[0]
    if len(idx) == 0: return idx, None, None, 1
    A = 1; s = np.zeros(len(idx), np.int64); Pm = np.ones((len(idx), 1))
    for f in fset:
        Af = FA[f]
        s = s * Af + C.sym[f][idx]
        Pm = (Pm[:, :, None] * C.P[f][idx][:, None, :]).reshape(len(idx), -1)
        A *= Af
    return idx, s, Pm, A


def chunk(idx, s, Pm, A, k, o, split):
    n = (len(s) - o) // k
    if n <= 0: return None
    s2 = s[o:o + n * k].reshape(n, k); P2 = Pm[o:o + n * k]
    sup = np.zeros(n, np.int64); Q = np.ones((n, 1))
    for j in range(k):
        sup = sup * A + s2[:, j]
        Q = (Q[:, :, None] * P2[j::k][:, None, :]).reshape(n, -1)
    sp = split[idx[o:o + n * k].reshape(n, k)]
    sp = np.where((sp == sp[:, :1]).all(1), sp[:, 0], -1)
    return sup, Q, sp


def fit_tilts(sup, Q, mfit, Asup):
    """mfit: boolean over chunks used to fit. Returns theta_u (Asup), theta_b (Asup x Asup)."""
    ii = np.nonzero(mfit)[0]
    n_b = np.bincount(sup[ii], minlength=Asup).astype(float)
    E_b = Q[ii].sum(0)
    tu = np.log((n_b + KAP) / (E_b + KAP))
    Qt = Q * np.exp(tu)[None, :]; Qt /= Qt.sum(1, keepdims=True)
    # pairs (t-1, t) both in fit set
    pi = ii[ii >= 1]; pi = pi[mfit[pi - 1]]
    nab = np.zeros((Asup, Asup)); Eab = np.zeros((Asup, Asup))
    np.add.at(nab, (sup[pi - 1], sup[pi]), 1)
    np.add.at(Eab, sup[pi - 1], Qt[pi])
    tb = np.log((nab + KAP) / (Eab + KAP))
    return tu, tb, Qt


def score_dir(sup, Q, mfit, mev, Asup):
    tu, tb, _ = fit_tilts(sup, Q, mfit, Asup)
    ii = np.nonzero(mev)[0]
    if len(ii) < 20: return 0.0, 0.0, len(ii)
    Qe = Q[ii]
    q = Qe * np.exp(tu)[None, :]; q /= q.sum(1, keepdims=True)
    x = sup[ii]
    gu = np.log(q[np.arange(len(ii)), x] / Qe[np.arange(len(ii)), x]).sum()
    # bigram where previous chunk is also in the evaluation set
    has = np.zeros(len(ii), bool)
    ok = ii >= 1
    has[ok] = mev[ii[ok] - 1]
    j = np.nonzero(has)[0]
    if len(j):
        prevx = sup[ii[j] - 1]
        r = q[j] * np.exp(tb[prevx]); r /= r.sum(1, keepdims=True)
        gb = np.log(r[np.arange(len(j)), x[j]] / q[j, x[j]]).sum()
    else:
        gb = 0.0
    return gu / LN2, gb / LN2, len(ii)


def _tilt_tokens(idx, s, Pm, A, fitsplit, split):
    m = np.isin(split[idx], fitsplit)
    n_b = np.bincount(s[m], minlength=A).astype(float)
    E_b = Pm[m].sum(0)
    t = np.exp(np.log((n_b + KAP) / (E_b + KAP)))
    P2 = Pm * t[None, :]
    return P2 / P2.sum(1, keepdims=True)


def score_mask(C, preds, fset, k, o, base=None, mode='disc'):
    """Score = held-out bits of the k-chunk super-symbol stream beyond the context model whose
    per-carrier marginals were first re-tilted to the mask's own carrier frequencies (so a predicate
    that merely selects a biased subset scores 0): u = within-chunk joint structure, b = chunk bigram."""
    if base is None: base = stream(C, preds, fset)
    idx, s, Pm, A = base
    if s is None: return None
    Asup = A ** k
    if Asup > 64 or len(s) < 40 * k: return None
    dirs = [((0,), 0, 1), ((1,), 1, 0)] if mode == 'disc' else [((0, 1), None, 2)]
    tot = {'u': 0.0, 'b': 0.0, 'n': 0}
    for fs_, a, b in dirs:
        Pt = _tilt_tokens(idx, s, Pm, A, fs_, C.split)
        ch = chunk(idx, s, Pt, A, k, o, C.split)
        if ch is None: return None
        sup, Q, sp = ch
        mf = np.isin(sp, fs_); me = sp == b
        if mf.sum() < 30 or me.sum() < 30: return None
        u, bb, n = score_dir(sup, Q, mf, me, Asup)
        tot['u'] += u; tot['b'] += bb; tot['n'] += n
    return tot


def search(C, combos, fsets, seed=0, limit=None, shard=(0, 1), log_every=2000, ck=None):
    rng = random.Random(seed)
    items = [(c, f) for c in combos for f in fsets]
    rng.shuffle(items)
    items = items[shard[0]::shard[1]]
    if limit: items = items[:limit]
    out = []
    for it, (c, f) in enumerate(items):
        base = stream(C, c, f)
        A = 1
        for ff in f: A *= FA[ff]
        if base[1] is None: continue
        for k in range(1, 6):
            if A ** k > 64: break
            for o in range(k):
                r = score_mask(C, c, f, k, o, base)
                if r is None: continue
                out.append((r['u'] + r['b'], r['u'], r['b'], r['n'], '+'.join(c), '+'.join(f), k, o))
        if ck and it % log_every == 0:
            with open(ck + '.progress', 'w') as fh: fh.write('%d/%d\n' % (it, len(items)))
    return out


# ---------------- planted messages ----------------
def letters(key, n, skip=0.3):
    L = vlib.load_ref(key, max_words=400000, skip_frac=skip)
    s = ''.join(''.join(r['words']) for r in L)
    s = ''.join(c for c in s.lower() if 'a' <= c <= 'z')
    return s[:n]


def plant(lines, preds_fn, feat, msg, seed, nbits=5):
    """Embed msg letters (random nbits code) as the value of binary feature `feat` at carrier tokens
    (tokens selected by preds_fn(tokens-corpus) & feature applicable), reading order."""
    rng = random.Random(seed)
    alph = sorted(set(msg))
    codes = rng.sample(range(2 ** nbits), len(alph))
    code = dict(zip(alph, codes))
    bits = []
    for c in msg:
        v = code[c]
        bits.extend([(v >> (nbits - 1 - j)) & 1 for j in range(nbits)])
    C = Corpus(lines, fit_scope='all', type_salts=TYPE_SALTS)
    m = preds_fn(C) & (C.sym[feat] >= 0)
    idx = np.nonzero(m)[0]
    nb = (len(idx) // nbits) * nbits
    bits = bits[:nb]
    new = [dict(r, words=list(r['words'])) for r in lines]
    fn = dict((f[0], f[1]) for f in FEATS)[feat]
    for j, b in zip(idx, bits):
        t = C.toks[j]
        w = SETTERS[feat](t['w'], b)
        assert fn(G(w))[0] == b, (t['w'], w, b)
        new[t['li']]['words'][t['wi']] = w
    return new, {'code': code, 'nletters': len(bits) // nbits, 'ncarriers': int(len(idx)), 'msg': msg[:len(bits) // nbits]}


# ---------------- substitution solver for decoded streams ----------------
def bigram_model(text):
    al = sorted(set(text))
    ix = {c: i for i, c in enumerate(al)}
    M = np.ones((len(al), len(al)))
    for a, b in zip(text, text[1:]): M[ix[a], ix[b]] += 1
    M = np.log(M / M.sum(1, keepdims=True))
    u = np.bincount([ix[c] for c in text], minlength=len(al)) + 1.0
    return al, M, np.log(u / u.sum())


def solve_sub(seq, model, iters=6000, restarts=6, seed=0):
    """seq: ints (cipher symbols). One-to-one substitution onto letters (extra symbols beyond the
    alphabet share letters). Swap hill-climb on log P(text) = unigram start + bigram chain."""
    al, M, U = model
    rng = np.random.default_rng(seed)
    syms = np.unique(seq); K = len(al); S = len(syms)
    sx = np.searchsorted(syms, seq)
    freq = np.bincount(sx, minlength=S)
    order_u = np.argsort(-U)
    slots = max(S, K)
    def ll(perm):
        y = perm[:S][sx] % K
        return M[y[:-1], y[1:]].sum() + U[y].sum() * 0.0
    best = (-1e18, None)
    for r in range(restarts):
        perm = np.arange(slots)
        if r == 0:
            rk = np.argsort(-freq)
            letters_by_freq = list(order_u) + [i for i in range(K, slots)]
            perm = np.zeros(slots, int); used = []
            for i, s_ in enumerate(rk): perm[s_] = letters_by_freq[i]
            rest = [x for x in range(slots) if x not in set(perm[:S])]
            perm[S:] = rest[:slots - S]
        else:
            perm = rng.permutation(slots)
        cur = ll(perm)
        for it in range(iters):
            i, j = rng.integers(slots, size=2)
            if i == j or (i >= S and j >= S): continue
            perm[i], perm[j] = perm[j], perm[i]
            v = ll(perm)
            if v >= cur: cur = v
            else: perm[i], perm[j] = perm[j], perm[i]
        if cur > best[0]: best = (cur, perm.copy())
    y = best[1][:S][sx] % K
    return best[0] / max(1, len(seq) - 1), ''.join(al[i] for i in y)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


TYPE_SALTS2 = TYPE_SALTS + ['t%02d' % i for i in range(14)]


def all_items(C, n_random=40000, seed=91):
    combos = mask_space([p for p in C.preds if not p.startswith('h') or p[3:] in TYPE_SALTS])
    fsets = feature_sets()
    items = [(c, f) for c in combos for f in fsets]
    rng = random.Random(seed)
    pn = sorted(C.preds)
    seen = set(items)
    while len(items) < len(combos) * len(fsets) + n_random:
        c = tuple(sorted(rng.sample(pn, rng.choice([2, 3]))))
        f = rng.choice(fsets)
        if (c, f) in seen: continue
        seen.add((c, f)); items.append((c, f))
    return items


def run_items(C, items):
    out = []
    for c, f in items:
        base = stream(C, c, f)
        if base[1] is None: continue
        A = base[3]
        for k in range(1, 6):
            if A ** k > 64: break
            for o in range(k):
                r = score_mask(C, c, f, k, o, base)
                if r is None: continue
                out.append((round(r['b'], 3), round(r['u'], 2), r['n'], '+'.join(c), '+'.join(f), k, o))
    return out


def build(name, seed=0):
    """Named corpora for v91."""
    import pickle
    p = os.path.join(CK, 'lines_%s.pkl' % name)
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    zl = voy_lines('ZL3b')
    meta = {}
    if name == 'ZL': L = zl
    elif name == 'IT': L = voy_lines('IT2a')
    elif name == 'N1': L = shuffle_within_line(zl, 11)
    elif name == 'N2': L = permute_lines_in_page(zl, 12)
    elif name == 'N3': L = markov_cover(zl, 13)
    elif name == 'P1':
        msg = letters('Latin-Caesar', 4000)
        L, meta = plant(markov_cover(zl, 21), lambda C: C.preds['wodd'], 'chsh', msg, 21)
        meta['mask'] = ('wodd', 'chsh', 5)
    elif name == 'P2':
        msg = letters('Italian-Manzoni', 4000)
        L, meta = plant(markov_cover(zl, 22), lambda C: C.preds['all'], 'q', msg, 22)
        meta['mask'] = ('all', 'q', 5)
    elif name == 'P3':  # real Voynich cover, sparse: second word of each line
        msg = letters('Latin-Descartes', 4000)
        L, meta = plant(zl, lambda C: C.preds['w1'], 'chsh', msg, 23)
        meta['mask'] = ('w1', 'chsh', 5)
    elif name == 'P4':  # real Voynich cover, every 3rd word of the paragraph (offset 1), line-internal only
        msg = letters('Italian-Dante', 4000)
        L, meta = plant(zl, lambda C: C.preds['tp3b'] & C.preds['inner'], 'ao', msg, 24)
        meta['mask'] = ('inner+tp3b', 'ao', 5)
    elif name == 'N3b':
        L = markov_cover(zl, 113)
    elif name == 'N2b':
        L = permute_lines_in_page(zl, 112)
    elif name == 'ITN1':
        L = shuffle_within_line(voy_lines('IT2a'), 31)
    else:
        raise ValueError(name)
    pickle.dump((L, meta), open(p, 'wb'))
    return L, meta


def decode_stream(C, preds, fset, k, o, splits=(0, 1, 2)):
    idx, s, Pm, A = stream(C, preds, fset)
    n = (len(s) - o) // k
    s2 = s[o:o + n * k].reshape(n, k)
    sup = np.zeros(n, np.int64)
    for j in range(k): sup = sup * A + s2[:, j]
    sp = C.split[idx[o:o + n * k].reshape(n, k)][:, 0]
    return sup[np.isin(sp, splits)]


def letter_acc(dec, msg):
    m = min(len(dec), len(msg))
    return sum(a == b for a, b in zip(dec[:m], msg[:m])) / max(1, m)
