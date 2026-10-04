#!/usr/bin/env python3
"""LA-19 shared code: THE PALACE SPOKE MANY TONGUES.

Arrow in the dark: the sign-groups of Linear A (above all the name-like entry words) come from several languages at
once, as in a trading port. We fit mixtures of K sign-level phonotactic models (bigram over sign identities with
BOS/EOS, i.e. initial-sign and final-sign preferences and length) to the list of word TYPES by collapsed Gibbs, for
K = 1..8, choose K by 5-fold held-out likelihood (paired 2-SE rule) and stability across restarts, and ask whether
the inferred 'languages' have support the model never saw (site, support, commodity, number size, position).
No sound values: signs are opaque identities. Sound values are used only to label the Linear B control.
"""
import os, sys, json, math, random, collections, ctypes, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(HERE, '..', 'data')
CK = os.path.join(LAD, 'la19_ckpt')
OUT = os.path.join(LAD, 'la19')
LOOPS = os.path.join(HERE, '..', 'loops')
C56 = '/home/user/Indus-/data/derived/dark/loop56_corpora/'
for d in (CK, OUT): os.makedirs(d, exist_ok=True)

_lib = ctypes.CDLL(os.path.join(HERE, 'la19_core.so'))
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C')
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C')
_lib.gibbs.restype = ctypes.c_double
_lib.gibbs.argtypes = [ctypes.c_int, _ip, _ip, ctypes.c_int, _ip, _ip, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                       ctypes.c_double, _dp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong, _ip, _dp, _dp,
                       ctypes.c_void_p]


def norm(s):
    return s.replace('₂', '2').replace('₃', '3').upper()


# ------------------------------------------------------------------ corpora
def la_tokens():
    """Every LA word token (>= 2 signs) with context attributes."""
    C = json.load(open(os.path.join(LAD, 'corpus.json')))
    out = []
    for c in C:
        toks = c['tokens']
        first_word_seen = False
        line_start = True
        for i, t in enumerate(toks):
            if t['t'] == 'nl': line_start = True; continue
            if t['t'] != 'word':
                if t['t'] != 'div': line_start = False
                continue
            w = tuple(norm(x) for x in t['s'])
            # what follows on the line
            nxt, logo, num = 'END', None, None
            for u in toks[i + 1:]:
                if u['t'] == 'div': continue
                if u['t'] == 'nl': nxt = 'NL'; break
                if u['t'] == 'word': nxt = 'WORD'; break
                if u['t'] == 'logo':
                    nxt = 'LOGO'; logo = u['v'].split('+')[0]
                    continue  # look for the number after the logogram
                if u['t'] == 'num':
                    num = u['v']
                    if nxt != 'LOGO': nxt = 'NUM'
                    break
                break
            if len(w) >= 2:
                out.append(dict(w=w, doc=c['id'], site=c['site'], support=c['support'], scribe=c.get('scribe', ''),
                                nxt=nxt, logo=logo, num=num, docfirst=not first_word_seen, linefirst=line_start))
            first_word_seen = True
            line_start = False
    return out


def la_types(entry_only=False):
    T = la_tokens()
    if entry_only: T = [t for t in T if t['num'] is not None or t['logo'] is not None]
    return sorted(set(t['w'] for t in T))


def jl(path):
    return [tuple(x for x in json.loads(l)['seq']) for l in open(path)]


def lb_names():
    """LB personnel names (DAMOS, deduplicated) -> list of (tuple, site label KN/PY/both)."""
    kn = set(tuple(x.upper() for x in s) for s in jl(C56 + 'linb_personnel_KN.jsonl') if '' not in s)
    py = set(tuple(x.upper() for x in s) for s in jl(C56 + 'linb_personnel_PY.jsonl') if '' not in s)
    allw = set(tuple(x.upper() for x in s) for s in jl(C56 + 'linb_personnel_dedup.jsonl') if '' not in s)
    out = []
    for w in sorted(allw):
        if len(w) < 2: continue
        out.append((w, 'KN' if (w in kn and w not in py) else 'PY' if (w in py and w not in kn) else 'BOTH' if w in kn else 'OTHER'))
    return out


GREEK_O = {'O', 'RO', 'TO', 'NO', 'WO', 'KO', 'MO', 'SO', 'DO', 'PO', 'QO', 'JO', 'ZO'}


def lb_greekness(w):
    """CONTROL LABEL ONLY (uses LB sound values): 'G' Greek-looking (nominative -os / -eus shapes),
    'N' non-Greek-looking (final -i / -u other than -e-u, or Cu-Cu reduplication), else '?'."""
    if len(w) >= 2 and w[-1] == 'U' and w[-2].endswith('E'): return 'G'
    if w[-1] in GREEK_O: return 'G'
    if w[-1].endswith('I') or (w[-1].endswith('U') and w[-1] != 'U'): return 'N'
    if w[-1] == 'U': return 'N'
    return '?'


def ur3_names():
    L = [s for s in jl(C56 + 'ur3_names_dedup.jsonl') if not any(x.startswith('(') or x in ('blank', 'line', '$)') for x in s)]
    return sorted(set(s for s in L if len(s) >= 2))


def ob_names():
    return sorted(set(s for s in jl(C56 + 'ob_names_dedup.jsonl') if len(s) >= 2))


def la11_lang(code, scheme='DROP'):
    d = json.load(open(os.path.join(LAD, 'la11', 'lang', code + '.json')))
    s = d['schemes'][scheme]
    test = sorted(set(tuple(x) for x in s['test'] if len(x) >= 2))
    return d, s, test


def la11_codes():
    return sorted(f[:-5] for f in os.listdir(os.path.join(LAD, 'la11', 'lang')) if f.endswith('.json'))


def bigram_gen(bi, n, rnd, minlen=2, maxlen=10, exclude=()):
    """Generate n distinct words from a ^/$ bigram count table {'a\tb': c}."""
    tr = collections.defaultdict(list)
    for k, c in bi.items():
        a, b = k.split('\t'); tr[a].append((b, c))
    tab = {a: ([b for b, _ in v], np.cumsum([c for _, c in v], dtype=float)) for a, v in tr.items()}
    out, seen, tries = [], set(exclude), 0
    while len(out) < n and tries < n * 200:
        tries += 1
        w, p = [], '^'
        while True:
            bs, cs = tab[p]
            x = bs[int(np.searchsorted(cs, rnd.random() * cs[-1], side='right'))]
            if x == '$' or len(w) >= maxlen: break
            w.append(x); p = x
        w = tuple(w)
        if len(w) >= minlen and w not in seen:
            seen.add(w); out.append(w)
    return out


# ------------------------------------------------------------------ encoding
class Enc:
    def __init__(self, words):
        self.signs = sorted(set(s for w in words for s in w))
        self.ix = {s: i for i, s in enumerate(self.signs)}
        self.V = len(self.signs)

    def pack(self, words):
        off = np.zeros(len(words) + 1, np.int32)
        sym = []
        for i, w in enumerate(words):
            sym.extend(self.ix[s] for s in w); off[i + 1] = len(sym)
        return off, np.array(sym if sym else [0], np.int32)

    def gvec(self, words, eps=0.5):
        g = np.full(self.V + 1, eps)
        for w in words:
            for s in w: g[self.ix[s]] += 1
            g[self.V] += 1
        return g / g.sum()


def gibbs(enc, train, held, K, alpha=1.0, beta=5.0, sweeps=200, burn=100, thin=5, seed=0, g=None, zinit=None):
    off, sym = enc.pack(train)
    if held: hoff, hsym = enc.pack(held)
    else: hoff, hsym = np.zeros(1, np.int32), np.zeros(1, np.int32)
    if g is None: g = enc.gvec(train)
    z = np.zeros(len(train), np.int32)
    hl = np.zeros(max(1, len(held)), np.float64)
    zp = np.zeros(len(train) * K, np.float64)
    zi = None
    if zinit is not None:
        zarr = np.ascontiguousarray(zinit, np.int32); zi = zarr.ctypes.data_as(ctypes.c_void_p)
    ll = _lib.gibbs(len(train), off, sym, len(held), hoff, hsym, enc.V, K, alpha, beta, np.ascontiguousarray(g),
                    sweeps, burn, thin, seed, z, hl, zp, zi)
    return dict(ll=ll, z=z, held=hl[:len(held)], zprob=zp.reshape(len(train), K))


# ------------------------------------------------------------------ single-language higher-order baseline
def ngram_heldout(enc, train, held, b1, b2, g=None):
    """K=1 interpolated trigram (Dirichlet backoff to bigram, bigram to g). Same BOS/EOS convention."""
    if g is None: g = enc.gvec(train)
    V = enc.V; B, E = V, V
    c2 = collections.Counter(); r2 = collections.Counter(); c3 = collections.Counter(); r3 = collections.Counter()
    for w in train:
        s = [B, B] + [enc.ix[x] for x in w] + [E]
        for j in range(2, len(s)):
            a, b, x = s[j - 2], s[j - 1], s[j]
            c2[(b, x)] += 1; r2[b] += 1; c3[(a, b, x)] += 1; r3[(a, b)] += 1
    out = []
    for w in held:
        s = [B, B] + [enc.ix[x] for x in w] + [E]; lp = 0
        for j in range(2, len(s)):
            a, b, x = s[j - 2], s[j - 1], s[j]
            p2 = (c2[(b, x)] + b1 * g[x]) / (r2[b] + b1)
            p3 = (c3[(a, b, x)] + b2 * p2) / (r3[(a, b)] + b2)
            lp += math.log(p3)
        out.append(lp)
    return np.array(out)


# ------------------------------------------------------------------ CV and K selection
BETAS = (1.0, 3.0, 10.0, 30.0)


def folds_of(n, nf, seed):
    r = np.random.RandomState(seed); p = r.permutation(n)
    return [np.sort(p[i::nf]) for i in range(nf)]


def cv_mixture(words, Ks=range(1, 9), betas=BETAS, nf=5, restarts=3, sweeps=200, burn=100, thin=5, seed=0, alpha=1.0):
    """-> dict K -> dict(best_beta, per-word held-out lp (array, aligned to words), total)."""
    enc = Enc(words)
    F = folds_of(len(words), nf, seed)
    res = {}
    for K in Ks:
        best = None
        for beta in (betas if K > 0 else betas):
            lpw = np.zeros(len(words))
            for fi, te in enumerate(F):
                tr = np.setdiff1d(np.arange(len(words)), te)
                trw = [words[i] for i in tr]; tew = [words[i] for i in te]
                g = Enc.gvec(enc, trw)
                acc = []
                for r in range(restarts if K > 1 else 1):
                    o = gibbs(enc, trw, tew, K, alpha, beta, sweeps, burn, thin, seed * 7919 + fi * 101 + r * 13 + K, g=g)
                    acc.append(o['held'])
                lpw[te] = np.mean(acc, 0)
            tot = lpw.sum()
            if best is None or tot > best['total']: best = dict(beta=beta, lpw=lpw, total=tot)
        res[K] = best
    return res


def cv_trigram(words, nf=5, seed=0, grid=((1, 1), (3, 3), (10, 3), (3, 10), (10, 10), (30, 10), (10, 30), (30, 30), (3, 1), (10, 1))):
    enc = Enc(words); F = folds_of(len(words), nf, seed); best = None
    for b1, b2 in grid:
        lpw = np.zeros(len(words))
        for te in F:
            tr = np.setdiff1d(np.arange(len(words)), te)
            trw = [words[i] for i in tr]
            lpw[te] = ngram_heldout(enc, trw, [words[i] for i in te], b1, b2, g=Enc.gvec(enc, trw))
        if best is None or lpw.sum() > best['total']: best = dict(b=(b1, b2), lpw=lpw, total=lpw.sum())
    return best


def choose_K(res, nse=2.0):
    Ks = sorted(res); tots = {K: res[K]['total'] for K in Ks}
    Kb = max(Ks, key=lambda k: tots[k])
    for K in Ks:
        d = res[K]['lpw'] - res[Kb]['lpw']
        se = d.std(ddof=1) * math.sqrt(len(d)) if K != Kb else 0.0
        if d.sum() >= -nse * se:
            return K, Kb
    return Kb, Kb


def ari(a, b):
    from math import comb
    ct = collections.Counter(zip(a, b)); A = collections.Counter(a); B = collections.Counter(b); n = len(a)
    s = sum(comb(v, 2) for v in ct.values()); sa = sum(comb(v, 2) for v in A.values()); sb = sum(comb(v, 2) for v in B.values())
    e = sa * sb / comb(n, 2); m = (sa + sb) / 2
    return (s - e) / (m - e) if m != e else 1.0


def stability(words, K, beta, restarts=12, sweeps=300, burn=150, thin=5, seed=0):
    enc = Enc(words); maps, lls, zps = [], [], []
    for r in range(restarts):
        o = gibbs(enc, words, [], K, 1.0, beta, sweeps, burn, thin, seed * 1000 + r)
        maps.append(o['zprob'].argmax(1)); lls.append(o['ll']); zps.append(o['zprob'])
    A = [ari(maps[i], maps[j]) for i in range(restarts) for j in range(i + 1, restarts)]
    return dict(ari_mean=float(np.mean(A)) if A else 1.0, ari_min=float(np.min(A)) if A else 1.0,
                maps=maps, lls=lls, zprobs=zps)


def align_to(ref, z, K):
    """relabel z to best match ref (Hungarian on overlap)."""
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((K, K))
    for a, b in zip(ref, z): M[a, b] += 1
    r, c = linear_sum_assignment(-M)
    mp = {cc: rr for rr, cc in zip(r, c)}
    return np.array([mp[x] for x in z])


def consensus(stab, K):
    """restart with highest joint ll as reference; soft membership = mean of aligned zprobs."""
    i0 = int(np.argmax(stab['lls'])); ref = stab['maps'][i0]
    P = np.zeros_like(stab['zprobs'][0])
    for zp, m in zip(stab['zprobs'], stab['maps']):
        from scipy.optimize import linear_sum_assignment
        M = np.zeros((K, K))
        for a, b in zip(ref, m): M[a, b] += 1
        r, c = linear_sum_assignment(-M)
        P[:, r] += zp[:, c]
    P /= len(stab['zprobs'])
    return P


def summary_row(name, res, tri=None):
    K, Kb = choose_K(res)
    n = len(res[1]['lpw'])
    gain = {k: (res[k]['total'] - res[1]['total']) / n for k in sorted(res)}
    out = dict(name=name, n=n, K_sel=K, K_best=Kb, gain_per_word={k: round(v, 4) for k, v in gain.items()},
               beta={k: res[k]['beta'] for k in res}, ll1=round(res[1]['total'], 1))
    if tri is not None:
        d = res[Kb]['lpw'] - tri['lpw']
        out['tri_minus_bestmix_per_word'] = round(-d.mean(), 4)
        out['tri_vs_mix_z'] = round(float(-d.sum() / (d.std(ddof=1) * math.sqrt(len(d)))), 2)
        out['tri_b'] = tri['b']
        d1 = res[K]['lpw'] - tri['lpw']
        out['tri_vs_mixsel_z'] = round(float(-d1.sum() / (d1.std(ddof=1) * math.sqrt(len(d1)))), 2)
    return out


def dump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
