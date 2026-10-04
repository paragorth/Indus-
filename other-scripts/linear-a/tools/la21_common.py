#!/usr/bin/env python3
"""LA-21 shared code: LET UNIVERSALS BUILD THE GRID.

Signs are opaque identities. A blind annealer (la21_core.c) assigns each sign to a cell of a grid with one pure-vowel
row plus R consonant rows and K vowel columns, scoring only language-independent tendencies (tier-factorised bigram
likelihood, consonant-row OCP, pure vowels word-initial, cell capacity). Sound values are used only to SCORE the
controls (Linear B, Japanese and Ancient Greek syllabified texts) and, for Linear A, as an outside check after the fact.
"""
import os, sys, json, math, random, collections, ctypes, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la5_common as C
CK = os.path.join(HERE, '..', 'data', 'la21_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')

_so = os.path.join(HERE, 'la21_core.so')
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(HERE, 'la21_core.c')):
    os.system(f"gcc -O3 -march=native -shared -fPIC -o {_so} {os.path.join(HERE, 'la21_core.c')} -lm")
_lib = ctypes.CDLL(_so)
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C')
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C')
_lib.anneal_many.restype = ctypes.c_int
_lib.anneal_many.argtypes = [ctypes.c_int] * 4 + [_ip, _ip, _ip] + [ctypes.c_double] * 3 + [ctypes.c_int, ctypes.c_long,
                             ctypes.c_double, ctypes.c_double, ctypes.c_ulonglong, _ip, _ip, _dp]

SIGRE = re.compile(r'^(D|J|K|M|N|P|Q|R|S|T|W|Z)?([AEIOU])$')
def truth_lb(sign):
    """(consonant, vowel) of a plain Linear B CV/V sign, else None (used only to score)."""
    m = SIGRE.match(sign)
    return (m.group(1) or '', m.group(2)) if m else None

# ------------------------------------------------------------------ corpora
def la_types():
    return sorted(set(w for _, _, w in C.words_of(C.la_docs(admin_only=False))))

def lb_types():
    return sorted(set(w for _, _, w in C.words_of(C.lb_docs())))

def counts_from_types(types, nsign):
    """top-nsign signs by type frequency; words split at other signs into fragments.
    -> signs, B (S x S), I, F (int32)."""
    c = collections.Counter(s for w in types for s in w)
    signs = [s for s, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:nsign]]
    ix = {s: i for i, s in enumerate(signs)}
    S = len(signs)
    B = np.zeros((S, S), np.int32); I = np.zeros(S, np.int32); F = np.zeros(S, np.int32)
    for w in types:
        frag = []
        for s in list(w) + [None]:
            if s is not None and s in ix: frag.append(ix[s]); continue
            if frag:
                I[frag[0]] += 1; F[frag[-1]] += 1
                for a, b in zip(frag, frag[1:]): B[a, b] += 1
            frag = []
    return signs, B, I, F

def shuffle_within(types, seed):
    r = random.Random(seed); out = []
    for w in types:
        w = list(w); r.shuffle(w); out.append(tuple(w))
    return out

def shuffle_global(types, seed):
    r = random.Random(seed); toks = [s for w in types for s in w]; r.shuffle(toks); out = []; k = 0
    for w in types: out.append(tuple(toks[k:k + len(w)])); k += len(w)
    return out

def la11_counts(code, scheme='DROP', nsign=65, target_bigrams=None, seed=0):
    """syllable bigram table of a la11 language (type counts), optionally multinomially thinned to target size."""
    d = json.load(open(os.path.join(HERE, '..', 'data', 'la11', 'lang', code + '.json')))['schemes'][scheme]
    uni = d['uni']; bi = d['bi']
    signs = [s for s, _ in sorted(uni.items(), key=lambda kv: (-kv[1], kv[0]))[:nsign]]
    ix = {s: i for i, s in enumerate(signs)}; S = len(signs)
    items = []
    for k, v in bi.items():
        a, b = k.split('\t'); items.append((a, b, v))
    if target_bigrams:
        r = np.random.default_rng(seed); tot = sum(v for _, _, v in items)
        p = np.array([v for _, _, v in items], float) / tot
        n = r.multinomial(target_bigrams, p)
        items = [(a, b, int(k)) for (a, b, _), k in zip(items, n)]
    B = np.zeros((S, S), np.int32); I = np.zeros(S, np.int32); F = np.zeros(S, np.int32)
    for a, b, v in items:
        if a == '^' and b in ix: I[ix[b]] += v
        elif b == '$' and a in ix: F[ix[a]] += v
        elif a in ix and b in ix: B[ix[a], ix[b]] += v
        elif a in ix and b not in ix: F[ix[a]] += v        # fragment ends at a dropped syllable
        elif b in ix and a not in ix: I[ix[b]] += v
    return signs, B, I, F

VOWS = set('aeiouəɨɯɔɛ')
def truth_syll(s):
    """(onset, nucleus) of a la11 syllable string: nucleus = last vowel letter (+length marks stripped)."""
    t = s.replace('ː', '').replace(':', '')
    i = len(t) - 1
    while i >= 0 and t[i] not in VOWS: i -= 1
    if i < 0: return None
    return (t[:i], t[i])

# ------------------------------------------------------------------ annealing
def anneal(signs_B_I_F, R1, K, nrest, moves, seed, cap=2, wocp=1.0, wvin=1.0, alpha=0.5, T0=8.0, T1=0.05):
    signs, B, I, F = signs_B_I_F
    S = len(signs)
    orow = np.zeros(nrest * S, np.int32); ocol = np.zeros(nrest * S, np.int32); osc = np.zeros(nrest * 5)
    rc = _lib.anneal_many(S, R1, K, cap, np.ascontiguousarray(B.ravel()), np.ascontiguousarray(I), np.ascontiguousarray(F),
                          wocp, wvin, alpha, nrest, moves, T0, T1, seed, orow, ocol, osc)
    assert rc == 0
    return orow.reshape(nrest, S), ocol.reshape(nrest, S), osc.reshape(nrest, 5)

def coassign(rows, cols):
    """fraction of runs in which each sign pair shares a row / a column; and 'pure-vowel row' frequency."""
    n, S = rows.shape
    PR = np.zeros((S, S)); PC = np.zeros((S, S))
    for k in range(n):
        PR += rows[k][:, None] == rows[k][None, :]
        PC += cols[k][:, None] == cols[k][None, :]
    return PR / n, PC / n, (rows == 0).mean(0)

def stability(PR, PC):
    """mean pair decisiveness 2|p-0.5| over sign pairs, and number of pairs with p >= 0.7."""
    iu = np.triu_indices(PR.shape[0], 1)
    return dict(rowdec=float(np.mean(np.abs(PR[iu] - .5) * 2)), coldec=float(np.mean(np.abs(PC[iu] - .5) * 2)),
                row70=int((PR[iu] >= .7).sum()), col70=int((PC[iu] >= .7).sum()))

def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv)); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    rk = np.empty(len(allv)); rk[order] = ranks
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def truth_scores(signs, PR, PC, truthf):
    """AUC of co-assignment frequency for true same-consonant / same-vowel pairs (plain CV signs only)."""
    T = [truthf(s) for s in signs]
    idx = [i for i, t in enumerate(T) if t is not None]
    sr, lr, sc, lc = [], [], [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            sr.append(PR[i, j]); lr.append(T[i][0] == T[j][0])
            sc.append(PC[i, j]); lc.append(T[i][1] == T[j][1])
    sr2, lr2 = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if T[i][0] and T[j][0]: sr2.append(PR[i, j]); lr2.append(T[i][0] == T[j][0])
    return dict(n=len(idx), row_auc=auc(sr, lr), col_auc=auc(sc, lc), crow_auc=auc(sr2, lr2),
                row_prec70=_prec(sr, lr, .7), col_prec70=_prec(sc, lc, .7),
                base_row=float(np.mean(lr)) if lr else float('nan'), base_col=float(np.mean(lc)) if lc else float('nan'))

def _prec(s, l, th):
    s = np.asarray(s); l = np.asarray(l, bool); m = s >= th
    return (int(l[m].sum()), int(m.sum()))

def groups(signs, P, th=0.7):
    """single-link-free groups: connected components of pairs with P >= th, then kept if cohesive (mean >= th)."""
    S = len(signs); par = list(range(S))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(S):
        for j in range(i + 1, S):
            if P[i, j] >= th: par[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(S): g[f(i)].append(i)
    out = []
    for m in g.values():
        if len(m) < 2: continue
        mp = np.mean([P[a, b] for a in m for b in m if a < b])
        out.append(([signs[i] for i in m], float(mp)))
    return sorted(out, key=lambda x: -x[1])
