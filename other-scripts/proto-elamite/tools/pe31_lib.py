"""pe31: interchangeable name signs.  Shared code.

Idea: if some signs inside name-like strings write sounds, spelling variation
should make some sign pairs interchangeable (the same string written with X or
with Y, in the same contexts).  Every pair of signs is scored inside name-like
strings with four type-level statistics:

  MP   minimal pairs: number of distinct frames (a string type with one position
       left open) attested with both X and Y.
  CTX  v35 interchangeability index: tablets split into two alternating halves,
       PPMI rows of the +-1/+-2 neighbour profile in each half,
       I = mean(cos(X1,Y2), cos(Y1,X2)) / sqrt(cos(X1,X2) cos(Y1,Y2)).
  POS  cosine of the position profiles (initial / medial / final x length 2,3,4+).
  TAB  tablet complementarity: log((observed tablets with both + .5) / (expected + .5));
       allographs chosen by scribe should be negative.

No sound values or readings are used anywhere.
"""
import json, math, os, re, random, sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, is_sign  # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe31_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)

_r = json.load(open(os.path.join(DATA, 'res_a_slots.json')))['rows']
FINAL = {x['sign'] for x in _r if x['z_final'] >= 3}


def base(s):
    return re.sub(r'~[A-Za-z0-9]+', '', s)


def ok(s):
    return is_sign(s) and 'X' not in s.replace('|', '') .split('+') and '...' not in s and s != 'x'


# ------------------------------------------------------------------ corpora
def pe_names(variants=True):
    """Name-like strings: entry lines (header line excluded), no x/lacuna, final class
    sign(s) stripped, at least 2 signs left.  Returns list of (tuple, tablet)."""
    T = load()
    out = []
    for t in T:
        lines = t['lines']
        for i, l in enumerate(lines):
            if i == 0 and not l['numerals']:
                continue
            if not l['numerals'] or l.get('lacuna'):
                continue
            sg = l['signs']
            if not sg or not all(ok(s) for s in sg):
                continue
            while sg and base(sg[-1]) in FINAL and len(sg) > 1:
                sg = sg[:-1]
            if len(sg) < 2:
                continue
            w = tuple(sg if variants else [base(s) for s in sg])
            out.append((w, t['id']))
    return out


def control(name):
    d = json.load(open(os.path.join(DATA, 'pe7_corpora.json')))[name]
    out = []
    for x in d:
        w = tuple(s for s in x['seq'] if s)
        if len(w) < 2:
            continue
        for tb in x['tablets'][:3]:          # a type spread over its first tablets (cap 3)
            out.append((w, tb))
    return out


def hbase(s):
    """phonetic base of a transliterated sign value (a2 -> a, uszur4 -> uszur, hu(RI) -> hu)"""
    s = re.sub(r'\(.*', '', s)
    s = re.sub(r'(x|[0-9])+$', '', s)
    return s.lower()


LINB_DOUBLETS = {('a', 'a2'), ('a', 'a3'), ('ra', 'ra2'), ('ra', 'ra3'), ('ro', 'ro2'), ('ta', 'ta2'),
                 ('pu', 'pu2'), ('ra2', 'ra3'), ('a2', 'a3')}


def truth_pairs(name, alph):
    P = set()
    for a, b in combinations(alph, 2):
        k = tuple(sorted((a, b)))
        if name == 'LINB':
            if k in LINB_DOUBLETS:
                P.add(k)
        elif hbase(a) == hbase(b) and hbase(a):
            P.add(k)
    return P


# ------------------------------------------------------------------ scores
def alphabet(data, minc):
    c = Counter(s for w, _ in data for s in w)
    return sorted(s for s, n in c.items() if n >= minc), c


def mp_counts(types, alph_set):
    """minimal-pair frame counts for all pairs, type level"""
    fr = defaultdict(set)
    for w in types:
        for i, s in enumerate(w):
            if s in alph_set:
                fr[w[:i] + ('_',) + w[i + 1:]].add(s)
    pc = Counter()
    for v in fr.values():
        if len(v) > 1:
            for a, b in combinations(sorted(v), 2):
                pc[(a, b)] += 1
    return pc


def ctx_rows(words, alph):
    idx = {g: i for i, g in enumerate(alph)}
    ctx = alph + ['<', '>', '?']
    cidx = {g: i for i, g in enumerate(ctx)}
    nC = len(ctx)
    C = np.zeros((len(alph), 4 * nC))
    for w in words:
        ww = ['<', '<'] + list(w) + ['>', '>']
        for k in range(2, len(ww) - 2):
            g = ww[k]
            if g not in idx:
                continue
            for off, slot in ((-1, 0), (1, 1), (-2, 2), (2, 3)):
                x = ww[k + off]
                C[idx[g], slot * nC + cidx.get(x, cidx['?'])] += 1
    tot = C.sum(); r = C.sum(1, keepdims=True); c = C.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.log((C * tot) / (r * c))
    P[~np.isfinite(P)] = 0
    P = np.maximum(P, 0)
    n = np.linalg.norm(P, axis=1, keepdims=True); n[n == 0] = 1
    return P / n


def ctx_index(data, alph, seed=0):
    tabs = sorted({t for _, t in data})
    rng = random.Random(seed); rng.shuffle(tabs)
    h = {t: i % 2 for i, t in enumerate(tabs)}
    A = [w for w, t in data if h[t] == 0]; B = [w for w, t in data if h[t] == 1]
    P1, P2 = ctx_rows(A, alph), ctx_rows(B, alph)
    X = P1 @ P2.T
    d = np.sqrt(np.maximum(np.outer(np.diag(X), np.diag(X)), 1e-6))
    I = 0.5 * (X + X.T) / d
    return np.clip(I, -2, 2)


def pos_profile(data, alph):
    idx = {g: i for i, g in enumerate(alph)}
    M = np.zeros((len(alph), 9))
    for w, _ in data:
        L = min(len(w), 4) - 2
        for i, s in enumerate(w):
            if s in idx:
                p = 0 if i == 0 else (2 if i == len(w) - 1 else 1)
                M[idx[s], L * 3 + p] += 1
    M = M / np.maximum(M.sum(1, keepdims=True), 1)
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
    M = M / n
    return M @ M.T


def tab_comp(data, alph):
    idx = {g: i for i, g in enumerate(alph)}
    tabs = defaultdict(set)
    for w, t in data:
        for s in w:
            if s in idx:
                tabs[t].add(idx[s])
    nT = len(tabs)
    B = np.zeros((nT, len(alph)))
    for k, (t, ss) in enumerate(tabs.items()):
        B[k, list(ss)] = 1
    O = B.T @ B
    f = B.sum(0)
    E = np.outer(f, f) / nT
    return np.log((O + .5) / (E + .5))


def scores(data, alph, seed=0):
    """matrix scores over alph; MP as matrix too"""
    types = sorted({w for w, _ in data})
    pc = mp_counts(types, set(alph))
    idx = {g: i for i, g in enumerate(alph)}
    MP = np.zeros((len(alph), len(alph)))
    for (a, b), n in pc.items():
        MP[idx[a], idx[b]] = MP[idx[b], idx[a]] = n
    return dict(MP=MP, CTX=ctx_index(data, alph, seed), POS=pos_profile(data, alph), TAB=tab_comp(data, alph))


def shuffle_within(data, rng):
    out = []
    for w, t in data:
        w = list(w); rng.shuffle(w); out.append((tuple(w), t))
    return out


def freq_z(S, cnt, alph, nbin=6):
    """z of each pair score against pairs in the same (log freq a, log freq b) bin:
    the frequency-matched random-pair reference."""
    n = len(alph)
    lf = np.log([cnt[a] for a in alph])
    q = np.quantile(lf, np.linspace(0, 1, nbin + 1)[1:-1])
    b = np.digitize(lf, q)
    iu = np.triu_indices(n, 1)
    key = np.minimum(b[iu[0]], b[iu[1]]) * nbin + np.maximum(b[iu[0]], b[iu[1]])
    v = S[iu]
    z = np.zeros_like(v, dtype=float)
    for k in np.unique(key):
        m = key == k
        mu, sd = v[m].mean(), v[m].std() + 1e-9
        z[m] = (v[m] - mu) / sd
    Z = np.zeros((n, n)); Z[iu] = z; Z = Z + Z.T
    return Z


def combo(Sd, cnt, alph, w=(1, 1, 1, 0)):
    ks = ('MP', 'CTX', 'POS', 'TAB')
    Z = sum(wi * freq_z(Sd[k], cnt, alph) for wi, k in zip(w, ks) if wi)
    return Z


def auc(vals, labels):
    vals = np.asarray(vals, float); labels = np.asarray(labels, bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float('nan')
    r = np.argsort(np.argsort(vals + np.random.default_rng(0).random(len(vals)) * 1e-9)) + 1
    n1 = labels.sum(); n0 = len(vals) - n1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def plant(data, signs, mode, rng, suffix="'"):
    """split each planted sign into two homophones: 'token' = coin per token,
    'tablet' = coin per tablet (scribe allograph)"""
    coin = {}
    out = []
    for w, t in data:
        nw = []
        for s in w:
            if s in signs:
                if mode == 'token':
                    c = rng.random() < .5
                else:
                    c = coin.setdefault((t, s), rng.random() < .5)
                nw.append(s + suffix if c else s)
            else:
                nw.append(s)
        out.append((tuple(nw), t))
    return out


def write_rows(path, rows, head=None):
    with open(path, 'w') as f:
        if head:
            f.write(head.rstrip() + '\n\n')
        f.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(x).replace('|', '/') if i else str(x) for i, x in enumerate(r)) + ' |\n')


# ------------------------------------------------------------------ merge evidence (cycle 1b)
def lr_counts(data, alph):
    idx = {g: i for i, g in enumerate(alph)}
    ctx = alph + ['<', '>', '?']
    cidx = {g: i for i, g in enumerate(ctx)}
    Lc = np.zeros((len(alph), len(ctx))); Rc = np.zeros((len(alph), len(ctx)))
    for w, _ in data:
        ww = ['<'] + list(w) + ['>']
        for k in range(1, len(ww) - 1):
            g = ww[k]
            if g not in idx:
                continue
            Lc[idx[g], cidx.get(ww[k - 1], cidx['?'])] += 1
            Rc[idx[g], cidx.get(ww[k + 1], cidx['?'])] += 1
    return Lc, Rc


def _dm_rows(C, a=0.5):
    from scipy.special import gammaln
    K = C.shape[-1]
    n = C.sum(-1)
    return -(gammaln(K * a) - gammaln(K * a + n) + (gammaln(a + C) - gammaln(a)).sum(-1)) / np.log(2)


def merge_evidence(data, alph):
    """bits saved by describing X and Y with ONE left and ONE right neighbour distribution
    (plus the cost of recording which of the two was written), Dirichlet-multinomial.
    Positive = the corpus prefers to treat X and Y as one sign."""
    from scipy.special import gammaln
    Lc, Rc = lr_counts(data, alph)
    sepL, sepR = _dm_rows(Lc), _dm_rows(Rc)
    n = len(alph)
    E = np.zeros((n, n))
    nx = Lc.sum(1)
    for i in range(n):
        mL = _dm_rows(Lc[i] + Lc[i + 1:]); mR = _dm_rows(Rc[i] + Rc[i + 1:])
        a, b = nx[i], nx[i + 1:]
        choose = -(gammaln(1.0) - gammaln(1.0 + a + b) + gammaln(.5 + a) + gammaln(.5 + b) - 2 * gammaln(.5)) / np.log(2)
        e = (sepL[i] + sepL[i + 1:] + sepR[i] + sepR[i + 1:]) - (mL + mR + choose)
        E[i, i + 1:] = e; E[i + 1:, i] = e
    return E
