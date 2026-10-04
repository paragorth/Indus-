"""v41: the spelling drift as a clock.

Pages carry odd-line and even-line halves so that every trait correlation is
computed across disjoint tokens. Traits are word-level spelling habits,
oriented so that +1 means 'more like the later (B / later-dated) state'.

Statistics
  R  = mean_i corr(trait_i on half 1, trait_i on half 2)   (page-level reliability)
  S  = mean_{i!=j} oriented corr(trait_i half 1, trait_j half 2)  (co-movement)
  S/R near 1 = one clock drives all traits; near 0 = independent traits.
Null: each trait column permuted independently across pages (within strata).
Seriation: simulated annealing over page orders maximising sum_i rho_i^2
(Spearman of trait i with position, free sign), many random restarts.
"""
import os, re, json, glob, html, math, random, unicodedata
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v41_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v30/ReF-v1.0.2'

# ---------------- Voynich ----------------
BENCH = ('cth', 'ckh', 'cph', 'cfh')
A_END = ('chy', 'shy', 'chol', 'shol', 'chor', 'shor')
B_END = ('chedy', 'shedy', 'chdy', 'shdy', 'edy')

# fixed, theory-driven trait list (v30 rules, v36 bench loss, v8 oddities)
# each: name -> (predicate on word, denominator predicate or None for all words)
def _ends(w, t):
    return any(w.endswith(x) for x in t)

VTRAITS = {
    'bench_init': (lambda w: w.startswith(BENCH), None),
    'bench_med': (lambda w: any(b in w[1:] for b in BENCH) and not w.startswith(BENCH), None),
    'Bend_share': (lambda w: _ends(w, B_END), lambda w: _ends(w, B_END) or _ends(w, A_END)),
    'ol_or_end': (lambda w: w.endswith(('ol', 'or')), None),
    'qo_init': (lambda w: w.startswith('qo'), None),
    'ee': (lambda w: 'ee' in w, None),
    'ed': (lambda w: 'ed' in w, None),
    'aiin_end': (lambda w: w.endswith('aiin'), None),
    'y_init': (lambda w: w.startswith('y'), None),
    'dy_end': (lambda w: w.endswith('dy'), None),
}


def vpages(name='ZL3b', min_tokens=30):
    from v8_lib import voynich_pages
    P = voynich_pages(min_tokens=min_tokens, ltypes=('P',), name=name)
    for p in P:
        p['h'] = [[w for i, l in enumerate(p['lines']) if i % 2 == k for w in l] for k in (0, 1)]
        p['all'] = [w for l in p['lines'] for w in l]
    return P


def rate_matrix(pages, traits, key='h', prior=1.0):
    """returns X[half] (pages x traits) of empirical logits, and N (denominators)."""
    names = list(traits)
    out = []
    for k in (0, 1):
        X = np.zeros((len(pages), len(names)))
        for r, p in enumerate(pages):
            ws = p[key][k] if key == 'h' else p[key]
            for c, nm in enumerate(names):
                f, d = traits[nm]
                den = [w for w in ws if (d is None or d(w))]
                kk = sum(1 for w in den if f(w))
                n = len(den)
                # shrink to the group mean is done later; empirical logit with prior
                X[r, c] = math.log((kk + prior * 0.5) / (n - kk + prior * 0.5))
        out.append(X)
    return out, names


def orient_from(pages_ref_early, pages_ref_late, traits):
    """sign of (late - early) pooled log-odds for each trait."""
    o = []
    for nm, (f, d) in traits.items():
        def lo(P):
            ws = [w for p in P for w in p['all'] if d is None or d(w)]
            k = sum(1 for w in ws if f(w)); n = len(ws)
            return math.log((k + .5) / (n - k + .5))
        o.append(np.sign(lo(pages_ref_late) - lo(pages_ref_early)) or 1.0)
    return np.array(o)


def demean(X, strata):
    X = X.copy()
    strata = np.asarray(strata)
    for s in set(strata.tolist()):
        m = strata == s
        X[m] -= X[m].mean(0)
    return X


def _z(X):
    s = X.std(0); s[s == 0] = 1
    return (X - X.mean(0)) / s


def coherence(X1, X2, o, strata=None):
    if strata is not None:
        X1 = demean(X1, strata); X2 = demean(X2, strata)
    Z1 = _z(X1) * o; Z2 = _z(X2) * o
    n, k = Z1.shape
    C = Z1.T @ Z2 / n                     # k x k cross-half correlations
    C = (C + C.T) / 2
    R = np.mean(np.diag(C))
    off = C[~np.eye(k, dtype=bool)]
    S = off.mean()
    return dict(R=float(R), S=float(S), ratio=float(S / R) if R > 0.02 else float('nan'),
                pos_pairs=float((off > 0).mean()), C=C)


def perm_within(n, strata, rng):
    idx = np.arange(n)
    if strata is None:
        return rng.permutation(n)
    strata = np.asarray(strata)
    out = idx.copy()
    for s in set(strata.tolist()):
        m = np.where(strata == s)[0]
        out[m] = m[rng.permutation(len(m))]
    return out


def coherence_null(X1, X2, o, strata=None, nperm=500, seed=0):
    rng = np.random.default_rng(seed)
    obs = coherence(X1, X2, o, strata)
    Ss = []
    for _ in range(nperm):
        Y1 = X1.copy(); Y2 = X2.copy()
        for j in range(X1.shape[1]):
            pi = perm_within(X1.shape[0], strata, rng)
            Y1[:, j] = X1[pi, j]; Y2[:, j] = X2[pi, j]
        Ss.append(coherence(Y1, Y2, o, strata)['S'])
    Ss = np.array(Ss)
    obs['S_null_mean'] = float(Ss.mean()); obs['S_null_sd'] = float(Ss.std() + 1e-12)
    obs['z'] = float((obs['S'] - Ss.mean()) / (Ss.std() + 1e-12))
    obs['p'] = float((1 + (Ss >= obs['S']).sum()) / (1 + nperm))
    return obs


# ---------------- seriation ----------------
def seriate(X, restarts=20, iters=20000, seed=0):
    """X pages x traits (already demeaned). Maximise sum_i corr(rank-trait_i, position)^2.
    Returns best order (array of page indices, position 0 first), all restart orders."""
    rng = np.random.default_rng(seed)
    n, k = X.shape
    Rk = np.column_stack([rankdata(X[:, j]) for j in range(k)])
    Rk = (Rk - Rk.mean(0)) / (Rk.std(0) + 1e-12)
    pos0 = np.arange(n, dtype=float); pos0 = (pos0 - pos0.mean()) / pos0.std()
    best, orders = None, []
    for r in range(restarts):
        perm = rng.permutation(n)           # perm[position] = page
        posof = np.empty(n); posof[perm] = pos0
        dot = Rk.T @ posof / n               # k-vector of correlations
        f = (dot ** 2).sum()
        T = 0.05
        for it in range(iters):
            a, b = rng.integers(0, n, 2)
            if a == b:
                continue
            pa, pb = perm[a], perm[b]
            d = (pos0[b] - pos0[a]) * (Rk[pa] - Rk[pb]) / n   # pa moves to b, pb to a
            nd = dot + d
            nf = (nd ** 2).sum()
            if nf >= f or rng.random() < math.exp((nf - f) / T):
                perm[a], perm[b] = pb, pa
                dot, f = nd, nf
            T *= 0.9997
        # orient: increasing mean of first trait loadings sign-agnostic -> later fixed by caller
        orders.append((f, perm.copy()))
        if best is None or f > best[0]:
            best = (f, perm.copy())
    return best[1], best[0], [o for _, o in orders]


def position_of(order):
    pos = np.empty(len(order)); pos[order] = np.arange(len(order))
    return pos


# ---------------- German (ReF MLU/RUB) ----------------
def gnorm(w):
    w = unicodedata.normalize('NFD', w.lower())
    return ''.join(ch for ch in w if unicodedata.category(ch)[0] in 'LM')


def ref_meta():
    out = []
    for f in sorted(glob.glob(os.path.join(SCR, 'ref-mlu', '*.xml')) + glob.glob(os.path.join(SCR, 'ref-rub', '*.xml'))):
        head = open(f, encoding='utf-8').read(6000)
        g = lambda k: (re.search(r'^' + k + r': (.*)$', head, re.M) or [None, ''])[1].strip()
        t = g('time'); dt = g('date')
        yr = None
        m = re.search(r'(1[3-6]\d\d)', dt)
        if m:
            yr = int(m.group(1))
            m2 = re.search(r'(1[3-6]\d\d)\s*[-–]\s*(\d{2,4})', dt)
            if m2:
                y2 = m2.group(2); y2 = int(m.group(1)[:4 - len(y2)] + y2)
                if y2 >= yr: yr = (yr + y2) / 2
        elif re.match(r'\d\d,\d', t):
            c, h = t.split(','); yr = (int(c) - 1) * 100 + (25 if h == '1' else 75)
        out.append(dict(sig=os.path.basename(f)[:-4], path=f, time=t, year=yr, medium=g('medium'),
                        ltype=g('language-type'), region=g('language-region'), area=g('language-area'),
                        text=g('text')))
    return out


def ref_words(path):
    txt = open(path, encoding='utf-8').read()
    ws = [gnorm(html.unescape(u)) for u in re.findall(r'<tok_dipl [^>]*utf="([^"]*)"', txt)]
    return [w for w in ws if w]


def chunk_pages(words, size=300, maxpages=None, line=10):
    P = []
    for i in range(0, len(words) - size + 1, size):
        ws = words[i:i + size]
        lines = [ws[j:j + line] for j in range(0, len(ws), line)]
        P.append(dict(lines=lines, all=ws,
                      h=[[w for t, l in enumerate(lines) if t % 2 == k for w in l] for k in (0, 1)]))
        if maxpages and len(P) >= maxpages:
            break
    return P


def ngram_traits(names):
    return {nm: ((lambda g: (lambda w: g in ('^' + w + '$')))(nm), None) for nm in names}


def write_rows(path, rows, header=None):
    with open(path, 'w') as fh:
        if header:
            fh.write(header.rstrip() + '\n\n')
        for r in rows:
            fh.write(r.rstrip() + '\n')


def plant(word, s, r):
    """planted A->B rewrite of one word with probability s (control only)."""
    if r.random() >= s:
        return word
    w = word
    for b, g in (('cth', 't'), ('ckh', 'k'), ('cph', 'p'), ('cfh', 'f')):
        if w.startswith(b):
            return g + w[3:]
    for a, b in (('chol', 'chedy'), ('chor', 'chedy'), ('chy', 'chedy'), ('shy', 'shedy'), ('ol', 'edy'), ('or', 'edy')):
        if w.endswith(a):
            return w[:-len(a)] + b
    return w


def stem_split(w):
    for e in sorted(A_END + B_END, key=len, reverse=True):
        if w.endswith(e) and len(w) > len(e):
            return w[:-len(e)], e
    return None, None


def make_traits(P):
    """10 fixed traits + stem-held-fixed B-ending share (stems seen with both an A and a B ending)."""
    st = defaultdict(set)
    for p in P:
        for w in p['all']:
            s, e = stem_split(w)
            if s:
                st[s].add('B' if e in B_END else 'A')
    shared = {s for s, v in st.items() if len(v) == 2}
    T = dict(VTRAITS)
    T['stem_Bshare'] = (lambda w: stem_split(w)[1] in B_END,
                        lambda w: stem_split(w)[0] in shared)
    return T, len(shared)
