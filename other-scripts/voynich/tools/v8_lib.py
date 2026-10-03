"""v8: seriation ('writing clock') helpers for the Voynich drift experiments.

Pages are dicts: id, order (binding index), quire, bifolio ($B), leaf ($F),
side (r/v), lang, hand, illus, lines (list of word lists).
"""
import os, re, json, math, random
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')


def page_headers(path=os.path.join(DATA, 'ZL3b-n.txt')):
    hdr = {}
    for L in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'^<(f\d+[rv]\d*)>\s*<!([^>]*)>', L)
        if m:
            d = dict(re.findall(r'\$(\w)=(\w+)', m.group(2)))
            hdr[m.group(1)] = d
    return hdr


def voynich_pages(min_tokens=30, ltypes=('P', 'C', 'R')):
    recs = json.load(open(os.path.join(DATA, 'derived', 'ZL3b_lines.json')))
    hdr = page_headers()
    pages, idx = {}, []
    for r in recs:
        if r['ltype'] not in ltypes:
            continue
        f = r['folio']
        if f not in pages:
            h = hdr.get(f, {})
            side = 'r' if 'r' in f[1:] else 'v'
            leafnum = int(re.match(r'f(\d+)', f).group(1))
            pages[f] = dict(id=f, order=len(idx), quire=h.get('Q'), bifolio=h.get('B'),
                            leaf=h.get('F'), leafnum=leafnum, side=side,
                            lang=h.get('L'), hand=h.get('H'), illus=h.get('I'), lines=[])
            idx.append(f)
        ws = [w for w in r['words'] if '?' not in w]
        if ws:
            pages[f]['lines'].append(ws)
    out = [pages[f] for f in idx if sum(len(l) for l in pages[f]['lines']) >= min_tokens]
    for i, p in enumerate(out):
        p['order'] = i
    return out


def ntok(p):
    return sum(len(l) for l in p['lines'])


# ---------------- features & seriation ----------------

def matrix(pages, min_count=5, which=None, max_types=None):
    """Page x type count matrix. which: None (all lines), 0 (even lines), 1 (odd lines)."""
    cnts = []
    for p in pages:
        c = Counter()
        for i, l in enumerate(p['lines']):
            if which is None or i % 2 == which:
                c.update(l)
        cnts.append(c)
    tot = Counter()
    for c in cnts:
        tot.update(c)
    vocab = [w for w, n in tot.most_common() if n >= min_count]
    if max_types:
        vocab = vocab[:max_types]
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(pages), len(vocab)))
    for r, c in enumerate(cnts):
        for w, n in c.items():
            if w in vi:
                X[r, vi[w]] = n
    return X, vocab


def hellinger_sim(X):
    P = X / np.maximum(X.sum(1, keepdims=True), 1)
    S = np.sqrt(P)
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    return S @ S.T


def spectral_order(X, k_nn=None):
    """Fiedler vector of normalized Laplacian of Hellinger affinity (optionally kNN-sparsified)."""
    A = hellinger_sim(X)
    np.fill_diagonal(A, 0)
    A = np.clip(A, 0, None)
    if k_nn:
        n = len(A)
        M = np.zeros_like(A)
        for i in range(n):
            nn = np.argsort(-A[i])[:k_nn]
            M[i, nn] = A[i, nn]
        A = np.maximum(M, M.T)
    d = A.sum(1)
    d[d == 0] = 1e-9
    Dm = np.diag(1 / np.sqrt(d))
    L = np.eye(len(A)) - Dm @ A @ Dm
    w, v = np.linalg.eigh(L)
    f = Dm @ v[:, 1]
    return f


def tsp_order(D, seed=0, restarts=3):
    """Open-path TSP: nearest neighbour from several starts + 2-opt. Returns order list, length."""
    n = len(D)
    rng = random.Random(seed)
    best = None
    starts = [rng.randrange(n) for _ in range(restarts)]
    for s in starts:
        un = set(range(n)); path = [s]; un.remove(s)
        while un:
            last = path[-1]
            nxt = min(un, key=lambda j: D[last, j])
            path.append(nxt); un.remove(nxt)
        path = two_opt(path, D)
        Ln = sum(D[path[i], path[i + 1]] for i in range(n - 1))
        if best is None or Ln < best[1]:
            best = (path, Ln)
    return best


def two_opt(path, D, max_iter=50):
    path = list(path); n = len(path)
    improved, it = True, 0
    while improved and it < max_iter:
        improved = False; it += 1
        for i in range(0, n - 2):
            a, b = path[i], path[i + 1]
            # vectorised search over j
            js = np.arange(i + 2, n)
            c = np.array(path)[js]
            dn = np.array([path[j + 1] if j + 1 < n else -1 for j in js])
            old = D[a, b] + np.where(dn >= 0, D[c, np.maximum(dn, 0)], 0)
            new = D[a, c] + np.where(dn >= 0, D[b, np.maximum(dn, 0)], 0)
            gain = old - new
            k = int(np.argmax(gain))
            if gain[k] > 1e-12:
                j = js[k]
                path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                improved = True
    return path


def dist(X):
    return np.clip(1 - hellinger_sim(X), 0, None)


def rankpos(order):
    r = np.empty(len(order)); r[np.array(order)] = np.arange(len(order)); return r


def absrho(a, b):
    r = spearmanr(a, b).correlation
    return abs(r) if r == r else 0.0


def split_half(pages, min_count=3):
    """Seriate even lines and odd lines separately; |rho| of Fiedler orders."""
    X0, _ = matrix(pages, min_count, which=0)
    X1, _ = matrix(pages, min_count, which=1)
    return absrho(spectral_order(X0), spectral_order(X1))


# ---------------- controls ----------------

def redeal(pages, strata_key=None, seed=0):
    """Shuffle all tokens across pages (keeping every line length), optionally within strata."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, p in enumerate(pages):
        groups[strata_key(p) if strata_key else 0].append(i)
    out = [dict(p) for p in pages]
    for g, ids in groups.items():
        toks = [w for i in ids for l in pages[i]['lines'] for w in l]
        rng.shuffle(toks)
        k = 0
        for i in ids:
            nl = []
            for l in pages[i]['lines']:
                nl.append(toks[k:k + len(l)]); k += len(l)
            out[i]['lines'] = nl
    return out


def latin_words(path=os.path.join(DATA, 'plain', 'la.txt')):
    t = open(path, encoding='utf-8', errors='replace').read().lower()
    t = t.replace('æ', 'ae').replace('œ', 'oe').replace('j', 'i').replace('v', 'u')
    return re.findall(r'[a-z]+', t)


def italian_words():
    from vlib import load_ref
    return [w for l in load_ref('Italian-Manzoni') for w in l['words']]


def text_to_pages(words, template, offset=0):
    """Cut a word stream into pages with the same line lengths as template pages."""
    out, k = [], offset
    for p in template:
        nl = []
        for l in p['lines']:
            nl.append(words[k:k + len(l)]); k += len(l)
        q = dict(p); q['lines'] = nl; q['true'] = p['order']
        out.append(q)
    return out


def drift_cipher(pages, seed=0, width=0.08, letters='abcdefghiklmnopqrstuxyz'):
    """Homophonic letter cipher whose variant choice drifts with true position t in [0,1].
    Each letter l switches from variant 0 to 1 around a random time c_l (logistic, width)."""
    rng = random.Random(seed)
    c = {l: rng.uniform(0.1, 0.9) for l in letters}
    n = len(pages)
    out = []
    for i, p in enumerate(pages):
        t = i / max(n - 1, 1)
        nl = []
        for l in p['lines']:
            nw = []
            for w in l:
                s = []
                for ch in w:
                    if ch in c:
                        pr = 1 / (1 + math.exp(-(t - c[ch]) / width))
                        s.append(ch.upper() if rng.random() < pr else ch)
                    else:
                        s.append(ch)
                nw.append(''.join(s))
            nl.append(nw)
        q = dict(p); q['lines'] = nl
        out.append(q)
    return out


def write_rows(cycle, rows, header=None):
    os.makedirs(LOOPS, exist_ok=True)
    path = os.path.join(LOOPS, f'v8_cycle{cycle}.txt')
    with open(path, 'w') as f:
        if header:
            f.write(header.rstrip() + '\n\n')
        f.write('| ID | Method and control | Result | Verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
    return path
