"""X-4 shared library: corpora as documents of lines of words, words encoded as strings of one character per
sign (so the Voynich v23 / v31 / v33 code can run unchanged on any script), fitted generators, row writer.
"""
import os, sys, json, math, random
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PE = os.path.dirname(HERE)
OS = os.path.dirname(PE)
CK = os.path.join(PE, 'data', 'x4_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(PE, 'loops')
sys.path.insert(0, os.path.join(OS, 'voynich', 'tools'))

LIST = ['LA', 'LA_E', 'PE_E', 'PC_E', 'LB', 'LB_E', 'UR3', 'UR3_E']
PROSE = ['GRC', 'LAT', 'ITA', 'DEU', 'CES']
ROLE = {'LA': 'target', 'LA_E': 'target', 'PE_E': 'target', 'VOY': 'voynich',
        'LB': 'list-lang', 'LB_E': 'list-lang', 'UR3': 'list-lang', 'UR3_E': 'list-lang', 'PC_E': 'list-lang',
        'GRC': 'prose', 'LAT': 'prose', 'ITA': 'prose', 'DEU': 'prose', 'CES': 'prose'}
_C = None


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def corpora():
    """name -> list of docs; doc = list of lines; line = list of word strings (1 char per sign)."""
    global _C
    if _C is None:
        raw = json.load(open(os.path.join(CK, 'corpora.json')))
        _C = {}
        for k, docs in raw.items():
            cnt = Counter(s for d in docs for l in d for w in l for s in w)
            m = {s: chr(0x4E00 + i) for i, (s, _) in enumerate(cnt.most_common())}
            _C[k] = [[[''.join(m[s] for s in w) for w in l] for l in d] for d in docs]
    return _C


def ntok(docs):
    return sum(len(l) for d in docs for l in d)


def subsample(docs, max_tok, seed=0):
    """random whole documents until max_tok words (document order kept)."""
    idx = list(range(len(docs))); random.Random(seed).shuffle(idx)
    keep, t = [], 0
    for i in idx:
        if t >= max_tok: break
        keep.append(i); t += sum(len(l) for l in docs[i])
    return [docs[i] for i in sorted(keep)]


def body_only(docs):
    """drop the first and last line of every document (headers, totals); keep docs with >= 1 line left."""
    return [d[1:-1] for d in docs if len(d) >= 3]


def stream(d):
    return [w for l in d for w in l]


# ------------------------------------------------------------------ fitted generators (keep doc / line shape)
def _cum(c):
    ks = list(c); v = np.cumsum([c[k] for k in ks]).astype(float); return ks, v / v[-1]


def _draw(kv, rng):
    ks, cp = kv
    return ks[min(int(np.searchsorted(cp, rng.random())), len(ks) - 1)]


def _reshape(docs, words):
    out, i = [], 0
    for d in docs:
        nd = []
        for l in d:
            nd.append(words[i:i + len(l)]); i += len(l)
        out.append(nd)
    return out


def gen_wbg(docs, rng):
    """word bigram over the document stream (doc start token), same doc and line shape."""
    T = defaultdict(Counter)
    for d in docs:
        s = ['<D>'] + stream(d) + ['</D>']
        for a, b in zip(s, s[1:]): T[a][b] += 1
    T = {k: _cum(v) for k, v in T.items()}
    words = []
    for d in docs:
        p = '<D>'
        n = len(stream(d)); got = 0
        while got < n:
            w = _draw(T.get(p) or T['<D>'], rng)
            if w == '</D>':
                p = '<D>'; continue                      # document ended early: start a new one
            words.append(w); p = w; got += 1
    return _reshape(docs, words)


def gen_tri(docs, rng):
    """sign trigram over the running text of each doc (word boundary ' ', doc start '^^')."""
    T = defaultdict(Counter)
    for d in docs:
        x = '^^' + ' '.join(stream(d)) + ' $'
        for i in range(2, len(x)): T[x[i - 2:i]][x[i]] += 1
    T = {k: _cum(v) for k, v in T.items()}
    words = []
    for d in docs:
        n = len(stream(d)); ctx = '^^'; cur = ''; got = 0; guard = 0
        while got < n and guard < 4000:
            guard += 1
            c = _draw(T.get(ctx) or T['^^'], rng)
            if c == '$':
                ctx = '^^'; continue                     # document ended early: start a new one
            if c == ' ':
                if cur: words.append(cur); cur = ''; got += 1
            else:
                cur += c
            ctx = ctx[1] + c
        while got < n: words.append(cur or _draw(T['^^'], rng).strip() or 'x'); got += 1; cur = ''
    return _reshape(docs, words)


def gen_slot(docs, rng):
    """in-word sign trigram, every word independent."""
    T = defaultdict(Counter)
    for d in docs:
        for w in stream(d):
            x = '^^' + w + '$'
            for i in range(2, len(x)): T[x[i - 2:i]][x[i]] += 1
    T = {k: _cum(v) for k, v in T.items()}
    words = []
    for _ in range(ntok(docs)):
        x = '^^'
        while len(x) < 24:
            c = _draw(T[x[-2:]], rng)
            if c == '$': break
            x += c
        words.append(x[2:] or _draw(T['^^'], rng))
    return _reshape(docs, words)


def gen_cpv(docs, rng, pcopy=0.5, window=40):
    """copy-and-vary: with prob pcopy copy one of the last `window` words of the doc and change one sign
    (unigram draw); otherwise a word drawn from the corpus unigram word law."""
    wl = _cum(Counter(w for d in docs for w in stream(d)))
    sl = _cum(Counter(s for d in docs for w in stream(d) for s in w))
    words = []
    for d in docs:
        hist = []
        for _ in range(len(stream(d))):
            if hist and rng.random() < pcopy:
                w = rng.choice(hist[-window:]); i = rng.randrange(len(w))
                w = w[:i] + _draw(sl, rng) + w[i + 1:]
            else:
                w = _draw(wl, rng)
            hist.append(w); words.append(w)
    return _reshape(docs, words)


GENS = {'WBG': gen_wbg, 'TRI': gen_tri, 'SLOT': gen_slot, 'CPV': gen_cpv}
