"""R2 random-PROGRAM machine: data layer.

Builds integer corpora for the C search engine (r2_search.c):
  real corpora (voynich, linear_a, proto_elamite),
  nulls (global token shuffle; Markov resynthesis from the real corpus),
  positive controls (planted grille generator; planted ledger generator),
  natural-language controls (Latin letters, Linear B signs).
For each corpus: tokens per document = line tokens joined by '|' (line end).  Every
position i is predicted from strictly earlier context.  Per position we write the
terminal features the tiny program language can read, the split (A/B/C by document),
and the probability of the true symbol under the Kneser-Ney baseline (A positions:
2-fold out-of-fold inside A; B and C: model trained on A; order chosen on B).
"""
import json, os, sys, math, random, struct
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_lib as L

OUT = os.path.join(L.VOY, 'results', 'r2')
os.makedirs(OUT, exist_ok=True)
VMAX = 250          # symbols with rank >= VMAX are merged into RARE (=VMAX)
PAD = 251
TERMS = ['P1', 'P2', 'P3', 'P4', 'P6', 'P8', 'AB', 'AW', 'AW2', 'FW', 'FL',
         'PW', 'PL', 'WI', 'LI', 'CW', 'LPW', 'LLW']
NT = len(TERMS)


# ---------------------------------------------------------------- corpora
def real(name, seed=0):
    c = L.load(name, seed)
    return [{'id': d['id'], 'split': d['split'], 'lines': d['lines']} for d in c['docs']]


def split_docs(docs, seed=0):
    for d in docs:
        d['split'] = L.split_of(d['id'], seed)
    return docs


def shuffle_null(docs, seed):
    rng = random.Random(seed)
    pool = [t for d in docs for l in d['lines'] for t in l if t not in L.FIX]
    rng.shuffle(pool)
    it = iter(pool)
    return [{'id': d['id'], 'split': d['split'],
             'lines': [[t if t in L.FIX else next(it) for t in l] for l in d['lines']]} for d in docs]


def markov_null(docs, seed, k):
    """Order-k MLE Markov chain over the token stream (with '|' line ends and doc start
    padding), trained on the whole real corpus; each document regenerated with the same
    number of lines (lines end where the chain emits '|')."""
    rng = random.Random(seed)
    tr = defaultdict(Counter)
    for d in docs:
        s = ['<s>'] * k + [t for l in d['lines'] for t in l + ['|']]
        for i in range(k, len(s)):
            tr[tuple(s[i - k:i])][s[i]] += 1
    tabs = {c: (list(v), list(np.cumsum(list(v.values())))) for c, v in tr.items()}
    out = []
    for d in docs:
        ctx = ['<s>'] * k
        lines, cur = [], []
        guard = 0
        while len(lines) < len(d['lines']) and guard < 20000:
            guard += 1
            ks, cs = tabs.get(tuple(ctx), (None, None))
            if ks is None:
                ctx = ['<s>'] * k; continue
            r = rng.random() * cs[-1]
            j = int(np.searchsorted(cs, r, side='right'))
            t = ks[min(j, len(ks) - 1)]
            ctx = ctx[1:] + [t]
            if t == '|' or len(cur) > 200:
                lines.append(cur); cur = []
            else:
                cur.append(t)
        out.append({'id': d['id'], 'split': d['split'], 'lines': lines})
    return out


def planted_grille(vdocs, seed=1, noise=0.2):
    """Voynich-sized table-and-grille text with a PROCEDURAL grille: a 12-row table of
    whole words (prefix+middle+suffix pieces cut from real Voynich words); the word in
    line l, slot w of a page is table row (l + w) mod 12, with a 'noise' share of slots
    filled by a random real Voynich word."""
    rng = random.Random(seed)
    words = [tuple(w) for d in L.load('voynich')['docs'] for l in d['words'] for w in l]
    P, M, S = Counter(), Counter(), Counter()
    for w in words:
        n = len(w); a = round(n / 3); b = round(2 * n / 3)
        P[w[:a]] += 1; M[w[a:b]] += 1; S[w[b:]] += 1
    top = lambda C: [x for x, _ in C.most_common(30) if x]
    p, m, s = top(P), top(M), top(S)
    table = [list(rng.choice(p) + rng.choice(m) + rng.choice(s)) for _ in range(12)]
    out = []
    for d in vdocs:
        lines = []
        for li, l in enumerate(d['lines']):
            nw = l.count('_') + 1
            toks = []
            for w in range(nw):
                if w: toks.append('_')
                toks.extend(list(rng.choice(words)) if rng.random() < noise else table[(li + w) % 12])
            lines.append(toks)
        out.append({'id': d['id'], 'split': d['split'], 'lines': lines})
    return out, [''.join(r) for r in table]


def planted_ledger(docs, seed=1, noise=0.2):
    """LA/PE-sized ledger: line 0 = header word; line i>=1 = COMMODITY _ NAME _ #,
    commodity = cycle[(i-1) mod 3] (a 3-step cycle over 3 logograms), names drawn from
    the real corpus words (Markov-free), 'noise' share of commodities random."""
    rng = random.Random(seed)
    words = []
    for d in docs:
        for l in d['lines']:
            w = []
            for t in l + ['_']:
                if t == '_':
                    if w and all(x not in L.FIX for x in w): words.append(tuple(w))
                    w = []
                else:
                    w.append(t)
    signs = Counter(t for d in docs for l in d['lines'] for t in l if t not in L.FIX)
    pool = [s for s, _ in signs.most_common(60)]
    cyc = rng.sample(pool, 3)
    head = rng.choice(words)
    out = []
    for d in docs:
        lines = [list(head)]
        for i in range(1, len(d['lines'])):
            c = rng.choice(pool) if rng.random() < noise else cyc[(i - 1) % 3]
            lines.append([c, '_'] + list(rng.choice(words)) + ['_', '#'])
        out.append({'id': d['id'], 'split': d['split'], 'lines': lines})
    return out, cyc


def latin(n_tokens=187000, seed=0):
    txt = open(os.path.join(L.VOY, 'pg218.txt'), encoding='utf-8').read()
    a = txt.find('*** START'); b = txt.find('*** END')
    txt = txt[txt.find('\n', a) + 1:b]
    lines = []
    for raw in txt.split('\n'):
        s = ''.join(c for c in raw.lower() if c.isalpha() or c == ' ')
        s = '_'.join(s.split())
        s = ''.join(c for c in s if c in 'abcdefghijklmnopqrstuvwxyz_')
        if len(s) >= 20:
            lines.append(list(s))
    docs, cur, n = [], [], 0
    for l in lines:
        cur.append(l); n += len(l)
        if len(cur) == 20:
            docs.append(cur); cur = []
        if n >= n_tokens: break
    return split_docs([{'id': f'lat{i}', 'lines': d} for i, d in enumerate(docs)], seed)


def linear_b(seed=0):
    C = json.load(open(os.path.join(L.LA, 'la8', 'corpus_LB.json')))
    out = []
    for x in C:
        toks = []
        for t in x['toks']:
            v = t[0]
            if toks: toks.append('_')
            if v.startswith('W:'): toks.extend(v[2:].split('-'))
            elif v.startswith('L:'): toks.append('L:' + v[2:].split(':')[0])
            elif v == 'NUM': toks.append('#')
            else: toks.append('?')
        if toks:
            out.append({'id': x['id'], 'lines': [toks]})
    return split_docs(out, seed)


# ---------------------------------------------------------------- KN baseline
class KN:
    """Interpolated Kneser-Ney (modified discount 0.75) over token ids, order k."""
    def __init__(self, seqs, k, V):
        self.k, self.V, self.D = k, V, 0.75
        self.c = [defaultdict(Counter) for _ in range(k + 1)]   # c[n][ctx][w]
        for s in seqs:
            s = [-1] * k + s
            for i in range(k, len(s)):
                for n in range(k + 1):
                    self.c[n][tuple(s[i - n:i])][s[i]] += 1
        # continuation counts for lower orders
        self.cc = [defaultdict(Counter) for _ in range(k + 1)]
        for n in range(1, k + 1):
            for ctx, cnt in self.c[n].items():
                for w in cnt:
                    self.cc[n - 1][ctx[1:]][w] += 1
        self.tot = {}
        self.memo = {}

    def p(self, ctx, w, n=None):
        if n is None: n = self.k
        if n == 0:
            cnt = self.cc[0][()] if self.k > 0 else self.c[0][()]
            T = self.tot.get(n)
            if T is None: T = self.tot[n] = sum(cnt.values())
            return (cnt.get(w, 0) + 0.5) / (T + 0.5 * self.V)
        ctx = tuple(ctx[-n:])
        key = (n, ctx, w)
        if key in self.memo: return self.memo[key]
        cnt = self.c[n][ctx] if n == self.k else self.cc[n][ctx]
        T = sum(cnt.values())
        lower = self.p(ctx, w, n - 1)
        if T == 0:
            r = lower
        else:
            r = max(cnt.get(w, 0) - self.D, 0) / T + self.D * len(cnt) / T * lower
        self.memo[key] = r
        return r


def kn_probs(model, seq):
    k = model.k
    s = [-1] * k + seq
    return [model.p(s[i - k:i], s[i]) for i in range(k, len(s))]


# ---------------------------------------------------------------- features
def features(seq_tokens, ids):
    """seq_tokens: list of string tokens of one doc incl. '|' line ends."""
    n = len(seq_tokens)
    F = np.full((n, NT), PAD, dtype=np.int32)
    lines_prev, cur_line = None, []
    words_done = []          # list of previous words (lists of ids), across lines
    cur_word = []
    li = wi = cw = 0
    llw = 0
    x = [ids[t] for t in seq_tokens]
    for i, t in enumerate(seq_tokens):
        f = F[i]
        for j, d in enumerate([1, 2, 3, 4, 6, 8]):
            if i - d >= 0: f[j] = x[i - d]
        pl = len(cur_line)
        if lines_prev is not None and pl < len(lines_prev): f[6] = lines_prev[pl]
        pw = len(cur_word)
        if words_done and pw < len(words_done[-1]): f[7] = words_done[-1][pw]
        if len(words_done) > 1 and pw < len(words_done[-2]): f[8] = words_done[-2][pw]
        if cur_word: f[9] = cur_word[0]
        if cur_line: f[10] = cur_line[0]
        f[11] = min(pw, 255); f[12] = min(pl, 255); f[13] = min(wi, 255); f[14] = min(li, 255)
        f[15] = cw & 255
        f[16] = min(len(words_done[-1]), 255) if words_done else 0
        f[17] = min(llw, 255)
        # advance
        if t == '|':
            if cur_word: words_done.append(cur_word); cw += 1
            lines_prev = cur_line + []; cur_line = []; cur_word = []
            llw = wi + (1 if pl > 0 else 0); wi = 0; li += 1
        elif t == '_':
            if cur_word: words_done.append(cur_word); cw += 1
            cur_word = []; wi += 1; cur_line.append(x[i])
        else:
            cur_word.append(x[i]); cur_line.append(x[i])
    return F, x


def build(name, docs, kmax):
    """Write <name>.bin and <name>.json (meta)."""
    seqs_tok = [[t for l in d['lines'] for t in l + ['|']] for d in docs]
    cnt = Counter(t for s in seqs_tok for t in s)
    order = [t for t, _ in cnt.most_common()]
    ids = {t: (r if r < VMAX else VMAX) for r, t in enumerate(order)}
    V = min(len(order), VMAX + 1)
    feats, xs, splits = [], [], []
    for d, s in zip(docs, seqs_tok):
        F, x = features(s, ids)
        feats.append(F); xs.append(x); splits.append(d['split'])
    # baseline
    A = [i for i, s in enumerate(splits) if s == 'A']
    rng = random.Random(7)
    fold = {i: rng.random() < 0.5 for i in A}
    best = None
    for k in range(1, kmax + 1):
        m = KN([xs[i] for i in A], k, V)
        lb = sum(math.log2(p) for i, s in enumerate(splits) if s == 'B' for p in kn_probs(m, xs[i]))
        nB = sum(len(xs[i]) for i, s in enumerate(splits) if s == 'B')
        print(f'  {name} KN k={k} B bits/tok {-lb / nB:.4f}', flush=True)
        if best is None or lb > best[1]: best = (k, lb, m)
    k, _, m = best
    m0 = KN([xs[i] for i in A if fold[i]], k, V)
    m1 = KN([xs[i] for i in A if not fold[i]], k, V)
    pbs = []
    for i, s in enumerate(splits):
        mod = m if s != 'A' else (m1 if fold[i] else m0)
        pbs.append(np.array(kn_probs(mod, xs[i]), dtype=np.float64))
    X = np.concatenate([np.array(x, dtype=np.int32) for x in xs])
    F = np.concatenate(feats)
    SP = np.concatenate([np.full(len(x), 'ABC'.index(s), dtype=np.int32) for x, s in zip(xs, splits)])
    DOC = np.concatenate([np.full(len(x), j, dtype=np.int32) for j, x in enumerate(xs)])
    PB = np.concatenate(pbs)
    N = len(X)
    with open(os.path.join(OUT, name + '.bin'), 'wb') as fh:
        fh.write(struct.pack('iiii', N, NT, V, len(docs)))
        fh.write(X.astype(np.int32).tobytes()); fh.write(SP.astype(np.int32).tobytes())
        fh.write(DOC.astype(np.int32).tobytes()); fh.write(PB.astype(np.float64).tobytes())
        fh.write(F.astype(np.int32).T.copy().tobytes())
    meta = {'name': name, 'N': N, 'V': V, 'docs': len(docs), 'kn_order': k,
            'symbols': order[:VMAX], 'doc_ids': [d['id'] for d in docs],
            'baseline_bits_per_tok': {s: float(-np.log2(PB[SP == j]).mean()) for j, s in enumerate('ABC')},
            'n_split': {s: int((SP == j).sum()) for j, s in enumerate('ABC')}}
    json.dump(meta, open(os.path.join(OUT, name + '.json'), 'w'), indent=1)
    print(name, 'N', N, 'V', V, 'k', k, meta['baseline_bits_per_tok'], flush=True)
    return meta


KMAX = {'voynich': 5, 'linear_a': 3, 'proto_elamite': 3}


def main(which):
    real_docs = {n: real(n) for n in KMAX}
    jobs = []
    for n, docs in real_docs.items():
        km = KMAX[n]
        jobs.append((n, lambda docs=docs: docs, km))
        for s in (1, 2):
            jobs.append((f'{n}_shuf{s}', lambda docs=docs, s=s: shuffle_null(docs, 100 + s), km))
            jobs.append((f'{n}_mark{s}', lambda docs=docs, s=s, km=km: markov_null(docs, 200 + s, max(1, km - 1)), km))
    truth = {}
    def pg():
        o, t = planted_grille(real_docs['voynich']); truth['voynich_grille'] = t; return o
    def pl(n):
        o, t = planted_ledger(real_docs[n]); truth[n + '_ledger'] = t; return o
    jobs.append(('voynich_grille', pg, 5))
    jobs.append(('linear_a_ledger', lambda: pl('linear_a'), 3))
    jobs.append(('proto_elamite_ledger', lambda: pl('proto_elamite'), 3))
    jobs.append(('latin', latin, 5))
    jobs.append(('linear_b', linear_b, 3))
    jobs.append(('voynich_it2a', lambda: [{'id': d['id'], 'split': d['split'], 'lines': d['lines']}
                                          for d in L.load_voynich('IT2a')['docs']], 5))
    for name, fn, km in jobs:
        if which and name not in which: continue
        if os.path.exists(os.path.join(OUT, name + '.bin')): continue
        build(name, fn(), km)
    if truth:
        p = os.path.join(OUT, 'planted_truth.json')
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(truth); json.dump(old, open(p, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
