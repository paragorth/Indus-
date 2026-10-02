"""Control texts: shuffles and simple generators. All keep the line/word-count
structure of the input lines so position tests can be run on them too.
Words are plain strings; in glyph mode pass unit-strings (one char per glyph)."""
import random
from collections import Counter, defaultdict

def _relayout(lines, new_words):
    out, i = [], 0
    for L in lines:
        k = len(L['words'])
        nl = dict(L); nl['words'] = new_words[i:i + k]; i += k
        out.append(nl)
    return out

def char_shuffle(lines, seed=1):
    """Permute all characters across the corpus; word lengths fixed."""
    rng = random.Random(seed)
    words = [w for L in lines for w in L['words']]
    chars = [c for w in words for c in w]; rng.shuffle(chars)
    out, i = [], 0
    for w in words:
        out.append(''.join(chars[i:i + len(w)])); i += len(w)
    return _relayout(lines, out)

def word_shuffle(lines, seed=1):
    rng = random.Random(seed)
    words = [w for L in lines for w in L['words']]; rng.shuffle(words)
    return _relayout(lines, words)

def within_line_shuffle(lines, seed=1):
    rng = random.Random(seed)
    out = []
    for L in lines:
        ws = list(L['words']); rng.shuffle(ws)
        nl = dict(L); nl['words'] = ws; out.append(nl)
    return out

def char_markov(lines, order=1, seed=1):
    """Order-k character Markov model with '_' word boundary, trained on the text."""
    rng = random.Random(seed)
    words = [w for L in lines for w in L['words']]
    trans = defaultdict(Counter)
    for w in words:
        s = '^' * order + w + '_'
        for i in range(order, len(s)):
            trans[s[i - order:i]][s[i]] += 1
    tabs = {k: (list(v.keys()), list(v.values())) for k, v in trans.items()}
    out = []
    for _ in words:
        ctx, w = '^' * order, ''
        while True:
            ks, vs = tabs[ctx]
            c = rng.choices(ks, vs)[0]
            if c == '_' or len(w) > 25:
                break
            w += c; ctx = (ctx + c)[-order:]
        out.append(w if w else rng.choice(words))
    return _relayout(lines, out)

def table_grille(lines, seed=1):
    """Rugg-style three-column table: each word split into prefix/middle/suffix
    thirds; new words = independent draws from the three column frequency tables."""
    rng = random.Random(seed)
    words = [w for L in lines for w in L['words']]
    P, M, S = Counter(), Counter(), Counter()
    for w in words:
        n = len(w); a = round(n / 3); b = round(2 * n / 3)
        P[w[:a]] += 1; M[w[a:b]] += 1; S[w[b:]] += 1
    cols = [(list(c.keys()), list(c.values())) for c in (P, M, S)]
    out = []
    for _ in words:
        w = ''.join(rng.choices(k, v)[0] for k, v in cols)
        out.append(w if w else rng.choice(words))
    return _relayout(lines, out)

def self_citation(lines, seed=1, window=60, p_mod=0.5):
    """Copy-and-modify generator: each word is a copy of a random word from the
    previous `window` generated words, modified (one substitution/insertion/
    deletion; new glyph drawn from the glyph-bigram table given its left neighbour) with prob p_mod. Seeded with the first
    `window` real words."""
    rng = random.Random(seed)
    words = [w for L in lines for w in L['words']]
    big = defaultdict(Counter)
    for w in words:
        for a, b in zip('^' + w, w):
            big[a][b] += 1
    bt = {a: (list(c.keys()), list(c.values())) for a, c in big.items()}
    def draw(prev):
        k, v = bt.get(prev, bt['^']); return rng.choices(k, v)[0]
    lens = [len(w) for w in words]
    out = list(words[:window])
    while len(out) < len(words):
        w = list(rng.choice(out[-window:]))
        if rng.random() < p_mod:
            target = rng.choice(lens)   # restoring force on length
            if len(w) < target:
                pos = rng.randrange(len(w) + 1); w.insert(pos, draw(w[pos - 1] if pos else '^'))
            elif len(w) > target and len(w) > 1:
                w.pop(rng.randrange(len(w)))
            elif w:
                pos = rng.randrange(len(w)); w[pos] = draw(w[pos - 1] if pos else '^')
        out.append(''.join(w) if w else rng.choice(words))
    return _relayout(lines, out)

GENERATORS = {
    'char_shuffle': char_shuffle,
    'char_bigram_markov': lambda L, seed=1: char_markov(L, 1, seed),
    'char_trigram_markov': lambda L, seed=1: char_markov(L, 2, seed),
    'table_grille': table_grille,
    'self_citation': self_citation,
}
