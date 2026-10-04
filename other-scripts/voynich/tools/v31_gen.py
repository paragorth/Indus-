"""v31 generators (class GEN). None is trained on the Voynich. Each returns one document (list of lines of words).

  self-citation (copy an earlier word, mutate it; Timm & Schinner's idea, re-implemented from its description)
  grille (Rugg's table-and-grille idea: prefix / root / suffix columns read through a moving mask)
  wheel (Llull-like combinatory wheel stepping through syllable combinations)
  char-trigram words (each word independent, trained on Latin or German)
  word-bigram resynthesis (trained on Latin or Italian)
  slot template (random slot grammar), uniform random strings
Test-only generators trained on the Voynich (v21 forgers) are built in v31_cycle1.py.
"""
import random, re, json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V = 'aeiou'; C = 'bcdfghklmnprstvz'


def syll(rng, alph_v=V, alph_c=C):
    k = rng.random()
    if k < 0.5: return rng.choice(alph_c) + rng.choice(alph_v)
    if k < 0.8: return rng.choice(alph_c) + rng.choice(alph_v) + rng.choice(alph_c)
    return rng.choice(alph_v) + rng.choice(alph_c)


def wrap(words, rng, lo=7, hi=12):
    out = []; i = 0
    while i < len(words):
        k = rng.randint(lo, hi); out.append(words[i:i + k]); i += k
    return out


def self_citation(seed, n=3000, window=40, pmut=0.7, nseed=8):
    rng = random.Random(seed)
    sub = {}
    letters = list(V + C)
    for c in letters:   # each unit has 1-2 'similar' units it may turn into
        sub[c] = rng.sample([x for x in letters if (x in V) == (c in V)], 2)
    words = [''.join(syll(rng) for _ in range(rng.randint(1, 3))) for _ in range(nseed)]
    while len(words) < n:
        w = rng.choice(words[-window:])
        if rng.random() < pmut:
            for _ in range(rng.choice([1, 1, 2])):
                r = rng.random(); i = rng.randrange(len(w))
                if r < 0.6: w = w[:i] + rng.choice(sub[w[i]]) + w[i + 1:]
                elif r < 0.8 and len(w) < 9: w = w + syll(rng)[:rng.randint(1, 2)]
                elif len(w) > 3: w = w[:i] + w[i + 1:]
        words.append(w)
    return wrap(words, rng)


def grille(seed, n=3000, rows=36):
    rng = random.Random(seed)
    T = [[''.join(syll(rng) for _ in range(rng.randint(0, 1))) or rng.choice(V),
          syll(rng) + (syll(rng) if rng.random() < .3 else ''),
          rng.choice(['', '', rng.choice(C), syll(rng)])] for _ in range(rows)]
    words = []; r = 0
    while len(words) < n:
        off = [rng.randint(0, 2) for _ in range(3)]
        for k in range(rng.randint(3, 8)):
            rr = (r + k) % rows
            words.append(T[rr][0] + T[(rr + off[1]) % rows][1] + T[(rr + off[2]) % rows][2])
        r = (r + rng.randint(1, 5)) % rows
    return wrap(words[:n], rng)


def wheel(seed, n=3000, k=12):
    rng = random.Random(seed)
    S = list({syll(rng) for _ in range(k * 2)})[:k]
    a = b = c = 0; words = []
    while len(words) < n:
        step = rng.choice([1, 1, 1, 2, 3])
        c += step
        if c >= k: c -= k; b += 1
        if b >= k: b = 0; a = (a + 1) % k
        L = rng.choice([2, 2, 3])
        words.append((S[a] + S[b] + S[c])[: 2 * L + 1] if L == 2 else S[a] + S[b] + S[c])
    return wrap(words, rng)


def char_trigram(train_words, seed, n=3000):
    rng = random.Random(seed)
    T = defaultdict(Counter)
    for w in train_words:
        x = '^^' + w + '$'
        for i in range(2, len(x)): T[x[i - 2:i]][x[i]] += 1
    T = {k: (list(v), list(v.values())) for k, v in T.items()}
    words = []
    while len(words) < n:
        x = '^^'
        while len(x) < 16:
            ks, ws = T[x[-2:]]
            c = rng.choices(ks, ws)[0]
            if c == '$': break
            x += c
        if len(x) > 2: words.append(x[2:])
    return wrap(words, rng)


def word_bigram(train_words, seed, n=3000):
    rng = random.Random(seed)
    T = defaultdict(Counter)
    for a, b in zip(train_words, train_words[1:]): T[a][b] += 1
    T = {k: (list(v), list(v.values())) for k, v in T.items()}
    w = rng.choice(train_words); words = []
    while len(words) < n:
        if w not in T: w = rng.choice(train_words)
        ks, ws = T[w]; w = rng.choices(ks, ws)[0]; words.append(w)
    return wrap(words, rng)


def slot_template(seed, n=3000, nslots=6):
    rng = random.Random(seed)
    slots = []
    for s in range(nslots):
        opts = [''.join(rng.choice(V if (s + j) % 2 else C) for j in range(rng.randint(1, 2))) for _ in range(rng.randint(2, 5))]
        slots.append((opts, rng.uniform(0.2, 0.9)))
    words = []
    while len(words) < n:
        w = ''.join(rng.choice(o) for o, p in slots if rng.random() < p)
        if w: words.append(w)
    return wrap(words, rng)


def uniform(train_words, seed, n=3000):
    rng = random.Random(seed)
    L = [len(w) for w in train_words]
    al = list(Counter(c for w in train_words for c in w))
    return wrap([''.join(rng.choice(al) for _ in range(rng.choice(L))) for _ in range(n)], rng)


def build(C):
    """C: corpora dict from v31_corpora (to borrow Latin / German / Italian training words)."""
    def words(k): return [w for d in C[k]['docs'] for l in d for w in l]
    la = words('L_Isidore'); de = words('L_msG_Bav2') if 'L_msG_Bav2' in C else words('L_pgKafka')
    it = words('L_pgDante')
    G = {}
    G['G_selfcit1'] = self_citation(1)
    G['G_selfcit2'] = self_citation(2, window=80, pmut=0.5)
    G['G_selfcit3'] = self_citation(3, window=20, pmut=0.9)
    G['G_grille1'] = grille(4); G['G_grille2'] = grille(5, rows=60)
    G['G_wheel1'] = wheel(6); G['G_wheel2'] = wheel(7, k=16)
    G['G_tri_la'] = char_trigram(la, 8); G['G_tri_de'] = char_trigram(de, 9)
    G['G_big_la'] = word_bigram(la, 10); G['G_big_it'] = word_bigram(it, 11)
    G['G_slot1'] = slot_template(12); G['G_slot2'] = slot_template(13, nslots=8)
    G['G_unif_la'] = uniform(la, 14)
    return G
