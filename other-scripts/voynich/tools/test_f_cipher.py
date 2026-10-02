#!/usr/bin/env python3
"""Test (f): which writing/cipher class best reproduces the Voynich profile?
Profile = h1, h2, h3 (bits, char stream incl. word space), mean and sd of word
length in symbols, alphabet size (log2), top-5 symbol share.
Candidates, each built from real-language samples (20k words):
  simple substitution (= the plaintext's own profile; relabelling changes nothing),
  homophonic substitution (each letter -> one of 2 symbols at random),
  verbose cipher (each letter -> fixed code of 1-2 symbols, random codebooks),
  verbose cipher with positional codes (letter code depends on word-initial /
     medial / final position -> mimics slot structure),
  syllabary (word split into C*V+ syllables; each syllable = one sign),
  abjad (vowels dropped),
  '+respaced' variants: plaintext word boundaries discarded and the symbol stream
     cut into chunks with the Voynich word-length distribution.
Distance = mean over the profile of |x - V| / scale, scale = SD of that
statistic across the six plain-language samples (floor 0.05).
Ranked list = hypothesis ranking only."""
import sys, os, random, math, re
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
from test_a_language import unitize

VOW = set('aeiouyàèéìòùáíóúäöüâêîôûë')

def profile(words):
    e = entropies(words)
    lens = [len(w) for w in words]; m = sum(lens) / len(lens)
    sd = (sum((l - m) ** 2 for l in lens) / len(lens)) ** 0.5
    c = Counter(ch for w in words for ch in w); n = sum(c.values())
    top5 = sum(v for _, v in c.most_common(5)) / n
    return {'h1': e['h1'], 'h2': e['h2'], 'h3': e['h3'], 'wlen_mean': m, 'wlen_sd': sd,
            'log2_alphabet': math.log2(len(c)), 'top5_share': top5}

SYM = [chr(c) for c in range(0x100, 0x200)]  # symbol pool (private chars)

def homophonic(words, rng):
    letters = sorted({c for w in words for c in w})
    pool = iter(SYM); m = {c: (next(pool), next(pool)) for c in letters}
    return [''.join(rng.choice(m[c]) for c in w) for w in words]

def verbose(words, rng, nsym=20, p2=0.5):
    letters = sorted({c for w in words for c in w}); syms = SYM[:nsym]; used = set(); m = {}
    for c in letters:
        while True:
            k = 2 if rng.random() < p2 else 1
            code = ''.join(rng.choice(syms) for _ in range(k))
            if code not in used: used.add(code); m[c] = code; break
    return [''.join(m[c] for c in w) for w in words]

def verbose_positional(words, rng, nsym=20):
    """separate codebooks for word-initial, medial, final letters (1-2 symbols)."""
    letters = sorted({c for w in words for c in w}); syms = SYM[:nsym]
    books = []
    for _ in range(3):
        m = {}
        for c in letters:
            k = 2 if rng.random() < 0.5 else 1
            m[c] = ''.join(rng.choice(syms[:12] if _ != 1 else syms[8:]) for _ in range(k))
        books.append(m)
    out = []
    for w in words:
        s = ''
        for i, c in enumerate(w):
            b = 0 if i == 0 else (2 if i == len(w) - 1 else 1)
            s += books[b][c]
        out.append(s)
    return out

def syllabary(words):
    out, sylmap = [], {}
    pool = iter([chr(c) for c in range(0x1000, 0x4000)])
    for w in words:
        sy = re.findall(r'[^aeiouyàèéìòùáíóúäöüâêîôûë]*[aeiouyàèéìòùáíóúäöüâêîôûë]+|[^aeiouyàèéìòùáíóúäöüâêîôûë]+$', w)
        s = ''
        for x in sy:
            if x not in sylmap: sylmap[x] = next(pool)
            s += sylmap[x]
        if s: out.append(s)
    return out

def abjad(words):
    out = [''.join(c for c in w if c not in VOW) for w in words]
    return [w for w in out if w]

def respace(words, lens, rng):
    """ignore the plaintext word boundaries; cut the symbol stream into chunks
    whose lengths are drawn from the Voynich word-length distribution."""
    stream = ''.join(words); out = []; i = 0
    while i < len(stream):
        k = rng.choice(lens); out.append(stream[i:i + k]); i += k
    return out

def run():
    N = 20000
    V = {'Voynich-ZL-glyph': words_of(unitize(load_voynich('ZL3b', ('P',), True)))[:N],
         'Voynich-ZL-EVA': words_of(load_voynich('ZL3b', ('P',), True))[:N],
         'Voynich-IT-glyph': words_of(unitize(load_voynich('IT2a', ('P',), True)))[:N]}
    langs = {}
    for k in REFS:
        skip = 0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0.0
        langs[k] = words_of(load_ref(k, max_words=N, skip_frac=skip))[:N]
    cand = {}
    rng = random.Random(11)
    vlens = [len(w) for w in V['Voynich-ZL-glyph']]
    for k, ws in langs.items():
        cand[f'simple-substitution+respaced:{k}'] = [profile(respace(ws, vlens, random.Random(s))) for s in range(3)]
        cand[f'verbose-random+respaced:{k}'] = [profile(respace(verbose(ws, random.Random(s)), vlens, random.Random(s))) for s in range(5)]
        cand[f'simple-substitution:{k}'] = [profile(ws)]
        cand[f'homophonic:{k}'] = [profile(homophonic(ws, random.Random(s))) for s in range(3)]
        cand[f'verbose-random:{k}'] = [profile(verbose(ws, random.Random(s))) for s in range(5)]
        cand[f'verbose-positional:{k}'] = [profile(verbose_positional(ws, random.Random(s))) for s in range(5)]
        cand[f'syllabary:{k}'] = [profile(syllabary(ws))]
        cand[f'abjad:{k}'] = [profile(abjad(ws))]
    lang_profiles = [profile(ws) for ws in langs.values()]
    keys = list(lang_profiles[0])
    scale = {s: max(0.05, (sum((p[s] - sum(q[s] for q in lang_profiles) / 6) ** 2 for p in lang_profiles) / 6) ** 0.5) for s in keys}
    out = {'scale': scale, 'voynich': {}, 'ranking': {}}
    for vn, vw in V.items():
        vp = profile(vw); out['voynich'][vn] = vp
        rows = []
        for cn, ps in cand.items():
            avg = {s: sum(p[s] for p in ps) / len(ps) for s in keys}
            d = sum(abs(avg[s] - vp[s]) / scale[s] for s in keys) / len(keys)
            rows.append((cn, round(d, 2), {s: round(avg[s], 3) for s in keys}))
        rows.sort(key=lambda r: r[1])
        out['ranking'][vn] = rows
        print('==', vn, {s: round(v, 3) for s, v in vp.items()})
        for r in rows[:12]: print('  ', r[1], r[0], r[2])
        # class-level summary: best per class
        best = {}
        for cn, d, _ in rows:
            c = cn.split(':')[0]
            best.setdefault(c, (d, cn))
        print('  best per class:', sorted(best.values()))
        out.setdefault('best_per_class', {})[vn] = sorted(best.values())
    save('test_f_cipher', out)

if __name__ == '__main__':
    run()
