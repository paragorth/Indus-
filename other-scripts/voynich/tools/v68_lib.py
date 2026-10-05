"""v68 THE MELODY IS THE PLAINTEXT (shared helpers).

The musical-notation reading of the Voynich (glyph = note, finals, modes, transposition) was already
tested against GregoBase chant with controls in v5 (negative).  v68 turns the idea round: the
melody is not what the glyphs *are*, it is what was *written down*.  v53 showed that evolved
meaning-preserving encoders of language plaintexts cannot reproduce three Voynich surface traits
(line-edge pools, neighbour-word coupling, adjacent repeats).  A melody supplies exactly such
traits for free (phrase-initial intonations and finals, stepwise continuity across syllables,
repeated notes and neumes).  So: write real chant through thousands of random neume codes and
ask whether chant-as-source reaches the Voynich where language-as-source through the same codes
does not.

Sources are lists of lines; a line is a list of units; a unit is a tuple of small ints.
  chant : unit = notes of one syllable (staff letters a..m -> 0..12), line = phrase between bars
  lang  : unit = word or syllable, symbols = letters (random letter -> int map per encoder)
Chant data: GregoBase dump 2019-10-24 (github.com/bacor/gregobasecorpus, commit ad5060fe, CC0),
  parsed in v5 (data/derived/v5_chant.json.gz).
"""
import json, math, os, random, sys, gzip
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v53_lib as L53

DATA = os.path.join(os.path.dirname(HERE), 'data')
CK = os.path.join(DATA, 'v68_ckpt')
os.makedirs(CK, exist_ok=True)

VGLYPHS = list('oedyainlrskqtpfCSTKPFgmhx')   # vglyphs alphabet (C=ch S=sh T=cth K=ckh P=cph F=cfh)


# ------------------------------------------------------------------ sources
def chant_units(seed=0):
    d = json.load(gzip.open(os.path.join(DATA, 'derived', 'v5_chant.json.gz'), 'rt'))
    out, seen = [], set()
    for x in d:
        m = (x['mode'] or '').strip()
        if m not in list('12345678'):
            continue
        key = (x['incipit'], m, len(x['phrases']))
        if key in seen:
            continue
        seen.add(key)
        lines = []
        for p in x['phrases']:
            l = [tuple(ord(c) - 97 for c in w if 'a' <= c <= 'm') for w in p]
            l = [u for u in l if u]
            if len(l) >= 2:
                lines.append(l)
        if lines:
            out.append({'id': x['id'], 'mode': m, 'part': x['part'], 'lines': lines})
    random.Random(seed).shuffle(out)
    return out


VOW = set('aeiouy')


def syllabify(w):
    """crude orthographic syllables: split before a consonant that precedes a vowel."""
    if len(w) <= 2:
        return [w]
    out, cur = [], ''
    for i, c in enumerate(w):
        cur += c
        nxt = w[i + 1] if i + 1 < len(w) else ''
        nxt2 = w[i + 2] if i + 2 < len(w) else ''
        if c in VOW and nxt and nxt not in VOW and nxt2 in VOW:
            out.append(cur); cur = ''
        elif c not in VOW and nxt and nxt not in VOW and nxt2 in VOW and any(x in VOW for x in cur):
            out.append(cur); cur = ''
    if cur:
        if out and not any(x in VOW for x in cur):
            out[-1] += cur
        else:
            out.append(cur)
    return out


def lang_lines(name, syll=False):
    C = L53.load_corpora()
    ls = []
    for l in C[name][2]:
        units = []
        for w in l:
            for s in (syllabify(w) if syll else [w]):
                units.append(tuple(ord(c) - 97 for c in s if 'a' <= c <= 'z'))
        units = [u for u in units if u]
        if units:
            ls.append(units)
    return ls


def chant_lines(pieces):
    return [l for p in pieces for l in p['lines']]


def rewrap(lines, width, rng):
    flat = [u for l in lines for u in l]
    out, i = [], 0
    while i < len(flat):
        w = max(3, int(rng.gauss(width, 2)))
        out.append(flat[i:i + w]); i += w
    return out


def neume_shuffle(lines, rng):
    out = []
    for l in lines:
        l = list(l); rng.shuffle(l); out.append(l)
    return out


def markov_melody(lines, rng):
    """note-level Markov-1 resynthesis over the whole corpus, segmented with the original unit lengths."""
    T = {}
    for l in lines:
        seq = [n for u in l for n in u]
        for a, b in zip(seq, seq[1:]):
            T.setdefault(a, []).append(b)
    starts = [l[0][0] for l in lines]
    out = []
    for l in lines:
        cur = rng.choice(starts); nl = []
        for u in l:
            nu = []
            for _ in u:
                nu.append(cur)
                cur = rng.choice(T.get(cur, starts))
            nl.append(tuple(nu))
        out.append(nl)
    return out


# ------------------------------------------------------------------ encoders
REPS = ['abs', 'abs', 'int', 'absint', 'contour', 'ends', 'absrun']
GROUPS = ['one', 'one', 'merge1', 'split6', 'split4', 'pair', 'rand13', 'rand13']


def random_encoder(rng):
    return {'rep': rng.choice(REPS), 'fold': rng.random() < 0.3, 'group': rng.choice(GROUPS),
            'trunc': rng.choice([0, 0, 0, 5, 7]), 'rank': rng.random() < 0.6,
            'p2': rng.choice([0.0, 0.15, 0.3, 0.5]), 'seed': rng.randrange(10 ** 9),
            'sympermute': rng.random() < 1.0}


def _rep_unit(u, rep):
    if rep == 'abs':
        return [('A', n) for n in u]
    if rep == 'absrun':                       # abs with repeated notes as a count token
        out, i = [], 0
        while i < len(u):
            j = i
            while j < len(u) and u[j] == u[i]:
                j += 1
            out.append(('A', u[i]))
            if j - i > 1:
                out.append(('R', min(j - i, 3)))
            i = j
        return out
    if rep == 'int':
        return [('S', u[0])] + [('I', max(-4, min(4, b - a))) for a, b in zip(u, u[1:])]
    if rep == 'absint':
        return [('A', u[0])] + [('I', max(-4, min(4, b - a))) for a, b in zip(u, u[1:])]
    if rep == 'contour':
        return [('A', u[0])] + [('D', (b > a) - (b < a)) for a, b in zip(u, u[1:])]
    if rep == 'ends':
        return [('A', u[0]), ('N', min(len(u), 4))] + ([('E', u[-1])] if len(u) > 1 else [])
    raise ValueError(rep)


def encode(lines, enc, fold_mod=7, symmap=None):
    """lines of int-tuples -> lines of glyph-string words under encoder enc."""
    rng = random.Random(enc['seed'])
    if symmap is not None:
        lines = [[tuple(symmap[n] for n in u) for u in l] for l in lines]
    if enc['fold']:
        lines = [[tuple(n % fold_mod for n in u) for u in l] for l in lines]
    # grouping
    gl = []
    for l in lines:
        g = []
        if enc['group'] == 'merge1':
            buf = ()
            for u in l:
                buf = buf + u
                if len(buf) >= 2:
                    g.append(buf); buf = ()
            if buf:
                g.append(buf)
        elif enc['group'] in ('pair', 'rand13'):
            i = 0
            while i < len(l):
                k = 2 if enc['group'] == 'pair' else rng.choice([1, 1, 2, 2, 3])
                g.append(tuple(n for u in l[i:i + k] for n in u)); i += k
        elif enc['group'].startswith('split'):
            k = int(enc['group'][5:])
            for u in l:
                for i in range(0, len(u), k):
                    g.append(u[i:i + k])
        else:
            g = list(l)
        if enc['trunc']:
            g = [u[:enc['trunc']] for u in g]
        gl.append(g)
    toks = [[_rep_unit(u, enc['rep']) for u in l] for l in gl]
    cnt = Counter(t for l in toks for u in l for t in u)
    syms = [t for t, _ in cnt.most_common()]
    # table: token -> glyph string (1 glyph, or 2 with prob p2); rank-matched to Voynich glyph freq or random
    glyph_rank = VGLYPHS[:]
    if not enc['rank']:
        rng.shuffle(glyph_rank)
    table, used = {}, set()
    gi = 0
    for t in syms:
        for _ in range(200):
            if rng.random() < enc['p2'] or gi >= len(glyph_rank):
                s = rng.choice(glyph_rank[:12]) + rng.choice(glyph_rank[:12])
            else:
                s = glyph_rank[gi]; gi += 1
            if s not in used:
                break
        used.add(s); table[t] = s
    return [[''.join(table[t] for t in u) for u in l if u] for l in toks]


def random_symmap(rng, n=26, k=13):
    """language letters -> 'pitch' ints 0..k-1 (random, so interval codes are arbitrary)."""
    return [rng.randrange(k) for _ in range(n)]


# ------------------------------------------------------------------ scoring
FEAT = ['h2', 'irr', 'mi_lf', 'mi_ll', 'mi_j', 'adjrep', 'linerep', 'ttr', 'hapax', 'wpl', 'mwl']


def sample_lines(lines, ntok, rng):
    """contiguous block of lines with about ntok tokens from a random start."""
    tot = sum(len(l) for l in lines)
    if tot <= ntok:
        return lines
    s = rng.randrange(len(lines))
    out, n, i = [], 0, s
    while n < ntok:
        out.append(lines[i % len(lines)]); n += len(lines[i % len(lines)]); i += 1
    return out


def profile(lines, nchunks=3, ntok=2000, seed=0):
    rng = random.Random(seed)
    ps = []
    for _ in range(nchunks):
        p = L53.panel(sample_lines(lines, ntok + 50, rng), ntok)
        if p:
            ps.append(p)
    if not ps:
        return None
    out = {k: sum(p[k] for p in ps) / len(ps) for k in FEAT}
    return out, ps


def score(lines, T, nchunks=3, seed=0):
    r = profile(lines, nchunks, seed=seed)
    if r is None:
        return 99.0, None
    m, ps = r
    ds = [L53.distance(p, T)[0] for p in ps]
    return sum(ds) / len(ds), m
