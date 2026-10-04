"""v6 cycle 2: geometric grid. Horizontal position = glyph count from the left margin (one unit per glyph, one per space).
(a) glyph grid: MI between a glyph and its right / lower / lower-right / lower-left neighbour;
(b) word grid by position: a word and the word in the next line that sits under its midpoint (and +-one word width);
(c) offset profile: same-prefix rate between a word and the next-line word at horizontal offset dx (is there a peak at 0?).
Nulls: lines permuted within paragraph (keeps rows) for vertical/diagonal; words permuted within line for horizontal.
Controls (same line widths in units as the Voynich paragraphs): verbose Latin written row-wise (prose, negative),
column-wise and diagonal-wise at glyph level (positives: a grille / table read down)."""
import sys, os, json, random, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v6_gridlib import voynich_paragraphs, latin_stream, encode_verbose, mi

REPS = int(os.environ.get('REPS', 20))

def voy_char_paras(name):
    return [['_'.join(''.join(vlib.glyphs(w)) for w in l) for l in p] for p in voynich_paragraphs(name)]

def control_char_paras(widths, stream, order):
    out, i = [], 0
    for ws in widths:
        R = len(ws); C = max(ws); cells = [(r, c) for r in range(R) for c in range(ws[r])]
        if order == 'row': cells.sort()
        elif order == 'col': cells.sort(key=lambda x: (x[1], x[0]))
        elif order == 'diag': cells.sort(key=lambda x: (x[1] - x[0], x[0]))
        g = [[' '] * w for w in ws]
        for (r, c) in cells:
            g[r][c] = stream[i]; i += 1
        out.append([''.join(l) for l in g])
    return out

def glyph_pairs(paras, kind):
    P = []
    for p in paras:
        for r, l in enumerate(p):
            for x, ch in enumerate(l):
                if kind == 'right':
                    if x + 1 < len(l): P.append((ch, l[x + 1]))
                elif r + 1 < len(p):
                    m = p[r + 1]; xx = x + {'down': 0, 'downright': 1, 'downleft': -1}[kind]
                    if 0 <= xx < len(m): P.append((ch, m[xx]))
    return P

def word_spans(line):
    out, x = [], 0
    for w in line.split('_'):
        if w: out.append((w, x, x + len(w)))
        x += len(w) + 1
    return out

def word_at(spans, x):
    for w, a, b in spans:
        if a <= x < b + 1: return w
    return None

def word_vert_pairs(paras, dx):
    P = []
    for p in paras:
        sp = [word_spans(l) for l in p]
        for r in range(len(p) - 1):
            for w, a, b in sp[r]:
                v = word_at(sp[r + 1], (a + b) / 2 + dx)
                if v: P.append((w, v))
    return P

def word_row_pairs(paras):
    P = []
    for p in paras:
        for l in p:
            ws = [w for w in l.split('_') if w]; P += list(zip(ws, ws[1:]))
    return P

def wstats(P, cnt):
    n = max(len(P), 1)
    cls = lambda w: w if cnt[w] >= 10 else '*' + w[0] + w[-1]
    return {'word': mi([(cls(a), cls(b)) for a, b in P]), 'junction': mi([(a[-1], b[0]) for a, b in P]),
            'first-first': mi([(a[0], b[0]) for a, b in P]),
            'same-word': sum(a == b for a, b in P) / n, 'same-prefix2': sum(a[:2] == b[:2] for a, b in P) / n}

def null(paras, kind, rng):
    if kind == 'line': return [rng.sample(p, len(p)) for p in paras]
    out = []
    for p in paras:
        q = []
        for l in p:
            ws = [w for w in l.split('_') if w]; rng.shuffle(ws); q.append('_'.join(ws))
        out.append(q)
    return out

def z(obs, sims):
    m = sum(sims) / len(sims); sd = (sum((s - m) ** 2 for s in sims) / (len(sims) - 1)) ** .5
    return obs - m, (obs - m) / sd if sd else 0.0

def run(paras, cnt):
    res = {}
    # (a) glyph grid
    for k in ('right', 'down', 'downright', 'downleft'):
        nk = 'inrow' if k == 'right' else 'line'
        obs = mi(glyph_pairs(paras, k))
        sims = [mi(glyph_pairs(null(paras, nk, random.Random(s)), k)) for s in range(REPS)]
        res['glyph-' + k] = dict(zip(('ex', 'z'), z(obs, sims)), obs=obs)
    # (b) words by position
    W = 5.3
    for k, P_of in (('row', lambda q: word_row_pairs(q)), ('below', lambda q: word_vert_pairs(q, 0)),
                    ('below-right', lambda q: word_vert_pairs(q, W + 1)), ('below-left', lambda q: word_vert_pairs(q, -W - 1))):
        nk = 'inrow' if k == 'row' else 'line'
        obs = wstats(P_of(paras), cnt); sims = [wstats(P_of(null(paras, nk, random.Random(s))), cnt) for s in range(REPS)]
        res['word-' + k] = {f: dict(zip(('ex', 'z'), z(obs[f], [s[f] for s in sims]))) for f in obs}
    # (c) offset profile, same-prefix2 and same-word, ratio to line-null
    prof = {}
    nulls = [null(paras, 'line', random.Random(s)) for s in range(10)]
    for dx in range(-16, 17, 2):
        P = word_vert_pairs(paras, dx); o = wstats(P, cnt)
        s = [wstats(word_vert_pairs(q, dx), cnt) for q in nulls]
        prof[dx] = {f: o[f] / (sum(x[f] for x in s) / len(s)) for f in ('same-word', 'same-prefix2')}
    res['offset'] = prof
    return res

out = {}
vz = voy_char_paras('ZL3b'); widths = [[len(l) for l in p] for p in vz]
lat = encode_verbose(latin_stream()); stream = list('_'.join(lat))
need = sum(map(sum, widths)); stream = (stream * 3)[:need]
corp = {'Voynich-ZL': vz, 'Voynich-IT': voy_char_paras('IT2a')}
for o in ('row', 'col', 'diag'):
    corp['Latin-verbose glyphs written ' + o] = control_char_paras(widths, stream, o)
for name, paras in corp.items():
    cnt = Counter(w for p in paras for l in p for w in l.split('_') if w)
    r = run(paras, cnt); out[name] = r
    print('\n==', name)
    for k in ('glyph-right', 'glyph-down', 'glyph-downright', 'glyph-downleft'):
        print('  %-16s MI excess %+.4f  z %6.1f' % (k, r[k]['ex'], r[k]['z']))
    for k in ('word-row', 'word-below', 'word-below-right', 'word-below-left'):
        print('  %-16s ' % k + '  '.join('%s %+.4f(z %.1f)' % (f, r[k][f]['ex'], r[k][f]['z']) for f in r[k]))
    print('  offset profile (obs/null) same-word:', ' '.join('%d:%.2f' % (d, v['same-word']) for d, v in r['offset'].items()))
    print('  offset profile (obs/null) same-prefix2:', ' '.join('%d:%.2f' % (d, v['same-prefix2']) for d, v in r['offset'].items()))
json.dump(out, open(os.path.join(vlib.RES, 'v6_cycle2.json'), 'w'), indent=0)
