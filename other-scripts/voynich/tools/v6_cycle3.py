"""v6 cycle 3: acrostic / edge-column test. Read the paragraph's left edge (and right edge) top to bottom.
Sequences: first glyph of each line, line-initial word, last glyph of each line, line-final word.
Statistic: MI between consecutive members of the edge column (lag 1 and lag 2), excess over permuting the lines
inside the paragraph (keeps every row and every positional marginal). Variant: paragraph-first line excluded
(then only lines 2..n are permuted). Also: 'junction' (last glyph of edge word r -> first glyph of edge word r+1).
Controls (verbose-encoded, same paragraph shapes): Latin prose (negative); letter acrostic (first letters of lines
spell a Latin text); word acrostic (first words of lines are a running Latin text, cover text after them)."""
import sys, os, json, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v6_gridlib import voynich_paragraphs, shapes_of, encode_verbose, mi, units_voy, units_plain

REPS = int(os.environ.get('REPS', 200))

def edge_seqs(paras, U, skip_first):
    seqs = {'first-glyph': [], 'initial-word': [], 'last-glyph': [], 'final-word': [], 'init-junction': []}
    for p in paras:
        q = p[1:] if skip_first else p
        seqs['first-glyph'].append([U(l[0])[0] for l in q])
        seqs['initial-word'].append([l[0] for l in q])
        seqs['last-glyph'].append([U(l[-1])[-1] for l in q])
        seqs['final-word'].append([l[-1] for l in q])
        seqs['init-junction'].append([(U(l[0])[0], U(l[0])[-1]) for l in q])
    return seqs

def col_mi(seqs, lag, kind, cls):
    if kind == 'init-junction':
        P = [(s[i][1], s[i + lag][0]) for s in seqs for i in range(len(s) - lag)]
    else:
        P = [(cls(s[i]), cls(s[i + lag])) for s in seqs for i in range(len(s) - lag)]
    return mi(P)

def test(paras, U, skip_first):
    cnt = Counter(w for p in paras for l in p for w in (l[0], l[-1]))
    cls = lambda w: w if (not isinstance(w, str) or len(w) <= 1 or cnt[w] >= 5) else '*'
    obs_s = edge_seqs(paras, U, skip_first); res = {}
    sims = {}
    for s in range(REPS):
        rng = random.Random(s)
        sh = [([p[0]] + rng.sample(p[1:], len(p) - 1)) if skip_first else rng.sample(p, len(p)) for p in paras]
        ss = edge_seqs(sh, U, skip_first)
        for k in ss:
            for lag in (1, 2):
                sims.setdefault((k, lag), []).append(col_mi(ss[k], lag, k, cls))
    for k in obs_s:
        for lag in (1, 2):
            o = col_mi(obs_s[k], lag, k, cls); xs = sims[(k, lag)]; m = sum(xs) / len(xs)
            sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** .5
            p = (1 + sum(x >= o for x in xs)) / (1 + len(xs))
            res['%s lag%d' % (k, lag)] = {'obs': o, 'ex': o - m, 'z': (o - m) / sd if sd else 0, 'p': p}
    return res

# ---- controls ----
def latin_words(key): return vlib.words_of(vlib.load_ref(key))

def letter_acrostic(shapes, cover, msg_letters):
    out, i, k = [], 0, 0
    for s in shapes:
        g = []
        for n in s:
            want = msg_letters[k % len(msg_letters)]; k += 1
            j = i
            while not cover[j % len(cover)].startswith(want) and j - i < 5000: j += 1
            g.append([cover[(j + t) % len(cover)] for t in range(n)]); i = j + n
        out.append(g)
    return out

def word_acrostic(shapes, cover, msg_words):
    out, i, k = [], 0, 0
    for s in shapes:
        g = []
        for n in s:
            g.append([msg_words[k % len(msg_words)]] + [cover[(i + t) % len(cover)] for t in range(n - 1)])
            k += 1; i += n - 1
        out.append(g)
    return out

def prose(shapes, cover):
    out, i = [], 0
    for s in shapes:
        g = []
        for n in s: g.append([cover[(i + t) % len(cover)] for t in range(n)]); i += n
        out.append(g)
    return out

def encode_grid(paras, code_words):
    flat = [w for p in paras for l in p for w in l]
    enc = dict(zip(flat, encode_verbose(flat))) if False else None
    E = encode_verbose(sorted(set(flat)) )
    m = dict(zip(sorted(set(flat)), E))
    return [[[m[w] for w in l] for l in p] for p in paras]

if __name__ == '__main__':
    vz = voynich_paragraphs('ZL3b'); shapes = shapes_of(vz)
    caes = latin_words('Latin-Caesar'); desc = latin_words('Latin-Descartes')
    msg_letters = [c for w in desc for c in w]
    corp = {'Voynich-ZL': (vz, units_voy), 'Voynich-IT': (voynich_paragraphs('IT2a'), units_voy),
            'Latin prose (neg)': (encode_grid(prose(shapes, caes), None), units_plain),
            'Latin letter-acrostic (pos)': (encode_grid(letter_acrostic(shapes, caes, msg_letters), None), units_plain),
            'Latin word-acrostic (pos)': (encode_grid(word_acrostic(shapes, caes, desc), None), units_plain)}
    for L in ('A', 'B'):
        corp['Voynich-ZL lang ' + L] = (voynich_paragraphs('ZL3b', lang=L), units_voy)
    out = {}
    for name, (paras, U) in corp.items():
        for sf in (False, True):
            r = test(paras, U, sf); out['%s skipfirst=%s' % (name, sf)] = r
            print('\n== %s (paragraph-first line %s)' % (name, 'excluded' if sf else 'included'))
            for k, v in r.items():
                print('  %-22s MI %.4f excess %+.4f z %6.2f p %.3f' % (k, v['obs'], v['ex'], v['z'], v['p']))
    json.dump(out, open(os.path.join(vlib.RES, 'v6_cycle3.json'), 'w'), indent=0)
