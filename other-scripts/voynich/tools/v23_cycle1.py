"""v23 cycle 1: the irreversibility battery on real texts, Voynich, surrogates and planted controls.

Usage: python3 v23_cycle1.py [corpus ...]   (writes data/v23_ckpt/c1_<corpus>.json)
"""
import sys, os, json, math, random, lzma, zlib, time
from collections import Counter
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L

NULL_DRAWS = 300


def build(name):
    if name in ('ZL', 'IT'):
        return L.voynich('ZL3b' if name == 'ZL' else 'IT2a')
    if name == 'ZL19': return L.subsample_pages(L.voynich('ZL3b'), 19000, seed=1)
    if name == 'LA': return L.ref('Latin-Caesar')
    if name == 'ITA': return L.ref('Italian-Manzoni')
    if name == 'DE': return L.ref('German-Kafka')
    if name == 'CS': return L.czech()
    if name == 'HE': return L.hebrew()
    if name == 'HEvis':      # Hebrew stored in visual (left-to-right display) order: every line reversed
        return [[[[w[::-1] for w in l[::-1]] for l in pa] for pa in p] for p in L.hebrew()]
    if name == 'PL_REV': return [L.rev_page(p) for p in L.ref('Latin-Descartes')]
    if name == 'PL_WREV': return [L.rev_words_only(p) for p in L.ref('Italian-Dante')]
    if name.startswith('MkG_'): return L.markov_glyph(L.voynich('ZL3b'), int(name[4:]), k=3)
    if name.startswith('MkW_'): return L.markov_word(L.voynich('ZL3b'), int(name[4:]))
    if name.startswith('RvG_'): return L.reversible_glyph(L.voynich('ZL3b'), int(name[4:]))
    if name.startswith('RvW_'): return L.reversible_word(L.voynich('ZL3b'), int(name[4:]))
    if name.startswith('RvGL_'): return L.reversible_glyph(L.ref('Latin-Caesar'), int(name[5:]))
    raise KeyError(name)


# ---------------------------------------------------------------- signed signatures (per page)
def model_sigs(C):
    out = {}
    pg_stream = [[L.stream(p)] for p in C]
    for n in (2, 4, 6):
        r = L.kn_arrow(pg_stream, n)
        out[f'G{n}_stream'] = r
    words = [[w for w in L.tokens(p)] for p in C]
    out['G4_inword'] = L.kn_arrow(words, 4)        # each word its own sequence: within-word arrow only
    # across-word only: sequence of (last glyph of word, first glyph of next word) junction chain
    junc = [[[x for w in l for x in (w[0], w[-1])] for l in L.page_lines(p)] for p in C]
    out['J2_junction'] = L.kn_arrow(junc, 2)
    ws = [[L.wstream(p)] for p in C]
    out['W2_word'] = L.kn_arrow(ws, 2, unk_min=2)
    out['W3_word'] = L.kn_arrow(ws, 3, unk_min=2)
    lens = [[[str(min(len(w), 9)) for w in L.tokens(p)]] for p in C]
    out['W3_length'] = L.kn_arrow(lens, 3)
    fl = [[[w[0] + w[-1] for w in L.tokens(p)]] for p in C]
    out['W2_firstlast'] = L.kn_arrow(fl, 2)
    # line level: each line coded by (first glyph of line, last glyph of line, length bin)
    lc = [[[l[0][0] + l[-1][-1] + str(min(len(l), 12) // 3) for pa in p for l in pa]] for p in C]
    out['L2_linecode'] = L.kn_arrow(lc, 2)
    # convert to per-page a_p in bits (bwd - fwd) and totals
    res = {}
    for k, r in out.items():
        a = [b - f for f, b, n in r]; ns = sum(n for _, _, n in r); hf = sum(f for f, _, _ in r) / ns
        res[k] = {'a': a, 'eff': sum(a) / ns, 'rel': sum(a) / ns / hf, 'unit': 'bits/sym (bwd-fwd)'}
    return res


def compress_sigs(C):
    res = {}
    for nm, fn in (('Z_lzma', lambda b: len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))),
                   ('Z_zlib', lambda b: len(zlib.compress(b, 9)))):
        a, ns = [], 0
        for p in C:
            s = L.stream(p)
            f = fn(s.encode('utf-8')); b = fn(s[::-1].encode('utf-8'))
            a.append(8 * (b - f)); ns += len(s)
        res[nm] = {'a': a, 'eff': sum(a) / ns, 'rel': None, 'unit': 'bits/char (rev-fwd)'}
    return res


def page_sigs(C, gf):
    """signed per-page statistics s(page) with a_p = s(p) - s(R p)"""
    def intro_reuse(seq):
        # mean over repeated types of (rel. first pos + rel. last pos - 1); antisymmetric under reversal
        n = len(seq)
        if n < 10: return None
        first, last, cnt = {}, {}, Counter(seq)
        for i, w in enumerate(seq):
            first.setdefault(w, i); last[w] = i
        v = [(first[w] + last[w]) / (n - 1) - 1 for w in cnt if cnt[w] >= 2]
        return float(np.mean(v)) if v else None

    def novelty_len(seq):
        n = len(seq)
        if n < 10: return None
        first, last = {}, {}
        for i, w in enumerate(seq):
            first.setdefault(w, i); last[w] = i
        isf = np.array([first[w] == i for i, w in enumerate(seq)], float)
        isl = np.array([last[w] == i for i, w in enumerate(seq)], float)
        ln = np.array([len(w) for w in seq], float)
        # mean length of first occurrences minus mean length of last occurrences (repeated types only)
        rep = np.array([first[w] != last[w] for w in seq])
        if rep.sum() < 4: return None
        return float(ln[(isf == 1) & rep].mean() - ln[(isl == 1) & rep].mean())

    def mutation(seq, key):
        # near-copies (edit distance 1) within 1..5 words: later member rarer (key=freq) / longer (key=len)
        s, m = 0, 0
        for i in range(len(seq)):
            for j in range(i + 1, min(len(seq), i + 6)):
                a, b = seq[i], seq[j]
                if L.ed1(a, b):
                    if key == 'freq':
                        d = L.sgn(gf[a] - gf[b])     # +1 if later (b) is rarer
                    else:
                        d = L.sgn(len(b) - len(a))   # +1 if later is longer
                    s += d; m += 1
        return s

    def ordinal_signed(series_list):
        c = L.ordinal_counts(series_list)
        return (c[(1, 1)] - c[(-1, -1)]) + (c[(1, 0)] - c[(0, -1)]) - (c[(0, 1)] - c[(-1, 0)])

    def para_intro(p):
        v = [intro_reuse([w for l in pa for w in l]) for pa in p]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    def line_trend(p):
        # within paragraphs: sum over consecutive lines of sign(len(next) - len(prev)) (in glyphs)
        s = 0
        for pa in p:
            ll = [sum(len(w) for w in l) for l in pa]
            s += sum(L.sgn(b - a) for a, b in zip(ll, ll[1:]))
        return s

    def word_trend_line(p):
        # within lines: correlation-free monotone trend of word length (sum sign of later-minus-earlier)
        s = 0
        for l in L.page_lines(p):
            x = [len(w) for w in l]
            s += sum(L.sgn(x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)))
        return s

    fns = {
        'I_intro_page': lambda p: intro_reuse(L.tokens(p)),
        'I_intro_para': para_intro,
        'I_novel_len': lambda p: novelty_len(L.tokens(p)),
        'M_mut_freq': lambda p: mutation(L.tokens(p), 'freq'),
        'M_mut_len': lambda p: mutation(L.tokens(p), 'len'),
        'O_len_line': lambda p: ordinal_signed([[len(w) for w in l] for l in L.page_lines(p)]),
        'O_freq_page': lambda p: ordinal_signed([[math.log(gf[w] + 1) for w in L.tokens(p)]]),
        'T_wlen_line': word_trend_line,
        'T_linelen_para': line_trend,
    }
    res = {}
    for k, fn in fns.items():
        a = []
        for p in C:
            x = fn(p); y = fn(L.rev_page(p))
            if k.startswith('M_'):
                # reversal maps word w -> w[::-1]; frequency/length of reversed words: use the same gf via key
                pass
            a.append(None if x is None or y is None else x - y)
        res[k] = {'a': a, 'eff': float(np.mean([v for v in a if v is not None])), 'rel': None, 'unit': 'stat units/page'}
    return res


# ---------------------------------------------------------------- unsigned (entropy production)
def unsigned_stats(C, gf):
    def glyph_pairs(pages, k):
        c = Counter()
        for p in pages:
            s = L.stream(p)
            c.update(zip(s, s[k:]))
        return c
    top = {w for w, _ in gf.most_common(300)}

    def word_pairs(pages, k):
        c = Counter()
        for p in pages:
            t = [w if w in top or w[::-1] in top else '<o>' for w in L.tokens(p)]
            c.update(zip(t, t[k:]))
        return c

    def lenpairs(pages):
        c = Counter()
        for p in pages:
            for l in L.page_lines(p):
                x = [min(len(w), 10) for w in l]; c.update(zip(x, x[1:]))
        return c

    def logf(w):
        return math.log(gf.get(w, 0) + gf.get(w[::-1], 0) + 1)

    st = {
        'E_glyph_lag1': lambda P: L.kl_asym(glyph_pairs(P, 1)),
        'E_glyph_lag2': lambda P: L.kl_asym(glyph_pairs(P, 2)),
        'E_glyph_lag4': lambda P: L.kl_asym(glyph_pairs(P, 4)),
        'E_word_lag1': lambda P: L.kl_asym(word_pairs(P, 1)),
        'E_word_lag2': lambda P: L.kl_asym(word_pairs(P, 2)),
        'E_len_lag1': lambda P: L.kl_asym(lenpairs(P)),
        'E_ord_len': lambda P: L.ordinal_irrev(L.ordinal_counts([[len(w) for w in L.tokens(p)] for p in P])),
        'E_ord_freq': lambda P: L.ordinal_irrev(L.ordinal_counts([[logf(w) for w in L.tokens(p)] for p in P])),
        'E_hvg_freq': lambda P: L.hvg_irrev([[logf(w) for w in L.tokens(p)] for p in P]),
        'E_hvg_len': lambda P: L.hvg_irrev([[len(w) for w in L.tokens(p)] for p in P]),
    }
    R = [L.rev_page(p) for p in C]
    res = {}
    rng = random.Random(7)
    draws = [[rng.random() < 0.5 for _ in C] for _ in range(NULL_DRAWS)]
    for k, f in st.items():
        obs = f(C)
        null = []
        for d in draws[: (NULL_DRAWS if not k.startswith('E_hvg') else 100)]:
            null.append(f([R[i] if d[i] else C[i] for i in range(len(C))]))
        null = np.array(null)
        res[k] = {'obs': obs, 'null_mean': float(null.mean()), 'null_sd': float(null.std()),
                  'excess': obs - float(null.mean()), 'z': (obs - null.mean()) / (null.std() + 1e-12),
                  'p': float((1 + np.sum(null >= obs)) / (len(null) + 1))}
    return res


def run(name):
    fn = os.path.join(L.CK, f'c1_{name}.json')
    if os.path.exists(fn): return name
    t = time.time()
    C = build(name)
    gf = L.gfreq(C)
    gf2 = Counter(gf)
    for w, c in list(gf.items()): gf2[w[::-1]] += 0  # keep keys
    # frequency for reversed words: a reversed word gets the frequency of its forward spelling
    class GF(dict):
        def __missing__(self, w): return gf.get(w[::-1], 0)
    gfr = GF(gf)
    res = {'name': name, 'npages': len(C), 'ntok': L.ntok(C)}
    sig = {}
    sig.update(model_sigs(C))
    sig.update(compress_sigs(C))
    sig.update(page_sigs(C, gfr))
    for k, v in sig.items():
        z, p = L.signflip(v['a'])
        v['z'], v['p'] = float(z), float(p)
    res['signed'] = sig
    res['unsigned'] = unsigned_stats(C, gf)
    res['sec'] = time.time() - t
    json.dump(res, open(fn, 'w'))
    print(name, 'done', round(res['sec']), flush=True)
    return name


if __name__ == '__main__':
    names = sys.argv[1:] or (['ZL', 'IT', 'ZL19', 'LA', 'ITA', 'DE', 'CS', 'HE', 'HEvis', 'PL_REV', 'PL_WREV',
                              'MkG_1', 'MkW_1'] + [f'RvG_{i}' for i in range(6)] + [f'RvW_{i}' for i in range(6)]
                             + [f'RvGL_{i}' for i in range(3)])
    with Pool(2) as pool:
        for n in pool.imap_unordered(run, names): pass
