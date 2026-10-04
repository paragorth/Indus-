"""v12 shared helpers: 'each line is two (or more) texts woven together'.

Every corpus is poured into the ZL3b paragraph-line layout (v11_lib.template: same folios, same line lengths),
so the folio-parity 2-fold split is identical for all corpora. A word is a string, one character per glyph
(Voynich: vlib.glyphs, benched gallows / ch / sh merged; verbose texts: symbols A-R of v5_corpora.verbose_table).

Corpora
  Voynich-ZL / Voynich-IT        P lines, uncertain words dropped
  vLatin / vItalian / vDante     single texts, verbose-encoded (NEGATIVE controls: must not split)
  pLatin                         plain Latin letters (negative)
  Shuf-line                      Voynich ZL, word order shuffled inside each line (negative / null)
  Weave-LF-alt                   verbose Latin and verbose filler, strict alternation (POSITIVE)
  Weave-LF-hmm                   same, stream chosen by a 2-state Markov switch, P(switch) = 0.6 (POSITIVE)
  Weave-LI-alt                   verbose Latin + verbose Italian, strict alternation (POSITIVE)
  Weave-VV-alt                   Voynich Currier-A word stream + Currier-B word stream (each in its own order),
                                 alternating: the hardest positive, both streams are real Voynich (POSITIVE)
  Gloss-pair                     verbose Latin word followed by a 'gloss' word that copies its last two
                                 letters + a fixed suffix pool: adjacent PAIRS linked, pairs not (phase control)
Streams in all weaves run on across line breaks (a woven continuous text).
"""
import os, sys, re, random, math, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
import v11_lib as L11
import v5_corpora as V5

CK = os.path.join(vlib.DATA, 'results', 'v12'); os.makedirs(CK, exist_ok=True)
OUT = os.path.join(vlib.ROOT, 'loops')


def _g(w):
    return ''.join(vlib.glyphs(w))


_TPL = None
def template():
    global _TPL
    if _TPL is None:
        _TPL = [dict(l, words=[_g(w) for w in l['words']]) for l in L11.template()]
    return _TPL


def ntok():
    return sum(len(l['words']) for l in template())


_TAB = None
def enc(w):
    global _TAB
    if _TAB is None:
        _TAB = {k: ''.join(v) for k, v in V5.verbose_table(7).items()}
    return ''.join(_TAB.get(c, '') for c in w)


def latin_plain():
    return L11.latin_words()


def italian_plain():
    return L11.ref_words('Italian-Manzoni')


def dante_plain():
    return L11.ref_words('Italian-Dante')


def filler_stream(n, seed=3, nlex=40):
    """A regular filler text: 40 random pseudo-words (2-5 letters), first-order Markov chain in which
    each word has 3 preferred successors (p 0.8 together); Zipf-ish start weights."""
    rng = random.Random(seed)
    lex = set()
    while len(lex) < nlex:
        lex.add(''.join(rng.choice('abcdefghilmnoprstuv') for _ in range(rng.randint(2, 5))))
    lex = sorted(lex); rng.shuffle(lex)
    succ = {w: rng.sample(lex, 3) for w in lex}
    zw = [1 / (i + 1) for i in range(nlex)]
    out = [rng.choices(lex, zw)[0]]
    while len(out) < n:
        out.append(rng.choice(succ[out[-1]]) if rng.random() < 0.8 else rng.choices(lex, zw)[0])
    return out


def weave(a, b, n, mode='alt', p_switch=0.6, seed=5):
    rng = random.Random(seed); ia = ib = 0; s = 0; out = []
    for k in range(n):
        if mode == 'alt':
            s = k % 2
        elif k > 0 and rng.random() < p_switch:
            s = 1 - s
        if s == 0:
            out.append(a[ia]); ia += 1
        else:
            out.append(b[ib]); ib += 1
    return out


def pour(words):
    out, k = [], 0
    for l in template():
        q = dict(l); q['words'] = words[k:k + len(l['words'])]; k += len(l['words']); out.append(q)
    assert k <= len(words) or True
    return out


def gloss_pairs(n, seed=4):
    rng = random.Random(seed); lat = latin_plain(); off = len(lat) // 7
    suf = ['um', 'is', 'en', 'or', 'a']
    out = []; i = off
    while len(out) < n:
        w = lat[i]; i += 1
        out.append(enc(w))
        out.append(enc(w[-2:] + rng.choice(suf)))
    return out[:n]


def corpus(name, seed=1):
    N = ntok()
    if name == 'Voynich-ZL':
        return template()
    if name == 'Voynich-IT':
        return [dict(l, words=[_g(w) for w in l['words']]) for l in L11.voy('IT2a')]
    if name == 'Shuf-line':
        rng = random.Random(seed); out = []
        for l in template():
            ws = list(l['words']); rng.shuffle(ws); out.append(dict(l, words=ws))
        return out
    if name == 'vLatin':
        w = latin_plain(); w = w[len(w) // 10:]; return pour([enc(x) for x in w])
    if name == 'pLatin':
        w = latin_plain(); w = w[len(w) // 10:]; return pour(w)
    if name == 'vItalian':
        return pour([enc(x) for x in italian_plain()])
    if name == 'vDante':
        return pour([enc(x) for x in dante_plain()])
    if name.startswith('Weave-LF'):
        w = latin_plain(); a = [enc(x) for x in w[len(w) // 10:]]
        b = [enc(x) for x in filler_stream(N, seed=3)]
        mode = 'alt' if name.endswith('alt') else 'hmm'
        return pour(weave(a, b, N, mode, seed=seed))
    if name == 'Weave-LI-alt':
        w = latin_plain(); a = [enc(x) for x in w[len(w) // 10:]]
        b = [enc(x) for x in italian_plain()]
        return pour(weave(a, b, N, 'alt'))
    if name == 'Weave-VV-alt':
        tpl = template()
        a = [x for l in tpl if l['lang'] == 'A' for x in l['words']]
        b = [x for l in tpl if l['lang'] != 'A' for x in l['words']]
        m = min(len(a), len(b))
        ww = weave(a[:m], b[:m], 2 * m, 'alt')
        ww = (ww * 2)[:N]
        return pour(ww)
    if name == 'Gloss-pair':
        return pour(gloss_pairs(N))
    raise KeyError(name)


POS = ['Weave-LF-alt', 'Weave-LF-hmm', 'Weave-LI-alt', 'Weave-VV-alt']
NEG = ['vLatin', 'vItalian', 'vDante', 'pLatin', 'Shuf-line']
VOY = ['Voynich-ZL', 'Voynich-IT']


# ---------------- word features and the backoff context model ----------------
def feats(w):
    """(first, second, third, last, length bucket)"""
    return (w[0], w[1] if len(w) > 1 else '#', w[2] if len(w) > 2 else '#', w[-1], str(min(len(w), 8)))


START = ('^', '^', '^', '^', '^')
# feature i of the word is predicted from (cond_in[i] = a feature of the SAME word already predicted, ctx feature ctx_ix[i])
COND_IN = [None, 0, 1, 2, 3]
CTX_IX = [3, 1, 2, 3, 4]   # first <- context last glyph (junction); second <- ctx second; ... length <- ctx length


class CtxModel:
    """P(word | ctx word) = prod_i P(x_i | x_{cond_in[i]}, ctx_{ctx_ix[i]}), Witten-Bell-like backoff
    P(x|a,c) -> P(x|a) -> P(x). Counts can be fractional (EM). One instance per stream label if wanted."""
    def __init__(self, beta=2.0):
        self.beta = beta
        self.c3 = [defaultdict(Counter) for _ in range(5)]
        self.c2 = [defaultdict(Counter) for _ in range(5)]
        self.c1 = [Counter() for _ in range(5)]
        self.V = [set() for _ in range(5)]

    def add(self, x, c, wt=1.0):
        for i in range(5):
            a = x[COND_IN[i]] if COND_IN[i] is not None else ''
            self.c3[i][(a, c[CTX_IX[i]])][x[i]] += wt
            self.c2[i][a][x[i]] += wt
            self.c1[i][x[i]] += wt
            self.V[i].add(x[i])

    def finalize(self):
        self.n3 = [{k: sum(v.values()) for k, v in d.items()} for d in self.c3]
        self.n2 = [{k: sum(v.values()) for k, v in d.items()} for d in self.c2]
        self.n1 = [sum(c.values()) for c in self.c1]
        self.cache = {}
        return self

    def lp(self, x, c):
        key = (x, c)
        r = self.cache.get(key)
        if r is not None:
            return r
        b = self.beta; tot = 0.0
        for i in range(5):
            a = x[COND_IN[i]] if COND_IN[i] is not None else ''
            V = len(self.V[i]) + 1
            p1 = (self.c1[i].get(x[i], 0) + 0.5) / (self.n1[i] + 0.5 * V)
            n2 = self.n2[i].get(a, 0)
            p2 = (self.c2[i][a].get(x[i], 0) + b * p1) / (n2 + b) if n2 else p1
            k3 = (a, c[CTX_IX[i]]); n3 = self.n3[i].get(k3, 0)
            p3 = (self.c3[i][k3].get(x[i], 0) + b * p2) / (n3 + b) if n3 else p2
            tot += math.log2(p3)
        self.cache[key] = tot
        return tot


def split(lines, fold):
    tr = [l for l in lines if l['fi'] % 2 != fold]; te = [l for l in lines if l['fi'] % 2 == fold]
    return tr, te


def ctx_pairs(lines, assign=None, carry=False, lag=None):
    """Yield (word feats, ctx feats, stream label). assign(words) -> labels; context = last earlier word with the
    same label in the line (START if none). carry=True: per-stream context carries over line breaks inside a
    paragraph. lag=k: context = the word k positions back (fixed lag)."""
    out = []
    last = {}
    for l in lines:
        ws = l['words']
        if not ws:
            continue
        if not carry or l.get('para_start'):
            last = {}
        F = [feats(w) for w in ws]
        if lag is not None:
            for t in range(len(ws)):
                out.append((F[t], F[t - lag] if t - lag >= 0 else START, 0))
            continue
        lab = assign(ws) if assign else [0] * len(ws)
        for t in range(len(ws)):
            s = lab[t]
            out.append((F[t], last.get(s, START), s))
            last[s] = F[t]
    return out


def link_bits(train_pairs, test_pairs, per_stream=True):
    """Held-out bits/token gained by the context: CE(no ctx) - CE(ctx), averaged over test tokens that HAVE a
    non-START context (also returns the all-token version)."""
    def build(pairs, use_ctx):
        ms = {}
        for x, c, s in pairs:
            k = s if per_stream else 0
            if k not in ms: ms[k] = CtxModel()
            ms[k].add(x, c if use_ctx else START)
        for m in ms.values(): m.finalize()
        return ms
    mc = build(train_pairs, True); m0 = build(train_pairs, False)
    g, n, gall = 0.0, 0, 0.0
    for x, c, s in test_pairs:
        k = s if per_stream else 0
        if k not in mc: continue
        d = mc[k].lp(x, c) - m0[k].lp(x, START)
        gall += d
        if c != START:
            g += d; n += 1
    return g / max(n, 1), gall / max(len(test_pairs), 1), n


def cv_link(lines, **kw):
    """2-fold (folio parity) link bits."""
    r = []
    for fold in (0, 1):
        tr, te = split(lines, fold)
        r.append(link_bits(ctx_pairs(tr, **kw), ctx_pairs(te, **kw)))
    return sum(x[0] for x in r) / 2, sum(x[1] for x in r) / 2, sum(x[2] for x in r)


def hamming1(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1
