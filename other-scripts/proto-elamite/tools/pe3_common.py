"""pe3: is the Proto-Elamite entry middle phonetically spelled?  Shared code.

No sound values are used anywhere.  Corpora are lists of tuples of sign tokens.
Calibration corpora (phonetic vs logographic name lists) come from
data/derived/dark/loop56_corpora and loop63_corpora.
"""
import json, math, os, random, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, entries, header, base, is_sign  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DARK = os.path.join(ROOT, 'data', 'derived', 'dark')
PEDATA = os.path.join(HERE, '..', 'data')

# slot signs from test a (data/res_a_slots.json, z >= 3)
_r = json.load(open(os.path.join(PEDATA, 'res_a_slots.json')))['rows']
FINAL = {x['sign'] for x in _r if x['z_final'] >= 3}
INIT = {x['sign'] for x in _r if x['z_init'] >= 3}

CALIB = {
    # name: (path, script type)
    'LinB':      ('loop56_corpora/linb_personnel_dedup', 'phon'),   # pure syllabary
    'OB':        ('loop56_corpora/ob_names_dedup', 'phon'),         # Akkadian names, syllabic cuneiform
    'UrIII':     ('loop56_corpora/ur3_names_dedup', 'mixed'),       # Sumerian names, morphographic + syllabic
    'CN_anc':    ('loop63_corpora/cn_ancient', 'logo'),
    'CN_full':   ('loop63_corpora/cn_full', 'logo'),
    'JP_given':  ('loop63_corpora/jp_given', 'logo'),
    'JP_person': ('loop63_corpora/jp_person', 'logo'),
}


def _clean_tok(t):
    t = t.strip()
    if not t or '$' in t or '(' in t or ')' in t or t in ('x', '...', '[...]'):
        return None
    return t.lower() if t.isascii() else t


def load_calib(name):
    path = os.path.join(DARK, CALIB[name][0] + '.jsonl')
    out = set()
    for line in open(path, encoding='utf-8'):
        seq = json.loads(line)['seq']
        toks = [_clean_tok(t) for t in seq]
        if any(t is None for t in toks):
            continue
        if toks:
            out.add(tuple(toks))
    return sorted(out)


def pe_entries(clean=True, keep_variants=False):
    T = load()
    return T, entries(T, require_clean=clean, base_signs=not keep_variants)


def strip_frame(s, mode='A'):
    """mode A: drop one final class sign and one initial prefix sign (test-a lists).
       mode B: keep the whole entry string.
       mode C: drop ALL leading INIT and trailing FINAL signs."""
    s = list(s)
    if mode == 'B':
        return tuple(s)
    if mode == 'A':
        if s and s[-1] in FINAL:
            s = s[:-1]
        if s and s[0] in INIT:
            s = s[1:]
        return tuple(s)
    while s and s[-1] in FINAL:
        s = s[:-1]
    while s and s[0] in INIT:
        s = s[1:]
    return tuple(s)


def pe_middles(mode='A', distinct=True, minlen=2, tablets=None, keep_variants=False):
    T, E = pe_entries(keep_variants=keep_variants)
    M = []
    for e in E:
        if tablets is not None and e['tablet'] not in tablets:
            continue
        if len(e['signs']) < 2 and mode != 'B':
            continue
        m = strip_frame(e['signs'], mode)
        if len(m) >= minlen:
            M.append(m)
    return sorted(set(M)) if distinct else M


def pe_headers(minlen=2, distinct=True):
    T = load()
    H = [tuple(h) for h in (header(t) for t in T) if h and 'x' not in h]
    H = [h for h in H if len(h) >= minlen]
    return sorted(set(H)) if distinct else H


# ---------------------------------------------------------------- sampling
BINS = [2, 3, 4, 5]          # 5 = 5+


def lbin(n):
    return min(n, 5)


def length_profile(corpus):
    c = Counter(lbin(len(s)) for s in corpus if len(s) >= 2)
    tot = sum(c.values())
    return {b: c[b] / tot for b in BINS}


def matched_sample(corpus, profile, n, rng):
    """Sample n distinct strings whose length-bin shares follow profile.
    Bins short of strings are filled from the nearest available bin."""
    by = defaultdict(list)
    for s in corpus:
        if len(s) >= 2:
            by[lbin(len(s))].append(s)
    want = {b: int(round(profile[b] * n)) for b in BINS}
    out, short = [], 0
    for b in BINS:
        pool = by[b]
        k = min(want[b], len(pool))
        out += rng.sample(pool, k)
        short += want[b] - k
    if short:
        rest = [s for b in BINS for s in by[b] if s not in set(out)]
        out += rng.sample(rest, min(short, len(rest)))
    return out, short


# ---------------------------------------------------------------- nulls
def shuffle_within(corpus, rng):
    out = []
    for s in corpus:
        s = list(s)
        rng.shuffle(s)
        out.append(tuple(s))
    return out


def shuffle_global(corpus, rng):
    toks = [t for s in corpus for t in s]
    rng.shuffle(toks)
    out, i = [], 0
    for s in corpus:
        out.append(tuple(toks[i:i + len(s)]))
        i += len(s)
    return out


def shuffle_column(corpus, rng):
    """Position-preserving null: tokens are permuted among strings of the same
    length at the same position.  Keeps every positional distribution, destroys
    dependence between neighbours."""
    by = defaultdict(list)
    for i, s in enumerate(corpus):
        by[len(s)].append(i)
    out = [None] * len(corpus)
    for L, idx in by.items():
        cols = []
        for p in range(L):
            col = [corpus[i][p] for i in idx]
            rng.shuffle(col)
            cols.append(col)
        for j, i in enumerate(idx):
            out[i] = tuple(cols[p][j] for p in range(L))
    return out


def markov2(corpus, rng, lam=(0.6, 0.3, 0.1)):
    """Interpolated order-2 Markov chain with start symbols, fixed lengths copied
    from the corpus (end symbol not modelled, so final-slot effects are not
    reproduced unless the chain itself creates them)."""
    uni = Counter(t for s in corpus for t in s)
    bi = defaultdict(Counter)
    tri = defaultdict(Counter)
    for s in corpus:
        p = ('<s>', '<s>') + tuple(s)
        for i in range(2, len(p)):
            bi[p[i - 1]][p[i]] += 1
            tri[(p[i - 2], p[i - 1])][p[i]] += 1
    vocab = list(uni)
    uw = [uni[v] for v in vocab]
    out = []
    for s in corpus:
        a, b = '<s>', '<s>'
        seq = []
        for _ in range(len(s)):
            r = rng.random()
            if r < lam[0] and tri[(a, b)]:
                c = tri[(a, b)]
                x = rng.choices(list(c), weights=list(c.values()))[0]
            elif r < lam[0] + lam[1] and bi[b]:
                c = bi[b]
                x = rng.choices(list(c), weights=list(c.values()))[0]
            else:
                x = rng.choices(vocab, weights=uw)[0]
            seq.append(x)
            a, b = b, x
        out.append(tuple(seq))
    return out


# ---------------------------------------------------------------- statistics
def inventory(corpus, ntok=800, rng=None):
    toks = [t for s in corpus for t in s]
    rng = rng or random.Random(0)
    sub = rng.sample(toks, min(ntok, len(toks)))
    c = Counter(toks)
    top20 = sum(v for _, v in c.most_common(20)) / len(toks)
    hapax = sum(1 for v in c.values() if v == 1) / len(c)
    return {'types@%d' % ntok: len(set(sub)), 'top20cov': top20, 'hapax_type_share': hapax}


def final_closure(corpus, k=10):
    fin = Counter(s[-1] for s in corpus)
    ini = Counter(s[0] for s in corpus)
    n = len(corpus)
    return {'fin_topk': sum(v for _, v in fin.most_common(k)) / n,
            'ini_topk': sum(v for _, v in ini.most_common(k)) / n}


def order_consistency(corpus, minc=3):
    adj = Counter()
    for s in corpus:
        for a, b in zip(s, s[1:]):
            if a != b:
                adj[(a, b)] += 1
    seen, vals, w = set(), [], []
    for (a, b), v in adj.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        tot = v + adj.get((b, a), 0)
        if tot >= minc:
            vals.append(max(v, adj.get((b, a), 0)) / tot)
    return {'order_cons': sum(vals) / len(vals) if vals else float('nan'), 'order_npairs': len(vals)}


def phonotactic_gaps(corpus, topn=25, minexp=1.0):
    """Among the topn signs, ordered pairs whose expected adjacent count
    (independence, position-free unigram shares) >= minexp: observed share of
    never-adjacent pairs minus the Poisson-expected share exp(-e).  Positive =
    more 'forbidden' pairs than chance.  Also adjacent-pair mutual information."""
    toks = Counter(t for s in corpus for t in s)
    N = sum(toks.values())
    nadj = sum(len(s) - 1 for s in corpus)
    top = [t for t, _ in toks.most_common(topn)]
    adj = Counter((a, b) for s in corpus for a, b in zip(s, s[1:]))
    cand = 0
    exc = 0.0
    for a in top:
        for b in top:
            e = nadj * toks[a] / N * toks[b] / N
            if e >= minexp:
                cand += 1
                exc += (adj[(a, b)] == 0) - math.exp(-e)
    L = Counter(a for a, b in adj.elements())
    Rr = Counter(b for a, b in adj.elements())
    tot = sum(adj.values())
    mi = sum(v / tot * math.log2(v * tot / (L[a] * Rr[b])) for (a, b), v in adj.items())
    return {'gap_excess': exc / cand if cand else float('nan'), 'gap_ncand': cand, 'adj_mi': mi}


def repeat_rate(corpus):
    n = sum(len(s) - 1 for s in corpus)
    r = sum(1 for s in corpus for a, b in zip(s, s[1:]) if a == b)
    return {'aa_rate': r / n if n else float('nan')}


def positional_freedom(corpus, minn=10):
    """For strings of length >= 3: per sign, normalised entropy over
    (initial, medial, final); mean over signs with n >= minn (token-weighted)."""
    pos = defaultdict(Counter)
    for s in corpus:
        if len(s) < 3:
            continue
        for i, t in enumerate(s):
            pos[t]['I' if i == 0 else 'F' if i == len(s) - 1 else 'M'] += 1
    num = den = 0
    for t, c in pos.items():
        n = sum(c.values())
        if n < minn:
            continue
        h = -sum(v / n * math.log(v / n, 3) for v in c.values() if v)
        num += h * n
        den += n
    return {'pos_free': num / den if den else float('nan')}


# ---------------------------------------------------------------- MDL segmentation
def _dl(units_corpus, lex, V0):
    cnt = Counter(u for s in units_corpus for u in s)
    N = sum(cnt.values())
    corpus_bits = -sum(v * math.log2(v / N) for v in cnt.values())
    lex_bits = sum((len(u) + 1) * math.log2(V0 + 1) for u in lex)
    return corpus_bits + lex_bits


def _merge(units_corpus, a, b):
    ab = a + b
    out = []
    for s in units_corpus:
        r, i = [], 0
        while i < len(s):
            if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                r.append(ab)
                i += 2
            else:
                r.append(s[i])
                i += 1
        out.append(tuple(r))
    return out


def mdl_segment(corpus, max_try=15, max_units=400):
    """Greedy MDL chunking: merge the most frequent adjacent unit pair while the
    two-part description length (lexicon of multi-sign units + unigram code of
    the segmented corpus) falls.  Returns gain in bits per sign, the accepted
    multi-sign units and the share of sign tokens covered by them."""
    uc = [tuple((t,) for t in s) for s in corpus]
    V0 = len({t for s in corpus for t in s})
    nsign = sum(len(s) for s in corpus)
    lex = set()
    dl0 = cur = _dl(uc, lex, V0)
    while len(lex) < max_units:
        pairs = Counter((a, b) for s in uc for a, b in zip(s, s[1:]))
        acc = False
        for (a, b), c in pairs.most_common(max_try):
            if c < 2:
                break
            nuc = _merge(uc, a, b)
            nlex = {u for s in nuc for u in s if len(u) > 1}
            d = _dl(nuc, nlex, V0)
            if d < cur:
                uc, lex, cur, acc = nuc, nlex, d, True
                break
        if not acc:
            break
    covered = sum(len(u) for s in uc for u in s if len(u) > 1)
    return {'mdl_gain': (dl0 - cur) / nsign, 'mdl_units': len(lex),
            'mdl_cov': covered / nsign}, lex, uc


def zscore(obs, null):
    m = sum(null) / len(null)
    sd = (sum((x - m) ** 2 for x in null) / max(1, len(null) - 1)) ** 0.5
    return (obs - m) / sd if sd > 0 else float('nan'), m


def fmt(x):
    return '%.3f' % x if isinstance(x, float) else str(x)
