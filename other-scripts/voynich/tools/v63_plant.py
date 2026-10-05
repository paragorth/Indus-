"""v63: corpora for the slip tests.  Planted 'uncorrected copy': clean text with slip-then-rewrite events
inserted (never struck), with the type mix and anticipation offsets measured on the 1,775 real struck words
of the Plaoul witnesses.  Also a copy-and-modify generator and nulls."""
import json, os, sys, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L

_CAT = None


def slipcat():
    global _CAT
    if _CAT is None:
        d = json.load(open(os.path.join(L.CK, 'slipcat.json')))
        _CAT = d
    return _CAT


def make_slip(words, i, rng, vocab, vfreq, by_end, cat_types, cat_p, offs, offs_p):
    """a slip A written before words[i] (B)."""
    B = words[i]
    c = rng.choices(cat_types, cat_p)[0]
    if c == 'ditto':
        return B, c
    if c == 'falsestart' and len(B) >= 2:
        return B[:rng.randint(1, len(B) - 1)], c
    if c == 'nearmiss':
        w = list(B); al = sorted(set(g for v in vocab[:200] for g in v))
        for _ in range(rng.randint(1, 2)):
            op = rng.random()
            if op < 0.5 and w:
                w[rng.randrange(len(w))] = rng.choice(al)
            elif op < 0.75 and len(w) > 1:
                del w[rng.randrange(len(w))]
            else:
                w.insert(rng.randrange(len(w) + 1), rng.choice(al))
        return tuple(w), c
    if c == 'anticip':
        k = rng.choices(offs, offs_p)[0]
        if i + k < len(words):
            return words[i + k], c
    if c == 'persev':
        k = rng.randint(1, 12)
        if i - k >= 0:
            return words[i - k], c
    if c == 'sameend' and len(B) >= 2 and by_end.get(B[-2:]):
        return rng.choice(by_end[B[-2:]]), c
    return rng.choices(vocab, vfreq)[0], 'other'


def plant_slips(streams, rate, seed=0):
    """returns new streams + per-token labels (1 = inserted slip)."""
    rng = random.Random(seed)
    d = slipcat()
    cat_types = list(d['cat']); cat_p = [d['cat'][c] for c in cat_types]
    offs = [int(k) for k in d['offs']]; offs_p = [d['offs'][k] for k in d['offs']]
    wc = Counter(w for s in streams for w in s['words'])
    vocab = [w for w, _ in wc.most_common()]; vfreq = [wc[w] for w in vocab]
    by_end = {}
    for w in vocab[:3000]:
        if len(w) >= 2: by_end.setdefault(w[-2:], []).append(w)
    out = []
    for s in streams:
        ws, lab = [], []
        for i, w in enumerate(s['words']):
            if i > 0 and rng.random() < rate:
                a, c = make_slip(s['words'], i, rng, vocab, vfreq, by_end, cat_types, cat_p, offs, offs_p)
                if a:
                    ws.append(a); lab.append(1)
            ws.append(w); lab.append(0)
        out.append(dict(s, words=ws, lab=lab))
    return out


def copygen(streams, rate=0.5, edits=1.0, win=12, seed=0):
    """shape-neutral copy-and-modify (self-citation) generator, paragraph lengths kept (as v29)."""
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    words = [w for s in streams for w in s['words']]
    gc = Counter(g for w in words for g in w)
    gk = list(gc); gp = np.cumsum([gc[k] for k in gk]); gp = gp / gp[-1]
    out, hist = [], []
    for s in streams:
        nl = []
        for _ in s['words']:
            if hist and rng.random() < rate:
                w = list(rng.choice(hist[-win:]))
                for _ in range(nrng.poisson(edits)):
                    i = rng.randrange(len(w)); w[i] = gk[int(np.searchsorted(gp, rng.random()))]
                w = tuple(w)
            else:
                w = rng.choice(words)
            nl.append(w); hist.append(w)
        out.append(dict(s, words=nl))
    return out


def shuffle_streams(streams, seed=0):
    rng = random.Random(seed)
    out = []
    for s in streams:
        w = list(s['words']); rng.shuffle(w); out.append(dict(s, words=w))
    return out


def markov1(streams, seed=0):
    """word-bigram resynthesis within paragraphs (first word from paragraph-initial unigram)."""
    rng = random.Random(seed)
    succ, first = {}, []
    for s in streams:
        ws = s['words']; first.append(ws[0])
        for a, b in zip(ws, ws[1:]):
            succ.setdefault(a, []).append(b)
    out = []
    for s in streams:
        w = [rng.choice(first)]
        while len(w) < len(s['words']):
            nx = succ.get(w[-1])
            w.append(rng.choice(nx) if nx else rng.choice(first))
        out.append(dict(s, words=w))
    return out


def latin_clean(nwords=36000, wit='svict', seed=7, opaque=True):
    S = [s for s in L.plaoul_streams('after') if s['folio'].startswith(wit + '_')]
    out, n = [], 0
    for s in S:
        out.append(s); n += len(s['words'])
        if n >= nwords: break
    return L.encode_streams(out, seed) if opaque else out


def latin_as_written(wit=None, opaque=True, seed=7):
    """the real first writing of all witnesses (struck words kept), + labels of struck words."""
    out = []
    for p in L.plaoul_witnesses():
        if wit and p['wit'] != wit: continue
        ws, lab = [], []
        for t in p['toks']:
            if t['st'] == 'a' or not t['before']: continue
            ws.append(tuple(t['before'])); lab.append(1 if t['st'] == 'd' else 0)
        if len(ws) >= 3:
            out.append(dict(folio=p['file'], words=ws, lab=lab))
    if opaque:
        enc = L.encode_streams(out, seed)
        for e, o in zip(enc, out):  # encode_streams may drop empty words; keep labels aligned
            if len(e['words']) != len(o['words']):
                tab = L.verbose_table(seed)
                keep = [i for i, w in enumerate(o['words']) if any(c in tab for c in w)]
                e['lab'] = [o['lab'][i] for i in keep]
        return enc
    return out


def column_shuffle(streams, seed=0, by_lang=None):
    """position-preserving null: words permuted among tokens with the same position in the line
    (0..7, 8+, and 'last') and line-length bin; keeps every positional trend, destroys adjacency."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for a, s in enumerate(streams):
        for i, (k, j, n) in enumerate(s['line']):
            pos = 'L' if j == n - 1 else min(j, 8)
            key = (pos, min(n, 12) // 3, (by_lang[a] if by_lang else 0))
            groups[key].append((a, i))
    out = [dict(s, words=list(s['words'])) for s in streams]
    for key, locs in groups.items():
        ws = [streams[a]['words'][i] for a, i in locs]
        rng.shuffle(ws)
        for (a, i), w in zip(locs, ws):
            out[a]['words'][i] = w
    return out
