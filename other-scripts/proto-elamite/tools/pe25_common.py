"""pe25 shared code: TWO HANDS ON ONE TABLET.

Idea: a tablet may be written by two people (a list-writer and a checker who
adds the total, the edge tag or an endorsement). Writer fingerprint = free
choices that the transliteration keeps and that do not obviously change
meaning: which graphic variant of a base sign is used (M387 vs M387~c ...),
and whether a sign+numeral line carries the separator comma.

Statistic (pooled over tablets): for two line sets X, Y of a tablet and every
base sign b with real variant choice in the corpus that occurs in both, the
share of (x, y) token pairs that use the same variant; averaged per
(tablet, base) cell.  'ctx' mode drops pairs whose two entries have the same
base-sign string (same item written twice, where agreement is meaning).

Hand index H = (agreement - stratum null) / (1 - stratum null), where the null
pairs set X of tablet i with set Y of another tablet j of the same stratum
(header base sign x publication series).  Same hand: H(obv,rev) ~ H(obv half,
obv half); a second hand on the reverse: H(obv,rev) -> 0.
pi = 1 - H(obv,rev)/H(obv,obv) = estimated share of reverses in another hand.
"""
import collections, json, os, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe25_ckpt')
os.makedirs(CK, exist_ok=True)


def base(s):
    return s.split('~')[0]


def load():
    d = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    return d


def series(t):
    p = t['designation'].split(',')[0].split()
    return ' '.join(p[:2])


def header_class(t):
    for l in t['lines']:
        if l['surface'] == 'obverse' and l['signs']:
            return base(l['signs'][0]) if l.get('header_comment') or not l['numerals'] else 'NOHDR'
    return 'NONE'


def variant_bases(d, min_tok=20, min_minor=0.05):
    c = collections.defaultdict(collections.Counter)
    for t in d:
        for l in t['lines']:
            for s in l['signs']:
                if s.startswith('M') and '|' not in s:
                    c[base(s)][s] += 1
    out = {}
    for b, cc in c.items():
        n = sum(cc.values())
        if n >= min_tok and len(cc) > 1 and 1 - cc.most_common(1)[0][1] / n >= min_minor:
            out[b] = {v: k / n for v, k in cc.items()}
    return out


def tokens(lines, vb):
    """list of (base, variant, entry_key) for variant-bearing tokens."""
    out = []
    for l in lines:
        key = ' '.join(base(s) for s in l['signs'])
        for s in l['signs']:
            b = base(s)
            if b in vb:
                out.append((b, s, key))
    return out


def comma_habit(lines):
    """(n sign+numeral lines, n of them without comma)."""
    n = k = 0
    for l in lines:
        if l['signs'] and l['numerals']:
            n += 1
            k += (not l['has_comma'])
    return n, k


def faces(t):
    obv = [l for l in t['lines'] if l['surface'] == 'obverse']
    rev = [l for l in t['lines'] if l['surface'] == 'reverse']
    edge = [l for l in t['lines'] if l['surface'] in ('top', 'left', 'bottom', 'right')]
    return obv, rev, edge


def cells(X, Y, ctx=False):
    """per-base agreement cells between token lists X and Y -> list of (agree, pairs)."""
    bx = collections.defaultdict(list)
    for b, v, k in X:
        bx[b].append((v, k))
    res = []
    by = collections.defaultdict(list)
    for b, v, k in Y:
        by[b].append((v, k))
    for b in bx:
        if b not in by:
            continue
        a = n = 0
        for v1, k1 in bx[b]:
            for v2, k2 in by[b]:
                if ctx and k1 == k2:
                    continue
                n += 1
                a += (v1 == v2)
        if n:
            res.append(a / n)
    return res


def pooled(cell_lists):
    allc = [c for cl in cell_lists for c in cl]
    return (float(np.mean(allc)) if allc else float('nan')), len(allc)


def hand_index(pairs_real, pairs_null, ctx=False):
    """pairs_*: list of (X, Y) token lists. returns agreement, null agreement, H, ncells."""
    a, n = pooled([cells(X, Y, ctx) for X, Y in pairs_real])
    a0, n0 = pooled([cells(X, Y, ctx) for X, Y in pairs_null])
    H = (a - a0) / (1 - a0) if a0 < 1 else float('nan')
    return a, a0, H, n


def strata(d, key):
    g = collections.defaultdict(list)
    for i, t in enumerate(d):
        g[key(t)].append(i)
    return g


def null_partner(i, groups, keyof, rng, need=None):
    """random j != i in the same stratum (optionally satisfying need(j))."""
    g = groups[keyof[i]]
    cand = [j for j in g if j != i and (need is None or need(j))]
    if not cand:
        return None
    return rng.choice(cand)


def split_halves(lines):
    """obverse body split into first and second half by line order."""
    body = [l for l in lines]
    h = len(body) // 2
    return body[:h], body[h:]


def interleave(lines):
    return lines[0::2], lines[1::2]


def dump(name, obj):
    with open(os.path.join(CK, name), 'w') as f:
        json.dump(obj, f, indent=1, default=str)


def row(path, rid, method, result, verdict):
    with open(path, 'a') as f:
        f.write('| %s | %s | %s | %s |\n' % (rid, method, result, verdict))
