"""Shared helpers for the v2 label tests (labels vs drawn objects)."""
import json, os, random, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vlib import glyphs

DER = os.path.join(ROOT, 'data', 'derived')


def load():
    labs = json.load(open(os.path.join(DER, 'v2_labels.json')))
    text = json.load(open(os.path.join(DER, 'v2_text_words.json')))
    for l in labs:
        l['g'] = glyphs(''.join(l['words']).replace('?', ''))
    return labs, text


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sims(a, b):
    """Three label-similarity scores in [0,1]: 1-normalised edit distance,
    shared first-2-glyph prefix, shared last-2-glyph suffix."""
    if not a or not b:
        return (0.0, 0.0, 0.0)
    ed = 1 - lev(a, b) / max(len(a), len(b))
    return (ed, float(a[:2] == b[:2]), float(a[-2:] == b[-2:]))


SIMNAMES = ('edit', 'prefix2', 'suffix2')


def simmat(labs):
    n = len(labs)
    S = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            S[i][j] = S[j][i] = sims(labs[i]['g'], labs[j]['g'])
    return S


def perm_within(groups_key, labs, attr, rng):
    """Return a copy of attr values permuted among labels sharing groups_key."""
    by = defaultdict(list)
    for i, l in enumerate(labs):
        by[l[groups_key]].append(i)
    out = [l[attr] for l in labs]
    for idx in by.values():
        vals = [out[i] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            out[i] = v
    return out


def pval(obs, null, greater=True):
    k = sum((x >= obs) if greater else (x <= obs) for x in null)
    return (k + 1) / (len(null) + 1)


def zscore(obs, null):
    m = sum(null) / len(null)
    sd = (sum((x - m) ** 2 for x in null) / len(null)) ** 0.5 or 1e-9
    return (obs - m) / sd
