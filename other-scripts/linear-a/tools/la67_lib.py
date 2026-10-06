#!/usr/bin/env python3
"""LA-67 THE KILL SWEEP: shared helpers.  Every grade-C guess in FINDINGS.md is re-tested with its own
stated would-kill line (or the strongest existing-data version: fresh seeds, frequency-matched decoys,
held-out tablet halves, permutation nulls).  The SAME criterion is applied to the decoys, so that
'survive' has a false-survival rate attached.  No Linear B values are used anywhere."""
import os, sys, json, math, random, hashlib, re
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from la60_common import load_la, split as la60_split, tab_of
CK = os.path.join(HERE, '..', 'data', 'la67_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')


def seed(name):
    return int(hashlib.sha256(('la67-' + name).encode()).hexdigest()[:8], 16)


def docs_all():
    return load_la()


def admin(D):
    return [d for d in D if not d['lib']]


def lines(d):
    out, cur = [], []
    for t in d['toks']:
        if t[0] == 'NL':
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def occ(D, x):
    """(doc index, token index) of every token whose value is x."""
    return [(i, j) for i, d in enumerate(D) for j, t in enumerate(d['toks']) if t[0] in 'WL' and t[1] == x]


def counts(D, kind=None):
    return Counter(t[1] for d in D for t in d['toks'] if t[0] in ('WL' if kind is None else kind))


def is_single(w):
    return '-' not in w


def matched_decoys(cnt, target, pool, n=20, exclude=()):
    """n frequency-nearest items from pool (log distance), deterministic tie-break."""
    f = max(cnt[target], 1)
    cand = [w for w in pool if w != target and w not in exclude and cnt[w] > 0]
    cand.sort(key=lambda w: (abs(math.log(cnt[w] / f)), hashlib.md5(w.encode()).hexdigest()))
    return cand[:n]


def halves(D, k):
    """Disjoint tablet halves (both sides of a tablet together), site-stratified, fresh seed."""
    return la60_split(D, 'la67-half-%d' % k)


def pct_rank(x, ref):
    """share of reference values >= x (one-sided P with +1 correction)."""
    ref = np.asarray([r for r in ref if r is not None and not (isinstance(r, float) and math.isnan(r))])
    if len(ref) == 0 or x is None:
        return None
    return float((np.sum(ref >= x) + 1) / (len(ref) + 1))


def decoy_false_rate(vals, alpha=0.05):
    """Apply the 'beats >= 1-alpha of the other decoys' criterion to each decoy in turn."""
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    hits = 0
    for i, v in enumerate(vals):
        p = pct_rank(v, vals[:i] + vals[i + 1:])
        hits += p <= alpha
    return hits / len(vals)


def wlog(fn, row):
    with open(fn, 'a') as f:
        f.write(row.rstrip() + '\n')
