#!/usr/bin/env python3
"""LA-16 shared code: PRIVATE SPELLINGS AS A ROSETTA STONE.

Hypothesis: if one writer spells a word X-B-C where another writes Y-B-C, then X and Y are interchangeable
for those writers (same sound, near sound or a variant spelling). So sign pairs that are swapped BETWEEN
writers more than WITHIN writers are inside-the-script sign equations.

Objects
  doc   = dict(id, site, hand, words=[tuple of signs, ...])  (hand = scribe label or None)
  pairs = every pair of word types of equal length that differ in exactly one position (X in w1, Y in w2),
          optionally also one-sign indels (Y = '_').
  For a labelling `lab` (hand of each doc), H[w] = token counts of word w by label. The cross-writer weight
  of a pair is c = 1 - sum_h H1[h]H2[h] / (n1 n2)  (share of token pairs written by different labels).
  Sign-pair statistic T(X,Y) = sum of c over its word pairs (only pairs whose two words both have labelled
  tokens). Null: labels permuted over documents (within site / series) -> per-sign-pair z and max-statistic.
No sound value is used anywhere in the search; values enter only when scoring the Linear B control.
"""
import os, sys, json, re, collections, random, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from la5_common import la_docs, strip_diac, norm_la, LAD
OUT = os.path.join(HERE, '..', 'data', 'la16')
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ corpora
def la_corpus(minlen=2):
    C = {c['id']: c for c in json.load(open(os.path.join(LAD, 'corpus.json')))}
    out = []
    for d in la_docs(admin_only=False):
        ws = [t[1] for L in d['lines'] for t in L if t[0] == 'W' and len(t[1]) >= minlen]
        if not ws: continue
        s = C[d['id']].get('scribe') or None
        out.append(dict(id=d['id'], site=d['site'], hand=s, words=ws, support=d['support']))
    return out


def lb_corpus(sites=('KN', 'PY'), minlen=2, keep_q=True):
    """DAMOS docs; hand = parenthesised hand in the heading ('KN Fp 1 (138)'); '-' -> None."""
    from la5_common import lb_docs
    raw = {}
    for line in open(os.path.join(LAD, 'damos_items.jsonl')):
        d = json.loads(line); h = d.get('heading') or ''
        m = re.search(r'\(([^)]*)\)\s*$', h)
        raw[h] = m.group(1).strip() if m else None
    out = []
    for d in lb_docs():
        if d['site'] not in sites: continue
        ws = [t[1] for L in d['lines'] for t in L if t[0] == 'W' and len(t[1]) >= minlen]
        if not ws: continue
        hd = raw.get(d['id'])
        if hd in (None, '-', ''): hd = None
        elif '?' in hd:
            hd = hd.replace('?', '').strip() if keep_q else None
        if hd: hd = d['site'] + ':' + hd
        out.append(dict(id=d['id'], site=d['site'], hand=hd, words=ws, support=d['series']))
    return out


# ------------------------------------------------------------------ pairs
def build_pairs(types, indel=False):
    """types: iterable of word tuples -> list of (i, j, X, Y) with index into sorted type list."""
    T = sorted(set(types)); ix = {w: k for k, w in enumerate(T)}
    idx = collections.defaultdict(list)
    for w in T:
        for i in range(len(w)):
            idx[(len(w), i, w[:i] + w[i + 1:])].append(w)
    P = set()
    for (L, i, m), ws in idx.items():
        if len(ws) < 2: continue
        for a in range(len(ws)):
            for b in range(a + 1, len(ws)):
                w1, w2 = ws[a], ws[b]
                x, y = w1[i], w2[i]
                if x > y: w1, w2, x, y = w2, w1, y, x
                P.add((ix[w1], ix[w2], x, y))
    if indel:
        for (L, i, m), ws in idx.items():
            if m in ix and len(m) >= 2:
                for w in ws: P.add((ix[m], ix[w], '_', w[i]))
    return T, sorted(P)


class Engine:
    """Vectorised cross-writer statistic for one corpus and one label field."""
    def __init__(self, docs, field='hand', indel=False, strat='site'):
        self.docs = [d for d in docs if d.get(field)]
        self.field = field
        types = [w for d in self.docs for w in d['words']]
        self.T, self.P = build_pairs(types, indel)
        self.tix = {w: k for k, w in enumerate(self.T)}
        labs = sorted({d[field] for d in self.docs}); self.labs = labs
        self.lix = {l: k for k, l in enumerate(labs)}
        # incidence: type x doc counts
        M = np.zeros((len(self.T), len(self.docs)), dtype=np.float32)
        for j, d in enumerate(self.docs):
            for w in d['words']: M[self.tix[w], j] += 1
        self.M = M
        self.lab0 = np.array([self.lix[d[field]] for d in self.docs])
        self.strata = collections.defaultdict(list)
        for j, d in enumerate(self.docs): self.strata[d.get(strat) or ''].append(j)
        self.strata = [np.array(v) for v in self.strata.values()]
        self.pi = np.array([p[0] for p in self.P], dtype=np.int64)
        self.pj = np.array([p[1] for p in self.P], dtype=np.int64)
        keys = sorted({(p[2], p[3]) for p in self.P}); self.keys = keys
        kix = {k: n for n, k in enumerate(keys)}
        self.pk = np.array([kix[(p[2], p[3])] for p in self.P], dtype=np.int64)
        self.npairs_k = np.bincount(self.pk, minlength=len(keys))

    def H(self, lab):
        D = np.zeros((len(self.docs), len(self.labs)), dtype=np.float32)
        D[np.arange(len(self.docs)), lab] = 1
        return self.M @ D

    def cross(self, lab):
        H = self.H(lab); n = H.sum(1)
        a, b = H[self.pi], H[self.pj]
        same = (a * b).sum(1); tot = n[self.pi] * n[self.pj]
        return 1 - same / tot

    def stat(self, lab):
        return np.bincount(self.pk, weights=self.cross(lab), minlength=len(self.keys))

    def perm(self, rnd):
        lab = self.lab0.copy()
        for s in self.strata:
            v = lab[s]; rnd.shuffle(v); lab[s] = v
        return lab

    def null(self, nperm, seed=0):
        rnd = np.random.default_rng(seed)
        return np.array([self.stat(self.perm(rnd)) for _ in range(nperm)])


def rank_report(E, real, N, top=15, minpairs=2):
    mu, sd = N.mean(0), N.std(0) + 1e-9
    z = (real - mu) / sd
    exc = real - mu
    ok = E.npairs_k >= minpairs
    zN = (N - mu) / sd
    zN[:, ~ok] = -np.inf
    zmax_null = zN.max(1)
    order = [k for k in np.argsort(-z) if ok[k]]
    rows = []
    for k in order[:top]:
        fw = float((zmax_null >= z[k]).mean())
        praw = float((N[:, k] >= real[k]).mean())
        rows.append(dict(key=E.keys[k], n=int(E.npairs_k[k]), real=float(real[k]), null=float(mu[k]),
                         z=float(z[k]), P=praw, FWER=fw))
    return z, exc, rows, zmax_null


# ------------------------------------------------------------------ LB truth (control only)
LB_DOUBLET = {frozenset(p) for p in [('A', 'A2'), ('A', 'A3'), ('RA', 'RA3'), ('RA', 'RA2'), ('RO', 'RO2'),
                                     ('TA', 'TA2'), ('PU', 'PU2'), ('E', 'JE'), ('O', 'JO'), ('O', 'WO'),
                                     ('A', 'JA'), ('A', 'WA'), ('I', 'JI'), ('U', 'WU'), ('E', 'WE'), ('I', 'WI'),
                                     ('A3', 'AI'), ('RI', 'RA2'), ('TI', 'TA2'), ('PA', 'PA2'), ('QA', 'PA2'),
                                     ('PE', 'PTE'), ('TE', 'PTE'), ('RE', 'RA3'), ('RI', 'RO2')]}


def lb_cv(s):
    s = s.upper()
    if s in ('A', 'E', 'I', 'O', 'U'): return '', s
    m = re.match(r'^([A-Z]*?)([AEIOU])([0-9]?)$', s)
    if not m: return None
    return m.group(1) + (m.group(3) and '#' + m.group(3)), m.group(2)


def lb_class(x, y):
    """'doublet' (same/near value pair), 'sameC' (vowel differs), 'sameV' (consonant differs), 'other'."""
    if frozenset((x, y)) in LB_DOUBLET: return 'doublet'
    a, b = lb_cv(x), lb_cv(y)
    if not a or not b: return 'other'
    if a[0] == b[0]: return 'sameC'
    if a[1] == b[1]: return 'sameV'
    return 'other'


def say_to(path):
    f = open(path, 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); f.write(s + '\n'); f.flush()
    return say
