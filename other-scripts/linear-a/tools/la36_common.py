#!/usr/bin/env python3
"""LA-36 shared code: THE TWO SIDES ARE BEFORE AND AFTER.

When the same recipient words appear on two lists (two sides of a tablet, two tablets of one
scribe, two tablets of one site), the change between the paired amounts may be a transaction
(delivered vs owed, issued vs returned, this month vs next, a commodity equivalence).

Lists: a document side (Linear A) or a DAMOS document (Linear B). Entry: a recipient word
(2+ signs, not a total word, used once on that list) with one amount of one commodity
(bare numbers: commodity BARE). Amounts are kept as (integer, fraction letters) so that the
fraction values V can be a free parameter.

Pair classes (Linear A): SIDES (a/b of one tablet), SCRIBE (same named scribe), SITE (same
site, otherwise), XSITE (different sites).

Transformation families on the aligned amounts (x_i, y_i):
  RATIO  y = R(r x), r = p/q (p, q <= 12, r != 1), R in exact/floor/round/ceil/half-floor
  DIFF   y = x + d, d != 0
  COMP   x + y = T (sum to a fixed total)
  AFF    y = r x + d (r != 1, d != 0), needs 3 agreeing entries
  (IDENT y = x is counted separately, it is the 'nothing changed' case)
Pair score s = max(k_RATIO - 1, k_DIFF - 1, k_COMP - 1, k_AFF - 2), counting a family only
when k >= 2 (AFF k >= 3).
"""
import json, os, re, sys, random, unicodedata
from collections import defaultdict, Counter
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la36_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from la6_common import la_entries, FRAC, LB_COM, LIQ_COM, DRYU, LIQU

CONV = dict(FRAC)
TOTW = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}

# ------------------------------------------------------------------ Linear A lists
def la_lists():
    """{doc: {'site','scribe','ents':[(word, com, int, fracs)], 'words':[all words in order]}}"""
    C = {c['id']: c for c in json.load(open(os.path.join(D, 'corpus.json')))}
    E = la_entries(use_frac=False)
    by = defaultdict(list)
    for e in E:
        if e['role'] != 'entry' or not e['label'] or e['label'].count('-') < 1: continue
        raws = e['raw']
        coms = {r[0] for r in raws}
        if len(coms) == 1 and raws:
            k = raws[0][0]
            iv = sum(r[1] for r in raws); fr = tuple(sorted(f for r in raws for f in r[2]))
        elif not raws and len(e['bare']) == 1 and e['bare'][0] > 0:
            k = 'BARE'
            # recover the fraction letters of the bare number from the corpus tokens
            iv = int(e['bare'][0]); fr = ()
        else:
            continue
        by[e['doc']].append([e['label'], k, iv, fr, e['idx']])
    # bare fraction letters: re-read tokens (la_entries with use_frac=False drops them)
    out = {}
    for d, ents in by.items():
        ins = C[d]
        bare_fr = _bare_fracs(ins)
        for en in ents:
            if en[1] == 'BARE' and en[4] in bare_fr:
                en[3] = bare_fr[en[4]]
        c = Counter(en[0] for en in ents)
        ents = [tuple(en[:4]) for en in ents if c[en[0]] == 1]
        if len(ents) >= 2:
            out[d] = {'site': ins['site'], 'scribe': ins.get('scribe') or '', 'ents': ents,
                      'words': list(ins['words']), 'stem': re.sub(r'[ab]$', '', d)}
    return out


def _bare_fracs(ins):
    """entry index -> fraction letters for entries carried by a bare number (mirrors la_entries)."""
    E = [e for e in la_entries_cache() if e['doc'] == ins['id']]
    res = {}
    # walk tokens again: k-th bare number -> its frac list
    nums = []
    cur_entry = 0
    # simple approach: map by order of bare numbers in tokens vs entries with bare values
    bare_tok = [t for t in ins['tokens'] if t['t'] == 'num']
    ents_with_bare = [e for e in E if e['bare']]
    # number tokens that were bare: match in order by integer value
    j = 0
    for e in ents_with_bare:
        for b in e['bare']:
            while j < len(bare_tok) and bare_tok[j]['v'] != int(b): j += 1
            if j < len(bare_tok):
                if len(e['bare']) == 1: res[e['idx']] = tuple(sorted(bare_tok[j]['frac']))
                j += 1
    return res


_EC = None
def la_entries_cache():
    global _EC
    if _EC is None: _EC = la_entries(use_frac=False)
    return _EC


def pair_class(a, b, L):
    A, B = L[a], L[b]
    if A['stem'] == B['stem']: return 'SIDES'
    if A['site'] != B['site']: return 'XSITE'
    if A['scribe'] and A['scribe'] == B['scribe']: return 'SCRIBE'
    return 'SITE'


def shared_pairs(L, min_shared=2):
    idx = defaultdict(set)
    for d, x in L.items():
        for en in x['ents']: idx[en[0]].add(d)
    cnt = Counter()
    for w, ds in idx.items():
        ds = sorted(ds)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)): cnt[(ds[i], ds[j])] += 1
    return sorted(p for p, c in cnt.items() if c >= min_shared)


def value(iv, fr, V):
    return float(iv + sum((V.get(f, Fr(1, 16)) for f in fr), Fr(0)))

# ------------------------------------------------------------------ transformation families
RAT = sorted({Fr(p, q) for p in range(1, 13) for q in range(1, 13) if Fr(p, q) != 1})
RATF = np.array([float(r) for r in RAT])
RMODES = ['exact', 'floor', 'round', 'ceil', 'half']
EPS = 1e-6


def _apply(rx, mode):
    if mode == 'exact': return rx
    if mode == 'floor': return np.floor(rx + EPS)
    if mode == 'round': return np.floor(rx + 0.5 + EPS)
    if mode == 'ceil': return np.ceil(rx - EPS)
    if mode == 'half': return np.floor(2 * rx + EPS) / 2


def _maxmult(Z):
    """Z: (P, n) -> max multiplicity of equal values per row."""
    eq = np.abs(Z[:, :, None] - Z[:, None, :]) < 1e-6
    return eq.sum(2).max(1)


def fam_scores(X, Y, want_detail=False):
    """X, Y: (P, n) aligned amounts per replicate. Returns dict of (P,) arrays of k per family."""
    P, n = X.shape
    out = {}
    # identity
    out['IDENT'] = (np.abs(X - Y) < 1e-6).sum(1)
    # ratio with rounding
    best = np.zeros(P, int); arg = np.full(P, -1)
    rx = X[:, None, :] * RATF[None, :, None]            # (P, R, n)
    for mi, m in enumerate(RMODES):
        k = (np.abs(_apply(rx, m) - Y[:, None, :]) < 1e-6) & (X[:, None, :] > 0)
        # rounding modes must not be trivially exact-only for zero
        kk = k.sum(2)                                   # (P, R)
        a = kk.argmax(1); v = kk.max(1)
        upd = v > best
        best = np.where(upd, v, best); arg = np.where(upd, mi * len(RAT) + a, arg)
    out['RATIO'] = best; out['RATIO_arg'] = arg
    dz = Y - X
    km = _maxmult(dz)
    # exclude d = 0 groups: if the max group is the zero group, recompute without zeros
    dz2 = np.where(np.abs(dz) < 1e-6, np.nan, dz)
    out['DIFF'] = _maxmult_nan(dz2)
    out['COMP'] = _maxmult(X + Y)
    # affine exact: y - r x = d, r != 1, d != 0
    dd = Y[:, None, :] - rx                              # (P, R, n)
    dd = np.where(np.abs(dd) < 1e-6, np.nan, dd)
    if n >= 3:
        eq = np.abs(dd[:, :, :, None] - dd[:, :, None, :]) < 1e-6
        out['AFF'] = eq.sum(3).max(2).max(1)
    else:
        out['AFF'] = np.zeros(P, int)
    return out


def _maxmult_nan(Z):
    eq = np.abs(Z[:, :, None] - Z[:, None, :]) < 1e-6
    r = eq.sum(2).max(1)
    return r


def pair_score(f):
    s = np.zeros_like(f['RATIO'])
    for k, c, mn in (('RATIO', 1, 2), ('DIFF', 1, 2), ('COMP', 1, 2), ('AFF', 2, 3)):
        v = np.where(f[k] >= mn, f[k] - c, 0)
        s = np.maximum(s, v)
    return s


def describe_ratio(arg):
    mi, a = divmod(int(arg), len(RAT))
    return f'{RAT[a]} ({RMODES[mi]})'

# ------------------------------------------------------------------ Linear B lists (DAMOS)
NUM = re.compile(r'^\[?(\d+)\]?$')
STOP = {'do-so-mo', 'o-na-to', 'ke-ke-me-na', 'ko-to-na', 'pa-ro', 'to-so', 'to-sa', 'o-da-a2', 'qe',
        'e-ke', 'to-so-de', 'pe-mo', 'we-te-i-we-te-i', 'po-se-da-o-ne', 'do-e-ro'}


def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


def lb_raw():
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        out.append((d.get('heading') or '', d.get('content') or ''))
    return out


def lb_line_amounts(ln):
    toks = ln.split()
    com = defaultdict(float); cur = None
    words = []
    for i, t in enumerate(toks):
        s = _strip(t).strip('[]⟦⟧')
        w = re.sub(r'[\[\]⟦⟧?!,]', '', _strip(t))
        if re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)+', w): words.append(w)
        b = s.split('+')[0]
        if b in LB_COM or re.fullmatch(r'\*1\d\d', b): cur = b; continue
        if cur is None: continue
        mm = NUM.match(s)
        if mm and i > 0:
            prev = _strip(toks[i - 1]).strip('[]')
            tab = LIQU if cur in LIQ_COM else DRYU
            if prev in tab: com[cur] += float(tab[prev]) * int(mm.group(1))
            elif prev in ('M', 'N'): com[cur] += int(mm.group(1)) * (1.0 if prev == 'M' else 1 / 30)
            elif prev.split('+')[0].strip('[]⟦⟧') in LB_COM or re.fullmatch(r'\*1\d\d', prev): com[cur] += int(mm.group(1))
    return words, dict(com)


def lb_lists():
    by = {}
    for h, content in lb_raw():
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?)', h)
        site, series = (m.group(1), m.group(2)) if m else (h[:2], '')
        ents = []; allw = []
        for ln in content.split('\n'):
            words, com = lb_line_amounts(ln)
            allw += words
            words = [w for w in words if w not in STOP]
            if len(com) == 1 and words:
                k, v = next(iter(com.items()))
                if v > 0: ents.append((words[0], k, v, ()))
        c = Counter(e[0] for e in ents)
        ents = [e for e in ents if c[e[0]] == 1]
        if len(ents) >= 2:
            by[h] = {'site': site, 'scribe': series, 'ents': ents, 'words': allw, 'stem': h}
    return by


def lb_class(a, b, L):
    A, B = L[a], L[b]
    if A['site'] != B['site']: return 'XSITE'
    if A['scribe'] == B['scribe']: return 'SCRIBE'   # same series = same dossier
    return 'SITE'
