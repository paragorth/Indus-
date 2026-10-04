#!/usr/bin/env python3
"""LA-41 shared code: WHAT THE SCRIBES EDITED.

Every document side becomes a list of ITEMS. An item is one entry:
  w    : tuple of signs of its word (or None for a bare logogram line)
  logo : commodity base (or None)
  num  : amount key (string, e.g. '20' or '2J' for Linear A, 'T7V3' for Linear B) or None
  val  : float amount (Linear A: integer part + conventional fractions only for ratio tests;
         Linear B: base units) or None
Families of near-copies are document pairs whose item lists align (Smith-Waterman local
alignment) with a high word-match score under several scoring schemes. Edits are read off the
alignment. No Linear B sound values are used for Linear A; they are used only to score the
Linear B control and as an outside check.
"""
import json, os, re, sys, math, random, unicodedata
from collections import defaultdict, Counter
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la41_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from la6_common import base_of, FRAC, LB_COM, LIQ_COM, DRYU, LIQU

TOTW = {('KU', 'RO'), ('KI', 'RO'), ('PO', 'TO', 'KU', 'RO')}


# ------------------------------------------------------------------ Linear A
def la_docs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = {}
    for ins in C:
        items = []
        cur = None
        for t in ins['tokens']:
            if t['t'] == 'word':
                cur = {'w': tuple(t['s']), 'logo': None, 'num': None, 'val': None}
                items.append(cur)
            elif t['t'] == 'logo':
                b = base_of(t['v'])
                if cur is None or cur['num'] is not None or cur['logo'] is not None:
                    cur = {'w': None, 'logo': b, 'num': None, 'val': None}
                    items.append(cur)
                else:
                    cur['logo'] = b
            elif t['t'] == 'num':
                key = str(t['v']) + ''.join(sorted(t['frac']))
                v = float(t['v'] + sum(FRAC.get(f, Fr(1, 16)) for f in t['frac']))
                if cur is None or cur['num'] is not None:
                    cur = {'w': None, 'logo': None, 'num': key, 'val': v}
                    items.append(cur)
                else:
                    cur['num'] = key; cur['val'] = v
        if items:
            out[ins['id']] = {'site': ins['site'], 'support': ins['support'], 'scribe': ins.get('scribe') or '',
                              'items': items, 'stem': re.sub(r'[ab]$', '', ins['id'])}
    return out


# ------------------------------------------------------------------ Linear B
def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')

WORD = re.compile(r'^[a-z0-9*]+(-[a-z0-9*]+)*$')
NUMT = re.compile(r'^\d+$')
MEAS = set('TVZSMNPQ')


def lb_docs(sites=('KN', 'PY')):
    """Each DAMOS document -> items. One item per word that is followed (within the line) by a
    logogram and/or number; words without a number on that line are attached as part of the item
    (the first word is the item word; others are context)."""
    out = {}
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?\d?)', h)
        if not m or m.group(1) not in sites: continue
        site, series = m.group(1), m.group(2)
        items = []
        for ln in (d.get('content') or '').split('\n'):
            toks = _strip(ln).split()
            cur = None; meas = None
            for t in toks:
                s = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.', 'v.', 'r.'): continue
                b = s.split('+')[0]
                if b in LB_COM or re.fullmatch(r'\*1\d\d', b) or (b.isupper() and len(b) >= 3 and b not in MEAS):
                    if cur is None or cur['logo'] is not None:
                        cur = {'w': None, 'logo': b, 'num': None, 'val': None, 'ctx': []}
                        items.append(cur)
                    else:
                        cur['logo'] = b
                    meas = None
                elif s in MEAS:
                    meas = s
                elif NUMT.match(s):
                    if cur is None:
                        cur = {'w': None, 'logo': None, 'num': None, 'val': None, 'ctx': []}
                        items.append(cur)
                    k = (meas or '') + s
                    cur['num'] = (cur['num'] or '') + k
                    tab = LIQU if cur['logo'] in LIQ_COM else DRYU
                    f = float(tab[meas]) if meas in tab else 1.0
                    cur['val'] = (cur['val'] or 0.0) + f * int(s)
                    meas = None
                elif WORD.match(s) and '-' in s:
                    if cur is not None and cur['w'] is not None and cur['num'] is None and cur['logo'] is None:
                        cur['ctx'].append(tuple(s.split('-')))
                    else:
                        cur = {'w': tuple(s.split('-')), 'logo': None, 'num': None, 'val': None, 'ctx': []}
                        items.append(cur)
        if items:
            out[re.sub(r'\s*\(\S*\)\s*$', '', h).strip()] = {'site': site, 'support': series, 'scribe': series,
                                             'items': items, 'stem': h}
    return out


# ------------------------------------------------------------------ alignment
def sign_edit(a, b):
    """Return ('same',) | ('sub', i, x, y) | ('ins', i, x) | ('del', i, x) | None for one-sign edits.
    Words must have >= 2 signs on both sides."""
    if a == b: return ('same',)
    if len(a) < 2 or len(b) < 2: return None
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            i = diff[0]; return ('sub', i, a[i], b[i])
        return None
    if abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        for i in range(len(l)):
            if l[:i] + l[i + 1:] == s:
                return ('ins', i, l[i]) if len(a) < len(b) else ('del', i, l[i])
    return None


SCHEMES = {
    # name: (exact word, one-sign variant, logo-only same, number bonus, gap, mismatch)
    'EXACT': (1.0, 0.0, 0.0, 0.0, -0.4, -0.5),
    'FUZZY': (1.0, 0.7, 0.0, 0.0, -0.4, -0.5),
    'FUZZLOG': (1.0, 0.7, 0.3, 0.0, -0.4, -0.5),
    'FUZZNUM': (1.0, 0.7, 0.3, 0.3, -0.4, -0.5),
    'LOOSEGAP': (1.0, 0.7, 0.3, 0.0, -0.15, -0.3),
}


def item_sim(x, y, sch, idf=None):
    ex, fz, lg, nb, gap, mm = sch
    s = None
    if x['w'] is not None and y['w'] is not None:
        e = sign_edit(x['w'], y['w'])
        if e is None: s = mm
        elif e[0] == 'same':
            s = ex * (idf.get(x['w'], 1.0) if idf else 1.0)
        else:
            s = fz if fz > 0 else mm
    elif x['w'] is None and y['w'] is None and x['logo'] and x['logo'] == y['logo']:
        s = lg if lg > 0 else mm
    else:
        s = mm
    if s > 0 and nb and x['num'] is not None and x['num'] == y['num']:
        s += nb
    return s


def sw_align(A, B, sch, idf=None):
    """Smith-Waterman local alignment over items. Returns (score, pairs [(i,j)])."""
    gap = sch[4]
    n, m = len(A), len(B)
    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    P = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0.0; bi = bj = 0
    for i in range(1, n + 1):
        Hi = H[i]; Hp = H[i - 1]; Pi = P[i]
        for j in range(1, m + 1):
            d = Hp[j - 1] + item_sim(A[i - 1], B[j - 1], sch, idf)
            u = Hp[j] + gap; l = Hi[j - 1] + gap
            v = d; p = 1
            if u > v: v, p = u, 2
            if l > v: v, p = l, 3
            if v <= 0: v, p = 0.0, 0
            Hi[j] = v; Pi[j] = p
            if v > best: best, bi, bj = v, i, j
    pairs = []
    i, j = bi, bj
    while i > 0 and j > 0 and P[i][j]:
        p = P[i][j]
        if p == 1: pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif p == 2: i -= 1
        else: j -= 1
    return best, pairs[::-1]


def set_match(A, B, fuzzy=True):
    """Order-free matching of item words (2+ signs, totals excluded): exact first, then one-sign
    variants; each item used once. Returns pairs sorted by A index."""
    ia = [i for i, it in enumerate(A) if it['w'] and len(it['w']) >= 2 and it['w'] not in TOTW]
    ib = [j for j, it in enumerate(B) if it['w'] and len(it['w']) >= 2 and it['w'] not in TOTW]
    used = set(); pairs = []
    rest = []
    for i in ia:
        j = next((j for j in ib if j not in used and B[j]['w'] == A[i]['w']), None)
        if j is None: rest.append(i)
        else: used.add(j); pairs.append((i, j))
    if fuzzy:
        for i in rest:
            j = next((j for j in ib if j not in used and sign_edit(A[i]['w'], B[j]['w']) is not None), None)
            if j is not None: used.add(j); pairs.append((i, j))
    return sorted(pairs)


def word_matches(A, B, pairs):
    """aligned pairs where words are equal or one-sign variants."""
    out = []
    for i, j in pairs:
        x, y = A[i], B[j]
        if x['w'] is not None and y['w'] is not None and sign_edit(x['w'], y['w']) is not None:
            out.append((i, j))
    return out


def fuzzy_keys(w):
    """keys shared by a word and all its one-sign variants (for candidate generation)."""
    if w is None or len(w) < 2: return set()
    ks = {('E',) + w}
    for i in range(len(w)):
        ks.add(('S', i) + w[:i] + ('_',) + w[i + 1:])     # substitution
        ks.add(('D',) + w[:i] + w[i + 1:])                 # deletion of sign i
    ks.add(('D',) + w)                                     # w itself as a deletion target
    return ks


def candidate_pairs(docs, min_items=2, min_shared=2, skip_words=TOTW):
    """doc pairs sharing >= min_shared distinct fuzzy-matching item words (words of 2+ signs)."""
    ids = [d for d in docs if sum(1 for it in docs[d]['items'] if it['w']) >= min_items]
    idx = defaultdict(set)
    for d in ids:
        for it in docs[d]['items']:
            if it['w'] and len(it['w']) >= 2 and it['w'] not in skip_words:
                for k in fuzzy_keys(it['w']): idx[k].add(d)
    # count shared words per pair (approximate by shared keys of exact form, dedup per word)
    cnt = Counter()
    for d in ids:
        seen = defaultdict(set)
        for it in docs[d]['items']:
            if it['w'] and len(it['w']) >= 2 and it['w'] not in skip_words:
                partners = set()
                for k in fuzzy_keys(it['w']): partners |= idx[k]
                for p in partners:
                    if p > d: seen[p].add(it['w'])
        for p, ws in seen.items():
            if len(ws) >= min_shared: cnt[(d, p)] = len(ws)
    return sorted(cnt)


def doc_word_idf(docs):
    df = Counter()
    for d in docs:
        for w in {it['w'] for it in docs[d]['items'] if it['w']}: df[w] += 1
    N = len(docs)
    return {w: min(2.0, max(0.3, math.log(N / c) / math.log(N / 2))) for w, c in df.items()}


def pair_stats(A, B, sch, idf=None):
    if sch in ('SET', 'SETEXACT'):
        pairs = set_match(A, B, fuzzy=(sch == 'SET')); sc = float(len(pairs))
    else:
        sc, pairs = sw_align(A, B, sch, idf)
    wm = word_matches(A, B, pairs)
    wm = [(i, j) for i, j in wm if A[i]['w'] not in TOTW and len(A[i]['w']) >= 2 and len(B[j]['w']) >= 2]
    nA = sum(1 for it in A if it['w'] and len(it['w']) >= 2 and it['w'] not in TOTW)
    nB = sum(1 for it in B if it['w'] and len(it['w']) >= 2 and it['w'] not in TOTW)
    cov = len(wm) / max(1, min(nA, nB))
    return sc, pairs, wm, cov


def is_family(sc, wm, cov, min_match=3, min_cov=0.5):
    return len(wm) >= min_match and cov >= min_cov


# ------------------------------------------------------------------ edits
def catalogue(A, B, pairs):
    """Edit operations between aligned lists A -> B (local alignment span only)."""
    ops = Counter(); subs = []; amt = []
    if not pairs: return ops, subs, amt
    i0, j0 = pairs[0]; i1, j1 = pairs[-1]
    ai = {i for i, _ in pairs}; bj = {j for _, j in pairs}
    ops['word_dropped'] = sum(1 for i in range(i0, i1 + 1) if i not in ai and A[i]['w'])
    ops['word_added'] = sum(1 for j in range(j0, j1 + 1) if j not in bj and B[j]['w'])
    for i, j in pairs:
        x, y = A[i], B[j]
        if x['w'] is not None and y['w'] is not None:
            e = sign_edit(x['w'], y['w'])
            if e is None: ops['word_replaced'] += 1; continue
            if e[0] == 'same': ops['word_same'] += 1
            else:
                ops['spelling_' + e[0]] += 1; subs.append((x['w'], y['w'], e))
        if x['logo'] and y['logo']:
            ops['logo_same' if x['logo'] == y['logo'] else 'logo_changed'] += 1
        elif bool(x['logo']) != bool(y['logo']):
            ops['logo_added_or_dropped'] += 1
        if x['num'] is not None and y['num'] is not None:
            same = x['num'] == y['num']
            tot = x['w'] in TOTW if x['w'] else False
            ops[('total_' if tot else 'amount_') + ('same' if same else 'changed')] += 1
            if not same and x['val'] and y['val']: amt.append((x['val'], y['val'], tot))
        elif (x['num'] is None) != (y['num'] is None):
            ops['amount_added_or_dropped'] += 1
    # order swaps: matching words outside the local alignment path that cross
    wa = {A[i]['w']: i for i in range(len(A)) if A[i]['w'] and A[i]['w'] not in TOTW}
    wb = {B[j]['w']: j for j in range(len(B)) if B[j]['w'] and B[j]['w'] not in TOTW}
    common = [w for w in wa if w in wb]
    sw = 0
    for a in range(len(common)):
        for b in range(a + 1, len(common)):
            if (wa[common[a]] - wa[common[b]]) * (wb[common[a]] - wb[common[b]]) < 0: sw += 1
    ops['order_swaps'] = sw
    return ops, subs, amt


# ------------------------------------------------------------------ planted copies
def plant_copies(docs, n_fam, rng, sub_table, p_amt=0.5, p_drop=0.15, p_add=0.1, p_swap=0.1, p_spell=0.15,
                 p_logo=0.05, min_items=4):
    """Copy n_fam random lists (>= min_items word items) with known edits; add as new docs.
    sub_table: {sign: replacement sign} planted spelling substitutions."""
    src = [d for d in docs if sum(1 for it in docs[d]['items'] if it['w'] and len(it['w']) >= 2) >= min_items]
    chosen = rng.sample(src, min(n_fam, len(src)))
    allwords = [it['w'] for d in docs for it in docs[d]['items'] if it['w'] and len(it['w']) >= 2]
    logos = [it['logo'] for d in docs for it in docs[d]['items'] if it['logo']]
    truth = {'pairs': [], 'subs': [], 'ops': Counter()}
    new = dict(docs)
    for k, d in enumerate(chosen):
        items = []
        for it in docs[d]['items']:
            if it['w'] and rng.random() < p_drop: truth['ops']['word_dropped'] += 1; continue
            it2 = dict(it)
            if it2['w'] and len(it2['w']) >= 2 and rng.random() < p_spell:
                cand = [i for i, s in enumerate(it2['w']) if s in sub_table]
                if cand:
                    i = rng.choice(cand); w = list(it2['w']); old = w[i]; w[i] = sub_table[old]
                    it2['w'] = tuple(w); truth['subs'].append((old, w[i])); truth['ops']['spelling_sub'] += 1
            if it2['num'] is not None and rng.random() < p_amt and it2['val']:
                f = rng.choice([0.5, 2.0, 0.8, 1.25, None])
                nv = max(1, round(it2['val'] * f)) if f else max(1, int(it2['val']) + rng.choice([-3, -2, -1, 1, 2, 3, 5]))
                if str(nv) != it2['num']:
                    it2['num'] = str(nv); it2['val'] = float(nv); truth['ops']['amount_changed'] += 1
            if it2['logo'] and rng.random() < p_logo:
                it2['logo'] = rng.choice(logos); truth['ops']['logo_changed'] += 1
            items.append(it2)
            if rng.random() < p_add:
                items.append({'w': rng.choice(allwords), 'logo': None, 'num': str(rng.randint(1, 20)), 'val': None})
                truth['ops']['word_added'] += 1
        for i in range(len(items) - 1):
            if rng.random() < p_swap:
                items[i], items[i + 1] = items[i + 1], items[i]; truth['ops']['swaps'] += 1
        nid = f'PLANT{k}_{d}'
        new[nid] = dict(docs[d]); new[nid]['items'] = items; new[nid]['stem'] = nid
        truth['pairs'].append((d, nid))
    return new, truth


def site_word_shuffle(docs, rng):
    """Permute item words among all items of the same site (lengths, numbers, logos kept)."""
    by = defaultdict(list)
    for d in docs:
        for it in docs[d]['items']:
            if it['w']: by[docs[d]['site']].append(it['w'])
    for s in by: rng.shuffle(by[s])
    pos = defaultdict(int); out = {}
    for d in docs:
        s = docs[d]['site']; items = []
        for it in docs[d]['items']:
            it2 = dict(it)
            if it['w']:
                it2['w'] = by[s][pos[s]]; pos[s] += 1
            items.append(it2)
        out[d] = dict(docs[d]); out[d]['items'] = items
    return out


def find_families(docs, scheme='FUZZY', min_match=3, min_cov=0.5, cands=None, idf=None):
    sch = SCHEMES.get(scheme, scheme)
    if cands is None: cands = candidate_pairs(docs)
    fam = []
    for a, b in cands:
        A, B = docs[a]['items'], docs[b]['items']
        sc, pairs, wm, cov = pair_stats(A, B, sch, idf)
        if is_family(sc, wm, cov, min_match, min_cov):
            fam.append((a, b, sc, len(wm), cov, pairs))
    return fam


def wlog(path, row):
    with open(path, 'a') as f: f.write(row.rstrip() + '\n')


# ------------------------------------------------------------------ calibrated family score
def site_df(docs):
    """{site: (N docs, Counter df)} over item words of 2+ signs."""
    out = {}
    by = defaultdict(list)
    for d in docs: by[docs[d]['site']].append(d)
    for s, ds in by.items():
        df = Counter()
        for d in ds:
            for w in {it['w'] for it in docs[d]['items'] if it['w']}: df[w] += 1
        out[s] = (len(ds), df)
    return out


def weight(w, site, SD):
    N, df = SD[site]
    return math.log((N + 1) / (df.get(w, 0) + 1))


def fam_score(docs, a, b, wm, SD):
    """sum over matched words of information weight (rarer shared word = stronger evidence);
    a one-sign variant counts 0.7 of the rarer form's weight."""
    A, B = docs[a]['items'], docs[b]['items']
    s = 0.0
    for i, j in wm:
        x, y = A[i]['w'], B[j]['w']
        wa = weight(x, docs[a]['site'], SD); wb = weight(y, docs[b]['site'], SD)
        s += min(wa, wb) if x == y else 0.7 * min(wa, wb)
    return s


ALL_SCHEMES = list(SCHEMES) + ['SET', 'SETEXACT']


def scored_pairs(docs, scheme, cands=None, SD=None, min_match=3, min_cov=0.5):
    sch = SCHEMES.get(scheme, scheme)
    if cands is None: cands = candidate_pairs(docs)
    if SD is None: SD = site_df(docs)
    out = []
    for a, b in cands:
        sc, pairs, wm, cov = pair_stats(docs[a]['items'], docs[b]['items'], sch)
        if len(wm) >= min_match and cov >= min_cov:
            out.append((a, b, fam_score(docs, a, b, wm, SD), len(wm), cov, pairs, wm))
    return out
