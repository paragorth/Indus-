#!/usr/bin/env python3
"""LA-12 'the tablets are pieces of a jigsaw': shared machinery.

A RECORD is one inscribed side (HT 9a, HT 9b ...) or, for the controls, one Linear B
document or one artificial fragment. A record holds ITEMS (section, tag, value, is_total).
  tag  = commodity logogram base (GRA, OLE, CYP ...), '*' when the number has no logogram.
BLOCKS are the sums a record could contribute to a ledger kept on another tablet:
  every section sum (all items, and per tag), the whole-record sum (all, per tag) and each
  written total. Duplicate (tag, value) blocks inside a record are merged.
TARGETS are written totals (KU-RO, PO-TO-KU-RO, KI-RO in Linear A; to-so / to-sa in Linear B).

Arithmetic modes (values are exact integer keys, so sums are exact):
  'V'  vector mode, no fraction values assumed: integer part and the COUNT of each fraction
       sign add separately (J + J stays 2J). Conservative: misses carries.
  'C'  conventional values (lineara.xyz-style FRAC from la6_common, used as data only),
       in units of 1/480. Linear B uses its known sub-units (dry 1/10,1/60,1/240; liquid 1/3,
       1/18, 1/72), same 1/480 grid would not hold, so LB 'C' uses units of 1/720.
Compatibility: a typed target (tag c) may be reached by blocks tagged c or '*';
  an untyped target by any block. A block mixing several tags is tagged 'MIX'.

Closure counts for a target T on record t (distinct records, none of them t):
  k1, k2, k3 = number of 1-, 2-, 3-subsets of blocks with sum == T
  c1, c2     = 'completion': T's own preceding section s_t (if 0 < s_t < T) plus 1 or 2
               blocks from other records == T (the list began on another tablet).
Nulls: neighbour values (same fraction part, integer part within +-25%, at least +-2),
  numbers shuffled across records (within tag), iid draws from the item distribution.
"""
import json, os, re, random
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(HERE, '..', 'data')
OUT = os.path.join(D, 'la12')
os.makedirs(OUT, exist_ok=True)
import la6_common as L6

LETTERS = ['J', 'E', 'D', 'B', 'K', 'JE', 'L2', 'F', 'A', 'H', 'L', 'DD', 'W', 'L4', 'X', 'Y', 'L6', 'L3',
           'T', 'V', 'Z', 'S', 'Vd', 'Zd', 'Vl', 'Zl', '?']
LI = {l: i for i, l in enumerate(LETTERS)}
RAD = 1 << 7          # per-letter count field (sums of <= 4 blocks stay < 128)
MV = RAD ** len(LETTERS)


class Q:
    """Additive quantity: integer part + letter counts."""
    __slots__ = ('n', 'f')

    def __init__(self, n=0, f=None):
        self.n = n; self.f = Counter(f or {})

    def __add__(self, o):
        if isinstance(o, Q):
            return Q(self.n + o.n, self.f + o.f)
        if o == 0:
            return Q(self.n, self.f)
        return NotImplemented
    __radd__ = __add__

    def key(self, mode, cval):
        if mode == 'V':
            k = self.n * MV
            for l, c in self.f.items():
                k += c * RAD ** LI.get(l, LI['?'])
            return k
        U = cval['_U']
        v = self.n * U
        for l, c in self.f.items():
            v += c * cval.get(l, cval['_def'])
        return v

    def iszero(self):
        return self.n == 0 and not any(self.f.values())


LA_CVAL = {'_U': 480, '_def': 30}
for _l, _v in L6.FRAC.items():
    LA_CVAL[_l] = int(_v * 480)
LB_CVAL = {'_U': 720, '_def': 10, 'T': 72, 'Vd': 12, 'Zd': 3, 'S': 240, 'Vl': 40, 'Zl': 10}


def _intpart(key, mode, cval):
    return key // MV if mode == 'V' else key // cval['_U']


# --------------------------------------------------------------------------- Linear A records
def la_records(site_prefix='HT', support='Tablet'):
    """Records for one site, using la6_common.la_entries with Q-valued quantities."""
    orig = L6.qval
    L6.qval = lambda t, use_frac=True: Q(int(t['v']), Counter(t['frac']))
    try:
        E = L6.la_entries()
    finally:
        L6.qval = orig
    C = {c['id']: c for c in json.load(open(os.path.join(D, 'corpus.json')))}
    fs = findspots()
    recs = {}
    for e in E:
        if not e['doc'].startswith(site_prefix) or (support and e['support'] != support):
            continue
        r = recs.setdefault(e['doc'], {'id': e['doc'], 'items': [], 'sec': 0, 'labels': [],
                                       'scribe': C[e['doc']]['scribe'], 'findspot': fs.get(e['doc'], ''),
                                       'words': C[e['doc']]['words']})
        is_tot = e['role'] in ('total', 'grand', 'deficit')
        kind = e['role'] if is_tot else 'entry'
        for c, v in e['com'].items():
            if not v.iszero():
                r['items'].append({'sec': r['sec'], 'tag': c, 'q': v, 'tot': is_tot, 'kind': kind, 'label': e['label']})
        for v in e['bare']:
            if not v.iszero():
                r['items'].append({'sec': r['sec'], 'tag': '*', 'q': v, 'tot': is_tot, 'kind': kind, 'label': e['label']})
        if is_tot:
            r['sec'] += 1
    return [r for r in recs.values() if r['items']]


def findspots():
    p = os.path.join(D, 'LinearAInscriptions.js')
    out = {}
    cur = None
    for line in open(p, encoding='utf-8'):
        m = re.match(r'^\["([^"]+)",\{', line)
        if m:
            cur = m.group(1)
        m = re.search(r'"findspot": "([^"]*)"', line)
        if m and cur:
            out[cur] = m.group(1)
    return out


# --------------------------------------------------------------------------- Linear B records
LB_DRY = {'GRA', 'HORD', 'FAR', 'NI', 'OLIV', 'CYP', 'AROM', 'CROC', 'KA±PO', 'PYC', 'SA', 'KO', 'MA', 'KU', 'SE', 'PO', 'ME', 'RI'}
LB_LIQ = {'OLE', 'VIN', 'ME±RI'}
LB_OTHER = {'VIR', 'MUL', 'OVIS', 'CAP', 'SUS', 'BOS', 'TELA', 'LANA', 'AES', 'AUR', 'TU±RO2', 'A±RE±PA', 'CORN', 'ARB',
            'OVISm', 'OVISf', 'CAPm', 'CAPf', 'SUSm', 'SUSf', 'BOSm', 'BOSf', 'EQU', 'ROTA', 'BIG', 'CUR', 'GAL', 'HAS',
            'JAC', 'SAG', 'TUN'}
UNITLET = {'T', 'V', 'Z', 'S'}


def lb_records(prefix='PY'):
    recs = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = re.sub(r'\s*\([^()]*\)\s*$', '', (d.get('heading') or '')).strip()
        if not h.startswith(prefix):
            continue
        items = []; sec = 0; words = []
        for l in (d.get('content') or '').split('\n'):
            toks = [L6.re.sub(r'[\[\]⟦⟧?]', '', t) for t in l.split()]
            is_tot = any(re.match(r'^to-s[oa]', t) for t in toks)
            cur = None; i = 0; any_q = False
            for t in toks:
                if re.match(r'^[a-z][a-z0-9\-]+$', t) and '-' in t:
                    words.append(t)
            while i < len(toks):
                t = toks[i]; b = t.split('+')[0].split(':')[0]
                if b in LB_DRY or b in LB_LIQ or b in LB_OTHER:
                    cur = b; i += 1
                    n = 0; lets = Counter(); has = False
                    if i < len(toks) and toks[i].isdigit():
                        n = int(toks[i]); has = True; i += 1
                    while i + 1 < len(toks) and toks[i] in UNITLET and toks[i + 1].isdigit():
                        u = toks[i]; c = int(toks[i + 1])
                        if cur in LB_LIQ:
                            lets[{'S': 'S', 'V': 'Vl', 'Z': 'Zl'}.get(u, '?')] += c
                        elif cur in LB_DRY:
                            lets[{'T': 'T', 'V': 'Vd', 'Z': 'Zd'}.get(u, '?')] += c
                        else:
                            lets['?'] += c
                        has = True; i += 2
                    if has:
                        items.append({'sec': sec, 'tag': cur, 'q': Q(n, lets), 'tot': is_tot,
                                      'kind': 'total' if is_tot else 'entry', 'label': None})
                        any_q = True
                    continue
                i += 1
            if is_tot and any_q:
                sec += 1
        if items:
            recs.append({'id': h, 'items': items, 'scribe': '', 'findspot': '', 'words': words})
    return recs


# --------------------------------------------------------------------------- blocks and targets
def _sumtag(its):
    tags = {i['tag'] for i in its}
    s = Q()
    for i in its:
        s = s + i['q']
    return s, (tags.pop() if len(tags) == 1 else 'MIX')


def blocks_targets(recs, mode, cval, include_deficit=True):
    """Return blocks: list of (key, tag, rid, desc) and targets: list of dicts."""
    blocks = []; targets = []
    for rid, r in enumerate(recs):
        seen = set()

        def add(q, tag, desc):
            if q.iszero():
                return
            k = q.key(mode, cval)
            if (k, tag) in seen:
                return
            seen.add((k, tag)); blocks.append((k, tag, rid, desc))
        ent = [i for i in r['items'] if not i['tot']]
        secs = sorted({i['sec'] for i in ent})
        for s in secs:
            its = [i for i in ent if i['sec'] == s]
            q, tag = _sumtag(its); add(q, tag, 'sec%d' % s)
            for tg in {i['tag'] for i in its}:
                q, _ = _sumtag([i for i in its if i['tag'] == tg]); add(q, tg, 'sec%d:%s' % (s, tg))
        if ent:
            q, tag = _sumtag(ent); add(q, tag, 'all')
            for tg in {i['tag'] for i in ent}:
                q, _ = _sumtag([i for i in ent if i['tag'] == tg]); add(q, tg, 'all:%s' % tg)
        for j, i in enumerate(r['items']):
            if i['tot']:
                add(i['q'], i['tag'], 'tot%d' % j)
                if i['kind'] == 'deficit' and not include_deficit:
                    continue
                own = [x for x in ent if x['sec'] == i['sec'] and (i['tag'] == '*' or x['tag'] in (i['tag'], '*'))]
                s, _ = _sumtag(own) if own else (Q(), None)
                targets.append({'rid': rid, 'rec': r['id'], 'key': i['q'].key(mode, cval), 'tag': i['tag'],
                                'kind': i['kind'], 'own': s.key(mode, cval) if own else 0, 'idx': j,
                                'int': i['q'].n, 'frackey': i['q'].key(mode, cval) - i['q'].n * (MV if mode == 'V' else cval['_U'])})
    return blocks, targets


class Index:
    """Sparse sum index for one compatibility class (list of blocks)."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.S1 = Counter(); self.byrec = defaultdict(Counter)
        for k, tag, rid, _ in blocks:
            self.S1[k] += 1; self.byrec[rid][k] += 1
        self.P2 = Counter()
        n = len(blocks)
        for a in range(n):
            ka, _, ra, _ = blocks[a]
            for b in range(a + 1, n):
                kb, _, rb, _ = blocks[b]
                if ra != rb:
                    self.P2[ka + kb] += 1
        self._t3 = {}

    def s1x(self, x, r):
        return self.S1.get(x, 0) - self.byrec[r].get(x, 0)

    def Q(self, x, r):        # pairs (distinct records) with one block from r summing to x
        return sum(c * self.s1x(x - k, r) for k, c in self.byrec[r].items())

    def T3(self, v):
        if v in self._t3:
            return self._t3[v]
        tot = 0
        for k, tag, rid, _ in self.blocks:
            x = v - k
            tot += self.P2.get(x, 0) - self.Q(x, rid)
        self._t3[v] = tot // 3
        return tot // 3

    def h(self, v, t, kmax=3):
        """k-subset counts summing to v, distinct records, none from record t."""
        h1 = self.s1x(v, t)
        h2 = self.P2.get(v, 0) - self.Q(v, t)
        out = [h1, h2]
        if kmax >= 3:
            # triples with exactly one block in t: sum over b in t of pairs avoiding t summing to v-b
            # triples with two blocks in t are impossible (distinct records)
            sub = 0
            for k, c in self.byrec[t].items():
                x = v - k
                sub += c * (self.P2.get(x, 0) - self.Q(x, t))
            out.append(self.T3(v) - sub)
        return out


def make_indexes(blocks):
    tags = {b[1] for b in blocks}
    idx = {'*': Index(blocks)}
    for tg in tags:
        if tg in ('*', 'MIX'):
            continue
        idx[tg] = Index([b for b in blocks if b[1] in (tg, '*')])
    return idx


def target_counts(idx, T, v=None, kmax=3):
    """[k1,k2,k3,c1,c2] for target T (at value v if given)."""
    v = T['key'] if v is None else v
    I = idx.get(T['tag'], idx['*']) if T['tag'] != '*' else idx['*']
    hk = I.h(v, T['rid'], kmax)
    own = T['own']
    if 0 < own < v:
        hc = I.h(v - own, T['rid'], 2)
    else:
        hc = [0, 0]
    return hk + hc


def neighbours(T, mode, cval, rng=None, kmin=2, frac=0.25):
    n = T['int']; unit = MV if mode == 'V' else cval['_U']
    lo = max(1, min(n - kmin, int(round(n * (1 - frac))))); hi = max(n + kmin, int(round(n * (1 + frac))))
    return [m * unit + T['frackey'] for m in range(lo, hi + 1) if m != n]


# --------------------------------------------------------------------------- nulls
def shuffle_numbers(recs, rng, within_tag=True, totals_too=True):
    """Permute item quantities across records (entries among entries, totals among totals)."""
    out = [dict(r, items=[dict(i) for i in r['items']]) for r in recs]
    groups = defaultdict(list)
    for r in out:
        for i in r['items']:
            if i['tot'] and not totals_too:
                continue
            groups[(i['tot'], i['tag'] if within_tag else '')].append(i)
    for g in groups.values():
        qs = [i['q'] for i in g]; rng.shuffle(qs)
        for i, q in zip(g, qs):
            i['q'] = q
    return out


def iid_numbers(recs, rng):
    out = [dict(r, items=[dict(i) for i in r['items']]) for r in recs]
    pool = defaultdict(list)
    for r in recs:
        for i in r['items']:
            pool[(i['tot'], i['tag'])].append(i['q'])
    for r in out:
        for i in r['items']:
            i['q'] = rng.choice(pool[(i['tot'], i['tag'])])
    return out


# --------------------------------------------------------------------------- planted cuts
def closes_internally(r, mode, cval):
    """Indices of total items that equal the sum of their own section (compatible tags)."""
    ok = []
    ent = [i for i in r['items'] if not i['tot']]
    for j, i in enumerate(r['items']):
        if i['tot'] and i['kind'] != 'deficit':
            own = [x for x in ent if x['sec'] == i['sec'] and (i['tag'] == '*' or x['tag'] in (i['tag'], '*'))]
            if len(own) >= 3:
                s, _ = _sumtag(own)
                if s.key(mode, cval) == i['q'].key(mode, cval):
                    ok.append(j)
    return ok


def cut(recs, mode, cval, rng, nplant=None, nfrag=(2, 3)):
    """Cut every record that has an internally-closing total into 2-3 fragments: the
    fragment holding the total keeps a random share of its section; the other entries
    of that section go to new fragment records. Returns (new recs, truth list)."""
    cand = [ri for ri, r in enumerate(recs) if closes_internally(r, mode, cval)]
    if nplant is not None and len(cand) > nplant:
        cand = rng.sample(cand, nplant)
    out = []; truth = []
    cs = set(cand)
    for ri, r in enumerate(recs):
        if ri not in cs:
            out.append(r); continue
        j = closes_internally(r, mode, cval)[0]
        tot = r['items'][j]
        sec_items = [i for i in r['items'] if not i['tot'] and i['sec'] == tot['sec']]
        others = [i for i in r['items'] if i not in sec_items]
        k = rng.choice(nfrag)
        sh = sec_items[:]; rng.shuffle(sh)
        # cut points: the total-fragment keeps >= 0 entries, every other fragment >= 1
        nkeep = rng.randint(0, max(0, len(sh) - (k - 1)))
        keep, rest = sh[:nkeep], sh[nkeep:]
        parts = [[] for _ in range(k - 1)]
        for t_, it in enumerate(rest):
            parts[t_ % (k - 1) if t_ < k - 1 else rng.randrange(k - 1)].append(it)
        base = dict(r, id=r['id'] + '#0', items=sorted(others + keep, key=lambda i: r['items'].index(i)))
        out.append(base)
        fids = []
        for p, its in enumerate(parts):
            fr = dict(r, id=r['id'] + '#%d' % (p + 1), items=[dict(i, sec=0) for i in its])
            out.append(fr); fids.append(fr['id'])
        truth.append({'rec': base['id'], 'frags': fids, 'k': k - 1, 'keep': nkeep})
    rng.shuffle(out)
    return out, truth
