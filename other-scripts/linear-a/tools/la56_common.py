#!/usr/bin/env python3
"""LA-56 'the scribes' mistakes show how they thought': inclusion-rule learning on written totals.

Not an error-mechanism fit (that was LA-9): here every item that COULD have entered a total
carries attributes (label word, first/last sign, commodity, same-as-total commodity, fraction
letters, position, size, role), and thousands of random INCLUSION RULES (exclude / subtract /
double items with an attribute, add sub-totals, after-total items or the other side, fraction
letters ignored / swapped / read as a whole unit, and random pairs of these) are scored by the
corpus log marginal likelihood of the written totals. Unknown fraction values are a nuisance:
D random value vectors are drawn from the LA-9 grid and the likelihood is averaged over them
(values shared across the corpus, not per section).

Universal section format (LA, LB, Ur III, planted):
  {'id', 'tab', 'dam': bool, 'total': (int, {L: n}), 'items': [ {'v': int, 'l': {L: n},
     'role': 'main'|'sub'|'after'|'other'|'osub', 'f': set of 'feat=value' strings} ]}
Default reading: main items count once, all other roles not at all.
No Linear B sound values are used for Linear A; letters are opaque fraction signs.
"""
import json, os, re, sys, random
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D_DIR = os.path.join(HERE, '..', 'data')
CK = os.path.join(D_DIR, 'la56_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)

U = 14400
_G = set()
for k in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 48, 50, 60, 64, 72, 80, 90,
          96, 100, 120, 144, 150, 180, 200, 240, 300, 360]:
    _G.add((1, k))
for n, k in [(2, 3), (3, 4), (3, 8), (5, 8), (3, 16), (5, 16), (7, 16), (2, 5), (3, 5), (3, 10),
             (3, 20), (5, 6), (7, 8), (5, 12), (7, 12)]:
    _G.add((n, k))
GRID = np.array(sorted(U * n // k for n, k in _G), dtype=np.int64)
DELTA = 0.05          # background probability that a section closes by accident / was not meant to close


def mag(v):
    return '0' if v == 0 else '1' if v == 1 else '2-9' if v < 10 else '10-99' if v < 100 else '100+'


# ------------------------------------------------------------------ Linear A
MARK = {'KU-RO', 'PO-TO-KU-RO'}
SIGN_AS_LOGO = os.environ.get('LA56_SIGNLOGO', '0') == '1'


def _w(t):
    return '-'.join(t['s']) if t['t'] == 'word' else None


def tab_of(i):
    return re.sub(r'[ab]$', '', i)


def _side_items(T, sign_logo=None):
    SL = SIGN_AS_LOGO if sign_logo is None else sign_logo
    return _side_items0(T, SL)


def _side_items0(T, SIGN_AS_LOGO):
    """All numbers of one side with attributes; marks totals (number right after KU-RO/PO-TO-KU-RO)."""
    out = []; word = None; logo = None; line = 0; lastword_line = -1
    tot_next = None
    for i, t in enumerate(T):
        if t['t'] == 'nl':
            line += 1; continue
        if t['t'] == 'word':
            w = _w(t)
            if w in MARK:
                tot_next = (w, i); word = None; continue
            # a single unknown sign (*304, *308 ...) written straight before a number acts as a commodity sign
            if SIGN_AS_LOGO and len(t['s']) == 1 and t['s'][0].startswith('*') and i + 1 < len(T) and T[i + 1]['t'] == 'num':
                logo = t['s'][0]
                if tot_next and i - tot_next[1] <= 2:
                    continue
                word = None; continue
            word = w; lastword_line = line; tot_next = tot_next if tot_next and i - tot_next[1] < 3 else None
            continue
        if t['t'] == 'logo':
            logo = t['v']; continue
        if t['t'] == 'num':
            is_tot = tot_next is not None and i - tot_next[1] <= 3
            out.append({'v': int(t['v']), 'l': dict(Counter(t['frac'])), 'word': word if not is_tot else tot_next[0],
                        'logo': logo, 'line': line, 'tot': tot_next[0] if is_tot else None, 'tok': i})
            if is_tot: tot_next = None
            # a word labels only the first number after it on its line
            word = None
    return out


def _feats(it, tcom, pos, n, multi):
    f = set()
    w = it['word'] or 'NONE'
    f.add('w=' + w)
    if w != 'NONE':
        s = w.split('-'); f.add('w1=' + s[0]); f.add('wl=' + s[-1]); f.add('wlen=%d' % min(len(s), 4))
    f.add('logo=' + (it['logo'] or 'NONE'))
    if tcom:
        f.add('com=' + ('SAME' if (it['logo'] or '').split('+')[0] == tcom.split('+')[0] else 'DIFF'))
    f.add('frac=' + ('Y' if it['l'] else 'N'))
    for L in it['l']: f.add('let=' + L)
    f.add('pos=' + ('first' if pos == 0 else 'last' if pos == n - 1 else 'mid'))
    f.add('mag=' + mag(it['v']))
    if multi: f.add('multi=Y')
    return f


def load_la(sign_logo=None):
    C = json.load(open(os.path.join(D_DIR, 'corpus.json')))
    by = {c['id']: c for c in C}
    sides = defaultdict(list)
    for c in C: sides[tab_of(c['id'])].append(c['id'])
    secs = []
    for ins in C:
        T = ins['tokens']
        if not any(_w(t) in MARK for t in T): continue
        its = _side_items(T, sign_logo)
        dam_side = any(t['t'] == 'unk' for t in T)
        # other side items (non-totals) and its totals
        oth = [o for o in sides[tab_of(ins['id'])] if o != ins['id']]
        oits = []
        for o in oth:
            for it in _side_items(by[o]['tokens'], sign_logo):
                oits.append(dict(it, _side=o))
        start = 0
        for k, it in enumerate(its):
            if not it['tot']: continue
            kind = it['tot']
            # total commodity: logogram between marker and number, else current
            tcom = it['logo']
            if kind == 'KU-RO':
                main = [x for x in its[start:k] if not x['tot']]
                sub = [x for x in its[:start] if x['tot']]
            else:   # PO-TO-KU-RO: every non-total item before it on this side
                main = [x for x in its[:k] if not x['tot']]
                sub = [x for x in its[:k] if x['tot']]
            nxt = next((j for j in range(k + 1, len(its)) if its[j]['tot']), len(its))
            after = [x for x in its[k + 1:nxt] if not x['tot']]
            start = k + 1
            if not main: continue
            lines = Counter(x['line'] for x in main)
            items = []
            for p, x in enumerate(main):
                items.append({'v': x['v'], 'l': x['l'], 'role': 'main',
                              'f': _feats(x, tcom, p, len(main), lines[x['line']] > 1), 'word': x['word']})
            for x in sub:
                items.append({'v': x['v'], 'l': x['l'], 'role': 'sub', 'f': {'role=sub'}, 'word': x['word']})
            for x in after:
                items.append({'v': x['v'], 'l': x['l'], 'role': 'after',
                              'f': {'role=after'} | _feats(x, tcom, 1, 3, False), 'word': x['word']})
            for x in oits:
                r = 'osub' if x['tot'] else 'other'
                items.append({'v': x['v'], 'l': x['l'], 'role': r, 'f': {'role=' + r}, 'word': x['word']})
            secs.append({'id': '%s:%s@%d' % (ins['id'], kind, it['tok']), 'tab': tab_of(ins['id']), 'kind': kind,
                         'dam': dam_side, 'total': (it['v'], it['l']), 'tcom': tcom, 'items': items})
    return secs


# ------------------------------------------------------------------ Linear B (DAMOS), all commodities kept
def load_lb():
    import la9_common as L9
    secs = []
    for line in open(os.path.join(D_DIR, 'damos_items.jsonl')):
        d = json.loads(line)
        cont = d.get('content') or ''
        if not re.search(r'\bto-s[oa]\b', cont): continue
        lines = [l for l in cont.split('\n') if l.strip()]
        quants = []; cur = None
        for ln, l in enumerate(lines):
            toks = [L9._clean(t) for t in l.split()]
            is_tot_line = any(re.match(r'^to-s[oa]', t) for t, _ in toks)
            seen = False; i = 0; lastw = None
            while i < len(toks):
                t, dam = toks[i]
                if re.match(r'^to-s[oa]', t): seen = True
                base = t.split('+')[0]
                is_com = base in L9.LB_DRY or base in L9.LB_LIQ or base in L9.LB_OTHER
                if is_com or (t in ('T', 'V', 'Z', 'S') and cur):
                    if is_com:
                        cur = base; i += 1; ival = 0; has = False
                        if i < len(toks) and toks[i][0].isdigit():
                            ival = int(toks[i][0]); dam |= toks[i][1]; has = True; i += 1
                    else:
                        ival = 0; has = False
                    lets = Counter()
                    while i + 1 < len(toks) and toks[i][0] in ('T', 'V', 'Z', 'S') and toks[i + 1][0].isdigit():
                        u = toks[i][0]; n = int(toks[i + 1][0]); dam |= toks[i + 1][1]
                        fam = 'L' if cur in L9.LB_LIQ else ('D' if cur in L9.LB_DRY else 'O')
                        lets[u + fam] += n; has = True; i += 2
                    if has:
                        quants.append({'ln': ln, 'com': cur, 'v': ival, 'l': dict(lets), 'dam': dam,
                                       'tot': is_tot_line and seen, 'word': lastw})
                    elif not is_com:
                        i += 1
                    continue
                if re.match(r'^[a-z]', t) and not re.match(r'^to-s[oa]', t): lastw = t
                i += 1
        start = 0
        for k, q in enumerate(quants):
            if not q['tot']: continue
            if k > 0 and quants[k - 1]['tot'] and quants[k - 1]['ln'] == q['ln']:
                continue
            main = [e for e in quants[start:k] if not e['tot']]
            start = k + 1
            if len(main) < 2: continue
            lines = Counter(e['ln'] for e in main)
            items = []
            for p, e in enumerate(main):
                x = {'word': e['word'], 'logo': e['com'], 'l': e['l'], 'v': e['v']}
                items.append({'v': e['v'], 'l': e['l'], 'role': 'main', 'word': e['word'],
                              'f': _feats(x, q['com'], p, len(main), lines[e['ln']] > 1)})
            secs.append({'id': d['heading'].split('(')[0].strip() + '@%d' % k, 'tab': d['heading'].split('(')[0].strip(),
                         'kind': 'to-so', 'dam': any(e['dam'] for e in main) or q['dam'], 'total': (q['v'], q['l']), 'tcom': q['com'], 'items': items})
    return secs


# ------------------------------------------------------------------ Ur III count lists (pe42 cases)
def load_ur3():
    from fractions import Fraction as Fr
    p = os.path.join(HERE, '..', '..', 'proto-elamite', 'data', 'pe42_ckpt', 'cases_UR3.json')
    M = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000}
    val = lambda nums: sum(n * M[c.split('@')[0]] for n, c in nums)
    secs = []
    for c in json.load(open(p)):
        h = c['hyps'][0][0]
        E = h['E']
        items = []
        for k, e in enumerate(E):
            w = '-'.join(e['signs'][:1]) if e['signs'] else None
            x = {'word': w.replace('-', '_') if w else None, 'logo': None, 'l': {}, 'v': val(e['nums'])}
            items.append({'v': x['v'], 'l': {}, 'role': 'main', 'word': x['word'],
                          'f': _feats(x, None, k, len(E), False)})
        secs.append({'id': c['id'], 'tab': c['id'], 'kind': 'szunigin', 'dam': False,
                     'total': (val(h['T']['nums']), {}), 'tcom': None, 'items': items})
    return secs


# ------------------------------------------------------------------ scoring engine
class Engine:
    """Pre-computes the value of every item under D random fraction-value draws."""

    def __init__(self, secs, D=1000, seed=0):
        self.secs = secs
        lets = sorted({L for s in secs for it in s['items'] for L in it['l']} | {L for s in secs for L in s['total'][1]})
        if not lets: D = 1
        self.lets = lets; self.li = {L: i for i, L in enumerate(lets)}
        rng = np.random.default_rng(seed)
        self.V = GRID[rng.integers(0, len(GRID), size=(D, max(1, len(lets))))]   # D x nL
        self.D = D
        self.pre = []
        for s in secs:
            n = len(s['items'])
            ints = np.array([it['v'] for it in s['items']], dtype=np.int64) * U
            Lm = np.zeros((n, max(1, len(lets))), dtype=np.int64)
            for j, it in enumerate(s['items']):
                for L, c in it['l'].items(): Lm[j, self.li[L]] += c
            tl = np.zeros(max(1, len(lets)), dtype=np.int64)
            for L, c in s['total'][1].items(): tl[self.li[L]] += c
            self.pre.append({'ints': ints, 'Lm': Lm, 'tint': s['total'][0] * U, 'tl': tl,
                             'base_w': np.array([1 if it['role'] == 'main' else 0 for it in s['items']], dtype=np.int64),
                             'feats': [it['f'] for it in s['items']]})
        self.base = np.stack([self._closes(k, self.pre[k]['base_w']) for k in range(len(secs))])  # S x D bool
        self.base_log = np.log(self.base + DELTA)
        self.base_sum = self.base_log.sum(0)

    def _closes(self, k, w, Ve=None, Vt=None):
        p = self.pre[k]
        Ve = self.V if Ve is None else Ve
        Vt = self.V if Vt is None else Vt
        ev = p['ints'][None, :] + (Ve @ p['Lm'].T)   # D x n
        tot = p['tint'] + Vt @ p['tl']
        return (ev @ w == tot).astype(np.float64)

    def ll(self, rows):
        """rows: {section index: D-vector of closes} replacing the base rows. Returns log marginal lik."""
        s = self.base_sum.copy()
        for k, r in rows.items():
            s += np.log(r + DELTA) - self.base_log[k]
        m = s.max()
        return m + np.log(np.mean(np.exp(s - m)))

    def base_ll(self):
        return self.ll({})

    def apply(self, rule, idx=None):
        """Returns {k: closes row} for sections the rule touches (idx limits to a subset)."""
        out = {}
        ks = range(len(self.secs)) if idx is None else idx
        for k in ks:
            r = apply_rule(self, k, rule)
            if r is not None: out[k] = r
        return out

    def ll_subset(self, rows, idx):
        s = self.base_log[idx].sum(0).copy()
        pos = {k: i for i, k in enumerate(idx)}
        for k, r in rows.items():
            if k in pos: s += np.log(r + DELTA) - self.base_log[k]
        m = s.max()
        return m + np.log(np.mean(np.exp(s - m)))


def weights(eng, k, rule):
    """Weight vector for section k under an item rule, or None if the rule does not touch it."""
    p = eng.pre[k]; w = p['base_w'].copy(); touched = False
    for part in rule:
        op = part[0]
        if op in ('EXCL', 'NEG', 'DBL', 'ONLY'):
            f = part[1]
            for j, fs in enumerate(p['feats']):
                if w[j] == 0 and op != 'ONLY': continue
                if eng.secs[k]['items'][j]['role'] != 'main': continue
                hit = f in fs
                if op == 'ONLY':
                    if not hit and w[j] != 0: w[j] = 0; touched = True
                elif hit:
                    w[j] = {'EXCL': 0, 'NEG': -1, 'DBL': 2}[op]; touched = True
        elif op == 'ADD':
            role = part[1]; f = part[2] if len(part) > 2 else None
            for j, it in enumerate(eng.secs[k]['items']):
                if it['role'] == role and (f is None or f in it['f']):
                    w[j] = 1; touched = True
    return w if touched else None


def apply_rule(eng, k, rule):
    frac = [p for p in rule if p[0] in ('IGN', 'EQ', 'WHOLE', 'TIGN', 'DROPFR')]
    item = [p for p in rule if p not in frac and p[0] != 'COUNT']
    w = weights(eng, k, item) if item else None
    if item and w is None and not frac: return None
    itouch = w is not None
    if w is None: w = eng.pre[k]['base_w']
    Ve = eng.V; Vt = eng.V; ftouch = False
    p = eng.pre[k]
    for q in frac:
        if q[0] in ('IGN', 'EQ', 'WHOLE'):
            L = eng.li.get(q[1])
            if L is None or not p['Lm'][:, L].any(): continue
            Ve = Ve.copy(); ftouch = True
            if q[0] == 'IGN': Ve[:, L] = 0
            elif q[0] == 'WHOLE': Ve[:, L] = U
            else:
                L2 = eng.li.get(q[2])
                if L2 is None: continue
                Ve[:, L] = eng.V[:, L2]
        elif q[0] == 'DROPFR' and p['Lm'].any():
            Ve = Ve * 0; ftouch = True
        elif q[0] == 'TIGN' and p['tl'].any():
            Vt = Vt * 0; ftouch = True
    if any(q[0] == 'COUNT' for q in rule):
        # total = number of counted items (a tally of lines, not of quantities)
        n = int((w != 0).sum())
        return np.full(eng.D, 1.0 if p['tint'] == n * U and not p['tl'].any() else 0.0)
    if not itouch and not ftouch: return None
    return eng._closes(k, w, Ve, Vt)


def rule_str(rule):
    return ' & '.join(':'.join(p) for p in rule)


def candidate_rules(secs, min_secs=2, n_pairs=3000, seed=0):
    """All single rules whose feature occurs in >= min_secs sections, fraction rules, and random pairs."""
    fsec = defaultdict(set)
    roles = set()
    lets = set()
    for k, s in enumerate(secs):
        for it in s['items']:
            if it['role'] == 'main':
                for f in it['f']: fsec[f].add(k)
            else:
                roles.add(it['role'])
                for f in it['f']:
                    if f.startswith('w=') or f.startswith('logo='): fsec[('R', it['role'], f)].add(k)
            lets |= set(it['l'])
        lets |= set(s['total'][1])
    singles = []
    for f, ks in fsec.items():
        if len(ks) < min_secs: continue
        if isinstance(f, tuple):
            singles.append((('ADD', f[1], f[2]),)); continue
        for op in ('EXCL', 'NEG', 'DBL'):
            singles.append(((op, f),))
        if f.startswith('com=') or f.startswith('logo=') or f.startswith('frac='):
            singles.append((('ONLY', f),))
    for r in roles: singles.append((('ADD', r),))
    lets = sorted(lets)
    for L in lets:
        singles.append((('IGN', L),)); singles.append((('WHOLE', L),))
        for L2 in lets:
            if L2 != L: singles.append((('EQ', L, L2),))
    singles.append((('DROPFR',),)); singles.append((('TIGN',),)); singles.append((('COUNT',),))
    rng = random.Random(seed)
    pairs = set()
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 20:
        tries += 1
        a, b = rng.sample(singles, 2)
        if a[0][0] in ('IGN', 'EQ', 'WHOLE') and b[0][0] in ('IGN', 'EQ', 'WHOLE') and a[0][1] == b[0][1]: continue
        pairs.add(tuple(sorted([a[0], b[0]])))
    return singles, sorted(pairs)


def score_rules(eng, rules, idx=None):
    """Gain in log marginal likelihood over the default reading, on sections idx (all if None)."""
    if idx is None: idx = list(range(len(eng.secs)))
    base = eng.ll_subset({}, idx)
    out = np.zeros(len(rules))
    for i, r in enumerate(rules):
        rows = eng.apply(r, idx)
        out[i] = eng.ll_subset(rows, idx) - base if rows else 0.0
    return out


def closes_report(eng):
    """Per section: probability (over value draws) that the default reading closes."""
    return eng.base.mean(1)


def eval_rows(eng, rules):
    """Rows (bool, per touched section) of every rule over the whole corpus."""
    out = []
    for r in rules:
        rows = eng.apply(r)
        out.append({k: v.astype(bool) for k, v in rows.items()})
    return out


def gains(eng, rows_list, idx):
    """Log marginal-likelihood gain over the default reading on section subset idx."""
    idx = list(idx); sidx = set(idx)
    B = eng.base_log[idx].sum(0)
    def lme(s):
        m = s.max(); return m + np.log(np.mean(np.exp(s - m)))
    b0 = lme(B)
    out = np.zeros(len(rows_list))
    for i, rows in enumerate(rows_list):
        s = B
        ch = False
        for k, r in rows.items():
            if k in sidx:
                if not ch: s = B.copy(); ch = True
                s += np.log(r + DELTA) - eng.base_log[k]
        out[i] = lme(s) - b0 if ch else 0.0
    return out


def split_tabs(secs, rng):
    tabs = sorted({s['tab'] for s in secs})
    rng.shuffle(tabs)
    A = set(tabs[:len(tabs) // 2])
    a = [k for k, s in enumerate(secs) if s['tab'] in A]
    b = [k for k, s in enumerate(secs) if s['tab'] not in A]
    return a, b
