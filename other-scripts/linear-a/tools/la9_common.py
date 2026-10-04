#!/usr/bin/env python3
"""LA-9 'read the mistakes': shared data for the scribal-error fingerprint tests.

Every dataset is a list of sections. A section is a list of entries plus one written total.
A quantity is (integer part, {letter: count}). Letters are sub-unit / fraction signs whose
values are HIDDEN from the machinery (for the controls the truth is known and kept aside).

  load_la()   Linear A KU-RO sections (corpus.json, same cut rule as totals_test.py).
              Letters = fraction signs (J, E, JE, ...). No Linear B sound values are used.
  load_lb()   Linear B to-so / to-sa sections (DAMOS). Letters = capacity sub-units:
              dry T, Vd, Zd and liquid S, Vl, Zl. Truth 1/10, 1/60, 1/240 ; 1/3, 1/18, 1/72.
  load_ur3()  Neo-Sumerian (Ur III-style) szunigin grain totals from the CDLI ATF dump
              (cdliatf_unblocked.atf, not committed; path in CDLI_ATF). Letters barig, ban2,
              sila3. Truth 1/5, 1/30, 1/300 gur.

Values are kept as exact integers in units of 1/U (U = 14400, the lcm of every grid denominator).
"""
import json, os, re, unicodedata
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
OUT = os.path.join(D, 'la9')
os.makedirs(OUT, exist_ok=True)
CDLI_ATF = os.environ.get('CDLI_ATF', '/tmp/cdliatf_unblocked.atf')   # set CDLI_ATF to the downloaded dump

U = 14400
DEC_PLACES = [1, 10, 100, 1000, 10000]
SEX_PLACES = [1, 10, 60, 600, 3600, 36000]

# candidate value grid (fractions of the main unit), same for every dataset
_GRID = set()
for k in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 48, 50, 60, 64, 72, 80, 90,
          96, 100, 120, 144, 150, 180, 200, 240, 300, 360]:
    _GRID.add((1, k))
for n, k in [(2, 3), (3, 4), (3, 8), (5, 8), (3, 16), (5, 16), (7, 16), (2, 5), (3, 5), (3, 10),
             (3, 20), (5, 6), (7, 8), (5, 12), (7, 12)]:
    _GRID.add((n, k))
GRID_FR = sorted(_GRID, key=lambda t: t[0] / t[1])
GRID = np.array([U * n // k for n, k in GRID_FR], dtype=np.int64)
GRID_LABEL = ['%d/%d' % t for t in GRID_FR]
assert all(U * n % k == 0 for n, k in GRID_FR)


def gi(label):
    return GRID_LABEL.index(label)


# --------------------------------------------------------------------------- Linear A
def load_la():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    MARK = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}
    w = lambda t: '-'.join(t['s']) if t['t'] == 'word' else None
    secs = []
    for ins in C:
        T = ins['tokens']; last = 0
        for i, t in enumerate(T):
            if w(t) in MARK:
                if w(t) == 'KU-RO':
                    tot = None
                    for j in range(i + 1, min(i + 4, len(T))):
                        if T[j]['t'] == 'num': tot = T[j]; break
                        if T[j]['t'] == 'word': break
                    ent = [x for x in T[last:i] if x['t'] == 'num']
                    if tot is not None and ent:
                        q = lambda x: (int(x['v']), dict(Counter(x['frac'])))
                        secs.append({'id': ins['id'], 'entries': [q(e) for e in ent], 'total': q(tot),
                                     'places': DEC_PLACES})
                last = i + 1
    return secs


# --------------------------------------------------------------------------- Linear B
LB_DRY = {'GRA', 'HORD', 'FAR', 'NI', 'OLIV', 'CYP', 'AROM', 'CROC', 'KA±PO', 'PYC', 'SA', 'KO', 'MA',
          'KU', 'SE', 'PO', 'ME', 'RI'}
LB_LIQ = {'OLE', 'VIN', 'ME±RI'}
LB_OTHER = {'VIR', 'MUL', 'OVIS', 'CAP', 'SUS', 'BOS', 'TELA', 'LANA', 'AES', 'AUR', 'TU±RO2', 'A±RE±PA',
            'CORN', 'ARB', 'OVISm', 'OVISf', 'CAPm', 'CAPf', 'SUSm', 'SUSf', 'BOSm', 'BOSf', 'EQU', 'ROTA',
            'BIG', 'CUR', 'GAL', 'HAS', 'JAC', 'SAG', 'TUN'}
LB_TRUTH = {'T': (1, 10), 'Vd': (1, 60), 'Zd': (1, 240), 'S': (1, 3), 'Vl': (1, 18), 'Zl': (1, 72)}


def _clean(tok):
    t = unicodedata.normalize('NFD', tok)
    t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
    dam = ('[' in t) or (']' in t)
    return t.strip('[]?'), dam


def load_lb():
    secs = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        cont = d.get('content') or ''
        if not re.search(r'\bto-s[oa]\b', cont): continue
        lines = [l for l in cont.split('\n') if l.strip()]
        quants = []          # (line_no, commodity, int, letters, damaged, is_total)
        cur = None
        for ln, l in enumerate(lines):
            raw = l.split()
            toks = [_clean(t) for t in raw]
            is_tot_line = any(re.match(r'^to-s[oa]', t) for t, _ in toks)
            seen_toso = False
            i = 0
            while i < len(toks):
                t, dam = toks[i]
                if re.match(r'^to-s[oa]', t): seen_toso = True
                base = t.split('+')[0]
                is_com = base in LB_DRY or base in LB_LIQ or base in LB_OTHER
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
                        fam = 'L' if cur in LB_LIQ else ('D' if cur in LB_DRY else None)
                        if fam is None: lets['?'] += n
                        elif u == 'T': lets['T' if fam == 'D' else '?'] += n
                        elif u == 'S': lets['S' if fam == 'L' else '?'] += n
                        else: lets[u + ('d' if fam == 'D' else 'l')] += n
                        has = True; i += 2
                    if has:
                        quants.append((ln, cur, ival, dict(lets), dam, is_tot_line and seen_toso))
                    elif not is_com:
                        i += 1
                    continue
                i += 1
        # sections
        start = 0
        for k, q in enumerate(quants):
            if q[5]:
                if k > 0 and quants[k - 1][5] and quants[k - 1][0] == q[0]:
                    continue      # only the first quantity on a to-so line is the total
                ent = [e for e in quants[start:k] if not e[5] and e[1] == q[1]]
                if ent and '?' not in q[3] and all('?' not in e[3] for e in ent):
                    secs.append({'id': d['heading'].split('(')[0].strip() + ' ' + str(q[1]),
                                 'entries': [(e[2], e[3]) for e in ent], 'total': (q[2], q[3]),
                                 'damaged': any(e[4] for e in ent) or q[4], 'places': DEC_PLACES})
                start = k + 1
    return secs


# --------------------------------------------------------------------------- Ur III (CDLI)
UR_NUM = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, "szar'u": 36000}
UR_TRUTH = {'barig': (1, 5), 'ban2': (1, 30), 'sila3': (1, 300)}
_numtok = re.compile(r"^(\d+(?:/\d+)?)\(([a-z0-9']+)(?:@[a-z])?\)$")
TOT_WORDS = {'szunigin', 'szunigin2', 'szu-nigin2', 'szu-nigin'}


def _ur_line_tokens(text):
    return [t.strip('#!?*_<>') for t in text.split()]


def parse_ur_quantity(toks):
    """Return (gur, {'barig':n,'ban2':n,'sila3':n}, commodity) or 'bad' or None (no capacity)."""
    gur = 0; lets = Counter(); pending = 0; sign = 1; cap = False; com = None; closed = False
    for t in toks:
        m = _numtok.match(t) or (re.match(r'^(\d+)$', t) and re.match(r'^(\d+)()$', t))
        if m:
            if closed: return 'bad'
            num, unit = m.group(1), m.group(2)
            if '/' in num: return 'bad'
            n = int(num) * sign
            if unit in ('barig', 'ban2'):
                gur += pending; pending = 0; lets[unit] += n; cap = True
            elif unit == '' or unit in UR_NUM:
                pending += n * UR_NUM.get(unit, 1)
            else:
                return 'bad'
            continue
        sign = 1
        if t == 'la2': sign = -1; continue
        if t == 'sila3':
            lets['sila3'] += pending; pending = 0; cap = True; continue
        if t == 'gur':
            gur += pending; pending = 0; cap = True; continue
        if t in ('gin2', 'ma-na', 'sar', 'iku', 'GAN2', 'bur3', 'esze3'): return 'bad'
        if (gur or lets or pending) and com is None and not t.startswith('{'):
            com = t
        if lets or gur or pending: closed = True
    if not cap: return None
    if pending: return 'bad'
    if any(v < 0 for v in lets.values()) or gur < 0: return 'bad'
    return gur, {k: v for k, v in lets.items() if v}, com


def load_ur3(max_texts=None):
    secs = []
    if not os.path.exists(CDLI_ATF): return secs
    def flush(pid, lines):
        # lines: list of (kind, payload) kind in {'q','tot','bad','dam','txt'}
        sect = []; block = []; bad = False
        def close_block():
            nonlocal sect, block, bad
            if block:
                if not bad:
                    if len(block) == 1:
                        ent = [q for q in sect]
                        tots = [(block[0], ent)]
                    else:
                        tots = [(b, [q for q in sect if q[2] == b[2]]) for b in block]
                    for b, ent in tots:
                        if len(ent) >= 2:
                            secs.append({'id': pid, 'entries': [(q[0], q[1]) for q in ent],
                                         'total': (b[0], b[1]), 'places': SEX_PLACES})
                sect = []; block = []; bad = False
        for kind, pay in lines:
            if kind == 'tot':
                if pay == 'bad': bad = True
                else: block.append(pay)
                continue
            close_block()
            if kind == 'q': sect.append(pay)
            elif kind in ('bad', 'dam'): bad = True
        close_block()
    pid = None; lines = []; n = 0
    for raw in open(CDLI_ATF, encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            if pid and lines: flush(pid, lines)
            pid = raw.split()[0][1:]; lines = []; n += 1
            if max_texts and n > max_texts: break
            continue
        s = raw.strip()
        if not s: continue
        if s.startswith('$'):
            if 'broken' in s or 'missing' in s or 'traces' in s: lines.append(('dam', None))
            continue
        m = re.match(r"^\d+'?\.\s*(.*)$", s)
        if not m: continue
        body = m.group(1)
        if '[' in body or ']' in body or '...' in body or re.search(r'(^|\s)x(\s|$|\()', body) \
                or re.search(r'\bn\(', body):
            toks = _ur_line_tokens(body)
            if toks and toks[0] in TOT_WORDS: lines.append(('tot', 'bad'))
            else:
                lines.append(('dam', None))
            continue
        toks = _ur_line_tokens(body)
        if toks and toks[0] in TOT_WORDS:
            q = parse_ur_quantity(toks[1:])
            if q is None: continue          # count total of another thing; ignore
            lines.append(('tot', q))
        else:
            q = parse_ur_quantity(toks)
            if q == 'bad': lines.append(('bad', None))
            elif q is not None: lines.append(('q', q))
    if pid and lines: flush(pid, lines)
    return secs


# --------------------------------------------------------------------------- packing
def pack(secs, letters=None):
    """Turn sections into integer arrays. Returns (letters, packed list)."""
    if letters is None:
        letters = sorted({l for s in secs for q in s['entries'] + [s['total']] for l in q[1]})
    L = {l: i for i, l in enumerate(letters)}
    P = []
    for s in secs:
        A = np.array([q[0] for q in s['entries']], dtype=np.int64)
        Cm = np.zeros((len(s['entries']), len(letters)), dtype=np.int64)
        for i, q in enumerate(s['entries']):
            for l, c in q[1].items(): Cm[i, L[l]] += c
        ct = np.zeros(len(letters), dtype=np.int64)
        for l, c in s['total'][1].items(): ct[L[l]] += c
        P.append({'id': s['id'], 'a': A, 'C': Cm, 'ta': int(s['total'][0]), 'ct': ct,
                  'places': np.array(s['places'], dtype=np.int64),
                  'letters_used': sorted(set(np.nonzero(Cm.sum(0) + ct)[0].tolist()))})
    return letters, P


def truth_vector(letters, truth):
    return np.array([gi('%d/%d' % truth[l]) for l in letters])
