#!/usr/bin/env python3
"""la50: THE FARMING YEAR IS THE ANSWER KEY.

Hypothesis: the fixed order in which goods are written in a list (la20, la46) follows the
agricultural calendar: the order in which each crop comes in, is processed and stored.

Data: commodity logogram names (conventional labels GRA, VIN, OLE, OLIV ...) and the sign NI
written alone (the fig sign of the conventional labels), with their written order. No readings.

Units ('baskets'): runs of commodity items inside one tablet that are not interrupted by a
multi-sign word (an entry name or a total). Secondary unit: the first-occurrence list of a
whole tablet (la20 style). Each basket with >= 2 distinct items gives ordered pairs, weighted
1/(n-1) so long baskets do not dominate.

Calendars (month of harvest / intake into store; 0 = Jan, fractional = mid-month). Fixed BEFORE
any score was computed; sources in CAL_SRC.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la50_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
MONTHS = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()

# ------------------------------------------------------------------ calendars
# Aegean (Crete) calendar: harvest / intake midpoints
AEGEAN = {
    'GRA': 5.0,    # barley May, wheat June (FAO crop calendar: TN, MA harvest Jun; JO Apr-May; Hesiod WD 383-4, 571-81)
    'FIC': 7.7,    # figs Aug-Sep (FAO JO fig harvest Aug; Cretan practice)
    'VIN': 8.5,    # vintage late Aug - early Oct, peak Sep (Crete; Hesiod WD 609-14)
    'OLIV': 10.5,  # olive picking late Oct - Jan (Crete; FAO JO olive harvest from Sep-Oct)
    'OLE': 11.5,   # oil pressing follows picking, Nov - Jan
    'LANA': 4.0,   # shearing between spring equinox and solstice (Varro RR 2.11), Crete Apr-May
    'LIV': 2.0,    # lambing / kidding Dec - Mar, young counted in spring (Varro RR 2.2)
    'CROC': 10.0,  # Crocus cartwrightianus / sativus flowers Oct - Nov
    'ME': 5.5,     # main honey harvest at the rising of the Pleiades, May - Jun (Columella 9.15; Varro 3.16)
    'SA': 5.5,     # flax pulled May - Jun (Pliny NH 19.16-17)
}
# Northern Europe (traditional English / north German year; medieval 'labours of the months',
# e.g. Walter of Henley; crops that do not grow there mapped to the local analogue).
NORTH = {
    'GRA': 7.5,    # harvest Aug
    'FIC': 8.5,    # orchard fruit Sep (no figs)
    'VIN': 9.5,    # Rhine / Mosel vintage Oct
    'OLIV': 7.5,   # oilseed (rape, linseed) Aug (no olives)
    'OLE': 9.0,    # seed-oil pressing autumn
    'LANA': 5.5,   # shearing June
    'LIV': 10.5,   # Martinmas slaughter Nov
    'CROC': 9.5,   # Saffron Walden harvest Oct
    'ME': 7.5,     # honey Aug
    'SA': 7.5,     # flax pulled Aug
}
CAL_SRC = ('FAO crop calendar API (api-cropcalendar.apps.fao.org: TN, MA, JO; wheat/barley Apr-Jun, '
           'fig Aug, olive Sep-Oct start); Hesiod Works and Days 383-4, 571-81, 609-14; Varro RR 2.2, 2.11, 3.16; '
           'Columella 9.15; Pliny NH 19.16-17; Cretan olive (explorecrete.com) and grape (crete.direct) harvests; '
           'northern: medieval labours of the months / Walter of Henley')


# ------------------------------------------------------------------ classes
def cls_la(v):
    v = v.strip('*[]')
    b = v.split('+')[0]
    if b in ('GRA',) or v.startswith('GRA'): return 'GRA'
    if b.startswith('VIN'): return 'VIN'
    if b.startswith('OLIV'): return 'OLIV'
    if b.startswith('OLE'): return 'OLE'
    if b in ('NI',): return 'FIC'
    if b in ('CAP', 'OVIS', 'SUS', 'BOS'): return 'LIV'
    if b in ('VIR', 'MUL', 'TELA', 'HIDE'): return None      # persons / cloth / hides: not crops
    return b                                                  # unidentified: kept under its own name


def cls_lb(v):
    v = v.strip('*[]')
    b = v.split('+')[0].split('±')[0]
    if b.startswith(('GRA', 'HORD', 'FAR')): return 'GRA'
    if b == 'VIN': return 'VIN'
    if b == 'OLIV': return 'OLIV'
    if b == 'OLE': return 'OLE'
    if b == 'NI': return 'FIC'
    if b in ('OVIS', 'CAP', 'SUS', 'BOS'): return 'LIV'
    if b == 'LANA': return 'LANA'
    if b == 'CROC': return 'CROC'
    if b == 'ME': return 'ME'
    if b == 'SA': return 'SA'
    if b == 'CYP': return 'CYP'
    if b in ('AROM', 'KO', 'PE', 'TU'): return b
    return None


def _baskets_from_items(seq):
    """seq: list of items or '|' breaks -> list of baskets (distinct, in order, len >= 2)"""
    out, cur = [], []
    for x in seq + ['|']:
        if x == '|':
            if len(cur) >= 2: out.append(cur)
            cur = []
        elif x not in cur:
            cur.append(x)
    return out


SUPPORTS = {'Tablet', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar'}


def base_id(i):
    return re.sub(r'(?<=\d)[a-e]$', '', i)


def load_la():
    """returns list of docs {id, site, seq (items with '|' breaks), flat (first-occurrence list)}"""
    c = json.load(open(os.path.join(DATA, 'corpus.json')))
    docs = {}
    for r in c:
        if r['support'] not in SUPPORTS: continue
        b = base_id(r['id'])
        d = docs.setdefault(b, {'id': b, 'site': r['site'], 'seq': []})
        for t in r['tokens']:
            if t['t'] == 'logo':
                k = cls_la(t['v'])
                if k: d['seq'].append(k)
            elif t['t'] == 'word':
                if t['s'] == ['NI']: d['seq'].append('FIC')
                elif len(t['s']) >= 2: d['seq'].append('|')
        d['seq'].append('|')    # sides a / b are separate baskets
    out = []
    for d in docs.values():
        fl = []
        for x in d['seq']:
            if x != '|' and x not in fl: fl.append(x)
        d['flat'] = fl
        d['baskets'] = _baskets_from_items(d['seq'])
        out.append(d)
    return out


def load_lb(sites=('KN', 'PY')):
    from la22_lb import parse_doc
    out = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        x = json.loads(l)
        if not x.get('content') or not x.get('heading'): continue
        s = x['heading'].split()[0]
        if s not in sites: continue
        seq = []
        for t in parse_doc(x['content']):
            if t['t'] == 'L' and len(t['c']) > 1:
                k = cls_lb(t['c'])
                if k: seq.append(k)
            elif t['t'] == 'w' and len(t['c']) >= 2:
                seq.append('|')
        fl = []
        for y in seq:
            if y != '|' and y not in fl: fl.append(y)
        out.append({'id': x['heading'], 'site': s, 'seq': seq, 'flat': fl,
                    'baskets': _baskets_from_items(seq)})
    return out


def units(docs, unit='basket'):
    """list of (doc index, ordered item list)"""
    if unit == 'basket':
        return [(i, b) for i, d in enumerate(docs) for b in d['baskets']]
    return [(i, d['flat']) for i, d in enumerate(docs) if len(d['flat']) >= 2]


# ------------------------------------------------------------------ scoring
def pairs(us, items):
    """weighted ordered pairs among `items` (a written before b). returns a, b, w, doc"""
    idx = {x: i for i, x in enumerate(items)}
    A, B, W, G = [], [], [], []
    for di, o in us:
        o2 = [x for x in o if x in idx]
        n = len(o2)
        if n < 2: continue
        w = 1.0 / (n - 1)
        for p in range(n):
            for q in range(p + 1, n):
                A.append(idx[o2[p]]); B.append(idx[o2[q]]); W.append(w); G.append(di)
    return np.array(A, int), np.array(B, int), np.array(W), np.array(G, int)


STARTS = np.arange(24) / 2.0     # candidate administrative year starts (half-months)


def conc(M, a, b, w, starts=STARTS, return_start=False):
    """M[n, k] item months for n calendars. Concordance (weighted share of pairs written in
    calendar order counted from year start s), maximised over s. Returns [n]."""
    M = np.atleast_2d(M)
    W = w.sum()
    best = np.full(M.shape[0], -1.0); bs = np.zeros(M.shape[0])
    for s in starts:
        K = (M - s) % 12.0
        ka, kb = K[:, a], K[:, b]
        v = ((ka < kb) * 1.0 + (ka == kb) * 0.5) @ w / W
        upd = v > best
        best[upd] = v[upd]; bs[upd] = s
    return (best, bs) if return_start else best


def circ_d(x, y):
    d = np.abs(x - y) % 12.0
    return np.minimum(d, 12.0 - d)


def cal_vec(cal, items):
    return np.array([cal[x] for x in items], float)


def fmt(p):
    return '%.3g' % p


def log(path, s):
    print(s, flush=True)
    with open(path, 'a') as f: f.write(s + '\n')
