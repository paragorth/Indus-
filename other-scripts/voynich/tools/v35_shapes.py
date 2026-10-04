"""v35 hand stroke decompositions, written from the drawn glyphs ONLY (montages of the Copiale font from Knight et al.
2011, FreeMonoTengwar, Noto Sans Shavian; Unicode names for the layout of Canadian syllabics), before any v35
statistic was computed.  Primitive vocabulary for Roman-like letters reuses v25's book-hand primitives (v25_shapes.LATIN).
Cherokee and Deseret get no hand decomposition (no stroke system visible; image descriptors only).
"""
import os, sys, unicodedata, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v25_shapes as S
from v25_shapes import _m

LT = S.LATIN

# ------------------------------------------------------------------ Copiale (transcription names)
COPIALE = {
    'a': LT['a'], 'b': LT['b'], 'c': LT['c'], 'd': LT['d'], 'e': LT['e'], 'g': LT['g'], 'h': LT['h'], 'i': LT['i'],
    'k': LT['k'], 'l': LT['l'], 'm': LT['m'], 'n': LT['n'], 'o': LT['o'], 'p': LT['p'], 'q': LT['q'], 'r': LT['r'],
    's': LT['s'], 't': LT['t'], 'u': LT['u'], 'x': LT['x'], 'y': LT['y'], 'z': LT['z'],
    'f': _m(TALL=1, DESCS=1, XBAR=1, HOOK=1, BOWL=1),
    'j': _m(MINIM=1, DOT=1, DESCS=1),
    'v': _m(CUP=1, HOOK=1), 'w': _m(CUP=2, HOOK=1),
    # circumflexed, dotted, underlined, umlauted variants
    'ah': _m(LT['a'], CIRC=1), 'eh': _m(LT['e'], CIRC=1), 'ih': _m(MINIM=1, CIRC=1), 'oh': _m(LT['o'], CIRC=1),
    'uh': _m(LT['u'], CIRC=1),
    'c.': _m(LT['c'], DOT=1), 'h.': _m(LT['h'], DOT=1), 'm.': _m(LT['m'], DOT=1), 'n.': _m(LT['n'], DOT=1),
    'o.': _m(LT['o'], DOT=1), 'p.': _m(LT['p'], DOT=1), 'r.': _m(LT['r'], DOT=1), 's.': _m(LT['s'], DOT=1),
    'x.': _m(LT['x'], DOT=1), 'y..': _m(LT['y'], DOT=2),
    'mu': _m(LT['m'], UNDER=1), 'nu': _m(LT['n'], UNDER=1), 'ru': _m(LT['r'], UNDER=1), 'uu': _m(LT['u'], UNDER=1),
    'hd': _m(TALL=1, ARCH=1, XBAR=1, HOOK=1),
    'o..': _m(BOWL=2, DOT=1),
    # symbols
    'del': _m(BOWL=1, CURVE=1, TALL=1), 'ds': _m(BOWL=1, CURVE=1, HOOK=1),
    'tri': _m(TRI=1), 'tri..': _m(TRI=1, DOT=2), 'star': _m(TRI=1, DIAG=2, TALL=1),
    'gam': _m(DIAG=2, BOWL=1, TALL=1), 'iot': _m(MINIM=1, HOOK=1), 'lam': _m(DIAG=2), 'pi': _m(XBAR=1, MINIM=2),
    'gat': _m(XBAR=1, MINIM=2, BIG=1), 'arr': _m(DIAG=1, HEAD=1), 'bas': _m(CURVE=1, HOOK=1),
    'car': _m(TALL=1, XBAR=2, DIAG=1), 'ki': _m(XBAR=2, DIAG=1), 'plus': _m(XBAR=1, MINIM=1),
    'cross': _m(XBAR=1, TALL=1, DESCS=1), 'fem': _m(BOWL=1, XBAR=1, DESCS=1), 'mal': _m(BOWL=1, XBAR=1, TALL=1),
    'ft': _m(TALL=1, DESCS=1, HOOK=1, XBAR=2, MINIM=2), 'no': _m(BOX=1, DIAG=1), 'sqp': _m(DESCS=1, BOX=1),
    'zzz': _m(DIAG=2, XBAR=2), 'zs': _m(DIAG=1, XBAR=2, HOOK=1), 'pipe': _m(TALL=1, DESCS=1, XBAR=1, HOOK=1),
    'longs': _m(DIAG=1, TALL=1, DESCS=1), 'grr': _m(TALL=2, XBAR=1, ARCH=1), 'grl': _m(TALL=2, XBAR=2, HOOK=1),
    'grc': _m(TALL=2, XBAR=3), 'hk': _m(TALL=1, HOOK=1), 'lip': _m(BOWL=1, XBAR=1), 'sqi': _m(TALL=1, XBAR=1, HOOK=1),
    'nee': _m(DIAG=2, BOWL=1, XBAR=1), ':': _m(DOT=2), '.': _m(DOT=1), '...': _m(DOT=3),
    'ni': _m(XBAR=2), 'gs': _m(BOWL=1, DESCS=1, HOOK=1, CURVE=1), 'bigx': _m(DIAG=2, XBAR=2),
    'bar': _m(TALL=1, DESCS=1), 'smil': _m(CURVE=1, DOT=2, LEFTB=1), 'smir': _m(CURVE=1, DOT=2, RIGHTB=1),
    'three': _m(CURVE=2), 'ns': _m(MINIM=1, ARCH=1, DESCS=1), 'toe': _m(XBAR=1, CUP=2), 'inf': _m(BOWL=2),
}

# ------------------------------------------------------------------ Tengwar (FreeMonoTengwar glyph names)
_GR = {1: dict(STEM_D=1, BOW=1), 2: dict(STEM_D=1, BOW=2), 3: dict(STEM_U=1, BOW=1), 4: dict(STEM_U=1, BOW=2),
       5: dict(STEM_S=1, BOW=2), 6: dict(STEM_S=1, BOW=1)}
_SE = {1: dict(RIGHT=1), 2: dict(RIGHT=1, CLOSED=1), 3: dict(LEFT=1), 4: dict(LEFT=1, CLOSED=1)}
_TABLE = [['tinco', 'parma', 'calma', 'quesse'], ['ando', 'umbar', 'anga', 'ungwe'], ['thule', 'formen', 'harma', 'hwesta'],
          ['anto', 'ampa', 'anca', 'unque'], ['numen', 'malta', 'noldo', 'nwalme'], ['ore', 'vala', 'anna', 'vilya']]
TENGWAR = {}
for gi, row in enumerate(_TABLE, 1):
    for si, n in enumerate(row, 1):
        TENGWAR[n] = _m(_GR[gi], _SE[si])
TENGWAR.update({
    'romen': _m(HOOK=1, DIAG=1, CURVE=1), 'lambe': _m(CURVE=1, HOOK=1), 'silme': _m(CURVE=1, LOOP=1),
    'esse': _m(CURVE=2), 'hyarmen': _m(STEM_U=1, DIAG=2), 'hwestaS': _m(STEM_U=1, BOW=1, LEFT=1, HOOK=1),
    'shortCarrier': _m(STEM_S=1), 'longCarrier': _m(STEM_D=1, HOOK=1),
    'tehtaA': _m(MARK=1, DOT=3), 'tehtaE': _m(MARK=1, ACUTE=1), 'tehtaI': _m(MARK=1, DOT=1),
    'tehtaO': _m(MARK=1, CURL=1, RIGHT=1), 'tehtaU': _m(MARK=1, CURL=1, LEFT=1), 'tehtaY': _m(MARK=1, DOT=2),
    'tehtaBar': _m(MARK=1, UNDERBAR=1),
})

# ------------------------------------------------------------------ Shavian (code point -> primitives, from the Noto montage)
_SHV = {
    0x50: _m(VLINE=1, HOOK=1, TALL=1), 0x51: _m(VLINE=1, FLAG=1, TALL=1), 0x52: _m(ARC=1, HOOK=1, TALL=1),
    0x53: _m(VLINE=1, HOOK=1, TALL=1), 0x54: _m(LOOP=1, ARC=1, TALL=1), 0x55: _m(SCURVE=1, TALL=1),
    0x56: _m(ARC=1, TALL=1), 0x57: _m(ARC=1, XBAR=1, TALL=1), 0x58: _m(DIAG=1, TALL=1), 0x59: _m(LOOP=1, DIAG=1, TALL=1),
    0x5A: _m(VLINE=1, HOOK=1, DEEP=1), 0x5B: _m(VLINE=1, FLAG=1, DEEP=1), 0x5C: _m(ARC=1, HOOK=1, DEEP=1),
    0x5D: _m(VLINE=1, HOOK=1, DEEP=1), 0x5E: _m(LOOP=1, ARC=1, DEEP=1), 0x5F: _m(SCURVE=1, DEEP=1),
    0x60: _m(ARC=1, DEEP=1), 0x61: _m(ARC=1, XBAR=1, DEEP=1), 0x62: _m(DIAG=1, DEEP=1), 0x63: _m(LOOP=1, DIAG=1, DEEP=1),
    0x64: _m(ARC=1, SHORT=1), 0x65: _m(SCURVE=1, SHORT=1), 0x66: _m(VLINE=1, SHORT=1), 0x67: _m(VLINE=1, HOOK=1, SHORT=1),
    0x68: _m(VLINE=1, HOOK=1, SHORT=1), 0x69: _m(VLINE=1, HOOK=1, SHORT=1), 0x6A: _m(VLINE=1, HOOK=1, SHORT=1),
    0x6B: _m(DIAG=2, SHORT=1), 0x6C: _m(DIAG=2, HOOK=1, SHORT=1), 0x6D: _m(SCURVE=1, HOOK=1, SHORT=1),
    0x6E: _m(ARC=1, SHORT=1), 0x6F: _m(ARC=1, DIAG=1, SHORT=1), 0x70: _m(VLINE=1, FLAG=1, SHORT=1),
    0x71: _m(ARC=1, XBAR=1, SHORT=1), 0x72: _m(ARC=1, HOOK=1, SHORT=1), 0x73: _m(XBAR=1, DIAG=1, SHORT=1),
    0x74: _m(LOOP=1, SHORT=1), 0x75: _m(DIAG=2, SHORT=1), 0x76: _m(DIAG=2, HOOK=1, SHORT=1),
    0x77: _m(SCURVE=1, HOOK=1, SHORT=1),
    0x78: _m(LOOP=1, ARC=1, HOOK=1, SHORT=1), 0x79: _m(LOOP=1, ARC=1, HOOK=1, SHORT=1), 0x7A: _m(LOOP=1, ARC=1, SHORT=1),
    0x7B: _m(LOOP=1, ARC=1, HOOK=1, SHORT=1), 0x7C: _m(ARC=2, SHORT=1), 0x7D: _m(ARC=2, VLINE=1, SHORT=1),
    0x7E: _m(VLINE=1, HOOK=1, ARC=1, SHORT=1), 0x7F: _m(DIAG=2, VLINE=1, SHORT=1)}
SHAVIAN = {chr(0x10400 + k): v for k, v in _SHV.items()}


# ------------------------------------------------------------------ Canadian syllabics (series shape + orientation)
def ucas(ch):
    n = unicodedata.name(ch, '').replace('CANADIAN SYLLABICS ', '')
    n = re.sub(r'^(WEST-CREE|NASKAPI|Y-CREE|EASTERN|WOODS-CREE|SAYISI|CARRIER|NUNAVIK|OJIBWAY|MEDIAL|BLACKFOOT|SOUTH-SLAVEY) ', '', n)
    if n == 'FINAL DOUBLE SHORT VERTICAL STROKES':
        return _m(VSTROKE=2, SMALL=1)
    if n == 'FINAL RING':
        return _m(RING=1, SMALL=1)
    if n == 'FINAL MIDDLE DOT':
        return _m(WDOT=1)
    if n in ('FINAL PLUS', 'FINAL GRAVE', 'FINAL ACUTE', 'FINAL RIGHT HALF RING'):
        return _m(**{n.split()[-1].replace(' ', ''): 1}, SMALL=1)
    m = re.match(r'^(SH|TH|KW|[PTKCMNSYLRFWH])?(W)?(AAI|AA|II|OO|AI|A|I|O|E)?$', n)
    if not m:
        return None
    cons, w, v = m.groups()
    out = {}
    if cons == 'KW' and v is None:
        out.update(SER_K=1, SMALL=1, WDOT=1)
        return out
    if cons == 'W':
        cons, w = None, 'W'
    out[f'SER_{cons or "V"}'] = 1
    if v is None:
        out['SMALL'] = 1
        out['ORI_A'] = 1  # finals are the a-form drawn small
        return out
    if w:
        out['WDOT'] = 1
    if v in ('AA', 'II', 'OO'):
        out['LDOT'] = 1
    out[f'ORI_{v[0]}'] = 1
    return out


def hand(name, alph):
    if name.startswith('copiale'):
        from v35_lib import COP_KEY
        return {g: COPIALE[COP_KEY[g]] for g in alph if COP_KEY.get(g) in COPIALE}
    if name == 'tengwar':
        from v35_lib import TW_NAME
        return {g: TENGWAR[TW_NAME[g]] for g in alph if TW_NAME.get(g) in TENGWAR}
    if name == 'shavian':
        return {g: SHAVIAN[g] for g in alph if g in SHAVIAN}
    if name == 'cree':
        return {g: ucas(g) for g in alph if ucas(g)}
    if name.startswith('voy'):
        return {g: S.VOYNICH[g] for g in alph if g in S.VOYNICH}
    return None
