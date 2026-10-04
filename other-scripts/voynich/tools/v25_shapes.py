"""v25 THE GLYPHS ARE BUILT FROM FEATURES -- stroke decompositions, fixed BEFORE any statistics were run.

Each glyph is a multiset (count vector) of visual stroke primitives, defined from the drawn letter
forms only (EVA glyph shapes for the Voynich; book-hand lowercase for Latin; lowercase Greek;
Hangul jamo as drawn in a syllable block).  No distributional fact was consulted in writing them.

Voynich primitives
  CURVE  open c-shaped stroke (e, a, the two halves of the ch bench, s)
  MINIM  short vertical stroke (i, a, n, r, m)
  LOOP   closed oval or loop (o, y, d, l, q, gallows loops, m/g tails)
  ASC    single tall ascender, not a gallows leg (d, l)
  DESC   tail below the line (y, g)
  LEGS   the two tall legs of a gallows (k t p f and the benched gallows)
  TLOOP  the second, t-type loop of a gallows (t p cth cph)
  WIDE   the broad extended top loop (p f cph cfh)
  BAR    the horizontal bench connecting two curves (ch sh and benched gallows)
  PLUME  the comma or flourish above (sh, s)
  FLAG   hook at the top of a minim (r)
  RTAIL  final tail curling to the right (n m g)
  CROSS  crossing diagonal stroke of the 4-shape (q)
  CURL   curved, back-bent ascender (d)
"""
from collections import Counter

V_PRIMS = ['CURVE', 'MINIM', 'LOOP', 'ASC', 'DESC', 'LEGS', 'TLOOP', 'WIDE', 'BAR', 'PLUME', 'FLAG', 'RTAIL', 'CROSS', 'CURL']
_gal = dict(LEGS=1, LOOP=1)
_bench = dict(CURVE=2, BAR=1)


def _m(*ds, **kw):
    c = Counter()
    for d in ds:
        c.update(d)
    c.update(kw)
    return dict(c)


VOYNICH = {
    'o': _m(LOOP=1),
    'e': _m(CURVE=1),
    'y': _m(LOOP=1, DESC=1),
    'a': _m(CURVE=1, MINIM=1),
    'd': _m(LOOP=1, ASC=1, CURL=1),
    'i': _m(MINIM=1),
    'ch': _m(_bench),
    'l': _m(LOOP=1, ASC=1),
    'k': _m(_gal),
    'r': _m(MINIM=1, FLAG=1),
    'n': _m(MINIM=1, RTAIL=1),
    'q': _m(LOOP=1, CROSS=1),
    't': _m(_gal, TLOOP=1),
    'sh': _m(_bench, PLUME=1),
    's': _m(CURVE=1, PLUME=1),
    'p': _m(_gal, TLOOP=1, WIDE=1),
    'm': _m(MINIM=1, RTAIL=1, LOOP=1),
    'cth': _m(_bench, _gal, TLOOP=1),
    'ckh': _m(_bench, _gal),
    'f': _m(_gal, WIDE=1),
    'cph': _m(_bench, _gal, TLOOP=1, WIDE=1),
    'g': _m(LOOP=1, DESC=1, RTAIL=1),
    'cfh': _m(_bench, _gal, WIDE=1),
}

# Latin book-hand lowercase (u/v merged as u, j as i)
L_PRIMS = ['BOWL', 'CURVE', 'TALL', 'DESCS', 'MINIM', 'ARCH', 'XBAR', 'DIAG', 'DOT', 'HOOK', 'CUP', 'SCURVE', 'RIGHTB', 'LEFTB']
LATIN = {
    'a': _m(BOWL=1, MINIM=1, LEFTB=1), 'b': _m(TALL=1, BOWL=1, RIGHTB=1), 'c': _m(CURVE=1),
    'd': _m(BOWL=1, TALL=1, LEFTB=1), 'e': _m(CURVE=1, XBAR=1), 'f': _m(TALL=1, HOOK=1, XBAR=1),
    'g': _m(BOWL=1, DESCS=1, HOOK=1, LEFTB=1), 'h': _m(TALL=1, ARCH=1), 'i': _m(MINIM=1, DOT=1),
    'k': _m(TALL=1, DIAG=2), 'l': _m(TALL=1), 'm': _m(MINIM=1, ARCH=2), 'n': _m(MINIM=1, ARCH=1),
    'o': _m(BOWL=1), 'p': _m(DESCS=1, BOWL=1, RIGHTB=1), 'q': _m(BOWL=1, DESCS=1, LEFTB=1),
    'r': _m(MINIM=1, HOOK=1), 's': _m(SCURVE=1), 't': _m(MINIM=1, XBAR=1, HOOK=1),
    'u': _m(MINIM=1, CUP=1), 'x': _m(DIAG=2), 'y': _m(DIAG=2, DESCS=1), 'z': _m(DIAG=1, XBAR=2),
}

# Greek lowercase (accents stripped)
GREEK = {
    'α': _m(BOWL=1, HOOK=1), 'β': _m(DESCS=1, TALL=1, BOWL=2), 'γ': _m(DIAG=2, DESCS=1),
    'δ': _m(BOWL=1, TALL=1, CURVE=1), 'ε': _m(CURVE=2), 'ζ': _m(TALL=1, SCURVE=1, DESCS=1),
    'η': _m(MINIM=1, ARCH=1, DESCS=1), 'θ': _m(BOWL=1, XBAR=1, TALL=1), 'ι': _m(MINIM=1),
    'κ': _m(MINIM=1, DIAG=2), 'λ': _m(DIAG=2, TALL=1), 'μ': _m(MINIM=2, CUP=1, DESCS=1),
    'ν': _m(DIAG=2), 'ξ': _m(SCURVE=2, TALL=1, DESCS=1), 'ο': _m(BOWL=1), 'π': _m(XBAR=1, MINIM=2),
    'ρ': _m(BOWL=1, DESCS=1), 'σ': _m(BOWL=1, XBAR=1), 'ς': _m(CURVE=1, DESCS=1, HOOK=1),
    'τ': _m(XBAR=1, MINIM=1), 'υ': _m(CUP=1), 'φ': _m(BOWL=1, TALL=1, DESCS=1),
    'χ': _m(DIAG=2, DESCS=1), 'ψ': _m(CUP=1, TALL=1, DESCS=1), 'ω': _m(CUP=2),
}

# Hangul jamo (compatibility jamo), strokes as drawn
_K = dict(
    G=dict(C7=1), N=dict(CL=1), D=dict(CL=1, TOP=1), R=dict(ZIG=1), M=dict(BOX=1), B=dict(LV=2, MIDH=1, BOTH=1),
    S=dict(WEDGE=1), O=dict(CIRC=1), J=dict(WEDGE=1, TOP=1), C=dict(WEDGE=1, TOP=1, TICK=1), K=dict(C7=1, MIDH=1),
    T=dict(CL=1, TOP=1, MIDH=1), P=dict(TOP=1, BOTH=1, LV=2), H=dict(CIRC=1, TOP=1, TICK=1),
)
HANGUL = {
    'ㄱ': _K['G'], 'ㄴ': _K['N'], 'ㄷ': _K['D'], 'ㄹ': _K['R'], 'ㅁ': _K['M'], 'ㅂ': _K['B'], 'ㅅ': _K['S'],
    'ㅇ': _K['O'], 'ㅈ': _K['J'], 'ㅊ': _K['C'], 'ㅋ': _K['K'], 'ㅌ': _K['T'], 'ㅍ': _K['P'], 'ㅎ': _K['H'],
    'ㄲ': _m(_K['G'], _K['G'], DBL=1), 'ㄸ': _m(_K['D'], _K['D'], DBL=1), 'ㅃ': _m(_K['B'], _K['B'], DBL=1),
    'ㅆ': _m(_K['S'], _K['S'], DBL=1), 'ㅉ': _m(_K['J'], _K['J'], DBL=1),
    'ㅏ': _m(VL=1, TR=1), 'ㅓ': _m(VL=1, TL=1), 'ㅗ': _m(HL=1, TU=1), 'ㅜ': _m(HL=1, TD=1),
    'ㅑ': _m(VL=1, TR=2), 'ㅕ': _m(VL=1, TL=2), 'ㅛ': _m(HL=1, TU=2), 'ㅠ': _m(HL=1, TD=2),
    'ㅡ': _m(HL=1), 'ㅣ': _m(VL=1),
    'ㅐ': _m(VL=2, TR=1), 'ㅔ': _m(VL=2, TL=1), 'ㅒ': _m(VL=2, TR=2), 'ㅖ': _m(VL=2, TL=2),
    'ㅘ': _m(HL=1, TU=1, VL=1, TR=1), 'ㅙ': _m(HL=1, TU=1, VL=2, TR=1), 'ㅚ': _m(HL=1, TU=1, VL=1),
    'ㅝ': _m(HL=1, TD=1, VL=1, TL=1), 'ㅞ': _m(HL=1, TD=1, VL=2, TL=1), 'ㅟ': _m(HL=1, TD=1, VL=1),
    'ㅢ': _m(HL=1, VL=1),
}

SHAPES = dict(voynich=VOYNICH, latin=LATIN, greek=GREEK, hangul=HANGUL)

# Named Voynich swap pairs from v5/v12 (tandem Hamming-1 neighbours above chance)
SWAPS = [('k', 't'), ('l', 'r'), ('l', 'o'), ('ch', 'cth'), ('ch', 'sh'), ('ch', 'k'), ('a', 'o')]

# Analogy families (parallelograms) defined from the shapes: the same stroke added to different bases
V_ANALOGIES = {
    'bench (+c_h)': [('k', 'ckh'), ('t', 'cth'), ('p', 'cph'), ('f', 'cfh')],
    'tloop (+TLOOP)': [('k', 't'), ('f', 'p'), ('ckh', 'cth'), ('cfh', 'cph')],
    'wide (+WIDE)': [('k', 'f'), ('t', 'p'), ('ckh', 'cfh'), ('cth', 'cph')],
    'plume (+PLUME)': [('ch', 'sh'), ('e', 's')],
    'rtail (+RTAIL)': [('i', 'n'), ('y', 'g')],
}
H_ANALOGIES = {
    'aspirate (+bar)': [('ㄱ', 'ㅋ'), ('ㄷ', 'ㅌ'), ('ㅈ', 'ㅊ')],
    'double': [('ㄱ', 'ㄲ'), ('ㄷ', 'ㄸ'), ('ㅂ', 'ㅃ'), ('ㅅ', 'ㅆ'), ('ㅈ', 'ㅉ')],
    'y-vowel (+tick)': [('ㅏ', 'ㅑ'), ('ㅓ', 'ㅕ'), ('ㅗ', 'ㅛ'), ('ㅜ', 'ㅠ')],
    '+i (ㅣ)': [('ㅏ', 'ㅐ'), ('ㅓ', 'ㅔ'), ('ㅗ', 'ㅚ'), ('ㅜ', 'ㅟ'), ('ㅡ', 'ㅢ')],
}
L_ANALOGIES = {
    '+arch': [('i', 'n'), ('n', 'm')],
    'descender vs ascender': [('b', 'p'), ('d', 'q')],
    '+xbar': [('c', 'e'), ('i', 't')],
}
G_ANALOGIES = {
    '+descender': [('ο', 'ρ'), ('ν', 'γ')],
    '+tall': [('ν', 'λ'), ('ο', 'δ')],
    '+xbar': [('ι', 'τ'), ('ο', 'σ')],
}
ANALOGIES = dict(voynich=V_ANALOGIES, hangul=H_ANALOGIES, latin=L_ANALOGIES, greek=G_ANALOGIES)
