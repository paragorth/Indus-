"""Shared helpers for the Proto-Elamite tests."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

C_CODES = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N29B', 'N39N', 'N28'}
B_CODES = {'N51', 'N51G', 'N54', 'N54G', 'N46'}
FRAC_CODES = {'N02', 'N08', 'N08A', 'N8B', 'N8A'}


def norm_code(c):
    c = c.replace('N1@', 'N01@')
    if c == 'N1':
        c = 'N01'
    return c


def load(mode=None):
    """mode: 'rd' (DEFAULT from pe74 on: read + damaged tokens; uncertain '?' signs become 'x', restored '[..]'
    and supplied '<..>' signs are removed, and lines that lose a numeral are marked lacuna), 'r' (read only) or
    'all' (every token as build_corpus.py wrote it, editorial restorations included; use it only to reproduce
    loops up to pe73).  See tools/pe74_parse.py.  Overridden by the environment variable PE_CORPUS_MODE."""
    mode = mode or os.environ.get('PE_CORPUS_MODE', 'rd')
    if mode != 'all':
        import pe74_parse
        T = json.load(open(pe74_parse.corpus_file(mode)))
    else:
        T = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    for t in T:
        for l in t['lines']:
            l['numerals'] = [[n, norm_code(c)] for n, c in l['numerals']]
    return T


def base(sign):
    """Strip ~variant letters; keep compounds as compounds of base signs."""
    if sign.startswith('|'):
        parts = re.split(r'([+.x&])', sign.strip('|'))
        return '|' + ''.join(re.sub(r'~[A-Za-z0-9]+', '', p) for p in parts) + '|'
    return re.sub(r'~[A-Za-z0-9]+', '', sign)


def is_sign(s):
    return s.startswith('M') or s.startswith('|')


def system_of(numerals):
    """Classify one numeral group by diagnostic codes."""
    codes = {c for _, c in numerals}
    if not codes:
        return None
    if any('@' in c for c in codes):
        return 'C*' if any(c.split('@')[0] in C_CODES for c in codes) else 'mod*'
    if codes & C_CODES:
        return 'C'
    if codes & B_CODES:
        return 'B'
    if 'N23' in codes:
        return 'N23'
    if codes & FRAC_CODES:
        return 'S-frac'
    return 'SDB'   # plain N01/N14/N34/N45/N48: sexagesimal, decimal or bisexagesimal


def first_numeric_index(t):
    for i, l in enumerate(t['lines']):
        if l['numerals']:
            return i
    return None


def entries(T, require_clean=False, base_signs=True):
    """Entries = lines with >=1 sign and >=1 numeral, excluding the tablet's
    first line (header) and lines off the obverse when the tablet has exactly
    one numeric line off the obverse (that one is treated as a total)."""
    out = []
    for t in T:
        lines = t['lines']
        off = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
        for i, l in enumerate(lines):
            if i == 0 and not l['numerals']:
                continue
            if l['surface'] != 'obverse' and len(off) == 1 and l is off[0]:
                continue
            sg = [s for s in l['signs'] if is_sign(s) or s == 'x']
            if not sg or not l['numerals']:
                continue
            if require_clean and ('x' in sg or l['lacuna']):
                continue
            if base_signs:
                sg = [base(s) if s != 'x' else s for s in sg]
            out.append({'tablet': t['id'], 'signs': sg, 'numerals': l['numerals'],
                        'system': system_of(l['numerals']), 'prov': t['provenience']})
    return out


def header(t):
    """Return header sign list: first line if it has signs and no numerals."""
    if not t['lines']:
        return None
    l = t['lines'][0]
    if l['numerals'] or not [s for s in l['signs'] if is_sign(s)]:
        return None
    return [base(s) if s != 'x' else s for s in l['signs']]
