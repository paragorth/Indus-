"""pe18 'the number system is an accent': shared data for all cycles.

Tablet records with: region (SUSA / PLAT / other), fine number-system label per numeric line,
seal flag (from pe_raw.atf), non-numeral token streams (base signs, variant forms), header,
format vector, publication volume.  No sign readings from anyone are used.
"""
import json, os, re, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, C_CODES, B_CODES, FRAC_CODES  # noqa

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe18_ckpt')
os.makedirs(CK, exist_ok=True)

PLATEAU = ('Tepe Yahya', 'Malyan', 'Tepe Sialk', 'Tepe Sofalin', 'Ozbaki', 'Shahr-i Sokhta')
SYSTEMS = ['C', 'C@', 'B', 'S@', 'N23', 'FRAC', 'DEC', 'SEX', 'AMB']


def region(p):
    if p.startswith('Susa'):
        return 'SUSA'
    if any(k in p for k in PLATEAU):
        return 'PLAT'
    return 'OTHER'


def site(p):
    for k in PLATEAU:
        if k in p:
            return k
    return 'Susa' if p.startswith('Susa') else (p or 'none')


def line_system(nums):
    codes = [c for _, c in nums]
    if not codes:
        return None
    bases = {c.split('@')[0] for c in codes}
    at = any('@' in c for c in codes)
    if bases & C_CODES:
        return 'C@' if at else 'C'
    if bases & B_CODES:
        return 'B'
    if at:
        return 'S@'
    if 'N23' in bases:
        return 'N23'
    if bases & FRAC_CODES:
        return 'FRAC'
    d = collections.Counter()
    for n, c in nums:
        if isinstance(n, int):
            d[c] += n
    if d.get('N14', 0) >= 6 or 'N45' in bases:
        return 'DEC'
    if 'N34' in bases or 'N48' in bases:
        return 'SEX'
    return 'AMB'


def seal_map():
    """P-number -> (sealed flag, list of seal ids) from the raw ATF."""
    out = {}
    cur = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&P'):
            cur = raw[1:8]
            out[cur] = [False, []]
            continue
        if cur and re.search(r'seal', raw, re.I) and not raw.startswith('@object'):
            if raw.startswith('&'):
                continue
            out[cur][0] = True
            out[cur][1] += re.findall(r'PES\d+', raw)
    return out


def tablets():
    T = load()
    S = seal_map()
    fin = collections.Counter()
    for t in T:
        for l in t['lines']:
            sg = [s for s in l['signs'] if is_sign(s)]
            if sg and l['numerals']:
                fin[base(sg[-1])] += 1
    CLASS = {s for s, c in fin.items() if c >= 30}
    out = []
    for t in T:
        if t['object_type'] != 'tablet':
            continue
        lines = t['lines']
        sysl = [line_system(l['numerals']) for l in lines if l['numerals']]
        sysc = collections.Counter(s for s in sysl if s)
        toks, forms, nocls = [], [], []
        ent_len, single, n_ent, finals = [], 0, 0, []
        for i, l in enumerate(lines):
            sg = [s for s in l['signs'] if is_sign(s)]
            for s in sg:
                b = base(s)
                toks.append(b)
                if b not in CLASS:
                    nocls.append(b)
                if '~' in s and not s.startswith('|'):
                    forms.append((b, s))
            if sg and l['numerals']:
                n_ent += 1
                ent_len.append(len(sg))
                single += len(sg) == 1
                finals.append(base(sg[-1]))
        hdr = None
        if lines and not lines[0]['numerals']:
            h = [base(s) for s in lines[0]['signs'] if is_sign(s)]
            hdr = h[0] if h else None
        surf = {l['surface'] for l in lines}
        cols = {(l['surface'], l['column']) for l in lines}
        off_num = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
        fmt = [np.log1p(len(lines)), np.mean(ent_len) if ent_len else 0.0,
               single / n_ent if n_ent else 0.0, len(cols), float('reverse' in surf),
               float(len(off_num) >= 1), (len(sysl) / len(lines)) if lines else 0.0,
               np.mean([l['damaged'] for l in lines]) if lines else 0.0,
               float(hdr is not None)]
        vol = re.sub(r',.*', '', t['designation']).strip()
        sm = S.get(t['id'], [False, []])
        out.append({'id': t['id'], 'prov': t['provenience'], 'region': region(t['provenience']),
                    'site': site(t['provenience']), 'vol': vol, 'n_lines': len(lines),
                    'n_num': len(sysl), 'n_ent': n_ent, 'sys': dict(sysc), 'toks': toks,
                    'nocls': nocls, 'finals': finals,
                    'dom': collections.Counter(finals).most_common(1)[0][0] if finals else 'none', 'forms': forms, 'hdr': hdr, 'fmt': fmt,
                    'sealed': sm[0], 'seals': sm[1], 'cap': bool(sysc.get('C', 0) + sysc.get('C@', 0))})
    return out, CLASS


def size_bin(t):
    n = t['n_num']
    return 0 if n <= 2 else 1 if n <= 5 else 2 if n <= 10 else 3


def strata_perm(labels, strata, rng):
    lab = np.array(labels).copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        lab[idx] = lab[rng.permutation(idx)]
    return lab
