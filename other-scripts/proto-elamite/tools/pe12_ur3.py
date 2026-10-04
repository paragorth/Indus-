"""pe12 positive control: Ur III weight entries ('N gin2 ku3-babbar', 'N ma-na siki').
The word after the number (unit: gin2 / ma-na / gu2 / sze) is a KNOWN size-class marker
(ma-na for >= 60 shekels), and the commodity word (silver vs wool / copper / bitumen)
is a soft, real size dependency.  Entries are built in the PE engine's format:
signs = [unit, commodity words...], value in shekels, digit counts mapped to PE-like codes.
Source: CDLI bulk ATF + catalogue in the scratchpad (not committed).
"""
import os, re, csv, json
from collections import Counter
from pe12_common import SCRATCH, CKPT

UNIT = {'gin2': 1.0, 'ma-na': 60.0, 'gu2': 3600.0}   # 'sze' dropped: also 'barley'
DIG = {'disz': ('N01', 1), 'asz': ('N01', 1), 'u': ('N14', 10), 'gesz2': ('N34', 60),
       "gesz'u": ('N45', 600), 'szar2': ('N48', 3600)}
NUM = re.compile(r'^(\d+(?:/\d+)?)\(([a-z\']+\d?)(?:@[a-z])?\)$')


def periods():
    cache = os.path.join(CKPT, 'ur3_ids.json')
    if os.path.exists(cache):
        return set(json.load(open(cache)))
    csv.field_size_limit(10 ** 9)
    ids = []
    for r in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if r['period'].startswith('Ur III'):
            ids.append('P%06d' % int(r['id_text']))
    json.dump(ids, open(cache, 'w'))
    return set(ids)


def parse_line(txt):
    s = re.sub(r'[_\[\]#?!<>]', '', txt)
    toks = s.split()
    if not toks or any(t in ('la2', 'szu-nigin2', 'szu-nigin', 'n') for t in toks):
        return None
    k, val, dig, first_unit, frac = 0, 0.0, Counter(), None, 0
    while k < len(toks):
        grp = 0.0
        j = k
        while j < len(toks) and NUM.match(toks[j]):
            n, d = NUM.match(toks[j]).groups()
            if '/' in n:
                a, b = n.split('/')
                grp += int(a) / int(b); frac += 1; dig['N08'] += 1
            else:
                if d not in DIG:
                    return None
                code, mult = DIG[d]
                grp += int(n) * mult; dig[code] += int(n)
            j += 1
        if j == k:
            break
        if j >= len(toks) or toks[j] not in UNIT:
            return None
        if first_unit is None:
            first_unit = toks[j]
        val += grp * UNIT[toks[j]]
        k = j + 1
    rest = [t for t in toks[k:] if not NUM.match(t) and t != '...'][:3]
    if 'gur' in rest or 'sze' in rest:
        return None
    if first_unit is None or not rest or val <= 0:
        return None
    return first_unit, rest, val, dig, frac


def ur3_entries():
    ids = periods()
    E, cur, tab, lines = [], None, None, []

    def flush():
        ents = [x for x in lines if x]
        for j, (unit, rest, val, dig, frac, li) in enumerate(ents):
            E.append(dict(tab=tab, signs=[unit] + rest, sys='S', v=val, v2=sum(dig.values()), dig=dig,
                          frac=int(frac > 0), line=li, hdr='none', first_after_hdr=False, idx=j, n=len(ents)))

    li = 0
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            if tab:
                flush()
            p = raw[1:8]
            tab = p if p in ids else None
            lines, li = [], 0
            continue
        if tab is None or not re.match(r"^\d+'?\.", raw):
            continue
        li += 1
        r = parse_line(raw.split('.', 1)[1])
        if r:
            lines.append(r + (li,))
    if tab:
        flush()
    return E


def pe_sized(E, n_target=5336, seed=0):
    """Random whole tablets until the entry count matches the PE corpus."""
    import numpy as np
    tabs = sorted(set(e['tab'] for e in E))
    np.random.default_rng(seed).shuffle(tabs)
    by = {}
    for e in E:
        by.setdefault(e['tab'], []).append(e)
    out = []
    for t in tabs:
        out.extend(by[t])
        if len(out) >= n_target:
            break
    return out
