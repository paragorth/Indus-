"""pe75 'THE CLAY WAS CUT TO A RULER': shared data.

Physical dimensions (height, width, thickness, mm) come straight from the CDLI catalogue
(cdli-gh/data release, cdli_cat.csv, fetched 25 Sep 2026, kept in the scratchpad, not
committed; a random sample is re-checked live against https://cdli.earth/artifacts/<id>.json).
Text features come from common.load() (damage-aware default 'rd').
Control corpora from the same catalogue: proto-cuneiform (Uruk IV/III) and Ur III tablets.
"""
import csv, json, os, sys, collections, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe75_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
CAT = os.path.join(SCRATCH, 'cdli_cat.csv')
os.makedirs(CK, exist_ok=True)

BAD_PRES = re.compile(r'fragment|%|broken', re.I)


def _f(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None


def catalogue():
    """id -> dict(h,w,t,pres,period,prov,otype) for PE, proto-cuneiform and Ur III tablets."""
    cache = os.path.join(CK, 'cat.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    csv.field_size_limit(10 ** 9)
    out = {}
    for row in csv.DictReader(open(CAT, encoding='utf-8')):
        per = row['period']
        if not (per.startswith('Proto-Elamite') or per.startswith('Uruk III') or per.startswith('Uruk IV')
                or per.startswith('Ur III')):
            continue
        if row['object_type'] != 'tablet':
            continue
        pid = 'P%06d' % int(row['id_text'])
        out[pid] = {'h': _f(row['height']), 'w': _f(row['width']), 't': _f(row['thickness']),
                    'pres': row['object_preservation'], 'period': per, 'prov': row['provenience'],
                    'genre': row['genre']}
    json.dump(out, open(cache, 'w'))
    return out


def pe_table():
    """One row per PE tablet with dims and text features available at the START of writing."""
    cat = catalogue()
    T = common.load()
    rows = []
    for t in T:
        d = cat.get(t['id'])
        if not d:
            continue
        L = t['lines']
        if not L:
            continue
        lost = any(l['lacuna'] for l in L) or any(l.get('n_dropped_signs') or l.get('n_dropped_nums') for l in L)
        hdr = common.header(t)
        first_num = next((l for l in L if l['numerals']), None)
        fsys = common.system_of(first_num['numerals']) if first_num else 'NONE'
        first_signs = [common.base(s) for s in (L[0]['signs'] if L else []) if common.is_sign(s)]
        off = [l for l in L if l['surface'] != 'obverse' and l['numerals']]
        nglyph = sum(len([s for s in l['signs'] if common.is_sign(s) or s == 'x']) + sum((n or 1) for n, _ in l["numerals"]) for l in L)
        signs = [common.base(s) for l in L for s in l['signs'] if common.is_sign(s)]
        faces = collections.Counter(l['surface'] for l in L)
        rows.append({
            'id': t['id'], 'h': d['h'], 'w': d['w'], 't': d['t'], 'pres': d['pres'],
            'complete_cat': not BAD_PRES.search(d['pres'] or ''),
            'intact': (not lost) and not BAD_PRES.search(d['pres'] or ''),
            'site': t.get('site'), 'n_lines': len(L), 'glyphs': nglyph,
            'n_num_lines': sum(1 for l in L if l['numerals']),
            'header': hdr or [], 'first_signs': first_signs, 'first_sys': fsys or 'NONE',
            'signs': sorted(set(signs)), 'obv': faces.get('obverse', 0), 'rev': faces.get('reverse', 0),
            'has_total': len(off) == 1,
            'systems': sorted({common.system_of(l['numerals']) for l in L if l['numerals']} - {None}),
            'max_num': max([sum((n or 1) for n, _ in l["numerals"]) for l in L if l['numerals']] or [0]),
        })
    return rows


def control_dims(prefix):
    cat = catalogue()
    return {k: v for k, v in cat.items() if v['period'].startswith(prefix)}
