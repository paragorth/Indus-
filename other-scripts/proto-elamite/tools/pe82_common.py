"""pe82 shared loaders: PE and proto-cuneiform tablets as ordered numeric lines with publication-batch labels.

Tablet: dict(id, batch, lines=[dict(sg=tuple(base signs, 'x' dropped), sys=str, nums=tuple, surf='o'|'r', val)])
PE batches: MDP06, MDP17, MDP26, MDP26S, TCL+MDP31, OTHER.  PC batches: U4 (Uruk IV, Uruk), U3 (Uruk III, Uruk), OTHER.
"""
import os, sys, json, math, random, collections, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe82_ckpt')
os.makedirs(CK, exist_ok=True)

PE_BATCH = {'MDP 06': 'MDP06', 'MDP 17': 'MDP17', 'MDP 26': 'MDP26', 'MDP 26S': 'MDP26S',
            'TCL 32': 'TCL31', 'MDP 31': 'TCL31'}

# pre-registered proto-cuneiform calibration sets (copied from pe77_common, fixed before pe77 ran)
from pe77_common import PC_WORLD, PC_CONV


def _lines(raw, signfilter, base=True):
    out = []
    for l in raw:
        if not l['numerals'] or l.get('lacuna'):
            continue
        nums = tuple((n, common.norm_code(c)) for n, c in l['numerals'])
        sg = tuple(common.base(s) if base else s for s in l['signs'] if signfilter(s))
        out.append(dict(sg=sg, sys=common.system_of([list(x) for x in nums]) or 'none', nums=nums,
                        surf='o' if l['surface'] == 'obverse' else 'r'))
    return out


def load_pe(mode=None):
    T = common.load(mode)
    out = []
    for t in T:
        b = PE_BATCH.get(t.get('volume'), 'OTHER') if t.get('site') == 'Susa' else 'OTHER'
        hdr = None
        for l in t['lines']:
            s = [common.base(x) for x in l['signs'] if common.is_sign(x)]
            if s:
                hdr = s[0]
                break
        L = _lines(t['lines'], common.is_sign)
        if L:
            out.append(dict(id=t['id'], batch=b, hdr=hdr, site=t.get('site'), lines=L))
    return out


def load_pc():
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    keep = lambda s: s not in ('x', 'X') and not s.startswith('x') and not s.startswith('X')
    out = []
    for t in T:
        prov = t.get('provenience', '')
        if prov.startswith('Uruk (mod. Warka)') and not prov.endswith('?'):
            b = 'U4' if t.get('period') == 'Uruk IV' else 'U3'
        else:
            b = 'OTHER'
        hdr = None
        for l in t['lines']:
            s = [common.base(x) for x in l['signs'] if keep(x)]
            if s:
                hdr = s[0]
                break
        L = _lines(t['lines'], keep)
        if L:
            out.append(dict(id=t['id'], batch=b, hdr=hdr, site=prov, lines=L))
    return out


def sha(fn):
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()
