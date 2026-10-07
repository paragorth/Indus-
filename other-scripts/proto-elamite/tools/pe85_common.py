"""pe85: WHY is the header end squared? Use-model tournament on the pe81/pe83 outline measurements.

No new photos: reuses data/pe83_ckpt/c1_feats.json, c1d_feats.json, lineart_feats.json (outline features) and
pe71_lib.load() (seal presence / identity, volume, publication number). Text attributes come from common.load().
"""
import sys, os, json, re, hashlib
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pe83_common as P

CK = os.path.join(common.DATA, 'pe85_ckpt')
LOOPS = os.path.join(os.path.dirname(common.DATA), 'loops')
os.makedirs(CK, exist_ok=True)

# pe63 dossiers (loops/pe63_cycle1.txt, pe63_final.txt); D7 (Sofalin) has no photo
DOSS = {'D1': ['P%06d' % i for i in range(8717, 8732)], 'D2': ['P%06d' % i for i in range(8796, 8803)],
        'D3': ['P009190', 'P009211', 'P009220', 'P009237', 'P009238', 'P009286', 'P009309'],
        'D4': ['P009056', 'P009137', 'P009138', 'P009139', 'P009140'], 'D5': ['P%06d' % i for i in range(8790, 8795)],
        'D6': ['P008100', 'P008125', 'P008193', 'P368479']}
DOSS_OF = {p: d for d, v in DOSS.items() for p in v}
M327FAM = ('M327',)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def m153_last(t):
    sl = [l for l in t['lines'] if l['surface'] in ('obverse', 'reverse') and [s for s in l['signs'] if common.is_sign(s)]]
    if not sl:
        return 0.0
    return float(any(s.startswith('|M153+') for s in sl[-1]['signs']))


def table(intact=True):
    import pe71_lib as L
    S = {r['id']: r for r in L.load()}
    F1 = json.load(open(os.path.join(P.CK, 'c1_feats.json'))); F2 = json.load(open(os.path.join(P.CK, 'c1d_feats.json')))
    LA = json.load(open(os.path.join(P.CK, 'lineart_feats.json')))
    out = []
    for r in P.rows():
        if r['id'] not in F1 or r['id'] not in F2:
            continue
        t = r['t']; l1 = t['lines'][0]
        r = dict(r, **F1[r['id']], **F2[r['id']])
        r['l1ok'] = float(not l1['lacuna'] and not l1['damaged'])
        if intact and not r['l1ok']:
            continue
        s = S.get(r['id'], {})
        h = common.header(t) or []
        r['tag'] = float(any(l['surface'] == 'top' for l in t['lines']))
        r['sealed'] = float(bool(s.get('sealed')))
        r['pes329'] = float('PES0329' in s.get('seals', [])); r['pes334'] = float('PES0334' in s.get('seals', []))
        r['seal_unit'] = s.get('unit')
        r['vol'] = s.get('vol') or t.get('volume') or 'other'
        r['pub'] = s.get('pub', np.nan)
        r['m288'] = float(bool(s.get('m288')))
        r['h157'] = float(bool(h) and h[0] == 'M157')
        r['h327'] = float(bool(h) and any(x.startswith('M327') or x.startswith('|M327') for x in h[:1]))
        r['hlen'] = float(len(h))
        r['m153'] = m153_last(t)
        r['doss'] = DOSS_OF.get(r['id'])
        r['indoss'] = float(r['doss'] is not None)
        r['outpost'] = float(r['site'] != 'Susa (mod. Shush)' and 'Susa' not in r['site'])
        sysd = s.get('sys', {}) or {}
        tot = sum(sysd.values()) or 1
        r['cap'] = sum(v for k, v in sysd.items() if k.startswith('C')) / tot
        r['asym'] = r['cf_top'] - r['cf_bot']
        r['revend'] = r.get('rev_img_lower', np.nan)
        la = LA.get(r['id']) or {}
        r['la_top'] = la.get('la_cf_top', np.nan); r['la_bot'] = la.get('la_cf_bot', np.nan)
        out.append(r)
    return out


def col(R, k):
    return np.array([np.nan if r.get(k) is None else r[k] for r in R], float)


def resid(y, Z):
    ok = np.isfinite(y)
    out = np.full(len(y), np.nan)
    a = rankdata(y[ok]); Zo = Z[ok]
    out[ok] = a - Zo @ np.linalg.lstsq(Zo, a, rcond=None)[0]
    return out


def zs(v):
    v = np.asarray(v, float); m = np.nanmean(v); s = np.nanstd(v)
    return np.nan_to_num((v - m) / (s if s > 0 else 1))


def perm_within(v, s, rng):
    v = v.copy()
    for k in np.unique(s):
        m = np.where(s == k)[0]; v[m] = v[rng.permutation(m)]
    return v
