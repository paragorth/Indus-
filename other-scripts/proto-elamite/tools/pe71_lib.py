"""pe71 'who signs with M153?': shared data and nulls.

Tablet records come from pe70 (data/pe70_ckpt/pe.pkl via pe70_common.get): opaque sign tokens, sealed flag,
seal ids (presence and identity only).  Line-level data from data/pe_corpus.json.  Publication numbers
from data/pe8_meta.json (find-lot proxy).  No sign reading and no seal picture is used.
"""
import json, os, re, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe70_common import get, bands  # noqa
from common import base, is_sign  # noqa

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe71_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)


def comps(tok):
    """component base signs of a compound token |A+B~x+C| -> ['A','B','C'] (X kept as 'X')."""
    if not tok.startswith('|'):
        return [tok]
    return [re.sub(r'~.*', '', p) for p in re.split(r'[+.x&]', tok.strip('|')) if p]


def load():
    R = get('pe')
    meta = json.load(open(os.path.join(DATA, 'pe8_meta.json')))
    C = {c['id']: c for c in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
    for r in R:
        m = meta.get(r['id'], {})
        try:
            r['pub'] = float(m.get('pub_num'))
        except Exception:
            r['pub'] = np.nan
        L = []
        for l in C[r['id']]['lines'] if r['id'] in C else []:
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            L.append(dict(signs=sg, nums=l['numerals'], surf=l['surface']))
        r['lines'] = L
        r['m288'] = any('M288' in l['signs'] for l in L)
    # sealing units: tablets sharing a seal id collapse to one unit (seal id sets that name the same tablets merge)
    g = collections.defaultdict(set)
    for i, r in enumerate(R):
        for s in r['seals']:
            g[s].add(i)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for i, r in enumerate(R):
        for s in r['seals'][1:]:
            parent[find(s)] = find(r['seals'][0])
    for i, r in enumerate(R):
        if r['sealed']:
            r['unit'] = find(r['seals'][0]) if r['seals'] else 'T:' + r['id']
        else:
            r['unit'] = None
    return R


def strata(R, how):
    b = bands(R)
    if how == 'volband':
        return np.array([r['vol'] + '|' + x for r, x in zip(R, b)])
    if how == 'doc':  # vol x size band x M157 header x has M288 (the pe70 sealed document type)
        return np.array(['%s|%s|%d|%d' % (r['vol'], x, r['hdr'] == 'M157', r['m288']) for r, x in zip(R, b)])
    raise ValueError(how)


def perm_index(st, rng, n):
    """n permutations (index arrays) within strata."""
    out = np.tile(np.arange(len(st)), (n, 1))
    groups = [np.where(st == s)[0] for s in np.unique(st)]
    for k in range(n):
        for i in groups:
            if len(i) > 1:
                out[k, i] = i[rng.permutation(len(i))]
    return out


def shift_index(R, rng, n):
    """find-lot null: within each volume, tablets ordered by publication number; labels circularly shifted
    by a random offset (keeps runs of sealed tablets intact).  Tablets without a number stay put."""
    out = np.tile(np.arange(len(R)), (n, 1))
    byv = collections.defaultdict(list)
    for i, r in enumerate(R):
        if not np.isnan(r['pub']):
            byv[r['vol']].append(i)
    for v, ix in byv.items():
        ix = np.array(sorted(ix, key=lambda i: R[i]['pub']))
        if len(ix) < 3:
            continue
        for k in range(n):
            s = rng.integers(1, len(ix))
            out[k, ix] = np.roll(ix, s)
    return out


def unit_codes(R):
    """integer unit code per tablet: -1 unsealed, else unit id"""
    u = sorted({r['unit'] for r in R if r['unit']})
    ui = {x: k for k, x in enumerate(u)}
    return np.array([ui[r['unit']] if r['unit'] else -1 for r in R])


def fam_stats(mask_idx, sealed, units, P):
    """observed and null (tablets sealed, distinct sealing units) for one family.
    P: (nperm, N) index arrays -- null label of tablet i is label of P[k, i]."""
    f = np.asarray(mask_idx)
    obs_t = int(sealed[f].sum())
    uu = units[f]; obs_u = len(set(uu[uu >= 0]))
    nt = sealed[P[:, f]].sum(1)
    U = np.sort(units[P[:, f]], axis=1)
    nu = ((U >= 0) & np.concatenate([np.ones((len(P), 1), bool), U[:, 1:] != U[:, :-1]], axis=1)).sum(1)
    return dict(n=len(f), sealed=obs_t, units=obs_u, e_t=float(nt.mean()), p_t=float((nt >= obs_t).mean()),
                e_u=float(nu.mean()), p_u=float((nu >= obs_u).mean()),
                z_u=float((obs_u - nu.mean()) / (nu.std() + 1e-9)))


def families(R, min_n=3):
    """component-sign families: sign S -> tablets with a compound containing S."""
    F = collections.defaultdict(set)
    for i, r in enumerate(R):
        for t in r['toks']:
            if t.startswith('|'):
                for c in set(comps(t)):
                    F[c].add(i)
    return {k: sorted(v) for k, v in F.items() if len(v) >= min_n}


def type_families(R, min_n=3):
    F = collections.defaultdict(set)
    for i, r in enumerate(R):
        for t in r['toks']:
            F[t].add(i)
    return {k: sorted(v) for k, v in F.items() if len(v) >= min_n}
