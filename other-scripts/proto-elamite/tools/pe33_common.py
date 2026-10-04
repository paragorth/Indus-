"""pe33 'the seal picture says what the tablet is about': shared data and tests."""
import json, os, sys, collections, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe18_common import tablets as pe18_tablets, size_bin  # noqa
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe33_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)
OFFICES = {'GRAIN': {'M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'},
           'CLASS': {'M387', 'M388', 'M218', 'M124', 'M009', 'M066', 'M057'},
           'BARE': {'M054', 'M367', 'M001', 'M370', 'M032', '|M036+1(N30D)|', 'M206', 'M269', 'M059', 'M102'}}
HERD = {'M362', 'M367', 'M346', 'M006'}
MOTIFS = ['BOVID', 'CAPRID', 'FELINE', 'ANTHRO', 'MONSTER', 'WATER', 'HUMAN', 'PREDATION', 'BUILDING']


def content(t):
    """content features of one tablet (pe18 record) -> set of strings"""
    f = set('S:' + s for s in t['toks'])
    f |= set('Y:' + s for s in t['sys'])
    if t['hdr']:
        f.add('H:' + t['hdr'])
    sg = set(t['toks'])
    for o, L in OFFICES.items():
        if sg & L:
            f.add('O:' + o)
    if sg & HERD:
        f.add('O:HERD')
    f.add('Z:%d' % size_bin(t))
    return f


def load(primary_only=True):
    M = json.load(open(os.path.join(DATA, 'pe33_seal_motifs.json')))['tablets']
    T, _ = pe18_tablets()
    byid = {t['id']: t for t in T}
    rows = []
    for m in M:
        if primary_only and not m['primary']:
            continue
        t = byid.get(m['id'])
        if t is None:
            continue
        seal = 'F%d' % min(m['figs'])
        rows.append({'id': m['id'], 'vol': t['vol'], 'seal': seal, 'motif': set(m['features']),
                     'content': content(t), 'size': size_bin(t), 't': t})
    return rows, T


def matrices(rows, min_n=4):
    feats = collections.Counter(f for r in rows for f in r['content'])
    n = len(rows)
    F = sorted(f for f, c in feats.items() if min_n <= c <= n - min_n and not f.startswith('Z:'))
    X = np.array([[f in r['content'] for f in F] for r in rows], dtype=np.int8)
    mot = [m for m in MOTIFS if min_n <= sum(m in r['motif'] for r in rows) <= n - min_n]
    Y = np.array([[m in r['motif'] for m in mot] for r in rows], dtype=np.int8)
    return X, F, Y, mot


def strata_of(rows, key=('vol', 'size')):
    return np.array(['|'.join(str(r[k]) for k in key) for r in rows])


def perm_tablet(Y, strata, rng):
    Yp = Y.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        Yp[idx] = Y[rng.permutation(idx)]
    return Yp


def perm_seal(Y, rows, rng, by_vol=True):
    """permute motif vectors among seals (blocks), within volume of the seal's majority."""
    seals = sorted({r['seal'] for r in rows})
    mot_of = {}
    vol_of = {}
    for i, r in enumerate(rows):
        mot_of.setdefault(r['seal'], Y[i])
        vol_of.setdefault(r['seal'], collections.Counter())[r['vol'].replace('MDP 26S', 'MDP 26')] += 1
    sv = {s: vol_of[s].most_common(1)[0][0] if by_vol else 'all' for s in seals}
    new = {}
    for v in set(sv.values()):
        S = [s for s in seals if sv[s] == v]
        P = rng.permutation(len(S))
        for a, b in zip(S, P):
            new[a] = mot_of[S[b]]
    return np.array([new[r['seal']] for r in rows])


def assoc(X, Y):
    """log odds ratio (0.5-corrected) and one-sided-agnostic |z| for every motif x feature pair."""
    X = X.astype(float); Y = Y.astype(float)
    a = Y.T @ X                     # motif & feat
    b = Y.T @ (1 - X)               # motif & not feat
    c = (1 - Y).T @ X
    d = (1 - Y).T @ (1 - X)
    lor = np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5)))
    se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
    return lor, lor / se
