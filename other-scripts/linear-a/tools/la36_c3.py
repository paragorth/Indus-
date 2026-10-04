#!/usr/bin/env python3
"""LA-36 cycle 3: MASSIVE RANDOM TRANSACTION RULES, HELD OUT.

(a) Global rule: a hypothesis H = (fraction value set V, transform t). V: 2,000 random value sets
    (11 letters from a pool of 22 fractions) + the conventional set; t: RATIO r = p/q <= 16 under
    5 roundings, DIFF |y - x| = d (d = k/4, k <= 120); either direction. ~3.7 M hypotheses.
    A hit = an aligned entry pair (not identical under V) obeying t. 20 random half splits of the
    units: the 20 best H on the training half are scored on the held-out half (mean hits).
    Null: the same pipeline on 30 data sets with alignment shuffled inside each list.
    Planted: 40 % of units given y = round(3/4 x) (conventional V). Control: PY Ma KE vs *146 and
    O vs *152, split by town (known 2/7, 1/2).
(b) Leave-one-entry-out inside units with >= 3 aligned entries: fit the best family on the other
    entries (k >= 2), predict the held-out amount (credit 1/|prediction set|). Null: N1.
    Control: PY Ma columns; planted rules.
(c) Fraction values: HT 9a/b pair score as a function of J (other letters conventional), and the
    J value chosen by the global search on SIDES+SCRIBE units.
"""
import json, os, sys, time
import numpy as np
from fractions import Fraction as Fr
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la36_common import *

LET = ['J', 'E', 'F', 'K', 'D', 'B', 'A', 'H', 'JE', 'L2', 'L6']
POOL = [Fr(1, k) for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)] + \
       [Fr(2, 3), Fr(3, 4), Fr(3, 8), Fr(5, 8), Fr(5, 16), Fr(3, 16), Fr(7, 16), Fr(2, 5), Fr(3, 10), Fr(5, 6)]
NV = int(os.environ.get('NV', 2000)); NSPLIT = 20; NNULL = int(os.environ.get('NNULL', 30)); TOP = 20
DGRID = np.arange(1, 121) / 4.0


def rand_V(rng):
    V = dict(CONV)
    for l in LET: V[l] = POOL[rng.integers(len(POOL))]
    return V


def hits_matrix(x, y):
    """x, y: (E,) -> (T, E) bool hits for all transforms, either direction, identical excluded."""
    ne = np.abs(x - y) > 1e-6
    rows = []
    for m in RMODES:
        for (a, b) in ((x, y), (y, x)):
            pass
    rx = x[None, :] * RATF[:, None]; ry = y[None, :] * RATF[:, None]
    for m in RMODES:
        h = (np.abs(apply_round(rx, m) - y[None]) < 1e-6) & (x[None] > 0)
        h |= (np.abs(apply_round(ry, m) - x[None]) < 1e-6) & (y[None] > 0)
        rows.append(h)
    rows.append(np.abs(np.abs(y - x)[None, :] - DGRID[:, None]) < 1e-6)
    H = np.concatenate(rows, 0)
    return H & ne[None]


def tname(t):
    nr = len(RAT)
    if t < 5 * nr:
        m, a = divmod(t, nr); return f'RATIO {RAT[a]} {RMODES[m]}'
    return f'DIFF {DGRID[t - 5 * nr]}'


def entries(U):
    """flatten: list of (unit index, x_raw, y_raw)"""
    return [(i, a, b) for i, u in enumerate(U) for a, b in zip(u['x'], u['y'])]


ALLLET = sorted(CONV)


def _mat(side):
    iv = np.array([a[0] for a in side], float)
    C = np.zeros((len(side), len(ALLLET)))
    for i, a in enumerate(side):
        for f in a[1]:
            if f in ALLLET: C[i, ALLLET.index(f)] += 1
    return iv, C


def all_hits(E, Vs):
    """(V, T, E) hits, vectorised over V (value = integer + letter counts @ letter values)."""
    ix, Cx = _mat([a for _, a, _ in E]); iy, Cy = _mat([b for _, _, b in E])
    Vm = np.array([[float(V.get(l, Fr(1, 16))) for l in ALLLET] for V in Vs])     # (V, L)
    X = ix[None] + Vm @ Cx.T; Y = iy[None] + Vm @ Cy.T                               # (V, E)
    ne = np.abs(X - Y) > 1e-6
    out = []
    for m in RMODES:
        rx = X[:, None, :] * RATF[None, :, None]; ry = Y[:, None, :] * RATF[None, :, None]
        h = (np.abs(apply_round(rx, m) - Y[:, None, :]) < 1e-6) & (X[:, None, :] > 0)
        h |= (np.abs(apply_round(ry, m) - X[:, None, :]) < 1e-6) & (Y[:, None, :] > 0)
        out.append(h)
    out.append(np.abs(np.abs(Y - X)[:, None, :] - DGRID[None, :, None]) < 1e-6)
    return np.concatenate(out, 1) & ne[:, None, :]


def heldout(HM, unit_of, nunits, rng, splits=NSPLIT):
    res = []; best_rules = Counter()
    for s in range(splits):
        tr_units = rng.random(nunits) < 0.5
        tr = tr_units[unit_of]; te = ~tr
        a = HM[:, :, tr].sum(2); b = HM[:, :, te].sum(2)
        flat = a.ravel(); idx = np.argsort(-flat, kind='stable')[:TOP]
        res.append(float(b.ravel()[idx].mean()))
        v, t = np.unravel_index(idx[0], a.shape); best_rules[tname(int(t))] += 1
    return float(np.mean(res)), best_rules


def shuffle_units(U, rng):
    out = []
    for u in U:
        v = dict(u); n = len(u['x'])
        pa = list(u['poolA']); pb = list(u['poolB'])
        ia = rng.permutation(len(pa))[:n]; ib = rng.permutation(len(pb))[:n]
        v['x'] = [pa[i] for i in ia]; v['y'] = [pb[i] for i in ib]
        out.append(v)
    return out


def part_a(U, tag, rng, Vs):
    E = entries(U); unit_of = np.array([e[0] for e in E])
    HM = all_hits(E, Vs)
    obs, rules = heldout(HM, unit_of, len(U), np.random.default_rng(1))
    nulls = []
    for k in range(NNULL):
        Us = shuffle_units(U, rng)
        HMs = all_hits(entries(Us), Vs)
        nulls.append(heldout(HMs, unit_of, len(U), np.random.default_rng(1))[0])
    nulls = np.array(nulls)
    r = {'entries': len(E), 'heldout_hits': obs, 'null_mean': float(nulls.mean()), 'null_sd': float(nulls.std()),
         'P': float((np.sum(nulls >= obs) + 1) / (NNULL + 1)), 'best_rules': rules.most_common(5)}
    # which J does the global best pick (all data)
    tot = HM.sum(2); v, t = np.unravel_index(np.argmax(tot), tot.shape)
    r['best_all'] = {'rule': tname(int(t)), 'hits': int(tot[v, t]), 'J': str(Vs[v]['J']), 'D': str(Vs[v]['D']), 'B': str(Vs[v]['B'])}
    print(tag, json.dumps(r, default=str), flush=True)
    return r


# ------------------------------------------------------------------ (b) leave-one-out
def predictions(xs, ys):
    """candidate predictions for y given training (xs, ys): transforms with max agreement k >= 2."""
    cands = {}
    n = len(xs)
    def add(key, f):
        k = sum(1 for a, b in zip(xs, ys) if abs(f(a) - b) < 1e-6 and abs(a - b) > 1e-6)
        if k >= 2: cands[key] = (k, f)
    for ri, r in enumerate(RATF):
        for m in RMODES:
            add(('R', ri, m), lambda a, r=r, m=m: float(apply_round(np.array(a * r), m)))
    for a, b in zip(xs, ys):
        d = b - a
        if abs(d) > 1e-6: add(('D', round(d, 6)), lambda z, d=d: z + d)
        T = a + b
        add(('C', round(T, 6)), lambda z, T=T: T - z)
    if not cands: return None
    kmax = max(k for k, _ in cands.values())
    return [f for k, f in cands.values() if k == kmax]


def loo(U, V):
    score = 0.0; tried = 0
    for u in U:
        if len(u['x']) < 3: continue
        x = vals(u['x'], V); y = vals(u['y'], V)
        for i in range(len(x)):
            m = np.ones(len(x), bool); m[i] = False
            fs = predictions(list(x[m]), list(y[m]))
            tried += 1
            if not fs: continue
            preds = {round(f(x[i]), 6) for f in fs}
            if round(y[i], 6) in preds and abs(x[i] - y[i]) > 1e-6: score += 1 / len(preds)
    return score, tried


def part_b(U, tag, rng, V=CONV, nnull=None):
    nnull = nnull or NNULL
    U = [u for u in U if len(u['x']) >= 3]
    obs, tried = loo(U, V)
    nl = np.array([loo(shuffle_units(U, rng), V)[0] for _ in range(nnull)])
    r = {'units': len(U), 'tried': tried, 'loo_score': obs, 'null_mean': float(nl.mean()),
         'P': float((np.sum(nl >= obs) + 1) / (nnull + 1))}
    print(tag, json.dumps(r), flush=True)
    return r


def plant(U, rng, frac=0.4, r=0.75, mode='round'):
    out = []
    for u in U:
        v = dict(u)
        if rng.random() < frac:
            x = vals(u['x'], CONV)
            yy = apply_round(x * r, mode)
            v['y'] = [(float(a), ()) for a in yy]
            v['poolB'] = v['y'] + list(u['poolB'])[len(v['y']):]
        out.append(v)
    return out


def ma_split_units(c1, c2):
    """one PY Ma commodity pair as a list of single-town units (so the split is by town)."""
    M = [u for u in ma_units() if u['id'] == f'PYMa:{c1}~{c2}'][0]
    return [{'id': k, 'cls': 'MA', 'x': [a], 'y': [b], 'poolA': M['poolA'], 'poolB': M['poolB']}
            for k, a, b in zip(M['keys'], M['x'], M['y'])]


def main():
    rng = np.random.default_rng(363)
    out = {}
    Vs = [dict(CONV)] + [rand_V(rng) for _ in range(NV)]
    LA1 = la_units(1); LA2 = la_units(2)
    t = time.time()
    PART = os.environ.get('PART', 'abc')
    if 'a' not in PART:
        pass
    else:
      out['a_LA_all'] = part_a(LA1, 'A_LA_all', rng, Vs)
      out['a_LA_dossier'] = part_a([u for u in LA1 if u['cls'] in ('SIDES', 'SCRIBE')], 'A_LA_SIDES+SCRIBE', rng, Vs)
      out['a_LA_cols'] = part_a([u for u in LA1 if u['cls'] in ('COLUMNS', 'ROWS')], 'A_LA_COLUMNS+ROWS', rng, Vs)
      out['a_plant'] = part_a(plant(LA1, rng), 'A_PLANT_3/4round_40%', rng, Vs[:200])
      out['a_plant20'] = part_a(plant(LA1, rng, 0.2), 'A_PLANT_3/4round_20%', rng, Vs[:200])
      for c1, c2 in (('*146', 'KE'), ('*152', 'O'), ('*146', '*152')):
        out[f'a_MA_{c1}_{c2}'] = part_a(ma_split_units(c1, c2), f'A_MA_{c1}~{c2}', rng, Vs[:50])
    print('time a', time.time() - t, flush=True)
    if 'b' in PART:
      out['b_LA'] = part_b(LA2, 'B_LA', rng)
      out['b_LA_noHT9'] = part_b([u for u in LA2 if 'HT9a' not in u['id']], 'B_LA_noHT9', rng)
      out['b_MA'] = part_b(ma_units(), 'B_MA', rng, nnull=10)
      out['b_LB'] = part_b(lb_units(), 'B_LB', rng, nnull=10)
      out['b_plant'] = part_b(plant(LA2, rng, 0.4), 'B_PLANT_40%', rng, nnull=10)
    # (c) J scan on HT 9a/b
    u9 = [u for u in LA2 if u['id'] == 'HT9a|HT9b'][0]
    scan = {}
    for j in POOL:
        V = dict(CONV); V['J'] = j
        d, s = detail(u9, V); scan[str(j)] = (s, d['ratio'])
    out['c_HT9_J'] = scan
    print('C_HT9_J', json.dumps(scan), flush=True)
    json.dump(out, open(os.path.join(CK, f'c3_{PART}.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
