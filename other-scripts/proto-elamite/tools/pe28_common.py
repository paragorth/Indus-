"""pe28 PROPAGATE THE RATES: constraint propagation ('crossword') over every count -> entry equation
in the corpus, seeded with the pe27 anchor rates.  Shared code.

Equation: a target line j (final sign B, system S = CAP in N39C units or CNT in units) and a window of
1-3 contiguous count lines among the 6 numeric lines before it, with final signs A_1..A_k and counts
x_1..x_k.  Hypothesis: y_j = sum_i r(A_i -> B, S) x_i.  Unknowns: the rates r(key), key = (A, B, S).
Rates are restricted to a notation grid (values a scribe could write as a small per-unit amount).

  build(tabs)          -> list of targets {t, y, sys, fin, windows: [ {key: x} ]}
  propagate(E, anch)   crossword solver: equations with one unknown key vote for its value;
                       a key is fixed when >= MINV tablets agree (and beat the runner-up);
                       fixed keys feed the next round (propagation)
  predict(E, rates)    held-out exact matches (only windows whose keys are all fixed)
  redeal(E, rng)       null: target values re-dealt among targets of the same (system, final sign)
                       across tablets (each sign keeps its values; the alignment with counts dies)
No sign readings from anyone are used.  Values: PE capacity in N39C units (a-priori set, confirmed by
pe27: N30C:N24:N39B:N01:N14 = 1:3:6:30:180).
"""
import os, sys, json, random
from fractions import Fraction as Fr
from collections import defaultdict, Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe27_common import pe_tablets, ur3_tablets, DATA  # noqa
from common import load  # noqa

CK = os.path.join(DATA, 'pe28_ckpt')
os.makedirs(CK, exist_ok=True)
LBACK, W = 6, 3
G_CAP = [Fr(x) for x in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 45, 48, 50, 60, 72,
                         75, 80, 90, 96, 100, 120, 144, 150, 180, 240, 300, 360, 480, 600, 720)] + [Fr(1, 2)]
G_CNT = [Fr(1, 6), Fr(1, 5), Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(3, 2), Fr(2), Fr(3), Fr(4), Fr(5),
         Fr(6), Fr(10), Fr(12), Fr(20), Fr(30), Fr(60)]   # rate 1 (CNT -> CNT) excluded: digit echoes (pe27 F7)
G_UR = [Fr(x) for x in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 45, 50, 60, 75, 90, 100, 120, 150,
                        180, 240, 300)] + [Fr(1, 2), Fr(1, 3), Fr(2, 3)]


def grid_for(sysn, ur=False):
    if ur:
        return G_UR
    return G_CAP if sysn == 'CAP' else G_CNT


GSET = {k: set(v) for k, v in (('CAP', G_CAP), ('CNT', G_CNT), ('UR', G_UR))}

# pe27 anchors (final signs, any variant stripped for the counted sign; M288 target)
ANCH_SIGNS = ['M388', 'M054', 'M124', 'M003']


def anchors_pe():
    a = {}
    for s in ANCH_SIGNS:
        a[(s, 'M288', 'CAP')] = Fr(60)       # 2(N39B) 1(N24) per unit
        a[(s, 'M288', 'CNT')] = Fr(1, 2)     # 1/2 N01 per unit when written in N01
    return a


def strip(s):
    return s.split('~')[0].rstrip('#?!')


def pe_seqs(capset='apriori'):
    """per tablet: (id, [ (line, sys, value, fin_counted_base, fin_target_variant) ])"""
    raw = {t['id']: t for t in load()}
    out = []
    for t in pe_tablets(capset=capset):
        L = []
        lines = raw[t['id']]['lines']
        for q in sorted(t['Q'], key=lambda q: q['line']):
            if q['sys'] not in ('CNT', 'CAP'):
                continue
            sg = [s for s in lines[q['line']]['signs'] if s.startswith('M') or s.startswith('|')]
            finv = sg[-1].rstrip('#?!') if sg else '-'
            L.append((q['line'], q['sys'], q['v'], q['fin'], finv))
        if len(L) >= 2:
            out.append((t['id'], L))
    return out


def ur_seqs():
    out = []
    for t in ur3_tablets():
        L = [(q['line'], q['sys'], Fr(q['v']), q['fin'], q['fin']) for q in sorted(t['Q'], key=lambda q: q['line'])]
        if len(L) >= 2:
            out.append((t['id'], L, t['rates']))
    return out


def build(seqs, ur=False):
    """targets with their candidate windows.  key = (counted sign, target sign, target system)"""
    E = []
    for ti, s in enumerate(seqs):
        tid, L = s[0], s[1]
        for j in range(1, len(L)):
            ln, sy, y, _, finv = L[j]
            if y <= 0 or finv == '-':
                continue
            if ur and sy != 'CAP':
                continue
            wins = []
            lo = max(0, j - LBACK)
            for a in range(j - 1, lo - 1, -1):
                acc = Counter()
                for b in range(a, max(lo - 1, a - W), -1):
                    if L[b][1] != 'CNT' or L[b][3] == '-':
                        break
                    acc[(L[b][3], finv, 'UR' if ur else sy)] += L[b][2]
                    wins.append(dict(acc))
            if wins:
                # dedupe identical windows
                seen, W2 = set(), []
                for w in wins:
                    k = tuple(sorted(w.items()))
                    if k not in seen:
                        seen.add(k); W2.append(w)
                E.append({'t': ti, 'tid': tid, 'y': Fr(y), 'sys': sy, 'fin': finv, 'line': ln, 'W': W2})
    return E


def propagate(E, anchors, minv=3, margin=2.0, max_rounds=20, gsys=None, record=False):
    """Crossword solver.  Returns (fixed rates, info per new key)."""
    fixed = dict(anchors)
    info = {}
    for rnd in range(max_rounds):
        votes = defaultdict(set)
        prop = defaultdict(set)
        for e in E:
            for w in e['W']:
                unk = [k for k in w if k not in fixed]
                if len(unk) != 1:
                    continue
                k = unk[0]
                rest = e['y'] - sum(fixed[kk] * x for kk, x in w.items() if kk in fixed)
                if rest <= 0:
                    continue
                v = rest / w[k]
                g = gsys or k[2]
                if v not in GSET[g]:
                    continue
                votes[(k, v)].add(e['t'])
                if len(w) > 1:
                    prop[(k, v)].add(e['t'])
        bykey = defaultdict(list)
        for (k, v), ts in votes.items():
            bykey[k].append((len(ts), v))
        new = {}
        for k, lst in bykey.items():
            lst.sort(key=lambda a: (-a[0], a[1]))
            n1, v1 = lst[0]
            n2 = lst[1][0] if len(lst) > 1 else 0
            if n1 >= minv and n1 >= margin * n2:
                new[k] = v1
                info[k] = {'v': v1, 'n': n1, 'n2': n2, 'round': rnd, 'prop': len(prop[(k, v1)]),
                           'tabs': sorted(votes[(k, v1)])}
        if not new:
            break
        fixed.update(new)
    return fixed, info


def predict(E, rates, anchors, only_new=True):
    """per target: did any fully-fixed window (with >= 1 non-anchor key if only_new) hit y exactly?
    returns (n_targets_scored, n_hits, list of hit details)"""
    n, h, det = 0, 0, []
    for e in E:
        cand = []
        for w in e['W']:
            if all(k in rates for k in w) and (not only_new or any(k not in anchors for k in w)):
                cand.append(sum(rates[k] * x for k, x in w.items()))
        if not cand:
            continue
        n += 1
        if e['y'] in cand:
            h += 1
            det.append((e['tid'], e['line'], e['fin'], str(e['y'])))
    return n, h, det


def redeal(E, rng):
    pools = defaultdict(list)
    for i, e in enumerate(E):
        pools[(e['sys'], e['fin'])].append(i)
    out = [dict(e) for e in E]
    for idx in pools.values():
        ys = [E[i]['y'] for i in idx]
        p = rng.permutation(len(ys))
        for i, j in zip(idx, p):
            out[i]['y'] = ys[j]
    return out


def keystr(k):
    return f'{k[0]}->{k[1]}[{k[2]}]'
