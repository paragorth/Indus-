"""pe28 cycle 5: COMMODITY-ONLY rates.  Cycle 3 showed the M288 rate holds whatever sign heads the person
line, so the rate may belong to the entry (commodity) sign B, not to the counted sign.  Key = (B, system).
Equation: y_B = r(B) * x, x = the count line immediately before (any final sign).
Leave-one-tablet-out: r(B) = the grid value with most supporting tablets among the others (>= 3, 2:1 over the
runner-up); predict the left-out tablet's B lines.  Hits split by count x = 1 (a fixed entry value, a formula
constant) and x >= 2 (a true per-unit rate must scale).  Nulls 1000x: y re-dealt within (system, B) [global]
and within (system, B, tablet scale bin) [scale]; per-key search correction.  Ur III control (keys = target
noun; Ur III rates depend on the counted noun, so a commodity-only fit should be weaker there)."""
import os, sys, json, time
import numpy as np
from math import log2
from collections import Counter, defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe28_common import *  # noqa

NR = int(os.environ.get('NR', 1000))
t0 = time.time()


def pairs(seqs, ur=False):
    P = []
    for ti, s in enumerate(seqs):
        L = s[1]
        for j in range(1, len(L)):
            ln, sy, y, _, fin = L[j]
            if ur and sy != 'CAP':
                continue
            if y <= 0 or fin == '-' or L[j - 1][1] != 'CNT':
                continue
            P.append({'t': ti, 'tid': s[0], 'line': ln, 'key': (fin, 'UR' if ur else sy), 'x': L[j - 1][2], 'y': Fr(y)})
    return P


def fit_rates(P, ur):
    votes = defaultdict(lambda: defaultdict(set))
    for p in P:
        r = p['y'] / p['x']
        if r in GSET['UR' if ur else p['key'][1]]:
            votes[p['key']][r].add(p['t'])
    return votes


def choose(vk, excl):
    lst = sorted(((len(ts - {excl}), r) for r, ts in vk.items()), key=lambda a: (-a[0], a[1]))
    if not lst or lst[0][0] < 3:
        return None
    if len(lst) > 1 and lst[0][0] < 2 * lst[1][0]:
        return None
    return lst[0][1]


def run(P, ntab, scale, label, ur=False):
    votes = fit_rates(P, ur)
    pred = []          # (index, predicted y)
    for i, p in enumerate(P):
        r = choose(votes[p['key']], p['t'])
        if r is not None:
            pred.append((i, r * p['x']))
    ys = [p['y'] for p in P]

    def sc(yy):
        h1, h2, per = 0, 0, Counter()
        for i, v in pred:
            if yy[i] == v:
                if P[i]['x'] == 1:
                    h1 += 1
                else:
                    h2 += 1
                    per[P[i]['key']] += 1
        return h1, h2, per
    r1, r2, rper = sc(ys)
    rng = np.random.default_rng(3)
    out = {'pairs': len(P), 'scored': len(pred), 'scored_x>=2': sum(1 for i, _ in pred if P[i]['x'] != 1),
           'hits_x1': r1, 'hits_x>=2': r2}
    for mode in ('global', 'scale'):
        pl = defaultdict(list)
        for i, p in enumerate(P):
            pl[(p['key'],) if mode == 'global' else (p['key'], scale[p['t']])].append(i)
        pools = [np.array(v) for v in pl.values() if len(v) > 1]
        N1, N2 = np.zeros(NR), np.zeros(NR)
        NP = defaultdict(lambda: np.zeros(NR))
        for r in range(NR):
            yy = list(ys)
            for idx in pools:
                for a, b in zip(idx, rng.permutation(idx)):
                    yy[a] = ys[b]
            a1, a2, per = sc(yy)
            N1[r], N2[r] = a1, a2
            for k, c in per.items():
                NP[k][r] = c
        keys = list(set(rper) | set(NP))
        M = np.array([NP[k] for k in keys]) if keys else np.zeros((0, NR))
        tail = np.zeros_like(M)
        for a in range(len(keys)):
            srt = np.sort(M[a]); tail[a] = 1 - np.searchsorted(srt, M[a], side='left') / NR
        mn = tail.min(0) if len(keys) else np.ones(NR)
        rows = []
        for k, c in rper.most_common(15):
            pk = float((1 + (NP[k] >= c).sum()) / (1 + NR))
            rows.append({'key': f'{k[0]}[{k[1]}]', 'hits_x>=2': c, 'null': round(float(NP[k].mean()), 2), 'p': pk,
                         'p_corr': float((1 + (mn <= pk).sum()) / (1 + NR)),
                         'rate': str(choose(votes[k], -1)),
                         'lines': [(P[i]['tid'], P[i]['line'], str(P[i]['x']), str(v)) for i, v in pred
                                   if P[i]['key'] == k and P[i]['x'] != 1 and ys[i] == v][:8]})
        out[mode] = {'null_x1': float(N1.mean()), 'p_x1': float((1 + (N1 >= r1).sum()) / (1 + NR)),
                     'null_x>=2': float(N2.mean()), 'p_x>=2': float((1 + (N2 >= r2).sum()) / (1 + NR)), 'keys': rows}
    print(label, {k: v for k, v in out.items() if k not in ('global', 'scale')}, round(time.time() - t0), flush=True)
    for mode in ('global', 'scale'):
        print('  ', mode, {k: v for k, v in out[mode].items() if k != 'keys'}, flush=True)
        for r in out[mode]['keys'][:8]:
            print('      ', {k: v for k, v in r.items() if k != 'lines'}, flush=True)
    return out


def scales(seqs):
    return [int(log2(max(float(np.median([float(q[2]) for q in s[1]])), 1))) for s in seqs]


if __name__ == '__main__':
    res = {}
    US = ur_seqs()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(US), 800, replace=False)
    USs = [US[i] for i in idx]
    res['ur3'] = run(pairs(USs, ur=True), len(USs), scales(USs), 'UR3', ur=True)
    S = pe_seqs()
    P = pairs(S)
    res['pe'] = run(P, len(S), scales(S), 'PE')
    # planted: commodity PLC = 36 N39C per unit on 25 random CAP lines after counts with x >= 2
    rp = np.random.default_rng(9)
    elig = [i for i, p in enumerate(P) if p['key'][1] == 'CAP' and p['x'] >= 2]
    P2 = [dict(p) for p in P]
    for i in rp.choice(elig, 25, replace=False):
        P2[i]['key'] = ('PLC', 'CAP'); P2[i]['y'] = Fr(36) * P2[i]['x']
    res['plant'] = run(P2, len(S), scales(S), 'PLANT')
    json.dump(res, open(os.path.join(CK, 'cycle5.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
