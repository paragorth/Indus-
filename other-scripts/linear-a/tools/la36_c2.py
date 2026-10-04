#!/usr/bin/env python3
"""LA-36 cycle 2: DOES A WORD MARK THE DIRECTION OF CHANGE?

Units: every pair of lists sharing >= 1 recipient word in the same commodity (LA sides; LB
DAMOS documents), plus the PY Ma line pairs (target line vs each later line, aligned by commodity).
Direction of a unit: sign of the median log(y / x) over the aligned entries (units with 0 dropped).
Marker words: words on one list and not on the other (shared recipient words excluded). For every
word seen in >= 3 directional units: k = units where the word sits on the larger side.
Statistic: best two-sided binomial P over words (and the number of words at P <= 0.05).
Null: each unit's direction flipped at random (5,000 replicates) -> search-corrected P.
Second null for LA: direction taken from a random same-site list pair (amount-scale confound).
Controls: PY Ma (a-pu-do-si / o-pe-ro / o-u-di-do-si lines are smaller than the target), the PY Es
dossier (do-so-mo vs e-ke ... pe-mo), Linear B generally; planted marker (a random LA word forced to
the smaller side in its units).
"""
import json, os, sys, math
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la36_common import *

NR = int(os.environ.get('NR', 5000))
LISTNULL = int(os.environ.get('LISTNULL', 1))


def list_units(L, cls_fn):
    U = []
    for a, b in shared_pairs(L, 1):
        A = {e[0]: e for e in L[a]['ents']}; B = {e[0]: e for e in L[b]['ents']}
        keys = [w for w in A if w in B and A[w][1] == B[w][1]]
        if not keys: continue
        x = [value(A[w][2], A[w][3], CONV) if not isinstance(A[w][2], float) else A[w][2] for w in keys]
        y = [value(B[w][2], B[w][3], CONV) if not isinstance(B[w][2], float) else B[w][2] for w in keys]
        r = [math.log(q / p) for p, q in zip(x, y) if p > 0 and q > 0]
        if not r: continue
        d = np.sign(np.median(r))
        wa = set(L[a]['words']) - set(keys) - set(L[b]['words'])
        wb = set(L[b]['words']) - set(keys) - set(L[a]['words'])
        U.append({'id': f'{a}|{b}', 'cls': cls_fn(a, b, L), 'd': d, 'wa': wa, 'wb': wb, 'site': L[a]['site'], 'A': a, 'B': b})
    return U


def ma_dir_units():
    U = []
    for h, lines in ma_lines():
        w0, c0, o0, ws0 = lines[0]
        for w, c, o, ws in lines[1:]:
            ks = [k for k in c if k in c0 and c0[k] > 0 and c[k] > 0]
            if not ks: continue
            d = np.sign(np.median([math.log(c[k] / c0[k]) for k in ks]))
            U.append({'id': h, 'cls': 'MA', 'd': d, 'wa': set(ws0) - set(ws), 'wb': set(ws) - set(ws0), 'site': 'PY',
                      'A': h + ':0', 'B': h + ':' + w + str(sorted(c.items()))})
    return U


def word_table(U, dirs):
    up = Counter(); n = Counter()
    for u, d in zip(U, dirs):
        if d == 0: continue
        for w in u['wb']: n[w] += 1; up[w] += d > 0
        for w in u['wa']: n[w] += 1; up[w] += d < 0
    return up, n


def best(U, dirs, words):
    up, n = word_table(U, dirs)
    ps = {w: binomtest(up[w], n[w], 0.5).pvalue for w in words}
    return ps, up, n


def analyse(U, tag, rng, min_n=3):
    dirs = np.array([u['d'] for u in U])
    up, n = word_table(U, dirs)
    words = sorted(w for w in n if n[w] >= min_n)
    if not words:
        print(tag, 'no words', flush=True); return {}
    ps, _, _ = best(U, dirs, words)
    pmin = min(ps.values()); n05 = sum(p <= 0.05 for p in ps.values())
    # null: direction flips; fast binomial via cached table
    cache = {}
    def bp(k, m):
        if (k, m) not in cache: cache[(k, m)] = binomtest(k, m, 0.5).pvalue
        return cache[(k, m)]
    nm = []; nn = []
    idx = defaultdict(list)
    for i, u in enumerate(U):
        if u['d'] == 0: continue
        for w in u['wb']:
            if w in ps: idx[w].append((i, 1))
        for w in u['wa']:
            if w in ps: idx[w].append((i, -1))
    lists = sorted({u['A'] for u in U} | {u['B'] for u in U}); li = {l: i for i, l in enumerate(lists)}
    ia = np.array([li[u['A']] for u in U]); ib = np.array([li[u['B']] for u in U])
    for _ in range(NR):
        if LISTNULL:
            z = rng.standard_normal(len(lists)); fl = np.sign(z[ib] - z[ia]) * (dirs != 0)
        else:
            fl = rng.choice([-1, 1], size=len(U)) * dirs
        pv = [bp(sum(1 for i, s in idx[w] if fl[i] * s > 0), len(idx[w])) for w in words]
        nm.append(min(pv)); nn.append(sum(p <= 0.05 for p in pv))
    P_best = (np.sum(np.array(nm) <= pmin) + 1) / (NR + 1)
    P_n = (np.sum(np.array(nn) >= n05) + 1) / (NR + 1)
    top = sorted(ps.items(), key=lambda t: t[1])[:8]
    nlist = {w: len({(u['B'] if w in u['wb'] else u['A']) for u in U if w in u['wa'] or w in u['wb']}) for w, _ in top}
    res = {'null': 'list' if LISTNULL else 'flip', 'units': int(np.sum(dirs != 0)), 'up_frac': float(np.mean(dirs[dirs != 0] > 0)), 'words': len(words),
           'pmin': float(pmin), 'P_best': float(P_best), 'n05': int(n05), 'n05_null': float(np.mean(nn)), 'P_n05': float(P_n),
           'top': [(w, int(up[w]), int(n[w]), nlist[w], round(p, 4)) for w, p in top]}
    print(tag, json.dumps(res, default=float), flush=True)
    return res


def analyse_lists(U, tag, rng, min_lists=3):
    """Word -> distinct lists carrying it; each list votes the majority direction of its units
    (its side larger = +1). Binomial over lists; list-level random-effect null."""
    lists = sorted({u['A'] for u in U} | {u['B'] for u in U}); li = {l: i for i, l in enumerate(lists)}
    ia = np.array([li[u['A']] for u in U]); ib = np.array([li[u['B']] for u in U])
    dirs = np.array([u['d'] for u in U])
    carry = defaultdict(set)
    for u in U:
        for w in u['wa']: carry[w].add(li[u['A']])
        for w in u['wb']: carry[w].add(li[u['B']])
    words = sorted(w for w in carry if len(carry[w]) >= min_lists)
    if not words: print(tag, 'no words'); return {}
    def votes(dd):
        v = np.zeros(len(lists))
        np.add.at(v, ib, dd); np.add.at(v, ia, -dd)
        return np.sign(v)
    cache = {}
    def bp(k, m):
        if (k, m) not in cache: cache[(k, m)] = binomtest(k, m, 0.5).pvalue if m else 1.0
        return cache[(k, m)]
    def scores(v):
        out = {}
        for w in words:
            vv = v[list(carry[w])]; vv = vv[vv != 0]
            out[w] = (bp(int((vv > 0).sum()), len(vv)), int((vv > 0).sum()), len(vv))
        return out
    obs = scores(votes(dirs))
    pmin = min(p for p, _, _ in obs.values()); n05 = sum(p <= 0.05 for p, _, _ in obs.values())
    nm, nn = [], []
    for _ in range(NR):
        z = rng.standard_normal(len(lists)); dd = np.sign(z[ib] - z[ia]) * (dirs != 0)
        sc = scores(votes(dd)); ps = [p for p, _, _ in sc.values()]
        nm.append(min(ps)); nn.append(sum(p <= 0.05 for p in ps))
    res = {'stat': 'lists', 'lists': len(lists), 'words': len(words), 'pmin': float(pmin),
           'P_best': float((np.sum(np.array(nm) <= pmin) + 1) / (NR + 1)), 'n05': int(n05),
           'n05_null': float(np.mean(nn)), 'P_n05': float((np.sum(np.array(nn) >= n05) + 1) / (NR + 1)),
           'top': [(w, k, m, round(p, 4)) for w, (p, k, m) in sorted(obs.items(), key=lambda t: t[1][0])[:8]]}
    print(tag, json.dumps(res, default=float), flush=True)
    return res


def main():
    rng = np.random.default_rng(362)
    out = {}
    L = la_lists()
    LA = list_units(L, pair_class)
    print('LA units', len(LA), Counter(u['cls'] for u in LA), flush=True)
    out['LA'] = analyse(LA, 'LA_all', rng)
    out['LA_same_tablet_or_scribe'] = analyse([u for u in LA if u['cls'] in ('SIDES', 'SCRIBE')], 'LA_SIDES+SCRIBE', rng, 2)
    out['LA_HT'] = analyse([u for u in LA if u['site'] == 'Haghia Triada'], 'LA_HT', rng)
    # planted marker: pick a word seen in >= 4 units, force it onto the smaller side
    up, n = word_table(LA, [u['d'] for u in LA])
    cand = sorted(w for w in n if n[w] >= 4)
    det = 0; trials = 20
    for t in range(trials):
        w = cand[rng.integers(len(cand))]
        Pl = []
        for u in LA:
            v = dict(u)
            if w in u['wb'] and u['d'] != 0: v['d'] = -1.0
            if w in u['wa'] and u['d'] != 0: v['d'] = 1.0
            Pl.append(v)
        global NR
        keep = NR; NR = 500
        r = analyse(Pl, f'PLANT_{t}_{w}', rng)
        NR = keep
        det += bool(r) and r['P_best'] <= 0.05 and r['top'][0][0] == w
    print('planted marker recovered', det, 'of', trials, flush=True)
    out['planted'] = [det, trials, cand]
    # Linear B
    LBL = lb_lists()
    LB = list_units(LBL, lb_class)
    print('LB units', len(LB), flush=True)
    out['LB'] = analyse(LB, 'LB_all', rng)
    out['LB_Es'] = analyse([u for u in LB if 'PY Es' in u['id']], 'LB_PY_Es', rng, 2)
    MA = ma_dir_units()
    print('MA units', len(MA), Counter(u['d'] for u in MA), flush=True)
    out['MA'] = analyse(MA, 'PY_Ma', rng, 2)
    out['L_LA'] = analyse_lists(LA, 'LISTS_LA', rng)
    out['L_LB'] = analyse_lists(LB, 'LISTS_LB', rng)
    out['L_MA'] = analyse_lists(MA, 'LISTS_MA', rng)
    det = 0
    for t in range(20):
        w = cand[rng.integers(len(cand))]
        Pl = []
        for u in LA:
            v = dict(u)
            if w in u['wb'] and u['d'] != 0: v['d'] = -1.0
            if w in u['wa'] and u['d'] != 0: v['d'] = 1.0
            Pl.append(v)
        keep = NR; NR = 300
        r = analyse_lists(Pl, f'LPLANT_{t}_{w}', rng)
        NR = keep
        det += bool(r) and r['P_best'] <= 0.05
    print('list-stat planted marker recovered', det, 'of 20', flush=True)
    out['L_planted'] = det
    json.dump(out, open(os.path.join(CK, f'c2_null{LISTNULL}.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
